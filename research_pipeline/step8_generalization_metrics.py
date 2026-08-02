"""
step8_generalization_metrics.py
================================
Computes multi-dimensional generalization diagnostic metrics for Leave-One-Refrigerant-Out (LORO).
Fulfills all 12 defense constraints:
  1. Nearest neighbor identification for each fingerprint
  2. Multi-fingerprint CCI (Morgan r=2, Morgan r=3, MACCS)
  3. Explicit IL pair identity with seen/unseen counts
  4. ILCR by count (ilcr_count) and by sample (ilcr_sample)
  5. Individual z-scores saved for all physical descriptors
  6. RMS normalized distances for D_chem and D_TP
  7. Strict check for target refrigerants in GNN results
  8. Interface for baseline model results (GCN, RF, XGB, MLP)
  9. Objective, label-free data output
 10. Reproducibility metadata (timestamp, FP settings)
 11. Fingerprint rank consistency correlation
 12. Pure diagnostic metric calculation (no manual composite formulas)
"""

import os
import sys
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs

# Resolve ROOT directory using pathlib
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

TARGET_REFRIGERANTS = ['R32', 'R152a', 'R125', 'R1234yf', 'R134a', 'R22']

FAMILY_MAP = {
    'R32': 'HFC',
    'R152a': 'HFC',
    'R125': 'HFC',
    'R1234yf': 'HFO',
    'R134a': 'HFC',
    'R22': 'HCFC'
}

def get_fingerprints(mol):
    """Generate 3 types of fingerprints for a given RDKit molecule."""
    fp_r2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_r3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
    fp_maccs = MACCSkeys.GenMACCSKeys(mol)
    return fp_r2, fp_r3, fp_maccs

def compute_metrics():
    print("============================================================")
    print("  Step 8: Generalization Metrics & Landscape Diagnostic Engine")
    print("============================================================")
    
    # 1. Read index_with_anion.csv
    csv_path = ROOT / 'index_with_anion.csv'
    df = pd.read_csv(csv_path)
    
    # Check GNN results file
    gnn_csv = ROOT / 'loro_gnn_results.csv'
    if gnn_csv.exists():
        gnn_df = pd.read_csv(gnn_csv)
    else:
        gnn_df = pd.DataFrame()
        
    # Check Baselines results file
    base_csv = ROOT / 'loro_baselines_results.csv'
    if base_csv.exists():
        base_df = pd.read_csv(base_csv)
    else:
        base_df = pd.DataFrame()
        
    # 2. Extract fingerprints for all unique refrigerants
    ref_df = df[['refrigerant', 'refri_smiles']].drop_duplicates()
    fps_r2, fps_r3, fps_maccs = {}, {}, {}
    for _, row in ref_df.iterrows():
        ref_name = row['refrigerant']
        smiles = row['refri_smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            r2, r3, maccs = get_fingerprints(mol)
            fps_r2[ref_name] = r2
            fps_r3[ref_name] = r3
            fps_maccs[ref_name] = maccs
        else:
            print(f"[Warning] Could not parse SMILES for {ref_name}: {smiles}")

    # 3. Load numpy raw data for descriptor z-scores
    # data[i] structure: [0-2] mol graphs, [3] T, [4] P, [5] ref_charge,
    #                     [6] ref_logp, [7] ani_mw, [8] cat_charge, [9] cat_tpsa
    raw_data = np.load(ROOT / 'processed_tri_data' / 'data.npy', allow_pickle=True)
    labels = np.load(ROOT / 'processed_tri_data' / 'label.npy', allow_pickle=True).flatten()

    all_t = np.array([float(raw_data[i][3]) for i in range(len(raw_data))])
    all_p = np.array([float(raw_data[i][4]) for i in range(len(raw_data))])
    all_ref_charge = np.array([float(raw_data[i][5]) for i in range(len(raw_data))])
    all_ref_logp   = np.array([float(raw_data[i][6]) for i in range(len(raw_data))])
    all_cat_charge = np.array([float(raw_data[i][8]) for i in range(len(raw_data))])
    all_cat_tpsa   = np.array([float(raw_data[i][9]) for i in range(len(raw_data))])

    landscape_rows = []
    
    for ref_name in TARGET_REFRIGERANTS:
        if ref_name not in fps_r2:
            print(f"[Warning] Skipping {ref_name} (SMILES missing or invalid)")
            continue

        # --- A. Train-Only CCI Calculation (Patch 1 & Patch 7) ---
        # Exclude target_ref from candidates (j in Training Refrigerants)
        candidates = [r for r in fps_r2.keys() if r != ref_name]
        
        best_sim_r2, nearest_r2 = -1.0, None
        for cand in candidates:
            sim = DataStructs.TanimotoSimilarity(fps_r2[ref_name], fps_r2[cand])
            if sim > best_sim_r2:
                best_sim_r2, nearest_r2 = sim, cand

        best_sim_r3, nearest_r3 = -1.0, None
        for cand in candidates:
            sim = DataStructs.TanimotoSimilarity(fps_r3[ref_name], fps_r3[cand])
            if sim > best_sim_r3:
                best_sim_r3, nearest_r3 = sim, cand

        best_sim_maccs, nearest_maccs = -1.0, None
        for cand in candidates:
            sim = DataStructs.TanimotoSimilarity(fps_maccs[ref_name], fps_maccs[cand])
            if sim > best_sim_maccs:
                best_sim_maccs, nearest_maccs = sim, cand

        # --- B. IL Coverage Ratio (ILCR) Calculation (Patch 2, 3, 4) ---
        # Unique IL identity defined by (cation_smiles, anion_smiles)
        test_mask = (df['refrigerant'] == ref_name)
        train_mask = ~test_mask

        test_indices = df[test_mask].index.tolist()
        train_indices = df[train_mask].index.tolist()

        test_il_pairs = set(df[test_mask][['cation_smiles', 'anion_smiles']].apply(tuple, axis=1))
        train_il_pairs = set(df[train_mask][['cation_smiles', 'anion_smiles']].apply(tuple, axis=1))

        seen_ils = test_il_pairs & train_il_pairs
        unseen_ils = test_il_pairs - train_il_pairs

        n_il_total = len(test_il_pairs)
        n_il_seen = len(seen_ils)
        n_il_unseen = len(unseen_ils)

        ilcr_count = n_il_seen / n_il_total if n_il_total > 0 else 0.0

        # Sample-weighted ILCR (ilcr_sample)
        test_df_sub = df[test_mask]
        seen_samples_count = test_df_sub.apply(
            lambda r: (r['cation_smiles'], r['anion_smiles']) in train_il_pairs, axis=1
        ).sum()
        ilcr_sample = seen_samples_count / len(test_df_sub) if len(test_df_sub) > 0 else 0.0

        # --- C. Descriptor Distribution Shift (Patch 3, 5, 6) ---
        # Calculate z-scores for test vs train distribution
        def calc_z(test_vals, train_vals):
            tr_mean = train_vals.mean()
            tr_std = train_vals.std()
            if tr_std == 0:
                return 0.0
            return float(abs(test_vals.mean() - tr_mean) / tr_std)

        z_ref_charge = calc_z(all_ref_charge[test_indices], all_ref_charge[train_indices])
        z_ref_logp   = calc_z(all_ref_logp[test_indices], all_ref_logp[train_indices])
        z_cat_charge = calc_z(all_cat_charge[test_indices], all_cat_charge[train_indices])
        z_cat_tpsa   = calc_z(all_cat_tpsa[test_indices], all_cat_tpsa[train_indices])

        z_t          = calc_z(all_t[test_indices], all_t[train_indices])
        z_p          = calc_z(all_p[test_indices], all_p[train_indices])

        # RMS normalized distance for D_chem (4 features) and D_TP (2 features)
        d_chem = float(np.sqrt(np.mean([z_ref_charge**2, z_ref_logp**2, z_cat_charge**2, z_cat_tpsa**2])))
        d_tp   = float(np.sqrt(np.mean([z_t**2, z_p**2])))

        # --- D. Model Performance Interfacing (Patch 7 & 8) ---
        r2_mean, r2_std, mae_mean = np.nan, np.nan, np.nan
        gcn_r2_mean, gcn_r2_std = np.nan, np.nan
        
        if not gnn_df.empty:
            gat_sub = gnn_df[(gnn_df['refrigerant'] == ref_name) & (gnn_df['model'] == 'GAT_v5')]
            if not gat_sub.empty:
                r2_mean = gat_sub.iloc[0]['r2_mean']
                r2_std = gat_sub.iloc[0]['r2_std']
                mae_mean = gat_sub.iloc[0]['mae_mean']

            gcn_sub = gnn_df[(gnn_df['refrigerant'] == ref_name) & (gnn_df['model'] == 'GCN_v5')]
            if not gcn_sub.empty:
                gcn_r2_mean = gcn_sub.iloc[0]['r2_mean']
                gcn_r2_std = gcn_sub.iloc[0]['r2_std']

        rf_r2, xgb_r2, mlp_r2 = np.nan, np.nan, np.nan
        if not base_df.empty:
            sub_rf = base_df[(base_df['refrigerant'] == ref_name) & (base_df['model'] == 'RF')]
            if not sub_rf.empty: rf_r2 = sub_rf.iloc[0]['r2']

            sub_xgb = base_df[(base_df['refrigerant'] == ref_name) & (base_df['model'] == 'XGBoost')]
            if not sub_xgb.empty: xgb_r2 = sub_xgb.iloc[0]['r2']

            sub_mlp = base_df[(base_df['refrigerant'] == ref_name) & (base_df['model'] == 'MLP')]
            if not sub_mlp.empty: mlp_r2 = sub_mlp.iloc[0]['r2']

        landscape_rows.append({
            'refrigerant': ref_name,
            'family': FAMILY_MAP.get(ref_name, 'Unknown'),
            'n_test': len(test_indices),
            
            # GAT_v5 performance
            'gat_r2_mean': r2_mean,
            'gat_r2_std': r2_std,
            'gat_mae_mean': mae_mean,

            # GCN_v5 performance
            'gcn_r2_mean': gcn_r2_mean,
            'gcn_r2_std': gcn_r2_std,

            # Baselines performance
            'rf_r2': rf_r2,
            'xgb_r2': xgb_r2,
            'mlp_r2': mlp_r2,

            # Train-Only CCI & Nearest Neighbors
            'cci_ref_morgan_r2': best_sim_r2,
            'nearest_ref_morgan_r2': nearest_r2,
            'cci_ref_morgan_r3': best_sim_r3,
            'nearest_ref_morgan_r3': nearest_r3,
            'cci_ref_maccs': best_sim_maccs,
            'nearest_ref_maccs': nearest_maccs,

            # IL Coverage Indicators
            'ilcr_count': ilcr_count,
            'ilcr_sample': ilcr_sample,
            'n_il_total': n_il_total,
            'n_il_seen': n_il_seen,
            'n_il_unseen': n_il_unseen,

            # Chemical & Condition Shift (RMS z-scores)
            'd_chem': d_chem,
            'd_tp': d_tp,
            'z_ref_charge': z_ref_charge,
            'z_ref_logp': z_ref_logp,
            'z_cat_charge': z_cat_charge,
            'z_cat_tpsa': z_cat_tpsa,
            'z_t': z_t,
            'z_p': z_p,

            # Metadata
            'fingerprint_version': 'Morgan_r2_r3_MACCS',
            'dataset_version': 'v5_index_with_anion',
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    res_df = pd.DataFrame(landscape_rows)
    out_csv = ROOT / 'loro_generalization_landscape.csv'
    res_df.to_csv(out_csv, index=False)
    print(f"\n[Done] Generalization landscape exported to: {out_csv}")

    # --- E. Terminal Diagnostics Summary ---
    print("\n======================= Generalization Diagnosis =======================")
    for _, r in res_df.iterrows():
        r_name = r['refrigerant']
        cci_val = r['cci_ref_morgan_r2']
        ilcr_val = r['ilcr_count']
        dchem_val = r['d_chem']
        r2_m = r['gat_r2_mean']
        r2_s = r['gat_r2_std']
        nn = r['nearest_ref_morgan_r2']
        
        r2_str = f"{r2_m:.4f} +/- {r2_s:.4f}" if not np.isnan(r2_m) else "N/A"
        print(f"  {r_name:8s} | CCI: {cci_val:.3f} (NN: {nn:7s}) | ILCR: {ilcr_val:.3f} | D_chem: {dchem_val:.3f} | GAT R2: {r2_str}")
    print("========================================================================\n")

    # --- F. Check Fingerprint Rank Consistency (Patch 11) ---
    if len(res_df) >= 3:
        rho_r2_r3, _ = spearmanr(res_df['cci_ref_morgan_r2'], res_df['cci_ref_morgan_r3'])
        rho_r2_maccs, _ = spearmanr(res_df['cci_ref_morgan_r2'], res_df['cci_ref_maccs'])
        print(f"  Fingerprint Rank Consistency Check:")
        print(f"    Spearman(Morgan_r2, Morgan_r3): {rho_r2_r3:.4f}")
        print(f"    Spearman(Morgan_r2, MACCS):     {rho_r2_maccs:.4f}")

    return res_df

if __name__ == '__main__':
    compute_metrics()
