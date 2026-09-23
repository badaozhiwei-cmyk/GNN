"""
phase1_hfo_zeroshot_probe.py — Phase 1: Unsaturated HFO/HCFO Zero-Shot Probe
=============================================================================
Evaluates Frozen Checkpoints across 3 Independent 5-Member Ensembles:
  - Ensemble V6-M0 (Seeds 42-46)
  - Ensemble V7-A  (Seeds 42-46)
  - Ensemble V7-B  (Seeds 42-46)

Strict Protocols:
  1. Zero retraining or fine-tuning (Pure zero-shot inference).
  2. Training-only HFC scaler used (no HFO leakage).
  3. Sample-level N=1106 across 5 species:
     - R1234yf: 685
     - R1234ze(E): 308
     - R1233zd(E): 90
     - R1336mzz(Z): 12
     - R1336mzz(E): 11
  4. Separate epistemic uncertainty (sigma_ens) and error-uncertainty correlation.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import Descriptors

import torch
from torch_geometric.data import Data, Batch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v7_shadow_experiment"))
sys.path.insert(0, str(ROOT / "GNN_for_property_prediction"))

from Model_v6 import IL_GAT_v6
from Model_v7 import IL_GAT_v7
from Dataset_v6 import combine_Graph, add_global
from prepare_tri_graph_data_v6 import mol2graph_components

SEEDS = [42, 43, 44, 45, 46]

HFO_CRITICAL = {
    'R1234YF': (367.85, 3.382, 0.276),
    'R1234ZE(E)': (382.51, 3.635, 0.313),
    'R1233ZD(E)': (438.86, 3.5828, 0.304),
    'R1336MZZ(E)': (403.53, 2.779, 0.4128),
    'R1336MZZ(Z)': (444.50, 2.9037, 0.386),
}

HFO_CANONICAL_SMILES = {
    'R1234YF': 'C=C(F)C(F)(F)F',
    'R1234ZE(E)': 'F/C=C/C(F)(F)F',
    'R1233ZD(E)': 'FC(F)(F)/C=C/Cl',
    'R1336MZZ(E)': 'FC(F)(F)/C=C/C(F)(F)F',
    'R1336MZZ(Z)': r'FC(F)(F)/C=C\C(F)(F)F',
}

def mol2graph_v6(mol_data):
    x = torch.tensor(mol_data[0], dtype=torch.long)
    edge_index = torch.tensor(mol_data[1], dtype=torch.long)
    if len(mol_data[2]) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros((1, 3), dtype=torch.long)
    else:
        edge_attr = torch.tensor(mol_data[2], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def mol2graph_v7(mol_data, target_edge_dim=4):
    x = torch.tensor(mol_data[0], dtype=torch.long)
    edge_index = torch.tensor(mol_data[1], dtype=torch.long)
    if len(mol_data[2]) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros((1, target_edge_dim), dtype=torch.long)
    else:
        raw_attr = torch.tensor(mol_data[2], dtype=torch.long)
        if target_edge_dim == 4 and raw_attr.size(1) == 3:
            zero_stereo = torch.zeros((raw_attr.size(0), 1), dtype=torch.long)
            edge_attr = torch.cat([raw_attr, zero_stereo], dim=1)
        else:
            edge_attr = raw_attr
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def main():
    print("=" * 80)
    print("  PHASE 1: HFO/HCFO ZERO-SHOT STRESS TEST & 5-MEMBER UQ AUDIT")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference Device: {device}")

    # 1. Load Data
    df_raw = pd.read_csv(ROOT / 'index_with_anion.csv')
    df_hfo = df_raw[df_raw['sheet'] == 'Table S4. VLE HFOs'].copy().reset_index()
    df_hfo.rename(columns={'index': 'orig_data_idx'}, inplace=True)
    N = len(df_hfo)
    assert N == 1106, f"Expected N=1106, got {N}"
    print(f"[Data Check] Table S4 samples loaded: N={N}")

    raw_data = np.load(ROOT / 'processed_tri_data' / 'data.npy', allow_pickle=True)

    mw_cache = {}
    def get_mw(smi):
        if smi not in mw_cache:
            mol = Chem.MolFromSmiles(smi)
            mw_cache[smi] = float(Descriptors.MolWt(mol)) if mol else 0.0
        return mw_cache[smi]

    # 2. Load Scalers (strictly training-only, 9 features)
    scaler_path = ROOT / 'results_hfc_all' / 'HFC_all_M0' / 'scalers.pkl'
    scalers = joblib.load(scaler_path)
    assert len(scalers) == 9
    scaler_means = np.array([float(s.mean_[0]) for s in scalers], dtype=np.float32)
    scaler_scales = np.array([max(float(s.scale_[0]), 1e-8) for s in scalers], dtype=np.float32)
    print(f"[Scaler Check] Loaded 9-dim training scalers: T_mean={scaler_means[0]:.2f}, P_mean={scaler_means[1]:.4f}")

    # 3. Assemble Samples
    v6_graphs = []
    v7_cats = []
    v7_anis = []
    v7_refs = []
    state_conds = []
    desc_conds = []
    all_conds = []
    targets = []
    species_list = []
    sample_ids = []

    for _, r in df_hfo.iterrows():
        orig_i = int(r['orig_data_idx'])
        raw_item = raw_data[orig_i]
        r_clean = str(r['refrigerant']).strip().upper()
        species_list.append(r_clean)
        sample_ids.append(f"{r['cation']}__{r['anion']}__{r['refrigerant']}__{r['T_K']}__{r['P_MPa']}")

        ref_smi = HFO_CANONICAL_SMILES.get(r_clean, str(r['refri_smiles']).strip())

        # V6 graphs
        cg_v6 = mol2graph_v6(raw_item[0])
        ag_v6 = mol2graph_v6(raw_item[1])
        if r_clean == 'R1234YF':
            rg_v6 = mol2graph_v6(mol2graph_components(ref_smi))
        else:
            rg_v6 = mol2graph_v6(raw_item[2])
        combined_g = add_global(combine_Graph([cg_v6, ag_v6, rg_v6]))
        v6_graphs.append(combined_g)

        # V7 decoupled graphs
        cg_v7 = mol2graph_v7(raw_item[0], target_edge_dim=4)
        ag_v7 = mol2graph_v7(raw_item[1], target_edge_dim=4)
        if r_clean == 'R1234YF':
            rg_v7 = mol2graph_v7(mol2graph_components(ref_smi), target_edge_dim=4)
        else:
            rg_v7 = mol2graph_v7(raw_item[2], target_edge_dim=4)
        v7_cats.append(cg_v7)
        v7_anis.append(ag_v7)
        v7_refs.append(rg_v7)

        # 9 Condition features: [T, P, ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa, ref_mw, cat_mw]
        t_val = float(r['T_K'])
        p_val = float(r['P_MPa'])
        ref_mw = get_mw(ref_smi)
        cat_mw = get_mw(r['cation_smiles'])
        ani_mw = get_mw(r['anion_smiles'])

        mol_ref = Chem.MolFromSmiles(ref_smi)
        ref_logp = float(Descriptors.MolLogP(mol_ref)) if mol_ref else 0.0
        try:
            ref_charge = float(Descriptors.MaxAbsPartialCharge(mol_ref)) if mol_ref else 0.0
        except Exception:
            ref_charge = 0.0

        mol_cat = Chem.MolFromSmiles(r['cation_smiles'])
        cat_tpsa = float(Descriptors.TPSA(mol_cat)) if mol_cat else 0.0
        try:
            cat_charge = float(Descriptors.MaxAbsPartialCharge(mol_cat)) if mol_cat else 0.0
        except Exception:
            cat_charge = 0.0

        raw_cond_9 = np.array([
            t_val, p_val,
            ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa,
            ref_mw, cat_mw
        ], dtype=np.float32)

        norm_cond_9 = (raw_cond_9 - scaler_means) / scaler_scales
        all_conds.append(norm_cond_9)
        state_conds.append(norm_cond_9[0:2])
        desc_conds.append(norm_cond_9[2:9])
        targets.append(float(r['x1']))

    targets = np.array(targets, dtype=np.float32)
    all_conds_tensor = torch.tensor(np.array(all_conds), dtype=torch.float32)
    state_conds_tensor = torch.tensor(np.array(state_conds), dtype=torch.float32)
    desc_conds_tensor = torch.tensor(np.array(desc_conds), dtype=torch.float32)
    species_arr = np.array(species_list)

    print(f"[Assembly Check] 1106 samples prepared. Species count: {pd.Series(species_list).value_counts().to_dict()}")

    # 4. Checkpoint paths
    v6_ckpt_dir = ROOT / "results_hfc_all" / "HFC_all_M0"
    v7a_ckpt_dir = Path(r"D:\折腾\v7_pilot_gpu_results\v7_shadow_experiment\checkpoints\V7-A")
    v7b_ckpt_dir = Path(r"D:\折腾\v7_pilot_gpu_results2\v7_shadow_experiment\checkpoints\V7-B")

    # 5. Run Inference
    results_dict = {
        'sample_id': sample_ids,
        'refrigerant': species_list,
        'true_x1': targets
    }

    batch_size = 64
    num_samples = len(targets)

    # 5.1 V6-M0 Inference (5 seeds)
    print("\n>>> Running V6-M0 Inference (5 Seeds)...")
    model_args_v6 = {
        'emb_dim': 300,
        'dropout_rate': 0.2,
        'cond_dim': 9,
        'pool': 'global',
        'use_layernorm': False,
        'use_adaptive_gate': False
    }
    v6_preds = {s: [] for s in SEEDS}
    for s in SEEDS:
        ckpt_path = v6_ckpt_dir / f"best_seed_{s}.pth"
        model_v6 = IL_GAT_v6(model_args_v6).to(device)
        raw_ckpt = torch.load(ckpt_path, map_location=device)
        st = raw_ckpt['model_state_dict'] if 'model_state_dict' in raw_ckpt else raw_ckpt
        model_v6.load_state_dict(st)
        model_v6.eval()

        s_preds = []
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                b_graphs = Batch.from_data_list(v6_graphs[i:i+batch_size]).to(device)
                b_cond = all_conds_tensor[i:i+batch_size].to(device)
                out = model_v6(b_graphs, b_cond).view(-1).cpu().numpy()
                s_preds.extend(out)
        v6_preds[s] = np.array(s_preds, dtype=np.float32)
        results_dict[f'pred_v6_seed_{s}'] = v6_preds[s]

    v6_mat = np.column_stack([v6_preds[s] for s in SEEDS])
    v6_ens_mean = np.mean(v6_mat, axis=1)
    v6_ens_sigma = np.std(v6_mat, axis=1, ddof=1)
    results_dict['pred_v6_ens_mean'] = v6_ens_mean
    results_dict['sigma_v6_ens'] = v6_ens_sigma
    results_dict['ae_v6_ens'] = np.abs(targets - v6_ens_mean)

    # 5.2 V7-A Inference (5 seeds)
    print(">>> Running V7-A Inference (5 Seeds)...")
    v7a_preds = {s: [] for s in SEEDS}
    for s in SEEDS:
        ckpt_path = v7a_ckpt_dir / f"best_seed_{s}.pth"
        model_v7a = IL_GAT_v7({"emb_dim": 300, "hidden_dim": 512, "desc_dropout_p": 0.0}).to(device)
        raw_ckpt = torch.load(ckpt_path, map_location=device)
        st = raw_ckpt['model_state_dict'] if 'model_state_dict' in raw_ckpt else raw_ckpt
        model_v7a.load_state_dict(st)
        model_v7a.eval()

        s_preds = []
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                b_dict = {
                    'cat': Batch.from_data_list(v7_cats[i:i+batch_size]).to(device),
                    'ani': Batch.from_data_list(v7_anis[i:i+batch_size]).to(device),
                    'ref': Batch.from_data_list(v7_refs[i:i+batch_size]).to(device),
                    'state': state_conds_tensor[i:i+batch_size].to(device),
                    'desc': desc_conds_tensor[i:i+batch_size].to(device)
                }
                out = model_v7a(b_dict).view(-1).cpu().numpy()
                s_preds.extend(out)
        v7a_preds[s] = np.array(s_preds, dtype=np.float32)
        results_dict[f'pred_v7a_seed_{s}'] = v7a_preds[s]

    v7a_mat = np.column_stack([v7a_preds[s] for s in SEEDS])
    v7a_ens_mean = np.mean(v7a_mat, axis=1)
    v7a_ens_sigma = np.std(v7a_mat, axis=1, ddof=1)
    results_dict['pred_v7a_ens_mean'] = v7a_ens_mean
    results_dict['sigma_v7a_ens'] = v7a_ens_sigma
    results_dict['ae_v7a_ens'] = np.abs(targets - v7a_ens_mean)

    # 5.3 V7-B Inference (5 seeds)
    print(">>> Running V7-B Inference (5 Seeds)...")
    v7b_preds = {s: [] for s in SEEDS}
    for s in SEEDS:
        ckpt_path = v7b_ckpt_dir / f"best_seed_{s}.pth"
        model_v7b = IL_GAT_v7({"emb_dim": 300, "hidden_dim": 512, "desc_dropout_p": 0.0}).to(device)
        raw_ckpt = torch.load(ckpt_path, map_location=device)
        st = raw_ckpt['model_state_dict'] if 'model_state_dict' in raw_ckpt else raw_ckpt
        model_v7b.load_state_dict(st)
        model_v7b.eval()

        s_preds = []
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                b_dict = {
                    'cat': Batch.from_data_list(v7_cats[i:i+batch_size]).to(device),
                    'ani': Batch.from_data_list(v7_anis[i:i+batch_size]).to(device),
                    'ref': Batch.from_data_list(v7_refs[i:i+batch_size]).to(device),
                    'state': state_conds_tensor[i:i+batch_size].to(device),
                    'desc': desc_conds_tensor[i:i+batch_size].to(device)
                }
                out = model_v7b(b_dict).view(-1).cpu().numpy()
                s_preds.extend(out)
        v7b_preds[s] = np.array(s_preds, dtype=np.float32)
        results_dict[f'pred_v7b_seed_{s}'] = v7b_preds[s]

    v7b_mat = np.column_stack([v7b_preds[s] for s in SEEDS])
    v7b_ens_mean = np.mean(v7b_mat, axis=1)
    v7b_ens_sigma = np.std(v7b_mat, axis=1, ddof=1)
    results_dict['pred_v7b_ens_mean'] = v7b_ens_mean
    results_dict['sigma_v7b_ens'] = v7b_ens_sigma
    results_dict['ae_v7b_ens'] = np.abs(targets - v7b_ens_mean)

    df_preds = pd.DataFrame(results_dict)
    pred_csv = ROOT / "v7_shadow_experiment" / "phase1_hfo_zeroshot_predictions.csv"
    df_preds.to_csv(pred_csv, index=False)
    print(f"\n[Saved] Sample-level predictions saved to: {pred_csv}")

    # 6. Statistical Metrics Computation
    def get_eval_row(name, y_true, y_pred, y_sigma=None):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 and np.var(y_true) > 1e-12 else 0.0
        row = {
            'subset': name,
            'N': len(y_true),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
        }
        if y_sigma is not None:
            ae = np.abs(y_true - y_pred)
            sp_rho, sp_p = spearmanr(ae, y_sigma)
            pe_r, pe_p = pearsonr(ae, y_sigma)
            row['mean_sigma'] = float(np.mean(y_sigma))
            row['spearman_rho_error_sigma'] = float(sp_rho)
            row['spearman_p'] = float(sp_p)
            row['pearson_r'] = float(pe_r)
        return row

    summary_rows = []
    # Overall sample-level
    summary_rows.append(get_eval_row('ALL_V6_Ensemble', targets, v6_ens_mean, v6_ens_sigma))
    summary_rows.append(get_eval_row('ALL_V7A_Ensemble', targets, v7a_ens_mean, v7a_ens_sigma))
    summary_rows.append(get_eval_row('ALL_V7B_Ensemble', targets, v7b_ens_mean, v7b_ens_sigma))

    # Species-level
    unique_species = sorted(list(set(species_list)))
    for sp in unique_species:
        mask = (species_arr == sp)
        sub_t = targets[mask]
        summary_rows.append(get_eval_row(f'{sp}_V6_Ensemble', sub_t, v6_ens_mean[mask], v6_ens_sigma[mask]))
        summary_rows.append(get_eval_row(f'{sp}_V7A_Ensemble', sub_t, v7a_ens_mean[mask], v7a_ens_sigma[mask]))
        summary_rows.append(get_eval_row(f'{sp}_V7B_Ensemble', sub_t, v7b_ens_mean[mask], v7b_ens_sigma[mask]))

    df_summary = pd.DataFrame(summary_rows)
    sum_csv = ROOT / "v7_shadow_experiment" / "phase1_hfo_zeroshot_summary.csv"
    df_summary.to_csv(sum_csv, index=False)
    print(f"[Saved] Summary evaluation metrics saved to: {sum_csv}")

    # Print Formatted Comparison Table
    print("\n" + "=" * 95)
    print("  TABLE: HFO/HCFO ZERO-SHOT BENCHMARK: V6 vs V7-A vs V7-B (5-Member Ensembles)")
    print("=" * 95)
    disp_cols = ['subset', 'N', 'mae', 'rmse', 'r2', 'mean_sigma', 'spearman_rho_error_sigma']
    print(df_summary[disp_cols].to_string(index=False))

if __name__ == '__main__':
    main()
