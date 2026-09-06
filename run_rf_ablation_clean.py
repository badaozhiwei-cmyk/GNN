"""
run_rf_ablation_clean.py — 纯粹的 RF 信号解耦实验 (Thermodynamics vs Molecular Topology)
实现：
1. RF_thermo_only: 仅输入 [Tr, Pr, omega] (3 维)
2. RF_mol_only:    仅输入 [Morgan Fingerprints (384维) + T + P]
3. RF_full:        输入 [Morgan Fingerprints + Mreduced 完整特征]
在 12 个制冷剂 LORO 上评估，严格揭示热力学对比态信号的独立贡献率。
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem

FEATURE_SCHEMA = {
    "T": 3, "P": 4, 
    "ref_charge": 5, "ref_logp": 6, "ani_mw": 7, "cat_charge": 8, "cat_tpsa": 9, 
    "ref_MW": 10, "cat_MW": 11,
    "Tc": 17, "Pc": 18, "omega": 19,
    "Tr": 20, "Pr": 21,
}

FAMILY_MAP = {
    'R32': 'HFC', 'R152A': 'HFC', 'R134A': 'HFC', 'R125': 'HFC', 'R143A': 'HFC',
    'R23': 'HFC', 'R41': 'HFC', 'R161': 'HFC', 'R134': 'HFC', 'R227EA': 'HFC',
    'R236FA': 'HFC', 'R245FA': 'HFC', 'R365MFC': 'HFC', 'R236EA': 'HFC',
}

def get_morgan_fp(smi, n_bits=128):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    return np.array([int(b) for b in fp.ToBitString()], dtype=np.float32)

def run_decoupling_study(data_dir='processed_tri_data_v6', complete_case_only=True):
    print("=" * 85)
    print(f"🔬 启动 RF 纯粹解耦基准：Thermodynamics-Only vs Molecular-Only vs Combined (CompleteCase={complete_case_only})")
    print("=" * 85)
    
    meta_path = os.path.join(data_dir, 'meta_info.csv')
    data_path = os.path.join(data_dir, 'data.npy')
    df_raw = pd.read_csv(meta_path)
    data_raw = np.load(data_path, allow_pickle=True)
    
    ref_col = 'refrigerant' if 'refrigerant' in df_raw.columns else 'Refrigerant'
    family_map_upper = {str(k).strip().upper(): v for k, v in FAMILY_MAP.items()}
    df_raw['family'] = df_raw[ref_col].astype(str).str.strip().str.upper().map(family_map_upper)
    mask = (df_raw['family'] == 'HFC')
    if complete_case_only and 'pair_energy_complete' in df_raw.columns:
        mask = mask & (df_raw['pair_energy_complete'] == True)
    
    active_indices = df_raw[mask].index.values
    df = df_raw.loc[active_indices].reset_index(drop=True)
    data = data_raw[active_indices]
    print(f"  [RF Universe] Samples in active evaluation: {len(df)}")
    
    from prepare_tri_graph_data_v6 import lookup_smiles
    fp_cache = {}
    def get_cached_fp(name):
        if name not in fp_cache:
            smi = lookup_smiles(name)
            fp_cache[name] = get_morgan_fp(smi, n_bits=128)
        return fp_cache[name]
        
    fp_mat = []
    for idx, row in df.iterrows():
        c_fp = get_cached_fp(row['IL cation'])
        a_fp = get_cached_fp(row['IL anion'])
        r_fp = get_cached_fp(row[ref_col])
        fp_mat.append(np.concatenate([c_fp, a_fp, r_fp]))
    fp_mat = np.array(fp_mat, dtype=np.float32)

    # 1. 特征集定义
    # Thermo-only: Tr, Pr, omega (索引 20, 21, 19)
    idx_thermo = [FEATURE_SCHEMA["Tr"], FEATURE_SCHEMA["Pr"], FEATURE_SCHEMA["omega"]]
    X_thermo_raw = np.array([[row[k] for k in idx_thermo] for row in data], dtype=np.float32)
    
    # Mol-only: 基础 T, P (索引 3, 4)
    idx_tp = [FEATURE_SCHEMA["T"], FEATURE_SCHEMA["P"]]
    X_tp_raw = np.array([[row[k] for k in idx_tp] for row in data], dtype=np.float32)
    
    # Full (Mreduced): T, P, 7 basics, Tr, Pr, omega (12 维)
    idx_full = [3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 19]
    X_full_cont_raw = np.array([[row[k] for k in idx_full] for row in data], dtype=np.float32)

    unique_refs = df[ref_col].unique()
    models = ['RF_thermo_only', 'RF_mol_only', 'RF_full']
    results = {m: {} for m in models}
    
    for ref in unique_refs:
        test_mask = (df[ref_col] == ref).values
        train_mask = ~test_mask
        y_train = df.loc[train_mask, 'x1'].values
        y_test  = df.loc[test_mask, 'x1'].values
        
        # --- 1. RF_thermo_only ---
        sc_th = StandardScaler()
        X_tr_th = sc_th.fit_transform(X_thermo_raw[train_mask])
        X_te_th = sc_th.transform(X_thermo_raw[test_mask])
        rf_th = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_th.fit(X_tr_th, y_train)
        pred_th = np.clip(rf_th.predict(X_te_th), 0.0, 1.0)
        results['RF_thermo_only'][ref] = mean_absolute_error(y_test, pred_th)
        
        # --- 2. RF_mol_only ---
        sc_tp = StandardScaler()
        X_tr_tp = sc_tp.fit_transform(X_tp_raw[train_mask])
        X_te_tp = sc_tp.transform(X_tp_raw[test_mask])
        X_tr_mol = np.hstack([fp_mat[train_mask], X_tr_tp])
        X_te_mol = np.hstack([fp_mat[test_mask], X_te_tp])
        rf_mol = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_mol.fit(X_tr_mol, y_train)
        pred_mol = np.clip(rf_mol.predict(X_te_mol), 0.0, 1.0)
        results['RF_mol_only'][ref] = mean_absolute_error(y_test, pred_mol)
        
        # --- 3. RF_full ---
        sc_full = StandardScaler()
        X_tr_fc = sc_full.fit_transform(X_full_cont_raw[train_mask])
        X_te_fc = sc_full.transform(X_full_cont_raw[test_mask])
        X_tr_full = np.hstack([fp_mat[train_mask], X_tr_fc])
        X_te_full = np.hstack([fp_mat[test_mask], X_te_fc])
        rf_full = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_full.fit(X_tr_full, y_train)
        pred_full = np.clip(rf_full.predict(X_te_full), 0.0, 1.0)
        results['RF_full'][ref] = mean_absolute_error(y_test, pred_full)

    # 打印逐物质对比大表
    rows = []
    for ref in unique_refs:
        rows.append({
            'Refrigerant': ref,
            'RF_thermo_only (3维)': results['RF_thermo_only'][ref],
            'RF_mol_only (FP+T,P)': results['RF_mol_only'][ref],
            'RF_full (FP+Mred)':    results['RF_full'][ref],
        })
    res_df = pd.DataFrame(rows)
    
    macro_th   = res_df['RF_thermo_only (3维)'].mean()
    macro_mol  = res_df['RF_mol_only (FP+T,P)'].mean()
    macro_full = res_df['RF_full (FP+Mred)'].mean()
    
    print("\n" + "=" * 95)
    print("📊 解耦大表：仅凭对比态参数 vs 仅凭分子拓扑 vs 二者结合 (LORO MAE)")
    print("=" * 95)
    print(res_df.to_string(index=False))
    print("-" * 95)
    print(f"  Macro-MAE [RF_thermo_only (仅3维对比态)]: {macro_th:.4f}")
    print(f"  Macro-MAE [RF_mol_only    (仅分子指纹+TP)]: {macro_mol:.4f}")
    print(f"  Macro-MAE [RF_full        (指纹+对比态)]:   {macro_full:.4f}")
    print("=" * 95)
    
    os.makedirs('paper_results', exist_ok=True)
    res_df.to_csv('paper_results/table_rf_decoupling_study.csv', index=False)
    print("✅ 纯粹解耦实验数据已落盘至 paper_results/table_rf_decoupling_study.csv！")

if __name__ == '__main__':
    run_decoupling_study()
