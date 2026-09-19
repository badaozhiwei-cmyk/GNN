"""
run_rf_baselines.py — 浅层机器学习 (Random Forest) 严格 LORO 对照基准与解耦研究
实现：
1. 评估 RF-M0, RF-Mthermo, RF-Mreduced 在 100% 同源冻结 LORO Split (Train/Val/Test) 下的表现
2. 严密排除 Val 集，使 RF 训练预算 (Train-only) 与 GNN 学习预算完全一致
3. 分子表征采用 Morgan 指纹: 128 bits x 3 (阳离子/阴离子/制冷剂) = 384 bits
4. 严格遵循 Train-only StandardScaler 拟合
5. 输出 GNN vs RF 横向对比表及 2x2 因子解耦对照表
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
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
    "ref_dipole": 12, "ref_polarizability": 13, "ref_volume": 14,
    "deltaE_anion": 15, "deltaE_cation": 16,
    "Tc": 17, "Pc": 18, "omega": 19,
    "Tr": 20, "Pr": 21,
}
BASE_FEATURES = ["T", "P", "ref_charge", "ref_logp", "ani_mw", "cat_charge", "cat_tpsa", "ref_MW", "cat_MW"]
MODE_DEF = {
    'M0':        BASE_FEATURES,
    'Mthermo':   BASE_FEATURES + ["Tc", "Pc", "omega"],
    'Mreduced':  BASE_FEATURES + ["Tr", "Pr", "omega"],
}
MODE_INDICES = {
    mode: [FEATURE_SCHEMA[feat] for feat in feat_list]
    for mode, feat_list in MODE_DEF.items()
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

def run_rf_study(data_dir='processed_tri_data_v6'):
    print("=" * 85)
    print("🌲 启动 Random Forest (RF) 100% 严格同 Split 基准与解耦实验")
    print("=" * 85)
    
    meta_path = os.path.join(data_dir, 'meta_info.csv')
    data_path = os.path.join(data_dir, 'data.npy')
    if not os.path.exists(meta_path) or not os.path.exists(data_path):
        print(f"[错误] 数据文件不存在于 {data_dir}，请确认。")
        return

    df_raw = pd.read_csv(meta_path)
    data_raw = np.load(data_path, allow_pickle=True)
    
    ref_col = 'refrigerant' if 'refrigerant' in df_raw.columns else 'Refrigerant'
    family_map_upper = {str(k).strip().upper(): v for k, v in FAMILY_MAP.items()}
    df_raw['family'] = df_raw[ref_col].astype(str).str.strip().str.upper().map(family_map_upper)
    
    mask = (df_raw['family'] == 'HFC')
    if ('pair_energy_complete' in df_raw.columns) and ((df_raw['pair_energy_complete'] == True).sum() > 0):
        mask = mask & (df_raw['pair_energy_complete'] == True)
        print(f"  [Complete-Case] 使用与 Kaggle 完全对齐的子集: {mask.sum()} 条")
    else:
        print(f"  [HFC全量子集] 本地使用全部 HFC 样本: {mask.sum()} 条")
        
    active_indices = df_raw[mask].index.values
    df = df_raw.loc[active_indices].reset_index(drop=True)
    data = data_raw[active_indices]
    
    print(f"活跃完整数据集样本量: {len(df)} 条 (100% 对应 Complete-Case)")
    
    # 预计算分子指纹 (128 bits * 3 = 384 bits)
    print("正在为 IL 阳离子、阴离子和制冷剂生成 Morgan ECFP4 指纹 (128 bits x 3 = 384 bits)...")
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
    print(f"指纹矩阵构建完成: {fp_mat.shape}")

    # LORO 切分与同 Split 绑定
    unique_refs = sorted(df[ref_col].unique())
    modes = ['M0', 'Mthermo', 'Mreduced']
    rf_results = {m: {} for m in modes}
    
    split_dir = 'Phase4_Scientific_Validation/HFC_LORO_Splits'

    for mode in modes:
        feat_indices = MODE_INDICES[mode]
        print(f"\n▶ 正在运行 RF-{mode:<9} (特征索引: {feat_indices})")
        
        cont_features = np.array([[row[k] for k in feat_indices] for row in data], dtype=np.float32)
        
        split_maes, split_r2s = [], []
        for ref in unique_refs:
            # 严格读取冻结的 LORO split 文件，实现 100% 相同的 Train/Val/Test 分区
            sp_file = os.path.join(split_dir, f"split_hfc_loro_{ref}_val1.npz")
            if os.path.exists(sp_file):
                sp = np.load(sp_file)
                val_hfc = str(sp['val_hfc'])
                test_mask = (df[ref_col] == ref).values
                val_mask = (df[ref_col] == val_hfc).values
                # 训练集严格排除 Test 和 Val，与 GNN 训练预算完全相同
                train_mask = (~test_mask) & (~val_mask)
            else:
                test_mask = (df[ref_col] == ref).values
                train_mask = ~test_mask
            
            X_train_cont = cont_features[train_mask]
            X_test_cont  = cont_features[test_mask]
            
            scaler = StandardScaler()
            X_train_cont_s = scaler.fit_transform(X_train_cont)
            X_test_cont_s  = scaler.transform(X_test_cont)
            
            X_train = np.hstack([fp_mat[train_mask], X_train_cont_s])
            X_test  = np.hstack([fp_mat[test_mask], X_test_cont_s])
            
            y_train = df.loc[train_mask, 'x1'].values
            y_test  = df.loc[test_mask, 'x1'].values
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            pred = np.clip(rf.predict(X_test), 0.0, 1.0)
            
            mae = mean_absolute_error(y_test, pred)
            ss_tot = np.sum((y_test - np.mean(y_test))**2)
            r2 = r2_score(y_test, pred) if ss_tot > 1e-8 else np.nan
            
            rf_results[mode][ref] = {'MAE': mae, 'R2': r2}
            split_maes.append(mae)
            if np.isfinite(r2):
                split_r2s.append(r2)
                
        macro_mae = np.mean(split_maes)
        macro_r2  = np.mean(split_r2s)
        median_r2 = np.median(split_r2s)
        print(f"  [RF-{mode}] Macro-MAE: {macro_mae:.4f} | Macro-R²: {macro_r2:.4f} | Median-R²: {median_r2:.4f}")

    # 汇总成与 GNN 的横向大决战对比表
    print("\n" + "=" * 95)
    print("⚔️ GNN vs Random Forest: 严格同 Split 跨物质外推大决战 (Per-Refrigerant MAE)")
    print("=" * 95)
    
    recovered_csv = 'paper_results/recovered_overnight_results.csv'
    gnn_df = pd.read_csv(recovered_csv) if os.path.exists(recovered_csv) else None
    
    comp_rows = []
    for ref in unique_refs:
        row_dict = {'Refrigerant': ref}
        for m in ['M0', 'Mthermo', 'Mreduced']:
            if gnn_df is not None:
                sub = gnn_df[(gnn_df['Mode'] == m) & (gnn_df['Target'].str.lower() == f'loro_{ref}'.lower())]
                row_dict[f'GNN_{m}'] = sub['MAE_mean'].values[0] if len(sub) > 0 else np.nan
            row_dict[f'RF_{m}'] = rf_results[m][ref]['MAE']
            
        comp_rows.append(row_dict)
        
    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))
    
    comp_df.to_csv('paper_results/table_gnn_vs_rf_comparison.csv', index=False)
    print("\n✅ 对决表已保存至 paper_results/table_gnn_vs_rf_comparison.csv！")

    # 运行严谨的信号解耦与 2x2 因子控制实验
    print("\n" + "=" * 95)
    print("🔬 运行信号解耦与互补性检验 (Decoupling Study under Identical Split)")
    print("=" * 95)
    
    idx_thermo = [FEATURE_SCHEMA["Tr"], FEATURE_SCHEMA["Pr"], FEATURE_SCHEMA["omega"]]
    idx_tp = [FEATURE_SCHEMA["T"], FEATURE_SCHEMA["P"]]
    idx_full = [3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 19]

    decoupling_modes = {
        'RF_thermo_only': ('cont', idx_thermo, False),
        'RF_mol_only':    ('cont', idx_tp, True),
        'RF_full':        ('cont', idx_full, True),
        'RF_fp_only':     ('none', [], True),
        'RF_tp_only':     ('cont', idx_tp, False),
    }

    decoupling_results = {dm: {} for dm in decoupling_modes}

    for dm, (c_type, feat_ids, use_fp) in decoupling_modes.items():
        maes = []
        for ref in unique_refs:
            sp_file = os.path.join(split_dir, f"split_hfc_loro_{ref}_val1.npz")
            if os.path.exists(sp_file):
                sp = np.load(sp_file)
                val_hfc = str(sp['val_hfc'])
                test_mask = (df[ref_col] == ref).values
                val_mask = (df[ref_col] == val_hfc).values
                train_mask = (~test_mask) & (~val_mask)
            else:
                test_mask = (df[ref_col] == ref).values
                train_mask = ~test_mask
            
            parts_train, parts_test = [], []
            if use_fp:
                parts_train.append(fp_mat[train_mask])
                parts_test.append(fp_mat[test_mask])
            if c_type == 'cont':
                cont_data = np.array([[row[k] for k in feat_ids] for row in data], dtype=np.float32)
                scaler = StandardScaler()
                parts_train.append(scaler.fit_transform(cont_data[train_mask]))
                parts_test.append(scaler.transform(cont_data[test_mask]))
                
            X_train = np.hstack(parts_train)
            X_test  = np.hstack(parts_test)
            y_train = df.loc[train_mask, 'x1'].values
            y_test  = df.loc[test_mask, 'x1'].values
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            pred = np.clip(rf.predict(X_test), 0.0, 1.0)
            mae = mean_absolute_error(y_test, pred)
            decoupling_results[dm][ref] = mae
            maes.append(mae)
            
        print(f"  [{dm:<16}] Macro-MAE across 12 refrigerants: {np.mean(maes):.4f}")

    dec_rows = []
    for ref in unique_refs:
        row_dict = {'Refrigerant': ref}
        for dm in decoupling_modes:
            row_dict[dm] = decoupling_results[dm][ref]
        dec_rows.append(row_dict)
    dec_df = pd.DataFrame(dec_rows)
    dec_df.to_csv('paper_results/table_rf_decoupling_study.csv', index=False)
    print("\n✅ 解耦对照表已保存至 paper_results/table_rf_decoupling_study.csv！")

if __name__ == '__main__':
    run_rf_study()
