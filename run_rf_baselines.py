"""
run_rf_baselines.py — 浅层机器学习 (Random Forest) 严格 LORO 对照基准
实现：
1. 评估 RF-M0, RF-Mstd, RF-Mthermo, RF-Mreduced 在同一 LORO 协议下的表现
2. 分子表征采用 ECFP4 (Morgan Fingerprints, 256 bits x 3 = 768 bits)
3. 严格遵循 Train-only StandardScaler 拟合
4. 横向对比 GNN vs RF 跨物质外推能力
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
    'Mstd':      BASE_FEATURES + ["T", "P", "omega"],
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

def get_morgan_fp(smi, n_bits=256):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((1,), dtype=np.int8)
    # 快速转为 numpy
    return np.array([int(b) for b in fp.ToBitString()], dtype=np.float32)

def run_rf_study(data_dir='processed_tri_data_v6'):
    print("=" * 85)
    print("🌲 启动 Random Forest (RF) 严格 LORO 基准对决实验")
    print("=" * 85)
    
    meta_path = os.path.join(data_dir, 'meta_info.csv')
    data_path = os.path.join(data_dir, 'data.npy')
    if not os.path.exists(meta_path) or not os.path.exists(data_path):
        print(f"[错误] 数据文件不存在于 {data_dir}，请确认。")
        return

    df_raw = pd.read_csv(meta_path)
    data_raw = np.load(data_path, allow_pickle=True)
    
    # 过滤 HFC
    ref_col = 'refrigerant' if 'refrigerant' in df_raw.columns else 'Refrigerant'
    family_map_upper = {str(k).strip().upper(): v for k, v in FAMILY_MAP.items()}
    df_raw['family'] = df_raw[ref_col].astype(str).str.strip().str.upper().map(family_map_upper)
    
    mask = (df_raw['family'] == 'HFC')
    has_complete = ('pair_energy_complete' in df_raw.columns) and ((df_raw['pair_energy_complete'] == True).sum() > 0)
    if has_complete:
        mask = mask & (df_raw['pair_energy_complete'] == True)
        print(f"  [Complete-Case] 使用与 Kaggle 完全对齐的子集: {mask.sum()} 条")
    else:
        print(f"  [HFC全量子集] 本地使用全部 HFC 样本: {mask.sum()} 条")
        
    active_indices = df_raw[mask].index.values
    df = df_raw.loc[active_indices].reset_index(drop=True)
    data = data_raw[active_indices]
    
    print(f"活跃完整数据集样本量: {len(df)} 条 (100% 对应 Complete-Case)")
    
    # 预计算分子指纹 (加速训练)
    print("正在为 IL 阳离子、阴离子和制冷剂生成 Morgan ECFP4 指纹...")
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
        fp_mat.append(np.concatenate([c_fp, a_fp, r_fp])) # 128 * 3 = 384 维
    fp_mat = np.array(fp_mat, dtype=np.float32)
    print(f"指纹矩阵构建完成: {fp_mat.shape}")

    # LORO 切分
    unique_refs = df[ref_col].unique()
    modes = ['M0', 'Mstd', 'Mthermo', 'Mreduced']
    rf_results = {m: {} for m in modes}
    
    for mode in modes:
        feat_indices = MODE_INDICES[mode]
        print(f"\n▶ 正在运行 RF-{mode:<9} (特征索引: {feat_indices})")
        
        # 提取当前模式的连续特征
        cont_features = np.array([[row[k] for k in feat_indices] for row in data], dtype=np.float32)
        
        split_maes, split_r2s = [], []
        for ref in unique_refs:
            test_mask = (df[ref_col] == ref).values
            train_mask = ~test_mask
            
            X_train_cont = cont_features[train_mask]
            X_test_cont  = cont_features[test_mask]
            
            # Scaler 严格只拟合训练集
            scaler = StandardScaler()
            X_train_cont_s = scaler.fit_transform(X_train_cont)
            X_test_cont_s  = scaler.transform(X_test_cont)
            
            # 拼接指纹特征与连续特征
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
    print("⚔️ GNN vs Random Forest: 跨物质泛化终极大决战 (Per-Refrigerant MAE)")
    print("=" * 95)
    
    # 读取昨晚抢救出的 GNN 结果
    recovered_csv = 'paper_results/recovered_overnight_results.csv'
    gnn_df = pd.read_csv(recovered_csv) if os.path.exists(recovered_csv) else None
    
    comp_rows = []
    for ref in unique_refs:
        row_dict = {'Refrigerant': ref}
        for m in ['M0', 'Mthermo', 'Mreduced']:
            if gnn_df is not None:
                sub = gnn_df[(gnn_df['Mode'] == m) & (gnn_df['Target'].str.contains(ref))]
                row_dict[f'GNN_{m}'] = sub['MAE_mean'].values[0] if len(sub) > 0 else np.nan
            row_dict[f'RF_{m}'] = rf_results[m][ref]['MAE']
            
        row_dict['RF_Mstd'] = rf_results['Mstd'][ref]['MAE']
        comp_rows.append(row_dict)
        
    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))
    
    comp_df.to_csv('paper_results/table_gnn_vs_rf_comparison.csv', index=False)
    print("\n✅ 对决表已保存至 paper_results/table_gnn_vs_rf_comparison.csv！")

if __name__ == '__main__':
    run_rf_study()
