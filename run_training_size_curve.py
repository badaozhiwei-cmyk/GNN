"""
run_training_size_curve.py — 训练集规模学习曲线 (Coverage-limited vs Representation-limited 判决实验)
实现：
1. 固定代表性测试目标：R245fa (边界分子), R134 (偶极对称异构体), R134a (强极性异构体)
2. 训练集数据比例梯队：25%, 50%, 75%, 100% (分层保比例抽取)
3. 分别计算各梯队在 M0 与 Mreduced 下的外推 MAE
4. 绘制并输出学习曲线判定结果 (Data-limited vs Representation-limited)
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import argparse
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

def get_morgan_fp(smi, n_bits=128):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    return np.array([int(b) for b in fp.ToBitString()], dtype=np.float32)

def run_curve(target_refs=['R245fa', 'R134', 'R134a'], fractions=[0.25, 0.50, 0.75, 1.00], data_dir='processed_tri_data_v6'):
    print("=" * 85)
    print("📈 启动训练集规模学习曲线 (Training-size Curve) 判定实验")
    print(f"  目标测试分子: {target_refs}")
    print(f"  训练集梯队:   {[f'{int(f*100)}%' for f in fractions]}")
    print("=" * 85)
    
    meta_path = os.path.join(data_dir, 'meta_info.csv')
    data_path = os.path.join(data_dir, 'data.npy')
    df_raw = pd.read_csv(meta_path)
    data_raw = np.load(data_path, allow_pickle=True)
    
    ref_col = 'refrigerant' if 'refrigerant' in df_raw.columns else 'Refrigerant'
    family_map_upper = {str(k).strip().upper(): v for k, v in FAMILY_MAP.items()}
    df_raw['family'] = df_raw[ref_col].astype(str).str.strip().str.upper().map(family_map_upper)
    mask = (df_raw['family'] == 'HFC')
    
    active_indices = df_raw[mask].index.values
    df = df_raw.loc[active_indices].reset_index(drop=True)
    data = data_raw[active_indices]
    
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

    modes = ['M0', 'Mreduced']
    all_records = []
    
    for target in target_refs:
        print(f"\n🎯 目标测试分子: {target}")
        test_mask = (df[ref_col].astype(str).str.upper() == target.upper()).values
        train_indices_full = np.where(~test_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        y_test = df.loc[test_mask, 'x1'].values
        
        for mode in modes:
            feat_indices = MODE_INDICES[mode]
            cont_features = np.array([[row[k] for k in feat_indices] for row in data], dtype=np.float32)
            
            for frac in fractions:
                # 随机采样训练子集 (3次不同随机种子以消除抽样方差)
                rep_maes = []
                for seed in [42, 43, 44]:
                    rng = np.random.RandomState(seed)
                    n_sample = max(1, int(len(train_indices_full) * frac))
                    sampled_train_idx = rng.choice(train_indices_full, size=n_sample, replace=False)
                    
                    scaler = StandardScaler()
                    X_tr_cont_s = scaler.fit_transform(cont_features[sampled_train_idx])
                    X_te_cont_s = scaler.transform(cont_features[test_indices])
                    
                    X_train = np.hstack([fp_mat[sampled_train_idx], X_tr_cont_s])
                    X_test  = np.hstack([fp_mat[test_indices], X_te_cont_s])
                    y_train = df.loc[sampled_train_idx, 'x1'].values
                    
                    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
                    rf.fit(X_train, y_train)
                    pred = np.clip(rf.predict(X_test), 0.0, 1.0)
                    rep_maes.append(mean_absolute_error(y_test, pred))
                    
                mean_mae = np.mean(rep_maes)
                std_mae  = np.std(rep_maes)
                
                all_records.append({
                    'Target': target,
                    'Mode': mode,
                    'Train_Fraction': f"{int(frac*100)}%",
                    'N_train': n_sample,
                    'MAE_mean': mean_mae,
                    'MAE_std': std_mae
                })
                print(f"  [{mode:<8}] Train: {int(frac*100):>3}% (N={n_sample:>4}) -> Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
                
    curve_df = pd.DataFrame(all_records)
    print("\n" + "=" * 90)
    print("📊 判定分析：学习曲线斜率与机理诊断")
    print("=" * 90)
    
    # 计算从 25% 到 100% 的误差变化率 (Delta MAE)
    verdict_rows = []
    for target in target_refs:
        for mode in modes:
            sub = curve_df[(curve_df['Target'] == target) & (curve_df['Mode'] == mode)]
            mae_25 = sub[sub['Train_Fraction'] == '25%']['MAE_mean'].values[0]
            mae_100 = sub[sub['Train_Fraction'] == '100%']['MAE_mean'].values[0]
            pct_change = (mae_100 - mae_25) / mae_25 * 100.0
            
            if pct_change < -15.0:
                verdict = "Coverage-Limited (数据量增加显著缓解误差)"
            elif abs(pct_change) <= 15.0:
                verdict = "Representation-Limited (增加数据无改善，受制于表征/物理边界)"
            else:
                verdict = "Negative scaling"
                
            verdict_rows.append({
                'Target': target,
                'Mode': mode,
                'MAE_25%': mae_25,
                'MAE_100%': mae_100,
                'Relative_Change': f"{pct_change:+.1f}%",
                'Scientific Verdict': verdict
            })
            
    verdict_df = pd.DataFrame(verdict_rows)
    print(verdict_df.to_string(index=False))
    
    os.makedirs('paper_results', exist_ok=True)
    curve_df.to_csv('paper_results/table_training_size_curve.csv', index=False)
    verdict_df.to_csv('paper_results/table_curve_scientific_verdict.csv', index=False)
    print("\n✅ 学习曲线与科学判决结果已保存至 paper_results/ 目录！")

if __name__ == '__main__':
    run_curve()
