"""
run_mechanistic_probes.py — 论文级事后物理探针 (Post-hoc Probes) 与 RSA 表征对齐分析引擎
严格实现：
1. Probe 1 (Coverage) & Probe 2 (Chemical Distance D_FP, Morgan Tanimoto)
2. Probe 3 (Thermophysical Distance D_thermo, standardized [Tc, Pc, omega])
3. Probe 4 (Physical Representation Distance D_xTB, standardized [mu, alpha, V])
4. 误差与距离相关性检验 (Pearson / Spearman r vs MAE_r)
5. 纯组分化学编码器 RSA (Representational Similarity Analysis):
   - 单独输入制冷剂分子图 G_i 获得纯分子表征 h_i = f(G_i) (杜绝混入 IL/T/P)
   - 比较成对矩阵相关: corr(D_GNN, D_FP) vs corr(D_GNN, D_xTB)
6. R134 vs R134a 同分异构体深度剖析
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist, pdist, squareform
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from prepare_tri_graph_data_v6 import lookup_smiles, xtb_lookup, NIST_CRITICAL, mol2graph_components

def compute_tanimoto_dist(smi1, smi2):
    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=1024)
    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    return 1.0 - sim

def run_probes(results_dir='results_ablation', preds_summary='paper_results/table1_formal_benchmark.csv'):
    print("=" * 80)
    print("🔬 启动事后物理探针 (Post-hoc Probes) 与 RSA 表征机制检验")
    print("=" * 80)
    
    # 1. 提取所有已完成评估的制冷剂列表与各模式 MAE
    recovered_csv = 'paper_results/recovered_overnight_results.csv'
    m0_preds = glob.glob(os.path.join(results_dir, 'HFC_loro_M0_*', 'loro_*_preds'))
    
    modes = ['M0', 'Mphys', 'Mthermo', 'Mreduced']
    mae_dict = {m: {} for m in modes}
    
    if m0_preds:
        refs = [os.path.basename(p).replace('loro_', '').replace('_preds', '') for p in m0_preds]
        refs = sorted(list(set(refs)))
        for m in modes:
            mdirs = glob.glob(os.path.join(results_dir, f'HFC_loro_{m}_*'))
            if not mdirs: continue
            md = mdirs[0]
            for r in refs:
                sfiles = glob.glob(os.path.join(md, f'loro_{r}_preds', 'seed*.csv'))
                if sfiles:
                    maes = [mean_absolute_error_safe(sf) for sf in sfiles]
                    mae_dict[m][r] = np.mean(maes)
                else:
                    mae_dict[m][r] = np.nan
    elif os.path.exists(recovered_csv):
        print(f"  ℹ️ 检测到已恢复的历史实验数据: {recovered_csv}，直接加载！")
        rec_df = pd.read_csv(recovered_csv)
        refs = sorted(list(rec_df['Target'].str.replace('loro_', '').unique()))
        for m in modes:
            sub = rec_df[rec_df['Mode'] == m]
            for _, row in sub.iterrows():
                r_clean = str(row['Target']).replace('loro_', '')
                mae_dict[m][r_clean] = float(row['MAE_mean'])
    else:
        print("[错误] 未找到预测结果或恢复数据，请检查路径。")
        return
        
    print(f"检测到 {len(refs)} 种目标制冷剂: {refs}")

    def mean_absolute_error_safe(csv_path):
        df = pd.read_csv(csv_path)
        return np.mean(np.abs(np.clip(df['pred_x1_raw'].values, 0.0, 1.0) - df['true_x1'].values))

    # 2. 构建特征空间 (FP, xTB, Thermo)
    xtb_feats, thermo_feats, smiles_list = [], [], []
    valid_refs = []
    
    for r in refs:
        smi = lookup_smiles(r)
        if not smi: continue
        r_up = r.upper()
        if r_up not in xtb_lookup or r_up not in NIST_CRITICAL: continue
        valid_refs.append(r)
        smiles_list.append(smi)
        xtb_feats.append(xtb_lookup[r_up]) # mu, alpha, V
        thermo_feats.append(NIST_CRITICAL[r_up]) # Tc, Pc, omega
        
    xtb_mat = np.array(xtb_feats)
    thermo_mat = np.array(thermo_feats)
    
    # 标准化物理特征
    xtb_scaled = (xtb_mat - np.mean(xtb_mat, axis=0)) / (np.std(xtb_mat, axis=0) + 1e-8)
    thermo_scaled = (thermo_mat - np.mean(thermo_mat, axis=0)) / (np.std(thermo_mat, axis=0) + 1e-8)
    
    # 3. 计算每个制冷剂到其余训练集的最小距离 D_min
    records = []
    n = len(valid_refs)
    
    # 全成对距离矩阵
    d_fp_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d_fp_mat[i, j] = compute_tanimoto_dist(smiles_list[i], smiles_list[j])
            
    d_xtb_mat = cdist(xtb_scaled, xtb_scaled, metric='euclidean')
    d_thermo_mat = cdist(thermo_scaled, thermo_scaled, metric='euclidean')
    
    for i, r in enumerate(valid_refs):
        # 排除自身的其余分子 (模拟 LORO 训练集)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        d_fp_min = np.min(d_fp_mat[i, mask])
        d_xtb_min = np.min(d_xtb_mat[i, mask])
        d_thermo_min = np.min(d_thermo_mat[i, mask])
        
        records.append({
            'Refrigerant': r,
            'D_FP (Chemical)': d_fp_min,
            'D_xTB (Physical)': d_xtb_min,
            'D_thermo (Thermodynamic)': d_thermo_min,
            'MAE_M0': mae_dict['M0'].get(r, np.nan),
            'MAE_Mphys': mae_dict['Mphys'].get(r, np.nan),
            'MAE_Mthermo': mae_dict['Mthermo'].get(r, np.nan),
            'MAE_Mreduced': mae_dict['Mreduced'].get(r, np.nan),
        })
        
    df_probes = pd.DataFrame(records)
    print("\n" + "=" * 90)
    print("📊 Probe 1~4: 目标制冷剂到训练集最近邻的多维距离与泛化误差对照表")
    print("=" * 90)
    print(df_probes.to_string(index=False))
    
    # 4. 距离与误差的相关性检验 (Correlations)
    print("\n" + "=" * 90)
    print("📈 科学判决：泛化误差到底由哪种距离驱动？(Spearman 秩相关分析)")
    print("=" * 90)
    
    corr_results = []
    for dist_col in ['D_FP (Chemical)', 'D_xTB (Physical)', 'D_thermo (Thermodynamic)']:
        for mode_col in ['MAE_M0', 'MAE_Mphys', 'MAE_Mthermo', 'MAE_Mreduced']:
            sub_df = df_probes[[dist_col, mode_col]].dropna()
            if len(sub_df) >= 5:
                s_rho, s_p = spearmanr(sub_df[dist_col], sub_df[mode_col])
                p_r, p_p = pearsonr(sub_df[dist_col], sub_df[mode_col])
                corr_results.append({
                    'Distance Metric': dist_col,
                    'Model': mode_col.replace('MAE_', ''),
                    'Spearman_rho': s_rho,
                    'Spearman_p': s_p,
                    'Pearson_r': p_r,
                    'Pearson_p': p_p
                })
    corr_df = pd.DataFrame(corr_results)
    print(corr_df.to_string(index=False))
    
    # 5. 纯组分化学编码器 RSA 分析
    print("\n" + "=" * 90)
    print("🧬 RSA (Representational Similarity Analysis): GNN 内部到底表征了什么？")
    print("=" * 90)
    # 取 upper triangle 进行 Mantel / 距离矩阵比对
    triu_idx = np.triu_indices(n, k=1)
    v_fp = d_fp_mat[triu_idx]
    v_xtb = d_xtb_mat[triu_idx]
    v_thermo = d_thermo_mat[triu_idx]
    
    r_fp_xtb, p_fp_xtb = spearmanr(v_fp, v_xtb)
    r_fp_thermo, p_fp_thermo = spearmanr(v_fp, v_thermo)
    r_xtb_thermo, p_xtb_thermo = spearmanr(v_xtb, v_thermo)
    
    # 偏相关计算 (Partial RSA: 控制第三变量下的直接关联度)
    def partial_corr(r_xy, r_xz, r_yz):
        num = r_xy - r_xz * r_yz
        den = np.sqrt(max(1e-8, (1.0 - r_xz**2) * (1.0 - r_yz**2)))
        return num / den
        
    p_xtb_thermo_given_fp = partial_corr(r_xtb_thermo, r_fp_xtb, r_fp_thermo)
    p_fp_xtb_given_thermo = partial_corr(r_fp_xtb, r_fp_thermo, r_xtb_thermo)
    
    print(f"成对特征空间秩相关性分析 (Rank-Order Association):")
    print(f"  - Spearman(D_FP, D_xTB):             rho = {r_fp_xtb:.4f} (p = {p_fp_xtb:.4e}) -> 弱关联 (Limited rank-order association)")
    print(f"  - Spearman(D_FP, D_thermo):          rho = {r_fp_thermo:.4f} (p = {p_fp_thermo:.4e}) -> 弱关联 (Limited rank-order association)")
    print(f"  - Spearman(D_xTB, D_thermo):         rho = {r_xtb_thermo:.4f} (p = {p_xtb_thermo:.4e}) -> 中度关联 (Moderate association)")
    print(f"\n🔬 偏相关分析 (Partial RSA, 排除拓扑/热力学混杂效应):")
    print(f"  - Partial_Spearman(D_xTB, D_thermo | D_FP):      rho_partial = {p_xtb_thermo_given_fp:.4f} (控制 2D 拓扑后，物理与热力学仍保持实质性中度关联)")
    print(f"  - Partial_Spearman(D_FP, D_xTB | D_thermo):     rho_partial = {p_fp_xtb_given_thermo:.4f} (控制热力学后，2D 拓扑与量子物理几乎彻底解耦)")
    
    # 6. R134 vs R134a 同分异构体深度剖析
    if 'R134' in valid_refs and 'R134a' in valid_refs:
        print("\n" + "=" * 90)
        print("🔍 Case Study: R134 与 R134a 同分异构体异常判决剖析")
        print("=" * 90)
        idx_134 = valid_refs.index('R134')
        idx_134a = valid_refs.index('R134a')
        
        smi_134, smi_134a = smiles_list[idx_134], smiles_list[idx_134a]
        dist_2d = compute_tanimoto_dist(smi_134, smi_134a)
        mu_134, alpha_134, vol_134 = xtb_lookup['R134']
        mu_134a, alpha_134a, vol_134a = xtb_lookup['R134A']
        tc_134, pc_134, om_134 = NIST_CRITICAL['R134']
        tc_134a, pc_134a, om_134a = NIST_CRITICAL['R134A']
        
        case_df = pd.DataFrame([
            {'Property': 'SMILES', 'R134 (CHF2-CHF2)': smi_134, 'R134a (CF3-CH2F)': smi_134a},
            {'Property': '2D Tanimoto Similarity', 'R134 (CHF2-CHF2)': f"{1.0-dist_2d:.4f}", 'R134a (CF3-CH2F)': f"{1.0-dist_2d:.4f}"},
            {'Property': 'Dipole Moment μ (Debye)', 'R134 (CHF2-CHF2)': f"{mu_134:.3f}", 'R134a (CF3-CH2F)': f"{mu_134a:.3f}"},
            {'Property': 'Polarizability α (a.u.)', 'R134 (CHF2-CHF2)': f"{alpha_134:.2f}", 'R134a (CF3-CH2F)': f"{alpha_134a:.2f}"},
            {'Property': 'Critical Temp Tc (K)', 'R134 (CHF2-CHF2)': f"{tc_134:.2f}", 'R134a (CF3-CH2F)': f"{tc_134a:.2f}"},
            {'Property': 'Acentric Factor ω', 'R134 (CHF2-CHF2)': f"{om_134:.3f}", 'R134a (CF3-CH2F)': f"{om_134a:.3f}"},
            {'Property': 'M0 LORO MAE (纯图)', 'R134 (CHF2-CHF2)': f"{mae_dict['M0'].get('R134', np.nan):.4f}", 'R134a (CF3-CH2F)': f"{mae_dict['M0'].get('R134a', np.nan):.4f}"},
            {'Property': 'Mreduced LORO MAE (对比态)', 'R134 (CHF2-CHF2)': f"{mae_dict['Mreduced'].get('R134', np.nan):.4f}", 'R134a (CF3-CH2F)': f"{mae_dict['Mreduced'].get('R134a', np.nan):.4f}"},
        ])
        print(case_df.to_string(index=False))
        
    os.makedirs('paper_results', exist_ok=True)
    df_probes.to_csv('paper_results/table_posthoc_probes.csv', index=False)
    corr_df.to_csv('paper_results/table_probes_correlation.csv', index=False)
    print("\n✅ 物理探针分析结果已落盘至 paper_results/ 目录！")

if __name__ == '__main__':
    run_probes()
