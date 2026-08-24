"""
=====================================================================
 Pilot Interaction Energy Signal Analyzer
 Evaluates:
   1. Structural Sensitivity: Does Delta_E differentiate ion chemistry?
   2. Signal Strength: Correlation with experimental x1 vs Monomer Dipole
   3. Controlled Multivariable Regression Gain: x1 ~ T + P + Delta_E
=====================================================================
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PILOT_CSV = os.path.join(SCRIPT_DIR, 'pilot_pair_interaction_results.csv')
DATASET_CSV = os.path.join(ROOT_DIR, 'index_with_anion.csv')
XTB_DESC_CSV = os.path.join(SCRIPT_DIR, 'xTB_Physics_Descriptors.csv')

def main():
    print("="*70)
    print("  Pilot Interaction Energy Signal Diagnostic")
    print("="*70)
    
    if not os.path.exists(PILOT_CSV):
        print(f"❌ 未找到 Pilot 结果文件: {PILOT_CSV}，请先运行 pilot_pair_interaction_xtb.py！")
        return
        
    df_pilot = pd.read_csv(PILOT_CSV)
    df_raw = pd.read_csv(DATASET_CSV)
    
    # 1. 结构化比较：Anion-Ref 与 Cation-Ref 矩阵
    print("\n[Layer 1: 结构化化学敏感度矩阵 (Structural Sensitivity Matrix)]")
    anion_pairs = df_pilot[df_pilot['Pair_Type'] == 'Anion-Ref']
    if not anion_pairs.empty:
        pivot_anion = anion_pairs.pivot(index='Refrigerant', columns='Ion_Name', values='Delta_E_int_kcal_mol')
        print("\n📊 阴离子配对相互作用能矩阵 (ΔE_anion, kcal/mol):")
        print(pivot_anion.to_markdown(floatfmt=".2f"))
        
    cation_pairs = df_pilot[df_pilot['Pair_Type'] == 'Cation-Ref']
    if not cation_pairs.empty:
        pivot_cat = cation_pairs.pivot(index='Refrigerant', columns='Ion_Name', values='Delta_E_int_kcal_mol')
        print("\n📊 阳离子配对相互作用能矩阵 (ΔE_cation, kcal/mol):")
        print(pivot_cat.to_markdown(floatfmt=".2f"))

    # 2. 与真实溶解度数据集进行对齐合并
    print("\n[Layer 2: 控制变量实验关联度检验 (Controlled Correlation with x1)]")
    # 筛选出属于这 3 种制冷剂与 5 种阴离子 / 2 种阳离子的实验数据行
    target_refs = df_pilot['Refrigerant'].unique()
    target_anions = df_pilot[df_pilot['Pair_Type'] == 'Anion-Ref']['Ion_Name'].unique()
    target_cats = df_pilot[df_pilot['Pair_Type'] == 'Cation-Ref']['Ion_Name'].unique()
    
    sub_df = df_raw[
        df_raw['refrigerant'].isin(target_refs) &
        df_raw['anion'].isin(target_anions) &
        df_raw['cation'].isin(target_cats)
    ].copy()
    
    if sub_df.empty:
        print("⚠️ 数据集中未找到同时匹配上述 3 制冷剂 + 5 阴离子 + 2 阳离子的交集行，放宽阳离子限制进行阴离子匹配...")
        sub_df = df_raw[
            df_raw['refrigerant'].isin(target_refs) &
            df_raw['anion'].isin(target_anions)
        ].copy()

    print(f"  -> 成功提取相关真实实验数据点: {len(sub_df)} 行")
    
    # 合并 Delta_E_anion
    anion_map = dict(zip(
        zip(anion_pairs['Ion_Name'], anion_pairs['Refrigerant']),
        anion_pairs['Delta_E_int_kcal_mol']
    ))
    sub_df['Delta_E_anion'] = sub_df.apply(lambda r: anion_map.get((r['anion'], r['refrigerant']), np.nan), axis=1)
    
    # 合并单体偶极矩作为 Baseline 对照
    if os.path.exists(XTB_DESC_CSV):
        desc_df = pd.read_csv(XTB_DESC_CSV)
        dipole_map = dict(zip(desc_df['Molecule'], desc_df['Dipole_Debye']))
        sub_df['ref_dipole'] = sub_df['refrigerant'].map(dipole_map)
    else:
        sub_df['ref_dipole'] = np.nan
        
    clean_df = sub_df.dropna(subset=['Delta_E_anion', 'x1', 'T_K', 'P_MPa', 'ref_dipole']).copy()
    print(f"  -> 完整配对可用样本量: {len(clean_df)} 条")
    
    # 统计相关性
    spear_de, p_de = stats.spearmanr(clean_df['Delta_E_anion'], clean_df['x1'])
    spear_dip, p_dip = stats.spearmanr(clean_df['ref_dipole'], clean_df['x1'])
    
    print(f"\n  单变量相关性对比:")
    print(f"  - 阴离子相互作用能 ρ(ΔE_anion, x1) = {spear_de:+.4f} (p = {p_de:.4e})")
    print(f"  - 单体制冷剂偶极矩 ρ(μ_ref, x1)     = {spear_dip:+.4f} (p = {p_dip:.4e})")

    # 3. 控制多变量线性回归增益检验
    print("\n[Layer 3: 控制多变量回归增益检验 (Controlled Multivariable Regression)]")
    
    # Baseline 1: 只用 T, P
    X_tp = clean_df[['T_K', 'P_MPa']].values
    y = clean_df['x1'].values
    
    m_tp = Ridge().fit(X_tp, y)
    r2_tp = r2_score(y, m_tp.predict(X_tp))
    
    # Baseline 2: T, P + 单体偶极矩
    X_dip = clean_df[['T_K', 'P_MPa', 'ref_dipole']].values
    m_dip = Ridge().fit(X_dip, y)
    r2_dip = r2_score(y, m_dip.predict(X_dip))
    
    # Proposal: T, P + 相互作用能
    X_de = clean_df[['T_K', 'P_MPa', 'Delta_E_anion']].values
    m_de = Ridge().fit(X_de, y)
    r2_de = r2_score(y, m_de.predict(X_de))
    
    # Combined: T, P + 偶极矩 + 相互作用能
    X_both = clean_df[['T_K', 'P_MPa', 'ref_dipole', 'Delta_E_anion']].values
    m_both = Ridge().fit(X_both, y)
    r2_both = r2_score(y, m_both.predict(X_both))

    print(f"  - Model (T, P):                         R² = {r2_tp:.4f}")
    print(f"  - Model (T, P + μ_ref):                 R² = {r2_dip:.4f} (增益: {r2_dip - r2_tp:+.4f})")
    print(f"  - Model (T, P + ΔE_anion):              R² = {r2_de:.4f} (增益: {r2_de - r2_tp:+.4f})")
    print(f"  - Model (T, P + μ_ref + ΔE_anion):      R² = {r2_both:.4f} (总增益: {r2_both - r2_tp:+.4f})")
    
    print("\n" + "="*70)
    print("💡 结论与决策建议:")
    if r2_de > r2_dip:
        print("  ✅ 明确阳性信号！ΔE_anion 提供的物化解释力显著优于单体偶极矩 μ_ref。")
        print("  -> 建议：立刻将 Pilot 扩展到全量 218 个配对体系，并将其作为第二代物理特征注入 GNN！")
    else:
        print("  ⚠️ ΔE_anion 在当前简单回归中未展现出压倒性优势，需结合非线性 GNN 进行深入消融。")
    print("="*70)

if __name__ == '__main__':
    main()
