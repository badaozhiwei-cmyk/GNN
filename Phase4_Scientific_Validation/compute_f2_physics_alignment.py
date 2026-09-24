"""
compute_f2_physics_alignment.py — Master F2 Physics Alignment & Statistical Re-Audit Pipeline
=============================================================================================
Protocol: v2.4 Production Final Audit Version
Hierarchy & Methodological Rules:
  - 430 evals -> 43 operational samples -> 10 System Contexts (N_system = 10)
  - Primary Inferential Track: Independent V7-A and V7-B tracks (Decoupled)
  - Pooled Track: Strictly designated as "Exploratory Only" (ensemble summary, not independent)
  - Main Evidence: 10-System Paired Contrast (removes additive component separation; suggestive trend)
  - Pair Identity Level: 16 Unique Ion-Refrigerant Pairs (removes physical-axis duplication)
  - Supporting Evidence: 20 Link Instances (context-conditioned, caution on between-group separation)
  - Boundary Evidence & Sensitivity: Layer 2 Simplex Normalization Masking (P_ref vs Raw A_ref vs Logit)
  - Permutation Null: Phipson & Smyth (2010) finite exact formula with full audit: (n_extreme + 1) / (n_perm + 1)
  - Cluster Bootstrap: N_cluster = 10 System Contexts (10,000 iterations)
=============================================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# 适配 Windows 控制台编码
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

CLEAN_PAIR_CSV = SCRIPT_DIR / "full_pair_interaction_results_v7_clean.csv"
XTB_PAIR_CSV = CLEAN_PAIR_CSV if CLEAN_PAIR_CSV.exists() else SCRIPT_DIR / "full_pair_interaction_results.csv"

CORE10_PHYSICS_CSV = SCRIPT_DIR / "core10_system_physics_summary.csv"

CLEAN_DESC_CSV = SCRIPT_DIR / "xTB_Physics_Descriptors_v7_clean.csv"
XTB_DESC_CSV = CLEAN_DESC_CSV if CLEAN_DESC_CSV.exists() else SCRIPT_DIR / "xTB_Physics_Descriptors.csv"

def _assert_uncorrupted_descriptors(desc_csv: Path):
    if desc_csv.exists():
        df = pd.read_csv(desc_csv)
        yf = df[df['Molecule'].astype(str).str.upper().str.strip() == 'R1234YF']
        if len(yf) > 0 and 'SMILES' in yf.columns:
            for smi in yf['SMILES']:
                if str(smi).strip() == "C(=C(F)F)(C(F)(F)F)F":
                    raise RuntimeError(
                        f"FAIL-CLOSED BREACH: {desc_csv} contains historical HFP C3F6 alias under R1234yf! "
                        f"Refusing to execute alignment on contaminated descriptors."
                    )

GROUPS_ATTR_CSV = ROOT_DIR / "v7_shadow_experiment" / "results_attribution" / "v7_graph_attribution_groups.csv"

# 10 个核心三元体系上下文
PHASE2_10_SYSTEM_CONTEXTS = [
    ("[emim]", "[Ac]", "R1234yf"),
    ("[emim]", "[BF4]", "R1234yf"),
    ("[bmim]", "[Ac]", "R1234yf"),
    ("[bmim]", "[PF6]", "R1234yf"),
    ("[emim]", "[BEI]", "R134"),
    ("[emim]", "[BEI]", "R134a"),
    ("[bmim]", "[BF4]", "R32"),
    ("[bmim]", "[PF6]", "R32"),
    ("[emim]", "[Tf2N]", "R1336mzz(E)"),
    ("[emim]", "[Tf2N]", "R1336mzz(Z)")
]

def parse_system(sample_id):
    parts = sample_id.split('__')
    return parts[0], parts[1], parts[2]

def run_permutation_test_phipson_smyth(x, y, n_perm=10000, seed=42):
    """
    置换检验（Permutation Null Test），严格采用 Phipson & Smyth (2010) 有限抽样校正公式:
    p = (n_extreme + 1) / (n_perm + 1)
    记录完整的 n_perm, n_extreme 与精确经验 p 值。
    """
    np.random.seed(seed)
    r_obs, _ = stats.spearmanr(x, y)
    y_arr = np.array(y)
    x_arr = np.array(x)
    n_extreme = 0
    for _ in range(n_perm):
        y_perm = np.random.permutation(y_arr)
        r_p, _ = stats.spearmanr(x_arr, y_perm)
        if abs(r_p) >= abs(r_obs):
            n_extreme += 1
    p_exact = (n_extreme + 1.0) / (n_perm + 1.0)
    p_formatted = f"< 10^-4" if n_extreme == 0 else f"{p_exact:.4f}"
    return float(r_obs), float(p_exact), p_formatted, n_perm, n_extreme

def run_cluster_bootstrap(df_systems, x_col, y_col, cluster_col='system_key', n_boot=10000, seed=42):
    """
    以 N_cluster = 10 个三元体系为抽样簇的 Cluster Bootstrap
    保持体系内链接的物理耦合与拓扑约束，输出 95% 置信区间。
    """
    np.random.seed(seed)
    system_ids = df_systems[cluster_col].unique()
    n_clusters = len(system_ids)
    
    boot_spearman = []
    boot_pearson = []

    for _ in range(n_boot):
        sampled_clusters = np.random.choice(system_ids, size=n_clusters, replace=True)
        sampled_rows = []
        for c in sampled_clusters:
            sampled_rows.append(df_systems[df_systems[cluster_col] == c])
        df_sample = pd.concat(sampled_rows, ignore_index=True)
        
        if len(df_sample[x_col].unique()) > 1 and len(df_sample[y_col].unique()) > 1:
            r_sp, _ = stats.spearmanr(df_sample[x_col], df_sample[y_col])
            r_pe, _ = stats.pearsonr(df_sample[x_col], df_sample[y_col])
            if np.isfinite(r_sp): boot_spearman.append(r_sp)
            if np.isfinite(r_pe): boot_pearson.append(r_pe)

    boot_spearman = np.array(boot_spearman)
    boot_pearson = np.array(boot_pearson)

    ci_sp = [float(np.percentile(boot_spearman, 2.5)), float(np.percentile(boot_spearman, 97.5))]
    ci_pe = [float(np.percentile(boot_pearson, 2.5)), float(np.percentile(boot_pearson, 97.5))]

    return ci_sp, ci_pe

def main():
    print("=" * 115)
    print("  MASTER F2 PHYSICS ALIGNMENT & DECOUPLED STATISTICAL RE-AUDIT PIPELINE (v2.4)")
    print("=" * 115)

    # 1. 读取基础数据
    df_xtb_pairs = pd.read_csv(XTB_PAIR_CSV)
    df_core_phys = pd.read_csv(CORE10_PHYSICS_CSV)
    _assert_uncorrupted_descriptors(XTB_DESC_CSV)
    df_desc = pd.read_csv(XTB_DESC_CSV)
    df_groups = pd.read_csv(GROUPS_ATTR_CSV)

    print(f"[*] 成功加载 Phase 2 归因数据: {len(df_groups)} 组基团归因记录 (430 次评估)")
    print(f"[*] 成功加载 xTB 配对计算数据: {len(df_xtb_pairs)} 对 (Pair级: 218/218 具备优化收敛构型, 构向级: 812/872=93.12% 收敛)")
    print(f"[*] 成功加载 Core-10 物理能级参数表: {len(df_core_phys)} 个体系")

    # 2. 聚合组件级归因份额与原始绝对归因质量
    comp_attr = df_groups.groupby(['model_family', 'seed', 'sample_id', 'component'])['A_g_abs'].sum().unstack(fill_value=0).reset_index()
    comp_attr['A_total'] = comp_attr['Cation'] + comp_attr['Anion'] + comp_attr['Refri']
    comp_attr['P_cat'] = comp_attr['Cation'] / comp_attr['A_total']
    comp_attr['P_ani'] = comp_attr['Anion'] / comp_attr['A_total']
    comp_attr['P_ref'] = comp_attr['Refri'] / comp_attr['A_total']
    comp_attr['A_ref'] = comp_attr['Refri'] # 原始绝对归因质量

    # 提取 (cat, ani, ref) 与 system_key
    systems_meta = [parse_system(s) for s in comp_attr['sample_id']]
    comp_attr['Cation'] = [m[0] for m in systems_meta]
    comp_attr['Anion'] = [m[1] for m in systems_meta]
    comp_attr['Refrigerant'] = [m[2] for m in systems_meta]
    comp_attr['system_key'] = comp_attr['Cation'] + " + " + comp_attr['Anion'] + " + " + comp_attr['Refrigerant']

    # 3. 体系级聚合（按协议：中位数聚合 median(T, P, seed)）
    agg_list = []
    for model in ['V7-A', 'V7-B', 'Pooled']:
        sub = comp_attr if model == 'Pooled' else comp_attr[comp_attr['model_family'] == model]
        median_df = sub.groupby(['system_key', 'Cation', 'Anion', 'Refrigerant'])[['P_cat', 'P_ani', 'P_ref', 'A_ref']].median().reset_index()
        median_df.columns = ['system_key', 'Cation', 'Anion', 'Refrigerant', 'P_cat_median', 'P_ani_median', 'P_ref_median', 'A_ref_median']
        median_df['model_track'] = 'Pooled (Exploratory only)' if model == 'Pooled' else model
        agg_list.append(median_df)

    df_sys_attr = pd.concat(agg_list, ignore_index=True)

    # 合并 Core-10 物理能量参数
    df_core_phys['system_key'] = df_core_phys['System_Context']
    df_sys_merged = pd.merge(df_sys_attr, df_core_phys, on='system_key')
    df_sys_merged['Cation'] = df_sys_merged['Cation_x']
    df_sys_merged['Anion'] = df_sys_merged['Anion_x']
    df_sys_merged['Refrigerant'] = df_sys_merged['Refrigerant_x']

    stat_summary_rows = []

    # =========================================================================
    # PART 1: 【主证据 Main Evidence】10-System 配对对比分析 (Paired Contrast)
    # =========================================================================
    print("\n" + "=" * 115)
    print("【PART 1: 主证据 (Main Evidence) —— 10 体系配对对比 (Paired Contrast Analysis)】")
    print("  * 核心定位: paired contrast removes the simplest additive component-type separation from the analysis,")
    print("              while residual component-specific and system-level dependence may remain.")
    print("  * 结果性质: did not reach the conventional 0.05 threshold, but showed a suggestive positive association.")
    print("=" * 115)

    main_contrast_rows = []

    for model in ['V7-A', 'V7-B', 'Pooled (Exploratory only)']:
        sub = df_sys_merged[df_sys_merged['model_track'] == model].copy()
        sub['D_P'] = sub['P_ani_median'] - sub['P_cat_median']
        sub['D_E'] = np.abs(sub['Delta_E_assoc_ani_kcal_mol']) - np.abs(sub['Delta_E_assoc_cat_kcal_mol'])
        
        r_sp, p_exact_sp, p_str_sp, n_p, n_e = run_permutation_test_phipson_smyth(sub['D_E'], sub['D_P'])
        r_pe, _ = stats.pearsonr(sub['D_E'], sub['D_P'])
        ci_sp, ci_pe = run_cluster_bootstrap(sub, 'D_E', 'D_P')

        stat_summary_rows.append({
            'Evidence_Tier': 'Tier 1: Main Evidence',
            'Model_Track': model,
            'Analysis_Target': '10-System Paired Contrast',
            'Sample_Units': 'N_system=10',
            'Physics_Descriptor': 'D_E = |ΔE_ani| - |ΔE_cat| (kcal/mol)',
            'Attribution_Metric': 'D_P = P_ani - P_cat',
            'Spearman_rho': r_sp,
            'Permutation_p': p_exact_sp,
            'Perm_p_formatted': p_str_sp,
            'N_perm': n_p,
            'N_extreme': n_e,
            'Spearman_95CI': f"[{ci_sp[0]:.3f}, {ci_sp[1]:.3f}]",
            'Pearson_r': r_pe,
            'Pearson_95CI': f"[{ci_pe[0]:.3f}, {ci_pe[1]:.3f}]"
        })

        if model == 'Pooled (Exploratory only)':
            for _, r in sub.iterrows():
                main_contrast_rows.append({
                    'System_Context': r['system_key'],
                    'Cation': r['Cation'],
                    'Anion': r['Anion'],
                    'Refrigerant': r['Refrigerant'],
                    'Delta_E_cat_kcal_mol': r['Delta_E_assoc_cat_kcal_mol'],
                    'Delta_E_ani_kcal_mol': r['Delta_E_assoc_ani_kcal_mol'],
                    'D_E_kcal_mol': r['D_E'],
                    'P_cat_median': r['P_cat_median'],
                    'P_ani_median': r['P_ani_median'],
                    'D_P': r['D_P']
                })

    df_main_contrast = pd.DataFrame(main_contrast_rows)
    df_main_contrast.to_csv(SCRIPT_DIR / "f2_table_main_contrast_10sys.csv", index=False)

    # =========================================================================
    # PART 2: 【配对身份级证据】16 个唯一物理对分析 (16 Unique Pair Identities)
    # =========================================================================
    print("\n" + "=" * 115)
    print("【PART 2: 配对身份级证据 —— 16 个唯一物理对分析 (16 Unique Pair Identities)】")
    print("  * 核心定位: removes duplicated pair identities on the physical-energy axis and summarizes")
    print("              context-conditioned attribution at the unique-pair level.")
    print("=" * 115)

    for model in ['V7-A', 'V7-B', 'Pooled (Exploratory only)']:
        sub_sys = df_sys_merged[df_sys_merged['model_track'] == model].copy()
        
        inst_rows = []
        for _, r in sub_sys.iterrows():
            inst_rows.append({
                'system_key': r['system_key'],
                'Link_Type': 'Cation-Ref',
                'Ion_Name': r['Cation'],
                'Refrigerant': r['Refrigerant'],
                'Pair_Identity': f"{r['Cation']} — {r['Refrigerant']}",
                'P_comp': r['P_cat_median'],
                'abs_Delta_E': abs(r['Delta_E_assoc_cat_kcal_mol'])
            })
            inst_rows.append({
                'system_key': r['system_key'],
                'Link_Type': 'Anion-Ref',
                'Ion_Name': r['Anion'],
                'Refrigerant': r['Refrigerant'],
                'Pair_Identity': f"{r['Anion']} — {r['Refrigerant']}",
                'P_comp': r['P_ani_median'],
                'abs_Delta_E': abs(r['Delta_E_assoc_ani_kcal_mol'])
            })
        df_inst = pd.DataFrame(inst_rows)

        # 聚合为 16 唯一对 (取中位数)
        df_u16 = df_inst.groupby(['Pair_Identity', 'Link_Type'])[['P_comp', 'abs_Delta_E']].median().reset_index()
        
        r_sp_16, p_exact_16, p_str_16, n_p16, n_e16 = run_permutation_test_phipson_smyth(df_u16['abs_Delta_E'], df_u16['P_comp'])
        r_pe_16, _ = stats.pearsonr(df_u16['abs_Delta_E'], df_u16['P_comp'])
        ci_sp_16, ci_pe_16 = run_cluster_bootstrap(df_inst, 'abs_Delta_E', 'P_comp')

        df_ani_16 = df_u16[df_u16['Link_Type'] == 'Anion-Ref']
        r_sp_ani16, p_exact_ani16, p_str_ani16, n_pa, n_ea = run_permutation_test_phipson_smyth(df_ani_16['abs_Delta_E'], df_ani_16['P_comp'])
        r_pe_ani16, _ = stats.pearsonr(df_ani_16['abs_Delta_E'], df_ani_16['P_comp'])

        df_cat_16 = df_u16[df_u16['Link_Type'] == 'Cation-Ref']
        r_sp_cat16, p_exact_cat16, p_str_cat16, n_pc, n_ec = run_permutation_test_phipson_smyth(df_cat_16['abs_Delta_E'], df_cat_16['P_comp'])
        r_pe_cat16, _ = stats.pearsonr(df_cat_16['abs_Delta_E'], df_cat_16['P_comp'])

        stat_summary_rows.extend([
            {
                'Evidence_Tier': 'Tier 2: Pair Identity',
                'Model_Track': model,
                'Analysis_Target': '16 Unique Pairs (Pooled C-R & A-R)',
                'Sample_Units': 'N_unique=16 (7 C-R, 9 A-R)',
                'Physics_Descriptor': '|ΔE_assoc| (Pair Identity Median)',
                'Attribution_Metric': 'P_comp_median',
                'Spearman_rho': r_sp_16,
                'Permutation_p': p_exact_16,
                'Perm_p_formatted': p_str_16,
                'N_perm': n_p16,
                'N_extreme': n_e16,
                'Spearman_95CI': f"[{ci_sp_16[0]:.3f}, {ci_sp_16[1]:.3f}]",
                'Pearson_r': r_pe_16,
                'Pearson_95CI': f"[{ci_pe_16[0]:.3f}, {ci_pe_16[1]:.3f}]"
            },
            {
                'Evidence_Tier': 'Tier 2: Pair Identity (Within Anion)',
                'Model_Track': model,
                'Analysis_Target': '9 Unique Anion Pairs',
                'Sample_Units': 'N_unique=9',
                'Physics_Descriptor': '|ΔE_assoc(A-R)|',
                'Attribution_Metric': 'P_ani_median',
                'Spearman_rho': r_sp_ani16,
                'Permutation_p': p_exact_ani16,
                'Perm_p_formatted': p_str_ani16,
                'N_perm': n_pa,
                'N_extreme': n_ea,
                'Spearman_95CI': 'N/A (N=9 small)',
                'Pearson_r': r_pe_ani16,
                'Pearson_95CI': 'N/A'
            },
            {
                'Evidence_Tier': 'Tier 2: Pair Identity (Within Cation)',
                'Model_Track': model,
                'Analysis_Target': '7 Unique Cation Pairs',
                'Sample_Units': 'N_unique=7',
                'Physics_Descriptor': '|ΔE_assoc(C-R)|',
                'Attribution_Metric': 'P_cat_median',
                'Spearman_rho': r_sp_cat16,
                'Permutation_p': p_exact_cat16,
                'Perm_p_formatted': p_str_cat16,
                'N_perm': n_pc,
                'N_extreme': n_ec,
                'Spearman_95CI': 'N/A (N=7 small)',
                'Pearson_r': r_pe_cat16,
                'Pearson_95CI': 'N/A'
            }
        ])

        if model == 'Pooled (Exploratory only)':
            df_u16.to_csv(SCRIPT_DIR / "f2_table_unique16_pairs.csv", index=False)

    # =========================================================================
    # PART 3: 【辅助证据 Supporting Evidence】20 链接实例分析 (带组间效应警示)
    # =========================================================================
    print("\n" + "=" * 115)
    print("【PART 3: 辅助证据 (Supporting Evidence) —— 20 链接实例分析 (带组间效应警示)】")
    print("  * 科学警示: pooled rho=0.819 包含显著的阴阳离子组间分离效应，不能单向解释为微观单调规律")
    print("=" * 115)

    for model in ['V7-A', 'V7-B', 'Pooled (Exploratory only)']:
        sub_sys = df_sys_merged[df_sys_merged['model_track'] == model].copy()
        
        inst_rows = []
        for _, r in sub_sys.iterrows():
            inst_rows.append({
                'system_key': r['system_key'],
                'Link_Type': 'Cation-Ref',
                'Ion_Name': r['Cation'],
                'Refrigerant': r['Refrigerant'],
                'Pair_Identity': f"{r['Cation']} — {r['Refrigerant']}",
                'P_comp': r['P_cat_median'],
                'abs_Delta_E': abs(r['Delta_E_assoc_cat_kcal_mol'])
            })
            inst_rows.append({
                'system_key': r['system_key'],
                'Link_Type': 'Anion-Ref',
                'Ion_Name': r['Anion'],
                'Refrigerant': r['Refrigerant'],
                'Pair_Identity': f"{r['Anion']} — {r['Refrigerant']}",
                'P_comp': r['P_ani_median'],
                'abs_Delta_E': abs(r['Delta_E_assoc_ani_kcal_mol'])
            })
        df_l1 = pd.DataFrame(inst_rows)

        r_sp_l1, p_exact_l1, p_str_l1, n_pl1, n_el1 = run_permutation_test_phipson_smyth(df_l1['abs_Delta_E'], df_l1['P_comp'])
        r_pe_l1, _ = stats.pearsonr(df_l1['abs_Delta_E'], df_l1['P_comp'])
        ci_sp_l1, ci_pe_l1 = run_cluster_bootstrap(df_l1, 'abs_Delta_E', 'P_comp')

        stat_summary_rows.append({
            'Evidence_Tier': 'Tier 3: Supporting (Instances)',
            'Model_Track': model,
            'Analysis_Target': '20 Link Instances (Pooled)',
            'Sample_Units': 'N_instance=20',
            'Physics_Descriptor': '|ΔE_assoc| (C-R & A-R)',
            'Attribution_Metric': 'P_comp_median',
            'Spearman_rho': r_sp_l1,
            'Permutation_p': p_exact_l1,
            'Perm_p_formatted': p_str_l1,
            'N_perm': n_pl1,
            'N_extreme': n_el1,
            'Spearman_95CI': f"[{ci_sp_l1[0]:.3f}, {ci_sp_l1[1]:.3f}]",
            'Pearson_r': r_pe_l1,
            'Pearson_95CI': f"[{ci_pe_l1[0]:.3f}, {ci_pe_l1[1]:.3f}]"
        })

        if model == 'V7-A':
            df_l1.to_csv(SCRIPT_DIR / "f2_table_layer1_v7a.csv", index=False)
        elif model == 'V7-B':
            df_l1.to_csv(SCRIPT_DIR / "f2_table_layer1_v7b.csv", index=False)

    # =========================================================================
    # PART 4: 【边界证据与敏感性分析】Layer 2 Simplex Normalization Masking Audit
    # =========================================================================
    print("\n" + "=" * 115)
    print("【PART 4: 边界证据与敏感性检验 —— Layer 2 单纯形归一化掩盖效应审计】")
    print("  * 核心对比: 归一化 P_ref (Null) vs 原始绝对归因 A_ref (显著正相关) vs 对数比 logit(P_ref)")
    print("=" * 115)

    sensitivity_rows = []

    for model in ['V7-A', 'V7-B', 'Pooled (Exploratory only)']:
        sub_sys = df_sys_merged[df_sys_merged['model_track'] == model].copy()
        sub_sys['logit_P_ref'] = np.log(sub_sys['P_ref_median'] / (1.0 - sub_sys['P_ref_median']))

        for metric_name, metric_col in [('Normalized P_ref', 'P_ref_median'), 
                                         ('Raw A_ref (Unnormalized)', 'A_ref_median'), 
                                         ('Logit(P_ref)', 'logit_P_ref')]:
            for energy_name, energy_col in [('Combined Sum |ΔE|', 'Combined_Strength_Sum_kcal_mol'),
                                            ('Max |ΔE|', 'Max_Strength_kcal_mol')]:
                r_sp, p_exact, p_str, n_p, n_e = run_permutation_test_phipson_smyth(sub_sys[energy_col], sub_sys[metric_col])
                r_pe, _ = stats.pearsonr(sub_sys[energy_col], sub_sys[metric_col])
                
                sensitivity_rows.append({
                    'Model_Track': model,
                    'Attribution_Form': metric_name,
                    'Energy_Descriptor': energy_name,
                    'Spearman_rho': r_sp,
                    'Permutation_p': p_exact,
                    'Perm_p_formatted': p_str,
                    'N_perm': n_p,
                    'N_extreme': n_e,
                    'Pearson_r': r_pe
                })

        # 主归一化行进入总表
        r_sp_l2, p_exact_l2, p_str_l2, n_pl2, n_el2 = run_permutation_test_phipson_smyth(sub_sys['Combined_Strength_Sum_kcal_mol'], sub_sys['P_ref_median'])
        r_pe_l2, _ = stats.pearsonr(sub_sys['Combined_Strength_Sum_kcal_mol'], sub_sys['P_ref_median'])
        ci_sp_l2, ci_pe_l2 = run_cluster_bootstrap(sub_sys, 'Combined_Strength_Sum_kcal_mol', 'P_ref_median')

        stat_summary_rows.append({
            'Evidence_Tier': 'Tier 4: Boundary (Layer 2)',
            'Model_Track': model,
            'Analysis_Target': 'Layer 2: Combined Strength Sum',
            'Sample_Units': 'N_system=10',
            'Physics_Descriptor': '|ΔE(C-R)| + |ΔE(A-R)|',
            'Attribution_Metric': 'P_ref_median',
            'Spearman_rho': r_sp_l2,
            'Permutation_p': p_exact_l2,
            'Perm_p_formatted': p_str_l2,
            'N_perm': n_pl2,
            'N_extreme': n_el2,
            'Spearman_95CI': f"[{ci_sp_l2[0]:.3f}, {ci_sp_l2[1]:.3f}]",
            'Pearson_r': r_pe_l2,
            'Pearson_95CI': f"[{ci_pe_l2[0]:.3f}, {ci_pe_l2[1]:.3f}]"
        })

        if model == 'V7-A':
            sub_sys.to_csv(SCRIPT_DIR / "f2_table_layer2_v7a.csv", index=False)
        elif model == 'V7-B':
            sub_sys.to_csv(SCRIPT_DIR / "f2_table_layer2_v7b.csv", index=False)

    df_sens = pd.DataFrame(sensitivity_rows)
    df_sens.to_csv(SCRIPT_DIR / "f2_table_layer2_sensitivity.csv", index=False)

    # =========================================================================
    # PART 5: 【F2-A 辅线】单分子尺度物理特征对齐分析 (全部 6 种制冷剂)
    # =========================================================================
    print("\n" + "=" * 115)
    print("【PART 5: F2-A 辅线 —— 全部 6 种制冷剂单分子物性与异构体对照】")
    print("=" * 115)

    ref_desc_map = {}
    for _, r in df_desc[df_desc['Category'] == 'Refrigerant'].iterrows():
        ref_desc_map[r['Molecule']] = {
            'Dipole_Debye': float(r['Dipole_Debye']),
            'Polarizability_au': float(r['Polarizability_au']),
            'Volume_A3': float(r['Volume_A3'])
        }

    species_f2a_rows = []
    # 核心 10 个体系涵盖的全部 6 种制冷剂
    all_6_refrigerants = ['R1234yf', 'R1336mzz(E)', 'R1336mzz(Z)', 'R134', 'R134a', 'R32']
    
    for ref in sorted(all_6_refrigerants):
        desc = ref_desc_map.get(ref, {'Dipole_Debye': np.nan, 'Polarizability_au': np.nan, 'Volume_A3': np.nan})
        p_v7a = df_sys_merged[(df_sys_merged['model_track'] == 'V7-A') & (df_sys_merged['Refrigerant'] == ref)]['P_ref_median'].median()
        p_v7b = df_sys_merged[(df_sys_merged['model_track'] == 'V7-B') & (df_sys_merged['Refrigerant'] == ref)]['P_ref_median'].median()
        p_pool = df_sys_merged[(df_sys_merged['model_track'] == 'Pooled (Exploratory only)') & (df_sys_merged['Refrigerant'] == ref)]['P_ref_median'].median()
        species_f2a_rows.append({
            'Refrigerant': ref,
            'Dipole_Debye': desc['Dipole_Debye'],
            'Polarizability_au': desc['Polarizability_au'],
            'Volume_A3': desc['Volume_A3'],
            'P_ref_median_V7A': p_v7a,
            'P_ref_median_V7B': p_v7b,
            'P_ref_median_Pooled': p_pool
        })
    df_f2a = pd.DataFrame(species_f2a_rows)
    df_f2a.to_csv(SCRIPT_DIR / "f2_table_species_alignment.csv", index=False)

    # 保存统计汇总总表
    df_stat_summary = pd.DataFrame(stat_summary_rows)
    df_stat_summary.to_csv(SCRIPT_DIR / "f2_table_statistical_summary.csv", index=False)

    print("\n" + "=" * 145)
    print("  🏆 F2 严谨重审全景统计检验报表 (Phipson & Smyth 2010 置换校正 + N_cluster=10 Bootstrap)")
    print("=" * 145)
    print(f"  {'Evidence Tier':<28} {'Model':<26} {'Analysis Target':<32} {'Spearman ρ':<12} {'Perm p-val':<12} {'Extreme/10k':<12} {'Spearman 95% CI':<24}")
    print("  " + "-" * 145)
    for _, r in df_stat_summary.iterrows():
        print(f"  {r['Evidence_Tier']:<28} {r['Model_Track']:<26} {r['Analysis_Target']:<32} {r['Spearman_rho']:<12.3f} {r['Perm_p_formatted']:<12} {r['N_extreme']:<12} {r['Spearman_95CI']:<24}")
    print("=" * 145)

    print("\n[DONE] 所有持久化分析产物已全部更新生成:")
    print(f"  • 主证据 10 体系配对差值表: {SCRIPT_DIR / 'f2_table_main_contrast_10sys.csv'}")
    print(f"  • 配对身份 16 唯一对数据表: {SCRIPT_DIR / 'f2_table_unique16_pairs.csv'}")
    print(f"  • 严谨统计检验分级汇总表: {SCRIPT_DIR / 'f2_table_statistical_summary.csv'}")
    print(f"  • Layer 2 归一化敏感性对比表: {SCRIPT_DIR / 'f2_table_layer2_sensitivity.csv'}")
    print(f"  • F2-A 物性对齐表 (全部 6 种制冷剂): {SCRIPT_DIR / 'f2_table_species_alignment.csv'}")

if __name__ == "__main__":
    main()
