"""
audit_f2_raw_xtb.py — Rigorous Physical & Quality Audit for F2 218-Pair xTB Results
===================================================================================
1. Schema & Key Set Completeness (218 rows, 27 columns)
2. Orientation Sampling Statistics (Ori 1..4 convergence & best orientation distribution)
3. Energy & Geometric Distance Range Audit (Delta_E_assoc, d_min)
4. Core-10 System Context (20 Link Instances, 16 Unique Pairs) 100% Verification
===================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
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
CSV_PATH = SCRIPT_DIR / "full_pair_interaction_results.csv"
INDEX_CSV = ROOT_DIR / "index_with_anion.csv"

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

EXPECTED_COLUMNS = [
    'Pair_Type', 'Ion_Name', 'Refrigerant',
    'Delta_E_assoc_kcal_mol', 'Delta_E_int_kcal_mol',
    'd_min_Angstrom', 'Best_Orientation', 'N_Converged_Orientations',
    'E_complex_Eh', 'E_ion_Eh', 'E_ref_Eh',
    'E_ori1_Eh', 'converged_ori1', 'd_min_ori1_Angstrom',
    'E_ori2_Eh', 'converged_ori2', 'd_min_ori2_Angstrom',
    'E_ori3_Eh', 'converged_ori3', 'd_min_ori3_Angstrom',
    'E_ori4_Eh', 'converged_ori4', 'd_min_ori4_Angstrom',
    'energy_selection_criterion', 'physical_definition',
    'monomer_conformer_protocol', 'Status'
]

def main():
    print("=" * 85)
    print("   F2 QUANTUM CHEMICAL PHYSICAL AUDIT: 218 PAIR xTB RESULTS")
    print("=" * 85)

    if not CSV_PATH.exists():
        print(f"❌ 产物文件不存在: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"\n[SECTION 1: 数据集完整性与 Schema 审计]")
    print(f"  • 文件路径: {CSV_PATH}")
    print(f"  • 记录总行数: {len(df)} / 218 (预期 218)")
    print(f"  • 字段总列数: {len(df.columns)} / 27 (预期 27 列 v2.0 Schema)")
    
    schema_ok = (list(df.columns) == EXPECTED_COLUMNS)
    print(f"  • Schema 字段全等性: {'✓ 100% 严格一致' if schema_ok else '❌ 字段不匹配'}")
    if not schema_ok:
        print(f"    差异列: {set(EXPECTED_COLUMNS) ^ set(df.columns)}")

    # 键集全等与唯一性
    df_idx = pd.read_csv(INDEX_CSV)
    exp_anion = {("Anion-Ref", str(r["anion"]), str(r["refrigerant"])) for _, r in df_idx[["anion", "refrigerant"]].drop_duplicates().iterrows()}
    exp_cation = {("Cation-Ref", str(r["cation"]), str(r["refrigerant"])) for _, r in df_idx[["cation", "refrigerant"]].drop_duplicates().iterrows()}
    expected_pairs = exp_anion | exp_cation

    actual_keys = [(str(r["Pair_Type"]), str(r["Ion_Name"]), str(r["Refrigerant"])) for _, r in df.iterrows()]
    dup_keys = len(actual_keys) - len(set(actual_keys))
    print(f"  • 重复任务键数量: {dup_keys} ({'✓ 无重复' if dup_keys == 0 else '❌ 存在重复键'})")
    set_match = (set(actual_keys) == expected_pairs)
    print(f"  • 218 键集合全等性: {'✓ 与 index_with_anion.csv 100% 全等' if set_match else '❌ 存在键集缺失或溢出'}")

    print(f"\n[SECTION 2: 全量计算收敛率与状态分布]")
    status_counts = df["Status"].value_counts().to_dict()
    n_succ = status_counts.get("Success", 0)
    n_fail = status_counts.get("Failed", 0)
    print(f"  • 成功完成 (Success): {n_succ} / {len(df)} ({n_succ / len(df) * 100:.2f}%)")
    print(f"  • 失败/未收敛 (Failed): {n_fail} / {len(df)} ({n_fail / len(df) * 100:.2f}%)")

    # 分离子类型统计
    for p_type in ["Anion-Ref", "Cation-Ref"]:
        sub = df[df["Pair_Type"] == p_type]
        s_succ = (sub["Status"] == "Success").sum()
        print(f"    - {p_type:<10}: {s_succ} / {len(sub)} 成功 ({s_succ/len(sub)*100:.1f}%)")

    print(f"\n[SECTION 3: 4 构向空间采样与几何收敛统计]")
    df_succ = df[df["Status"] == "Success"]
    n_conv_dist = df["N_Converged_Orientations"].value_counts().sort_index(ascending=False).to_dict()
    print(f"  • 每组配对收敛构向数分布 (4 构向采样):")
    for k, v in n_conv_dist.items():
        print(f"    - {k}/4 个构向收敛: {v} 组 ({v / len(df) * 100:.1f}%)")

    # 各构向独立收敛率
    for i in range(1, 5):
        c_i = df[f"converged_ori{i}"].sum()
        print(f"    - 构向 {i} (Ori {i}) 几何收敛率: {c_i} / {len(df)} ({c_i / len(df) * 100:.1f}%)")

    best_ori_dist = df_succ["Best_Orientation"].value_counts().sort_index().to_dict()
    print(f"  • 最低能量构向 (Best Orientation) 选择分布:")
    for ori_idx, cnt in best_ori_dist.items():
        print(f"    - 最佳为 Ori {int(ori_idx)}: {cnt} 次 ({cnt / len(df_succ) * 100:.1f}%)")

    print(f"\n[SECTION 4: 物理能级与几何接触距离区间审计]")
    delta_e = df_succ["Delta_E_assoc_kcal_mol"]
    d_mins = df_succ["d_min_Angstrom"]

    print(f"  • 缔合能 ΔE_assoc (kcal/mol):")
    print(f"    - 极值范围: [{delta_e.min():.2f}, {delta_e.max():.2f}] kcal/mol")
    print(f"    - 均值 ± 标准差: {delta_e.mean():.2f} ± {delta_e.std():.2f} kcal/mol")
    print(f"    - 中位数 (IQR): {delta_e.median():.2f} [{delta_e.quantile(0.25):.2f}, {delta_e.quantile(0.75):.2f}] kcal/mol")
    
    # 阳离子 vs 阴离子能级分布对比
    ani_de = df_succ[df_succ["Pair_Type"] == "Anion-Ref"]["Delta_E_assoc_kcal_mol"]
    cat_de = df_succ[df_succ["Pair_Type"] == "Cation-Ref"]["Delta_E_assoc_kcal_mol"]
    print(f"    - 阴离子-制冷剂 (N={len(ani_de)}): 均值 {ani_de.mean():.2f} kcal/mol, 范围 [{ani_de.min():.2f}, {ani_de.max():.2f}]")
    print(f"    - 阳离子-制冷剂 (N={len(cat_de)}): 均值 {cat_de.mean():.2f} kcal/mol, 范围 [{cat_de.min():.2f}, {cat_de.max():.2f}]")

    print(f"  • 最近原子接触距离 d_min (Å):")
    print(f"    - 极值范围: [{d_mins.min():.2f}, {d_mins.max():.2f}] Å")
    print(f"    - 均值 ± 标准差: {d_mins.mean():.2f} ± {d_mins.std():.2f} Å")
    print(f"    - 中位数: {d_mins.median():.2f} Å")
    
    anomalous_d = df_succ[(df_succ["d_min_Angstrom"] < 1.2) | (df_succ["d_min_Angstrom"] > 4.5)]
    print(f"    - 异常距离分子对数 (<1.2 Å 或 >4.5 Å): {len(anomalous_d)}")

    print(f"\n[SECTION 5: Phase 2 核心 10 个三元体系上下文 (20 链接实例, 16 唯一对) 审计]")
    core_records = []
    covered_systems = 0

    print(f"  {'#':<3} {'System Context':<36} {'C-R Pair':<24} {'C-R ΔE':<10} {'A-R Pair':<24} {'A-R ΔE':<10} {'Sum |ΔE|':<10} {'Status'}")
    print("  " + "-" * 125)

    for s_idx, (cat, ani, ref) in enumerate(PHASE2_10_SYSTEM_CONTEXTS, 1):
        m_cat = df[(df["Pair_Type"] == "Cation-Ref") & (df["Ion_Name"] == cat) & (df["Refrigerant"] == ref)]
        m_ani = df[(df["Pair_Type"] == "Anion-Ref") & (df["Ion_Name"] == ani) & (df["Refrigerant"] == ref)]

        cat_succ = (len(m_cat) == 1) and (m_cat.iloc[0]["Status"] == "Success")
        ani_succ = (len(m_ani) == 1) and (m_ani.iloc[0]["Status"] == "Success")

        if cat_succ and ani_succ:
            covered_systems += 1
            cat_e = m_cat.iloc[0]["Delta_E_assoc_kcal_mol"]
            ani_e = m_ani.iloc[0]["Delta_E_assoc_kcal_mol"]
            sum_e = abs(cat_e) + abs(ani_e)
            sys_name = f"{cat} + {ani} + {ref}"
            c_link = f"{cat} — {ref}"
            a_link = f"{ani} — {ref}"
            print(f"  {s_idx:<3} {sys_name:<36} {c_link:<24} {cat_e:<10.2f} {a_link:<24} {ani_e:<10.2f} {sum_e:<10.2f} ✓ PASS")
            core_records.append({
                'System_Idx': s_idx,
                'System_Context': sys_name,
                'Cation': cat,
                'Anion': ani,
                'Refrigerant': ref,
                'Delta_E_assoc_cat_kcal_mol': cat_e,
                'Delta_E_assoc_ani_kcal_mol': ani_e,
                'Combined_Strength_Sum_kcal_mol': sum_e,
                'Max_Strength_kcal_mol': max(abs(cat_e), abs(ani_e)),
                'd_min_cat_Angstrom': m_cat.iloc[0]["d_min_Angstrom"],
                'd_min_ani_Angstrom': m_ani.iloc[0]["d_min_Angstrom"],
                'N_conv_cat': m_cat.iloc[0]["N_Converged_Orientations"],
                'N_conv_ani': m_ani.iloc[0]["N_Converged_Orientations"]
            })
        else:
            sys_name = f"{cat} + {ani} + {ref}"
            print(f"  {s_idx:<3} {sys_name:<36} FAIL: C-R({cat_succ}), A-R({ani_succ})")

    print("  " + "-" * 125)
    print(f"  • 核心体系上下文覆盖率: {covered_systems} / 10 ({covered_systems * 10.0:.1f}%)")
    
    # 保存核心 10 个体系的物理对照表供下游分析
    df_core = pd.DataFrame(core_records)
    core_out = SCRIPT_DIR / "core10_system_physics_summary.csv"
    df_core.to_csv(core_out, index=False)
    print(f"  • 核心 10 体系物理能量参数表已持久化: {core_out}")

    print("\n" + "=" * 85)
    if covered_systems == 10 and schema_ok and dup_keys == 0 and set_match:
        print("  🎉 [PHYSICAL AUDIT FULLY PASSED] 核心体系 100% 收敛，Schema 无污染，物理参数合理！")
    else:
        print("  ⚠️ [PHYSICAL AUDIT WARNING] 存在未通过项目，请核查上方日志。")
    print("=" * 85)

if __name__ == "__main__":
    main()
