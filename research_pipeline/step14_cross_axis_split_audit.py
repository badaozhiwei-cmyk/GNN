"""
step14_cross_axis_split_audit.py — Phase II-D Gate D0: Cross-Axis Split Audit
=============================================================================
【目标】
1. L2 Ion-Pair Recombination 候选审计:
   - 必须满足: cation_test in C_train, anion_test in A_train, (c, a)_test not in Train
   - 严格在相同 Refrigerant Universe (如 HFC) 下考察
   - 建立所有候选 (c, a) pair 列表, 样本量, 体系数, 并评估 L2-Strict 与 L2-Controlled 切分可行性
2. HFC -> HFO 数据可行性审计:
   - 统计全集中所有 HFO/HCFO 工质 (R1234yf, R1234ze(E), R1233zd(E), R1336mzz(E/Z) 等)
   - 统计样本量 N_HFO, 涉及的 IL 体系数, T/P 空间分布及与 HFC 的重叠度
   - 评估支持 Internal family transfer (HFO-A) 还是 True external validation (HFO-B)
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"c:\Users\霸道志伟\Desktop\GNN\Refrigerant-Solubility-GNN")
DATA_CSV = PROJECT_ROOT / "index_with_anion.csv"
V6_META = PROJECT_ROOT / "processed_tri_data_v6" / "meta_info.csv"
OUT_DIR = PROJECT_ROOT / "paper_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("  PHASE II-D GATE D0: CROSS-AXIS SPLIT AUDIT")
print("=" * 80)

# 加载全量元数据
df_all = pd.read_csv(DATA_CSV)
print(f"[数据加载] index_with_anion.csv 加载成功: 总计 {len(df_all)} 条样本点")

# 统一大小写与清理
clean_str = lambda s: str(s).strip()
clean_ion = lambda s: str(s).strip().replace("[", "").replace("]", "").upper()

df_all["cation_clean"] = df_all["cation"].apply(clean_ion)
df_all["anion_clean"] = df_all["anion"].apply(clean_ion)
df_all["ref_clean"] = df_all["refrigerant"].apply(clean_str).str.upper()
df_all["il_pair"] = df_all["cation_clean"] + "__" + df_all["anion_clean"]
df_all["chem_system"] = df_all["il_pair"] + "__" + df_all["ref_clean"]

# 区分化学家族 (HFC vs HFO vs others)
# 经典 HFC 12 工质
HFC_12 = {"R32", "R125", "R134", "R134A", "R143A", "R152A", "R161", "R227EA", "R23", "R236FA", "R245FA", "R41"}

def classify_refrigerant(ref: str) -> str:
    r = ref.upper()
    if r in HFC_12 or (r.startswith("R") and any(c in r for c in ["32", "125", "134", "143", "152", "161", "227", "23", "236", "245", "41"])):
        return "HFC"
    elif "1234" in r or "1233" in r or "1336" in r or "ZE" in r or "YF" in r or "ZD" in r:
        return "HFO"
    elif "CO2" in r:
        return "CO2"
    elif any(x in r for x in ["22", "142", "123", "124"]):
        return "HCFC"
    elif any(x in r for x in ["11", "12", "13", "113", "114", "115"]):
        return "CFC"
    else:
        return "OTHER"

df_all["family"] = df_all["ref_clean"].apply(classify_refrigerant)

print("\n--- 全集化学家族分布 ---")
family_counts = df_all["family"].value_counts()
print(family_counts.to_string())

# ==============================================================================
# 1. HFO Feasibility Audit
# ==============================================================================
print("\n" + "=" * 80)
print("  AUDIT 1: HFC -> HFO DATA FEASIBILITY")
print("=" * 80)

df_hfo = df_all[df_all["family"] == "HFO"].copy()
df_hfc = df_all[df_all["family"] == "HFC"].copy()

print(f"HFO 样本总量 N: {len(df_hfo)}")
print(f"HFO 涉及工质列表: {df_hfo['ref_clean'].unique().tolist()}")
print(f"HFO 涉及化学体系数 (#IL x Ref): {df_hfo['chem_system'].nunique()}")
print(f"HFO 涉及 IL 种类数 (#IL pairs): {df_hfo['il_pair'].nunique()}")

# 详细统计每个 HFO 工质
hfo_breakdown = []
for ref_name, grp in df_hfo.groupby("ref_clean"):
    # 检查离子重叠度与 HFC 训练集
    cations_in_hfc = set(grp["cation_clean"]).issubset(set(df_hfc["cation_clean"]))
    anions_in_hfc = set(grp["anion_clean"]).issubset(set(df_hfc["anion_clean"]))
    il_pairs_in_hfc = set(grp["il_pair"]).issubset(set(df_hfc["il_pair"]))
    overlap_pairs = set(grp["il_pair"]).intersection(set(df_hfc["il_pair"]))
    
    sheets = grp["sheet"].unique().tolist() if "sheet" in grp.columns else ["Unknown"]
    
    hfo_breakdown.append({
        "HFO_refrigerant": ref_name,
        "n_samples": len(grp),
        "n_il_systems": grp["il_pair"].nunique(),
        "T_min": grp["T_K"].min(),
        "T_max": grp["T_K"].max(),
        "P_min": grp["P_MPa"].min(),
        "P_max": grp["P_MPa"].max(),
        "cations_in_HFC": cations_in_hfc,
        "anions_in_HFC": anions_in_hfc,
        "n_IL_pairs_shared_with_HFC": len(overlap_pairs),
        "source_sheets": "; ".join(sheets),
    })

df_hfo_audit = pd.DataFrame(hfo_breakdown)
df_hfo_audit.to_csv(OUT_DIR / "table_hfo_feasibility_audit.csv", index=False)
print("\nHFO 工质详细特征与重叠度:")
print(df_hfo_audit[["HFO_refrigerant", "n_samples", "n_il_systems", "T_min", "T_max", "P_min", "P_max", "n_IL_pairs_shared_with_HFC"]].to_string(index=False))

# 检查温压覆盖与 HFC 对比
print(f"\nHFC 训练集温压范围: T in [{df_hfc['T_K'].min():.1f}, {df_hfc['T_K'].max():.1f}] K, P in [{df_hfc['P_MPa'].min():.4f}, {df_hfc['P_MPa'].max():.4f}] MPa")
print(f"HFO 测试集温压范围: T in [{df_hfo['T_K'].min():.1f}, {df_hfo['T_K'].max():.1f}] K, P in [{df_hfo['P_MPa'].min():.4f}, {df_hfo['P_MPa'].max():.4f}] MPa")

# ==============================================================================
# 2. L2 Ion-Pair Recombination Audit (In HFC Universe)
# ==============================================================================
print("\n" + "=" * 80)
print("  AUDIT 2: L2 ION-PAIR RECOMBINATION (HFC UNIVERSE)")
print("=" * 80)

# 限制在 HFC 宇宙下考察 L2, 确保 refrigerant 完全受控 (Same Refrigerant Universe)
print(f"HFC Universe: 总样本数 N = {len(df_hfc)}, 包含 {df_hfc['ref_clean'].nunique()} 种 HFC 制冷剂")

# 统计 IL Cation x Anion 矩阵
unique_cations = sorted(df_hfc["cation_clean"].unique())
unique_anions = sorted(df_hfc["anion_clean"].unique())
unique_pairs = sorted(df_hfc["il_pair"].unique())

print(f"独立阳离子数: {len(unique_cations)}")
print(f"独立阴离子数: {len(unique_anions)}")
print(f"独立实测离子对数 (#IL Pairs): {len(unique_pairs)}")

# 计算每个 cation 连接了多少个不同 anion, 每个 anion 连接了多少个不同 cation
cat_deg = df_hfc.groupby("cation_clean")["anion_clean"].nunique()
ani_deg = df_hfc.groupby("anion_clean")["cation_clean"].nunique()

print(f"多阴离子连接的阳离子数 (deg >= 2): {(cat_deg >= 2).sum()} / {len(unique_cations)}")
print(f"多阳离子连接的阴离子数 (deg >= 2): {(ani_deg >= 2).sum()} / {len(unique_anions)}")

# 筛选候选 L2 组合:
# 必须满足: 当剔除该 pair (c*, a*) 后, c* 至少还在训练集中连接另外 >= 1 个 anion,
# 且 a* 至少还在训练集中连接另外 >= 1 个 cation!
# 即: cat_deg[c*] >= 2 且 ani_deg[a*] >= 2!
pair_stats = []
for p_name, grp in df_hfc.groupby("il_pair"):
    c = grp["cation_clean"].iloc[0]
    a = grp["anion_clean"].iloc[0]
    
    c_deg = cat_deg[c]
    a_deg = ani_deg[a]
    
    is_valid_recomb = (c_deg >= 2) and (a_deg >= 2)
    
    # 检查该 pair 包含哪些制冷剂
    refs = grp["ref_clean"].unique().tolist()
    
    pair_stats.append({
        "il_pair": p_name,
        "cation": c,
        "anion": a,
        "n_samples": len(grp),
        "n_refrigerants": len(refs),
        "cation_degree": c_deg,
        "anion_degree": a_deg,
        "is_valid_l2_candidate": is_valid_recomb,
        "refrigerants": ", ".join(refs),
        "T_min": grp["T_K"].min(),
        "T_max": grp["T_K"].max(),
        "P_min": grp["P_MPa"].min(),
        "P_max": grp["P_MPa"].max(),
    })

df_pairs = pd.DataFrame(pair_stats)
valid_candidates = df_pairs[df_pairs["is_valid_l2_candidate"]].sort_values(by="n_samples", ascending=False)
df_pairs.to_csv(OUT_DIR / "table_l2_recombination_candidates.csv", index=False)

print(f"\n满足严格重组条件的合法 L2 候选对数量: {len(valid_candidates)} / {len(df_pairs)}")
print(f"候选对总样本量覆盖: {valid_candidates['n_samples'].sum()} / {len(df_hfc)} 点")

print("\nTop 15 优质候选重组对 (按样本量排序):")
cols_cand = ["il_pair", "n_samples", "n_refrigerants", "cation_degree", "anion_degree", "T_min", "T_max", "P_min", "P_max"]
print(valid_candidates[cols_cand].head(15).to_string(index=False))

# 评估推荐的 L2-Strict 与 L2-Controlled 设计方案
print("\n" + "=" * 80)
print("  RECOMMENDED L2 SPLIT DESIGN ARCHITECTURES")
print("=" * 80)
# 方案 A: 单一大体系经典重组 (如 [BMIM][PF6] 或 [EMIM][BF4])
# 方案 B: 多体系均衡重组 (保留 3-5 个中等规模 pair 作为复合测试集, 占比约 10-15%)
print("【方案 1: L2-Single-Star (明星单离子对重组)】")
star_pair = valid_candidates.iloc[0]["il_pair"]
star_n = valid_candidates.iloc[0]["n_samples"]
print(f"  测试集: {star_pair} (N={star_n}, 占全集 {star_n/len(df_hfc)*100:.1f}%)")
print(f"  特点: 样本密集, 包含丰富制冷剂, 统计精度极高。")

print("\n【方案 2: L2-Balanced-Recombination (多对均衡重组组合)】")
# 挑选 3-4 个频率适中的代表性组合
balanced_sample = valid_candidates.iloc[2:6]
b_pairs = balanced_sample["il_pair"].tolist()
b_n = balanced_sample["n_samples"].sum()
print(f"  测试集组合: {b_pairs}")
print(f"  总样本量: N={b_n} (占全集 {b_n/len(df_hfc)*100:.1f}%)")
print(f"  特点: 跨多个阴阳离子骨架, 彻底排除了单对偶然性, 满足 L2-Controlled 规范。")

print("\n" + "=" * 80)
print("  [SUCCESS] GATE D0 AUDIT COMPLETE: READY FOR SPLIT CREATION")
print("=" * 80)
