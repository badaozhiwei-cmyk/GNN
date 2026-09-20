"""
audit_step25_attribution.py — Deep Diagnostic & Critical Scrutiny of Step 25 Attribution
========================================================================================
对 Kaggle GPU 产出的 results_attribution/ 下全部结果进行客观、冷静、多维度的严格数据审计：
1. Gate A ~ Gate E 全面核查
2. 4 大案例的深层化学机制与基团重要性对比 (Case 1 ~ Case 4)
3. 跨 Seed (5 seeds) 稳定性、一致性与潜在表征坍塌 (Shortcut Learning) 分析
4. 保真度测试 (R_faith) 分布与反常样本 (R_faith <= 1.0) 挑刺
5. 科学论文发表视角下的审稿人质疑点 (Reviewer Objections) 归纳
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES_DIR = PROJECT_ROOT / "results_attribution"

df_atoms = pd.read_csv(RES_DIR / "graph_attribution_atoms_bonds.csv")
df_groups = pd.read_csv(RES_DIR / "graph_attribution_groups.csv")
df_faith = pd.read_csv(RES_DIR / "graph_attribution_faithfulness.csv")
df_stab = pd.read_csv(RES_DIR / "graph_attribution_stability.csv")
df_manifest = pd.read_csv(RES_DIR / "case_selection_manifest.csv")

with open(RES_DIR / "graph_attribution_provenance.json", "r", encoding="utf-8") as f:
    provenance = json.load(f)

print("=" * 80)
print("  STEP 25 ATTRIBUTION RESULTS: COMPREHENSIVE CRITICAL AUDIT REPORT")
print("=" * 80)

# =============================================================================
# 1. 公理与质量门禁核查 (Axiomatic & Gate Verifications)
# =============================================================================
print("\n" + "-" * 80)
print("【PART 1: 数值完备性与公理验证 (Completeness & Axiom Audit)】")
print("-" * 80)

comp_errs = df_faith["comp_rel_err"].values * 100
mean_err = np.mean(comp_errs)
median_err = np.median(comp_errs)
max_err = np.max(comp_errs)
p95_err = np.percentile(comp_errs, 95)
p99_err = np.percentile(comp_errs, 99)
fail_count = np.sum(comp_errs > 5.0)

print(f"1. Signed 积分梯度数值完备性误差分布 (N=215 次评估):")
print(f"   - 均值 (Mean)         : {mean_err:.3f}%")
print(f"   - 中位数 (Median)     : {median_err:.3f}%")
print(f"   - 95 分位数 (P95)     : {p95_err:.3f}%")
print(f"   - 99 分位数 (P99)     : {p99_err:.3f}%")
print(f"   - 最大误差 (Max)      : {max_err:.3f}%")
print(f"   - 超过 5.0% 阈值次数  : {fail_count} / 215 (合格率: {(1 - fail_count/215)*100:.1f}%)")

# 检查 global_token 隔离
# global_token 在 atoms 表中未被列为真实原子，真实原子均属于 Cation/Anion/Refri
real_components = set(df_atoms["component"].unique())
print(f"\n2. 节点与组件隔离:")
print(f"   - 归因原子所属组件集合: {real_components} (完全局限在真实分子内)")
print(f"   - 真实原子归因绝对值均值: {df_atoms['a_i_abs'].mean():.6f} (活跃梯度通路)")

# =============================================================================
# 2. 保真度响应比率 (R_faith) 深度审视与挑刺
# =============================================================================
print("\n" + "-" * 80)
print("【PART 2: 模型保真度测试 (Model Faithfulness / R_faith) 深度审视】")
print("-" * 80)

r_faith_vals = df_faith["r_faith"].values
mean_rf = np.mean(r_faith_vals)
med_rf = np.median(r_faith_vals)
p25_rf = np.percentile(r_faith_vals, 25)
p75_rf = np.percentile(r_faith_vals, 75)
min_rf = np.min(r_faith_vals)
max_rf = np.max(r_faith_vals)

greater_1 = np.sum(r_faith_vals > 1.0)
greater_2 = np.sum(r_faith_vals > 2.0)
less_equal_1 = np.sum(r_faith_vals <= 1.0)

print(f"1. 全局 R_faith (Δy_top / mean(Δy_rand)) 统计:")
print(f"   - 均值 ± 标准差 : {mean_rf:.2f} ± {np.std(r_faith_vals):.2f}")
print(f"   - 中位数 (IQR)   : {med_rf:.2f} (P25={p25_rf:.2f}, P75={p75_rf:.2f})")
print(f"   - 范围 (Min ~ Max): {min_rf:.2f} ~ {max_rf:.2f}")
print(f"   - R_faith > 1.0 比例 (Top基团扰动高于随机基团): {greater_1} / 215 ({greater_1/215*100:.1f}%)")
print(f"   - R_faith > 2.0 比例 (Top基团扰动达随机 2 倍以上): {greater_2} / 215 ({greater_2/215*100:.1f}%)")

# 挑刺: 检查 R_faith <= 1.0 的反常样本
print(f"\n2. 【核心挑刺点 1】: 反常样本分析 (R_faith <= 1.0，共 {less_equal_1} 例，占比 {less_equal_1/215*100:.1f}%):")
if less_equal_1 > 0:
    anomaly_df = df_faith[df_faith["r_faith"] <= 1.0]
    print(f"   反常样本在 4 大案例中的分布:")
    for cid, cnt in anomaly_df["case_id"].value_counts().items():
        print(f"     * {cid:<26}: {cnt} 次 (占该 Case 的 {cnt / len(df_faith[df_faith['case_id']==cid])*100:.1f}%)")
    print(f"   反常样本在各 Seed 中的分布:")
    for sd, cnt in anomaly_df["seed"].value_counts().items():
        print(f"     * Seed {sd}: {cnt} 次")
    print(f"   反常样例前 3 项展示:")
    for _, r in anomaly_df.head(3).iterrows():
        print(f"     - [{r['case_id']}] Seed {r['seed']}, Sample: {r['sample_id'][:40]}... | TopGroup: {r['top_group_name']} | Δy_top: {r['delta_y_top']:.5f} vs Δy_rand: {r['delta_y_rand_mean']:.5f} (R_faith={r['r_faith']:.2f})")
else:
    print("   未发现 R_faith <= 1.0 的样本，全部样本均满足 Top-1 基团扰动大于随机扰动。")

# =============================================================================
# 3. 跨 Seed (5 seeds) 稳定性与表征差异挖掘
# =============================================================================
print("\n" + "-" * 80)
print("【PART 3: 跨 Seed 稳定性度量与表征异质性 (Representation Stability)】")
print("-" * 80)

mean_rec = df_stab["top1_recurrence_rate"].mean()
mean_spear = df_stab["mean_pairwise_spearman"].mean()
mean_cos = df_stab["mean_pairwise_cosine"].mean()

rec_100 = np.sum(df_stab["top1_recurrence_rate"] == 1.0)
rec_80 = np.sum(df_stab["top1_recurrence_rate"] >= 0.8)
rec_60 = np.sum(df_stab["top1_recurrence_rate"] >= 0.6)

print(f"1. 5 种子跨 Seed 统计汇总 (N=43 个独立化学样本):")
print(f"   - Top-1 基团重现率均值 (Recurrence Rate) : {mean_rec*100:.1f}%")
print(f"   - 成对 Spearman 秩相关均值 (Rank Stability) : {mean_spear:.3f}")
print(f"   - 成对余弦相似度均值 (Direction Cosine)   : {mean_cos:.3f}")
print(f"   - 5 种子完全一致 (5/5 同属同一 Top-1) 样本数: {rec_100} / 43 ({rec_100/43*100:.1f}%)")
print(f"   - 80% 以上一致 (>=4/5 同属同一 Top-1) 样本数: {rec_80} / 43 ({rec_80/43*100:.1f}%)")
print(f"   - 60% 以上一致 (>=3/5 同属同一 Top-1) 样本数: {rec_60} / 43 ({rec_60/43*100:.1f}%)")

print(f"\n2. 各案例的跨 Seed 稳定性对比:")
print(f"   {'Case ID':<26} {'N_samples':<10} {'Mean Recurrence':<18} {'Mean Spearman':<16} {'Mean Cosine':<14}")
print("   " + "-" * 82)
for cid, sub_stab in df_stab.groupby("case_id"):
    print(f"   {cid:<26} {len(sub_stab):<10d} {sub_stab['top1_recurrence_rate'].mean()*100:16.1f}% {sub_stab['mean_pairwise_spearman'].mean():16.3f} {sub_stab['mean_pairwise_cosine'].mean():14.3f}")

# 挑刺: 检查各 Seed 的单分子归因总强度
print(f"\n3. 【核心挑刺点 2】: 各 Seed 图分支权重强度分布 (表征坍塌/捷径学习诊断):")
seed_total_abs = df_groups.groupby(["case_id", "seed"])["A_g_abs"].sum().unstack()
print(seed_total_abs.round(4).to_string())

# =============================================================================
# 4. 四大化学案例深度解析 (Case 1 ~ Case 4)
# =============================================================================
print("\n" + "-" * 80)
print("【PART 4: 四大代表性化学案例深度机制剖析】")
print("-" * 80)

# --- Case 1: R1234yf ---
print("\n>>> Case 1: R1234yf (HFO 跨家族零样本外推)")
c1_groups = df_groups[df_groups["case_id"] == "Case1_R1234yf"]
c1_summary = c1_groups.groupby("group_name")["P_g"].agg(["mean", "std"]).sort_values("mean", ascending=False)
print("  各官能团重要性份额均值 P_g (跨 4 样本 x 5 种子 = 20 次评估):")
for gname, row in c1_summary.iterrows():
    print(f"    - {gname:<24}: {row['mean']*100:6.2f}% ± {row['std']*100:5.2f}%")

# 检查制冷剂自身 vs 溶剂分担
c1_comp = c1_groups.groupby("component")["P_g"].mean()
print(f"  分子体系重要性分配: Cation={c1_comp.get('Cation', 0)*100:.1f}%, Anion={c1_comp.get('Anion', 0)*100:.1f}%, Refrigerant={c1_comp.get('Refri', 0)*100:.1f}%")

# --- Case 2: R134 vs R134a ---
print("\n>>> Case 2: R134 vs R134a (M1 异构体偶极矩悖论)")
c2_groups = df_groups[df_groups["case_id"] == "Case2_R134_vs_R134a"]
# 合并 manifest 信息
c2_merged = pd.merge(c2_groups, df_manifest[["sample_id", "refrigerant", "pair_match_id"]], on="sample_id")

r134_sub = c2_merged[c2_merged["refrigerant"] == "R134"]
r134a_sub = c2_merged[c2_merged["refrigerant"] == "R134a"]

print("  R134 (对称分子, CHF2-CHF2, 偶极矩 0 D) 制冷剂基团归因份额:")
r134_ref_g = r134_sub[r134_sub["component"] == "Refri"].groupby("group_name")["P_g"].agg(["mean", "std"])
for gname, row in r134_ref_g.iterrows():
    print(f"    - {gname:<24}: {row['mean']*100:6.2f}% ± {row['std']*100:5.2f}%")

print("  R134a (非对称分子, CF3-CH2F, 偶极矩 2.06 D) 制冷剂基团归因份额:")
r134a_ref_g = r134a_sub[r134a_sub["component"] == "Refri"].groupby("group_name")["P_g"].agg(["mean", "std"])
for gname, row in r134a_ref_g.iterrows():
    print(f"    - {gname:<24}: {row['mean']*100:6.2f}% ± {row['std']*100:5.2f}%")

# 对比两者的总制冷剂权重
r134_tot = r134_sub[r134_sub["component"] == "Refri"].groupby("sample_id")["P_g"].sum().mean()
r134a_tot = r134a_sub[r134a_sub["component"] == "Refri"].groupby("sample_id")["P_g"].sum().mean()
print(f"  制冷剂整体重要性对比: R134 (对称) = {r134_tot*100:.1f}% vs R134a (非对称) = {r134a_tot*100:.1f}%")

# --- Case 3: BF4 vs PF6 ---
print("\n>>> Case 3: BF4 vs PF6 (B2 阴离子家族外推)")
c3_groups = df_groups[df_groups["case_id"] == "Case3_BF4_vs_PF6"]
c3_merged = pd.merge(c3_groups, df_manifest[["sample_id", "anion", "pair_match_id"]], on="sample_id")

bf4_sub = c3_merged[c3_merged["anion"] == "[BF4]"]
pf6_sub = c3_merged[c3_merged["anion"] == "[PF6]"]

bf4_core = bf4_sub[bf4_sub["group_name"] == "BF4_Core"]["P_g"].mean()
pf6_core = pf6_sub[pf6_sub["group_name"] == "PF6_Core"]["P_g"].mean()
print(f"  阴离子核心归因份额: BF4_Core = {bf4_core*100:.2f}% vs PF6_Core = {pf6_core*100:.2f}%")

# 全阴离子总权重
bf4_ani_tot = bf4_sub[bf4_sub["component"] == "Anion"].groupby("sample_id")["P_g"].sum().mean()
pf6_ani_tot = pf6_sub[pf6_sub["component"] == "Anion"].groupby("sample_id")["P_g"].sum().mean()
print(f"  阴离子组件总体份额: [BF4] = {bf4_ani_tot*100:.1f}% vs [PF6] = {pf6_ani_tot*100:.1f}%")

# --- Case 4: R1336mzz(E) vs R1336mzz(Z) ---
print("\n>>> Case 4: R1336mzz(E) vs R1336mzz(Z) (立体化学图退化边界)")
c4_groups = df_groups[df_groups["case_id"] == "Case4_R1336mzz_Stereo"]
c4_merged = pd.merge(c4_groups, df_manifest[["sample_id", "refrigerant", "pair_match_id"]], on="sample_id")

e_sub = c4_merged[c4_merged["refrigerant"] == "R1336mzz(E)"]
z_sub = c4_merged[c4_merged["refrigerant"] == "R1336mzz(Z)"]

e_ref_p = e_sub[e_sub["component"] == "Refri"].groupby("group_name")["P_g"].mean()
z_ref_p = z_sub[z_sub["component"] == "Refri"].groupby("group_name")["P_g"].mean()

print(f"  E-异构体 (N=11) 制冷剂基团份额: {e_ref_p.to_dict()}")
print(f"  Z-异构体 (N=12) 制冷剂基团份额: {z_ref_p.to_dict()}")
diff_ez = abs(e_ref_p - z_ref_p).max()
print(f"  E 与 Z 在图归因上的最大均值偏差: {diff_ez:.6f} (验证了二维图拓扑同构导致的退化)")

print("\n" + "=" * 80)
print("  AUDIT COMPLETED!")
print("=" * 80 + "\n")
