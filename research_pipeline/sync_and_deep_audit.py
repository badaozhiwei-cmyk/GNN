"""
sync_and_deep_audit.py — Synchronize GPU outputs and execute cold critical audit
==============================================================================
"""

import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path("D:/折腾/可解释性")

# 1. 复制并重命名为标准 csv
shutil.copy2(SRC_DIR / "graph_attribution_atoms_bonds.txt", ROOT / "results_attribution/graph_attribution_atoms_bonds.csv")
shutil.copy2(SRC_DIR / "graph_attribution_faithfulness.txt", ROOT / "results_attribution/graph_attribution_faithfulness.csv")
shutil.copy2(SRC_DIR / "graph_attribution_groups.txt", ROOT / "results_attribution/graph_attribution_groups.csv")
shutil.copy2(SRC_DIR / "step25_integrity_audit_v2.txt", ROOT / "diagnostic_outputs/step25_integrity_audit_v2.csv")
shutil.copy2(SRC_DIR / "stratified_activity_summary.txt", ROOT / "diagnostic_outputs/stratified_activity_summary.csv")

print("[INFO] 5 个文件已成功同步至本地工作区并规范化为 .csv！")

df_v2 = pd.read_csv(ROOT / "diagnostic_outputs/step25_integrity_audit_v2.csv")
df_groups = pd.read_csv(ROOT / "results_attribution/graph_attribution_groups.csv")
df_strat = pd.read_csv(ROOT / "diagnostic_outputs/stratified_activity_summary.csv")

print("=" * 85)
print("  STEP 25 INTEGRITY AUDIT V2: COLD, UNBIASED & CRITICAL DEEP DIVE")
print("=" * 85)

# -----------------------------------------------------------------------------
# 1. 宏观分层与各 Seed 活跃性渗透率
# -----------------------------------------------------------------------------
print("\n" + "-" * 85)
print("【维度 1: 5-Seed 活跃性渗透率分布 (哪些 Seed 真正学到了图？)】")
print("-" * 85)
seed_activity = pd.crosstab(df_v2["seed"], df_v2["graph_active_flag"], margins=True)
seed_activity.columns = ["Inactive (False)", "Active (True)", "Total"]
print(seed_activity)

print("\n各 Case 在 5 个 Seed 中的活跃样本数 (Active / Total):")
case_seed_act = df_v2.groupby(["case_id", "seed"])["graph_active_flag"].sum().unstack()
case_seed_tot = df_v2.groupby(["case_id", "seed"])["graph_active_flag"].count().unstack()
summary_str = ""
for cid in case_seed_act.index:
    print(f"  * {cid:<26}: Seed 42={case_seed_act.loc[cid, 42]}/{case_seed_tot.loc[cid, 42]} | Seed 43={case_seed_act.loc[cid, 43]}/{case_seed_tot.loc[cid, 43]} | Seed 44={case_seed_act.loc[cid, 44]}/{case_seed_tot.loc[cid, 44]} | Seed 45={case_seed_act.loc[cid, 45]}/{case_seed_tot.loc[cid, 45]} | Seed 46={case_seed_act.loc[cid, 46]}/{case_seed_tot.loc[cid, 46]}")

# -----------------------------------------------------------------------------
# 1.5 阈值尺度敏感度与双峰分离检验 (Threshold Sensitivity & Bimodal Separation)
# -----------------------------------------------------------------------------
print("\n" + "-" * 85)
print("【维度 1.5: 判定阈值稳健性检验 (不同阈值下的活跃样本数扫描)】")
print("-" * 85)
sens_series = df_v2["graph_output_sensitivity"]
print("敏感度分位数分布:")
for q, val in sens_series.quantile([0.0, 0.5, 0.75, 0.8, 0.83, 0.84, 0.85, 0.9, 1.0]).items():
    print(f"  * P{int(q*100):02d} = {val:.6e}")

print("\n在不同阈值 Tau 下被判定为 Graph-Active 的样本数:")
for tau in [1e-8, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2]:
    n_act = int((sens_series >= tau).sum())
    print(f"  * Tau = {tau:.0e} : Active = {n_act:>3d} / 215 (占 {n_act/215*100:5.2f}%)")

# -----------------------------------------------------------------------------
# 2. 挑刺点 A: 为什么活跃组的 R_faith 暴增至 146？(数值死区与极小分母诊断)
# -----------------------------------------------------------------------------
print("\n" + "-" * 85)
print("【维度 2: 活跃组保真度数值病态 (Pathological R_faith) 挑刺分析】")
print("-" * 85)
active_df = df_v2[df_v2["graph_active_flag"] == True]
print(f"活跃子网络 (N={len(active_df)}) 的保真度数值分布:")
print(f"  - delta_y_top (Top-1 基团遮蔽预测偏差):")
print(f"      均值={active_df['delta_y_top'].mean():.5f}, 中位数={active_df['delta_y_top'].median():.5f}, Min={active_df['delta_y_top'].min():.5f}, Max={active_df['delta_y_top'].max():.5f}")
print(f"  - delta_y_rand_comp (同组分随机原子遮蔽平均偏差):")
print(f"      均值={active_df['delta_y_rand_comp'].mean():.5f}, 中位数={active_df['delta_y_rand_comp'].median():.5f}, Min={active_df['delta_y_rand_comp'].min():.5f}, Max={active_df['delta_y_rand_comp'].max():.5f}")
print(f"  - R_faith_comp (delta_y_top / delta_y_rand_comp):")
print(f"      中位数={active_df['r_faith_component_matched'].median():.2f}, 均值={active_df['r_faith_component_matched'].mean():.2f}, P25={active_df['r_faith_component_matched'].quantile(0.25):.2f}, P75={active_df['r_faith_component_matched'].quantile(0.75):.2f}")
print(f"      Max={active_df['r_faith_component_matched'].max():.2f}, Min={active_df['r_faith_component_matched'].min():.2f}")

# 挑刺: 检查 R_faith 极大值的成因
extreme_rf = active_df[active_df["r_faith_component_matched"] > 50]
print(f"\n  [挑刺实锤]: R_faith > 50 的极端样本有 {len(extreme_rf)} 例 (占活跃组 {len(extreme_rf)/len(active_df)*100:.1f}%)!")
print("  极端样例展示:")
for _, r in extreme_rf.head(3).iterrows():
    print(f"    - [{r['case_id']}] Seed {r['seed']}, {r['sample_id'][:35]}... | Top: {r['top_group_component']}:{r['top_group_name']} | Δy_top: {r['delta_y_top']:.5f} vs Δy_rand: {r['delta_y_rand_comp']:.6f} -> R_faith={r['r_faith_component_matched']:.2f}")

# -----------------------------------------------------------------------------
# 3. 挑刺点 B: 180 次 Inactive 评估的具体“死因”
# -----------------------------------------------------------------------------
print("\n" + "-" * 85)
print("【维度 3: 180 次非活跃评估的具体机制死因 (Mechanism of Failure)】")
print("-" * 85)
inactive_df = df_v2[df_v2["graph_active_flag"] == False]
reasons = inactive_df["graph_activity_reason"].value_counts()
for r_name, cnt in reasons.items():
    print(f"  - {r_name:<50}: {cnt:>3d} 次 ({cnt/len(inactive_df)*100:5.1f}%)")

# 检查逐层衰减
print(f"\n  各层全局节点差量均值 (Inactive 组):")
print(f"    * L1 Diff Mean: {inactive_df['global_node_sensitivity_L1'].mean():.6e}")
print(f"    * L2 Diff Mean: {inactive_df['global_node_sensitivity_L2'].mean():.6e}")
print(f"    * L3 Diff Mean: {inactive_df['global_node_sensitivity_L3'].mean():.6e}")

# -----------------------------------------------------------------------------
# 4. 活跃子网络下的四大案例化学归因 (真实化学信号核定)
# -----------------------------------------------------------------------------
print("\n" + "-" * 85)
print("【维度 4: 仅在活跃子网络 (Graph-Active, N=35) 上的化学归因再核算】")
print("-" * 85)
active_samples = active_df[["case_id", "sample_id", "seed"]]
merged_active_groups = pd.merge(df_groups, active_samples, on=["case_id", "sample_id", "seed"])

for cid, sub in merged_active_groups.groupby("case_id"):
    print(f"\n>>> {cid} (活跃评估次数: {len(sub.groupby(['sample_id', 'seed']))}):")
    top_g = sub.groupby(["component", "group_name"])["P_g"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    for (comp, gname), row in top_g.head(5).iterrows():
        print(f"    * [{comp:<6}] {gname:<22}: {row['mean']*100:6.2f}% ± {row['std']*100:5.2f}%")
