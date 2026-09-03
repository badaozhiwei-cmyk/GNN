"""
cluster_bootstrap_delta_mae.py — 论文级跨物质消融差异 Cluster Bootstrap 与非参数显著性检验
实现：
1. 主统计量：Delta MAE = MAE(M0) - MAE(Mreduced)
2. 10,000 次 Cluster Bootstrap (按 12 个物质有放回抽样)，计算 95% 置信区间
3. 配对 Wilcoxon 符号秩检验 (Wilcoxon Signed-Rank Test)
4. 配对二项式符号检验 (Binomial Sign Test)
5. 稳健中位数差值 Bootstrap (Delta MedianAE)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

def run_significance_test(csv_path='paper_results/recovered_overnight_results.csv', n_bootstraps=10000, seed=42):
    print("=" * 85)
    print("🔬 启动严密统计检验：Delta MAE 物质簇级 Bootstrap 与非参数显著性分析")
    print("=" * 85)
    
    df = pd.read_csv(csv_path)
    m0 = df[df['Mode'] == 'M0'].set_index('Target')['MAE_mean']
    mr = df[df['Mode'] == 'Mreduced'].set_index('Target')['MAE_mean']
    
    common_targets = [t for t in m0.index if t in mr.index]
    targets = [t.replace('loro_', '') for t in common_targets]
    
    mae_m0 = np.array([m0.loc[t] for t in common_targets])
    mae_mr = np.array([mr.loc[t] for t in common_targets])
    delta_mae = mae_m0 - mae_mr # 正值表示 Mreduced 优于 M0
    
    n_refs = len(delta_mae)
    print(f"参与配对检验的制冷剂总数: {n_refs} 种")
    
    # 1. 10,000 次 Cluster Bootstrap
    rng = np.random.RandomState(seed)
    boot_means = []
    boot_medians = []
    for _ in range(n_bootstraps):
        sample_idx = rng.choice(n_refs, size=n_refs, replace=True)
        boot_means.append(np.mean(delta_mae[sample_idx]))
        boot_medians.append(np.median(delta_mae[sample_idx]))
        
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    
    ci_med_lower = np.percentile(boot_medians, 2.5)
    ci_med_upper = np.percentile(boot_medians, 97.5)
    
    # 2. 配对 Wilcoxon 符号秩检验
    w_stat, w_pval = wilcoxon(mae_m0, mae_mr, alternative='greater')
    
    # 3. 二项式符号检验 (Sign Test)
    wins = np.sum(delta_mae > 0)
    sign_res = binomtest(k=wins, n=n_refs, p=0.5, alternative='greater')
    
    print("\n" + "=" * 85)
    print("📋 统计检验汇总结果 (严密审稿级指标)")
    print("=" * 85)
    print(f"1. 平均误差改善幅度 (Mean Delta MAE):       {np.mean(delta_mae):+.4f}")
    print(f"   - Cluster Bootstrap 95% 置信区间:        [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"   - 零假说排除判定 (Zero in 95% CI?):      {'❌ 否 (95% CI 完全严格位于正半轴，极其显著！)' if ci_lower > 0 else '⚠️ 是'}")
    print(f"\n2. 中位数改善幅度 (Median Delta MAE):     {np.median(delta_mae):+.4f}")
    print(f"   - Median Bootstrap 95% 置信区间:         [{ci_med_lower:+.4f}, {ci_med_upper:+.4f}]")
    print(f"\n3. 胜率与符号检验 (Sign Test):")
    print(f"   - 正向改善物质占比:                     {wins}/{n_refs} ({wins/n_refs*100:.1f}%)")
    print(f"   - 二项检验 p 值 (Binomial p-value):     p = {sign_res.pvalue:.4e} {'(p < 0.05 显著)' if sign_res.pvalue < 0.05 else ''}")
    print(f"\n4. 配对 Wilcoxon 秩检验 (Wilcoxon Signed-Rank Test):")
    print(f"   - 检验统计量 W:                         W = {w_stat:.1f}")
    print(f"   - 单尾显著性 p 值 (One-tailed p-value):  p = {w_pval:.4e} {'(p < 0.01 高度显著)' if w_pval < 0.01 else ''}")
    print("=" * 85)
    
    # 输出表格保存
    res_df = pd.DataFrame([{
        'Metric': 'Delta MAE (M0 - Mreduced)',
        'Mean': np.mean(delta_mae),
        'CI95_Lower': ci_lower,
        'CI95_Upper': ci_upper,
        'Median': np.median(delta_mae),
        'Median_CI95_Lower': ci_med_lower,
        'Median_CI95_Upper': ci_med_upper,
        'Win_Count': f"{wins}/{n_refs}",
        'Win_Rate(%)': wins/n_refs*100.0,
        'Binomial_p': sign_res.pvalue,
        'Wilcoxon_p': w_pval,
        'Statistically_Significant': ci_lower > 0 and w_pval < 0.05
    }])
    res_df.to_csv('paper_results/table_cluster_bootstrap_significance.csv', index=False)
    print("✅ 统计检验表已保存至 paper_results/table_cluster_bootstrap_significance.csv！")

if __name__ == '__main__':
    run_significance_test()
