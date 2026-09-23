"""
audit_b42_vs_a42.py — Four-Layer Mechanistic & Paired Error Audit: V7-B(42) vs V7-A(42)
=======================================================================================
Reads:
- V7-A(42): D:/折腾/v7_pilot_gpu_results/v7_shadow_experiment/checkpoints/V7-A/val_predictions_seed_42.csv
- V7-B(42): D:/折腾/v7_pilot_gpu_results1/v7_shadow_experiment/checkpoints/V7-B/val_predictions_seed_42.csv
- Meta: processed_tri_data_hfc2739/meta_info.csv
Computes:
1. Metric summary table (V6 vs V7-A vs V7-B)
2. Paired per-sample error differences (ΔAE = AE_A - AE_B)
3. Bootstrap 95% CI and Wilcoxon test
4. Refrigerant-by-refrigerant breakdown (specifically targeting R134, R245fa, R23)
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

def bootstrap_ci(arr: np.ndarray, n_boot=2000, ci=95):
    boot_means = []
    boot_medians = []
    np.random.seed(42)
    n = len(arr)
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
        boot_medians.append(np.median(sample))
    
    alpha = (100 - ci) / 2.0
    mean_ci = (np.percentile(boot_means, alpha), np.percentile(boot_means, 100 - alpha))
    med_ci = (np.percentile(boot_medians, alpha), np.percentile(boot_medians, 100 - alpha))
    return mean_ci, med_ci

def main():
    dir_a = Path(r"D:\折腾\v7_pilot_gpu_results\v7_shadow_experiment")
    dir_b = Path(r"D:\折腾\v7_pilot_gpu_results1\v7_shadow_experiment")
    split_file = ROOT / "splits" / "HFC_all_split.npz"
    data_dir = ROOT / "processed_tri_data_hfc2739"

    sp = np.load(split_file)
    val_idx = sp['val'].astype(int)
    meta_df = pd.read_csv(data_dir / "meta_info.csv").iloc[val_idx].reset_index(drop=True)

    # Load predictions
    df_a = pd.read_csv(dir_a / "checkpoints" / "V7-A" / "val_predictions_seed_42.csv")
    df_b = pd.read_csv(dir_b / "checkpoints" / "V7-B" / "val_predictions_seed_42.csv")

    targets = df_a['target'].values
    preds_a = df_a['pred'].values
    preds_b = df_b['pred'].values

    # Paired errors
    ae_a = np.abs(targets - preds_a)
    ae_b = np.abs(targets - preds_b)
    delta_ae = ae_a - ae_b  # >0 means B is better, <0 means A is better

    mae_a = float(np.mean(ae_a))
    mae_b = float(np.mean(ae_b))
    r2_a = float(1.0 - np.sum((targets - preds_a)**2) / np.sum((targets - np.mean(targets))**2))
    r2_b = float(1.0 - np.sum((targets - preds_b)**2) / np.sum((targets - np.mean(targets))**2))

    win_count = int(np.sum(delta_ae > 0))
    loss_count = int(np.sum(delta_ae < 0))
    win_rate = win_count / len(delta_ae) * 100.0

    mean_ci, med_ci = bootstrap_ci(delta_ae)
    _, wilcoxon_p = stats.wilcoxon(ae_a, ae_b)

    print("=" * 80)
    print("  LAYER 1 & 2: V7-A(42) vs V7-B(42) PAIRED PERFORMANCE COMPARISON (N=274)")
    print("=" * 80)
    print(f"V7-A(42) MAE : {mae_a:.5f} | R2: {r2_a:.5f}")
    print(f"V7-B(42) MAE : {mae_b:.5f} | R2: {r2_b:.5f} (Δ = {mae_b - mae_a:+.5f}, {(mae_b - mae_a)/mae_a*100:+.2f}%)")
    print(f"Win Rate (B > A) : {win_rate:.2f}% ({win_count} / {len(delta_ae)}) | Losses: {loss_count}")
    print(f"Median ΔAE       : {np.median(delta_ae):+.5f} [95% CI: {med_ci[0]:+.5f}, {med_ci[1]:+.5f}]")
    print(f"Mean ΔAE         : {np.mean(delta_ae):+.5f} [95% CI: {mean_ci[0]:+.5f}, {mean_ci[1]:+.5f}]")
    print(f"Wilcoxon p-value : {wilcoxon_p:.4e}")

    # Chemical Breakdown
    meta_df['ae_a'] = ae_a
    meta_df['ae_b'] = ae_b
    meta_df['delta_ae'] = delta_ae

    print("\n" + "=" * 80)
    print("  LAYER 3: DISSECTION BY REFRIGERANT — V7-A vs V7-B")
    print("=" * 80)
    ref_grp = meta_df.groupby('refrigerant').agg(
        N=('x1', 'count'),
        A_MAE=('ae_a', 'mean'),
        B_MAE=('ae_b', 'mean'),
        Win_Rate=('delta_ae', lambda x: np.mean(x > 0) * 100.0),
        Mean_ΔAE=('delta_ae', 'mean')
    ).reset_index()
    ref_grp['Rel_Change_Pct'] = (ref_grp['B_MAE'] - ref_grp['A_MAE']) / ref_grp['A_MAE'] * 100.0
    ref_grp = ref_grp.sort_values(by='Rel_Change_Pct')
    print(ref_grp.to_string(index=False))

    # Top regressions (samples where B was notably worse than A)
    meta_df['pred_a'] = preds_a
    meta_df['pred_b'] = preds_b
    worst_regressions = meta_df.sort_values(by='delta_ae').head(8)
    print("\n" + "=" * 80)
    print("  CRITICAL AUDIT: TOP 8 REGRESSIONS (Where V7-B was notably worse than V7-A)")
    print("=" * 80)
    print(worst_regressions[['refrigerant', 'cation', 'anion', 'T_K', 'P_MPa', 'x1', 'pred_a', 'pred_b', 'ae_a', 'ae_b', 'delta_ae']].to_string(index=False))

    # Top improvements (samples where B was notably better than A)
    top_improvements = meta_df.sort_values(by='delta_ae', ascending=False).head(8)
    print("\n" + "=" * 80)
    print("  CRITICAL AUDIT: TOP 8 IMPROVEMENTS (Where V7-B was notably better than V7-A)")
    print("=" * 80)
    print(top_improvements[['refrigerant', 'cation', 'anion', 'T_K', 'P_MPa', 'x1', 'pred_a', 'pred_b', 'ae_a', 'ae_b', 'delta_ae']].to_string(index=False))

    out_csv = ROOT / "v7_shadow_experiment" / "b42_vs_a42_paired_dissection.csv"
    meta_df.to_csv(out_csv, index=False)
    print(f"\nSaved dissection to: {out_csv}")

if __name__ == '__main__':
    main()
