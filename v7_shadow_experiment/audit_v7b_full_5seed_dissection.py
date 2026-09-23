"""
audit_v7b_full_5seed_dissection.py — Full 5-Seed Paired Mechanistic Audit: V7-B vs V7-A
=======================================================================================
Reads:
- V7-A: D:/折腾/v7_pilot_gpu_results/v7_shadow_experiment/
- V7-B: D:/折腾/v7_pilot_gpu_results2/v7_shadow_experiment/
- V6:   results_hfc_all/HFC_all_M0/
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
    dir_b = Path(r"D:\折腾\v7_pilot_gpu_results2\v7_shadow_experiment")
    data_dir = ROOT / "processed_tri_data_hfc2739"
    split_file = ROOT / "splits" / "HFC_all_split.npz"

    sp = np.load(split_file)
    val_idx = sp['val'].astype(int)
    meta_df = pd.read_csv(data_dir / "meta_info.csv").iloc[val_idx].reset_index(drop=True)

    summary_a = pd.read_csv(dir_a / "pilot_summary_V7-A.csv")
    summary_b = pd.read_csv(dir_b / "pilot_summary_V7-B.csv")

    seeds = [42, 43, 44, 45, 46]

    # Load all predictions
    preds_a = {}
    preds_b = {}
    targets = None

    for s in seeds:
        df_pa = pd.read_csv(dir_a / "checkpoints" / "V7-A" / f"val_predictions_seed_{s}.csv")
        df_pb = pd.read_csv(dir_b / "checkpoints" / "V7-B" / f"val_predictions_seed_{s}.csv")
        preds_a[s] = df_pa['pred'].values
        preds_b[s] = df_pb['pred'].values
        if targets is None:
            targets = df_pa['target'].values

    # 1. Seed-by-Seed Paired Metrics
    seed_paired_records = []
    for s in seeds:
        sa = summary_a[summary_a['seed'] == s].iloc[0]
        sb = summary_b[summary_b['seed'] == s].iloc[0]

        ya = preds_a[s]
        yb = preds_b[s]

        ae_a = np.abs(targets - ya)
        ae_b = np.abs(targets - yb)
        delta_ae = ae_a - ae_b  # >0 means B is better

        win_rate = np.mean(delta_ae > 0) * 100.0
        mean_ci, med_ci = bootstrap_ci(delta_ae)
        _, p_val = stats.wilcoxon(ae_a, ae_b)

        gd_ratio_a = sa['median_delta_y_graph'] / max(sa['median_delta_y_desc'], 1e-8)
        gd_ratio_b = sb['median_delta_y_graph'] / max(sb['median_delta_y_desc'], 1e-8)

        seed_paired_records.append({
            'seed': s,
            'mae_a': sa['mae_raw'],
            'mae_b': sb['mae_raw'],
            'mae_diff': sb['mae_raw'] - sa['mae_raw'],
            'mae_rel_pct': (sb['mae_raw'] - sa['mae_raw']) / sa['mae_raw'] * 100.0,
            'r2_a': sa['r2_raw'],
            'r2_b': sb['r2_raw'],
            'dy_graph_a': sa['median_delta_y_graph'],
            'dy_graph_b': sb['median_delta_y_graph'],
            'dy_desc_a': sa['median_delta_y_desc'],
            'dy_desc_b': sb['median_delta_y_desc'],
            'gd_ratio_a': gd_ratio_a,
            'gd_ratio_b': gd_ratio_b,
            'win_rate_pct': win_rate,
            'median_delta_ae': np.median(delta_ae),
            'mean_delta_ae': np.mean(delta_ae),
            'wilcoxon_p': p_val,
            'best_ep_a': int(sa['best_epoch']),
            'best_ep_b': int(sb['best_epoch']),
        })

    df_paired = pd.DataFrame(seed_paired_records)

    print("=" * 80)
    print("  TABLE 1: 5-SEED PAIRWISE COMPARISON: V7-A vs V7-B")
    print("=" * 80)
    disp_cols = ['seed', 'mae_a', 'mae_b', 'mae_rel_pct', 'dy_graph_a', 'dy_graph_b', 'dy_desc_a', 'dy_desc_b', 'gd_ratio_b', 'win_rate_pct', 'best_ep_b']
    print(df_paired[disp_cols].to_string(index=False))

    # 2. Summary Statistics (All 5 seeds vs Excluding Seed 45 outlier)
    print("\n" + "=" * 80)
    print("  TABLE 2: AGGREGATED METRICS SUMMARY")
    print("=" * 80)
    print(f"V7-A 5-Seed Mean MAE : {df_paired['mae_a'].mean():.5f} ± {df_paired['mae_a'].std():.5f}")
    print(f"V7-B 5-Seed Mean MAE : {df_paired['mae_b'].mean():.5f} ± {df_paired['mae_b'].std():.5f}")
    
    # 4-seed robust subset (Seeds 42, 43, 44, 46)
    sub4 = df_paired[df_paired['seed'] != 45]
    print(f"V7-A Robust 4-Seed MAE: {sub4['mae_a'].mean():.5f} ± {sub4['mae_a'].std():.5f}")
    print(f"V7-B Robust 4-Seed MAE: {sub4['mae_b'].mean():.5f} ± {sub4['mae_b'].std():.5f} (Δ = {sub4['mae_b'].mean()-sub4['mae_a'].mean():+.5f}, {(sub4['mae_b'].mean()-sub4['mae_a'].mean())/sub4['mae_a'].mean()*100:+.2f}%)")

    print(f"\nV7-A 5-Seed Median Δy_graph: {df_paired['dy_graph_a'].mean():.5f} ± {df_paired['dy_graph_a'].std():.5f}")
    print(f"V7-B 5-Seed Median Δy_graph: {df_paired['dy_graph_b'].mean():.5f} ± {df_paired['dy_graph_b'].std():.5f}")
    print(f"V7-A 5-Seed Median Δy_desc : {df_paired['dy_desc_a'].mean():.5f} ± {df_paired['dy_desc_a'].std():.5f}")
    print(f"V7-B 5-Seed Median Δy_desc : {df_paired['dy_desc_b'].mean():.5f} ± {df_paired['dy_desc_b'].std():.5f}")
    print(f"V7-B 5-Seed Mean G/D Ratio : {df_paired['gd_ratio_b'].mean():.2f}")

    # 3. Ensemble Level (5-Seed Ensemble)
    ens_ya = np.mean([preds_a[s] for s in seeds], axis=0)
    ens_yb = np.mean([preds_b[s] for s in seeds], axis=0)
    ens_aea = np.abs(targets - ens_ya)
    ens_aeb = np.abs(targets - ens_yb)
    ens_delta = ens_aea - ens_aeb

    ens_mean_ci, ens_med_ci = bootstrap_ci(ens_delta)
    _, ens_p = stats.wilcoxon(ens_aea, ens_aeb)

    print("\n" + "=" * 80)
    print("  TABLE 3: 5-SEED ENSEMBLE LEVEL PAIRED TEST")
    print("=" * 80)
    print(f"V7-A Ensemble MAE : {np.mean(ens_aea):.5f}")
    print(f"V7-B Ensemble MAE : {np.mean(ens_aeb):.5f} (Δ = {np.mean(ens_aeb)-np.mean(ens_aea):+.5f}, {(np.mean(ens_aeb)-np.mean(ens_aea))/np.mean(ens_aea)*100:+.2f}%)")
    print(f"Ensemble Win Rate (B > A): {np.mean(ens_delta > 0)*100:.2f}% ({np.sum(ens_delta > 0)}/274)")
    print(f"Median ΔAE        : {np.median(ens_delta):+.5f} [95% CI: {ens_med_ci[0]:+.5f}, {ens_med_ci[1]:+.5f}]")
    print(f"Mean ΔAE          : {np.mean(ens_delta):+.5f} [95% CI: {ens_mean_ci[0]:+.5f}, {ens_mean_ci[1]:+.5f}]")
    print(f"Wilcoxon p-value  : {ens_p:.4e}")

    # 4. Dissection by Refrigerant (5-Seed Ensemble)
    meta_df['ens_aea'] = ens_aea
    meta_df['ens_aeb'] = ens_aeb
    meta_df['ens_delta'] = ens_delta

    print("\n" + "=" * 80)
    print("  TABLE 4: REFRIGERANT DISSECTION: 5-SEED ENSEMBLE (V7-A vs V7-B)")
    print("=" * 80)
    ref_grp = meta_df.groupby('refrigerant').agg(
        N=('x1', 'count'),
        A_MAE=('ens_aea', 'mean'),
        B_MAE=('ens_aeb', 'mean'),
        Win_Rate=('ens_delta', lambda x: np.mean(x > 0) * 100.0),
        Mean_ΔAE=('ens_delta', 'mean')
    ).reset_index()
    ref_grp['Rel_Change_Pct'] = (ref_grp['B_MAE'] - ref_grp['A_MAE']) / ref_grp['A_MAE'] * 100.0
    ref_grp = ref_grp.sort_values(by='Rel_Change_Pct')
    print(ref_grp.to_string(index=False))

    out_csv = ROOT / "v7_shadow_experiment" / "v7b_full_5seed_dissection.csv"
    df_paired.to_csv(out_csv, index=False)
    print(f"\nSaved 5-seed dissection to: {out_csv}")

if __name__ == '__main__':
    main()
