"""
audit_v7a_vs_v6_paired.py — Deep Paired Error & Critical Dissection: V7-A vs V6-M0
===================================================================================
Analyzes the official 5-seed GPU results in D:/折腾/v7_pilot_gpu_results against V6-M0.
Performs:
1. Seed-level and ensemble-level paired error differences (ΔAE = AE_v6 - AE_v7a).
2. Win Rate (% samples where V7-A has lower absolute error).
3. Bootstrap 95% CI on median and mean ΔAE.
4. Chemical group dissection (which refrigerants improved, which deteriorated).
5. Error distribution and outlier analysis ("挑刺").
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'GNN_for_property_prediction'))

from Dataset_v6 import IL_set_v6
from Model_v6 import IL_GAT_v6
from torch_geometric.loader import DataLoader


def get_v6_predictions(seeds: list[int], v6_dir: Path, data_dir: Path, val_idx: np.ndarray, train_idx: np.ndarray):
    whole_set = IL_set_v6(path=str(data_dir), args={'descriptor_mode': 'M0'})
    whole_set.fit_scalers(train_idx, save_dir=str(v6_dir))
    val_set = torch.utils.data.Subset(whole_set, val_idx)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v6_preds_by_seed = {}

    for s in seeds:
        ckpt_path = v6_dir / f"best_seed_{s}.pth"
        model_args = {
            'emb_dim': 300,
            'dropout_rate': 0.2,
            'cond_dim': 9,
            'pool': 'global',
            'use_layernorm': False,
            'use_adaptive_gate': False
        }
        model = IL_GAT_v6(model_args).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        st = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(st)
        model.eval()

        preds = []
        with torch.no_grad():
            for graph, cond, label in val_loader:
                graph = graph.to(device)
                cond = cond.to(device)
                out = model(graph, cond).flatten()
                preds.extend(out.cpu().numpy().tolist())
        v6_preds_by_seed[s] = np.array(preds)

    return v6_preds_by_seed


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
    gpu_res_dir = Path(r"D:\折腾\v7_pilot_gpu_results\v7_shadow_experiment")
    v7a_ckpt_dir = gpu_res_dir / "checkpoints" / "V7-A"
    data_dir = ROOT / 'processed_tri_data_hfc2739'
    split_file = ROOT / 'splits' / 'HFC_all_split.npz'
    v6_dir = ROOT / 'results_hfc_all' / 'HFC_all_M0'

    sp = np.load(split_file)
    train_idx = sp['train'].astype(int)
    val_idx = sp['val'].astype(int)

    meta_df = pd.read_csv(data_dir / 'meta_info.csv').iloc[val_idx].reset_index(drop=True)

    seeds = [42, 43, 44, 45, 46]

    print("=" * 80)
    print("  LOADING V6-M0 OFFICIAL PREDICTIONS ON 274 VALIDATION SAMPLES")
    print("=" * 80)
    v6_preds = get_v6_predictions(seeds, v6_dir, data_dir, val_idx, train_idx)

    print("=" * 80)
    print("  LOADING V7-A GPU PREDICTIONS ON 274 VALIDATION SAMPLES")
    print("=" * 80)
    v7a_preds = {}
    targets = None
    for s in seeds:
        pred_csv = v7a_ckpt_dir / f"val_predictions_seed_{s}.csv"
        df = pd.read_csv(pred_csv)
        v7a_preds[s] = df['pred'].values
        if targets is None:
            targets = df['target'].values

    # 1. Per-Seed Paired Metrics
    seed_stats = []
    for s in seeds:
        y_true = targets
        p6 = v6_preds[s]
        p7 = v7a_preds[s]

        ae_v6 = np.abs(y_true - p6)
        ae_v7 = np.abs(y_true - p7)
        delta_ae = ae_v6 - ae_v7  # >0 means V7-A is better

        mae_v6 = np.mean(ae_v6)
        mae_v7 = np.mean(ae_v7)
        r2_v6 = 1.0 - np.sum((y_true - p6)**2) / np.sum((y_true - np.mean(y_true))**2)
        r2_v7 = 1.0 - np.sum((y_true - p7)**2) / np.sum((y_true - np.mean(y_true))**2)

        win_count = int(np.sum(delta_ae > 0))
        loss_count = int(np.sum(delta_ae < 0))
        win_rate = win_count / len(delta_ae) * 100.0

        mean_ci, med_ci = bootstrap_ci(delta_ae)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(ae_v6, ae_v7)

        seed_stats.append({
            'seed': s,
            'mae_v6': mae_v6,
            'mae_v7': mae_v7,
            'mae_diff': mae_v7 - mae_v6,
            'mae_rel_pct': (mae_v7 - mae_v6) / mae_v6 * 100.0,
            'r2_v6': r2_v6,
            'r2_v7': r2_v7,
            'win_rate_pct': win_rate,
            'win_count': win_count,
            'loss_count': loss_count,
            'median_delta_ae': float(np.median(delta_ae)),
            'med_ci_low': med_ci[0],
            'med_ci_high': med_ci[1],
            'mean_delta_ae': float(np.mean(delta_ae)),
            'mean_ci_low': mean_ci[0],
            'mean_ci_high': mean_ci[1],
            'wilcoxon_p': wilcoxon_p
        })

    df_seed_stats = pd.DataFrame(seed_stats)
    print("\n" + "=" * 80)
    print("  SEED-BY-SEED PAIRED COMPARISON MATRIX (N=274)")
    print("=" * 80)
    print(df_seed_stats[['seed', 'mae_v6', 'mae_v7', 'mae_diff', 'mae_rel_pct', 'win_rate_pct', 'median_delta_ae', 'mean_delta_ae', 'wilcoxon_p']].to_string(index=False))

    # 2. Ensemble Level
    ens_p6 = np.mean([v6_preds[s] for s in seeds], axis=0)
    ens_p7 = np.mean([v7a_preds[s] for s in seeds], axis=0)
    ens_ae_v6 = np.abs(targets - ens_p6)
    ens_ae_v7 = np.abs(targets - ens_p7)
    ens_delta_ae = ens_ae_v6 - ens_ae_v7
    ens_win_rate = np.mean(ens_delta_ae > 0) * 100.0
    ens_mean_ci, ens_med_ci = bootstrap_ci(ens_delta_ae)
    _, ens_p = stats.wilcoxon(ens_ae_v6, ens_ae_v7)

    print("\n" + "=" * 80)
    print("  5-SEED ENSEMBLE LEVEL PAIRED COMPARISON")
    print("=" * 80)
    print(f"V6 Ensemble MAE : {np.mean(ens_ae_v6):.5f}")
    print(f"V7 Ensemble MAE : {np.mean(ens_ae_v7):.5f} (Δ = {np.mean(ens_ae_v7)-np.mean(ens_ae_v6):.5f}, {(np.mean(ens_ae_v7)-np.mean(ens_ae_v6))/np.mean(ens_ae_v6)*100:.2f}%)")
    print(f"Ensemble Win Rate: {ens_win_rate:.2f}% ({np.sum(ens_delta_ae > 0)}/274)")
    print(f"Median ΔAE       : {np.median(ens_delta_ae):.5f} [95% CI: {ens_med_ci[0]:.5f}, {ens_med_ci[1]:.5f}]")
    print(f"Mean ΔAE         : {np.mean(ens_delta_ae):.5f} [95% CI: {ens_mean_ci[0]:.5f}, {ens_mean_ci[1]:.5f}]")
    print(f"Wilcoxon p-value : {ens_p:.4e}")

    # 3. Chemical Dissection ("挑刺")
    meta_df['ens_ae_v6'] = ens_ae_v6
    meta_df['ens_ae_v7'] = ens_ae_v7
    meta_df['delta_ae'] = ens_delta_ae

    print("\n" + "=" * 80)
    print("  DISSECTION BY REFRIGERANT: WHERE DID V7 IMPROVE OR WORSEN?")
    print("=" * 80)
    ref_grp = meta_df.groupby('refrigerant').agg(
        N=('x1', 'count'),
        V6_MAE=('ens_ae_v6', 'mean'),
        V7_MAE=('ens_ae_v7', 'mean'),
        Win_Rate=('delta_ae', lambda x: np.mean(x > 0) * 100.0),
        Mean_ΔAE=('delta_ae', 'mean')
    ).reset_index()
    ref_grp['Rel_Change_Pct'] = (ref_grp['V7_MAE'] - ref_grp['V6_MAE']) / ref_grp['V6_MAE'] * 100.0
    ref_grp = ref_grp.sort_values(by='Rel_Change_Pct')
    print(ref_grp.to_string(index=False))

    # 4. Outliers: Top 10 worst errors in V7-A
    meta_df['abs_err_v7'] = ens_ae_v7
    meta_df['pred_v7'] = ens_p7
    meta_df['pred_v6'] = ens_p6
    worst_v7 = meta_df.sort_values(by='abs_err_v7', ascending=False).head(10)
    print("\n" + "=" * 80)
    print("  CRITICAL AUDIT: TOP 10 WORST ERRORS IN V7-A ENSEMBLE")
    print("=" * 80)
    print(worst_v7[['refrigerant', 'cation', 'anion', 'T_K', 'P_MPa', 'x1', 'pred_v7', 'pred_v6', 'abs_err_v7', 'delta_ae']].to_string(index=False))

    # Save detailed per-sample comparison
    out_csv = ROOT / "v7_shadow_experiment" / "v7a_vs_v6_paired_dissection.csv"
    meta_df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed dissection to: {out_csv}")


if __name__ == '__main__':
    main()
