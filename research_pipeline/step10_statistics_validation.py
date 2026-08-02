"""
step10_statistics_validation.py
================================
Full statistical validation for LORO generalization landscape.

1. Full pairwise Spearman correlation matrix (CCI, ILCR, D_chem) × (R²_mean, R²_std)
2. R134a sensitivity analysis: repeat all correlations excluding the isomer outlier
3. Fingerprint robustness (multi-fingerprint rank consistency)
4. Clean summary table for paper inclusion
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Resolve ROOT directory using pathlib
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

PREDICTORS = [
    ('cci_ref_morgan_r2', 'CCI'),
    ('ilcr_count',        'ILCR'),
    ('d_chem',            'D_chem'),
]

RESPONSES = [
    ('gat_r2_mean', 'R²_mean'),
    ('gat_r2_std',  'R²_std'),
]


def _run_correlation_block(df_sub, label, predictors, responses):
    """Compute Spearman and Pearson for all predictor×response pairs."""
    n = len(df_sub)
    print(f"\n  --- {label} (n={n}) ---")
    
    results = []
    for pcol, pname in predictors:
        for rcol, rname in responses:
            x = df_sub[pcol].values
            y = df_sub[rcol].values
            # Skip if any NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 3:
                print(f"    {pname:8s} vs {rname:8s}: insufficient data (n={mask.sum()})")
                continue
            x_clean, y_clean = x[mask], y[mask]
            rho, p_rho = spearmanr(x_clean, y_clean)
            r_pear, p_pear = pearsonr(x_clean, y_clean)
            
            # Significance marker
            sig = ''
            if p_rho < 0.01:
                sig = '**'
            elif p_rho < 0.05:
                sig = '*'
            elif p_rho < 0.10:
                sig = '†'
            
            print(f"    {pname:8s} vs {rname:8s}:  Spearman ρ = {rho:+.4f} (p={p_rho:.4f}){sig}  |  Pearson r = {r_pear:+.4f} (p={p_pear:.4f})")
            results.append({
                'subset': label,
                'predictor': pname,
                'response': rname,
                'n': int(mask.sum()),
                'spearman_rho': rho,
                'spearman_p': p_rho,
                'pearson_r': r_pear,
                'pearson_p': p_pear,
                'significant_005': p_rho < 0.05,
            })
    return results


def validate_statistics():
    print("=" * 70)
    print("  Step 10: Full Statistical Validation & Sensitivity Analysis")
    print("=" * 70)

    csv_path = ROOT / 'loro_generalization_landscape.csv'
    if not csv_path.exists():
        print(f"[Error] {csv_path} does not exist. Run step8 first.")
        return

    df = pd.read_csv(csv_path)
    valid_df = df[~df['gat_r2_mean'].isna()].copy()

    if len(valid_df) < 3:
        print(f"[Notice] Only {len(valid_df)} refrigerants have R² data. Need ≥3.")
        return

    # ================================================================
    # 1. Full pairwise correlation matrix (all 8 refrigerants)
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 1: Full Pairwise Correlation Matrix")
    print("=" * 70)

    all_results = []

    results_all = _run_correlation_block(valid_df, "All refrigerants", PREDICTORS, RESPONSES)
    all_results.extend(results_all)

    # ================================================================
    # 2. Sensitivity: Exclude R134a (isomer outlier)
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 2: Sensitivity Analysis (Excluding R134a isomer outlier)")
    print("=" * 70)
    print("  Rationale: R134a fails due to position isomer confusion with")
    print("  training-set R134 (CHF₂CHF₂ vs CH₂FCF₃), not coverage/shift.")

    no134a = valid_df[valid_df['refrigerant'] != 'R134a'].copy()
    results_no134a = _run_correlation_block(no134a, "Excluding R134a", PREDICTORS, RESPONSES)
    all_results.extend(results_no134a)

    # ================================================================
    # 3. Fingerprint Robustness
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 3: Fingerprint Robustness Check")
    print("=" * 70)

    rho_r2_r3, p_r2_r3 = spearmanr(df['cci_ref_morgan_r2'], df['cci_ref_morgan_r3'])
    rho_r2_maccs, p_r2_maccs = spearmanr(df['cci_ref_morgan_r2'], df['cci_ref_maccs'])

    print(f"  Morgan r=2 vs Morgan r=3 : Spearman ρ = {rho_r2_r3:.4f} (p={p_r2_r3:.4f})")
    print(f"  Morgan r=2 vs MACCS Keys : Spearman ρ = {rho_r2_maccs:.4f} (p={p_r2_maccs:.4f})")

    # ================================================================
    # 4. Summary Data Table (for paper)
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 4: Summary Data Table")
    print("=" * 70)

    summary_cols = ['refrigerant', 'family', 'cci_ref_morgan_r2', 'ilcr_count',
                    'd_chem', 'gat_r2_mean', 'gat_r2_std']
    summary = valid_df[summary_cols].copy()
    summary.columns = ['Refrigerant', 'Family', 'CCI', 'ILCR', 'D_chem', 'R2_mean', 'R2_std']
    summary = summary.sort_values('R2_mean', ascending=False)
    print(summary.to_string(index=False, float_format='%.4f'))

    # ================================================================
    # 5. Export correlation results to CSV
    # ================================================================
    corr_df = pd.DataFrame(all_results)
    corr_path = ROOT / 'loro_correlation_matrix.csv'
    corr_df.to_csv(corr_path, index=False)
    print(f"\n[Done] Full correlation matrix exported to: {corr_path}")

    # ================================================================
    # 6. Key findings summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  KEY FINDINGS SUMMARY")
    print("=" * 70)

    sig_all = [r for r in results_all if r['spearman_p'] < 0.05]
    sig_no134a = [r for r in results_no134a if r['spearman_p'] < 0.05]

    if sig_all:
        print("\n  Significant correlations (p<0.05, all 8 refrigerants):")
        for r in sig_all:
            print(f"    ✓ {r['predictor']} vs {r['response']}: ρ={r['spearman_rho']:+.4f} (p={r['spearman_p']:.4f})")
    else:
        print("\n  No significant correlations found across all 8 refrigerants.")

    if sig_no134a:
        print(f"\n  Significant correlations (p<0.05, excluding R134a, n={len(no134a)}):")
        for r in sig_no134a:
            print(f"    ✓ {r['predictor']} vs {r['response']}: ρ={r['spearman_rho']:+.4f} (p={r['spearman_p']:.4f})")
    else:
        print(f"\n  No significant correlations found excluding R134a (n={len(no134a)}).")

    marginal_all = [r for r in results_all if 0.05 <= r['spearman_p'] < 0.10]
    marginal_no134a = [r for r in results_no134a if 0.05 <= r['spearman_p'] < 0.10]

    if marginal_all:
        print(f"\n  Marginal correlations (0.05≤p<0.10, all 8):")
        for r in marginal_all:
            print(f"    † {r['predictor']} vs {r['response']}: ρ={r['spearman_rho']:+.4f} (p={r['spearman_p']:.4f})")

    if marginal_no134a:
        print(f"\n  Marginal correlations (0.05≤p<0.10, excluding R134a):")
        for r in marginal_no134a:
            print(f"    † {r['predictor']} vs {r['response']}: ρ={r['spearman_rho']:+.4f} (p={r['spearman_p']:.4f})")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    validate_statistics()
