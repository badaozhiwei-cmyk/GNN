"""
step10_statistics_validation.py
================================
Performs statistical validation and fingerprint robustness analysis for LORO results.
Calculates Spearman/Pearson correlation and multi-fingerprint rank consistency.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Resolve ROOT directory using pathlib
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

def validate_statistics():
    print("============================================================")
    print("  Step 10: Statistical Validation & Fingerprint Robustness")
    print("============================================================")

    csv_path = ROOT / 'loro_generalization_landscape.csv'
    if not csv_path.exists():
        print(f"[Error] {csv_path} does not exist. Please run step8_generalization_metrics.py first.")
        return

    df = pd.read_csv(csv_path)

    # Filter rows with valid GAT R2
    valid_df = df[~df['gat_r2_mean'].isna()]
    
    if len(valid_df) < 3:
        print(f"[Notice] Currently only {len(valid_df)} refrigerants have GAT R2 values.")
        print("[Info] Full statistical correlation tests require complete 8-refrigerant benchmark runs.")
    else:
        print(f"\n1. Correlation Tests (n={len(valid_df)} refrigerants):")
        
        # CCI vs R2_mean correlation
        rho_cci, p_rho_cci = spearmanr(valid_df['cci_ref_morgan_r2'], valid_df['gat_r2_mean'])
        r_cci, p_r_cci = pearsonr(valid_df['cci_ref_morgan_r2'], valid_df['gat_r2_mean'])
        print(f"   CCI vs Zero-Shot R2_mean:")
        print(f"     Spearman rho = {rho_cci:.4f} (p = {p_rho_cci:.4f})")
        print(f"     Pearson  r   = {r_cci:.4f} (p = {p_r_cci:.4f})")

        # ILCR vs R2_std (Uncertainty) correlation
        if not valid_df['gat_r2_std'].isna().all():
            rho_ilcr, p_rho_ilcr = spearmanr(valid_df['ilcr_count'], valid_df['gat_r2_std'])
            print(f"\n   ILCR vs Prediction Uncertainty (R2_std):")
            print(f"     Spearman rho = {rho_ilcr:.4f} (p = {p_rho_ilcr:.4f})")

    # 2. Multi-Fingerprint Rank Consistency Check (All 8 refrigerants)
    print("\n2. Fingerprint Sensitivity & Robustness Analysis (All 8 Refrigerants):")
    rho_r2_r3, _ = spearmanr(df['cci_ref_morgan_r2'], df['cci_ref_morgan_r3'])
    rho_r2_maccs, _ = spearmanr(df['cci_ref_morgan_r2'], df['cci_ref_maccs'])

    print(f"   Rank Correlation across Fingerprint Representations:")
    print(f"     Morgan r=2 vs Morgan r=3 : Spearman rho = {rho_r2_r3:.4f}")
    print(f"     Morgan r=2 vs MACCS Keys : Spearman rho = {rho_r2_maccs:.4f}")

    # Summary table output
    print("\n3. Robustness Summary Table:")
    summary_cols = ['refrigerant', 'nearest_ref_morgan_r2', 'cci_ref_morgan_r2', 'cci_ref_morgan_r3', 'cci_ref_maccs']
    print(df[summary_cols].to_string(index=False))
    print("============================================================\n")

if __name__ == '__main__':
    validate_statistics()
