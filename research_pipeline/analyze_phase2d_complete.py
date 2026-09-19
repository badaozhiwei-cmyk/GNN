"""
Deep objective analysis of Phase II-D complete GPU results
Compatible with both local project paths and external export folders.
"""
import sys, os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent

print("=" * 90)
print("1. L2 COMPOSITIONAL BENCHMARK (4-PAIR UNSEEN RECOMBINATION)")
print("=" * 90)

l2_m0_file = ROOT / "results_split_L2" / "L2_M0" / "ensemble_predictions.csv"
l2_mr_file = ROOT / "results_split_L2" / "L2_Mreduced" / "ensemble_predictions.csv"

if l2_m0_file.exists() and l2_mr_file.exists():
    df_m0 = pd.read_csv(l2_m0_file)
    df_mr = pd.read_csv(l2_mr_file)
    
    print(f"L2 Test Samples: {len(df_m0)}")
    
    df_m0['pair'] = df_m0['sample_id'].apply(lambda x: '__'.join(x.split('__')[:2]))
    df_mr['pair'] = df_mr['sample_id'].apply(lambda x: '__'.join(x.split('__')[:2]))
    pairs = df_m0['pair'].unique()
    print(f"\n{'Pair':<20s} {'N':<6s} {'M0 MAE':<10s} {'M0 R2':<10s} {'Mred MAE':<10s} {'Mred R2':<10s} {'Delta MAE':<10s}")
    print("-" * 80)
    for p in pairs:
        sub0 = df_m0[df_m0['pair'] == p]
        subr = df_mr[df_mr['pair'] == p]
        
        y_true = sub0['true_x1'].values
        p0 = sub0['pred_mu'].values
        pr = subr['pred_mu'].values
        
        mae0 = np.mean(np.abs(y_true - p0))
        maer = np.mean(np.abs(y_true - pr))
        
        var_y = np.var(y_true)
        r20 = 1.0 - np.mean((y_true - p0)**2) / (var_y + 1e-12) if var_y > 1e-6 else 0.0
        r2r = 1.0 - np.mean((y_true - pr)**2) / (var_y + 1e-12) if var_y > 1e-6 else 0.0
        
        delta = maer - mae0
        print(f"{p:<20s} {len(sub0):<6d} {mae0:<10.4f} {r20:<10.4f} {maer:<10.4f} {r2r:<10.4f} {delta:<+10.4f}")

    err0 = np.abs(df_m0['true_x1'].values - df_m0['pred_mu'].values)
    sig0 = df_m0['pred_sigma'].values
    errr = np.abs(df_mr['true_x1'].values - df_mr['pred_mu'].values)
    sigr = df_mr['pred_sigma'].values
    
    r0, _ = pearsonr(sig0, err0)
    rho0, _ = spearmanr(sig0, err0)
    rr, _ = pearsonr(sigr, errr)
    rhor, _ = spearmanr(sigr, errr)
    print(f"\nL2 M0 Uncertainty-Error: Pearson r={r0:.4f}, Spearman rho={rho0:.4f}")
    print(f"L2 Mreduced Uncertainty-Error: Pearson r={rr:.4f}, Spearman rho={rhor:.4f}")

print("\n" + "=" * 90)
print("2. HFO / HCFO CROSS-FAMILY ZERO-SHOT STRESS TEST (N=1106)")
print("=" * 90)

hfo_hfc_all_csv = ROOT / "paper_results" / "hfo_zeroshot_predictions_HFC_all.csv"
if hfo_hfc_all_csv.exists():
    hfo_df = pd.read_csv(hfo_hfc_all_csv)
    
    for md in ['M0', 'Mreduced']:
        sub = hfo_df[hfo_df['descriptor_mode'] == md]
        print(f"\n--- Model: HFC_all | Mode: {md} (Total N={len(sub)}) ---")
        print(f"{'Species':<16s} {'Class':<8s} {'N':<6s} {'MAE':<10s} {'RMSE':<10s} {'R2':<10s} {'Mean Sigma':<12s} {'Spearman':<10s}")
        print("-" * 85)
        
        for sp, grp in sub.groupby('refrigerant'):
            yt = grp['true_x1'].values
            yp = grp['pred_mu'].values
            sig = grp['pred_sigma'].values
            err = np.abs(yt - yp)
            
            mae = np.mean(err)
            rmse = np.sqrt(np.mean((yt - yp)**2))
            var_y = np.var(yt)
            r2 = 1.0 - np.mean((yt - yp)**2) / (var_y + 1e-12) if var_y > 1e-6 else 0.0
            msig = np.mean(sig)
            
            rho, _ = spearmanr(sig, err) if len(grp) > 3 and np.std(sig) > 1e-8 else (0.0, 0.0)
            chem_cls = "HCFO" if "ZD" in sp else "HFO"
            print(f"{sp:<16s} {chem_cls:<8s} {len(grp):<6d} {mae:<10.4f} {rmse:<10.4f} {r2:<10.4f} {msig:<12.4f} {rho:<10.4f}")

print("\n" + "=" * 90)
print("3. MODEL SOURCE COMPARISON: HFC_all vs L2 (Sensitivity Check)")
print("=" * 90)

hfo_l2_csv = ROOT / "paper_results" / "hfo_zeroshot_predictions_L2.csv"
if hfo_l2_csv.exists():
    l2_hfo_df = pd.read_csv(hfo_l2_csv)
    for md in ['M0', 'Mreduced']:
        sub_hfc = hfo_df[hfo_df['descriptor_mode'] == md]
        sub_l2 = l2_hfo_df[l2_hfo_df['descriptor_mode'] == md]
        
        mae_hfc = np.mean(np.abs(sub_hfc['true_x1'].values - sub_hfc['pred_mu'].values))
        mae_l2 = np.mean(np.abs(sub_l2['true_x1'].values - sub_l2['pred_mu'].values))
        
        r2_hfc = 1.0 - np.mean((sub_hfc['true_x1'].values - sub_hfc['pred_mu'].values)**2) / np.var(sub_hfc['true_x1'].values)
        r2_l2 = 1.0 - np.mean((sub_l2['true_x1'].values - sub_l2['pred_mu'].values)**2) / np.var(sub_l2['true_x1'].values)
        
        print(f"Mode {md:9s} Overall: HFC_all MAE={mae_hfc:.4f} (R2={r2_hfc:.4f})  |  L2 MAE={mae_l2:.4f} (R2={r2_l2:.4f})")
