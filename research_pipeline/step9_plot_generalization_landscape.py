"""
step9_plot_generalization_landscape.py
======================================
Generates publication-quality dual-panel Figures (Figure 3a & Figure 3b)
from `loro_generalization_landscape.csv`.

Figure 3a: Chemical Coverage controls zero-shot transferability (CCI vs R2_mean ± std)
Figure 3b: Multicomponent coverage controls prediction reliability (ILCR vs R2_std, size=D_chem)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# Resolve ROOT directory using pathlib
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

def plot_landscape():
    print("============================================================")
    print("  Step 9: Plotting Dual-Panel Generalization Landscape")
    print("============================================================")
    
    csv_path = ROOT / 'loro_generalization_landscape.csv'
    if not csv_path.exists():
        print(f"[Error] {csv_path} does not exist. Please run step8_generalization_metrics.py first.")
        return

    df = pd.read_csv(csv_path)

    # Check if GAT R2 data is available
    if df['gat_r2_mean'].isna().all():
        print("[Warning] GAT R2 values are all NaN in loro_generalization_landscape.csv.")
        print("[Info] Please run GAT LORO benchmarks on Kaggle to populate R2 values before plotting final figures.")
        # Fill with dummy/placeholder data for local plot structure verification if needed
        # But we will handle NaN gracefully in scatter plot.

    fig_dir = ROOT / 'figure_v5'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Font and styling setup
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    chosen_font = 'Arial' if 'Arial' in available_fonts else 'sans-serif'
    
    plt.rcParams['font.family'] = chosen_font
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2

    family_colors = {
        'HFC': '#1f77b4',   # Blue
        'HFO': '#2ca02c',   # Green
        'HCFC': '#ff7f0e'   # Orange
    }

    # ============================================================
    # Figure 3a: Chemical Coverage controls zero-shot transferability
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)

    # Plot points by family
    for fam in ['HFC', 'HFO', 'HCFC']:
        fam_df = df[df['family'] == fam]
        if not fam_df.empty and not fam_df['gat_r2_mean'].isna().all():
            # Size scaled by n_test (between 80 and 220)
            sizes = np.clip(fam_df['n_test'] / 3.5, 80, 220)
            
            ax.errorbar(
                fam_df['cci_ref_morgan_r2'],
                fam_df['gat_r2_mean'],
                yerr=fam_df['gat_r2_std'].fillna(0.0),
                fmt='none',
                ecolor='#666666',
                elinewidth=1.5,
                capsize=4,
                capthick=1.2,
                zorder=3
            )
            
            ax.scatter(
                fam_df['cci_ref_morgan_r2'],
                fam_df['gat_r2_mean'],
                c=family_colors[fam],
                label=fam,
                s=sizes,
                alpha=0.9,
                edgecolors='black',
                linewidths=1.0,
                zorder=4
            )

    # Annotate refrigerant names
    for _, row in df.iterrows():
        if not np.isnan(row['gat_r2_mean']):
            r_name = row['refrigerant']
            ax.annotate(
                r_name,
                (row['cci_ref_morgan_r2'], row['gat_r2_mean']),
                xytext=(row['cci_ref_morgan_r2'] + 0.008, row['gat_r2_mean'] + 0.02),
                fontsize=11,
                fontweight='bold',
                color='#222222',
                zorder=5
            )

    # Reference lines
    ax.axhline(0.0, color='#d62728', linestyle='--', linewidth=1.5, label='R² = 0', zorder=2)
    ax.axhline(0.7, color='#2ca02c', linestyle='--', linewidth=1.5, label='High Reliability (R² = 0.7)', zorder=2)

    ax.set_xlabel('Chemical Coverage Index (CCI)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Zero-Shot R² (Mean ± Std)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Chemical Coverage Controls Zero-Shot Transferability', fontsize=13, fontweight='bold', pad=15)

    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10, loc='lower right')
    
    plt.tight_layout()
    fig3a_path = fig_dir / 'Figure3a_CCI_vs_R2.png'
    plt.savefig(fig3a_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Done] Figure 3a saved to: {fig3a_path}")

    # ============================================================
    # Figure 3b: Multicomponent coverage controls prediction reliability
    # ============================================================
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)

    valid_std_df = df[~df['gat_r2_std'].isna()]
    if not valid_std_df.empty:
        # Size scaled by D_chem (Chemical Shift)
        sizes = np.clip(valid_std_df['d_chem'] * 120 + 60, 60, 300)

        for fam in ['HFC', 'HFO', 'HCFC']:
            fam_df = valid_std_df[valid_std_df['family'] == fam]
            if not fam_df.empty:
                s_fam = np.clip(fam_df['d_chem'] * 120 + 60, 60, 300)
                ax.scatter(
                    fam_df['ilcr_count'],
                    fam_df['gat_r2_std'],
                    c=family_colors[fam],
                    label=fam,
                    s=s_fam,
                    alpha=0.85,
                    edgecolors='black',
                    linewidths=1.2,
                    zorder=4
                )

        # Annotate refrigerant names
        for _, row in valid_std_df.iterrows():
            r_name = row['refrigerant']
            ax.annotate(
                r_name,
                (row['ilcr_count'], row['gat_r2_std']),
                xytext=(row['ilcr_count'] + 0.008, row['gat_r2_std'] + 0.015),
                fontsize=11,
                fontweight='bold',
                color='#222222',
                zorder=5
            )

    ax.set_xlabel('Ionic-Liquid Coverage Ratio (ILCR)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Prediction Uncertainty (R² Std across 5 seeds)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Multicomponent Coverage Controls Prediction Reliability', fontsize=13, fontweight='bold', pad=15)

    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10, loc='upper right')

    plt.tight_layout()
    fig3b_path = fig_dir / 'Figure3b_Uncertainty_landscape.png'
    plt.savefig(fig3b_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Done] Figure 3b saved to: {fig3b_path}")

if __name__ == '__main__':
    plot_landscape()
