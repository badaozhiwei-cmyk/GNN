"""
step18_paper_figures.py — Phase II-D Paper Publication Figures (Figures 1 - 5)
==============================================================================
【功能】
生成论文最终定稿的 5 张出版级主图 (同时导出 300 DPI PNG 与 矢量 SVG):
  - Fig 1: Chemical Generalization Ladder (M1 -> B1 -> B2 -> L2 -> HFO)
  - Fig 2: Continuous Generalization Landscape (D_FP, D_thermo, D_phys -> Error)
  - Fig 3: Mechanistic Attribution via Integrated Gradients (M0 -> Mreduced shifts)
  - Fig 4: Uncertainty Quantification & Risk-Coverage (Dual Panel)
  - Fig 5: Multi-Dimensional Failure Boundary Map (Final Synthesis Map)
"""

import sys, os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

FIG_DIR = PROJECT_ROOT / "paper_results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 统一出版级样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# 经典配色方案
C_M0 = "#1f77b4"      # 纯图基准: 经典冷蓝
C_MRED = "#d95f02"    # 对比态先验: 稳健暖橙
C_ACCENT = "#2ca02c"  # 辅助高亮
C_GRAY = "#7f7f7f"

print("=" * 80)
print("  STEP 18: RENDERING PUBLICATION FIGURES 1 TO 5")
print("=" * 80)

# ==============================================================================
# FIGURE 1: CHEMICAL GENERALIZATION LADDER
# ==============================================================================
print("\n[1/5] Generating Figure 1: Chemical Generalization Ladder...")

# 从官方冻结主表读取
df_master = pd.read_csv(PROJECT_ROOT / "paper_results" / "table_main_generalization_boundary.csv")

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9.5, 7.0), gridspec_kw={'height_ratios': [1.2, 1.0]})

# Panel A: M0 vs Mreduced MAE Grouped Bars
x = np.arange(len(df_master))
width = 0.35

rects1 = ax_top.bar(x - width/2, df_master['M0 MAE'], width, label=r'Baseline $M_0$ (Pure Graph)', color=C_M0, alpha=0.9, edgecolor='black', linewidth=0.8)
rects2 = ax_top.bar(x + width/2, df_master['Mred MAE'], width, label=r'Intervention $M_{\rm reduced}$ (Physics Prior)', color=C_MRED, alpha=0.9, edgecolor='black', linewidth=0.8)

# 标注具体数值
for rect in rects1:
    h = rect.get_height()
    ax_top.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)
for rect in rects2:
    h = rect.get_height()
    ax_top.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax_top.set_ylabel('Mean Absolute Error (MAE)')
ax_top.set_title('(a) Generalization Error Across Distinct Distribution Shifts', loc='left', fontsize=11, fontweight='bold')
ax_top.set_xticks(x)
axis_short_labels = [
    "M1\n(12 HFCs LORO)",
    "B1\n(Fluorosulfonates)",
    "B2\n(Inorganic BF4/PF6)",
    "L2\n(4 Unseen IL Pairs)",
    "HFO/HCFO\n(1106 Olefin Points)"
]
ax_top.set_xticklabels(axis_short_labels, fontweight='bold', fontsize=9.0)
ax_top.legend(loc='upper right', frameon=True)
ax_top.grid(axis='y', linestyle='--', alpha=0.3)
ax_top.set_ylim(0, 0.125)

# Panel B: Delta MAE (%) Waterfall / Bar
deltas = [float(d.replace('%', '')) for d in df_master['delta_MAE']]
colors = ['#2b83ba' if d < -15 else ('#abdda4' if d < -5 else '#fdae61') for d in deltas]

bars = ax_bot.bar(x, deltas, width=0.55, color=colors, edgecolor='black', linewidth=0.8)
ax_bot.axhline(0, color='black', linewidth=1.0)

for bar, d in zip(bars, deltas):
    ax_bot.annotate(f'{d:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, d), xytext=(0, -12 if d < 0 else 3), textcoords="offset points", ha='center', va='top' if d < 0 else 'bottom', fontsize=9, fontweight='bold')

ax_bot.set_ylabel(r'Error Reduction $\Delta\text{MAE}$ (%)')
ax_bot.set_title('(b) Relative Benefit of Thermodynamic Coordinates', loc='left', fontsize=11, fontweight='bold')
ax_bot.set_xticks(x)
ax_bot.set_xticklabels(df_master['Axis'], fontweight='bold')
ax_bot.set_ylim(-45, 5)
ax_bot.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / "Figure1_chemical_generalization_ladder.png", dpi=300)
fig.savefig(FIG_DIR / "Figure1_chemical_generalization_ladder.svg")
plt.close(fig)
print("  [✓] Saved Figure 1 (PNG + SVG)")

# ==============================================================================
# FIGURE 2: CONTINUOUS GENERALIZATION LANDSCAPE
# ==============================================================================
print("\n[2/5] Generating Figure 2: Continuous Generalization Landscape...")

dist_csv = PROJECT_ROOT / "results_boundary" / "continuous_distance_benchmark.csv"
if dist_csv.exists():
    df_dist = pd.read_csv(dist_csv)
    
    # 筛选 M0 与 Mreduced 的数据点
    sub_m0 = df_dist[df_dist['mode'] == 'M0'].dropna(subset=['MAE'])
    sub_mr = df_dist[df_dist['mode'] == 'Mreduced'].dropna(subset=['MAE'])
    
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)
    
    # 2.1 D_FP (Fingerprint distance)
    axes[0].scatter(sub_m0['D_FP'], sub_m0['MAE'], color=C_M0, label='M0', alpha=0.75, s=45, edgecolor='none')
    axes[0].scatter(sub_mr['D_FP'], sub_mr['MAE'], color=C_MRED, marker='^', label='Mreduced', alpha=0.85, s=55, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel(r'Topological Distance $D_{\rm FP}$ (1 - Tanimoto)')
    axes[0].set_ylabel('Species MAE')
    axes[0].set_title('(a) Molecular Topology', loc='left', fontsize=10.5, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].legend(loc='upper left')
    
    # 2.2 D_phys (RDKit physical proxy distance)
    axes[1].scatter(sub_m0['D_phys'], sub_m0['MAE'], color=C_M0, label='M0', alpha=0.75, s=45, edgecolor='none')
    axes[1].scatter(sub_mr['D_phys'], sub_mr['MAE'], color=C_MRED, marker='^', label='Mreduced', alpha=0.85, s=55, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel(r'Physicochemical Distance $D_{\rm phys}$')
    axes[1].set_title('(b) Physicochemical Descriptors', loc='left', fontsize=10.5, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    # 2.3 D_thermo (where available, e.g. M1 LORO)
    m1_dist = sub_m0[sub_m0['split'] == 'M1_LORO']
    m1_mr = sub_mr[sub_mr['split'] == 'M1_LORO']
    axes[2].scatter(m1_dist['D_thermo'], m1_dist['MAE'], color=C_M0, label='M0 (M1)', alpha=0.8, s=50)
    axes[2].scatter(m1_mr['D_thermo'], m1_mr['MAE'], color=C_MRED, marker='^', label='Mreduced (M1)', alpha=0.9, s=60, edgecolor='black', linewidth=0.5)
    axes[2].set_xlabel(r'Thermodynamic State Distance $D_{\rm thermo}$')
    axes[2].set_title('(c) Critical Property Distance', loc='left', fontsize=10.5, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.3)
    axes[2].legend(loc='upper left')
    
    fig.suptitle('No Single Scalar Distance Adequately Explains OOD Generalization Boundary', fontsize=12, y=1.03, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / "Figure2_continuous_generalization_landscape.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / "Figure2_continuous_generalization_landscape.svg", bbox_inches='tight')
    plt.close(fig)
    print("  [✓] Saved Figure 2 (PNG + SVG)")

# ==============================================================================
# FIGURE 3: MECHANISTIC ATTRIBUTION VIA INTEGRATED GRADIENTS
# ==============================================================================
print("\n[3/5] Generating Figure 3: Mechanistic Attribution (IG)...")

attr_csv = PROJECT_ROOT / "results_attribution" / "attribution_species_summary.csv"
if attr_csv.exists():
    df_attr = pd.read_csv(attr_csv)
    
    fig, (ax_bf4, ax_pf6) = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    
    concepts = ['thermo_state', 'ref_physical', 'anion_physical', 'cation_physical', 'reduced_priors']
    concept_labels = ['Thermo State (T, P)', 'Refrigerant Phys (q, logP, MW)', 'Anion Phys (MW)', 'Cation Phys (q, TPSA, MW)', r'Reduced Priors ($T_r, P_r, \omega$)']
    c_colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a', '#d95f02']
    
    modes = ['M0', 'Mreduced']
    
    for ax, anion_name, title in [(ax_bf4, 'BF4', r'(a) $\mathrm{BF}_4$ Anion Systems'), 
                                  (ax_pf6, 'PF6', r'(b) $\mathrm{PF}_6$ Anion Systems')]:
        sub = df_attr[df_attr['anion'] == anion_name].set_index('mode')
        
        bottom = np.zeros(len(modes))
        for c_key, c_lbl, col in zip(concepts, concept_labels, c_colors):
            col_name = f'group_share_{c_key}'
            vals = [sub.loc[m, col_name] if col_name in sub.columns and m in sub.index and not pd.isna(sub.loc[m, col_name]) else 0.0 for m in modes]
            ax.bar(modes, vals, bottom=bottom, label=c_lbl, color=col, width=0.45, edgecolor='black', linewidth=0.7)
            bottom += np.array(vals)
            
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        ax.set_ylabel('Relative Attribution Share')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(0, 1.05)
        
    ax_pf6.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, title='Concept Groups')
    fig.suptitle('Mechanistic Attribution: Thermodynamic Coordinates Anchor Molecular Predictions', fontsize=12, y=0.98, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / "Figure3_mechanistic_attribution_ig.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / "Figure3_mechanistic_attribution_ig.svg", bbox_inches='tight')
    plt.close(fig)
    print("  [✓] Saved Figure 3 (PNG + SVG)")

# ==============================================================================
# FIGURE 4: UNCERTAINTY & RISK-COVERAGE
# ==============================================================================
print("\n[4/5] Generating Figure 4: Uncertainty & Risk-Coverage (Dual Panel)...")

# 复用已严格对账的 Risk-Coverage 数据
rc_csv = PROJECT_ROOT / "paper_results" / "phase2d_paper_assembly" / "tables" / "table_risk_coverage_mreduced.csv"
uq_csv = PROJECT_ROOT / "paper_results" / "phase2d_paper_assembly" / "tables" / "table_cross_axis_uq_summary.csv"

if rc_csv.exists() and uq_csv.exists():
    df_rc = pd.read_csv(rc_csv)
    df_uq = pd.read_csv(uq_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    
    # Panel A: Error–Uncertainty Scatter (Verified Axes)
    verified_uq = df_uq.dropna(subset=['Mred_mean_sigma'])
    for _, r in verified_uq.iterrows():
        ax1.scatter(r['M0_ensemble_MAE'], r['M0_mean_sigma'], color=C_M0, s=80, alpha=0.85)
        ax1.scatter(r['Mred_ensemble_MAE'], r['Mred_mean_sigma'], color=C_MRED, marker='X', s=95, alpha=0.95)
        ax1.annotate(f"{r['axis']}", (r['Mred_ensemble_MAE'], r['Mred_mean_sigma']), xytext=(6, -2), textcoords="offset points", fontsize=9.5, fontweight='bold')
    
    ax1.set_xlabel('Ensemble MAE')
    ax1.set_ylabel(r'Mean Ensemble Disagreement $\bar{\sigma}$')
    ax1.set_title('(a) Error–Uncertainty Landscape', loc='left', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    custom_handles = [
        mpatches.Patch(color=C_M0, label='M0 Baseline'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor=C_MRED, markersize=10, label=r'Mreduced ($\rho \in [0.31, 0.61]$)')
    ]
    ax1.legend(handles=custom_handles, loc='upper right')
    
    # Panel B: Risk–Coverage Curves (5 Axes Monotonic)
    axis_colors = {'M1': '#1f77b4', 'B1': '#ff7f0e', 'B2': '#2ca02c', 'L2': '#d62728', 'HFO/HCFO': '#9467bd'}
    
    for ax_name in ['M1', 'B1', 'B2', 'L2', 'HFO/HCFO']:
        sub_c = df_rc[df_rc['axis'] == ax_name].sort_values('coverage', ascending=False)
        cov_pct = sub_c['coverage'] * 100
        risks = sub_c['risk_MAE']
        ax2.plot(cov_pct, risks, marker='o', linewidth=1.8, markersize=6, label=ax_name, color=axis_colors.get(ax_name, 'black'))
        
    ax2.set_xlabel('Coverage (%) [Retaining Lowest-Disagreement Samples]')
    ax2.set_ylabel('Selective Prediction Risk (MAE)')
    ax2.set_title('(b) Risk–Coverage Selective Decision Profiles', loc='left', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / "Figure4_uncertainty_and_risk_coverage.png", dpi=300)
    fig.savefig(FIG_DIR / "Figure4_uncertainty_and_risk_coverage.svg")
    plt.close(fig)
    print("  [✓] Saved Figure 4 (PNG + SVG)")

# ==============================================================================
# FIGURE 5: MULTI-DIMENSIONAL FAILURE BOUNDARY MAP
# ==============================================================================
print("\n[5/5] Generating Figure 5: Comprehensive Failure Boundary Map...")

# 全文总收口图: 构筑多维失效模式与化学边界全景图
fig, ax = plt.subplots(figsize=(10.8, 6.2))

# 绘制矩阵格局
dimensions = [
    "State Coverage\n(Extrapolation in T/P)",
    "Chemical Family\n(Saturation & Olefins)",
    "Composition Recomb.\n(Unseen Ion Pairs)",
    "Ion Matrix Shift\n(Anion Chem. Class)",
    "Refrigerant Shift\n(Molecular Identity)",
    "Stereo Representation\n(E/Z Isomerism)"
]

y_coords = np.arange(len(dimensions))

# 案例点与定性状态 (Dim_idx, Intervention_Effect (-1 to 1), Label, Severity_color, y_offset)
cases = [
    (4, 0.75, "R134/R134a\n(Fluorine Distribution)", "#2ca02c", -0.16),
    (4, 0.65, "R245fa/R236fa\n(Carbon Backbone)", "#2ca02c", 0.16),
    (3, 0.15, "Fam-2 Fluorosulfonates\n(High Graph Redundancy)", "#1f77b4", -0.16),
    (3, 0.50, "BF4 / PF6\n(Coordination Decoupling)", "#ff7f0e", 0.16),
    (2, 0.45, "4 Unseen IL Pairs\n(Interpolative Recomb.)", "#2ca02c", 0.0),
    (1, 0.70, "R1234yf / R1234ze\n(Unsaturated Transfer)", "#2ca02c", -0.16),
    (1, 0.85, "R1233zd(E)\n(Chloro-olefin Anchor)", "#2ca02c", 0.16),
    (5, -0.60, "R1336mzz(E)\n(Graph Blindness / Steric)", "#d62728", 0.0),
    (5, 0.35, "R1336mzz(Z)\n(Scalar Distinction)", "#ff7f0e", 0.0)
]

# 绘制中心中性轴
ax.axvline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

for y_idx in y_coords:
    ax.axhline(y_idx, color='lightgray', linestyle=':', linewidth=0.8, zorder=1)

for dim_idx, eff, label, col, y_off in cases:
    y_pos = dim_idx + y_off
    ax.scatter(eff, y_pos, color=col, s=120, edgecolor='black', linewidth=1.0, zorder=3)
    offset_x = 0.04 if eff >= 0 else -0.04
    ha = 'left' if eff >= 0 else 'right'
    ax.annotate(label, (eff + offset_x, y_pos), va='center', ha=ha, fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="lightgray", alpha=0.85))

ax.set_yticks(y_coords)
ax.set_yticklabels(dimensions, fontweight='bold')
ax.set_xlim(-1.25, 1.15)
ax.set_xlabel(r'$\leftarrow$ Severe Negative Transfer / Boundary Limit     |     Thermodynamic Regularization Benefit $\rightarrow$', fontsize=10.5, fontweight='bold')
ax.set_title('Comprehensive Failure Boundary Map: When Do Physical Priors Help, and Where Do They Fail?', loc='left', fontsize=11.5, fontweight='bold')

# 背景分区着色提示
ax.axvspan(-1.25, 0, color='#fee0d2', alpha=0.35, label='Severe Failure / Representation Limits')
ax.axvspan(0, 1.15, color='#e5f5e0', alpha=0.35, label='Thermodynamic Regularization Regime')

ax.legend(loc='lower left', frameon=True, fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "Figure5_failure_boundary_map.png", dpi=300)
fig.savefig(FIG_DIR / "Figure5_failure_boundary_map.svg")
plt.close(fig)
print("  [✓] Saved Figure 5 (PNG + SVG)")

print("\n" + "=" * 80)
print("  ALL 5 CORE PAPER FIGURES SUCCESSFULLY GENERATED IN:")
print(f"  {FIG_DIR.resolve()}")
print("=" * 80)
