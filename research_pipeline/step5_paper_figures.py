"""
step5_paper_figures.py
======================
【目的】
  汇总所有模型的训练结果，生成论文级别的对比表和主图。
  本脚本在所有训练脚本跑完后运行，只做"读 CSV → 绘图"，
  不依赖 PyTorch / PyG，本地 Windows 可以直接运行。

【输入文件（需提前存在）】
  gat_splitA_results.csv   / gat_splitB_results.csv
  gin_splitA_results.csv   / gin_splitB_results.csv
  mpnn_splitA_results.csv  / mpnn_splitB_results.csv
  Shallow_Machine_Learning_for_property_prediction/
    ml_baselines_splitA_results.csv
    ml_baselines_splitB_results.csv
  checkpoints_splitA/pred_ensemble.csv
  checkpoints_splitB/pred_ensemble.csv
  checkpoints_gin_splitA/pred_ensemble.csv  ... 等

【输出文件】
  table2_model_comparison.csv    → 论文 Table 2（直接复制）
  figure/Fig2_model_comparison.png  → 主对比柱状图
  figure/Fig3_ood_generalization.png → Split A vs B 泛化差距
  figure/Fig4_parity_plots.png       → 所有模型散点图（2行4列）

【运行方法】
  python step5_paper_figures.py
"""

import os

# ── 自动定位项目根目录（脚本在 research_pipeline/ 子文件夹中）──────
import pathlib as _pl
ROOT = str(_pl.Path(__file__).resolve().parent.parent)
import os as _os; _os.chdir(ROOT)
# ─────────────────────────────────────────────────────────────────────
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

warnings.filterwarnings('ignore')

# ── 全局字体设置（论文级）────────────────────────────────────
rcParams['font.family']  = 'DejaVu Sans'
rcParams['font.size']    = 11
rcParams['axes.linewidth'] = 1.2

os.makedirs('figure', exist_ok=True)

# ============================================================
# 1. 读取各模型结果 CSV，构建统一的对比 DataFrame
# ============================================================

def read_gnn_results(csv_path, model_name):
    """读取 GNN Runner 生成的 per-seed 结果 CSV"""
    if not os.path.exists(csv_path):
        print(f"  ⚠️  未找到 {csv_path}，跳过 {model_name}")
        return None
    df = pd.read_csv(csv_path)
    row = {
        'Model':      model_name,
        'MAE_mean':   df['MAE'].mean(),
        'MAE_std':    df['MAE'].std(),
        'R2_mean':    df['R2'].mean(),
        'R2_std':     df['R2'].std(),
        'RMSE_mean':  df['RMSE'].mean(),
        'RMSE_std':   df['RMSE'].std(),
        'n_seeds':    len(df),
    }
    return row


def read_ml_results(csv_path):
    """读取 ML 基线结果 CSV（每行一个模型，无 std）"""
    if not os.path.exists(csv_path):
        print(f"  ⚠️  未找到 {csv_path}，跳过 ML 基线")
        return []
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'Model':      r['model'],
            'MAE_mean':   r['MAE'],
            'MAE_std':    np.nan,
            'R2_mean':    r['R2'],
            'R2_std':     np.nan,
            'RMSE_mean':  r['RMSE'],
            'RMSE_std':   np.nan,
            'n_seeds':    1,
        })
    return rows

# ── ML 基线 CSV 路径（注意：放在子目录下）────────────────────
ML_DIR = 'Shallow_Machine_Learning_for_property_prediction'

print("=" * 60)
print("  Step 5：汇总所有模型结果")
print("=" * 60)

results_A = []
results_B = []
results_D = []

# GNN 模型
for model_name, prefix in [('GAT (ours)', 'gat'), ('GIN', 'gin'), ('MPNN', 'mpnn')]:
    r = read_gnn_results(f'{prefix}_splitA_results.csv', model_name)
    if r: results_A.append(r)
    r = read_gnn_results(f'{prefix}_splitB_results.csv', model_name)
    if r: results_B.append(r)
    r = read_gnn_results(f'{prefix}_splitD_results.csv', model_name)
    if r: results_D.append(r)

# ML 基线
ml_A = read_ml_results(os.path.join(ML_DIR, 'ml_baselines_splitA_results.csv'))
ml_B = read_ml_results(os.path.join(ML_DIR, 'ml_baselines_splitB_results.csv'))
ml_D = read_ml_results(os.path.join(ML_DIR, 'ml_baselines_splitD_results.csv'))
results_A.extend(ml_A)
results_B.extend(ml_B)
results_D.extend(ml_D)

df_A = pd.DataFrame(results_A)
df_B = pd.DataFrame(results_B)
df_D = pd.DataFrame(results_D)

if df_A.empty and df_B.empty and df_D.empty:
    print("\n⚠️  没有找到任何结果 CSV，请先运行所有训练脚本。")
    print("   预期文件：gat_splitA_results.csv / gin_splitB_results.csv / gat_splitD_results.csv 等")
    raise SystemExit(0)

print(f"\n  读取完成：Split A {len(df_A)} 个模型 | Split B {len(df_B)} 个模型 | Split D {len(df_D)} 个模型")

# ============================================================
# 2. 生成 Table 2（论文级别格式）
# ============================================================

def fmt(mean, std, precision=4):
    """格式化 mean ± std，如果 std 是 nan 只显示 mean"""
    if pd.isna(std) or std == 0:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"

print("\n" + "=" * 65)
print("  Table 2：模型性能对比")
print("=" * 65)

def build_table(df, split_name):
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        rows.append({
            'Model':       r['Model'],
            'R² (mean±std)':   fmt(r['R2_mean'],   r['R2_std']),
            'MAE (mean±std)':  fmt(r['MAE_mean'],  r['MAE_std']),
            'RMSE (mean±std)': fmt(r['RMSE_mean'], r['RMSE_std']),
        })
    tbl = pd.DataFrame(rows)
    print(f"\n  [Split {split_name}]")
    print(tbl.to_string(index=False))
    return tbl

tbl_A = build_table(df_A, 'A (Random)')
tbl_B = build_table(df_B, 'B (Anion OOD)')
tbl_D = build_table(df_D, 'D (Cation OOD)')

# 保存 CSV（多 header：先 Split A，再 Split B，再 Split D）
with open('table2_model_comparison.csv', 'w', encoding='utf-8-sig') as f:
    f.write('Split A (Random 70/10/20)\n')
    if not tbl_A.empty: tbl_A.to_csv(f, index=False)
    f.write('\nSplit B (Anion Family OOD)\n')
    if not tbl_B.empty: tbl_B.to_csv(f, index=False)
    f.write('\nSplit D (Cation Family OOD)\n')
    if not tbl_D.empty: tbl_D.to_csv(f, index=False)

print(f"\n  📄 Table 2 已保存：table2_model_comparison.csv")

# ============================================================
# 3. Figure 2：多模型性能对比柱状图
# ============================================================
print("\n绘制 Fig 2（多模型对比柱状图）...")

# 定义固定颜色（按模型类型）
MODEL_COLORS = {
    'GAT (ours)': '#E74C3C',  # 红：本文主模型
    'GIN':        '#3498DB',  # 蓝
    'MPNN':       '#27AE60',  # 绿
    'Random Forest': '#F39C12',   # 橙
    'XGBoost':    '#8E44AD',  # 紫
    'LightGBM':   '#1ABC9C',  # 青
    'MLP':        '#95A5A6',  # 灰
}

def get_color(name):
    for k, v in MODEL_COLORS.items():
        if k.lower() in name.lower():
            return v
    return '#BDC3C7'

def plot_comparison(df_A, df_B, df_D, metric, ylabel, save_path, higher_better=True):
    """绘制 Split A / Split B / Split D 三向对比"""
    # 取都有的模型
    common = set(df_A['Model'].tolist()) if not df_A.empty else set()
    if not df_B.empty:
        common = common & set(df_B['Model'].tolist()) if common else set(df_B['Model'].tolist())
    if not df_D.empty:
        common = common & set(df_D['Model'].tolist()) if common else set(df_D['Model'].tolist())
        
    order  = [m for m in (df_A['Model'].tolist() if not df_A.empty else df_B['Model'].tolist()) if m in common]

    n   = len(order)
    x   = np.arange(n)
    w   = 0.25 # 调窄一些以容纳 3 根柱子

    fig, ax = plt.subplots(figsize=(max(12, n * 1.6), 6))
    col_mean = f'{metric}_mean'
    col_std  = f'{metric}_std'

    vals_A = df_A.set_index('Model').loc[order, col_mean].values if not df_A.empty else np.zeros(n)
    stds_A = df_A.set_index('Model').loc[order, col_std].values if not df_A.empty else np.zeros(n)
    
    vals_B = df_B.set_index('Model').loc[order, col_mean].values if not df_B.empty else np.zeros(n)
    stds_B = df_B.set_index('Model').loc[order, col_std].values if not df_B.empty else np.zeros(n)
    
    vals_D = df_D.set_index('Model').loc[order, col_mean].values if not df_D.empty else np.zeros(n)
    stds_D = df_D.set_index('Model').loc[order, col_std].values if not df_D.empty else np.zeros(n)

    colors = [get_color(m) for m in order]

    bars_A = ax.bar(x - w, vals_A,
                    yerr=np.nan_to_num(stds_A), width=w,
                    color=colors, alpha=0.90,
                    edgecolor='white', linewidth=0.7,
                    error_kw=dict(elinewidth=1.2, capsize=3),
                    label='Split A (Random)')
                    
    bars_B = ax.bar(x, vals_B,
                    yerr=np.nan_to_num(stds_B), width=w,
                    color=colors, alpha=0.60,
                    edgecolor='white', linewidth=0.7, hatch='//',
                    error_kw=dict(elinewidth=1.2, capsize=3),
                    label='Split B (Anion OOD)')
                    
    bars_D = ax.bar(x + w, vals_D,
                    yerr=np.nan_to_num(stds_D), width=w,
                    color=colors, alpha=0.40,
                    edgecolor='white', linewidth=0.7, hatch='..',
                    error_kw=dict(elinewidth=1.2, capsize=3),
                    label='Split D (Cation OOD)')

    # 数值标注
    for bar, v in zip(bars_A, vals_A):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (max(vals_A)*0.01 if not np.isnan(bar.get_height()) else 0),
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=45)
    for bar, v in zip(bars_B, vals_B):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (max(vals_B)*0.01 if not np.isnan(bar.get_height()) else 0),
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=45)
    for bar, v in zip(bars_D, vals_D):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (max(vals_D)*0.01 if not np.isnan(bar.get_height()) else 0),
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=10, rotation=20, ha='right')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{ylabel}: All Models — Split A (Solid) vs Split B (Hatched) vs Split D (Dotted)',
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 已保存：{save_path}")

if not df_A.empty or not df_B.empty or not df_D.empty:
    plot_comparison(df_A, df_B, df_D, 'R2',   'R²',   'figure/Fig2a_R2_comparison.png',  higher_better=True)
    plot_comparison(df_A, df_B, df_D, 'MAE',  'MAE',  'figure/Fig2b_MAE_comparison.png', higher_better=False)
    plot_comparison(df_A, df_B, df_D, 'RMSE', 'RMSE', 'figure/Fig2c_RMSE_comparison.png',higher_better=False)

# ============================================================
# 4. Figure 3：OOD 泛化差距（Δ = Split A - Split B）
# ============================================================
print("\n绘制 Fig 3（OOD 泛化差距）...")

if not df_A.empty and not df_B.empty:
    common = set(df_A['Model'].tolist()) & set(df_B['Model'].tolist())
    order  = [m for m in df_A['Model'].tolist() if m in common]

    r2_A  = df_A.set_index('Model').loc[order, 'R2_mean'].values
    r2_B  = df_B.set_index('Model').loc[order, 'R2_mean'].values
    delta = r2_A - r2_B   # 正值 = OOD 时性能下降，绝对值越小 = 泛化越好

    colors_delta = ['#E74C3C' if d > 0.05 else '#27AE60' for d in delta]

    fig, ax = plt.subplots(figsize=(max(9, len(order) * 1.3), 5))
    bars = ax.bar(range(len(order)), delta, color=colors_delta,
                  alpha=0.85, edgecolor='white', linewidth=0.7)

    for bar, v in zip(bars, delta):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003 * np.sign(v),
                f'{v:+.3f}', ha='center',
                va='bottom' if v >= 0 else 'top',
                fontsize=9, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.axhline(y=0.05, color='orange', linewidth=1, linestyle=':', alpha=0.7,
               label='5% threshold')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=10, rotation=20, ha='right')
    ax.set_ylabel('ΔR² = R²(Split A) − R²(Split B)', fontsize=12)
    ax.set_title('Fig 3: OOD Generalization Gap — Smaller is Better',
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figure/Fig3_ood_generalization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  📊 Fig 3 已保存：figure/Fig3_ood_generalization.png")

# ============================================================
# 5. Figure 4：所有模型 Parity Plot（2 行 × N 列）
#    读取各模型的 pred_ensemble.csv
# ============================================================
print("\n绘制 Fig 4（Parity Plot 矩阵）...")

PRED_PATHS = {
    'GAT':   {'A': 'checkpoints_splitA/pred_ensemble.csv',
               'B': 'checkpoints_splitB/pred_ensemble.csv'},
    'GIN':   {'A': 'checkpoints_gin_splitA/pred_ensemble.csv',
               'B': 'checkpoints_gin_splitB/pred_ensemble.csv'},
    'MPNN':  {'A': 'checkpoints_mpnn_splitA/pred_ensemble.csv',
               'B': 'checkpoints_mpnn_splitB/pred_ensemble.csv'},
    'RF':    {'A': f'{ML_DIR}/ml_baselines_splitA_predictions/Random_Forest_pred.csv',
               'B': f'{ML_DIR}/ml_baselines_splitB_predictions/Random_Forest_pred.csv'},
    'LightGBM': {'A': f'{ML_DIR}/ml_baselines_splitA_predictions/LightGBM_pred.csv',
                  'B': f'{ML_DIR}/ml_baselines_splitB_predictions/LightGBM_pred.csv'},
}

# 过滤实际存在的文件
PRED_PATHS_EXIST = {}
for mname, paths in PRED_PATHS.items():
    if os.path.exists(paths['A']) or os.path.exists(paths['B']):
        PRED_PATHS_EXIST[mname] = paths

n_models = len(PRED_PATHS_EXIST)
if n_models > 0:
    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    for col, (mname, paths) in enumerate(PRED_PATHS_EXIST.items()):
        for row, split in enumerate(['A', 'B']):
            ax   = axes[row][col]
            path = paths[split]
            if not os.path.exists(path):
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, color='gray')
                ax.axis('off')
                continue

            pred_df = pd.read_csv(path)
            true_y  = pred_df['true'].values
            # 支持不同列名（pred / pred_ensemble）
            pred_col = 'pred_ensemble' if 'pred_ensemble' in pred_df.columns else 'pred'
            pred_y  = pred_df[pred_col].values

            r2   = r2_score(true_y, pred_y)
            mae  = mean_absolute_error(true_y, pred_y)
            rmse = np.sqrt(mean_squared_error(true_y, pred_y))

            color = get_color(mname)
            ax.scatter(true_y, pred_y, alpha=0.35, s=8,
                       color=color, rasterized=True)
            lims = [min(min(true_y), min(pred_y)) - 0.02,
                    max(max(true_y), max(pred_y)) + 0.02]
            ax.plot(lims, lims, 'k--', lw=1.2)
            ax.set_xlim(lims); ax.set_ylim(lims)

            ax.text(0.05, 0.93,
                    f'R²={r2:.3f}\nMAE={mae:.3f}\nRMSE={rmse:.3f}',
                    transform=ax.transAxes, fontsize=8.5,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            ax.grid(True, linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if row == 0:
                ax.set_title(mname, fontsize=12, fontweight='bold', pad=6)
            if col == 0:
                ax.set_ylabel(f'Split {"A (Random)" if split=="A" else "B (OOD)"}\nPredicted x₁',
                              fontsize=9)
            if row == 1:
                ax.set_xlabel('Experimental x₁', fontsize=9)

    plt.suptitle('Fig 4: Parity Plots — All Models (Split A: Random / Split B: Anion OOD)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('figure/Fig4_parity_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  📊 Fig 4 已保存：figure/Fig4_parity_plots.png")
else:
    print("  ⚠️  未找到任何 pred_ensemble.csv，跳过 Fig 4")

# ============================================================
# 6. 最终汇总打印
# ============================================================
print("\n" + "=" * 60)
print("  ✅ Step 5 全部完成！")
print("=" * 60)
print("\n  论文主要产出文件清单：")
print("  ─────────────────────────────────────────────────")
print("  📄 table2_model_comparison.csv    → 论文 Table 2")
print("  📊 figure/Fig2a_R2_comparison.png  → Fig 2a (R²对比)")
print("  📊 figure/Fig2b_MAE_comparison.png → Fig 2b (MAE对比)")
print("  📊 figure/Fig2c_RMSE_comparison.png→ Fig 2c (RMSE对比)")
print("  📊 figure/Fig3_ood_generalization.png → Fig 3 (泛化差距)")
print("  📊 figure/Fig4_parity_plots.png    → Fig 4 (散点图矩阵)")
print("  📊 figure/Fig5_fgca_global_*.png   → Fig 5 (FGCA排行)")
print("  📊 figure/Fig6_fgca_casestudy*.png → Fig 6 (案例热力图)")
print("  📊 figure/FigS1~S4_*.png           → Supplementary Figs")
print("  ─────────────────────────────────────────────────")
print("\n  🎉 全部脚本执行完毕，可以开始写论文了！")
