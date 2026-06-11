"""
消融实验：GIN 层数对比
======================
自动训练 num_gin_layer = 2, 3, 4, 5 的模型并汇总结果。
直接在 Colab 里运行：
    %cd /content/GNN/GNN_for_property_prediction
    !python ablation_layers.py

结果保存至 ablation_results_layers.csv
"""

import os, sys, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── 复用 v2 的数据集和模型 ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from GIN_Runner_v2 import (
    IL_set_v2, Runner, set_seed, plot_results,
    EarlyStopping
)
from Model import GIN

# ============================================================
# 基础配置（只改 num_gin_layer，其余全部固定！）
# ============================================================
BASE_ARGS = {
    'data_path':     '../processed_tri_data/',
    'batch_size':    64,
    'lr':            0.001,
    'epoch':         100,
    'weight_decay':  1e-6,
    'emb_dim':       300,
    'feat_dim':      512,
    'drop_ratio':    0.2,
    'pool':          'mean',
    'T_0':           30,
    'T_mult':        2,
    'patience':      20,
}

# 要测试的层数列表
LAYER_LIST = [2, 3, 4, 5]
SEEDS      = [42, 7, 1]      # 3 个 seed 取平均，与 v2 保持一致
SPLIT_SEED = 42               # 固定数据划分，保证对比公平！

# ============================================================
# 主程序
# ============================================================
def run_ablation():
    print("=" * 60)
    print("  GIN 层数消融实验")
    print(f"  测试层数: {LAYER_LIST}")
    print(f"  每层随机种子: {SEEDS}")
    print("=" * 60)

    # ── 加载数据（只加载一次）──
    print("\n[加载数据集]...")
    whole_set = IL_set_v2(path=BASE_ARGS['data_path'])
    n = len(whole_set)
    train_size = int(n * 0.7)
    test_size  = int(n * 0.2)
    dev_size   = n - train_size - test_size

    # 固定数据划分！不同层数用完全相同的 train/dev/test
    train_set, dev_set, test_set = random_split(
        whole_set, [train_size, dev_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )
    test_loader = DataLoader(test_set, batch_size=BASE_ARGS['batch_size'], shuffle=False)
    print(f"  Train: {train_size}  Dev: {dev_size}  Test: {test_size}\n")

    # ── 汇总表 ──
    summary_rows = []

    for n_layers in LAYER_LIST:
        print(f"\n{'─'*60}")
        print(f"  ▶ num_gin_layer = {n_layers}")
        print(f"{'─'*60}")

        # 修改层数，其他参数不变
        args = copy.deepcopy(BASE_ARGS)
        args['num_gin_layer'] = n_layers

        seed_r2s, seed_rmses, seed_maes = [], [], []
        all_preds = []
        test_true = None

        for seed in SEEDS:
            set_seed(seed)
            train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True)
            dev_loader   = DataLoader(dev_set,   batch_size=args['batch_size'], shuffle=False)

            runner = Runner(args, seed=seed)
            runner.train(train_loader, dev_loader)
            pred, true = runner.test(test_loader)

            r2   = r2_score(true, pred)
            rmse = mean_squared_error(true, pred) ** 0.5
            mae  = mean_absolute_error(true, pred)

            print(f"    Seed {seed}: R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
            seed_r2s.append(r2)
            seed_rmses.append(rmse)
            seed_maes.append(mae)
            all_preds.append(pred)
            test_true = true

        # 3-seed 集成
        ens_pred = np.mean(all_preds, axis=0).tolist()
        ens_r2   = r2_score(test_true, ens_pred)
        ens_rmse = mean_squared_error(test_true, ens_pred) ** 0.5
        ens_mae  = mean_absolute_error(test_true, ens_pred)

        print(f"\n  → 均值 R²: {np.mean(seed_r2s):.4f} (±{np.std(seed_r2s):.4f})")
        print(f"  → 集成 R²: {ens_r2:.4f}  RMSE: {ens_rmse:.4f}  MAE: {ens_mae:.4f}")

        # 保存散点图
        plot_results(
            test_true, ens_pred,
            f"GIN {n_layers}-Layer Ensemble (R²={ens_r2:.4f})",
            f"ablation_layer{n_layers}_ensemble"
        )

        summary_rows.append({
            'num_layers':     n_layers,
            'mean_R2':        round(np.mean(seed_r2s), 4),
            'std_R2':         round(np.std(seed_r2s),  4),
            'ensemble_R2':    round(ens_r2,   4),
            'ensemble_RMSE':  round(ens_rmse, 4),
            'ensemble_MAE':   round(ens_mae,  4),
        })

    # ============================================================
    # 汇总结果表格
    # ============================================================
    df = pd.DataFrame(summary_rows)
    df.to_csv('ablation_results_layers.csv', index=False)

    print(f"\n{'='*60}")
    print("  消融实验完成！汇总结果：")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n  📄 已保存至: ablation_results_layers.csv")

    # ── 画对比柱状图 ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ['ensemble_R2', 'ensemble_RMSE', 'ensemble_MAE']
    ylabels = ['R²', 'RMSE', 'MAE']
    colors  = ['#2ecc71' if r == df['ensemble_R2'].max() else '#3498db'
               for r in df['ensemble_R2']]

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        bars = ax.bar([f"{n}层" for n in df['num_layers']],
                      df[metric], color=colors, edgecolor='white', linewidth=1.2)
        ax.set_title(f'GIN 层数 vs {ylabel}', fontsize=13, fontweight='bold')
        ax.set_xlabel('GIN 层数', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        # 在柱子上标数值
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        ax.set_ylim(bottom=df[metric].min() * 0.95)

    plt.suptitle('Ablation Study: Number of GIN Layers', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('ablation_layers_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  📊 对比图已保存至: ablation_layers_comparison.png")


if __name__ == '__main__':
    run_ablation()
