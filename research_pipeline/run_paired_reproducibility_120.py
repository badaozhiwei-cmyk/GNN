"""
run_paired_reproducibility_120.py — 成对重训与双轨审计主运行器
================================================================================
执行 120 次标准 LORO 实验：
  - 历史加法基线 (Additive Baseline): 12 折 × 5 种子 = 60 次
  - 原点约束模型 (Henry-Head):         12 折 × 5 种子 = 60 次

全部使用修正后的样本加权验证损失 (Sample-Weighted Validation Loss)，
保持 12 折 LORO 切分、内层冷媒分组验证、随机种子和超参数严格对称。
训练结束后自动在 1,390 个独立物理状态上核算 Raw vs Clipped 双轨指标主表。
================================================================================
"""

import os
import sys
import subprocess
import glob
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable

REFS = [
    'R32', 'R152a', 'R134a', 'R125', 'R143a', 'R23',
    'R41', 'R161', 'R134', 'R227ea', 'R236fa', 'R245fa'
]
SEEDS = [42, 123, 456, 789, 2024]


def run_model_training(model_name, use_henry, out_dir):
    print(f"\n{'='*70}")
    print(f"🚀 开始执行模型重训: {model_name} (总计 12 折 × 5 种子 = 60 次)")
    print(f"   输出目录: {out_dir}")
    print(f"   use_henry_head: {use_henry}")
    print(f"{'='*70}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        PYTHON_EXE,
        os.path.join(ROOT_DIR, "run_ablation.py"),
        "--family", "HFC",
        "--mode", "loro",
        "--descriptor_mode", "Mreduced",
        "--seeds", str(len(SEEDS)),
        "--keep_checkpoints",
        "--out_dir", out_dir
    ]
    if use_henry:
        cmd.append("--use_henry_head")

    print(f"执行指令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"✅ {model_name} 训练完成！")


def evaluate_1390_states(preds_dir, model_name):
    """
    读取 60 份预测文件，在 1,390 个独立物理状态上分别计算各种子指标，再汇总平均。
    同时计算 Raw 与 Clipped 结果。
    """
    csv_files = glob.glob(os.path.join(preds_dir, "*_preds", "seed*.csv"))
    if len(csv_files) < 60:
        print(f"⚠️ 警告: {preds_dir} 下预测文件不足 60 份 (实际找到 {len(csv_files)} 份)")

    dfs = [pd.read_csv(f) for f in csv_files]
    all_df = pd.concat(dfs, ignore_index=True)

    # 物理状态主键
    state_cols = ['IL cation', 'IL anion', 'refrigerant', 'T (K)', 'P (MPa)']

    # 逐种子计算指标
    seed_macro_raw, seed_macro_clip = [], []
    seed_micro_raw, seed_micro_clip = [], []
    seed_rmse_raw, seed_rmse_clip   = [], []
    seed_r2_raw, seed_r2_clip       = [], []
    seed_oob_high, seed_oob_low     = [], []

    for seed in SEEDS:
        sub = all_df[all_df['seed'] == seed].copy()
        if len(sub) == 0:
            continue
        # 归并为 1,390 个物理状态（平均重复测量）
        grouped = sub.groupby(state_cols).agg({
            'true_x1': 'mean',
            'pred_x1_raw': 'mean',
            'pred_x1_clipped': 'mean'
        }).reset_index()

        y_true = grouped['true_x1'].values
        y_raw  = grouped['pred_x1_raw'].values
        y_clip = grouped['pred_x1_clipped'].values

        # 统计越界
        seed_oob_high.append(int((sub['pred_x1_raw'] > 1.0).sum()))
        seed_oob_low.append(int((sub['pred_x1_raw'] < 0.0).sum()))

        # Micro 指标
        seed_micro_raw.append(mean_absolute_error(y_true, y_raw))
        seed_micro_clip.append(mean_absolute_error(y_true, y_clip))
        seed_rmse_raw.append(np.sqrt(mean_squared_error(y_true, y_raw)))
        seed_rmse_clip.append(np.sqrt(mean_squared_error(y_true, y_clip)))
        seed_r2_raw.append(r2_score(y_true, y_raw))
        seed_r2_clip.append(r2_score(y_true, y_clip))

        # Macro-MAE (冷媒等权)
        macro_raw_list, macro_clip_list = [], []
        for ref in grouped['refrigerant'].unique():
            ref_sub = grouped[grouped['refrigerant'] == ref]
            macro_raw_list.append(mean_absolute_error(ref_sub['true_x1'], ref_sub['pred_x1_raw']))
            macro_clip_list.append(mean_absolute_error(ref_sub['true_x1'], ref_sub['pred_x1_clipped']))
        seed_macro_raw.append(np.mean(macro_raw_list))
        seed_macro_clip.append(np.mean(macro_clip_list))

    return {
        'model': model_name,
        'macro_mae_raw': np.mean(seed_macro_raw),
        'macro_mae_clipped': np.mean(seed_macro_clip),
        'micro_mae_raw': np.mean(seed_micro_raw),
        'micro_mae_clipped': np.mean(seed_micro_clip),
        'micro_rmse_raw': np.mean(seed_rmse_raw),
        'micro_rmse_clipped': np.mean(seed_rmse_clip),
        'micro_r2_raw': np.mean(seed_r2_raw),
        'micro_r2_clipped': np.mean(seed_r2_clip),
        'oob_gt_1': np.sum(seed_oob_high),
        'oob_lt_0': np.sum(seed_oob_low)
    }


def main():
    baseline_dir = os.path.join(ROOT_DIR, "results_ablation", "reproduced_baseline_Mreduced")
    henry_dir    = os.path.join(ROOT_DIR, "results_ablation", "reproduced_henry_Mreduced")

    # 1. 运行历史加法基线重训 (60 次)
    if len(glob.glob(os.path.join(baseline_dir, "*_preds", "seed*.csv"))) < 60:
        run_model_training("Historical_Additive_Baseline", use_henry=False, out_dir=baseline_dir)
    else:
        print(f"ℹ️ 检测到基线 60 份预测已存在，跳过训练直接审计: {baseline_dir}")

    # 2. 运行 Henry-Head 重训 (60 次)
    if len(glob.glob(os.path.join(henry_dir, "*_preds", "seed*.csv"))) < 60:
        run_model_training("Origin_Constrained_Henry_Head", use_henry=True, out_dir=henry_dir)
    else:
        print(f"ℹ️ 检测到 Henry-Head 60 份预测已存在，跳过训练直接审计: {henry_dir}")

    # 3. 统计审计主表
    print(f"\n{'='*70}")
    print("📊 正在在 1,390 个独立物理状态上核算双轨指标 (Raw vs Clipped)...")
    base_metrics  = evaluate_1390_states(baseline_dir, "Historical_Additive_Baseline")
    henry_metrics = evaluate_1390_states(henry_dir, "Origin_Constrained_Henry_Head")

    summary_df = pd.DataFrame([base_metrics, henry_metrics])
    out_table = os.path.join(ROOT_DIR, "paper_results", "table_reproduced_paired_loro_120.csv")
    os.makedirs(os.path.dirname(out_table), exist_ok=True)
    summary_df.to_csv(out_table, index=False)

    print(f"\n{'='*70}")
    print("🏆 成对重训 120 组实验终极双轨结果主表 (1,390 独立物理状态)")
    print(f"{'='*70}")
    print(f"指标                      加法基线 (Raw)   Henry (Raw)   | 加法基线 (Clip)  Henry (Clip)")
    print(f"--------------------------------------------------------------------------------")
    print(f"Macro-MAE (冷媒等权)       {base_metrics['macro_mae_raw']:.6f}       {henry_metrics['macro_mae_raw']:.6f}   | {base_metrics['macro_mae_clipped']:.6f}        {henry_metrics['macro_mae_clipped']:.6f}")
    print(f"Micro-MAE (状态等权)       {base_metrics['micro_mae_raw']:.6f}       {henry_metrics['micro_mae_raw']:.6f}   | {base_metrics['micro_mae_clipped']:.6f}        {henry_metrics['micro_mae_clipped']:.6f}")
    print(f"Micro-RMSE (均方根误差)    {base_metrics['micro_rmse_raw']:.6f}       {henry_metrics['micro_rmse_raw']:.6f}   | {base_metrics['micro_rmse_clipped']:.6f}        {henry_metrics['micro_rmse_clipped']:.6f}")
    print(f"Micro-R² (决定系数)        {base_metrics['micro_r2_raw']:.6f}       {henry_metrics['micro_r2_raw']:.6f}   | {base_metrics['micro_r2_clipped']:.6f}        {henry_metrics['micro_r2_clipped']:.6f}")
    print(f"越界计数 (> 1.0)           {base_metrics['oob_gt_1']}                 {henry_metrics['oob_gt_1']}            | 0                0")
    print(f"越界计数 (< 0.0)           {base_metrics['oob_lt_0']}                 {henry_metrics['oob_lt_0']}            | 0                0")
    print(f"{'='*70}")
    print(f"主表已保存至: {out_table}")


if __name__ == '__main__':
    main()
