r"""
run_kaggle_paired_120.py
================================================================================
Kaggle GPU 一键主调度脚本：成对重训 120 组 LORO 实验 (加法基线 60 + Henry 60)
================================================================================
1. 自动检测并安装 torch_geometric 依赖
2. 自动检测 GPU
3. 运行历史加法基线 (Additive Baseline): 12 折 × 5 种子 = 60 次
4. 运行原点约束模型 (Henry-Head):         12 折 × 5 种子 = 60 次
5. 在 1,390 个独立物理状态上自动完成 Raw vs Clipped 双轨指标核算
6. 自动将全部预测与结果主表打包为 reproduced_paired_loro_120_results.zip
================================================================================
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import subprocess
import zipfile
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 80)
print("  🚀 KAGGLE GPU: PAIRED LORO 120 REPRODUCIBILITY RUNNER")
print("  Baseline (60 runs) vs Origin-Constrained Henry-Head (60 runs)")
print("  Sample-Weighted Validation Loss & Dual Evaluation (Raw vs Clipped)")
print("=" * 80)

# 1. 检查并安装依赖
print("\n[1/5] 检查并配置依赖环境...")
try:
    import torch_geometric
    print("  ✅ torch_geometric 已就绪！")
except ImportError:
    print("  ⚙️ 正在安装 torch_geometric...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"], check=True)

import torch
print(f"  • PyTorch Version: {torch.__version__}")
print(f"  • CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  • GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("  ⚠️ Warning: 未检测到 GPU，将以 CPU 运行！")

baseline_dir = "results_baseline_reproduced"
henry_dir    = "results_henry_reproduced"

# 2. 运行历史加法基线 (60 runs)
print("\n" + "=" * 80)
print(f">>> [2/5] 启动历史加法基线 12 折 LORO 5 种子训练 (共 60 次作业)...")
print("=" * 80)
cmd_baseline = [
    sys.executable, "run_ablation.py",
    "--mode", "loro",
    "--descriptor_mode", "Mreduced",
    "--family", "HFC",
    "--epoch", "100",
    "--patience", "15",
    "--seeds", "5",
    "--keep_checkpoints",
    "--out_dir", baseline_dir
]
print(f"执行指令: {' '.join(cmd_baseline)}")
subprocess.run(cmd_baseline, check=True)

# 3. 运行 Henry-Head (60 runs)
print("\n" + "=" * 80)
print(f">>> [3/5] 启动 Henry-Head 12 折 LORO 5 种子训练 (共 60 次作业)...")
print("=" * 80)
cmd_henry = [
    sys.executable, "run_ablation.py",
    "--mode", "loro",
    "--descriptor_mode", "Mreduced",
    "--family", "HFC",
    "--epoch", "100",
    "--patience", "15",
    "--seeds", "5",
    "--use_henry_head",
    "--keep_checkpoints",
    "--out_dir", henry_dir
]
print(f"执行指令: {' '.join(cmd_henry)}")
subprocess.run(cmd_henry, check=True)

# 4. 双轨指标自动核算
print("\n" + "=" * 80)
print(">>> [4/5] 正在在 1,390 个独立物理状态上核算双轨指标 (Raw vs Clipped)...")
print("=" * 80)

def evaluate_dir(preds_dir, model_name):
    csv_files = glob.glob(os.path.join(preds_dir, "*_preds", "seed*.csv"))
    dfs = [pd.read_csv(f) for f in csv_files]
    all_df = pd.concat(dfs, ignore_index=True)

    state_cols = ['IL cation', 'IL anion', 'refrigerant', 'T (K)', 'P (MPa)']
    SEEDS = [42, 123, 456, 789, 2024]

    seed_macro_raw, seed_macro_clip = [], []
    seed_micro_raw, seed_micro_clip = [], []
    seed_rmse_raw, seed_rmse_clip   = [], []
    seed_r2_raw, seed_r2_clip       = [], []
    seed_oob_high, seed_oob_low     = [], []

    for seed in SEEDS:
        sub = all_df[all_df['seed'] == seed].copy()
        if len(sub) == 0:
            continue
        grouped = sub.groupby(state_cols).agg({
            'true_x1': 'mean',
            'pred_x1_raw': 'mean',
            'pred_x1_clipped': 'mean'
        }).reset_index()

        y_true = grouped['true_x1'].values
        y_raw  = grouped['pred_x1_raw'].values
        y_clip = grouped['pred_x1_clipped'].values

        seed_oob_high.append(int((sub['pred_x1_raw'] > 1.0).sum()))
        seed_oob_low.append(int((sub['pred_x1_raw'] < 0.0).sum()))

        seed_micro_raw.append(mean_absolute_error(y_true, y_raw))
        seed_micro_clip.append(mean_absolute_error(y_true, y_clip))
        seed_rmse_raw.append(np.sqrt(mean_squared_error(y_true, y_raw)))
        seed_rmse_clip.append(np.sqrt(mean_squared_error(y_true, y_clip)))
        seed_r2_raw.append(r2_score(y_true, y_raw))
        seed_r2_clip.append(r2_score(y_true, y_clip))

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

base_res = evaluate_dir(baseline_dir, "Historical_Additive_Baseline")
henry_res = evaluate_dir(henry_dir, "Origin_Constrained_Henry_Head")
summary_df = pd.DataFrame([base_res, henry_res])
summary_df.to_csv("table_reproduced_paired_loro_120.csv", index=False)

print(summary_df.to_string())

# 5. 打包输出
print("\n" + "=" * 80)
print(">>> [5/5] 正在打包全部 120 份预测与模型结果为 ZIP 文件...")
print("=" * 80)

zip_out = "reproduced_paired_loro_120_results.zip"
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write("table_reproduced_paired_loro_120.csv")
    for d in [baseline_dir, henry_dir]:
        for root, dirs, files in os.walk(d):
            for f in files:
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, ".")
                zf.write(full_p, rel_p)

print(f"\n🎉 恭喜！成对重训 120 次作业圆满完成！")
print(f"📦 完整结果已打包为: {zip_out}")
print("👉 请在 Kaggle Notebook 右侧 'Output' 面板中直接下载该 ZIP 文件！")
