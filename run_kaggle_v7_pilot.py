"""
run_kaggle_v7_pilot.py — Master One-Click Kaggle GPU Controller for V7 Shadow Pilot
===================================================================================
【在 Kaggle Notebook (开启 T4 GPU) 中执行】:
    %cd /kaggle/working
    # 1. 浅克隆仓库最新 Commit (极速下载，不下载多余历史):
    !git clone --depth 1 https://github.com/badaozhiwei-cmyk/GNN.git
    %cd /kaggle/working/GNN
    !git pull origin main

    # 2. 仅安装真正必需的轻量图神经网络库 (训练无需 rdkit):
    !pip install -q torch_geometric

    # 3. 一键启动 V7-A Seed 42 训练:
    !python run_kaggle_v7_pilot.py --model V7-A --seeds 42

【流水线说明】:
1. 自动检测 GPU 硬件与 CUDA 环境；
2. 调度执行 V7 试点训练 (支持 V7-A 或 V7-B, 默认执行 Seed 42 完整收敛训练)；
3. 伴随记录 Huber Loss、Val MAE/R2、Median Δy_graph、Graph-Active Rate；
4. 自动将权重、Scaler、预测表与评估历史打包为 v7_pilot_gpu_results.zip，供一键下载至本地。
"""

from __future__ import annotations
import argparse
import os
import sys
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

print("=" * 80)
print("  KAGGLE GPU: V7 SHADOW EXPERIMENT PILOT MASTER CONTROLLER")
print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Kaggle GPU V7 Pilot Controller")
    parser.add_argument('--model', type=str, default='V7-A', choices=['V7-A', 'V7-B'], help="Target V7 variant")
    parser.add_argument('--seeds', type=str, default='42', help="Target seed(s), e.g. '42' or '42,43,44,45,46'")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    # 1. 硬件检测
    import torch
    is_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if is_cuda else "CPU (Warning: Running on CPU)"
    device_arg = "cuda" if is_cuda else "cpu"
    print(f"\n[硬件检测] 计算设备: {device_name} (CUDA={is_cuda})")
    if not is_cuda:
        print("  ⚠️ 提示: 检测到当前未开启 GPU，建议在 Kaggle 右侧 Settings -> Accelerator 中开启 GPU T4 x2。")

    # 2. 调度执行训练
    cmd = [
        sys.executable, "v7_shadow_experiment/run_v7_pilot.py",
        "--model", args.model,
        "--seeds", args.seeds,
        "--epoch", str(args.epoch),
        "--patience", str(args.patience),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--device", device_arg
    ]

    print(f"\n>>> [TASK START] 正在启动 {args.model} 种子 [{args.seeds}] 的完整训练...")
    print("    CMD: " + " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n❌ [ERROR] 训练脚本异常退出，返回码: {res.returncode}")
        sys.exit(res.returncode)
    print(f"\n>>> [TASK DONE] 训练成功完成！")

    # 3. 自动打包产物
    print("\n" + "=" * 80)
    print("  PACKAGING V7 PILOT RESULTS INTO ZIP FOR EASY DOWNLOAD")
    print("=" * 80)

    zip_filename = ROOT / "v7_pilot_gpu_results.zip"
    ckpt_dir = ROOT / "v7_shadow_experiment" / "checkpoints"
    shadow_dir = ROOT / "v7_shadow_experiment"

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 打包 checkpoints 目录下的全部权重、scaler 与 history
        if ckpt_dir.exists():
            for file in ckpt_dir.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(ROOT)
                    zipf.write(file, arcname)
                    print(f"  Added: {arcname}")

        # 打包 pilot summary csv
        for file in shadow_dir.glob("*.csv"):
            arcname = file.relative_to(ROOT)
            zipf.write(file, arcname)
            print(f"  Added: {arcname}")

    print(f"\n[ALL TASKS COMPLETED SUCCESSFULLY!]")
    print(f"产物压缩包已就绪: {zip_filename.resolve()}")
    print("请直接在 Kaggle Notebook 输出区下载 'v7_pilot_gpu_results.zip' 并解压至本地项目根目录！")

if __name__ == "__main__":
    main()
