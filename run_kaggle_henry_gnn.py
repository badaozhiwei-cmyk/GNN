r"""
run_kaggle_henry_gnn.py
=======================
Kaggle GPU 一键主调度脚本：
1. 自动检测并安装 torch_geometric 依赖
2. 自动检测 GPU
3. 运行 12 折留一冷媒 (12-fold LORO) 全量 5 种子 (Seeds: 42~46) 训练
4. 自动将结果目录打包为 results_henry_gnn_12folds.zip 供右侧 Output 直接下载
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import subprocess
import zipfile
from pathlib import Path

print("=" * 80)
print("  🚀 KAGGLE GPU: HENRY-HEAD GNN 12-FOLD LORO 5-SEED MASTER RUNNER")
print("  Formula: y = P_raw * Softplus(MLP(x_g, cond))")
print("=" * 80)

# 1. 检查并安装依赖
print("\n[1/3] 检查并配置依赖环境...")
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
    print("  ⚠️ Warning: 未检测到 GPU，将以 CPU 降级运行！")

# 2. 启动 12 折 LORO 全量 5 种子训练
out_dir = "results_henry_gnn_12folds"
print("\n" + "=" * 80)
print(f">>> [2/3] 启动全量 12 折 LORO 训练 (Mreduced, 5 Seeds: 42-46, Output: {out_dir})...")
print("=" * 80)

cmd = [
    sys.executable, "run_ablation.py",
    "--mode", "loro",
    "--descriptor_mode", "Mreduced",
    "--family", "HFC",
    "--epoch", "100",
    "--patience", "15",
    "--seeds", "5",
    "--use_henry_head",
    "--out_dir", out_dir
]

print(f"执行指令: {' '.join(cmd)}")
subprocess.run(cmd, check=True)

# 3. 打包输出
print("\n" + "=" * 80)
print(">>> [3/3] 正在打包结果目录为 ZIP 文件以便下载...")
print("=" * 80)

zip_out = "results_henry_gnn_12folds.zip"
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, ".")
            zf.write(full_p, rel_p)

print(f"\n🎉 恭喜！12 折全量训练圆满完成！")
print(f"📦 结果文件已打包为: {zip_out}")
print("👉 请在 Kaggle Notebook 右侧 'Output' 面板中直接下载该 ZIP 文件！")
