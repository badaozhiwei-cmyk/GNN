"""
run_kaggle_step25_attribution.py — Kaggle GPU One-Click Dedicated Attribution Runner
=====================================================================================
【在 Kaggle Notebook (开启 T4 GPU) 中运行】:
    %cd /kaggle/working/GNN
    !git pull origin main
    !pip install -q torch_geometric rdkit
    !python run_kaggle_step25_attribution.py

【自动化流水线】:
1. 自动检查 HFC_all_M0 与 B2_M0 模型权重；若未预先生成，则自动调用 GPU 快速拟合 (~2 分钟);
2. 自动调用 GPU 完成 43 个目标样本在 5 个种子下的全量图与官能团积分归因 (215 次评测, ~30 秒);
3. 评估 Top-1 关键基团的特征遮蔽保真度 (R_faith) 与 5 种子跨 Seed 稳定性;
4. 严格核验 Gate A ~ Gate E 质量门禁;
5. 自动打包全部生成物至 step25_attribution_results.zip，供一键下载至本地。
"""

from __future__ import annotations
import os
import sys
import time
import zipfile
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

print("=" * 80)
print("  KAGGLE GPU: DEDICATED STEP 25 ATTRIBUTION & FAITHFULNESS RUNNER")
print("=" * 80)

# 1. 检测 PyTorch 与 GPU 状态
import torch
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"[硬件检测] 当前计算设备: {device_name} (CUDA Available: {torch.cuda.is_available()})")

# 2. 依赖检查点自动检测与按需补全训练
ckpt_hfc = ROOT / "results_hfc_all/HFC_all_M0/best_seed_42.pth"
if not ckpt_hfc.exists():
    print("\n>>> [STEP 1/3] 检测到 HFC_all_M0 检查点缺失，正在自动调用 GPU 进行 5 种子生产模型基准训练 (~1.5 分钟)...")
    subprocess.run(
        [sys.executable, "research_pipeline/run_hfc_all_production.py", "--mode", "M0", "--seeds", "42,43,44,45,46"],
        check=True
    )
else:
    print(f"[INFO] 已检测到 HFC_all_M0 检查点 ({ckpt_hfc})，跳过训练。")

ckpt_b2 = ROOT / "results_split_B/B2_M0/best_seed_42.pth"
if not ckpt_b2.exists():
    print("\n>>> [STEP 2/3] 检测到 B2_M0 检查点缺失，正在自动调用 GPU 进行 5 种子 B2 OOD 基准训练 (~1 分钟)...")
    subprocess.run(
        [sys.executable, "research_pipeline/run_split_B_benchmark.py", "--split", "B2", "--mode", "M0", "--seeds", "42,43,44,45,46"],
        check=True
    )
else:
    print(f"[INFO] 已检测到 B2_M0 检查点 ({ckpt_b2})，跳过训练。")

# 3. 执行全量归因评估
cmd = [
    sys.executable,
    "research_pipeline/step25_graph_substructure_attribution.py",
    "--run_manifest",
    "--steps", "25",
    "--n_random_trials", "30",
]
if not torch.cuda.is_available():
    cmd.append("--cpu")

print("\n>>> [STEP 3/3] 正在启动全量归因与保真度评估流水线 (~30 秒)...")
t0 = time.time()
res = subprocess.run(cmd)
t_elapsed = time.time() - t0

if res.returncode != 0:
    print(f"\n❌ [ERROR] 归因计算失败，退出码: {res.returncode}")
    sys.exit(res.returncode)

print(f"\n✅ [SUCCESS] 全量归因与 Gate A~E 核验成功！总耗时: {t_elapsed:.1f} 秒")

# 4. 自动执行 Step 25 Integrity Audit v2 (张量等价性证明 + SMARTS覆盖率 + 客观分层)
print("\n>>> [STEP 4/4] 正在执行 Step 25 Integrity Audit v2 深度审计与打包...")
subprocess.run([sys.executable, "research_pipeline/audit_step25_v2.py"], check=True)

# 5. 打包结果文件
out_zip = ROOT / "step25_attribution_results.zip"
res_dir = ROOT / "results_attribution"
print(f"\n>>> 正在打包核心结果文件至 {out_zip.name}...")

target_files = [
    "case_selection_manifest.csv",
    "graph_attribution_atoms_bonds.csv",
    "graph_attribution_groups.csv",
    "graph_attribution_faithfulness.csv",
    "graph_attribution_stability.csv",
    "graph_attribution_provenance.json",
]

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in target_files:
        p = res_dir / fname
        if p.exists():
            zf.write(p, f"results_attribution/{fname}")
            print(f"  + results_attribution/{fname} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  ! WARNING: Missing {p}")

debug_zip = ROOT / "step25_debug_package.zip"
print("=" * 80)
print(f"🎉 全部计算与审计打包完成！")
print(f"👉 产出包 1: {out_zip.name} ({out_zip.stat().st_size / 1024:.1f} KB)")
if debug_zip.exists():
    print(f"👉 产出包 2: {debug_zip.name} ({debug_zip.stat().st_size / 1024:.1f} KB - 包含完整 v2 审计诊断与 metrics)")
print("👉 请在 Kaggle 界面右侧 Output 面板下载产物。")
print("=" * 80)
