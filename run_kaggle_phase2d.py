"""
run_kaggle_phase2d.py — Master One-Click GPU Pipeline for Phase II-D
=====================================================================
【Kaggle 执行指令】
  在 Kaggle Notebook (开启 T4 GPU) 中直接执行:
    python run_kaggle_phase2d.py

【自动化任务流 (~8-10 分钟全部跑完)】
  1. 训练 L2 组分重组基准 (10 runs: M0 & Mreduced x 5 seeds) -> results_split_L2/
  2. 训练 All-HFC 生产基线 (10 runs: M0 & Mreduced x 5 seeds) -> results_hfc_all/
  3. 全量 1106 点不饱和工质 (HFO/HCFO) 零样本前向推断:
     - 主模型: HFC_all (无任何阴离子/冷媒约束的全量饱和 HFC 生产模型)
     - 对照模型: L2 (未见 4 对离子对)
     - 对照模型: B1 (未见磺酸盐家族)
  4. 自动打包全部产物为 phase2d_gpu_results.zip
"""

from __future__ import annotations
import os
import sys
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

print("=" * 80)
print("  PHASE II-D MASTER GPU EXECUTION CONTROLLER")
print("=" * 80)

def run_cmd(cmd_list: list[str], desc: str):
    print(f"\n>>> [TASK START] {desc}")
    print("    CMD: " + " ".join(cmd_list))
    res = subprocess.run(cmd_list, check=True)
    print(f">>> [TASK DONE] {desc} (Exit Code: {res.returncode})")

# 1. 训练 L2 (10 runs)
run_cmd(
    [sys.executable, "research_pipeline/run_split_L2_benchmark.py", "--mode", "all", "--seeds", "42,43,44,45,46"],
    "Step 1: Training L2 Compositional Recombination Benchmark (10 runs)"
)

# 2. 训练 All-HFC 生产模型 (10 runs)
run_cmd(
    [sys.executable, "research_pipeline/run_hfc_all_production.py", "--mode", "all", "--seeds", "42,43,44,45,46"],
    "Step 2: Training All-HFC Production Baseline Models (10 runs)"
)

# 3. 运行 1106 点不饱和工质 (HFO/HCFO) Zero-Shot 探测
# 3.1 主模型: HFC_all
run_cmd(
    [sys.executable, "research_pipeline/step16_hfo_zeroshot_probe.py", "--model_source", "HFC_all", "--target_scope", "all"],
    "Step 3.1: Zero-Shot Unsaturated Probe with Primary Model (HFC_all)"
)

# 3.2 敏感性对照: L2
run_cmd(
    [sys.executable, "research_pipeline/step16_hfo_zeroshot_probe.py", "--model_source", "L2", "--target_scope", "all"],
    "Step 3.2: Zero-Shot Unsaturated Probe with Sensitivity Check (L2)"
)

# 3.3 敏感性对照: B1 (若存在权重则执行，若使用 Lite 包则跳过)
b1_ckpt = ROOT / "results_split_B" / "B1_M0" / "best_seed_42.pth"
if b1_ckpt.exists():
    run_cmd(
        [sys.executable, "research_pipeline/step16_hfo_zeroshot_probe.py", "--model_source", "B1", "--target_scope", "all"],
        "Step 3.3: Zero-Shot Unsaturated Probe with Sensitivity Check (B1)"
    )
else:
    print("\n[INFO] Skipping Step 3.3 (B1 checkpoints not in upload pack; B1 anchor probe already preserved locally).")

# 4. 自动打包产物
print("\n" + "=" * 80)
print("  PACKAGING RESULTS INTO ZIP FOR EASY DOWNLOAD")
print("=" * 80)

zip_filename = ROOT / "phase2d_gpu_results.zip"
with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    # 打包 results_split_L2
    l2_dir = ROOT / "results_split_L2"
    if l2_dir.exists():
        for file in l2_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(ROOT)
                zipf.write(file, arcname)
                print(f"  Added: {arcname}")

    # 打包 results_hfc_all
    hfc_dir = ROOT / "results_hfc_all"
    if hfc_dir.exists():
        for file in hfc_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(ROOT)
                zipf.write(file, arcname)
                print(f"  Added: {arcname}")

    # 打包 paper_results 中的 HFO/HCFO 结果
    paper_dir = ROOT / "paper_results"
    if paper_dir.exists():
        for file in paper_dir.glob("*hfo*"):
            arcname = file.relative_to(ROOT)
            zipf.write(file, arcname)
            print(f"  Added: {arcname}")

print(f"\n[ALL COMPLETED SUCCESSFULLY!]")
print(f"Result archive created: {zip_filename.resolve()}")
print("Download phase2d_gpu_results.zip and extract into your local project root!")
