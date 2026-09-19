"""
run_hfo_probe_only.py — Kaggle GPU One-Click Dedicated HFO Zero-Shot Runner
==========================================================================
【使用方法】
在 Kaggle Notebook (GPU T4) 中运行:
    !python run_hfo_probe_only.py

若环境中已有 results_hfc_all 权重，15 秒内直接完成前向推断并生成 hfo_corrected_results.zip。
若无，则自动进行 All-HFC 生产基准训练 (~3-4 分钟) 后推断。
"""

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("=" * 80)
print("  KAGGLE GPU: DEDICATED HFO/HCFO ZERO-SHOT PROBE (CORRECTED R1234yf)")
print("=" * 80)

# 1. 检查是否存在 HFC_all 检查点权重，若无则快速训练
hfc_summary = ROOT / "results_hfc_all" / "split_HFC_all_summary.csv"
ckpt_file = ROOT / "results_hfc_all" / "HFC_all_M0" / "best_seed_42.pth"
if hfc_summary.exists() and ckpt_file.exists():
    print("\n[INFO] Found existing results_hfc_all checkpoints. Skipping training!")
else:
    print("\n>>> [STEP 1] Training All-HFC production baseline on GPU (10 runs)...")
    subprocess.run(
        [sys.executable, "research_pipeline/run_hfc_all_production.py", "--mode", "all", "--seeds", "42,43,44,45,46"],
        check=True
    )

# 2. 执行 1106 点 Zero-Shot 前向推断
print("\n>>> [STEP 2] Running 1106-point Zero-Shot Forward Inference on GPU...")
subprocess.run(
    [sys.executable, "research_pipeline/step16_hfo_zeroshot_probe.py", "--model_source", "HFC_all", "--target_scope", "all"],
    check=True
)

# 3. 打包结果为轻量 zip
out_zip = ROOT / "hfo_corrected_results.zip"
paper_dir = ROOT / "paper_results"
print(f"\n>>> [STEP 3] Packaging HFO probe results into {out_zip.name}...")

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in paper_dir.glob("*hfo*"):
        if f.is_file() and not f.name.endswith(".zip"):
            zf.write(f, f.name)
            print(f"  + {f.name} ({f.stat().st_size / 1024:.1f} KB)")

print("=" * 80)
print(f"[SUCCESS] HFO probe completed! Download {out_zip.name} and extract into your workspace.")
print("=" * 80)
