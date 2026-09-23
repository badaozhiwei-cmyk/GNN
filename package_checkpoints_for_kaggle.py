"""
package_checkpoints_for_kaggle.py — Pack Local V7 Checkpoints for Kaggle Dataset Upload
======================================================================================
功能：
将本地 D:\折腾 中保存完好的 V7-A 与 V7-B 全量 10 个检查点及 Scaler，
自动打包为 v7_checkpoints_for_kaggle.zip，供用户上传为 Kaggle Dataset。
"""

from __future__ import annotations
import os
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent

v7a_dir = Path(r"D:\折腾\v7_pilot_gpu_results\v7_shadow_experiment\checkpoints\V7-A")
v7b_dir = Path(r"D:\折腾\v7_pilot_gpu_results2\v7_shadow_experiment\checkpoints\V7-B")

out_zip = ROOT / "v7_checkpoints_for_kaggle.zip"

print("=" * 80)
print("  PACKAGING LOCAL V7 CHECKPOINTS FOR KAGGLE UPLOAD")
print("=" * 80)

if not v7a_dir.exists():
    print(f"❌ 错误: 未找到 V7-A 目录: {v7a_dir}")
    sys.exit(1)

if not v7b_dir.exists():
    print(f"❌ 错误: 未找到 V7-B 目录: {v7b_dir}")
    sys.exit(1)

seeds = [42, 43, 44, 45, 46]

files_to_pack = []

# V7-A files
files_to_pack.append((v7a_dir / "scalers.pkl", "V7-A/scalers.pkl"))
for s in seeds:
    pth = v7a_dir / f"best_seed_{s}.pth"
    if pth.exists():
        files_to_pack.append((pth, f"V7-A/best_seed_{s}.pth"))
    else:
        print(f"⚠️ 警告: 缺少 V7-A Seed {s}: {pth}")

# V7-B files
files_to_pack.append((v7b_dir / "scalers.pkl", "V7-B/scalers.pkl"))
for s in seeds:
    pth = v7b_dir / f"best_seed_{s}.pth"
    if pth.exists():
        files_to_pack.append((pth, f"V7-B/best_seed_{s}.pth"))
    else:
        print(f"⚠️ 警告: 缺少 V7-B Seed {s}: {pth}")

print(f"[*] 准备打包 {len(files_to_pack)} 个核心权重与 Scaler 文件...")
print(f"[*] 输出压缩包: {out_zip}")

# 使用 zipfile.ZIP_STORED (不压缩，仅归档) 可以瞬间完成打包，且上传 Kaggle 解压极快
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as zf:
    for src, arc in files_to_pack:
        print(f"  + 写入: {arc} ({src.stat().st_size / 1024 / 1024:.1f} MB)...")
        zf.write(src, arc)

total_gb = out_zip.stat().st_size / 1024 / 1024 / 1024
print("=" * 80)
print(f"✅ 打包成功完成！文件位置: {out_zip} ({total_gb:.2f} GB)")
print("=" * 80)
