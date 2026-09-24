"""
make_kaggle_v7_retrain_pack.py — 打包 V7 干净重训的独立 Kaggle 执行包 (~4.5MB)
============================================================================
打包内容:
1. 数据集: processed_tri_data_hfc2739/ (data.npy, label.npy, meta_info.csv)
2. 划分文件: splits/HFC_grouped_state_split.npz (Grouped-L0 基准), splits/HFC_all_split.npz
3. V7 代码: v7_shadow_experiment/ (Dataset_v7.py, Model_v7.py, run_v7_pilot.py, __init__.py)
4. Kaggle 调度脚本: run_kaggle_v7_pilot.py
5. 一键执行脚本: run_kaggle_all.py (全量 10 个模型权重自动重训与打包)
"""

import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_OUT = ROOT / "kaggle_v7_retrain_pack.zip"

FILES_TO_PACK = [
    # 1. 核心数据集 (已剔除 R1234yf 污染)
    "processed_tri_data_hfc2739/data.npy",
    "processed_tri_data_hfc2739/label.npy",
    "processed_tri_data_hfc2739/meta_info.csv",
    "processed_tri_data_hfc2739/index_with_anion.csv",

    # 2. 划分文件
    "splits/HFC_grouped_state_split.npz",
    "splits/HFC_grouped_state_split_bound.json",
    "splits/HFC_all_split.npz",

    # 3. V7 架构代码
    "v7_shadow_experiment/__init__.py",
    "v7_shadow_experiment/Dataset_v7.py",
    "v7_shadow_experiment/Model_v7.py",
    "v7_shadow_experiment/run_v7_pilot.py",

    # 4. 控制器
    "run_kaggle_v7_pilot.py",
]

RUN_ALL_SCRIPT = '''# run_kaggle_all.py — Kaggle GPU Master Execution Script for Clean V7 Retraining
import os
import sys
import subprocess
import zipfile
from pathlib import Path

print("=" * 80)
print("  KAGGLE GPU: V7 CLEAN RETRAINING MASTER RUNNER (V7-A & V7-B, 5 SEEDS)")
print("  Split Target: Grouped-L0 (splits/HFC_grouped_state_split.npz)")
print("=" * 80)

# 1. 环境依赖检查与安装
print("[1/3] 检查并安装 torch_geometric 依赖...")
try:
    import torch_geometric
    print("  torch_geometric 已安装！")
except ImportError:
    print("  正在安装 torch_geometric...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"], check=True)

# 1.5 运行时血统契约预检
print("\\n[*] 运行血统契约预检 (Lineage Contract Preflight)...")
import hashlib, json
split_p = Path("splits/HFC_grouped_state_split.npz")
bound_p = Path("splits/HFC_grouped_state_split_bound.json")
if bound_p.exists() and split_p.exists():
    with open(bound_p, "r", encoding="utf-8") as f_b:
        b_meta = json.load(f_b)
    s_sha = hashlib.sha256(split_p.read_bytes()).hexdigest()
    assert s_sha == b_meta["split_npz_sha256"], "Split SHA256 does not match bound metadata!"
    print(f"  ✅ 划分文件与血统绑定元数据 100% 吻合 (SHA: {s_sha[:16]}...)")

# 2. 运行 V7-A (5 种子: 42, 43, 44, 45, 46)
print("\\n" + "=" * 80)
print(">>> [2/3] 启动 V7-A 全量 5 种子训练...")
print("=" * 80)
cmd_v7a = [
    sys.executable, "run_kaggle_v7_pilot.py",
    "--model", "V7-A",
    "--seeds", "42,43,44,45,46",
    "--split-file", "splits/HFC_grouped_state_split.npz"
]
subprocess.run(cmd_v7a, check=True)

# 3. 运行 V7-B (5 种子: 42, 43, 44, 45, 46)
print("\\n" + "=" * 80)
print(">>> [3/3] 启动 V7-B 全量 5 种子训练...")
print("=" * 80)
cmd_v7b = [
    sys.executable, "run_kaggle_v7_pilot.py",
    "--model", "V7-B",
    "--seeds", "42,43,44,45,46",
    "--split-file", "splits/HFC_grouped_state_split.npz"
]
subprocess.run(cmd_v7b, check=True)

# 4. 汇总打包所有结果
print("\\n" + "=" * 80)
print("  PACKAGING ALL 10 RETRAINED CHECKPOINTS & SUMMARIES")
print("=" * 80)

ROOT = Path(".").resolve()
zip_filename = ROOT / "v7_retrained_gpu_results.zip"
ckpt_dir = ROOT / "v7_shadow_experiment" / "checkpoints"
shadow_dir = ROOT / "v7_shadow_experiment"

with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    if ckpt_dir.exists():
        for file in ckpt_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(ROOT)
                zipf.write(file, arcname)
                print(f"  Added: {arcname}")

    for file in shadow_dir.glob("pilot_summary_*.csv"):
        arcname = file.relative_to(ROOT)
        zipf.write(file, arcname)
        print(f"  Added: {arcname}")

print(f"\\n🎉 全量重训成功完成！产物压缩包已就绪: {zip_filename.resolve()}")
print("请在 Kaggle Notebook 输出区下载 'v7_retrained_gpu_results.zip' 并解压至本地项目根目录！")
'''

def main():
    print("=" * 80)
    print("  PACKAGING V7 RETRAINING STANDALONE PACK FOR KAGGLE")
    print("=" * 80)

    with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
        for rel in FILES_TO_PACK:
            f = ROOT / rel
            if f.exists():
                zf.write(f, rel)
                print(f"  + {rel} ({f.stat().st_size / 1024:.1f} KB)")
            else:
                raise FileNotFoundError(f"Missing required file: {f}")

        # 写入一键启动脚本
        zf.writestr("run_kaggle_all.py", RUN_ALL_SCRIPT)
        print("  + run_kaggle_all.py (Kaggle 全量重训一键启动脚本)")

    zip_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
    print("=" * 80)
    print(f"[SUCCESS] Kaggle execution package created: {ZIP_OUT.name} ({zip_mb:.2f} MB)")
    print("=" * 80)

if __name__ == '__main__':
    main()
