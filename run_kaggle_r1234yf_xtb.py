# run_kaggle_r1234yf_xtb.py — Kaggle Master Runner for 12 R1234yf + 16 Sentinels Seam Audit
import os
import sys
import subprocess
import zipfile
from pathlib import Path

print("=" * 80)
print("  KAGGLE: 12 R1234yf RECOMPUTATION + 16 SENTINELS SEAM AUDIT (GFN2-xTB)")
print("=" * 80)

# 1. 安装 Python 依赖
print("[1/3] 检查并安装 RDKit...")
try:
    import rdkit
    print("  RDKit 已就绪！")
except ImportError:
    print("  正在安装 RDKit...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], check=True)

# 2. 自动下载并配置 Linux xTB 6.6.1 官方二进制引擎
print("\n[2/3] 配置 Linux xTB 6.6.1 计算引擎...")
xtb_tar = "/kaggle/working/xtb-6.6.1-linux-x86_64.tar.xz"
if not os.path.exists("/kaggle/working/xtb-dist/bin/xtb"):
    print("  下载 GFN2-xTB 官方二进制发布包...")
    subprocess.run([
        "wget", "-q",
        "https://github.com/grimme-lab/xtb/releases/download/v6.6.1/xtb-6.6.1-linux-x86_64.tar.xz",
        "-O", xtb_tar
    ], check=True)
    subprocess.run(["tar", "-xf", xtb_tar, "-C", "/kaggle/working"], check=True)
    if os.path.exists("/kaggle/working/xtb-6.6.1"):
        subprocess.run(["mv", "/kaggle/working/xtb-6.6.1", "/kaggle/working/xtb-dist"], check=True)

os.environ['PATH'] = "/kaggle/working/xtb-dist/bin:" + os.environ.get('PATH', '')
os.environ['XTBPATH'] = "/kaggle/working/xtb-dist/share/xtb"

# 验证 xTB 可用性与版本
ver_check = subprocess.run(["xtb", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(f"  xTB 版本检测:\n  {ver_check.stdout.splitlines()[0] if ver_check.stdout else 'xTB ready'}")

# 3. 执行核心计算脚本
print("\n[3/3] 启动 12 对 R1234yf 干净重算与 16 哨兵 4 取向优化全轨迹接缝审计...")
subprocess.run([sys.executable, "Phase4_Scientific_Validation/recompute_r1234yf_and_sentinels_xtb.py"], check=True)

# 4. 自动打包产物至 /kaggle/working/xtb_clean_results.zip
output_zip = "/kaggle/working/xtb_clean_results.zip"
files_to_zip = [
    "Phase4_Scientific_Validation/full_pair_interaction_results_v7_clean.csv",
    "Phase4_Scientific_Validation/sentinel_seam_audit_results.csv",
    "Phase4_Scientific_Validation/sentinel_orientation_trajectories.csv",
    "Phase4_Scientific_Validation/xTB_Physics_Descriptors_v7_clean.csv",
    "Phase4_Scientific_Validation/xtb_batch_continuity_certificate.json"
]

print("\n" + "=" * 80)
print("  PACKAGING CLEAN XTB ARTIFACTS")
print("=" * 80)
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf_out:
    for rel_f in files_to_zip:
        if os.path.exists(rel_f):
            zf_out.write(rel_f, rel_f)
            print(f"  + 写入压缩包: {rel_f}")
        else:
            print(f"  ⚠️ 警告: 预期产物未找到: {rel_f}")

print(f"\n🎉 任务 A 全部完成！产物压缩包已就绪: {output_zip}")
print("请在 Kaggle Notebook 右侧 Output 栏直接下载 'xtb_clean_results.zip'！")
print("=" * 80)
