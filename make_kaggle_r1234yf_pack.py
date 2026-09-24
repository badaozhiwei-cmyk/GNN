"""
make_kaggle_r1234yf_pack.py — 打包 12 个 R1234yf xTB 配对重算的超轻量 Kaggle 执行包 (< 1MB)
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_OUT = ROOT / "kaggle_r1234yf_pack.zip"

FILES_TO_PACK = [
    "Phase4_Scientific_Validation/recompute_r1234yf_and_sentinels_xtb.py",
    "Phase4_Scientific_Validation/compute_full_pair_interaction_xtb.py",
    "Phase4_Scientific_Validation/chemical_identity_contract.py",
    "Phase4_Scientific_Validation/refrigerant_identity_manifest.csv",
    "Phase4_Scientific_Validation/ionic_liquid_identity_manifest.csv",
    "Phase4_Scientific_Validation/full_pair_interaction_results.csv",
    "Phase4_Scientific_Validation/sentinel_selection_manifest.csv",
    "Phase4_Scientific_Validation/xTB_Physics_Descriptors.csv",
    "index_with_anion.csv"
]

print("=" * 80)
print("  PACKAGING TARGETED R1234yf XTB + 16 SENTINELS SEAM AUDIT FOR KAGGLE")
print("=" * 80)

with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    for rel in FILES_TO_PACK:
        f = ROOT / rel
        if f.exists():
            zf.write(f, rel)
            print(f"  + {rel} ({f.stat().st_size / 1024:.1f} KB)")
        else:
            raise FileNotFoundError(f"Missing required file: {f}")

    # 写入一个顶层启动脚本 run_kaggle.py
    run_script_content = '''# Kaggle Runner for 12 R1234yf + 16 Sentinels Seam Audit
import os, sys, subprocess

print("📦 正在安装与配置运行环境 (RDKit, xTB)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], check=True)

# 自动下载 Linux xTB 二进制
xtb_tar = "/kaggle/working/xtb-6.6.1-linux-x86_64.tar.xz"
if not os.path.exists("/kaggle/working/xtb-dist"):
    print("⬇️ 下载 GFN2-xTB 官方二进制发布包...")
    subprocess.run(["wget", "-q", "https://github.com/grimme-lab/xtb/releases/download/v6.6.1/xtb-6.6.1-linux-x86_64.tar.xz", "-O", xtb_tar], check=True)
    subprocess.run(["tar", "-xf", xtb_tar, "-C", "/kaggle/working"], check=True)
    subprocess.run(["mv", "/kaggle/working/xtb-6.6.1", "/kaggle/working/xtb-dist"], check=True)

os.environ['PATH'] = "/kaggle/working/xtb-dist/bin:" + os.environ.get('PATH', '')
os.environ['XTBPATH'] = "/kaggle/working/xtb-dist/share/xtb"

print("🚀 启动 12 对 R1234yf + 16 对分层哨兵接缝审计 (GFN2-xTB tight)...")
subprocess.run([sys.executable, "Phase4_Scientific_Validation/recompute_r1234yf_and_sentinels_xtb.py"], check=True)

print("🎉 运行完成！全新清洁工件已生成:")
print("  1. Phase4_Scientific_Validation/full_pair_interaction_results_v7_clean.csv")
print("  2. Phase4_Scientific_Validation/sentinel_seam_audit_results.csv")
print("  3. Phase4_Scientific_Validation/sentinel_orientation_trajectories.csv")
print("  4. Phase4_Scientific_Validation/xTB_Physics_Descriptors_v7_clean.csv")
print("  5. Phase4_Scientific_Validation/xtb_batch_continuity_certificate.json")

import zipfile
output_zip = "/kaggle/working/xtb_clean_results.zip"
files_to_zip = [
    "Phase4_Scientific_Validation/full_pair_interaction_results_v7_clean.csv",
    "Phase4_Scientific_Validation/sentinel_seam_audit_results.csv",
    "Phase4_Scientific_Validation/sentinel_orientation_trajectories.csv",
    "Phase4_Scientific_Validation/xTB_Physics_Descriptors_v7_clean.csv",
    "Phase4_Scientific_Validation/xtb_batch_continuity_certificate.json"
]
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf_out:
    for rel_f in files_to_zip:
        if os.path.exists(rel_f):
            zf_out.write(rel_f, rel_f)
            print(f"  + 写入压缩包: {rel_f}")
print(f"\\n📦 产物压缩包已就绪: {output_zip}")
print("请在 Kaggle Notebook 输出区直接下载 'xtb_clean_results.zip' 并解压至本地！")
'''
    zf.writestr("run_kaggle.py", run_script_content)
    print("  + run_kaggle.py (Kaggle 专用启动脚本)")

zip_kb = ZIP_OUT.stat().st_size / 1024
print("=" * 80)
print(f"[SUCCESS] Kaggle execution package created: {ZIP_OUT.name} ({zip_kb:.1f} KB)")
print("=" * 80)
