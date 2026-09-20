"""
package_step25_debug.py — Package complete Step 25 debug suite for rigorous audit
================================================================================
按用户指定的目录规范打包 step25_debug_package.zip，同时在本地生成 seed_metrics/ 目录副本。
"""

import sys
import shutil
import zipfile
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent

# 1. 准备本地 seed_metrics 目录
seed_metrics_dir = ROOT / "seed_metrics"
seed_metrics_dir.mkdir(parents=True, exist_ok=True)

shutil.copy2(
    ROOT / "results_hfc_all/HFC_all_M0/seed_val_metrics.csv",
    seed_metrics_dir / "HFC_all_M0_seed_metrics.csv"
)
shutil.copy2(
    ROOT / "results_hfc_all/HFC_all_Mreduced/seed_val_metrics.csv",
    seed_metrics_dir / "HFC_all_Mreduced_seed_metrics.csv"
)
shutil.copy2(
    ROOT / "results_split_B/B2_M0/seed_metrics.csv",
    seed_metrics_dir / "B2_M0_seed_metrics.csv"
)

# 2. 构建 step25_debug_package.zip
zip_path = ROOT / "step25_debug_package.zip"
print(f">>> 正在创建 {zip_path.name}...")

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    # A. results_attribution (6 files)
    res_dir = ROOT / "results_attribution"
    for fname in [
        "case_selection_manifest.csv",
        "graph_attribution_atoms_bonds.csv",
        "graph_attribution_groups.csv",
        "graph_attribution_faithfulness.csv",
        "graph_attribution_stability.csv",
        "graph_attribution_provenance.json",
    ]:
        p = res_dir / fname
        if p.exists():
            zf.write(p, f"results_attribution/{fname}")
            print(f"  [+] results_attribution/{fname} ({p.stat().st_size / 1024:.1f} KB)")

    # B. diagnostic & audit scripts
    for sname in [
        "diagnostic_seed_shortcut.py",
        "diagnostic_virtual_node.py",
        "audit_step25_attribution.py",
    ]:
        p = ROOT / "research_pipeline" / sname
        if p.exists():
            zf.write(p, sname)
            print(f"  [+] {sname}")

    # C. diagnostic_outputs
    diag_out_dir = ROOT / "diagnostic_outputs"
    for p in sorted(diag_out_dir.glob("*")):
        if p.is_file():
            zf.write(p, f"diagnostic_outputs/{p.name}")
            print(f"  [+] diagnostic_outputs/{p.name} ({p.stat().st_size / 1024:.1f} KB)")

    # D. seed_metrics
    for p in sorted(seed_metrics_dir.glob("*.csv")):
        zf.write(p, f"seed_metrics/{p.name}")
        print(f"  [+] seed_metrics/{p.name} ({p.stat().st_size / 1024:.1f} KB)")

print(f"\n[SUCCESS] 打包完成！文件保存于: {zip_path}")
print(f"   压缩包总大小: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
