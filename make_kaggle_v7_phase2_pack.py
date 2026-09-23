"""
make_kaggle_v7_phase2_pack.py — Build Dedicated Phase 2 Graph-IG Kaggle Upload Pack
====================================================================================
打包所有在 Kaggle GPU 上运行 Phase 2 归因所必需的文件（轻量级，~25MB，不包含 3.5GB 权重）：
1. Kaggle 主控脚本 run_kaggle_phase2_v7_graph_ig.py
2. 核心解释性引擎 v7_shadow_experiment/phase2_v7_graph_ig.py
3. 模型与数据接口 Model_v7.py, Dataset_v7.py
4. 43 样本清单 results_attribution/case_selection_manifest.csv
5. 图结构底层数据 processed_tri_data_hfc2739/ 与 processed_tri_data/
"""

from __future__ import annotations
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_OUT = ROOT / "kaggle_v7_phase2_pack.zip"

INCLUDE_FILES = [
    "run_kaggle_phase2_v7_graph_ig.py",
    "v7_shadow_experiment/phase2_v7_graph_ig.py",
    "v7_shadow_experiment/Model_v7.py",
    "v7_shadow_experiment/Dataset_v7.py",
    "index_with_anion.csv",
    "results_attribution/case_selection_manifest.csv",
    "processed_tri_data_hfc2739/data.npy",
    "processed_tri_data_hfc2739/meta_info.csv",
    "processed_tri_data/data.npy",
]

print("=" * 80)
print("  BUILDING DEDICATED PHASE 2 GRAPH-IG KAGGLE PACK (< 30MB)")
print("=" * 80)

with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel in INCLUDE_FILES:
        f = ROOT / rel
        if f.exists():
            zf.write(f, rel)
            print(f"  + {rel} ({f.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  ! 警告: 缺少文件 {rel}")

zip_mb = ZIP_OUT.stat().st_size / 1024 / 1024
print("=" * 80)
print(f"[SUCCESS] Kaggle upload pack built: {ZIP_OUT.name} ({zip_mb:.2f} MB)")
print("=" * 80)
