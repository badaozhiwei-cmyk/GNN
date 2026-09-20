"""
make_kaggle_step25_pack.py — Build Dedicated Step 25 Attribution Kaggle Upload Pack
===================================================================================
打包所有在 Kaggle GPU 上运行 Step 25 归因评估所必需的文件：
1. 一键运行器 run_kaggle_step25_attribution.py
2. 核心归因引擎 research_pipeline/step25_graph_substructure_attribution.py
3. 样本清单 results_attribution/case_selection_manifest.csv
4. 模型与数据定义 (Model_v6, Dataset_v6, prepare_tri_graph_data_v6)
5. 底层数据文件 (processed_tri_data, processed_tri_data_hfc2739, index_with_anion.csv)
6. 10 个目标检查点权重与 scalers (HFC_all_M0 与 B2_M0 各种子)
7. 冻结主线绝缘文件 (FINAL_FREEZE.json, table_main_generalization_boundary.csv)
"""

from __future__ import annotations
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_OUT = ROOT / "kaggle_step25_attribution_pack.zip"

INCLUDE_PATTERNS = [
    "run_kaggle_step25_attribution.py",
    "research_pipeline/step25_graph_substructure_attribution.py",
    "results_attribution/case_selection_manifest.csv",
    "GNN_for_property_prediction/Model_v6.py",
    "GNN_for_property_prediction/Dataset_v6.py",
    "prepare_tri_graph_data_v6.py",
    "index_with_anion.csv",
    "processed_tri_data_hfc2739/data.npy",
    "processed_tri_data_hfc2739/meta_info.csv",
    "processed_tri_data/data.npy",
    "results_hfc_all/HFC_all_M0/best_seed_*.pth",
    "results_hfc_all/HFC_all_M0/scalers.pkl",
    "results_split_B/B2_M0/best_seed_*.pth",
    "results_split_B/B2_M0/scalers.pkl",
    "paper_results/FINAL_FREEZE.json",
    "paper_results/table_main_generalization_boundary.csv",
]

print("=" * 80)
print("  BUILDING DEDICATED STEP 25 ATTRIBUTION KAGGLE UPLOAD PACK")
print("=" * 80)

files_to_pack = set()

for pat in INCLUDE_PATTERNS:
    if "*" in pat:
        parts = pat.split("/")
        base_dir = ROOT / "/".join(parts[:-1])
        glob_pat = parts[-1]
        if base_dir.exists():
            for f in base_dir.glob(glob_pat):
                if f.is_file():
                    files_to_pack.add(f)
    else:
        f = ROOT / pat
        if f.is_file():
            files_to_pack.add(f)
        elif f.is_dir():
            for sub in f.rglob("*"):
                if sub.is_file():
                    files_to_pack.add(sub)

with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(list(files_to_pack)):
        rel = f.relative_to(ROOT)
        zf.write(f, rel)
        print(f"  + {rel} ({f.stat().st_size / 1024:.1f} KB)")

zip_mb = ZIP_OUT.stat().st_size / 1024 / 1024
print("=" * 80)
print(f"[SUCCESS] Kaggle upload pack built: {ZIP_OUT.name} ({zip_mb:.2f} MB)")
print("=" * 80)
