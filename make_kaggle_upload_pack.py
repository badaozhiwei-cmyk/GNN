"""
make_kaggle_upload_pack.py — Build lightweight Kaggle upload zip
================================================================
打包所有运行 run_kaggle_phase2d.py 所必需的文件，体积控制在 20MB 以内。
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_OUT = ROOT / "kaggle_upload_pack.zip"

INCLUDE_PATTERNS = [
    "run_kaggle_phase2d.py",
    "index_with_anion.csv",
    "GNN_for_property_prediction/Model_v6.py",
    "GNN_for_property_prediction/Dataset_v6.py",
    "research_pipeline/run_split_L2_benchmark.py",
    "research_pipeline/run_hfc_all_production.py",
    "research_pipeline/step16_hfo_zeroshot_probe.py",
    "processed_tri_data_hfc2739/data.npy",
    "processed_tri_data_hfc2739/label.npy",
    "processed_tri_data_hfc2739/meta_info.csv",
    "processed_tri_data/data.npy",
    "processed_tri_data/label.npy",
    "processed_tri_data/meta_info.csv",
    "splits/L2_controlled_composite.npz",
    "splits/HFC_all_split.npz",
    "splits/L2_star_anchor.npz",
    "splits/L2_pair_assignment.csv",
    "results_split_B/B1_M0/best_seed_*.pth",
    "results_split_B/B1_M0/scalers.pkl",
    "results_split_B/B1_Mreduced/best_seed_*.pth",
    "results_split_B/B1_Mreduced/scalers.pkl",
]

print("=" * 70)
print("  BUILDING KAGGLE UPLOAD PACK")
print("=" * 70)

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
print("=" * 70)
print(f"[SUCCESS] Kaggle upload pack built: {ZIP_OUT.name} ({zip_mb:.2f} MB)")
print("=" * 70)
