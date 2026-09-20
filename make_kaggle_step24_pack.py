import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_OUT = ROOT / "kaggle_step24_stereo_pack.zip"

INCLUDE_FILES = [
    "research_pipeline/step24_stereo_ablation.py",
    "research_pipeline/step24_stereo_preflight_final.py",
    "GNN_for_property_prediction/Model_v6.py",
    "GNN_for_property_prediction/Dataset_v6.py",
    "prepare_tri_graph_data_v6.py",
    "index_with_anion.csv",
    "processed_tri_data_hfc2739/data.npy",
    "processed_tri_data_hfc2739/label.npy",
    "processed_tri_data_hfc2739/meta_info.csv",
    "splits/HFC_all_split.npz",
    "paper_results/hfo_zeroshot_predictions_HFC_all.csv",
]

print("=" * 70)
print("  BUILDING DEDICATED STEP 24 STEREO KAGGLE PACK (< 5MB)")
print("=" * 70)

with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel_path in INCLUDE_FILES:
        f = ROOT / rel_path
        if f.exists():
            zf.write(f, rel_path)
            print(f"  + {rel_path} ({f.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  ! WARNING: Missing {rel_path}")

zip_mb = ZIP_OUT.stat().st_size / 1024 / 1024
print("=" * 70)
print(f"[SUCCESS] Dedicated Step 24 pack built: {ZIP_OUT.name} ({zip_mb:.2f} MB)")
print("=" * 70)
