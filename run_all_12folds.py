# run_all_12folds.py — Full 12-Fold LORO Master Runner on Kaggle GPU
import subprocess, sys
from pathlib import Path

print("=" * 80)
print("  STEP 2: RUNNING ALL 12 FOLDS LORO ON V7-B (5 SEEDS: 42,43,44,45,46)")
print("=" * 80)

cmd = [
    sys.executable, str(Path(__file__).resolve().parent / "run_v7_loro.py"),
    "--model", "V7-B",
    "--seeds", "42,43,44,45,46",
    "--epoch", "100",
    "--patience", "15",
    "--device", "cuda",
    "--output-dir", "results_v7_loro_full"
]
subprocess.run(cmd, check=True)
print("\n>>> ALL 12 FOLDS FINISHED! Summary saved to results_v7_loro_full/loro_metrics_summary.csv")
