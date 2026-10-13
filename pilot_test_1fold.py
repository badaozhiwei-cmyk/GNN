# pilot_test_1fold.py — 1-Fold Fast Pilot on Kaggle GPU (e.g. R125)
import subprocess, sys
from pathlib import Path

print("=" * 80)
print("  STEP 1: RUNNING 1-FOLD PILOT ON R125 (V7-B, SEED 42)")
print("=" * 80)

cmd = [
    sys.executable, str(Path(__file__).resolve().parent / "run_v7_loro.py"),
    "--target-ref", "R125",
    "--model", "V7-B",
    "--seeds", "42",
    "--epoch", "100",
    "--patience", "15",
    "--device", "cuda"
]
subprocess.run(cmd, check=True)
print("\n>>> PILOT FINISHED! Check results_v7_loro/ for metrics and predictions!")
