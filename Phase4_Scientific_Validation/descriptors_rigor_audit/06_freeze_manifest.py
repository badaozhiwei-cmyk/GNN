import os
import sys
import hashlib
import json
import pkg_resources
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT_DIR / "Phase4_Scientific_Validation"
AUDIT_OUT = Path(__file__).resolve().parent / "audit_outputs"
AUDIT_OUT.mkdir(parents=True, exist_ok=True)

INPUT_CSV = ROOT_DIR / "index_with_anion.csv"
DESC_CSV = WORK_DIR / "xTB_Physics_Descriptors.csv"
PROV_CSV = AUDIT_OUT / "computation_provenance.csv"

def sha256(path):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def tree_sha256(directory):
    if not directory.is_dir():
        return None
    h = hashlib.sha256()
    for path in sorted(p for p in directory.rglob('*') if p.is_file()):
        h.update(str(path.relative_to(directory)).encode('utf-8'))
        h.update(path.read_bytes())
    return h.hexdigest()

def get_pkg_version(pkg_name):
    try:
        return pkg_resources.get_distribution(pkg_name).version
    except:
        return "Unknown"

def main():
    print("Generating descriptor manifest freeze...")
    
    # 1. Hashes
    input_hash = sha256(INPUT_CSV)
    desc_hash = sha256(DESC_CSV)
    prov_hash = sha256(PROV_CSV)
    script_hashes = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    logs_hash = tree_sha256(WORK_DIR / 'xtb_logs')
    
    # 2. Extract Failed Molecules from Provenance
    failed = []
    n_success = 0
    n_failed = 0
    if PROV_CSV.is_file():
        prov = pd.read_csv(PROV_CSV)
        n_success = int((prov['calculation_status'] == 'success').sum())
        n_failed = int((prov['calculation_status'] == 'failed').sum())
        
        failed_df = prov[prov['calculation_status'] == 'failed']
        for _, r in failed_df.iterrows():
            failed.append({
                "Molecule": r["Molecule"],
                "Reason": str(r.get("failure_reason", "unknown"))
            })
            
    # 3. Environment
    import rdkit
    import sklearn
    env = {
        "Python": sys.version.split()[0],
        "RDKit": rdkit.__version__,
        "Scikit-Learn": sklearn.__version__,
        "Pandas": pd.__version__
    }
    
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "description": "Final rigid geometry xTB descriptor freeze for Mphys evaluation.",
        "input_inventory": {
            "file": "index_with_anion.csv",
            "sha256": input_hash
        },
        "descriptor_csv": {
            "file": "xTB_Physics_Descriptors.csv",
            "sha256": desc_hash
        },
        "provenance_audit": {
            "file": "computation_provenance.csv",
            "sha256": prov_hash
        },
        "forensic_inputs": {"xtb_logs_tree_sha256": logs_hash, "audit_scripts_sha256": script_hashes},
        "statistics": {
            "n_success": n_success,
            "n_failed": n_failed,
            "failed_molecules": failed
        },
        "methodology": {
            "xtb_version": "6.6.1",
            "optimization": "GFN2-xTB --opt tight",
            "single_point": "GFN2-xTB --alpha",
            "conformer": "ETKDGv3 seeds=0..19; MMFF94 pre-opt; Boltzmann weighting at 298.15 K",
            "volume_method": "RDKit ComputeMolVolume (gridSpacing=0.2, boxMargin=2.0) on xTB geometry mapped onto SMILES topology",
            "loocv": "train-fold-only PCA/scaling; RidgeCV alphas=1e-3..1e3; RF n_estimators=100, seed=42",
            "bootstrap": "1000 resamples, seed=42"
        },
        "environment": env
    }
    
    manifest_path = AUDIT_OUT / "descriptor_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    main()
