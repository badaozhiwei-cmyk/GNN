"""Forensic provenance audit for the unified xTB descriptor table.

The input molecule inventory is the authoritative 66-row table.  Calculation
results and logs are left-joined onto it, so failed or missing calculations
cannot disappear from the audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT.parent / "index_with_anion.csv"
DESCRIPTORS = ROOT / "xTB_Physics_Descriptors.csv"
LOG_DIR = ROOT / "xtb_logs"
OUT_DIR = Path(__file__).resolve().parent / "audit_outputs"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", str(name))


def section(text: str, start: str, end: str | None = None) -> str:
    pattern = re.escape(start) + r"(.*?)(?=" + (re.escape(end) if end else r"\Z") + r")"
    m = re.search(pattern, text, flags=re.I | re.S)
    return m.group(1) if m else ""


def parse_returncode(text: str) -> tuple[int | None, str]:
    # The current pipeline does not persist subprocess.returncode in logs.
    patterns = [r"(?:return\s*code|returncode|exit\s*code)\s*[:=]\s*(-?\d+)"]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return int(m.group(1)), "logged_metadata"
    return None, "unavailable_in_log"


def audit_log(path: Path) -> dict:
    if not path.is_file():
        return {"log_present": False, "log_sha256": None,
                "opt_returncode_source": "unavailable_in_log",
                "sp_returncode_source": "unavailable_in_log",
                "opt_converged": False, "sp_terminated_normally": False,
                "geometry_source": "FAILED", "failure_reason": "log_missing"}
    text = path.read_text(encoding="utf-8", errors="replace")
    opt = section(text, "=== OPT STDOUT ===", "=== OPT STDERR ===")
    opt_err = section(text, "=== OPT STDERR ===", "=== SP+ALPHA STDOUT ===")
    sp = section(text, "=== SP+ALPHA STDOUT ===", "=== SP+ALPHA STDERR ===")
    sp_err = section(text, "=== SP+ALPHA STDERR ===")
    opt_all, sp_all = opt + "\n" + opt_err, sp + "\n" + sp_err
    opt_rc, opt_rc_source = parse_returncode(opt_all)
    sp_rc, sp_rc_source = parse_returncode(sp_all)
    opt_normal = bool(re.search(r"normal termination of xtb", opt_all, re.I))
    sp_normal = bool(re.search(r"normal termination of xtb", sp_all, re.I))
    opt_xyz = path.parent.parent / "xtb_work" / path.stem / "xtbopt.xyz"
    optimized = opt_xyz.is_file() or bool(re.search(r"optimized geometry written to", opt_all, re.I))
    reasons = []
    low = text.lower()
    if "scf" in low and ("oscillat" in low or "not converg" in low):
        reasons.append("SCF oscillation/non-convergence")
    if "segmentation fault" in low:
        reasons.append("segmentation fault")
    if "failed" in low or "abnormal termination" in low:
        reasons.append("abnormal termination")
    if not reasons and not (opt_normal and sp_normal and optimized):
        reasons.append("incomplete termination evidence")
    return {"log_present": True, "log_sha256": sha256(path),
            "opt_returncode": opt_rc, "opt_returncode_source": opt_rc_source,
            "sp_returncode": sp_rc, "sp_returncode_source": sp_rc_source,
            "opt_normal_termination": opt_normal,
            "sp_normal_termination": sp_normal,
            "opt_geometry_present": optimized,
            "opt_converged": bool(opt_normal and optimized and opt_rc == 0),
            "sp_terminated_normally": bool(sp_normal and sp_rc == 0),
            "geometry_source": "GFN2-xTB tight opt" if optimized else "FAILED",
            "failure_reason": "; ".join(reasons) if reasons else "none"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=INPUT)
    ap.add_argument("--descriptors", type=Path, default=DESCRIPTORS)
    ap.add_argument("--logs", type=Path, default=LOG_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(args.input).drop_duplicates(["refrigerant", "refri_smiles", "cation", "cation_smiles", "anion", "anion_smiles"])
    rows = []
    for category, name_col, smiles_col, charge in [
        ("Refrigerant", "refrigerant", "refri_smiles", 0),
        ("Cation", "cation", "cation_smiles", 1),
        ("Anion", "anion", "anion_smiles", -1),
    ]:
        u = inv[[name_col, smiles_col]].drop_duplicates()
        u.columns = ["Molecule", "SMILES"]
        u["Category"], u["Charge"] = category, charge
        rows.append(u)
    master = pd.concat(rows, ignore_index=True).drop_duplicates(["Category", "Molecule", "SMILES"])
    desc = pd.read_csv(args.descriptors) if args.descriptors.is_file() else pd.DataFrame()
    if not desc.empty:
        desc = desc.drop_duplicates(["Category", "Molecule"], keep="last")
        merged = master.merge(desc, on=["Category", "Molecule"], how="left", suffixes=("_inventory", ""))
    else:
        merged = master.copy()
    audit = []
    for _, r in master.iterrows():
        log = args.logs / f"{safe_filename(r.Molecule)}.log"
        d = audit_log(log)
        d.update({"Category": r.Category, "Molecule": r.Molecule,
                  "SMILES": r.SMILES, "Charge": r.Charge})
        audit.append(d)
    audit_df = pd.DataFrame(audit)
    out = merged.merge(audit_df, on=["Category", "Molecule", "SMILES", "Charge"], how="left")
    numeric = ["Dipole_Debye", "Polarizability_au", "Volume_A3"]
    out["parser_complete"] = out[[c for c in numeric if c in out]].notna().all(axis=1) if numeric[0] in out else False
    out["calculation_status"] = out.apply(lambda x: "success" if x.parser_complete and x.log_present and x.opt_converged and x.sp_terminated_normally else ("failed" if x.failure_reason != "log_missing" else "unverified"), axis=1)
    out.to_csv(args.output_dir / "computation_provenance.csv", index=False)
    summary = {"n_inventory": int(len(master)), "n_success": int((out.calculation_status == "success").sum()),
               "n_failed": int((out.calculation_status == "failed").sum()), "n_unverified": int((out.calculation_status == "unverified").sum()),
               "failed_molecules": out.loc[out.calculation_status == "failed", "Molecule"].tolist()}
    (args.output_dir / "provenance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
