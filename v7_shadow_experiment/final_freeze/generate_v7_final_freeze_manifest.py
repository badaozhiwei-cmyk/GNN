#!/usr/bin/env python3
"""Freeze the 15 official V6/V7 checkpoints with SHA256 provenance.

This version deliberately avoids heuristic model detection. The three checkpoint
roots are supplied explicitly, and each root must contain exactly:
    best_seed_42.pth ... best_seed_46.pth

Example (Windows PowerShell):
    uv run python generate_v7_final_freeze_manifest_fixed.py `
      --v6-dir "results_hfc_all/HFC_all_M0" `
      --v7a-dir "D:/折腾/v7_pilot_gpu_results/v7_shadow_experiment/checkpoints/V7-A" `
      --v7b-dir "D:/折腾/v7_pilot_gpu_results2/v7_shadow_experiment/checkpoints/V7-B" `
      --out-dir "v7_shadow_experiment/final_freeze"

Exit code 0 is emitted ONLY when all 15 expected files exist and are unique.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

SEEDS = (42, 43, 44, 45, 46)
MODELS = ("V6-M0", "V7-A", "V7-B")
EXPECTED = [(m, s) for m in MODELS for s in SEEDS]


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_info(repo: Path) -> tuple[str | None, str | None]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.STDOUT
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=repo, text=True, stderr=subprocess.STDOUT
        ).strip()
        return head, status
    except Exception:
        return None, None


def resolve(repo: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (repo / p)


def collect(model: str, root: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    if not root.exists():
        return [], [f"{model}: directory not found: {root}"]
    if not root.is_dir():
        return [], [f"{model}: not a directory: {root}"]

    for seed in SEEDS:
        expected = root / f"best_seed_{seed}.pth"
        if not expected.exists():
            errors.append(f"{model}-Seed{seed}: missing {expected}")
            continue
        if not expected.is_file():
            errors.append(f"{model}-Seed{seed}: not a file: {expected}")
            continue
        rows.append({
            "model": model,
            "seed": seed,
            "checkpoint_path": str(expected.resolve()),
            "file_size_bytes": expected.stat().st_size,
            "sha256": sha256(expected),
        })

    return rows, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6-dir", required=True)
    ap.add_argument("--v7a-dir", required=True)
    ap.add_argument("--v7b-dir", required=True)
    ap.add_argument("--out-dir", default="v7_shadow_experiment/final_freeze")
    args = ap.parse_args()

    repo = Path.cwd().resolve()
    out_dir = resolve(repo, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = {
        "V6-M0": resolve(repo, args.v6_dir),
        "V7-A": resolve(repo, args.v7a_dir),
        "V7-B": resolve(repo, args.v7b_dir),
    }

    rows: list[dict] = []
    errors: list[str] = []
    for model, root in roots.items():
        r, e = collect(model, root)
        rows.extend(r)
        errors.extend(e)

    seen = {(r["model"], r["seed"]) for r in rows}
    missing_runs = [f"{m}-Seed{s}" for m, s in EXPECTED if (m, s) not in seen]

    # There should be exactly one explicitly addressed file for each run.
    duplicates = []
    counts = {}
    for r in rows:
        key = (r["model"], r["seed"])
        counts[key] = counts.get(key, 0) + 1
    for key, n in counts.items():
        if n != 1:
            duplicates.append(f"{key[0]}-Seed{key[1]}: {n} files")

    git_head, git_status = git_info(repo)
    status = "PASS" if not errors and not missing_runs and not duplicates and len(rows) == 15 else "INCOMPLETE"

    csv_path = out_dir / "checkpoint_hash_manifest.csv"
    fieldnames = ["model", "seed", "checkpoint_path", "file_size_bytes", "sha256"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (MODELS.index(r["model"]), r["seed"])))

    summary = {
        "status": status,
        "expected_runs": 15,
        "identified_checkpoints": len(rows),
        "missing_runs": missing_runs,
        "duplicates": duplicates,
        "errors": errors,
        "git_head": git_head,
        "git_worktree_status": git_status,
        "official_seed_set": list(SEEDS),
        "official_models": list(MODELS),
        "notes": [
            "All five seeds are official primary runs; Seed45 is never excluded from primary reporting.",
            "V7-B four-seed exclusion of Seed45 is sensitivity analysis only.",
            "SHA256 values are computed from the actual checkpoint bytes on the local machine.",
            "A PASS requires all 15 exact best_seed_{42..46}.pth files to exist at the explicit roots.",
        ],
    }
    json_path = out_dir / "checkpoint_hash_manifest.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
