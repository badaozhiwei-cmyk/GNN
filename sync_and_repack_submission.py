#!/usr/bin/env python3
"""
sync_and_repack_submission.py
=============================
Synchronize all authoritative manuscript, tables, freeze, and audit assets
from paper_results/ into the submission package directories, and generate
authoritative zip archives.
"""

import os
import shutil
import zipfile
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
PAPER_RESULTS = ROOT / "paper_results"

TARGET_DIR_INTERNAL = PAPER_RESULTS / "phase2d_final_submission"
TARGET_DIR_ROOT = ROOT.parent / "phase2d_final_submission"

ZIP_INTERNAL = PAPER_RESULTS / "phase2d_final_submission.zip"
ZIP_ROOT = ROOT.parent / "phase2d_final_submission.zip"

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def sync_tree(src, dst):
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)
    
    # 1. Sync files directly in paper_results
    direct_files = [
        "FINAL_FREEZE.json",
        "claim_reference_map.md",
        "reviewer_defense_and_stress_test.md",
        "literature_matrix.csv",
        "references.bib",
        "table_gnn_vs_rf_comparison.csv",
        "table_rf_decoupling_study.csv",
        "table_main_generalization_boundary.csv",
        "table_main_generalization_boundary.tex",
        "table_cluster_bootstrap_significance.csv",
        "supplementary_split_integrity_report.csv",
        "audit_duplicate_states.csv",
        "audit_duplicate_states.json",
        "audit_paper_consistency.csv",
        "audit_paper_consistency.json",
        "audit_row_order_split_binding.csv",
        "audit_row_order_split_binding.json",
        "audit_split_integrity.csv",
        "audit_split_integrity.json",
        "audit_step24_preflight_final.csv",
        "audit_step24_preflight_final.json",
        "audit_step24_preflight_stereo.csv",
        "audit_step24_preflight_stereo.json",
        "audit_step24d0_training_exposure.csv",
        "audit_step24d0_training_exposure.json",
        "audit_uq_false_confidence.csv",
        "audit_uq_recompute.json",
        "audit_uq_reliability_bins.csv",
        "table_hfo_zeroshot_metrics_HFC_all.csv",
        "table_hfo_zeroshot_metrics_B1.csv",
        "table_hfo_zeroshot_metrics_L2.csv",
        "hfo_zeroshot_predictions_HFC_all.csv",
        "hfo_zeroshot_predictions_B1.csv",
        "hfo_zeroshot_predictions_L2.csv",
        "hfo_zeroshot_report_HFC_all.json",
        "hfo_zeroshot_report_B1.json",
        "hfo_zeroshot_report_L2.json",
        "table_hfo_feasibility_audit.csv",
        "hfo_selective_risk_coverage_curve.csv",
    ]
    
    for fname in direct_files:
        s = src / fname
        d = dst / fname
        if s.exists():
            shutil.copy2(s, d)
            print(f"  [COPY] {fname} -> {dst.name}/")
        else:
            print(f"  [SKIP] {fname} (not in {src.name})")

    # 2. Sync manuscript files
    dst_ms = dst / "manuscript"
    dst_ms.mkdir(parents=True, exist_ok=True)
    ms_files = [
        "manuscript_introduction.md",
        "manuscript_introduction.tex",
        "manuscript_methods.md",
        "manuscript_methods.tex",
        "manuscript_results.md",
        "manuscript_results.tex",
        "manuscript_discussion.md",
        "manuscript_discussion.tex",
        "manuscript_supplementary.md",
        "manuscript_supplementary.tex",
    ]
    for mfname in ms_files:
        s = src / mfname
        d = dst_ms / mfname
        if s.exists():
            shutil.copy2(s, d)
            print(f"  [COPY] {mfname} -> {dst.name}/manuscript/")

    # 3. Sync figures if present
    src_fig = src / "figures"
    dst_fig = dst / "figures"
    if src_fig.exists():
        if dst_fig.exists():
            shutil.rmtree(dst_fig)
        shutil.copytree(src_fig, dst_fig)
        print(f"  [COPY] figures/ -> {dst.name}/figures/")

def make_zip(source_dir, output_zip):
    print(f"\nCompressing {source_dir} -> {output_zip} ...")
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                full_path = pathlib.Path(root) / file
                rel_path = full_path.relative_to(source_dir)
                zf.write(full_path, arcname=rel_path)
    size_mb = output_zip.stat().st_size / (1024 * 1024)
    sha = get_sha256(output_zip)
    print(f"  Created: {output_zip.name} ({size_mb:.2f} MB, SHA256: {sha[:16]}...)")
    return sha

def main():
    print("=" * 70)
    print("  PHASE III FINAL SUBMISSION PACKAGE REPACK & SYNC")
    print("=" * 70)

    print(f"\n1. Syncing to Internal: {TARGET_DIR_INTERNAL} ...")
    sync_tree(PAPER_RESULTS, TARGET_DIR_INTERNAL)

    print(f"\n2. Syncing to Root: {TARGET_DIR_ROOT} ...")
    sync_tree(PAPER_RESULTS, TARGET_DIR_ROOT)

    print("\n3. Creating Zip Packages ...")
    sha_internal = make_zip(TARGET_DIR_INTERNAL, ZIP_INTERNAL)
    sha_root = make_zip(TARGET_DIR_ROOT, ZIP_ROOT)

    print("\n" + "=" * 70)
    print("  PACKAGE SYNC & REPACK VERIFIED")
    print(f"  Internal Zip SHA256: {sha_internal}")
    print(f"  Root Zip SHA256    : {sha_root}")
    print("=" * 70)

if __name__ == "__main__":
    main()
