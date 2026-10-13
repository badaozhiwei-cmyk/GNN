#!/usr/bin/env python3
"""Independent HFO provenance/matching audit using exact T_K/P_MPa values.

Example:
  python match_hfo_exact.py \
    --source source/index_with_anion.csv \
    --predictions predictions/canonical_v7_ood_predictions_all.csv \
    --summary summaries/hfo_species_breakdown.csv \
    --manifest predictions/canonical_v7_ood_manifest.json \
    --mapping-out hfo_sample_level_mapping_corrected.csv

Requires pandas and numpy. The script intentionally does NOT round temperature or
pressure before matching. It treats source x1 vs prediction y_true float32 storage
precision explicitly instead of claiming literal binary equality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

HFO_RE = r"1234|1233|1336"
KEYS = ["_cat_key", "_anion_key", "_ref_key", "T_K", "P_MPa"]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def norm_chem(x: object) -> str:
    # Normalize brackets and case only; do not round or coarsen chemical/state keys.
    return str(x).strip().replace("[", "").replace("]", "").upper()

def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))

def r2(y: np.ndarray, p: np.ndarray) -> float:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / denom) if denom > 0 else float("nan")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path, help="index_with_anion.csv")
    ap.add_argument("--predictions", required=True, type=Path, help="canonical_v7_ood_predictions_all.csv")
    ap.add_argument("--summary", type=Path, help="hfo_species_breakdown.csv")
    ap.add_argument("--manifest", type=Path, help="canonical_v7_ood_manifest.json")
    ap.add_argument("--mapping-out", type=Path, default=Path("hfo_sample_level_mapping_corrected.csv"))
    args = ap.parse_args()

    for p in [args.source, args.predictions, *([args.summary] if args.summary else []), *([args.manifest] if args.manifest else [])]:
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")

    source_all = pd.read_csv(args.source)
    pred_all = pd.read_csv(args.predictions)
    required_source = {"cation", "anion", "refrigerant", "T_K", "P_MPa", "x1"}
    required_pred = {"sample_id", "domain", "model", "seed", "y_true", "y_pred_clip", "cation", "anion", "refrigerant", "T_K", "P_MPa", "ckpt_sha256", "split_sha256", "scaler_sha256", "git_commit"}
    if missing := required_source - set(source_all.columns):
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    if missing := required_pred - set(pred_all.columns):
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    # Preserve original CSV row position. The sample_id suffix should point to it.
    source_all = source_all.reset_index(names="source_index_row")
    hfo_src = source_all[source_all["refrigerant"].astype(str).str.contains(HFO_RE, case=False, regex=True, na=False)].copy()
    hfo_pred = pred_all[pred_all["domain"].astype(str).str.upper().eq("HFO")].copy()
    unique_pred = hfo_pred.drop_duplicates(subset=["sample_id"]).copy()

    # Every sample must have exactly one row per model/seed pair and invariant metadata.
    records_per_sample = hfo_pred.groupby("sample_id").size()
    if records_per_sample.empty or not records_per_sample.eq(10).all():
        raise AssertionError(f"Expected 10 model/seed prediction rows per HFO sample; count distribution: {records_per_sample.value_counts().to_dict()}")
    if hfo_pred[["sample_id", "model", "seed"]].duplicated().any():
        raise AssertionError("Duplicate (sample_id, model, seed) records found")
    for col in ["y_true", "cation", "anion", "refrigerant", "T_K", "P_MPa"]:
        if (hfo_pred.groupby("sample_id")[col].nunique(dropna=False) > 1).any():
            raise AssertionError(f"Field {col} varies within a sample_id")

    # Exact 5-tuple join. Crucially, do not round pressure: 0.3263 and 0.3265 are distinct.
    for df in (hfo_src, unique_pred):
        df["_cat_key"] = df["cation"].map(norm_chem)
        df["_anion_key"] = df["anion"].map(norm_chem)
        df["_ref_key"] = df["refrigerant"].map(norm_chem)

    source_keys = hfo_src[KEYS]
    pred_keys = unique_pred[KEYS]
    if source_keys.duplicated().any():
        raise AssertionError("Source HFO exact 5-tuple is not unique")
    if pred_keys.duplicated().any():
        raise AssertionError("Prediction HFO exact 5-tuple is not unique")

    merged = hfo_src.merge(
        unique_pred,
        on=KEYS,
        how="outer",
        indicator=True,
        suffixes=("_source", "_pred"),
        validate="one_to_one",
    )
    merge_counts = merged["_merge"].value_counts().to_dict()
    if merge_counts.get("both", 0) != len(hfo_src) or merge_counts.get("left_only", 0) != 0 or merge_counts.get("right_only", 0) != 0 or len(hfo_src) != len(unique_pred):
        raise AssertionError(f"HFO records do not match one-to-one: {merge_counts}; source={len(hfo_src)}, predictions={len(unique_pred)}")

    # Validate the canonical sample ID's trailing source-row index and full metadata.
    suffix = pd.to_numeric(merged["sample_id"].astype(str).str.rsplit("__", n=1).str[-1], errors="coerce")
    if suffix.isna().any() or not np.array_equal(suffix.to_numpy(dtype=np.int64), merged["source_index_row"].to_numpy(dtype=np.int64)):
        raise AssertionError("sample_id trailing source-row indices do not exactly match source CSV row positions")
    for stem in ["cation", "anion", "refrigerant"]:
        s = merged[f"{stem}_source"].map(norm_chem)
        p = merged[f"{stem}_pred"].map(norm_chem)
        if not s.eq(p).all():
            raise AssertionError(f"Chemical metadata mismatch in {stem}")
    # T_K and P_MPa are exact merge keys; every joined row therefore has exact state equality.

    label_diff = (merged["x1"] - merged["y_true"]).abs()
    x1_float32 = merged["x1"].astype(np.float32).astype(np.float64)
    float32_diff = (x1_float32 - merged["y_true"]).abs()
    if not float32_diff.max() <= 1e-7:
        raise AssertionError(f"Source labels do not agree with float32 representation: max diff={float32_diff.max()}")

    out = pd.DataFrame({
        "source_index_row": merged["source_index_row"],
        "sample_id": merged["sample_id"],
        "cation": merged["cation_source"],
        "anion": merged["anion_source"],
        "refrigerant_source": merged["refrigerant_source"],
        "refrigerant_prediction": merged["refrigerant_pred"],
        "T_K": merged["T_K"],
        "P_MPa": merged["P_MPa"],
        "x1_source": merged["x1"],
        "y_true_prediction": merged["y_true"],
        "abs_diff_float64": label_diff,
        "abs_diff_after_source_float32_cast": float32_diff,
        "prediction_rows_model_seed": merged["sample_id"].map(records_per_sample),
        "match_status": "EXACT_UNROUNDED_5TUPLE_MATCH",
    }).sort_values("source_index_row")
    args.mapping_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.mapping_out, index=False, float_format="%.12g")

    species_counts = hfo_src["refrigerant"].value_counts().sort_index()
    summary_checks = []
    if args.summary:
        summary = pd.read_csv(args.summary)
        # Recreate summary statistics by averaging the five clipped predictions per sample.
        for _, sr in summary.iterrows():
            species_key = str(sr["Species"]).upper()
            sp = hfo_pred[hfo_pred["refrigerant"].astype(str).str.upper().eq(species_key)]
            if sp.empty:
                raise AssertionError(f"Summary species not found in predictions: {sr['Species']}")
            values = []
            for model in ["V7-A", "V7-B"]:
                sm = sp[sp["model"].eq(model)].groupby("sample_id").agg(y_true=("y_true", "first"), y_pred=("y_pred_clip", "mean"), sigma=("y_pred_clip", "std"))
                mae_calc, r2_calc, sigma_calc = mae(sm.y_true.to_numpy(), sm.y_pred.to_numpy()), r2(sm.y_true.to_numpy(), sm.y_pred.to_numpy()), float(sm.sigma.mean())
                mae_ref = float(sr["V7A_MAE"] if model == "V7-A" else sr["V7B_MAE"])
                r2_ref = float(sr["V7A_R2"] if model == "V7-A" else sr["V7B_R2"])
                sigma_ref = float(sr["V7A_Sigma"] if model == "V7-A" else sr["V7B_Sigma"])
                values.append((abs(mae_calc - mae_ref), abs(r2_calc - r2_ref), abs(sigma_calc - sigma_ref)))
            summary_checks.append((sr["Species"], int(sr["N"]), max(v[0] for v in values), max(v[1] for v in values), max(v[2] for v in values)))
        if int(summary["N"].sum()) != len(hfo_src):
            raise AssertionError("Species summary N does not sum to source HFO count")
        if any(max(v[2:]) > 2e-6 for v in summary_checks):
            # Values in the provided table are stored to finite precision; tolerate rounding.
            raise AssertionError(f"Species summary metrics do not reproduce within tolerance: {summary_checks}")

    print("HFO source rows:", len(hfo_src))
    print("HFO prediction rows:", len(hfo_pred))
    print("Unique HFO sample IDs:", unique_pred["sample_id"].nunique())
    print("Rows per HFO sample ID:", sorted(records_per_sample.unique().tolist()))
    print("Exact unrounded 5-tuple merge:", merge_counts)
    print("sample_id suffix vs source CSV row index: all matched")
    print("max |source x1 - prediction y_true|:", f"{label_diff.max():.12g}")
    print("max |float32(source x1) - prediction y_true|:", f"{float32_diff.max():.12g}")
    print("Corrected mapping rows/unique IDs:", len(out), out["sample_id"].nunique())
    print("Source HFO species counts:", species_counts.to_dict())
    print("Source SHA-256:", sha256(args.source))
    print("Predictions SHA-256:", sha256(args.predictions))
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        meta = manifest.get("metadata", {})
        if meta.get("training_split_sha256") and not pred_all["split_sha256"].eq(meta["training_split_sha256"]).all():
            raise AssertionError("Prediction split SHA does not match manifest")
        if meta.get("scaler_sha256") and not pred_all["scaler_sha256"].eq(meta["scaler_sha256"]).all():
            raise AssertionError("Prediction scaler SHA does not match manifest")
        if meta.get("git_commit") and not pred_all["git_commit"].eq(meta["git_commit"]).all():
            raise AssertionError("Prediction git commit does not match manifest")
        for (model, seed), group in pred_all.groupby(["model", "seed"]):
            expected = meta.get("checkpoint_sha256", {}).get(f"{model}_seed{seed}")
            got = group["ckpt_sha256"].unique()
            if expected is None or len(got) != 1 or got[0] != expected:
                raise AssertionError(f"Checkpoint SHA mismatch for {model}, seed {seed}")
        domain_counts = {k: int(v["sample_count"]) for k, v in manifest.get("domains", {}).items()}
        got_domain_counts = {str(k): int(v["sample_id"].nunique()) for k, v in pred_all.groupby("domain")}
        if domain_counts != got_domain_counts:
            raise AssertionError(f"Manifest domain counts differ: manifest={domain_counts}, predictions={got_domain_counts}")
        print("Manifest checkpoint/split/scaler/commit/domain metadata: internally consistent")
    if summary_checks:
        print("Species summary metric checks (species, N, max abs diff MAE/R2/sigma):")
        for row in summary_checks:
            print("  ", row)
    print("Corrected mapping written to:", args.mapping_out.resolve())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
