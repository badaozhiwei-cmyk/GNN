from __future__ import annotations

"""Phase II-A: Continuous Chemical Generalization Boundary.

Build species-level chemical distances for:
  1) M1 refrigerant LORO (12 held-out refrigerants)
  2) Split B1/B2 anion-family OOD (9 held-out anions)

Primary distances:
  - D_FP: 1 - nearest-neighbour Morgan Tanimoto similarity
  - D_thermo: nearest-neighbour Euclidean distance after train-species z-scoring
  - D_phys: nearest-neighbour Euclidean distance over RDKit physical proxies

IMPORTANT:
  D_phys is deliberately named D_phys, not D_xTB. It uses RDKit-computable
  proxies (MW, MolLogP, TPSA, HBA, rotatable bonds) and is NOT an xTB distance.

The script does not assume a breakpoint exists. It produces:
  - species-level distance/error table
  - binned summaries
  - Spearman correlations (Distance -> MAE)
  - Inter-distance correlations (D_FP vs D_thermo vs D_phys)
  - optional broken-stick search, only as an exploratory analysis
  - failure-probability summaries using an explicit absolute-error threshold
"""

from pathlib import Path
from typing import Iterable
import argparse
import json
import warnings

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, Descriptors, Lipinski
    RDLogger.DisableLog('rdApp.*')
except Exception as exc:  # pragma: no cover
    raise RuntimeError("RDKit is required for Phase II-A.") from exc

try:
    from scipy.stats import spearmanr, pearsonr
except Exception as exc:  # pragma: no cover
    raise RuntimeError("SciPy is required for Phase II-A.") from exc


SEED_MODES = ["M0", "Mthermo", "Mreduced"]

# Production M1 targets in frozen order.
M1_TARGETS = [
    "R32", "R134A", "R125", "R23", "R41", "R152A",
    "R134", "R161", "R143A", "R245FA", "R236FA", "R227EA"
]

# Explicit family definitions used in Split B.
FAM2 = {"FS", "HFPS", "OTF", "PFBS", "TFES", "TPES", "TTES"}
FAM3 = {"BF4", "PF6"}


def load_nist_critical_from_archived_script(repo: Path) -> dict[str, tuple[float, float, float]]:
    """Safely extract the literal NIST_CRITICAL dictionary from project assets."""
    try:
        import sys
        sys.path.insert(0, str(repo))
        from v6_metadata_utils import NIST_CRITICAL as nc
        return {norm_token(k): tuple(float(x) for x in v) for k, v in nc.items()}
    except Exception:
        pass

    import ast
    import re

    p = repo / "archived_scripts" / "prepare_tri_graph_data_v3.py"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing NIST provenance source: {p}. "
            "Use the exact archived script or supply --nist-json."
        )
    text = p.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"NIST_CRITICAL\s*=\s*(\{.*?\})", text, flags=re.S)
    if not m:
        raise ValueError(f"Could not locate NIST_CRITICAL literal in {p}")
    node = ast.parse("NIST_CRITICAL = " + m.group(1), mode="exec").body[0]
    value = ast.literal_eval(node.value)
    out: dict[str, tuple[float, float, float]] = {}
    for k, v in value.items():
        nk = norm_token(k)
        if len(v) != 3:
            raise ValueError(f"Invalid NIST_CRITICAL entry for {k}: {v}")
        out[nk] = tuple(float(x) for x in v)
    return out


def norm_token(value: object) -> str:
    s = str(value).strip().upper()
    if s in {"NAN", "NONE", "", "NA"}:
        return ""
    return s.replace("[", "").replace("]", "").replace("-", "")


def norm_smiles(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    return mol


def fp(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def tanimoto_distance(m1, m2) -> float:
    return 1.0 - float(DataStructs.TanimotoSimilarity(m1, m2))


def physical_vector(mol: Chem.Mol) -> np.ndarray:
    return np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
        ],
        dtype=float,
    )


def zscore_euclid(target: np.ndarray, refs: np.ndarray, eps: float = 1e-12) -> float:
    mu = np.nanmean(refs, axis=0)
    sd = np.nanstd(refs, axis=0, ddof=0)
    sd = np.where(sd < eps, 1.0, sd)
    z_t = (target - mu) / sd
    z_r = (refs - mu) / sd
    return float(np.min(np.linalg.norm(z_r - z_t, axis=1)))


def phys_distance(target: Chem.Mol, refs: list[Chem.Mol]) -> float:
    t = physical_vector(target)
    r = np.vstack([physical_vector(m) for m in refs])
    return zscore_euclid(t, r)


def fp_distance(target: Chem.Mol, refs: list[Chem.Mol]) -> float:
    t = fp(target)
    return min(tanimoto_distance(t, fp(m)) for m in refs)


def thermo_distance(target_vec: np.ndarray, ref_vecs: np.ndarray) -> float:
    return zscore_euclid(target_vec.astype(float), ref_vecs.astype(float))


def species_smiles_map(index_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Map (species_type, normalized_species) -> SMILES with consistency checks."""
    out: dict[tuple[str, str], str] = {}
    for _, row in index_df.iterrows():
        r = norm_token(row.get("refrigerant", ""))
        a = norm_token(row.get("anion", ""))
        rs = norm_smiles(row.get("refri_smiles", ""))
        a_s = norm_smiles(row.get("anion_smiles", ""))
        if r and rs:
            out.setdefault(("refrigerant", r), rs)
        if a and a_s:
            out.setdefault(("anion", a), a_s)
    return out


def load_index(repo: Path) -> pd.DataFrame:
    p = repo / "index_with_anion.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)
    required = {"refrigerant", "refri_smiles", "anion", "anion_smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"index_with_anion.csv missing columns: {sorted(missing)}")
    return df


def load_m1_summary(repo: Path) -> pd.DataFrame:
    p = repo / "paper_results" / "recovered_overnight_results.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)
    required = {"Mode", "Target", "MAE_mean", "R2_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"M1 summary missing columns: {sorted(missing)}")
    return df


def load_split_audit(repo: Path) -> pd.DataFrame:
    p = repo / "results_split_B" / "audit_cluster_aware_significance.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)
    return df


def load_family_assignment(repo: Path) -> pd.DataFrame:
    p = repo / "splits_anion_ood" / "split_B_family_assignment.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return pd.read_csv(p)


def find_pred_dirs(repo: Path, split: str, mode: str) -> Path:
    d = repo / "results_split_B" / f"{split}_{mode}"
    if not d.exists():
        raise FileNotFoundError(f"Missing {d}")
    return d


def collect_split_species_metrics(repo: Path, split: str, mode: str) -> pd.DataFrame:
    d = find_pred_dirs(repo, split, mode)
    dfs = []
    for seed in [42, 43, 44, 45, 46]:
        p = d / f"pred_seed_{seed}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        x = pd.read_csv(p)
        required = {"sample_id", "true_x1", "pred_x1_raw"}
        missing = required - set(x.columns)
        if missing:
            raise ValueError(f"{p} missing columns: {sorted(missing)}")
        dfs.append(x)
    base = dfs[0].copy()
    for other in dfs[1:]:
        if not base["sample_id"].equals(other["sample_id"]):
            raise AssertionError(f"Seed sample_id order mismatch in {split}_{mode}")
        if not np.allclose(base["true_x1"].to_numpy(), other["true_x1"].to_numpy(), equal_nan=True):
            raise AssertionError(f"true_x1 mismatch in {split}_{mode}")
    preds = np.vstack([x["pred_x1_raw"].to_numpy(float) for x in dfs])
    ens = preds.mean(axis=0)
    err = np.abs(base["true_x1"].to_numpy(float) - ens)

    # In sample_id, format is [cation]__[anion]__refrigerant__T__P
    parsed = base["sample_id"].astype(str).str.split("__")
    base["anion_clean"] = parsed.str[1].map(norm_token)
    base["refrigerant_clean"] = parsed.str[2].map(norm_token)
    base["ensemble_pred_raw"] = ens
    base["abs_error_raw"] = err

    if base["anion_clean"].eq("").any() or base["refrigerant_clean"].eq("").any():
        raise ValueError(f"Could not parse species from sample_id for {split}_{mode}")

    group_cols = ["anion_clean"] if split in {"B1", "B2"} else ["refrigerant_clean"]
    g = base.groupby(group_cols, sort=True)
    out = g.agg(
        MAE=("abs_error_raw", "mean"),
        n_test_points=("abs_error_raw", "size"),
    ).reset_index()
    out["mode"] = mode
    out["split"] = split
    out["species_type"] = "anion" if split in {"B1", "B2"} else "refrigerant"

    # System count inferred as unique cation__anion__refrigerant prefix.
    tmp = base["sample_id"].astype(str).str.split("__").str[:3].str.join("__")
    base["chemical_system"] = tmp
    if split in {"B1", "B2"}:
        sys_counts = base.groupby("anion_clean")["chemical_system"].nunique().rename("n_test_systems")
        out = out.merge(sys_counts, left_on="anion_clean", right_index=True, how="left")
        out = out.rename(columns={"anion_clean": "species"})
    else:
        sys_counts = base.groupby("refrigerant_clean")["chemical_system"].nunique().rename("n_test_systems")
        out = out.merge(sys_counts, left_on="refrigerant_clean", right_index=True, how="left")
        out = out.rename(columns={"refrigerant_clean": "species"})

    return out[["species", "species_type", "split", "mode", "MAE", "n_test_points", "n_test_systems"]]


def infer_m1_species_r2_mae(repo: Path) -> pd.DataFrame:
    """Load frozen M1 summary and normalize target names (stripping 'loro_' prefix)."""
    df = load_m1_summary(repo).copy()
    # Strip prefix 'loro_' or 'LORO_' then normalize
    df["species"] = df["Target"].astype(str).str.replace(r"(?i)^loro_", "", regex=True).map(norm_token)
    df["species_type"] = "refrigerant"
    df["split"] = "M1_LORO"
    df = df.rename(columns={"MAE_mean": "MAE", "R2_mean": "R2"})
    return df[["species", "species_type", "split", "Mode", "MAE", "R2"]].rename(columns={"Mode": "mode"})


def build_distance_table(
    index_df: pd.DataFrame,
    targets: list[str],
    species_type: str,
    ref_by_target: dict[str, list[str]],
    nist_critical: dict[str, tuple[float, float, float]] | None = None
) -> pd.DataFrame:
    smap = species_smiles_map(index_df)
    rows = []
    for target in targets:
        key = (species_type, target)
        if key not in smap:
            raise KeyError(f"Missing SMILES for {species_type}={target}")
        ref_names = ref_by_target[target]
        ref_mols = []
        missing = []
        for r in ref_names:
            k = (species_type, r)
            if k not in smap:
                missing.append(r)
            else:
                ref_mols.append(mol_from_smiles(smap[k]))
        if missing:
            raise KeyError(f"Missing reference SMILES for {species_type}={target}: {missing}")
        target_mol = mol_from_smiles(smap[key])

        d_fp = fp_distance(target_mol, ref_mols)
        d_phys = phys_distance(target_mol, ref_mols)

        if species_type == "refrigerant":
            if nist_critical is None:
                raise ValueError("nist_critical must be provided for refrigerant species")
            try:
                tvec = np.array(nist_critical[target], dtype=float)
                refvecs = np.vstack([np.array(nist_critical[r], dtype=float) for r in ref_names])
            except KeyError as exc:
                raise KeyError(f"Missing NIST constants for M1 target/reference: {exc}") from exc
            d_thermo = thermo_distance(tvec, refvecs)
        else:
            # Anions do not have natural NIST critical constants in this benchmark.
            d_thermo = np.nan

        rows.append(
            {
                "species": target,
                "species_type": species_type,
                "D_FP": d_fp,
                "D_thermo": d_thermo,
                "D_phys": d_phys,
                "n_train_species_reference": len(ref_names),
            }
        )
    return pd.DataFrame(rows)


def build_all_distance_rows(
    index_df: pd.DataFrame,
    family_df: pd.DataFrame,
    nist_critical: dict[str, tuple[float, float, float]]
) -> pd.DataFrame:
    m1_ref = {t: [r for r in M1_TARGETS if r != t] for t in M1_TARGETS}

    # Build all unique anions in the curated universe.
    all_anions = sorted({norm_token(x) for x in family_df["anion_clean"].tolist() if norm_token(x)})
    if len(all_anions) != 23:
        raise AssertionError(f"Expected 23 unique anions from family assignment, got {len(all_anions)}")

    b1_ref = {t: [a for a in all_anions if a not in FAM2] for t in sorted(FAM2)}
    b2_ref = {t: [a for a in all_anions if a not in FAM3] for t in sorted(FAM3)}

    m1 = build_distance_table(index_df, M1_TARGETS, "refrigerant", m1_ref, nist_critical=nist_critical)
    b1 = build_distance_table(index_df, sorted(FAM2), "anion", b1_ref, nist_critical=None)
    b2 = build_distance_table(index_df, sorted(FAM3), "anion", b2_ref, nist_critical=None)
    b1["split"] = "B1"
    b2["split"] = "B2"
    m1["split"] = "M1_LORO"
    return pd.concat([m1, b1, b2], ignore_index=True)


def build_metric_table(repo: Path) -> pd.DataFrame:
    rows = []
    m1 = infer_m1_species_r2_mae(repo)
    rows.append(m1)
    for split in ["B1", "B2"]:
        for mode in SEED_MODES:
            rows.append(collect_split_species_metrics(repo, split, mode))
    return pd.concat(rows, ignore_index=True)


def merge_distance_metrics(distance_df: pd.DataFrame, metric_df: pd.DataFrame) -> pd.DataFrame:
    out = metric_df.merge(
        distance_df,
        on=["species", "species_type", "split"],
        how="inner",
        validate="many_to_one",
    )
    return out


def make_bins(series: pd.Series, q: int = 4) -> pd.Categorical:
    vals = series.dropna()
    if vals.nunique() < 2:
        return pd.Categorical(["all"] * len(series))
    try:
        bins = pd.qcut(series, q=q, duplicates="drop")
    except ValueError:
        bins = pd.cut(series, bins=q, duplicates="drop")
    return bins


def binned_summary(df: pd.DataFrame, distance_col: str, q: int = 4) -> pd.DataFrame:
    x = df.dropna(subset=[distance_col, "MAE"]).copy()
    if x.empty:
        return pd.DataFrame()
    x["distance_bin"] = make_bins(x[distance_col], q=q)
    g = x.groupby(["split", "mode", "distance_bin"], observed=False)
    out = g.agg(
        n_species=("species", "nunique"),
        distance_mean=(distance_col, "mean"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
    ).reset_index()
    out["MAE_se"] = out["MAE_std"] / np.sqrt(out["n_species"].clip(lower=1))
    out["distance_metric"] = distance_col
    return out


def correlation_summary(df: pd.DataFrame, distance_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for (split, mode), g in df.groupby(["split", "mode"], dropna=False):
        for col in distance_cols:
            x = g[[col, "MAE"]].dropna()
            if len(x) < 3 or x[col].nunique() < 2 or x["MAE"].nunique() < 2:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(x[col], x["MAE"])
            rows.append({
                "split": split,
                "mode": mode,
                "distance_metric": col,
                "n_species": len(x),
                "spearman_rho": rho,
                "p_value": p
            })
    return pd.DataFrame(rows)


def distance_inter_correlation(distance_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Pearson and Spearman correlation between distance measures."""
    records = []
    pairs = [("D_FP", "D_thermo"), ("D_FP", "D_phys"), ("D_thermo", "D_phys")]
    for split, g in distance_df.groupby("split"):
        for d1, d2 in pairs:
            sub = g[[d1, d2]].dropna()
            if len(sub) >= 4:
                sp_rho, sp_p = spearmanr(sub[d1], sub[d2])
                pe_r, pe_p = pearsonr(sub[d1], sub[d2])
                records.append({
                    "split": split,
                    "distance_pair": f"{d1}_vs_{d2}",
                    "n_species": len(sub),
                    "spearman_rho": sp_rho,
                    "spearman_p": sp_p,
                    "pearson_r": pe_r,
                    "pearson_p": pe_p,
                })
    return pd.DataFrame(records)


def parse_optional_failure_threshold(value: float | None) -> float:
    if value is None:
        return float("nan")
    x = float(value)
    if not np.isfinite(x) or x <= 0:
        raise ValueError("--failure-threshold must be a positive finite scalar")
    return x


def failure_probability(df: pd.DataFrame, distance_col: str, threshold: float) -> pd.DataFrame:
    x = df.dropna(subset=[distance_col, "MAE"]).copy()
    if x.empty or not np.isfinite(threshold):
        return pd.DataFrame()
    x["failure"] = (x["MAE"] > threshold).astype(int)
    x["distance_bin"] = make_bins(x[distance_col], q=4)
    return (
        x.groupby(["split", "mode", "distance_bin"], observed=False)
        .agg(
            n_species=("species", "nunique"),
            failure_probability=("failure", "mean"),
            distance_mean=(distance_col, "mean"),
            threshold=("failure", lambda s: threshold)
        )
        .reset_index()
        .assign(distance_metric=distance_col)
    )


def broken_stick_search(df: pd.DataFrame, distance_col: str, min_left: int = 3, min_right: int = 3) -> pd.DataFrame:
    """Exploratory continuous broken-stick fit."""
    rows = []
    for (split, mode), g in df.groupby(["split", "mode"], dropna=False):
        x = g[[distance_col, "MAE"]].dropna().sort_values(distance_col)
        if len(x) < (min_left + min_right):
            continue
        xx = x[distance_col].to_numpy(float)
        yy = x["MAE"].to_numpy(float)
        A = np.column_stack([np.ones_like(xx), xx])
        beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
        sse1 = float(np.sum((yy - A @ beta) ** 2))
        best = None
        for i in range(min_left, len(xx) - min_right + 1):
            bp = float((xx[i - 1] + xx[i]) / 2.0)
            h = np.maximum(xx - bp, 0.0)
            X = np.column_stack([np.ones_like(xx), xx, h])
            b, *_ = np.linalg.lstsq(X, yy, rcond=None)
            sse = float(np.sum((yy - X @ b) ** 2))
            if best is None or sse < best[0]:
                best = (sse, bp, b[1], b[1] + b[2])
        if best is None:
            continue
        sse2, bp, slope1, slope2 = best
        rows.append({
            "split": split,
            "mode": mode,
            "distance_metric": distance_col,
            "n_species": len(x),
            "linear_SSE": sse1,
            "broken_stick_SSE": sse2,
            "delta_SSE": sse1 - sse2,
            "candidate_breakpoint": bp,
            "slope_before": slope1,
            "slope_after": slope2,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--failure-threshold", type=float, default=None,
                        help="Optional precomputed absolute-error threshold; if omitted, failure-probability output is empty.")
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = (args.out or (repo / "results_boundary")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Phase II-A: Continuous Chemical Generalization Boundary")
    print(f"Repo: {repo}")
    print(f"Output: {out}")
    print("Rule: no breakpoint is assumed to exist.")
    print("D_phys = RDKit proxy distance; it is NOT an xTB distance.")
    print("=" * 78)

    index_df = load_index(repo)
    family_df = load_family_assignment(repo)
    _ = load_split_audit(repo)  # existence/schema provenance check

    unique_hfc = sorted({norm_token(x) for x in index_df["refrigerant"].tolist() if norm_token(x)})
    unique_anions = sorted({norm_token(x) for x in index_df["anion"].tolist() if norm_token(x)})
    print(f"Index rows: {len(index_df)}")
    print(f"Unique refrigerants in index: {len(unique_hfc)}")
    print(f"Unique anions in index: {len(unique_anions)}")

    nist_critical = load_nist_critical_from_archived_script(repo)
    distance_df = build_all_distance_rows(index_df, family_df, nist_critical)
    metric_df = build_metric_table(repo)
    merged = merge_distance_metrics(distance_df, metric_df)

    # Attach R2 from Split B summary safely (preserving M1 R2 via fillna)
    split_summary_p = repo / "results_split_B" / "split_B_benchmark_summary.csv"
    if split_summary_p.exists():
        ss = pd.read_csv(split_summary_p)
        mode_r2 = ss[["split", "mode", "ensemble_r2_raw"]].copy()
        merged = merged.merge(mode_r2, on=["split", "mode"], how="left")
        if "R2" in merged.columns:
            merged["R2"] = merged["R2"].fillna(merged["ensemble_r2_raw"])
        else:
            merged["R2"] = merged["ensemble_r2_raw"]
        merged = merged.drop(columns=["ensemble_r2_raw"])
    elif "R2" not in merged.columns:
        merged["R2"] = np.nan

    merged["MAE_std_across_seeds"] = np.nan

    # Column order for downstream interface.
    merged = merged[
        [
            "species", "species_type", "split", "mode",
            "D_FP", "D_thermo", "D_phys",
            "MAE", "R2", "MAE_std_across_seeds",
            "n_test_points", "n_test_systems", "n_train_species_reference",
        ]
    ].sort_values(["split", "mode", "species"]).reset_index(drop=True)

    merged.to_csv(out / "continuous_distance_benchmark.csv", index=False)

    cor = correlation_summary(merged, ["D_FP", "D_thermo", "D_phys"])
    cor.to_csv(out / "distance_mae_spearman.csv", index=False)

    # Output inter-distance correlations (D_FP vs D_thermo vs D_phys)
    inter_cor = distance_inter_correlation(distance_df)
    inter_cor.to_csv(out / "distance_inter_correlation.csv", index=False)

    bins = []
    for dc in ["D_FP", "D_thermo", "D_phys"]:
        b = binned_summary(merged, dc, q=4)
        if not b.empty:
            bins.append(b)
    if bins:
        pd.concat(bins, ignore_index=True).to_csv(out / "distance_binned_mae.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "distance_binned_mae.csv", index=False)

    threshold = parse_optional_failure_threshold(args.failure_threshold)
    failure_rows = []
    for dc in ["D_FP", "D_thermo", "D_phys"]:
        f = failure_probability(merged, dc, threshold)
        if not f.empty:
            failure_rows.append(f)
    if failure_rows:
        failure = pd.concat(failure_rows, ignore_index=True)
    else:
        failure = pd.DataFrame()
    failure.to_csv(out / "distance_failure_probability.csv", index=False)

    break_rows = []
    for dc in ["D_FP", "D_thermo", "D_phys"]:
        b = broken_stick_search(merged, dc)
        if not b.empty:
            break_rows.append(b)
    if break_rows:
        pd.concat(break_rows, ignore_index=True).to_csv(out / "exploratory_broken_stick.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "exploratory_broken_stick.csv", index=False)

    manifest = {
        "phase": "Phase II-A",
        "m1_targets": M1_TARGETS,
        "b1_targets": sorted(FAM2),
        "b2_targets": sorted(FAM3),
        "n_unique_anions_expected": 23,
        "distance_definitions": {
            "D_FP": "nearest-neighbour 1 - Morgan Tanimoto, radius=2, nBits=2048",
            "D_thermo": "nearest-neighbour Euclidean distance after train-species z-scoring of (Tc,Pc,omega); applicable to M1 refrigerant axis only",
            "D_phys": "nearest-neighbour Euclidean distance after train-species z-scoring of (MW, MolLogP, TPSA, HBA, rotatable bonds); RDKit proxy, NOT xTB",
        },
        "failure_threshold": threshold,
        "failure_threshold_definition": "Externally supplied absolute-error threshold; do not infer from species-level summary MAE.",
        "notes": [
            "No breakpoint is assumed to exist.",
            "Broken-stick output is exploratory and must not be reported as a definitive D* without evidence.",
            "M1 and Split B are analysed on their own chemical axes; no common universal distance scale is assumed.",
            "D_thermo uses Tc/Pc/omega for refrigerants only. It is N/A for anions unless an explicit anion thermodynamic descriptor source is supplied.",
        ],
    }
    (out / "phaseIIA_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Hard assertions
    expected = set(M1_TARGETS) | FAM2 | FAM3
    got = set(distance_df["species"])
    if got != expected:
        raise AssertionError(f"Distance target mismatch. Missing={sorted(expected-got)}, extra={sorted(got-expected)}")
    if distance_df[["D_FP", "D_phys"]].isna().any().any():
        raise AssertionError("NaN in required D_FP/D_phys distances")
    if not ((distance_df["D_FP"] >= 0).all() and (distance_df["D_FP"] <= 1).all()):
        raise AssertionError("D_FP outside [0,1]")

    print("\n[PASS] Distance table written:", out / "continuous_distance_benchmark.csv")
    print("[PASS] Correlations written:", out / "distance_mae_spearman.csv")
    print("[PASS] Inter-distance correlations written:", out / "distance_inter_correlation.csv")
    print("[PASS] Binned analysis written:", out / "distance_binned_mae.csv")
    print("[PASS] Failure analysis written:", out / "distance_failure_probability.csv")
    print("[PASS] Exploratory broken-stick written:", out / "exploratory_broken_stick.csv")
    print("[PASS] Manifest written:", out / "phaseIIA_manifest.json")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
