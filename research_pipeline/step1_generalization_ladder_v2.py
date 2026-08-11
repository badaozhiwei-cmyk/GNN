"""Generate leakage-audited L0-L4 chemical generalization splits.

Protocol
--------
L0: random condition-point interpolation. Exact duplicate measurements remain
    in the same partition, while chemical systems may occur in all partitions.
L1: high-temperature/pressure extrapolation within systems seen in training.
L2: unseen cation-anion recombinations; both ions remain individually seen.
L3: refrigerant-family shift from saturated fluorocarbons to C=C refrigerants.
L4: leave-one-refrigerant-out (LORO), emitted as one split per refrigerant.

All indices refer directly to index_with_anion.csv / processed_tri_data arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parent.parent
SEED = 42
SYSTEM_COLS = ["cation", "anion", "refrigerant"]
POINT_COLS = SYSTEM_COLS + ["T_K", "P_MPa"]


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _split_groups(
    df: pd.DataFrame,
    candidate_indices: list[int],
    group_cols: list[str],
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Randomly assign entire groups to train/val/test."""
    subset = df.loc[candidate_indices]
    groups = [g.index.to_list() for _, g in subset.groupby(group_cols, sort=False, dropna=False)]
    _rng(seed).shuffle(groups)

    target_val = round(len(candidate_indices) * fractions[1])
    target_test = round(len(candidate_indices) * fractions[2])
    val, test, train = [], [], []
    for group in groups:
        if len(test) < target_test:
            test.extend(group)
        elif len(val) < target_val:
            val.extend(group)
        else:
            train.extend(group)
    return sorted(train), sorted(val), sorted(test)


def _split_train_val_by_points(
    df: pd.DataFrame, indices: list[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    train, val, _ = _split_groups(
        df, indices, POINT_COLS, (1.0 - val_fraction, val_fraction, 0.0), seed
    )
    return train, val


def _is_unsaturated_fluorocarbon(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    has_cc_double = any(
        bond.GetBondTypeAsDouble() == 2.0
        and bond.GetBeginAtom().GetSymbol() == "C"
        and bond.GetEndAtom().GetSymbol() == "C"
        for bond in mol.GetBonds()
    )
    has_f = any(atom.GetSymbol() == "F" for atom in mol.GetAtoms())
    return has_cc_double and has_f


def _audit(
    df: pd.DataFrame,
    train: list[int],
    val: list[int],
    test: list[int],
    protocol: str,
) -> dict:
    partitions = [set(train), set(val), set(test)]
    assert not (partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2])

    exact_leaks = 0
    for _, group in df.groupby(POINT_COLS, sort=False, dropna=False):
        memberships = sum(bool(set(group.index) & part) for part in partitions)
        exact_leaks += memberships > 1

    train_df, test_df = df.loc[train], df.loc[test]
    train_systems = set(map(tuple, train_df[SYSTEM_COLS].to_numpy()))
    test_systems = set(map(tuple, test_df[SYSTEM_COLS].to_numpy()))
    train_il = set(map(tuple, train_df[["cation", "anion"]].to_numpy()))
    test_il = set(map(tuple, test_df[["cation", "anion"]].to_numpy()))

    return {
        "protocol": protocol,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "exact_duplicate_groups_crossing_partitions": int(exact_leaks),
        "test_systems": len(test_systems),
        "test_systems_seen_in_train": len(test_systems & train_systems),
        "test_il_pairs": len(test_il),
        "test_il_pairs_seen_in_train": len(test_il & train_il),
        "test_cations": int(test_df["cation"].nunique()),
        "test_cations_seen_in_train": len(set(test_df["cation"]) & set(train_df["cation"])),
        "test_anions": int(test_df["anion"].nunique()),
        "test_anions_seen_in_train": len(set(test_df["anion"]) & set(train_df["anion"])),
        "train_refrigerants": sorted(train_df["refrigerant"].unique().tolist()),
        "test_refrigerants": sorted(test_df["refrigerant"].unique().tolist()),
        "train_y_mean": float(train_df["x1"].mean()),
        "train_y_std": float(train_df["x1"].std()),
        "test_y_mean": float(test_df["x1"].mean()),
        "test_y_std": float(test_df["x1"].std()),
    }


def _save_split(path: Path, train: list[int], val: list[int], test: list[int], **metadata) -> None:
    np.savez(
        path,
        train=np.asarray(train, dtype=np.int64),
        val=np.asarray(val, dtype=np.int64),
        test=np.asarray(test, dtype=np.int64),
        metadata_json=json.dumps(metadata, ensure_ascii=True),
    )


def build_l0(df: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    return _split_groups(df, df.index.tolist(), POINT_COLS, (0.8, 0.1, 0.1), SEED)


def build_l1(df: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    """Hold out the high-condition tail of sufficiently sampled systems."""
    train, val, test = [], [], []
    rng = _rng(SEED + 1)
    for _, system in df.groupby(SYSTEM_COLS, sort=False):
        points = []
        for _, point in system.groupby(POINT_COLS, sort=False, dropna=False):
            points.append(point.index.to_list())

        if len(points) < 5:
            train.extend(system.index.to_list())
            continue

        point_rows = pd.DataFrame(
            [{"T": df.loc[g[0], "T_K"], "P": df.loc[g[0], "P_MPa"], "group": g} for g in points]
        )
        for col in ["T", "P"]:
            span = point_rows[col].max() - point_rows[col].min()
            point_rows[f"{col}_norm"] = 0.0 if span == 0 else (
                point_rows[col] - point_rows[col].min()
            ) / span
        point_rows["score"] = point_rows[["T_norm", "P_norm"]].max(axis=1)
        point_rows = point_rows.sort_values(["score", "P", "T"], ascending=False)

        n_test = max(1, round(len(point_rows) * 0.2))
        remaining = point_rows.iloc[n_test:].copy()
        n_val = max(1, round(len(remaining) * 0.1)) if len(remaining) >= 3 else 0
        val_positions = set(rng.choice(len(remaining), size=n_val, replace=False).tolist()) if n_val else set()

        test.extend(i for group in point_rows.iloc[:n_test]["group"] for i in group)
        for pos, group in enumerate(remaining["group"]):
            (val if pos in val_positions else train).extend(group)

    return sorted(train), sorted(val), sorted(test)


def build_l2(df: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    pair_series = df["cation"].astype(str) + "||" + df["anion"].astype(str)
    all_pairs = sorted(pair_series.unique())
    for trial_seed in range(SEED, SEED + 5000):
        rng = _rng(trial_seed)
        n_pairs = max(1, round(0.15 * len(all_pairs)))
        held_pairs = set(rng.choice(all_pairs, n_pairs, replace=False).tolist())
        test_mask = pair_series.isin(held_pairs)
        test_df, remain_df = df[test_mask], df[~test_mask]
        ratio = len(test_df) / len(df)
        if not 0.12 <= ratio <= 0.22:
            continue
        if not set(test_df["cation"]).issubset(set(remain_df["cation"])):
            continue
        if not set(test_df["anion"]).issubset(set(remain_df["anion"])):
            continue
        train, val = _split_train_val_by_points(df, remain_df.index.tolist(), 0.1, trial_seed)
        return train, val, sorted(test_df.index.tolist())
    raise RuntimeError("Unable to construct a valid L2 split after 5000 trials")


def build_l3(df: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    is_hfo = df["refri_smiles"].map(_is_unsaturated_fluorocarbon)
    test = sorted(df[is_hfo].index.tolist())
    train, val = _split_train_val_by_points(df, df[~is_hfo].index.tolist(), 0.1, SEED + 3)
    return train, val, test


def build_l4_splits(df: pd.DataFrame, min_samples: int) -> dict[str, tuple[list[int], list[int], list[int]]]:
    results = {}
    counts = df["refrigerant"].value_counts()
    for offset, target in enumerate(sorted(counts[counts >= min_samples].index.tolist())):
        test = sorted(df[df["refrigerant"] == target].index.tolist())
        remaining = df[df["refrigerant"] != target]
        train, val = _split_train_val_by_points(
            df, remaining.index.tolist(), 0.1, SEED + 100 + offset
        )
        results[target] = (train, val, test)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-loro-samples", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(ROOT / "index_with_anion.csv").reset_index(drop=True)
    required = set(POINT_COLS + ["refri_smiles", "x1"])
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    reports = {}
    builders = {
        "L0": (build_l0, "random interpolation; exact duplicates grouped"),
        "L1": (build_l1, "within-system high T/P extrapolation"),
        "L2": (build_l2, "unseen cation-anion recombination"),
        "L3": (build_l3, "saturated refrigerants to C=C fluorocarbon family shift"),
    }
    for level, (builder, protocol) in builders.items():
        train, val, test = builder(df)
        audit = _audit(df, train, val, test, protocol)
        assert audit["exact_duplicate_groups_crossing_partitions"] == 0
        if level == "L1":
            assert audit["test_systems_seen_in_train"] == audit["test_systems"]
        if level == "L2":
            assert audit["test_il_pairs_seen_in_train"] == 0
            assert audit["test_cations_seen_in_train"] == audit["test_cations"]
            assert audit["test_anions_seen_in_train"] == audit["test_anions"]
        if level == "L3":
            assert set(audit["train_refrigerants"]).isdisjoint(audit["test_refrigerants"])
        _save_split(ROOT / f"split_{level}_indices.npz", train, val, test, level=level, **audit)
        reports[level] = audit

    loro_dir = ROOT / "splits_loro"
    loro_dir.mkdir(exist_ok=True)
    for target, (train, val, test) in build_l4_splits(df, args.min_loro_samples).items():
        protocol = f"leave-one-refrigerant-out: {target}"
        audit = _audit(df, train, val, test, protocol)
        assert target not in audit["train_refrigerants"]
        assert audit["test_refrigerants"] == [target]
        assert audit["exact_duplicate_groups_crossing_partitions"] == 0
        _save_split(loro_dir / f"split_L4_{target}.npz", train, val, test, level="L4", held_out=target, **audit)
        reports[f"L4_{target}"] = audit

    report_path = ROOT / "split_report_v3.json"
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(reports)} audited splits and {report_path}")
    for name, report in reports.items():
        print(
            f"{name:12s} train={report['n_train']:4d} val={report['n_val']:4d} "
            f"test={report['n_test']:4d} duplicate_leaks="
            f"{report['exact_duplicate_groups_crossing_partitions']}"
        )


if __name__ == "__main__":
    main()
