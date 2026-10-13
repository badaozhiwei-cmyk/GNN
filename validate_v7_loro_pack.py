"""No-training preflight validation for the bundled V7 LORO data and split files."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "processed_tri_data_hfc2739"
SPLITS = ROOT / "splits" / "HFC_loro_v7_2739"
meta = pd.read_csv(DATA / "meta_info.csv")
data = np.load(DATA / "data.npy", allow_pickle=True)
labels = np.load(DATA / "label.npy", allow_pickle=True).astype(float).reshape(-1)
assert len(meta) == len(data) == len(labels) == 2739
assert np.array_equal(meta["npy_idx"].to_numpy(), np.arange(len(meta)))
assert np.allclose(labels, meta["x1"].to_numpy(float), rtol=0, atol=1e-6)
files = sorted(SPLITS.glob("loro_*.npz"))
assert len(files) == 12, f"Expected 12 split files, found {len(files)}"

for p in files:
    with np.load(p, allow_pickle=True) as z:
        target_val = z["target_ref"]
        target = str(target_val.item() if hasattr(target_val, "item") else target_val)
        train, val, test = [np.asarray(z[k], dtype=int).reshape(-1) for k in ("train", "val", "test")]
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test), p.name
    assert set(train) | set(val) | set(test) == set(range(len(meta))), p.name
    assert target not in set(meta.iloc[train]["refrigerant"].astype(str)), p.name
    assert target not in set(meta.iloc[val]["refrigerant"].astype(str)), p.name
    assert set(meta.iloc[test]["refrigerant"].astype(str)) == {target}, p.name
    tr = meta.iloc[train]
    va = meta.iloc[val]
    def keys(df):
        return set(zip(df.cation.astype(str), df.anion.astype(str), df.refrigerant.astype(str),
                       df.T_K.astype(float).round(2), df.P_MPa.astype(float).round(3)))
    assert keys(tr).isdisjoint(keys(va)), f"physical-state group overlap: {p.name}"
    print(f"PASS {p.name}: target={target}, train={len(train)}, val={len(val)}, test={len(test)}")
print(f"\nPASS all 12 LORO splits; rows={len(meta)}; labels aligned within 1e-6")
print("Note: sample_id is not unique in this metadata; use npy_idx/row alignment, not sample_id alone, as the row key.")
