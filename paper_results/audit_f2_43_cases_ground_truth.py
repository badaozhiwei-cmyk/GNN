"""
audit_f2_43_cases_ground_truth.py — Authoritative Ground Truth Audit for 43 State Points
========================================================================================
Rigorous verification of sample roles across 4 Case Studies (43 State Points):
1. Ensures strict 1:1 injection (len(sub) == 1) for every physical five-tuple.
2. Verifies npy_idx line-number alignment against canonical meta_info.csv and split npz.
3. Outputs self-contained audit table and SHA-256 provenance manifest.
"""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("  F2 43 STATE POINTS GROUND TRUTH AUDIT & PROVENANCE LOCK")
    print("=" * 80)

    hfc_meta_path = PROJECT_ROOT / "datasets" / "hfc_2739_v7" / "meta_info.csv"
    hfc_split_path = PROJECT_ROOT / "splits" / "HFC_grouped_state_split.npz"
    master_index_path = PROJECT_ROOT / "index_with_anion.csv"
    attr_cases_path = PROJECT_ROOT / "v7_shadow_experiment" / "results_attribution" / "v7_graph_attribution_groups.csv"

    print(f"[*] HFC Metadata: {hfc_meta_path} (SHA256: {compute_sha256(hfc_meta_path)})")
    print(f"[*] Canonical Split: {hfc_split_path} (SHA256: {compute_sha256(hfc_split_path)})")

    # Load canonical data
    hfc_meta = pd.read_csv(hfc_meta_path)
    assert len(hfc_meta) == 2739, f"Expected 2739 rows, got {len(hfc_meta)}"
    assert list(hfc_meta['npy_idx']) == list(range(2739)), "npy_idx must strictly equal 0..2738"

    raw_df = pd.read_csv(master_index_path)
    hfo_meta = raw_df[raw_df['sheet'] == 'Table S4. VLE HFOs'].copy().reset_index(drop=True)
    assert len(hfo_meta) == 1106, f"Expected 1106 HFO rows, got {len(hfo_meta)}"

    split_data = np.load(hfc_split_path)
    train_idx = set(split_data['train'])
    val_idx = set(split_data['val'])
    assert len(train_idx) == 2464 and len(val_idx) == 275
    assert (train_idx & val_idx) == set()
    assert (train_idx | val_idx) == set(range(2739))

    df_attr = pd.read_csv(attr_cases_path)
    cases = df_attr[['case_id', 'sample_id']].drop_duplicates().sort_values('case_id').reset_index(drop=True)
    assert len(cases) == 43, f"Expected 43 sample points, got {len(cases)}"
    assert cases['case_id'].nunique() == 4, f"Expected 4 unique case_ids, got {cases['case_id'].nunique()}"

    clean = lambda s: str(s).strip().replace('[', '').replace(']', '').upper()

    records = []
    for _, row in cases.iterrows():
        cid = row['case_id']
        sid = str(row['sample_id']).strip()
        parts = sid.split('__')
        c_cat, c_ani, c_ref = clean(parts[0]), clean(parts[1]), clean(parts[2])
        c_t, c_p = float(parts[3]), float(parts[4])

        # 1. Match HFC
        sub_hfc = hfc_meta[
            (hfc_meta['cation'].apply(clean) == c_cat) &
            (hfc_meta['anion'].apply(clean) == c_ani) &
            (hfc_meta['refrigerant'].apply(clean) == c_ref) &
            (np.isclose(hfc_meta['T_K'], c_t, atol=1e-2)) &
            (np.isclose(hfc_meta['P_MPa'], c_p, atol=1e-3))
        ]

        # 2. Match HFO
        sub_hfo = hfo_meta[
            (hfo_meta['cation'].apply(clean) == c_cat) &
            (hfo_meta['anion'].apply(clean) == c_ani) &
            (hfo_meta['refrigerant'].apply(clean) == c_ref) &
            (np.isclose(hfo_meta['T_K'], c_t, atol=1e-2)) &
            (np.isclose(hfo_meta['P_MPa'], c_p, atol=1e-3))
        ]

        if len(sub_hfc) > 0:
            assert len(sub_hfc) == 1, f"Ambiguous multiple HFC matches ({len(sub_hfc)}) for {sid}"
            assert len(sub_hfo) == 0, f"Cross-dataset conflict for {sid}"
            match_idx = int(sub_hfc.iloc[0]['npy_idx'])
            is_train = match_idx in train_idx
            is_val = match_idx in val_idx
            assert is_train ^ is_val, f"Sample {sid} (idx {match_idx}) must be either train or val"
            role = "HFC_Train" if is_train else "HFC_Val"

            records.append({
                'case_id': cid,
                'sample_id': sid,
                'refrigerant': c_ref,
                'data_role': role,
                'canonical_npy_idx': match_idx,
                'source_dataset': 'HFC-2739 (Table S3)',
                'is_in_training_set': is_train,
                'is_in_validation_set': is_val,
                'is_external_hfo': False,
                'candidates_count': 1
            })
        elif len(sub_hfo) > 0:
            assert len(sub_hfo) == 1, f"Ambiguous multiple HFO matches ({len(sub_hfo)}) for {sid}"
            records.append({
                'case_id': cid,
                'sample_id': sid,
                'refrigerant': c_ref,
                'data_role': 'HFO_External_Stress',
                'canonical_npy_idx': -1,
                'source_dataset': 'HFO-1106 (Table S4)',
                'is_in_training_set': False,
                'is_in_validation_set': False,
                'is_external_hfo': True,
                'candidates_count': 1
            })
        else:
            raise ValueError(f"Unmatched physical state point: {sid}")

    df_out = pd.DataFrame(records)
    out_csv = PROJECT_ROOT / "paper_results" / "f2_43_cases_ground_truth_audit.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n[+] Audit Passed Successfully:")
    print(df_out['data_role'].value_counts().to_string())
    print(f"[+] Output CSV: {out_csv} (SHA256: {compute_sha256(out_csv)})")
    print("=" * 80)

if __name__ == "__main__":
    main()
