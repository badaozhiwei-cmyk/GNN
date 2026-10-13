# P2 HFO independent recheck artifacts

This small package contains an independently corrected provenance/matching audit. It does **not** replace or modify the original `P2_HFO_provenance_bundle.zip`.

Files:
- `P2_HFO_independent_audit_report.md`: findings, reproducible counts, confirmed defect in the rounded-pressure mapping, and scope limits.
- `hfo_sample_level_mapping_corrected.csv`: corrected 1,106-row mapping produced by exact unrounded `(cation, anion, refrigerant, T_K, P_MPa)` matching. All 1,106 `sample_id` values are unique.
- `match_hfo_exact.py`: portable audit script; accepts explicit file paths rather than a hard-coded Windows directory.

To rerun, extract the **original** `P2_HFO_provenance_bundle.zip`, then run from the directory containing its `source/`, `predictions/`, and `summaries/` folders:

```bash
python match_hfo_exact.py \
  --source source/index_with_anion.csv \
  --predictions predictions/canonical_v7_ood_predictions_all.csv \
  --summary summaries/hfo_species_breakdown.csv \
  --manifest predictions/canonical_v7_ood_manifest.json \
  --mapping-out hfo_sample_level_mapping_corrected.csv
```

Dependencies: Python 3, pandas, numpy. The script checks the exact match, source-row suffix, float32 label agreement, summary metrics, and manifest metadata consistency. It does not rerun neural-network inference because the model checkpoint files and evaluator are not present in the original bundle.
