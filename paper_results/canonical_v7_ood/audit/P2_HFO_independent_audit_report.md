# P2 HFO Provenance Bundle — Independent Recheck

**Audit date:** 2026-10-13  
**Scope:** Uploaded `P2_HFO_provenance_bundle.zip`; source-to-prediction HFO matching, species counts, summary metric reproducibility, bundle scripts, and manifest consistency.

## Executive result

The **HFO species counts and the underlying source-to-prediction pairing are confirmed** when matched with the full, unrounded chemical/state tuple. However, the supplied `hfo_sample_level_mapping.csv` and its rounded-pressure matching logic contain one incorrect row. The original audit scripts also do not faithfully reproduce the claimed one-to-one match as written. The attached `hfo_sample_level_mapping_corrected.csv` and `match_hfo_exact.py` resolve this by retaining full-precision `T_K` and `P_MPa` values.

## 1. Integrity and internal metadata

The uploaded ZIP passed `zipfile.testzip()` (no archive corruption). The independently computed ZIP SHA-256 is:

`6af8d0c91c1fb3ceef2eb638d62ca3e112d95bd135dc174baaeb9dcc44a5a5d4`

The SHA-256 values of the seven files inside the bundle match the hashes included in the uploaded transcript. The source CSV hash is `1151ef5371f23d4b0f7ca5a6e4fa53343788bb2f58a9bb6fab2a376dd7f848f1`; the canonical prediction CSV hash is `7882ca7bfcd7305d4553acbb99bf5224ff21f550f624a877cb383286cea5dd1a`.

The 10 checkpoint-hash strings in prediction rows match the manifest values for V7-A/V7-B and seeds 42–46. Prediction-row split SHA-256, scaler SHA-256, Git commit, and per-domain unique sample counts also agree with the manifest. This verifies **internal metadata consistency**, not that model inference can be rerun: the actual `.pth` checkpoints and `eval_canonical_v7_ood.py` are not included in this bundle.

## 2. HFO counts and exact one-to-one matching

- Source table: **4,444 rows**.
- HFO source rows: **1,106**, with **1,106 unique exact chemical/state tuples**.
- HFO prediction rows: **11,060** = 1,106 sample IDs × 10 model/seed records per sample.
- Every HFO sample ID has 10 records: V7-A and V7-B, each at seeds 42, 43, 44, 45, and 46.
- Joining on normalized cation, anion, refrigerant name, and the **unrounded** `T_K` and `P_MPa` values yields 1,106 `both`, 0 `left_only`, and 0 `right_only` rows.
- Each `sample_id` trailing row-index suffix matches the corresponding source CSV row index, and the molecular/state metadata agree.

### Source species distribution

| Refrigerant | N | Share of HFO rows |
|---|---:|---:|
| R1234yf | 685 | 61.93% |
| R1234ze(E) | 308 | 27.85% |
| R1233zd(E) | 90 | 8.14% |
| R1336mzz(Z) | 12 | 1.08% |
| R1336mzz(E) | 11 | 0.99% |
| **Total** | **1,106** | **100.00%** |

These counts are reproduced directly from `index_with_anion.csv` and agree with the `N` values in `hfo_species_breakdown.csv`.

## 3. Label consistency: exact wording matters

Comparing the stored CSV numbers directly gives:

- max `abs(x1 - y_true)`: **2.7179718 × 10⁻⁸**;
- max `abs(float32(x1) - y_true)`: **9.7144515 × 10⁻¹⁷**.

The small difference is consistent with float32 representation of the source label in the prediction records. Therefore, describe the check as **agreement within float32 numerical precision**, not literal binary equality of the two CSV columns.

## 4. Defect found in the supplied mapping table

The supplied mapping table has 1,106 rows but only **1,105 unique `canonical_sample_id` values**. Source row **2840** (pressure `0.3263 MPa`, `x1=0.0207`) was assigned row 2807's ID, which encodes pressure `0.3265 MPa` and suffix `2807`. The mapped `label_abs_diff` for row 2840 is approximately **0.0001000001**, although `match_status` says `EXACT_MATCH`.

The cause is the rounded key: both `0.3263` and `0.3265` become `0.326` when rounded to three decimals. This creates a 2×2 many-to-many join for those two observations. The rounded join produces 1,108 rows rather than 1,106 and cannot substantiate a one-to-one bijection. The exact, unrounded join resolves this cleanly; the underlying source and predictions contain both observations correctly.

## 5. Problems in the bundled audit scripts

1. `audit_p2_hfo_provenance.py` hard-codes a Windows repository path. More importantly, it parses `sample_id` as though `parts[0]` were the cation; actual IDs begin with `HFO_[cation]`, so that part includes the `HFO_` prefix. This parser does not reconstruct the tuple correctly.
2. `match_hfo_direct.py` also hard-codes a Windows path and rounds `P_MPa` to three decimals. In this data that yields 1,108 matched merge rows because of the pressure collision; its success condition expects exactly 1,106 and therefore does not confirm the claimed match.
3. The supplied mapping file labels all 1,106 rows `EXACT_MATCH` despite the duplicated ID and the 0.0001 label discrepancy described above.

The replacement `match_hfo_exact.py` is path-agnostic, does not round the state fields, checks one-to-one cardinality and row-index suffixes, and writes the corrected mapping supplied with this report.

## 6. Species summary metrics reproduce

I recomputed the species table from the prediction file by averaging `y_pred_clip` over the five seeds for each sample, then calculating sample-level MAE and R²; `Sigma` was recomputed as the mean per-sample standard deviation across those five clipped predictions. The resulting MAE, R², and Sigma values reproduce every row of `hfo_species_breakdown.csv` to numerical precision (differences below 1 × 10⁻¹⁵ in the run here).

R² values for R1336mzz(E) (**N=11**) and R1336mzz(Z) (**N=12**) are highly negative and should be reported as small-sample descriptive results rather than stable estimates of species-level performance.

## 7. What this package does and does not establish

**Established by the supplied files and this recheck:** source species counts; exact one-to-one HFO record pairing; label agreement at float32 precision; reproducibility of the existing per-species metric table; internal agreement between prediction metadata and manifest hash strings.

**Not established by this bundle alone:** independent rerunning of the actual neural-network forward pass or verification that the listed checkpoint weights generated the predictions. For that stronger claim, supply the evaluator script named in the manifest (`eval_canonical_v7_ood.py`, expected SHA-256 `026c0b0158ad1723a96cbabde97231f8430bdc0359f50f885a8cde85fc7918f1`) and the ten checkpoint `.pth` files whose SHA values the manifest records.
