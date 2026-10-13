# Pre-run code review — V7 LORO inspection package

## Result
The bundled 2,739-row dataset and all 12 split files passed independent static preflight checks in the review environment. Do not run the original runner version: the reviewed bundle replaces it with a runner that fixes the API/path errors below and adds hard assertions.

## Confirmed split/data checks
- `data.npy`, `label.npy`, and `meta_info.csv` each have 2,739 rows.
- `meta_info.csv:npy_idx` equals `0..2738`; `label.npy` agrees with `meta_info.csv:x1` within absolute tolerance `1e-6` (observed maximum absolute difference about `3e-8`).
- All 12 NPZ files partition the 2,739 row indices exactly once, without train/validation/test overlap.
- Each held-out test fold contains only its named target refrigerant; the target is absent from that fold's train and validation samples.
- No train/validation overlap was found using the physical-state key (cation, anion, refrigerant, T_K rounded to 2 decimals, P_MPa rounded to 3 decimals).
- `sample_id` is not unique in the supplied metadata, so do not use `sample_id` alone to join predictions; retain `npy_idx`/row index and the complete state identity.

## Definite faults in the original runner
1. It called `ds_val.load_scalers(load_dir=...)` and `ds_test.load_scalers(load_dir=...)`, but `Dataset_v7.load_scalers` accepts `scaler_path` positionally and expects the `scalers.pkl` file, not a directory.
2. It called `IL_GAT_v7(desc_dropout_p=desc_p)`, but the model constructor is `IL_GAT_v7(args=None)`. The dropout setting must be passed through `args={"desc_dropout_p": ...}`.
3. The package flattens `run_v7_loro.py` at bundle root, while the original script calculated `ROOT` as the parent directory's parent. That pointed to the wrong data directory after bundle extraction. The reviewed runner detects the bundled root or the original repo subdirectory layout.

## How to proceed
1. Run `python check_runtime.py` to confirm required ML packages and GPU availability.
2. Run `python validate_v7_loro_pack.py` after extracting the reviewed bundle.
3. Run `python pilot_test_1fold.py` on the intended GPU environment.
3. Inspect the R125 summary row, predictions, `run_manifest.json`, and scaler/checkpoint files. Only then launch the 12-fold runner.
4. `run_all_12folds.py` runs V7-B with five seeds (60 training runs), not both V7-A and V7-B (120 runs). Run V7-A separately with a separate output directory only if the planned analysis needs that comparison.

## Evidence boundary
The model training itself was not executed during this package review. This review environment has CPU-only PyTorch and does not have `torch_geometric`, so it cannot validate a real model forward pass or training runtime. The split results and code API issues above were checked directly against the uploaded files.

The V6 result CSV and historical training scripts are not included in this ZIP. Therefore, claims that the V6 scores use exactly the same 2,739-row source data, preprocessing, fold composition, target clipping convention, and metric computation still require separate verification before calling a V6–V7 comparison apples-to-apples.
