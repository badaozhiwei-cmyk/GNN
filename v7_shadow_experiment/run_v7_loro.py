"""V7 Leave-One-Refrigerant-Out runner (reviewed package layout).

This runner supports either a flattened self-contained bundle (this file next to
Model_v7.py, Dataset_v7.py and processed_tri_data_hfc2739/) or the repository
layout where this file lives in v7_shadow_experiment/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
if (HERE / "processed_tri_data_hfc2739" / "meta_info.csv").is_file():
    ROOT = HERE
elif (HERE.parent / "processed_tri_data_hfc2739" / "meta_info.csv").is_file():
    ROOT = HERE.parent
else:
    # Keep a useful error below rather than silently looking in an unrelated parent.
    ROOT = HERE

for module_dir in (ROOT, ROOT / "v7_shadow_experiment", HERE, HERE / "v7_shadow_experiment"):
    if module_dir.exists() and str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from Dataset_v7 import DecoupledTriDataset_v7, get_v7_dataloader  # noqa: E402
from Model_v7 import IL_GAT_v7  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_and_validate_data(data_root: Path):
    meta_path = data_root / "meta_info.csv"
    data_path = data_root / "data.npy"
    label_path = data_root / "label.npy"
    missing = [str(p) for p in (meta_path, data_path, label_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required data files: " + ", ".join(missing))

    meta = pd.read_csv(meta_path)
    raw_data = np.load(data_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True).astype(np.float64).reshape(-1)
    n = len(meta)
    if len(raw_data) != n or len(labels) != n:
        raise ValueError(f"Data/meta/label row mismatch: data={len(raw_data)}, meta={n}, labels={len(labels)}")
    if "npy_idx" not in meta.columns:
        raise ValueError("meta_info.csv lacks required npy_idx column")
    npy_idx = meta["npy_idx"].to_numpy()
    if not np.array_equal(npy_idx, np.arange(n)):
        raise ValueError("meta_info.csv npy_idx is not sequentially aligned with data.npy/label.npy rows")
    if "x1" in meta.columns and not np.allclose(labels, meta["x1"].to_numpy(dtype=float), rtol=0, atol=1e-6):
        raise ValueError("label.npy does not align with meta_info.csv x1 (tolerance 1e-6)")
    required_cols = {"refrigerant", "cation", "anion", "T_K", "P_MPa"}
    missing_cols = required_cols - set(meta.columns)
    if missing_cols:
        raise ValueError(f"meta_info.csv missing columns: {sorted(missing_cols)}")
    return meta, labels, {"meta_info.csv": sha256_file(meta_path), "data.npy": sha256_file(data_path), "label.npy": sha256_file(label_path)}


def group_keys(frame: pd.DataFrame):
    # Matches the physical-state grouping convention used in the bundled split audit:
    # same cation/anion/refrigerant and temperature/pressure rounded to 2/3 decimals.
    return set(zip(
        frame["cation"].astype(str),
        frame["anion"].astype(str),
        frame["refrigerant"].astype(str),
        frame["T_K"].astype(float).round(2),
        frame["P_MPa"].astype(float).round(3),
    ))


def validate_split(split_file: Path, meta: pd.DataFrame):
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")
    with np.load(split_file, allow_pickle=True) as z:
        required = {"train", "val", "test", "target_ref"}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"{split_file.name}: missing keys {sorted(missing)}")
        train_idx = np.asarray(z["train"], dtype=np.int64).reshape(-1)
        val_idx = np.asarray(z["val"], dtype=np.int64).reshape(-1)
        test_idx = np.asarray(z["test"], dtype=np.int64).reshape(-1)
        raw_target = z["target_ref"]
        target_ref = str(raw_target.item() if hasattr(raw_target, "item") else raw_target)

    n = len(meta)
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx) == 0:
            raise ValueError(f"{split_file.name}: {name} indices are empty")
        if idx.min() < 0 or idx.max() >= n:
            raise IndexError(f"{split_file.name}: {name} indices outside 0..{n-1}")
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"{split_file.name}: duplicate indices in {name}")

    if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
        raise ValueError(f"{split_file.name}: train/val/test index sets overlap")
    union = set(train_idx) | set(val_idx) | set(test_idx)
    if union != set(range(n)):
        raise ValueError(f"{split_file.name}: split does not cover every one of the {n} rows exactly once")

    train_refs = set(meta.iloc[train_idx]["refrigerant"].astype(str))
    val_refs = set(meta.iloc[val_idx]["refrigerant"].astype(str))
    test_refs = set(meta.iloc[test_idx]["refrigerant"].astype(str))
    if target_ref in train_refs or target_ref in val_refs:
        raise ValueError(f"{split_file.name}: target refrigerant {target_ref} appears in train/val")
    if test_refs != {target_ref}:
        raise ValueError(f"{split_file.name}: test contains refrigerants {sorted(test_refs)}, expected only {target_ref}")

    if group_keys(meta.iloc[train_idx]) & group_keys(meta.iloc[val_idx]):
        raise ValueError(f"{split_file.name}: physical-state group overlap between train and val")

    return train_idx, val_idx, test_idx, target_ref


def compute_metrics(true_y: np.ndarray, pred_y: np.ndarray) -> dict[str, float]:
    true_y = np.asarray(true_y).reshape(-1)
    pred_y = np.asarray(pred_y).reshape(-1)
    pred_clip = np.clip(pred_y, 0.0, 1.0)
    def safe_r2(y, p):
        return float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 1e-12 else float("nan")
    return {
        "mae_raw": float(mean_absolute_error(true_y, pred_y)),
        "rmse_raw": float(np.sqrt(mean_squared_error(true_y, pred_y))),
        "r2_raw": safe_r2(true_y, pred_y),
        "mae_clip": float(mean_absolute_error(true_y, pred_clip)),
        "rmse_clip": float(np.sqrt(mean_squared_error(true_y, pred_clip))),
        "r2_clip": safe_r2(true_y, pred_clip),
    }


def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            for key in ("cat", "ani", "ref"):
                batch[key] = batch[key].to(device)
            for key in ("state", "desc", "label"):
                batch[key] = batch[key].to(device)
            out = model(batch)
            loss = criterion(out, batch["label"])
            total_loss += loss.item() * batch["label"].size(0)
            preds.append(out.detach().cpu().numpy().reshape(-1))
            targets.append(batch["label"].detach().cpu().numpy().reshape(-1))
    if not targets:
        raise RuntimeError("Evaluation loader yielded no batches")
    all_preds = np.concatenate(preds)
    all_targets = np.concatenate(targets)
    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = float(total_loss / len(all_targets))
    return metrics, all_preds, all_targets


def run_single_fold(split_file: Path, model_name: str, seed: int, args, output_base: Path, meta: pd.DataFrame, data_hashes):
    set_seed(seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("WARNING: CUDA requested but unavailable; running this fold on CPU.")

    train_idx, val_idx, test_idx, target_ref = validate_split(split_file, meta)
    print(f"\n{'=' * 75}\nLORO target={target_ref} | model={model_name} | seed={seed}\n"
          f"Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)} | device={device}\n{'=' * 75}")

    out_dir = output_base / f"{target_ref}_{model_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = ROOT / "processed_tri_data_hfc2739"

    ds_train = DecoupledTriDataset_v7(str(data_root), valid_indices=train_idx)
    # ds_train.data is already reduced to train rows, hence local indices 0..n_train-1 are correct.
    ds_train.fit_scalers(train_indices=list(range(len(ds_train))), save_dir=str(out_dir))
    scaler_path = out_dir / "scalers.pkl"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Scaler file was not created: {scaler_path}")

    ds_val = DecoupledTriDataset_v7(str(data_root), valid_indices=val_idx)
    ds_val.load_scalers(str(scaler_path))
    ds_test = DecoupledTriDataset_v7(str(data_root), valid_indices=test_idx)
    ds_test.load_scalers(str(scaler_path))

    train_loader = get_v7_dataloader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader = get_v7_dataloader(ds_val, batch_size=args.batch_size, shuffle=False)
    test_loader = get_v7_dataloader(ds_test, batch_size=args.batch_size, shuffle=False)

    desc_p = args.desc_dropout_p if model_name == "V7-B" else 0.0
    model_args = {"emb_dim": 300, "hidden_dim": 512, "desc_dropout_p": float(desc_p),
                  "dropout_rate": 0.2, "use_sigmoid": False}
    model = IL_GAT_v7(args=model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=args.delta)

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_counter = 0
    started = time.time()

    for ep in range(1, args.epoch + 1):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            for key in ("cat", "ani", "ref"):
                batch[key] = batch[key].to(device)
            for key in ("state", "desc", "label"):
                batch[key] = batch[key].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = criterion(out, batch["label"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite train loss at epoch {ep} for {target_ref}/{model_name}/seed{seed}")
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch["label"].size(0)
        scheduler.step()

        val_metrics, _, _ = evaluate_loader(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if ep % 10 == 0 or ep == 1 or patience_counter >= args.patience:
            print(f"Epoch {ep:03d} | train_loss={train_loss_sum / len(train_idx):.5f} | "
                  f"val_loss={val_loss:.5f} | val_MAE={val_metrics['mae_raw']:.4f} | val_R2={val_metrics['r2_raw']:.4f}")
        if patience_counter >= args.patience:
            print(f"Early stop at epoch {ep}; best epoch={best_epoch}, best val loss={best_val_loss:.5f}")
            break

    duration = time.time() - started
    if best_weights is None:
        raise RuntimeError(f"No finite validation checkpoint for {target_ref}/{model_name}/seed{seed}")
    ckpt_path = out_dir / "best_model.pth"
    torch.save(best_weights, ckpt_path)
    model.load_state_dict(best_weights)

    # Test is accessed only after the best epoch has been selected using val loss.
    test_metrics, test_preds, test_targets = evaluate_loader(model, test_loader, criterion, device)
    print(f"FINAL HELD-OUT TEST {target_ref}: MAE(raw)={test_metrics['mae_raw']:.4f}; "
          f"RMSE(raw)={test_metrics['rmse_raw']:.4f}; R2(raw)={test_metrics['r2_raw']:.4f}; "
          f"MAE(clip)={test_metrics['mae_clip']:.4f}; best_epoch={best_epoch}; {duration:.1f}s")

    test_meta = meta.iloc[test_idx].copy()
    test_meta["y_true"] = test_targets
    test_meta["y_pred_raw"] = test_preds
    test_meta["y_pred_clip"] = np.clip(test_preds, 0.0, 1.0)
    test_meta["abs_error_raw"] = np.abs(test_meta["y_true"] - test_meta["y_pred_raw"])
    test_meta["abs_error_clip"] = np.abs(test_meta["y_true"] - test_meta["y_pred_clip"])
    test_meta.to_csv(out_dir / f"predictions_{target_ref}.csv", index=False)

    run_config = {
        "target_ref": target_ref, "model": model_name, "seed": seed,
        "model_args": model_args, "epoch_cap": args.epoch, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay, "patience": args.patience,
        "huber_delta": args.delta, "device": str(device), "best_epoch": best_epoch,
        "split_file": str(split_file.relative_to(ROOT) if split_file.is_relative_to(ROOT) else split_file),
        "split_sha256": sha256_file(split_file), "data_sha256": data_hashes,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    return {
        "target_ref": target_ref, "model": model_name, "seed": seed,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "best_epoch": best_epoch, "duration_sec": duration, "val_loss_best": best_val_loss,
        "test_mae_raw": test_metrics["mae_raw"], "test_rmse_raw": test_metrics["rmse_raw"],
        "test_r2_raw": test_metrics["r2_raw"], "test_mae_clip": test_metrics["mae_clip"],
        "test_rmse_clip": test_metrics["rmse_clip"], "test_r2_clip": test_metrics["r2_clip"],
        "split_sha256": sha256_file(split_file), "checkpoint": str(ckpt_path.relative_to(ROOT)),
    }


def main():
    parser = argparse.ArgumentParser(description="V7 LORO runner with split/data checks")
    parser.add_argument("--split-file", type=str, default="", help="Single .npz split; optional")
    parser.add_argument("--splits-dir", type=str, default="splits/HFC_loro_v7_2739")
    parser.add_argument("--target-ref", type=str, default="", help="e.g. R125")
    parser.add_argument("--model", type=str, default="V7-B", choices=["V7-A", "V7-B"])
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--desc_dropout_p", type=float, default=0.3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", type=str, default="results_v7_loro")
    args = parser.parse_args()

    data_root = ROOT / "processed_tri_data_hfc2739"
    meta, labels, data_hashes = load_and_validate_data(data_root)
    print(f"Preflight data check passed: N={len(meta)}; refrigerants={meta['refrigerant'].nunique()}; ROOT={ROOT}")

    try:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be comma-separated integers, e.g. 42,43,44") from exc
    if not seeds:
        raise ValueError("No seeds supplied")

    splits_dir = ROOT / args.splits_dir
    if args.split_file:
        supplied = Path(args.split_file)
        split_files = [supplied if supplied.is_absolute() else ROOT / supplied]
    elif args.target_ref:
        split_files = [splits_dir / f"loro_{args.target_ref}.npz"]
    else:
        split_files = sorted(splits_dir.glob("loro_*.npz"))
    if not split_files:
        raise FileNotFoundError(f"No LORO split files found for requested configuration under {splits_dir}")

    output_arg = Path(args.output_dir)
    out_base = output_arg if output_arg.is_absolute() else ROOT / output_arg
    out_base.mkdir(parents=True, exist_ok=True)
    all_records = []
    for split_file in split_files:
        for seed in seeds:
            record = run_single_fold(split_file, args.model, seed, args, out_base, meta, data_hashes)
            all_records.append(record)
            pd.DataFrame(all_records).to_csv(out_base / "loro_metrics_summary.csv", index=False)
    print(f"\nLORO run complete. Summary: {out_base / 'loro_metrics_summary.csv'}")


if __name__ == "__main__":
    main()
