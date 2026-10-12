"""
run_p1_controlled_ablation.py — V7 Phase 1 Controlled Architectural Ablation Runner
===================================================================================
Rigorous 5-Seed Controlled Comparison:
  Model A: 3-Node Interaction GAT (Canonical V7-A)
  Model B: Capacity-Matched Concat + MLP Baseline
Under identical training conditions, split, and optimizer protocol.
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import random
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from Dataset_v7 import DecoupledTriDataset_v7, get_v7_dataloader
from Model_v7 import IL_GAT_v7
from Model_v7_concat_mlp import IL_ConcatMLP_v7

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime_contract(data_root: Path, split_path: Path, bound_path: Path):
    """Fail closed if packaged data or split no longer matches its provenance contract."""
    with open(bound_path, "r", encoding="utf-8") as f:
        bound = json.load(f)

    source = bound.get("source_dataset", {})
    required = {
        "data.npy": source.get("data_npy_sha256"),
        "label.npy": source.get("label_npy_sha256"),
        "meta_info.csv": source.get("meta_info_sha256"),
    }
    for filename, expected in required.items():
        path = data_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required source file missing: {path}")
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise RuntimeError(
                f"Source hash mismatch for {filename}: expected={expected}, actual={actual}"
            )

    actual_split_hash = sha256_file(split_path)
    expected_split_hash = bound.get("split_npz_sha256")
    if not expected_split_hash or actual_split_hash != expected_split_hash:
        raise RuntimeError(
            f"Split hash mismatch: expected={expected_split_hash}, actual={actual_split_hash}"
        )

    data = np.load(data_root / "data.npy", allow_pickle=True)
    labels = np.load(data_root / "label.npy", allow_pickle=True)
    meta = pd.read_csv(data_root / "meta_info.csv")
    n_total = int(bound.get("n_total", -1))
    if n_total < 1 or len(data) != n_total or len(labels) != n_total or len(meta) != n_total:
        raise RuntimeError(
            f"Dataset length mismatch: contract={n_total}, data={len(data)}, "
            f"labels={len(labels)}, meta={len(meta)}"
        )
    if "npy_idx" not in meta.columns:
        raise RuntimeError("meta_info.csv lacks required npy_idx column")
    if meta["npy_idx"].tolist() != list(range(n_total)):
        raise RuntimeError("meta_info.csv npy_idx is not exactly 0..n_total-1 in row order")

    split = np.load(split_path)
    if "train" not in split.files or "val" not in split.files:
        raise RuntimeError(f"Split archive lacks train/val arrays: {split.files}")
    train = split["train"].astype(int)
    val = split["val"].astype(int)
    train_set, val_set = set(train.tolist()), set(val.tolist())
    expected_train = int(bound.get("n_train", -1))
    expected_val = int(bound.get("n_val", -1))
    if len(train_set) != len(train) or len(val_set) != len(val):
        raise RuntimeError("Duplicate indices detected in train or val split")
    if len(train) != expected_train or len(val) != expected_val:
        raise RuntimeError(
            f"Split length mismatch: expected train/val={expected_train}/{expected_val}, "
            f"actual={len(train)}/{len(val)}"
        )
    if train_set & val_set:
        raise RuntimeError("Train/validation index overlap detected")
    if train_set | val_set != set(range(n_total)):
        raise RuntimeError("Train/validation split does not cover 0..n_total-1 exactly")
    if np.any(train < 0) or np.any(val < 0) or np.any(train >= n_total) or np.any(val >= n_total):
        raise RuntimeError("Out-of-range split index detected")

    print("[PROVENANCE] Source and split hashes verified.")
    print(f"[PROVENANCE] n={n_total}; train={len(train)}; val={len(val)}; overlap=0; coverage=complete.")
    print(f"[PROVENANCE] data.npy   SHA256={required['data.npy']}")
    print(f"[PROVENANCE] label.npy  SHA256={required['label.npy']}")
    print(f"[PROVENANCE] meta_info  SHA256={required['meta_info.csv']}")
    print(f"[PROVENANCE] split.npz  SHA256={actual_split_hash}")
    return bound


def compute_metrics(true_y: np.ndarray, pred_y: np.ndarray) -> dict[str, float]:
    pred_clip = np.clip(pred_y, 0.0, 1.0)
    mae_raw  = mean_absolute_error(true_y, pred_y)
    r2_raw   = r2_score(true_y, pred_y) if len(true_y) > 1 and np.var(true_y) > 1e-12 else 0.0
    rmse_raw = np.sqrt(mean_squared_error(true_y, pred_y))
    mae_clip = mean_absolute_error(true_y, pred_clip)
    r2_clip  = r2_score(true_y, pred_clip) if len(true_y) > 1 and np.var(true_y) > 1e-12 else 0.0
    rmse_clip= np.sqrt(mean_squared_error(true_y, pred_clip))
    return {
        'mae_raw': float(mae_raw), 'r2_raw': float(r2_raw), 'rmse_raw': float(rmse_raw),
        'mae_clip': float(mae_clip), 'r2_clip': float(r2_clip), 'rmse_clip': float(rmse_clip)
    }

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    delta_y_graphs, delta_y_descs = [], []

    with torch.no_grad():
        for batch in loader:
            for k in ['cat', 'ani', 'ref']:
                batch[k] = batch[k].to(device)
            batch['state'] = batch['state'].to(device)
            batch['desc']  = batch['desc'].to(device)
            batch['label'] = batch['label'].to(device)

            out = model(batch)
            loss = criterion(out, batch['label'])
            total_loss += loss.item() * batch['label'].size(0)

            preds.append(out.cpu().numpy())
            targets.append(batch['label'].cpu().numpy())

            out_no_graph = model(batch, override_zero_graph=True)
            dy_g = torch.abs(out - out_no_graph).cpu().numpy()
            delta_y_graphs.extend(dy_g)

            out_no_desc = model(batch, override_mask_desc=0.0)
            dy_d = torch.abs(out - out_no_desc).cpu().numpy()
            delta_y_descs.extend(dy_d)

    all_preds = np.concatenate(preds)
    all_targets = np.concatenate(targets)
    avg_loss = total_loss / len(all_targets)

    metrics = compute_metrics(all_targets, all_preds)
    metrics['val_loss'] = float(avg_loss)
    metrics['median_delta_y_graph'] = float(np.median(delta_y_graphs))
    metrics['mean_delta_y_graph']   = float(np.mean(delta_y_graphs))
    metrics['graph_active_rate']    = float(np.mean(np.array(delta_y_graphs) >= 1e-3))
    metrics['median_delta_y_desc']  = float(np.median(delta_y_descs))
    metrics['mean_delta_y_desc']    = float(np.mean(delta_y_descs))
    return metrics, all_preds, all_targets

def train_single_seed(model_type: str, seed: int, args, data_root: Path, split_path: Path):
    print("\n" + "=" * 80)
    print(f"  RUNNING P1 TRIAL: Model = {model_type} | Seed = {seed}")
    print("=" * 80)
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    run_dir = PROJECT_ROOT / "v7_shadow_experiment" / "checkpoints_p1" / f"{model_type}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & Scalers
    sp_data = np.load(split_path)
    train_idx = sp_data['train'].astype(int)
    val_idx = sp_data['val'].astype(int)

    ds_train = DecoupledTriDataset_v7(str(data_root), valid_indices=train_idx)
    ds_train.fit_scalers(train_indices=list(range(len(train_idx))), save_dir=str(run_dir))

    ds_val = DecoupledTriDataset_v7(str(data_root), valid_indices=val_idx)
    ds_val.load_scalers(str(run_dir / "scalers.pkl"))

    train_loader = get_v7_dataloader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = get_v7_dataloader(ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 2. Model Instantiation & Parameter Verification
    model_args = {
        'emb_dim': 300,
        'hidden_dim': 512,
        'desc_dropout_p': args.desc_dropout_p,
        'dropout_rate': 0.2,
        'edge_dim': 4,
        'use_sigmoid': False
    }

    if model_type == "Interaction_GAT":
        model = IL_GAT_v7(model_args).to(device)
        expected_params = 29260989
    elif model_type == "Concat_MLP":
        model = IL_ConcatMLP_v7(model_args).to(device)
        expected_params = 29252959
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Model]: {model_type} | Trainable Params: {n_params:,} (Expected: {expected_params:,})")
    assert n_params == expected_params, f"Parameter count mismatch: {n_params} != {expected_params}"

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=args.delta)

    best_val_loss = float('inf')
    best_metrics = {}
    patience_counter = 0

    start_time = time.time()
    for epoch in range(1, args.epoch + 1):
        model.train()
        train_loss = 0.0
        n_train_samples = 0

        for batch in train_loader:
            for k in ['cat', 'ani', 'ref']:
                batch[k] = batch[k].to(device)
            batch['state'] = batch['state'].to(device)
            batch['desc']  = batch['desc'].to(device)
            batch['label'] = batch['label'].to(device)

            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch['label'])
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch['label'].size(0)
            n_train_samples += batch['label'].size(0)

        scheduler.step()
        avg_train_loss = train_loss / n_train_samples

        val_metrics, val_preds, val_targets = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['val_loss']

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                  f"Val MAE: {val_metrics['mae_raw']:.4f} | Val R2: {val_metrics['r2_raw']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics.copy()
            best_metrics['best_epoch'] = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'seed': seed,
                'model_type': model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'model_args': model_args,
                'n_params': n_params
            }, run_dir / f"best_model.pth")

            pd.DataFrame({
                'val_idx': val_idx,
                'y_true': val_targets,
                'y_pred': val_preds
            }).to_csv(run_dir / f"val_predictions.csv", index=False)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    [*] Early stopping triggered at epoch {epoch}")
                break

    best_metrics['duration_sec'] = time.time() - start_time
    best_metrics['model_type'] = model_type
    best_metrics['seed'] = seed
    best_metrics['params'] = n_params
    return best_metrics

def main():
    parser = argparse.ArgumentParser(description="V7 Phase 1 Controlled Architectural Ablation")
    parser.add_argument("--model_type", type=str, default="all", choices=["Interaction_GAT", "Concat_MLP", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--desc_dropout_p", type=float, default=0.0, help="Controlled at 0.0 for clean comparison")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_root = PROJECT_ROOT / "processed_tri_data_hfc2739"
    split_path = PROJECT_ROOT / "splits" / "HFC_grouped_state_split.npz"
    bound_path = PROJECT_ROOT / "splits" / "HFC_grouped_state_split_bound.json"

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split path not found: {split_path}")
    if not bound_path.exists():
        raise FileNotFoundError(f"Split bound json not found: {bound_path}")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError(f"Duplicate seeds are not allowed: {args.seeds}")

    bound = validate_runtime_contract(data_root, split_path, bound_path)
    models_to_run = ["Interaction_GAT", "Concat_MLP"] if args.model_type == "all" else [args.model_type]

    records = []
    for m_type in models_to_run:
        for seed in args.seeds:
            res = train_single_seed(m_type, seed, args, data_root, split_path)
            records.append(res)

    df_res = pd.DataFrame(records)
    out_dir = PROJECT_ROOT / "paper_results" / "p1_ablation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / "p1_ablation_5seeds_raw.csv", index=False)

    print("\n" + "=" * 80)
    print("  PHASE 1 ARCHITECTURAL ABLATION SUMMARY")
    print("=" * 80)
    for m_type in models_to_run:
        sub = df_res[df_res['model_type'] == m_type]
        mae_m, mae_s = sub['mae_raw'].mean(), sub['mae_raw'].std(ddof=1)
        r2_m, r2_s   = sub['r2_raw'].mean(), sub['r2_raw'].std(ddof=1)
        rmse_m, rmse_s = sub['rmse_raw'].mean(), sub['rmse_raw'].std(ddof=1)
        print(f"  {m_type:18s} | MAE: {mae_m:.4f} ± {mae_s:.4f} | R2: {r2_m:.4f} ± {r2_s:.4f} | RMSE: {rmse_m:.4f} ± {rmse_s:.4f}")

    # Persist a run manifest containing the exact data/split contract used for this run.
    manifest = {
        "project": "V7 P1 Controlled Architectural Ablation",
        "models": models_to_run,
        "seeds": args.seeds,
        "args": vars(args),
        "data_root": str(data_root),
        "split_file": str(split_path.name),
        "split_sha256": sha256_file(split_path),
        "source_hashes": {
            "data.npy": sha256_file(data_root / "data.npy"),
            "label.npy": sha256_file(data_root / "label.npy"),
            "meta_info.csv": sha256_file(data_root / "meta_info.csv"),
        },
        "n_total": bound["n_total"],
        "n_train": bound["n_train"],
        "n_val": bound["n_val"],
    }
    with open(out_dir / "p1_ablation_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if len(models_to_run) == 2 and len(args.seeds) == 5 and set(args.seeds) == {42, 43, 44, 45, 46}:
        piv = df_res.pivot(index="seed", columns="model_type", values=[
            "mae_raw", "rmse_raw", "r2_raw", "mae_clip", "rmse_clip", "r2_clip", "val_loss"
        ]).sort_index()
        metric_pairs = [
            ("mae_raw", "ΔMAE_raw (Concat_MLP - Interaction_GAT)"),
            ("rmse_raw", "ΔRMSE_raw (Concat_MLP - Interaction_GAT)"),
            ("r2_raw", "ΔR2_raw (Concat_MLP - Interaction_GAT)"),
            ("mae_clip", "ΔMAE_clip (Concat_MLP - Interaction_GAT)"),
            ("rmse_clip", "ΔRMSE_clip (Concat_MLP - Interaction_GAT)"),
            ("r2_clip", "ΔR2_clip (Concat_MLP - Interaction_GAT)"),
            ("val_loss", "ΔHuber_val_loss (Concat_MLP - Interaction_GAT)"),
        ]
        paired_rows = {"seed": piv.index.to_numpy()}
        summary_rows = []
        t_crit = float(stats.t.ppf(0.975, df=4))
        for key, label in metric_pairs:
            diffs = (piv[key]["Concat_MLP"] - piv[key]["Interaction_GAT"]).astype(float)
            paired_rows[label] = diffs.to_numpy()
            mean_diff = float(diffs.mean())
            sd_diff = float(diffs.std(ddof=1))
            margin = float(t_crit * sd_diff / np.sqrt(len(diffs)))
            summary_rows.append({
                "metric_difference": label,
                "n_seed_pairs": int(len(diffs)),
                "mean_difference": mean_diff,
                "sd_difference": sd_diff,
                "ci95_low_t_df4": mean_diff - margin,
                "ci95_high_t_df4": mean_diff + margin,
                "ci_method": "paired seed-level t interval; df=4",
            })
            print(f"  {label}: mean={mean_diff:+.6f}; SD={sd_diff:.6f}; "
                  f"95% CI [{mean_diff-margin:+.6f}, {mean_diff+margin:+.6f}]")
        pd.DataFrame(paired_rows).to_csv(out_dir / "p1_ablation_paired_differences.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(out_dir / "p1_ablation_paired_ci_summary.csv", index=False)
        print("[OUTPUT] Paired differences and 95% CI saved as CSV (seed-level uncertainty only).")
    else:
        print("[NOTE] Paired CI CSV not generated: requires both models and each seed 42–46 exactly once.")
    print("=" * 80)

if __name__ == "__main__":
    main()
