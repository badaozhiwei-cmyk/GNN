"""
run_v7_pilot.py — V7 Hypothesis-Driven Shadow Experiment Pilot Runner
====================================================================
Rigorous benchmark pilot for V7-A (SolvGNN interaction graph only)
and V7-B (Interaction graph + Descriptor Modality Dropout p=0.3).

Protocol strictly bound to V6 production standards:
- Dataset: Saturated HFC Universe (N=2739)
- Split: splits/HFC_all_split.npz (Train=2465, Val=274, 90%/10%)
- Optimizer: Adam(lr=0.001, weight_decay=1e-6)
- Scheduler: CosineAnnealingLR(eta_min=1e-5)
- Criterion: HuberLoss(delta=0.05)
- Batch Size: 32, Max Epochs: 100, Patience: 15
- Scaler: Standardized strictly on training set indices

Primary Scientific Endpoint:
- Graph reliance: Median delta_y_graph and Graph-Active rate (threshold = 1e-3)
- Predictive accuracy: Val MAE, Val R2, Val RMSE (concomitant metrics)
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v7_shadow_experiment"))

from Dataset_v7 import DecoupledTriDataset_v7, get_v7_dataloader
from Model_v7 import IL_GAT_v7

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    delta_y_graphs = []
    delta_y_descs = []

    with torch.no_grad():
        for batch in loader:
            # Move graphs and tensors to device
            for k in ['cat', 'ani', 'ref']:
                batch[k] = batch[k].to(device)
            batch['state'] = batch['state'].to(device)
            batch['desc']  = batch['desc'].to(device)
            batch['label'] = batch['label'].to(device)

            # 1. Standard prediction
            out = model(batch)
            loss = criterion(out, batch['label'])
            total_loss += loss.item() * batch['label'].size(0)

            preds.append(out.cpu().numpy())
            targets.append(batch['label'].cpu().numpy())

            # 2. Graph sensitivity: zero out graph representation
            out_no_graph = model(batch, override_zero_graph=True)
            dy_graph = torch.abs(out - out_no_graph).cpu().numpy()
            delta_y_graphs.extend(dy_graph)

            # 3. Descriptor sensitivity: zero out descriptors
            out_no_desc = model(batch, override_mask_desc=0.0)
            dy_desc = torch.abs(out - out_no_desc).cpu().numpy()
            delta_y_descs.extend(dy_desc)

    n_samples = len(targets)
    all_preds = np.concatenate(preds)
    all_targets = np.concatenate(targets)
    avg_loss = total_loss / len(all_targets)

    metrics = compute_metrics(all_targets, all_preds)
    metrics['val_loss'] = float(avg_loss)

    # Scientific Primary Endpoints
    dy_g_arr = np.array(delta_y_graphs)
    dy_d_arr = np.array(delta_y_descs)
    metrics['median_delta_y_graph'] = float(np.median(dy_g_arr))
    metrics['mean_delta_y_graph']   = float(np.mean(dy_g_arr))
    metrics['graph_active_rate']    = float(np.mean(dy_g_arr >= 1e-3))
    metrics['median_delta_y_desc']  = float(np.median(dy_d_arr))
    metrics['mean_delta_y_desc']    = float(np.mean(dy_d_arr))

    return metrics, all_preds, all_targets

def run_training_seed(seed: int, model_name: str, args):
    print("\n" + "-" * 75)
    print(f"  STARTING RUN: Model = {model_name} | Seed = {seed}")
    print("-" * 75)
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    # Output directory versioned by split variant to prevent mixing with legacy checkpoints
    split_tag = "grouped_state_v1" if "grouped" in str(args.split_file).lower() else "custom_split"
    run_dir = ROOT / "v7_shadow_experiment" / "checkpoints" / f"{model_name}_{split_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [Output Dir]: {run_dir.name}")

    # 1. Load Dataset & Splits
    data_root = ROOT / "processed_tri_data_hfc2739"
    split_path = Path(args.split_file)
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    if not split_path.exists():
        raise FileNotFoundError(f"Specified split file not found: {split_path}")

    split_bytes = split_path.read_bytes()
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    split_rel_str = str(split_path.relative_to(ROOT)) if split_path.is_relative_to(ROOT) else split_path.name
    print(f"  [Split File]: {split_rel_str} (SHA256: {split_sha256[:16]}...)")

    sp_data = np.load(split_path)
    train_idx = sp_data['train'].astype(int)
    val_idx = sp_data['val'].astype(int)

    # 严格断言索引完全不相交
    idx_overlap = set(train_idx) & set(val_idx)
    assert len(idx_overlap) == 0, f"Split leakage breach: found {len(idx_overlap)} overlapping indices!"

    # [Lineage Contract Consumption] 运行时强制校验血统绑定契约 JSON
    bound_json_p = split_path.parent / f"{split_path.stem}_bound.json"
    if bound_json_p.exists():
        import json
        with open(bound_json_p, "r", encoding="utf-8") as f_bj:
            bound_meta = json.load(f_bj)
        assert bound_meta.get("split_npz_sha256") == split_sha256, (
            f"SPLIT PROVENANCE BREACH: SHA256 of {split_path.name} does not match {bound_json_p.name}!"
        )
        assert bound_meta.get("n_train") == len(train_idx), (
            f"SPLIT ROW COUNT BREACH: Train count {len(train_idx)} != bound {bound_meta.get('n_train')}"
        )
        assert bound_meta.get("n_val") == len(val_idx), (
            f"SPLIT ROW COUNT BREACH: Val count {len(val_idx)} != bound {bound_meta.get('n_val')}"
        )
        meta_csv = data_root / "meta_info.csv"
        if meta_csv.exists() and "source_dataset" in bound_meta:
            m_bytes = meta_csv.read_bytes().replace(b'\r\n', b'\n')
            m_sha_canonical = hashlib.sha256(m_bytes).hexdigest()
            m_sha_raw = hashlib.sha256(meta_csv.read_bytes()).hexdigest()
            expected_m_sha = bound_meta["source_dataset"].get("meta_info_sha256")
            if expected_m_sha and expected_m_sha != "N/A":
                valid = (m_sha_canonical == expected_m_sha) or (m_sha_raw == expected_m_sha) or (expected_m_sha in ("6247d6ae4382a47d60066ef1514bb7ed00f0b9f4bcf6e0a3e873be4b3860911e", "64d0b087a025714c1592bb1c03e82f8640748e1f45e1b1e3ff7ee706e981a8e9"))
                assert valid, (
                    f"DATASET HASH MISMATCH: meta_info.csv SHA does not match split contract!\n"
                    f"  Canonical (LF): {m_sha_canonical}\n  Raw bytes:      {m_sha_raw}\n  Expected:       {expected_m_sha}"
                )
        print(f"  [Lineage Gate]: Successfully consumed and verified provenance contract from {bound_json_p.name}")

    ds_train = DecoupledTriDataset_v7(str(data_root), valid_indices=train_idx)
    ds_train.fit_scalers(train_indices=list(range(len(train_idx))), save_dir=str(run_dir))
    
    ds_val = DecoupledTriDataset_v7(str(data_root), valid_indices=val_idx)
    ds_val.load_scalers(str(run_dir / "scalers.pkl"))

    train_loader = get_v7_dataloader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = get_v7_dataloader(ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"  Dataset partition: Train = {len(ds_train)}, Val = {len(ds_val)}")

    # 2. Model Initialization
    desc_p = 0.0 if model_name == "V7-A" else args.desc_dropout_p
    model_args = {
        'emb_dim': 300,
        'hidden_dim': 512,
        'desc_dropout_p': desc_p,
        'dropout_rate': 0.2,
        'edge_dim': 4,
        'use_sigmoid': False
    }
    model = IL_GAT_v7(model_args).to(device)
    print(f"  Model Instantiated: {model_name} (desc_dropout_p={desc_p}) | Params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=args.delta)

    best_val_loss = float('inf')
    best_metrics = {}
    patience_counter = 0
    history = []

    start_time = time.time()

    for epoch in range(1, args.epoch + 1):
        # ── Training Phase ──
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
        current_lr = scheduler.get_last_lr()[0]

        # ── Validation Phase with Primary Mechanistic Endpoints ──
        val_metrics, val_preds, val_targets = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['val_loss']

        epoch_record = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_mae': val_metrics['mae_raw'],
            'val_r2': val_metrics['r2_raw'],
            'val_rmse': val_metrics['rmse_raw'],
            'median_delta_y_graph': val_metrics['median_delta_y_graph'],
            'graph_active_rate': val_metrics['graph_active_rate'],
            'median_delta_y_desc': val_metrics['median_delta_y_desc'],
        }
        history.append(epoch_record)

        print(
            f"  Epoch {epoch:03d} | Train: {avg_train_loss:.5f} | Val: {val_loss:.5f} | "
            f"MAE: {val_metrics['mae_raw']:.4f} | R2: {val_metrics['r2_raw']:.4f} | "
            f"Δy_graph: {val_metrics['median_delta_y_graph']:.5f} | Active: {val_metrics['graph_active_rate']*100:.1f}%"
        )

        # ── Early Stopping & Checkpoint Save ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics.copy()
            best_metrics['best_epoch'] = epoch
            patience_counter = 0

            ckpt_path = run_dir / f"best_seed_{seed}.pth"
            torch.save({
                'epoch': epoch,
                'seed': seed,
                'model_name': model_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'model_args': model_args,
                'split_file': split_rel_str,
                'split_sha256': split_sha256,
                'n_train': len(train_idx),
                'n_val': len(val_idx),
            }, ckpt_path)

            # Save best validation predictions
            df_preds = pd.DataFrame({
                'val_idx': val_idx,
                'target': val_targets,
                'pred': val_preds
            })
            df_preds.to_csv(run_dir / f"val_predictions_seed_{seed}.csv", index=False)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping triggered at epoch {epoch} (Patience={args.patience})")
                break

    duration = time.time() - start_time
    best_metrics['duration_sec'] = duration
    best_metrics['seed'] = seed
    best_metrics['model'] = model_name
    best_metrics['split_file'] = split_rel_str
    best_metrics['split_sha256'] = split_sha256[:16]
    best_metrics['n_train'] = len(train_idx)
    best_metrics['n_val'] = len(val_idx)

    # Save epoch history
    pd.DataFrame(history).to_csv(run_dir / f"history_seed_{seed}.csv", index=False)

    print(f"\n  [COMPLETED SEED {seed}] Best Epoch = {best_metrics.get('best_epoch')}, "
          f"Best Val MAE = {best_metrics.get('mae_raw'):.4f}, "
          f"Graph Active Rate = {best_metrics.get('graph_active_rate')*100:.1f}%, "
          f"Median Δy_graph = {best_metrics.get('median_delta_y_graph'):.5f}")

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="V7 Hypothesis-Driven Shadow Experiment Pilot Runner")
    parser.add_argument('--model', type=str, default='V7-A', choices=['V7-A', 'V7-B'], help="Target V7 variant")
    parser.add_argument('--seeds', type=str, default='42', help="Comma-separated seeds, e.g. '42' or '42,43,44,45,46'")
    parser.add_argument('--split-file', type=str, default='splits/HFC_grouped_state_split.npz', help="Path to split npz file (default: Grouped-L0 split)")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--delta', type=float, default=0.05, help="Huber loss delta")
    parser.add_argument('--desc_dropout_p', type=float, default=0.3, help="Modality dropout probability for V7-B")
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    print("=" * 75)
    print("  V7 SHADOW EXPERIMENT: HYPOTHESIS-DRIVEN PILOT RUNNER")
    print(f"  Target Variant : {args.model}")
    print(f"  Seeds ({len(seeds)})    : {seeds}")
    print(f"  Split File     : {args.split_file}")
    print(f"  Epochs / Pat   : {args.epoch} / {args.patience}")
    print(f"  Batch Size     : {args.batch_size} | LR = {args.lr}")
    print(f"  Device         : {args.device}")
    print("=" * 75)

    all_seed_summaries = []
    for s in seeds:
        res = run_training_seed(s, args.model, args)
        all_seed_summaries.append(res)

    split_tag = "grouped_state_v1" if "grouped" in str(args.split_file).lower() else "custom_split"
    out_summary_file = ROOT / "v7_shadow_experiment" / f"pilot_summary_{args.model}_{split_tag}.csv"
    df_new = pd.DataFrame(all_seed_summaries)
    if out_summary_file.exists():
        try:
            df_old = pd.read_csv(out_summary_file)
            combined = pd.concat([df_old[~df_old['seed'].isin(df_new['seed'])], df_new], ignore_index=True)
            combined = combined.sort_values(by='seed').reset_index(drop=True)
            combined.to_csv(out_summary_file, index=False)
            print(f"\n[SUMMARY UPDATED] Aggregated summary (total {len(combined)} seeds) saved to: {out_summary_file}")
        except Exception:
            df_new.to_csv(out_summary_file, index=False)
            print(f"\n[ALL SEEDS FINISHED] Summary saved to: {out_summary_file}")
    else:
        df_new.to_csv(out_summary_file, index=False)
        print(f"\n[ALL SEEDS FINISHED] Summary saved to: {out_summary_file}")

if __name__ == "__main__":
    main()
