"""
run_hfc_all_production.py — All-HFC Production Model Runner
===========================================================
【实验使命】
  在完整 Saturated HFC Universe (N=2739, 12 种冷媒, 23 种阴离子) 上训练 5-Seed 生产模型。
  模式:
    - M0: 9 维基础特征
    - Mreduced: 12 维特征 (M0 + Tr, Pr, omega 对比态热力学先验)
  种子: 42, 43, 44, 45, 46
  产出存入: results_hfc_all/
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import random
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.append(str(ROOT / 'GNN_for_property_prediction'))

from Dataset_v6 import (
    FEATURE_SCHEMA,
    BASE_FEATURES,
    MODE_DEF,
    MODE_INDICES,
    MODE_COND_DIM,
    IL_set_v6
)
from Model_v6 import IL_GAT_v6

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def compute_metrics(true_y: np.ndarray, pred_y: np.ndarray) -> dict[str, float]:
    pred_y_clipped = np.clip(pred_y, 0.0, 1.0)
    mae_raw  = mean_absolute_error(true_y, pred_y)
    r2_raw   = r2_score(true_y, pred_y) if len(true_y) > 1 and np.var(true_y) > 1e-12 else 0.0
    rmse_raw = np.sqrt(mean_squared_error(true_y, pred_y))
    
    mae_clip  = mean_absolute_error(true_y, pred_y_clipped)
    r2_clip   = r2_score(true_y, pred_y_clipped) if len(true_y) > 1 and np.var(true_y) > 1e-12 else 0.0
    rmse_clip = np.sqrt(mean_squared_error(true_y, pred_y_clipped))
    
    return {
        'mae_raw': float(mae_raw), 'r2_raw': float(r2_raw), 'rmse_raw': float(rmse_raw),
        'mae_clip': float(mae_clip), 'r2_clip': float(r2_clip), 'rmse_clip': float(rmse_clip)
    }

def main():
    parser = argparse.ArgumentParser(description="All-HFC Production Model Runner")
    parser.add_argument('--mode', type=str, default='all', choices=['M0', 'Mreduced', 'all'])
    parser.add_argument('--seeds', type=str, default='42,43,44,45,46')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--dry_run', action='store_true', help="Preflight check only")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    modes = ['M0', 'Mreduced'] if args.mode == 'all' else [args.mode]

    data_dir = ROOT / 'processed_tri_data_hfc2739'
    splits_dir = ROOT / 'splits'
    out_base = ROOT / 'results_hfc_all'
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  ALL-HFC PRODUCTION MODEL BENCHMARK CONTROLLER")
    print(f"  Target Modes : {modes}")
    print(f"  Seeds ({len(seeds)})    : {seeds}")
    print(f"  Epochs / Pat : {args.epoch} / {args.patience}")
    print(f"  Dry-run Only : {args.dry_run}")
    print("=" * 75)

    split_file = splits_dir / "HFC_all_split.npz"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")

    sp_data = np.load(split_file)
    train_idx = sp_data['train'].astype(int)
    val_idx   = sp_data['val'].astype(int)

    meta_file = data_dir / 'meta_info.csv'
    df_meta = pd.read_csv(meta_file)
    print(f"  [Meta Loaded]  : {len(df_meta)} rows aligned with Saturated HFC Universe")
    print(f"  [Split Indices]: Train={len(train_idx)}, Val={len(val_idx)}")

    if args.dry_run:
        print("\n[PREFLIGHT AUDIT: DRY-RUN VERIFICATION PASSED]")
        for md in modes:
            print(f"  Mode {md:<9}: cond_dim = {MODE_COND_DIM[md]}, indices = {MODE_INDICES[md]}")
        return

    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch_geometric.loader import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [Compute Device]: {device}")

    summary_records = []

    for md in modes:
        cond_dim = MODE_COND_DIM[md]
        run_tag = f"HFC_all_{md}"
        run_dir = out_base / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        exp_config = {
            "split": "HFC_all_split",
            "descriptor_mode": md,
            "cond_dim": cond_dim,
            "feature_indices": MODE_INDICES[md],
            "seeds": seeds,
            "epoch": args.epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "split_sha256": sha256_file(split_file),
        }
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(exp_config, f, indent=2)

        print(f"\n{'='*70}")
        print(f"  Executing Training: {run_tag} | Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"{'='*70}")

        whole_set = IL_set_v6(path=str(data_dir), args={'descriptor_mode': md})

        # 严格仅在训练集拟合 StandardScaler
        whole_set.fit_scalers(train_idx, save_dir=str(run_dir))

        train_set = torch.utils.data.Subset(whole_set, train_idx)
        val_set   = torch.utils.data.Subset(whole_set, val_idx)

        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        seed_metrics = []

        for seed in seeds:
            set_seed(seed)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

            model_args = {
                'emb_dim': 300,
                'dropout_rate': 0.2,
                'cond_dim': cond_dim,
                'pool': 'global',
                'use_layernorm': False,
                'use_adaptive_gate': False
            }
            model = IL_GAT_v6(model_args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)
            criterion = nn.HuberLoss(delta=0.05)

            best_val_loss = float('inf')
            best_ckpt = run_dir / f"best_seed_{seed}.pth"
            early_stop_count = 0

            for epoch in range(1, args.epoch + 1):
                model.train()
                for graph, cond, label in train_loader:
                    graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                    optimizer.zero_grad()
                    out = model(graph, cond)
                    loss = criterion(out.flatten(), label.flatten())
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                model.eval()
                v_loss = 0.0
                with torch.no_grad():
                    for graph, cond, label in val_loader:
                        graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                        out = model(graph, cond)
                        v_loss += criterion(out.flatten(), label.flatten()).item()
                v_loss /= len(val_loader)

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    early_stop_count = 0
                    torch.save({'model_state_dict': model.state_dict()}, best_ckpt)
                else:
                    early_stop_count += 1
                    if early_stop_count >= args.patience:
                        break

            # 验证集评估
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            val_preds, val_trues = [], []
            with torch.no_grad():
                for graph, cond, label in val_loader:
                    graph, cond = graph.to(device), cond.to(device)
                    out = model(graph, cond)
                    val_preds.extend(out.flatten().cpu().numpy().tolist())
                    val_trues.extend(label.flatten().cpu().numpy().tolist())

            val_preds = np.array(val_preds)
            val_trues = np.array(val_trues)
            m = compute_metrics(val_trues, val_preds)
            print(f"  [Seed {seed}] Epochs: {epoch:2d} | Best Val Loss: {best_val_loss:.5f} | Val MAE: {m['mae_clip']:.4f}, R2: {m['r2_clip']:.4f}")
            seed_metrics.append({'seed': seed, 'best_val_loss': float(best_val_loss), **m})

        df_sm = pd.DataFrame(seed_metrics)
        df_sm.to_csv(run_dir / "seed_val_metrics.csv", index=False)

        summary_records.append({
            'mode': md,
            'mean_val_mae': df_sm['mae_clip'].mean(),
            'std_val_mae': df_sm['mae_clip'].std(ddof=1),
            'mean_val_r2': df_sm['r2_clip'].mean(),
            'std_val_r2': df_sm['r2_clip'].std(ddof=1),
        })

    df_sum = pd.DataFrame(summary_records)
    summary_csv = out_base / "split_HFC_all_summary.csv"
    df_sum.to_csv(summary_csv, index=False)
    print(f"\n[ALL DONE] Training summary written to {summary_csv}")
    print(df_sum.to_string(index=False))

if __name__ == '__main__':
    main()
