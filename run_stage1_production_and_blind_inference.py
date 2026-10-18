"""
run_stage1_production_and_blind_inference.py
=============================================
Step 4 Stage 1 Execution Pipeline:
1. Train 10 final frozen models (5 Baseline + 5 Henry-Head, Seeds 42-46) on HFC 1,403 grouped split.
2. Standardize condition features strictly using training set statistics.
3. Perform feature-only blind inference on the 599-point external holdout set.
4. Export raw predictions, verify 100% coverage (2,995 rows each), and seal with SHA-256 hashes.
"""

from __future__ import annotations
import os
import sys
import time
import json
import random
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.append(str(ROOT / 'GNN_for_property_prediction'))

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from Model_v6 import IL_GAT_v6
from Dataset_v6 import BASE_FEATURES, IL_set_v6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=30, delta=1e-5):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_single_model(model_type: str, seed: int, train_loader, val_loader, 
                       model_args: dict, device: torch.device, out_ckpt_dir: Path):
    set_seed(seed)
    is_henry = (model_type == 'henry')
    args = dict(model_args)
    args['use_henry_head'] = is_henry

    model = IL_GAT_v6(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=args['epoch'], eta_min=1e-5)
    criterion = nn.HuberLoss(delta=0.05)
    early_stopping = EarlyStopping(patience=args['patience'])

    best_val_loss = float('inf')
    best_epoch = 0
    best_state_dict = None

    print(f"\n[{model_type.upper():8s} | Seed {seed}] Starting training for max {args['epoch']} epochs...")
    t_start = time.time()

    for epoch in range(1, args['epoch'] + 1):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for graph, cond, label in train_loader:
            graph = graph.to(device)
            cond = cond.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(graph, cond)
            loss = criterion(pred.flatten(), label.flatten())
            loss.backward()
            optimizer.step()

            batch_n = label.numel()
            train_loss_sum += loss.item() * batch_n
            train_samples += batch_n

        scheduler.step()

        # Validation phase (sample-weighted)
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for graph, cond, label in val_loader:
                graph = graph.to(device)
                cond = cond.to(device)
                label = label.to(device)

                pred = model(graph, cond)
                batch_n = label.numel()
                val_loss_sum += criterion(pred.flatten(), label.flatten()).item() * batch_n
                val_samples += batch_n

        avg_train = train_loss_sum / train_samples
        avg_val = val_loss_sum / val_samples

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{args['epoch']} | Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f} | Best Val: {best_val_loss:.5f} (ep {best_epoch})")

        early_stopping(avg_val)
        if early_stopping.early_stop:
            print(f"  ⏹ Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch} (Val Loss: {best_val_loss:.5f})")
            break

    duration = time.time() - t_start
    print(f"  Completed in {duration:.1f}s. Saving best checkpoint from epoch {best_epoch}...")

    ckpt_path = out_ckpt_dir / f"{model_type}_seed_{seed}.pth"
    torch.save({
        'model_state_dict': best_state_dict,
        'model_type': model_type,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'duration_sec': round(duration, 2),
        'model_args': args
    }, ckpt_path)

    ckpt_sha = sha256_file(ckpt_path)
    print(f"  [SAVED] {ckpt_path.name} | SHA-256: {ckpt_sha}")

    return {
        'model_type': model_type,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'duration_sec': round(duration, 2),
        'checkpoint_file': ckpt_path.name,
        'sha256': ckpt_sha
    }


def main():
    print("=" * 80)
    print("  STEP 4 STAGE 1: PRODUCTION TRAINING & BLIND EXTERNAL INFERENCE")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Execution Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU Name         : {torch.cuda.get_device_name(0)}")

    out_base = ROOT / 'paper_results' / 'stage4_external_holdout'
    out_base.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_base / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. 验证切分文件与特征输入文件
    split_p = ROOT / 'splits' / 'HFC_1403_grouped_state_split.npz'
    feat_p = out_base / 'X_599_features_only.csv'
    
    assert split_p.exists(), f"Missing split file: {split_p}"
    assert feat_p.exists(), f"Missing feature file: {feat_p}"

    split_sha = sha256_file(split_p)
    feat_sha = sha256_file(feat_p)
    print(f"\n[INTEGRITY AUDIT]")
    print(f"  Split File       : {split_p.name} | SHA-256: {split_sha}")
    print(f"  Feature-Only File: {feat_p.name} | SHA-256: {feat_sha}")

    # 2. 加载训练数据与拟合 Scaler
    print(f"\n[TRAINING DATA & PREPROCESSING]")
    sp = np.load(split_p)
    train_idx = sp['train'].tolist()
    val_idx = sp['val'].tolist()
    print(f"  Grouped Split Indices: Train={len(train_idx)}, Val={len(val_idx)}")

    model_args = {
        'batch_size': 32,
        'lr': 1e-3,
        'epoch': 100,
        'weight_decay': 1e-6,
        'emb_dim': 300,
        'dropout_rate': 0.2,
        'patience': 30,
        'pool': 'global',
        'add_global': True,
        'no_mol_embedding': False,
        'descriptor_mode': 'Mreduced',
        'cond_dim': 12,
        'use_interaction': True,
        'use_sigmoid': False,
        'use_cond_dropout': False,
        'cond_dropout_p': 0.1,
        'use_layernorm': True,
        'use_graph_ln': False,
        'use_adaptive_gate': True,
        'base_feature_names': BASE_FEATURES,
        'gate_init_bias': 2.0,
        'feature_clip': None
    }

    # 载入 1,403 全量训练集并严格只在 train_idx 上拟合 Scaler
    train_dataset = IL_set_v6(path=str(ROOT / 'processed_tri_data_v6'), args=model_args, valid_indices=train_idx)
    val_dataset   = IL_set_v6(path=str(ROOT / 'processed_tri_data_v6'), args=model_args, valid_indices=val_idx)

    # 拟合 scaler 严格基于 train_idx
    train_dataset.fit_scalers(list(range(len(train_dataset))), save_dir=str(out_base))
    scaler_path = out_base / 'scalers.pkl'
    canonical_scaler = out_base / 'canonical_train_scaler.joblib'
    if scaler_path.exists():
        os.replace(scaler_path, canonical_scaler)
    scaler_sha = sha256_file(canonical_scaler)
    print(f"  [SAVED] Scaler fitted strictly on train split: {canonical_scaler.name} | SHA-256: {scaler_sha}")

    # 将拟合好的 scaler 赋予 validation 数据集
    val_dataset.load_scalers(str(canonical_scaler))

    train_loader = DataLoader(train_dataset, batch_size=model_args['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=model_args['batch_size'], shuffle=False)

    # 3. 训练 10 个模型 (5 Baseline + 5 Henry-Head, 成对交替)
    seeds = [42, 43, 44, 45, 46]
    training_manifest = []

    print(f"\n[TRAINING 10 FROZEN PRODUCTION MODELS]")
    for seed in seeds:
        # Baseline
        base_info = train_single_model('baseline', seed, train_loader, val_loader, model_args, device, ckpt_dir)
        training_manifest.append(base_info)

        # Henry-Head
        henry_info = train_single_model('henry', seed, train_loader, val_loader, model_args, device, ckpt_dir)
        training_manifest.append(henry_info)

    # 4. 在 599 纯特征集上执行盲推
    print(f"\n[STAGE 1: FEATURE-ONLY BLIND INFERENCE (599 SAMPLES)]")
    feat_df = pd.read_csv(feat_p)
    assert len(feat_df) == 599, f"Expected 599 rows, got {len(feat_df)}"
    assert 'x1' not in feat_df.columns, "CRITICAL ERROR: x1 label detected in feature file!"

    indices_599 = list(range(3845, 4444))
    test_dataset = IL_set_v6(path=str(ROOT / 'datasets' / 'full_4444_v7'), args=model_args, valid_indices=indices_599)
    test_dataset.load_scalers(str(canonical_scaler))
    assert len(test_dataset) == 599

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 批量推理
    baseline_records = []
    henry_records = []

    for seed in seeds:
        print(f"  Running inference for Seed {seed}...")
        # 1. Baseline
        ckpt_b = torch.load(ckpt_dir / f"baseline_seed_{seed}.pth", map_location=device)
        model_b = IL_GAT_v6({**model_args, 'use_henry_head': False}).to(device)
        model_b.load_state_dict(ckpt_b['model_state_dict'])
        model_b.eval()

        # 2. Henry
        ckpt_h = torch.load(ckpt_dir / f"henry_seed_{seed}.pth", map_location=device)
        model_h = IL_GAT_v6({**model_args, 'use_henry_head': True}).to(device)
        model_h.load_state_dict(ckpt_h['model_state_dict'])
        model_h.eval()

        b_preds = []
        h_preds = []
        with torch.no_grad():
            for graph, cond, _ in test_loader:
                graph = graph.to(device)
                cond = cond.to(device)

                p_b = model_b(graph, cond).flatten().cpu().numpy()
                p_h = model_h(graph, cond).flatten().cpu().numpy()

                b_preds.extend(p_b.tolist())
                h_preds.extend(p_h.tolist())

        assert len(b_preds) == 599
        assert len(h_preds) == 599

        for i in range(599):
            sid = str(feat_df['sample_id'].iloc[i])
            baseline_records.append({
                'sample_id': sid,
                'seed': seed,
                'y_pred_raw': float(b_preds[i])
            })
            henry_records.append({
                'sample_id': sid,
                'seed': seed,
                'y_pred_raw': float(h_preds[i])
            })

    # 5. 导出预测文件并执行全面覆盖与数值检查 (Checklist C)
    df_pred_b = pd.DataFrame(baseline_records)
    df_pred_h = pd.DataFrame(henry_records)

    pred_b_path = out_base / 'predictions_baseline_599_raw.csv'
    pred_h_path = out_base / 'predictions_henry_599_raw.csv'

    df_pred_b.to_csv(pred_b_path, index=False)
    df_pred_h.to_csv(pred_h_path, index=False)

    print(f"\n[PREDICTION COVERAGE & INTEGRITY AUDIT]")
    for name, df_p, p_path in [('Baseline', df_pred_b, pred_b_path), ('Henry-Head', df_pred_h, pred_h_path)]:
        n_rows = len(df_p)
        n_unique_sids = df_p['sample_id'].nunique()
        n_nan = df_p['y_pred_raw'].isna().sum()
        n_inf = np.isinf(df_p['y_pred_raw']).sum()

        print(f"  {name:10s} Total Rows         : {n_rows} (Expected 2,995: 5 seeds x 599 samples)")
        print(f"  {name:10s} Unique Sample IDs  : {n_unique_sids} (Expected 599)")
        print(f"  {name:10s} NaN Count          : {n_nan}")
        print(f"  {name:10s} Inf Count          : {n_inf}")

        assert n_rows == 2995, f"{name} row count mismatch: {n_rows} != 2995"
        assert n_unique_sids == 599, f"{name} unique sample_id mismatch: {n_unique_sids} != 599"
        assert n_nan == 0, f"{name} contains NaN predictions!"
        assert n_inf == 0, f"{name} contains Inf predictions!"

    sha_b = sha256_file(pred_b_path)
    sha_h = sha256_file(pred_h_path)

    print(f"\n[STAGE 1 PREDICTION FILES FROZEN]")
    print(f"  predictions_baseline_599_raw.csv SHA-256: {sha_b}")
    print(f"  predictions_henry_599_raw.csv    SHA-256: {sha_h}")

    # 6. 生成阶段一权威执行 Manifest
    manifest = {
        'protocol': 'Pre-isolated External Confirmatory Evaluation (Step 4 Stage 1)',
        'execution_timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'device': str(device),
        'git_commit': '07bacb26f865d77574a8533d76e7ef74caf34d78',
        'training_split': {
            'file': 'splits/HFC_1403_grouped_state_split.npz',
            'sha256': split_sha,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx)
        },
        'scaler': {
            'file': 'paper_results/stage4_external_holdout/canonical_train_scaler.joblib',
            'sha256': scaler_sha,
            'fitted_on': 'train_split_only (1,262 samples)'
        },
        'input_features': {
            'file': 'paper_results/stage4_external_holdout/X_599_features_only.csv',
            'sha256': feat_sha,
            'samples': 599,
            'label_presence': 'STRICTLY_NONE'
        },
        'checkpoints_trained_10': training_manifest,
        'predictions_frozen': {
            'baseline_raw': {
                'file': 'paper_results/stage4_external_holdout/predictions_baseline_599_raw.csv',
                'sha256': sha_b,
                'total_rows': len(df_pred_b)
            },
            'henry_raw': {
                'file': 'paper_results/stage4_external_holdout/predictions_henry_599_raw.csv',
                'sha256': sha_h,
                'total_rows': len(df_pred_h)
            }
        },
        'status': 'STAGE_1_PREDICTIONS_LOCKED_AWAITING_VERIFICATION'
    }

    manifest_path = out_base / 'STAGE1_EXECUTION_MANIFEST.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n  [SAVED] Stage 1 Manifest written to: {manifest_path.name}")
    print("=" * 80)

    # 7. 自动生成交付给 GPT 独立核验的轻量级预测包 (约 100 KB)
    delivery_zip = ROOT / 'predictions_only_stage1.zip'
    import zipfile
    with zipfile.ZipFile(delivery_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_b_path, arcname=pred_b_path.name)
        zf.write(pred_h_path, arcname=pred_h_path.name)
        zf.write(manifest_path, arcname=manifest_path.name)
        zf.write(canonical_scaler, arcname=canonical_scaler.name)
    print(f"  [DELIVERY PACK READY] {delivery_zip.name} ({delivery_zip.stat().st_size:,} bytes)")

    print("  STAGE 1 SUCCESSFULLY COMPLETED & PERMANENTLY LOCKED!")
    print("=" * 80)


if __name__ == '__main__':
    main()
