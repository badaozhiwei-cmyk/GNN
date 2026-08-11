"""
step4_gat_loro_runner.py
========================
Runs GAT_v5 model on Leave-One-Refrigerant-Out (LORO) splits.
Evaluates GNN's zero-shot generalization capabilities when a specific
refrigerant is held out from the training set.

Supported experimental domains:
  --split-mode full      all remaining refrigerant families are training data
  --split-mode hfc-only  saturated-HFC-only nested LORO (--val-id 1/2/3)

[v2] Fixed critical bugs:
  1. Use copy.deepcopy for best_model_state to avoid weight pollution
  2. Add EarlyStopping (patience=20) matching GAT_Runner_v5
  3. Add tqdm progress bar for each epoch
  4. Align all hyperparameters to GAT_Runner_v5
"""

import argparse
import copy
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pathlib as pl
import json
from datetime import datetime, timezone
from rdkit import Chem

ROOT = str(pl.Path(__file__).resolve().parent.parent)
os.chdir(ROOT)
sys.path.append(os.path.join(ROOT, 'GNN_for_property_prediction'))

from Dataset_v5 import IL_set_v5
from Model_v5 import IL_GAT_v5, IL_GCN_v5

# ============================================================
# EarlyStopping (aligned with GAT_Runner_v5.py)
# ============================================================
class EarlyStopping:
    def __init__(self, patience=20, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _scalar_text(value) -> str:
    """Read a scalar string stored in an npz without enabling pickle."""
    return str(np.asarray(value).item())


def _is_saturated_hfc(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    allowed_elements = {'C', 'H', 'F'}
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if 'F' not in symbols or any(symbol not in allowed_elements for symbol in symbols):
        return False
    return not any(
        bond.GetBondTypeAsDouble() > 1.0
        for bond in mol.GetBonds()
    )


def load_loro_split(df, target_ref: str, split_mode: str, val_id: int):
    """Load and scientifically audit either full-space or HFC-only LORO."""
    if split_mode == 'full':
        split_path = pl.Path(ROOT) / 'splits_loro' / f'split_L4_{target_ref}.npz'
        if not split_path.exists():
            raise FileNotFoundError(
                f"Audited full LORO split not found: {split_path}. Run "
                "research_pipeline/step1_generalization_ladder_v2.py first."
            )
        split = np.load(split_path, allow_pickle=False)
        required = {'train', 'val', 'test', 'metadata_json'}
        if not required.issubset(split.files):
            raise KeyError(f"Full LORO split has invalid schema: {split.files}")
        train_indices = split['train'].astype(int).tolist()
        val_indices = split['val'].astype(int).tolist()
        test_indices = split['test'].astype(int).tolist()
        split_metadata = json.loads(_scalar_text(split['metadata_json']))
        experiment_id = 'full'
    else:
        if val_id not in {1, 2, 3}:
            raise ValueError("--val-id must be 1, 2, or 3 for hfc-only mode")
        split_path = (
            pl.Path(ROOT) / 'Phase4_Scientific_Validation' / 'HFC_LORO_Splits'
            / f'split_hfc_loro_{target_ref}_val{val_id}.npz'
        )
        if not split_path.exists():
            raise FileNotFoundError(
                f"HFC-only LORO split not found: {split_path}. Run "
                "Phase4_Scientific_Validation/generate_hfc_loro_splits.py first."
            )
        split = np.load(split_path, allow_pickle=False)
        required = {'train_idx', 'val_idx', 'test_idx', 'test_hfc', 'val_hfc', 'train_hfcs'}
        if not required.issubset(split.files):
            raise KeyError(f"HFC-only split has invalid schema: {split.files}")
        train_indices = split['train_idx'].astype(int).tolist()
        val_indices = split['val_idx'].astype(int).tolist()
        test_indices = split['test_idx'].astype(int).tolist()
        train_hfcs = [str(value) for value in split['train_hfcs'].tolist()]
        val_hfc = _scalar_text(split['val_hfc'])
        test_hfc = _scalar_text(split['test_hfc'])
        if test_hfc != target_ref:
            raise ValueError(f"Split target is {test_hfc}, requested target is {target_ref}")
        split_metadata = {
            'protocol': f'HFC-only leave-one-refrigerant-out: {target_ref}',
            'split_mode': 'hfc-only',
            'val_id': val_id,
            'held_out_refrigerant': test_hfc,
            'validation_refrigerant': val_hfc,
            'training_refrigerants': train_hfcs,
        }
        experiment_id = f'hfc_only_val{val_id}'

    partition_sets = [set(train_indices), set(val_indices), set(test_indices)]
    if partition_sets[0] & partition_sets[1] or partition_sets[0] & partition_sets[2] or partition_sets[1] & partition_sets[2]:
        raise AssertionError("Train/validation/test indices overlap")
    if target_ref in set(df.loc[train_indices, 'refrigerant']):
        raise AssertionError(f"Held-out refrigerant {target_ref} leaked into training")
    if set(df.loc[test_indices, 'refrigerant']) != {target_ref}:
        raise AssertionError("Test partition contains refrigerants other than the requested target")

    if split_mode == 'hfc-only':
        actual_train = set(df.loc[train_indices, 'refrigerant'])
        actual_val = set(df.loc[val_indices, 'refrigerant'])
        if actual_train != set(split_metadata['training_refrigerants']):
            raise AssertionError("HFC-only training refrigerants disagree with split metadata")
        if actual_val != {split_metadata['validation_refrigerant']}:
            raise AssertionError("HFC-only validation refrigerant disagrees with split metadata")
        all_hfcs = actual_train | actual_val | {target_ref}
        ref_smiles = df.drop_duplicates('refrigerant').set_index('refrigerant')['refri_smiles']
        invalid = [ref for ref in sorted(all_hfcs) if not _is_saturated_hfc(ref_smiles.loc[ref])]
        if invalid:
            raise AssertionError(f"Non-saturated-HFC chemistry found in HFC-only split: {invalid}")

    split_metadata.update({
        'split_mode': split_mode,
        'val_id': val_id if split_mode == 'hfc-only' else None,
        'n_train': len(train_indices),
        'n_val': len(val_indices),
        'n_test': len(test_indices),
    })
    return train_indices, val_indices, test_indices, split_metadata, split_path, experiment_id

# ============================================================
# LORO Runner
# ============================================================
def run_gat_loro(target_ref: str, seeds: int = 1, epochs: int = 100, batch_size: int = 64, lr: float = 1e-3, model_type: str = 'gat', num_workers: int = 0, split_mode: str = 'full', val_id: int = 1):
    model_name = 'GAT_v5' if model_type == 'gat' else 'GCN_v5'
    print(f"\n============================================================")
    print(f"  {model_name} LORO Benchmark | Held-out Refrigerant: {target_ref}")
    print(f"  Split Mode: {split_mode} | Validation ID: {val_id if split_mode == 'hfc-only' else 'N/A'}")
    print(f"  Batch Size: {batch_size} | Num Workers: {num_workers}")
    print(f"============================================================")
    
    df = pd.read_csv('index_with_anion.csv').reset_index(drop=True)
    train_indices, val_indices, test_indices, split_metadata, split_path, experiment_id = load_loro_split(
        df, target_ref, split_mode, val_id
    )
    
    print(f"  LORO Split -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test ({target_ref}): {len(test_indices)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Dataset (aligned with GAT_Runner_v5.py)
    dataset_args = {
        'add_global': True,
        'use_ani_mw': False,
        'no_mol_embedding': False
    }
    data_path = os.path.join(ROOT, 'processed_tri_data/')
    whole_set = IL_set_v5(path=data_path, args=dataset_args)
    
    save_dir = os.path.join(
        ROOT, 'checkpoints_v5', 'loro', experiment_id, target_ref, model_type
    )
    os.makedirs(save_dir, exist_ok=True)
    whole_set.fit_scalers(train_indices, save_dir=save_dir)
    
    train_set = torch.utils.data.Subset(whole_set, train_indices)
    val_set   = torch.utils.data.Subset(whole_set, val_indices)
    test_set  = torch.utils.data.Subset(whole_set, test_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Model args (aligned with GAT_Runner_v5.py)
    model_args = {
        'emb_dim': 300,
        'pool': 'global',
        'use_ani_mw': False,
        'dropout_rate': 0.2,
        'no_mol_embedding': False,
        'add_global': True,
    }
    
    r2_list, mae_list = [], []
    raw_r2_list, raw_mae_list = [], []
    prediction_rows = []
    
    for seed_id in range(seeds):
        random_seed = seed_id + 42
        set_seed(random_seed)
        
        ModelClass = IL_GAT_v5 if model_type == 'gat' else IL_GCN_v5
        model = ModelClass(model_args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.HuberLoss(delta=1.0)
        early_stopping = EarlyStopping(patience=20)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        print(f"\n  [Seed {random_seed}] Training {model_name} ({epochs} epochs, patience=20)...")
        for epoch in range(1, epochs + 1):
            # --- Train ---
            model.train()
            train_loss = 0.0
            
            bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False,
                       desc=f"Epoch {epoch:>3d}")
            for graph, cond, label in train_loader:
                graph, cond, label = graph.to(device, non_blocking=True), cond.to(device, non_blocking=True), label.to(device, non_blocking=True)
                optimizer.zero_grad()
                out = model(graph, cond)
                loss = criterion(out.flatten(), label.flatten())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                bar.update()
            bar.close()
            
            scheduler.step()
            
            # --- Validate ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for graph, cond, label in val_loader:
                    graph, cond, label = graph.to(device, non_blocking=True), cond.to(device, non_blocking=True), label.to(device, non_blocking=True)
                    out = model(graph, cond)
                    val_loss += criterion(out.flatten(), label.flatten()).item()
            
            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_state = copy.deepcopy(model.state_dict())
            
            if epoch % 10 == 0:
                print(f"    Epoch {epoch:>3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
            
            early_stopping(avg_val)
            if early_stopping.early_stop:
                print(f"    Early stopping at epoch {epoch}")
                break
                
        # --- Test ---
        model.load_state_dict(best_model_state)
        model.eval()
        raw_preds, targets = [], []
        with torch.no_grad():
            for graph, cond, label in test_loader:
                graph, cond, label = graph.to(device, non_blocking=True), cond.to(device, non_blocking=True), label.to(device, non_blocking=True)
                out = model(graph, cond)
                pred_vals = out.flatten().cpu().numpy()
                raw_preds.extend(pred_vals.tolist())
                targets.extend(label.flatten().cpu().numpy().tolist())
        preds = np.clip(np.asarray(raw_preds), 0.0, 1.0)
        targets = np.asarray(targets)
        r2  = r2_score(targets, preds)
        mae = mean_absolute_error(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        raw_r2 = r2_score(targets, raw_preds)
        raw_mae = mean_absolute_error(targets, raw_preds)
        raw_rmse = np.sqrt(mean_squared_error(targets, raw_preds))
        r2_list.append(r2)
        mae_list.append(mae)
        raw_r2_list.append(raw_r2)
        raw_mae_list.append(raw_mae)
        print(
            f"  Seed {random_seed} -> LORO ({target_ref}) "
            f"raw R2: {raw_r2:.4f}, clipped R2: {r2:.4f}, "
            f"raw MAE: {raw_mae:.4f}, clipped MAE: {mae:.4f}"
        )

        for sample_idx, target, raw_pred, clipped_pred in zip(
            test_indices, targets, raw_preds, preds
        ):
            prediction_rows.append({
                'model': model_name,
                'refrigerant': target_ref,
                'split_mode': split_mode,
                'val_id': val_id if split_mode == 'hfc-only' else None,
                'seed_id': seed_id,
                'random_seed': random_seed,
                'sample_index': sample_idx,
                'y_true': float(target),
                'y_pred_raw': float(raw_pred),
                'y_pred_clipped': float(clipped_pred),
            })
        
        # Save best checkpoint for later embedding analysis
        ckpt_path = os.path.join(save_dir, f'best_model_seed{random_seed}.pt')
        torch.save(best_model_state, ckpt_path)
        
    mean_r2 = np.mean(r2_list)
    std_r2 = np.std(r2_list)
    mean_mae = np.mean(mae_list)
    mean_raw_r2 = np.mean(raw_r2_list)
    mean_raw_mae = np.mean(raw_mae_list)
    print(f"\n  [Result] LORO ({target_ref}) {model_name}: R2 = {mean_r2:.4f} +/- {std_r2:.4f}")
    
    # Auto-save results to CSV
    results_dir = os.path.join(ROOT, 'results_v5', 'loro', experiment_id)
    os.makedirs(results_dir, exist_ok=True)
    res_path = os.path.join(results_dir, 'loro_gnn_results.csv')
    res_row = pd.DataFrame([{
        'model': model_name,
        'refrigerant': target_ref,
        'split_mode': split_mode,
        'val_id': val_id if split_mode == 'hfc-only' else None,
        'experiment_id': experiment_id,
        'r2_mean': mean_r2,
        'r2_std': std_r2,
        'mae_mean': mean_mae,
        'raw_r2_mean': mean_raw_r2,
        'raw_r2_std': np.std(raw_r2_list),
        'raw_mae_mean': mean_raw_mae,
        'n_seeds': seeds,
        'n_test': len(test_indices),
        'split_file': os.path.relpath(split_path, ROOT),
        'split_protocol': split_metadata['protocol'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }])
    if not os.path.exists(res_path):
        res_row.to_csv(res_path, index=False)
    else:
        existing_df = pd.read_csv(res_path)
        existing_df = existing_df[~(
            (existing_df['refrigerant'] == target_ref)
            & (existing_df['model'] == model_name)
            & (existing_df['split_mode'] == split_mode)
            & (existing_df['val_id'].fillna(-1) == (val_id if split_mode == 'hfc-only' else -1))
        )]
        combined_df = pd.concat([existing_df, res_row], ignore_index=True)
        combined_df.to_csv(res_path, index=False)
    print(f"  [Saved] Results updated in {res_path}")

    # Auto-save raw seed-level details to CSV (for boxplot / Supplementary)
    seed_path = os.path.join(results_dir, 'loro_gnn_seed_details.csv')
    seed_rows = []
    for seed_id, (r2_val, mae_val, raw_r2_val, raw_mae_val) in enumerate(
        zip(r2_list, mae_list, raw_r2_list, raw_mae_list)
    ):
        seed_rows.append({
            'model': model_name,
            'refrigerant': target_ref,
            'split_mode': split_mode,
            'val_id': val_id if split_mode == 'hfc-only' else None,
            'experiment_id': experiment_id,
            'seed_id': seed_id,
            'random_seed': seed_id + 42,
            'r2': r2_val,
            'mae': mae_val,
            'raw_r2': raw_r2_val,
            'raw_mae': raw_mae_val,
        })
    seed_df = pd.DataFrame(seed_rows)
    if not os.path.exists(seed_path):
        seed_df.to_csv(seed_path, index=False)
    else:
        existing_seed_df = pd.read_csv(seed_path)
        existing_seed_df = existing_seed_df[~(
            (existing_seed_df['refrigerant'] == target_ref)
            & (existing_seed_df['model'] == model_name)
            & (existing_seed_df['split_mode'] == split_mode)
            & (existing_seed_df['val_id'].fillna(-1) == (val_id if split_mode == 'hfc-only' else -1))
        )]
        combined_seed_df = pd.concat([existing_seed_df, seed_df], ignore_index=True)
        combined_seed_df.to_csv(seed_path, index=False)
    print(f"  [Saved] Seed-level details updated in {seed_path}")

    prediction_dir = os.path.join(results_dir, 'predictions')
    os.makedirs(prediction_dir, exist_ok=True)
    prediction_path = os.path.join(prediction_dir, f'{target_ref}_{model_type}_predictions.csv')
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)

    run_manifest = {
        'model': model_name,
        'held_out_refrigerant': target_ref,
        'split_mode': split_mode,
        'val_id': val_id if split_mode == 'hfc-only' else None,
        'experiment_id': experiment_id,
        'split_file': os.path.relpath(split_path, ROOT),
        'split_metadata': split_metadata,
        'random_seeds': list(range(42, 42 + seeds)),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'dataset_args': dataset_args,
        'model_args': model_args,
        'metrics_primary': 'raw',
        'prediction_file': os.path.relpath(prediction_path, ROOT),
    }
    manifest_path = os.path.join(prediction_dir, f'{target_ref}_{model_type}_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(run_manifest, handle, indent=2, ensure_ascii=False)
    print(f"  [Saved] Predictions: {prediction_path}")
    print(f"  [Saved] Manifest: {manifest_path}")
    
    return mean_r2, mean_mae

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='R32', help='Held-out refrigerant')
    parser.add_argument('--seeds', type=int, default=1, help='Number of seeds')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs per seed')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers')
    parser.add_argument('--model', type=str, default='gat', choices=['gat', 'gcn'],
                        help='Model type: gat (GAT_v5) or gcn (GCN_v5 ablation)')
    parser.add_argument(
        '--split-mode', default='full', choices=['full', 'hfc-only'],
        help='full: all-refrigerant L4 LORO; hfc-only: saturated-HFC nested LORO'
    )
    parser.add_argument(
        '--val-id', type=int, default=1, choices=[1, 2, 3],
        help='Nested validation split ID for --split-mode hfc-only'
    )
    args = parser.parse_args()
    
    run_gat_loro(
        args.ref,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model,
        num_workers=args.num_workers,
        split_mode=args.split_mode,
        val_id=args.val_id,
    )
