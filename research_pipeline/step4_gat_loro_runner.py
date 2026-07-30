"""
step4_gat_loro_runner.py
========================
Runs GAT_v5 model on Leave-One-Refrigerant-Out (LORO) splits.
Evaluates GNN's zero-shot generalization capabilities when a specific
refrigerant is held out from the training set.

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

ROOT = str(pl.Path(__file__).resolve().parent.parent)
os.chdir(ROOT)
sys.path.append(os.path.join(ROOT, 'GNN_for_property_prediction'))

from Dataset_v5 import IL_set_v5
from Model_v5 import IL_GAT_v5

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

# ============================================================
# LORO Runner
# ============================================================
def run_gat_loro(target_ref: str, seeds: int = 1, epochs: int = 100, batch_size: int = 64, lr: float = 1e-3):
    print(f"\n============================================================")
    print(f"  GAT_v5 LORO Benchmark | Held-out Refrigerant: {target_ref}")
    print(f"============================================================")
    
    df = pd.read_csv('index_with_anion.csv')
    test_indices = df[df['refrigerant'] == target_ref].index.tolist()
    train_val_indices = df[df['refrigerant'] != target_ref].index.tolist()
    
    np.random.seed(42)
    np.random.shuffle(train_val_indices)
    
    val_size = int(len(train_val_indices) * 0.1)
    val_indices = train_val_indices[:val_size]
    train_indices = train_val_indices[val_size:]
    
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
    
    save_dir = f"checkpoints_v5/LORO_{target_ref}"
    whole_set.fit_scalers(train_indices, save_dir=save_dir)
    
    train_set = torch.utils.data.Subset(whole_set, train_indices)
    val_set   = torch.utils.data.Subset(whole_set, val_indices)
    test_set  = torch.utils.data.Subset(whole_set, test_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    
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
    
    for seed in range(seeds):
        set_seed(seed + 42)
        
        model = IL_GAT_v5(model_args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.HuberLoss(delta=1.0)
        early_stopping = EarlyStopping(patience=20)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        print(f"\n  [Seed {seed}] Training GAT_v5 ({epochs} epochs, patience=20)...")
        for epoch in range(1, epochs + 1):
            # --- Train ---
            model.train()
            train_loss = 0.0
            
            bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False,
                       desc=f"Epoch {epoch:>3d}")
            for graph, cond, label in train_loader:
                graph, cond, label = graph.to(device), cond.to(device), label.to(device)
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
                    graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                    out = model(graph, cond)
                    val_loss += criterion(out.flatten(), label.flatten()).item()
            
            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                # [CRITICAL FIX] deepcopy prevents weight pollution from subsequent epochs
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
        preds, targets = [], []
        with torch.no_grad():
            for graph, cond, label in test_loader:
                graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                out = model(graph, cond)
                pred_vals = np.clip(out.flatten().cpu().numpy(), 0.0, 1.0)
                preds.extend(pred_vals.tolist())
                targets.extend(label.flatten().cpu().numpy().tolist())
                
        r2  = r2_score(targets, preds)
        mae = mean_absolute_error(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        r2_list.append(r2)
        mae_list.append(mae)
        print(f"  Seed {seed} -> LORO ({target_ref}) R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
    mean_r2 = np.mean(r2_list)
    mean_mae = np.mean(mae_list)
    print(f"\n  🎉 LORO ({target_ref}) Final: R2 = {mean_r2:.4f} +/- {np.std(r2_list):.4f}")
    
    # [便利增强] 自动将结果追加写入 gat_loro_results.csv
    res_path = 'gat_loro_results.csv'
    res_row = pd.DataFrame([{
        'refrigerant': target_ref,
        'r2': mean_r2,
        'mae': mean_mae,
        'n_test': len(test_indices)
    }])
    if not os.path.exists(res_path):
        res_row.to_csv(res_path, index=False)
    else:
        # 覆盖相同制冷剂的旧记录或追加新记录
        existing_df = pd.read_csv(res_path)
        existing_df = existing_df[existing_df['refrigerant'] != target_ref]
        combined_df = pd.concat([existing_df, res_row], ignore_index=True)
        combined_df.to_csv(res_path, index=False)
    print(f"  📊 结果已自动更新至 gat_loro_results.csv")
    
    return mean_r2, mean_mae

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='R32', help='Held-out refrigerant')
    parser.add_argument('--seeds', type=int, default=1, help='Number of seeds')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs per seed')
    args = parser.parse_args()
    
    run_gat_loro(args.ref, seeds=args.seeds, epochs=args.epochs)
