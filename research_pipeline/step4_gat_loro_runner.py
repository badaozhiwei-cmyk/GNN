"""
step4_gat_loro_runner.py
========================
Runs GAT_v5 model on Leave-One-Refrigerant-Out (LORO) splits.
Evaluates GNN's zero-shot generalization capabilities when a specific
refrigerant is held out from the training set.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import pathlib as pl

ROOT = str(pl.Path(__file__).resolve().parent.parent)
os.chdir(ROOT)
sys.path.append(os.path.join(ROOT, 'GNN_for_property_prediction'))

from Dataset_v5 import IL_Dataset_v5
from Model_v5 import IL_GAT_v5

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
    
    print(f"  LORO Split → Train: {len(train_indices)}, Val: {len(val_indices)}, Test ({target_ref}): {len(test_indices)}")
    
    # Save temporary npz for dataset loader compatibility
    split_filename = f"split_LORO_{target_ref}_indices.npz"
    np.savez(split_filename, train=train_indices, val=val_indices, test=test_indices)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Initialize Dataset v5 with LORO split
    dataset_args = {'use_ani_mw': False}
    dataset = IL_Dataset_v5(level=f"LORO_{target_ref}", args=dataset_args)
    
    train_loader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset.val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset.test_dataset,  batch_size=batch_size, shuffle=False)
    
    model_args = {
        'emb_dim': 300,
        'pool': 'global',
        'use_ani_mw': False,
        'dropout_rate': 0.2
    }
    
    r2_list, mae_list = [], []
    
    for seed in range(seeds):
        torch.manual_seed(seed + 42)
        model = IL_GAT_v5(model_args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            for graph, cond, label in train_loader:
                graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                optimizer.zero_grad()
                out = model(graph, cond).squeeze()
                loss = criterion(out, label.squeeze())
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for graph, cond, label in val_loader:
                    graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                    out = model(graph, cond).squeeze()
                    val_loss += criterion(out, label.squeeze()).item() * len(label)
            val_loss /= len(val_indices)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                
        # Test evaluation
        model.load_state_dict(best_model_state)
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for graph, cond, label in test_loader:
                graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                out = model(graph, cond).squeeze()
                preds.extend(out.cpu().numpy().tolist())
                targets.extend(label.cpu().numpy().tolist())
                
        preds = np.clip(preds, 0.0, 1.0)
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(targets, preds)
        mae = mean_absolute_error(targets, preds)
        r2_list.append(r2)
        mae_list.append(mae)
        print(f"  Seed {seed} → LORO ({target_ref}) R²: {r2:.4f}, MAE: {mae:.4f}")
        
    print(f"\n  ✅ LORO ({target_ref}) Final Mean R²: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
    
    # Cleanup temp npz
    if os.path.exists(split_filename):
        os.remove(split_filename)
        
    return np.mean(r2_list), np.mean(mae_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='R32', help='Held-out refrigerant')
    parser.add_argument('--seeds', type=int, default=1, help='Number of seeds')
    parser.add_argument('--epochs', type=int, default=80, help='Epochs per seed')
    args = parser.parse_args()
    
    run_gat_loro(args.ref, seeds=args.seeds, epochs=args.epochs)
