"""
step5_loro_baselines.py
========================
Evaluates Baseline Models (RandomForest, XGBoost, Descriptor-MLP)
on identical Leave-One-Refrigerant-Out (LORO) splits.
Provides parameter-matched deep learning and strong boosting baselines
to ensure rigorous comparison with GAT_v5.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pathlib as pl
import torch
import torch.nn as nn

ROOT = str(pl.Path(__file__).resolve().parent.parent)
os.chdir(ROOT)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Descriptor MLP Architecture
class DescriptorMLP(nn.Module):
    def __init__(self, in_dim):
        super(DescriptorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

def run_baselines_loro(target_ref: str):
    print(f"\n============================================================")
    print(f"  Baseline LORO Benchmark | Held-out Refrigerant: {target_ref}")
    print(f"============================================================")
    
    df = pd.read_csv('index_with_anion.csv')
    test_mask = (df['refrigerant'] == target_ref)
    train_mask = ~test_mask
    
    test_indices = df[test_mask].index.tolist()
    train_indices = df[train_mask].index.tolist()
    
    print(f"  LORO Split -> Train: {len(train_indices)}, Test ({target_ref}): {len(test_indices)}")
    
    # Load physical descriptors from numpy data file
    # data[i] structure: [0-2] mol graphs, [3] T, [4] P, [5] ref_charge,
    #                     [6] ref_logp, [7] ani_mw, [8] cat_charge, [9] cat_tpsa
    raw_data = np.load(os.path.join(ROOT, 'processed_tri_data', 'data.npy'), allow_pickle=True)
    labels = np.load(os.path.join(ROOT, 'processed_tri_data', 'label.npy'), allow_pickle=True).flatten()
    
    # Extract 6 scalar features: T, P, ref_charge, ref_logp, cat_charge, cat_tpsa (exclude ani_mw)
    feature_names = ['T', 'P', 'ref_charge', 'ref_logp', 'cat_charge', 'cat_tpsa']
    feature_indices = [3, 4, 5, 6, 8, 9]  # skip index 7 (ani_mw) to match GAT_v5 default
    
    all_features = np.array([[float(raw_data[i][j]) for j in feature_indices] for i in range(len(raw_data))])
    all_labels = labels.astype(np.float64)
    
    X_train = all_features[train_indices]
    y_train = all_labels[train_indices]
    X_test  = all_features[test_indices]
    y_test  = all_labels[test_indices]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    results = {}
    
    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_preds = np.clip(rf.predict(X_test_scaled), 0.0, 1.0)
    rf_r2 = r2_score(y_test, rf_preds)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    results['RF'] = (rf_r2, rf_mae)
    print(f"  [RF] Random Forest  R2: {rf_r2:.4f}, MAE: {rf_mae:.4f}")
    
    # 2. XGBoost
    if HAS_XGB:
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train_scaled, y_train)
        xgb_preds = np.clip(xgb_model.predict(X_test_scaled), 0.0, 1.0)
        xgb_r2 = r2_score(y_test, xgb_preds)
        xgb_mae = mean_absolute_error(y_test, xgb_preds)
        results['XGBoost'] = (xgb_r2, xgb_mae)
        print(f"  [XGB] XGBoost      R2: {xgb_r2:.4f}, MAE: {xgb_mae:.4f}")

        
    # 3. Descriptor MLP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp = DescriptorMLP(X_train_scaled.shape[1]).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.HuberLoss()
    
    X_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_te_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    mlp.train()
    for epoch in range(150):
        optimizer.zero_grad()
        out = mlp(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()
        
    mlp.eval()
    with torch.no_grad():
        mlp_preds = np.clip(mlp(X_te_t).cpu().numpy().flatten(), 0.0, 1.0)
    mlp_r2 = r2_score(y_test, mlp_preds)
    mlp_mae = mean_absolute_error(y_test, mlp_preds)
    results['MLP'] = (mlp_r2, mlp_mae)
    print(f"  [MLP] Descriptor MLP R2: {mlp_r2:.4f}, MAE: {mlp_mae:.4f}")
    
    # Save to CSV
    res_path = 'loro_baselines_results.csv'
    res_rows = []
    for model_name, (r2_val, mae_val) in results.items():
        res_rows.append({
            'refrigerant': target_ref,
            'model': model_name,
            'r2': r2_val,
            'mae': mae_val,
            'n_test': len(test_indices)
        })
    res_df = pd.DataFrame(res_rows)
    
    if not os.path.exists(res_path):
        res_df.to_csv(res_path, index=False)
    else:
        existing = pd.read_csv(res_path)
        existing = existing[~((existing['refrigerant'] == target_ref) & (existing['model'].isin(results.keys())))]
        combined = pd.concat([existing, res_df], ignore_index=True)
        combined.to_csv(res_path, index=False)
        
    print(f"  [Saved] Baseline results updated in {res_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='R32')
    args = parser.parse_args()
    
    run_baselines_loro(args.ref)
