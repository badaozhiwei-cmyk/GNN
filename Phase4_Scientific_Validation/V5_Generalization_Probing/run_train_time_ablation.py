"""
run_train_time_ablation.py
===================================================
A rigorously designed Train-Time Feature Ablation script.
Replaces specified chemical features with "Neutral Surrogate Categories" (Mean/Mode imputation)
DURING data loading, then trains a fresh model from scratch.
"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import os
import sys
import pandas as pd
import pathlib as pl

# 保证能导到外层目录的 Model_v5 和 Dataset_v5 (它们在 GNN_for_property_prediction 文件夹下)
current_script_dir = str(pl.Path(__file__).resolve().parent)
root_dir = str(pl.Path(current_script_dir).parent.parent)
sys.path.append(os.path.join(root_dir, 'GNN_for_property_prediction'))

from Dataset_v5 import IL_set_v5
from Model_v5 import IL_GAT_v5
from GAT_Runner_v5 import set_seed, EarlyStopping, Runner

# ============================================================
# Neutral Surrogate Categories (Mean/Mode Imputation)
# ============================================================
# 不修改 Model_v5 的词表，而是使用最无害的现有索引作为占位符
SURROGATE_TOKENS = {
    "atomic_num": 0,    # 0 = Dummy Atom in RDKit (完美的中性占位符)
    "hybridization": 0, # 0 = UNSPECIFIED 
    "aromatic": 0,      # 0 = 非芳香性
    "degree": 0,        # 0 = 无连接
    "charge": 1,        # 1 = 中性电荷 (0 formal charge) -> 最关键的修复！
    "eneg_bucket": 3,   # 3 = 中间桶位 (类似 Mean Imputation)
    "radius_bucket": 3  # 3 = 中间桶位 (类似 Mean Imputation)
}

ABLATION_MASKS = {
    "Full":                  [1, 1, 1, 1, 1, 1, 1],  
    "No_atomic_physics":     [1, 1, 1, 1, 1, 0, 0],  # 移除半径、电负性
    "No_electronic":         [1, 1, 1, 1, 0, 0, 1],  # 移除电荷、电负性
    "No_local_structure":    [1, 0, 0, 0, 1, 1, 1],  # 移除杂化、芳香性、连接度
    "Element_identity_only": [1, 0, 0, 0, 0, 0, 0],  # 仅保留原子种类
    "No_descriptors":        [1, 1, 1, 1, 1, 1, 1],  # 图特征全保留，仅掩码物理描述符
    "Graph_only":            [1, 0, 0, 0, 0, 0, 0]   # (终极剥离) 仅原子种类 + 拓扑 + T/P，无描述符
}

# ============================================================
# Dataset Wrapper
# ============================================================
class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, mode):
        self.base_dataset = base_dataset
        self.mode = mode
        self.mask_array = ABLATION_MASKS[mode]
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        graph, cond, label = self.base_dataset[idx]
        
        # 深拷贝，避免污染原数据集缓存
        graph = graph.clone()
        cond = cond.clone()
        
        # Apply mask
        for i, keep in enumerate(self.mask_array):
            if keep == 0:
                if i == 0: surrogate = SURROGATE_TOKENS["atomic_num"]
                elif i == 1: surrogate = SURROGATE_TOKENS["hybridization"]
                elif i == 2: surrogate = SURROGATE_TOKENS["aromatic"]
                elif i == 3: surrogate = SURROGATE_TOKENS["degree"]
                elif i == 4: surrogate = SURROGATE_TOKENS["charge"]
                elif i == 5: surrogate = SURROGATE_TOKENS["eneg_bucket"]
                elif i == 6: surrogate = SURROGATE_TOKENS["radius_bucket"]
                
                # graph.x shape is [num_nodes, 7]
                graph.x[:, i] = surrogate
                
        # 剥离 Cond 中的物理描述符 (保留索引 0 和 1 也就是 T 和 P)
        if self.mode in ["No_descriptors", "Graph_only"]:
            # StandardScaler 标准化后均值为 0，直接填 0 相当于 Mean Imputation
            cond[2:] = 0.0
                
        return graph, cond, label

# ============================================================
# Main Script
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train-Time Feature Ablation for V5")
    parser.add_argument("--level", type=str, default="L0", help="Split level (L0, L1, L2, L3, L4)")
    parser.add_argument("--mode", type=str, required=True, choices=list(ABLATION_MASKS.keys()), help="Ablation mode")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds (use 1 for quick trend check)")
    parser.add_argument("--epoch", type=int, default=100, help="Max epochs")
    
    cmd_args = parser.parse_args()
    
    Args = {
        'data_path':     os.path.join(root_dir, 'processed_tri_data/'),
        'batch_size':    64,
        'lr':            0.001,
        'epoch':         cmd_args.epoch,       
        'weight_decay':  1e-6,
        'emb_dim':       300,
        'dropout_rate':  0.2,
        'patience':      20,
        'pool':          'global', # 冻结架构参数
        'use_ani_mw':    False,
        'no_mol_embedding': False,
        'add_global':    True
    }

    LEVEL = cmd_args.level
    NUM_SEEDS = cmd_args.seeds
    SEEDS = list(range(NUM_SEEDS))
    MODE = cmd_args.mode
    
    # 存到独立目录，不覆盖原有的 V5 checkpoints
    SAVE_DIR = os.path.join(root_dir, f"checkpoints_v5/ablation/{LEVEL}/{MODE}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🧪 Train-Time Feature Ablation | Target: {LEVEL}")
    print(f"  Mode: {MODE} | Ensemble Seeds: {NUM_SEEDS}")
    print(f"{'='*60}\n")

    # 1. 严格保持与 Full 模型一致的切分
    split_file = os.path.join(root_dir, f"split_{LEVEL}_indices.npz")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"找不到 {split_file}，请确保在根目录下有该切分文件！")
        
    loaded_idx = np.load(split_file)
    train_indices = loaded_idx['train'].tolist()
    val_indices   = loaded_idx['val'].tolist()
    test_indices  = loaded_idx['test'].tolist()
    
    print(f"正在加载基础数据集 (v5)...")
    base_dataset = IL_set_v5(path=Args['data_path'], args=Args)
    # Fit Scaler 并保存在当前 Ablation 的文件夹内
    base_dataset.fit_scalers(train_indices, save_dir=SAVE_DIR)
    
    print(f"应用 Neutral Surrogate Masking 策略 -> Mode: {MODE}")
    if MODE == 'Full':
        # 对于 Full 模式，完全绕过 MaskedDataset，避免 .clone() 改变张量内存布局导致 CUDA 算子出现浮点误差积累
        masked_dataset = base_dataset
    else:
        masked_dataset = MaskedDataset(base_dataset, mode=MODE)

    train_set = torch.utils.data.Subset(masked_dataset, train_indices)
    dev_set   = torch.utils.data.Subset(masked_dataset, val_indices)
    test_set  = torch.utils.data.Subset(masked_dataset, test_indices)

    # 保持与 GAT_Runner_v5 完全一致的 DataLoader 配置，test_loader 在外，train/dev 在循环内
    test_loader  = DataLoader(test_set,  batch_size=Args['batch_size'], shuffle=False)
    
    print(f"  数据集 {LEVEL} 划分 → Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}\n")

    all_preds = []
    test_true = None
    ensemble_results = []

    for seed in SEEDS:
        print(f"\n{'─'*60}")
        print(f"  🚀 重新训练模型 | {MODE} | Seed {seed}")
        print(f"{'─'*60}")
        set_seed(seed)
        
        # 必须在 set_seed 之后初始化 train_loader，保证每个 seed 下 batch 的 shuffle 顺序和原版完全一致！
        train_loader = DataLoader(train_set, batch_size=Args['batch_size'], shuffle=True)
        dev_loader   = DataLoader(dev_set,   batch_size=Args['batch_size'], shuffle=False)
        
        # 使用原版 Runner (完美保证模型架构、优化器、调度器、Loss 全不改变)
        runner = Runner(Args, seed=seed, save_dir=SAVE_DIR)
        
        best_val = runner.train(train_loader, dev_loader)
        
        pred_y, true_y = runner.test(test_loader)
        
        r2 = r2_score(true_y, pred_y)
        mae = mean_absolute_error(true_y, pred_y)
        rmse = np.sqrt(mean_squared_error(true_y, pred_y))
        
        ensemble_results.append((r2, mae, rmse))
        all_preds.append(pred_y)
        if test_true is None:
            test_true = true_y

    print(f"\n{'='*60}")
    print(f" 🎉 Ablation {MODE} ({NUM_SEEDS} Seeds) 总结报告:")
    for i, res in enumerate(ensemble_results):
        print(f"  Seed {i} - R2: {res[0]:.4f} | MAE: {res[1]:.4f} | RMSE: {res[2]:.4f}")
        
    if NUM_SEEDS > 1:
        avg_preds = np.mean(all_preds, axis=0)
        e_r2 = r2_score(test_true, avg_preds)
        e_mae = mean_absolute_error(test_true, avg_preds)
        e_rmse = np.sqrt(mean_squared_error(test_true, avg_preds))
        print(f"\n  🏆 Ensemble Performance -> R2: {e_r2:.4f} | MAE: {e_mae:.4f} | RMSE: {e_rmse:.4f}")
        
    print(f"{'='*60}\n")
