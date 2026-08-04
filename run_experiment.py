import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import pathlib as pl

# 确保能找到子模块
current_script_dir = str(pl.Path(__file__).resolve().parent)
sys.path.append(os.path.join(current_script_dir, 'GNN_for_property_prediction'))

from Dataset_v5 import IL_set_v5
from GAT_Runner_v5 import Runner, set_seed

FAMILY_MAP = {
    'R32': 'HFC', 'R152a': 'HFC', 'R134a': 'HFC', 'R125': 'HFC', 'R143a': 'HFC', 
    'R23': 'HFC', 'R41': 'HFC', 'R161': 'HFC', 'R134': 'HFC', 'R227ea': 'HFC', 
    'R236fa': 'HFC', 'R245fa': 'HFC', 'R365mfc': 'HFC', 'R236ea': 'HFC', 
    'R116': 'FC', 'R14': 'FC', 'R218': 'FC', 
    'R1234yf': 'HFO', 'R1234ze(E)': 'HFO', 'R1233zd(E)': 'HFO', 'R1243zf': 'HFO', 
    'R22': 'HCFC', 'R124': 'HCFC', 'R123': 'HCFC', 'R142b': 'HCFC', 'R141b': 'HCFC', 
    'R22B1': 'HBFC', 'R11': 'CFC', 'R12': 'CFC', 'R13': 'CFC'
}

def generate_dataset_stats(df, ref_col='refrigerant'):
    stats = []
    # Full dataset
    stats.append({'Dataset': 'Full', 'Refrigerants': df[ref_col].nunique(), 'Samples': len(df)})
    
    # By family
    for fam in df['family'].dropna().unique():
        fam_df = df[df['family'] == fam]
        stats.append({'Dataset': fam, 'Refrigerants': fam_df[ref_col].nunique(), 'Samples': len(fam_df)})
        
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv('dataset_statistics.csv', index=False)
    print("✅ Generated dataset_statistics.csv")

def main():
    parser = argparse.ArgumentParser(description="Unified ChemEng GNN Experiment Runner")
    parser.add_argument("--family", type=str, required=True, choices=['HFC', 'HFO', 'ALL'], help="Chemical family to filter")
    parser.add_argument("--mode", type=str, required=True, choices=['random', 'loro', 'family_loro'], help="Split mode")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--epoch", type=int, default=150, help="Max epochs")
    
    args = parser.parse_args()
    
    # 1. Load and prepare data
    df = pd.read_csv('index_with_anion.csv')
    ref_col = 'refrigerant' if 'refrigerant' in df.columns else 'Refrigerant'
    
    if 'family' not in df.columns:
        df['family'] = df[ref_col].map(FAMILY_MAP)
        
    generate_dataset_stats(df, ref_col)
    
    # Filter by family
    if args.family != 'ALL':
        df = df[df['family'] == args.family]
        if len(df) == 0:
            raise ValueError(f"No samples found for family {args.family}")
    
    print(f"\n[{args.family}] Total samples after filtering: {len(df)}")
    
    # Model Arguments
    model_args = {
        'data_path': os.path.join(current_script_dir, 'processed_tri_data/'),
        'batch_size': 64,
        'lr': 0.001,
        'epoch': args.epoch,       
        'weight_decay': 1e-6,
        'emb_dim': 300,
        'dropout_rate': 0.2,
        'patience': 25,
        'pool': 'global',
        'use_ani_mw': True,
        'no_mol_embedding': False,
        'add_global': True
    }
    
    print("Loading Graph Dataset...")
    Whole_set = IL_set_v5(path=model_args['data_path'], args=model_args)
    
    # CRITICAL: Verify perfect alignment between CSV and PyG dataset
    assert len(pd.read_csv('index_with_anion.csv')) == len(Whole_set), f"Dataset length mismatch! CSV: {len(df)} vs PyG: {len(Whole_set)}"
    
    out_dir = f"results/{args.family}_{args.mode}"
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Determine splits
    splits_to_run = [] # List of tuples: (split_name, train_idx, val_idx, test_idx)
    
    if args.mode == 'random':
        train_val_idx, test_idx = train_test_split(df.index.values, test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1, random_state=42)
        splits_to_run.append(('random_split', train_idx.tolist(), val_idx.tolist(), test_idx.tolist()))
        
    elif args.mode in ['loro', 'family_loro']:
        unique_refs = df[ref_col].unique()
        for ref in unique_refs:
            test_idx = df[df[ref_col] == ref].index.values
            train_val_idx = df[df[ref_col] != ref].index.values
            if len(train_val_idx) == 0 or len(test_idx) == 0:
                continue
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1, random_state=42)
            splits_to_run.append((f'loro_{ref}', train_idx.tolist(), val_idx.tolist(), test_idx.tolist()))

    # 3. Execution Loop
    summary_results = []
    
    for split_name, train_idx, val_idx, test_idx in splits_to_run:
        print(f"\n{'='*60}")
        print(f"Running Split: {split_name} | Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        print(f"{'='*60}")
        
        train_set = torch.utils.data.Subset(Whole_set, train_idx)
        val_set   = torch.utils.data.Subset(Whole_set, val_idx)
        test_set  = torch.utils.data.Subset(Whole_set, test_idx)
        
        # Fit Scalers ONCE per split using training data
        Whole_set.fit_scalers(train_idx, save_dir=out_dir)
        
        test_loader = DataLoader(test_set, batch_size=model_args['batch_size'], shuffle=False)
        
        split_r2_list = []
        split_mae_list = []
        
        for seed in range(42, 42 + args.seeds):
            set_seed(seed)
            train_loader = DataLoader(train_set, batch_size=model_args['batch_size'], shuffle=True)
            val_loader   = DataLoader(val_set, batch_size=model_args['batch_size'], shuffle=False)
            
            runner = Runner(model_args, seed=seed, save_dir=f"{out_dir}/{split_name}")
            runner.train(train_loader, val_loader)
            
            test_pred, test_true = runner.test(test_loader)
            
            # Export individual seed predictions
            seed_df = pd.DataFrame({'true_x1': test_true, 'pred_x1': test_pred})
            os.makedirs(f"{out_dir}/{split_name}_preds", exist_ok=True)
            seed_df.to_csv(f"{out_dir}/{split_name}_preds/seed{seed}.csv", index=False)
            
            r2 = r2_score(test_true, test_pred)
            mae = mean_absolute_error(test_true, test_pred)
            split_r2_list.append(r2)
            split_mae_list.append(mae)
            
        summary_results.append({
            'Target': split_name,
            'Family': args.family,
            'Mode': args.mode,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'R2_mean': np.mean(split_r2_list),
            'R2_std': np.std(split_r2_list),
            'MAE_mean': np.mean(split_mae_list),
            'MAE_std': np.std(split_mae_list)
        })
        
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(f"{out_dir}/summary.csv", index=False)
    print(f"\n✅ All completed! Results saved to {out_dir}/summary.csv")

if __name__ == '__main__':
    main()
