"""
Dataset_v7.py — Decoupled Three-Component Dataset & Dataloader for V7
=====================================================================
Provides decoupled molecular subgraphs for Cation, Anion, and Refrigerant,
eliminating the single global virtual node bottleneck (RANGE, Nat. Commun. 2026).
Explicitly separates condition inputs into thermodynamic state variables (T, P)
and molecular shortcut descriptors (7 descriptors), supporting modality-level
dropout (ModDrop, arXiv:1501.00102; MEGNet, Chem. Mater. 2019).
"""

import os
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler

# V7 Schema: Explicitly separated State vs Descriptors
STATE_INDICES = [3, 4]  # T, P
DESC_INDICES  = [5, 6, 7, 8, 9, 10, 11]  # ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa, ref_MW, cat_MW
ALL_M0_INDICES = STATE_INDICES + DESC_INDICES

class DecoupledTriDataset_v7(Dataset):
    def __init__(self, data_root, args=None, valid_indices=None):
        super(DecoupledTriDataset_v7, self).__init__()
        self.args = args or {}
        self.target_edge_dim = self.args.get('edge_dim', 4)
        
        data_path = os.path.join(data_root, 'data.npy')
        label_path = os.path.join(data_root, 'label.npy')
        
        raw_data = np.load(data_path, allow_pickle=True)
        raw_label = np.load(label_path, allow_pickle=True)
        
        if valid_indices is not None:
            self.data = raw_data[valid_indices]
            self.label = raw_label[valid_indices]
        else:
            self.data = raw_data
            self.label = raw_label
            
        self.length = len(self.label)
        self.means = None
        self.scales = None
        
    def fit_scalers(self, train_indices, save_dir=None):
        """Fit standard scalers on training set for all 9 M0 features."""
        n_features = len(ALL_M0_INDICES)
        self.means = np.zeros(n_features, dtype=np.float32)
        self.scales = np.ones(n_features, dtype=np.float32)
        
        scalers = []
        for i, idx in enumerate(ALL_M0_INDICES):
            vals = np.array([self.data[k][idx] for k in train_indices], dtype=np.float32).reshape(-1, 1)
            sc = StandardScaler()
            sc.fit(vals)
            scalers.append(sc)
            self.means[i] = float(sc.mean_[0])
            self.scales[i] = max(float(sc.scale_[0]), 1e-8)
            
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(scalers, os.path.join(save_dir, 'scalers.pkl'))
            
    def load_scalers(self, scaler_path):
        scalers = joblib.load(scaler_path)
        n_features = len(ALL_M0_INDICES)
        self.means = np.zeros(n_features, dtype=np.float32)
        self.scales = np.ones(n_features, dtype=np.float32)
        for i, sc in enumerate(scalers):
            self.means[i] = float(sc.mean_[0])
            self.scales[i] = max(float(sc.scale_[0]), 1e-8)

    def mol2graph(self, mol_raw):
        x = torch.tensor(mol_raw[0], dtype=torch.long)
        edge_index = torch.tensor(mol_raw[1], dtype=torch.long)
        
        if len(mol_raw[2]) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.zeros((1, self.target_edge_dim), dtype=torch.long)
        else:
            raw_attr = torch.tensor(mol_raw[2], dtype=torch.long)
            if self.target_edge_dim == 4 and raw_attr.size(1) == 3:
                zero_stereo = torch.zeros((raw_attr.size(0), 1), dtype=torch.long)
                edge_attr = torch.cat([raw_attr, zero_stereo], dim=1)
            else:
                edge_attr = raw_attr
                
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Decoupled Graphs (No global node!)
        g_cat = self.mol2graph(item[0])
        g_ani = self.mol2graph(item[1])
        g_ref = self.mol2graph(item[2])
        
        # 2. Extract and standardize condition features
        raw_cond = [item[i] for i in ALL_M0_INDICES]
        if self.means is not None:
            norm_cond = [(raw_cond[i] - self.means[i]) / self.scales[i] for i in range(len(raw_cond))]
        else:
            norm_cond = raw_cond
            
        # Split into State (T, P) vs Descriptors (7 molecular shortcuts)
        state_cond = torch.tensor(norm_cond[0:2], dtype=torch.float32)  # [T, P]
        desc_cond  = torch.tensor(norm_cond[2:9], dtype=torch.float32)  # 7 descriptors
        
        label = torch.tensor(self.label[idx], dtype=torch.float32)
        
        return {
            'cat': g_cat,
            'ani': g_ani,
            'ref': g_ref,
            'state': state_cond,
            'desc': desc_cond,
            'label': label
        }


def collate_v7(batch_list):
    """
    Batches Cation, Anion, and Refrigerant graphs independently,
    and collates state and descriptor condition vectors.
    """
    batch_cat = Batch.from_data_list([item['cat'] for item in batch_list])
    batch_ani = Batch.from_data_list([item['ani'] for item in batch_list])
    batch_ref = Batch.from_data_list([item['ref'] for item in batch_list])
    
    batch_state = torch.stack([item['state'] for item in batch_list], dim=0)
    batch_desc  = torch.stack([item['desc'] for item in batch_list], dim=0)
    batch_label = torch.stack([item['label'] for item in batch_list], dim=0)
    
    return {
        'cat': batch_cat,
        'ani': batch_ani,
        'ref': batch_ref,
        'state': batch_state,
        'desc': batch_desc,
        'label': batch_label
    }

def get_v7_dataloader(dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_v7,
        drop_last=drop_last
    )
