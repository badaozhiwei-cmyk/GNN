"""
Dataset_v5.py
=============
【目的】
  修复原版 Dataset.py 中 USE_ANI_MW 被写死的问题。
  将 Scaler 整合进 Dataset 中，供 Runner 统一调用。
"""
import os
import joblib
import numpy as np
import torch
from torch_geometric.data import Batch, Data, Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

args_global = {
    'add_global': True,
    'bi_direction': True
}

def combine_Graph(Graph_list):
    combined = Batch.from_data_list(Graph_list)
    combined_Graph = Data(x=combined.x, edge_index=combined.edge_index, edge_attr=combined.edge_attr, mol_type=combined.batch)
    return combined_Graph

def add_global(graph):
    node = torch.tensor([0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)
    x = torch.cat([graph.x, node], dim=0)
    num_node = x.shape[0] - 1
    new_node = x.shape[0] - 1
    start = []
    end = []
    attr = []
    for i in range(num_node):
        start.append(i)
        end.append(new_node)
        attr.append([0, 0, 0])
    if args_global['bi_direction'] == True:
        for i in range(num_node):
            start.append(new_node)
            end.append(i)
            attr.append([0, 0, 0])

    start = torch.tensor(start).reshape(1, -1)
    end = torch.tensor(end).reshape(1, -1)
    new_edge = torch.cat([start, end], dim=0)
    edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
    attr = torch.tensor(attr)
    edge_attr = torch.cat([graph.edge_attr, attr], dim=0)
    
    if hasattr(graph, 'mol_type'):
        global_mol_type = torch.tensor([3], dtype=torch.long)
        new_mol_type = torch.cat([graph.mol_type, global_mol_type], dim=0)
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mol_type=new_mol_type)
    else:
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return g


class IL_set_v5(torch.utils.data.Dataset):
    """
    v5 版本数据集：
    1. 接受 args 动态控制 use_ani_mw
    2. 内置 StandardScaler 管理，支持按照 split 保存和加载
    """
    def __init__(self, path, args):
        super(IL_set_v5, self).__init__()
        self.args = args
        self.use_ani_mw = args.get('use_ani_mw', False)
        
        data_path = os.path.join(path, 'data.npy')
        label_path = os.path.join(path, 'label.npy')

        self.data = np.load(data_path, allow_pickle=True)
        self.label = np.load(label_path, allow_pickle=True)
        self.length = self.label.shape[0]
        self.scalers = None

    def fit_scalers(self, train_indices, save_dir):
        """Fit scalers ONLY on training data to prevent data leakage."""
        self.scalers = [StandardScaler() for _ in range(7)]
        self.means = np.zeros(7, dtype=np.float32)
        self.scales = np.ones(7, dtype=np.float32)
        
        for feature_idx in range(7):
            raw_vals = np.array([self.data[i][feature_idx + 3] for i in train_indices], dtype=np.float32).reshape(-1, 1)
            self.scalers[feature_idx].fit(raw_vals)
            self.means[feature_idx] = float(self.scalers[feature_idx].mean_[0])
            self.scales[feature_idx] = float(self.scalers[feature_idx].scale_[0])
        
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scalers, os.path.join(save_dir, 'scalers.pkl'))
        print(f"  [Dataset] 7个物理量的 StandardScaler 拟合完成并已保存至 {save_dir}/scalers.pkl")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.data[idx]
        cation = self.mol2graph(data[0])
        anion  = self.mol2graph(data[1])
        refri  = self.mol2graph(data[2])
        
        Combine_Graph = combine_Graph([cation, anion, refri])
        if self.args.get('add_global', True):
            Combine_Graph = add_global(Combine_Graph)

        # 获取7个物理特征
        T, P = data[3], data[4]
        ref_charge   = data[5]
        ref_logp     = data[6]
        ani_mw       = data[7]
        cat_charge   = data[8]
        cat_tpsa     = data[9]
        
        raw_cond = [T, P, ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa]
        
        # [极速向量化标准化] 使用预提取的 means/scales 替代繁重的 sklearn transform
        if hasattr(self, 'means') and self.means is not None:
            scaled_cond = [(raw_cond[i] - self.means[i]) / self.scales[i] for i in range(7)]
        elif self.scalers is not None:
            scaled_cond = [(raw_cond[i] - float(self.scalers[i].mean_[0])) / float(self.scalers[i].scale_[0]) for i in range(7)]
        else:
            scaled_cond = raw_cond

        if self.use_ani_mw:
            condition = torch.tensor(scaled_cond, dtype=torch.float)
        else:
            # 移除 ani_mw (index 4)
            scaled_cond.pop(4)
            condition = torch.tensor(scaled_cond, dtype=torch.float)

        label = torch.tensor(self.label[idx], dtype=torch.float)

        return Combine_Graph, condition, label

    def mol2graph(self, mol):
        x = torch.tensor(mol[0], dtype=torch.long)
        edge_index = torch.tensor(mol[1], dtype=torch.long)
        
        if len(mol[2]) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.zeros((1, 3), dtype=torch.long)
        else:
            edge_attr = torch.tensor(mol[2], dtype=torch.long)

        Graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return Graph
