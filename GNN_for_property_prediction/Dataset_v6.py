"""
Dataset_v6.py — 消融实验专用数据加载器
=====================================
【v6 核心升级】
  支持 descriptor_mode 参数，根据模式动态选择条件向量的特征子集。
  配合 prepare_tri_graph_data_v3.py 生成的 12 维条件数据使用。

  四种消融模式:
    M0    : 原始 7 维 (T, P, ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa)
    Msize : 7 + 2 = 9 维 (M0 + ref_MolWt, cat_MolWt)
    Mmu   : 7 + 1 = 8 维 (M0 + ref_dipole)
    Mphys : 7 + 3 = 10 维 (M0 + ref_dipole, ref_polarizability, ref_volume)

  v3 数据布局 (data[i] 的索引):
    [0] cation_graph  [1] anion_graph  [2] refri_graph
    [3] T  [4] P  [5] ref_charge  [6] ref_logp  [7] ani_mw
    [8] cat_charge  [9] cat_tpsa
    [10] ref_MolWt  [11] cat_MolWt
    [12] ref_dipole  [13] ref_polarizability  [14] ref_volume
"""
import os
import joblib
import numpy as np
import torch
from torch_geometric.data import Batch, Data, Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ============================================================
# 每种消融模式对应的条件特征索引
# ============================================================
MODE_INDICES = {
    'M0':    [3, 4, 5, 6, 7, 8, 9],                    # 7 维：原始基线
    'Msize': [3, 4, 5, 6, 7, 8, 9, 10, 11],            # 9 维：+ ref_MolWt, cat_MolWt
    'Mmu':   [3, 4, 5, 6, 7, 8, 9, 12],                # 8 维：+ ref_dipole
    'Mphys': [3, 4, 5, 6, 7, 8, 9, 12, 13, 14],        # 10 维：+ ref_dipole, ref_pol, ref_vol
}

MODE_COND_DIM = {k: len(v) for k, v in MODE_INDICES.items()}
# M0: 7, Msize: 9, Mmu: 8, Mphys: 10


args_global = {
    'add_global': True,
    'bi_direction': True
}


def combine_Graph(Graph_list):
    combined = Batch.from_data_list(Graph_list)
    combined_Graph = Data(x=combined.x, edge_index=combined.edge_index,
                          edge_attr=combined.edge_attr, mol_type=combined.batch)
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


class IL_set_v6(torch.utils.data.Dataset):
    """
    v6 版本数据集：
    1. 支持 descriptor_mode 参数 (M0/Msize/Mmu/Mphys)
    2. 根据模式动态选择条件特征子集
    3. 内置 StandardScaler 管理，支持按照 split 保存和加载
    4. 暴露 cond_dim 属性供 Model_v6 读取
    """
    def __init__(self, path, args):
        super(IL_set_v6, self).__init__()
        self.args = args

        # 读取消融模式，默认 M0（与 v5 行为一致）
        self.descriptor_mode = args.get('descriptor_mode', 'M0')
        if self.descriptor_mode not in MODE_INDICES:
            raise ValueError(
                f"未知的 descriptor_mode: '{self.descriptor_mode}'. "
                f"支持的模式: {list(MODE_INDICES.keys())}"
            )

        self.feature_indices = MODE_INDICES[self.descriptor_mode]
        self.cond_dim = len(self.feature_indices)

        # 打印模式信息
        print(f"  [Dataset_v6] 消融模式: {self.descriptor_mode}")
        print(f"  [Dataset_v6] 条件维度: {self.cond_dim}")
        print(f"  [Dataset_v6] 使用特征索引: {self.feature_indices}")

        data_path = os.path.join(path, 'data.npy')
        label_path = os.path.join(path, 'label.npy')

        self.data = np.load(data_path, allow_pickle=True)
        self.label = np.load(label_path, allow_pickle=True)
        self.length = self.label.shape[0]
        self.scalers = None

        # ── 数据完整性检查 ──
        # 确认数据中的条件特征数量足够
        sample = self.data[0]
        n_elements = len(sample)
        max_idx = max(self.feature_indices)
        if max_idx >= n_elements:
            raise ValueError(
                f"数据完整性错误！descriptor_mode='{self.descriptor_mode}' "
                f"需要索引 {max_idx}，但数据每行只有 {n_elements} 个元素。\n"
                f"请确认你使用的是 prepare_tri_graph_data_v3.py 生成的数据 "
                f"(应保存在 processed_tri_data_v3/ 目录下)。"
            )

        # ── NaN/NA 检查 ──
        # 对所有用到的特征索引，检查是否存在 NaN 值
        for idx in self.feature_indices:
            sample_val = sample[idx]
            if sample_val is None or (isinstance(sample_val, float) and np.isnan(sample_val)):
                raise ValueError(
                    f"数据质量错误！第一个样本的索引 {idx} 包含 NaN/None。"
                    f"请检查 prepare_tri_graph_data_v3.py 的输出。"
                )

    def fit_scalers(self, train_indices, save_dir):
        """
        Fit scalers ONLY on training data to prevent data leakage.
        只对当前模式使用的特征进行 scaler 拟合。
        """
        n_features = self.cond_dim
        self.scalers = [StandardScaler() for _ in range(n_features)]
        self.means = np.zeros(n_features, dtype=np.float32)
        self.scales = np.ones(n_features, dtype=np.float32)

        for feat_pos, data_idx in enumerate(self.feature_indices):
            raw_vals = np.array(
                [self.data[i][data_idx] for i in train_indices],
                dtype=np.float32
            ).reshape(-1, 1)

            # ── 防泄漏检查：确认没有 NaN ──
            if np.any(np.isnan(raw_vals)):
                raise ValueError(
                    f"Scaler 拟合失败！特征索引 {data_idx} (模式位置 {feat_pos}) "
                    f"在训练集中存在 NaN 值。"
                )

            self.scalers[feat_pos].fit(raw_vals)
            self.means[feat_pos] = float(self.scalers[feat_pos].mean_[0])
            self.scales[feat_pos] = float(self.scalers[feat_pos].scale_[0])

        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scalers, os.path.join(save_dir, 'scalers.pkl'))
        print(f"  [Dataset_v6] {n_features} 个特征的 StandardScaler 拟合完成"
              f"（模式: {self.descriptor_mode}），已保存至 {save_dir}/scalers.pkl")

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

        # ── 根据 descriptor_mode 提取对应的条件特征 ──
        raw_cond = [data[idx_] for idx_ in self.feature_indices]

        # 标准化
        if hasattr(self, 'means') and self.means is not None:
            scaled_cond = [
                (raw_cond[i] - self.means[i]) / self.scales[i]
                for i in range(self.cond_dim)
            ]
        elif self.scalers is not None:
            scaled_cond = [
                (raw_cond[i] - float(self.scalers[i].mean_[0])) / float(self.scalers[i].scale_[0])
                for i in range(self.cond_dim)
            ]
        else:
            scaled_cond = raw_cond

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
