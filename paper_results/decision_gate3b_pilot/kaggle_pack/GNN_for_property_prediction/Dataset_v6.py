"""
Dataset_v6.py — 消融实验专用数据加载器 (Unified V6 22-dim Schema)
=================================================================
【v6 核心规范】
  严格按照 22 维统一数据布局（3 分子图 + 19 连续标量）进行动态条件特征映射。
  支持 valid_indices 传入，实现 Complete-Case 公平基准子集的无缝加载与 NaN 隔离。

  消融模式:
    M0           : 9 维 (T, P + 7 个单分子基础物性)
    Mphys        : 9 + 3 = 12 维 (M0 + ref_dipole, ref_polarizability, ref_volume)
    Mthermo      : 9 + 3 = 12 维 (M0 + Tc, Pc, omega)
    Mreduced     : 9 + 3 = 12 维 (M0 + Tr, Pr, omega)
    Minteract    : 9 + 2 = 11 维 (M0 + deltaE_anion, deltaE_cation)
    Mreduced_pure: 10 维 (7 个物性 + Tr, Pr, omega)
    M_all        : 17 维 (全物理描述符汇聚)
"""
import os
import joblib
import numpy as np
import torch
from torch_geometric.data import Batch, Data, Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ============================================================
# 终极 22 维特征的全局索引簿 (绝对防爆防错位)
# ============================================================
FEATURE_SCHEMA = {
    "T": 3, "P": 4, 
    "ref_charge": 5, "ref_logp": 6, "ani_mw": 7, "cat_charge": 8, "cat_tpsa": 9, 
    "ref_MW": 10, "cat_MW": 11,
    "ref_dipole": 12, "ref_polarizability": 13, "ref_volume": 14,
    "deltaE_anion": 15, "deltaE_cation": 16,
    "Tc": 17, "Pc": 18, "omega": 19,
    "Tr": 20, "Pr": 21,
}

BASE_FEATURES = ["T", "P", "ref_charge", "ref_logp", "ani_mw", "cat_charge", "cat_tpsa", "ref_MW", "cat_MW"]

MODE_DEF = {
    'M0':        BASE_FEATURES,
    'Mstd':      BASE_FEATURES + ["T", "P", "omega"],
    'Mphys':     BASE_FEATURES + ["ref_dipole", "ref_polarizability", "ref_volume"],
    'Mthermo':   BASE_FEATURES + ["Tc", "Pc", "omega"],
    'Mreduced':  BASE_FEATURES + ["Tr", "Pr", "omega"],
    'Minteract': BASE_FEATURES + ["deltaE_anion", "deltaE_cation"],
    'Mreduced_pure': ["ref_charge", "ref_logp", "ani_mw", "cat_charge", "cat_tpsa", "ref_MW", "cat_MW", "Tr", "Pr", "omega"],
    'M_all':     BASE_FEATURES + ["ref_dipole", "ref_polarizability", "ref_volume", "deltaE_anion", "deltaE_cation", "Tc", "Pc", "omega"]
}

# 自动映射为网络所需的整数索引列表
MODE_INDICES = {
    mode: [FEATURE_SCHEMA[feat] for feat in feat_list]
    for mode, feat_list in MODE_DEF.items()
}

MODE_COND_DIM = {k: len(v) for k, v in MODE_INDICES.items()}
# M0: 7, Msize: 9, Mmu: 8, Malpha: 8, MV: 8, Mphys: 10


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
    1. 支持 descriptor_mode 参数 (M0/Mphys/Mthermo/Mreduced/Minteract 等)
    2. 根据模式动态选择条件特征子集
    3. 支持 valid_indices 传入，只对活跃实验子集执行 NaN 拦截和训练
    4. 内置 StandardScaler 管理，支持按照 split 保存和加载
    5. 暴露 cond_dim 属性供 Model_v6 读取
    """
    def __init__(self, path, args, valid_indices=None):
        super(IL_set_v6, self).__init__()
        self.args = args

        # 读取消融模式，默认 M0
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

        raw_data = np.load(data_path, allow_pickle=True)
        raw_label = np.load(label_path, allow_pickle=True)

        # 支持子集切片 (如 Complete-Case 过滤)
        if valid_indices is not None:
            self.data = raw_data[valid_indices]
            self.label = raw_label[valid_indices]
        else:
            self.data = raw_data
            self.label = raw_label

        self.length = self.label.shape[0]
        self.scalers = None

        # ── 数据完整性检查 ──
        sample = self.data[0]
        n_elements = len(sample)
        required_elements = max(self.feature_indices) + 1
        if n_elements < required_elements:
            raise ValueError(
                f"数据布局错误：模式 {self.descriptor_mode} 需要至少 {required_elements} 个元素，实际为 {n_elements}。"
            )

        # ── 活跃子集 NaN/NA 检查（严格防止坏行进入训练） ──
        for data_idx in self.feature_indices:
            vals = np.asarray([row[data_idx] for row in self.data], dtype=np.float64)
            if not np.all(np.isfinite(vals)):
                raise ValueError(
                    f"数据质量错误：特征索引 {data_idx} 在当前实验子集中含 NaN/Inf。"
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
            self.scales[feat_pos] = max(float(self.scalers[feat_pos].scale_[0]), 1e-8)

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
        feature_clip = self.args.get('feature_clip')
        if feature_clip is not None:
            condition = torch.clamp(condition, -feature_clip, feature_clip)
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
