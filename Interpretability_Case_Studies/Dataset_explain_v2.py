"""
Dataset_explain_v2.py
===================================================
制冷剂三图体系专用可解释性数据集加载?(V2)

?CO2 旧版 Dataset_explain.py 的核心差异：
  - 旧版: 双图（阳离子 + 阴离子）+ 2D 条件向量 [T, P]
  - V2:   三图（阳离子 + 阴离?+ 制冷剂）+ 7D 条件向量
            [T, P, Ref_Charge, Ref_LogP, Ani_MW, Cat_RotBonds, Cat_LogP]
            （与 Dataset.py 归一化逻辑完全一致）
  - V2:   额外返回 num_bond，供 Explainer 区分"真实??全局虚拟?

数据来源: processed_tri_data/data.npy + label.npy
          （由 prepare_tri_graph_data.py 生成，与训练时完全相同）
"""

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from sklearn.preprocessing import StandardScaler


# ── 全局配置（与 Dataset.py 保持一致）──────────────────────────
args = {
    'add_global': True,
    'bi_direction': True
}


# ══════════════════════════════════════════════════════════════
# 工具函数（直接复制自 Dataset.py，保证逻辑完全一致）
# ══════════════════════════════════════════════════════════════

def combine_Graph(Graph_list):
    """将多个子图（??制冷剂）拼接为单一大图"""
    combined = Batch.from_data_list(Graph_list)
    # [修复] 注入 mol_type，利?PyG ?batch 张量? 代表阳离子，1 代表阴离子，2 代表制冷剂）
    combined_Graph = Data(
        x=combined.x,
        edge_index=combined.edge_index,
        edge_attr=combined.edge_attr,
        mol_type=combined.batch
    )
    return combined_Graph


def add_global(graph):
    """
    添加一个全局虚拟节点（特征全零），并与所有真实原子节点双向相连?
    这个全局节点?GNN 提取图级别表示的"汇聚??
    返回添加全局节点后的图，以及新增?全局?数量（用?Explainer 解析掩码）?
    """
    # V2: 全局节点的特征维度必须与真实原子一致，?7 ?
    node = torch.tensor([0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)
    x = torch.cat([graph.x, node], dim=0)
    num_node = x.shape[0] - 1      # 真实原子?
    new_node  = x.shape[0] - 1     # 全局节点的索?

    start, end, attr = [], [], []

    # 真实原子 ?全局节点（正向）
    for i in range(num_node):
        start.append(i)
        end.append(new_node)
        attr.append([0, 0, 0])

    # 全局节点 ?真实原子（反向）
    if args['bi_direction']:
        for i in range(num_node):
            start.append(new_node)
            end.append(i)
            attr.append([0, 0, 0])

    new_edge   = torch.cat([
        torch.tensor(start).reshape(1, -1),
        torch.tensor(end).reshape(1, -1)
    ], dim=0)
    edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
    edge_attr  = torch.cat([graph.edge_attr, torch.tensor(attr)], dim=0)

    # [修复] 为全局虚拟节点分配特殊?mol_type = 3
    if hasattr(graph, 'mol_type'):
        global_mol_type = torch.tensor([3], dtype=torch.long)
        new_mol_type = torch.cat([graph.mol_type, global_mol_type], dim=0)
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mol_type=new_mol_type)
    else:
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return g


# ══════════════════════════════════════════════════════════════
# 数据集类：三图制冷剂体系（供 GNNExplainer 使用?
# ══════════════════════════════════════════════════════════════

class IL_set_v2(torch.utils.data.Dataset):
    """
    制冷?离子液体三图数据集（可解释性专用版）?

    __getitem__ 返回?
        Combine_Graph : PyG Data，三图拼?+ 全局节点
        condition     : FloatTensor [5]，归一化后?[T, P, Ref_Charge, Ref_LogP, Ani_MW]
        label         : FloatTensor scalar，溶解度 x1
        num_bond      : int，添加全局节点之前的原始边?
                        （Explainer 用它区分哪些 edge_mask 属于真实化学键）
    """

    def __init__(self, data_npy_path: str, label_npy_path: str):
        """
        Args:
            data_npy_path  : processed_tri_data/data.npy 的路?
            label_npy_path : processed_tri_data/label.npy 的路?
        """
        super().__init__()

        self.data  = np.load(data_npy_path,  allow_pickle=True)
        self.label = np.load(label_npy_path, allow_pickle=True)
        self.length = self.label.shape[0]

        # ── ?5 ?RDKit 特征?StandardScaler（与 Dataset.py 完全一致）──
        # index[5]=Ref_Charge, index[6]=Ref_LogP, index[7]=Ani_MW,
        # index[8]=Cat_Charge, index[9]=Cat_TPSA  ?V2 依据热图更新
        raw_T           = np.array([self.data[i][3] for i in range(self.length)], dtype=np.float32).reshape(-1, 1)
        raw_P           = np.array([self.data[i][4] for i in range(self.length)], dtype=np.float32).reshape(-1, 1)
        raw_ref_charge  = np.array([self.data[i][5] for i in range(self.length)], dtype=np.float32).reshape(-1, 1)
        raw_ref_logp    = np.array([self.data[i][6] for i in range(self.length)], dtype=np.float32).reshape(-1, 1)
        raw_ani_mw      = np.array([self.data[i][7] for i in range(self.length)], dtype=np.float32).reshape(-1, 1)
        raw_cat_charge  = np.array([self.data[i][8] for i in range(self.length)], dtype=np.float32).reshape(-1, 1)  # V2
        raw_cat_tpsa    = np.array([self.data[i][9] for i in range(self.length)], dtype=np.float32).reshape(-1, 1)  # V2

        self.scaler_T           = StandardScaler().fit(raw_T)
        self.scaler_P           = StandardScaler().fit(raw_P)
        self.scaler_ref_charge  = StandardScaler().fit(raw_ref_charge)
        self.scaler_ref_logp    = StandardScaler().fit(raw_ref_logp)
        self.scaler_ani_mw      = StandardScaler().fit(raw_ani_mw)
        self.scaler_cat_charge  = StandardScaler().fit(raw_cat_charge)  # V2
        self.scaler_cat_tpsa    = StandardScaler().fit(raw_cat_tpsa)    # V2

        print(f"[Dataset_explain_v2] 加载完毕，共 {self.length} 条数据。")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        d = self.data[idx]

        # ── Step 1: 读取三个分子?──────────────────────────────
        cation = self._mol2graph(d[0])
        anion  = self._mol2graph(d[1])
        refri  = self._mol2graph(d[2])

        # ── Step 2: 三图合并 ──────────────────────────────────
        Combine_Graph = combine_Graph([cation, anion, refri])

        # 记录添加全局节点之前的边数（Explainer 解析掩码时需要）
        num_bond = Combine_Graph.edge_index.shape[1]

        # ── Step 3: 添加全局节点 ──────────────────────────────
        if args['add_global']:
            Combine_Graph = add_global(Combine_Graph)

        # ── Step 4: 组装 7 维条件向量（归一化）────────────────
        T          = float(self.scaler_T.transform([[d[3]]])[0][0])
        P          = float(self.scaler_P.transform([[d[4]]])[0][0])
        ref_charge = float(self.scaler_ref_charge.transform([[d[5]]])[0][0])
        ref_logp   = float(self.scaler_ref_logp.transform([[d[6]]])[0][0])
        ani_mw     = float(self.scaler_ani_mw.transform([[d[7]]])[0][0])
        cat_charge   = float(self.scaler_cat_charge.transform([[d[8]]])[0][0])  # V2
        cat_tpsa     = float(self.scaler_cat_tpsa.transform([[d[9]]])[0][0])    # V2

        condition = torch.tensor([T, P, ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa], dtype=torch.float)

        # ── Step 5: 标签 ────────────────────────────────────
        label = torch.tensor(float(self.label[idx]), dtype=torch.float)

        return Combine_Graph, condition, label, num_bond

    def _mol2graph(self, mol):
        """?(node_feat, edge_index, edge_attr) 元组转为 PyG Data"""
        x          = torch.tensor(mol[0], dtype=torch.long)
        edge_index = torch.tensor(mol[1], dtype=torch.long)

        # 单原子离子（?[I-]、[Cl-]）没有化学键，需要特殊处?
        if len(mol[2]) == 0:
            edge_attr = torch.zeros((0, 3), dtype=torch.long)
        else:
            edge_attr = torch.tensor(mol[2], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @staticmethod
    def collate_fn(batch):
        """DataLoader collate：batch 中每个元素是 (Graph, cond, label, num_bond)"""
        from torch_geometric.data import Batch as PyGBatch
        graphs    = [item[0] for item in batch]
        conds     = torch.stack([item[1] for item in batch])
        labels    = torch.stack([item[2] for item in batch])
        num_bonds = [item[3] for item in batch]

        batched_graph = PyGBatch.from_data_list(graphs)
        return batched_graph, conds, labels, num_bonds


# ── 快速验?────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    root = os.path.join(os.path.dirname(__file__), '..', 'processed_tri_data')
    ds = IL_set_v2(
        data_npy_path  = os.path.join(root, 'data.npy'),
        label_npy_path = os.path.join(root, 'label.npy')
    )
    G, cond, label, nb = ds[0]
    print(f"  图节点数: {G.x.shape[0]}, 边数: {G.edge_index.shape[1]}")
    print(f"  num_bond (真实): {nb}")
    print(f"  条件向量: {cond}")
    print(f"  标签 x1: {label.item():.4f}")

def smiles_to_graph(smiles_string):
    import sys, os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    from prepare_tri_graph_data import mol2graph_components
    mol = mol2graph_components(smiles_string)
    if mol is None: return None
    import torch
    from torch_geometric.data import Data
    x = torch.tensor(mol[0], dtype=torch.long)
    edge_index = torch.tensor(mol[1], dtype=torch.long)
    if len(mol[2]) == 0:
        edge_attr = torch.zeros((0, 3), dtype=torch.long)
    else:
        edge_attr = torch.tensor(mol[2], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
