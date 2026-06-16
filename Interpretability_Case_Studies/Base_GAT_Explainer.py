import os
import sys
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data, Batch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))

from Model import IL_GAT
from Dataset_explain_v2 import IL_set_v2

# ─────────────────────────────────────────────────────────
# 化学元素映射与辅助函数
# ─────────────────────────────────────────────────────────
ELEMENT_SYMBOL = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

ELECTRONEG = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
COV_RADIUS = {1: 31, 5: 84, 6: 77, 7: 71, 8: 66, 9: 64, 15: 107, 16: 105, 17: 102, 35: 120, 53: 139}

def bucketize(val, min_v, max_v, n_buckets=8):
    ratio = (val - min_v) / (max_v - min_v + 1e-8)
    return min(int(ratio * n_buckets), n_buckets - 1)

def get_atom_features(atom):
    hybrid = int(atom.GetHybridization())
    if hybrid >= 8: hybrid = 7
    aro = 1 if atom.GetIsAromatic() else 0
    degree = atom.GetDegree()
    if degree >= 7: degree = 6
    charge = atom.GetFormalCharge() + 1
    if charge > 2: charge = 2
    if charge < 0: charge = 0
    atomic_num = atom.GetAtomicNum()
    eneg_bucket = bucketize(ELECTRONEG.get(atomic_num, 2.55), 2.04, 3.98, 8)
    radius_bucket = bucketize(COV_RADIUS.get(atomic_num, 77), 31, 139, 8)
    return [atomic_num, hybrid, aro, degree, charge, eneg_bucket, radius_bucket]

def get_bond_features(bond):
    bond_type_dict = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2, 
                      Chem.rdchem.BondType.TRIPLE: 3, Chem.rdchem.BondType.AROMATIC: 4}
    return [
        bond_type_dict.get(bond.GetBondType(), 1),
        1 if bond.IsInRing() else 0,
        1 if bond.GetIsAromatic() else 0
    ]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.long)
    
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e_attr = get_bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e_attr, e_attr]
        
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def build_tri_graph_with_global(c_smi, a_smi, r_smi):
    c_data, a_data, r_data = smiles_to_graph(c_smi), smiles_to_graph(a_smi), smiles_to_graph(r_smi)
    if not (c_data and a_data and r_data): return None, 0
        
    nc, na, nr = c_data.x.shape[0], a_data.x.shape[0], r_data.x.shape[0]
    
    x = torch.cat([c_data.x, a_data.x, r_data.x], dim=0)
    edge_index = torch.cat([c_data.edge_index, a_data.edge_index + nc, r_data.edge_index + nc + na], dim=1)
    edge_attr = torch.cat([c_data.edge_attr, a_data.edge_attr, r_data.edge_attr], dim=0)
    num_real_bonds = edge_index.shape[1]
    
    mol_type = torch.cat([
        torch.zeros(nc, dtype=torch.long),
        torch.ones(na, dtype=torch.long),
        torch.full((nr,), 2, dtype=torch.long)
    ], dim=0)
    
    # 添加全局虚拟节点 (GNN 模型需要)
    num_real_nodes = x.shape[0]
    global_node_feat = torch.zeros((1, x.shape[1]), dtype=torch.long)
    global_node_feat[0, 0] = 0 
    x_new = torch.cat([x, global_node_feat], dim=0)
    
    global_edges = []
    global_edge_attrs = []
    for i in range(num_real_nodes):
        global_edges.append([num_real_nodes, i])
        global_edge_attrs.append([0, 0, 0])
        
    if len(global_edges) > 0:
        ge = torch.tensor(global_edges, dtype=torch.long).t().contiguous()
        ga = torch.tensor(global_edge_attrs, dtype=torch.long)
        edge_index_new = torch.cat([edge_index, ge], dim=1)
        edge_attr_new = torch.cat([edge_attr, ga], dim=0)
    else:
        edge_index_new = edge_index
        edge_attr_new = edge_attr

    mol_type_new = torch.cat([mol_type, torch.tensor([3], dtype=torch.long)])

    data = Data(x=x_new, edge_index=edge_index_new, edge_attr=edge_attr_new)
    data.mol_type = mol_type_new
    return data, num_real_bonds


# ─────────────────────────────────────────────────────────
# 核心 GAT 分析类 (重构为 反事实微扰法 Counterfactual Perturbation)
# ─────────────────────────────────────────────────────────
class UniversalGATExplainer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[初始化] 加载 GAT 权重: {model_path} on {self.device}")
        print(f"[引擎切换] 正在使用端到端反事实微扰法 (Counterfactual Perturbation Explainer)")
        
        Args = {'emb_dim': 300, 'dropout_rate': 0.2, 'n_features': 7}
        self.model = IL_GAT(Args).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        self.model.eval()
        
        # 加载 IL_set_v2 以获取缩放器 (Scaler)
        data_path = os.path.join(ROOT, 'processed_tri_data')
        self.dataset = IL_set_v2(data_npy_path=os.path.join(data_path, 'data.npy'), 
                                 label_npy_path=os.path.join(data_path, 'label.npy'))

    def compute_condition(self, c_smi, a_smi, r_smi, T, P):
        try:
            m_c = Chem.MolFromSmiles(c_smi)
            cat_charge = float(Descriptors.MaxAbsPartialCharge(m_c))
            cat_tpsa   = float(Descriptors.TPSA(m_c))
        except: cat_charge, cat_tpsa = 0.0, 0.0
            
        try:
            m_a = Chem.MolFromSmiles(a_smi)
            ani_mw = float(Descriptors.MolWt(m_a))
        except: ani_mw = 0.0
            
        try:
            m_r = Chem.MolFromSmiles(r_smi)
            ref_charge = float(Descriptors.MaxAbsPartialCharge(m_r))
            ref_logp   = float(Descriptors.MolLogP(m_r))
        except: ref_charge, ref_logp = 0.0, 0.0

        T_s = float(self.dataset.scaler_T.transform([[T]])[0][0])
        P_s = float(self.dataset.scaler_P.transform([[P]])[0][0])
        rc_s = float(self.dataset.scaler_ref_charge.transform([[ref_charge]])[0][0])
        rl_s = float(self.dataset.scaler_ref_logp.transform([[ref_logp]])[0][0])
        am_s = float(self.dataset.scaler_ani_mw.transform([[ani_mw]])[0][0])
        cc_s = float(self.dataset.scaler_cat_charge.transform([[cat_charge]])[0][0])
        ct_s = float(self.dataset.scaler_cat_tpsa.transform([[cat_tpsa]])[0][0])

        return torch.tensor([T_s, P_s, rc_s, rl_s, am_s, cc_s, ct_s], dtype=torch.float)

    def get_attention_scores(self, c_smi, a_smi, r_smi, T, P):
        """
        核心物理重构：反事实微扰法
        保留原函数名以兼容 Scheme A/B/C，但内部逻辑彻底改为：
        遮蔽每个原子特征 -> 观测最终溶解度预测值的绝对变化量 |Y_base - Y_mask|
        """
        G, num_bond = build_tri_graph_with_global(c_smi, a_smi, r_smi)
        if G is None: return None, None, None
            
        cond = self.compute_condition(c_smi, a_smi, r_smi, T, P)
        cond_device = cond.unsqueeze(0).to(self.device)
        
        # 1. 计算基准预测值 (Y_base)
        G_batch = Batch.from_data_list([G]).to(self.device)
        with torch.no_grad():
            y_base = self.model(G_batch, cond_device).item()
        
        num_real_atom = G.x.shape[0] - 1
        node_scores = np.zeros(num_real_atom)
        
        # 2. 依次突变每个原子计算反事实预测值 (Y_mask)
        for i in range(num_real_atom):
            G_mask = G.clone()
            # 【反事实突变】将目标原子整体强行变为标准端基氢原子 (H)
            # 彻底覆盖：[原子序数1, 杂化0, 芳香性0, 连接度1, 形式电荷1(即0), 电负性0, 半径0]
            G_mask.x[i] = torch.tensor([1, 0, 0, 1, 1, 0, 0], dtype=torch.long)
            
            G_mask_batch = Batch.from_data_list([G_mask]).to(self.device)
            
            with torch.no_grad():
                y_mask = self.model(G_mask_batch, cond_device).item()
                
            # 计算因果差值
            node_scores[i] = np.abs(y_base - y_mask)

        atom_types = G.x[:num_real_atom, 0].cpu().numpy()
        mol_type = G.mol_type[:num_real_atom].cpu().numpy()
        
        return node_scores, atom_types, mol_type

    def explain(self, title, c_smi, a_smi, r_smi, T, P, save_name):
        print(f"\n==================================================")
        print(f"🚀 开始分析: {title}")
        print(f"   [Cation]: {c_smi}")
        print(f"   [Anion]:  {a_smi}")
        print(f"   [Refri]:  {r_smi} (T={T}K, P={P}bar)")
        
        G, num_bond = build_tri_graph_with_global(c_smi, a_smi, r_smi)
        if G is None:
            print("❌ 分子图构建失败！请检查 SMILES。")
            return
            
        cond = self.compute_condition(c_smi, a_smi, r_smi, T, P)
        cond_device = cond.unsqueeze(0).to(self.device)
        
        # 1. 计算基准预测值
        G_batch = Batch.from_data_list([G]).to(self.device)
        with torch.no_grad():
            y_base = self.model(G_batch, cond_device).item()
        print(f"📊 [GNN基准预测溶解度 x1]: {y_base:.4f}")
        
        # 2. 运行反事实微扰计算
        num_real_atom = G.x.shape[0] - 1
        node_scores = np.zeros(num_real_atom)
        
        for i in range(num_real_atom):
            G_mask = G.clone()
            # 彻底覆盖：[原子序数1, 杂化0, 芳香性0, 连接度1, 形式电荷1(即0), 电负性0, 半径0]
            G_mask.x[i] = torch.tensor([1, 0, 0, 1, 1, 0, 0], dtype=torch.long)
            
            G_mask_batch = Batch.from_data_list([G_mask]).to(self.device)
            with torch.no_grad():
                y_mask = self.model(G_mask_batch, cond_device).item()
            node_scores[i] = np.abs(y_base - y_mask)

        # 为了作图美观，归一化到 0-1
        if node_scores.max() > 0:
            node_scores = node_scores / node_scores.max()

        atom_types = G.x[:num_real_atom, 0].cpu().numpy()
        mol_type = G.mol_type[:num_real_atom].cpu().numpy()
        
        # 打印排行榜
        print("\n🏆 微扰重要性排行榜 (Counterfactual Importance):")
        sorted_idx = np.argsort(node_scores)[::-1]
        print(f"{'排名':<5} | {'归属部位':<14} | {'元素':<5} | {'相对得分':<8}")
        print("-" * 45)
        
        for rank, idx in enumerate(sorted_idx):
            at_num = int(atom_types[idx])
            sym = ELEMENT_SYMBOL.get(at_num, f"Z{at_num}")
            score = node_scores[idx]
            
            if mol_type[idx] == 0: part = "阳离子(Cat)"
            elif mol_type[idx] == 1: part = "阴离子(Ani)"
            elif mol_type[idx] == 2: part = "制冷剂(Ref)"
            else: part = "未知"
                
            print(f"#{rank+1:<4} | {part:<14} | {sym:<5} | {score:.4f}")
            
        # 绘图
        g = nx.Graph()
        labels_dict = {}
        for i in range(num_real_atom):
            g.add_node(i)
            at_num = int(atom_types[i])
            sym = ELEMENT_SYMBOL.get(at_num, f"Z{at_num}")
            labels_dict[i] = f"{sym}\n({node_scores[i]:.2f})"
            
        node_a = G.edge_index[0][:num_bond].cpu().numpy()
        node_b = G.edge_index[1][:num_bond].cpu().numpy()
        for u, v in zip(node_a, node_b):
            if u < num_real_atom and v < num_real_atom:
                g.add_edge(u, v)

        plt.figure(figsize=(10, 8))
        pos = nx.kamada_kawai_layout(g)
        nx.draw_networkx_edges(g, pos, alpha=0.4, edge_color='gray', width=1.5)
        nodes = nx.draw_networkx_nodes(
            g, pos, nodelist=list(g.nodes), node_size=800,
            node_color=node_scores, cmap=plt.cm.Reds, edgecolors='black', linewidths=1.2
        )
        nx.draw_networkx_labels(g, pos, labels=labels_dict, font_size=9, font_color='black', font_weight='bold')
        
        cbar = plt.colorbar(nodes, pad=0.02, shrink=0.8)
        cbar.set_label('Counterfactual Perturbation Importance', rotation=270, labelpad=15)
        plt.title(f"{title} (Target: {r_smi})", fontsize=14, pad=15)
        plt.axis("off")
        plt.tight_layout()
        
        out_dir = os.path.join(ROOT, 'Interpretability_Case_Studies', 'Results')
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"\n✅ 分析图已保存至: {save_path}")
