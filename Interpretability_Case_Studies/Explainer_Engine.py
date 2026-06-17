import os
import sys
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Batch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))
sys.path.insert(0, ROOT)

from Model import IL_GAT
from Interpretability_Case_Studies.Dataset_explain_v2 import smiles_to_graph, combine_Graph, add_global

ELEMENT_SYMBOL = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

class Explainer_Engine:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[引擎初始化] 加载 GAT 权重: {model_path} on {self.device}")
        
        Args = {'emb_dim': 300, 'dropout_rate': 0.2, 'n_features': 7}
        self.model = IL_GAT(Args).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        self.model.eval()

        # 为了获取标准的归一?Scaler
        from Interpretability_Case_Studies.Dataset_explain_v2 import IL_set_v2
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

    def _build_strict_graph(self, c_smi, a_smi, r_smi):
        c_data = smiles_to_graph(c_smi)
        a_data = smiles_to_graph(a_smi)
        r_data = smiles_to_graph(r_smi)
        if not (c_data and a_data and r_data): return None, 0
        
        # 注意：batch 索引由 combine_Graph() 内部的 Batch.from_data_list() 自动生成
        # 阳离子=0, 阴离子=1, 制冷剂=2，顺序由传参顺序决定
        
        num_real_bonds = c_data.edge_index.shape[1] + a_data.edge_index.shape[1] + r_data.edge_index.shape[1]
        combined = combine_Graph([c_data, a_data, r_data])
        G = add_global(combined)
        return G, num_real_bonds

    def _get_embeddings(self, G):
        with torch.no_grad():
            x = G.x.to(self.device)
            h = self.model.x_embedding1(x[:, 0]) + \
                self.model.x_embedding2(x[:, 1]) + \
                self.model.x_embedding3(x[:, 2]) + \
                self.model.x_embedding4(x[:, 3]) + \
                self.model.x_embedding5(x[:, 4]) + \
                self.model.x_embedding6(x[:, 5]) + \
                self.model.x_embedding7(x[:, 6])
            return h

    def ig_attribution(self, G_batch, cond_device, num_real_atom, steps=50):
        # 1. 提取 baseline ?input 对应的连?Embedding
        # IG 基线：全零向量（代表完全没有特征输入的虚无状态）
        h_input = self._get_embeddings(G_batch).detach()
        h_baseline = torch.zeros_like(h_input)
        
        # 准备注入 hook
        h_interp_current = None
        def pre_hook(module, args):
            # args[0] 是输入到 l1 ?x，args[1] ?edge_index
            # 我们?args[0] 强制替换为我们正在做 IG 积分的当前步?h_interp
            return (h_interp_current, args[1])
            
        handle = self.model.l1.register_forward_pre_hook(pre_hook)
        
        integrated_grads = torch.zeros_like(h_input)
        
        # 2. 黎曼和积分
        for step in range(1, steps + 1):
            alpha = step / steps
            # 线性插值
            h_interp = h_baseline + alpha * (h_input - h_baseline)
            h_interp = h_interp.clone().detach().requires_grad_(True)
            
            # 将插值注入
            h_interp_current = h_interp
            
            # 前向传播（l1 的输入会被 pre_hook 替换）
            # 注意：传 G_batch 作为外壳，以提供 edge_index、batch 等拓扑信息
            out = self.model(G_batch, cond_device)
            
            # 梯度回传
            self.model.zero_grad()
            out.backward()
            
            # 累加梯度
            integrated_grads += h_interp.grad
            
        handle.remove()
        
        # 3. 计算最终的 IG 归因 = (input - baseline) * avg_grad
        avg_grads = integrated_grads / steps
        attributions = (h_input - h_baseline) * avg_grads
        
        # 对于每个原子，将其所有特征维度上的归因求和
        node_scores = attributions[:num_real_atom].sum(dim=1).cpu().detach().numpy()
        
        # IG 可能会有负数（代表阻碍吸收），为了画图（重要性），取绝对值
        return np.abs(node_scores)

    def get_attention_scores(self, c_smi, a_smi, r_smi, T, P):
        G, num_bond = self._build_strict_graph(c_smi, a_smi, r_smi)
        if G is None: return None, None, None
            
        cond = self.compute_condition(c_smi, a_smi, r_smi, T, P)
        cond_device = cond.unsqueeze(0).to(self.device)
        G_batch = Batch.from_data_list([G]).to(self.device)
        
        num_real_atom = G.x.shape[0] - 1
        
        # 默认使用绝对严谨?IG 算法
        node_scores = self.ig_attribution(G_batch, cond_device, num_real_atom, steps=50)

        atom_types = G.x[:num_real_atom, 0].cpu().numpy()
        mol_type = G.mol_type[:num_real_atom].cpu().numpy()
        
        return node_scores, atom_types, mol_type

    def explain(self, title, c_smi, a_smi, r_smi, T, P, save_name):
        print(f"\n==================================================")
        print(f"🚀 开始科学归因分 {title}")
        print(f"   [引擎]: Integrated Gradients (IG)")
        print(f"   [Cation]: {c_smi}")
        print(f"   [Anion]:  {a_smi}")
        print(f"   [Refri]:  {r_smi} (T={T}K, P={P}bar)")
        
        G, num_bond = self._build_strict_graph(c_smi, a_smi, r_smi)
        if G is None: return
            
        cond = self.compute_condition(c_smi, a_smi, r_smi, T, P)
        cond_device = cond.unsqueeze(0).to(self.device)
        G_batch = Batch.from_data_list([G]).to(self.device)
        
        with torch.no_grad():
            y_base = self.model(G_batch, cond_device).item()
        print(f"📊 [GNN基准预测溶解x1]: {y_base:.4f}")
        
        num_real_atom = G.x.shape[0] - 1
        node_scores = self.ig_attribution(G_batch, cond_device, num_real_atom, steps=50)

        if node_scores.max() > 0:
            node_scores = node_scores / node_scores.max()

        atom_types = G.x[:num_real_atom, 0].cpu().numpy()
        mol_type = G.mol_type[:num_real_atom].cpu().numpy()
        
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
            
        # 构建 1x3 的三分屏画布
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        titles = ["Cation", "Anion", "Refrigerant"]
        
        subgraphs = {0: nx.Graph(), 1: nx.Graph(), 2: nx.Graph()}
        labels_dict = {0: {}, 1: {}, 2: {}}
        scores_dict = {0: [], 1: [], 2: []}
        nodelist_dict = {0: [], 1: [], 2: []}
        
        for i in range(num_real_atom):
            m_type = int(mol_type[i])
            if m_type in subgraphs:
                subgraphs[m_type].add_node(i)
                at_num = int(atom_types[i])
                sym = ELEMENT_SYMBOL.get(at_num, f"Z{at_num}")
                labels_dict[m_type][i] = f"{sym}\n({node_scores[i]:.2f})"
                scores_dict[m_type].append(node_scores[i])
                nodelist_dict[m_type].append(i)
                
        node_a = G.edge_index[0][:num_bond].cpu().numpy()
        node_b = G.edge_index[1][:num_bond].cpu().numpy()
        for u, v in zip(node_a, node_b):
            if u < num_real_atom and v < num_real_atom:
                if mol_type[u] == mol_type[v] and mol_type[u] in subgraphs:
                    subgraphs[mol_type[u]].add_edge(u, v)

        vmax = node_scores.max() if node_scores.max() > 0 else 1.0
        vmin = 0.0
        
        nodes_ref = None
        for m_type in [0, 1, 2]:
            ax = axes[m_type]
            g_sub = subgraphs[m_type]
            
            if len(g_sub.nodes) == 0:
                ax.axis("off")
                continue
                
            pos = nx.kamada_kawai_layout(g_sub)
            nx.draw_networkx_edges(g_sub, pos, ax=ax, alpha=0.4, edge_color='gray', width=1.5)
            nodes = nx.draw_networkx_nodes(
                g_sub, pos, nodelist=nodelist_dict[m_type], ax=ax, node_size=800,
                node_color=scores_dict[m_type], cmap=plt.cm.Reds, vmin=vmin, vmax=vmax,
                edgecolors='black', linewidths=1.2
            )
            nx.draw_networkx_labels(g_sub, pos, labels=labels_dict[m_type], ax=ax, font_size=10, font_color='black', font_weight='bold')
            ax.set_title(titles[m_type], fontsize=16, pad=10, weight='bold')
            ax.axis("off")
            if nodes_ref is None: nodes_ref = nodes
            
        fig.subplots_adjust(right=0.9, top=0.85, wspace=0.1)
        if nodes_ref is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(nodes_ref, cax=cbar_ax)
            cbar.set_label('IG Attribution Importance', rotation=270, labelpad=15)
        
        plt.suptitle(f"{title} (Target: {r_smi})", fontsize=20, weight='bold')
        
        out_dir = os.path.join(ROOT, 'Interpretability_Case_Studies', 'Results')
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
