import os
import sys
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'Interpretability_Case_Studies'))

from Explainer_Engine import Explainer_Engine, ELEMENT_SYMBOL
from smarts_dict import get_group_matches
from torch_geometric.data import Batch

def plot_occlusion_heatmap(title, c_smi, a_smi, r_smi, T, P, save_name):
    model_path = os.path.join(ROOT, 'GNN_for_property_prediction', 'checkpoints_v2', 'best_gat_seed_1.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    explainer = Explainer_Engine(model_path)
    model = explainer.model
    device = explainer.device
    
    print(f"\n==================================================")
    print(f"🎨 开始物理敲除法可解释性绘图: {title}")
    print(f"   [Cation]: {c_smi}")
    print(f"   [Anion]:  {a_smi}")
    print(f"   [Refri]:  {r_smi} (T={T}K, P={P}bar)")
    
    G, num_bond = explainer._build_strict_graph(c_smi, a_smi, r_smi)
    if G is None: return
        
    cond = explainer.compute_condition(c_smi, a_smi, r_smi, T, P)
    cond_device = cond.unsqueeze(0).to(device)
    G_batch = Batch.from_data_list([G]).to(device)
    
    # Baseline
    model.eval()
    with torch.no_grad():
        y_base = model(G_batch, cond_device).item()
        
    print(f"📊 [GNN基准预测溶解度(x1)]: {y_base:.4f}")
    
    def get_occluded_prediction(target_indices):
        h_input = explainer._get_embeddings(G_batch).detach()
        h_mod = h_input.clone()
        for idx in target_indices:
            h_mod[idx] = 0.0 # Knock out
            
        def pre_hook(module, args):
            return (h_mod, args[1])
        handle = model.l1.register_forward_pre_hook(pre_hook)
        
        with torch.no_grad():
            out = model(G_batch, cond_device).item()
            
        handle.remove()
        return out

    try:
        m_c = Chem.MolFromSmiles(c_smi)
        m_a = Chem.MolFromSmiles(a_smi)
        m_r = Chem.MolFromSmiles(r_smi)
        n_c = m_c.GetNumAtoms()
        n_a = m_a.GetNumAtoms()
    except:
        return
        
    c_groups = get_group_matches(c_smi)
    a_groups = get_group_matches(a_smi)
    r_groups = get_group_matches(r_smi)
    
    combined_groups = {}
    for g_name, indices in c_groups.items():
        combined_groups[f"Cat_{g_name}"] = indices
    for g_name, indices in a_groups.items():
        combined_groups[f"Ani_{g_name}"] = [i + n_c for i in indices]
    for g_name, indices in r_groups.items():
        combined_groups[f"Ref_{g_name}"] = [i + n_c + n_a for i in indices]
        
    num_real_atom = G.x.shape[0] - 1
    atom_scores = np.zeros(num_real_atom)
    
    print("\n[敲除法基团掉分记录]")
    for g_name, indices in combined_groups.items():
        if not indices: continue
        y_occ = get_occluded_prediction(indices)
        drop = y_base - y_occ
        print(f"  - {g_name:<20}: Drop = {drop:+.4f}")
        
        # 将掉分赋给该基团的所有原子
        # 如果 drop < 0 (敲除反而升高)，我们画图时可以截断为 0，或者保留负值。这里为了展示驱动力，我们截断负值或取绝对值
        # 物理上，drop > 0 表示该基团促进了吸收。
        val = max(drop, 0.0) 
        for idx in indices:
            atom_scores[idx] = val
            
    # Normalize scores for coloring
    if atom_scores.max() > 0:
        atom_scores = atom_scores / atom_scores.max()
        
    atom_types = G.x[:num_real_atom, 0].cpu().numpy()
    mol_type = G.mol_type[:num_real_atom].cpu().numpy()
    
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
            labels_dict[m_type][i] = f"{sym}\n({atom_scores[i]:.2f})"
            scores_dict[m_type].append(atom_scores[i])
            nodelist_dict[m_type].append(i)
            
    node_a = G.edge_index[0][:num_bond].cpu().numpy()
    node_b = G.edge_index[1][:num_bond].cpu().numpy()
    for u, v in zip(node_a, node_b):
        if u < num_real_atom and v < num_real_atom:
            if mol_type[u] == mol_type[v] and mol_type[u] in subgraphs:
                subgraphs[mol_type[u]].add_edge(u, v)
                
    vmax = atom_scores.max() if atom_scores.max() > 0 else 1.0
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
        cbar.set_label('Thermodynamic Importance (Drop in Prediction)', rotation=270, labelpad=15)
    
    plt.suptitle(f"{title} (Occlusion Method)", fontsize=20, weight='bold')
    
    out_dir = os.path.join(ROOT, 'scripts_phase3', 'Results')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'{save_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 热力图已保存至: {save_path}")

def main():
    # 典型案例测试
    plot_occlusion_heatmap(
        title="BMIM-Tf2N absorbing R1234yf",
        c_smi="CCCC[n+]1cccc(C)c1",
        a_smi="FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F",
        r_smi="C(=C(F)F)(C(F)(F)F)F",
        T=298.15, P=1.0,
        save_name="Scheme_A_Tf2N_Occlusion"
    )
    
    plot_occlusion_heatmap(
        title="BMIM-BF4 absorbing R1234yf",
        c_smi="CCCC[n+]1cccc(C)c1",
        a_smi="[B-](F)(F)(F)F",
        r_smi="C(=C(F)F)(C(F)(F)F)F",
        T=298.15, P=1.0,
        save_name="Scheme_A_BF4_Occlusion"
    )

if __name__ == '__main__':
    main()
