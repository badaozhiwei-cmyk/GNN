import os
import sys
import torch
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Batch

# ── 引入上级目录的模块 ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))

from Model import IL_GAT
from Dataset_explain_v2 import IL_set_v2

ELEMENT_SYMBOL = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_idx', type=int, default=100, help='分析的样本序号')
    parser.add_argument('--model_path', type=str, required=True, help='GAT 权重路径')
    args_cli = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 [设备] {device}")
    
    # ── 1. 载入数据集 ──
    data_path = os.path.join(ROOT, 'processed_tri_data')
    dataset = IL_set_v2(data_npy_path=os.path.join(data_path, 'data.npy'), 
                        label_npy_path=os.path.join(data_path, 'label.npy'))
    
    G, cond, label, num_bonds_list = dataset[args_cli.sample_idx]
    num_bond = num_bonds_list

    # ── 2. 加载 GAT 模型 ──
    Args = {'emb_dim': 300, 'dropout_rate': 0.2, 'n_features': 7}
    model = IL_GAT(Args).to(device)
    ckpt = torch.load(args_cli.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    # ── 3. 使用 PyTorch Hook 窃取 Attention Weights ──
    # 彻底告别梯度黑盒！直接捕获网络内部流动的注意力矩阵
    attentions = {}
    def hook_fn(layer_name):
        def hook(module, inp, out):
            # out 是 GATv2Conv 的返回元组: (x, (edge_index, alpha))
            attentions[layer_name] = out[1]
        return hook

    model.l1.register_forward_hook(hook_fn('l1'))
    model.l2.register_forward_hook(hook_fn('l2'))
    model.l3.register_forward_hook(hook_fn('l3'))

    # ── 4. 前向传播 (仅运行一次) ──
    # 将单张图转为 Batch 格式送入网络
    G_batch = Batch.from_data_list([G]).to(device)
    cond_device = cond.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(G_batch, cond_device)
        
    print(f"📊 [预测溶解度] {pred.item():.4f}  |  [真实溶解度] {label:.4f}")

    # ── 5. 将注意力分配回原子节点 ──
    # 提取最深层 (l3) 的注意力，代表网络对最终预测做出的高级决策
    edge_index, alpha = attentions['l3']
    edge_index = edge_index.cpu()
    alpha = alpha.cpu().mean(dim=-1).numpy() # 平均多头注意力

    # 最后一个节点为虚拟节点（汇聚图级别特征使用），不参与化学展示
    num_real_atom = G.x.shape[0] - 1
    if num_real_atom <= 0:
        num_real_atom = G.x.shape[0]
        
    node_scores = np.zeros(num_real_atom)
    
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        # 注意力 alpha_{u,v} 表示节点 u 把注意力投射给节点 v 的概率
        # 因此，节点 v 被注视的次数越多，其中心地位越高
        if u < num_real_atom and v < num_real_atom:
            node_scores[v] += alpha[i]

    # 将最高得分归一化到 1.0 方便颜色映射
    if node_scores.max() > 0:
        node_scores = node_scores / node_scores.max()

    atom_types = G.x[:num_real_atom, 0].cpu().numpy()

    # ── 6. 绘制分子骨架热力图 ──
    g = nx.Graph()
    labels_dict = {}
    for i in range(num_real_atom):
        g.add_node(i)
        at_num = int(atom_types[i])
        sym = ELEMENT_SYMBOL.get(at_num, f"Z{at_num}")
        labels_dict[i] = f"{sym}\n({node_scores[i]:.2f})"
        
    # 添加分子化学键 (只遍历真实边，排除全局虚拟边)
    node_a = G.edge_index[0][:num_bond].cpu().numpy()
    node_b = G.edge_index[1][:num_bond].cpu().numpy()
    for u, v in zip(node_a, node_b):
        if u < num_real_atom and v < num_real_atom:
            g.add_edge(u, v)

    plt.figure(figsize=(10, 8))
    pos = nx.kamada_kawai_layout(g) # 自动力学分子布局
    
    nx.draw_networkx_edges(g, pos, alpha=0.4, edge_color='gray', width=1.5)
    
    nodes = nx.draw_networkx_nodes(
        g, pos,
        nodelist=list(g.nodes),
        node_size=800,
        node_color=node_scores,
        cmap=plt.cm.Reds,
        edgecolors='black',
        linewidths=1.2
    )
    
    nx.draw_networkx_labels(g, pos, labels=labels_dict, font_size=9, font_color='black', font_weight='bold')
    
    cbar = plt.colorbar(nodes, pad=0.02, shrink=0.8)
    cbar.set_label('GAT Native Attention Weight (Normalized)', rotation=270, labelpad=15)
    
    plt.title(f"GAT Native Attention - Real Insight of the Model\n(Sample {args_cli.sample_idx})", fontsize=14, pad=15)
    plt.axis("off")
    plt.tight_layout()
    
    out_dir = os.path.join(ROOT, 'Explainer_for_ionic_molecule', 'fragment_explain_result')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'gat_attention_idx_{args_cli.sample_idx}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\n✅ [出图] 纯天然的 GAT 注意力图已生成:\n   => {save_path}")
    
    print("\n🏆 GAT 网络“视线”最聚焦的 TOP 5 核心原子:")
    sorted_idx = np.argsort(node_scores)[::-1]
    for rank, idx in enumerate(sorted_idx[:5]):
        at_num = int(atom_types[idx])
        sym = ELEMENT_SYMBOL.get(at_num, f"Z{at_num}")
        print(f"   第 {rank+1} 名: 【{sym:2s}】(原子编号 {idx:2d}) | 相对吸引力得分 = {node_scores[idx]:.4f}")

if __name__ == '__main__':
    main()
