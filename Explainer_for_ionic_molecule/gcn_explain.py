"""
gcn_explain.py
===================================================
制冷剂-离子液体三图体系 单分子结构重要性热力图可视化 - GCN专版

主要功能：
  1. 通过命令行参数指定任意样本索引（--sample_idx）
  2. 对该样本加载 GCN 权重视角，运行 GNNExplainer 掩码优化
  3. 基于原子重要性分数绘制该分子的 2D 骨架热力图（重要原子呈红色渐变）
  4. 自动标注原子化学符号（如 C, F, O, N）

使用示例：
  python gcn_explain.py --sample_idx 100 --model_path ../GNN_for_property_prediction/checkpoints/best_gcn_seed_42.pth
"""

import os
import sys
import argparse
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Batch

# ── 将 GNN_for_property_prediction 加入搜索路径 ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))

from Model import IL_Net_GCN
from Explainer_v2 import IL_Explainer_v2
from Dataset import IL_set  # GCN 是在原始 V1 数据集上训练的，所以这里必须用 V1 数据集

# GCN 模型超参数
Args = {
    'emb_dim': 300,
    'dropout_rate': 0.4, # GCN Runner 中使用的 dropout 是 0.4
}

ELEMENT_SYMBOL = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

def main(sample_idx: int, model_path: str, data_root: str, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[运行设备] {device}")
    
    # ── 1. 加载模型 ──────────────────────────
    print(f"[加载模型]: {model_path}")
    model = IL_Net_GCN(Args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # ── 2. 加载数据集 ──────────────────────────
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"❌ 数据文件夹不存在: {data_root}")
        
    dataset = IL_set(path=data_root)
    
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"❌ 样本索引越界！合理范围为 0 到 {len(dataset)-1}")
        
    print(f"[提取样本] 索引号: {sample_idx}")
    G, cond, label = dataset[sample_idx]
    
    # 获取原始分子图的键数
    # Dataset.py 中的 V1 并没有返回 num_bonds_list，我们需要自己算
    # 阳离子图、阴离子图、制冷剂图 在拼接前的边数
    # 其实也可以直接通过查看 batch==2 之前的边的数量来分离，但 GNNExplainer 是解释整个大图。
    
    # ── 3. 运行 Explainer ──────────────────────
    print(f"[计算解释] 正在以 GCN 视角优化掩码，共 {epochs} 轮...")
    explainer = IL_Explainer_v2(model, epochs=epochs, lr=0.01)
    
    # 临时打个 Batch 维度以喂给 Explainer
    G_batch = Batch.from_data_list([G]).to(device)
    cond_batch = cond.unsqueeze(0).to(device)
    
    node_feat_mask, edge_mask = explainer.explain_graph(G_batch, cond_batch)
    
    # ── 4. 解析原子重要性 ──────────────────────
    # 真实原子个数：全局图由于加入了 self_loops 和虚拟全局节点，稍微复杂。
    # 我们知道 G.x 包含所有真实原子，最后的几个是全局节点。
    # 这里直接提取 G.x 中 batch==0,1,2 对应的原子，忽略虚拟节点。
    real_nodes = (G.batch < 3).nonzero().view(-1)
    num_real_atom = len(real_nodes)
    
    # 将特征掩码和节点对应起来（node_feat_mask 只是一维特征权重，无法映射到具体节点上）
    # 在这个版本中，我们需要根据 GNNExplainer V2 算出的 edge_mask，把相连边的分数汇聚到节点上。
    # 提取有向边
    edge_idx = G.edge_index.cpu().numpy()
    edge_imp = edge_mask.cpu().numpy()
    
    atom_imp = np.zeros(num_real_atom)
    for i in range(len(edge_imp)):
        u, v = edge_idx[0, i], edge_idx[1, i]
        if u < num_real_atom and v < num_real_atom:
            # 每条边的重要性分一半给两头的原子
            atom_imp[u] += edge_imp[i] * 0.5
            atom_imp[v] += edge_imp[i] * 0.5

    # 归一化原子得分
    if atom_imp.max() > 0:
        atom_imp = atom_imp / atom_imp.max()
        
    # 获取原子类型 (原子序数)
    atom_types = G.x[:num_real_atom, 0].cpu().numpy()
    
    # ── 5. 构筑 NetworkX 分子图 ────────────────
    g = nx.Graph()
    
    # 添加原子节点并记录其化学符号作为展示 Label
    labels = {}
    for i in range(num_real_atom):
        g.add_node(i)
        at_num = int(atom_types[i])
        symbol = ELEMENT_SYMBOL.get(at_num, f"Z{at_num}")
        labels[i] = f"{symbol}\n({i})"
        
    # 添加原始化学键连线
    for u, v in zip(edge_idx[0], edge_idx[1]):
        if u < num_real_atom and v < num_real_atom:
            g.add_edge(u, v)
            
    # ── 6. 绘图与保存 ──────────────────────────
    plt.figure(figsize=(10, 8))
    
    pos = nx.kamada_kawai_layout(g)
    nx.draw_networkx_edges(g, pos, alpha=0.3, edge_color='gray', width=1.5)
    
    nodes = nx.draw_networkx_nodes(
        g, pos,
        nodelist=list(g.nodes),
        node_size=600,
        node_color=atom_imp,
        cmap=plt.cm.Reds,
        edgecolors='black',
        linewidths=0.8
    )
    
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, font_color='black', font_weight='bold')
    
    cbar = plt.colorbar(nodes, pad=0.02, shrink=0.8)
    cbar.set_label('Atom Importance Score (GCN-Explainer)', rotation=270, labelpad=15, fontsize=10)
    
    # V1 数据集中，T 是原始值，直接读取 cond[0]
    T = float(cond[0])
    P = float(cond[1])
    
    plt.title(
        f"GCN Atom Importance Heatmap for Sample {sample_idx}\n"
        f"Experimental Condition: T = {T:.1f} K, P = {P:.3f} MPa | Pred Solubility x1 = {label:.4f}",
        fontsize=11, pad=15
    )
    plt.axis("off")
    plt.tight_layout()
    
    out_dir = os.path.join(os.path.dirname(__file__), 'fragment_explain_result')
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, f'gcn_single_hotmap_idx_{sample_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 [成功] GCN单分子结构重要性热力图已绘制并保存至:\n   {save_path}")
    print("\n💡 [分析小结] GCN 认为本体系中重要性最高的前 5 个原子为:")
    sorted_idx = np.argsort(atom_imp)[::-1]
    for rank, idx in enumerate(sorted_idx[:5]):
        if rank >= len(atom_imp): break
        at_num = int(atom_types[idx])
        sym = ELEMENT_SYMBOL.get(at_num, f"Z{at_num}")
        print(f"   第 {rank+1} 名: 原子编号 {idx:2d} ({sym:3s}) | 贡献得分 = {atom_imp[idx]:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='制冷剂体系单样本 GCN-Explainer 结构热力图分析')
    parser.add_argument('--sample_idx', type=int, default=100, help='需要分析的样本索引，范围 0~4443')
    parser.add_argument('--model_path', type=str, default=os.path.join(ROOT, 'GNN_for_property_prediction', 'checkpoints', 'best_gcn_seed_42.pth'), help='GCN 的权重路径')
    parser.add_argument('--data_root', type=str, default=os.path.join(ROOT, 'processed_tri_data/'))
    parser.add_argument('--epochs', type=int, default=100, help='解释掩码优化轮数')
    args_cli = parser.parse_args()
    
    main(args_cli.sample_idx, args_cli.model_path, args_cli.data_root, args_cli.epochs)
