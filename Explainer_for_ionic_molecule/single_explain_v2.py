"""
single_explain_v2.py
===================================================
制冷剂-离子液体三图体系 单分子结构重要性热力图可视化 (V2)

主要功能：
  1. 通过命令行参数指定任意样本索引（--sample_idx）
  2. 对该样本运行 GNNExplainer 掩码优化
  3. 基于原子重要性分数绘制该分子的 2D 骨架热力图（重要原子呈红色渐变）
  4. 自动标注原子化学符号（如 C, F, O, N），无需修改任何代码！

使用示例：
  python single_explain_v2.py --sample_idx 100 --model_path ../checkpoints/best_seed_1.pth
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

from Model import GIN
from Explainer_v2 import IL_Explainer_v2
from Dataset_explain_v2 import IL_set_v2

# GIN 模型超参数
Args = {
    'num_gin_layer': 5,
    'emb_dim': 300,
    'feat_dim': 512,
    'drop_ratio': 0.2,
    'pool': 'mean',
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
    model = GIN(Args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # ── 2. 加载数据集 ──────────────────────────
    data_npy  = os.path.join(data_root, 'data.npy')
    label_npy = os.path.join(data_root, 'label.npy')
    
    if not os.path.exists(data_npy):
        raise FileNotFoundError(
            f"❌ 数据文件不存在: {data_npy}\n"
            f"   请在 GNN 根目录下运行 `python prepare_tri_graph_data.py` 重新生成制冷剂三图数据集！"
        )
        
    dataset = IL_set_v2(data_npy_path=data_npy, label_npy_path=label_npy)
    
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"❌ 样本索引越界！合理范围为 0 到 {len(dataset)-1}")
        
    print(f"[提取样本] 索引号: {sample_idx}")
    G, cond, label, num_bonds_list = dataset[sample_idx]
    num_bond = num_bonds_list
    
    # ── 3. 运行 Explainer ──────────────────────
    print(f"[计算解释] 正在优化掩码，共 {epochs} 轮...")
    explainer = IL_Explainer_v2(model, epochs=epochs, lr=0.01)
    
    # 临时打个 Batch 维度以喂给 Explainer
    G_batch = Batch.from_data_list([G]).to(device)
    cond_batch = cond.unsqueeze(0).to(device)
    
    node_feat_mask, edge_mask = explainer.explain_graph(G_batch, cond_batch)
    
    # ── 4. 解析原子重要性 ──────────────────────
    global_mask = edge_mask[num_bond:].cpu()
    num_real_atom = global_mask.shape[0] // 2
    
    fwd = global_mask[:num_real_atom]
    bwd = global_mask[num_real_atom:]
    atom_imp = ((fwd + bwd) / 2).numpy() # shape: [num_real_atom]
    
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
        
    # 添加原始化学键连线 (忽略全局虚拟边)
    node_a = G.edge_index[0][:num_bond].cpu().numpy()
    node_b = G.edge_index[1][:num_bond].cpu().numpy()
    for u, v in zip(node_a, node_b):
        if u < num_real_atom and v < num_real_atom:
            g.add_edge(u, v)
            
    # ── 6. 绘图与保存 ──────────────────────────
    plt.figure(figsize=(10, 8))
    
    # 使用 Kamada-Kawai 算法进行分子骨架排版，比 Spring 算法更规整漂亮
    pos = nx.kamada_kawai_layout(g)
    
    # 绘制化学键边
    nx.draw_networkx_edges(g, pos, alpha=0.3, edge_color='gray', width=1.5)
    
    # 绘制原子节点 (使用 Reds 渐变色盘，越红代表重要性越高)
    nodes = nx.draw_networkx_nodes(
        g, pos,
        nodelist=list(g.nodes),
        node_size=600,
        node_color=atom_imp,
        cmap=plt.cm.Reds,
        edgecolors='black',
        linewidths=0.8
    )
    
    # 标注化学元素符号和节点编号
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, font_color='black', font_weight='bold')
    
    # 伴随色带
    cbar = plt.colorbar(nodes, pad=0.02, shrink=0.8)
    cbar.set_label('Atom Importance Score (GNNExplainer)', rotation=270, labelpad=15, fontsize=10)
    
    # 还原原本的温度与压力（反归一化或直接提取）
    T = float(cond[0])
    P = float(cond[1])
    
    plt.title(
        f"Atom Importance Heatmap for Sample {sample_idx}\n"
        f"Experimental Condition: T = {T:.1f} K, P = {P:.3f} MPa | Pred Solubility x1 = {label:.4f}",
        fontsize=11, pad=15
    )
    plt.axis("off")
    plt.tight_layout()
    
    # 创建结果文件夹
    out_dir = os.path.join(os.path.dirname(__file__), 'fragment_explain_result')
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, f'single_hotmap_idx_{sample_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 [成功] 单分子结构重要性热力图已绘制并保存至:\n   {save_path}")
    print("\n💡 [分析小结] 本体系中重要性最高的前 5 个原子为:")
    sorted_idx = np.argsort(atom_imp)[::-1]
    for rank, idx in enumerate(sorted_idx[:5]):
        at_num = int(atom_types[idx])
        sym = ELEMENT_SYMBOL.get(at_num, f"Z{at_num}")
        print(f"   第 {rank+1} 名: 原子编号 {idx:2d} ({sym:3s}) | 贡献得分 = {atom_imp[idx]:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='制冷剂体系单样本 GNNExplainer 结构热力图分析 V2')
    parser.add_argument('--sample_idx', type=int, default=100, help='需要分析的样本索引，范围 0~4443')
    parser.add_argument('--model_path', type=str, default=os.path.join(ROOT, 'checkpoints', 'best_seed_1.pth'))
    parser.add_argument('--data_root', type=str, default=os.path.join(ROOT, 'processed_tri_data'))
    parser.add_argument('--epochs', type=int, default=100, help='解释掩码优化轮数')
    args_cli = parser.parse_args()
    
    main(args_cli.sample_idx, args_cli.model_path, args_cli.data_root, args_cli.epochs)
