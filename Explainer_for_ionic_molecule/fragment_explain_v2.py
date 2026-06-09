"""
fragment_explain_v2.py
===================================================
制冷剂-离子液体三图体系 GNNExplainer 片段/元素重要性分析 (V2 修复版)

与旧版相比的三大算法修复：
  1. 融入真实化学键边掩码: edge_mask[:num_bond] 不再被丢弃
  2. 分子内归一化: 每个原子的重要性与同一样本内的平均値比较而非全数据集均値
  3. 子图分离归因: 阳离子/阴离子/制冷剂中的相同元素分开统计贡献度
使用方式：
  cd Explainer_for_ionic_molecule
  python fragment_explain_v2.py --model_path ../checkpoints/best_seed_1.pth --num_samples 500
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import DataLoader

# ── 将 GNN_for_property_prediction 加入搜索路径 ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))

from Model import GIN
from Explainer_v2 import IL_Explainer_v2
from Dataset_explain_v2 import IL_set_v2

# GIN 模型超参数（与训练/预测时完全一致）
Args = {
    'num_gin_layer': 5,
    'emb_dim': 300,
    'feat_dim': 512,
    'drop_ratio': 0.2,
    'pool': 'mean',
}

# 按照元素序数分类汇总（对应 G.x[:, 0]）
# C=6, N=7, O=8, F=9, P=15, S=16, Cl=17, Br=35, B=5, I=53
ELEMENT_MAP = {
    5:  'B (Boron, e.g., BF4-)',
    6:  'C (Carbon skeleton / alkyl chain)',
    7:  'N (Nitrogen, e.g., Imidazole/Tf2N-)',
    8:  'O (Oxygen, e.g., Sulfonate/Carbonyl)',
    9:  'F (Fluorine, key polar atom)',
    15: 'P (Phosphorus, e.g., Phosphonium)',
    16: 'S (Sulfur, e.g., Sulfonyl group)',
    17: 'Cl (Chlorine, e.g., HCFCs/CFCs)',
    35: 'Br (Bromine)',
    53: 'I (Iodine)',
}


def plot_element_importance(result: dict, save_path: str):
    """
    绘制按元素分组的原子重要性相对平均贡献度水平条形图
    """
    # 过滤掉没有在当前数据集中出现的元素 (Nan值)
    clean_result = {k: v for k, v in result.items() if not np.isnan(v)}
    if not clean_result:
        print("  [提示] 没有收集到有效的元素贡献数据，跳过画图")
        return
    
    # 按照重要性分数升序排序，使条形图从上往下呈现梯度
    sorted_items = sorted(clean_result.items(), key=lambda x: x[1])
    elements = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    plt.figure(figsize=(9, 6))
    
    # 使用渐变色，分数越高的颜色越深/亮
    import matplotlib.cm as cm
    max_score = max(scores) if max(scores) > 0 else 1.0
    colors = cm.viridis(np.array(scores) / max_score)
    
    plt.barh(elements, scores, color=colors, height=0.6, edgecolor='black', linewidth=0.5)
    plt.xlabel('Absolute Importance Score (GNNExplainer Mask)', fontsize=12)
    plt.ylabel('Chemical Elements', fontsize=12)
    plt.title('Element-Level Absolute Importance on Solubility\n(Tri-Graph Refrigerant V2)', fontsize=13, pad=15)
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [图表] 已保存元素重要性贡献图: {save_path}")


def main(model_path: str, data_root: str, explainer_epochs: int = 100, num_samples: int = -1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[运行设备] {device}")

    # ── 1. 加载模型（带状态字典自适应） ──────────────────────────
    print(f"[加载模型] 路径: {model_path}")
    model = GIN(Args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # ── 2. 加载数据集 ──────────────────────────────────────────
    data_npy  = os.path.join(data_root, 'data.npy')
    label_npy = os.path.join(data_root, 'label.npy')
    dataset = IL_set_v2(data_npy_path=data_npy, label_npy_path=label_npy)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=IL_set_v2.collate_fn
    )

    # ── 3. 初始化 Explainer_v2（回归专用版） ────────────────────
    explainer = IL_Explainer_v2(
        model, epochs=explainer_epochs, lr=0.01
    )

    # ── 4. 主循环：逐样本推演解释 ───────────────────────────────
    # V2 改进：按子图角色分离汇总（阳离子/阴离子/制冷剂 分开记录）
    role_keys = ['cat', 'ani', 'ref']
    elem_imp = {role: {v: [] for v in ELEMENT_MAP.values()} for role in role_keys}
    node_feat_imp = np.zeros(7)

    total_to_run = len(loader) if num_samples <= 0 else min(num_samples, len(loader))
    print(f"[数据规模] 总数据量: {len(loader)} 条，本次分析样本数上限: {total_to_run} 条")
    bar = tqdm(total=total_to_run, desc='Explaining', dynamic_ncols=True)

    explained_count = 0
    for G, cond, label, num_bonds_list in loader:
        if num_samples > 0 and explained_count >= num_samples:
            break

        G    = G.to(device)
        cond = cond.to(device)
        num_bond = num_bonds_list[0]

        try:
            node_feat_mask, edge_mask = explainer.explain_graph(G, cond)
        except Exception as e:
            bar.update()
            continue

        explained_count += 1
        node_feat_imp += node_feat_mask.cpu().numpy()

        # ==================================================================
        # 修复点1：融入真实化学键边掩码 + 全局边掩码
        # ==================================================================
        edge_mask_cpu = edge_mask.cpu()
        num_real_atom = G.x.shape[0] - 1 

        bond_mask = edge_mask_cpu[:num_bond]
        edge_index_cpu = G.edge_index.cpu()
        bond_atom_score = np.zeros(num_real_atom)
        bond_atom_count = np.zeros(num_real_atom, dtype=int)
        for e_idx in range(num_bond):
            src = int(edge_index_cpu[0, e_idx])
            dst = int(edge_index_cpu[1, e_idx])
            score = float(bond_mask[e_idx])
            if src < num_real_atom:
                bond_atom_score[src] += score
                bond_atom_count[src] += 1
            if dst < num_real_atom:
                bond_atom_score[dst] += score
                bond_atom_count[dst] += 1
        bond_atom_avg = np.where(bond_atom_count > 0, bond_atom_score / bond_atom_count, 0.0)

        global_mask = edge_mask_cpu[num_bond:]
        fwd = global_mask[:num_real_atom].numpy()
        bwd = global_mask[num_real_atom:].numpy()
        global_atom_score = (fwd + bwd) / 2.0
        atom_imp = 0.5 * bond_atom_avg + 0.5 * global_atom_score

        # ==================================================================
        # 修复点2：分子内归一化 (修正：直接使用掩码分数，不再减去平均值导致支柱元素归零)
        # ==================================================================
        atom_imp_normalized = atom_imp

        # ==================================================================
        # 修复点3：按子图分离归因
        # ==================================================================
        atom_types = G.x[:num_real_atom, 0].cpu().numpy()
        subgraph_labels = np.full(num_real_atom, -1, dtype=int)
        if num_bond > 0:
            parent = list(range(num_real_atom))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb: parent[ra] = rb
            
            for e_idx in range(num_bond):
                s = int(edge_index_cpu[0, e_idx])
                t = int(edge_index_cpu[1, e_idx])
                if s < num_real_atom and t < num_real_atom:
                    union(s, t)
            
            root_to_id = {}
            label_counter = 0
            for a in range(num_real_atom):
                r = find(a)
                if r not in root_to_id:
                    root_to_id[r] = label_counter
                    label_counter += 1
                subgraph_labels[a] = root_to_id[r]
        else:
            subgraph_labels[:] = 0
        
        unique_labels, counts = np.unique(subgraph_labels, return_counts=True)
        sorted_by_size = unique_labels[np.argsort(counts)[::-1]]
        n_components = len(sorted_by_size)
        comp_to_role = {}
        if n_components >= 3:
            comp_to_role[sorted_by_size[0]] = 'cat'
            comp_to_role[sorted_by_size[1]] = 'ani'
            comp_to_role[sorted_by_size[2]] = 'ref'
        elif n_components == 2:
            comp_to_role[sorted_by_size[0]] = 'cat'
            comp_to_role[sorted_by_size[1]] = 'ani'
        elif n_components == 1:
            comp_to_role[sorted_by_size[0]] = 'cat'
        
        for atom_idx in range(num_real_atom):
            at = int(atom_types[atom_idx])
            if at not in ELEMENT_MAP: continue
            elem_label = ELEMENT_MAP[at]
            comp_id = subgraph_labels[atom_idx]
            role = comp_to_role.get(comp_id, 'cat')
            elem_imp[role][elem_label].append(float(atom_imp_normalized[atom_idx]))

        bar.update()

    bar.close()

    # ── 5. 汇总平均相对贡献度 ──────────────────────
    result = {}
    for elem_label in ELEMENT_MAP.values():
        all_scores = []
        for role in role_keys:
            all_scores.extend(elem_imp[role][elem_label])
        result[elem_label] = float(np.mean(all_scores)) if all_scores else float('nan')
    
    print("\n── 子图分离归因详细结果 ──")
    role_names = {'cat': '阳离子', 'ani': '阴离子', 'ref': '制冷剂'}
    for elem_label in ['F (Fluorine, key polar atom)', 'I (Iodine)', 'Cl (Chlorine, e.g., HCFCs/CFCs)', 'Br (Bromine)']:
        scores_by_role = {
            role_names[r]: f"{np.mean(elem_imp[r][elem_label]):.4f}" if elem_imp[r][elem_label] else 'N/A'
            for r in role_keys
        }
        print(f"  {elem_label}: {scores_by_role}")

    # ── 6. 创建输出目录并保存原始数据 ──────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), 'fragment_explain_result')
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'element_importance_v2_raw.npy'),
            np.array(list(elem_imp.items()), dtype=object), allow_pickle=True)
    np.save(os.path.join(out_dir, 'node_feat_imp_v2.npy'),
            node_feat_imp, allow_pickle=False)

    # ── 7. 绘制并保存元素重要性水平条形图 ──
    plot_element_importance(result, os.path.join(out_dir, 'element_score_v2.png'))

    # ── 8. 绘制并保存 7 大节点特征维度相对重要性图 ──
    feat_names = ['atomic_number', 'hybridization', 'aromatic', 'degree', 'charge',
                  'electronegativity', 'cov_radius']  # V2 新增 2 个
    
    # 改为计算每个特征的百分比贡献度
    total_imp = np.sum(node_feat_imp)
    relative_feat_imp = (node_feat_imp / total_imp * 100) if total_imp > 0 else node_feat_imp
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.bar(feat_names, relative_feat_imp, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Importance Contribution (%)', fontsize=12)
    ax.set_title('Node Feature Importance Contribution\n(Tri-Graph Refrigerant V2)', fontsize=13, pad=15)
    plt.tight_layout()
    nf_path = os.path.join(out_dir, 'node_feature_v2.png')
    plt.savefig(nf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [图表] 已保存原子特征重要性图: {nf_path}")

    print("\n[完成] 批量可解释性特征推演计算完毕！")
    print(f"  数据结果已保存在目录: {out_dir}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='制冷剂体系 GNNExplainer 批量可解释性 V2')
    parser.add_argument(
        '--model_path',
        type=str,
        default=os.path.join(ROOT, 'GNN_for_property_prediction', 'checkpoints_v2', 'best_seed_1.pth'),
        help='模型权重文件路径'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=os.path.join(ROOT, 'processed_tri_data'),
        help='processed_tri_data 数据目录'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='GNNExplainer 优化轮数 (默认100次)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=-1,
        help='限制参与推演的样本数 (默认-1表示全量)'
    )
    args_cli = parser.parse_args()

    result = main(
        model_path    = args_cli.model_path,
        data_root     = args_cli.data_root,
        explainer_epochs = args_cli.epochs,
        num_samples   = args_cli.num_samples
    )

    print("\n── 📊 化学元素重要性排序汇总 ──")
    # 按相对贡献度由高到低排序打印
    sorted_res = sorted(result.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)
    for frag, score in sorted_res:
        if not np.isnan(score):
            print(f"  {frag:<40s}  {score:+.6f}")
        else:
            print(f"  {frag:<40s}  (当前无数据)")