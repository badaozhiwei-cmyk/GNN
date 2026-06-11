"""
ig_explain_v2.py
===================================================
制冷剂-离子液体三图体系 Integrated Gradients (IG) 可解释性分析

核心算法：Integrated Gradients (Sundararajan et al., ICML 2017)
  IG(x)_i = (x_i - x'_i) * ∫₀¹ ∂F(x' + α(x-x'))/∂x_i dα

与 GNNExplainer 相比的核心优势：
  1. 基于梯度，不需要任何优化循环，速度极快（每个样本 < 0.1s）
  2. 对 GNN 过度平滑（Over-smoothing）不敏感：只要输入特征对输出有影响，梯度就不会为零
  3. 满足公理性保证：Completeness（归因之和 = 预测变化量）和 Sensitivity
  4. 无需安装 Captum，用 PyTorch 原生求导手动实现

使用方式：
  cd Explainer_for_ionic_molecule
  python ig_explain_v2.py --model_path ../GNN_for_property_prediction/best_seed_1.pth --num_samples 300
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from torch_geometric.data import DataLoader, Data

# ── 将 GNN_for_property_prediction 加入搜索路径 ──────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))

from Model import GIN
from Dataset_explain_v2 import IL_set_v2

# GIN 超参数（与训练时完全一致）
Args = {
    'num_gin_layer': 5,
    'emb_dim': 300,
    'feat_dim': 512,
    'drop_ratio': 0.2,
    'pool': 'mean',
}

# 原子序数 -> 元素标签
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

FEAT_NAMES = ['atomic_number', 'hybridization', 'aromatic', 'degree',
              'charge', 'electronegativity', 'cov_radius']


# ═══════════════════════════════════════════════════════════════════════
# 核心算法：Integrated Gradients（手动实现，无需 Captum）
# ═══════════════════════════════════════════════════════════════════════

def integrated_gradients(forward_func, x, baseline=None, n_steps=50):
    """
    计算 x 相对于模型输出的 Integrated Gradients 归因。

    Args:
        forward_func: callable(x_tensor) -> scalar_tensor
            接收节点特征张量 x，返回模型预测标量
        x        : torch.Tensor, shape [num_nodes, num_features]，原始输入（float，无 grad）
        baseline : torch.Tensor, 与 x 同形状，基线输入（默认全零）
        n_steps  : int，积分离散步数，通常 50 即可，需要更精确可用 100

    Returns:
        attributions: torch.Tensor, shape [num_nodes, num_features]
            正值表示该特征推动预测值上升，负值表示拉低
    """
    if baseline is None:
        baseline = torch.zeros_like(x)

    delta = (x - baseline).detach()          # 插值方向，不参与计算图
    accumulated_grads = torch.zeros_like(x)  # 累积梯度

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        # 在 baseline 和 x 之间做线性插值，并开启梯度追踪
        x_interp = (baseline + alpha * delta).detach().requires_grad_(True)

        out = forward_func(x_interp)
        if out.numel() > 1:
            out = out.sum()

        out.backward()

        if x_interp.grad is not None:
            accumulated_grads = accumulated_grads + x_interp.grad.detach()

    # 黎曼近似：IG = (x - baseline) × (累积梯度 / n_steps)
    attributions = delta * (accumulated_grads / n_steps)
    return attributions


# ═══════════════════════════════════════════════════════════════════════
# 子图分离（与 fragment_explain_v2 相同的 Union-Find 算法）
# ═══════════════════════════════════════════════════════════════════════

def assign_subgraph_roles(num_real_atom, edge_index_cpu, num_bond):
    """
    用并查集把图分成连通分量，并按节点数降序标记为阳离子/阴离子/制冷剂角色。
    返回: subgraph_labels [num_real_atom], comp_to_role dict
    """
    subgraph_labels = np.full(num_real_atom, 0, dtype=int)

    if num_bond > 0:
        parent = list(range(num_real_atom))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

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

    unique_labels, counts = np.unique(subgraph_labels, return_counts=True)
    sorted_by_size = unique_labels[np.argsort(counts)[::-1]]
    n_comp = len(sorted_by_size)

    comp_to_role = {}
    role_list = ['cat', 'ani', 'ref']
    for i, label in enumerate(sorted_by_size[:3]):
        comp_to_role[label] = role_list[i]

    return subgraph_labels, comp_to_role


# ═══════════════════════════════════════════════════════════════════════
# 绘图函数
# ═══════════════════════════════════════════════════════════════════════

def plot_element_importance(result: dict, save_path: str):
    clean = {k: v for k, v in result.items() if not np.isnan(v)}
    if not clean:
        print("  [提示] 无有效数据，跳过元素图绘制")
        return

    sorted_items = sorted(clean.items(), key=lambda x: x[1])
    elements = [it[0] for it in sorted_items]
    scores   = [it[1] for it in sorted_items]

    plt.figure(figsize=(10, 6))
    max_s = max(scores) if max(scores) > 0 else 1.0
    colors = cm.RdYlGn(np.array(scores) / max_s)

    plt.barh(elements, scores, color=colors, height=0.6,
             edgecolor='black', linewidth=0.5)
    plt.xlabel('Relative IG Attribution (Element Mean / Molecule Mean)', fontsize=11)
    plt.ylabel('Chemical Elements', fontsize=12)
    plt.title('Element-Level Importance via Integrated Gradients\n'
              '(Per-Molecule Normalized, Tri-Graph V2)', fontsize=13, pad=15)
    plt.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7,
                label='Molecule Average (=1.0)')
    plt.legend(fontsize=9)
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [图表] 已保存元素 IG 归因图: {save_path}")


def plot_feature_importance(feat_pct, save_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = cm.Blues(np.linspace(0.35, 0.85, len(FEAT_NAMES)))
    ax.bar(FEAT_NAMES, feat_pct, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Attribution Contribution (%)', fontsize=12)
    ax.set_title('Node Feature Importance via Integrated Gradients\n'
                 '(Tri-Graph Refrigerant V2)', fontsize=13, pad=15)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [图表] 已保存节点特征 IG 归因图: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════

def main(model_path: str, data_root: str, n_steps: int = 50, num_samples: int = -1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[运行设备] {device}")

    # ── 1. 加载模型 ──────────────────────────────────────────────────────
    print(f"[加载模型] {model_path}")
    model = GIN(Args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # ── 2. 加载数据集 ────────────────────────────────────────────────────
    dataset = IL_set_v2(
        data_npy_path=os.path.join(data_root, 'data.npy'),
        label_npy_path=os.path.join(data_root, 'label.npy')
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=IL_set_v2.collate_fn)

    # ── 3. 初始化汇总容器 ─────────────────────────────────────────────────
    role_keys = ['cat', 'ani', 'ref']
    elem_imp = {role: {v: [] for v in ELEMENT_MAP.values()} for role in role_keys}
    feat_imp_total = np.zeros(len(FEAT_NAMES))

    total_to_run = len(loader) if num_samples <= 0 else min(num_samples, len(loader))
    print(f"[数据规模] 总量: {len(loader)} 条，本次上限: {total_to_run} 条，IG步数: {n_steps}")
    bar = tqdm(total=total_to_run, desc='IG Explaining', dynamic_ncols=True)

    # ── 4. 主循环 ─────────────────────────────────────────────────────────
    explained_count = 0
    for G, cond, label, num_bonds_list in loader:
        if num_samples > 0 and explained_count >= num_samples:
            break

        G    = G.to(device)
        cond = cond.to(device)

        x_indices = G.x.long().to(device)
        num_nodes = x_indices.shape[0]
        num_bond  = num_bonds_list[0]

        # 修复 Embedding 报错：
        # 模型第一层是 nn.Embedding，必须输入整数。IG 需要求导必须是连续小数。
        # 所以我们先计算出 7 个特征的 Embedding 向量，然后对这些连续向量求导。
        with torch.no_grad():
            E1 = model.x_embedding1(x_indices[:, 0])
            E2 = model.x_embedding2(x_indices[:, 1])
            E3 = model.x_embedding3(x_indices[:, 2])
            E4 = model.x_embedding4(x_indices[:, 3])
            E5 = model.x_embedding5(x_indices[:, 4])
            E6 = model.x_embedding6(x_indices[:, 5])
            E7 = model.x_embedding7(x_indices[:, 6])
            # E_orig: [num_nodes, 7, emb_dim]
            E_orig = torch.stack([E1, E2, E3, E4, E5, E6, E7], dim=1).detach()

        _batch = G.batch if (hasattr(G, 'batch') and G.batch is not None) else torch.zeros(num_nodes, dtype=torch.long, device=device)
        _edge_index = G.edge_index.to(device)
        _edge_attr = G.edge_attr.to(device) if hasattr(G, 'edge_attr') and G.edge_attr is not None else None

        # 构造绕过 Embedding 层的 forward 函数，直接从连续向量 E 开始前向传播
        def forward_func(E):
            # E shape: [num_nodes, 7, emb_dim]
            h = E.sum(dim=1)  # 模拟 Model.py 里 7 个 embedding 的加和
            
            for layer in range(model.num_layer):
                h = model.gnns[layer](h, _edge_index, _edge_attr)
                h = model.batch_norms[layer](h)
                h = torch.nn.functional.dropout(torch.nn.functional.relu(h), model.drop_ratio, training=model.training)
                
            h = model.feat_lin(h)
            h_pair = model.extract(h, _batch)
            h_final = torch.cat([h_pair, cond], dim=1)
            return model.pred_head(h_final)

        try:
            # 计算 IG 归因，baseline 设为全零 Embedding
            attributions = integrated_gradients(forward_func, E_orig,
                                                baseline=torch.zeros_like(E_orig), n_steps=n_steps)
            # attributions shape: [num_nodes, 7, emb_dim]
            
            # 每个原子的总归因 = 各特征在各个维度上的 |IG| 之和
            atom_attr = attributions.abs().sum(dim=(1, 2)).cpu().numpy()   # [num_nodes]
            # 7 大特征各自的归因 = 所有节点在各个维度上的 |IG| 平均
            feat_attr = attributions.abs().sum(dim=2).mean(dim=0).cpu().numpy()  # [7]
        except Exception as e:
            if explained_count == 0:
                print(f"\n[警告] 样本报错，跳过。原因: {type(e).__name__}: {e}")
            bar.update()
            continue

        explained_count += 1
        feat_imp_total += feat_attr

        # ── 5. 子图分离 ────────────────────────────────────────────────────
        # 最后一个节点为虚拟汇聚节点，不参与化学元素分析
        num_real_atom = max(num_nodes - 1, 1)
        atom_types    = G.x[:num_real_atom, 0].cpu().numpy()
        edge_index_cpu = G.edge_index.cpu()

        subgraph_labels, comp_to_role = assign_subgraph_roles(
            num_real_atom, edge_index_cpu, num_bond
        )

        # ── 6. 分子级归一化聚合 ────────────────────────────────────────────
        atom_attr_real = atom_attr[:num_real_atom]
        mol_mean = float(np.mean(atom_attr_real))
        if mol_mean <= 1e-12:
            mol_mean = 1e-12

        mol_elem_raw = {}
        for atom_idx in range(num_real_atom):
            at = int(atom_types[atom_idx])
            if at not in ELEMENT_MAP:
                continue
            elem_label = ELEMENT_MAP[at]
            comp_id = subgraph_labels[atom_idx]
            role = comp_to_role.get(comp_id, 'cat')
            mol_elem_raw.setdefault((role, elem_label), []).append(
                float(atom_attr_real[atom_idx])
            )

        for (role, elem_label), scores_in_mol in mol_elem_raw.items():
            relative_imp = float(np.mean(scores_in_mol)) / mol_mean
            elem_imp[role][elem_label].append(relative_imp)

        bar.update()

    bar.close()
    print(f"\n[统计] 成功解释样本数: {explained_count}")

    # ── 7. 汇总结果 ───────────────────────────────────────────────────────
    result = {}
    for elem_label in ELEMENT_MAP.values():
        all_scores = []
        for role in role_keys:
            all_scores.extend(elem_imp[role][elem_label])
        result[elem_label] = float(np.mean(all_scores)) if all_scores else float('nan')

    # ── 8. 保存输出 ───────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), 'fragment_explain_result')
    os.makedirs(out_dir, exist_ok=True)

    plot_element_importance(result, os.path.join(out_dir, 'element_score_ig.png'))

    feat_pct = (feat_imp_total / feat_imp_total.sum() * 100
                if feat_imp_total.sum() > 0 else feat_imp_total)
    plot_feature_importance(feat_pct, os.path.join(out_dir, 'node_feature_ig.png'))

    print(f"\n[完成] IG 分析完毕！结果保存在: {out_dir}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrated Gradients 可解释性分析 V2')
    parser.add_argument(
        '--model_path', type=str,
        default=os.path.join(ROOT, 'GNN_for_property_prediction',
                             'checkpoints_v2', 'best_seed_1.pth'),
        help='模型权重路径'
    )
    parser.add_argument(
        '--data_root', type=str,
        default=os.path.join(ROOT, 'processed_tri_data'),
        help='processed_tri_data 数据目录'
    )
    parser.add_argument(
        '--n_steps', type=int, default=50,
        help='IG 积分步数（默认50，需精确可改100）'
    )
    parser.add_argument(
        '--num_samples', type=int, default=-1,
        help='限制样本数，-1 表示全量'
    )
    args_cli = parser.parse_args()

    result = main(
        model_path=args_cli.model_path,
        data_root=args_cli.data_root,
        n_steps=args_cli.n_steps,
        num_samples=args_cli.num_samples,
    )

    print("\n── 📊 化学元素 IG 归因排序（高→低）──")
    sorted_res = sorted(
        result.items(),
        key=lambda x: x[1] if not np.isnan(x[1]) else -999,
        reverse=True
    )
    for frag, score in sorted_res:
        if not np.isnan(score):
            marker = ' ← 低于分子均值' if score < 1.0 else ''
            print(f"  {frag:<40s}  {score:+.4f}{marker}")
        else:
            print(f"  {frag:<40s}  (当前无数据)")
