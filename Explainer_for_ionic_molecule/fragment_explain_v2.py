"""
fragment_explain_v2.py
===================================================
制冷剂-离子液体三图体系 GNNExplainer 片段重要性分析 (V2)

与 CO2 旧版 fragment_explain.py 的核心差异：
  1. 片段字典 (Frag_importance)：换成制冷剂体系的关键官能团
       • 制冷剂侧：CHF₂、CF₃、CH₂F、CF₂ 等氟代基团
       • 阴离子侧：磺酰亚胺基团、磺酸根、氟磺酸基、磷酸酯基
       • 阳离子侧：咪唑环、吡啶环、季磷、烷基链
  2. 数据集：从 Dataset_explain_v2.IL_set_v2 加载三图数据
  3. 模型路径：自动检测 checkpoints/ 下最新的 best_model_para.pth
  4. 输出文件：frag_importance_v2.npy / fragment_score_v2.png

运行方式（在 Kaggle 或本地）:
  cd Explainer_for_ionic_molecule
  python fragment_explain_v2.py --model_path ../checkpoints/best_model_para.pth
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import DataLoader

# ── 将 GNN_for_property_prediction 加入搜索路径 ───────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))

from Model import GIN
from Explainer_v2 import IL_Explainer_v2
from Dataset_explain_v2 import IL_set_v2

# ── GIN 训练时的 Args（key名必须与 GIN_Runner.py 和 Model.py 完全一致）────
# 注意：这里用的是制冷剂三图体系的训练参数（GIN_Runner.py 中的 Args）
Args = {
    'num_gin_layer': 5,    # Model.py 读取 args['num_gin_layer']
    'emb_dim': 300,
    'feat_dim': 512,
    'drop_ratio': 0.2,
    'pool': 'mean',
}

# ══════════════════════════════════════════════════════════════
# 制冷剂体系专属片段字典
# 每个 key 是一个 SMILES 子结构（用 RDKit SMARTS 也可）
# 每个 value 是一个空列表，运行时会收集每条数据里该片段的贡献分数
# ══════════════════════════════════════════════════════════════
# 按原子元素类型分组（用原子序数 x[:,0] 直接识别，无需 SMILES）
# 原子序数对应: C=6, N=7, O=8, F=9, P=15, S=16, Cl=17, Br=35, B=5, I=53
# 这套分组方法在制冷剂-离子液体体系中意义明确：
#   F  → 制冷剂的核心氟代基团，决定 C-H 酸性和偶极强度
#   N  → 咪唑/吡啶阳离子环心，以及 Tf2N⁻ 的亚氨基中心
#   S  → 磺酰基（Tf2N⁻, OTF⁻ 等阴离子的活性位点）
#   O  → 羧酸根、磷酸酯基、磺酸根的氧原子（氢键受体）
#   P  → 季磷阳离子（P66614+, P4442+）/ 磷酸酯阴离子
#   C  → 烷基链（影响空间位阻和自由体积）
#   Cl → 含氯制冷剂（R22, R114 等 HCFC/CFC 类）
#   Br → 含溴卤代烃（R22B1 等）
#   B  → 四氟硼酸根 BF4⁻
#   I  → 碘离子 I⁻
ELEMENT_MAP = {
    5:  'B  (硼，BF4⁻)',
    6:  'C  (碳，烷基链/制冷剂骨架)',
    7:  'N  (氮，咪唑/吡啶/Tf2N⁻)',
    8:  'O  (氧，羧酸根/磷酸酯/磺酸根)',
    9:  'F  (氟，氟代制冷剂关键元素)',
    15: 'P  (磷，季磷阳离子/磷酸酯阴离子)',
    16: 'S  (硫，磺酰基阴离子活性中心)',
    17: 'Cl (氯，HCFC/CFC 类制冷剂)',
    35: 'Br (溴，含溴卤代烃)',
    53: 'I  (碘，碘离子阴离子)',
}
# 初始化每种元素的重要性收集列表
Element_importance = {v: [] for v in ELEMENT_MAP.values()}


def plot_element_importance(result: dict, save_path: str):
    """
    绘制按元素分组的原子重要性水平条形图。
    result: {element_label: mean_relative_score}
    相对分 > 0 表示该元素对预测的贡献高于平均水平（关键位点）
    相对分 < 0 表示该元素贡献低于平均水平（惰性位点）
    """
    items = sorted(
        [(k, v) for k, v in result.items() if not np.isnan(v)],
        key=lambda x: x[1], reverse=True
    )
    if not items:
        print("  [警告] 没有有效的元素分数，跳过绘图。")
        return

    labels = [x[0] for x in items]
    scores = [x[1] for x in items]
    colors = ['#e74c3c' if s > 0 else '#3498db' for s in scores]  # 正=红，负=蓝

    fig, ax = plt.subplots(figsize=(11, max(6, len(labels) * 0.6)))
    bars = ax.barh(labels, scores, color=colors, edgecolor='white')

    for bar, score in zip(bars, scores):
        x_pos = bar.get_width() + 0.0005 if score >= 0 else bar.get_width() - 0.0005
        ha = 'left' if score >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{score:+.4f}', va='center', ha=ha, fontsize=9)

    ax.set_xlabel('Relative Atom Importance Score\n(positive = above average, key interaction site)', fontsize=11)
    ax.set_title('Refrigerant-IL System: GNNExplainer Element Attribution\n'
                 '(V2 Tri-Graph, per-element grouping by atomic number)', fontsize=12)
    ax.axvline(0, color='black', linewidth=1.0, linestyle='--')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [图表] 已保存: {save_path}")


def main(model_path: str, data_root: str, explainer_epochs: int = 100):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[运行设备] {device}")

    # ── 1. 加载模型 ────────────────────────────────────────────
    print(f"[模型] 加载: {model_path}")
    model = GIN(Args).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── 2. 加载数据集 ──────────────────────────────────────────
    data_npy  = os.path.join(data_root, 'data.npy')
    label_npy = os.path.join(data_root, 'label.npy')
    dataset = IL_set_v2(data_npy_path=data_npy, label_npy_path=label_npy)

    # batch_size=1：每次解释一个样本（GNNExplainer 的必要条件）
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=IL_set_v2.collate_fn
    )

    # ── 3. 初始化 Explainer_v2（回归专用版）────────────────────
    # IL_Explainer_v2 重写了 __loss__：用 MSE 替代原版的分类 NLL Loss
    # 这样梯度方向才能正确指导掩码优化
    explainer = IL_Explainer_v2(
        model, epochs=explainer_epochs, lr=0.01
    )

    # ── 4. 主循环：逐样本解释 ──────────────────────────────────
    elem_imp  = {v: [] for v in ELEMENT_MAP.values()}  # 每种元素的相对重要性列表
    node_feat_imp = np.zeros(5)                         # 5 种节点特征的累加重要性

    bar = tqdm(total=len(loader), desc='Explaining', dynamic_ncols=True)

    for G, cond, label, num_bonds_list in loader:
        G    = G.to(device)
        cond = cond.to(device)
        num_bond = num_bonds_list[0]

        try:
            node_feat_mask, edge_mask = explainer.explain_graph(G, cond)
        except Exception as e:
            bar.update()
            continue

        node_feat_imp += node_feat_mask.cpu().numpy()

        # ── 解析原子级掩码 ─────────────────────────────────────
        # 全局虚拟边的掩码反映了每个原子对预测值的贡献程度。
        # 因为 add_global 给每个真实原子 i 添加了两条虚拟边（i→global, global→i），
        # 所以 edge_mask[num_bond:] 的前半段 = 正向掩码，后半段 = 反向掩码
        global_mask = edge_mask[num_bond:].cpu()
        num_real_atom = global_mask.shape[0] // 2       # 不含全局节点本身
        fwd = global_mask[:num_real_atom]
        bwd = global_mask[num_real_atom:]
        atom_imp = ((fwd + bwd) / 2).numpy()            # shape: [num_real_atom]
        mean_imp = atom_imp.mean()                      # 全图原子重要性均值（基准）

        # ── 按元素类型分组，计算相对重要性 ────────────────────
        # x[:, 0] 存的就是原子序数（来自 prepare_tri_graph_data.py 的图构建）
        # 最后一个节点是全局虚拟节点（原子序数=0），跳过
        atom_types = G.x[:num_real_atom, 0].cpu().numpy()   # shape: [num_real_atom]

        for atom_idx, (at, imp) in enumerate(zip(atom_types, atom_imp)):
            at = int(at)
            if at in ELEMENT_MAP:
                elem_label = ELEMENT_MAP[at]
                # 相对重要性 = 该原子得分 - 全图均值
                # 正值：该原子是高于平均水平的关键位点
                # 负值：该原子是惰性的背景结构
                elem_imp[elem_label].append(float(imp - mean_imp))

        bar.update()

    bar.close()

    # ── 5. 汇总每种元素的平均相对重要性 ───────────────────────
    result = {}
    for elem_label, scores in elem_imp.items():
        result[elem_label] = float(np.mean(scores)) if scores else float('nan')

    # ── 6. 保存原始数据 ────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), 'fragment_explain_result')
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'element_importance_v2_raw.npy'),
            np.array(list(elem_imp.items()), dtype=object), allow_pickle=True)
    np.save(os.path.join(out_dir, 'node_feat_imp_v2.npy'),
            node_feat_imp, allow_pickle=False)

    # ── 7. 绘图 ────────────────────────────────────────────────
    plot_element_importance(result, os.path.join(out_dir, 'element_score_v2.png'))

    # ── 8. 节点特征重要性图 ─────────────────────────────────────
    feat_names = ['atomic_number', 'atomic_degree', 'charge', 'hybridization', 'Aromatic']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(feat_names, node_feat_imp, color='steelblue')
    ax.set_ylabel('Cumulative Importance Score')
    ax.set_title('Node Feature Importance\n(Tri-Graph Refrigerant V2)')
    plt.tight_layout()
    nf_path = os.path.join(out_dir, 'node_feature_v2.png')
    plt.savefig(nf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [图表] 已保存: {nf_path}")

    print("\n[完成] 片段重要性分析完毕！")
    print(f"  输出目录: {out_dir}")
    return result


# ── 命令行入口 ──────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='制冷剂体系 GNNExplainer 片段重要性 V2')
    parser.add_argument(
        '--model_path',
        type=str,
        default=os.path.join(ROOT, 'checkpoints', 'best_model_para.pth'),
        help='训练好的模型权重路径（默认: checkpoints/best_model_para.pth）'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=os.path.join(ROOT, 'processed_tri_data'),
        help='processed_tri_data 目录路径'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='GNNExplainer 优化轮数（默认: 100，越大越精确但越慢）'
    )
    args_cli = parser.parse_args()

    result = main(
        model_path    = args_cli.model_path,
        data_root     = args_cli.data_root,
        explainer_epochs = args_cli.epochs
    )

    print("\n── 最终片段重要性汇总 ──")
    for frag, score in sorted(result.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True):
        print(f"  {frag:<20s}  {score:.4f}")
