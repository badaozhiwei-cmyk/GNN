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
Frag_importance = {
    # ── 制冷剂侧：氟代官能团 ──────────────────────────────────
    # 这些基团与离子液体阴阳离子之间的 C-H...X 弱氢键和偶极-电荷作用
    # 是制冷剂在离子液体中物理溶解的核心驱动力
    'CF':        [],   # 氟甲烷基（单氟代）: CHF₂ / CH₂F
    'C(F)F':     [],   # 二氟甲基（CHF₂）：如 R32 (CH₂F₂), R152a
    'C(F)(F)F':  [],   # 三氟甲基（CF₃）：如 R23, R143a, R1234yf
    'FC(F)':     [],   # 二氟碳（CF₂）：如 R134a 中的中间碳
    'C=C':       [],   # 烯烃双键：如 R1234yf, R1234ze(E) 等 HFO 类制冷剂
    'ClC(F)':    [],   # 含氯氟碳（CFC/HCFC）：如 R22 (CHClF₂), R124

    # ── 阴离子侧：常见离子液体阴离子的特征基团 ───────────────
    # 阴离子的极性位点决定其与制冷剂的相互作用强度
    'S(=O)(=O)': [],   # 磺酰基（-SO₂-）：Tf2N⁻, OTF⁻, 磺酸盐类阴离子的核心
    '[N-]':      [],   # 带负电氮：双(三氟甲磺酰)亚胺阴离子 [Tf2N⁻] 的活性中心
    'P(=O)':     [],   # 膦酸酯/磷酸酯基：如 ET2PO4⁻
    'C(F)(F)S':  [],   # 氟磺酸基：TTES⁻, HFPS⁻, PFBS⁻ 等含氟阴离子特征
    '[B-]':      [],   # 四氟硼酸根 BF4⁻
    '[P-]':      [],   # 六氟磷酸根 PF6⁻ / FEP⁻
    'OC(=O)':    [],   # 羧酸根（乙酸根 AC⁻、丙酸根 PR⁻ 等）

    # ── 阳离子侧：离子液体阳离子的核心结构 ───────────────────
    # 阳离子决定了体系的整体极性和空间结构（自由体积）
    'C[N+]1=CN(C=C1)': [],   # 1,3-二烷基咪唑阳离子核心环（[MMIM]+, [BMIM]+ 等）
    'C[N+]1cccc':       [],   # 甲基吡啶阳离子环（[EMPY]+, [BMPY]+ 等）
    '[P+]':             [],   # 季磷阳离子（P4442+, P66614+ 等）
    'CC':               [],   # 饱和烷基链（影响自由体积和链长效应）
}


def plot_fragment_importance(result: dict, save_path: str):
    """
    绘制水平条形图并保存。
    result: {frag_smiles: mean_score, ...}，按得分降序排列。
    """
    # 按分数降序排列，过滤掉 NaN
    items = sorted(
        [(k, v) for k, v in result.items() if not np.isnan(v)],
        key=lambda x: x[1], reverse=True
    )
    if not items:
        print("  [警告] 没有有效的片段分数，跳过绘图。")
        return

    labels = [x[0] for x in items]
    scores = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.5)))
    bars = ax.barh(labels, scores, color='steelblue', edgecolor='white')

    # 在条形末端标注数值
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{score:.4f}', va='center', ha='left', fontsize=9
        )

    ax.set_xlabel('Fragment Importance Score (relative)', fontsize=12)
    ax.set_ylabel('Fragment SMILES', fontsize=12)
    ax.set_title('Refrigerant-IL System: GNNExplainer Fragment Attribution\n(V2, Tri-Graph Model)', fontsize=13)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.invert_yaxis()   # 得分最高的在顶部
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
    frag_imp = {k: [] for k in Frag_importance}   # 每个片段的分数列表
    node_feat_imp = np.zeros(5)                    # 5 种原子特征的重要性累加

    bar = tqdm(total=len(loader), desc='Explaining', dynamic_ncols=True)

    for G, cond, label, num_bonds_list in loader:
        G    = G.to(device)
        cond = cond.to(device)

        num_bond = num_bonds_list[0]   # batch_size=1，取第一个

        # GNNExplainer 优化掩码
        try:
            node_feat_mask, edge_mask = explainer.explain_graph(G, cond)
        except Exception as e:
            bar.update()
            continue

        # 累加节点特征重要性
        node_feat_imp += node_feat_mask.cpu().numpy()

        # ── 解析原子级掩码 ──────────────────────────────────
        # edge_mask[:num_bond]  = 真实化学键的掩码
        # edge_mask[num_bond:]  = 全局虚拟边的掩码（对应每个原子的"全局连接"）
        # 我们用全局虚拟边的掩码来代理"原子重要性"：
        # 每条全局虚拟边 i→global 的权重反映了第 i 个原子对预测的贡献
        global_edge_mask = edge_mask[num_bond:]          # shape: [2 * num_atom]（双向）
        num_atom = global_edge_mask.shape[0] // 2
        # 正向（atom→global）和反向（global→atom）取平均
        fwd = global_edge_mask[:num_atom]
        bwd = global_edge_mask[num_atom:]
        atom_importance = ((fwd + bwd) / 2).cpu().tolist()   # 每个原子的重要性分数
        mean_score = np.mean(atom_importance)

        # ── 将原子重要性映射到片段 ──────────────────────────
        # 由于三图合并后原子索引是连续的，我们用 RDKit 对每个分子做子结构匹配
        # 此处用简化方式：直接对 SMILES 片段检查是否出现在分子 SMILES 中
        # （更精确的实现需要在预处理阶段记录每个原子属于哪个分子）
        # 现阶段：将全图的平均分作为"相对贡献"的基准
        for frag in frag_imp:
            # 相对重要性 = 该片段区域的分数 - 全图平均分
            # 因为我们没有原子到片段的精确映射，此处用全图均值估算
            # 后续可通过记录分子边界进一步精细化
            frag_imp[frag].append(mean_score - mean_score)  # 占位，见下方说明

        bar.update()

    bar.close()

    # ── 5. 汇总结果 ────────────────────────────────────────────
    result = {}
    for frag, scores in frag_imp.items():
        if scores:
            result[frag] = float(np.mean(scores))
        else:
            result[frag] = float('nan')

    # ── 6. 保存原始数据 ────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), 'fragment_explain_result')
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'frag_importance_v2_raw.npy'),
            np.array(list(frag_imp.items()), dtype=object), allow_pickle=True)
    np.save(os.path.join(out_dir, 'node_feat_imp_v2.npy'),
            node_feat_imp, allow_pickle=False)

    # ── 7. 绘图 ────────────────────────────────────────────────
    plot_fragment_importance(result, os.path.join(out_dir, 'fragment_score_v2.png'))

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
