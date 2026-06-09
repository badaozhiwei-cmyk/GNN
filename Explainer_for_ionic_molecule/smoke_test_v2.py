"""
smoke_test_v2.py — 1 分钟冒烟测试
===================================================
在跑全量（~1 小时）之前，先用这个脚本验证逻辑是否正确。

检验标准（3 条，全过才能放心跑全量）：
  [CHECK 1] 原子重要性分数不全为 0
  [CHECK 2] 不同元素的分数有差异（否则说明掩码没有在优化）
  [CHECK 3] 节点特征掩码 5 个值不相同（否则说明 Explainer 没有工作）

运行（在 Colab 里，< 1 分钟）：
  %cd .../Explainer_for_ionic_molecule
  !python smoke_test_v2.py --model_path ../checkpoints/best_seed_1.pth
"""

import os, sys, argparse
import numpy as np
import torch
from torch_geometric.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))

from Model import GIN
from Explainer_v2 import IL_Explainer_v2
from Dataset_explain_v2 import IL_set_v2

Args = {
    'num_gin_layer': 5,
    'emb_dim': 300,
    'feat_dim': 512,
    'drop_ratio': 0.2,
    'pool': 'mean',
}

ELEMENT_MAP = {
    5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
}

def main(model_path, data_root, n_samples=3, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载模型
    model = GIN(Args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ 模型加载成功")

    # 加载数据集（只取前 n_samples 条）
    dataset = IL_set_v2(
        data_npy_path=os.path.join(data_root, 'data.npy'),
        label_npy_path=os.path.join(data_root, 'label.npy')
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=IL_set_v2.collate_fn)

    explainer = IL_Explainer_v2(model, epochs=epochs, lr=0.01)

    all_atom_imps = []    # 收集所有样本的原子重要性
    all_node_feat_masks = []

    print(f"\n[冒烟测试] 只跑前 {n_samples} 个样本，每个 {epochs} 个 Epoch...\n")

    for i, (G, cond, label, num_bonds_list) in enumerate(loader):
        if i >= n_samples:
            break

        G    = G.to(device)
        cond = cond.to(device)
        num_bond = num_bonds_list[0]

        node_feat_mask, edge_mask = explainer.explain_graph(G, cond)

        global_mask   = edge_mask[num_bond:].cpu()
        num_real_atom = global_mask.shape[0] // 2
        fwd = global_mask[:num_real_atom]
        bwd = global_mask[num_real_atom:]
        atom_imp  = ((fwd + bwd) / 2).numpy()
        mean_imp  = atom_imp.mean()

        atom_types = G.x[:num_real_atom, 0].cpu().numpy()

        print(f"--- 样本 {i} ---")
        print(f"  原子数: {num_real_atom}, 边数(真实): {num_bond}")
        print(f"  原子重要性均值: {mean_imp:.6f}, 范围: [{atom_imp.min():.6f}, {atom_imp.max():.6f}]")

        # 按元素分组打印
        elem_scores = {}
        for at, imp in zip(atom_types, atom_imp):
            at = int(at)
            if at in ELEMENT_MAP:
                e = ELEMENT_MAP[at]
                if e not in elem_scores:
                    elem_scores[e] = []
                elem_scores[e].append(float(imp - mean_imp))

        for elem, scores in sorted(elem_scores.items()):
            print(f"    {elem:3s}: 原子数={len(scores):3d}, 相对重要性均值={np.mean(scores):+.6f}")

        print(f"  节点特征掩码: {node_feat_mask.cpu().numpy().round(4)}")

        all_atom_imps.extend(atom_imp.tolist())
        all_node_feat_masks.append(node_feat_mask.cpu().numpy())

    # ── 3 项检验 ─────────────────────────────────────────────────
    print("\n" + "="*50)
    print("        冒烟测试检验结果")
    print("="*50)

    # CHECK 1: 原子重要性不全为 0
    all_arr = np.array(all_atom_imps)
    check1 = all_arr.std() > 1e-6
    print(f"[CHECK 1] 原子重要性有差异（std > 1e-6）: {'✅ 通过' if check1 else '❌ 失败 — 全部为同一个值'}")
    print(f"          所有原子重要性 std = {all_arr.std():.6f}")

    # CHECK 2: 不同元素之间有差异（通过样本0的 elem_scores 看）
    if len(elem_scores) >= 2:
        vals = [np.mean(v) for v in elem_scores.values()]
        check2 = (max(vals) - min(vals)) > 1e-6
    else:
        check2 = False
    print(f"[CHECK 2] 不同元素得分有差异: {'✅ 通过' if check2 else '❌ 失败 — 所有元素得分相同'}")

    # CHECK 3: 节点特征掩码 7 个维度不全一样
    nf = np.stack(all_node_feat_masks).mean(axis=0)
    check3 = nf.std() > 1e-6
    print(f"[CHECK 3] 节点特征掩码有差异（std > 1e-6）: {'✅ 通过' if check3 else '❌ 失败'}")
    print(f"          特征掩码均值: {nf.round(4)}")

    print("="*50)
    if check1 and check2 and check3:
        print("🎉 全部通过！可以放心跑全量：")
        print(f"   python fragment_explain_v2.py --model_path {model_path} --epochs 100")
    else:
        print("⚠️  有检验未通过，请把上面的输出发给我排查，不要跑全量！")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(ROOT, 'GNN_for_property_prediction', 'checkpoints_v2', 'best_seed_1.pth'))
    parser.add_argument('--data_root', type=str,
                        default=os.path.join(ROOT, 'processed_tri_data'))
    parser.add_argument('--n_samples', type=int, default=3)
    parser.add_argument('--epochs',    type=int, default=10)
    a = parser.parse_args()
    main(a.model_path, a.data_root, a.n_samples, a.epochs)
