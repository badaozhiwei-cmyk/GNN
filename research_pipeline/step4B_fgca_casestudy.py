"""
step4B_fgca_casestudy.py
========================
【目的】
  对 4 个精选的 IL-制冷剂体系进行 FGCA 局部可解释性分析：
  - 绘制原子级别热力图（红色越深 = 对吸收贡献越大）
  - 同时用 Integrated Gradients（IG）做对比，验证方向一致性
  - 产出论文 Fig 6 和 Table S2

【案例设计】
  Case 1: [BMIM][Tf2N] + R1234yf  → 高吸收基准
  Case 2: [BMIM][BF4]  + R1234yf  → 低吸收（同阳离子，不同阴离子）→ 证明阴离子作用
  Case 3: [BMIM][Tf2N] + R32      → 同一IL，不同制冷剂（HFC）→ 证明制冷剂作用
  Case 4: [EMIM][Tf2N] + R1234yf  → 更短烷基链阳离子 → 证明阳离子链长的影响

【运行方法】
  python step4B_fgca_casestudy.py --split B --seed 42
"""

import argparse
import os

# ── 自动定位项目根目录（脚本在 research_pipeline/ 子文件夹中）──────
import pathlib as _pl
ROOT = str(_pl.Path(__file__).resolve().parent.parent)
import os as _os; _os.chdir(ROOT)
# ─────────────────────────────────────────────────────────────────────

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from rdkit import Chem
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(ROOT, 'Interpretability_Case_Studies'))
from Explainer_Engine import Explainer_Engine, ELEMENT_SYMBOL
from smarts_dict import get_group_matches

# ============================================================
# 命令行参数
# ============================================================
parser = argparse.ArgumentParser(description='Step 4B — FGCA Case Study')
parser.add_argument('--split', type=str, default='B', choices=['A', 'B', 'D'])
parser.add_argument('--seed',  type=int, default=42)
args_cli = parser.parse_args()

SPLIT = args_cli.split
SEED  = args_cli.seed
CKPT  = os.path.join(ROOT, f'checkpoints_split{SPLIT}', f'best_gat_seed{SEED}.pth')

os.makedirs(os.path.join(ROOT, 'figure'), exist_ok=True)
os.makedirs(os.path.join(ROOT, 'scripts_phase3', 'Results'), exist_ok=True)

print(f"\n{'='*60}")
print(f"  Step 4B: FGCA Case Study + IG Comparison")
print(f"  Split: {SPLIT} | Seed: {SEED}")
print(f"  Checkpoint: {CKPT}")
print(f"{'='*60}\n")

if not os.path.exists(CKPT):
    raise FileNotFoundError(
        f"找不到 {CKPT}\n请先运行: python GAT_Runner_v4.py --split {SPLIT}"
    )

# ============================================================
# 1. 加载模型
# ============================================================
explainer = Explainer_Engine(CKPT)
model     = explainer.model
device    = explainer.device
model.eval()

# ============================================================
# 2. 精选案例定义
# ============================================================
CASES = [
    {
        'id':    'Case1',
        'title': '[BMIM][Tf₂N] + R1234yf\n(High absorption — benchmark)',
        'c_smi': 'CCCC[n+]1cccc(C)c1',
        'a_smi': 'FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F',
        'r_smi': 'C(=C(F)F)(C(F)(F)F)F',
        'T': 298.15, 'P': 1.0,
    },
    {
        'id':    'Case2',
        'title': '[BMIM][BF₄] + R1234yf\n(Low absorption — anion contrast)',
        'c_smi': 'CCCC[n+]1cccc(C)c1',
        'a_smi': '[B-](F)(F)(F)F',
        'r_smi': 'C(=C(F)F)(C(F)(F)F)F',
        'T': 298.15, 'P': 1.0,
    },
    {
        'id':    'Case3',
        'title': '[BMIM][Tf₂N] + R32\n(Same IL, HFC refrigerant)',
        'c_smi': 'CCCC[n+]1cccc(C)c1',
        'a_smi': 'FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F',
        'r_smi': 'C(F)F',
        'T': 298.15, 'P': 1.0,
    },
    {
        'id':    'Case4',
        'title': '[EMIM][Tf₂N] + R1234yf\n(Shorter alkyl chain — cation contrast)',
        'c_smi': 'CC[n+]1cccc(C)c1',
        'a_smi': 'FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F',
        'r_smi': 'C(=C(F)F)(C(F)(F)F)F',
        'T': 298.15, 'P': 1.0,
    },
]

# ============================================================
# 3. 核心函数
# ============================================================
def get_occluded_prediction(G_batch, cond_device, target_indices):
    h_input = explainer._get_embeddings(G_batch).detach()
    h_mod   = h_input.clone()
    for idx in target_indices:
        h_mod[idx] = 0.0

    def pre_hook(module, args):
        return (h_mod, args[1])

    handle = model.l1.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        out = model(G_batch, cond_device).item()
    handle.remove()
    return out


def compute_ig_scores(G_batch, cond_device, n_steps=50):
    """Integrated Gradients（简化版）：沿零→原始 Embedding 路径积分梯度"""
    h_real = explainer._get_embeddings(G_batch).detach()
    ig_sum = torch.zeros(h_real.shape[0], device=device)

    for step in range(n_steps):
        model.zero_grad()
        alpha   = (step + 0.5) / n_steps
        h_interp = (alpha * h_real).requires_grad_(True)

        def pre_hook_ig(module, args, _h=h_interp):
            return (_h, args[1])

        handle = model.l1.register_forward_pre_hook(pre_hook_ig)
        out = model(G_batch, cond_device)
        handle.remove()
        out.backward()
        ig_sum += h_interp.grad.detach().sum(dim=1)

    ig_scores = ig_sum * h_real.squeeze(0).sum(dim=1) / n_steps
    return ig_scores.cpu().numpy()


def run_case(case):
    c_smi, a_smi, r_smi = case['c_smi'], case['a_smi'], case['r_smi']
    T, P = case['T'], case['P']

    G, num_bond = explainer._build_strict_graph(c_smi, a_smi, r_smi)
    if G is None:
        print(f"  ⚠️  {case['id']}: 图构建失败，跳过")
        return None

    cond        = explainer.compute_condition(c_smi, a_smi, r_smi, T, P)
    cond_device = cond.unsqueeze(0).to(device)
    G_batch     = Batch.from_data_list([G]).to(device)

    with torch.no_grad():
        y_base = model(G_batch, cond_device).item()
    print(f"  {case['id']}: 基准预测 x₁ = {y_base:.4f}")

    m_c = Chem.MolFromSmiles(c_smi)
    m_a = Chem.MolFromSmiles(a_smi)
    m_r = Chem.MolFromSmiles(r_smi)
    n_c = m_c.GetNumAtoms()
    n_a = m_a.GetNumAtoms()
    n_r = m_r.GetNumAtoms()
    num_real = n_c + n_a + n_r

    # ── FGCA 基团级分数 ──────────────────────────────────────
    c_groups = get_group_matches(c_smi)
    a_groups = get_group_matches(a_smi)
    r_groups = get_group_matches(r_smi)

    combined = {}
    for g, idxs in c_groups.items(): combined[f"Cat_{g}"] = idxs
    for g, idxs in a_groups.items(): combined[f"Ani_{g}"] = [i + n_c for i in idxs]
    for g, idxs in r_groups.items(): combined[f"Ref_{g}"] = [i + n_c + n_a for i in idxs]

    fgca_atom_scores = np.zeros(num_real)
    group_records    = []

    for g_name, indices in combined.items():
        if 'Other_Atoms' in g_name or not indices:
            continue
        y_occ = get_occluded_prediction(G_batch, cond_device, indices)
        drop  = y_base - y_occ
        group_records.append({'group': g_name, 'drop': drop, 'indices': indices})
        val = max(drop, 0.0)
        for idx in indices:
            if idx < num_real:
                fgca_atom_scores[idx] = val

    # ── IG 原子级分数 ─────────────────────────────────────────
    try:
        ig_raw    = compute_ig_scores(G_batch, cond_device, n_steps=50)
        ig_scores = np.abs(ig_raw[:num_real])
    except Exception as e:
        print(f"  ⚠️  IG 计算失败: {e}，使用零向量替代")
        ig_scores = np.zeros(num_real)

    fgca_norm = fgca_atom_scores / (fgca_atom_scores.max() + 1e-8)
    ig_norm   = ig_scores / (ig_scores.max() + 1e-8)

    return {
        'case': case, 'G': G, 'num_bond': num_bond,
        'num_real': num_real, 'n_c': n_c, 'n_a': n_a,
        'y_base': y_base,
        'fgca_norm': fgca_norm, 'ig_norm': ig_norm,
        'group_records': group_records,
    }


def plot_case(result, save_dir):
    """绘制三分子热力图（FGCA 上行 / IG 下行）"""
    case     = result['case']
    G        = result['G']
    num_bond = result['num_bond']
    num_real = result['num_real']
    n_c, n_a = result['n_c'], result['n_a']

    atom_types = G.x[:num_real, 0].cpu().numpy()
    mol_type   = G.mol_type[:num_real].cpu().numpy()

    def build_subgraph(target_type, score_arr):
        g, ls, ss, nl = nx.Graph(), {}, [], []
        for i in range(num_real):
            if int(mol_type[i]) == target_type:
                g.add_node(i)
                sym  = ELEMENT_SYMBOL.get(int(atom_types[i]), f"Z{int(atom_types[i])}")
                ls[i] = sym
                ss.append(score_arr[i])
                nl.append(i)
        node_a = G.edge_index[0][:num_bond].cpu().numpy()
        node_b = G.edge_index[1][:num_bond].cpu().numpy()
        for u, v in zip(node_a, node_b):
            if u < num_real and v < num_real:
                if int(mol_type[u]) == target_type == int(mol_type[v]):
                    g.add_edge(u, v)
        return g, ls, ss, nl

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    titles      = ['Cation', 'Anion', 'Refrigerant']
    row_labels  = ['FGCA (Occlusion)', 'IG (Integrated Gradients)']
    score_arrays= [result['fgca_norm'], result['ig_norm']]
    cmaps       = [plt.cm.Reds, plt.cm.Blues]

    for row, (scores, cmap, rl) in enumerate(zip(score_arrays, cmaps, row_labels)):
        for col, (title, m_type) in enumerate(zip(titles, [0, 1, 2])):
            ax = axes[row][col]
            g, ls, ss, nl = build_subgraph(m_type, scores)
            if len(g.nodes) == 0:
                ax.axis('off')
                continue
            pos = nx.kamada_kawai_layout(g)
            nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.35,
                                   edge_color='gray', width=1.5)
            nx.draw_networkx_nodes(
                g, pos, nodelist=nl, ax=ax, node_size=700,
                node_color=ss, cmap=cmap, vmin=0, vmax=1,
                edgecolors='black', linewidths=1.0
            )
            nx.draw_networkx_labels(g, pos, labels=ls, ax=ax,
                                    font_size=9, font_weight='bold')
            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
            if col == 0:
                ax.set_ylabel(rl, fontsize=11, rotation=90, labelpad=10)

        sm = plt.cm.ScalarMappable(cmap=cmaps[row], norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[row, :], shrink=0.6, pad=0.02)
        cbar.set_label('Normalized Importance', rotation=270, labelpad=12, fontsize=10)

    plt.suptitle(
        f"{case['title']}\nBaseline x₁ = {result['y_base']:.4f}",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(save_dir, f"{case['id']}_heatmap.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 热力图已保存：{out}")


# ============================================================
# 4. 运行所有案例
# ============================================================
all_group_records = []

for case in CASES:
    print(f"\n{'─'*50}")
    print(f"  运行 {case['id']}: {case['title'].split(chr(10))[0]}")
    result = run_case(case)
    if result is None:
        continue

    save_dir = os.path.join(ROOT, 'scripts_phase3', 'Results')
    plot_case(result, save_dir)

    for rec in result['group_records']:
        all_group_records.append({
            'case':   case['id'],
            'title':  case['title'].split('\n')[0],
            'group':  rec['group'],
            'drop':   rec['drop'],
            'y_base': result['y_base'],
        })

# ============================================================
# 5. 保存 Table S2
# ============================================================
if all_group_records:
    records_df = pd.DataFrame(all_group_records)
    out_table  = os.path.join(ROOT, 'scripts_phase3',
                               f'TableS2_fgca_casestudy_split{SPLIT}.csv')
    records_df.to_csv(out_table, index=False)
    print(f"\n  📄 Table S2 已保存：{out_table}")

# ============================================================
# 6. Fig 6：四案例对比柱状图
# ============================================================
if not all_group_records:
    print("⚠️  没有案例成功运行，跳过 Fig 6")
else:
    records_df = pd.DataFrame(all_group_records)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_idx, case in enumerate(CASES):
        ax     = axes[ax_idx]
        subset = records_df[records_df['case'] == case['id']].copy()
        if subset.empty:
            ax.axis('off')
            continue
        subset     = subset.sort_values('drop', ascending=False).head(10)
        bar_labels = [g.replace('Cat_', 'Cat: ').replace('Ani_', 'Ani: ')
                       .replace('Ref_', 'Ref: ') for g in subset['group']]
        bar_colors = ['#E74C3C' if d > 0 else '#3498DB' for d in subset['drop']]
        ax.barh(range(len(subset)), subset['drop'].values,
                color=bar_colors, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(bar_labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Drop in x₁ (FGCA)', fontsize=9)
        ax.set_title(
            f"{case['id']}: {case['title'].split(chr(10))[0]}"
            f"\nx₁ = {subset['y_base'].iloc[0]:.4f}",
            fontsize=9.5, fontweight='bold'
        )
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Fig 6: FGCA Case Study — Top Functional Groups per IL System',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig6_out = os.path.join(ROOT, 'figure', f'Fig6_fgca_casestudy_split{SPLIT}.png')
    plt.savefig(fig6_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Fig 6 对比图已保存：{fig6_out}")

print(f"\n✅ Step 4B 全部完成！")
