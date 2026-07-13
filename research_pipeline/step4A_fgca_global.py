"""
step4A_fgca_global.py
=====================
【FGCA — Functional Group Counterfactual Analysis（全局版）】

【相比 priority4_global_occlusion.py 的改动】
  1. 支持 --split A/B 参数，自动加载对应的 GAT checkpoint（Split B 训练的模型）
  2. 支持 --seeds 指定用哪个种子的模型（默认用 ensemble 均值）
  3. 产出额外的论文级 Bar Chart（Fig 5）
  4. 结果 CSV 格式更完整（含正贡献/负贡献分开统计）

【运行方法（Kaggle）】
  python step4A_fgca_global.py --split B
  python step4A_fgca_global.py --split B --seed 42  # 只用单个种子模型

【输出文件】
  scripts_phase3/global_group_importance_split{B}.csv
  figure/Fig5_fgca_global_ranking_split{B}.png
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
import matplotlib.cm as cm
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(ROOT, 'Interpretability_Case_Studies'))
from Explainer_Engine import Explainer_Engine
from smarts_dict import get_group_matches

# ============================================================
# 命令行参数
# ============================================================
parser = argparse.ArgumentParser(description='Step 4A — FGCA Global Ranking')
parser.add_argument('--split', type=str, default='B', choices=['A', 'B', 'D'],
                    help='使用哪种划分训练的 GAT 模型')
parser.add_argument('--seed',  type=int, default=42,
                    help='使用哪个种子的 checkpoint（默认 42）')
parser.add_argument('--top_n', type=int, default=20,
                    help='排行榜只展示前 N 个基团（默认 20）')
args_cli = parser.parse_args()

SPLIT  = args_cli.split
SEED   = args_cli.seed
TOP_N  = args_cli.top_n
CKPT   = os.path.join(ROOT, f'checkpoints_split{SPLIT}', f'best_gat_seed{SEED}.pth')

os.makedirs(os.path.join(ROOT, 'scripts_phase3'), exist_ok=True)
os.makedirs(os.path.join(ROOT, 'figure'), exist_ok=True)

print(f"\n{'='*60}")
print(f"  Step 4A: FGCA Global Ranking")
print(f"  Split: {SPLIT} | Seed: {SEED}")
print(f"  Checkpoint: {CKPT}")
print(f"{'='*60}\n")

if not os.path.exists(CKPT):
    raise FileNotFoundError(
        f"找不到 {CKPT}\n"
        f"请先运行: python GAT_Runner_v4.py --split {SPLIT}"
    )

# ============================================================
# 1. 加载模型
# ============================================================
print("正在加载 GAT 模型...")
explainer = Explainer_Engine(CKPT)
model     = explainer.model
device    = explainer.device
model.eval()
print(f"  模型加载完毕，运行在 {device}")

# ============================================================
# 2. 加载 index_with_anion.csv（Step 0 产出）
#    它已包含所有字段，不用重新读 Excel + 重建 SMILES dict
# ============================================================
index_csv = os.path.join(ROOT, 'index_with_anion.csv')
if not os.path.exists(index_csv):
    raise FileNotFoundError("找不到 index_with_anion.csv，请先运行 step0_verify_alignment.py")

df_all = pd.read_csv(index_csv)
print(f"数据集共 {len(df_all)} 条")

# 如果是 Split B，FGCA 分析应该在【训练集】上做全局统计
# （测试集的阴离子是模型从未见过的，用训练集验证"已学到的规律"更合理）
npz_path = os.path.join(ROOT, f'split_{SPLIT}_indices.npz')
if os.path.exists(npz_path):
    split_data  = np.load(npz_path)
    train_idx   = split_data['train']
    df_analysis = df_all.iloc[train_idx].reset_index(drop=True)
    print(f"Split {SPLIT} 训练集：{len(df_analysis)} 条（FGCA 在训练集上统计）")
else:
    df_analysis = df_all
    print("未找到 .npz 文件，在全量数据上统计")

# ============================================================
# 3. 核心：Pre-hook 特征敲除函数
# ============================================================
def get_occluded_prediction(G_batch, cond_device, target_indices):
    """将指定节点的初始 Embedding 清零，重新前向传播，返回被扰动后的预测值"""
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

# ============================================================
# 4. 全局扫描
# ============================================================
group_stats = {}   # {group_name: [drop1, drop2, ...]}
skipped     = 0

print(f"\n开始全局 FGCA 扫描（共 {len(df_analysis)} 条）...")
for _, row in tqdm(df_analysis.iterrows(), total=len(df_analysis), desc="FGCA Sweep"):
    c_smi = str(row['cation_smiles'])
    a_smi = str(row['anion_smiles'])
    r_smi = str(row['refri_smiles'])
    T     = float(row['T_K'])
    P     = float(row['P_MPa'])

    # 构建图
    G, num_bond = explainer._build_strict_graph(c_smi, a_smi, r_smi)
    if G is None:
        skipped += 1
        continue

    cond        = explainer.compute_condition(c_smi, a_smi, r_smi, T, P)
    cond_device = cond.unsqueeze(0).to(device)
    G_batch     = Batch.from_data_list([G]).to(device)

    with torch.no_grad():
        y_base = model(G_batch, cond_device).item()

    # 获取各分子的原子数（用于索引偏移）
    try:
        n_c = Chem.MolFromSmiles(c_smi).GetNumAtoms()
        n_a = Chem.MolFromSmiles(a_smi).GetNumAtoms()
    except Exception:
        skipped += 1
        continue

    # 阳离子基团（索引不偏移）
    c_groups = get_group_matches(c_smi)
    # 阴离子基团（索引偏移 n_c）
    a_groups = get_group_matches(a_smi)
    # 制冷剂基团（索引偏移 n_c + n_a）
    r_groups = get_group_matches(r_smi)

    combined = {}
    for g, idxs in c_groups.items():
        combined[f"Cat_{g}"] = idxs
    for g, idxs in a_groups.items():
        combined[f"Ani_{g}"] = [i + n_c for i in idxs]
    for g, idxs in r_groups.items():
        combined[f"Ref_{g}"] = [i + n_c + n_a for i in idxs]

    for g_name, indices in combined.items():
        if 'Other_Atoms' in g_name or not indices:
            continue
        y_occ = get_occluded_prediction(G_batch, cond_device, indices)
        drop  = y_base - y_occ
        group_stats.setdefault(g_name, []).append(drop)

print(f"\n扫描完成，跳过 {skipped} 条无效数据")

# ============================================================
# 5. 汇总统计
# ============================================================
records = []
for g_name, drops in group_stats.items():
    drops = np.array(drops)
    records.append({
        'Group':         g_name,
        'Count':         len(drops),
        'Mean_Drop':     float(np.mean(drops)),
        'Std_Drop':      float(np.std(drops)),
        'Median_Drop':   float(np.median(drops)),
        'Pos_Rate':      float((drops > 0).mean()),  # 正贡献比率（drop>0 意味着对吸收有促进）
        'Max_Drop':      float(np.max(drops)),
    })

res_df = pd.DataFrame(records).sort_values('Mean_Drop', ascending=False).reset_index(drop=True)

print(f"\n{'='*65}")
print("  FGCA 全局基团重要性排行（按 Mean_Drop 降序）")
print(f"{'='*65}")
print(f"  {'Group':<25} {'Count':>6} {'Mean Drop':>10} {'Std':>8} {'Pos%':>7}")
print(f"  {'-'*60}")
for _, r in res_df.head(TOP_N).iterrows():
    print(f"  {r['Group']:<25} {r['Count']:>6.0f} {r['Mean_Drop']:>10.4f} "
          f"{r['Std_Drop']:>8.4f} {r['Pos_Rate']*100:>6.1f}%")

# 保存 CSV
csv_out = os.path.join(ROOT, 'scripts_phase3', f'global_group_importance_split{SPLIT}.csv')
res_df.to_csv(csv_out, index=False)
print(f"\n  📄 结果已保存：{csv_out}")

# ============================================================
# 6. 绘制论文级 Bar Chart（Fig 5）
# ============================================================
top_df = res_df.head(TOP_N).copy()

# 颜色：按来源区分（Cat=蓝，Ani=红，Ref=绿）
def get_color(name):
    if name.startswith('Cat_'): return '#3498DB'
    if name.startswith('Ani_'): return '#E74C3C'
    if name.startswith('Ref_'): return '#27AE60'
    return '#95A5A6'

colors  = [get_color(g) for g in top_df['Group']]
labels  = [g.replace('Cat_', 'Cat: ').replace('Ani_', 'Ani: ').replace('Ref_', 'Ref: ')
           for g in top_df['Group']]
means   = top_df['Mean_Drop'].values
stds    = top_df['Std_Drop'].values

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(range(len(top_df)), means,
               xerr=stds, color=colors, alpha=0.85,
               edgecolor='white', linewidth=0.6,
               error_kw=dict(elinewidth=1.2, capsize=3, ecolor='#555'))

# 数值标注
for i, (bar, v) in enumerate(zip(bars, means)):
    ax.text(max(v, 0) + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{v:.4f}', va='center', ha='left', fontsize=8.5)

ax.set_yticks(range(len(top_df)))
ax.set_yticklabels(labels, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel('Mean Prediction Drop (Δx₁) upon Group Occlusion', fontsize=12)
ax.set_title(
    f'FGCA: Functional Group Importance Ranking (Split-{SPLIT}, N={len(df_analysis)} samples)',
    fontsize=13, fontweight='bold', pad=12
)
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498DB', label='Cation groups'),
    Patch(facecolor='#E74C3C', label='Anion groups'),
    Patch(facecolor='#27AE60', label='Refrigerant groups'),
]
ax.legend(handles=legend_elements, fontsize=10, loc='lower right')

plt.tight_layout()
fig_out = os.path.join(ROOT, 'figure', f'Fig5_fgca_global_ranking_split{SPLIT}.png')
plt.savefig(fig_out, dpi=300, bbox_inches='tight')
plt.close()
print(f"  📊 Fig 5 已保存：{fig_out}")

print(f"\n✅ Step 4A 完成！")
