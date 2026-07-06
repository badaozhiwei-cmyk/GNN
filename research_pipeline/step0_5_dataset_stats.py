"""
step0_5_dataset_stats.py
========================
【目的】
  1. 生成论文 Table 1：数据集整体统计概览
  2. 生成 Figure S1：各阴离子家族样本分布柱状图
  3. 生成 Figure S2：溶解度 x1 分布直方图（Train/Test 分开展示）
  4. 预检 Split B 的测试集比例（目标 15~25%），决定哪些阴离子划入测试集

【输入】
  index_with_anion.csv（Step 0 的产出，需先运行 step0_verify_alignment.py）

【输出】
  - figure/FigS1_anion_family_distribution.png
  - figure/FigS2_x1_distribution.png
  - dataset_table1.csv（Table 1 数字，复制进论文）
  - split_B_proportion_check.txt（确认测试集比例是否合理）

【运行方法】
  python step0_5_dataset_stats.py
"""

import os

# ── 自动定位项目根目录（脚本在 research_pipeline/ 子文件夹中）──────
import pathlib as _pl
ROOT = str(_pl.Path(__file__).resolve().parent.parent)
import os as _os; _os.chdir(ROOT)
# ─────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ── 字体设置（论文级别）────────────────────────────────────
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.2

os.makedirs('figure', exist_ok=True)

# ──────────────────────────────────────────────────────────
# 1. 读取索引映射表
# ──────────────────────────────────────────────────────────
CSV_PATH = 'index_with_anion.csv'
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"找不到 {CSV_PATH}，请先运行 step0_verify_alignment.py 生成该文件。"
    )

df = pd.read_csv(CSV_PATH)
print(f"成功读取 {CSV_PATH}，共 {len(df)} 条数据")

# ──────────────────────────────────────────────────────────
# 2. 定义阴离子家族映射
#    这是 Split B 划分的核心依据，后续 step1 会复用同一份字典
# ──────────────────────────────────────────────────────────
# 规则：把 Excel 里的阴离子缩写（大写后）映射到家族名
ANION_FAMILY_MAP = {
    # ── F1: 含氟有机磺酸盐／磺酰亚胺（高吸收，划入 Test）──────────
    'TF2N':   'F1_Sulfonimide',   # bis(trifluoromethanesulfonyl)imide
    'TFSI':   'F1_Sulfonimide',   # 同 Tf2N 的另一种写法
    'NTF2':   'F1_Sulfonimide',
    'OTF':    'F1_Sulfonimide',   # trifluoromethanesulfonate (Triflate)
    'TFO':    'F1_Sulfonimide',
    'TTES':   'F1_Sulfonimide',
    'HFPS':   'F1_Sulfonimide',
    'PFBS':   'F1_Sulfonimide',
    'TFES':   'F1_Sulfonimide',
    'TPES':   'F1_Sulfonimide',
    'FS':     'F1_Sulfonimide',

    # ── F2: 氟代烷基磷酸盐（高吸收，划入 Test）──────────────────
    'FEP':    'F2_FluoroAlkyl',   # tris(pentafluoroethyl)trifluorophosphate
    'BEI':    'F2_FluoroAlkyl',   # bis(pentafluoroethylsulfonyl)imide
    'TMEM':   'F2_FluoroAlkyl',   # tris(trifluoromethylsulfonyl)methide
    'PFP':    'F2_FluoroAlkyl',   # perfluoropentanoate

    # ── F3: 无机球形氟（低吸收，划入 Train）──────────────────────
    'BF4':    'F3_InorganicF',
    'PF6':    'F3_InorganicF',

    # ── A1: 有机酸根／磷酸酯（中等吸收，划入 Train）──────────────
    'AC':     'A1_OrganicAcid',
    'DCA':    'A1_OrganicAcid',   # dicyanamide
    'SCN':    'A1_OrganicAcid',   # thiocyanate
    'PR':     'A1_OrganicAcid',   # propionate
    'PE':     'A1_OrganicAcid',   # pentanoate
    'ET2PO4': 'A1_OrganicAcid',
    'TMPP':   'A1_OrganicAcid',

    # ── A2: 卤素／无机简单阴离子（低吸收，划入 Train）────────────
    'CL':     'A2_Halide',
    'BR':     'A2_Halide',
    'I':      'A2_Halide',
    'NO3':    'A2_Halide',
    'SCO':    'A2_Halide',
}

def assign_family(anion_name: str) -> str:
    """把 Excel 里的阴离子名字映射到家族，未知的归入 Other"""
    key = str(anion_name).strip().upper().replace('[', '').replace(']', '').replace('-', '')
    return ANION_FAMILY_MAP.get(key, 'Other')

df['anion_family'] = df['anion'].apply(assign_family)

# ──────────────────────────────────────────────────────────
# 3. Table 1：数据集整体统计
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("【Table 1：数据集统计概览】")
print("=" * 55)

stats = {
    'Total samples':          len(df),
    'Unique cations':         df['cation'].nunique(),
    'Unique anions':          df['anion'].nunique(),
    'Unique refrigerants':    df['refrigerant'].nunique(),
    'T range (K)':            f"{df['T_K'].min():.1f} ~ {df['T_K'].max():.1f}",
    'P range (MPa)':          f"{df['P_MPa'].min():.4f} ~ {df['P_MPa'].max():.4f}",
    'x1 range':               f"{df['x1'].min():.4f} ~ {df['x1'].max():.4f}",
    'x1 mean ± std':          f"{df['x1'].mean():.4f} ± {df['x1'].std():.4f}",
}

for k, v in stats.items():
    print(f"  {k:<30}: {v}")

# 保存为 CSV
table1_df = pd.DataFrame(list(stats.items()), columns=['Item', 'Value'])
table1_df.to_csv('dataset_table1.csv', index=False, encoding='utf-8-sig')
print("\n  📄 Table 1 已保存：dataset_table1.csv")

# ──────────────────────────────────────────────────────────
# 4. Split B 比例检查
# ──────────────────────────────────────────────────────────
# 测试集 = F1 + F2 家族；训练集 = F3 + A1 + A2 + Other
TEST_FAMILIES  = {'F1_Sulfonimide', 'F2_FluoroAlkyl'}
TRAIN_FAMILIES = {'F3_InorganicF', 'A1_OrganicAcid', 'A2_Halide', 'Other'}

test_mask  = df['anion_family'].isin(TEST_FAMILIES)
train_mask = ~test_mask

n_test  = test_mask.sum()
n_train = train_mask.sum()
test_pct = n_test / len(df) * 100

print("\n" + "=" * 55)
print("【Split B 测试集比例预检】")
print("=" * 55)
print(f"  测试集（F1+F2）：{n_test} 条  ({test_pct:.1f}%)")
print(f"  训练集（其余） ：{n_train} 条  ({100-test_pct:.1f}%)")

check_lines = [
    "Split B 测试集比例预检报告",
    "=" * 40,
    f"总样本数：{len(df)}",
    f"测试集（F1_Sulfonimide + F2_FluoroAlkyl）：{n_test} 条 ({test_pct:.1f}%)",
    f"训练集（其余家族）：{n_train} 条 ({100-test_pct:.1f}%)",
    "",
]

if 15 <= test_pct <= 25:
    msg = f"✅ 测试集比例 {test_pct:.1f}% 在目标范围 15~25% 内，Split B 划分合理！"
    print(f"\n  {msg}")
elif test_pct < 15:
    msg = (f"⚠️  测试集比例仅 {test_pct:.1f}%，偏小。\n"
           f"     建议将 DCA 或 SCN 家族（A1 中的部分）也划入测试集，增大测试集比例。")
    print(f"\n  {msg}")
else:
    msg = (f"⚠️  测试集比例达 {test_pct:.1f}%，偏大（训练集太少）。\n"
           f"     建议将 OTF / TTES 等 F1 子类移回训练集，减少测试集大小。")
    print(f"\n  {msg}")

check_lines.append(msg)
check_lines.append("")
check_lines.append("各阴离子家族详细分布：")

family_counts = df.groupby('anion_family').agg(
    样本数=('x1', 'count'),
    x1均值=('x1', 'mean'),
    x1中位数=('x1', 'median'),
).sort_values('样本数', ascending=False)

print("\n【各阴离子家族分布】")
print(family_counts.to_string())
check_lines.append(family_counts.to_string())

with open('split_B_proportion_check.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(check_lines))
print("\n  📄 比例检查报告已保存：split_B_proportion_check.txt")

# ──────────────────────────────────────────────────────────
# 5. Figure S1：阴离子家族分布柱状图
# ──────────────────────────────────────────────────────────
# 颜色方案：测试集家族用暖色（红/橙），训练集家族用冷色（蓝/绿/灰）
FAMILY_COLORS = {
    'F1_Sulfonimide': '#E74C3C',  # 红：测试集（OOD）
    'F2_FluoroAlkyl': '#E74C3C',  # 红：测试集（OOD）
    'F3_InorganicF':  '#3498DB',  # 蓝：训练集
    'A1_OrganicAcid': '#3498DB',  # 蓝：训练集
    'A2_Halide':      '#3498DB',  # 蓝：训练集
    'Other':          '#3498DB',  # 蓝：训练集
}

# 按各家族包含的具体阴离子展示（更细粒度）
anion_family_detail = df.groupby(['anion_family', 'anion']).size().reset_index(name='count')
anion_family_detail = anion_family_detail.sort_values(['anion_family', 'count'], ascending=[True, False])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- 左图：按家族汇总 ---
ax1 = axes[0]
fam_counts = df['anion_family'].value_counts()
colors_bar = [FAMILY_COLORS.get(f, '#95A5A6') for f in fam_counts.index]
bars = ax1.bar(range(len(fam_counts)), fam_counts.values, color=colors_bar,
               edgecolor='white', linewidth=0.8, alpha=0.9)

for i, (bar, val) in enumerate(zip(bars, fam_counts.values)):
    pct = val / len(df) * 100
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
             f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xticks(range(len(fam_counts)))
ax1.set_xticklabels(
    [f.replace('_', '\n') for f in fam_counts.index],
    fontsize=9, rotation=15, ha='right'
)
ax1.set_ylabel('Number of data points', fontsize=12)
ax1.set_title('(a) Sample distribution by anion family', fontsize=13, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 图例：区分 Train / Test
test_patch  = mpatches.Patch(color='#E74C3C', label='Test set (OOD)')
train_patch = mpatches.Patch(color='#3498DB', label='Train set')
ax1.legend(handles=[test_patch, train_patch], fontsize=10, loc='upper right')

# --- 右图：各阴离子具体数量（Top 20）---
ax2 = axes[1]
anion_counts = df['anion'].value_counts().head(20)

# 根据所属家族决定颜色
def get_anion_color(anion_name):
    fam = assign_family(anion_name)
    return FAMILY_COLORS.get(fam, '#95A5A6')

colors_anion = [get_anion_color(a) for a in anion_counts.index]
bars2 = ax2.barh(range(len(anion_counts)), anion_counts.values,
                 color=colors_anion, edgecolor='white', linewidth=0.8, alpha=0.9)

for bar, val in zip(bars2, anion_counts.values):
    ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
             str(val), va='center', ha='left', fontsize=9)

ax2.set_yticks(range(len(anion_counts)))
ax2.set_yticklabels(anion_counts.index, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('Number of data points', fontsize=12)
ax2.set_title('(b) Top 20 anions by sample count', fontsize=13, fontweight='bold')
ax2.grid(axis='x', linestyle='--', alpha=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
fig_path = 'figure/FigS1_anion_family_distribution.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  📊 Figure S1 已保存：{fig_path}")

# ──────────────────────────────────────────────────────────
# 6. Figure S2：x1 溶解度分布直方图（Train vs Test 分开）
# ──────────────────────────────────────────────────────────
x1_train = df.loc[train_mask, 'x1'].values
x1_test  = df.loc[test_mask,  'x1'].values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 左图：整体分布 KDE + 直方图 ---
ax1 = axes[0]
bins = np.linspace(0, 1, 40)
ax1.hist(x1_train, bins=bins, color='#3498DB', alpha=0.6, label=f'Train ({len(x1_train)} pts)', density=True)
ax1.hist(x1_test,  bins=bins, color='#E74C3C', alpha=0.6, label=f'Test-OOD ({len(x1_test)} pts)', density=True)
ax1.set_xlabel('Refrigerant solubility x₁', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('(a) x₁ distribution: Train vs OOD-Test', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(linestyle='--', alpha=0.4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- 右图：按家族的箱线图 ---
ax2 = axes[1]
family_order = ['F1_Sulfonimide', 'F2_FluoroAlkyl', 'F3_InorganicF',
                'A1_OrganicAcid', 'A2_Halide', 'Other']
box_data   = [df.loc[df['anion_family'] == f, 'x1'].values for f in family_order]
box_labels = [f.replace('_', '\n') for f in family_order]
box_colors = [FAMILY_COLORS.get(f, '#95A5A6') for f in family_order]

bp = ax2.boxplot(box_data, patch_artist=True, notch=False,
                 medianprops=dict(color='black', linewidth=2))

for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax2.set_xticklabels(box_labels, fontsize=8, rotation=15, ha='right')
ax2.set_ylabel('Refrigerant solubility x₁', fontsize=12)
ax2.set_title('(b) x₁ distribution by anion family', fontsize=13, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 标注 Train / Test 分隔线
ax2.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
ax2.text(1.0, ax2.get_ylim()[1] * 0.95, '← Test (OOD)',
         ha='center', color='#E74C3C', fontsize=9, fontweight='bold')
ax2.text(4.5, ax2.get_ylim()[1] * 0.95, 'Train →',
         ha='center', color='#3498DB', fontsize=9, fontweight='bold')

plt.tight_layout(pad=2.0)
fig_path2 = 'figure/FigS2_x1_distribution.png'
plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
plt.close()
print(f"  📊 Figure S2 已保存：{fig_path2}")

# ──────────────────────────────────────────────────────────
# 7. 按制冷剂类型统计（为 Split C 提前准备）
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("【制冷剂类型分布（为 Split C 预检）】")
print("=" * 55)

# 根据 sheet 来源区分 HFC / HFO / Other
def get_refri_type(sheet_name):
    if 'HFC' in str(sheet_name): return 'HFC'
    if 'HFO' in str(sheet_name): return 'HFO'
    return 'Other'

df['refri_type'] = df['sheet'].apply(get_refri_type)
refri_counts = df['refri_type'].value_counts()
print(refri_counts.to_string())

hfc_n = refri_counts.get('HFC', 0)
hfo_n = refri_counts.get('HFO', 0)
total = len(df)
print(f"\n  HFC 占比：{hfc_n/total*100:.1f}%  |  HFO 占比：{hfo_n/total*100:.1f}%")
print("  （Split C：HFC↔HFO 互测，放 Supplementary）")

print("\n" + "=" * 55)
print("✅ Step 0.5 全部完成！请查看：")
print("   - dataset_table1.csv")
print("   - split_B_proportion_check.txt")
print("   - figure/FigS1_anion_family_distribution.png")
print("   - figure/FigS2_x1_distribution.png")
print("=" * 55)
