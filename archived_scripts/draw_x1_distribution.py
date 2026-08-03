# -*- coding: utf-8 -*-
"""
Step 1: 绘制溶解度 x1 分布直方图及箱线图，量化重叠特征
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 matplotlib 风格，使图表更 premium 
sns.set_theme(style="ticks", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']  # 支持中文和英文
plt.rcParams['axes.unicode_minus'] = False 

# ── 1. 读取数据 ──
data_path = 'ZLJ_DATA.xlsx'
dfs = []
for sheet in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']:
    try:
        tmp = pd.read_excel(data_path, sheet_name=sheet, skiprows=2)
        dfs.append(tmp)
    except Exception as e:
        print(f"读取工作表 {sheet} 失败: {e}")

df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])
df['Anion'] = df['IL anion'].astype(str).str.strip()
df['Cation'] = df['IL cation'].astype(str).str.strip()

print(f"成功载入数据，有效记录数: {len(df)}")

# ── 2. 定义化学家族 ──
anion_family_map = {
    '[Tf2N]':  'F1_Sulfonyl_Imide',
    '[BEI]':   'F1_Sulfonyl_Imide',
    '[TMeM]':  'F1_Sulfonyl_Imide',

    '[OTf]':   'F2_Fluoroalkyl_Sulfonate',
    '[TFES]':  'F2_Fluoroalkyl_Sulfonate',
    '[TPES]':  'F2_Fluoroalkyl_Sulfonate',
    '[TTES]':  'F2_Fluoroalkyl_Sulfonate',
    '[HFPS]':  'F2_Fluoroalkyl_Sulfonate',
    '[PFBS]':  'F2_Fluoroalkyl_Sulfonate',
    '[FS]':    'F2_Fluoroalkyl_Sulfonate',

    '[PF6]':   'F3_Fluorinated_Inorganic',
    '[BF4]':   'F3_Fluorinated_Inorganic',
    '[FEP]':   'F3_Fluorinated_Inorganic',  # 原分类中在 F3

    '[Ac]':     'O4_Organic_Oxy',
    '[Pe]':     'O4_Organic_Oxy',
    '[Pr]':     'O4_Organic_Oxy',
    '[Et2PO4]': 'O4_Organic_Oxy',
    '[MeSO4]':  'O4_Organic_Oxy',
    '[PFP]':    'O4_Organic_Oxy',

    '[TMPP]':  'P5_Phosphinate',

    '[Cl]':    'H6_Halide',
    '[I]':     'H6_Halide',
    '[SCN]':   'H6_Halide',
}

df['Anion_Family'] = df['Anion'].map(anion_family_map)

# ── 3. 构造 Scheme 3 修正版划分 (F2 + FEP 为测试集，其余为训练集) ──
test_anions = ['[OTf]', '[TFES]', '[TPES]', '[TTES]', '[HFPS]', '[PFBS]', '[FS]', '[FEP]']
df['Split'] = np.where(df['Anion'].isin(test_anions), 'Test Set (F2 + FEP)', 'Train+Val Set')

train_val = df[df['Split'] == 'Train+Val Set']
test = df[df['Split'] == 'Test Set (F2 + FEP)']

# ── 4. 计算统计特征 ──
print("\n" + "=" * 60)
print("  SCHEME 3 (MODIFIED) DATA SPLIT STATISTICS")
print("=" * 60)
print(f"Train+Val Set (F1, F3 w/o FEP, O4, P5, H6): N = {len(train_val)} ({len(train_val)/len(df)*100:.2f}%)")
print(f"Test Set (F2 + FEP):                      N = {len(test)} ({len(test)/len(df)*100:.2f}%)")

stats_train = train_val['x1'].describe()
stats_test = test['x1'].describe()

print("\n[Train+Val Set x1 stats]")
print(f"  Mean: {stats_train['mean']:.4f} | Std: {stats_train['std']:.4f} | Min: {stats_train['min']:.4f} | Max: {stats_train['max']:.4f} | Median: {stats_train['50%']:.4f}")
print("[Test Set x1 stats]")
print(f"  Mean: {stats_test['mean']:.4f} | Std: {stats_test['std']:.4f} | Min: {stats_test['min']:.4f} | Max: {stats_test['max']:.4f} | Median: {stats_test['50%']:.4f}")

# ── 5. 重叠度量化分析 ──
min_train, max_train = train_val['x1'].min(), train_val['x1'].max()
min_test, max_test = test['x1'].min(), test['x1'].max()

overlap_min = max(min_train, min_test)
overlap_max = min(max_train, max_test)

# 计算测试集落入训练集溶解度区间的样本比例
test_in_train_range = test[(test['x1'] >= min_train) & (test['x1'] <= max_train)]
overlap_pct = len(test_in_train_range) / len(test) * 100

print("\n[Overlap Analysis]")
print(f"  Train+Val range: [{min_train:.4f}, {max_train:.4f}]")
print(f"  Test range:      [{min_test:.4f}, {max_test:.4f}]")
print(f"  Overlap range:   [{overlap_min:.4f}, {overlap_max:.4f}]")
print(f"  Test samples within Train+Val range: {len(test_in_train_range)} / {len(test)} ({overlap_pct:.2f}%)")

# ── 6. 绘图 ──
output_dir_local = 'figure'
os.makedirs(output_dir_local, exist_ok=True)
output_dir_artifact = r'C:\Users\霸道志伟\.gemini\antigravity\brain\69cd4834-b843-44e9-b76c-493869656c43'

# --- 图 1: 直方图与 KDE 曲线 ---
plt.figure(figsize=(10, 6), dpi=300)
# 选用优雅且高对比的颜色：Slate Blue & Coral
sns.histplot(data=df, x='x1', hue='Split', kde=True, bins=40, stat='density', common_norm=False,
             palette={'Train+Val Set': '#4A90E2', 'Test Set (F2 + FEP)': '#FF6B6B'}, alpha=0.5, edgecolor='w', linewidth=0.5)

plt.axvline(x=min_train, color='#4A90E2', linestyle='--', alpha=0.7, label=f'Train Min ({min_train:.4f})')
plt.axvline(x=max_train, color='#4A90E2', linestyle='-.', alpha=0.7, label=f'Train Max ({max_train:.4f})')
plt.axvline(x=min_test, color='#FF6B6B', linestyle='--', alpha=0.7, label=f'Test Min ({min_test:.4f})')
plt.axvline(x=max_test, color='#FF6B6B', linestyle='-.', alpha=0.7, label=f'Test Max ({max_test:.4f})')

plt.title('Refrigerant Solubility (x1) Distribution: Train+Val vs Test Set', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Refrigerant Mole Fraction in IL (x1)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)
plt.tight_layout()

# 保存图 1
fig1_path_local = os.path.join(output_dir_local, 'x1_distribution_histogram.png')
fig1_path_artifact = os.path.join(output_dir_artifact, 'x1_distribution_histogram.png')
plt.savefig(fig1_path_local, bbox_inches='tight')
try:
    plt.savefig(fig1_path_artifact, bbox_inches='tight')
    print(f"✓ 图1已成功保存至 Artifacts 目录: {fig1_path_artifact}")
except Exception as e:
    print(f"⚠️ 无法保存图1至 Artifacts 目录: {e}")
plt.close()

# --- 图 2: 箱线图 (以阴离子家族为主，并突出展示 FEP 的归属合理性) ---
# 为了箱线图更清晰，我们把原分类中属于 F3 家族的 FEP 单独抽出来作为一个分类
df_box = df.copy()
df_box['Anion_Family_Box'] = df_box['Anion_Family'].copy()
df_box.loc[df_box['Anion'] == '[FEP]', 'Anion_Family_Box'] = 'FEP (Separated)'

# 排序以使图表更有逻辑性：F1 -> F2 -> FEP -> F3 (w/o FEP) -> O4 -> P5 -> H6
box_order = [
    'F1_Sulfonyl_Imide', 
    'F2_Fluoroalkyl_Sulfonate', 
    'FEP (Separated)', 
    'F3_Fluorinated_Inorganic', 
    'O4_Organic_Oxy', 
    'P5_Phosphinate', 
    'H6_Halide'
]

box_labels = [
    'F1: Sulfonyl Imide\n(Tf2N, BEI...)',
    'F2: Fluoroalkyl Sulfonate\n(OTf, TFES...)',
    'FEP\n(Separated from F3)',
    'F3: Fluorinated Inorganic\n(BF4, PF6 only)',
    'O4: Organic Oxy\n(Ac, MeSO4...)',
    'P5: Phosphinate\n(TMPP)',
    'H6: Halide\n(Cl, I, SCN)'
]

plt.figure(figsize=(12, 7), dpi=300)
# 使用柔和的配色以突显重点，给 FEP 配一个明亮的特殊颜色以作对比
box_colors = ['#5D9CEC', '#4FC1E9', '#FC6E51', '#A0D568', '#EC87C0', '#AC92EC', '#CCD1D9']
palette_box = dict(zip(box_order, box_colors))

sns.boxplot(data=df_box, x='Anion_Family_Box', y='x1', order=box_order, palette=palette_box, width=0.5, showmeans=True,
            meanprops={"marker":"d", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":6})
sns.stripplot(data=df_box, x='Anion_Family_Box', y='x1', order=box_order, color='black', alpha=0.15, size=2, jitter=0.2)

plt.xticks(ticks=range(len(box_order)), labels=box_labels, fontsize=10, rotation=15)
plt.title('Solubility (x1) Distribution by Anion Family & FEP Comparison', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Anion Family', fontsize=12)
plt.ylabel('Refrigerant Mole Fraction (x1)', fontsize=12)
plt.grid(True, axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()

# 保存图 2
fig2_path_local = os.path.join(output_dir_local, 'x1_distribution_boxplot.png')
fig2_path_artifact = os.path.join(output_dir_artifact, 'x1_distribution_boxplot.png')
plt.savefig(fig2_path_local, bbox_inches='tight')
try:
    plt.savefig(fig2_path_artifact, bbox_inches='tight')
    print(f"✓ 图2已成功保存至 Artifacts 目录: {fig2_path_artifact}")
except Exception as e:
    print(f"⚠️ 无法保存图2至 Artifacts 目录: {e}")
plt.close()

print("\n[绘图完成] 图像已成功保存至本地项目目录: figure/")
print("=" * 60)
