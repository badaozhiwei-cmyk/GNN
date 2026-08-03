# -*- coding: utf-8 -*-
"""
Step 1 Lightweight: 仅使用 pandas/numpy 在本地计算 x1 的分布统计量与重叠度
"""
import pandas as pd
import numpy as np

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

print(f"载入数据成功，总记录数: {len(df)}")

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
    '[FEP]':   'F3_Fluorinated_Inorganic',

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
print("\n" + "=" * 70)
print("             SCHEME 3 (MODIFIED) SPLIT STATISTICAL SUMMARY")
print("=" * 70)
print(f"Train+Val Set (F1, F3 w/o FEP, O4, P5, H6): N = {len(train_val)} ({len(train_val)/len(df)*100:.2f}%)")
print(f"Test Set (F2 + FEP):                      N = {len(test)} ({len(test)/len(df)*100:.2f}%)")

desc_train = train_val['x1'].describe()
desc_test = test['x1'].describe()

print("\n[整体溶解度描述符统计]")
print(f"指标          Train+Val Set          Test Set")
print(f"--------------------------------------------------")
print(f"样本量 (N)    {len(train_val):<22d} {len(test):<22d}")
print(f"均值 (Mean)   {desc_train['mean']:<22.6f} {desc_test['mean']:<22.6f}")
print(f"标准差 (Std)  {desc_train['std']:<22.6f} {desc_test['std']:<22.6f}")
print(f"最小值 (Min)  {desc_train['min']:<22.6f} {desc_test['min']:<22.6f}")
print(f"25% 分位数    {desc_train['25%']:<22.6f} {desc_test['25%']:<22.6f}")
print(f"中位数 (Med)  {desc_train['50%']:<22.6f} {desc_test['50%']:<22.6f}")
print(f"75% 分位数    {desc_train['75%']:<22.6f} {desc_test['75%']:<22.6f}")
print(f"最大值 (Max)  {desc_train['max']:<22.6f} {desc_test['max']:<22.6f}")

# ── 5. 重叠度量化分析 ──
min_train, max_train = train_val['x1'].min(), train_val['x1'].max()
min_test, max_test = test['x1'].min(), test['x1'].max()
overlap_min = max(min_train, min_test)
overlap_max = min(max_train, max_test)

test_in_train = test[(test['x1'] >= min_train) & (test['x1'] <= max_train)]
overlap_pct = len(test_in_train) / len(test) * 100

print("\n" + "=" * 70)
print("                         OVERLAP ANALYSIS")
print("=" * 70)
print(f"Train+Val 溶解度区间: [{min_train:.6f}, {max_train:.6f}]")
print(f"Test 溶解度区间:      [{min_test:.6f}, {max_test:.6f}]")
print(f"重叠溶解度区间:       [{overlap_min:.6f}, {overlap_max:.6f}]")
print(f"测试集中落在训练集区间内的样本数: {len(test_in_train)} / {len(test)}")
print(f"测试集重叠覆盖率 (Overlap Percentage): {overlap_pct:.2f}%")

# ── 6. 细分家族分析（论证 FEP 归属） ──
print("\n" + "=" * 70)
print("                   ANION FAMILY DISTRIBUTION COMPARISON")
print("=" * 70)

df_box = df.copy()
df_box['Family_Box'] = df_box['Anion_Family'].copy()
df_box.loc[df_box['Anion'] == '[FEP]', 'Family_Box'] = 'FEP (Separated)'

box_order = [
    'F1_Sulfonyl_Imide', 
    'F2_Fluoroalkyl_Sulfonate', 
    'FEP (Separated)', 
    'F3_Fluorinated_Inorganic', 
    'O4_Organic_Oxy', 
    'P5_Phosphinate', 
    'H6_Halide'
]

family_names = {
    'F1_Sulfonyl_Imide': 'F1: 双(氟磺酰)亚胺类 (Tf2N, BEI 等)',
    'F2_Fluoroalkyl_Sulfonate': 'F2: 氟代烷基磺酸盐 (OTf, TFES 等)',
    'FEP (Separated)': 'FEP: 独立阴离子 (从原 F3 拆分)',
    'F3_Fluorinated_Inorganic': 'F3: 氟代无机阴离子 (BF4, PF6)',
    'O4_Organic_Oxy': 'O4: 有机含氧酸盐 (Ac, MeSO4 等)',
    'P5_Phosphinate': 'P5: 膦酸酯类 (TMPP)',
    'H6_Halide': 'H6: 卤素/拟卤素类 (Cl, I, SCN)'
}

for fam in box_order:
    sub = df_box[df_box['Family_Box'] == fam]
    if len(sub) == 0:
        continue
    f_desc = sub['x1'].describe()
    print(f"\n▶ {family_names[fam]}:")
    print(f"  样本数 N = {len(sub):<5d} | 均值 = {f_desc['mean']:.4f} | 标准差 = {f_desc['std']:.4f}")
    print(f"  区间 = [{f_desc['min']:.4f}, {f_desc['max']:.4f}] | 中位数 = {f_desc['50%']:.4f}")
    # 打印每个家族内具体的阴离子
    anions_in_fam = sorted(sub['Anion'].unique())
    print(f"  包含阴离子: {', '.join(anions_in_fam)}")

print("\n" + "=" * 70)
