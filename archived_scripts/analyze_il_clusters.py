"""
以阴离子为主轴的化学家族聚类分析
目标：为数据集划分提供化学上合理的体系分组
"""
import pandas as pd
import numpy as np

# ── 读取数据 ──
dfs = []
for sheet in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']:
    try:
        tmp = pd.read_excel('ZLJ_DATA.xlsx', sheet_name=sheet, skiprows=2)
        dfs.append(tmp)
    except:
        pass

df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])
df['Anion'] = df['IL anion'].astype(str).str.strip()
df['Cation'] = df['IL cation'].astype(str).str.strip()
df['IL'] = df['Cation'] + ' + ' + df['Anion']

print(f"Total valid records: {len(df)}")
print(f"Unique anions: {df['Anion'].nunique()}")
print(f"Unique cations: {df['Cation'].nunique()}")

# ══════════════════════════════════════════════════════════════════
# 按化学结构定义阴离子家族
# 核心思路：同一家族内的阴离子具有相似的化学骨架和溶解机理
# ══════════════════════════════════════════════════════════════════
anion_family_map = {
    # ── 家族1：双(氟磺酰)亚胺类 (Bis-sulfonyl imide family) ──
    # 共同特征：N中心，两侧对称的 SO2-CF3 臂，电荷高度离域
    # 溶解机理：柔性构象 + CF3提供亲氟作用 + 电荷离域降低内聚能
    '[Tf2N]':  'F1_Sulfonyl_Imide',
    '[BEI]':   'F1_Sulfonyl_Imide',
    '[TMeM]':  'F1_Sulfonyl_Imide',

    # ── 家族2：氟代烷基磺酸盐 (Fluoroalkyl sulfonate family) ──
    # 共同特征：Rf-SO3 骨架，氟代烷基链长度不同
    # 溶解机理：氟链提供亲氟口袋，磺酸根提供离子稳定性
    '[OTf]':   'F2_Fluoroalkyl_Sulfonate',
    '[TFES]':  'F2_Fluoroalkyl_Sulfonate',
    '[TPES]':  'F2_Fluoroalkyl_Sulfonate',
    '[TTES]':  'F2_Fluoroalkyl_Sulfonate',
    '[HFPS]':  'F2_Fluoroalkyl_Sulfonate',
    '[PFBS]':  'F2_Fluoroalkyl_Sulfonate',
    '[FS]':    'F2_Fluoroalkyl_Sulfonate',

    # ── 家族3：氟代无机阴离子 (Fluorinated inorganic anion family) ──
    # 共同特征：中心原子(P/B)被F包围，刚性球状结构
    # 溶解机理：球状结构→自由体积受限，但F提供弱亲氟作用
    '[PF6]':   'F3_Fluorinated_Inorganic',
    '[BF4]':   'F3_Fluorinated_Inorganic',
    '[FEP]':   'F3_Fluorinated_Inorganic',

    # ── 家族4：有机羧酸盐/磷酸盐/硫酸盐 (Organic oxy-anion family) ──
    # 共同特征：含C-O或P-O键，极性较高
    # 溶解机理：极性高→内聚能高→排斥非极性气体
    '[Ac]':     'O4_Organic_Oxy',
    '[Pe]':     'O4_Organic_Oxy',
    '[Pr]':     'O4_Organic_Oxy',
    '[Et2PO4]': 'O4_Organic_Oxy',
    '[MeSO4]':  'O4_Organic_Oxy',
    '[PFP]':    'O4_Organic_Oxy',   # 虽然有氟，但骨架是羧酸盐

    # ── 家族5：膦酸酯类 (Phosphinate family) ──
    # TMPP 是长烷基链磷酸酯，结构独特，主要靠空间体积吸收
    '[TMPP]':  'P5_Phosphinate',

    # ── 家族6：卤素/拟卤素类 (Halide / Pseudohalide family) ──
    # 共同特征：简单单原子或线性阴离子，无氟
    '[Cl]':    'H6_Halide',
    '[I]':     'H6_Halide',
    '[SCN]':   'H6_Halide',
}

df['Anion_Family'] = df['Anion'].map(anion_family_map)

# 检查是否有未映射的阴离子
unmapped = df[df['Anion_Family'].isna()]['Anion'].unique()
if len(unmapped) > 0:
    print(f"\n⚠️ 未映射的阴离子: {unmapped}")

# ══════════════════════════════════════════════════════════════════
# 输出各家族的统计信息
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  ANION FAMILY CLUSTERING (以阴离子化学家族为主轴)")
print("=" * 90)

family_order = ['F1_Sulfonyl_Imide', 'F2_Fluoroalkyl_Sulfonate', 'F3_Fluorinated_Inorganic',
                'O4_Organic_Oxy', 'P5_Phosphinate', 'H6_Halide']

family_names_cn = {
    'F1_Sulfonyl_Imide':        '双氟磺酰亚胺类',
    'F2_Fluoroalkyl_Sulfonate': '氟代烷基磺酸盐',
    'F3_Fluorinated_Inorganic': '氟代无机阴离子',
    'O4_Organic_Oxy':           '有机含氧酸盐',
    'P5_Phosphinate':           '膦酸酯类',
    'H6_Halide':                '卤素/拟卤素类',
}

for fam in family_order:
    sub = df[df['Anion_Family'] == fam]
    if len(sub) == 0:
        continue
    mean_x1 = sub['x1'].mean()
    std_x1 = sub['x1'].std()
    n_samples = len(sub)
    n_anions = sub['Anion'].nunique()
    n_ils = sub['IL'].nunique()
    pct = n_samples / len(df) * 100

    cn_name = family_names_cn.get(fam, fam)
    print(f"\n{'─' * 90}")
    print(f"  [{fam}] {cn_name}")
    print(f"  样本数: {n_samples} ({pct:.1f}%)  |  阴离子种类: {n_anions}  |  IL组合: {n_ils}")
    print(f"  溶解度: Mean={mean_x1:.4f}, Std={std_x1:.4f}")
    print(f"  {'─' * 60}")

    # 按阴离子细分
    for anion in sub['Anion'].unique():
        a_sub = sub[sub['Anion'] == anion]
        a_mean = a_sub['x1'].mean()
        a_n = len(a_sub)
        cations = sorted(a_sub['Cation'].unique())
        cat_str = ', '.join(cations[:5])
        if len(cations) > 5:
            cat_str += f' ...+{len(cations)-5}more'
        print(f"    {anion:<12s} Mean={a_mean:.4f}  N={a_n:4d}  搭配阳离子: {cat_str}")

# ══════════════════════════════════════════════════════════════════
# 提出三种划分方案
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("  PROPOSED TRAIN/TEST SPLIT SCHEMES (以阴离子家族为主轴)")
print("=" * 90)

schemes = [
    {
        'name': 'Scheme 1: 留出氟代无机阴离子 (BF4+PF6+FEP)',
        'test_families': ['F3_Fluorinated_Inorganic'],
        'rationale': '球状刚性阴离子 vs 柔性链阴离子。测试模型能否从柔性体系推断刚性体系。'
    },
    {
        'name': 'Scheme 2: 留出有机含氧酸盐+卤素 (无氟阴离子全部留出)',
        'test_families': ['O4_Organic_Oxy', 'H6_Halide'],
        'rationale': '所有不含氟的阴离子做测试。极端挑战：模型能否从含氟体系推断无氟体系？'
    },
    {
        'name': 'Scheme 3: 留出氟代磺酸盐 (OTf/TFES/TPES等)',
        'test_families': ['F2_Fluoroalkyl_Sulfonate'],
        'rationale': '测试集中等偏高溶解度。训练集包含高(亚胺)和低(BF4/PF6)两端，考验模型插值。'
    },
    {
        'name': 'Scheme 4: 留出双氟磺酰亚胺类 (Tf2N/BEI/TMeM) — 最极端',
        'test_families': ['F1_Sulfonyl_Imide'],
        'rationale': '留出数据量最大的家族(1479条,33%)做测试。测试模型在从未见过最主流阴离子时的表现。'
    },
]

for scheme in schemes:
    test_mask = df['Anion_Family'].isin(scheme['test_families'])
    train_df = df[~test_mask]
    test_df = df[test_mask]

    # 检查阳离子泄露情况
    train_cations = set(train_df['Cation'].unique())
    test_cations = set(test_df['Cation'].unique())
    shared_cations = train_cations & test_cations
    test_only_cations = test_cations - train_cations

    print(f"\n{'─' * 90}")
    print(f"  {scheme['name']}")
    print(f"  理由: {scheme['rationale']}")
    print(f"  训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  测试集包含阴离子: {sorted(test_df['Anion'].unique())}")
    print(f"  测试集溶解度范围: Mean={test_df['x1'].mean():.4f}")
    print(f"  训练集溶解度范围: Mean={train_df['x1'].mean():.4f}")
    print(f"  阳离子重叠情况: {len(shared_cations)}/{len(test_cations)} 个测试阳离子在训练集出现过")
    if len(test_only_cations) > 0:
        print(f"  ⚠️ 测试集独有阳离子(完全未见): {test_only_cations}")
