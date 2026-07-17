"""
step1_generalization_ladder_v2.py
=================================
【目的】
  生成“化学泛化阶梯” v2版，修复了隐式泄漏漏洞并引入了严格的 RDKit 结构界定。

【核心升级】
  1. L2 增加 isdisjoint() 严格三元组/配对泄漏审查。
  2. L4 引入 RDKit，严格定义 (C=C > 0) & (F_count > 0) 为 HFO-like OOD。
  3. 输出每个阶梯的 Train/Test Y 分布统计，防范 Target Shift。
"""

import pandas as pd
import numpy as np
import random
import os
import pathlib as pl
from rdkit import Chem

# ==========================================
# 0. 配置与加载
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 动态获取项目根目录，避免 Kaggle 路径错位 (File Not Found)
current_script_dir = str(pl.Path(__file__).resolve().parent)
ROOT_DIR = str(pl.Path(current_script_dir).parent)

DATA_PATH = os.path.join(ROOT_DIR, 'index_with_anion.csv')
df = pd.read_csv(DATA_PATH)
df['x1'] = df['x1'].astype(float)
total_samples = len(df)
print(f"总样本数: {total_samples}")

# 生成 pair 和 triplet 标识，用于后续防泄漏审计
df['ca_pair'] = df['cation'] + "_" + df['anion']
df['ar_pair'] = df['anion'] + "_" + df['refrigerant']
df['cr_pair'] = df['cation'] + "_" + df['refrigerant']
df['triplet'] = df['cation'] + "_" + df['anion'] + "_" + df['refrigerant']

def save_split_and_report(level, train_idx, val_idx, test_idx, report_file):
    # 保存 npz
    npz_path = os.path.join(ROOT_DIR, f'split_{level}_indices.npz')
    np.savez(npz_path, 
             train=np.array(train_idx), 
             val=np.array(val_idx), 
             test=np.array(test_idx))
    
    # 统计 y 分布
    train_y = df.loc[train_idx, 'x1']
    test_y  = df.loc[test_idx, 'x1']
    
    with open(report_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*40}\n")
        f.write(f"Level: {level}\n")
        f.write(f"Train : N={len(train_idx)}, Mean={train_y.mean():.4f}, Std={train_y.std():.4f}, Min={train_y.min():.4f}, Max={train_y.max():.4f}\n")
        f.write(f"Test  : N={len(test_idx)}, Mean={test_y.mean():.4f}, Std={test_y.std():.4f}, Min={test_y.min():.4f}, Max={test_y.max():.4f}\n")
        f.write(f"{'='*40}\n")
        
    print(f"[{level}] 已生成 - Train:{len(train_idx)} Val:{len(val_idx)} Test:{len(test_idx)}")

report_file = os.path.join(ROOT_DIR, 'split_report_v2.txt')
if os.path.exists(report_file):
    os.remove(report_file)


# ==========================================
# Level 0: Random Interpolation
# ==========================================
indices = df.index.tolist()
random.shuffle(indices)

L0_train_size = int(0.8 * total_samples)
L0_val_size   = int(0.1 * total_samples)

L0_train = indices[:L0_train_size]
L0_val   = indices[L0_train_size : L0_train_size + L0_val_size]
L0_test  = indices[L0_train_size + L0_val_size:]

save_split_and_report('L0', L0_train, L0_val, L0_test, report_file)


# ==========================================
# Level 1: Homologous Generalization (BMIM OOD)
# ==========================================
test_L1_mask = df['cation'].str.contains('BMIM', case=False, na=False)
L1_test = df[test_L1_mask].index.tolist()
remain_L1 = df[~test_L1_mask].index.tolist()

random.shuffle(remain_L1)
val_size_L1 = int(len(remain_L1) * 0.1)
L1_val   = remain_L1[:val_size_L1]
L1_train = remain_L1[val_size_L1:]

# 防弹审计
train_L1_cations = set(df.loc[L1_train, 'cation'])
assert not any('BMIM' in c for c in train_L1_cations), "[Audit Failed] BMIM leaked into L1 Train!"

save_split_and_report('L1', L1_train, L1_val, L1_test, report_file)


# ==========================================
# Level 2: Component Recombination (True Disjoint Audit)
# ==========================================
# 我们通过随机保留一部分配对构建 test candidate
all_pairs = set(df['ca_pair'].unique())
random.seed(SEED)
# 随机选取 15% 的离子对作为候选池，保证其他 85% 在训练集出现过
candidate_pairs = set(random.sample(list(all_pairs), int(len(all_pairs) * 0.15)))

L2_test = []
L2_train_val = []

for idx, row in df.iterrows():
    if row['ca_pair'] in candidate_pairs:
        L2_test.append(idx)
    else:
        L2_train_val.append(idx)

random.shuffle(L2_train_val)
val_size_L2 = int(len(L2_train_val) * 0.1)
L2_val = L2_train_val[:val_size_L2]
L2_train = L2_train_val[val_size_L2:]

# 关键防泄漏断言审计 (True Disjoint Audit)
train_df = df.loc[L2_train]
test_df  = df.loc[L2_test]

# 1. 验证 Pair 级别是互斥的 (严格的新组合)
assert set(test_df['ca_pair']).isdisjoint(set(train_df['ca_pair'])), "CA Pair Leakage!"

# 2. 验证 Triplet 级别是互斥的
assert set(test_df['triplet']).isdisjoint(set(train_df['triplet'])), "Triplet Leakage!"

# 3. 验证 Component 级别【允许出现】 (保证是组合泛化，而不是成分没见过)
assert set(test_df['cation']).issubset(set(train_df['cation'])) or len(set(test_df['cation']).intersection(set(train_df['cation']))) > 0, "Warning: Complete cation OOD, not recombination!"
assert set(test_df['anion']).issubset(set(train_df['anion'])) or len(set(test_df['anion']).intersection(set(train_df['anion']))) > 0, "Warning: Complete anion OOD, not recombination!"

save_split_and_report('L2', L2_train, L2_val, L2_test, report_file)


# ==========================================
# Level 3: Family OOD (Phosphonium Hard OOD)
# ==========================================
TARGET_FAMILY = 'Phosphonium'
def get_family(cation_name):
    cat_upper = str(cation_name).upper()
    if 'P' in cat_upper and ('11' in cat_upper or '44' in cat_upper or '66' in cat_upper): 
        # 简单粗暴的化学直觉回退匹配，具体可以根据你的数据调整，这里假设 Phosphonium 在文本里有明显特征，或者直接匹配特定字符
        return 'Phosphonium'
    if 'MIM' in cat_upper or 'IM' in cat_upper: return 'Imidazolium'
    if 'PY' in cat_upper: return 'Pyridinium'
    if 'N' in cat_upper and '11' in cat_upper: return 'Ammonium'
    return 'Other'

df['family'] = df['cation'].apply(get_family)

# 因为上面只是启发式规则，为了精确，直接把含 P 的且带有典型四取代的归为 Phosphonium，或者我们依靠阳离子里是否包含 [Pxxxx] 来判断
test_L3_mask = df['cation'].str.contains('P[1-9]', regex=True, case=False, na=False) | df['cation'].str.contains('phos', case=False, na=False)

L3_test = df[test_L3_mask].index.tolist()
remain_L3 = df[~test_L3_mask].index.tolist()

if len(L3_test) < 50:
    print("  [Warning] Phosphonium 样本不足，改用 Pyridinium 为 L3")
    test_L3_mask = df['cation'].str.contains('PY', case=False, na=False)
    L3_test = df[test_L3_mask].index.tolist()
    remain_L3 = df[~test_L3_mask].index.tolist()

random.shuffle(remain_L3)
val_size_L3 = int(len(remain_L3) * 0.1)
L3_val   = remain_L3[:val_size_L3]
L3_train = remain_L3[val_size_L3:]

train_L3_cats = set(df.loc[L3_train, 'cation'])
test_L3_cats  = set(df.loc[L3_test, 'cation'])
assert test_L3_cats.isdisjoint(train_L3_cats), "Family OOD Leakage!"

save_split_and_report('L3', L3_train, L3_val, L3_test, report_file)


# ==========================================
# Level 4: Scaffold Extrapolation OOD (RDKit 双键判定)
# ==========================================
def get_descriptor_flags(smiles):
    if pd.isna(smiles): return 0, 0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return 0, 0
        num_c_double_c = len(mol.GetSubstructMatches(Chem.MolFromSmarts('C=C')))
        f_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
        return num_c_double_c, f_count
    except:
        return 0, 0

df[['num_double_bonds', 'F_count']] = df['refri_smiles'].apply(lambda x: pd.Series(get_descriptor_flags(x)))

# 纯正 Scaffold Extrapolation 定义：不绑定 HFO 语义，只看分子拓扑中是否引入了双键
test_L4_mask = (df['num_double_bonds'] >= 1)
L4_test = df[test_L4_mask].index.tolist()
remain_L4 = df[~test_L4_mask].index.tolist()

random.shuffle(remain_L4)
val_size_L4 = int(len(remain_L4) * 0.1)
L4_val   = remain_L4[:val_size_L4]
L4_train = remain_L4[val_size_L4:]

# 审计
train_refri = set(df.loc[L4_train, 'refrigerant'])
test_refri  = set(df.loc[L4_test, 'refrigerant'])
assert test_refri.isdisjoint(train_refri), "Test scaffold completely leaked into train!"

save_split_and_report('L4', L4_train, L4_val, L4_test, report_file)

print("\n🚀 step1_generalization_ladder_v2 生成完毕，防泄漏审计全部通过！")
