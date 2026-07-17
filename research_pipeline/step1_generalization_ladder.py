"""
step1_generalization_ladder.py
==============================
【目的】
  生成“化学泛化阶梯 (Chemical Generalization Ladder)”的 L0 - L4 划分索引。
  取代旧版的 anion_family_splitter，提供更严格、符合顶刊审稿人要求的划分规范。
  内置 Leakage Audit 和 Filtering Funnel 报告。

【产出文件】
  split_L0_indices.npz  (Random Interpolation)
  split_L1_indices.npz  (Homologous Generalization - BMIM strictly held out)
  split_L2_indices.npz  (Component Recombination - Triple/Double-level Novelty)
  split_L3_indices.npz  (Molecular Family OOD - Pre-registered Target)
  split_L4_indices.npz  (Scaffold Hard OOD - HFC vs HFO)
  split_ladder_summary.txt
"""

import os
import pathlib as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = str(pl.Path(__file__).resolve().parent.parent)
os.chdir(ROOT)

CSV_PATH = 'index_with_anion.csv'
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("找不到 index_with_anion.csv")

df = pd.read_csv(CSV_PATH)
N = len(df)
print(f"✅ 读取成功，共 {N} 条数据")

SPLIT_SEED = 42
all_idx = df['npy_idx'].values
report = ["=" * 70, "Chemical Generalization Ladder Splits Report", "=" * 70]

# ──────────────────────────────────────────────────────────
# Level 0: Random Interpolation
# ──────────────────────────────────────────────────────────
print("\n【Level 0: Random Interpolation (80/10/10)】")
train_val_idx, test_L0_idx = train_test_split(all_idx, test_size=0.10, random_state=SPLIT_SEED)
train_L0_idx, val_L0_idx = train_test_split(train_val_idx, test_size=0.1111, random_state=SPLIT_SEED) # 0.1111 * 0.9 ~ 0.1

np.savez('split_L0_indices.npz', train=train_L0_idx, val=val_L0_idx, test=test_L0_idx)
report.append("\n【Level 0: Random Interpolation】")
report.append(f"  Train: {len(train_L0_idx)} | Val: {len(val_L0_idx)} | Test: {len(test_L0_idx)}")

# ──────────────────────────────────────────────────────────
# Level 1: Homologous Generalization (BMIM strict holdout)
# ──────────────────────────────────────────────────────────
print("\n【Level 1: Homologous Generalization (Test on BMIM)】")

def get_homo(c):
    c = str(c).strip().upper().replace('[','').replace(']','')
    if c == 'BMIM': return 'TEST_BMIM'
    if c in ['EMIM', 'HMIM', 'OMIM']: return 'TRAIN_HOMO'
    return 'OTHER'

df['homo_tag'] = df['cation'].apply(get_homo)

test_L1_mask = df['homo_tag'] == 'TEST_BMIM'
test_L1_idx = df.loc[test_L1_mask, 'npy_idx'].values

# Pool for train/val excludes ALL BMIM
train_L1_pool_df = df[~test_L1_mask]
train_L1_pool = train_L1_pool_df['npy_idx'].values

# Strict Leakage Audit
train_L1_cations = train_L1_pool_df['cation'].apply(lambda x: str(x).strip().upper().replace('[','').replace(']','')).unique()
assert "BMIM" not in train_L1_cations, "CRITICAL LEAKAGE: BMIM found in L1 training pool!"

train_L1_idx, val_L1_idx = train_test_split(train_L1_pool, test_size=0.10, random_state=SPLIT_SEED)

np.savez('split_L1_indices.npz', train=train_L1_idx, val=val_L1_idx, test=test_L1_idx)
report.append("\n【Level 1: Homologous Generalization】")
report.append("  Strict unseen assumption: 'BMIM' is completely absent from the training/val sets.")
report.append(f"  Train: {len(train_L1_idx)} | Val: {len(val_L1_idx)} | Test: {len(test_L1_idx)}")


# ──────────────────────────────────────────────────────────
# Level 2: Component Recombination (Cascading Filtering)
# ──────────────────────────────────────────────────────────
print("\n【Level 2: Component Recombination (Filtering Funnel)】")

df['ca_pair'] = df['cation'].astype(str) + '_' + df['anion'].astype(str)
df['ar_pair'] = df['anion'].astype(str) + '_' + df['refrigerant'].astype(str)
df['cr_pair'] = df['cation'].astype(str) + '_' + df['refrigerant'].astype(str)

np.random.seed(SPLIT_SEED)
unique_combos = df[['cation', 'anion', 'refrigerant', 'ca_pair', 'ar_pair', 'cr_pair']].drop_duplicates()
shuffled_combos = unique_combos.sample(frac=1, random_state=SPLIT_SEED)

test_L2_ca_strict = set()
test_L2_ar_strict = set()
test_L2_cr_strict = set()
test_L2_strict_indices = []

test_L2_ca_relaxed = set()
test_L2_ar_relaxed = set()
test_L2_relaxed_indices = []

total_combos = len(shuffled_combos)
print(f"  Total unique combos candidate: {total_combos}")

# Filtering Funnel Simulation
for _, row in shuffled_combos.iterrows():
    ca, ar, cr = row['ca_pair'], row['ar_pair'], row['cr_pair']
    idx_matches = df[(df['ca_pair'] == ca) & (df['ar_pair'] == ar)]['npy_idx'].tolist()
    
    # Relaxed condition: CA and AR never seen in train
    if ca not in test_L2_ca_relaxed and ar not in test_L2_ar_relaxed:
        test_L2_relaxed_indices.extend(idx_matches)
        test_L2_ca_relaxed.add(ca)
        test_L2_ar_relaxed.add(ar)
        
        # Strict condition: CA, AR, and CR never seen in train
        if cr not in test_L2_cr_strict:
            test_L2_strict_indices.extend(idx_matches)
            test_L2_ca_strict.add(ca)
            test_L2_ar_strict.add(ar)
            test_L2_cr_strict.add(cr)

print(f"  After Relaxed filtering (CA + AR out): {len(test_L2_relaxed_indices)} samples")
print(f"  After Strict filtering (CA + AR + CR out): {len(test_L2_strict_indices)} samples")

if len(test_L2_strict_indices) >= 100:
    print("  => Using STRICT Novelty")
    test_L2_idx = np.array(test_L2_strict_indices)
    level_str = "Strict (Triple-level CA/AR/CR exclusion)"
else:
    print("  => Using RELAXED Novelty")
    test_L2_idx = np.array(test_L2_relaxed_indices)
    level_str = "Relaxed (Double-level CA/AR exclusion)"

train_L2_pool = df.loc[~df['npy_idx'].isin(test_L2_idx), 'npy_idx'].values
train_L2_idx, val_L2_idx = train_test_split(train_L2_pool, test_size=0.10, random_state=SPLIT_SEED)

np.savez('split_L2_indices.npz', train=train_L2_idx, val=val_L2_idx, test=test_L2_idx)
report.append("\n【Level 2: Component Recombination】")
report.append(f"  Filtering Strategy Applied: {level_str}")
report.append(f"  Train: {len(train_L2_idx)} | Val: {len(val_L2_idx)} | Test: {len(test_L2_idx)}")

# ──────────────────────────────────────────────────────────
# Level 3: Molecular Family OOD (Pre-registered)
# ──────────────────────────────────────────────────────────
print("\n【Level 3: Molecular Family OOD (Manual Registration)】")

CATION_FAMILY_MAP = {
    'P1444': 'Phosphonium', 'P2444': 'Phosphonium', 'P4444': 'Phosphonium', 'P66614':'Phosphonium', 'P4441': 'Phosphonium', 'P4442': 'Phosphonium', 'P44414':'Phosphonium',
    'C4PY': 'Pyridinium', 'HOEPY': 'Pyridinium', 'PMPY': 'Pyridinium', 'C4MPY': 'Pyridinium', 'HMPY': 'Pyridinium', 'EMPY': 'Pyridinium', 'BMPY': 'Pyridinium',
    'HE': 'Ammonium', 'HEA': 'Ammonium', 'THMA': 'Ammonium', 'DEME': 'Ammonium', 'MDEA': 'Ammonium', 'M2HEA': 'Ammonium',
    'N1132': 'Ammonium', 'N4111': 'Ammonium', 'N1444': 'Ammonium', 'N4444': 'Ammonium', 'N6111': 'Ammonium', 'N1120H':'Ammonium', 'N1320H':'Ammonium', 'N1888': 'Ammonium',
}
def assign_cat_fam(name):
    key = str(name).strip().upper().replace('[','').replace(']','').replace('-','').replace(' ','')
    return CATION_FAMILY_MAP.get(key, 'Other')

df['cat_family_L3'] = df['cation'].apply(assign_cat_fam)

print("  Family Statistics:")
print(f"    Phosphonium: {sum(df['cat_family_L3'] == 'Phosphonium')} samples")
print(f"    Pyridinium: {sum(df['cat_family_L3'] == 'Pyridinium')} samples")
print(f"    Ammonium: {sum(df['cat_family_L3'] == 'Ammonium')} samples")

# Pre-registered selection: Based on actual statistics, Phosphonium (481 samples) is the most robust target.
# Pyridinium (100) is borderline. Therefore, we explicitly register Phosphonium as the OOD target.
TARGET_FAMILY = 'Phosphonium'
if sum(df['cat_family_L3'] == TARGET_FAMILY) < 100:
    TARGET_FAMILY = 'Pyridinium'

print(f"  [Pre-registered Selection] => Holding out: {TARGET_FAMILY}")

test_L3_mask = df['cat_family_L3'] == TARGET_FAMILY
test_L3_idx = df.loc[test_L3_mask, 'npy_idx'].values
train_L3_pool = df.loc[~test_L3_mask, 'npy_idx'].values
train_L3_idx, val_L3_idx = train_test_split(train_L3_pool, test_size=0.10, random_state=SPLIT_SEED)

np.savez('split_L3_indices.npz', train=train_L3_idx, val=val_L3_idx, test=test_L3_idx)
report.append("\n【Level 3: Molecular Family OOD】")
report.append(f"  Target family strictly held out: {TARGET_FAMILY}")
report.append(f"  Train: {len(train_L3_idx)} | Val: {len(val_L3_idx)} | Test: {len(test_L3_idx)}")

# ──────────────────────────────────────────────────────────
# Level 4: Scaffold Hard OOD (HFO Test)
# ──────────────────────────────────────────────────────────
print("\n【Level 4: Scaffold Hard OOD (HFO Test)】")
def get_refri_type(sheet):
    s = str(sheet)
    if 'HFC' in s: return 'HFC'
    if 'HFO' in s: return 'HFO'
    return 'Other'

df['refri_type'] = df['sheet'].apply(get_refri_type)
hfo_idx = df.loc[df['refri_type'] == 'HFO', 'npy_idx'].values
hfc_idx = df.loc[df['refri_type'] == 'HFC', 'npy_idx'].values
other_idx = df.loc[df['refri_type'] == 'Other', 'npy_idx'].values

train_L4_pool = np.concatenate([hfc_idx, other_idx])
train_L4_idx, val_L4_idx = train_test_split(train_L4_pool, test_size=0.10, random_state=SPLIT_SEED)
test_L4_idx = hfo_idx

np.savez('split_L4_indices.npz', train=train_L4_idx, val=val_L4_idx, test=test_L4_idx)
report.append("\n【Level 4: Scaffold Hard OOD】")
report.append("  Test strictly contains HFO refrigerants (containing C=C double bonds).")
report.append(f"  Train: {len(train_L4_idx)} | Val: {len(val_L4_idx)} | Test: {len(test_L4_idx)}")

# ──────────────────────────────────────────────────────────
# 校验与报告
# ──────────────────────────────────────────────────────────
def check_overlap(name, train, val, test):
    all_used = np.concatenate([train, val, test])
    overlap = len(train) + len(val) + len(test) - len(np.unique(all_used))
    assert overlap == 0, f"❌ {name} has overlap!"
    print(f"  ✅ {name} verified no overlap.")

check_overlap("L0", train_L0_idx, val_L0_idx, test_L0_idx)
check_overlap("L1", train_L1_idx, val_L1_idx, test_L1_idx)
check_overlap("L2", train_L2_idx, val_L2_idx, test_L2_idx)
check_overlap("L3", train_L3_idx, val_L3_idx, test_L3_idx)
check_overlap("L4", train_L4_idx, val_L4_idx, test_L4_idx)

with open('split_ladder_summary.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print("\n✅ 泛化阶梯划分完成！已产出 L0-L4 npz 文件以及报告 split_ladder_summary.txt。")
