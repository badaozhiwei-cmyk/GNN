"""
step15_b_generate_hfc_all_split.py — Generate All-HFC Production Model Split
=============================================================================
【使命】
  为全量 HFC 生产模型 (All-HFC Production Model) 构建 90% Train / 10% Val 切分。
  该模型吸收全部 12 种饱和 HFC 冷媒、全部 23 种阴离子与全部 50 种离子对的所有热力学知识，
  作为 HFO/HCFO 零样本跨家族应力测试的主评估基线。
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
META_CSV = PROJECT_ROOT / "processed_tri_data_hfc2739" / "meta_info.csv"
OUT_FILE = PROJECT_ROOT / "splits" / "HFC_all_split.npz"

df = pd.read_csv(META_CSV)
print(f"[数据加载] {META_CSV.name}: 共 {len(df)} 条样本")

clean_ion = lambda s: str(s).strip().replace("[", "").replace("]", "").upper()
df['c'] = df['cation'].apply(clean_ion)
df['a'] = df['anion'].apply(clean_ion)
df['ref'] = df['refrigerant'].str.strip().str.upper()
df['chem_system'] = df['c'] + "__" + df['a'] + "__" + df['ref']

rng = np.random.default_rng(42)
n_total = len(df)
n_val_target = round(n_total * 0.10) # 274

val_indices = []
train_pool = []

for cs, grp in df.groupby('chem_system'):
    idx = grp.index.to_numpy().copy()
    rng.shuffle(idx)
    k = int(np.round(len(idx) * 0.10))
    if k > 0:
        val_indices.extend(idx[:k])
        train_pool.extend(idx[k:])
    else:
        train_pool.extend(idx)

# 精确调整到 274
val_set = set(val_indices)
train_pool = [i for i in range(n_total) if i not in val_set]
rng.shuffle(train_pool)

diff = n_val_target - len(val_indices)
if diff > 0:
    val_indices.extend(train_pool[:diff])
elif diff < 0:
    val_indices = val_indices[:n_val_target]

val_idx = np.sort(np.array(val_indices))
train_idx = np.sort(np.array([i for i in range(n_total) if i not in set(val_idx)]))

assert len(set(train_idx) & set(val_idx)) == 0, "Train-Val overlap error!"
assert len(train_idx) + len(val_idx) == n_total, "Incomplete coverage error!"

np.savez_compressed(
    OUT_FILE,
    train=train_idx,
    val=val_idx,
    test=np.array([], dtype=int),
    universe="Saturated_HFC2739_All",
    n_total=n_total
)

print(f"[成功] 生成切分: {OUT_FILE.name}")
print(f"       Train: {len(train_idx)} 点 ({len(train_idx)/n_total*100:.2f}%)")
print(f"       Val  : {len(val_idx)} 点 ({len(val_idx)/n_total*100:.2f}%)")
print(f"       涵盖冷媒种类: {df.iloc[train_idx]['ref'].nunique()} / 12")
print(f"       涵盖阴离子数: {df.iloc[train_idx]['a'].nunique()} / 23")
print(f"       涵盖阳离子数: {df.iloc[train_idx]['c'].nunique()} / {df['c'].nunique()}")
