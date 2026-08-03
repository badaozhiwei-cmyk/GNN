"""
per_refrigerant_eval.py
========================
按制冷剂拆分 L0 预测结果，快速定位模型的"强项"和"盲区"。
使用 HistGradientBoostingRegressor 作为快速代理模型。
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pathlib as pl
import os

ROOT = str(pl.Path(__file__).resolve().parent.parent)
os.chdir(ROOT)

df = pd.read_csv('index_with_anion.csv')

# Load L0 split
d = np.load('split_L0_indices.npz')
train_idx, test_idx = d['train'], d['test']

# Simple but effective categorical features
df['cat_id'] = pd.factorize(df['cation'])[0]
df['ani_id'] = pd.factorize(df['anion'])[0]
df['ref_id'] = pd.factorize(df['refrigerant'])[0]

X = df[['T_K', 'P_MPa', 'cat_id', 'ani_id', 'ref_id']].values
y = df['x1'].values

model = HistGradientBoostingRegressor(max_iter=500, random_state=42)
model.fit(X[train_idx], y[train_idx])
preds = model.predict(X[test_idx])

# Overall
overall_r2 = r2_score(y[test_idx], preds)
overall_mae = mean_absolute_error(y[test_idx], preds)
print(f"=== L0 Overall (Tree Baseline) ===")
print(f"R2={overall_r2:.4f}  MAE={overall_mae:.4f}")
print(f"Test samples: {len(test_idx)}")

# Per-refrigerant breakdown
test_df = df.iloc[test_idx].copy()
test_df['pred'] = preds

print(f"\n{'='*80}")
print(f"  Per-Refrigerant Prediction Quality Breakdown")
print(f"{'='*80}")

header = f"{'Refrigerant':20s}  {'N_test':>6s}  {'R2':>8s}  {'MAE':>8s}  {'y_mean':>8s}  {'y_std':>8s}"
print(header)
print("-" * 80)

results = []
for ref in test_df['refrigerant'].unique():
    sub = test_df[test_df['refrigerant'] == ref]
    mae = mean_absolute_error(sub['x1'], sub['pred'])
    if len(sub) >= 5:
        r2 = r2_score(sub['x1'], sub['pred'])
    else:
        r2 = float('nan')
    results.append({
        'refrigerant': ref,
        'n_test': len(sub),
        'r2': r2,
        'mae': mae,
        'y_mean': sub['x1'].mean(),
        'y_std': sub['x1'].std()
    })

results.sort(key=lambda x: -x['n_test'])
for r in results:
    r2_str = f"{r['r2']:.4f}" if not np.isnan(r['r2']) else "  N/A "
    print(f"{r['refrigerant']:20s}  {r['n_test']:6d}  {r2_str:>8s}  {r['mae']:.4f}  {r['y_mean']:.4f}  {r['y_std']:.4f}")

# Summary by chemical type
print(f"\n{'='*80}")
print(f"  Summary by Chemical Type")
print(f"{'='*80}")

from rdkit import Chem

type_map = {}
for ref in df['refrigerant'].unique():
    smi = str(df[df['refrigerant'] == ref]['refri_smiles'].iloc[0])
    mol = Chem.MolFromSmiles(smi)
    if mol:
        n_double = sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2.0)
        n_cl = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'Cl')
        n_br = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'Br')
        if n_double > 0:
            type_map[ref] = 'HFO'
        elif n_cl > 0 or n_br > 0:
            type_map[ref] = 'HCFC'
        else:
            type_map[ref] = 'HFC'

for ctype in ['HFC', 'HFO', 'HCFC']:
    refs_in_type = [r for r in results if type_map.get(r['refrigerant']) == ctype]
    if refs_in_type:
        total_n = sum(r['n_test'] for r in refs_in_type)
        valid_r2 = [r['r2'] for r in refs_in_type if not np.isnan(r['r2'])]
        avg_r2 = np.mean(valid_r2) if valid_r2 else float('nan')
        avg_mae = np.mean([r['mae'] for r in refs_in_type])
        r2_str = f"{avg_r2:.4f}" if not np.isnan(avg_r2) else "N/A"
        print(f"  {ctype:6s}  N_test={total_n:4d}  Avg_R2={r2_str}  Avg_MAE={avg_mae:.4f}  ({len(refs_in_type)} refrigerants)")
