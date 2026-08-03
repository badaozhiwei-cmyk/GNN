"""
loro_eval.py
============
Leave-One-Refrigerant-Out (LORO) Generalization Benchmark.
Evaluates model's ability to predict Refrigerant/refrigerant solubility in ionic liquids
when a specific refrigerant is COMPLETELY unseen during training.
"""
import pandas as pd
import numpy as np
import os
import pathlib as pl
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

ROOT = str(pl.Path(__file__).resolve().parent.parent)
os.chdir(ROOT)

print("正在加载数据集 (index_with_anion.csv)...")
df = pd.read_csv('index_with_anion.csv')

# Precompute Morgan Fingerprints and Descriptors for all unique refrigerants, cations, anions
def get_fp(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    return np.array(gen.GetFingerprint(mol), dtype=np.float32)

print("预计算分子的 Morgan 指纹...")
cat_fps = {c: get_fp(df[df['cation']==c]['cation_smiles'].iloc[0]) for c in df['cation'].unique()}
ani_fps = {a: get_fp(df[df['anion']==a]['anion_smiles'].iloc[0]) for a in df['anion'].unique()}
ref_fps = {r: get_fp(df[df['refrigerant']==r]['refri_smiles'].iloc[0]) for r in df['refrigerant'].unique()}

X_rows = []
for i in range(len(df)):
    row = df.iloc[i]
    c_fp = cat_fps[row['cation']]
    a_fp = ani_fps[row['anion']]
    r_fp = ref_fps[row['refrigerant']]
    T = float(row['T_K'])
    P = float(row['P_MPa'])
    feat = np.concatenate([c_fp, a_fp, r_fp, [T, P]])
    X_rows.append(feat)

X = np.vstack(X_rows)
y = df['x1'].values
refrigerants = df['refrigerant'].values

# Focus on refrigerants with N >= 100
target_refs = df['refrigerant'].value_counts()[df['refrigerant'].value_counts() >= 100].index.tolist()

print(f"\n==========================================================================")
print(f"  Leave-One-Refrigerant-Out (LORO) 外推泛化基准评估 (N >= 100)")
print(f"==========================================================================")
header = f"{'Held-out Refrigerant':22s}  {'Type':6s}  {'N_test':>6s}  {'R2':>8s}  {'MAE':>8s}  {'RMSE':>8s}  {'y_mean':>8s}"
print(header)
print("-" * 80)

results = []

for ref in target_refs:
    test_mask = (refrigerants == ref)
    train_mask = ~test_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test   = X[test_mask], y[test_mask]
    
    # Determine chemical type
    smi = str(df[df['refrigerant'] == ref]['refri_smiles'].iloc[0])
    mol = Chem.MolFromSmiles(smi)
    n_double = sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2.0) if mol else 0
    n_cl = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'Cl') if mol else 0
    if n_double > 0: ctype = 'HFO'
    elif n_cl > 0: ctype = 'HCFC'
    else: ctype = 'HFC'
    model = RandomForestRegressor(n_estimators=60, max_features='sqrt', n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0.0, 1.0)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    results.append({
        'refrigerant': ref,
        'type': ctype,
        'n_test': len(y_test),
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'y_mean': np.mean(y_test)
    })
    
    print(f"{ref:22s}  {ctype:6s}  {len(y_test):6d}  {r2:8.4f}  {mae:8.4f}  {rmse:8.4f}  {np.mean(y_test):8.4f}")

res_df = pd.DataFrame(results)
res_df.to_csv('loro_eval_results.csv', index=False)
print(f"\n[SUCCESS] 结果已保存至 loro_eval_results.csv")
print(f"平均 LORO R2 (主力制冷剂): {res_df['r2'].mean():.4f}")
print(f"平均 LORO MAE (主力制冷剂): {res_df['mae'].mean():.4f}")
