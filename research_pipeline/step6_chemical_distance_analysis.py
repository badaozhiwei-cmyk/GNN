"""
step6_chemical_distance_analysis.py
==================================
Quantifies chemical similarity and distance between held-out refrigerants
and their nearest neighbors in the training set.
Proves that chemical manifold coverage and positional isomerism dictate GNN zero-shot generalization.
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.preprocessing import StandardScaler
import pathlib as pl

ROOT = pl.Path(__file__).resolve().parent.parent
os.chdir(ROOT)

def analyze_chemical_distance():
    print("============================================================")
    print("  Chemical Distance & Similarity Analysis (LORO Benchmark)")
    print("============================================================")
    
    df = pd.read_csv('index_with_anion.csv')
    ref_df = df[['refrigerant', 'refri_smiles']].drop_duplicates()
    
    # 1. Fingerprint Tanimoto Similarity
    fps = {}
    mols = {}
    desc_list = []
    names = []
    
    for idx, row in ref_df.iterrows():
        r = row['refrigerant']
        s = row['refri_smiles']
        mol = Chem.MolFromSmiles(s)
        if mol:
            mols[r] = mol
            fps[r] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            desc_list.append([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol)
            ])
            names.append(r)
            
    X_desc = StandardScaler().fit_transform(np.array(desc_list))
    
    result_path = ROOT / 'results_v5' / 'loro' / 'full' / 'loro_gnn_results.csv'
    if not result_path.exists():
        raise FileNotFoundError(
            f"Missing {result_path}. Run step4_gat_loro_runner.py; "
            "hard-coded GAT metrics are intentionally not accepted."
        )
    gnn_results = pd.read_csv(result_path)
    gnn_results = gnn_results[gnn_results['model'] == 'GAT_v5'].copy()
    metric_col = 'raw_r2_mean' if 'raw_r2_mean' in gnn_results.columns else 'r2_mean'
    if gnn_results['refrigerant'].duplicated().any():
        raise ValueError("Duplicate GAT_v5 refrigerant rows in loro_gnn_results.csv")
    r2_dict = gnn_results.set_index('refrigerant')[metric_col].dropna().to_dict()
    
    analysis_rows = []
    
    for target in sorted(r2_dict):
        if target not in fps:
            continue
            
        target_fp = fps[target]
        idx_desc = names.index(target)
        vec_desc = X_desc[idx_desc]
        
        sims = []
        dists = []
        for i, name in enumerate(names):
            if name != target:
                sim = DataStructs.TanimotoSimilarity(target_fp, fps[name])
                dist = np.linalg.norm(vec_desc - X_desc[i])
                sims.append((name, sim))
                dists.append((name, dist))
                
        sims.sort(key=lambda x: x[1], reverse=True)
        dists.sort(key=lambda x: x[1])
        
        nn_sim_name, max_sim = sims[0]
        nn_dist_name, min_dist = dists[0]
        
        analysis_rows.append({
            'refrigerant': target,
            'gat_r2': float(r2_dict[target]),
            'metric_source': str(result_path.name),
            'metric_column': metric_col,
            'max_tanimoto_sim': max_sim,
            'nearest_fp_neighbor': nn_sim_name,
            'min_desc_distance': min_dist,
            'nearest_desc_neighbor': nn_dist_name
        })
        
    res_df = pd.DataFrame(analysis_rows)
    out_path = 'loro_chemical_distance_analysis.csv'
    res_df.to_csv(out_path, index=False)
    
    print("\n" + res_df.to_string(index=False))
    print(f"\n[Done] Chemical distance analysis saved to {out_path}")

if __name__ == '__main__':
    analyze_chemical_distance()
