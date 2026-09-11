"""
prepare_hfc2739_v6_data.py
==========================
将 Table S3 (Saturated HFC Universe: 2739 条样本, 12 种饱和冷媒, 全部 23 种阴离子, 6 大先验化学家族) 组装为标准的 V6 22 维数据结构。
直接复用已有的三分子图结构，并填入精确的 NIST 临界常数与对比态坐标。
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# 严格对齐 V6 的 22 维特征字段规范
FEATURE_SCHEMA = {
    "T": 3, "P": 4, 
    "ref_charge": 5, "ref_logp": 6, "ani_mw": 7, "cat_charge": 8, "cat_tpsa": 9, 
    "ref_MW": 10, "cat_MW": 11,
    "ref_dipole": 12, "ref_polarizability": 13, "ref_volume": 14,
    "deltaE_anion": 15, "deltaE_cation": 16,
    "Tc": 17, "Pc": 18, "omega": 19,
    "Tr": 20, "Pr": 21,
}

NIST_CRITICAL = {
    'R23': (299.29, 4.832, 0.263), 'R32': (351.26, 5.782, 0.277), 'R41': (317.28, 5.897, 0.201),
    'R125': (339.17, 3.618, 0.305), 'R134A': (374.21, 4.059, 0.327), 'R134': (391.75, 4.641, 0.312),
    'R143A': (345.86, 3.761, 0.262), 'R152A': (386.41, 4.517, 0.275), 'R161': (375.25, 5.091, 0.217),
    'R227EA': (374.90, 2.925, 0.357), 'R236FA': (398.07, 3.200, 0.377), 'R245FA': (427.16, 3.651, 0.378),
}

def main():
    print("============================================================")
    print("  Assembling V6 22-dim Dataset for HFC Universe (N=2739)")
    print("============================================================")
    
    raw_csv = 'index_with_anion.csv'
    df = pd.read_csv(raw_csv)
    hfc_mask = (df['sheet'] == 'Table S3. VLE HFCs')
    hfc_indices = df[hfc_mask].index.to_numpy()
    df_hfc = df[hfc_mask].copy().reset_index(drop=True)
    
    print(f"Loading graphs from processed_tri_data/data.npy for {len(hfc_indices)} HFC rows...")
    raw_data = np.load('processed_tri_data/data.npy', allow_pickle=True)
    raw_labels = np.load('processed_tri_data/label.npy', allow_pickle=True)
    
    final_data = []
    final_labels = []
    
    # 缓存分子量避免重复计算
    mw_cache = {}
    def get_mw(smi):
        if smi not in mw_cache:
            mol = Chem.MolFromSmiles(smi)
            mw_cache[smi] = float(Descriptors.MolWt(mol)) if mol else 0.0
        return mw_cache[smi]
        
    for new_idx, orig_idx in enumerate(hfc_indices):
        item = raw_data[orig_idx]
        c_graph, a_graph, r_graph = item[0], item[1], item[2]
        
        row = df_hfc.iloc[new_idx]
        T_val = float(row['T_K'])
        P_val = float(row['P_MPa'])
        ref_charge = float(item[5])
        ref_logp   = float(item[6])
        ani_mw     = float(item[7])
        cat_charge = float(item[8])
        cat_tpsa   = float(item[9])
        
        # 计算分子量
        ref_mw = get_mw(row['refri_smiles'])
        cat_mw = get_mw(row['cation_smiles'])
        
        # xTB 与结合能 (无单分子 xTB 时填 0.0)
        ref_dipole = 0.0
        ref_polarizability = 0.0
        ref_volume = 0.0
        de_anion = 0.0
        de_cation = 0.0
        
        # NIST 临界常数与对比态
        r_clean = str(row['refrigerant']).strip().upper().replace('-', '')
        Tc, Pc, omega = NIST_CRITICAL[r_clean]
        Tr = T_val / Tc
        Pr = P_val / Pc
        
        # 严格组装 22 维向量 (顺序与 Dataset_v6 FEATURE_SCHEMA 严格对齐)
        # 3: T, 4: P, 5: ref_charge, 6: ref_logp, 7: ani_mw, 8: cat_charge, 9: cat_tpsa
        # 10: ref_MW, 11: cat_MW, 12: ref_dipole, 13: ref_polarizability, 14: ref_volume
        # 15: deltaE_anion, 16: deltaE_cation, 17: Tc, 18: Pc, 19: omega, 20: Tr, 21: Pr
        cond_vec = [
            T_val, P_val,
            ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa,
            ref_mw, cat_mw,
            ref_dipole, ref_polarizability, ref_volume,
            de_anion, de_cation,
            Tc, Pc, omega,
            Tr, Pr
        ]
        
        final_data.append([c_graph, a_graph, r_graph] + cond_vec)
        final_labels.append(float(raw_labels[orig_idx]))
        
    out_dir = Path('processed_tri_data_hfc2739')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / 'data.npy', np.array(final_data, dtype=object))
    np.save(out_dir / 'label.npy', np.asarray(final_labels, dtype=np.float32))
    df_hfc.to_csv(out_dir / 'meta_info.csv', index=False)
    df_hfc.to_csv(out_dir / 'index_with_anion.csv', index=False)
    
    print(f"SUCCESS: Saved {len(final_data)} items to {out_dir}/")
    print(f"Features dimension check: {len(final_data[0])} (expected 22)")
    assert all(len(x) == 22 for x in final_data), "Schema assertion error: row length != 22"

if __name__ == '__main__':
    main()
