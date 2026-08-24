"""
prepare_tri_graph_data_v4.py — 融合超分子配对相互作用能的物理数据预处理
=====================================================================
【v4 核心升级】
  在 v3 的 12 维特征基础上，新增两组真实的【离子–制冷剂超分子配对相互作用能】：
    [15] Delta_E_anion  : 阴离子–制冷剂 GFN2-xTB 真实配对结合能 (kcal/mol)
    [16] Delta_E_cation : 阳离子–制冷剂 GFN2-xTB 真实配对结合能 (kcal/mol)
  
  数据输出目录：processed_tri_data_v4/
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import math

print("开始执行 Tri-Graph 模型的数据预处理 (V4 - 超分子配对相互作用能增强版)...")

# 1. 建立分子 SMILES 字典
smiles_csv_path = 'Original_Data/smiles.csv' if os.path.exists('Original_Data/smiles.csv') else 'smiles.csv'
il_df = pd.read_csv(smiles_csv_path)
il_df.columns = [c.strip() for c in il_df.columns]

smiles_dict = {}
for idx, row in il_df.iterrows():
    abbr = str(row['Abbreviation']).strip().upper()  
    abbr_no_bracket = abbr.replace('[', '').replace(']', '') 
    smi = str(row['Smiles']).strip() 
    smiles_dict[abbr] = smi          
    smiles_dict[abbr_no_bracket] = smi 

extra_smiles = {
    'R32':        'C(F)F',                           
    'R134A':      'C(C(F)(F)F)F',                    
    'R143A':      'CC(F)(F)F',                       
    'R125':       'C(F)(F)(C(F)(F)F)',               
    'R114':       'C(C(F)(F)Cl)(F)(F)Cl',           
    'R1234YF':    'C(=C(F)F)(C(F)(F)F)F',           
    'R1234ZE(E)': 'F/C=C/C(F)(F)F',                 
    'R152A':      'CC(F)F',                          
    'R23':        'C(F)(F)F',                        
    'R41':        'CF',                              
    'AC':         'CC(=O)[O-]',                      
    'Tf2N':       'FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F', 
    'R22':        'ClC(F)F',                         
    'R22B1':      'BrC(F)F',                         
    'R14':        'FC(F)(F)F',                       
    'R116':       'FC(F)(F)C(F)(F)F',               
    'R124':       'FC(F)(F)C(Cl)F',                 
    'R124A':      'ClC(F)C(F)(F)F',                 
    'R114A':      'ClC(Cl)(F)C(F)(F)F',             
    'R134':       'FC(F)C(F)F',                      
    'R227EA':     'FC(F)(C(F)(F)F)C(F)(F)F',         
    'R236EA':     'FC(F)C(F)C(F)(F)F',               
    'R236FA':     'FC(F)(F)CC(F)(F)F',               
    'R245FA':     'FCC(F)(F)C(F)(F)F',               
    'R365MFC':    'CC(F)(F)CC(F)(F)F',               
    'R218':       'FC(F)(F)C(F)(F)C(F)(F)F',         
    'R123':       'ClC(Cl)C(F)(F)F',                 
    'R141B':      'CC(Cl)(Cl)F',                     
    'R142B':      'CC(Cl)(F)F',                      
    'R1233ZD(E)': 'Cl/C=C/C(F)(F)F',                 
    'R1243ZF':    'C=CC(F)(F)F',                     
    'R1336MZZ(E)':'F/C(C(F)(F)F)=C/C(F)(F)F',        
    'R1336MZZ(Z)':'F/C(C(F)(F)F)=C\\C(F)(F)F',       
    'R11':        'ClC(Cl)(Cl)F',                    
    'R12':        'ClC(Cl)(F)F',                     
    'R13':        'ClC(F)(F)F',                      
    'R161':       'CCF'                              
}
for k, v in extra_smiles.items():
    smiles_dict[k.upper()] = v
    smiles_dict[k.upper().replace('[', '').replace(']', '')] = v

def lookup_smiles(name):
    clean_name = str(name).strip().upper()
    clean_no_bracket = clean_name.replace('[', '').replace(']', '')
    if clean_name in smiles_dict: return smiles_dict[clean_name]
    if clean_no_bracket in smiles_dict: return smiles_dict[clean_no_bracket]
    return None

# 特征字典
ELECTRONEG = {
    'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
    'B': 2.04, 'Al': 1.61, 'Fe': 1.83, 'Zn': 1.65
}
COV_RADIUS = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    'B': 0.84, 'Al': 1.21, 'Fe': 1.32, 'Zn': 1.22
}
ENEG_BINS   = [0.0, 2.0, 2.3, 2.6, 3.0, 3.3, 3.6, 4.0]
RADIUS_BINS = [0.0, 0.5, 0.7, 0.8, 1.0, 1.15, 1.3, 1.5]

def bucketize(val, bins):
    for i, b in enumerate(bins):
        if val <= b: return i
    return len(bins)

def get_atom_features(atom):
    sym = atom.GetSymbol()
    type_idx = atom.GetAtomicNum()
    h_idx = int(atom.GetHybridization())
    aro_idx = 1 if atom.GetIsAromatic() else 0
    deg_idx = atom.GetDegree()
    chg_idx = atom.GetFormalCharge() + 1
    eneg_val = ELECTRONEG.get(sym, 2.5)
    eneg_idx = bucketize(eneg_val, ENEG_BINS)
    rad_val = COV_RADIUS.get(sym, 0.8)
    rad_idx = bucketize(rad_val, RADIUS_BINS)
    return [type_idx, h_idx, aro_idx, deg_idx, chg_idx, eneg_idx, rad_idx]

def get_bond_features(bond):
    bt = bond.GetBondType()
    b_type = 1 if bt == Chem.rdchem.BondType.SINGLE else (
             2 if bt == Chem.rdchem.BondType.DOUBLE else (
             3 if bt == Chem.rdchem.BondType.TRIPLE else (
             4 if bt == Chem.rdchem.BondType.AROMATIC else 0)))
    b_ring = 1 if bond.IsInRing() else 0
    b_aro  = 1 if bond.GetIsAromatic() else 0
    return [b_type, b_ring, b_aro]

def mol2graph_components(smiles):
    if not smiles: return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_features.extend([bf, bf])
    if len(edge_indices) == 0:
        edge_index = [[], []]
        edge_attr = []
    else:
        edge_index = list(zip(*edge_indices))
        edge_attr = edge_features
    return [node_features, edge_index, edge_attr]

# 2. 读取单体 xTB 物理描述符
xtb_lookup = {}
xtb_path = 'Phase4_Scientific_Validation/xTB_Physics_Descriptors.csv'
if os.path.exists(xtb_path):
    xtb_df = pd.read_csv(xtb_path)
    xtb_ref = xtb_df[xtb_df['Category'] == 'Refrigerant']
    for _, row in xtb_ref.iterrows():
        mol_name = str(row['Molecule']).strip().upper()
        xtb_lookup[mol_name] = (row['Dipole_Debye'], row['Polarizability_au'], row['Volume_A3'])

# 3. 读取超分子配对结合能
pair_lookup = {}
pair_csv = 'Phase4_Scientific_Validation/full_pair_interaction_results.csv'
if os.path.exists(pair_csv):
    df_pair = pd.read_csv(pair_csv)
    for _, r in df_pair.iterrows():
        k = (str(r['Pair_Type']), str(r['Ion_Name']).strip().upper(), str(r['Refrigerant']).strip().upper())
        pair_lookup[k] = float(r['Delta_E_int_kcal_mol']) if pd.notna(r['Delta_E_int_kcal_mol']) else 0.0
    print(f"[Info] 成功加载超分子配对结合能表: {len(pair_lookup)} 个配对项")
else:
    print(f"[Warning] 未找到配对结合能表 {pair_csv}，请先运行 compute_full_pair_interaction_xtb.py！")

# 4. 读取原始相平衡数据
excel_name = 'ZLJ_DATA.xlsx'
if not os.path.exists(excel_name) and os.path.exists('../' + excel_name):
    excel_name = '../' + excel_name

dfs = []
for sheet in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']:
    try:
        tmp_df = pd.read_excel(excel_name, sheet_name=sheet, skiprows=2)
        dfs.append(tmp_df)
    except Exception as e:
        pass

df_vle = pd.concat(dfs, ignore_index=True).dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])

final_data = []    
final_labels = []  
meta_data = []     

for idx, row in df_vle.iterrows():
    c_name = str(row['IL cation']).strip()
    a_name = str(row['IL anion']).strip()
    r_name = str(row['Refrigerant']).strip()

    c_smi = lookup_smiles(c_name)   
    a_smi = lookup_smiles(a_name)    
    r_smi = lookup_smiles(r_name) 
    
    if None in (c_smi, a_smi, r_smi): continue
    c_graph = mol2graph_components(c_smi)
    a_graph = mol2graph_components(a_smi)
    r_graph = mol2graph_components(r_smi)
    if None in (c_graph, a_graph, r_graph): continue
    
    r_name_upper = r_name.upper()
    if r_name_upper not in xtb_lookup: continue
    ref_dipole, ref_polarizability, ref_volume = xtb_lookup[r_name_upper]
    
    # 提取配对相互作用能
    de_anion = pair_lookup.get(('Anion-Ref', a_name.upper(), r_name_upper), 0.0)
    de_cation = pair_lookup.get(('Cation-Ref', c_name.upper(), r_name_upper), 0.0)
    
    ref_mol = Chem.MolFromSmiles(r_smi) 
    ani_mol = Chem.MolFromSmiles(a_smi) 
    cat_mol = Chem.MolFromSmiles(c_smi) 
    
    ref_charge = float(Descriptors.MaxAbsPartialCharge(ref_mol)) if ref_mol else 0.0
    ref_logp   = float(Descriptors.MolLogP(ref_mol))             if ref_mol else 0.0
    ani_mw     = float(Descriptors.MolWt(ani_mol))               if ani_mol else 0.0
    try: cat_charge = float(Descriptors.MaxAbsPartialCharge(cat_mol)) if cat_mol else 0.0
    except: cat_charge = 0.0
    cat_tpsa   = float(Descriptors.TPSA(cat_mol)) if cat_mol else 0.0
    ref_molwt  = float(Descriptors.MolWt(ref_mol)) if ref_mol else 0.0
    cat_molwt  = float(Descriptors.MolWt(cat_mol)) if cat_mol else 0.0

    final_data.append([
        c_graph, a_graph, r_graph,        # indices 0,1,2: graphs
        float(row['T (K)']),               # index 3: T
        float(row['P (MPa)']),             # index 4: P
        ref_charge,                        # index 5: ref_charge
        ref_logp,                          # index 6: ref_logp
        ani_mw,                            # index 7: ani_mw
        cat_charge,                        # index 8: cat_charge
        cat_tpsa,                          # index 9: cat_tpsa
        ref_molwt,                         # index 10: ref_MolWt
        cat_molwt,                         # index 11: cat_MolWt
        ref_dipole,                        # index 12: ref_dipole
        ref_polarizability,                # index 13: ref_polarizability
        ref_volume,                        # index 14: ref_volume
        de_anion,                          # index 15: NEW - Delta_E_anion (kcal/mol)
        de_cation,                         # index 16: NEW - Delta_E_cation (kcal/mol)
    ])
    
    final_labels.append(float(row['x1']))
    meta_data.append({
        'IL cation': row['IL cation'],
        'IL anion': row['IL anion'],
        'Refrigerant': row['Refrigerant'],
        'T (K)': row['T (K)'],
        'P (MPa)': row['P (MPa)'],
        'x1': row['x1']
    })

out_dir = 'processed_tri_data_v4'
os.makedirs(out_dir, exist_ok=True)
np.save(f'{out_dir}/data.npy', np.array(final_data, dtype=object))
np.save(f'{out_dir}/label.npy', np.array(final_labels, dtype=object))

meta_df = pd.DataFrame(meta_data)
meta_df.to_csv(f'{out_dir}/meta_info.csv', index=False)
meta_df.to_csv(f'{out_dir}/index_with_anion.csv', index=False)

print(f"🎉 V4 数据预处理完成！共保存 {len(final_data)} 条样本至 {out_dir}/")
