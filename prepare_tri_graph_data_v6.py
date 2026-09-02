"""
prepare_tri_graph_data_v6.py — 顶刊规范终极数据管道 (Unified V6 Schema)
包含 3张图 + 19个标量 (共 22 维)。
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import math

print("🚀 开始执行 V6 终极 Tri-Graph 数据预处理 (22 维 Schema)...")

# 1. 加载 SMILES
smiles_csv_path = 'Original_Data/smiles.csv' if os.path.exists('Original_Data/smiles.csv') else 'smiles.csv'
il_df = pd.read_csv(smiles_csv_path)
il_df.columns = [c.strip() for c in il_df.columns]
smiles_dict = {}
for idx, row in il_df.iterrows():
    abbr = str(row['Abbreviation']).strip().upper()  
    smiles_dict[abbr] = str(row['Smiles']).strip() 
    smiles_dict[abbr.replace('[', '').replace(']', '')] = str(row['Smiles']).strip() 

# 补充缺失制冷剂 (缩略)
extra_smiles = {
    'R32':'C(F)F', 'R134A':'C(C(F)(F)F)F', 'R143A':'CC(F)(F)F', 'R125':'C(F)(F)(C(F)(F)F)',
    'R152A':'CC(F)F', 'R23':'C(F)(F)F', 'R41':'CF', 'R134':'FC(F)C(F)F', 'R161':'CCF',
    'R227EA':'FC(F)(F)C(F)C(F)(F)F', 'R236FA':'FC(F)(F)CC(F)(F)F', 'R245FA':'FC(F)(F)CC(F)F',
    'R114': 'C(C(F)(F)Cl)(F)(F)Cl', 'R1234YF': 'C(=C(F)F)(C(F)(F)F)F', 'R1234ZE(E)': 'F/C=C/C(F)(F)F',
}
for k, v in extra_smiles.items():
    smiles_dict[k] = v
    smiles_dict[k.replace('[', '').replace(']', '')] = v

def lookup_smiles(name):
    clean = str(name).strip().upper().replace('[', '').replace(']', '')
    return smiles_dict.get(clean, smiles_dict.get(str(name).strip().upper(), None))

# ==========================================
# 2. RDKit 特征提取函数
# ==========================================
ELECTRONEG = {1:2.20, 5:2.04, 6:2.55, 7:3.04, 8:3.44, 9:3.98, 15:2.19, 16:2.58, 17:3.16, 35:2.96, 53:2.66}
ENEG_MIN, ENEG_MAX = 2.04, 3.98  
COV_RADIUS = {1:31, 5:84, 6:77, 7:71, 8:66, 9:64, 15:107, 16:105, 17:102, 35:120, 53:139}
RAD_MIN, RAD_MAX = 31, 139  

def bucketize(val, min_v, max_v, n_buckets=8):
    ratio = (val - min_v) / (max_v - min_v + 1e-8)
    return min(int(ratio * n_buckets), n_buckets - 1)

def get_atom_features(atom):
    hybrid = int(atom.GetHybridization())
    if hybrid >= 8: hybrid = 7
    aro = 1 if atom.GetIsAromatic() else 0
    degree = atom.GetDegree()
    if degree >= 7: degree = 6
    charge = atom.GetFormalCharge() + 1
    if charge > 2: charge = 2
    if charge < 0: charge = 0
    atomic_num = atom.GetAtomicNum()
    eneg_val = ELECTRONEG.get(atomic_num, 2.55)
    eneg_bucket = bucketize(eneg_val, ENEG_MIN, ENEG_MAX, n_buckets=8)
    rad_val = COV_RADIUS.get(atomic_num, 77)
    radius_bucket = bucketize(rad_val, RAD_MIN, RAD_MAX, n_buckets=8)
    return [atomic_num, hybrid, aro, degree, charge, eneg_bucket, radius_bucket]

def get_bond_features(bond):
    bond_type_dict = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2, Chem.rdchem.BondType.TRIPLE: 3, Chem.rdchem.BondType.AROMATIC: 4}
    return [bond_type_dict.get(bond.GetBondType(), 1), 1 if bond.IsInRing() else 0, 1 if bond.GetIsAromatic() else 0]

def mol2graph_components(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return None 
    node_f = [get_atom_features(atom) for atom in mol.GetAtoms()]
    edge_index = [[], []] 
    edge_attr = []        
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx(); j = bond.GetEndAtomIdx() 
        f = get_bond_features(bond) 
        edge_index[0].extend([i, j])
        edge_index[1].extend([j, i])
        edge_attr.extend([f, f])
    if len(edge_attr) == 0:
        edge_index = [[], []]
        edge_attr = []
    return [node_f, edge_index, edge_attr]

# ==========================================
# 3. 加载 xTB 单分子物理描述符 (mu, alpha, V)
# ==========================================
xtb_lookup = {}
xtb_path = 'Phase4_Scientific_Validation/xTB_Physics_Descriptors.csv'
if os.path.exists(xtb_path):
    xtb_df = pd.read_csv(xtb_path)
    for _, row in xtb_df[xtb_df['Category'] == 'Refrigerant'].iterrows():
        xtb_lookup[str(row['Molecule']).strip().upper()] = (row['Dipole_Debye'], row['Polarizability_au'], row['Volume_A3'])
else:
    print(f"[警告] 找不到 xTB 文件: {xtb_path}")

# ==========================================
# 4. 加载 xTB 离子-制冷剂 配对结合能 (Delta E)
# ==========================================
pair_lookup = {}
pair_csv = 'Phase4_Scientific_Validation/full_pair_interaction_results.csv'
if os.path.exists(pair_csv):
    df_pair = pd.read_csv(pair_csv)
    for _, r in df_pair.iterrows():
        k = (str(r['Pair_Type']), str(r['Ion_Name']).strip().upper(), str(r['Refrigerant']).strip().upper())
        pair_lookup[k] = float(r['Delta_E_int_kcal_mol']) if pd.notna(r['Delta_E_int_kcal_mol']) else 0.0
else:
    print(f"[警告] 找不到超分子结合能文件: {pair_csv}")

# ==========================================
# 5. 加载 NIST 临界参数 (Tc, Pc, omega)
# ==========================================
NIST_CRITICAL = {
    'R23': (299.29, 4.832, 0.263), 'R32': (351.26, 5.782, 0.277), 'R41': (317.28, 5.897, 0.201),
    'R125': (339.17, 3.618, 0.305), 'R134A': (374.21, 4.059, 0.327), 'R134': (391.75, 4.641, 0.312),
    'R143A': (345.86, 3.761, 0.262), 'R152A': (386.41, 4.517, 0.275), 'R161': (375.25, 5.091, 0.217),
    'R227EA': (374.90, 2.925, 0.357), 'R236FA': (398.07, 3.200, 0.377), 'R245FA': (427.16, 3.651, 0.378),
    'R1234YF': (367.85, 3.382, 0.276), 'R1234ZE(E)': (382.51, 3.635, 0.313)
}

# ==========================================
# 6. 主循环生成数据
# ==========================================
excel_name = 'ZLJ_DATA.xlsx'
if not os.path.exists(excel_name): excel_name = '../' + excel_name
df_vle = pd.concat([pd.read_excel(excel_name, sheet_name=s, skiprows=2) for s in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']], ignore_index=True)
df_vle = df_vle.dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])

final_data, final_labels, meta_data = [], [], []
total_processed, saved_count = 0, 0

for idx, row in df_vle.iterrows():
    total_processed += 1
    c_name, a_name, r_name = str(row['IL cation']).strip(), str(row['IL anion']).strip(), str(row['Refrigerant']).strip()
    r_upper = r_name.upper()
    c_smi, a_smi, r_smi = lookup_smiles(c_name), lookup_smiles(a_name), lookup_smiles(r_name)
    
    if None in (c_smi, a_smi, r_smi): continue
    c_graph, a_graph, r_graph = mol2graph_components(c_smi), mol2graph_components(a_smi), mol2graph_components(r_smi)
    if None in (c_graph, a_graph, r_graph): continue
    
    # 获取 xTB 标量
    if r_upper not in xtb_lookup: continue
    ref_dipole, ref_polarizability, ref_volume = xtb_lookup[r_upper]
    
    # 获取相互作用能
    de_anion = pair_lookup.get(('Anion-Ref', a_name.upper(), r_upper), 0.0)
    de_cation = pair_lookup.get(('Cation-Ref', c_name.upper(), r_upper), 0.0)
    
    # 获取 NIST 热力学
    if r_upper not in NIST_CRITICAL: continue
    Tc, Pc, omega = NIST_CRITICAL[r_upper]
    T_val, P_val = float(row['T (K)']), float(row['P (MPa)'])
    Tr, Pr = T_val / Tc, P_val / Pc
    
    # 获取 RDKit 基础特征
    ref_mol, ani_mol, cat_mol = Chem.MolFromSmiles(r_smi), Chem.MolFromSmiles(a_smi), Chem.MolFromSmiles(c_smi)
    ref_charge = float(Descriptors.MaxAbsPartialCharge(ref_mol)) if ref_mol else 0.0
    ref_logp   = float(Descriptors.MolLogP(ref_mol)) if ref_mol else 0.0
    ani_mw     = float(Descriptors.MolWt(ani_mol)) if ani_mol else 0.0
    try: cat_charge = float(Descriptors.MaxAbsPartialCharge(cat_mol)) if cat_mol else 0.0
    except: cat_charge = 0.0
    cat_tpsa   = float(Descriptors.TPSA(cat_mol)) if cat_mol else 0.0
    ref_mw     = float(Descriptors.MolWt(ref_mol)) if ref_mol else 0.0
    cat_mw     = float(Descriptors.MolWt(cat_mol)) if cat_mol else 0.0

    # 严谨按照 22 维 Schema 拼装
    final_data.append([
        c_graph, a_graph, r_graph,    # 0, 1, 2
        T_val, P_val,                 # 3, 4
        ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa, ref_mw, cat_mw, # 5~11
        ref_dipole, ref_polarizability, ref_volume,                         # 12, 13, 14
        de_anion, de_cation,                                                # 15, 16
        Tc, Pc, omega, Tr, Pr                                               # 17~21
    ])
    final_labels.append(float(row['x1']))
    meta_data.append({'IL cation': c_name, 'IL anion': a_name, 'Refrigerant': r_name, 'T (K)': T_val, 'P (MPa)': P_val, 'x1': row['x1']})
    saved_count += 1

out_dir = 'processed_tri_data_v6'
os.makedirs(out_dir, exist_ok=True)
np.save(f'{out_dir}/data.npy', np.array(final_data, dtype=object))
np.save(f'{out_dir}/label.npy', np.asarray(final_labels, dtype=np.float32))
pd.DataFrame(meta_data).to_csv(f'{out_dir}/meta_info.csv', index=False)
pd.DataFrame(meta_data).to_csv(f'{out_dir}/index_with_anion.csv', index=False)

print(f"🎉 成功生成 22 维无歧义数据集，共保存 {saved_count} 条，保存在 {out_dir}/ 下！")
print("特征校验：", len(final_data[0]), "维 (预期 22)")
