import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import math

print("开始执行 Tri-Graph 模型的数据预处理 (V3 - Ablation Experiments 增强版)...")
print("12维环境与物理化学描述符拼接版本")

# ==========================================
# 模块 1：建立“分子花名册” (字典)
# ==========================================
print("正在从 smiles.csv 加载已有的离子液体 SMILES 字典...")

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
    'R161':       'CCF',                             
    'R218':       'FC(F)(F)C(F)(F)C(F)(F)F',        
    'R227EA':     'FC(F)(F)C(F)C(F)(F)F',           
    'R236FA':     'FC(F)(F)CC(F)(F)F',              
    'R245FA':     'FC(F)(F)CC(F)F',                 
    'R1233ZD(E)': 'FC(F)(F)/C=C/Cl',               
    'R1336MZZ(E)':'FC(F)(F)/C=C/C(F)(F)F',         
    'R1336MZZ(Z)':'FC(F)(F)/C=C\\C(F)(F)F', 
    
    'P4442':      'CCCC[P+](CCCC)(CCCC)CC',                  
    'P66614':     'CCCCCC[P+](CCCCCC)(CCCCCC)CCCCCCCCCCCCCC',
    'DOIM':       'CCCCCCCCn1cc[n+](CCCCCCCC)c1',            
    'P44414':     'CCCC[P+](CCCC)(CCCC)CCCCCCCCCCCCCC',      
    'EMPY':       'CC[n+]1cccc(C)c1',                        
    'BMPY':       'CCCC[n+]1cccc(C)c1',                      
    'DMPIM':      'CCCn1cc[n+](C)c1C',                       
    'P4441':      'CCCC[P+](CCCC)(CCCC)C',                   
    'C8H4F13C1IM':'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CCn1cc[n+](C)c1', 
    
    'ET2PO4':     'CCOP(=O)([O-])OCC',                               
    'BEI':        'FC(F)(F)C(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)C(F)(F)F', 
    'TTES':       'FC(F)(F)OC(F)C(F)(F)S(=O)(=O)[O-]',               
    'HFPS':       'FC(F)(F)C(F)C(F)(F)S(=O)(=O)[O-]',                
    'PFBS':       'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)[O-]',      
    'TMPP':       'CC(C)(C)CC(C)CCP(=O)([O-])CC(C)CC(C)(C)C',        
    'FS':         'FC(F)(F)C(F)OC(F)(F)C(F)(F)S(=O)(=O)[O-]',        
    'FEP':        'F[P-](F)(F)(C(F)(F)C(F)(F)F)(C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F', 
    'PR':         'CCC(=O)[O-]',                                     
    'OTF':        'FC(F)(F)S(=O)(=O)[O-]',                           
    'TPES':       'FC(F)(F)C(F)(F)OC(F)C(F)(F)S(=O)(=O)[O-]',        
    'I':          '[I-]',                                            
    'TFES':       'FC(F)C(F)(F)S(=O)(=O)[O-]',                       
    'PFP':        'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)[O-]',          
    'PE':         'CCCCC(=O)[O-]',                                   
    'TMEM':       'FC(F)(F)S(=O)(=O)[C-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F', 
}

for k, v in extra_smiles.items():
    smiles_dict[k.upper()] = v
    smiles_dict[k.upper().replace('[','').replace(']','')] = v

# ==========================================
# 模块 2：RDKit 特征提取
# ==========================================
ELECTRONEG = {
    1: 2.20,   
    5: 2.04,   
    6: 2.55,   
    7: 3.04,   
    8: 3.44,   
    9: 3.98,   
    15: 2.19,  
    16: 2.58,  
    17: 3.16,  
    35: 2.96,  
    53: 2.66,  
}
ENEG_MIN, ENEG_MAX = 2.04, 3.98  

COV_RADIUS = {
    1: 31,    
    5: 84,    
    6: 77,    
    7: 71,    
    8: 66,    
    9: 64,    
    15: 107,  
    16: 105,  
    17: 102,  
    35: 120,  
    53: 139,  
}
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
    bond_type_dict = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2, 
                      Chem.rdchem.BondType.TRIPLE: 3, Chem.rdchem.BondType.AROMATIC: 4}
    return [
        bond_type_dict.get(bond.GetBondType(), 1), 
        1 if bond.IsInRing() else 0,              
        1 if bond.GetIsAromatic() else 0          
    ]

# ==========================================
# 模块 3：组装分子图
# ==========================================
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

def lookup_smiles(name):
    name = str(name).strip().upper()
    name_no_bracket = name.replace('[', '').replace(']', '')
    if name in smiles_dict: return smiles_dict[name]
    if name_no_bracket in smiles_dict: return smiles_dict[name_no_bracket]
    name_no_hyphen = name_no_bracket.replace('-', '')
    if name_no_hyphen in smiles_dict: return smiles_dict[name_no_hyphen]
    return None

# ==========================================
# 模块 4：xTB 数据加载及数据处理
# ==========================================
print("正在加载 xTB 物理描述符文件...")
xtb_lookup = {}
xtb_path = 'Phase4_Scientific_Validation/xTB_Physics_Descriptors.csv'
if os.path.exists(xtb_path):
    xtb_df = pd.read_csv(xtb_path)
    # 只取制冷剂部分
    xtb_ref = xtb_df[xtb_df['Category'] == 'Refrigerant']
    for _, row in xtb_ref.iterrows():
        mol_name = str(row['Molecule']).strip().upper()
        # 获取 xTB 计算结果
        dipole = row['Dipole_Debye']
        polar = row['Polarizability_au']
        vol = row['Volume_A3']
        xtb_lookup[mol_name] = (dipole, polar, vol)
else:
    print(f"警告：未找到 xTB 数据文件 {xtb_path}")


excel_name = 'ZLJ_DATA.xlsx'
if not os.path.exists(excel_name):
    if os.path.exists('../'+excel_name):
        excel_name = '../'+excel_name

print(f"正在从 {excel_name} 读取制冷剂气液相平衡数据 (VLE)...")
dfs = []
for sheet in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']:
    try:
        tmp_df = pd.read_excel(excel_name, sheet_name=sheet, skiprows=2)
        dfs.append(tmp_df)
    except Exception as e:
        print(f"跳过页签 {sheet}: {e}")

if not dfs:
    print("严重错误: 空空如也。请检查你放对 ZLJ_DATA.xlsx 没？")
    exit()

df_vle = pd.concat(dfs, ignore_index=True)
df_vle = df_vle.dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])

final_data = []    
final_labels = []  
meta_data = []     

# 用于统计的信息
total_processed = 0
total_saved = 0
skipped_due_to_smiles = 0
skipped_due_to_rdkit = 0
skipped_due_to_xtb = 0
xtb_missing_mols = set()

for idx, row in df_vle.iterrows():
    total_processed += 1
    
    c_name = str(row['IL cation']).strip()
    a_name = str(row['IL anion']).strip()
    r_name = str(row['Refrigerant']).strip()

    c_smi = lookup_smiles(c_name)   
    a_smi = lookup_smiles(a_name)    
    r_smi = lookup_smiles(r_name) 
    
    if None in (c_smi, a_smi, r_smi): 
        skipped_due_to_smiles += 1
        continue
    
    c_graph = mol2graph_components(c_smi)
    a_graph = mol2graph_components(a_smi)
    r_graph = mol2graph_components(r_smi)
    
    if None in (c_graph, a_graph, r_graph): 
        skipped_due_to_rdkit += 1
        continue
    
    # 获取 xTB 特征（仅限制冷剂）
    r_name_upper = r_name.upper()
    if r_name_upper not in xtb_lookup:
        # 硬错误拦截，不可隐式填充为 0
        print(f"[xTB 缺失错误] 未找到制冷剂 {r_name_upper} 的 xTB 数据，跳过该行。")
        xtb_missing_mols.add(r_name_upper)
        skipped_due_to_xtb += 1
        continue
        
    ref_dipole, ref_polarizability, ref_volume = xtb_lookup[r_name_upper]
    
    # 检查是否有 NaN
    if any(pd.isna(x) or math.isnan(x) for x in (ref_dipole, ref_polarizability, ref_volume)):
        print(f"[xTB 数据无效] 制冷剂 {r_name_upper} 的 xTB 数据包含 NaN，跳过该行。")
        xtb_missing_mols.add(r_name_upper)
        skipped_due_to_xtb += 1
        continue
        
    # V3: 计算 RDKit 分子属性
    ref_mol = Chem.MolFromSmiles(r_smi) 
    ani_mol = Chem.MolFromSmiles(a_smi) 
    cat_mol = Chem.MolFromSmiles(c_smi) 
    
    # 1. Ref_Charge
    ref_charge = float(Descriptors.MaxAbsPartialCharge(ref_mol)) if ref_mol else 0.0
    # 2. Ref_LogP
    ref_logp   = float(Descriptors.MolLogP(ref_mol))             if ref_mol else 0.0
    # 3. Ani_MW
    ani_mw     = float(Descriptors.MolWt(ani_mol))               if ani_mol else 0.0
    # 4. Cat_Charge
    try:
        cat_charge = float(Descriptors.MaxAbsPartialCharge(cat_mol)) if cat_mol else 0.0
    except:
        cat_charge = 0.0
    # 5. Cat_TPSA
    cat_tpsa = float(Descriptors.TPSA(cat_mol)) if cat_mol else 0.0
    
    # V3 新增特征：制冷剂分子量和阳离子分子量
    ref_molwt = float(Descriptors.MolWt(ref_mol)) if ref_mol else 0.0
    cat_molwt = float(Descriptors.MolWt(cat_mol)) if cat_mol else 0.0

    final_data.append([
        c_graph, a_graph, r_graph,        # indices 0,1,2: graphs
        float(row['T (K)']),               # index 3: T
        float(row['P (MPa)']),             # index 4: P
        ref_charge,                        # index 5: ref_charge (original)
        ref_logp,                          # index 6: ref_logp (original)
        ani_mw,                            # index 7: ani_mw (original)
        cat_charge,                        # index 8: cat_charge (original)
        cat_tpsa,                          # index 9: cat_tpsa (original)
        ref_molwt,                         # index 10: NEW - refrigerant MolWt
        cat_molwt,                         # index 11: NEW - cation MolWt
        ref_dipole,                        # index 12: NEW - refrigerant xTB dipole
        ref_polarizability,                # index 13: NEW - refrigerant xTB polarizability
        ref_volume,                        # index 14: NEW - refrigerant xTB volume
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
    
    total_saved += 1

out_dir = 'processed_tri_data_v3'
os.makedirs(out_dir, exist_ok=True)
np.save(f'{out_dir}/data.npy', np.array(final_data, dtype=object))
np.save(f'{out_dir}/label.npy', np.array(final_labels, dtype=object))

meta_df = pd.DataFrame(meta_data)
meta_df.to_csv(f'{out_dir}/meta_info.csv', index=False)
meta_df.to_csv(f'{out_dir}/index_with_anion.csv', index=False)

print("\n" + "="*50)
print("V3 数据处理总结 (12维条件特征)")
print("="*50)
print(f"总计处理行数: {total_processed}")
print(f"最终保存行数: {total_saved}")
print(f"跳过行数 (字典未命中): {skipped_due_to_smiles}")
print(f"跳过行数 (RDKit 失败): {skipped_due_to_rdkit}")
print(f"跳过行数 (xTB 数据缺失/异常): {skipped_due_to_xtb}")
if xtb_missing_mols:
    print(f"-> 缺失的制冷剂列表: {', '.join(xtb_missing_mols)}")

print("\n[特征索引说明]")
feature_names = [
    "0: Temperature T (K)",
    "1: Pressure P (MPa)",
    "2: Ref_Charge",
    "3: Ref_LogP",
    "4: Ani_MW",
    "5: Cat_Charge",
    "6: Cat_TPSA",
    "7: Ref_MolWt (NEW)",
    "8: Cat_MolWt (NEW)",
    "9: Ref_xTB_Dipole (NEW)",
    "10: Ref_xTB_Polarizability (NEW)",
    "11: Ref_xTB_Volume (NEW)"
]
print("本次合并的 12 维连续特征顺序为:")
for f in feature_names:
    print(f"  {f}")
print("="*50)
print(f"所有输出均已保存至 {out_dir}/ 目录中！")
