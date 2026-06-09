import os
import pandas as pd
import numpy as np
from rdkit import Chem

excel_name = 'ZLJ_DATA.xlsx'
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
    'R32':        'C(F)F',                           # 二氟甲烷
    'R134A':      'C(C(F)(F)F)F',                    # 1,1,1,2-四氟乙烷
    'R143A':      'CC(F)(F)F',                       # 1,1,1-三氟乙烷
    'R125':       'C(F)(F)(C(F)(F)F)',               # 五氟乙烷
    'R114':       'C(C(F)(F)Cl)(F)(F)Cl',           # 1,2-二氯-1,1,2,2-四氟乙烷
    'R1234YF':    'C(=C(F)F)(C(F)(F)F)F',           # 2,3,3,3-四氟丙烯
    'R1234ZE(E)': 'F/C=C/C(F)(F)F',                 # (E)-1,3,3,3-四氟丙烯
    'R152A':      'CC(F)F',                          # 1,1-二氟乙烷
    'R23':        'C(F)(F)F',                        # 三氟甲烷
    'R41':        'CF',                              # 氟甲烷
    'AC':         'CC(=O)[O-]',                      # 某种缺失的阴离子补充
    'Tf2N':       'FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F', # 另一种极易遗漏报错的阴离子
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

def lookup_smiles(name):
    name = str(name).strip().upper()
    name_no_bracket = name.replace('[', '').replace(']', '')
    if name in smiles_dict: return smiles_dict[name]
    if name_no_bracket in smiles_dict: return smiles_dict[name_no_bracket]
    return None

dfs = []
for sheet in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']:
    try:
        tmp_df = pd.read_excel(excel_name, sheet_name=sheet, skiprows=2)
        dfs.append(tmp_df)
    except:
        pass
df_vle = pd.concat(dfs, ignore_index=True).dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])

matched_rows = []
for idx, row in df_vle.iterrows():
    c_smi = lookup_smiles(row['IL cation'])
    a_smi = lookup_smiles(row['IL anion'])
    r_smi = lookup_smiles(row['Refrigerant'])
    if None in (c_smi, a_smi, r_smi): continue
    matched_rows.append(row)

target_row = matched_rows[100]
print("=== Target Row 100 Info ===")
print("IL cation:", target_row['IL cation'])
print("IL anion:", target_row['IL anion'])
print("Refrigerant:", target_row['Refrigerant'])
print("T (K):", target_row['T (K)'])
print("P (MPa):", target_row['P (MPa)'])
print("x1:", target_row['x1'])

c_smi = lookup_smiles(target_row['IL cation'])
a_smi = lookup_smiles(target_row['IL anion'])
r_smi = lookup_smiles(target_row['Refrigerant'])

c_mol = Chem.MolFromSmiles(c_smi)
a_mol = Chem.MolFromSmiles(a_smi)
r_mol = Chem.MolFromSmiles(r_smi)

c_atoms = [atom.GetSymbol() for atom in c_mol.GetAtoms()]
a_atoms = [atom.GetSymbol() for atom in a_mol.GetAtoms()]
r_atoms = [atom.GetSymbol() for atom in r_mol.GetAtoms()]

print("\n=== Atoms count ===")
print(f"Cation atoms (total={len(c_atoms)}):", c_atoms)
print(f"Anion atoms (total={len(a_atoms)}):", a_atoms)
print(f"Refrigerant atoms (total={len(r_atoms)}):", r_atoms)

print("\n=== Combined Node Index Map ===")
current_idx = 0
for i, symbol in enumerate(c_atoms):
    print(f"Node {current_idx} -> Cation {symbol}")
    current_idx += 1
for i, symbol in enumerate(a_atoms):
    print(f"Node {current_idx} -> Anion {symbol}")
    current_idx += 1
for i, symbol in enumerate(r_atoms):
    print(f"Node {current_idx} -> Refrigerant {symbol}")
    current_idx += 1
