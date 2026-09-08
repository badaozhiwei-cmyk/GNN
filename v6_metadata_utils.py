"""
v6_metadata_utils.py — 纯轻量分子物性元数据工具模块 (Zero Side-Effects)
===================================================================
【功能说明】
提供论文机制探针与图构建所需的基础物理/化学查表工具：
1. lookup_smiles: 离子液体与制冷剂 SMILES 查询
2. xtb_lookup: xTB 量化物理描述符 (mu, alpha, V)
3. NIST_CRITICAL: NIST 临界参数 (Tc, Pc, omega)
4. compute_tanimoto_dist: 分子指纹谷本距离
严禁包含任何数据读写、npy 生成或模型训练的顶层执行逻辑！
"""
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

base_dir = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. 加载 SMILES 字典
# ==========================================
smiles_dict = {}
smiles_csv_candidates = [
    os.path.join(base_dir, 'Original_Data', 'IL_smiles.csv'),
    os.path.join(base_dir, '..', 'Original_Data', 'IL_smiles.csv'),
    'Original_Data/IL_smiles.csv',
]
smiles_csv_path = None
for p in smiles_csv_candidates:
    if os.path.exists(p):
        smiles_csv_path = p
        break

if smiles_csv_path:
    il_df = pd.read_csv(smiles_csv_path)
    il_df.columns = [c.strip() for c in il_df.columns]
    for idx, row in il_df.iterrows():
        abbr = str(row['Abbreviation']).strip().upper()
        smiles_dict[abbr] = str(row['Smiles']).strip()
        smiles_dict[abbr.replace('[', '').replace(']', '')] = str(row['Smiles']).strip()

extra_smiles = {
    'R32':'C(F)F', 'R134A':'C(C(F)(F)F)F', 'R143A':'CC(F)(F)F', 'R125':'C(F)(F)(C(F)(F)F)',
    'R152A':'CC(F)F', 'R23':'C(F)(F)F', 'R41':'CF', 'R134':'FC(F)C(F)F', 'R161':'CCF',
    'R227EA':'FC(F)(F)C(F)C(F)(F)F', 'R236FA':'FC(F)(F)CC(F)(F)F', 'R245FA':'FC(F)(F)CC(F)F',
    'R114': 'C(C(F)(F)Cl)(F)(F)Cl',
    'R1234YF': 'C=C(F)C(F)(F)F',
    'R1234ZE(E)': 'F/C=C/C(F)(F)F',
}
for k, v in extra_smiles.items():
    smiles_dict[k] = v
    smiles_dict[k.replace('[', '').replace(']', '')] = v

def lookup_smiles(name):
    clean = str(name).strip().upper().replace('[', '').replace(']', '')
    return smiles_dict.get(clean, smiles_dict.get(str(name).strip().upper(), None))

# ==========================================
# 2. 加载 xTB 单分子物理描述符 (mu, alpha, V)
# ==========================================
xtb_lookup = {}
xtb_candidates = [
    os.path.join(base_dir, 'Phase4_Scientific_Validation', 'xTB_Physics_Descriptors.csv'),
    os.path.join(base_dir, '..', 'Phase4_Scientific_Validation', 'xTB_Physics_Descriptors.csv'),
    'Phase4_Scientific_Validation/xTB_Physics_Descriptors.csv',
]
xtb_path = None
for p in xtb_candidates:
    if os.path.exists(p):
        xtb_path = p
        break

if xtb_path:
    xtb_df = pd.read_csv(xtb_path)
    for _, row in xtb_df[xtb_df['Category'] == 'Refrigerant'].iterrows():
        xtb_lookup[str(row['Molecule']).strip().upper()] = (
            float(row['Dipole_Debye']), 
            float(row['Polarizability_au']), 
            float(row['Volume_A3'])
        )

# ==========================================
# 3. 加载 NIST 临界参数 (Tc, Pc, omega)
# ==========================================
NIST_CRITICAL = {
    'R23': (299.29, 4.832, 0.263), 'R32': (351.26, 5.782, 0.277), 'R41': (317.28, 5.897, 0.201),
    'R125': (339.17, 3.618, 0.305), 'R134A': (374.21, 4.059, 0.327), 'R134': (391.75, 4.641, 0.312),
    'R143A': (345.86, 3.761, 0.262), 'R152A': (386.41, 4.517, 0.275), 'R161': (375.25, 5.091, 0.217),
    'R227EA': (374.90, 2.925, 0.357), 'R236FA': (398.07, 3.200, 0.377), 'R245FA': (427.16, 3.651, 0.378),
    'R1234YF': (367.85, 3.382, 0.276), 'R1234ZE(E)': (382.51, 3.635, 0.313)
}

# ==========================================
# 4. Morgan 指纹 Tanimoto 距离计算
# ==========================================
def compute_tanimoto_dist(smi1, smi2):
    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=1024)
    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    return 1.0 - sim
