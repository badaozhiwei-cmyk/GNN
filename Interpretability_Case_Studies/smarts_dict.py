import rdkit
from rdkit import Chem

# ==========================================
# SMARTS 模式字典：用于从原子级别的归因分数，映射并聚合到基团级别
# ==========================================

GROUP_SMARTS = {
    # ----------------------------------
    # 1. 阳离子特征基团 (Cation Groups)
    # ----------------------------------
    'Imidazolium_Ring': '[nR1]1[cR1][cR1][n+R1][cR1]1',  # 咪唑环核心 (注意：有时候带正电的N可能位置不同)
    'Imidazolium_Alt':  'n1cc[n+]c1',                      # 咪唑环备用匹配
    'Pyridinium_Ring':  '[n+R1]1[cR1][cR1][cR1][cR1][cR1]1', # 吡啶环核心
    'Phosphonium':      '[P+]',                            # 季鏻盐核心
    'Ammonium':         '[N+](C)(C)(C)C',                  # 季铵盐核心
    'Alkyl_Chain':      '[CX4;!R]',                        # 直链或支链烷基碳 (非环sp3碳)
    
    # ----------------------------------
    # 2. 阴离子特征基团 (Anion Groups)
    # ----------------------------------
    'Sulfonyl_SO2':     'S(=O)(=O)',                       # 磺酰基 (Tf2N 等)
    'Carboxylate_COO':  'C(=O)[O-]',                       # 羧酸根 (OAc 等)
    'BF4_Core':         '[B-](F)(F)(F)F',                  # 四氟硼酸根
    'PF6_Core':         '[P-](F)(F)(F)(F)(F)F',            # 六氟磷酸根
    'Phosphate_PO4':    'P(=O)([O-])(O)(O)',               # 磷酸根类
    
    # ----------------------------------
    # 3. 制冷剂特征基团 (Refrigerant Groups)
    # ----------------------------------
    'Trifluoromethyl_CF3': 'C(F)(F)F',                     # 三氟甲基
    'Difluoromethyl_CHF2': '[CH1](F)F',                    # 二氟甲基 (明确带一个氢)
    'Fluoromethyl_CH2F':   '[CH2]F',                       # 氟甲基 (明确带两个氢)
    'Double_Bond_C=C':     'C=C',                          # 碳碳双键 (HFOs 的标志性结构)
    'Chlorine_Cl':         '[Cl]',                         # 氯原子 (HCFCs)
    'Bromine_Br':          '[Br]',                         # 溴原子
    'Iodine_I':            '[I]',                          # 碘原子
    
    # ----------------------------------
    # 4. 杂项 (Miscellaneous)
    # ----------------------------------
    'Hydroxyl_OH':         '[OX2H]',                       # 羟基 (如果有醇类共溶剂)
    'Ether_O':             '[OX2](C)C',                    # 醚键
}

# 优先级列表（防止一个原子被分到多个基团导致重复计算）
# 例如 CF3 既匹配 Alkyl_Chain 也匹配 Trifluoromethyl_CF3，我们希望它优先匹配 CF3
GROUP_PRIORITY = [
    # 最高优先级：含氟特异性基团
    'Trifluoromethyl_CF3', 'Difluoromethyl_CHF2', 'Fluoromethyl_CH2F',
    'BF4_Core', 'PF6_Core',
    # 中高优先级：杂原子和特异骨架
    'Imidazolium_Ring', 'Imidazolium_Alt', 'Pyridinium_Ring', 'Phosphonium', 'Ammonium',
    'Sulfonyl_SO2', 'Carboxylate_COO', 'Phosphate_PO4',
    'Double_Bond_C=C', 'Chlorine_Cl', 'Bromine_Br', 'Iodine_I',
    # 最低优先级：普通烷基碳链
    'Alkyl_Chain'
]

def get_group_matches(smiles):
    """
    输入 SMILES，返回各个基团在该分子中包含的原子的索引集合
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return {}
    
    matched_atoms = set()
    group_mapping = {} # group_name -> list of atom indices
    
    # 按优先级进行匹配
    for group_name in GROUP_PRIORITY:
        smarts = GROUP_SMARTS[group_name]
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            matches = mol.GetSubstructMatches(patt)
            # 展平 matches 并排除已被更高优先级基团认领的原子
            valid_indices = []
            for match in matches:
                for atom_idx in match:
                    if atom_idx not in matched_atoms:
                        valid_indices.append(atom_idx)
                        matched_atoms.add(atom_idx)
            
            if valid_indices:
                group_mapping[group_name] = valid_indices
                
    # 收集剩余未匹配的原子作为 "Other_Atoms"
    other_atoms = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in matched_atoms:
            other_atoms.append(idx)
            
    if other_atoms:
        group_mapping['Other_Atoms'] = other_atoms
        
    return group_mapping

if __name__ == '__main__':
    print("=" * 50)
    print(" 🧪 Loaded SMARTS Functional Group Dictionary")
    print("=" * 50)
    
    # Test
    test_smiles = [
        ('Tf2N', 'FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F'),
        ('BMIM', 'CCCC[n+]1cccc(C)c1'),
        ('R134a', 'C(C(F)(F)F)F'),
        ('R1234yf', 'C(=C(F)F)(C(F)(F)F)F')
    ]
    
    for name, smi in test_smiles:
        print(f"\n[Test] Matching groups in {name} ({smi}):")
        mapping = get_group_matches(smi)
        mol = Chem.MolFromSmiles(smi)
        for g, atoms in mapping.items():
            symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in atoms]
            print(f"  ✅ {g:<20} -> atoms: {atoms} ({symbols})")
