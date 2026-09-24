"""
v6_metadata_utils.py — Central Molecular Property & Metadata Interface
======================================================================
Protocol: Scientific Integrity Framework (v1.0)
Rule: Code NEVER defines chemistry; code CONSUMES chemistry from the
      central Chemical Identity Registry.
======================================================================
"""

import os
import sys
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

base_dir = Path(__file__).resolve().parent

# 引入中央化学身份契约引擎
sys.path.insert(0, str(base_dir / "Phase4_Scientific_Validation"))
from chemical_identity_contract import (
    get_refrigerant_identity,
    assert_refrigerant_contract,
    _df_registry
)

# ==========================================
# 1. 动态生成 NIST 临界参数 (Tc, Pc, omega)
# ==========================================
# 彻底消除硬编码！完全从中央注册表 _df_registry 动态投影
NIST_CRITICAL = {}
for _, row in _df_registry.iterrows():
    cname_clean = str(row['canonical_name']).strip().upper().replace('[', '').replace(']', '')
    tc = float(row['Tc_K'])
    pc = float(row['Pc_MPa'])
    om = float(row['omega'])
    NIST_CRITICAL[cname_clean] = (tc, pc, om)
    # 额外兼容无连字符 key (如 R1234ZEE)
    NIST_CRITICAL[cname_clean.replace('-', '').replace('_', '')] = (tc, pc, om)

# ==========================================
# 2. 动态生成 xTB 单分子物理描述符 (mu, alpha, V)
# ==========================================
xtb_lookup = {}
for _, row in _df_registry.iterrows():
    cname_clean = str(row['canonical_name']).strip().upper().replace('[', '').replace(']', '')
    if pd.notna(row.get('Dipole_Debye')) and pd.notna(row.get('Polarizability_au')) and pd.notna(row.get('Volume_A3')):
        xtb_lookup[cname_clean] = (
            float(row['Dipole_Debye']),
            float(row['Polarizability_au']),
            float(row['Volume_A3'])
        )
        xtb_lookup[cname_clean.replace('-', '').replace('_', '')] = xtb_lookup[cname_clean]

# ==========================================
# 3. 离子液体 SMILES 字典加载 (消费自中央 IL 注册表)
# ==========================================
_il_manifest_path = base_dir / "Phase4_Scientific_Validation" / "ionic_liquid_identity_manifest.csv"
if not _il_manifest_path.exists():
    raise FileNotFoundError(f"未找到离子液体权威注册表: {_il_manifest_path}")

_df_il = pd.read_csv(_il_manifest_path)
_il_smiles_dict = {}
for _, r in _df_il.iterrows():
    cname = str(r['canonical_name']).strip().upper()
    cname_nb = cname.replace('[', '').replace(']', '')
    smi = str(r['canonical_smiles']).strip()
    _il_smiles_dict[cname] = smi
    _il_smiles_dict[cname_nb] = smi
    _il_smiles_dict[cname.replace('-', '').replace('_', '')] = smi

def lookup_smiles(name: str) -> str:
    """
    统一查询物种规范 SMILES:
    1. 优先在中央制冷剂注册表中消费 (强校验)
    2. 其次在中央离子液体注册表中消费 (强校验)
    3. 若均未查到，抛出 KeyError 熔断，严禁静默返回 None！
    """
    clean = str(name).strip().upper()
    clean_no_bracket = clean.replace('[', '').replace(']', '')

    # 优先查制冷剂中央注册表
    try:
        ident = get_refrigerant_identity(clean_no_bracket)
        return ident['canonical_smiles']
    except KeyError:
        pass

    # 查离子液体中央注册表
    if clean in _il_smiles_dict:
        return _il_smiles_dict[clean]
    if clean_no_bracket in _il_smiles_dict:
        return _il_smiles_dict[clean_no_bracket]

    raise KeyError(f"🚨 [Lookup Error] 物种 '{name}' 既不在制冷剂注册表，也不在离子液体注册表中！")

# ==========================================
# 4. Morgan 指纹 Tanimoto 距离计算
# ==========================================
def compute_tanimoto_dist(smi1: str, smi2: str) -> float:
    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)
    if m1 is None or m2 is None:
        raise ValueError(f"Invalid SMILES for Tanimoto computation: '{smi1}' or '{smi2}'")
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=1024)
    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    return 1.0 - sim

if __name__ == '__main__':
    print("🧪 正在测试重构后的 v6_metadata_utils...")
    r1234_smi = lookup_smiles('R1234yf')
    print(f"✅ R1234yf 消费自中央注册表: {r1234_smi} (期望: C=C(F)C(F)(F)F)")
    assert r1234_smi == 'C=C(F)C(F)(F)F', "🚨 R1234yf SMILES 严重错位！"
    
    tc, pc, om = NIST_CRITICAL['R1234YF']
    print(f"✅ R1234yf NIST 临界参数: Tc={tc} K, Pc={pc} MPa, omega={om}")
    assert abs(tc - 367.85) < 0.1, "🚨 R1234yf Tc 不符！"
    
    print("✅ v6_metadata_utils 全部测试通过，硬编码已彻底消除！")
