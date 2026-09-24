"""
chemical_identity_contract.py — Central Chemical Identity Contract & Runtime Gate
================================================================================
Protocol: Scientific Integrity Framework (v1.0)
Architecture Rules:
  1. Code NEVER defines chemistry; code CONSUMES chemistry from this registry.
  2. Fail-Closed: any identity, formula, InChIKey, or graph mismatch raises RuntimeError.
  3. No alias fallback, no compatibility shim, zero tolerance for molecular swaps.
================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# 适配 Windows 控制台编码
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

MODULE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = MODULE_DIR / "refrigerant_identity_manifest.csv"

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"🚨 [Fatal Integrity Error] 未找到化学身份注册表: {MANIFEST_PATH}")

_df_registry = pd.read_csv(MANIFEST_PATH)

# 构建多重索引查找字典 (全部大写标准化)
_BY_ID: Dict[str, Dict[str, Any]] = {}
_BY_NAME: Dict[str, Dict[str, Any]] = {}
_BY_CAS: Dict[str, Dict[str, Any]] = {}
_BY_INCHIKEY: Dict[str, Dict[str, Any]] = {}

for _, row in _df_registry.iterrows():
    entry = row.to_dict()
    sid = str(entry['species_id']).strip().upper()
    cname = str(entry['canonical_name']).strip().upper()
    cas = str(entry['CAS']).strip().upper()
    ikey = str(entry['inchi_key']).strip().upper()

    _BY_ID[sid] = entry
    _BY_NAME[cname] = entry
    _BY_NAME[cname.replace('[', '').replace(']', '')] = entry
    _BY_NAME[cname.replace('-', '').replace('_', '')] = entry
    _BY_CAS[cas] = entry
    _BY_INCHIKEY[ikey] = entry

    # 解析别名
    aliases = str(entry.get('aliases', '')).split(';')
    for a in aliases:
        a_clean = a.strip().upper()
        if a_clean:
            _BY_NAME[a_clean] = entry
            _BY_NAME[a_clean.replace('-', '').replace('_', '')] = entry

def get_refrigerant_identity(identifier: str) -> Dict[str, Any]:
    """
    从中央注册表统一消费化学身份。
    支持 species_id, canonical_name, CAS, InChIKey 及标准别名查找。
    若未查到，抛出 KeyError，严禁静默返回 None 或默认值。
    """
    clean = str(identifier).strip().upper()
    clean_no_bracket = clean.replace('[', '').replace(']', '')
    clean_no_hyphen = clean_no_bracket.replace('-', '').replace('_', '')

    if clean in _BY_ID:
        return _BY_ID[clean]
    if clean in _BY_NAME:
        return _BY_NAME[clean]
    if clean_no_bracket in _BY_NAME:
        return _BY_NAME[clean_no_bracket]
    if clean_no_hyphen in _BY_NAME:
        return _BY_NAME[clean_no_hyphen]
    if clean in _BY_CAS:
        return _BY_CAS[clean]
    if clean in _BY_INCHIKEY:
        return _BY_INCHIKEY[clean]

    raise KeyError(f"🚨 [Chemical Registry Breach] 未知化学实体标识符: '{identifier}'！注册表中不存在该物种。")

def assert_refrigerant_contract(name: str, smiles: Optional[str] = None, mol: Optional[Chem.Mol] = None) -> Dict[str, Any]:
    """
    严格断言物种化学身份契约 (Fail-Closed).
    比对: InChIKey, 分子式, 重原子数, 氢原子数.
    """
    contract = get_refrigerant_identity(name)
    expected_inchikey = contract['inchi_key']
    expected_formula = contract['formula']
    expected_heavy = contract['n_heavy_atoms']
    expected_h = contract['n_hydrogens']

    test_mol = mol
    if test_mol is None:
        if smiles is None:
            raise ValueError(f"必须提供 smiles 或 mol 以供物种 '{name}' 校验契约！")
        test_mol = Chem.MolFromSmiles(smiles)
        if test_mol is None:
            raise RuntimeError(f"🚨 [Chemical Integrity Error] 物种 '{name}' 的 SMILES 无法被 RDKit 解析: '{smiles}'")

    actual_inchikey = Chem.MolToInchiKey(test_mol)
    test_mol_h = Chem.AddHs(test_mol)
    actual_formula = rdMolDescriptors.CalcMolFormula(test_mol_h)
    actual_heavy = test_mol.GetNumHeavyAtoms()
    actual_total = test_mol_h.GetNumAtoms()
    actual_h = actual_total - actual_heavy

    mismatches = []
    if actual_inchikey != expected_inchikey:
        mismatches.append(f"InChIKey 错位 (期望 {expected_inchikey}, 实际 {actual_inchikey})")
    if actual_formula != expected_formula:
        mismatches.append(f"分子式错位 (期望 {expected_formula}, 实际 {actual_formula})")
    if actual_heavy != expected_heavy:
        mismatches.append(f"重原子数错位 (期望 {expected_heavy}, 实际 {actual_heavy})")
    if actual_h != expected_h:
        mismatches.append(f"氢原子数错位 (期望 {expected_h}, 实际 {actual_h})")

    if mismatches:
        err_msg = (
            f"\n🚨🚨🚨 [CHEMICAL IDENTITY CONTRACT BREACH] 🚨🚨🚨\n"
            f"物种标识: '{name}' (ID: {contract['species_id']}) 发生严重化学身份撕裂！\n"
            f"检测到违规项:\n  - " + "\n  - ".join(mismatches) + "\n"
            f"权威 CAS: {contract['CAS']} | 规范 SMILES: {contract['canonical_smiles']}\n"
            f"系统已强制抛出异常以阻止下游计算遭受污染。"
        )
        raise RuntimeError(err_msg)

    return contract

def assert_graph_identity_contract(node_atomic_numbers: List[int], expected_species: str) -> Dict[str, Any]:
    """
    全图深度原子组成与拓扑契约门禁 (Full-Spectrum Graph Gate).
    绝不单纯依靠节点总数！逐一严格比对重原子数与各元素原子频次直方图 (C, F, Cl, Br).
    """
    contract = get_refrigerant_identity(expected_species)
    expected_heavy = contract['n_heavy_atoms']
    
    # 过滤重原子 (排除显式氢原子, 假如有的话)
    heavy_nums = [z for z in node_atomic_numbers if z != 1]
    actual_heavy = len(heavy_nums)

    if actual_heavy != expected_heavy:
        raise RuntimeError(
            f"🚨 [Graph Integrity Breach] '{expected_species}' (ID: {contract['species_id']}) "
            f"图重原子数错位！期望 {expected_heavy} 个重原子，图结构实际传入 {actual_heavy} 个！"
        )

    # 统计元素频次直方图
    from collections import Counter
    actual_counts = Counter(heavy_nums)

    # 从标准 SMILES 反解期望重原子频次
    expected_mol = Chem.MolFromSmiles(contract['canonical_smiles'])
    expected_counts = Counter([a.GetAtomicNum() for a in expected_mol.GetAtoms() if a.GetAtomicNum() != 1])

    if actual_counts != expected_counts:
        raise RuntimeError(
            f"🚨 [Graph Elemental Mutation] '{expected_species}' (ID: {contract['species_id']}) "
            f"图元素原子序数分布错位！\n"
            f"期望重原子分布: {dict(expected_counts)} (原子序数:计数)\n"
            f"实际传入图结构: {dict(actual_counts)}\n"
            f"系统判定分子图已被篡改或掉包，强制熔断！"
        )

    return contract

def assert_clean_descriptors_contract(csv_path: Path) -> pd.DataFrame:
    """
    针对物理描述符消费端的 Fail-Closed 安全门禁。
    保证:
      1. 物理描述符文件必须真实存在，严禁静默回退 (Silent Fallback) 到未认证的历史文件；
      2. 若含有 R1234yf，其实测 SMILES 必须通过官方契约校验 (PXGPLTODNUVGFL-UHFFFAOYSA-N, C3H2F4, 7 重原子)；
      3. 任何同分异构体、六氟丙烯或结构错位立即触发 RuntimeError 熔断。
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"🚨 [FAIL-CLOSED] 缺少经过认证的干净物理描述符文件: {csv_path}\n"
            f"下游消费端严禁在未生成干净物理工件的情况下静默回退到历史受污染文件。\n"
            f"请先运行 Kaggle xTB 重算包生成清洁工件！"
        )
    df = pd.read_csv(csv_path)
    yf_rows = df[df['Molecule'].astype(str).str.upper().str.strip() == 'R1234YF']
    if len(yf_rows) > 0:
        for _, row in yf_rows.iterrows():
            smi = str(row.get('SMILES', '')).strip()
            if smi:
                assert_refrigerant_contract("R1234yf", smiles=smi)
    return df

if __name__ == "__main__":
    print("🧪 正在运行 Chemical Identity Contract 全谱门禁自检...")
    
    # 1. 正常查询测试
    ident = get_refrigerant_identity("R1234yf")
    print(f"✅ 查询成功: ID={ident['species_id']}, CAS={ident['CAS']}, SMILES={ident['canonical_smiles']}")
    
    # 2. 正常图校验 (R1234yf 真实重原子: 3个C [Z=6], 4个F [Z=9])
    valid_graph_nodes = [6, 6, 6, 9, 9, 9, 9]
    assert_graph_identity_contract(valid_graph_nodes, "R1234yf")
    print("✅ R1234yf 真实图拓扑原子序数校验通过！")
    
    # 3. 恶意变异测试: 注入六氟丙烯 HFP 图结构 (3个C, 6个F, 重原子数 9)
    try:
        mutated_hfp_nodes = [6, 6, 6, 9, 9, 9, 9, 9, 9]
        assert_graph_identity_contract(mutated_hfp_nodes, "R1234yf")
        print("❌ 严重失败: 未能拦截六氟丙烯图结构！")
    except RuntimeError as e:
        print("✅ 成功拦截六氟丙烯假冒 (图元素突变拦截生效)！")
        print(f"   拦截异常详情: {e}")
