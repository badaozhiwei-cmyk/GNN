"""
step0_verify_alignment.py
=========================
【目的】
  1. 重现 prepare_tri_graph_data.py 的"跳过逻辑"，找出 Excel 里哪些行最终
     进入了 data.npy，并记录它们的顺序索引。
  2. 对 7 个抽查点，比较 Excel 原始记录 与 data.npy 里存的物理量(T, P, ani_mw)
     是否完全一致，以此证明图数据和 Excel 行严格对齐。
  3. 【额外产出】生成 index_with_anion.csv：
     每一行 = (npy_idx, excel_row, cation, anion, refrigerant, x1, anion_smiles)
     供 Step 1 划分代码直接读取，不用重跑 prepare 脚本。

【运行方法】
  在项目根目录下运行：
      python step0_verify_alignment.py

【输出文件】
  - index_with_anion.csv（Step 1 划分的输入）
  - alignment_check_report.txt（对齐验证报告）
"""

import os

# ── 自动定位项目根目录（脚本在 research_pipeline/ 子文件夹中）──────
import pathlib as _pl
ROOT = str(_pl.Path(__file__).resolve().parent.parent)
import os as _os; _os.chdir(ROOT)
# ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# ──────────────────────────────────────────────
# 1. 重建 smiles_dict（和 prepare_tri_graph_data.py 完全一致）
# ──────────────────────────────────────────────
print("=" * 60)
print("Step 0：数据对齐验证 + 索引映射表生成")
print("=" * 60)

smiles_csv_path = 'Original_Data/smiles.csv' if os.path.exists('Original_Data/smiles.csv') else 'smiles.csv'
il_df = pd.read_csv(smiles_csv_path)
il_df.columns = [c.strip() for c in il_df.columns]

smiles_dict = {}
for _, row in il_df.iterrows():
    abbr = str(row['Abbreviation']).strip().upper()
    abbr_nb = abbr.replace('[', '').replace(']', '')
    smi = str(row['Smiles']).strip()
    smiles_dict[abbr] = smi
    smiles_dict[abbr_nb] = smi

extra_smiles = {
    'R32': 'C(F)F', 'R134A': 'C(C(F)(F)F)F', 'R143A': 'CC(F)(F)F',
    'R125': 'C(F)(F)(C(F)(F)F)', 'R114': 'C(C(F)(F)Cl)(F)(F)Cl',
    'R1234YF': 'C(=C(F)F)(C(F)(F)F)F', 'R1234ZE(E)': 'F/C=C/C(F)(F)F',
    'R152A': 'CC(F)F', 'R23': 'C(F)(F)F', 'R41': 'CF',
    'AC': 'CC(=O)[O-]',
    'Tf2N': 'FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F',
    'R22': 'ClC(F)F', 'R22B1': 'BrC(F)F', 'R14': 'FC(F)(F)F',
    'R116': 'FC(F)(F)C(F)(F)F', 'R124': 'FC(F)(F)C(Cl)F',
    'R124A': 'ClC(F)C(F)(F)F', 'R114A': 'ClC(Cl)(F)C(F)(F)F',
    'R134': 'FC(F)C(F)F', 'R161': 'CCF', 'R218': 'FC(F)(F)C(F)(F)C(F)(F)F',
    'R227EA': 'FC(F)(F)C(F)C(F)(F)F', 'R236FA': 'FC(F)(F)CC(F)(F)F',
    'R245FA': 'FC(F)(F)CC(F)F', 'R1233ZD(E)': 'FC(F)(F)/C=C/Cl',
    'R1336MZZ(E)': 'FC(F)(F)/C=C/C(F)(F)F', 'R1336MZZ(Z)': 'FC(F)(F)/C=C\\C(F)(F)F',
    'P4442': 'CCCC[P+](CCCC)(CCCC)CC',
    'P66614': 'CCCCCC[P+](CCCCCC)(CCCCCC)CCCCCCCCCCCCCC',
    'DOIM': 'CCCCCCCCn1cc[n+](CCCCCCCC)c1',
    'P44414': 'CCCC[P+](CCCC)(CCCC)CCCCCCCCCCCCCC',
    'EMPY': 'CC[n+]1cccc(C)c1', 'BMPY': 'CCCC[n+]1cccc(C)c1',
    'DMPIM': 'CCCn1cc[n+](C)c1C', 'P4441': 'CCCC[P+](CCCC)(CCCC)C',
    'C8H4F13C1IM': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CCn1cc[n+](C)c1',
    'ET2PO4': 'CCOP(=O)([O-])OCC',
    'BEI': 'FC(F)(F)C(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)C(F)(F)F',
    'TTES': 'FC(F)(F)OC(F)C(F)(F)S(=O)(=O)[O-]',
    'HFPS': 'FC(F)(F)C(F)C(F)(F)S(=O)(=O)[O-]',
    'PFBS': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)[O-]',
    'TMPP': 'CC(C)(C)CC(C)CCP(=O)([O-])CC(C)CC(C)(C)C',
    'FS': 'FC(F)(F)C(F)OC(F)(F)C(F)(F)S(=O)(=O)[O-]',
    'FEP': 'F[P-](F)(F)(C(F)(F)C(F)(F)F)(C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F',
    'PR': 'CCC(=O)[O-]', 'OTF': 'FC(F)(F)S(=O)(=O)[O-]',
    'TPES': 'FC(F)(F)C(F)(F)OC(F)C(F)(F)S(=O)(=O)[O-]',
    'I': '[I-]', 'TFES': 'FC(F)C(F)(F)S(=O)(=O)[O-]',
    'PFP': 'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)[O-]',
    'PE': 'CCCCC(=O)[O-]',
    'TMEM': 'FC(F)(F)S(=O)(=O)[C-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F',
}
for k, v in extra_smiles.items():
    smiles_dict[k.upper()] = v
    smiles_dict[k.upper().replace('[', '').replace(']', '')] = v

def lookup_smiles(name):
    name = str(name).strip().upper()
    nb = name.replace('[', '').replace(']', '')
    if name in smiles_dict: return smiles_dict[name]
    if nb in smiles_dict: return smiles_dict[nb]
    nh = nb.replace('-', '')
    if nh in smiles_dict: return smiles_dict[nh]
    return None

# ──────────────────────────────────────────────
# 2. 重现 prepare 脚本的"过滤 + 顺序遍历"逻辑，
#    同时记录每条数据在 Excel 中的原始信息
# ──────────────────────────────────────────────
excel_name = 'ZLJ_DATA.xlsx'
if not os.path.exists(excel_name):
    excel_name = '../ZLJ_DATA.xlsx'

print(f"\n正在读取 {excel_name} ...")
dfs = []
for sheet in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']:
    try:
        tmp = pd.read_excel(excel_name, sheet_name=sheet, skiprows=2)
        tmp['_sheet'] = sheet  # 记录来源 sheet，方便追溯
        dfs.append(tmp)
        print(f"  ✔ 读取 {sheet}：{len(tmp)} 行")
    except Exception as e:
        print(f"  ✘ 跳过 {sheet}：{e}")

df_vle = pd.concat(dfs, ignore_index=True)
df_vle = df_vle.dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])
print(f"\nExcel 有效行数（dropna后）：{len(df_vle)}")

# 逐行重现 prepare 脚本的过滤逻辑，记录最终进入 data.npy 的行
records = []   # 每条: {npy_idx, excel_row_pos, cation, anion, refrigerant, T, P, ani_mw, x1, anion_smiles}
npy_idx = 0

for pos, (excel_row_pos, row) in enumerate(df_vle.iterrows()):
    c_smi = lookup_smiles(row['IL cation'])
    a_smi = lookup_smiles(row['IL anion'])
    r_smi = lookup_smiles(row['Refrigerant'])
    if None in (c_smi, a_smi, r_smi):
        continue  # 和 prepare 脚本一样跳过

    # 尝试解析 RDKit（有极少数 SMILES 可能解析失败）
    c_mol = Chem.MolFromSmiles(c_smi)
    a_mol = Chem.MolFromSmiles(a_smi)
    r_mol = Chem.MolFromSmiles(r_smi)
    if None in (c_mol, a_mol, r_mol):
        continue  # 同 prepare 脚本

    # 计算 ani_mw（用于后续校验）
    ani_mw = float(Descriptors.MolWt(a_mol))

    records.append({
        'npy_idx':       npy_idx,
        'excel_pos':     pos,          # 在 concat 后 DataFrame 里的位置（0-based）
        'cation':        str(row['IL cation']).strip(),
        'anion':         str(row['IL anion']).strip(),
        'refrigerant':   str(row['Refrigerant']).strip(),
        'T_K':           float(row['T (K)']),
        'P_MPa':         float(row['P (MPa)']),
        'ani_mw_calc':   ani_mw,
        'x1':            float(row['x1']),
        'anion_smiles':  a_smi,
        'cation_smiles': c_smi,
        'refri_smiles':  r_smi,
        'sheet':         row.get('_sheet', ''),
    })
    npy_idx += 1

print(f"重现过滤后，实际保留条数：{len(records)}")

# ──────────────────────────────────────────────
# 3. 加载真实 data.npy，和重建的 records 做对比
# ──────────────────────────────────────────────
data_npy = np.load('processed_tri_data/data.npy', allow_pickle=True)
label_npy = np.load('processed_tri_data/label.npy', allow_pickle=True)
print(f"\ndata.npy 实际条数：{len(data_npy)}")
print(f"label.npy 实际条数：{len(label_npy)}")

# ──────────────────────────────────────────────
# 4. 核心校验：抽查 7 个索引，对比 T / P / ani_mw / x1
# ──────────────────────────────────────────────
CHECK_IDXS = [0, 10, 100, 500, 1000, 2000, min(4000, len(records) - 1)]

report_lines = []
report_lines.append("=" * 60)
report_lines.append("数据对齐验证报告")
report_lines.append(f"data.npy 条数：{len(data_npy)}  |  重建 records 条数：{len(records)}")
report_lines.append("=" * 60)

all_pass = True

print("\n【对齐抽查】")
print(f"{'idx':>6}  {'字段':>10}  {'Excel重建':>15}  {'data.npy':>15}  {'通过?':>6}")
print("-" * 60)

for idx in CHECK_IDXS:
    if idx >= len(records) or idx >= len(data_npy):
        print(f"  idx={idx} 超出范围，跳过")
        continue

    rec = records[idx]
    npy_row = data_npy[idx]

    # data.npy 结构: [c_graph, a_graph, r_graph, T, P, ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa]
    T_npy      = float(npy_row[3])
    P_npy      = float(npy_row[4])
    ani_mw_npy = float(npy_row[7])
    x1_npy     = float(label_npy[idx])

    checks = {
        'T (K)':    (rec['T_K'],        T_npy,      0.01),
        'P (MPa)':  (rec['P_MPa'],      P_npy,      0.001),
        'ani_mw':   (rec['ani_mw_calc'],ani_mw_npy, 0.5),
        'x1':       (rec['x1'],         x1_npy,     1e-6),
    }

    row_pass = True
    for field, (v_excel, v_npy, tol) in checks.items():
        ok = abs(v_excel - v_npy) < tol
        status = "✅" if ok else "❌"
        if not ok:
            row_pass = False
            all_pass = False
        print(f"  {idx:>4}  {field:>10}  {v_excel:>15.5f}  {v_npy:>15.5f}  {status}")

    anion_name = rec['anion']
    report_lines.append(f"\n[idx={idx}] anion={anion_name} | {'全部通过✅' if row_pass else '存在不一致❌'}")
    for field, (v_excel, v_npy, tol) in checks.items():
        ok = abs(v_excel - v_npy) < tol
        report_lines.append(f"  {field}: Excel={v_excel:.5f}  npy={v_npy:.5f}  {'OK' if ok else 'FAIL'}")

print("-" * 60)
if all_pass:
    print("\n🎉 全部通过！data.npy 与 Excel 完全对齐，可以放心进行 Step 1 划分。")
    report_lines.append("\n\n结论：全部校验通过，data.npy 与 Excel 完全对齐。✅")
else:
    print("\n⚠️  存在不一致！请检查 prepare_tri_graph_data.py 是否有改动。")
    report_lines.append("\n\n结论：存在不一致，需要排查。❌")

# ──────────────────────────────────────────────
# 5. 保存产出文件
# ──────────────────────────────────────────────

# 5a. 验证报告
report_path = 'alignment_check_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"\n📄 验证报告已保存：{report_path}")

# 5b. 索引映射表（Step 1 划分的输入）
index_df = pd.DataFrame(records)
index_df = index_df[['npy_idx', 'excel_pos', 'cation', 'anion', 'refrigerant',
                      'T_K', 'P_MPa', 'x1', 'anion_smiles', 'cation_smiles',
                      'refri_smiles', 'sheet']]
index_path = 'index_with_anion.csv'
index_df.to_csv(index_path, index=False, encoding='utf-8-sig')
print(f"📄 索引映射表已保存：{index_path}（共 {len(index_df)} 行）")

# ──────────────────────────────────────────────
# 6. 简要统计（为 Step 0.5 提前预热）
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("【数据集简要统计】（Step 0.5 预热）")
print("=" * 60)
print(f"  总样本数       : {len(index_df)}")
print(f"  唯一阳离子数   : {index_df['cation'].nunique()}")
print(f"  唯一阴离子数   : {index_df['anion'].nunique()}")
print(f"  唯一制冷剂数   : {index_df['refrigerant'].nunique()}")
print(f"  T 范围 (K)     : {index_df['T_K'].min():.1f} ~ {index_df['T_K'].max():.1f}")
print(f"  P 范围 (MPa)   : {index_df['P_MPa'].min():.4f} ~ {index_df['P_MPa'].max():.4f}")
print(f"  x1 范围        : {index_df['x1'].min():.4f} ~ {index_df['x1'].max():.4f}")

print("\n【各阴离子样本数 Top 15】")
anion_counts = index_df['anion'].value_counts()
print(anion_counts.head(15).to_string())

print("\n✅ Step 0 完成！请查看 alignment_check_report.txt 和 index_with_anion.csv")
