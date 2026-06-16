import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from Group_Explainer_Engine import Group_Explainer
from smarts_dict import get_group_matches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------
# 1. SMILES Dictionary
# ------------------------------------------
SMILES_DICT = {}
def build_smiles_dict():
    csv_path = os.path.join(ROOT, 'Original_Data', 'smiles.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            abbr = str(row['Abbreviation']).strip().upper()
            smi = str(row['Smiles']).strip()
            SMILES_DICT[abbr] = smi
            SMILES_DICT[abbr.replace('[', '').replace(']', '')] = smi

    extra = {
        'R32': 'C(F)F', 'R134A': 'C(C(F)(F)F)F', 'R143A': 'CC(F)(F)F',
        'R125': 'C(F)(F)(C(F)(F)F)', 'R114': 'C(C(F)(F)Cl)(F)(F)Cl',
        'R1234YF': 'C(=C(F)F)(C(F)(F)F)F', 'R1234ZE(E)': 'F/C=C/C(F)(F)F',
        'R152A': 'CC(F)F', 'R23': 'C(F)(F)F', 'R41': 'CF',
        'R22': 'ClC(F)F', 'R22B1': 'BrC(F)F', 'R14': 'FC(F)(F)F',
        'R116': 'FC(F)(F)C(F)(F)F', 'R124': 'FC(F)(F)C(Cl)F', 'R124A': 'ClC(F)C(F)(F)F',
        'R114A': 'ClC(Cl)(F)C(F)(F)F', 'R134': 'FC(F)C(F)F', 'R161': 'CCF',
        'R218': 'FC(F)(F)C(F)(F)C(F)(F)F', 'R227EA': 'FC(F)(F)C(F)C(F)(F)F',
        'R236FA': 'FC(F)(F)CC(F)(F)F', 'R1336MZZ(E)': 'F/C(C(F)(F)F)=C(\F)C(F)(F)F',
        'R1336MZZ(Z)': 'F/C(C(F)(F)F)=C(/F)C(F)(F)F', 'R1233ZD(E)': 'F/C=C(\Cl)C(F)(F)F',
        'R245FA': 'FCC(F)C(F)(F)F',
        'AC': 'CC(=O)[O-]', 'TF2N': 'FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F'
    }
    SMILES_DICT.update(extra)

def lookup(name):
    name = str(name).strip().upper()
    if name in SMILES_DICT: return SMILES_DICT[name]
    nb = name.replace('[', '').replace(']', '')
    if nb in SMILES_DICT: return SMILES_DICT[nb]
    return None

def main():
    build_smiles_dict()
    data_path = os.path.join(ROOT, 'Original_Data', 'original_data.xlsx')
    if not os.path.exists(data_path):
        print("未找到 original_data.xlsx")
        return

    df = pd.read_excel(data_path, sheet_name='Sheet1')
    
    # ------------------------------------------
    # 2. 定义子宇宙 (Sub-Universe) 规则
    # ------------------------------------------
    print("过滤子宇宙: [咪唑类阳离子] + [HFOs类制冷剂]")
    mask_cation = df['Cation'].str.contains('mim', case=False, na=False)
    mask_ref = df['Refrigerant'].str.contains('1234|1233|1336', case=False, na=False)
    sub_df = df[mask_cation & mask_ref].reset_index(drop=True)
    print(f"匹配到 {len(sub_df)} 条数据，准备开始全量 IG 积分...")
    
    if len(sub_df) == 0: return

    explainer = Group_Explainer()
    
    # 全局累加字典
    global_group_scores = defaultdict(float)
    valid_count = 0

    # ------------------------------------------
    # 3. 运行批量解释
    # ------------------------------------------
    for idx, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc="积分进度"):
        c_name = row['Cation']
        a_name = row['Anion']
        r_name = row['Refrigerant']
        T, P = row['T/K'], row['P/bar']
        
        c_smi = lookup(c_name)
        a_smi = lookup(a_name)
        r_smi = lookup(r_name)
        
        if not (c_smi and a_smi and r_smi):
            continue
            
        # 调用基础 Explainer (不画图, 仅获取分数)
        node_scores, atom_types, mol_type = explainer.get_attention_scores(c_smi, a_smi, r_smi, T, P)
        if node_scores is None: continue

        if node_scores.max() > 0:
            node_scores = node_scores / node_scores.max()

        from rdkit import Chem
        c_mol = Chem.MolFromSmiles(c_smi)
        a_mol = Chem.MolFromSmiles(a_smi)
        r_mol = Chem.MolFromSmiles(r_smi)
        
        c_num = c_mol.GetNumAtoms() if c_mol else 0
        a_num = a_mol.GetNumAtoms() if a_mol else 0
        r_num = r_mol.GetNumAtoms() if r_mol else 0

        c_groups = get_group_matches(c_smi)
        a_groups = get_group_matches(a_smi)
        r_groups = get_group_matches(r_smi)

        # 阳离子
        for g_name, atoms in c_groups.items():
            score = sum([node_scores[i] for i in atoms])
            global_group_scores[f"{g_name} (Cat)"] += score
            
        # 阴离子
        a_offset = c_num
        for g_name, atoms in a_groups.items():
            score = sum([node_scores[i + a_offset] for i in atoms])
            global_group_scores[f"{g_name} (Ani)"] += score
            
        # 制冷剂
        r_offset = c_num + a_num
        for g_name, atoms in r_groups.items():
            score = sum([node_scores[i + r_offset] for i in atoms])
            global_group_scores[f"{g_name} (Ref)"] += score
            
        valid_count += 1
        
    # ------------------------------------------
    # 4. 可视化分析结果 (Pie Chart)
    # ------------------------------------------
    print(f"\n==========================================")
    print(f" 分析完成！成功积分了 {valid_count} 个样本。")
    print(f"==========================================")
    
    if valid_count == 0: return
    
    # 过滤掉极小的得分项 (小于总分0.5%)
    total_score = sum(global_group_scores.values())
    filtered_groups = {k: v for k, v in global_group_scores.items() if (v / total_score) > 0.005}
    
    # 将剩下的微小项归入 "Others"
    other_score = total_score - sum(filtered_groups.values())
    if other_score > 0:
        filtered_groups["Other Trace Groups"] = other_score

    # 排序用于画图
    sorted_items = sorted(filtered_groups.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_items]
    sizes = [x[1] for x in sorted_items]

    # 配色逻辑：Cat(蓝系), Ani(红系), Ref(绿系)
    colors = []
    import matplotlib.cm as cm
    for l in labels:
        if "(Cat)" in l: colors.append('#3498db')
        elif "(Ani)" in l: colors.append('#e74c3c')
        elif "(Ref)" in l: colors.append('#2ecc71')
        else: colors.append('#95a5a6')

    plt.figure(figsize=(10, 8))
    explode = [0.05] * len(labels) # 分离每一块
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=False, startangle=140,
            textprops={'fontsize': 10, 'weight': 'bold'})
            
    plt.title(f"Micro-driving Forces for Imidazolium + HFOs System\n(Averaged over {valid_count} samples)", 
              fontsize=14, pad=20, weight='bold')
    
    plt.axis('equal')
    
    out_dir = os.path.join(ROOT, 'Interpretability_Case_Studies', 'Results')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'Global_Group_Pie_Chart.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 子宇宙全景分析图 (饼图) 已保存至: {save_path}")

if __name__ == "__main__":
    main()
