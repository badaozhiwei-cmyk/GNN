import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# 引入基座
from Explainer_Engine import Explainer_Engine

# ==========================================
# 1. 字典复刻（保证完全独立运行）
# ==========================================
SMILES_DICT = {}
def build_smiles_dict():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(ROOT, 'Original_Data', 'smiles.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            abbr = str(row['Abbreviation']).strip().upper()
            smi = str(row['Smiles']).strip()
            SMILES_DICT[abbr] = smi
            SMILES_DICT[abbr.replace('[', '').replace(']', '')] = smi

    # 补充制冷剂与特殊离子
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

# ==========================================
# 2. 元素常量
# ==========================================
ELEMENT_MAP = {1: 'H', 6: 'C', 8: 'O', 9: 'F', 17: 'Cl', 35: 'Br'}

def main():
    print("=" * 70)
    print(" 🌌 启动子宇宙全局元素积分分析 (Sub-Universe Attention Integrator)")
    print("=" * 70)
    
    # 步骤一：初始化工具
    print("\n[进度 1/5] 正在构建 SMILES 化学翻译字典...")
    build_smiles_dict()
    
    print("[进度 2/5] 正在加载 GAT 全局权重...")
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(ROOT, 'GNN_for_property_prediction', 'checkpoints_v2', 'best_gat_seed_1.pth')
    explainer = Explainer_Engine(model_path)
    
    # 步骤二：读取数据
    print("\n[进度 3/5] 正在读取全宇?4444 条原始数?(ZLJ_DATA.xlsx)...")
    excel_path = os.path.join(ROOT, 'ZLJ_DATA.xlsx')
    dfs = []
    for sheet in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']:
        try:
            dfs.append(pd.read_excel(excel_path, sheet_name=sheet, skiprows=2))
        except: pass
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])
    total_len = len(df)
    print(f"   => 成功读取全宇宙记录数: {total_len} ")
    
    # 步骤三：切割子宇?(咪唑 + 氟系)
    print("\n[进度 4/5] 正在施加环境场约束：切割『咪唑类 + 高氟阴离子』子宇宙...")
    # 阳离子带 mim
    mask_cation = df['IL cation'].str.contains('mim', case=False, na=False)
    # 阴离子带?(枚举最典型的高氟阴离子)
    F_ANIONS = ['[Tf2N]', '[PF6]', '[BF4]', '[OTf]', '[PFBS]', '[PFP]', '[FEP]', '[HFPS]']
    mask_anion = df['IL anion'].isin(F_ANIONS)
    
    sub_df = df[mask_cation & mask_anion]
    sub_len = len(sub_df)
    print(f"   => 🎯 精准锁定子宇宙！该特定大体系下共?{sub_len} 条数?(占比 {sub_len/total_len*100:.1f}%)")
    
    if sub_len == 0:
        print("无数据！请检查过滤条件")
        return

    # 步骤四：开始暴力积?    print("\n[进度 5/5] 🚀 开始批?GAT 推断，收集制冷剂吸收的『靶向注意力碎片?..")
    
    # 记录制冷剂内部，各个元素的总得?    element_score_bank = defaultdict(float)
    valid_count = 0
    
    for idx, row in tqdm(sub_df.iterrows(), total=sub_len, desc="GAT 积分"):
        cat = row['IL cation']
        ani = row['IL anion']
        ref = row['Refrigerant']
        T = row['T (K)']
        P = row['P (MPa)'] * 10 # 转为 bar
        
        c_smi = lookup(cat)
        a_smi = lookup(ani)
        r_smi = lookup(ref)
        
        if not (c_smi and a_smi and r_smi): continue
            
        # 调用 GAT 获取分数
        scores, atom_types, mol_types = explainer.get_attention_scores(c_smi, a_smi, r_smi, T, P)
        if scores is None: continue
            
        # 只提取制冷剂 (mol_type == 2) 的原子
        for i in range(len(scores)):
            if mol_types[i] == 2:
                element = ELEMENT_MAP.get(int(atom_types[i]), 'Other')
                element_score_bank[element] += scores[i]
                
        valid_count += 1

    print(f"\n==================================================")
    print(f"🎉 积分完成?成功解析 {valid_count} 条有效数?")
    print("==================================================")
    
    total_score = sum(element_score_bank.values())
    print("\n🏆 『咪?高氟体系』制冷剂各元素吸引力绝对占比排行榜：")
    for elem, sc in sorted(element_score_bank.items(), key=lambda x: x[1], reverse=True):
        ratio = sc / total_score * 100
        print(f"  ?{elem} 元素: {ratio:5.2f}% (总积? {sc:.2f})")

    # 步骤五：画饼图
    labels = []
    sizes = []
    for elem, sc in sorted(element_score_bank.items(), key=lambda x: x[1], reverse=True):
        labels.append(f"{elem} ({sc/total_score*100:.1f}%)")
        sizes.append(sc)
        
    plt.figure(figsize=(8, 8))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, pctdistance=0.85, shadow=True, textprops={'fontsize': 12, 'weight': 'bold'})
    
    # 画个白圈，做成甜甜圈?    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title("GAT Native Attention Target Proportion\n(Sub-Universe: Imidazolium + High-F Anions)", fontsize=16, weight='bold')
    plt.tight_layout()
    
    out_dir = os.path.join(ROOT, 'Interpretability_Case_Studies', 'Results')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'Sub_Universe_Attention_Integration.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\n?全局注意力占比饼图已生成，保存在：{save_path}")

if __name__ == "__main__":
    main()
