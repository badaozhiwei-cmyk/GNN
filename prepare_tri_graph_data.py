import pandas as pd
import numpy as np
from rdkit import Chem
import os

print("开始执行 Tri-Graph 模型的数据预处理 (V2 - 已修复 CUDA 越界错误)...")

# ==========================================
# 模块 1：建立“分子花名册” (字典)
# 作用：Excel 里面的名字是缩写（比如 R32），计算机看不懂。
# 我们需要把这些“缩写名字”翻译成用来表示化学结构的“SMILES 字符串”（比如 C(F)F）。
# ==========================================
print("正在从 smiles.csv 加载已有的离子液体 SMILES 字典...")

# 寻找原始字典的路径（它包含了论文原本研究的那些阴阳离子）
smiles_csv_path = 'Original_Data/smiles.csv' if os.path.exists('Original_Data/smiles.csv') else 'smiles.csv'
il_df = pd.read_csv(smiles_csv_path)

# 去除列名两边的空格以防读错
il_df.columns = [c.strip() for c in il_df.columns]

# 定义一个空的花名册字典
smiles_dict = {}

# 把 smiles.csv 里面的内容一条条填进字典
for idx, row in il_df.iterrows():
    abbr = str(row['Abbreviation']).strip().upper()  # 获取缩写名并大写，如 [BMIM]
    abbr_no_bracket = abbr.replace('[', '').replace(']', '') # 把中括号去掉，如 BMIM
    smi = str(row['Smiles']).strip() # 获取对应的 SMILES 结构式
    smiles_dict[abbr] = smi          # [BMIM] -> 对应的SMILES
    smiles_dict[abbr_no_bracket] = smi # BMIM -> 对应的SMILES，这样两边无论写哪个都能查到

# 手动打补丁：因为原论文做的是二氧化碳，没涉及到制冷剂
# 我们在这里手动补齐本项目中用到的各种常见制冷剂的结构式
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
    
    # --- 本次全盘排查补齐的 9 种罕见阳离子 ---
    'P4442':      'CCCC[P+](CCCC)(CCCC)CC',                  # tributyl(ethyl)phosphonium
    'P66614':     'CCCCCC[P+](CCCCCC)(CCCCCC)CCCCCCCCCCCCCC',# trihexyl(tetradecyl)phosphonium
    'DOIM':       'CCCCCCCCn1cc[n+](CCCCCCCC)c1',            # 1,3-dioctylimidazolium
    'P44414':     'CCCC[P+](CCCC)(CCCC)CCCCCCCCCCCCCC',      # tributyl(tetradecyl)phosphonium
    'EMPY':       'CC[n+]1cccc(C)c1',                        # 1-ethyl-3-methylpyridinium
    'BMPY':       'CCCC[n+]1cccc(C)c1',                      # 1-butyl-3-methylpyridinium
    'DMPIM':      'CCCn1cc[n+](C)c1C',                       # 1,2-dimethyl-3-propylimidazolium
    'P4441':      'CCCC[P+](CCCC)(CCCC)C',                   # tributyl(methyl)phosphonium
    'C8H4F13C1IM':'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CCn1cc[n+](C)c1', # 氟代长链咪唑
    
    # --- 本次全盘排查补齐的 16 种特殊阴离子 ---
    'ET2PO4':     'CCOP(=O)([O-])OCC',                               # diethylphosphate
    'BEI':        'FC(F)(F)C(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)C(F)(F)F', # bis(pentafluoroethylsulfonyl)imide
    'TTES':       'FC(F)(F)OC(F)C(F)(F)S(=O)(=O)[O-]',               # 1,1,2-trifluoro-2-(trifluoromethoxy)ethanesulfonate
    'HFPS':       'FC(F)(F)C(F)C(F)(F)S(=O)(=O)[O-]',                # 1,1,2,3,3,3-hexafluoropropanesulfonate
    'PFBS':       'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)[O-]',      # perfluorobutanesulfonate
    'TMPP':       'CC(C)(C)CC(C)CCP(=O)([O-])CC(C)CC(C)(C)C',        # bis(2,4,4-trimethylpentyl)phosphinate
    'FS':         'FC(F)(F)C(F)OC(F)(F)C(F)(F)S(=O)(=O)[O-]',        # 2-(1,2,2,2-tetrafluoroethoxy)...ethanesulfonate
    'FEP':        'F[P-](F)(F)(C(F)(F)C(F)(F)F)(C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F', # tris(pentafluoroethyl)trifluorophosphate
    'PR':         'CCC(=O)[O-]',                                     # propionate
    'OTF':        'FC(F)(F)S(=O)(=O)[O-]',                           # trifluoromethanesulfonate
    'TPES':       'FC(F)(F)C(F)(F)OC(F)C(F)(F)S(=O)(=O)[O-]',        # 1,1,2-trifluoro-2-(perfluoroethoxy)ethanesulfonate
    'I':          '[I-]',                                            # iodide
    'TFES':       'FC(F)C(F)(F)S(=O)(=O)[O-]',                       # 1,1,2,2-tetrafluoroethanesulfonate
    'PFP':        'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)[O-]',          # perfluoropentanoate
    'PE':         'CCCCC(=O)[O-]',                                   # pentanoate
    'TMEM':       'FC(F)(F)S(=O)(=O)[C-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F', # tris(trifluoromethylsulfonyl)methide
}

# 把这些补丁也全部塞进上面的基础大字典里
for k, v in extra_smiles.items():
    smiles_dict[k.upper()] = v
    smiles_dict[k.upper().replace('[','').replace(']','')] = v

# ==========================================
# 模块 2：RDKit 特征提取（把化学结构变成神经网络认识的数字）
# 作用：我们对分子里的每一个“原子”和每一根“化学键”进行体检，给他们打分。
# ==========================================
def get_atom_features(atom):
    """
    【专业增强版】：在不增加模型维度的情况下，融入核心化学特征
    """
    # 1. 杂化方式
    hybrid = int(atom.GetHybridization())
    if hybrid >= 8: hybrid = 7
    
    # 2. 芳香性
    aro = 1 if atom.GetIsAromatic() else 0
    
    # 3. 原子度 Degree
    degree = atom.GetDegree()
    if degree >= 7: degree = 6
    
    # 4. 电荷
    charge = atom.GetFormalCharge() + 1 
    if charge > 2: charge = 2
    if charge < 0: charge = 0
    
    # 5. 【新增】氢键供体 (HBD)
    # N, O, S 且带有氢原子
    is_hbd = 1 if atom.GetSymbol() in ['N', 'O', 'S'] and atom.GetTotalNumHs() > 0 else 0
    
    # 6. 【新增】氢键受体 (HBA)
    # N, O, F (具有孤对电子)
    is_hba = 1 if atom.GetSymbol() in ['N', 'O', 'F'] else 0
    
    # 7. 【新增】电负性等级 (Electronegativity Rank)
    # 常见元素电负性映射 (Pauling scale -> Rank 1-5)
    en_map = {
        'F': 5, 'O': 5,
        'N': 4, 'CL': 4,
        'C': 3, 'S': 3, 'BR': 3, 'I': 3,
        'P': 2, 'H': 2, 'B': 2,
        'LI': 1, 'NA': 1, 'K': 1, 'MG': 1, 'CA': 1
    }
    en_rank = en_map.get(atom.GetSymbol().upper(), 3) # 默认 3 (中等)

    # 恢复到最原始的 5 维特征
    return [atom.GetAtomicNum(), hybrid, aro, degree, charge]

def get_bond_features(bond):
    """
    化学键的体检表。
    """
    # 化学键类型打分表：单键得1分，双键得2分，三键得3分，特殊的芳香键得4分
    bond_type_dict = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2, 
                      Chem.rdchem.BondType.TRIPLE: 3, Chem.rdchem.BondType.AROMATIC: 4}
    return [
        bond_type_dict.get(bond.GetBondType(), 1), # 获取键的类型分
        1 if bond.IsInRing() else 0,              # 这根键是不是闭合环形状上的（1为是）
        1 if bond.GetIsAromatic() else 0          # 这根键是否属于芳香系统
    ]

# ==========================================
# 模块 3：组装分子图
# 作用：拿着得到的特征体检表，把一个分子内部的点和连线全部建出来，就像拼乐高积木一样。
# ==========================================
def mol2graph_components(smiles_string):
    # 召唤 RDKit 库大法，把一串 SMILES 文字具象化为虚拟分子的对象
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return None # 如果这串代码 RDKit 无法解析报错了，这组构件组装失败，抛空。
    
    # 第一步：获取所有的“点”（即拿着这个分子的每一个原子，去填写一次体检表）
    node_f = [get_atom_features(atom) for atom in mol.GetAtoms()]
    
    # 第二步：建立“谁连着谁”的图谱连线手册
    edge_index = [[], []] # 像个二维数组，左边放起始点，右边放终点
    edge_attr = []        # 同步记录这根线的特征连带属性
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx(); j = bond.GetEndAtomIdx() # 找到相连的原子的索引编号
        f = get_bond_features(bond) # 提取连通这两者的“化学键”的特征分数
        
        # 图算法里，如果要互相连通，就意味着我们要建一正一反向的两条边
        # 意思就是 i 走向 j ；同样 j 也能走向 i
        edge_index[0].extend([i, j])
        edge_index[1].extend([j, i])
        
        # 因为建立了两条来回的路，这跟键本身的特性咱们也要复制一次写上去
        edge_attr.extend([f, f])
    
    # ！！！极其关键的防报错补丁！！！
    # 如果处理的是像 [Cl-] 单原子离子，它就是孤身一个原子毫无化学键（没路），那 edge 列表就会是空的！
    # 到了最后喂给模型的时候，会让模型不知道它的尺度而引发崩盘。强制补入个安全边界。
    if len(edge_attr) == 0:
        edge_index = [[], []]
        edge_attr = []
        
    # 最后把组装好的三大件：[点的特征列表、边连线手册地图、边的特征列表] 原包返回出去
    return [node_f, edge_index, edge_attr]

# 查阅字典小工具：为了包容我们 Excel 里各种不讲规矩的手输名字
def lookup_smiles(name):
    name = str(name).strip().upper()
    name_no_bracket = name.replace('[', '').replace(']', '')
    if name in smiles_dict: return smiles_dict[name]
    if name_no_bracket in smiles_dict: return smiles_dict[name_no_bracket]
    name_no_hyphen = name_no_bracket.replace('-', '')
    if name_no_hyphen in smiles_dict: return smiles_dict[name_no_hyphen]
    return None

# ==========================================
# 模块 4：批处理 Excel 数据源并最终封盘 
# 作用：逐行读取你的研究 Excel 图表，配对查询字典，然后配上它的环境大背景，一起入袋打包。
# ==========================================
excel_name = 'ZLJ_DATA.xlsx'
if not os.path.exists(excel_name):
    if os.path.exists('../'+excel_name):
        excel_name = '../'+excel_name

print(f"正在从 {excel_name} 读取制冷剂气液相平衡数据 (VLE)...")
dfs = []
# 开始遍历你想读的子图表 (Sheets)
for sheet in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']:
    try:
        # skiprows=2 表示丢掉表格最上面用于排版的无效两行再看
        tmp_df = pd.read_excel(excel_name, sheet_name=sheet, skiprows=2)
        dfs.append(tmp_df)
    except Exception as e:
        print(f"跳过页签 {sheet}: {e}")

if not dfs:
    print("严重错误: 空空如也。请检查你放对 ZLJ_DATA.xlsx 没？")
    exit()

# 拿着剪刀把他们这三页数据首尾相接拼接成一个超级大总表
df_vle = pd.concat(dfs, ignore_index=True)
# 删掉含有空白选项的不良数据行，现已强制要求必须有 T 和 P
df_vle = df_vle.dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])

final_data = []    # 这是以后交给模型上场的数据筐
final_labels = []  # 对应的就是模型训练所期盼追求的那个准确结果标签框

# 开始流水线工作，处理这个超级大总表里的每一行的数据
for idx, row in df_vle.iterrows():
    
    # 第 1 关：拿着这一行的名字，问字典要它的 SMILES 代码
    c_smi = lookup_smiles(row['IL cation'])   # 查阳离子代码
    a_smi = lookup_smiles(row['IL anion'])    # 查阴离子代码
    r_smi = lookup_smiles(row['Refrigerant']) # 查制冷剂代码
    
    # 哪怕只要因为这三个人中有一个人不守规矩填错了，名字没在花名册上，那这行数据对不起只能狠心扔进垃圾桶！！
    # （这也是为什么你 4000 数据只用得了两三千的最主要原因泄露之处，有很多冷门阴阳离子在你的字典压根没列出来）
    if None in (c_smi, a_smi, r_smi): continue
    
    # 第 2 关：把查出代码的他们，放进加工机生产出节点、连线的实体属性矩阵图
    c_graph = mol2graph_components(c_smi)
    a_graph = mol2graph_components(a_smi)
    r_graph = mol2graph_components(r_smi)
    
    # 同理，万一有 SMILES 代码被写错了使得 RDKit 加载生成不出实物，这一行也得被牺牲当做垃圾扔掉
    if None in (c_graph, a_graph, r_graph): continue
    
    # 第 3 关：组装箱子并写入“周围环境变量”。
    # 强制读取，不再接受任何默认值
    final_data.append([c_graph, a_graph, r_graph, float(row['T (K)']), float(row['P (MPa)'])])
    
    # 同步把测试真正的答案也塞进对应的标签框，这就是将来它作为指导监督的任务目标 (x1=溶解度)
    final_labels.append(float(row['x1']))

# 一切成功完毕，创建最终归属地文件夹以便交工。
os.makedirs('processed_tri_data', exist_ok=True)
# numpy 直接出马，压缩他们入一个冷冰冰高效的二进制封装盒子里——我们常常读到的 data.npy
np.save('processed_tri_data/data.npy', np.array(final_data, dtype=object))
np.save('processed_tri_data/label.npy', np.array(final_labels, dtype=object))

# 给用户的最终宣告
print(f"帅呆了！一共从汪洋大海里提纯出了 {len(final_data)} 个没有任何瑕疵的数据结构存了下来。")
