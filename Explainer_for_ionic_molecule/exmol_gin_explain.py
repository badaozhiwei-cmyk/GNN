import os
import sys
import torch
import numpy as np
import argparse
from rdkit import Chem
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors

try:
    import exmol
except ImportError:
    print("请先安装 exmol: pip install exmol selfies")
    sys.exit(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))

from Model import GIN
from Dataset_explain_v2 import IL_set_v2

# ================= 特征提取 (与数据处理完全一致，防止OOD) =================
ELECTRONEG = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
COV_RADIUS = {1: 31, 5: 84, 6: 77, 7: 71, 8: 66, 9: 64, 15: 107, 16: 105, 17: 102, 35: 120, 53: 139}

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
    eneg_bucket = bucketize(ELECTRONEG.get(atomic_num, 2.55), 2.04, 3.98, 8)
    radius_bucket = bucketize(COV_RADIUS.get(atomic_num, 77), 31, 139, 8)
    return [atomic_num, hybrid, aro, degree, charge, eneg_bucket, radius_bucket]

def get_bond_features(bond):
    bond_type_dict = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2, 
                      Chem.rdchem.BondType.TRIPLE: 3, Chem.rdchem.BondType.AROMATIC: 4}
    return [
        bond_type_dict.get(bond.GetBondType(), 1),
        1 if bond.IsInRing() else 0,
        1 if bond.GetIsAromatic() else 0
    ]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.long)
    
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e_attr = get_bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e_attr, e_attr]
        
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ================= 构建三图拼接 =================
def build_tri_graph(c_smi, a_smi, r_smi):
    c_data, a_data, r_data = smiles_to_graph(c_smi), smiles_to_graph(a_smi), smiles_to_graph(r_smi)
    if not (c_data and a_data and r_data): return None
        
    nc, na, nr = c_data.x.shape[0], a_data.x.shape[0], r_data.x.shape[0]
    
    x = torch.cat([c_data.x, a_data.x, r_data.x], dim=0)
    edge_index = torch.cat([c_data.edge_index, a_data.edge_index + nc, r_data.edge_index + nc + na], dim=1)
    edge_attr = torch.cat([c_data.edge_attr, a_data.edge_attr, r_data.edge_attr], dim=0)
    
    mol_type = torch.cat([
        torch.zeros(nc, dtype=torch.long),
        torch.ones(na, dtype=torch.long),
        torch.full((nr,), 2, dtype=torch.long)
    ], dim=0)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.mol_type = mol_type
    return data

# ================= 黑盒预测管线 (Wrapper for exmol) =================
def create_model_wrapper(model, device, c_smi, a_smi, cond_tensor, scaler_ref_charge, scaler_ref_logp):
    """
    专门为 exmol 打造的预测管线：
    在固定的阳离子、阴离子和实验条件(T,P)下，只突变制冷剂的结构，看溶解度变化。
    """
    def wrapper(smiles_input, selfies=None):
        # 兼容 batched=False (传入字符串) 和 batched=True (传入列表)
        is_single = isinstance(smiles_input, str)
        smiles_list = [smiles_input] if is_single else smiles_input

        preds = []
        for smi in smiles_list:
            data = build_tri_graph(c_smi, a_smi, smi)
            mol = Chem.MolFromSmiles(smi)
            if data is None or mol is None:
                preds.append(0.0) # 非法分子惩罚
                continue
            
            # 动态重新计算突变分子的宏观物理特征
            try:
                new_charge = float(Descriptors.MaxAbsPartialCharge(mol))
            except:
                new_charge = 0.0
            new_logp = float(Descriptors.MolLogP(mol))
            
            # 使用原数据集的 scaler 标准化
            scaled_charge = scaler_ref_charge.transform([[new_charge]])[0][0]
            scaled_logp = scaler_ref_logp.transform([[new_logp]])[0][0]
            
            # 复制一份 condition，避免修改原对象，并替换对应的特征位置
            new_cond = cond_tensor.clone()
            new_cond[2] = scaled_charge
            new_cond[3] = scaled_logp

            batch = Batch.from_data_list([data]).to(device)
            c = new_cond.unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(batch, c).flatten().item()
            preds.append(out)
            
        return preds[0] if is_single else np.array(preds)
    return wrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--sample_idx', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 [加载 GIN 模型] {args.model_path}")
    
    gin_args = {'num_gin_layer': 3, 'emb_dim': 300, 'feat_dim': 512, 'drop_ratio': 0.2, 'pool': 'mean'}
    model = GIN(gin_args).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    # 加载标准化条件
    data_path = os.path.join(ROOT, 'processed_tri_data')
    dataset = IL_set_v2(data_npy_path=os.path.join(data_path, 'data.npy'), 
                        label_npy_path=os.path.join(data_path, 'label.npy'))
    _, cond_tensor, orig_label, _ = dataset[args.sample_idx]
    
    # 模拟环境：以经典的 BMIM Tf2N 离子液体为例
    c_smi = "CCCC[n+]1cccc(C)c1" # BMIM
    a_smi = "FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F" # TF2N
    
    # 我们的目标干预对象：制冷剂 R32 (可以从数据集中抓取，这里为了演示使用固定的)
    base_ref_smi = "C(F)F" 
    
    wrapper = create_model_wrapper(model, device, c_smi, a_smi, cond_tensor, dataset.scaler_ref_charge, dataset.scaler_ref_logp)
    orig_pred = wrapper([base_ref_smi])[0]
    print(f"🧪 原始分子 [{base_ref_smi}] 在该条件下的预测溶解度: {orig_pred:.4f}")
    
    print("\n🧬 开始执行基于 SELFIES 语法的 exmol 离散空间反事实突变...")
    print("   (在底层自动切碎重组分子，且保证 100% 拓扑和价态合法)")
    
    # exmol 的精华：在离散空间里采集成百上千个合法新分子，寻找能让溶解度飙升的结构！
    samples = exmol.sample_space(base_ref_smi, wrapper, batched=False, num_samples=1000)
    
    print("🎯 正在寻找靶向偏移的反事实解释...")
    # 我们设定一个目标：寻找溶解度偏离原分子最远的最佳反事实解释
    cfs = exmol.cf_explain(samples) 
    
    out_dir = os.path.join(ROOT, 'Explainer_for_ionic_molecule', 'exmol_result')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'exmol_cf_result_idx_{args.sample_idx}.png')
    
    # 一键出图，画出怎么改基团才能提升溶解度
    exmol.plot_cf(cfs)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\n✅ exmol 逆向靶向生成分析出炉！\n图表已保存至: {save_path}")

if __name__ == '__main__':
    main()
