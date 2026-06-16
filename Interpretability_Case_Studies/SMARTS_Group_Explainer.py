# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Batch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))
sys.path.insert(0, ROOT)

from Model import IL_GAT
from Interpretability_Case_Studies.Dataset_explain_v2 import smiles_to_graph, combine_Graph, add_global

class SMARTS_Group_Explainer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[引擎初始化] SMARTS 基团分析引擎就绪 on {self.device}")
        
        Args = {'emb_dim': 300, 'dropout_rate': 0.2, 'n_features': 7}
        self.model = IL_GAT(Args).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        self.model.eval()

        from Interpretability_Case_Studies.Dataset_explain_v2 import IL_set_v2
        data_path = os.path.join(ROOT, 'processed_tri_data')
        self.dataset = IL_set_v2(data_npy_path=os.path.join(data_path, 'data.npy'), 
                                 label_npy_path=os.path.join(data_path, 'label.npy'))

    def compute_condition(self, c_smi, a_smi, r_smi, T, P):
        try: m_c = Chem.MolFromSmiles(c_smi); cat_charge = float(Descriptors.MaxAbsPartialCharge(m_c)); cat_tpsa = float(Descriptors.TPSA(m_c))
        except: cat_charge, cat_tpsa = 0.0, 0.0
            
        try: m_a = Chem.MolFromSmiles(a_smi); ani_mw = float(Descriptors.MolWt(m_a))
        except: ani_mw = 0.0
            
        try: m_r = Chem.MolFromSmiles(r_smi); ref_charge = float(Descriptors.MaxAbsPartialCharge(m_r)); ref_logp = float(Descriptors.MolLogP(m_r))
        except: ref_charge, ref_logp = 0.0, 0.0

        T_s = float(self.dataset.scaler_T.transform([[T]])[0][0])
        P_s = float(self.dataset.scaler_P.transform([[P]])[0][0])
        rc_s = float(self.dataset.scaler_ref_charge.transform([[ref_charge]])[0][0])
        rl_s = float(self.dataset.scaler_ref_logp.transform([[ref_logp]])[0][0])
        am_s = float(self.dataset.scaler_ani_mw.transform([[ani_mw]])[0][0])
        cc_s = float(self.dataset.scaler_cat_charge.transform([[cat_charge]])[0][0])
        ct_s = float(self.dataset.scaler_cat_tpsa.transform([[cat_tpsa]])[0][0])

        return torch.tensor([[T_s, P_s, rc_s, rl_s, am_s, cc_s, ct_s]], dtype=torch.float).to(self.device)

    def _build_strict_graph(self, c_smi, a_smi, r_smi):
        c_data = smiles_to_graph(c_smi)
        a_data = smiles_to_graph(a_smi)
        r_data = smiles_to_graph(r_smi)
        if not (c_data and a_data and r_data): return None
        
        c_data.batch = torch.zeros(c_data.x.shape[0], dtype=torch.long)
        a_data.batch = torch.ones(a_data.x.shape[0], dtype=torch.long)
        r_data.batch = torch.full((r_data.x.shape[0],), 2, dtype=torch.long)
        
        combined = combine_Graph([c_data, a_data, r_data])
        G = add_global(combined)
        return G

    def find_group_indices(self, smi, smarts):
        """利用 RDKit 获取 SMARTS 匹配的原子索引（基团级的原子联合体）"""
        mol = Chem.MolFromSmiles(smi)
        patt = Chem.MolFromSmarts(smarts)
        if not mol or not patt: return []
        matches = mol.GetSubstructMatches(patt)
        # 展平所有匹配的原子索引
        indices = []
        for match in matches:
            indices.extend(match)
        return list(set(indices))

    def evaluate_group_contribution(self, c_smi, a_smi, r_smi, T, P, target='anion', smarts='C(F)(F)F'):
        """
        进行基团级的联合反事实微扰
        target: 'cation', 'anion', or 'ref' (制冷剂)
        smarts: 要突变的目标基团，比如 -CF3 是 'C(F)(F)F'
        """
        G = self._build_strict_graph(c_smi, a_smi, r_smi)
        cond = self.compute_condition(c_smi, a_smi, r_smi, T, P)
        
        G_batch = Batch.from_data_list([G]).to(self.device)
        
        # 1. 计算原态溶解度
        with torch.no_grad():
            y_base = self.model(G_batch, cond).item()
            
        # 2. 寻找匹配基团在整张图中的绝对索引
        mol_smi = {'cation': c_smi, 'anion': a_smi, 'ref': r_smi}[target]
        
        local_indices = self.find_group_indices(mol_smi, smarts)
        if not local_indices:
            return y_base, y_base, 0.0 # 没有匹配到该基团
            
        # 转换为全局图节点的索引
        offset = 0
        if target == 'anion':
            offset = Chem.MolFromSmiles(c_smi).GetNumAtoms()
        elif target == 'ref':
            offset = Chem.MolFromSmiles(c_smi).GetNumAtoms() + Chem.MolFromSmiles(a_smi).GetNumAtoms()
            
        global_indices = [idx + offset for idx in local_indices]
        
        # 3. 实施联合反事实微扰 (Co-mutation)
        G_mask = G.clone()
        for idx in global_indices:
            # 策略：将其特征替换为非极性状态 (消除电负性和半径影响)
            # 原子类型变为C(6)或H(1)，这里选择保守的特征消融
            G_mask.x[idx][0] = 1 # 原子类型 H
            G_mask.x[idx][5] = 0 # 电负性清零
            G_mask.x[idx][6] = 0 # 共价半径清零
            
        G_mask_batch = Batch.from_data_list([G_mask]).to(self.device)
        
        # 4. 计算微扰后的溶解度
        with torch.no_grad():
            y_mask = self.model(G_mask_batch, cond).item()
            
        # 5. 计算基团绝对贡献值
        delta_y = y_base - y_mask
        
        return y_base, y_mask, delta_y
