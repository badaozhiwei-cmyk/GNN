import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from Explainer_Engine import Explainer_Engine as GAT_Explainer
from smarts_dict import get_group_matches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Group_Explainer(GAT_Explainer):
    """
    基于 IG 的基团级别可解释性引擎。
    继承自底层的原子级别 GAT_Explainer，并利用 RDKit 的 SMARTS 匹配将原子得分聚合为基团得分。
    """
    def __init__(self, model_path=None):
        if model_path is None:
            # 默认去加载最新训练好的模型
            model_path = os.path.join(ROOT, 'GNN_for_property_prediction', 'pretrained_model', 'GAT_300', 'best_model_para.pth')
            # 备用路径：如果运行了新的 runner
            alt_path = os.path.join(ROOT, 'GNN_for_property_prediction', 'checkpoints_v2', 'best_gat_seed_1.pth')
            if os.path.exists(alt_path):
                model_path = alt_path
        super().__init__(model_path)

    def aggregate_to_groups(self, node_scores, c_smi, a_smi, r_smi):
        """
        公共方法：将原子级 IG 得分聚合为基团级得分。
        
        Args:
            node_scores: numpy array，已归一化的原子级 IG 得分
            c_smi: 阳离子 SMILES
            a_smi: 阴离子 SMILES
            r_smi: 制冷剂 SMILES
            
        Returns:
            group_scores: defaultdict(float)，键为 "GroupName (Cat/Ani/Ref)"，值为聚合得分
        """
        from rdkit import Chem
        
        c_mol = Chem.MolFromSmiles(c_smi)
        a_mol = Chem.MolFromSmiles(a_smi)
        r_mol = Chem.MolFromSmiles(r_smi)
        
        c_num = c_mol.GetNumAtoms() if c_mol else 0
        a_num = a_mol.GetNumAtoms() if a_mol else 0

        # 获取各分子的基团匹配
        c_groups = get_group_matches(c_smi)
        a_groups = get_group_matches(a_smi)
        r_groups = get_group_matches(r_smi)

        group_scores = defaultdict(float)

        # 阳离子（偏移量 = 0）
        for g_name, atoms in c_groups.items():
            if len(atoms) > 0:
                score = sum([node_scores[idx] for idx in atoms]) / len(atoms)
                group_scores[f"{g_name} (Cat)"] += score
            
        # 阴离子（偏移量 = 阳离子原子数）
        a_offset = c_num
        for g_name, atoms in a_groups.items():
            if len(atoms) > 0:
                score = sum([node_scores[idx + a_offset] for idx in atoms]) / len(atoms)
                group_scores[f"{g_name} (Ani)"] += score
            
        # 制冷剂（偏移量 = 阳离子 + 阴离子原子数）
        r_offset = c_num + a_num
        for g_name, atoms in r_groups.items():
            if len(atoms) > 0:
                score = sum([node_scores[idx + r_offset] for idx in atoms]) / len(atoms)
                group_scores[f"{g_name} (Ref)"] += score

        return group_scores

    def group_explain(self, title, c_smi, a_smi, r_smi, T, P, save_name):
        print(f"\n==================================================")
        print(f"🔬 运行基团级别可解释性分析: {title}")
        print(f"   [Cation]: {c_smi}")
        print(f"   [Anion]:  {a_smi}")
        print(f"   [Refri]:  {r_smi} (T={T}K, P={P}bar)")
        
        # 1. 获取原子级别的 IG 得分
        node_scores, atom_types, mol_type = self.get_attention_scores(c_smi, a_smi, r_smi, T, P)
        if node_scores is None:
            print("❌ Graph construction failed.")
            return

        # 归一化得分
        if node_scores.max() > 0:
            node_scores = node_scores / node_scores.max()

        # 2. 调用公共聚合方法
        group_scores = self.aggregate_to_groups(node_scores, c_smi, a_smi, r_smi)

        # 3. 打印结果
        print("\n📊 基团级重要性得分 (Group-level IG Attribution):")
        sorted_groups = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
        
        total_score = sum(group_scores.values())
        print(f"{'基团名称 (所属分子)':<30} | {'绝对总分':<10} | {'贡献占比':<10}")
        print("-" * 55)
        for g_name, score in sorted_groups:
            pct = (score / total_score * 100) if total_score > 0 else 0
            print(f"{g_name:<30} | {score:<10.4f} | {pct:>5.1f}%")

        # 5. 可视化条形图
        self._plot_group_bar(title, sorted_groups, save_name)
        
        return sorted_groups

    def _plot_group_bar(self, title, sorted_groups, save_name):
        # 过滤掉得分为0的项
        groups = [x[0] for x in sorted_groups if x[1] > 0.01]
        scores = [x[1] for x in sorted_groups if x[1] > 0.01]
        
        if not groups:
            return
            
        # 翻转使得分数最高的在最上面
        groups = groups[::-1]
        scores = scores[::-1]

        # 根据所属分子给不同的颜色
        colors = []
        for g in groups:
            if "(Cat)" in g: colors.append('#3498db') # 蓝色
            elif "(Ani)" in g: colors.append('#e74c3c') # 红色
            elif "(Ref)" in g: colors.append('#2ecc71') # 绿色
            else: colors.append('gray')

        plt.figure(figsize=(10, 6))
        bars = plt.barh(groups, scores, color=colors, edgecolor='black', alpha=0.8)
        
        # 添加数据标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                     f'{width:.2f}', ha='left', va='center', fontsize=10)

        plt.xlabel('Average IG Score per Atom (Density of Importance)', fontsize=12)
        plt.title(f'Functional Group Importance Density\n{title}', fontsize=14, pad=15)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', edgecolor='black', alpha=0.8, label='Cation Groups'),
            Patch(facecolor='#e74c3c', edgecolor='black', alpha=0.8, label='Anion Groups'),
            Patch(facecolor='#2ecc71', edgecolor='black', alpha=0.8, label='Refrigerant Groups')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        out_dir = os.path.join(ROOT, 'Interpretability_Case_Studies', 'Results')
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f'{save_name}_groups.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"\n📈 基团重要性条形图已保存: {save_path}")

if __name__ == '__main__':
    explainer = Group_Explainer()
    # Test with BMIM Tf2N + R1234yf
    c_smi = 'CCCCn1cc[n+](C)c1' # [BMIM]
    a_smi = 'FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F' # [Tf2N]
    r_smi = 'C(=C(F)F)(C(F)(F)F)F' # R1234yf
    T = 298.15
    P = 1.0
    
    explainer.group_explain("BMIM-Tf2N absorbing R1234yf", c_smi, a_smi, r_smi, T, P, "test_group_bmim_tf2n_r1234yf")
