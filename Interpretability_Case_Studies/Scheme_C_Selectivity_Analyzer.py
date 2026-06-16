import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 引入基座
from Explainer_Engine import Explainer_Engine

def main():
    print("=" * 70)
    print(" ⚖️ 启动理想选择性分析器 (Selectivity Analyzer)")
    print("=" * 70)

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(ROOT, 'GNN_for_property_prediction', 'checkpoints_v2', 'best_gat_seed_1.pth')
    
    if not os.path.exists(model_path):
        print(f"找不到权重文�?{model_path}，请确保您在 Kaggle 上正确配置了路径�?)
        return
        
    explainer = Explainer_Engine(model_path)
    
    # 【实验设定：锁定离子液体，对比两种相近制冷剂�?    # 分离主场：[BMIM][Tf2N]
    c_smi = "CCCC[n+]1cccc(C)c1" # [BMIM]
    a_smi = "FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F" # [Tf2N]
    
    # 宏观环境
    T = 298.15
    P = 1.0

    # 竞争气体�?    r1_name = "R32"
    r1_smi = "C(F)F"
    
    r2_name = "R134a"
    r2_smi = "C(C(F)(F)F)F"

    print(f"\n[任务] 正在分析 [BMIM][Tf2N] 吸收 {r1_name} vs {r2_name} 时的注意力重分配...")

    # 获取 R32 的注意力
    scores1, atom_types1, mol_types1 = explainer.get_attention_scores(c_smi, a_smi, r1_smi, T, P)
    # 获取 R134a 的注意力
    scores2, atom_types2, mol_types2 = explainer.get_attention_scores(c_smi, a_smi, r2_smi, T, P)

    if scores1 is None or scores2 is None:
        print("推断失败，请检查模型与 SMILES�?)
        return

    # 分离出阳离子 (mol=0) �?阴离�?(mol=1) 的总注意力
    cat_score1 = sum([s for s, m in zip(scores1, mol_types1) if m == 0])
    ani_score1 = sum([s for s, m in zip(scores1, mol_types1) if m == 1])
    
    cat_score2 = sum([s for s, m in zip(scores2, mol_types2) if m == 0])
    ani_score2 = sum([s for s, m in zip(scores2, mol_types2) if m == 1])

    # 为了更公平，计算在“离子液体内部”的注意力占�?    total_il1 = cat_score1 + ani_score1
    total_il2 = cat_score2 + ani_score2

    cat_ratio1 = (cat_score1 / total_il1) * 100
    ani_ratio1 = (ani_score1 / total_il1) * 100

    cat_ratio2 = (cat_score2 / total_il2) * 100
    ani_ratio2 = (ani_score2 / total_il2) * 100

    print(f"\n当吸�?{r1_name} �?")
    print(f"  �?[BMIM]+ 关注�? {cat_ratio1:.1f}%")
    print(f"  �?[Tf2N]- 关注�? {ani_ratio1:.1f}%")

    print(f"\n当吸�?{r2_name} �?")
    print(f"  �?[BMIM]+ 关注�? {cat_ratio2:.1f}%")
    print(f"  �?[Tf2N]- 关注�? {ani_ratio2:.1f}%")

    # ==========================
    # 开始画对比分组柱状�?    # ==========================
    labels = ['Cation: [BMIM]+', 'Anion: [Tf2N]-']
    r32_ratios = [cat_ratio1, ani_ratio1]
    r134a_ratios = [cat_ratio2, ani_ratio2]

    x = np.arange(len(labels))  # label locations
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, r32_ratios, width, label=r1_name, color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width/2, r134a_ratios, width, label=r2_name, color='#ff7f0e', edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Attention Share within Ionic Liquid (%)', fontsize=12, weight='bold')
    ax.set_title('Attention Reallocation for Refrigerant Selectivity\n(R32 vs R134a in [BMIM][Tf2N])', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', weight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    out_dir = os.path.join(ROOT, 'Interpretability_Case_Studies', 'Results')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'Scheme_C_Selectivity_Analyzer.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\n�?方案 C 对比柱状图已生成，保存在：{save_path}")

if __name__ == "__main__":
    main()
