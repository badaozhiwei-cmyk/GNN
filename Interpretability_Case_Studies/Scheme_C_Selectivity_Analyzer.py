import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 升级引擎：从原子级 Explainer_Engine 升级到基团级 Group_Explainer
from Group_Explainer_Engine import Group_Explainer

def main():
    print("=" * 70)
    print(" ⚖️ 启动理想选择性分析器 (Selectivity Analyzer) — 基团级版本")
    print("=" * 70)

    explainer = Group_Explainer()
    
    # 【实验设定：锁定离子液体，对比两种相近制冷剂】
    # 分离主场：[BMIM][Tf2N]
    c_smi = "CCCC[n+]1cccc(C)c1" # [BMIM]
    a_smi = "FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F" # [Tf2N]
    
    # 宏观环境
    T = 298.15
    P = 1.0

    # 竞争气体对
    r1_name = "R32"
    r1_smi = "C(F)F"
    
    r2_name = "R134a"
    r2_smi = "C(C(F)(F)F)F"

    print(f"\n[任务] 正在分析 [BMIM][Tf2N] 吸收 {r1_name} vs {r2_name} 时的基团级注意力重分配...")

    # ============================================================
    # Step 1: 同时输出原子图 + 基团条形图（与 Scheme A 深度一致）
    # ============================================================
    explainer.explain(
        title=f"Scheme C - {r1_name} in [BMIM][Tf2N]",
        c_smi=c_smi, a_smi=a_smi, r_smi=r1_smi, T=T, P=P,
        save_name=f"Scheme_C_{r1_name}"
    )
    explainer.group_explain(
        title=f"Scheme C - {r1_name} in [BMIM][Tf2N]",
        c_smi=c_smi, a_smi=a_smi, r_smi=r1_smi, T=T, P=P,
        save_name=f"Scheme_C_{r1_name}"
    )

    explainer.explain(
        title=f"Scheme C - {r2_name} in [BMIM][Tf2N]",
        c_smi=c_smi, a_smi=a_smi, r_smi=r2_smi, T=T, P=P,
        save_name=f"Scheme_C_{r2_name}"
    )
    explainer.group_explain(
        title=f"Scheme C - {r2_name} in [BMIM][Tf2N]",
        c_smi=c_smi, a_smi=a_smi, r_smi=r2_smi, T=T, P=P,
        save_name=f"Scheme_C_{r2_name}"
    )

    # ============================================================
    # Step 2: 基团级选择性对比柱状图
    # ============================================================
    # 获取 R32 的基团级得分
    scores1, _, _ = explainer.get_attention_scores(c_smi, a_smi, r1_smi, T, P)
    scores2, _, _ = explainer.get_attention_scores(c_smi, a_smi, r2_smi, T, P)

    if scores1 is None or scores2 is None:
        print("推断失败，请检查模型与 SMILES")
        return

    # 归一化逻辑：对于“选择性比较”，最科学的方法是计算“相对百分比”
    # 将注意力得分除以该分子全图最高分，统一基础尺度
    if scores1.max() > 0:
        scores1 = scores1 / scores1.max()
    if scores2.max() > 0:
        scores2 = scores2 / scores2.max()
        
    # 获取 R32 环境下的基团总分
    gs1 = explainer.aggregate_to_groups(scores1, c_smi, a_smi, r1_smi)
    # 获取 R134a 环境下的基团总分
    gs2 = explainer.aggregate_to_groups(scores2, c_smi, a_smi, r2_smi)

    # 收集所有基团名称的并集（仅取离子液体部分 Cat + Ani）
    il_groups = sorted(set(
        [k for k in gs1 if "(Cat)" in k or "(Ani)" in k] +
        [k for k in gs2 if "(Cat)" in k or "(Ani)" in k]
    ))
    
    # 提取绝对分数
    vals1_abs = [gs1.get(g, 0.0) for g in il_groups]
    vals2_abs = [gs2.get(g, 0.0) for g in il_groups]
    
    # 【核心数学升级】：计算每个基团在离子液体内部争夺注意力的“百分比占比”
    sum_il1 = sum(vals1_abs)
    sum_il2 = sum(vals2_abs)
    
    vals1 = [v / sum_il1 * 100 if sum_il1 > 0 else 0 for v in vals1_abs]
    vals2 = [v / sum_il2 * 100 if sum_il2 > 0 else 0 for v in vals2_abs]

    # ============================================================
    # Step 3: 画分组柱状图
    # ============================================================
    x = np.arange(len(il_groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, vals1, width, label=r1_name, color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width/2, vals2, width, label=r2_name, color='#ff7f0e', edgecolor='black')

    ax.set_ylabel('Aggregated IG Score (Normalized)', fontsize=12, weight='bold')
    ax.set_title('Group-Level Attention Reallocation for Refrigerant Selectivity\n'
                 f'({r1_name} vs {r2_name} in [BMIM][Tf2N])', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(il_groups, fontsize=9, rotation=35, ha='right')
    ax.legend(fontsize=12)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, weight='bold')

    autolabel(rects1)
    autolabel(rects2)

    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(ROOT, 'Interpretability_Case_Studies', 'Results')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'Scheme_C_Group_Selectivity.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    # ============================================================
    # Step 4: 打印定量对比表格
    # ============================================================
    print(f"\n{'基团':<30} | {r1_name:<10} | {r2_name:<10} | {'差异 (Δ)':<10}")
    print("-" * 65)
    for g, v1, v2 in zip(il_groups, vals1, vals2):
        delta = v1 - v2
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"{g:<30} | {v1:<10.4f} | {v2:<10.4f} | {delta:>+.4f} {arrow}")

    print(f"\n✅ 基团级选择性对比柱状图已保存至: {save_path}")

if __name__ == "__main__":
    main()
