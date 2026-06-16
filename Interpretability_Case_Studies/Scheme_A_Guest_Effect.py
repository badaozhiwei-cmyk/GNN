import os
import sys

# 引入基座
from Base_GAT_Explainer import UniversalGATExplainer

def main():
    # 假设权重放在 GNN_for_property_prediction/checkpoints_v2/best_gat_seed_1.pth
    # 或者如果不确定，提醒用户
    model_path = os.path.join('..', 'GNN_for_property_prediction', 'checkpoints_v2', 'best_gat_seed_1.pth')
    if not os.path.exists(model_path):
        print(f"找不到权重文件 {model_path}，请确保您已经跑完了 GAT_Runner_v3.py")
        return
        
    explainer = UniversalGATExplainer(model_path)
    
    # 【实验设定：同主场，换客人】
    # 离子液体主场锁定：[BMIM][Tf2N]
    c_smi = "CCCC[n+]1cccc(C)c1" # BMIM
    a_smi = "FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F" # Tf2N
    
    # 宏观环境锁定
    T = 298.15
    P = 1.0

    print("\n=======================================================")
    print(" 🧪 [方案 A：制冷剂客体效应 (Guest Effect)] 开始执行")
    print("=======================================================")
    
    # 1. 挑战者 A：R32 (高极性，氢键供体)
    explainer.explain(
        title="Scheme A - Target 1: R32 (Polar, H-bond donor)",
        c_smi=c_smi, a_smi=a_smi, r_smi="C(F)F", T=T, P=P,
        save_name="Scheme_A_R32"
    )
    
    # 2. 挑战者 B：R134a (大体积，强吸电子氟团)
    explainer.explain(
        title="Scheme A - Target 2: R134a (Bulky, High-F)",
        c_smi=c_smi, a_smi=a_smi, r_smi="C(C(F)(F)F)F", T=T, P=P,
        save_name="Scheme_A_R134a"
    )
    
    # 3. 挑战者 C：R22 (含氯原子，色散力增强)
    explainer.explain(
        title="Scheme A - Target 3: R22 (Chlorine atom added)",
        c_smi=c_smi, a_smi=a_smi, r_smi="ClC(F)F", T=T, P=P,
        save_name="Scheme_A_R22"
    )

if __name__ == "__main__":
    main()
