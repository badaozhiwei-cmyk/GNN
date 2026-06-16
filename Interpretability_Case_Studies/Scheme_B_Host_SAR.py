import os
import sys

# 引入基座
from Explainer_Engine import Explainer_Engine

def main():
    model_path = os.path.join('..', 'GNN_for_property_prediction', 'checkpoints_v2', 'best_gat_seed_1.pth')
    if not os.path.exists(model_path):
        print(f"找不到权重文�?{model_path}，请确保您已经跑完了 GAT_Runner_v3.py")
        return
        
    explainer = Explainer_Engine(model_path)
    
    # 【实验设定：同客人，换环境�?    # 靶向气体锁定：R32
    r_smi = "C(F)F" 
    T = 298.15
    P = 1.0

    print("\n=======================================================")
    print(" 🧪 [方案 B：离子液体构效关�?(Host SAR)] 开始执�?)
    print("=======================================================")
    
    # 阳离子全部锁定为 [BMIM]
    c_smi = "CCCC[n+]1cccc(C)c1"
    
    # 1. 环境 A：无氟阴离子 [OAc] (醋酸�?
    explainer.explain(
        title="Scheme B - Env 1: [BMIM][OAc] (F-free Anion)",
        c_smi=c_smi, a_smi="CC(=O)[O-]", r_smi=r_smi, T=T, P=P,
        save_name="Scheme_B_OAc"
    )
    
    # 2. 环境 B：低氟阴离子 [BF4] (四氟硼酸�?
    explainer.explain(
        title="Scheme B - Env 2: [BMIM][BF4] (Low-F Anion)",
        c_smi=c_smi, a_smi="F[B-](F)(F)F", r_smi=r_smi, T=T, P=P,
        save_name="Scheme_B_BF4"
    )

    # 3. 环境 C：高氟巨型阴离子 [Tf2N] (双三氟甲磺酰亚胺)
    explainer.explain(
        title="Scheme B - Env 3: [BMIM][Tf2N] (High-F Anion)",
        c_smi=c_smi, a_smi="FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F", r_smi=r_smi, T=T, P=P,
        save_name="Scheme_B_Tf2N"
    )

if __name__ == "__main__":
    main()
