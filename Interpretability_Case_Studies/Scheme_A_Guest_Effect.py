import os
import sys

from Group_Explainer_Engine import Group_Explainer

def main():
    explainer = Group_Explainer()
    
    # 相同阴阳离子体系：[BMIM][Tf2N]
    c_smi = "CCCC[n+]1cccc(C)c1" # BMIM
    a_smi = "FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F" # Tf2N
    
    # 物理条件
    T = 298.15
    P = 1.0

    print("\n=======================================================")
    print(" 🧪 [方案 A：客体效应 (Guest Effect)] 运行基团级分析")
    print("=======================================================")
    
    # 1. 对比目标 1: R32 (Polar, H-bond donor)
    explainer.explain(
        title="Scheme A - Target 1: R32 (Polar, H-bond donor)",
        c_smi=c_smi, a_smi=a_smi, r_smi="C(F)F", T=T, P=P,
        save_name="Scheme_A_R32"
    )
    explainer.group_explain(
        title="Scheme A - Target 1: R32 (Polar, H-bond donor)",
        c_smi=c_smi, a_smi=a_smi, r_smi="C(F)F", T=T, P=P,
        save_name="Scheme_A_R32"
    )
    
    # 2. 对比目标 2: R134a (Bulky, High-F)
    explainer.explain(
        title="Scheme A - Target 2: R134a (Bulky, High-F)",
        c_smi=c_smi, a_smi=a_smi, r_smi="C(C(F)(F)F)F", T=T, P=P,
        save_name="Scheme_A_R134a"
    )
    explainer.group_explain(
        title="Scheme A - Target 2: R134a (Bulky, High-F)",
        c_smi=c_smi, a_smi=a_smi, r_smi="C(C(F)(F)F)F", T=T, P=P,
        save_name="Scheme_A_R134a"
    )
    
    # 3. 对比目标 3: R22 (Chlorine atom added)
    explainer.explain(
        title="Scheme A - Target 3: R22 (Chlorine atom added)",
        c_smi=c_smi, a_smi=a_smi, r_smi="ClC(F)F", T=T, P=P,
        save_name="Scheme_A_R22"
    )
    explainer.group_explain(
        title="Scheme A - Target 3: R22 (Chlorine atom added)",
        c_smi=c_smi, a_smi=a_smi, r_smi="ClC(F)F", T=T, P=P,
        save_name="Scheme_A_R22"
    )

if __name__ == "__main__":
    main()
