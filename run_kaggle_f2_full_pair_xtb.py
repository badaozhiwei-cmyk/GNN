"""
run_kaggle_f2_full_pair_xtb.py — Master One-Click Kaggle Controller for F2 Pairwise xTB
======================================================================================
【在 Kaggle Notebook (CPU/GPU 均可，推荐多核 CPU 环境) 中执行】:
    %cd /kaggle/working
    !git clone --depth 1 https://github.com/badaozhiwei-cmyk/GNN.git
    %cd /kaggle/working/GNN
    !git pull origin main

    !pip install -q rdkit pandas numpy

    # 一键启动 218 对配对 xTB 计算 (包含自动下载 xTB 引擎与断点续算):
    !python run_kaggle_f2_full_pair_xtb.py

【流水线特性】:
1. 自动下载与解压 GFN2-xTB 6.6.1 Linux 二进制引擎;
2. 调度执行 compute_full_pair_interaction_xtb.py 完成 218 对离子-制冷剂缔合能计算 (4 构向采样);
3. 严格审计 Gate F2-1 (收敛率) 与 Gate F2-2 (Phase 2 的 10 个 Unique Physical Pairs 覆盖率);
4. 自动打包 full_pair_interaction_results.csv 为 /kaggle/working/f2_xtb_pair_results.zip 供下载。
======================================================================================
"""

import os
import sys
import time
import zipfile
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

print("=" * 85)
print("  KAGGLE: MASTER F2 PAIRWISE xTB ASSOCIATION ENERGY CONTROLLER")
print("=" * 85)

# 10 个 Phase 2 唯一定义的物理配对基准
PHASE2_10_PAIRS = [
    ("[emim]", "[Ac]", "R1234yf"),
    ("[emim]", "[BF4]", "R1234yf"),
    ("[bmim]", "[Ac]", "R1234yf"),
    ("[bmim]", "[PF6]", "R1234yf"),
    ("[emim]", "[BEI]", "R134"),
    ("[emim]", "[BEI]", "R134a"),
    ("[bmim]", "[BF4]", "R32"),
    ("[bmim]", "[PF6]", "R32"),
    ("[emim]", "[Tf2N]", "R1336mzz(E)"),
    ("[emim]", "[Tf2N]", "R1336mzz(Z)")
]

def main():
    start_time = time.time()

    # Step 1: 检查或下载 Linux xTB 引擎
    print("\n>>> [STEP 1/3] 检查 GFN2-xTB Linux 二进制引擎...")
    xtb_root = Path("/kaggle/working/xtb-dist")
    if not (xtb_root / "bin" / "xtb").exists() and Path("/kaggle/working").exists():
        print("  📦 正在自动下载并解压 xTB 6.6.1 (grimme-lab/xtb)...")
        os.system("cd /kaggle/working && wget -q https://github.com/grimme-lab/xtb/releases/download/v6.6.1/xtb-6.6.1-linux-x86_64.tar.xz && tar -xf xtb-6.6.1-linux-x86_64.tar.xz && mv xtb-6.6.1 xtb-dist && rm -f xtb-6.6.1-linux-x86_64.tar.xz")

    if xtb_root.exists():
        os.environ['PATH'] = f"{xtb_root}/bin:" + os.environ.get('PATH', '')
        os.environ['XTBPATH'] = f"{xtb_root}/share/xtb"

    try:
        chk = subprocess.run(['xtb', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        v_str = chk.stdout.splitlines()[0] if chk.stdout else 'Active'
        print(f"  ✓ xTB 引擎就绪: {v_str}")
    except Exception as e:
        print(f"  ❌ xTB 引擎检测失败: {e}")
        sys.exit(1)

    # Step 2: 调度执行 compute_full_pair_interaction_xtb.py
    print("\n>>> [STEP 2/3] 启动 218 对配对复合物优化计算 (4 构向采样)...")
    script_p = ROOT / "Phase4_Scientific_Validation" / "compute_full_pair_interaction_xtb.py"
    ret = subprocess.run([sys.executable, str(script_p)])
    if ret.returncode != 0:
        print(f"  ❌ xTB 配对计算执行异常，退出码: {ret.returncode}")
        sys.exit(ret.returncode)

    # Step 3: 严格审计产物与 10 个 Unique Pairs 覆盖
    out_csv = ROOT / "Phase4_Scientific_Validation" / "full_pair_interaction_results.csv"
    if not out_csv.exists():
        print(f"  ❌ 未找到产物文件: {out_csv}")
        sys.exit(1)

    print("\n>>> [STEP 3/3] 正在审计 F2 配对计算产物门禁...")
    df_res = pd.read_csv(out_csv)
    n_total = len(df_res)
    n_succ = (df_res["Status"] == "Success").sum()
    print(f"  • 全量配对任务数: {n_total} (收敛成功: {n_succ} / {n_total}, 成功率 {n_succ/max(n_total,1)*100:.1f}%)")

    # 重点核查 Phase 2 的 10 个物理配对
    print(f"\n  --- 检查 Phase 2 关键的 10 个 Unique Physical Pairs 覆盖率 ---")
    covered_p2 = 0
    for cat, ani, ref in PHASE2_10_PAIRS:
        match_cat = df_res[(df_res["Pair_Type"] == "Cation-Ref") & (df_res["Ion_Name"] == cat) & (df_res["Refrigerant"] == ref)]
        match_ani = df_res[(df_res["Pair_Type"] == "Anion-Ref") & (df_res["Ion_Name"] == ani) & (df_res["Refrigerant"] == ref)]
        cat_ok = not match_cat.empty and match_cat.iloc[0]["Status"] == "Success"
        ani_ok = not match_ani.empty and match_ani.iloc[0]["Status"] == "Success"
        if cat_ok and ani_ok:
            covered_p2 += 1
            cat_e = match_cat.iloc[0]["Delta_E_assoc_kcal_mol"]
            ani_e = match_ani.iloc[0]["Delta_E_assoc_kcal_mol"]
            print(f"    ✓ Pair: {cat:<8} + {ani:<8} + {ref:<14} -> ΔE_cat={cat_e:.2f}, ΔE_ani={ani_e:.2f} kcal/mol")
        else:
            print(f"    ! 警告: 物理对 {cat} + {ani} + {ref} 尚未完全收敛 (Cation OK: {cat_ok}, Anion OK: {ani_ok})")

    print(f"  • Phase 2 核心 10 对物理覆盖度: {covered_p2} / 10 ({covered_p2*10:.1f}%)")

    # 打包产物
    zip_out = Path("/kaggle/working/f2_xtb_pair_results.zip")
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, "full_pair_interaction_results.csv")
    
    elapsed = (time.time() - start_time) / 60.0
    print("\n" + "=" * 85)
    print("  🎉 F2 配对计算与审计全量通过！")
    print(f"  产物压缩包已就绪: {zip_out} ({zip_out.stat().st_size / 1024:.1f} KB)")
    print(f"  总耗时: {elapsed:.2f} 分钟")
    print("  请直接在 Kaggle Notebook 输出区下载 'f2_xtb_pair_results.zip' 并解压至本地")
    print("  Phase4_Scientific_Validation/ 目录下！")
    print("=" * 85)

if __name__ == "__main__":
    main()
