"""
recompute_r1234yf_xtb_pairs.py — 靶向重新计算 12 个 R1234yf 离子配对结合能
=============================================================================
协议遵循: F2_Physical_Alignment_Design_Protocol (v2.0 Rigorous)
化学真理源: refrigerant_identity_manifest.csv (C3H2F4, SMILES C=C(F)C(F)(F)F)

核心流程:
1. 校验输入: 强制断言 R1234yf 化学身份为真实 2,3,3,3-四氟丙烯 (InChIKey: FXRLMCRCYDHQFW-UHFFFAOYSA-N)。
2. 加载现有数据: 读取已有的 218 对计算结果，保留 206 对未经污染的非 R1234yf 配对。
3. 靶向单体与复合物优化: 仅对 R1234yf 单体及其 12 个离子配对 (4 构向采样) 执行 GFN2-xTB tight 几何优化。
4. 结果合并与全量门禁: 生成更新后的 full_pair_interaction_results.csv (218 行, 100% Success, 27-column schema)。
"""

import os
import sys
import shutil
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path

# 统一编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 自动定位工程根目录
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from chemical_identity_contract import get_refrigerant_identity, assert_refrigerant_contract
from compute_full_pair_interaction_xtb import (
    build_monomer_3d,
    run_xtb_opt,
    create_dimer_orientations,
    write_xyz,
    compute_min_distance,
    parse_energy,
    check_convergence,
    HARTREE_TO_KCAL,
    EXPECTED_COLUMNS
)

# 目标 12 个离子配对白名单
TARGET_ANION_IONS = ['[Ac]', '[BF4]', '[OTf]', '[SCN]', '[Tf2N]', '[PF6]', '[Cl]']
TARGET_CATION_IONS = ['[emim]', '[bmim]', '[hmim]', '[omim]', '[P66614]']

def main():
    print("=" * 80)
    print("  TARGETED XTB RECOMPUTATION: 12 R1234yf PAIRS (TRUE C3H2F4 GEOMETRY)")
    print("=" * 80)

    # 1. 强制契约校验
    yf_ident = get_refrigerant_identity("R1234yf")
    assert yf_ident['formula'] == 'C3H2F4', f"Registry formula error: {yf_ident['formula']}"
    assert yf_ident['n_heavy_atoms'] == 7, f"Heavy atom error: {yf_ident['n_heavy_atoms']}"
    assert yf_ident['canonical_smiles'] == 'C=C(F)C(F)(F)F', f"SMILES error: {yf_ident['canonical_smiles']}"
    print(f"🔒 [Contract Gate PASS] R1234yf chemical identity locked: {yf_ident['formula']} ({yf_ident['canonical_smiles']})")

    # 2. 检查 xTB 环境
    xtb_exe = shutil.which("xtb")
    if not xtb_exe:
        # 尝试 Kaggle 自动解压路径
        if os.path.exists('/kaggle/working/xtb-dist/bin/xtb'):
            xtb_exe = '/kaggle/working/xtb-dist/bin/xtb'
            os.environ['PATH'] = '/kaggle/working/xtb-dist/bin:' + os.environ.get('PATH', '')
            os.environ['XTBPATH'] = '/kaggle/working/xtb-dist/share/xtb'
        else:
            print("🚨 错误: 找不到 xtb 可执行文件！")
            print("如果是在本地运行，请配置 xTB 到 PATH；如果是在 Kaggle 运行，本脚本将自动下载 xTB。")
            if os.path.exists('/kaggle/working'):
                print("📦 正在自动下载并解压 xTB Linux 二进制引擎 (GFN2-xTB)...")
                os.system("cd /kaggle/working && wget -q https://github.com/grimme-lab/xtb/releases/download/v6.6.1/xtb-6.6.1-linux-x86_64.tar.xz && tar -xf xtb-6.6.1-linux-x86_64.tar.xz && mv xtb-6.6.1 xtb-dist && rm -f xtb-6.6.1-linux-x86_64.tar.xz")
                os.environ['PATH'] = '/kaggle/working/xtb-dist/bin:' + os.environ.get('PATH', '')
                os.environ['XTBPATH'] = '/kaggle/working/xtb-dist/share/xtb'
                xtb_exe = shutil.which("xtb")
            
            if not xtb_exe:
                raise RuntimeError("xTB executable could not be resolved.")

    print(f"  ✓ xTB Engine found at: {xtb_exe}")

    # 3. 准备输出文件与备份
    pair_csv = SCRIPT_DIR / "full_pair_interaction_results.csv"
    if not pair_csv.exists():
        raise FileNotFoundError(f"Missing authoritative pair results to update: {pair_csv}")

    backup_csv = SCRIPT_DIR / "full_pair_interaction_results_backup_pre_r1234yf_fix.csv"
    if not backup_csv.exists():
        shutil.copy2(pair_csv, backup_csv)
        print(f"  ✓ Created pre-fix backup: {backup_csv.name}")

    df_existing = pd.read_csv(pair_csv)
    assert len(df_existing) == 218, f"Expected 218 rows in existing results, got {len(df_existing)}"

    # 过滤出 206 对未经污染的样本
    df_clean_206 = df_existing[df_existing['Refrigerant'] != 'R1234yf'].copy()
    assert len(df_clean_206) == 206, f"Expected 206 clean pairs, got {len(df_clean_206)}"
    print(f"  ✓ Retained {len(df_clean_206)} clean, non-R1234yf historical calculations.")

    work_dir = SCRIPT_DIR / "r1234yf_recompute_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 4. 单体优化: R1234yf 单体
    print("\n[Step 1/2] 优化真实 R1234yf 单体几何结构...")
    ref_sub = work_dir / "monomer_R1234yf"
    ref_sub.mkdir(parents=True, exist_ok=True)
    ref_init_xyz = ref_sub / "init.xyz"
    
    ref_mol_3d = build_monomer_3d(yf_ident['canonical_smiles'])
    if ref_mol_3d is None:
        raise RuntimeError("Failed to build 3D conformer for R1234yf!")
    
    # 提取坐标并落盘 init.xyz
    conf = ref_mol_3d.GetConformer()
    syms = [atom.GetSymbol() for atom in ref_mol_3d.GetAtoms()]
    coords = conf.GetPositions()
    write_xyz(syms, coords, str(ref_init_xyz))

    e_ref, conv_ref, _, _ = run_xtb_opt(str(ref_init_xyz), 0, str(ref_sub))
    assert conv_ref, "R1234yf monomer geometry optimization failed to converge!"
    print(f"  ✓ R1234yf Monomer Optimized: E = {e_ref:.8f} Eh (Converged: {conv_ref})")

    # 5. 加载或优化各目标离子的单体能量
    print("\n[Step 2/2] 优化 12 个配对复合物 (每个配对 4 个空间初始取向)...")
    # 从已有数据中提取离子的单体参考能量
    ion_energies = {}
    for _, row in df_clean_206.iterrows():
        ion_energies[row['Ion_Name']] = float(row['E_ion_Eh'])

    # 加载权威索引以获取离子 SMILES
    index_csv = ROOT_DIR / "index_with_anion.csv"
    df_index = pd.read_csv(index_csv)

    recomputed_rows = []
    
    # 构建 12 个待算任务
    tasks = []
    for ion in TARGET_ANION_IONS:
        sub = df_index[(df_index['anion'] == ion) & (df_index['refrigerant'] == 'R1234yf')]
        smi = sub['anion_smiles'].iloc[0]
        tasks.append(('Anion-Ref', ion, smi, -1, 'R1234yf', yf_ident['canonical_smiles'], 0, -1))

    for ion in TARGET_CATION_IONS:
        sub = df_index[(df_index['cation'] == ion) & (df_index['refrigerant'] == 'R1234yf')]
        smi = sub['cation_smiles'].iloc[0]
        tasks.append(('Cation-Ref', ion, smi, 1, 'R1234yf', yf_ident['canonical_smiles'], 0, 1))

    assert len(tasks) == 12, f"Expected exactly 12 tasks, got {len(tasks)}"

    for task_idx, (pair_type, ion_name, ion_smi, ion_q, ref_name, ref_smi, ref_q, dimer_q) in enumerate(tasks, 1):
        pair_tag = f"{pair_type}_{ion_name}_{ref_name}".replace('[', '').replace(']', '').replace('-', '_')
        print(f"\n--- [{task_idx}/12] Computing Pair: {pair_type} {ion_name} + {ref_name} (Charge: {dimer_q}) ---")

        # 确保离子单体能量存在
        if ion_name not in ion_energies:
            ion_sub = work_dir / f"monomer_{pair_tag}_ion"
            ion_sub.mkdir(parents=True, exist_ok=True)
            ion_init = ion_sub / "init.xyz"
            m_ion = build_monomer_3d(ion_smi)
            c_ion = m_ion.GetConformer()
            write_xyz([a.GetSymbol() for a in m_ion.GetAtoms()], c_ion.GetPositions(), str(ion_init))
            e_i, c_i, _, _ = run_xtb_opt(str(ion_init), ion_q, str(ion_sub))
            assert c_i, f"Ion monomer optimization failed for {ion_name}"
            ion_energies[ion_name] = e_i

        e_ion = ion_energies[ion_name]

        # 4 方向采样
        ion_mol = build_monomer_3d(ion_smi)
        ref_mol = build_monomer_3d(ref_smi)
        syms_dimer, orientations, n_ion_atoms = create_dimer_orientations(ion_mol, ref_mol)

        ori_data = {}
        converged_candidates = []

        for o_idx, coords in enumerate(orientations, 1):
            sub_work = work_dir / f"pair_{pair_tag}_ori{o_idx}"
            sub_work.mkdir(parents=True, exist_ok=True)
            init_xyz = sub_work / "init.xyz"
            write_xyz(syms_dimer, coords, str(init_xyz))

            e_opt, conv, final_xyz, _ = run_xtb_opt(str(init_xyz), dimer_q, str(sub_work))
            d_min = compute_min_distance(final_xyz, n_ion_atoms) if final_xyz else None

            ori_data[f"E_ori{o_idx}_Eh"] = e_opt
            ori_data[f"converged_ori{o_idx}"] = conv
            ori_data[f"d_min_ori{o_idx}_Angstrom"] = d_min

            if e_opt is not None and conv:
                converged_candidates.append({
                    'orient_idx': o_idx,
                    'E_complex': e_opt,
                    'd_min': d_min
                })

        assert len(converged_candidates) >= 1, f"🚨 致命失败: 配对 {pair_tag} 在 4 个构向采样下无一收敛！"

        best = min(converged_candidates, key=lambda x: x['E_complex'])
        e_complex = best['E_complex']
        best_d_min = best['d_min']
        best_ori = best['orient_idx']
        delta_e_assoc = (e_complex - (e_ion + e_ref)) * HARTREE_TO_KCAL

        print(f"   🏆 最优构向 (Ori {best_ori}): ΔE_assoc = {delta_e_assoc:.2f} kcal/mol | d_min = {best_d_min:.2f} Å (收敛 {len(converged_candidates)}/4)")

        res_row = {
            'Pair_Type': pair_type,
            'Ion_Name': ion_name,
            'Refrigerant': ref_name,
            'Delta_E_assoc_kcal_mol': delta_e_assoc,
            'Delta_E_int_kcal_mol': delta_e_assoc,
            'd_min_Angstrom': best_d_min,
            'Best_Orientation': best_ori,
            'N_Converged_Orientations': len(converged_candidates),
            'E_complex_Eh': e_complex,
            'E_ion_Eh': e_ion,
            'E_ref_Eh': e_ref,
            **ori_data,
            'energy_selection_criterion': 'lowest-energy converged structure among four sampled initial orientations',
            'physical_definition': 'Association energy relative to isolated optimized monomers (includes geometry relaxation)',
            'monomer_conformer_protocol': 'deterministic single-conformer monomer reference (ETKDGv3 seed=42 + MMFF/UFF + GFN2-xTB tight)',
            'Status': 'Success'
        }
        recomputed_rows.append(res_row)

    # 6. 合并 206 干净历史行 + 12 重新计算的干净行
    df_recomputed = pd.DataFrame(recomputed_rows)
    df_final = pd.concat([df_clean_206, df_recomputed], ignore_index=True)

    # 校验最终 218 集合完整性与 27 列 Schema
    assert len(df_final) == 218, f"Expected 218 pairs in final merged file, got {len(df_final)}"
    assert list(df_final.columns) == EXPECTED_COLUMNS, "Final columns mismatch EXPECTED_COLUMNS!"
    assert (df_final['Status'] == 'Success').all(), "Not all rows have Status == Success!"
    assert df_final['Delta_E_assoc_kcal_mol'].notna().all(), "Found NaN in Delta_E_assoc!"

    df_final.to_csv(pair_csv, index=False)
    print("\n" + "=" * 80)
    print(f"✅ 成功完成 12 对 R1234yf 配对物理结合能重算，并完整合并入权威结果表: {pair_csv}")
    print(f"📊 总配对数: {len(df_final)} | 100% 收敛成功率 | 27 列科学元数据 Schema 严格合规")
    print("=" * 80)

if __name__ == '__main__':
    main()
