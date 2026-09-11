"""
build_split_B_family_ood.py
===========================
【使命】
  严格构建并审计溶剂侧 OOD (Solvent-Side / Anion-Family OOD) 的 B1 与 B2 划分。
  纯数据划分与严密反泄漏门禁工具，不包含任何模型训练代码。

【先验家族定义 (A Priori Taxonomy)】
  Anion families are defined a priori according to shared charged-core structural motifs:
  - Fam-1_Sulfonimide: [(SO2)2N]- imide or [(SO2)3C]- methide
  - Fam-2_Fluorosulfonate: -SO3- sulfonate attached to fluorinated chain/ether
  - Fam-3_InorganicFluoride: octahedral/tetrahedral purely inorganic fluoro-complexes ([PF6]-, [BF4]-)
  - Fam-4_PhosphorusOrganic: phosphinates and fluoroalkylphosphates
  - Fam-5_Carboxylate_Sulfate: carboxylate -COO- or alkyl sulfate
  - Fam-6_Halide_Pseudohalide: simple halides and pseudohalides

【划分目标】
  - Split B1: Fam-2 (全氟烷基磺酸盐，7种阴离子整族留出，目标占比 15%~25%)
  - Split B2: Fam-3 (无机球形对称氟化物，2种阴离子整族留出，目标占比 15%~25%)
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 设定项目根目录
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# ──────────────────────────────────────────────────────────
# 1. 先验化学母核定义 (A Priori Taxonomy)
# ──────────────────────────────────────────────────────────
ANION_TAXONOMY = {
    # ── Fam-1: 磺酰亚胺与甲基化物 ──
    'TF2N':     ('Fam-1_Sulfonimide', 'Bis(trifluoromethylsulfonyl)imide', '[(SO2)2N]- central core'),
    'BEI':      ('Fam-1_Sulfonimide', 'Bis(pentafluoroethylsulfonyl)imide', '[(SO2)2N]- central core'),
    'TMEM':     ('Fam-1_Sulfonimide', 'Tris(trifluoromethylsulfonyl)methide', '[(SO2)3C]- central core'),
    
    # ── Fam-2: 全氟烷基磺酸盐 (Split B1 留出目标) ──
    'OTF':      ('Fam-2_Fluorosulfonate', 'Trifluoromethanesulfonate', '-SO3- sulfonate core'),
    'TFES':     ('Fam-2_Fluorosulfonate', '1,1,2,2-Tetrafluoroethanesulfonate', '-SO3- sulfonate core'),
    'HFPS':     ('Fam-2_Fluorosulfonate', '1,1,2,3,3,3-Hexafluoropropanesulfonate', '-SO3- sulfonate core'),
    'TPES':     ('Fam-2_Fluorosulfonate', 'Fluoroether sulfonate', '-SO3- sulfonate core'),
    'PFBS':     ('Fam-2_Fluorosulfonate', 'Nonafluorobutanesulfonate', '-SO3- sulfonate core'),
    'TTES':     ('Fam-2_Fluorosulfonate', 'Fluoroether sulfonate', '-SO3- sulfonate core'),
    'FS':       ('Fam-2_Fluorosulfonate', 'Fluorinated sulfonate', '-SO3- sulfonate core'),
    
    # ── Fam-3: 无机球形氟化物 (Split B2 留出目标) ──
    'PF6':      ('Fam-3_InorganicFluoride', 'Hexafluorophosphate', 'Octahedral [PF6]- core'),
    'BF4':      ('Fam-3_InorganicFluoride', 'Tetrafluoroborate', 'Tetrahedral [BF4]- core'),
    
    # ── Fam-4: 磷系有机阴离子 ──
    'TMPP':     ('Fam-4_PhosphorusOrganic', 'Bis(2,4,4-trimethylpentyl)phosphinate', 'Phosphinate P(=O)[O-] core'),
    'FEP':      ('Fam-4_PhosphorusOrganic', 'Tris(pentafluoroethyl)trifluorophosphate', 'Organofluoro [P(C2F5)3F3]- core'),
    'ET2PO4':   ('Fam-4_PhosphorusOrganic', 'Diethyl phosphate', 'Phosphate P(=O)[O-] core'),
    
    # ── Fam-5: 羧酸盐与简单弱酸根 ──
    'AC':       ('Fam-5_Carboxylate_Sulfate', 'Acetate', 'Carboxylate -COO- core'),
    'PFP':      ('Fam-5_Carboxylate_Sulfate', 'Pentafluoropropionate', 'Perfluorocarboxylate -COO- core'),
    'MESO4':    ('Fam-5_Carboxylate_Sulfate', 'Methyl sulfate', 'Alkyl sulfate -OSO3- core'),
    'PR':       ('Fam-5_Carboxylate_Sulfate', 'Propionate', 'Carboxylate -COO- core'),
    'PE':       ('Fam-5_Carboxylate_Sulfate', 'Pentanoate', 'Carboxylate -COO- core'),
    
    # ── Fam-6: 卤素与拟卤素 ──
    'CL':       ('Fam-6_Halide_Pseudohalide', 'Chloride', 'Monoatomic [Cl-]'),
    'SCN':      ('Fam-6_Halide_Pseudohalide', 'Thiocyanate', 'Pseudohalide [S-C#N]'),
    'I':        ('Fam-6_Halide_Pseudohalide', 'Iodide', 'Monoatomic [I-]'),
}

def clean_name(x):
    return str(x).strip().upper().replace('[', '').replace(']', '').replace('-', '')

def assign_taxonomy(anion_name):
    key = clean_name(anion_name)
    if key in ANION_TAXONOMY:
        fam, full_name, motif = ANION_TAXONOMY[key]
        return pd.Series([fam, full_name, motif], index=['anion_family', 'anion_full_name', 'charged_core_motif'])
    else:
        raise ValueError(f"CRITICAL: Anion {anion_name} (clean key: {key}) not found in a priori taxonomy!")


# ──────────────────────────────────────────────────────────
# 2. 审计与硬断言引擎 (Audit & Assertion Engine)
# ──────────────────────────────────────────────────────────
def run_strict_assertions(df, train_idx, val_idx, test_idx, target_family, split_name):
    """
    运行硬断言。任何一条不满足直接抛出 AssertionError 终止执行。
    """
    print(f"\n[Assert] Running hard assertions for {split_name} (Target: {target_family})...")
    
    train_set = set(train_idx)
    val_set   = set(val_idx)
    test_set  = set(test_idx)
    
    # 1. 索引无重叠且覆盖完全
    assert len(train_set & val_set) == 0, f"[{split_name}] FATAL: Train and Val index overlap!"
    assert len(train_set & test_set) == 0, f"[{split_name}] FATAL: Train and Test index overlap!"
    assert len(val_set & test_set) == 0, f"[{split_name}] FATAL: Val and Test index overlap!"
    
    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]
    test_df  = df.iloc[test_idx]
    
    # 2. 测试集家族纯度硬断言 (100% 属于目标家族)
    test_families = set(test_df['anion_family'])
    assert test_families == {target_family}, (
        f"[{split_name}] FATAL: Test set contains non-target families! Found: {test_families}, Expected: {{{target_family}}}"
    )
    
    # 3. 训练集与验证集家族隔离硬断言 (绝不含目标家族)
    train_families = set(train_df['anion_family'])
    val_families   = set(val_df['anion_family'])
    assert target_family not in train_families, (
        f"[{split_name}] FATAL: Target family {target_family} LEAKED into train set!"
    )
    assert target_family not in val_families, (
        f"[{split_name}] FATAL: Target family {target_family} LEAKED into val set!"
    )
    
    # 4. 阴离子物种零交集硬断言 (Anion Species Disjoint)
    train_anions = set(train_df['anion_clean'])
    val_anions   = set(val_df['anion_clean'])
    test_anions  = set(test_df['anion_clean'])
    assert train_anions.isdisjoint(test_anions), (
        f"[{split_name}] FATAL: Anion species overlap between Train and Test! Overlap: {train_anions & test_anions}"
    )
    assert val_anions.isdisjoint(test_anions), (
        f"[{split_name}] FATAL: Anion species overlap between Val and Test! Overlap: {val_anions & test_anions}"
    )
    
    # 5. 零实验状态点泄漏硬断言 (State-Condition Exact Duplicate Disjoint)
    # 五元组：阳离子 + 阴离子 + 制冷剂 + 温度 + 压力
    pt_cols = ['cation_clean', 'anion_clean', 'refrigerant_clean', 'T_K', 'P_MPa']
    train_pts = set(tuple(x) for x in train_df[pt_cols].values)
    val_pts   = set(tuple(x) for x in val_df[pt_cols].values)
    test_pts  = set(tuple(x) for x in test_df[pt_cols].values)
    assert train_pts.isdisjoint(test_pts), f"[{split_name}] FATAL: Exact thermodynamic state points leak into Test!"
    assert val_pts.isdisjoint(test_pts), f"[{split_name}] FATAL: Exact thermodynamic state points leak between Val and Test!"
    
    # 6. 测试集比例黄金区间断言 (15% ~ 25%)
    test_ratio = len(test_idx) / len(df)
    assert 0.15 <= test_ratio <= 0.25, (
        f"[{split_name}] FATAL: Test ratio {test_ratio:.3f} outside [0.15, 0.25] golden corridor!"
    )
    
    print(f"  [Pass] All 6 strict assertion suites PASSED for {split_name} [OK]")
    print(f"         Test Ratio = {test_ratio*100:.1f}% ({len(test_idx)}/{len(df)})")
    print(f"         Test Anions ({len(test_anions)}): {sorted(list(test_anions))}")
    print(f"         Train Anions ({len(train_anions)}): {sorted(list(train_anions))}")


# ──────────────────────────────────────────────────────────
# 3. 覆盖度与诊断报告生成器
# ──────────────────────────────────────────────────────────
def compute_coverage_report(df, train_idx, val_idx, test_idx, split_name, target_family):
    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]
    test_df  = df.iloc[test_idx]
    
    report = {
        "split_name": split_name,
        "target_held_out_family": target_family,
        "n_total": len(df),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "test_fraction": len(test_idx) / len(df),
        "unique_cations": {
            "train": int(train_df['cation'].nunique()),
            "val": int(val_df['cation'].nunique()),
            "test": int(test_df['cation'].nunique()),
            "test_overlap_with_train": len(set(test_df['cation']) & set(train_df['cation'])),
            "test_novel_cations": len(set(test_df['cation']) - set(train_df['cation'])),
        },
        "unique_refrigerants": {
            "train": int(train_df['refrigerant'].nunique()),
            "val": int(val_df['refrigerant'].nunique()),
            "test": int(test_df['refrigerant'].nunique()),
            "test_overlap_with_train": len(set(test_df['refrigerant']) & set(train_df['refrigerant'])),
            "test_novel_refrigerants": len(set(test_df['refrigerant']) - set(train_df['refrigerant'])),
            "test_refrigerants_list": sorted(test_df['refrigerant'].unique().tolist()),
        },
        "unique_anions": {
            "train": sorted(train_df['anion_clean'].unique().tolist()),
            "val": sorted(val_df['anion_clean'].unique().tolist()),
            "test": sorted(test_df['anion_clean'].unique().tolist()),
            "overlap": len(set(train_df['anion_clean']) & set(test_df['anion_clean']))
        },
        "state_variables": {
            "T_K": {
                "train_min": float(train_df['T_K'].min()),
                "train_max": float(train_df['T_K'].max()),
                "test_min": float(test_df['T_K'].min()),
                "test_max": float(test_df['T_K'].max()),
            },
            "P_MPa": {
                "train_min": float(train_df['P_MPa'].min()),
                "train_max": float(train_df['P_MPa'].max()),
                "test_min": float(test_df['P_MPa'].min()),
                "test_max": float(test_df['P_MPa'].max()),
            },
            "x1": {
                "train_mean": float(train_df['x1'].mean()),
                "train_std": float(train_df['x1'].std()),
                "test_mean": float(test_df['x1'].mean()),
                "test_std": float(test_df['x1'].std()),
            }
        }
    }
    return report


# ──────────────────────────────────────────────────────────
# 4. 主执行流程
# ──────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Split B Feasibility & Leakage Audit Engine (Protocol 0)")
    print("=" * 70)
    
    # 1. 加载 4444 全量索引表
    raw_csv = 'index_with_anion.csv'
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Missing {raw_csv}!")
    df_raw = pd.read_csv(raw_csv)
    print(f"  [Data] Loaded {raw_csv}: {len(df_raw)} total rows")
    
    # 2. 映射化学先验家族
    tax_df = df_raw['anion'].apply(assign_taxonomy)
    df_raw = pd.concat([df_raw, tax_df], axis=1)
    df_raw['anion_clean'] = df_raw['anion'].apply(clean_name)
    df_raw['cation_clean'] = df_raw['cation'].apply(clean_name)
    df_raw['refrigerant_clean'] = df_raw['refrigerant'].apply(clean_name)
    
    # 3. 锁定基底宇宙 (HFC Universe: Table S3, N=2739)
    # 单一变量原则：冷媒保持为纯饱和 HFC，无冷媒家族漂移混杂，纯粹考核阴离子家族 OOD！
    df_hfc = df_raw[df_raw['sheet'] == 'Table S3. VLE HFCs'].copy().reset_index(drop=True)
    print(f"\n  [Universe] Saturated HFC Universe: {len(df_hfc)} rows (Strictly control refrigerant family shift)")

    # 统计阴离子在 2739 基底宇宙与 4444 全量库中的精确分布 (消除统计口径混淆)
    hfc_counts = df_hfc.groupby('anion_clean').size().rename('sample_count_hfc2739')
    all_counts = df_raw.groupby('anion_clean').size().rename('sample_count_total4444')
    
    out_dir = Path('splits_anion_ood')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tax_summary = df_raw.groupby(['anion_family', 'anion_clean', 'charged_core_motif']).size().reset_index(name='_drop')
    tax_summary = tax_summary.drop(columns=['_drop'])
    tax_summary['sample_count_hfc2739'] = tax_summary['anion_clean'].map(hfc_counts).fillna(0).astype(int)
    tax_summary['sample_count_total4444'] = tax_summary['anion_clean'].map(all_counts).fillna(0).astype(int)
    tax_summary.to_csv(out_dir / 'split_B_family_assignment.csv', index=False, encoding='utf-8-sig')
    print(f"  [Taxonomy] Saved a priori taxonomy assignment: {out_dir / 'split_B_family_assignment.csv'}")
    
    splits_config = [
        ("B1_Fam2_Fluorosulfonate", "Fam-2_Fluorosulfonate"),
        ("B2_Fam3_InorganicFluoride", "Fam-3_InorganicFluoride"),
    ]
    
    full_audit_reports = {}
    leakage_log_lines = [
        "Split B Formal Leakage & Feasibility Audit Log",
        "=" * 60,
        f"Base Universe: Table S3. VLE HFCs (N={len(df_hfc)})",
        "Scientific Objective: Orthogonal Solvent-side (Anion-Family) OOD Benchmarking",
        ""
    ]
    
    coverage_rows = []
    
    for split_tag, target_family in splits_config:
        print(f"\n{'─'*60}")
        print(f"  Building Split: {split_tag} (Target Family: {target_family})")
        print(f"{'─'*60}")
        
        test_mask = (df_hfc['anion_family'] == target_family)
        test_idx = df_hfc[test_mask].index.to_numpy()
        
        # 训练+验证集候选池
        train_val_idx = df_hfc[~test_mask].index.to_numpy()
        
        # 验证集切分策略：从非目标家族中，采用分层不泄漏点抽样 10%
        # 种子严格锁定 42
        rng = np.random.default_rng(42)
        n_val = max(1, round(len(train_val_idx) * 0.10))
        
        shuffled = train_val_idx.copy()
        rng.shuffle(shuffled)
        val_idx = np.sort(shuffled[:n_val])
        train_idx = np.sort(shuffled[n_val:])
        
        # 运行硬断言（任何违规立即抛出 AssertionError 终止）
        run_strict_assertions(df_hfc, train_idx, val_idx, test_idx, target_family, split_tag)
        
        # 保存 .npz 文件
        npz_file = out_dir / f"split_{split_tag}.npz"
        np.savez_compressed(
            npz_file,
            train=train_idx,
            val=val_idx,
            test=test_idx,
            target_family=target_family,
            universe="HFC_Table_S3",
            n_total=len(df_hfc)
        )
        print(f"  [Save] Split indices written to: {npz_file}")
        
        # 计算审计与覆盖度报告
        rep = compute_coverage_report(df_hfc, train_idx, val_idx, test_idx, split_tag, target_family)
        full_audit_reports[split_tag] = rep
        
        leakage_log_lines.append(f"[{split_tag}] Audit Status: PASSED (Verified zero family/species/condition leakage)")
        leakage_log_lines.append(f"  Target Family: {target_family}")
        leakage_log_lines.append(f"  Test Samples: {len(test_idx)} ({len(test_idx)/len(df_hfc)*100:.2f}%)")
        leakage_log_lines.append(f"  Train Samples: {len(train_idx)}, Val Samples: {len(val_idx)}")
        leakage_log_lines.append(f"  Test Anions ({len(rep['unique_anions']['test'])}): {rep['unique_anions']['test']}")
        leakage_log_lines.append(f"  Train Anions ({len(rep['unique_anions']['train'])}): {rep['unique_anions']['train']}")
        leakage_log_lines.append(f"  Cation Overlap: {rep['unique_cations']['test_overlap_with_train']}/{rep['unique_cations']['test']}")
        leakage_log_lines.append(f"  Refrigerant Overlap: {rep['unique_refrigerants']['test_overlap_with_train']}/{rep['unique_refrigerants']['test']}")
        leakage_log_lines.append("")
        
        coverage_rows.append({
            "split": split_tag,
            "target_family": target_family,
            "N_total": len(df_hfc),
            "N_train": len(train_idx),
            "N_val": len(val_idx),
            "N_test": len(test_idx),
            "test_pct": f"{len(test_idx)/len(df_hfc)*100:.2f}%",
            "test_anions_count": len(rep['unique_anions']['test']),
            "train_anions_count": len(rep['unique_anions']['train']),
            "test_cations_count": rep['unique_cations']['test'],
            "test_refrigerants_count": rep['unique_refrigerants']['test'],
            "T_span_test": f"{rep['state_variables']['T_K']['test_min']:.1f} ~ {rep['state_variables']['T_K']['test_max']:.1f} K",
            "P_span_test": f"{rep['state_variables']['P_MPa']['test_min']:.3f} ~ {rep['state_variables']['P_MPa']['test_max']:.3f} MPa",
        })
        
        # 交叉热力表 (Cation x Refrigerant in Test Set)
        test_df = df_hfc.iloc[test_idx]
        cross_tab = pd.crosstab(test_df['cation'], test_df['refrigerant'])
        cross_tab.to_csv(out_dir / f"coverage_matrix_{split_tag}_cation_x_refrigerant.csv")
        print(f"  [Matrix] Saved Cation x Refrigerant matrix: {out_dir / f'coverage_matrix_{split_tag}_cation_x_refrigerant.csv'}")

    # 4. 汇总持久化
    with open(out_dir / 'split_B_audit_report.json', 'w', encoding='utf-8') as f:
        json.dump(full_audit_reports, f, indent=2, ensure_ascii=False)
        
    with open(out_dir / 'split_B_leakage_check.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(leakage_log_lines))
        
    pd.DataFrame(coverage_rows).to_csv(out_dir / 'split_B_coverage_report.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 70)
    print("  [SUCCESS] All Split B Gatekeeper Audits Successfully Completed!")
    print(f"  Artifacts saved in directory: {out_dir.resolve()}")
    print("=" * 70)

if __name__ == '__main__':
    main()
