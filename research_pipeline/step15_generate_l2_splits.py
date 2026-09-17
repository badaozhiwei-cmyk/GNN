"""
step15_generate_l2_splits.py — Phase II-D: L2 Ion-Pair Recombination Split Generator
=================================================================================
【科学使命】
  严格构建并审计化学组分泛化（Compositional Generalization / Ion-Pair Recombination）的 L2 切分。
  考察“组分在训练集中均出现，但特定阴阳离子配对从未出现”的泛化能力：
    c_test in C_train, a_test in A_train, (c, a)_test not in Train

【基底宇宙】
  Saturated HFC Universe (Table S3, N=2739, 12 种饱和 HFC 冷媒, 23 种阴离子, 50 种 IL 对)

【切分架构】
  1. 主实验：L2-Controlled-Composite (方案 1: 3x2 阶乘均衡重组组合, N=374, 13.65%)
     测试对: [EMIM][BF4] (93), [BMIM][OTF] (50), [EMIM][OTF] (114), [HMIM][BF4] (117)
     覆盖阳离子: EMIM, BMIM, HMIM (训练集中与 TF2N, PF6, CL 等大量配对)
     覆盖阴离子: BF4, OTF (训练集中与 BMIM, HMIM 分别配对: [BMIM][BF4], [HMIM][OTF])
     覆盖冷媒: 9 种 HFC
  2. 附录参考：L2-Star-Anchor (单对高密度重组, [HMIM][TF2N], N=416, 15.19%)

【产物】
  - splits/L2_controlled_composite.npz
  - splits/L2_star_anchor.npz
  - splits/L2_pair_assignment.csv
  - paper_results/l2_split_audit.json
  - paper_results/table_l2_test_coverage.csv
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 设定项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

META_CSV = PROJECT_ROOT / "processed_tri_data_hfc2739" / "meta_info.csv"
SPLITS_DIR = PROJECT_ROOT / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_RES_DIR = PROJECT_ROOT / "paper_results"
PAPER_RES_DIR.mkdir(parents=True, exist_ok=True)

# 统一清洗函数
clean_str = lambda s: str(s).strip()
clean_ion = lambda s: str(s).strip().replace("[", "").replace("]", "").upper()

def run_strict_l2_assertions(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    target_pairs: list[str],
    split_name: str
):
    """
    运行严格的 L2 组分泛化门禁硬断言。
    任何一条不满足直接抛出 AssertionError 阻断执行。
    """
    print(f"\n[Assert] Running strict L2 gatekeeper assertions for {split_name}...")
    train_set = set(train_idx)
    val_set   = set(val_idx)
    test_set  = set(test_idx)

    # 1. 索引无重叠且覆盖完全
    assert len(train_set & val_set) == 0, f"[{split_name}] FATAL: Train and Val index overlap!"
    assert len(train_set & test_set) == 0, f"[{split_name}] FATAL: Train and Test index overlap!"
    assert len(val_set & test_set) == 0, f"[{split_name}] FATAL: Val and Test index overlap!"
    assert len(train_set | val_set | test_set) == len(df), f"[{split_name}] FATAL: Incomplete index coverage!"

    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]
    test_df  = df.iloc[test_idx]

    # 2. 测试集离子对纯度硬断言 (100% 属于目标 pairs)
    test_actual_pairs = set(test_df['pair'])
    expected_pairs = set(target_pairs)
    assert test_actual_pairs == expected_pairs, (
        f"[{split_name}] FATAL: Test set pairs mismatch! Found: {test_actual_pairs}, Expected: {expected_pairs}"
    )

    # 3. 训练集与验证集离子对隔离硬断言 (绝不含目标 pairs)
    train_pairs = set(train_df['pair'])
    val_pairs   = set(val_df['pair'])
    assert train_pairs.isdisjoint(expected_pairs), (
        f"[{split_name}] FATAL: Target pairs {train_pairs & expected_pairs} LEAKED into train set!"
    )
    assert val_pairs.isdisjoint(expected_pairs), (
        f"[{split_name}] FATAL: Target pairs {val_pairs & expected_pairs} LEAKED into val set!"
    )

    # 4. 组分存在性断言 (Compositional Generalization 铁律: seen components)
    # c_test in C_train, a_test in A_train, ref_test in Ref_train
    test_cats = set(test_df['c'])
    train_cats = set(train_df['c'])
    assert test_cats.issubset(train_cats), (
        f"[{split_name}] FATAL: Novel cation in test! {test_cats - train_cats} not in train!"
    )

    test_anis = set(test_df['a'])
    train_anis = set(train_df['a'])
    assert test_anis.issubset(train_anis), (
        f"[{split_name}] FATAL: Novel anion in test! {test_anis - train_anis} not in train!"
    )

    test_refs = set(test_df['ref'])
    train_refs = set(train_df['ref'])
    assert test_refs.issubset(train_refs), (
        f"[{split_name}] FATAL: Novel refrigerant in test! {test_refs - train_refs} not in train!"
    )

    # 5. 训练集中组件独立度断言 (每个测试阳离子/阴离子在训练集中必须有 >= 1 个其他伙伴)
    cat_train_deg = train_df.groupby('c')['a'].nunique()
    ani_train_deg = train_df.groupby('a')['c'].nunique()
    for c in test_cats:
        assert cat_train_deg.get(c, 0) >= 1, f"[{split_name}] FATAL: Cation {c} has 0 partners in train!"
    for a in test_anis:
        assert ani_train_deg.get(a, 0) >= 1, f"[{split_name}] FATAL: Anion {a} has 0 partners in train!"

    # 6. 零实验状态点泄漏断言 (五元组: c, a, ref, T, P)
    pt_cols = ['c', 'a', 'ref', 'T_K', 'P_MPa']
    train_pts = set(tuple(x) for x in train_df[pt_cols].values)
    val_pts   = set(tuple(x) for x in val_df[pt_cols].values)
    test_pts  = set(tuple(x) for x in test_df[pt_cols].values)
    assert train_pts.isdisjoint(test_pts), f"[{split_name}] FATAL: Thermodynamic state duplicate in test!"
    assert val_pts.isdisjoint(test_pts), f"[{split_name}] FATAL: Thermodynamic state duplicate in val!"

    # 7. 测试集比例黄金区间 (10% ~ 20%)
    test_ratio = len(test_idx) / len(df)
    assert 0.10 <= test_ratio <= 0.20, (
        f"[{split_name}] FATAL: Test ratio {test_ratio:.3f} outside [0.10, 0.20] golden corridor!"
    )

    print(f"  [Pass] All 7 strict assertions PASSED for {split_name} [OK]")
    print(f"         Test Ratio = {test_ratio*100:.2f}% ({len(test_idx)}/{len(df)})")
    print(f"         Test Cations ({len(test_cats)}): {sorted(list(test_cats))}")
    print(f"         Test Anions ({len(test_anis)}): {sorted(list(test_anis))}")
    print(f"         Test Refrigerants ({len(test_refs)}): {sorted(list(test_refs))}")


def generate_splits():
    print("=" * 75)
    print("  PHASE II-D STEP 15: GENERATE L2 ION-PAIR RECOMBINATION SPLITS")
    print("=" * 75)

    if not META_CSV.exists():
        raise FileNotFoundError(f"Missing meta file: {META_CSV}")

    df = pd.read_csv(META_CSV)
    print(f"[数据加载] {META_CSV.name}: 总计 {len(df)} 条样本")

    df['c'] = df['cation'].apply(clean_ion)
    df['a'] = df['anion'].apply(clean_ion)
    df['ref'] = df['refrigerant'].apply(clean_str).str.upper()
    df['pair'] = df['c'] + "__" + df['a']
    df['chem_system'] = df['pair'] + "__" + df['ref']

    # -------------------------------------------------------------------------
    # 1. 主切分: L2_controlled_composite (方案 1)
    # -------------------------------------------------------------------------
    target_pairs_composite = ["EMIM__BF4", "BMIM__OTF", "EMIM__OTF", "HMIM__BF4"]
    split_tag_comp = "L2_controlled_composite"
    
    print(f"\n--- 构建主切分: {split_tag_comp} ---")
    print(f"目标重组对 ({len(target_pairs_composite)}): {target_pairs_composite}")

    test_mask_comp = df['pair'].isin(target_pairs_composite)
    test_idx_comp = df[test_mask_comp].index.to_numpy()
    train_val_idx_comp = df[~test_mask_comp].index.to_numpy()

    # 验证集切分：采用分层抽样 (stratified across chemical systems in train_val)
    # 种子严格锁定 42
    rng = np.random.default_rng(42)
    n_val_target = max(1, round(len(train_val_idx_comp) * 0.10))

    # 按 chem_system 分组抽样，保证 val 覆盖全面且无偏差
    val_indices = []
    shuffled_pool = []
    for cs, grp in df.iloc[train_val_idx_comp].groupby('chem_system'):
        indices = grp.index.to_numpy().copy()
        rng.shuffle(indices)
        # 每个 system 抽取约 10% (至少 1 点如果样本多)
        k = int(np.round(len(indices) * 0.10))
        if k > 0:
            val_indices.extend(indices[:k])
            shuffled_pool.extend(indices[k:])
        else:
            shuffled_pool.extend(indices)

    # 若抽样不足或略微超过，通过全局随机池补充或调整以精确达到 target 10%
    val_set_cur = set(val_indices)
    shuffled_pool = [i for i in train_val_idx_comp if i not in val_set_cur]
    rng.shuffle(shuffled_pool)

    diff = n_val_target - len(val_indices)
    if diff > 0:
        val_indices.extend(shuffled_pool[:diff])
    elif diff < 0:
        val_indices = val_indices[:n_val_target]

    val_idx_comp = np.sort(np.array(val_indices))
    train_idx_comp = np.sort(np.array([i for i in train_val_idx_comp if i not in set(val_idx_comp)]))

    # 运行硬断言
    run_strict_l2_assertions(df, train_idx_comp, val_idx_comp, test_idx_comp, target_pairs_composite, split_tag_comp)

    # 保存 .npz 文件
    npz_comp_file = SPLITS_DIR / f"{split_tag_comp}.npz"
    np.savez_compressed(
        npz_comp_file,
        train=train_idx_comp,
        val=val_idx_comp,
        test=test_idx_comp,
        target_pairs=target_pairs_composite,
        split_name=split_tag_comp,
        universe="HFC_Table_S3",
        n_total=len(df)
    )
    print(f"  [Save] {npz_comp_file.name} saved successfully!")

    # -------------------------------------------------------------------------
    # 2. 补充切分: L2_star_anchor ([HMIM][TF2N])
    # -------------------------------------------------------------------------
    target_pairs_star = ["HMIM__TF2N"]
    split_tag_star = "L2_star_anchor"
    print(f"\n--- 构建补充切分: {split_tag_star} ---")
    test_mask_star = df['pair'].isin(target_pairs_star)
    test_idx_star = df[test_mask_star].index.to_numpy()
    train_val_idx_star = df[~test_mask_star].index.to_numpy()

    # 验证集切分 10%
    rng_star = np.random.default_rng(42)
    n_val_star = max(1, round(len(train_val_idx_star) * 0.10))
    shuffled_star = train_val_idx_star.copy()
    rng_star.shuffle(shuffled_star)
    val_idx_star = np.sort(shuffled_star[:n_val_star])
    train_idx_star = np.sort(shuffled_star[n_val_star:])

    run_strict_l2_assertions(df, train_idx_star, val_idx_star, test_idx_star, target_pairs_star, split_tag_star)

    npz_star_file = SPLITS_DIR / f"{split_tag_star}.npz"
    np.savez_compressed(
        npz_star_file,
        train=train_idx_star,
        val=val_idx_star,
        test=test_idx_star,
        target_pairs=target_pairs_star,
        split_name=split_tag_star,
        universe="HFC_Table_S3",
        n_total=len(df)
    )
    print(f"  [Save] {npz_star_file.name} saved successfully!")

    # -------------------------------------------------------------------------
    # 3. 生成样本级切分归属表: L2_pair_assignment.csv
    # -------------------------------------------------------------------------
    df_assign = df[['sample_id', 'npy_idx', 'cation', 'anion', 'refrigerant', 'T_K', 'P_MPa', 'x1']].copy()
    df_assign['pair'] = df['pair']
    df_assign['split_composite'] = 'train'
    df_assign.loc[val_idx_comp, 'split_composite'] = 'val'
    df_assign.loc[test_idx_comp, 'split_composite'] = 'test'

    df_assign['split_star'] = 'train'
    df_assign.loc[val_idx_star, 'split_star'] = 'val'
    df_assign.loc[test_idx_star, 'split_star'] = 'test'

    assign_csv = SPLITS_DIR / "L2_pair_assignment.csv"
    df_assign.to_csv(assign_csv, index=False)
    print(f"\n  [Save] Assignment CSV saved: {assign_csv.name} ({len(df_assign)} rows)")

    # -------------------------------------------------------------------------
    # 4. 生成测试集覆盖度表: table_l2_test_coverage.csv
    # -------------------------------------------------------------------------
    train_comp_df = df.iloc[train_idx_comp]
    coverage_list = []
    for p in target_pairs_composite:
        sub = df[df['pair'] == p]
        c = sub['c'].iloc[0]
        a = sub['a'].iloc[0]
        refs = sorted(sub['ref'].unique().tolist())
        
        # 训练集中共享该阳离子的阴离子伙伴
        shared_cat_partners = sorted(train_comp_df[train_comp_df['c'] == c]['a'].unique().tolist())
        # 训练集中共享该阴离子的阳离子伙伴
        shared_ani_partners = sorted(train_comp_df[train_comp_df['a'] == a]['c'].unique().tolist())

        coverage_list.append({
            "test_pair": p,
            "cation": c,
            "anion": a,
            "n_samples": len(sub),
            "test_pct_of_universe": f"{len(sub)/len(df)*100:.2f}%",
            "n_refrigerants": len(refs),
            "refrigerants_list": "; ".join(refs),
            "T_min_K": float(sub['T_K'].min()),
            "T_max_K": float(sub['T_K'].max()),
            "P_min_MPa": float(sub['P_MPa'].min()),
            "P_max_MPa": float(sub['P_MPa'].max()),
            "x1_min": float(sub['x1'].min()),
            "x1_max": float(sub['x1'].max()),
            "shared_cation_anion_partners_in_train": "; ".join(shared_cat_partners),
            "shared_anion_cation_partners_in_train": "; ".join(shared_ani_partners),
        })

    df_cov = pd.DataFrame(coverage_list)
    cov_csv = PAPER_RES_DIR / "table_l2_test_coverage.csv"
    df_cov.to_csv(cov_csv, index=False)
    print(f"  [Save] Test coverage table saved: {cov_csv.name}")

    # -------------------------------------------------------------------------
    # 5. 生成完整审计报告 JSON
    # -------------------------------------------------------------------------
    audit_dict = {
        "universe": "Saturated HFC Table S3 (N=2739)",
        "split_primary": {
            "name": split_tag_comp,
            "architecture": "3x2 Factorial Controlled Composite",
            "target_pairs": target_pairs_composite,
            "n_train": len(train_idx_comp),
            "n_val": len(val_idx_comp),
            "n_test": len(test_idx_comp),
            "test_fraction": len(test_idx_comp) / len(df),
            "unique_test_cations": sorted(list(set(df.iloc[test_idx_comp]['c']))),
            "unique_test_anions": sorted(list(set(df.iloc[test_idx_comp]['a']))),
            "unique_test_refrigerants": sorted(list(set(df.iloc[test_idx_comp]['ref']))),
            "gatekeeper_assertions": "PASSED (7/7)",
        },
        "split_star_anchor": {
            "name": split_tag_star,
            "architecture": "Single-Star High-Density Anchor",
            "target_pairs": target_pairs_star,
            "n_train": len(train_idx_star),
            "n_val": len(val_idx_star),
            "n_test": len(test_idx_star),
            "test_fraction": len(test_idx_star) / len(df),
            "gatekeeper_assertions": "PASSED (7/7)",
        }
    }
    audit_json = PAPER_RES_DIR / "l2_split_audit.json"
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(audit_dict, f, indent=2, ensure_ascii=False)
    print(f"  [Save] Audit JSON saved: {audit_json.name}")

    print("\n" + "=" * 75)
    print("  [SUCCESS] PHASE II-D STEP 15 EXECUTION COMPLETE")
    print("=" * 75)

if __name__ == '__main__':
    generate_splits()
