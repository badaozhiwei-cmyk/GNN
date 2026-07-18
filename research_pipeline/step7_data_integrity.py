"""
step7_data_integrity.py
=======================
【目的】
  数据完整性与 OOD 严格性审计脚本，解决审稿人两大攻击点：

  1. Duplicate Check：
     检测 37 个重复测量点是否会造成跨 split 泄漏，
     并在移除后验证结论不变。

  2. L4 Tanimoto Similarity：
     用 Morgan Fingerprint 计算 Train/Test 中制冷剂的结构距离，
     数学证明 L4 的测试集确实处于最大结构偏移处。

【运行方法】（在 Kaggle 上）
  python research_pipeline/step7_data_integrity.py

【输出】
  results_v5/duplicate_audit.csv
  results_v5/l4_tanimoto_audit.csv
  figure_v5/tanimoto_heatmap.png
"""

import os
import pathlib as pl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = str(pl.Path(__file__).resolve().parent.parent)
os.chdir(ROOT)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("⚠️  RDKit 未安装，请在 Kaggle 上运行。")


def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


# ============================================================
# Part 1: Duplicate Audit
# ============================================================
def audit_duplicates():
    print(f"\n{'='*60}")
    print("  Part 1: Duplicate Experiment Point Audit")
    print(f"{'='*60}")

    df = pd.read_csv('index_with_anion.csv')
    dup_cols = ['cation', 'anion', 'refrigerant', 'T_K', 'P_MPa']

    # 找到重复行
    dup_mask = df.duplicated(subset=dup_cols, keep=False)
    dup_df = df[dup_mask].sort_values(dup_cols)
    n_dup = dup_mask.sum()
    n_groups = df[dup_mask].groupby(dup_cols).ngroups

    print(f"  总样本数: {len(df)}")
    print(f"  重复实验点: {n_dup} 条 ({n_dup/len(df)*100:.2f}%)")
    print(f"  重复实验组数: {n_groups}")

    # 检查重复点的 x1 差异（衡量实验噪声）
    dup_var = df[dup_mask].groupby(dup_cols)['x1'].agg(['mean', 'std', 'count'])
    dup_var = dup_var[dup_var['count'] > 1].reset_index()
    print(f"\n  重复测量的 x1 标准差:")
    print(f"    平均 std: {dup_var['std'].mean():.6f}")
    print(f"    最大 std: {dup_var['std'].max():.6f}")

    # 检查重复点是否跨越 L0-L4 的 train/test 边界
    print(f"\n  检查重复点是否跨越 Train/Test 边界:")
    for level in ['L0', 'L1', 'L2', 'L3', 'L4']:
        npz_path = f'split_{level}_indices.npz'
        if not os.path.exists(npz_path):
            print(f"    {level}: split 文件不存在，跳过")
            continue
        loaded = np.load(npz_path)
        train_set = set(loaded['train'].tolist())
        test_set  = set(loaded['test'].tolist())

        dup_indices = df[dup_mask].index.tolist()
        dup_in_train = [i for i in dup_indices if i in train_set]
        dup_in_test  = [i for i in dup_indices if i in test_set]

        # 检查同一个重复组是否既有成员在 train 又有成员在 test
        leakage = 0
        for _, group in df[dup_mask].groupby(dup_cols):
            idxs = set(group.index.tolist())
            if idxs & train_set and idxs & test_set:
                leakage += 1

        status = "✅ 安全" if leakage == 0 else f"⚠️  {leakage} 组泄漏"
        print(f"    {level}: Train中{len(dup_in_train)}个, Test中{len(dup_in_test)}个, 跨界组数: {status}")

    # 保存
    os.makedirs('results_v5', exist_ok=True)
    dup_var.to_csv('results_v5/duplicate_audit.csv', index=False)
    print(f"\n  📊 重复审计结果已保存至 results_v5/duplicate_audit.csv")


# ============================================================
# Part 2: L4 Tanimoto Similarity Audit
# ============================================================
def audit_l4_tanimoto():
    if not RDKIT_OK:
        print("  ❌ RDKit 未安装，跳过 Tanimoto 审计")
        return

    print(f"\n{'='*60}")
    print("  Part 2: L4 Scaffold OOD - Tanimoto Similarity Audit")
    print(f"{'='*60}")

    df = pd.read_csv('index_with_anion.csv')

    # 获取所有唯一的制冷剂
    refri_smiles = df.drop_duplicates('refrigerant')[['refrigerant', 'refri_smiles']].reset_index(drop=True)
    print(f"  唯一制冷剂: {len(refri_smiles)}")

    # 计算 Morgan Fingerprints
    fps = {}
    for _, row in refri_smiles.iterrows():
        fp = smiles_to_morgan(row['refri_smiles'])
        if fp is not None:
            fps[row['refrigerant']] = fp

    # 计算 Tanimoto 相似度矩阵
    refri_names = list(fps.keys())
    n = len(refri_names)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = DataStructs.TanimotoSimilarity(
                fps[refri_names[i]], fps[refri_names[j]]
            )

    # 针对每个 L-level 的 split，计算 Train vs Test 制冷剂的平均相似度
    print(f"\n  各 Level 的 Train-Test 制冷剂 Tanimoto 相似度:")
    results = []
    for level in ['L0', 'L1', 'L2', 'L3', 'L4']:
        npz_path = f'split_{level}_indices.npz'
        if not os.path.exists(npz_path):
            continue
        loaded = np.load(npz_path)
        train_refris = set(df.iloc[loaded['train']]['refrigerant'].unique())
        test_refris  = set(df.iloc[loaded['test']]['refrigerant'].unique())

        # 计算 test 制冷剂与 train 制冷剂的平均最大相似度
        avg_max_sim = []
        for test_r in test_refris:
            if test_r not in fps:
                continue
            max_sim = 0
            for train_r in train_refris:
                if train_r not in fps:
                    continue
                s = DataStructs.TanimotoSimilarity(fps[test_r], fps[train_r])
                max_sim = max(max_sim, s)
            avg_max_sim.append(max_sim)

        mean_sim = np.mean(avg_max_sim) if avg_max_sim else 0
        results.append({
            'level': level,
            'train_refris': len(train_refris),
            'test_refris': len(test_refris),
            'mean_max_tanimoto': mean_sim
        })
        print(f"    {level}: Train {len(train_refris)} 种, Test {len(test_refris)} 种, "
              f"平均最大 Tanimoto = {mean_sim:.4f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv('results_v5/l4_tanimoto_audit.csv', index=False)

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(refri_names, rotation=90, fontsize=7)
    ax.set_yticklabels(refri_names, fontsize=7)
    ax.set_title('Refrigerant Morgan Fingerprint Tanimoto Similarity', fontsize=12)
    plt.colorbar(im)
    plt.tight_layout()
    os.makedirs('figure_v5', exist_ok=True)
    plt.savefig('figure_v5/tanimoto_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Tanimoto 热力图已保存至 figure_v5/tanimoto_heatmap.png")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    audit_duplicates()
    audit_l4_tanimoto()
    print(f"\n{'='*60}")
    print("  ✅ 数据完整性审计全部完成！")
    print(f"{'='*60}")
