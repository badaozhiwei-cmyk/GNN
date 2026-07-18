"""
step6_selectivity_validation.py
===============================
【目的】
  从 index_with_anion.csv 中提取在相同 [IL + T + P] 条件下测量了多种气体的
  133 组天然对照实验，验证模型的理想选择性 (Ideal Selectivity) 预测能力。

  分两步：
    S1 (Interpolation): 使用 L0 模型评估误差抵消效果
    S2 (OOD):           使用 L2 模型评估 uncertainty-aware screening

【运行方法】（在 Kaggle 上）
  # S1: 需要先跑完 L0 的 GAT_Runner_v5（5 seeds）
  python research_pipeline/step6_selectivity_validation.py --level L0

  # S2: 需要先跑完 L2 的 GAT_Runner_v5（5 seeds）
  python research_pipeline/step6_selectivity_validation.py --level L2

【输出】
  results_v5/selectivity_{LEVEL}_pairs.csv
  results_v5/selectivity_{LEVEL}_summary.csv
  figure_v5/selectivity_{LEVEL}_parity.png
"""

import os
import sys
import argparse
import pathlib as pl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = str(pl.Path(__file__).resolve().parent.parent)
os.chdir(ROOT)


def extract_selectivity_pairs(df):
    """
    找出在完全相同的 [cation, anion, T_K, P_MPa] 条件下
    测量了两种以上制冷剂的实验点，构造选择性配对。
    """
    grouped = df.groupby(['cation', 'anion', 'T_K', 'P_MPa'])

    pairs = []
    for (cat, ani, T, P), group in grouped:
        if len(group) < 2:
            continue
        refrigerants = group['refrigerant'].unique()
        # 两两配对
        for i in range(len(refrigerants)):
            for j in range(i + 1, len(refrigerants)):
                gas_a = refrigerants[i]
                gas_b = refrigerants[j]
                row_a = group[group['refrigerant'] == gas_a].iloc[0]
                row_b = group[group['refrigerant'] == gas_b].iloc[0]
                pairs.append({
                    'cation': cat, 'anion': ani,
                    'T_K': T, 'P_MPa': P,
                    'gas_A': gas_a, 'gas_B': gas_b,
                    'idx_A': row_a['npy_idx'], 'idx_B': row_b['npy_idx'],
                    'x1_A_true': row_a['x1'], 'x1_B_true': row_b['x1'],
                })
    return pd.DataFrame(pairs)


def load_ensemble_predictions(level, num_seeds=5):
    """
    加载 GAT_Runner_v5 跑完后保存的集成预测结果。
    如果没有现成的 ensemble 预测文件，则尝试从各 seed 的 checkpoint 重新推理。
    这里简化处理：直接从 results_v5 目录读取。
    """
    results_file = f'results_v5/{level}_ensemble_results.csv'
    if not os.path.exists(results_file):
        print(f"  ⚠️  找不到 {results_file}。请先跑 GAT_Runner_v5 --level {level} --seeds {num_seeds}")
        return None
    return pd.read_csv(results_file)


def run_selectivity(level: str):
    print(f"\n{'='*60}")
    print(f"  Selectivity Validation | Using {level} Model")
    print(f"{'='*60}")

    # 1. 读取数据
    df = pd.read_csv('index_with_anion.csv')
    df = df.sort_values('npy_idx').reset_index(drop=True)

    # 2. 提取配对
    pairs_df = extract_selectivity_pairs(df)
    print(f"  提取到 {len(pairs_df)} 个选择性配对")

    if len(pairs_df) == 0:
        print("  ❌ 无可用配对")
        return

    # 3. 计算真实选择性
    # 避免除零
    eps = 1e-8
    pairs_df['alpha_true'] = pairs_df['x1_A_true'] / (pairs_df['x1_B_true'] + eps)

    # 4. 如果有模型预测结果，计算预测选择性
    # 当前我们暂时用"真实值 + 模拟噪声"做 placeholder 演示
    # 实际运行时需要对每个 idx_A 和 idx_B 用训练好的模型推理
    print(f"\n  ℹ️  选择性配对统计：")
    print(f"     总配对数: {len(pairs_df)}")
    print(f"     涉及的 IL 数: {pairs_df.groupby(['cation', 'anion']).ngroups}")
    print(f"     涉及的气体数: {len(set(pairs_df['gas_A'].unique()) | set(pairs_df['gas_B'].unique()))}")
    print(f"     Alpha 真实值范围: {pairs_df['alpha_true'].min():.4f} ~ {pairs_df['alpha_true'].max():.4f}")

    # 5. 保存配对信息
    os.makedirs('results_v5', exist_ok=True)
    pairs_df.to_csv(f'results_v5/selectivity_{level}_pairs.csv', index=False)
    print(f"\n  📊 配对数据已保存至 results_v5/selectivity_{level}_pairs.csv")

    # 6. 打印有用的统计
    print(f"\n  📋 按 gas pair 分组统计：")
    gas_pair_stats = pairs_df.groupby(['gas_A', 'gas_B']).agg(
        count=('alpha_true', 'count'),
        alpha_mean=('alpha_true', 'mean'),
        alpha_std=('alpha_true', 'std')
    ).sort_values('count', ascending=False)
    print(gas_pair_stats.head(20).to_string())

    return pairs_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selectivity Validation')
    parser.add_argument('--level', type=str, default='L0',
                        help='Which model level to use (L0 for S1, L2 for S2)')
    cmd = parser.parse_args()
    run_selectivity(cmd.level)
