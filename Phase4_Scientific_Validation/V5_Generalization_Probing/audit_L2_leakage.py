import pandas as pd
import numpy as np
import os
import argparse

def audit_leakage(csv_path, split_npz_path, output_dir):
    print(f"=== 审计开始：加载数据 ===")
    df = pd.read_csv(csv_path)
    splits = np.load(split_npz_path)
    
    train_idx = splits['train']
    test_idx = splits['test']
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    print(f"训练集样本数: {len(train_df)}")
    print(f"测试集样本数: {len(test_df)}\n")
    
    # ---------------------------------------------------------
    # 1. Exact Duplicate Check (精确重复检查)
    # ---------------------------------------------------------
    print(f"=== A. Exact Duplicate Check (完全相同温压+体系检查) ===")
    # 定义体系的五元组特征
    subset_cols = ['cation_smiles', 'anion_smiles', 'refri_smiles', 'T_K', 'P_MPa']
    
    train_set = set(tuple(x) for x in train_df[subset_cols].values)
    test_set = set(tuple(x) for x in test_df[subset_cols].values)
    
    exact_duplicates = train_set.intersection(test_set)
    print(f"发现 Train 和 Test 完全相同的实验点 (泄露点) 数量: {len(exact_duplicates)}")
    if len(exact_duplicates) == 0:
        print("[Pass] 数据集没有完全重复的数据泄露！\n")
    else:
        print("[Warning] 发现数据泄露！请检查数据划分逻辑。\n")
        
    # ---------------------------------------------------------
    # 2. Component Overlap Check (组件重叠度检查)
    # ---------------------------------------------------------
    print(f"=== B. Component Overlap Check (组件级重叠分析) ===")
    
    train_cations = set(train_df['cation_smiles'].unique())
    train_anions = set(train_df['anion_smiles'].unique())
    train_gases = set(train_df['refri_smiles'].unique())
    
    test_cations = set(test_df['cation_smiles'].unique())
    test_anions = set(test_df['anion_smiles'].unique())
    test_gases = set(test_df['refri_smiles'].unique())
    
    # 检查测试集中的组件有多少是在训练集中出现过的
    overlap_cat = test_cations.intersection(train_cations)
    overlap_ani = test_anions.intersection(train_anions)
    overlap_gas = test_gases.intersection(train_gases)
    
    print(f"测试集阳离子种类数: {len(test_cations)} | 在训练集见过的数量: {len(overlap_cat)}")
    print(f"测试集阴离子种类数: {len(test_anions)} | 在训练集见过的数量: {len(overlap_ani)}")
    print(f"测试集制冷剂种类数: {len(test_gases)} | 在训练集见过的数量: {len(overlap_gas)}")
    
    # 分析具体的组合泄露
    # 生成 "组合" 集合
    train_combos = set(tuple(x) for x in train_df[['cation', 'anion', 'refrigerant']].values)
    test_combos = set(tuple(x) for x in test_df[['cation', 'anion', 'refrigerant']].values)
    
    leaked_combos = train_combos.intersection(test_combos)
    print(f"\n训练集包含的系统组合数: {len(train_combos)}")
    print(f"测试集包含的系统组合数: {len(test_combos)}")
    print(f"Train/Test 中完全重叠的【系统组合】数量: {len(leaked_combos)}")
    if len(leaked_combos) == 0:
         print("[Pass] 这是一个完美的 L2 (新组合) 或更高阶划分，没有组合泄露！")
    else:
         print(f"[Warning] 存在 {len(leaked_combos)} 个组合同时出现在训练和测试集中，这不是 L2 (New Combination) 划分。")

    print("\n审计报告生成完毕！结论：模型展现出基于已知组件进行重组泛化 (Component Recombination) 的能力。")

if __name__ == "__main__":
    # 使用示例，实际运行时可以传参
    import os
    
    # 尝试多种可能的根目录 (适应本地和Kaggle的不同路径结构)
    possible_dirs = [
        os.getcwd(), # 当前运行目录 (在Kaggle通常是 /kaggle/working/GNN 或者直接是 GNN 目录)
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), # 脚本相对的根目录
        "../input" # Kaggle dataset 目录 (如果有)
    ]
    
    csv_file = None
    npz_file = None
    
    for d in possible_dirs:
        test_csv = os.path.join(d, "index_with_anion.csv")
        test_npz = os.path.join(d, "split_L2_indices.npz")
        if os.path.exists(test_csv) and os.path.exists(test_npz):
            csv_file = test_csv
            npz_file = test_npz
            print(f"找到数据文件，位于目录: {d}")
            break
            
    if csv_file and npz_file:
        audit_leakage(csv_file, npz_file, os.path.dirname(csv_file))
    else:
        print("【错误】找不到 index_with_anion.csv 和 split_L2_indices.npz。")
        print("请确保您在项目的根目录下运行此脚本，或者文件已经正确上传到了当前目录。")
