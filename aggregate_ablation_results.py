"""
aggregate_ablation_results.py — 汇总并对比 4 个消融模型结果
=========================================================
【作用】
  读取 results_ablation/ 下的 4 个模式结果：
    - HFC_loro_M0/summary.csv
    - HFC_loro_Msize/summary.csv
    - HFC_loro_Mmu/summary.csv
    - HFC_loro_Mphys/summary.csv

  自动生成：
    1. ablation_summary_table.csv (横向对比大宽表，包含每个制冷剂在 M0/Msize/Mmu/Mphys 的 R2 和 MAE)
    2. 控制台打印 LaTeX / Markdown 格式的论文对比表格
    3. 自动检验核心假设：
       - Mmu vs M0 (偶极矩是否有提升)
       - Mmu vs Msize (偶极矩是否优于纯尺寸)
       - Mphys vs Mmu (完整物理量是否有额外收益)
"""

import os
import pandas as pd
import numpy as np

MODES = ['M0', 'Msize', 'Mmu', 'Mphys']
FAMILY = 'HFC'
SPLIT_MODE = 'loro'
BASE_DIR = 'results_ablation'

def load_results():
    data = {}
    for mode in MODES:
        path = os.path.join(BASE_DIR, f"{FAMILY}_{SPLIT_MODE}_{mode}", "summary.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            data[mode] = df
            print(f"✅ 加载成功: {path} ({len(df)} 个制冷剂)")
        else:
            print(f"⚠️ 未找到: {path} (请先运行该模式)")
    return data

def main():
    print("="*70)
    print("  GNN 消融实验 (HFC LORO) 结果汇总分析")
    print("="*70)
    
    results = load_results()
    if not results:
        print("❌ 未找到任何消融实验结果，请先在 Kaggle 上运行实验！")
        return
    
    # 提取所有出现的制冷剂 Target
    all_targets = set()
    for df in results.values():
        all_targets.update(df['Target'].tolist())
    all_targets = sorted(list(all_targets))
    
    # 构建汇总表
    rows = []
    for target in all_targets:
        ref_name = target.replace('loro_', '')
        row = {'Refrigerant': ref_name}
        for mode in MODES:
            if mode in results:
                df = results[mode]
                sub = df[df['Target'] == target]
                if not sub.empty:
                    row[f'{mode}_R2'] = sub['R2_mean'].values[0]
                    row[f'{mode}_R2_std'] = sub['R2_std'].values[0]
                    row[f'{mode}_MAE'] = sub['MAE_mean'].values[0]
                else:
                    row[f'{mode}_R2'] = np.nan
                    row[f'{mode}_R2_std'] = np.nan
                    row[f'{mode}_MAE'] = np.nan
        rows.append(row)
    
    comp_df = pd.DataFrame(rows)
    
    # 添加整体平均行
    mean_row = {'Refrigerant': 'OVERALL_MEAN'}
    for mode in MODES:
        if f'{mode}_R2' in comp_df.columns:
            mean_row[f'{mode}_R2'] = comp_df[f'{mode}_R2'].mean()
            mean_row[f'{mode}_MAE'] = comp_df[f'{mode}_MAE'].mean()
    comp_df = pd.concat([comp_df, pd.DataFrame([mean_row])], ignore_index=True)
    
    # 保存大宽表
    out_csv = os.path.join(BASE_DIR, "ablation_summary_table.csv")
    os.makedirs(BASE_DIR, exist_ok=True)
    comp_df.to_csv(out_csv, index=False)
    print(f"\n📁 详细对比宽表已保存至: {out_csv}")
    
    # 打印精美 Markdown 对比表格
    print("\n" + "="*70)
    print("📊 论文级别消融对比表格 (R² 均值对比)")
    print("="*70)
    
    r2_cols = ['Refrigerant'] + [f'{m}_R2' for m in MODES if f'{m}_R2' in comp_df.columns]
    print(comp_df[r2_cols].to_markdown(index=False, floatfmt=".4f"))
    
    print("\n" + "="*70)
    print("📊 论文级别消融对比表格 (MAE 均值对比)")
    print("="*70)
    mae_cols = ['Refrigerant'] + [f'{m}_MAE' for m in MODES if f'{m}_MAE' in comp_df.columns]
    print(comp_df[mae_cols].to_markdown(index=False, floatfmt=".4f"))
    
    # 核心假设检验
    print("\n" + "="*70)
    print("🧪 科学假设检验与讨论诊断")
    print("="*70)
    if 'M0_R2' in comp_df.columns and 'Mmu_R2' in comp_df.columns:
        m0_mean = comp_df.loc[comp_df['Refrigerant'] == 'OVERALL_MEAN', 'M0_R2'].values[0]
        mmu_mean = comp_df.loc[comp_df['Refrigerant'] == 'OVERALL_MEAN', 'Mmu_R2'].values[0]
        diff = mmu_mean - m0_mean
        print(f"1. [核心假设 Mμ vs M0]: 均值 R² 变化 = {diff:+.4f}")
        if diff > 0:
            print("   -> ✅ 支持假设：3D 偶极矩为 GNN 提供了有效信息增量，提升了外推泛化！")
        else:
            print("   -> ⚠️ 偶极矩未带来显著整体提升，需分制冷剂深入排查。")

    if 'Msize_R2' in comp_df.columns and 'Mmu_R2' in comp_df.columns:
        msize_mean = comp_df.loc[comp_df['Refrigerant'] == 'OVERALL_MEAN', 'Msize_R2'].values[0]
        mmu_mean = comp_df.loc[comp_df['Refrigerant'] == 'OVERALL_MEAN', 'Mmu_R2'].values[0]
        diff_size = mmu_mean - msize_mean
        print(f"2. [控制检验 Mμ vs Msize]: 均值 R² 变化 = {diff_size:+.4f}")
        if diff_size > 0:
            print("   -> ✅ 强力论证：提升并非来自简单的分子量特征增加，而是源自真实静电物理信息！")
        else:
            print("   -> ⚠️ 分子量控制组表现与偶极矩接近。")

    if 'Mphys_R2' in comp_df.columns and 'Mmu_R2' in comp_df.columns:
        mphys_mean = comp_df.loc[comp_df['Refrigerant'] == 'OVERALL_MEAN', 'Mphys_R2'].values[0]
        mmu_mean = comp_df.loc[comp_df['Refrigerant'] == 'OVERALL_MEAN', 'Mmu_R2'].values[0]
        diff_phys = mphys_mean - mmu_mean
        print(f"3. [全量检验 Mphys vs Mμ]: 均值 R² 变化 = {diff_phys:+.4f}")
        if abs(diff_phys) < 0.02:
            print("   -> 💡 结论：Mphys 与 Mμ 表现接近，证明偶极矩已抓取了 3D 量子特征的主要核心增量！")
        elif diff_phys > 0:
            print("   -> 💡 结论：极化率和体积在偶极矩之外提供了进一步的物理互补。")
    print("="*70)

if __name__ == '__main__':
    main()
