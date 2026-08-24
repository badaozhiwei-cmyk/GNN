"""
aggregate_ablation_results.py — 顶刊规范消融实验多维指标汇总分析引擎 (3-Tier Framework)
========================================================================================
【三层评价体系设计 (Nature MI / AIChE J. / JCIM 规范)】
  第 1 层 [全局宏观主指标]:
    - R²_pooled   : 全折 12 个制冷剂预测值微观拼接的全局决定系数 (消除单分子方差扭曲)
    - Macro-MAE   : 各制冷剂等权平均绝对误差 (防止 R32/R134a 等大样本分子主导)
    - Macro-RMSE  : 各制冷剂等权均方根误差
  
  第 2 层 [单分子物理精度主指标]:
    - MAE (mean ± SD)   : 各制冷剂绝对物理误差
    - RMSE (mean ± SD)  : 各制冷剂方差敏感误差
    - AARD% (mean ± SD) : 平均绝对相对偏差百分比
  
  第 3 层 [泛化边界诊断指标]:
    - Single-fold R² (mean ± SD) : 用于诊断是“单分子方差失效”还是“预测精度失效”

【支持模式】
  M0, Msize, Mmu, Mphys, M_interact, M_all
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

MODES = ['M0', 'Msize', 'Mmu', 'Mphys', 'M_interact', 'M_all']
FAMILY = 'HFC'
SPLIT_MODE = 'loro'
BASE_DIR = 'results_ablation'

def compute_aard(y_true, y_pred, eps=1e-4):
    y_true_safe = np.maximum(y_true, eps)
    return np.mean(np.abs(y_true - y_pred) / y_true_safe) * 100.0

def load_seed_predictions(mode_dir):
    """
    读取某个模式下所有 fold 和 seed 的原始预测文件，用于计算 Pooled R² 和各 fold 的多 seed 指标
    """
    preds_pattern = os.path.join(mode_dir, "loro_*_preds", "seed*.csv")
    files = glob.glob(preds_pattern)
    if not files:
        return None
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def analyze_mode(mode):
    mode_dir = os.path.join(BASE_DIR, f"{FAMILY}_{SPLIT_MODE}_{mode}")
    summary_csv = os.path.join(mode_dir, "summary.csv")
    
    if not os.path.exists(mode_dir):
        return None

    # 读取各折汇总数据
    df_summary = pd.read_csv(summary_csv) if os.path.exists(summary_csv) else None
    
    # 读取原始各 seed 预测
    all_preds_df = load_seed_predictions(mode_dir)
    
    # 1. 计算每个 Seed 的 Pooled R² 与全局指标，再求 mean ± std
    pooled_r2_seeds = []
    macro_mae_seeds = []
    macro_rmse_seeds = []
    
    per_ref_metrics = {}
    
    if all_preds_df is not None and not all_preds_df.empty:
        unique_seeds = all_preds_df['seed'].unique()
        unique_refs = all_preds_df['refrigerant'].unique()
        
        for seed in unique_seeds:
            s_df = all_preds_df[all_preds_df['seed'] == seed]
            
            # 全折 Pooled R²
            p_r2 = r2_score(s_df['true_x1'], s_df['pred_x1'])
            pooled_r2_seeds.append(p_r2)
            
            # 各制冷剂单独指标
            ref_maes = []
            ref_rmses = []
            for ref in unique_refs:
                r_df = s_df[s_df['refrigerant'] == ref]
                if not r_df.empty:
                    mae = mean_absolute_error(r_df['true_x1'], r_df['pred_x1'])
                    rmse = np.sqrt(mean_squared_error(r_df['true_x1'], r_df['pred_x1']))
                    aard = compute_aard(r_df['true_x1'].values, r_df['pred_x1'].values)
                    try:
                        r2_single = r2_score(r_df['true_x1'], r_df['pred_x1'])
                    except:
                        r2_single = np.nan
                    
                    if ref not in per_ref_metrics:
                        per_ref_metrics[ref] = {'MAE': [], 'RMSE': [], 'AARD': [], 'R2_single': [], 'N': len(r_df)}
                    
                    per_ref_metrics[ref]['MAE'].append(mae)
                    per_ref_metrics[ref]['RMSE'].append(rmse)
                    per_ref_metrics[ref]['AARD'].append(aard)
                    per_ref_metrics[ref]['R2_single'].append(r2_single)
                    
                    ref_maes.append(mae)
                    ref_rmses.append(rmse)
            
            macro_mae_seeds.append(np.mean(ref_maes))
            macro_rmse_seeds.append(np.mean(ref_rmses))
            
    return {
        'mode': mode,
        'summary_df': df_summary,
        'pooled_r2_mean': np.mean(pooled_r2_seeds) if pooled_r2_seeds else np.nan,
        'pooled_r2_std': np.std(pooled_r2_seeds) if pooled_r2_seeds else np.nan,
        'macro_mae_mean': np.mean(macro_mae_seeds) if macro_mae_seeds else np.nan,
        'macro_mae_std': np.std(macro_mae_seeds) if macro_mae_seeds else np.nan,
        'macro_rmse_mean': np.mean(macro_rmse_seeds) if macro_rmse_seeds else np.nan,
        'macro_rmse_std': np.std(macro_rmse_seeds) if macro_rmse_seeds else np.nan,
        'per_ref': per_ref_metrics
    }

def main():
    print("="*80)
    print("  🏆 GNN 消融实验 (HFC LORO) 顶刊三层指标汇总分析引擎")
    print("="*80)
    
    results = {}
    for m in MODES:
        res = analyze_mode(m)
        if res:
            results[m] = res
            print(f"✅ 加载成功: {m:<10} | Pooled R²: {res['pooled_r2_mean']:.4f} (±{res['pooled_r2_std']:.4f}) | Macro MAE: {res['macro_mae_mean']:.4f}")
        else:
            pass

    if not results:
        print("❌ 未在 results_ablation/ 目录下找到任何消融结果，请先在 Kaggle 运行实验！")
        return

    # ============================================================
    # 1. 打印第 1 层：全局宏观主指标总览表 (Tier 1: Global Master Table)
    # ============================================================
    print("\n" + "━"*80)
    print("  📊 [Tier 1] 全局宏观主指标对比 (Global Master Metrics across all LORO folds)")
    print("━"*80)
    
    tier1_rows = []
    for m, res in results.items():
        tier1_rows.append({
            'Model': m,
            'Pooled R² (mean±std)': f"{res['pooled_r2_mean']:.4f} ± {res['pooled_r2_std']:.4f}",
            'Macro MAE': f"{res['macro_mae_mean']:.4f} ± {res['macro_mae_std']:.4f}",
            'Macro RMSE': f"{res['macro_rmse_mean']:.4f} ± {res['macro_rmse_std']:.4f}",
        })
    df_t1 = pd.DataFrame(tier1_rows)
    print(df_t1.to_string(index=False))

    # ============================================================
    # 2. 打印第 2 & 3 层：各制冷剂细分指标表 (Tier 2 & 3: Per-Refrigerant Breakdown)
    # ============================================================
    print("\n" + "━"*80)
    print("  🔬 [Tier 2 & 3] 各制冷剂详细精度与边界诊断表 (MAE / RMSE / AARD% / R²_single)")
    print("━"*80)
    
    all_refs = set()
    for res in results.values():
        all_refs.update(res['per_ref'].keys())
    all_refs = sorted(list(all_refs))

    for m, res in results.items():
        print(f"\n▶ 模式: 【 {m} 】")
        ref_rows = []
        for ref in all_refs:
            if ref in res['per_ref']:
                info = res['per_ref'][ref]
                ref_rows.append({
                    'Refrigerant': ref,
                    'N': info['N'],
                    'MAE (↓)': f"{np.mean(info['MAE']):.4f} ± {np.std(info['MAE']):.4f}",
                    'RMSE (↓)': f"{np.mean(info['RMSE']):.4f} ± {np.std(info['RMSE']):.4f}",
                    'AARD (%)': f"{np.mean(info['AARD']):.2f}% ± {np.std(info['AARD']):.2f}%",
                    'Single R² (Diag)': f"{np.mean(info['R2_single']):.4f} ± {np.std(info['R2_single']):.4f}",
                })
        df_ref = pd.DataFrame(ref_rows)
        print(df_ref.to_string(index=False))

    # ============================================================
    # 3. 导出全量对比大宽表
    # ============================================================
    out_table_path = os.path.join(BASE_DIR, "ablation_3tier_summary_table.csv")
    
    # 构造大宽表
    wide_rows = []
    for ref in all_refs:
        row = {'Refrigerant': ref}
        for m in MODES:
            if m in results and ref in results[m]['per_ref']:
                info = results[m]['per_ref'][ref]
                row[f'{m}_MAE'] = np.mean(info['MAE'])
                row[f'{m}_RMSE'] = np.mean(info['RMSE'])
                row[f'{m}_AARD'] = np.mean(info['AARD'])
                row[f'{m}_Single_R2'] = np.mean(info['R2_single'])
            else:
                row[f'{m}_MAE'] = np.nan
                row[f'{m}_RMSE'] = np.nan
                row[f'{m}_AARD'] = np.nan
                row[f'{m}_Single_R2'] = np.nan
        wide_rows.append(row)
    
    df_wide = pd.DataFrame(wide_rows)
    df_wide.to_csv(out_table_path, index=False)
    print(f"\n🎉 顶刊规范 3-Tier 汇总分析表已保存至: {out_table_path}")

if __name__ == '__main__':
    main()
