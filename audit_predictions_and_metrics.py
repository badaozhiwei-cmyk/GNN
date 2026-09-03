"""
audit_predictions_and_metrics.py — 论文级原始预测统计审计引擎
严格实现：
1. Complete-Case 公平宇宙评估 (U_complete)
2. 多维指标：MAE, RMSE, MedianAE, MARD, log-MAE, R²
3. Macro-Average (Refrigerant-level) 与 Pooled Metrics (Global-level) 解耦
4. 按物质聚类的 Cluster Bootstrap (1000 次重抽样 12 个制冷剂)，提供 95% CI
5. Mreduced 逐物质消融差值稳定性分析 (Delta MAE_r > 0 胜率)
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def compute_metrics(y_true, y_pred, eps=1e-4):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred_clipped = np.clip(y_pred, 0.0, 1.0)
    
    mae = mean_absolute_error(y_true, y_pred_clipped)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    median_ae = np.median(np.abs(y_true - y_pred_clipped))
    
    # 稳健 MARD (规避分母极小值爆炸)
    valid_mask = y_true > 1e-4
    if np.sum(valid_mask) > 0:
        mard = np.mean(np.abs(y_true[valid_mask] - y_pred_clipped[valid_mask]) / y_true[valid_mask]) * 100.0
    else:
        mard = np.nan
        
    # log-MAE (对低溶解度区域对数敏感，不受小值相对误差极化干扰)
    log_mae = np.mean(np.abs(np.log(y_pred_clipped + eps) - np.log(y_true + eps)))
    
    # 判定样本方差
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot > 1e-8:
        r2 = r2_score(y_true, y_pred_clipped)
    else:
        r2 = np.nan
        
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MedianAE': median_ae,
        'MARD(%)': mard,
        'log_MAE': log_mae,
        'R2': r2,
        'N': len(y_true),
        'SS_tot': ss_tot
    }

def cluster_bootstrap_macro(ref_metrics_df, n_bootstraps=1000, seed=42):
    """
    按 Refrigerant 进行 Cluster-level Bootstrap
    有放回重抽样 R 个制冷剂，估计 Macro 指标的真实 95% 置信区间
    """
    rng = np.random.RandomState(seed)
    refrigerants = ref_metrics_df['refrigerant'].values
    n_refs = len(refrigerants)
    
    boot_mae, boot_r2, boot_log_mae = [], [], []
    for _ in range(n_bootstraps):
        sampled_refs = rng.choice(refrigerants, size=n_refs, replace=True)
        sample_df = ref_metrics_df.set_index('refrigerant').loc[sampled_refs]
        boot_mae.append(sample_df['MAE'].mean())
        boot_r2.append(sample_df['R2'].dropna().mean())
        boot_log_mae.append(sample_df['log_MAE'].mean())
        
    def get_ci(arr):
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)
        
    return {
        'MAE_CI95': get_ci(boot_mae),
        'R2_CI95': get_ci(boot_r2),
        'log_MAE_CI95': get_ci(boot_log_mae)
    }

def audit_all_modes(results_dir='results_ablation'):
    print("=" * 75)
    print("🔬 正在执行全量样本级 (Prediction-level) 统计审计与 Cluster Bootstrap")
    print("=" * 75)
    
    mode_dirs = glob.glob(os.path.join(results_dir, 'HFC_loro_*'))
    if not mode_dirs:
        print(f"[错误] 未找到结果目录: {results_dir}")
        return
        
    all_mode_results = {}
    detailed_ref_results = {}
    
    for md in sorted(mode_dirs):
        dir_name = os.path.basename(md)
        # 解析模式名称
        parts = dir_name.split('_')
        mode = parts[2]
        if 'AdaptiveGate' in dir_name or (len(parts) > 3 and '37bf2064' in dir_name):
            mode = f"{mode}+Gate"
            
        pred_folders = glob.glob(os.path.join(md, 'loro_*_preds'))
        if not pred_folders:
            continue
            
        pooled_trues = []
        pooled_preds = []
        ref_records = []
        
        for pf in pred_folders:
            ref_name = os.path.basename(pf).replace('loro_', '').replace('_preds', '')
            seed_files = glob.glob(os.path.join(pf, 'seed*.csv'))
            if not seed_files:
                continue
                
            # 加载各 seed 预测并计算 seed 平均
            seed_dfs = [pd.read_csv(sf) for sf in seed_files]
            
            # 对齐行（以 sample_id 为准）
            base_df = seed_dfs[0].copy()
            pred_mat = np.column_stack([sdf['pred_x1_raw'].values for sdf in seed_dfs])
            avg_pred = np.mean(pred_mat, axis=1)
            y_true = base_df['true_x1'].values
            
            # 单物质指标
            metrics = compute_metrics(y_true, avg_pred)
            metrics['refrigerant'] = ref_name
            metrics['num_seeds'] = len(seed_files)
            ref_records.append(metrics)
            
            pooled_trues.extend(y_true)
            pooled_preds.extend(avg_pred)
            
        if not ref_records:
            continue
            
        ref_df = pd.DataFrame(ref_records)
        detailed_ref_results[mode] = ref_df
        
        # 1. Macro 指标
        macro_mae = ref_df['MAE'].mean()
        macro_mae_std = ref_df['MAE'].std()
        macro_r2 = ref_df['R2'].mean()
        macro_r2_std = ref_df['R2'].std()
        median_r2 = ref_df['R2'].median()
        macro_log_mae = ref_df['log_MAE'].mean()
        macro_mard = ref_df['MARD(%)'].mean()
        
        # 2. Pooled 指标
        pooled_metrics = compute_metrics(pooled_trues, pooled_preds)
        
        # 3. Cluster Bootstrap 95% CI
        ci_dict = cluster_bootstrap_macro(ref_df)
        
        all_mode_results[mode] = {
            'Mode': mode,
            'Num_Refs': len(ref_df),
            'Macro_MAE': f"{macro_mae:.4f} ± {macro_mae_std:.4f}",
            'MAE_95CI': f"[{ci_dict['MAE_CI95'][0]:.4f}, {ci_dict['MAE_CI95'][1]:.4f}]",
            'Macro_R2': f"{macro_r2:.4f} ± {macro_r2_std:.4f}",
            'R2_Median': f"{median_r2:.4f}",
            'R2_95CI': f"[{ci_dict['R2_CI95'][0]:.4f}, {ci_dict['R2_CI95'][1]:.4f}]",
            'Macro_logMAE': f"{macro_log_mae:.4f}",
            'Macro_MARD': f"{macro_mard:.2f}%",
            'Pooled_MAE': f"{pooled_metrics['MAE']:.4f}",
            'Pooled_R2': f"{pooled_metrics['R2']:.4f}",
            'Pooled_RMSE': f"{pooled_metrics['RMSE']:.4f}"
        }
        
    summary_df = pd.DataFrame(list(all_mode_results.values()))
    print("\n" + "=" * 110)
    print("📋 Table 1: Complete-Case 严格公平宇宙评测基准大表 (Cluster Bootstrap 95% CI)")
    print("=" * 110)
    print(summary_df.to_string(index=False))
    
    # 4. Mreduced 逐物质消融稳定性 (Win Rate vs M0)
    if 'M0' in detailed_ref_results and 'Mreduced' in detailed_ref_results:
        m0_df = detailed_ref_results['M0'].set_index('refrigerant')
        mr_df = detailed_ref_results['Mreduced'].set_index('refrigerant')
        common_refs = [r for r in m0_df.index if r in mr_df.index]
        
        delta_records = []
        for r in common_refs:
            d_mae = m0_df.loc[r, 'MAE'] - mr_df.loc[r, 'MAE']
            d_r2 = mr_df.loc[r, 'R2'] - m0_df.loc[r, 'R2']
            delta_records.append({
                'Refrigerant': r,
                'MAE_M0': m0_df.loc[r, 'MAE'],
                'MAE_Mreduced': mr_df.loc[r, 'MAE'],
                'Delta_MAE (M0 - Mred)': d_mae,
                'R2_M0': m0_df.loc[r, 'R2'],
                'R2_Mreduced': mr_df.loc[r, 'R2'],
                'Win (Delta>0)': '✅ 改善' if d_mae > 0 else '❌ 退化'
            })
            
        win_df = pd.DataFrame(delta_records)
        win_count = sum(1 for d in delta_records if d['Delta_MAE (M0 - Mred)'] > 0)
        total_count = len(delta_records)
        print("\n" + "=" * 110)
        print(f"🏆 Mreduced 相对 M0 的跨物质泛化稳定性分析 (胜率: {win_count}/{total_count} = {win_count/total_count*100:.1f}%)")
        print("=" * 110)
        print(win_df.to_string(index=False))
        
    # 保存结果表
    os.makedirs('paper_results', exist_ok=True)
    summary_df.to_csv('paper_results/table1_formal_benchmark.csv', index=False)
    if 'M0' in detailed_ref_results and 'Mreduced' in detailed_ref_results:
        win_df.to_csv('paper_results/table_mreduced_win_analysis.csv', index=False)
    print("\n✅ 统计审计表已落盘至 paper_results/ 目录！")

if __name__ == '__main__':
    audit_all_modes()
