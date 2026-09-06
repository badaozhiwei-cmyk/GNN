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
        config_file = os.path.join(md, 'config.json')
        if os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                cfg = json.load(f)
            mode = cfg.get('descriptor_mode', 'Unknown')
            if cfg.get('use_adaptive_gate', False):
                mode = f"{mode}+Gate"
        else:
            parts = dir_name.split('_')
            mode = parts[2] if len(parts) > 2 else dir_name
            
        pred_folders = glob.glob(os.path.join(md, 'loro_*_preds'))
        if not pred_folders:
            continue
            
        pooled_trues = []
        pooled_preds_ensemble = []
        ref_records = []
        
        for pf in pred_folders:
            ref_name = os.path.basename(pf).replace('loro_', '').replace('_preds', '')
            seed_files = sorted(glob.glob(os.path.join(pf, 'seed*.csv')))
            if not seed_files:
                continue
                
            # ── 1. 严格显式按 sample_id 对齐 ──
            seed_dfs = []
            for sf in seed_files:
                sdf = pd.read_csv(sf)
                required = {'sample_id', 'true_x1', 'pred_x1_raw'}
                if not required.issubset(sdf.columns):
                    raise ValueError(f"{sf} 缺失必要列: {required - set(sdf.columns)}")
                if sdf['sample_id'].duplicated().any():
                    raise ValueError(f"{sf} 存在重复的 sample_id！")
                seed_dfs.append(sdf[['sample_id', 'true_x1', 'pred_x1_raw']].set_index('sample_id'))
                
            base_ids = seed_dfs[0].index
            for i, sdf in enumerate(seed_dfs[1:], start=1):
                if not sdf.index.equals(base_ids):
                    raise ValueError(f"🚨 {pf} 中 seed 文件样本 ID 顺序不一致: seed{i} vs seed0")
                    
            y_true = seed_dfs[0].loc[base_ids, 'true_x1'].to_numpy()
            pred_mat = np.column_stack([sdf.loc[base_ids, 'pred_x1_raw'].to_numpy() for sdf in seed_dfs])
            
            # ── 2. 双轨制计算：单 Seed 分别评估求均值 vs 5-seed 集成 ──
            seed_maes = [mean_absolute_error(y_true, np.clip(pred_mat[:, s], 0, 1)) for s in range(pred_mat.shape[1])]
            seed_r2s = [r2_score(y_true, np.clip(pred_mat[:, s], 0, 1)) for s in range(pred_mat.shape[1])]
            
            avg_pred = np.mean(pred_mat, axis=1)
            metrics_ensemble = compute_metrics(y_true, avg_pred)
            
            metrics_record = {
                'refrigerant': ref_name,
                'num_seeds': len(seed_files),
                'MAE_seed_mean': np.mean(seed_maes),
                'MAE_seed_std': np.std(seed_maes),
                'R2_seed_mean': np.mean(seed_r2s),
                'R2_seed_std': np.std(seed_r2s),
                'MAE_ensemble': metrics_ensemble['MAE'],
                'R2_ensemble': metrics_ensemble['R2'],
                'MARD(%)_ensemble': metrics_ensemble['MARD(%)'],
                'log_MAE_ensemble': metrics_ensemble['log_MAE'],
            }
            ref_records.append(metrics_record)
            
            pooled_trues.extend(y_true)
            pooled_preds_ensemble.extend(avg_pred)
            
        if not ref_records:
            continue
            
        ref_df = pd.DataFrame(ref_records)
        detailed_ref_results[mode] = ref_df
        
        # ── 3. 宏观统计：同时呈现标准 Seed-Mean 与 Ensemble ──
        macro_mae_seed = ref_df['MAE_seed_mean'].mean()
        macro_r2_seed = ref_df['R2_seed_mean'].mean()
        macro_mae_ens = ref_df['MAE_ensemble'].mean()
        macro_r2_ens = ref_df['R2_ensemble'].mean()
        
        pooled_ens = compute_metrics(pooled_trues, pooled_preds_ensemble)
        
        ci_dict = cluster_bootstrap_macro(ref_df.rename(columns={'MAE_seed_mean': 'MAE', 'R2_seed_mean': 'R2', 'log_MAE_ensemble': 'log_MAE'}))
        
        all_mode_results[mode] = {
            'Mode': mode,
            'Num_Refs': len(ref_df),
            'Macro_MAE (Seed-Mean)': f"{macro_mae_seed:.4f}",
            'MAE_95CI': f"[{ci_dict['MAE_CI95'][0]:.4f}, {ci_dict['MAE_CI95'][1]:.4f}]",
            'Macro_R2 (Seed-Mean)': f"{macro_r2_seed:.4f}",
            'Macro_MAE (Ensemble)': f"{macro_mae_ens:.4f}",
            'Macro_R2 (Ensemble)': f"{macro_r2_ens:.4f}",
            'Pooled_MAE (Ensemble)': f"{pooled_ens['MAE']:.4f}",
            'Pooled_R2 (Ensemble)': f"{pooled_ens['R2']:.4f}",
            'Pooled_RMSE (Ensemble)': f"{pooled_ens['RMSE']:.4f}"
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
