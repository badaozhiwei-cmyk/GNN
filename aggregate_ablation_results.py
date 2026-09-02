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
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

MODES = ['M0', 'Mphys', 'Mthermo', 'Mreduced', 'Minteract', 'Mreduced_pure', 'M_all']
FAMILY = 'HFC'
SPLIT_MODE = 'loro'
BASE_DIR = 'results_ablation'
EXPECTED_HFC_REFS = {
    'R23', 'R32', 'R41', 'R125', 'R134', 'R134a',
    'R143a', 'R152a', 'R161', 'R227ea', 'R236fa', 'R245fa'
}

def compute_aard(y_true, y_pred, eps=1e-4):
    y_true_safe = np.maximum(y_true, eps)
    return np.mean(np.abs(y_true - y_pred) / y_true_safe) * 100.0

def load_seed_predictions(mode_dir, expected_seeds):
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
        if set(df.columns) < {'seed', 'refrigerant', 'true_x1', 'pred_x1'}:
            raise ValueError(f'Malformed prediction file: {f}')
        file_seeds = set(df['seed'].astype(int))
        if len(file_seeds) != 1:
            raise ValueError(f'Prediction file must contain exactly one seed: {f}')
        file_refs = set(df['refrigerant'].astype(str))
        if len(file_refs) != 1:
            raise ValueError(f'Prediction file must contain exactly one refrigerant: {f}')
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined['seed'].isin(expected_seeds)].copy()
    combined['pred_x1'] = np.clip(combined['pred_x1'], 0.0, 1.0)
    return combined

def config_matches_mode(config, mode):
    if config.get('family') != FAMILY or config.get('mode') != SPLIT_MODE:
        return False
    if mode == 'Mphys_gated':
        return config.get('descriptor_mode') == 'Mphys' and config.get('use_adaptive_gate') is True
    return config.get('descriptor_mode') == mode and config.get('use_adaptive_gate') is not True


def discover_mode_dir(mode, selected_hash=None):
    candidates = []
    pattern = os.path.join(BASE_DIR, f'{FAMILY}_{SPLIT_MODE}_*_*')
    for mode_dir in glob.glob(pattern):
        config_path = os.path.join(mode_dir, 'config.json')
        if not os.path.isfile(config_path):
            continue
        with open(config_path, 'r', encoding='utf-8') as handle:
            config = json.load(handle)
        if not config_matches_mode(config, mode):
            continue
        directory_hash = os.path.basename(mode_dir).rsplit('_', 1)[-1]
        if selected_hash and directory_hash != selected_hash:
            continue
        candidates.append((mode_dir, config, directory_hash))

    if not candidates:
        return None
    if len(candidates) > 1:
        choices = ', '.join(item[2] for item in candidates)
        raise RuntimeError(
            f'Multiple configurations found for {mode}: {choices}. '
            f'Choose one with --config {mode}=HASH.'
        )
    return candidates[0]


def validate_prediction_completeness(all_preds_df, expected_seeds, mode_dir):
    if all_preds_df is None or all_preds_df.empty:
        raise RuntimeError(f'No predictions found in {mode_dir}')
    actual_refs = set(all_preds_df['refrigerant'].astype(str))
    if actual_refs != EXPECTED_HFC_REFS:
        missing = sorted(EXPECTED_HFC_REFS - actual_refs)
        extra = sorted(actual_refs - EXPECTED_HFC_REFS)
        raise RuntimeError(f'Incomplete folds in {mode_dir}; missing={missing}, extra={extra}')
    for ref in EXPECTED_HFC_REFS:
        actual_seeds = set(all_preds_df.loc[all_preds_df['refrigerant'] == ref, 'seed'].astype(int))
        if actual_seeds != set(expected_seeds):
            raise RuntimeError(
                f'Incomplete seeds for {ref} in {mode_dir}: '
                f'expected={expected_seeds}, actual={sorted(actual_seeds)}'
            )


def analyze_mode(mode, selected_hash=None):
    discovered = discover_mode_dir(mode, selected_hash)
    if discovered is None:
        return None
    mode_dir, config, config_hash = discovered
    summary_csv = os.path.join(mode_dir, "summary.csv")
    expected_seeds = [int(seed) for seed in config['seeds']]

    # 读取各折汇总数据
    df_summary = pd.read_csv(summary_csv) if os.path.exists(summary_csv) else None
    
    # 读取原始各 seed 预测
    all_preds_df = load_seed_predictions(mode_dir, expected_seeds)
    validate_prediction_completeness(all_preds_df, expected_seeds, mode_dir)
    
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
        'config_hash': config_hash,
        'config': config,
        'summary_df': df_summary,
        'pooled_r2_mean': np.mean(pooled_r2_seeds) if pooled_r2_seeds else np.nan,
        'pooled_r2_std': np.std(pooled_r2_seeds) if pooled_r2_seeds else np.nan,
        'macro_mae_mean': np.mean(macro_mae_seeds) if macro_mae_seeds else np.nan,
        'macro_mae_std': np.std(macro_mae_seeds) if macro_mae_seeds else np.nan,
        'macro_rmse_mean': np.mean(macro_rmse_seeds) if macro_rmse_seeds else np.nan,
        'macro_rmse_std': np.std(macro_rmse_seeds) if macro_rmse_seeds else np.nan,
        'per_ref': per_ref_metrics
    }

def parse_config_selections(values):
    selections = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'Invalid --config value {value!r}; expected MODE=HASH')
        mode, config_hash = value.split('=', 1)
        selections[mode] = config_hash
    return selections


def main():
    parser = argparse.ArgumentParser(description='Aggregate complete ablation configurations')
    parser.add_argument(
        '--config', action='append', default=[], metavar='MODE=HASH',
        help='Select an exact configuration when multiple hashes exist (repeatable)'
    )
    args = parser.parse_args()
    selections = parse_config_selections(args.config)
    print("="*80)
    print("  🏆 GNN 消融实验 (HFC LORO) 顶刊三层指标汇总分析引擎")
    print("="*80)
    
    results = {}
    for m in MODES:
        res = analyze_mode(m, selections.get(m))
        if res:
            results[m] = res
            print(f"✅ 加载成功: {m:<10} [{res['config_hash']}] | Pooled R²: {res['pooled_r2_mean']:.4f} (±{res['pooled_r2_std']:.4f}) | Macro MAE: {res['macro_mae_mean']:.4f}")
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
    # 3. 打印核心对比表：12 个制冷剂 MAE 修复对比表 (Table 2: Per-Refrigerant MAE Reduction)
    # ============================================================
    print("\n" + "━"*80)
    print("  🎯 [Table 2] 12 个制冷剂 MAE 逐级修复横向对比表 (M0 -> Msize -> Mmu -> Mphys)")
    print("━"*80)
    
    mae_comp_rows = []
    for ref in all_refs:
        n_samples = results[list(results.keys())[0]]['per_ref'][ref]['N'] if ref in results[list(results.keys())[0]]['per_ref'] else 0
        r_row = {'Refrigerant': ref, 'N': n_samples}
        m0_val = None
        for m in MODES:
            if m in results and ref in results[m]['per_ref']:
                val = np.mean(results[m]['per_ref'][ref]['MAE'])
                r_row[f"{m}_MAE"] = f"{val:.4f}"
                if m == 'M0': m0_val = val
                if m in ['Mmu', 'Mphys', 'Mphys_gated', 'Mthermo', 'Mreduced', 'M_interact'] and m0_val is not None and m0_val > 0:
                    gain = (val - m0_val) / m0_val * 100.0
                    r_row[f"Δ_{m}%"] = f"{gain:+.1f}%"
            else:
                r_row[f"{m}_MAE"] = "N/A"
        mae_comp_rows.append(r_row)
    
    df_mae_comp = pd.DataFrame(mae_comp_rows)
    print(df_mae_comp.to_string(index=False))

    # ============================================================
    # 4. 导出全量对比大宽表
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
