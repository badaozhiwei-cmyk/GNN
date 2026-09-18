"""
step16_hfo_zeroshot_probe.py — Phase II-D: HFO Zero-Shot Cross-Family Stress Test
================================================================================
【科学定位】
  零样本跨家族应力测试 (Zero-Shot Cross-Family Stress Test)。
  严格遵循用户与合作者裁决：
    1. 严禁在 HFO 上做任何微调或重训，纯粹采用饱和 HFC 训练的模型进行 Zero-Shot 前向推断；
    2. 主攻核心锚点: [emim][Tf2N] 中的 R1336mzz(E) (N=11) 与 R1336mzz(Z) (N=12) 顺反异构对 (N=23)；
    3. 全景应力测试: Table S4 全量 1106 个 HFO 样本点 (R1234yf, R1234ze(E), R1233zd(E), R1336mzz(E), R1336mzz(Z))；
    4. 考察对比态热力学描述符 (Tr, Pr, omega) 能否缓冲 C-C -> C=C 的跨家族化学漂移，
       以及模型分歧不确定性 sigma 能否敏感感知跨家族 OOD。

【产物】
  - paper_results/hfo_zeroshot_sample_predictions.csv
  - paper_results/table_hfo_zeroshot_metrics.csv
  - paper_results/hfo_zeroshot_analysis_report.json
"""

from __future__ import annotations
import argparse
import os
import sys
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem
from rdkit.Chem import Descriptors

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'GNN_for_property_prediction'))

import torch
from torch_geometric.data import Batch, Data
from Dataset_v6 import (
    FEATURE_SCHEMA,
    BASE_FEATURES,
    MODE_DEF,
    MODE_INDICES,
    MODE_COND_DIM,
    combine_Graph,
    add_global
)
from Model_v6 import IL_GAT_v6

# 国际标准参考方程临界参数 (NIST / CoolProp / Akasaka-Lemmon)
HFO_CRITICAL = {
    'R1234YF': (367.85, 3.382, 0.276),
    'R1234ZE(E)': (382.51, 3.635, 0.313),
    'R1233ZD(E)': (438.86, 3.5828, 0.304),
    'R1336MZZ(E)': (403.53, 2.779, 0.4128),
    'R1336MZZ(Z)': (444.50, 2.9037, 0.386),
}

# 国际标准权威 SMILES 映射表 (纠正历史数据中 R1234yf 误填为全氟丙烯 C3F6 的严重偏差)
HFO_CANONICAL_SMILES = {
    'R1234YF': 'C=C(F)C(F)(F)F',           # 2,3,3,3-tetrafluoropropene (C3H2F4, MW=114.041)
    'R1234ZE(E)': 'F/C=C/C(F)(F)F',         # trans-1,3,3,3-tetrafluoropropene (C3H2F4, MW=114.041)
    'R1233ZD(E)': 'FC(F)(F)/C=C/Cl',        # trans-1-chloro-3,3,3-trifluoropropene (C3H2ClF3, MW=130.496)
    'R1336MZZ(E)': 'FC(F)(F)/C=C/C(F)(F)F',  # trans-1,1,1,4,4,4-hexafluoro-2-butene (C4H2F6, MW=164.048)
    'R1336MZZ(Z)': r'FC(F)(F)/C=C\C(F)(F)F', # cis-1,1,1,4,4,4-hexafluoro-2-butene (C4H2F6, MW=164.048)
}

from prepare_tri_graph_data_v6 import mol2graph_components

def mol2graph(mol_data):
    x = torch.tensor(mol_data[0], dtype=torch.long)
    edge_index = torch.tensor(mol_data[1], dtype=torch.long)
    if len(mol_data[2]) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros((1, 3), dtype=torch.long)
    else:
        edge_attr = torch.tensor(mol_data[2], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def compute_metrics(true_y: np.ndarray, pred_y: np.ndarray) -> dict[str, float]:
    pred_y_clipped = np.clip(pred_y, 0.0, 1.0)
    mae_raw  = mean_absolute_error(true_y, pred_y)
    r2_raw   = r2_score(true_y, pred_y) if len(true_y) > 1 and np.var(true_y) > 1e-12 else 0.0
    rmse_raw = np.sqrt(mean_squared_error(true_y, pred_y))
    
    mae_clip  = mean_absolute_error(true_y, pred_y_clipped)
    r2_clip   = r2_score(true_y, pred_y_clipped) if len(true_y) > 1 and np.var(true_y) > 1e-12 else 0.0
    rmse_clip = np.sqrt(mean_squared_error(true_y, pred_y_clipped))
    
    return {
        'mae_raw': float(mae_raw), 'r2_raw': float(r2_raw), 'rmse_raw': float(rmse_raw),
        'mae_clip': float(mae_clip), 'r2_clip': float(r2_clip), 'rmse_clip': float(rmse_clip)
    }

def main():
    parser = argparse.ArgumentParser(description="Unsaturated Refrigerant (HFO/HCFO) Zero-Shot Probe Runner")
    parser.add_argument('--model_source', type=str, default='HFC_all', choices=['HFC_all', 'B1', 'L2'],
                        help="HFC-trained baseline model checkpoint source folder (default: HFC_all)")
    parser.add_argument('--target_scope', type=str, default='all', choices=['anchor_23', 'all'],
                        help="Target test set scope: anchor_23 (R1336mzz E/Z) or all 1106 unsaturated samples")
    args = parser.parse_args()

    print("=" * 80)
    print("  PHASE II-D STEP 16: UNSATURATED REFRIGERANT (HFO/HCFO) ZERO-SHOT STRESS TEST")
    print(f"  Model Source Checkpoint: {args.model_source}")
    print(f"  Evaluation Scope       : {args.target_scope}")
    print("=" * 80)

    # 1. 筛选并构建 HFO 样本元数据
    df_raw = pd.read_csv('index_with_anion.csv')
    df_hfo = df_raw[df_raw['sheet'] == 'Table S4. VLE HFOs'].copy().reset_index()
    df_hfo.rename(columns={'index': 'orig_data_idx'}, inplace=True)

    if args.target_scope == 'anchor_23':
        df_hfo = df_hfo[df_hfo['refrigerant'].str.upper().str.contains('1336')].copy().reset_index(drop=True)

    print(f"[数据加载] 选定 HFO 样本量 N={len(df_hfo)}")
    print(f"工质分布: {df_hfo['refrigerant'].value_counts().to_dict()}")

    # 2. 从 processed_tri_data 加载底层三分子图
    raw_data = np.load('processed_tri_data/data.npy', allow_pickle=True)
    raw_labels = np.load('processed_tri_data/label.npy', allow_pickle=True)

    # 缓存分子量
    mw_cache = {}
    def get_mw(smi):
        if smi not in mw_cache:
            mol = Chem.MolFromSmiles(smi)
            mw_cache[smi] = float(Descriptors.MolWt(mol)) if mol else 0.0
        return mw_cache[smi]

    # 组装 22 维完整特征向量
    hfo_samples = []
    for row_idx, r in df_hfo.iterrows():
        orig_i = int(r['orig_data_idx'])
        raw_item = raw_data[orig_i]
        r_clean = str(r['refrigerant']).strip().upper()

        ref_smi = HFO_CANONICAL_SMILES.get(r_clean, str(r['refri_smiles']).strip())

        cg = mol2graph(raw_item[0])
        ag = mol2graph(raw_item[1])
        # 修正历史数据中 R1234yf (误存为全氟丙烯 C3F6) 的图拓扑，其余工质与原图严格一致
        if r_clean == 'R1234YF':
            rg = mol2graph(mol2graph_components(ref_smi))
        else:
            rg = mol2graph(raw_item[2])

        combined_g = add_global(combine_Graph([cg, ag, rg]))

        t_val = float(r['T_K'])
        p_val = float(r['P_MPa'])
        if r_clean not in HFO_CRITICAL:
            raise KeyError(f"Missing critical parameters for {r_clean}")

        tc, pc, omega = HFO_CRITICAL[r_clean]
        tr = t_val / tc
        pr = p_val / pc

        ref_mw = get_mw(ref_smi)
        cat_mw = get_mw(r['cation_smiles'])
        ani_mw = get_mw(r['anion_smiles'])

        mol_ref = Chem.MolFromSmiles(ref_smi)
        ref_logp = float(Descriptors.MolLogP(mol_ref)) if mol_ref else 0.0
        try:
            ref_charge = float(Descriptors.MaxAbsPartialCharge(mol_ref)) if mol_ref else 0.0
        except Exception:
            ref_charge = 0.0

        mol_cat = Chem.MolFromSmiles(r['cation_smiles'])
        cat_tpsa = float(Descriptors.TPSA(mol_cat)) if mol_cat else 0.0
        try:
            cat_charge = float(Descriptors.MaxAbsPartialCharge(mol_cat)) if mol_cat else 0.0
        except Exception:
            cat_charge = 0.0

        cond_22 = [
            0, 0, 0, # 0,1,2 reserved for graphs
            t_val, p_val,
            ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa,  # FIXED: Gasteiger charges
            ref_mw, cat_mw,
            0.0, 0.0, 0.0,   # 12~14: xTB (M0/Mreduced not used)
            0.0, 0.0,         # 15~16: deltaE (M0/Mreduced not used)
            tc, pc, omega,
            tr, pr
        ]

        hfo_samples.append({
            'orig_idx': orig_i,
            'sample_id': f"{r['cation']}__{r['anion']}__{r['refrigerant']}__{t_val:.6g}__{p_val:.6g}",
            'cation': str(r['cation']).strip().strip('[]').upper(),
            'anion': str(r['anion']).strip().strip('[]').upper(),
            'refrigerant': r_clean,
            'T_K': t_val,
            'P_MPa': p_val,
            'true_x1': float(r['x1']),
            'graph': combined_g,
            'cond_22': cond_22
        })

    print(f"[特征组装] 成功组装 {len(hfo_samples)} 条 22 维三分子图与状态向量")

    SPECIES_CLASS = {
        'R1234YF': 'HFO',
        'R1234ZE(E)': 'HFO',
        'R1233ZD(E)': 'HCFO',
        'R1336MZZ(E)': 'HFO',
        'R1336MZZ(Z)': 'HFO',
    }

    # 3. 评测执行循环 (M0 与 Mreduced)
    seeds = [42, 43, 44, 45, 46]
    modes = ['M0', 'Mreduced']

    if args.model_source == 'HFC_all':
        model_root = PROJECT_ROOT / 'results_hfc_all'
        prefix = 'HFC_all'
    elif args.model_source == 'B1':
        model_root = PROJECT_ROOT / 'results_split_B'
        prefix = 'B1'
    elif args.model_source == 'L2':
        model_root = PROJECT_ROOT / 'results_split_L2'
        prefix = 'L2'
    else:
        raise ValueError(f"Unknown model_source: {args.model_source}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[运行设备] {device}")

    all_predictions = []
    summary_metrics = []

    for md in modes:
        mode_tag = f"{prefix}_{md}"
        mode_dir = model_root / mode_tag
        scaler_file = mode_dir / 'scalers.pkl'
        if not scaler_file.exists():
            raise FileNotFoundError(f"Missing scalers file: {scaler_file}")

        import joblib
        scalers = joblib.load(scaler_file)
        means = np.array([float(s.mean_[0]) for s in scalers], dtype=np.float32)
        scales = np.array([max(float(s.scale_[0]), 1e-8) for s in scalers], dtype=np.float32)

        feat_indices = MODE_INDICES[md]
        cond_dim = MODE_COND_DIM[md]

        # 载入 5-Seed 模型
        models = {}
        for s in seeds:
            ckpt_p = mode_dir / f"best_seed_{s}.pth"
            if not ckpt_p.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_p}")
            m_args = {
                'emb_dim': 300,
                'dropout_rate': 0.2,
                'cond_dim': cond_dim,
                'pool': 'global',
                'use_layernorm': False,
                'use_adaptive_gate': False
            }
            model = IL_GAT_v6(m_args).to(device)
            ckpt = torch.load(ckpt_p, map_location=device)
            st = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
            model.load_state_dict(st)
            model.eval()
            models[s] = model

        print(f"\n>>> 正在前向推断: 模式 {md} (来自 {prefix} 冻结模型, 5 个种子就绪)")

        # 逐样本推断
        sample_preds_this_mode = []
        for s_info in hfo_samples:
            g_batch = Batch.from_data_list([s_info['graph']]).to(device)
            raw_cond_sub = np.array([s_info['cond_22'][i] for i in feat_indices], dtype=np.float32)
            norm_cond = (raw_cond_sub - means) / scales
            cond_t = torch.tensor(norm_cond, dtype=torch.float32, device=device).unsqueeze(0)

            seed_outputs = []
            with torch.no_grad():
                for s in seeds:
                    pred_val = float(models[s](g_batch, cond_t).flatten().cpu().numpy()[0])
                    seed_outputs.append(pred_val)

            seed_outputs = np.array(seed_outputs)
            clipped_outputs = np.clip(seed_outputs, 0.0, 1.0)
            ens_mu = float(clipped_outputs.mean())
            ens_sigma = float(clipped_outputs.std(ddof=1)) if len(seeds) > 1 else 0.0
            abs_err = abs(s_info['true_x1'] - ens_mu)

            rec = {
                'sample_id': s_info['sample_id'],
                'model_source': args.model_source,
                'descriptor_mode': md,
                'refrigerant': s_info['refrigerant'],
                'chemical_class': SPECIES_CLASS.get(s_info['refrigerant'], 'Other'),
                'cation': s_info['cation'],
                'anion': s_info['anion'],
                'T_K': s_info['T_K'],
                'P_MPa': s_info['P_MPa'],
                'true_x1': s_info['true_x1'],
                'pred_mu': ens_mu,
                'pred_sigma': ens_sigma,
                'abs_error': abs_err,
            }
            for i_s, s in enumerate(seeds):
                rec[f'pred_seed_{s}'] = float(seed_outputs[i_s])
                rec[f'pred_seed_{s}_clip'] = float(clipped_outputs[i_s])
            
            sample_preds_this_mode.append(rec)
            all_predictions.append(rec)

        df_mode_pred = pd.DataFrame(sample_preds_this_mode)

        # 统计分物种及总体指标
        # 1. 顺反异构核心锚点统计 (R1336mzz E vs Z vs Both)
        sub_1336_e = df_mode_pred[df_mode_pred['refrigerant'] == 'R1336MZZ(E)']
        sub_1336_z = df_mode_pred[df_mode_pred['refrigerant'] == 'R1336MZZ(Z)']
        sub_1336_all = df_mode_pred[df_mode_pred['refrigerant'].isin(['R1336MZZ(E)', 'R1336MZZ(Z)'])]

        m_e = compute_metrics(sub_1336_e['true_x1'].values, sub_1336_e['pred_mu'].values) if len(sub_1336_e) > 0 else {}
        m_z = compute_metrics(sub_1336_z['true_x1'].values, sub_1336_z['pred_mu'].values) if len(sub_1336_z) > 0 else {}
        m_1336 = compute_metrics(sub_1336_all['true_x1'].values, sub_1336_all['pred_mu'].values) if len(sub_1336_all) > 0 else {}
        m_all = compute_metrics(df_mode_pred['true_x1'].values, df_mode_pred['pred_mu'].values)

        # 不确定性与误差相关性
        if len(df_mode_pred) > 2 and df_mode_pred['pred_sigma'].std() > 1e-12:
            r_val, _ = pearsonr(df_mode_pred['pred_sigma'], df_mode_pred['abs_error'])
            rho_val, _ = spearmanr(df_mode_pred['pred_sigma'], df_mode_pred['abs_error'])
        else:
            r_val, rho_val = 0.0, 0.0

        summary_metrics.append({
            'model_source': args.model_source,
            'descriptor_mode': md,
            'scope': 'Overall_Unsaturated_Universe',
            'chemical_class': 'HFO+HCFO',
            'N': len(df_mode_pred),
            'MAE': m_all['mae_clip'],
            'RMSE': m_all['rmse_clip'],
            'R2': m_all['r2_clip'],
            'mean_sigma': float(df_mode_pred['pred_sigma'].mean()),
            'pearson_r_sigma_err': float(r_val),
            'spearman_rho_sigma_err': float(rho_val),
        })

        if len(sub_1336_all) > 0:
            summary_metrics.append({
                'model_source': args.model_source,
                'descriptor_mode': md,
                'scope': 'Anchor_R1336mzz_Combined',
                'chemical_class': 'HFO',
                'N': len(sub_1336_all),
                'MAE': m_1336['mae_clip'],
                'RMSE': m_1336['rmse_clip'],
                'R2': m_1336['r2_clip'],
                'mean_sigma': float(sub_1336_all['pred_sigma'].mean()),
                'pearson_r_sigma_err': float(pearsonr(sub_1336_all['pred_sigma'], sub_1336_all['abs_error'])[0]) if len(sub_1336_all)>2 else 0.0,
                'spearman_rho_sigma_err': float(spearmanr(sub_1336_all['pred_sigma'], sub_1336_all['abs_error'])[0]) if len(sub_1336_all)>2 else 0.0,
            })
            summary_metrics.append({
                'model_source': args.model_source,
                'descriptor_mode': md,
                'scope': 'Anchor_R1336mzz(E)',
                'chemical_class': 'HFO',
                'N': len(sub_1336_e),
                'MAE': m_e.get('mae_clip', 0.0),
                'RMSE': m_e.get('rmse_clip', 0.0),
                'R2': m_e.get('r2_clip', 0.0),
                'mean_sigma': float(sub_1336_e['pred_sigma'].mean()),
                'pearson_r_sigma_err': 0.0,
                'spearman_rho_sigma_err': 0.0,
            })
            summary_metrics.append({
                'model_source': args.model_source,
                'descriptor_mode': md,
                'scope': 'Anchor_R1336mzz(Z)',
                'chemical_class': 'HFO',
                'N': len(sub_1336_z),
                'MAE': m_z.get('mae_clip', 0.0),
                'RMSE': m_z.get('rmse_clip', 0.0),
                'R2': m_z.get('r2_clip', 0.0),
                'mean_sigma': float(sub_1336_z['pred_sigma'].mean()),
                'pearson_r_sigma_err': 0.0,
                'spearman_rho_sigma_err': 0.0,
            })

        # 分物种逐一输出
        for ref_sp, grp in df_mode_pred.groupby('refrigerant'):
            m_sp = compute_metrics(grp['true_x1'].values, grp['pred_mu'].values)
            summary_metrics.append({
                'model_source': args.model_source,
                'descriptor_mode': md,
                'scope': f"Species_{ref_sp}",
                'chemical_class': SPECIES_CLASS.get(ref_sp, 'Other'),
                'N': len(grp),
                'MAE': m_sp['mae_clip'],
                'RMSE': m_sp['rmse_clip'],
                'R2': m_sp['r2_clip'],
                'mean_sigma': float(grp['pred_sigma'].mean()),
                'pearson_r_sigma_err': 0.0,
                'spearman_rho_sigma_err': 0.0,
            })

    # 4. 持久化报告与预测结果
    out_dir = PROJECT_ROOT / 'paper_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    df_preds_all = pd.DataFrame(all_predictions)
    pred_csv = out_dir / f"hfo_zeroshot_predictions_{args.model_source}.csv"
    df_preds_all.to_csv(pred_csv, index=False)
    print(f"\n[Save] Sample predictions saved to: {pred_csv.name} ({len(df_preds_all)} rows)")

    df_summary = pd.DataFrame(summary_metrics)
    summary_csv = out_dir / f"table_hfo_zeroshot_metrics_{args.model_source}.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"[Save] Evaluation summary table saved to: {summary_csv.name}")

    print("\n" + "=" * 80)
    print(f"  HFO ZERO-SHOT PROBE SUMMARY ({args.model_source})")
    print("=" * 80)
    print(df_summary[['descriptor_mode', 'scope', 'N', 'MAE', 'RMSE', 'R2', 'mean_sigma']].to_string(index=False))

    # 生成 JSON 报告
    report_dict = {
        "model_source": args.model_source,
        "n_samples": len(df_hfo),
        "target_scope": args.target_scope,
        "summary": summary_metrics
    }
    json_path = out_dir / f"hfo_zeroshot_report_{args.model_source}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    print(f"[Save] Full JSON report saved to: {json_path.name}")
    print("=" * 80)

if __name__ == '__main__':
    main()
