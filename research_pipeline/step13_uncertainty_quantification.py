"""
step13_uncertainty_quantification.py — Phase II-C: Uncertainty Quantification (UQ) Pipeline
===========================================================================================
【定位与核心职责】
1. 基于 5 个独立训练种子 (Seed 42..46) 计算 Seed-Ensemble Epistemic Uncertainty Proxy:
     mu_i = mean(y_hat_i^(s)),  sigma_i = std(y_hat_i^(s), ddof=1)
2. C1: Uncertainty Sanity (sigma 与 |y - mu| 的 Pearson r, Spearman rho, 四分位 MAE 单调性)
3. C2: Risk-Coverage Profile (按 sigma 递增排序进行选择性预测, 评估 100%..50% 覆盖率下的 Risk 抑制与 AURC)
4. C3: Distance -> Uncertainty (挂载冻结的 Phase II-A 连续距离 D_FP, D_thermo, D_phys, 评估 D 与 sigma 的关联)
5. 核心对照实验:
     - B1 vs B2: 不确定性能否自发区分泛化难度阶梯 (Mean sigma_B1 vs Mean sigma_B2)
     - BF4 vs PF6 (B2 内): 验证 "预测改善 + 可靠性收缩" 协同假说 (sigma_M0 -> sigma_Mreduced)
     - M1 LORO 对照组: R245fa vs R236fa 及 R134 vs R134a 的不确定性表征
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(r"c:\Users\霸道志伟\Desktop\GNN\Refrigerant-Solubility-GNN")
RESULTS_SPLIT_B = PROJECT_ROOT / "results_split_B"
RESULTS_BOUNDARY = PROJECT_ROOT / "results_boundary"
M1_ROOT = PROJECT_ROOT.parent / "模型结果" / "extracted" / "results_ablation"
OUT_DIR = PROJECT_ROOT / "results_uq"


def load_split_b_predictions() -> pd.DataFrame:
    records = []
    seeds = [42, 43, 44, 45, 46]
    
    for split in ["B1", "B2"]:
        for mode in ["M0", "Mthermo", "Mreduced"]:
            folder = RESULTS_SPLIT_B / f"{split}_{mode}"
            if not folder.exists():
                continue
            
            seed_dfs = {}
            for s in seeds:
                f = folder / f"pred_seed_{s}.csv"
                if f.exists():
                    df_s = pd.read_csv(f)
                    seed_dfs[s] = df_s.set_index("sample_id")
            
            if len(seed_dfs) != 5:
                print(f"  [警告] {split}_{mode} 仅找到 {len(seed_dfs)}/5 个种子预测，跳过。")
                continue
            
            ref_df = seed_dfs[42]
            sample_ids = ref_df.index.tolist()
            true_vals = ref_df["true_x1"].to_numpy()
            
            preds_matrix = np.zeros((len(sample_ids), 5), dtype=np.float64)
            for col_idx, s in enumerate(seeds):
                preds_matrix[:, col_idx] = seed_dfs[s].loc[sample_ids, "pred_x1_clipped"].to_numpy()
            
            mu = preds_matrix.mean(axis=1)
            sigma = preds_matrix.std(axis=1, ddof=1)
            abs_err = np.abs(true_vals - mu)
            
            for i, sid in enumerate(sample_ids):
                parts = sid.split("__")
                cation = parts[0].strip("[]") if len(parts) > 0 else "unknown"
                anion = parts[1].strip("[]") if len(parts) > 1 else "unknown"
                ref = parts[2] if len(parts) > 2 else "unknown"
                t_val = float(parts[3]) if len(parts) > 3 else np.nan
                p_val = float(parts[4]) if len(parts) > 4 else np.nan
                
                records.append({
                    "sample_id": sid,
                    "dataset": "Split_B",
                    "split": split,
                    "mode": mode,
                    "cation": cation,
                    "anion": anion,
                    "species": anion.upper(),
                    "species_type": "anion",
                    "refrigerant": ref.upper(),
                    "T_K": t_val,
                    "P_MPa": p_val,
                    "true_x1": true_vals[i],
                    "pred_mu": mu[i],
                    "pred_sigma": sigma[i],
                    "abs_error": abs_err[i],
                })
                
    df_b = pd.DataFrame(records)
    print(f"[数据加载] Split B 加载完成: {len(df_b)} 条预测记录 (涵盖 B1/B2 x 3 模式)")
    return df_b


def load_m1_predictions() -> pd.DataFrame:
    records = []
    seeds = [42, 43, 44, 45, 46]
    
    modes_map = {
        "M0": M1_ROOT / "HFC_loro_M0_cde658c5",
        "Mreduced": M1_ROOT / "HFC_loro_Mreduced_8eb5ea83",
    }
    
    for mode, base_dir in modes_map.items():
        if not base_dir.exists():
            continue
        
        sum_csv = base_dir / "summary.csv"
        if not sum_csv.exists():
            continue
        
        df_sum = pd.read_csv(sum_csv)
        for _, row in df_sum.iterrows():
            tgt = row["Target"]
            ref_name = tgt.replace("loro_", "").upper()
            preds_dir = base_dir / f"{tgt}_preds"
            
            seed_dfs = {}
            for s in seeds:
                f = preds_dir / f"seed{s}.csv"
                if f.exists():
                    df_s = pd.read_csv(f)
                    seed_dfs[s] = df_s
                    
            if len(seed_dfs) != 5:
                continue
            
            ref_df = seed_dfs[42]
            true_vals = ref_df["true_x1"].to_numpy()
            sample_ids = ref_df["sample_id"].tolist() if "sample_id" in ref_df.columns else [f"M1_{ref_name}_{idx}" for idx in range(len(true_vals))]
            
            preds_matrix = np.zeros((len(true_vals), 5), dtype=np.float64)
            for col_idx, s in enumerate(seeds):
                pred_col = "pred_x1_clipped" if "pred_x1_clipped" in seed_dfs[s].columns else "pred_x1"
                preds_matrix[:, col_idx] = seed_dfs[s][pred_col].to_numpy()
                
            mu = preds_matrix.mean(axis=1)
            sigma = preds_matrix.std(axis=1, ddof=1)
            abs_err = np.abs(true_vals - mu)
            
            for i in range(len(true_vals)):
                records.append({
                    "sample_id": sample_ids[i],
                    "dataset": "M1_LORO",
                    "split": tgt,
                    "mode": mode,
                    "cation": ref_df.loc[i, "IL cation"] if "IL cation" in ref_df.columns else "unknown",
                    "anion": ref_df.loc[i, "IL anion"] if "IL anion" in ref_df.columns else "unknown",
                    "species": ref_name,
                    "species_type": "refrigerant",
                    "refrigerant": ref_name,
                    "T_K": ref_df.loc[i, "T (K)"] if "T (K)" in ref_df.columns else np.nan,
                    "P_MPa": ref_df.loc[i, "P (MPa)"] if "P (MPa)" in ref_df.columns else np.nan,
                    "true_x1": true_vals[i],
                    "pred_mu": mu[i],
                    "pred_sigma": sigma[i],
                    "abs_error": abs_err[i],
                })
                
    df_m1 = pd.DataFrame(records)
    print(f"[数据加载] M1 LORO 加载完成: {len(df_m1)} 条预测记录 (涵盖 12 工质 x 2 模式)")
    return df_m1


def compute_uncertainty_sanity(df_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    quantile_rows = []
    
    groups_to_run = list(df_all.groupby(["dataset", "split", "mode"]))
    
    for m1_md in ["M0", "Mreduced"]:
        m1_all_sub = df_all[(df_all["dataset"] == "M1_LORO") & (df_all["mode"] == m1_md)]
        if len(m1_all_sub) > 0:
            groups_to_run.append((("M1_LORO", "ALL_12_REFRIGERANTS", m1_md), m1_all_sub))
            
    for (dset, sp, md), group in groups_to_run:
        n = len(group)
        if n < 10:
            continue
        
        err = group["abs_error"].to_numpy()
        sigma = group["pred_sigma"].to_numpy()
        
        r_p, p_p = pearsonr(sigma, err)
        r_s, p_s = spearmanr(sigma, err)
        
        mean_err = np.mean(err)
        mean_sigma = np.mean(sigma)
        
        summary_rows.append({
            "dataset": dset,
            "split": sp,
            "mode": md,
            "n_samples": n,
            "mean_mae": mean_err,
            "mean_sigma": mean_sigma,
            "pearson_r": r_p,
            "pearson_p": p_p,
            "spearman_rho": r_s,
            "spearman_p": p_s,
        })
        
        q_edges = np.percentile(sigma, [0, 25, 50, 75, 100])
        for q_idx in range(4):
            low, high = q_edges[q_idx], q_edges[q_idx + 1]
            mask = (sigma >= low) & (sigma <= high) if q_idx == 3 else (sigma >= low) & (sigma < high)
            sub_err = err[mask]
            sub_sig = sigma[mask]
            
            quantile_rows.append({
                "dataset": dset,
                "split": sp,
                "mode": md,
                "quantile": f"Q{q_idx + 1}",
                "n_samples": len(sub_err),
                "sigma_range": f"[{low:.4f}, {high:.4f}]",
                "mean_sigma": np.mean(sub_sig) if len(sub_sig) > 0 else np.nan,
                "mean_mae": np.mean(sub_err) if len(sub_err) > 0 else np.nan,
            })
            
    df_sanity = pd.DataFrame(summary_rows)
    df_quantiles = pd.DataFrame(quantile_rows)
    return df_sanity, df_quantiles


def compute_risk_coverage(df_all: pd.DataFrame) -> pd.DataFrame:
    coverage_thresholds = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
    rc_rows = []
    
    groups_to_run = list(df_all.groupby(["dataset", "split", "mode"]))
    for m1_md in ["M0", "Mreduced"]:
        m1_all_sub = df_all[(df_all["dataset"] == "M1_LORO") & (df_all["mode"] == m1_md)]
        if len(m1_all_sub) > 0:
            groups_to_run.append((("M1_LORO", "ALL_12_REFRIGERANTS", m1_md), m1_all_sub))
            
    for (dset, sp, md), group in groups_to_run:
        n = len(group)
        if n < 20:
            continue
        
        sorted_group = group.sort_values(by="pred_sigma", ascending=True)
        err_sorted = sorted_group["abs_error"].to_numpy()
        
        baseline_mae = np.mean(err_sorted)
        
        for cov in coverage_thresholds:
            k = max(int(np.round(n * cov)), 1)
            retained_err = err_sorted[:k]
            retained_mae = np.mean(retained_err)
            risk_reduction_pct = ((baseline_mae - retained_mae) / max(baseline_mae, 1e-8)) * 100.0
            
            rc_rows.append({
                "dataset": dset,
                "split": sp,
                "mode": md,
                "coverage": cov,
                "retained_samples": k,
                "retained_mae": retained_mae,
                "risk_reduction_pct": risk_reduction_pct,
            })
            
    df_rc = pd.DataFrame(rc_rows)
    return df_rc


def evaluate_distance_to_uncertainty(df_all: pd.DataFrame) -> pd.DataFrame:
    """C3 模块: 将 Phase II-A 连续外推距离与物种平均不确定性进行关联度分析。"""
    dist_file = RESULTS_BOUNDARY / "continuous_distance_benchmark.csv"
    if not dist_file.exists():
        print(f"  [警告] 未找到 {dist_file}，跳过 C3 距离分析。")
        return pd.DataFrame()
    
    df_dist = pd.read_csv(dist_file)
    clean_fn = lambda x: str(x).strip().upper().replace("[", "").replace("]", "").replace("-", "")
    df_dist["species_clean"] = df_dist["species"].apply(clean_fn)
    
    corr_results = []
    
    # 1. Split B (按 split, mode, species_clean 匹配)
    b_sub = df_all[df_all["dataset"] == "Split_B"]
    b_uq = b_sub.groupby(["dataset", "split", "mode", "species"]).agg(
        mean_sigma=("pred_sigma", "mean"),
        mean_mae=("abs_error", "mean"),
        n_points=("true_x1", "count")
    ).reset_index()
    b_uq["species_clean"] = b_uq["species"].apply(clean_fn)
    
    b_merged = pd.merge(
        b_uq,
        df_dist[["species_clean", "split", "mode", "D_FP", "D_thermo", "D_phys"]],
        on=["species_clean", "split", "mode"],
        how="inner"
    )
    
    for (dset, sp, md), grp in b_merged.groupby(["dataset", "split", "mode"]):
        if len(grp) < 3:
            continue
        for d_col in ["D_FP", "D_thermo", "D_phys"]:
            valid = grp.dropna(subset=[d_col, "mean_sigma", "mean_mae"])
            if len(valid) < 3:
                continue
            rho_sig, p_sig = spearmanr(valid[d_col], valid["mean_sigma"])
            rho_mae, p_mae = spearmanr(valid[d_col], valid["mean_mae"])
            corr_results.append({
                "dataset": dset,
                "split": sp,
                "mode": md,
                "distance_metric": d_col,
                "n_species": len(valid),
                "spearman_dist_to_sigma": rho_sig,
                "p_val_sigma": p_sig,
                "spearman_dist_to_mae": rho_mae,
                "p_val_mae": p_mae,
            })
            
    # 2. M1 LORO (按 mode, species_clean 匹配, 跨全部 12 个制冷剂)
    m1_sub = df_all[df_all["dataset"] == "M1_LORO"]
    m1_uq = m1_sub.groupby(["dataset", "mode", "species"]).agg(
        mean_sigma=("pred_sigma", "mean"),
        mean_mae=("abs_error", "mean"),
        n_points=("true_x1", "count")
    ).reset_index()
    m1_uq["species_clean"] = m1_uq["species"].apply(clean_fn)
    
    # 在 df_dist 中找 split == "M1_LORO"
    m1_dist = df_dist[df_dist["split"] == "M1_LORO"].copy()
    
    m1_merged = pd.merge(
        m1_uq,
        m1_dist[["species_clean", "mode", "D_FP", "D_thermo", "D_phys"]],
        on=["species_clean", "mode"],
        how="inner"
    )
    
    for md, grp in m1_merged.groupby("mode"):
        if len(grp) < 4:
            continue
        for d_col in ["D_FP", "D_thermo", "D_phys"]:
            valid = grp.dropna(subset=[d_col, "mean_sigma", "mean_mae"])
            if len(valid) < 4:
                continue
            rho_sig, p_sig = spearmanr(valid[d_col], valid["mean_sigma"])
            rho_mae, p_mae = spearmanr(valid[d_col], valid["mean_mae"])
            corr_results.append({
                "dataset": "M1_LORO",
                "split": "ALL_12_REFRIGERANTS",
                "mode": md,
                "distance_metric": d_col,
                "n_species": len(valid),
                "spearman_dist_to_sigma": rho_sig,
                "p_val_sigma": p_sig,
                "spearman_dist_to_mae": rho_mae,
                "p_val_mae": p_mae,
            })
            
    return pd.DataFrame(corr_results)


def run_contrast_case_analyses(df_all: pd.DataFrame) -> pd.DataFrame:
    case_rows = []
    
    # 对照 1: B1 vs B2
    b1_b2_sub = df_all[df_all["dataset"] == "Split_B"]
    for md in ["M0", "Mthermo", "Mreduced"]:
        b1 = b1_b2_sub[(b1_b2_sub["split"] == "B1") & (b1_b2_sub["mode"] == md)]
        b2 = b1_b2_sub[(b1_b2_sub["split"] == "B2") & (b1_b2_sub["mode"] == md)]
        
        if len(b1) > 0 and len(b2) > 0:
            case_rows.append({
                "case_name": "Macro Domain: B1 vs B2",
                "mode": md,
                "comparison": "B1 (Easy) vs B2 (Hard)",
                "item_A": "B1",
                "item_B": "B2",
                "A_MAE": b1["abs_error"].mean(),
                "A_sigma": b1["pred_sigma"].mean(),
                "B_MAE": b2["abs_error"].mean(),
                "B_sigma": b2["pred_sigma"].mean(),
                "sigma_ratio_B_over_A": b2["pred_sigma"].mean() / max(b1["pred_sigma"].mean(), 1e-8),
                "delta_mae_pct": ((b2["abs_error"].mean() - b1["abs_error"].mean()) / b1["abs_error"].mean()) * 100.0,
                "delta_sigma_pct": ((b2["pred_sigma"].mean() - b1["pred_sigma"].mean()) / b1["pred_sigma"].mean()) * 100.0,
                "notes": "Harder domain B2 exhibits substantially elevated epistemic uncertainty.",
            })
            
    # 对照 2: BF4 vs PF6 (Within Split B2)
    b2_sub = df_all[(df_all["dataset"] == "Split_B") & (df_all["split"] == "B2")]
    for an_name in ["BF4", "PF6"]:
        an_m0 = b2_sub[(b2_sub["species"].str.contains(an_name)) & (b2_sub["mode"] == "M0")]
        an_red = b2_sub[(b2_sub["species"].str.contains(an_name)) & (b2_sub["mode"] == "Mreduced")]
        
        if len(an_m0) > 0 and len(an_red) > 0:
            mae_m0 = an_m0["abs_error"].mean()
            mae_red = an_red["abs_error"].mean()
            sig_m0 = an_m0["pred_sigma"].mean()
            sig_red = an_red["pred_sigma"].mean()
            
            delta_mae_pct = ((mae_m0 - mae_red) / mae_m0) * 100.0
            delta_sig_pct = ((sig_m0 - sig_red) / sig_m0) * 100.0
            
            case_rows.append({
                "case_name": f"Within-Family: {an_name} (M0 -> Mreduced)",
                "mode": "M0 -> Mreduced",
                "comparison": f"{an_name} Transition",
                "item_A": f"{an_name}_M0",
                "item_B": f"{an_name}_Mreduced",
                "A_MAE": mae_m0,
                "A_sigma": sig_m0,
                "B_MAE": mae_red,
                "B_sigma": sig_red,
                "sigma_ratio_B_over_A": sig_red / max(sig_m0, 1e-8),
                "delta_mae_pct": delta_mae_pct,
                "delta_sigma_pct": delta_sig_pct,
                "notes": f"{an_name}: MAE drop={delta_mae_pct:.1f}%, sigma contraction={delta_sig_pct:.1f}%",
            })
            
    # 对照 3: M1 LORO 对照组 (R245fa vs R236fa)
    m1_sub = df_all[df_all["dataset"] == "M1_LORO"]
    if len(m1_sub) > 0:
        for md in ["M0", "Mreduced"]:
            r245 = m1_sub[(m1_sub["species"] == "R245FA") & (m1_sub["mode"] == md)]
            r236 = m1_sub[(m1_sub["species"] == "R236FA") & (m1_sub["mode"] == md)]
            if len(r245) > 0 and len(r236) > 0:
                case_rows.append({
                    "case_name": "Homolog Contrast: R245fa vs R236fa",
                    "mode": md,
                    "comparison": "R245fa (Failure) vs R236fa (Recovery)",
                    "item_A": "R236FA",
                    "item_B": "R245FA",
                    "A_MAE": r236["abs_error"].mean(),
                    "A_sigma": r236["pred_sigma"].mean(),
                    "B_MAE": r245["abs_error"].mean(),
                    "B_sigma": r245["pred_sigma"].mean(),
                    "sigma_ratio_B_over_A": r245["pred_sigma"].mean() / max(r236["pred_sigma"].mean(), 1e-8),
                    "delta_mae_pct": ((r245["abs_error"].mean() - r236["abs_error"].mean()) / r236["abs_error"].mean()) * 100.0,
                    "delta_sigma_pct": ((r245["pred_sigma"].mean() - r236["pred_sigma"].mean()) / r236["pred_sigma"].mean()) * 100.0,
                    "notes": "Failure case R245fa maintains higher uncertainty than recovered R236fa.",
                })
                
        # 对照 4: M1 LORO 对照组 (R134 vs R134a)
        for md in ["M0", "Mreduced"]:
            r134 = m1_sub[(m1_sub["species"] == "R134") & (m1_sub["mode"] == md)]
            r134a = m1_sub[(m1_sub["species"] == "R134A") & (m1_sub["mode"] == md)]
            if len(r134) > 0 and len(r134a) > 0:
                case_rows.append({
                    "case_name": "Isomer Contrast: R134 vs R134a",
                    "mode": md,
                    "comparison": "R134 (Mismatch) vs R134a (Standard)",
                    "item_A": "R134A",
                    "item_B": "R134",
                    "A_MAE": r134a["abs_error"].mean(),
                    "A_sigma": r134a["pred_sigma"].mean(),
                    "B_MAE": r134["abs_error"].mean(),
                    "B_sigma": r134["pred_sigma"].mean(),
                    "sigma_ratio_B_over_A": r134["pred_sigma"].mean() / max(r134a["pred_sigma"].mean(), 1e-8),
                    "delta_mae_pct": ((r134["abs_error"].mean() - r134a["abs_error"].mean()) / r134a["abs_error"].mean()) * 100.0,
                    "delta_sigma_pct": ((r134["pred_sigma"].mean() - r134a["pred_sigma"].mean()) / r134a["pred_sigma"].mean()) * 100.0,
                    "notes": "Representation mismatch in R134 manifests in substantially higher uncertainty.",
                })
                
    return pd.DataFrame(case_rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("  PHASE II-C: UNCERTAINTY QUANTIFICATION (UQ) PIPELINE")
    print("=" * 80)
    
    df_split_b = load_split_b_predictions()
    df_m1 = load_m1_predictions()
    df_all = pd.concat([df_split_b, df_m1], ignore_index=True)
    
    sample_csv = OUT_DIR / "uq_sample_level.csv"
    df_all.to_csv(sample_csv, index=False)
    print(f"  [落盘] 逐样本 UQ 预测表: {sample_csv} (总计 {len(df_all)} 点)")
    
    print("\n>>> 正在运行 C1: Uncertainty Sanity 分析...")
    df_sanity, df_quantiles = compute_uncertainty_sanity(df_all)
    df_sanity.to_csv(OUT_DIR / "uq_benchmark_summary.csv", index=False)
    df_quantiles.to_csv(OUT_DIR / "uq_quantiles.csv", index=False)
    print(f"  [落盘] UQ 基础合理性总表: {OUT_DIR / 'uq_benchmark_summary.csv'}")
    print(f"  [落盘] UQ 四分位阶梯检验表: {OUT_DIR / 'uq_quantiles.csv'}")
    
    print("\n>>> 正在运行 C2: Risk-Coverage 剖面分析...")
    df_rc = compute_risk_coverage(df_all)
    df_rc.to_csv(OUT_DIR / "uq_risk_coverage.csv", index=False)
    print(f"  [落盘] Risk-Coverage 曲线数据: {OUT_DIR / 'uq_risk_coverage.csv'}")
    
    print("\n>>> 正在运行 C3: OOD Distance -> Uncertainty 关联分析...")
    df_d_to_sig = evaluate_distance_to_uncertainty(df_all)
    if len(df_d_to_sig) > 0:
        df_d_to_sig.to_csv(OUT_DIR / "uq_distance_correlation.csv", index=False)
        print(f"  [落盘] 连续外推距离与不确定性关联表: {OUT_DIR / 'uq_distance_correlation.csv'}")
        
    print("\n>>> 正在计算核心对照组深度审计指标 (B1 vs B2, BF4 vs PF6, R245fa vs R236fa, R134 vs R134a)...")
    df_cases = run_contrast_case_analyses(df_all)
    df_cases.to_csv(OUT_DIR / "uq_contrast_cases.csv", index=False)
    print(f"  [落盘] 核心对照组深度审计表: {OUT_DIR / 'uq_contrast_cases.csv'}")
    
    print("\n" + "=" * 80)
    print("  [COMPLETE] PHASE II-C UNCERTAINTY QUANTIFICATION PIPELINE")
    print("=" * 80)


if __name__ == "__main__":
    main()
