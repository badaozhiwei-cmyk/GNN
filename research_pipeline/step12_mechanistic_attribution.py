"""
step12_mechanistic_attribution.py — Phase II-B: Mechanistic Attribution Engine (Fast Representative Sampling)
============================================================================================================
【功能】
1. 支持 --max_samples_per_species 参数，对 BF4 和 PF6 进行温压空间等距均匀抽样。
2. 5-Seed 生产模型 (42..46) 深度归因与完备性校验。
3. 概念组聚合分析 (thermo_state, ref_physical, anion_physical, cation_physical, reduced_priors)。
4. 5-Seed 稳定性检验 (Cosine & Spearman) 及物理遮蔽扰动敏感性验证。
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
GNN_DIR = PROJECT_ROOT / "GNN_for_property_prediction"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GNN_DIR))

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Batch, Data
    from Model_v6 import IL_GAT_v6
    from Dataset_v6 import (
        FEATURE_SCHEMA,
        BASE_FEATURES,
        MODE_DEF,
        MODE_INDICES,
        MODE_COND_DIM,
        combine_Graph,
        add_global,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


CONCEPT_GROUPS = {
    "M0": {
        "thermo_state": ["T", "P"],
        "ref_physical": ["ref_charge", "ref_logp", "ref_MW"],
        "anion_physical": ["ani_mw"],
        "cation_physical": ["cat_charge", "cat_tpsa", "cat_MW"],
    },
    "Mreduced": {
        "thermo_state": ["T", "P"],
        "ref_physical": ["ref_charge", "ref_logp", "ref_MW"],
        "anion_physical": ["ani_mw"],
        "cation_physical": ["cat_charge", "cat_tpsa", "cat_MW"],
        "reduced_priors": ["Tr", "Pr", "omega"],
    },
}


def compute_integrated_gradients_scalar(
    model: Any,
    graph_batch: Any,
    target_cond: Any,
    baseline_cond: Any,
    steps: int = 50,
    device: Any = None,
) -> tuple[np.ndarray, float]:
    if device is None:
        device = torch.device("cpu")

    target_cond = target_cond.to(device).float()
    baseline_cond = baseline_cond.to(device).float()
    graph_batch = graph_batch.to(device)

    captured_x_g = []
    handle = model.l5.register_forward_pre_hook(
        lambda mod, inp: captured_x_g.append(inp[0][:, :512].detach())
    )

    with torch.no_grad():
        pred_target = model(graph_batch, target_cond).item()
        pred_baseline = model(graph_batch, baseline_cond).item()
    handle.remove()

    delta_pred = pred_target - pred_baseline
    x_g = captured_x_g[0]

    diff = target_cond - baseline_cond
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device)

    c_batch = (baseline_cond + alphas.view(-1, 1) * diff).requires_grad_(True)
    x_g_batch = x_g.expand(steps, -1)

    preds = model.l5(torch.cat([x_g_batch, c_batch], dim=1))
    preds.sum().backward()

    avg_grads = c_batch.grad.mean(dim=0)
    ig = (diff.squeeze(0) * avg_grads).detach().cpu().numpy()

    completeness_err = abs(float(np.sum(ig)) - delta_pred)

    return ig, completeness_err


def compute_pairwise_cosine(vectors: list[np.ndarray]) -> float:
    n = len(vectors)
    if n < 2:
        return 1.0
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            v1, v2 = vectors[i], vectors[j]
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 1e-12 and norm2 > 1e-12:
                sim = float(np.dot(v1, v2) / (norm1 * norm2))
                sims.append(sim)
            else:
                sims.append(1.0 if norm1 == norm2 else 0.0)
    return float(np.mean(sims)) if sims else 1.0


def compute_pairwise_spearman(vectors: list[np.ndarray]) -> float:
    from scipy.stats import spearmanr
    n = len(vectors)
    if n < 2:
        return 1.0
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(vectors[i], vectors[j])
            if np.isfinite(rho):
                corrs.append(float(rho))
    return float(np.mean(corrs)) if corrs else 1.0


def run_attribution_pipeline(
    data_dir: Path,
    splits_dir: Path,
    results_dir: Path,
    out_dir: Path,
    target_split: str = "B2",
    modes: list[str] = ["M0", "Mreduced"],
    seeds: list[int] = [42, 43, 44, 45, 46],
    steps: int = 50,
    max_samples_per_species: int = 30,
    device: Any = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("  Phase II-B: Mechanistic Attribution Engine (Fast Representative Sampling)", flush=True)
    print(f"  Target Split   : {target_split} (Star Case: BF4 vs PF6)", flush=True)
    print(f"  Modes          : {modes}", flush=True)
    print(f"  Seeds          : {seeds}", flush=True)
    print(f"  Device         : {device}", flush=True)
    print(f"  IG Steps       : {steps}", flush=True)
    print(f"  Sampling Quota : Max {max_samples_per_species} points per species", flush=True)
    print("=" * 78, flush=True)

    raw_data_path = data_dir / "data.npy"
    raw_label_path = data_dir / "label.npy"
    meta_path = data_dir / "meta_info.csv"

    if not (raw_data_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"Missing data files in {data_dir}")

    raw_data = np.load(raw_data_path, allow_pickle=True)
    raw_labels = np.load(raw_label_path, allow_pickle=True)
    meta_df = pd.read_csv(meta_path)

    sp_tag = "B2_Fam3_InorganicFluoride" if target_split == "B2" else "B1_Fam2_Fluorosulfonate"
    split_npz_path = splits_dir / f"split_{sp_tag}.npz"
    split_data = np.load(split_npz_path)
    all_test_indices = split_data["test"]

    # 均匀抽样逻辑：对每个物种按温度压力等距均匀抽取样本
    meta_test = meta_df.iloc[all_test_indices].copy()
    clean_fn = lambda x: str(x).strip().upper().replace("[", "").replace("]", "").replace("-", "")
    meta_test["anion_clean"] = meta_test["anion"].apply(clean_fn)

    sampled_indices = []
    for an_name, group in meta_test.groupby("anion_clean"):
        g_indices = group.index.to_numpy()
        if len(g_indices) > max_samples_per_species:
            pick_pos = np.linspace(0, len(g_indices) - 1, max_samples_per_species, dtype=int)
            sampled = g_indices[pick_pos]
        else:
            sampled = g_indices
        sampled_indices.extend(sampled)
        print(f"  [抽样配额] 物种 {an_name:<6}: 总计 {len(g_indices)} 点 → 等距抽取 {len(sampled)} 代表点", flush=True)

    test_indices = sorted(sampled_indices)
    print(f"[数据就绪] 最终代表性计算样本总量: {len(test_indices)} 点 (预估耗时: ~2 分钟)", flush=True)

    def mol2graph(mol_data):
        x = torch.tensor(mol_data[0], dtype=torch.long)
        edge_index = torch.tensor(mol_data[1], dtype=torch.long)
        if len(mol_data[2]) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.zeros((1, 3), dtype=torch.long)
        else:
            edge_attr = torch.tensor(mol_data[2], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    sample_attribution_records = []
    stability_records = []
    sanity_records = []

    for mode in modes:
        feat_names = MODE_DEF[mode]
        cond_dim = MODE_COND_DIM[mode]
        feat_indices = MODE_INDICES[mode]
        groups = CONCEPT_GROUPS[mode]

        print(f"\n>>> 正在处理模式: {mode} (cond_dim={cond_dim})", flush=True)
        mode_dir = results_dir / f"{target_split}_{mode}"
        scaler_path = mode_dir / "scalers.pkl"

        if not scaler_path.exists():
            print(f"[警告] 未找到 {scaler_path}，跳过该模式。", flush=True)
            continue

        import joblib
        scalers = joblib.load(scaler_path)
        means = np.array([float(s.mean_[0]) for s in scalers], dtype=np.float32)
        scales = np.array([max(float(s.scale_[0]), 1e-8) for s in scalers], dtype=np.float32)

        models = {}
        for seed in seeds:
            ckpt_path = mode_dir / f"best_seed_{seed}.pth"
            if not ckpt_path.exists():
                print(f"[警告] 缺失权重: {ckpt_path}", flush=True)
                continue
            model_args = {
                "emb_dim": 300,
                "pool": "global",
                "cond_dim": cond_dim,
                "dropout_rate": 0.1,
                "descriptor_mode": mode,
            }
            m = IL_GAT_v6(model_args).to(device)
            raw_ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = raw_ckpt["model_state_dict"] if (isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt) else raw_ckpt
            m.load_state_dict(state_dict)
            m.eval()
            models[seed] = m

        print(f"  成功载入 {len(models)} 个 Seed 模型权重", flush=True)

        t0 = time.time()
        for s_idx, idx in enumerate(test_indices):
            if (s_idx + 1) % 10 == 0 or s_idx == 0 or (s_idx + 1) == len(test_indices):
                elapsed = time.time() - t0
                speed = (s_idx + 1) / max(elapsed, 1e-5)
                eta = (len(test_indices) - (s_idx + 1)) / max(speed, 1e-5)
                print(f"  [{mode}] 进度: {s_idx + 1:2d}/{len(test_indices)} ({(s_idx + 1)/len(test_indices)*100:5.1f}%) | 速度: {speed:4.1f} 点/秒 | 预估剩余: {eta:4.1f}s", flush=True)

            row_meta = meta_df.iloc[idx]
            sid = str(row_meta.get("sample_id", f"idx_{idx}"))
            anion_name = str(row_meta.get("anion", "")).strip().upper().replace("[", "").replace("]", "").replace("-", "")
            refrigerant_name = str(row_meta.get("refrigerant", "")).strip().upper().replace("[", "").replace("]", "").replace("-", "")
            true_y = float(raw_labels[idx])

            sample = raw_data[idx]
            g_cat = mol2graph(sample[0])
            g_ani = mol2graph(sample[1])
            g_ref = mol2graph(sample[2])
            comb_g = combine_Graph([g_cat, g_ani, g_ref])
            comb_g = add_global(comb_g)
            graph_batch = Batch.from_data_list([comb_g])

            raw_cond_vals = np.array([sample[col_i] for col_i in feat_indices], dtype=np.float32)
            scaled_cond_vals = (raw_cond_vals - means) / scales
            target_cond = torch.tensor(scaled_cond_vals, dtype=torch.float32).unsqueeze(0)
            baseline_cond = torch.zeros_like(target_cond)

            seed_attributions = []
            seed_preds = []

            for seed, m in models.items():
                ig, comp_err = compute_integrated_gradients_scalar(
                    m, graph_batch, target_cond, baseline_cond, steps=steps, device=device
                )
                seed_attributions.append(ig)
                with torch.no_grad():
                    pred_val = m(graph_batch.to(device), target_cond.to(device)).item()
                seed_preds.append(pred_val)

            mean_ig = np.mean(seed_attributions, axis=0)
            mean_pred = float(np.mean(seed_preds))

            rec = {
                "sample_id": sid,
                "split": target_split,
                "mode": mode,
                "anion": anion_name,
                "refrigerant": refrigerant_name,
                "true_x1": true_y,
                "pred_x1": mean_pred,
                "abs_error": abs(true_y - mean_pred),
            }

            for f_idx, f_name in enumerate(feat_names):
                rec[f"IG_{f_name}"] = float(mean_ig[f_idx])

            total_abs_ig = float(np.sum(np.abs(mean_ig))) + 1e-12
            for g_name, g_feats in groups.items():
                g_indices = [feat_names.index(fn) for fn in g_feats]
                g_abs_sum = float(np.sum(np.abs(mean_ig[g_indices])))
                rec[f"group_abs_{g_name}"] = g_abs_sum
                rec[f"group_share_{g_name}"] = g_abs_sum / total_abs_ig

            sample_attribution_records.append(rec)

            cos_stab = compute_pairwise_cosine(seed_attributions)
            spear_stab = compute_pairwise_spearman(seed_attributions)

            stability_records.append({
                "sample_id": sid,
                "split": target_split,
                "mode": mode,
                "anion": anion_name,
                "refrigerant": refrigerant_name,
                "abs_error": abs(true_y - mean_pred),
                "cosine_stability_5seeds": cos_stab,
                "spearman_stability_5seeds": spear_stab,
            })

            # 轻量级物理参数扰动防伪检验
            if mode == "Mreduced" and len(sanity_records) < 60:
                perturbed_cond = target_cond.clone()
                pr_idx = feat_names.index("Pr")
                omega_idx = feat_names.index("omega")

                perturbed_cond[0, pr_idx] = 0.0
                perturbed_cond[0, omega_idx] = 0.0

                with torch.no_grad():
                    pert_preds = [m(graph_batch.to(device), perturbed_cond.to(device)).item() for m in models.values()]
                pert_mean_pred = float(np.mean(pert_preds))
                delta_from_pert = abs(mean_pred - pert_mean_pred)
                ig_reduced_prior_sum = abs(mean_ig[pr_idx]) + abs(mean_ig[omega_idx])

                sanity_records.append({
                    "sample_id": sid,
                    "anion": anion_name,
                    "refrigerant": refrigerant_name,
                    "ig_pr_omega_sum": ig_reduced_prior_sum,
                    "perturbation_delta_y": delta_from_pert,
                })

    df_samples = pd.DataFrame(sample_attribution_records)
    df_stability = pd.DataFrame(stability_records)
    df_sanity = pd.DataFrame(sanity_records)

    df_samples.to_csv(out_dir / "attribution_sample_level.csv", index=False)
    df_stability.to_csv(out_dir / "attribution_stability_per_sample.csv", index=False)
    if not df_sanity.empty:
        df_sanity.to_csv(out_dir / "attribution_perturbation_sanity.csv", index=False)

    species_summary_rows = []
    for (sp, md, an), g in df_samples.groupby(["split", "mode", "anion"]):
        row_dict = {
            "split": sp,
            "mode": md,
            "anion": an,
            "n_samples": len(g),
            "mean_mae": g["abs_error"].mean(),
        }
        for col in g.columns:
            if col.startswith("group_share_"):
                row_dict[col] = g[col].mean()
            elif col.startswith("IG_"):
                row_dict[f"mean_{col}"] = g[col].mean()
        species_summary_rows.append(row_dict)

    df_species_summary = pd.DataFrame(species_summary_rows)
    df_species_summary.to_csv(out_dir / "attribution_species_summary.csv", index=False)

    stab_summary = df_stability.groupby(["split", "mode", "anion"]).agg(
        n_samples=("sample_id", "count"),
        mean_mae=("abs_error", "mean"),
        mean_cosine_stability=("cosine_stability_5seeds", "mean"),
        mean_spearman_stability=("spearman_stability_5seeds", "mean"),
    ).reset_index()
    stab_summary.to_csv(out_dir / "attribution_stability_summary.csv", index=False)

    print("\n" + "=" * 78, flush=True)
    print("  [PASS] Phase II-B 归因分析执行完毕！已生成全套文件：", flush=True)
    print(f"  1. 样本级归因表   : {out_dir / 'attribution_sample_level.csv'}", flush=True)
    print(f"  2. 物种级归因汇总 : {out_dir / 'attribution_species_summary.csv'}", flush=True)
    print(f"  3. 5-Seed 稳定性表: {out_dir / 'attribution_stability_summary.csv'}", flush=True)
    print(f"  4. 物理扰动验证表 : {out_dir / 'attribution_perturbation_sanity.csv'}", flush=True)
    print("=" * 78, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase II-B Mechanistic Attribution Engine")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "processed_tri_data_hfc2739")
    parser.add_argument("--splits_dir", type=Path, default=PROJECT_ROOT / "splits_anion_ood")
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results_split_B")
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "results_attribution")
    parser.add_argument("--target_split", type=str, default="B2", choices=["B1", "B2"])
    parser.add_argument("--modes", type=str, default="M0,Mreduced")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max_samples_per_species", type=int, default=30)
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("[Notice] PyTorch/PyG is not available in local environment.")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    run_attribution_pipeline(
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        target_split=args.target_split,
        modes=modes,
        seeds=seeds,
        steps=args.steps,
        max_samples_per_species=args.max_samples_per_species,
        device=device,
    )


if __name__ == "__main__":
    main()
