"""
run_r1234yf_graph_ig.py — 针对 4 个 R1234yf 样本的专用 Graph-IG 重新归因脚本
=============================================================================
覆盖 Case 1 的 4 个代表性测试点:
  - [emim][Ac] + R1234yf
  - [emim][BF4] + R1234yf
  - [bmim][Ac] + R1234yf
  - [bmim][PF6] + R1234yf
在 2 个模型家族 (V7-A, V7-B) × 5 个官方随机种子 (42, 43, 44, 45, 46) 下执行。
总计 40 次精确积分梯度计算 (Riemann Steps = 50)。
"""

import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# 统一编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v7_shadow_experiment"))
sys.path.insert(0, str(ROOT / "Phase4_Scientific_Validation"))

from artifact_freshness_gate import validate_dataset_freshness
from phase2_v7_graph_ig import (
    get_checkpoint_dir,
    load_sample_from_manifest_row,
    compute_v7_graph_ig,
    match_smarts_hierarchy,
    IL_GAT_v7
)

def main():
    print("=" * 80)
    print("  RUNNING TARGETED GRAPH-IG ON 4 CORRECTED R1234YF SAMPLES (40 EVALS)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Device: {device}")

    # 1. 强契约校验
    validate_dataset_freshness(ROOT / "datasets" / "hfc_2739_v7", expected_dataset_id="HFC_2739_V7_CANONICAL")
    validate_dataset_freshness(ROOT / "datasets" / "full_4444_v7", expected_dataset_id="FULL_4444_V7_CANONICAL")
    print("✓ Dataset Freshness Gates Passed.")

    # 2. 读取数据与清单
    manifest_p = ROOT / "results_attribution" / "case_selection_manifest.csv"
    df_manifest = pd.read_csv(manifest_p)
    raw_hfc_data = np.load(ROOT / "datasets" / "hfc_2739_v7" / "data.npy", allow_pickle=True)
    raw_full_data = np.load(ROOT / "datasets" / "full_4444_v7" / "data.npy", allow_pickle=True)

    # 筛选 4 个 R1234yf 样本
    r1234_manifest = df_manifest[df_manifest['refrigerant'] == 'R1234yf'].copy()
    assert len(r1234_manifest) == 4, f"Expected 4 R1234yf cases, got {len(r1234_manifest)}"
    print(f"✓ Found {len(r1234_manifest)} target R1234yf samples in manifest.")

    models = ["V7-A", "V7-B"]
    seeds = [42, 43, 44, 45, 46]
    steps = 50

    out_records = []
    t_start = time.time()
    counter = 0
    total = len(models) * len(seeds) * len(r1234_manifest)

    for model_family in models:
        ckpt_dir = get_checkpoint_dir(model_family)
        print(f"\n📂 [{model_family}] 载入权重目录: {ckpt_dir}")
        scaler_p = ckpt_dir / "scalers.pkl"
        scalers = joblib.load(scaler_p)
        scaler_means = np.array([float(s.mean_[0]) for s in scalers], dtype=np.float32)
        scaler_scales = np.array([max(float(s.scale_[0]), 1e-8) for s in scalers], dtype=np.float32)

        for seed in seeds:
            ckpt_p = ckpt_dir / f"best_seed_{seed}.pth"
            raw_ckpt = torch.load(ckpt_p, map_location=device)
            model_args = raw_ckpt["model_args"]
            model = IL_GAT_v7(model_args).to(device)
            st = raw_ckpt['model_state_dict'] if isinstance(raw_ckpt, dict) and 'model_state_dict' in raw_ckpt else raw_ckpt
            model.load_state_dict(st)
            model.eval()

            for m_idx, row in r1234_manifest.iterrows():
                counter += 1
                t0 = time.time()
                smp = load_sample_from_manifest_row(
                    row, raw_hfc_data, raw_full_data, scaler_means, scaler_scales, device
                )

                ig_res = compute_v7_graph_ig(
                    model, smp["g_cat"], smp["g_ani"], smp["g_ref"],
                    smp["state_t"], smp["desc_t"], steps=steps
                )

                # 提取各组件与主要基团贡献
                cat_abs = float(np.sum(ig_res["cat"]["atom_abs"]))
                ani_abs = float(np.sum(ig_res["ani"]["atom_abs"]))
                ref_abs = float(np.sum(ig_res["ref"]["atom_abs"]))
                tot_abs = cat_abs + ani_abs + ref_abs

                _, _, ref_groups = match_smarts_hierarchy(smp["mol_ref"])
                cf3_atoms = ref_groups.get("CF3", [])
                alkene_atoms = ref_groups.get("Alkene_C=C", [])

                cf3_signed = float(np.sum([ig_res["ref"]["atom_signed"][a] for a in cf3_atoms])) if cf3_atoms else 0.0
                cf3_abs = float(np.sum([ig_res["ref"]["atom_abs"][a] for a in cf3_atoms])) if cf3_atoms else 0.0
                alkene_signed = float(np.sum([ig_res["ref"]["atom_signed"][a] for a in alkene_atoms])) if alkene_atoms else 0.0
                alkene_abs = float(np.sum([ig_res["ref"]["atom_abs"][a] for a in alkene_atoms])) if alkene_atoms else 0.0

                elapsed = time.time() - t0
                print(f"[{counter:02d}/{total}] {model_family} S{seed} | {row['sample_id']} | "
                      f"DeltaY: {ig_res['pred_delta']:.4f} | CF3: {cf3_signed:+.4f} | C=C: {alkene_signed:+.4f} | {elapsed:.2f}s")

                out_records.append({
                    "model": model_family,
                    "seed": seed,
                    "sample_id": row["sample_id"],
                    "cation": row["cation"],
                    "anion": row["anion"],
                    "refrigerant": row["refrigerant"],
                    "T_K": row["T_K"],
                    "P_MPa": row["P_MPa"],
                    "true_x1": row["x1"],
                    "y_target": ig_res["pred_raw"],
                    "y_base": ig_res["pred_base"],
                    "pred_delta": ig_res["pred_delta"],
                    "comp_abs_err": ig_res["comp_abs_err"],
                    "comp_rel_err": ig_res["comp_rel_err"],
                    "cat_abs_sum": cat_abs,
                    "ani_abs_sum": ani_abs,
                    "ref_abs_sum": ref_abs,
                    "cf3_signed": cf3_signed,
                    "cf3_abs": cf3_abs,
                    "alkene_signed": alkene_signed,
                    "alkene_abs": alkene_abs,
                })

    df_out = pd.DataFrame(out_records)
    out_csv = ROOT / "results_attribution" / "r1234yf_corrected_attributions.csv"
    df_out.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print(f"✅ 成功完成全部 40 次 R1234yf 干净图归因计算！总耗时: {(time.time() - t_start):.1f}s")
    print(f"📁 结果已保存至: {out_csv}")
    print("=" * 80)

    # 统计汇总
    print("\n【R1234yf 各模型归因统计摘要】")
    summary = df_out.groupby("model").agg({
        "cf3_signed": ["mean", "std"],
        "alkene_signed": ["mean", "std"],
        "ref_abs_sum": "mean",
        "ani_abs_sum": "mean",
        "cat_abs_sum": "mean",
        "comp_rel_err": "mean"
    })
    print(summary)

if __name__ == '__main__':
    main()
