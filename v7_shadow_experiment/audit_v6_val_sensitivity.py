"""
audit_v6_val_sensitivity.py — Audit V6-M0 Sensitivity on Exact 274 Validation Samples
=====================================================================================
Strictly align evaluation universe:
Computes Graph-Active Rate (|Δy_graph| >= 1e-3), Median Δy_graph, and Median Δy_desc
for V6-M0 across all 5 seeds (42, 43, 44, 45, 46) on the exact same 274 validation samples.
Uses forward hook on model.l5 to guarantee 100% exact reproduction of official forward pass.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'GNN_for_property_prediction'))

from Dataset_v6 import IL_set_v6
from Model_v6 import IL_GAT_v6


def main():
    print("=" * 80)
    print("  AUDITING V6-M0 COUNTERFACTUAL SENSITIVITY ON EXACT 274 VAL SAMPLES")
    print("=" * 80)

    data_dir = ROOT / 'processed_tri_data_hfc2739'
    split_file = ROOT / 'splits' / 'HFC_all_split.npz'
    v6_dir = ROOT / 'results_hfc_all' / 'HFC_all_M0'

    sp_data = np.load(split_file)
    train_idx = sp_data['train'].astype(int)
    val_idx = sp_data['val'].astype(int)
    print(f"Validation sample count: {len(val_idx)} (Index disjoint with Train: {len(train_idx)})")

    # Load V6 Dataset
    whole_set = IL_set_v6(path=str(data_dir), args={'descriptor_mode': 'M0'})
    # Load scalers strictly from existing v6_dir without refitting
    whole_set.fit_scalers(train_idx, save_dir=str(v6_dir))
    val_set = torch.utils.data.Subset(whole_set, val_idx)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    seeds = [42, 43, 44, 45, 46]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Audit device: {device}\n")

    results = []

    for seed in seeds:
        ckpt_path = v6_dir / f"best_seed_{seed}.pth"
        if not ckpt_path.exists():
            print(f"❌ Missing checkpoint: {ckpt_path}")
            continue

        model_args = {
            'emb_dim': 300,
            'dropout_rate': 0.2,
            'cond_dim': 9,
            'pool': 'global',
            'use_layernorm': False,
            'use_adaptive_gate': False
        }
        model = IL_GAT_v6(model_args).to(device)
        raw_ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = raw_ckpt['model_state_dict'] if 'model_state_dict' in raw_ckpt else raw_ckpt
        model.load_state_dict(state_dict)
        model.eval()

        delta_y_graphs = []
        delta_y_descs = []
        all_preds = []
        all_targets = []

        # Hook to capture exact input to self.l5 (x_concat)
        captured_x_concat = []

        def hook_fn(module, input):
            captured_x_concat.append(input[0])

        hook_handle = model.l5.register_forward_pre_hook(hook_fn)

        with torch.no_grad():
            for graph, cond, label in val_loader:
                graph = graph.to(device)
                cond = cond.to(device)
                label = label.to(device)

                captured_x_concat.clear()
                # 1. Official forward pass
                out = model(graph, cond).flatten()
                x_concat = captured_x_concat[0]  # shape: (batch, 512 + 9)

                # Split graph representation and condition
                x_g = x_concat[:, :512]
                c_val = x_concat[:, 512:]

                # 2. Counterfactual zero-graph prediction
                x_concat_zero_g = torch.cat([torch.zeros_like(x_g), c_val], dim=1)
                out_zero_g = model.l5(x_concat_zero_g).flatten()
                dy_g = torch.abs(out - out_zero_g).cpu().numpy()
                delta_y_graphs.extend(dy_g)

                # 3. Counterfactual zero-descriptor prediction (keep T, P at indices 0, 1; zero out 2..8)
                c_val_zero_d = c_val.clone()
                c_val_zero_d[:, 2:] = 0.0
                x_concat_zero_d = torch.cat([x_g, c_val_zero_d], dim=1)
                out_zero_d = model.l5(x_concat_zero_d).flatten()
                dy_d = torch.abs(out - out_zero_d).cpu().numpy()
                delta_y_descs.extend(dy_d)

                all_preds.extend(out.cpu().numpy())
                all_targets.extend(label.cpu().numpy())

        hook_handle.remove()

        dy_g_arr = np.array(delta_y_graphs)
        dy_d_arr = np.array(delta_y_descs)
        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)

        val_mae = float(np.mean(np.abs(preds_arr - targets_arr)))
        active_rate = float(np.mean(dy_g_arr >= 1e-3))
        med_dy_g = float(np.median(dy_g_arr))
        med_dy_d = float(np.median(dy_d_arr))

        res = {
            'seed': seed,
            'val_mae': val_mae,
            'active_rate_pct': active_rate * 100.0,
            'active_count': int(np.sum(dy_g_arr >= 1e-3)),
            'total_samples': len(dy_g_arr),
            'median_delta_y_graph': med_dy_g,
            'median_delta_y_desc': med_dy_d,
        }
        results.append(res)
        print(f"Seed {seed:02d} | Val MAE: {val_mae:.4f} | "
              f"Graph-Active: {res['active_count']}/{res['total_samples']} ({active_rate*100:.1f}%) | "
              f"Median Δy_graph: {med_dy_g:.5e} | Median Δy_desc: {med_dy_d:.5f}")

    df_res = pd.DataFrame(results)
    out_csv = ROOT / "v7_shadow_experiment" / "v6_val274_sensitivity_audit.csv"
    df_res.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print("  AUDIT COMPLETE: Unified V6-M0 Baseline on Exact 274 Validation Samples")
    print("=" * 80)
    print(df_res.to_string(index=False))


if __name__ == '__main__':
    main()
