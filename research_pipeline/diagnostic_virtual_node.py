"""
diagnostic_virtual_node.py — Rigorous Layer-by-Layer Attention & Representation Audit
=====================================================================================
逐层解构 GATv2 各层中真实原子向全局虚拟节点的信息传递：
1. 真实原子 -> 全局节点注意力总和 vs 全局节点自环 (Self-loop) 注意力
2. 各 GATv2 隐藏层后全局节点表征差量 ||x^(l)_global(target) - x^(l)_global(base)||
3. MLP Head 输入端图表征与标量特征对最终预测的逐级影响
"""

from __future__ import annotations
import sys
import json
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GNN_for_property_prediction"))
sys.path.insert(0, str(ROOT / "research_pipeline"))

from prepare_tri_graph_data_v6 import mol2graph_components
from Dataset_v6 import combine_Graph, add_global, MODE_COND_DIM
from Model_v6 import IL_GAT_v6
from step25_graph_substructure_attribution import (
    extract_node_and_edge_embeddings,
    forward_from_embeddings,
)

def mol2graph(mol_data):
    x = torch.tensor(mol_data[0], dtype=torch.long)
    edge_index = torch.tensor(mol_data[1], dtype=torch.long)
    if len(mol_data[2]) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros((1, 3), dtype=torch.long)
    else:
        edge_attr = torch.tensor(mol_data[2], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def audit_sample_across_seeds(
    sample_row: pd.Series,
    case_label: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ref_smi = str(sample_row["refri_smiles"])
    cat_smi = str(sample_row["cation_smiles"])
    ani_smi = str(sample_row["anion_smiles"])

    cg = mol2graph(mol2graph_components(cat_smi))
    ag = mol2graph(mol2graph_components(ani_smi))
    rg = mol2graph(mol2graph_components(ref_smi))
    comb_g = add_global(combine_Graph([cg, ag, rg]))
    comb_g.batch = torch.zeros(comb_g.x.size(0), dtype=torch.long)

    n_total_nodes = comb_g.x.size(0)
    global_node_idx = n_total_nodes - 1
    normal_node_indices = list(range(global_node_idx))

    # 标量特征构造
    raw_cond = np.array([
        float(sample_row["T_K"]), float(sample_row["P_MPa"]),
        Descriptors.MaxAbsPartialCharge(Chem.MolFromSmiles(ref_smi)),
        Descriptors.MolLogP(Chem.MolFromSmiles(ref_smi)),
        Descriptors.MolWt(Chem.MolFromSmiles(ani_smi)),
        Descriptors.MaxAbsPartialCharge(Chem.MolFromSmiles(cat_smi)),
        Descriptors.TPSA(Chem.MolFromSmiles(cat_smi)),
        Descriptors.MolWt(Chem.MolFromSmiles(ref_smi)),
        Descriptors.MolWt(Chem.MolFromSmiles(cat_smi)),
    ], dtype=np.float32)

    scaler_p = ROOT / sample_row["model_dir"] / "scalers.pkl"
    sc = joblib.load(scaler_p)
    means = np.array([float(s.mean_[0]) for s in sc], dtype=np.float32)
    scales = np.array([max(float(s.scale_[0]), 1e-8) for s in sc], dtype=np.float32)
    scaled_cond = torch.tensor((raw_cond - means) / scales, dtype=torch.float32).unsqueeze(0)

    summary_rows = []
    layer_rows = []

    for seed in [42, 43, 44, 45, 46]:
        ckpt_p = ROOT / sample_row["model_dir"] / f"best_seed_{seed}.pth"
        model_args = {
            "emb_dim": 300,
            "pool": "global",
            "cond_dim": 9,
            "dropout_rate": 0.2,
            "use_layernorm": False,
            "use_adaptive_gate": False,
        }
        model = IL_GAT_v6(model_args)
        raw_ckpt = torch.load(ckpt_p, map_location="cpu")
        st = raw_ckpt["model_state_dict"] if (isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt) else raw_ckpt
        model.load_state_dict(st)
        model.eval()

        h_target, e_target = extract_node_and_edge_embeddings(model, comb_g)
        h_base = torch.zeros_like(h_target)
        is_global = comb_g.mol_type == 3
        h_base[is_global] = model.global_token.detach()
        e_base = torch.zeros_like(e_target)

        with torch.no_grad():
            y_target, _ = forward_from_embeddings(model, h_target, e_target, comb_g, scaled_cond)
            y_base, _ = forward_from_embeddings(model, h_base, e_base, comb_g, scaled_cond)

            # 逐层前向与注意力分解
            def forward_and_capture(h_in, e_in):
                layers_out = []
                attns = []
                x = h_in
                
                # Layer 1
                x, (e_idx_1, a_1) = model.l1(x, comb_g.edge_index, edge_attr=e_in, return_attention_weights=True)
                x = model.act(x)
                layers_out.append(x.clone())
                attns.append((e_idx_1, a_1))
                
                # Layer 2
                x, (e_idx_2, a_2) = model.l2(x, comb_g.edge_index, edge_attr=e_in, return_attention_weights=True)
                x = model.act(x)
                layers_out.append(x.clone())
                attns.append((e_idx_2, a_2))

                # Layer 3
                x, (e_idx_3, a_3) = model.l3(x, comb_g.edge_index, edge_attr=e_in, return_attention_weights=True)
                x = model.act(x)
                layers_out.append(x.clone())
                attns.append((e_idx_3, a_3))

                xg = model.extract(x, comb_g)
                return xg, layers_out, attns

            xg_t, layers_t, attns_t = forward_and_capture(h_target, e_target)
            xg_b, layers_b, attns_b = forward_and_capture(h_base, e_base)

            diff_xg = float(torch.norm(xg_t - xg_b))
            norm_xg = float(torch.norm(xg_t))
            dy = abs(y_target.item() - y_base.item())

            summary_rows.append({
                "case": case_label,
                "sample_id": sample_row["sample_id"],
                "model_dir": sample_row["model_dir"],
                "seed": seed,
                "y_target": float(y_target.item()),
                "y_base": float(y_base.item()),
                "delta_y": float(dy),
                "norm_xg_target": float(norm_xg),
                "diff_xg": float(diff_xg),
                "graph_active": bool(dy > 1e-4),
            })

            # 剖析各层注意力权重分布
            for l_idx, ((edge_idx_l, alpha_l), xt_l, xb_l) in enumerate(zip(attns_t, layers_t, layers_b), start=1):
                # alpha_l 形状: (|E|, heads) -> 跨头求平均
                alpha_mean = alpha_l.mean(dim=1).cpu().numpy()
                src_nodes = edge_idx_l[0].cpu().numpy()
                dst_nodes = edge_idx_l[1].cpu().numpy()

                # 筛选以 global_node_idx 为汇聚目标的边 (dst == global_node_idx)
                in_edges_mask = (dst_nodes == global_node_idx)
                srcs_into_global = src_nodes[in_edges_mask]
                weights_into_global = alpha_mean[in_edges_mask]

                # 自环边 vs 原子入边
                self_loop_mask = (srcs_into_global == global_node_idx)
                atom_edge_mask = (srcs_into_global != global_node_idx)

                self_loop_attn = float(weights_into_global[self_loop_mask].sum()) if self_loop_mask.any() else 0.0
                atom_attn_sum = float(weights_into_global[atom_edge_mask].sum()) if atom_edge_mask.any() else 0.0

                # 该层全局节点的表征差量
                g_node_diff = float(torch.norm(xt_l[global_node_idx] - xb_l[global_node_idx]))
                g_node_norm = float(torch.norm(xt_l[global_node_idx]))

                layer_rows.append({
                    "case": case_label,
                    "seed": seed,
                    "layer": l_idx,
                    "atom_attn_sum": atom_attn_sum,
                    "self_loop_attn": self_loop_attn,
                    "atom_attn_share_pct": (atom_attn_sum / max(atom_attn_sum + self_loop_attn, 1e-12)) * 100,
                    "global_node_diff": g_node_diff,
                    "global_node_norm": g_node_norm,
                })

    return summary_rows, layer_rows


def main():
    print("=" * 90)
    print("  RIGOROUS DIAGNOSTIC: LAYER-BY-LAYER ATTENTION & REPRESENTATION FLOW")
    print("=" * 90)

    manifest_p = ROOT / "results_attribution/case_selection_manifest.csv"
    df_manifest = pd.read_csv(manifest_p)

    out_dir = ROOT / "diagnostic_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_layers = []

    # 1. 评估 Case 1 样本
    c1_row = df_manifest[df_manifest["case_id"] == "Case1_R1234yf"].iloc[0]
    print(f"\n[Case 1: R1234yf on HFC_all_M0] Sample: {c1_row['sample_id']}")
    s1, l1 = audit_sample_across_seeds(c1_row, "Case1_R1234yf")
    all_summaries.extend(s1)
    all_layers.extend(l1)

    # 2. 评估 Case 3 样本
    c3_row = df_manifest[df_manifest["case_id"] == "Case3_BF4_vs_PF6"].iloc[0]
    print(f"\n[Case 3: BF4 vs PF6 on B2_M0] Sample: {c3_row['sample_id']}")
    s3, l3 = audit_sample_across_seeds(c3_row, "Case3_BF4_vs_PF6")
    all_summaries.extend(s3)
    all_layers.extend(l3)

    df_sum = pd.DataFrame(all_summaries)
    df_lay = pd.DataFrame(all_layers)

    # 打印全局表征对比
    print("\n" + "-" * 90)
    print("1. 全局预测差量 (delta_y) 与 图表征范数差量 (diff_xg) 汇总:")
    print("-" * 90)
    print(f"{'Case':<22} {'Seed':<6} {'y_target':<11} {'y_base':<11} {'delta_y':<12} {'diff_xg':<12} {'norm_xg':<12} {'Active?'}")
    print("-" * 90)
    for _, r in df_sum.iterrows():
        act_str = "YES (Deep GNN)" if r["graph_active"] else "NO (Collapsed)"
        print(f"{r['case']:<22} {r['seed']:<6} {r['y_target']:<11.5f} {r['y_base']:<11.5f} {r['delta_y']:<12.5e} {r['diff_xg']:<12.5f} {r['norm_xg_target']:<12.5f} {act_str}")

    # 打印逐层注意力与表征流动
    print("\n" + "-" * 90)
    print("2. GATv2 逐层注意力分配与全局节点差量 (Atom Attn vs Self-Loop Attn):")
    print("-" * 90)
    print(f"{'Case':<20} {'Seed':<6} {'Layer':<6} {'Atom Attn Sum':<16} {'Self-Loop Attn':<16} {'Atom Share %':<14} {'G-Node Diff':<14}")
    print("-" * 90)
    for _, r in df_lay.iterrows():
        print(f"{r['case']:<20} {r['seed']:<6} L{r['layer']:<5} {r['atom_attn_sum']:<16.6f} {r['self_loop_attn']:<16.6f} {r['atom_attn_share_pct']:<14.2f}% {r['global_node_diff']:<14.6f}")

    # 保存结构化文件
    sum_csv = out_dir / "virtual_node_global_summary.csv"
    lay_csv = out_dir / "virtual_node_layerwise_breakdown.csv"
    df_sum.to_csv(sum_csv, index=False)
    df_lay.to_csv(lay_csv, index=False)
    print(f"\n[INFO] 结果已成功输出至:")
    print(f"  - {sum_csv}")
    print(f"  - {lay_csv}")

if __name__ == "__main__":
    main()
