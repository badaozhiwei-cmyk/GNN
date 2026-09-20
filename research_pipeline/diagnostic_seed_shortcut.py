"""
diagnostic_seed_shortcut.py — Diagnostic for GNN Graph vs Scalar Shortcut
========================================================================
检查 5 个 seed 模型的权重范数以及对图嵌入和标量特征的依赖程度。
"""

import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for model_name in ["HFC_all_M0", "B2_M0"]:
    model_dir = ROOT / ("results_hfc_all" if "HFC" in model_name else "results_split_B") / model_name
    print("=" * 80)
    print(f"DIAGNOSTIC: {model_name} (Path: {model_dir})")
    print("=" * 80)
    
    if not model_dir.exists():
        print(f"Directory not found: {model_dir}")
        continue

    for seed in [42, 43, 44, 45, 46]:
        ckpt_p = model_dir / f"best_seed_{seed}.pth"
        if not ckpt_p.exists():
            print(f"Seed {seed}: Checkpoint missing")
            continue
        
        raw_ckpt = torch.load(ckpt_p, map_location="cpu")
        st = raw_ckpt["model_state_dict"] if (isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt) else raw_ckpt
        
        # 提取 l5.0.weight (全连接第一层)
        # 输入维度: 512 (graph embedding) + cond_dim (scalar features)
        w_head = st["l5.0.weight"] # (1024, 512 + cond_dim)
        w_graph = w_head[:, :512]
        w_cond = w_head[:, 512:]
        
        norm_graph = float(torch.norm(w_graph))
        norm_cond = float(torch.norm(w_cond))
        ratio = norm_graph / max(norm_cond, 1e-8)
        
        # 提取 GATv2 第一层与第三层权重范数
        w_l1 = float(torch.norm(st["l1.lin_src.weight"])) if "l1.lin_src.weight" in st else 0.0
        w_l3 = float(torch.norm(st["l3.lin_src.weight"])) if "l3.lin_src.weight" in st else 0.0
        
        # 提取 global_token 范数
        norm_gt = float(torch.norm(st["global_token"])) if "global_token" in st else 0.0
        
        print(f"Seed {seed}:")
        print(f"  Head Weight Norm -> Graph(512d): {norm_graph:.4f} | Cond({w_cond.size(1)}d): {norm_cond:.4f} | Ratio(G/C): {ratio:.4f}")
        print(f"  GNN Layers Norm  -> L1: {w_l1:.4f} | L3: {w_l3:.4f} | global_token: {norm_gt:.4f}")
