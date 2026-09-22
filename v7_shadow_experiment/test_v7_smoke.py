"""
test_v7_smoke.py — End-to-End Smoke Test for V7 Architecture
============================================================
Verifies:
1. Decoupled data loading via Dataset_v7 and DataLoader_v7.
2. Forward pass for V7-A (dropout=0.0) and V7-B (dropout=0.3).
3. Gradient flow across all components (Local IL/Ref GATs, Inter-MPNN, Readout, Head).
4. Parameter count audit.
"""

import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v7_shadow_experiment"))

from Dataset_v7 import DecoupledTriDataset_v7, get_v7_dataloader
from Model_v7 import IL_GAT_v7

def run_smoke_test():
    print("=" * 70)
    print("【V7 SHADOW EXPERIMENT: END-TO-END SMOKE TEST】")
    print("=" * 70)
    
    # 1. Dataset Loading
    data_root = ROOT / "processed_tri_data_hfc2739"
    split_path = ROOT / "splits" / "HFC_all_split.npz"
    
    print("[1/5] Initializing DecoupledTriDataset_v7...")
    ds = DecoupledTriDataset_v7(str(data_root))
    print(f"      Total dataset size: {len(ds)} samples")
    
    # Fit scalers on first 50 samples for smoke testing
    ds.fit_scalers(train_indices=list(range(min(50, len(ds)))))
    loader = get_v7_dataloader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    
    print(f"      Batch Cat: Nodes={batch['cat'].x.size(0)}, Edges={batch['cat'].edge_index.size(1)}")
    print(f"      Batch Ani: Nodes={batch['ani'].x.size(0)}, Edges={batch['ani'].edge_index.size(1)}")
    print(f"      Batch Ref: Nodes={batch['ref'].x.size(0)}, Edges={batch['ref'].edge_index.size(1)}, EdgeAttr={batch['ref'].edge_attr.size()}")
    print(f"      State Cond (T, P)  : Shape={batch['state'].shape}")
    print(f"      Desc Cond (7 Descs): Shape={batch['desc'].shape}")
    print(f"      Labels             : Shape={batch['label'].shape}")
    
    # 2. Model V7-A (Architecture only, desc_dropout=0.0)
    print("\n[2/5] Instantiating V7-A (desc_dropout_p=0.0)...")
    model_a = IL_GAT_v7({"emb_dim": 300, "hidden_dim": 512, "desc_dropout_p": 0.0})
    model_a.train()
    out_a = model_a(batch)
    print(f"      Forward success! Output shape: {out_a.shape}, values: {out_a.detach().numpy()}")
    
    # 3. Model V7-B (Architecture + ModDrop 0.3)
    print("\n[3/5] Instantiating V7-B (desc_dropout_p=0.3)...")
    model_b = IL_GAT_v7({"emb_dim": 300, "hidden_dim": 512, "desc_dropout_p": 0.3})
    model_b.train()
    out_b = model_b(batch)
    print(f"      Forward success! Output shape: {out_b.shape}, values: {out_b.detach().numpy()}")
    
    # Eval mode & override mask check
    model_b.eval()
    with torch.no_grad():
        out_eval = model_b(batch)
        out_masked = model_b(batch, override_mask_desc=0.0)
    print(f"      Eval mode output: {out_eval.numpy()}")
    print(f"      Masked desc output (perturbed): {out_masked.numpy()}")
    delta_desc = (out_eval - out_masked).abs().mean().item()
    print(f"      Mean absolute delta upon zeroing descriptors: {delta_desc:.5f}")
    
    # 4. Backward Pass & Gradient Flow
    print("\n[4/5] Testing Gradient Flow on V7-B...")
    model_b.train()
    criterion = torch.nn.MSELoss()
    pred = model_b(batch)
    loss = criterion(pred, batch['label'])
    loss.backward()
    
    grad_checks = {
        'Shared IL GAT l1': model_b.il_l1.lin_l.weight.grad is not None,
        'Solute Ref GAT l1': model_b.ref_l1.lin_l.weight.grad is not None,
        'Inter-MPNN Conv': model_b.inter_conv.lin_l.weight.grad is not None,
        'Inter-Edge Embed': model_b.inter_edge_embed.weight.grad is not None,
        'System Projection': model_b.sys_proj[0].weight.grad is not None,
        'Head Linear 1': model_b.head[0].weight.grad is not None,
    }
    all_grads_ok = all(grad_checks.values())
    for k, ok in grad_checks.items():
        print(f"      - {k:20s}: Gradient Active? {'[PASS]' if ok else '[FAIL]'}")
    assert all_grads_ok, "Critical: Some layer gradients are missing!"
    
    # 5. Parameter Count
    print("\n[5/5] Parameter Count Audit...")
    total_params = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
    print(f"      Total Trainable Parameters: {total_params / 1e6:.2f} M ({total_params:,} parameters)")
    
    print("\n" + "=" * 70)
    print("【SMOKE TEST RESULT: 100% PASSED! V7 ARCHITECTURE READY!】")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_test()
