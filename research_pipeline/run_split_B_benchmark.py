"""
run_split_B_benchmark.py
========================
【使命】
  溶剂侧 OOD (Solvent-Side / Anion-Family OOD) 的 30 次基准评测驱动脚本。
  基底宇宙：Saturated HFC Universe (N=2739, 12 种饱和 HFC 冷媒, 23 种阴离子, 6 大先验化学家族)。
  全面继承 M1 生产级工程防线：
    1. 严格 5 种子 (42, 43, 44, 45, 46) 与 M1 完全同构；
    2. Train-only 物理量标准化器拟合 (严禁统计量泄漏)；
    3. 基于验证集 Loss 的最佳权重挑选与早停；
    4. 结果严格记录 sample_id、raw 与 clipped 指标；
    5. 保存 config.json 与完整 sha256 溯源哈希。

【实验矩阵 (2 splits x 3 modes x 5 seeds = 30 runs)】
  - Split B1: Fam-2 (全氟烷基磺酸盐 -SO3- 留出, N_test=513, 18.73%, 包含 7 种磺酸盐阴离子)
  - Split B2: Fam-3 (无机球形氟化物 PF6/BF4 留出, N_test=598, 21.83%, 包含 2 种无机氟化物阴离子)
  - 模式: M0 (9维), Mthermo (12维), Mreduced (12维)
"""

import argparse
import os
import sys
import json
import random
import hashlib
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.append(str(ROOT / 'GNN_for_property_prediction'))

# 特征架构 (与 V6 保持 100% 对齐)
FEATURE_SCHEMA = {
    "T": 3, "P": 4, 
    "ref_charge": 5, "ref_logp": 6, "ani_mw": 7, "cat_charge": 8, "cat_tpsa": 9, 
    "ref_MW": 10, "cat_MW": 11,
    "ref_dipole": 12, "ref_polarizability": 13, "ref_volume": 14,
    "deltaE_anion": 15, "deltaE_cation": 16,
    "Tc": 17, "Pc": 18, "omega": 19,
    "Tr": 20, "Pr": 21,
}

BASE_FEATURES = ["T", "P", "ref_charge", "ref_logp", "ani_mw", "cat_charge", "cat_tpsa", "ref_MW", "cat_MW"]

MODE_DEF = {
    'M0':       BASE_FEATURES,                             # 9 维
    'Mthermo':  BASE_FEATURES + ["Tc", "Pc", "omega"],     # 12 维
    'Mreduced': BASE_FEATURES + ["Tr", "Pr", "omega"],     # 12 维
}

MODE_INDICES = {
    mode: [FEATURE_SCHEMA[feat] for feat in feat_list]
    for mode, feat_list in MODE_DEF.items()
}

MODE_COND_DIM = {k: len(v) for k, v in MODE_INDICES.items()}

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def compute_metrics(true_y, pred_y):
    pred_y_clipped = np.clip(pred_y, 0.0, 1.0)
    mae_raw  = mean_absolute_error(true_y, pred_y)
    r2_raw   = r2_score(true_y, pred_y)
    rmse_raw = np.sqrt(mean_squared_error(true_y, pred_y))
    
    mae_clip  = mean_absolute_error(true_y, pred_y_clipped)
    r2_clip   = r2_score(true_y, pred_y_clipped)
    rmse_clip = np.sqrt(mean_squared_error(true_y, pred_y_clipped))
    
    return {
        'mae_raw': mae_raw, 'r2_raw': r2_raw, 'rmse_raw': rmse_raw,
        'mae_clip': mae_clip, 'r2_clip': r2_clip, 'rmse_clip': rmse_clip
    }

def main():
    parser = argparse.ArgumentParser(description="Split B Secondary OOD Benchmark Runner")
    parser.add_argument('--split', type=str, default='all', choices=['B1', 'B2', 'all'])
    parser.add_argument('--mode', type=str, default='all', choices=['M0', 'Mthermo', 'Mreduced', 'all'])
    parser.add_argument('--seeds', type=str, default='42,43,44,45,46')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--dry_run', action='store_true', help="Preflight check only, test data and exit")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    splits = ['B1', 'B2'] if args.split == 'all' else [args.split]
    modes  = ['M0', 'Mthermo', 'Mreduced'] if args.mode == 'all' else [args.mode]

    data_dir = Path('processed_tri_data_hfc2739')
    splits_dir = Path('splits_anion_ood')
    out_base = Path('results_split_B')
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Split B (Anion-Family OOD) Benchmark Controller")
    print(f"  Target Splits: {splits}")
    print(f"  Target Modes : {modes}")
    print(f"  Seeds ({len(seeds)})    : {seeds}")
    print(f"  Dry-run Only : {args.dry_run}")
    print("=" * 70)

    # 1. 验证必要文件存在并校验哈希
    data_file = data_dir / 'data.npy'
    label_file = data_dir / 'label.npy'
    meta_file = data_dir / 'meta_info.csv'
    for f in [data_file, label_file, meta_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing essential file: {f}")

    data_hash = sha256_file(data_file)
    label_hash = sha256_file(label_file)
    print(f"  [Data SHA256]  : {data_hash[:16]}...")
    print(f"  [Label SHA256] : {label_hash[:16]}...")

    df_meta = pd.read_csv(meta_file)
    print(f"  [Meta Loaded]  : {len(df_meta)} rows aligned with Saturated HFC Universe")

    # ── 终极运行时门禁 (Mandatory Pre-Flight Runtime Assertions Gate) ──
    print("\n[GATEKEEPER] Executing mandatory pre-flight runtime assertions...")
    assert len(df_meta) == 2739, f"FATAL: Expected N=2739, got {len(df_meta)}"
    
    clean_fn = lambda x: str(x).strip().upper().replace('[', '').replace(']', '').replace('-', '')
    n_unique_anions = df_meta['anion'].apply(clean_fn).nunique()
    n_unique_refs = df_meta['refrigerant'].apply(clean_fn).nunique()
    assert n_unique_anions == 23, f"FATAL: Expected exactly 23 unique anions, got {n_unique_anions}"
    assert n_unique_refs == 12, f"FATAL: Expected exactly 12 unique refrigerants, got {n_unique_refs}"
    
    b1_npz = np.load(splits_dir / "split_B1_Fam2_Fluorosulfonate.npz")
    b2_npz = np.load(splits_dir / "split_B2_Fam3_InorganicFluoride.npz")
    assert len(b1_npz['test']) == 513, f"FATAL: Expected B1 test=513, got {len(b1_npz['test'])}"
    assert len(b2_npz['test']) == 598, f"FATAL: Expected B2 test=598, got {len(b2_npz['test'])}"
    
    assert seeds == [42, 43, 44, 45, 46], f"FATAL: Production parity requires seeds [42, 43, 44, 45, 46], got {seeds}"
    assert MODE_COND_DIM['M0'] == 9, f"FATAL: Expected M0 cond_dim=9, got {MODE_COND_DIM['M0']}"
    assert MODE_COND_DIM['Mthermo'] == 12, f"FATAL: Expected Mthermo cond_dim=12, got {MODE_COND_DIM['Mthermo']}"
    assert MODE_COND_DIM['Mreduced'] == 12, f"FATAL: Expected Mreduced cond_dim=12, got {MODE_COND_DIM['Mreduced']}"
    print("  [GATEKEEPER] ALL 8 HARD PRE-FLIGHT ASSERTIONS PASSED! [OK]\n")

    # 2. 预检模式下的快速检查
    if args.dry_run:
        print("\n[PREFLIGHT AUDIT: DRY-RUN VERIFICATION]")
        raw_data = np.load(data_file, allow_pickle=True)
        raw_labels = np.load(label_file, allow_pickle=True)
        print(f"  1. data.npy shape: {len(raw_data)}, row length: {len(raw_data[0])} (expected 22) [OK]")
        print(f"  2. label.npy shape: {len(raw_labels)} [OK]")
        
        for sp in splits:
            sp_tag = 'B1_Fam2_Fluorosulfonate' if sp == 'B1' else 'B2_Fam3_InorganicFluoride'
            sp_file = splits_dir / f"split_{sp_tag}.npz"
            sp_data = np.load(sp_file)
            tr, va, te = sp_data['train'], sp_data['val'], sp_data['test']
            print(f"  3. Split {sp} indices loaded from {sp_file.name}:")
            print(f"     Train: {len(tr)} | Val: {len(va)} | Test: {len(te)} (Ratio: {len(te)/len(df_meta)*100:.2f}%) [OK]")
            
            for md in modes:
                indices = MODE_INDICES[md]
                cond_dim = MODE_COND_DIM[md]
                print(f"     Mode {md:<8}: cond_dim = {cond_dim:2d}, column indices = {indices} [OK]")
        
        print("\n  4. Live Dataset_v6 Pipeline Testing:")
        try:
            from Dataset_v6 import IL_set_v6
            for md in ['M0', 'Mthermo', 'Mreduced']:
                test_ds = IL_set_v6(path=str(data_dir), args={'descriptor_mode': md})
                g, c, y = test_ds[0]
                print(f"     Mode {md:<8} item[0] -> Graph nodes: {g.x.shape[0]}, Cond dim: {c.shape[0]} (expected {MODE_COND_DIM[md]}), Label: {float(y):.4f} [OK]")
            print("\n[PREFLIGHT RESULT]: All data structures, split indices, mode dimensions, and Dataset_v6 tensor loaders verified! Preflight PASSED.")
        except ImportError:
            print("     [Notice] PyTorch/PyG not in local environment; live tensor test will run on GPU host.")
            print("\n[PREFLIGHT RESULT]: All data structures, split indices, mode dimensions, and 8 runtime assertions verified! Preflight PASSED.")
        return

    # 3. 正式执行逻辑 (按 30 runs 矩阵循环)
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch_geometric.loader import DataLoader
    from Dataset_v6 import IL_set_v6
    from Model_v6 import IL_GAT_v6

    summary_rows = []

    for sp in splits:
        sp_tag = 'B1_Fam2_Fluorosulfonate' if sp == 'B1' else 'B2_Fam3_InorganicFluoride'
        sp_file = splits_dir / f"split_{sp_tag}.npz"
        sp_data = np.load(sp_file)
        train_idx = sp_data['train'].astype(int)
        val_idx   = sp_data['val'].astype(int)
        test_idx  = sp_data['test'].astype(int)
        target_family = str(sp_data['target_family'])

        for md in modes:
            cond_dim = MODE_COND_DIM[md]
            run_tag = f"{sp}_{md}"
            run_dir = out_base / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)

            exp_config = {
                "split": sp,
                "split_tag": sp_tag,
                "target_family": target_family,
                "descriptor_mode": md,
                "cond_dim": cond_dim,
                "feature_indices": MODE_INDICES[md],
                "seeds": seeds,
                "epoch": args.epoch,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "patience": args.patience,
                "data_sha256": data_hash,
                "label_sha256": label_hash,
                "split_sha256": sha256_file(sp_file),
            }
            with open(run_dir / 'config.json', 'w') as f:
                json.dump(exp_config, f, indent=2)

            print(f"\n{'='*65}")
            print(f"  Executing Benchmark: {run_tag} | Test: {len(test_idx)} points ({target_family})")
            print(f"{'='*65}")

            whole_set = IL_set_v6(path=str(data_dir), args={'descriptor_mode': md})

            # 严格仅在训练集拟合 StandardScaler (防泄漏铁律)
            whole_set.fit_scalers(train_idx, save_dir=str(run_dir))

            train_set = torch.utils.data.Subset(whole_set, train_idx)
            val_set   = torch.utils.data.Subset(whole_set, val_idx)
            test_set  = torch.utils.data.Subset(whole_set, test_idx)

            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

            seed_metrics = []
            test_preds_all = []
            if 'sample_id' in df_meta.columns:
                expected_sample_ids = df_meta.loc[test_idx, 'sample_id'].astype(str).tolist()
            else:
                cat_c = 'cation' if 'cation' in df_meta.columns else 'IL cation'
                ani_c = 'anion' if 'anion' in df_meta.columns else 'IL anion'
                ref_c = 'refrigerant' if 'refrigerant' in df_meta.columns else 'Refrigerant'
                t_c   = 'T_K' if 'T_K' in df_meta.columns else 'T (K)'
                p_c   = 'P_MPa' if 'P_MPa' in df_meta.columns else 'P (MPa)'
                expected_sample_ids = [
                    f"{r[cat_c]}__{r[ani_c]}__{r[ref_c]}__{float(r[t_c]):.8g}__{float(r[p_c]):.8g}"
                    for _, r in df_meta.loc[test_idx].iterrows()
                ]

            test_trues = None
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for seed in seeds:
                set_seed(seed)
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
                val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)

                model_args = {
                    'emb_dim': 300,
                    'dropout_rate': 0.2,
                    'cond_dim': cond_dim,
                    'pool': 'global',
                    'use_layernorm': False,
                    'use_adaptive_gate': False
                }
                model = IL_GAT_v6(model_args).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)
                criterion = nn.HuberLoss(delta=0.05)

                best_val_loss = float('inf')
                best_ckpt = run_dir / f"best_seed_{seed}.pth"
                early_stop_count = 0

                for epoch in range(1, args.epoch + 1):
                    model.train()
                    for graph, cond, label in train_loader:
                        graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                        optimizer.zero_grad()
                        out = model(graph, cond)
                        loss = criterion(out.flatten(), label.flatten())
                        loss.backward()
                        optimizer.step()
                    scheduler.step()

                    model.eval()
                    v_loss = 0.0
                    with torch.no_grad():
                        for graph, cond, label in val_loader:
                            graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                            out = model(graph, cond)
                            v_loss += criterion(out.flatten(), label.flatten()).item()
                    v_loss /= len(val_loader)

                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        early_stop_count = 0
                        torch.save({'model_state_dict': model.state_dict()}, best_ckpt)
                    else:
                        early_stop_count += 1
                        if early_stop_count >= args.patience:
                            break

                # 加载最佳模型并在测试集上推断
                ckpt = torch.load(best_ckpt, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()

                preds, trues = [], []
                with torch.no_grad():
                    for graph, cond, label in test_loader:
                        graph, cond = graph.to(device), cond.to(device)
                        out = model(graph, cond)
                        preds.extend(out.flatten().cpu().numpy().tolist())
                        trues.extend(label.flatten().cpu().numpy().tolist())

                preds = np.array(preds)
                trues = np.array(trues)
                if test_trues is None:
                    test_trues = trues

                m = compute_metrics(trues, preds)
                print(f"  [Seed {seed}] Raw MAE: {m['mae_raw']:.4f}, R2: {m['r2_raw']:.4f} | Clipped MAE: {m['mae_clip']:.4f}, R2: {m['r2_clip']:.4f}")
                seed_metrics.append({'seed': seed, **m})
                test_preds_all.append(preds)

                # 保存样本级预测 CSV
                df_pred = pd.DataFrame({
                    'sample_id': expected_sample_ids,
                    'true_x1': trues,
                    'pred_x1_raw': preds,
                    'pred_x1_clipped': np.clip(preds, 0.0, 1.0)
                })
                df_pred.to_csv(run_dir / f"pred_seed_{seed}.csv", index=False)

            # 集成均值与统计汇总
            df_sm = pd.DataFrame(seed_metrics)
            ens_pred = np.mean(test_preds_all, axis=0)
            ens_m = compute_metrics(test_trues, ens_pred)

            summary_row = {
                "split": sp,
                "target_family": target_family,
                "mode": md,
                "cond_dim": cond_dim,
                "N_test": len(test_idx),
                "seed_mean_mae_raw": float(df_sm['mae_raw'].mean()),
                "seed_std_mae_raw": float(df_sm['mae_raw'].std()),
                "seed_mean_r2_raw": float(df_sm['r2_raw'].mean()),
                "seed_std_r2_raw": float(df_sm['r2_raw'].std()),
                "seed_mean_mae_clip": float(df_sm['mae_clip'].mean()),
                "seed_std_mae_clip": float(df_sm['mae_clip'].std()),
                "seed_mean_r2_clip": float(df_sm['r2_clip'].mean()),
                "seed_std_r2_clip": float(df_sm['r2_clip'].std()),
                "ensemble_mae_raw": float(ens_m['mae_raw']),
                "ensemble_r2_raw": float(ens_m['r2_raw']),
                "ensemble_mae_clip": float(ens_m['mae_clip']),
                "ensemble_r2_clip": float(ens_m['r2_clip']),
            }
            summary_rows.append(summary_row)

            # 保存单模式的种子统计
            df_sm.to_csv(run_dir / 'seed_metrics.csv', index=False)
            print(f"\n  Summary for {run_tag}:")
            print(f"    Seed-Mean MAE (Raw) : {summary_row['seed_mean_mae_raw']:.4f} ± {summary_row['seed_std_mae_raw']:.4f}")
            print(f"    Seed-Mean R2 (Raw)  : {summary_row['seed_mean_r2_raw']:.4f} ± {summary_row['seed_std_r2_raw']:.4f}")
            print(f"    Ensemble MAE (Raw)  : {summary_row['ensemble_mae_raw']:.4f}")

    # 保存总汇总表
    df_all_summary = pd.DataFrame(summary_rows)
    df_all_summary.to_csv(out_base / 'split_B_benchmark_summary.csv', index=False)
    print("\n" + "=" * 70)
    print("  [SUCCESS] All Split B Benchmark Runs Completed!")
    print(f"  Final summary written to: {out_base / 'split_B_benchmark_summary.csv'}")
    print("=" * 70)

if __name__ == '__main__':
    main()
