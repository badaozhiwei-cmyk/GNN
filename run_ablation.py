"""
run_ablation.py — GNN 消融实验驱动器
====================================
【目的】
  在 HFC 家族上使用 LORO 切分，对比 4 种描述符模式的 GNN 模型性能。
  证明 3D 物理描述符（尤其是偶极矩）对溶解度预测的贡献。

【四个消融模型】
  M0    : 原始 7 维 RDKit 基线
  Msize : 7 + 2 = 9 维 (+ ref_MolWt, cat_MolWt)
  Mmu   : 7 + 1 = 8 维 (+ ref_dipole)
  Mphys : 7 + 3 = 10 维 (+ ref_dipole, ref_polarizability, ref_volume)

【用法】
  # 单模式运行
  python run_ablation.py --family HFC --mode loro --descriptor_mode Mmu --seeds 3

  # 快速测试
  python run_ablation.py --family HFC --mode loro --descriptor_mode M0 --seeds 1 --epoch 2

【注意】
  必须先运行 prepare_tri_graph_data_v3.py 生成 processed_tri_data_v3/ 目录下的数据。
  原始 v5 代码（run_experiment.py, Dataset_v5, Model_v5）未做任何修改。
"""
import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import sys
import pathlib as pl
import hashlib
import json
import subprocess

# 确保能找到子模块
current_script_dir = str(pl.Path(__file__).resolve().parent)
sys.path.append(os.path.join(current_script_dir, 'GNN_for_property_prediction'))

from Dataset_v6 import IL_set_v6, MODE_COND_DIM, BASE_FEATURES
from GAT_Runner_v5 import set_seed  # set_seed 通用，不需要新版

# ============================================================
# Runner (复用 v5 的训练逻辑，但使用 v6 Model)
# ============================================================
from Model_v6 import IL_GAT_v6
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_or_unknown():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=current_script_dir,
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


class AblationRunner:
    """消融实验专用 Runner，使用 Model_v6 (动态 cond_dim)"""
    def __init__(self, args, seed=42, save_dir="checkpoints_ablation"):
        self.args = args
        self.seed = seed
        self.save_dir = save_dir
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = IL_GAT_v6(args).to(self._device)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=args['lr'],
            weight_decay=args['weight_decay']
        )
        self._scheduler = CosineAnnealingLR(self._optimizer, T_max=args['epoch'], eta_min=1e-5)
        self._criterion = nn.HuberLoss(delta=0.05)

    def _save(self, title):
        os.makedirs(self.save_dir, exist_ok=True)
        path = f"{self.save_dir}/{title}_seed_{self.seed}.pth"
        torch.save({'model_state_dict': self._model.state_dict()}, path)

    def _load_best(self):
        path = f"{self.save_dir}/best_seed_{self.seed}.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Best checkpoint missing; refusing to evaluate last-epoch weights: {path}")
        ckpt = torch.load(path, map_location=self._device)
        self._model.load_state_dict(ckpt['model_state_dict'])

    def train(self, train_loader, dev_loader):
        from GAT_Runner_v5 import EarlyStopping
        early_stopping = EarlyStopping(patience=self.args['patience'])
        best_v_loss = float('inf')

        for epoch in range(1, self.args['epoch'] + 1):
            self._model.train()
            train_loss = 0.0

            bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False,
                       desc=f"Epoch {epoch:>3d}")
            for graph, cond, label in train_loader:
                graph = graph.to(self._device)
                cond = cond.to(self._device)
                label = label.to(self._device)
                self._optimizer.zero_grad()
                y = self._model(graph, cond)
                loss = self._criterion(y.flatten(), label.flatten())
                loss.backward()
                self._optimizer.step()
                train_loss += loss.item()
                bar.update()
            bar.close()

            self._scheduler.step()

            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for graph, cond, label in dev_loader:
                    graph = graph.to(self._device)
                    cond = cond.to(self._device)
                    label = label.to(self._device)
                    y = self._model(graph, cond)
                    val_loss += self._criterion(y.flatten(), label.flatten()).item()

            if len(train_loader.dataset) == 0 or len(dev_loader.dataset) == 0:
                raise RuntimeError("Empty train/validation loader; cannot train or early-stop safely")
            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss / len(dev_loader)

            if avg_val < best_v_loss:
                best_v_loss = avg_val
                self._save('best')

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:>3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

            early_stopping(avg_val)
            if early_stopping.early_stop:
                print(f"  ⏹ Early stopping at epoch {epoch}")
                break

        return best_v_loss

    def predict(self, loader):
        self._load_best()
        self._model.eval()
        raw_pred_y, true_y = [], []
        all_gate_vals = []
        with torch.no_grad():
            for graph, cond, label in loader:
                graph = graph.to(self._device)
                cond = cond.to(self._device)
                label = label.to(self._device)
                pred = self._model(graph, cond)
                pred_vals = pred.flatten().cpu().numpy()
                raw_pred_y.extend(pred_vals.tolist())
                true_y.extend(label.cpu().numpy().tolist())
                # 收集门控激活值（如果启用了自适应门控）
                if hasattr(self._model, '_gate_values') and self._model._gate_values is not None:
                    all_gate_vals.append(self._model._gate_values.cpu().numpy())
        # 存储门控值供后续可解释性分析（不修改返回签名）
        self._last_gate_values = np.vstack(all_gate_vals) if all_gate_vals else None
        return np.asarray(raw_pred_y), np.asarray(true_y)

    def test(self, test_loader):
        raw_pred_y, true_y = self.predict(test_loader)
        pred_y = np.clip(np.asarray(raw_pred_y), 0.0, 1.0)
        mae = mean_absolute_error(true_y, pred_y)
        r2  = r2_score(true_y, pred_y)
        print(f"  Seed {self.seed} -> GNN R²: {r2:.4f}, MAE: {mae:.4f}")
        return np.asarray(raw_pred_y), np.asarray(true_y)


# ============================================================
# 制冷剂家族映射表
# ============================================================
FAMILY_MAP = {
    'R32': 'HFC', 'R152a': 'HFC', 'R134a': 'HFC', 'R125': 'HFC', 'R143a': 'HFC',
    'R23': 'HFC', 'R41': 'HFC', 'R161': 'HFC', 'R134': 'HFC', 'R227ea': 'HFC',
    'R236fa': 'HFC', 'R245fa': 'HFC', 'R365mfc': 'HFC', 'R236ea': 'HFC',
    'R116': 'FC', 'R14': 'FC', 'R218': 'FC',
    'R1234yf': 'HFO', 'R1234ze(E)': 'HFO', 'R1233zd(E)': 'HFO', 'R1243zf': 'HFO',
    'R22': 'HCFC', 'R124': 'HCFC', 'R123': 'HCFC', 'R142b': 'HCFC', 'R141b': 'HCFC',
    'R22B1': 'HBFC', 'R11': 'CFC', 'R12': 'CFC', 'R13': 'CFC'
}


def main():
    parser = argparse.ArgumentParser(description="GNN Ablation Study Runner (v6)")
    parser.add_argument("--family", type=str, required=True,
                        choices=['HFC', 'HFO', 'ALL'],
                        help="Chemical family to filter")
    parser.add_argument("--mode", type=str, required=True,
                        choices=['random', 'loro'],
                        help="Split mode")
    parser.add_argument("--descriptor_mode", type=str, required=True,
                        choices=['M0', 'Msize', 'Mmu', 'Malpha', 'MV', 'Mphys', 'Mthermo', 'Mreduced', 'Mreduced_pure', 'M_interact', 'M_all'],
                        help="Ablation descriptor mode")
    parser.add_argument("--data_dir", type=str, default="processed_tri_data_v6",
                        help="Path to preprocessed tri-graph dataset directory (default: processed_tri_data_v6)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--epoch", type=int, default=80,
                        help="Max epochs (default: 80)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64, recommend 128 for CPU)")
    parser.add_argument("--target_ref", type=str, default=None,
                        help="Optional: Run only a single specific refrigerant (e.g. R134a)")

    # ── 第一招：组件级交互池化 ──
    parser.add_argument("--use_interaction", action="store_true",
                        help="启用 IL×Ref 组件级交互池化（h_il*h_ref + h_il-h_ref）")

    # ── 第二招：低成本修复三件套 ──
    parser.add_argument("--use_sigmoid", action="store_true",
                        help="输出层加 sigmoid，物理约束预测值在 [0,1]")
    parser.add_argument("--use_cond_dropout", action="store_true",
                        help="条件特征 Dropout，破坏偶极矩等标量的捷径学习")
    parser.add_argument("--cond_dropout_p", type=float, default=0.3,
                        help="条件特征 Dropout 概率")
    parser.add_argument("--use_layernorm", action="store_true",
                        help="MLP Head 使用 LayerNorm 替代 BatchNorm")
    parser.add_argument("--use_rf_blend", action="store_true",
                        help="启用 RF+GNN 验证集自适应动态加权融合 (自动防发散气囊)")
    # ── 第三招：自适应门控 ──
    parser.add_argument("--use_adaptive_gate", action="store_true",
                        help="启用自适应门控物理描述符注入 (Adaptive Gated Descriptor Injection)")
    parser.add_argument("--gate_init_bias", type=float, default=-2.0,
                        help="门控偏置初始值 (default: -2.0, σ(-2)≈0.12)")
    parser.add_argument("--feature_clip", type=float, default=None,
                        help="可选：将标准化条件特征裁剪到 [-X, X]；默认不裁剪")

    args = parser.parse_args()
    if args.feature_clip is not None and args.feature_clip <= 0:
        parser.error('--feature_clip must be positive')
    # CPU 多核极致并行提速
    if not torch.cuda.is_available():
        torch.set_num_threads(4)

    # ============================================================
    # 1. 加载数据 (严格指定数据目录)
    # ============================================================
    data_path = os.path.join(current_script_dir, args.data_dir)
    if not os.path.exists(os.path.join(data_path, 'data.npy')):
        if os.path.exists(os.path.join(args.data_dir, 'data.npy')):
            data_path = args.data_dir
        else:
            raise FileNotFoundError(
                f"找不到 {data_path}/data.npy！\n"
                f"请先运行: python prepare_tri_graph_data_v3.py"
            )

    meta_csv = os.path.join(data_path, 'index_with_anion.csv')
    if not os.path.exists(meta_csv):
        meta_csv = os.path.join(data_path, 'meta_info.csv')
    df_raw = pd.read_csv(meta_csv)

    ref_col = 'refrigerant' if 'refrigerant' in df_raw.columns else 'Refrigerant'
    if 'family' not in df_raw.columns:
        df_raw['family'] = df_raw[ref_col].map(FAMILY_MAP)

    # 家族过滤
    df = df_raw.copy()
    if args.family != 'ALL':
        df = df[df['family'] == args.family]
        if len(df) == 0:
            raise ValueError(f"No samples found for family {args.family}")

    gate_tag = " + AdaptiveGate" if args.use_adaptive_gate else ""
    print(f"\n{'='*60}")
    print(f"  GNN Ablation Study")
    print(f"  Family: {args.family} | Mode: {args.mode}")
    print(f"  Descriptor: {args.descriptor_mode} (cond_dim={MODE_COND_DIM[args.descriptor_mode]}){gate_tag}")
    print(f"  Seeds: {args.seeds} | Epochs: {args.epoch} | Batch: {args.batch_size}")
    print(f"  Samples after filtering: {len(df)}")
    print(f"{'='*60}\n")

    # ============================================================
    # 2. 构建 Model Args（传入 cond_dim）
    # ============================================================
    cond_dim = MODE_COND_DIM[args.descriptor_mode]

    model_args = {
        'data_path': data_path,
        'batch_size': args.batch_size,
        'lr': 0.001,
        'epoch': args.epoch,
        'weight_decay': 1e-6,
        'emb_dim': 300,
        'dropout_rate': 0.2,
        'patience': args.patience,
        'pool': 'global',
        'add_global': True,
        'no_mol_embedding': False,
        'descriptor_mode': args.descriptor_mode,
        'cond_dim': cond_dim,
        # ── 增强开关 ──
        'use_interaction': args.use_interaction,
        'use_sigmoid': args.use_sigmoid,
        'use_cond_dropout': args.use_cond_dropout,
        'cond_dropout_p': args.cond_dropout_p,
        'use_layernorm': args.use_layernorm,
        # ── 第三招：自适应门控 ──
        'use_adaptive_gate': args.use_adaptive_gate,
        'base_feature_names': BASE_FEATURES,  # 动态传给 Model，彻底消除魔法常数
        'gate_init_bias': args.gate_init_bias,
        'feature_clip': args.feature_clip,
    }

    print("Loading Graph Dataset (v6)...")
    Whole_set = IL_set_v6(path=model_args['data_path'], args=model_args)

    # 验证数据集大小与 CSV 对齐
    assert len(df_raw) == len(Whole_set), \
        f"Dataset length mismatch! CSV: {len(df_raw)} vs PyG: {len(Whole_set)}"
    # 2b. 实验配置哈希（防止不同配置复用旧预测）
    # ============================================================
    exp_config = {
        "schema_version": 2,
        "family": args.family,
        "mode": args.mode,
        "descriptor_mode": args.descriptor_mode,
        "use_interaction": args.use_interaction,
        "use_sigmoid": args.use_sigmoid,
        "use_cond_dropout": args.use_cond_dropout,
        "cond_dropout_p": args.cond_dropout_p,
        "use_layernorm": args.use_layernorm,
        "use_rf_blend": args.use_rf_blend,
        "use_adaptive_gate": args.use_adaptive_gate,
        "gate_init_bias": args.gate_init_bias if args.use_adaptive_gate else None,
        "epoch": args.epoch,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "seeds": list(range(42, 42 + args.seeds)),
        "lr": model_args['lr'],
        "weight_decay": model_args['weight_decay'],
        "emb_dim": model_args['emb_dim'],
        "gnn_dropout": model_args['dropout_rate'],
        "pool": model_args['pool'],
        "huber_delta": 0.05,
        "scheduler": "CosineAnnealingLR",
        "scheduler_eta_min": 1e-5,
        "feature_clip": args.feature_clip,
        "data_version": os.path.basename(os.path.normpath(data_path)),
        "data_sha256": sha256_file(os.path.join(data_path, 'data.npy')),
        "label_sha256": sha256_file(os.path.join(data_path, 'label.npy')),
        "meta_sha256": sha256_file(meta_csv),
        "git_commit": git_commit_or_unknown(),
        "split_seed": 42,
        "val_protocol": "group_shuffle_by_refrigerant_v1",
        "val_group_fraction": 0.18,
    }
    config_str = json.dumps(exp_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    out_dir = f"results_ablation/{args.family}_{args.mode}_{args.descriptor_mode}_{config_hash}"
    os.makedirs(out_dir, exist_ok=True)

    # 保存配置文件（可溯源）
    config_path = f"{out_dir}/config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            old_config = json.load(f)
        if old_config != exp_config:
            raise RuntimeError(
                f"🚨 CONFIG COLLISION! Directory {out_dir} has different config.\n"
                f"Old: {old_config}\nNew: {exp_config}\n"
                f"This should never happen with hash-based naming. Check for manual changes."
            )
    else:
        with open(config_path, 'w') as f:
            json.dump(exp_config, f, indent=2)
    print(f"  Config hash: {config_hash} | Dir: {out_dir}")
    splits_dir = os.path.join(out_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # ============================================================
    # 3. 构建 LORO 切分
    # ============================================================
    splits_to_run = []

    if args.mode == 'random':
        train_val_idx, test_idx = train_test_split(
            df.index.values, test_size=0.1, random_state=42)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=1/9, random_state=42)
        splits_to_run.append(('random_split', train_idx.tolist(),
                              val_idx.tolist(), test_idx.tolist()))

    elif args.mode == 'loro':
        unique_refs = df[ref_col].unique()
        if args.target_ref:
            matching = [r for r in unique_refs if r.upper() == args.target_ref.upper()]
            if not matching:
                raise ValueError(f"Target refrigerant '{args.target_ref}' not found in {args.family} family! Available: {list(unique_refs)}")
            unique_refs = matching
            print(f"🎯 Targeted single refrigerant mode: {unique_refs[0]}")
        for ref in unique_refs:
            test_idx = df[df[ref_col] == ref].index.values
            train_val_idx = df[df[ref_col] != ref].index.values
            if len(train_val_idx) == 0 or len(test_idx) == 0:
                continue

            # ── 验证集按制冷剂分组划分（审稿铁律）──
            # 保证 val 中的制冷剂完全不出现在 train 中，
            # 使 Early Stopping 选择的是跨制冷剂外推最优 checkpoint。
            split_path = os.path.join(splits_dir, f'loro_{ref}.npz')
            if os.path.exists(split_path):
                frozen = np.load(split_path)
                train_idx = frozen['train'].astype(int).tolist()
                val_idx = frozen['val'].astype(int).tolist()
                frozen_test = frozen['test'].astype(int).tolist()
                if frozen_test != test_idx.astype(int).tolist():
                    raise RuntimeError(f'Frozen test indices disagree with current data: {split_path}')
            else:
                train_val_df = df_raw.loc[train_val_idx]
                gss = GroupShuffleSplit(
                    n_splits=1, test_size=0.18, random_state=42
                )
                tr_pos, va_pos = next(gss.split(
                    train_val_df,
                    groups=train_val_df[ref_col]
                ))
                train_idx = train_val_idx[tr_pos].astype(int).tolist()
                val_idx = train_val_idx[va_pos].astype(int).tolist()
                np.savez_compressed(
                    split_path, train=np.asarray(train_idx),
                    val=np.asarray(val_idx), test=np.asarray(test_idx, dtype=int)
                )

            # 验证 train/val 制冷剂零重叠
            tr_refs = set(df_raw.loc[train_idx, ref_col])
            va_refs = set(df_raw.loc[val_idx, ref_col])
            assert tr_refs.isdisjoint(va_refs), \
                f"🚨 VAL LEAKAGE! Overlap: {tr_refs & va_refs}"
            print(f"  [GroupVal] Train refs: {len(tr_refs)}, Val refs: {va_refs}")

            splits_to_run.append((f'loro_{ref}', train_idx,
                                  val_idx, test_idx.tolist()))

    # ============================================================
    # 4. 训练和评估循环
    # ============================================================
    summary_path = os.path.join(out_dir, 'summary.csv')
    split_info_path = os.path.join(out_dir, 'split_information.csv')
    summary_results = (
        pd.read_csv(summary_path).to_dict('records')
        if os.path.exists(summary_path) else []
    )
    split_info_results = (
        pd.read_csv(split_info_path).to_dict('records')
        if os.path.exists(split_info_path) else []
    )

    for split_name, train_idx, val_idx, test_idx in splits_to_run:
        print(f"\n{'='*60}")
        print(f"Running Split: {split_name} | "
              f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # ── 数据泄漏防护 ──
        unique_train_refs = df_raw.loc[train_idx, ref_col].unique()
        unique_val_refs   = df_raw.loc[val_idx, ref_col].unique()
        unique_test_refs  = df_raw.loc[test_idx, ref_col].unique()
        assert set(unique_train_refs).isdisjoint(unique_test_refs), "Train/test refrigerant overlap"
        assert set(unique_val_refs).isdisjoint(unique_test_refs), "Validation/test refrigerant overlap"

        if args.mode == 'loro':
            target_ref = split_name.replace('loro_', '')
            assert target_ref not in unique_train_refs, \
                f"🚨 LEAKAGE! {target_ref} found in training set!"
            assert list(unique_test_refs) == [target_ref], \
                f"🚨 TEST PURITY FAILED! Expected only {target_ref}, found {unique_test_refs}"
            print(f"[DEBUG] LORO Purity Check: PASSED ✅ (Train refs: {len(unique_train_refs)}, Test ref: {target_ref})")

        split_info_results = [r for r in split_info_results if r['split'] != split_name]
        split_info_results.append({
            'split': split_name,
            'descriptor_mode': args.descriptor_mode,
            'cond_dim': cond_dim,
            'train_refs': len(unique_train_refs),
            'test_ref': split_name.replace('loro_', '') if args.mode == 'loro' else 'mixed',
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx),
        })
        print(f"{'='*60}")

        train_set = torch.utils.data.Subset(Whole_set, train_idx)
        val_set   = torch.utils.data.Subset(Whole_set, val_idx)
        test_set  = torch.utils.data.Subset(Whole_set, test_idx)

        split_dir = f"{out_dir}/{split_name}"
        preds_dir = f"{split_dir}_preds"

        # ── 检查是否已完成全部 Seed（断点秒级续算）──
        all_seeds_exist = True
        loaded_split_r2 = []
        loaded_split_mae = []
        for seed in range(42, 42 + args.seeds):
            seed_file = f"{preds_dir}/seed{seed}.csv"
            if os.path.exists(seed_file):
                try:
                    s_df = pd.read_csv(seed_file)
                    if len(s_df) == len(test_idx):
                        r2 = r2_score(s_df['true_x1'], np.clip(s_df['pred_x1'], 0, 1))
                        mae = mean_absolute_error(s_df['true_x1'], np.clip(s_df['pred_x1'], 0, 1))
                        loaded_split_r2.append(r2)
                        loaded_split_mae.append(mae)
                    else:
                        all_seeds_exist = False
                        break
                except Exception:
                    all_seeds_exist = False
                    break
            else:
                all_seeds_exist = False
                break

        if all_seeds_exist and len(loaded_split_r2) == args.seeds:
            print(f"  ⚡ [Resume] 检测到 {split_name} 已完成 ({args.seeds} seeds)，直接复用！(R²={np.mean(loaded_split_r2):.4f}, MAE={np.mean(loaded_split_mae):.4f})")
            summary_results = [r for r in summary_results if r['Target'] != split_name]
            summary_results.append({
                    'Target': split_name,
                    'Family': args.family,
                    'Mode': args.mode,
                    'Descriptor': args.descriptor_mode,
                    'cond_dim': cond_dim,
                    'n_train': len(train_idx),
                    'n_test': len(test_idx),
                    'R2_mean': np.mean(loaded_split_r2),
                    'R2_std': np.std(loaded_split_r2),
                    'MAE_mean': np.mean(loaded_split_mae),
                    'MAE_std': np.std(loaded_split_mae),
                })
            continue

        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(preds_dir, exist_ok=True)

        # ── Scaler: 只用 train 拟合（防泄漏铁律）──
        Whole_set.fit_scalers(train_idx, save_dir=split_dir)

        test_loader = DataLoader(test_set, batch_size=model_args['batch_size'],
                                 shuffle=False)

        split_r2_list = []
        split_mae_list = []

        for seed in range(42, 42 + args.seeds):
            set_seed(seed)
            train_loader = DataLoader(train_set, batch_size=model_args['batch_size'],
                                      shuffle=True)
            val_loader   = DataLoader(val_set, batch_size=model_args['batch_size'],
                                      shuffle=False)

            runner = AblationRunner(model_args, seed=seed, save_dir=split_dir)
            runner.train(train_loader, val_loader)

            if args.use_rf_blend:
                # 提取 tabular 特征用于当前 fold 内的 RF 训练（严格防泄漏）
                feature_indices = Whole_set.feature_indices
                X_train_raw = np.array([[Whole_set.data[i][k] for k in feature_indices] for i in train_idx], dtype=np.float32)
                X_val_raw   = np.array([[Whole_set.data[i][k] for k in feature_indices] for i in val_idx], dtype=np.float32)
                X_test_raw  = np.array([[Whole_set.data[i][k] for k in feature_indices] for i in test_idx], dtype=np.float32)
                y_train_raw = np.array([float(Whole_set.label[i]) for i in train_idx], dtype=np.float32)
                y_val_raw   = np.array([float(Whole_set.label[i]) for i in val_idx], dtype=np.float32)

                rf_scaler = StandardScaler()
                X_train_s = rf_scaler.fit_transform(X_train_raw)
                X_val_s   = rf_scaler.transform(X_val_raw)
                X_test_s  = rf_scaler.transform(X_test_raw)

                rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
                rf.fit(X_train_s, y_train_raw)
                rf_val_pred  = np.clip(rf.predict(X_val_s), 0.0, 1.0)
                rf_test_pred = np.clip(rf.predict(X_test_s), 0.0, 1.0)

                gnn_val_pred, _ = runner.predict(val_loader)
                gnn_test_pred, test_true = runner.predict(test_loader)
                gnn_val_pred  = np.clip(gnn_val_pred, 0.0, 1.0)
                gnn_test_pred = np.clip(gnn_test_pred, 0.0, 1.0)

                best_lambda = 0.0
                best_val_mae = float('inf')
                for lam in np.linspace(0.0, 1.0, 21):
                    blend_val = (1.0 - lam) * rf_val_pred + lam * gnn_val_pred
                    mae_val = mean_absolute_error(y_val_raw, blend_val)
                    if mae_val < best_val_mae:
                        best_val_mae = mae_val
                        best_lambda = lam

                test_pred = (1.0 - best_lambda) * rf_test_pred + best_lambda * gnn_test_pred
                test_pred = np.clip(test_pred, 0.0, 1.0)

                r2_gnn = r2_score(test_true, gnn_test_pred)
                r2_rf  = r2_score(test_true, rf_test_pred)
                r2_blend = r2_score(test_true, test_pred)
                mae_blend = mean_absolute_error(test_true, test_pred)
                print(f"  Seed {seed} -> GNN R²: {r2_gnn:.4f} | RF R²: {r2_rf:.4f} | 🛡️ Blend R²: {r2_blend:.4f} (λ={best_lambda:.2f}, MAE={mae_blend:.4f})")
            else:
                test_pred, test_true = runner.test(test_loader)

            # 验证长度对齐
            assert len(test_idx) == len(test_true) == len(test_pred), \
                f"Length mismatch: idx({len(test_idx)}) true({len(test_true)}) pred({len(test_pred)})"

            # 导出每个 seed 的预测
            seed_df = pd.DataFrame({
                'seed': seed,
                'refrigerant': df_raw.loc[test_idx, ref_col].values,
                'true_x1': test_true,
                'pred_x1': test_pred
            })
            seed_df.to_csv(f"{preds_dir}/seed{seed}.csv", index=False)

            # ── 门控可解释性：导出每个样本的门控激活值 ──
            if hasattr(runner, '_last_gate_values') and runner._last_gate_values is not None:
                gate_cols = [f'gate_{i}' for i in range(runner._last_gate_values.shape[1])]
                gate_df = pd.DataFrame(runner._last_gate_values, columns=gate_cols)
                gate_df['refrigerant'] = df_raw.loc[test_idx, ref_col].values
                gate_df['seed'] = seed
                gate_df.to_csv(f"{preds_dir}/gate_values_seed{seed}.csv", index=False)
                # 打印平均门控值
                mean_gates = runner._last_gate_values.mean(axis=0)
                gate_str = ", ".join([f"g{i}={v:.3f}" for i, v in enumerate(mean_gates)])
                print(f"  🔑 Gate activations: {gate_str}")

            r2  = r2_score(test_true, np.clip(test_pred, 0, 1))
            mae = mean_absolute_error(test_true, np.clip(test_pred, 0, 1))
            split_r2_list.append(r2)
            split_mae_list.append(mae)

        # 更新 summary 并实时落盘
        summary_results = [r for r in summary_results if r['Target'] != split_name]
        summary_results.append({
            'Target': split_name,
            'Family': args.family,
            'Mode': args.mode,
            'Descriptor': args.descriptor_mode,
            'cond_dim': cond_dim,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'R2_mean': np.mean(split_r2_list),
            'R2_std': np.std(split_r2_list),
            'MAE_mean': np.mean(split_mae_list),
            'MAE_std': np.std(split_mae_list),
        })
        pd.DataFrame(summary_results).to_csv(f"{out_dir}/summary.csv", index=False)

    # ============================================================
    # 5. 保存结果
    # ============================================================
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(f"{out_dir}/summary.csv", index=False)

    split_info_df = pd.DataFrame(split_info_results)
    split_info_df.to_csv(f"{out_dir}/split_information.csv", index=False)

    # 打印总览
    print(f"\n{'='*60}")
    print(f"  ✅ Ablation Complete: {args.descriptor_mode}")
    print(f"  Results saved to: {out_dir}/summary.csv")
    print(f"{'='*60}")

    if len(summary_df) > 0:
        print(f"\n  Overall {args.descriptor_mode} Performance:")
        print(f"  R² mean: {summary_df['R2_mean'].mean():.4f} "
              f"(±{summary_df['R2_mean'].std():.4f})")
        print(f"  MAE mean: {summary_df['MAE_mean'].mean():.4f} "
              f"(±{summary_df['MAE_mean'].std():.4f})")
        print(f"\nPer-refrigerant breakdown:")
        print(summary_df[['Target', 'R2_mean', 'R2_std', 'MAE_mean', 'MAE_std']].to_string(index=False))


if __name__ == '__main__':
    main()
