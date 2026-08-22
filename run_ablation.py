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
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import pathlib as pl

# 确保能找到子模块
current_script_dir = str(pl.Path(__file__).resolve().parent)
sys.path.append(os.path.join(current_script_dir, 'GNN_for_property_prediction'))

from Dataset_v6 import IL_set_v6, MODE_COND_DIM
from GAT_Runner_v5 import set_seed  # set_seed 通用，不需要新版

# ============================================================
# Runner (复用 v5 的训练逻辑，但使用 v6 Model)
# ============================================================
from Model_v6 import IL_GAT_v6
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


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
        self._criterion = nn.HuberLoss(delta=1.0)

    def _save(self, title):
        os.makedirs(self.save_dir, exist_ok=True)
        path = f"{self.save_dir}/{title}_seed_{self.seed}.pth"
        torch.save({'model_state_dict': self._model.state_dict()}, path)

    def _load_best(self):
        path = f"{self.save_dir}/best_seed_{self.seed}.pth"
        if os.path.exists(path):
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

    def test(self, test_loader):
        self._load_best()
        self._model.eval()
        raw_pred_y, true_y = [], []
        with torch.no_grad():
            for graph, cond, label in test_loader:
                graph = graph.to(self._device)
                cond = cond.to(self._device)
                label = label.to(self._device)
                pred = self._model(graph, cond)
                pred_vals = pred.flatten().cpu().numpy()
                raw_pred_y.extend(pred_vals.tolist())
                true_y.extend(label.cpu().numpy().tolist())

        pred_y = np.clip(np.asarray(raw_pred_y), 0.0, 1.0)
        mae = mean_absolute_error(true_y, pred_y)
        r2  = r2_score(true_y, pred_y)
        print(f"  Seed {self.seed} -> R²: {r2:.4f}, MAE: {mae:.4f}")
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
                        choices=['M0', 'Msize', 'Mmu', 'Mphys'],
                        help="Ablation descriptor mode")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--epoch", type=int, default=150,
                        help="Max epochs")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience")
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

    args = parser.parse_args()

    # ============================================================
    # 1. 加载数据（使用 v3 版本的 processed_tri_data）
    # ============================================================
    data_path = os.path.join(current_script_dir, 'processed_tri_data_v3/')
    if not os.path.exists(os.path.join(data_path, 'data.npy')):
        raise FileNotFoundError(
            f"找不到 {data_path}data.npy！\n"
            f"请先运行: python prepare_tri_graph_data_v3.py"
        )

    # 加载 meta_info 用于切分（与 processed_tri_data_v3 对齐）
    meta_csv = os.path.join(data_path, 'index_with_anion.csv')
    if not os.path.exists(meta_csv):
        # 回退到 meta_info.csv
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

    print(f"\n{'='*60}")
    print(f"  GNN Ablation Study")
    print(f"  Family: {args.family} | Mode: {args.mode}")
    print(f"  Descriptor: {args.descriptor_mode} (cond_dim={MODE_COND_DIM[args.descriptor_mode]})")
    print(f"  Seeds: {args.seeds} | Epochs: {args.epoch}")
    print(f"  Samples after filtering: {len(df)}")
    print(f"{'='*60}\n")

    # ============================================================
    # 2. 构建 Model Args（传入 cond_dim）
    # ============================================================
    cond_dim = MODE_COND_DIM[args.descriptor_mode]

    model_args = {
        'data_path': data_path,
        'batch_size': 64,
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
    }

    print("Loading Graph Dataset (v6)...")
    Whole_set = IL_set_v6(path=model_args['data_path'], args=model_args)

    # 验证数据集大小与 CSV 对齐
    assert len(df_raw) == len(Whole_set), \
        f"Dataset length mismatch! CSV: {len(df_raw)} vs PyG: {len(Whole_set)}"

    out_dir = f"results_ablation/{args.family}_{args.mode}_{args.descriptor_mode}"
    os.makedirs(out_dir, exist_ok=True)

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

            # 90% 训练集，10% 验证集（保留全量分子与数据规模 1881 条）
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.1, random_state=42)

            splits_to_run.append((f'loro_{ref}', train_idx.tolist(),
                                  val_idx.tolist(), test_idx.tolist()))

    # ============================================================
    # 4. 训练和评估循环
    # ============================================================
    summary_results = []
    split_info_results = []

    for split_name, train_idx, val_idx, test_idx in splits_to_run:
        print(f"\n{'='*60}")
        print(f"Running Split: {split_name} | "
              f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # ── 数据泄漏防护 ──
        unique_train_refs = df_raw.iloc[train_idx][ref_col].unique()
        unique_test_refs  = df_raw.iloc[test_idx][ref_col].unique()

        if args.mode == 'loro':
            target_ref = split_name.replace('loro_', '')
            assert target_ref not in unique_train_refs, \
                f"🚨 LEAKAGE! {target_ref} found in training set!"
            assert list(unique_test_refs) == [target_ref], \
                f"🚨 TEST PURITY FAILED! Expected only {target_ref}, found {unique_test_refs}"
            print(f"[DEBUG] LORO Purity Check: PASSED ✅ (Train refs: {len(unique_train_refs)}, Test ref: {target_ref})")

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
        os.makedirs(split_dir, exist_ok=True)

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

            test_pred, test_true = runner.test(test_loader)

            # 验证长度对齐
            assert len(test_idx) == len(test_true) == len(test_pred), \
                f"Length mismatch: idx({len(test_idx)}) true({len(test_true)}) pred({len(test_pred)})"

            # 导出每个 seed 的预测
            seed_df = pd.DataFrame({
                'seed': seed,
                'refrigerant': df_raw.iloc[test_idx][ref_col].values,
                'true_x1': test_true,
                'pred_x1': test_pred
            })
            preds_dir = f"{split_dir}_preds"
            os.makedirs(preds_dir, exist_ok=True)
            seed_df.to_csv(f"{preds_dir}/seed{seed}.csv", index=False)

            r2  = r2_score(test_true, np.clip(test_pred, 0, 1))
            mae = mean_absolute_error(test_true, np.clip(test_pred, 0, 1))
            split_r2_list.append(r2)
            split_mae_list.append(mae)

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
