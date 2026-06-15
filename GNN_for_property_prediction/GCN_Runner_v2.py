"""
GCN_Runner_v2.py — 快速优化实验版本
===================================================
核心改动:
  [改动1] Dataset_v2: T 和 P 同样经过 StandardScaler 标准化
  [改动2] 损失函数: L1Loss (MAE) → HuberLoss (Smooth L1, delta=1.0)
  [改动3] LR 调度: CosineAnnealingLR (100 Epochs, 无重启)
  [改动4] 模型: 保持 IL_Net_GCN 不变，主要对齐外围训练组件。

作者: GNN 优化方案 v2
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import random
import os
import pandas as pd

from Dataset import IL_set
from Model import IL_Net_GCN


# ============================================================
# [改动1] Dataset_v2: 继承 IL_set，追加对 T 和 P 的标准化
# ============================================================
class IL_set_v2(IL_set):
    """
    在原始 IL_set 基础上，对温度 T 和压力 P 进行 StandardScaler 标准化。
    """
    def __init__(self, path):
        super().__init__(path)

        raw_T = np.array([self.data[i][3] for i in range(self.length)], dtype=np.float32).reshape(-1, 1)
        raw_P = np.array([self.data[i][4] for i in range(self.length)], dtype=np.float32).reshape(-1, 1)
        self.scaler_T = StandardScaler().fit(raw_T)
        self.scaler_P = StandardScaler().fit(raw_P)

        T_mean, T_std = self.scaler_T.mean_[0], self.scaler_T.scale_[0]
        P_mean, P_std = self.scaler_P.mean_[0], self.scaler_P.scale_[0]
        print(f"  [v2 数据增强] T 标准化: 均值={T_mean:.1f}K, 标准差={T_std:.1f}K")
        print(f"  [v2 数据增强] P 标准化: 均值={P_mean:.4f}MPa, 标准差={P_std:.4f}MPa")

    def __getitem__(self, idx):
        Combine_Graph, condition, label = super().__getitem__(idx)
        T_scaled = float(self.scaler_T.transform([[self.data[idx][3]]])[0][0])
        P_scaled = float(self.scaler_P.transform([[self.data[idx][4]]])[0][0])
        condition[0] = T_scaled
        condition[1] = P_scaled
        return Combine_Graph, condition, label


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  Random seed set to: {seed}")


class EarlyStopping:
    def __init__(self, patience=20, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# ============================================================
# 超参数配置
# ============================================================
Args = {
    'data_path':     '../processed_tri_data/',
    'batch_size':    64,
    'lr':            0.001,
    'epoch':         100,       
    'weight_decay':  1e-6,

    # GCN 模型超参数 (保持与原始 GCN_Runner.py 相同)
    'emb_dim':       300,
    'dropout_rate':  0.4,

    # Early Stopping
    'patience':      20,
}


# ============================================================
# Runner
# ============================================================
class Runner:
    def __init__(self, args, seed=42):
        self.args = args
        self.seed = seed
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Running on: {self._device}")

        self._model = IL_Net_GCN(args).to(self._device)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=args['lr'],
            weight_decay=args['weight_decay']
        )

        self._scheduler = CosineAnnealingLR(
            self._optimizer,
            T_max=args['epoch'],
            eta_min=1e-5
        )

        # HuberLoss 替代 L1Loss
        self._criterion = nn.HuberLoss(delta=1.0)

    def _save(self, title):
        os.makedirs('checkpoints_v2', exist_ok=True)
        path = f"checkpoints_v2/{title}_gcn_seed_{self.seed}.pth"
        torch.save({'model_state_dict': self._model.state_dict()}, path)

    def _load_best(self):
        path = f"checkpoints_v2/best_gcn_seed_{self.seed}.pth"
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self._device)
            self._model.load_state_dict(ckpt['model_state_dict'])

    def train(self, train_loader, dev_loader):
        model = self._model
        optimizer = self._optimizer
        scheduler = self._scheduler
        early_stopping = EarlyStopping(patience=self.args['patience'])
        best_v_loss = float('inf')

        for epoch in range(1, self.args['epoch'] + 1):
            model.train()
            train_loss = 0.0

            bar = tqdm(total=len(train_loader), dynamic_ncols=True,
                       leave=False, desc=f"Epoch {epoch:>3d} Train")
            for batch_idx, (graph, cond, label) in enumerate(train_loader):
                graph = graph.to(self._device)
                cond  = cond.to(self._device)
                label = label.to(self._device)

                optimizer.zero_grad()
                y    = model(graph, cond)
                loss = self._criterion(y.flatten(), label.flatten())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                bar.set_postfix(
                    loss=f"{train_loss/(batch_idx+1):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}"
                )
                bar.update()
            bar.close()

            scheduler.step()

            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for graph, cond, label in dev_loader:
                    graph = graph.to(self._device)
                    cond  = cond.to(self._device)
                    label = label.to(self._device)
                    y = model(graph, cond)
                    val_loss += self._criterion(y.flatten(), label.flatten()).item()

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(dev_loader)

            if avg_val < best_v_loss:
                best_v_loss = avg_val
                self._save('best')

            print(f"  Epoch {epoch:>3d}/{self.args['epoch']} | "
                  f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            early_stopping(avg_val)
            if early_stopping.early_stop:
                print(f"  ⏹ Early stopping at epoch {epoch}")
                break

        return best_v_loss

    def test(self, test_loader):
        self._load_best()
        model = self._model
        model.eval()
        pred_y, true_y = [], []
        with torch.no_grad():
            for graph, cond, label in tqdm(test_loader, desc="Testing", leave=False):
                graph = graph.to(self._device)
                cond  = cond.to(self._device)
                pred  = model(graph, cond)
                pred_y.extend(pred.flatten().cpu().numpy().tolist())
                true_y.extend(label.numpy().tolist())

        mae = mean_absolute_error(true_y, pred_y)
        r2  = r2_score(true_y, pred_y)
        print(f"  ✅ Test → MAE: {mae:.4f}, R²: {r2:.4f}")
        return pred_y, true_y


def plot_results(true_y, pred_y, title, filename):
    plt.figure(figsize=(7, 7))
    plt.scatter(true_y, pred_y, alpha=0.45, color='darkorange', s=18, label='Predictions')
    lims = [min(min(true_y), min(pred_y)), max(max(true_y), max(pred_y))]
    plt.plot(lims, lims, 'r--', lw=1.5, label='Ideal (y=x)')
    plt.xlabel('Experimental x₁', fontsize=12)
    plt.ylabel('Predicted x₁',    fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs('figure_v2', exist_ok=True)
    plt.savefig(f"figure_v2/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 图已保存: figure_v2/{filename}.png")


# ============================================================
# 主程序入口
# ============================================================
if __name__ == '__main__':
    seeds = [42, 7, 1]

    # ── 固定测试集切分 (与 GIN 保持相同的 SPLIT_SEED=42，确保对比公平) ──
    SPLIT_SEED = 42
    print(f"\n{'='*55}")
    print(f"  GCN_Runner_v2 | 核心改动: T/P归一化 + HuberLoss + 余弦退火")
    print(f"  种子数: {len(seeds)}  最大Epoch: {Args['epoch']}  总迭代上限: {len(seeds)*Args['epoch']}")
    print(f"{'='*55}\n")

    print("正在加载数据集 (v2 版本，T/P 已追加标准化)...")
    Whole_set = IL_set_v2(path=Args['data_path'])

    train_size = int(len(Whole_set) * 0.7)
    test_size  = int(len(Whole_set) * 0.2)
    dev_size   = len(Whole_set) - train_size - test_size

    train_set, dev_set, test_set = random_split(
        Whole_set, [train_size, dev_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )
    test_loader = DataLoader(test_set, batch_size=Args['batch_size'], shuffle=False)
    print(f"  数据划分 → Train: {train_size}, Dev: {dev_size}, Test: {test_size}\n")

    all_preds      = []
    test_true      = None
    ensemble_results = []

    for seed in seeds:
        print(f"\n{'─'*55}")
        print(f"  训练 GCN Seed: {seed}")
        print(f"{'─'*55}")
        set_seed(seed)

        train_loader = DataLoader(train_set, batch_size=Args['batch_size'], shuffle=True)
        dev_loader   = DataLoader(dev_set,   batch_size=Args['batch_size'], shuffle=False)

        runner = Runner(Args, seed=seed)
        runner.train(train_loader, dev_loader)

        test_pred, test_true = runner.test(test_loader)
        all_preds.append(test_pred)

        mae = mean_absolute_error(test_true, test_pred)
        r2  = r2_score(test_true, test_pred)
        ensemble_results.append({'seed': seed, 'mae': mae, 'r2': r2})

        plot_results(test_true, test_pred,
                     f"GCN v2 Seed {seed} (R²={r2:.4f})",
                     f"gcn_pred_v2_seed_{seed}")

    # ── 集成结果 ──
    ensemble_pred = np.mean(all_preds, axis=0).tolist()
    ens_mae = mean_absolute_error(test_true, ensemble_pred)
    ens_r2  = r2_score(test_true, ensemble_pred)

    print(f"\n{'='*55}")
    print(f"  GCN 单体模型结果汇总:")
    df = pd.DataFrame(ensemble_results)
    print(df.to_string(index=False))
    print(f"\n  GCN 平均单体 R²: {df['r2'].mean():.4f} (±{df['r2'].std():.4f})")

    print(f"\n  🏆 GCN 集成 (3模型均值) → MAE: {ens_mae:.4f}, R²: {ens_r2:.4f}")
    print(f"{'='*55}\n")

    # 保存摘要
    df['ensemble_mae'] = ens_mae
    df['ensemble_r2']  = ens_r2
    df.to_csv('gcn_ensemble_results_v2.csv', index=False)
    print("  📄 结果已保存至: gcn_ensemble_results_v2.csv")

    plot_results(test_true, ensemble_pred,
                 f"GCN v2 Ensemble (3 Seeds, R²={ens_r2:.4f})",
                 "gcn_ensemble_v2_final")
