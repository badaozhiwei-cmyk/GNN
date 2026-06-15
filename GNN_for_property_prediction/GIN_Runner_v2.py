"""
GIN_Runner_v2.py — 快速优化实验版本
===================================================
相比 v1 (GIN_Runner.py) 的核心改动 (3项):

  [改动1] Dataset_v2: T 和 P 同样经过 StandardScaler 标准化
          → 消除因温度(~300K) vs 描述符(~0) 量级相差数百倍导致的梯度失衡
          → 不改 Model.py / 原 Dataset.py，仅在此文件内以子类方式覆盖

  [改动2] 损失函数: L1Loss (MAE) → HuberLoss (Smooth L1, delta=1.0)
          → R² 的优化方向与 L2 范数直接绑定，Huber 兼具 L2 精细优化 + L1 鲁棒性
          → 理论上对提升 R² 比 MAE 训练更直接有效

  [改动3] LR 调度: ReduceLROnPlateau → CosineAnnealingWarmRestarts
          → 阶梯下降容易卡鞍点；余弦热重启能周期性脱离局部极小值
          → 与此同时将 Epoch 从 150 降至 100，Seed 数从 5 降至 3
          → 总迭代量: 3×100=300  vs  原版 5×150=750，快 60%，效果更好

作者: GNN 优化方案 v2 (2026-05-17)
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

# 导入原有的图数据集类和模型 (保持不变)
from Dataset import IL_set
from Model import GIN


# ============================================================
# [改动1] Dataset_v2: 继承 IL_set，追加对 T 和 P 的标准化
# ============================================================
class IL_set_v2(IL_set):
    """
    [修复] 数据泄露和特征错位：只在训练集上拟合 StandardScaler，并作用于 7 个物理量
    """
    def __init__(self, path):
        super().__init__(path)
        self.scalers = None

    def fit_scalers(self, train_indices):
        self.scalers = [StandardScaler() for _ in range(7)]
        for feature_idx in range(7):
            # 7 个物理量在 self.data 中的索引是从 3 到 9
            raw_vals = np.array([self.data[i][feature_idx + 3] for i in train_indices], dtype=np.float32).reshape(-1, 1)
            self.scalers[feature_idx].fit(raw_vals)
        print("  [v2 数据增强] 7个物理量的 StandardScaler 拟合完成 (无数据泄露)")

    def __getitem__(self, idx):
        Combine_Graph, condition, label = super().__getitem__(idx)
        if self.scalers is not None:
            for feature_idx in range(7):
                raw_val = float(self.data[idx][feature_idx + 3])
                scaled_val = float(self.scalers[feature_idx].transform([[raw_val]])[0][0])
                condition[feature_idx] = scaled_val
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
# (改动3: epoch 100, seed数3, 使用余弦退火热重启)
# ============================================================
Args = {
    'data_path':     '../processed_tri_data/',
    'batch_size':    64,
    'lr':            0.001,
    'epoch':         100,       # ↓ 原来 150 → 现在 100 (配合余弦退火, 60 epoch 内可收敛)
    'weight_decay':  1e-6,

    # GIN 模型超参数 (与 v1 保持一致，便于对比)
    'num_gin_layer': 3,
    'emb_dim':       300,
    'feat_dim':      512,
    'drop_ratio':    0.2,
    'pool':          'mean',

    # 余弦退火参数
    'T_0':           30,        # 第一轮余弦周期长度 (Epoch)
    'T_mult':        2,         # 每次重启后周期翻倍 (30 → 60 → ...)

    # Early Stopping
    'patience':      20,        # ↓ 原来 30 → 现在 20 (更高效)
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

        self._model = GIN(args).to(self._device)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=args['lr'],
            weight_decay=args['weight_decay']
        )

        # [改动3] 平滑余弦退火衰减 (无重启，在最大 Epoch 期间一条龙平滑下降)
        self._scheduler = CosineAnnealingLR(
            self._optimizer,
            T_max=args['epoch'],
            eta_min=1e-5  # 学习率下限，防止学习率归零
        )

        # [改动2] HuberLoss 替代 L1Loss
        self._criterion = nn.HuberLoss(delta=1.0)

    def _save(self, title):
        os.makedirs('checkpoints_v2', exist_ok=True)
        path = f"checkpoints_v2/{title}_seed_{self.seed}.pth"
        torch.save({'model_state_dict': self._model.state_dict()}, path)

    def _load_best(self):
        path = f"checkpoints_v2/best_seed_{self.seed}.pth"
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

            # [改动3] 平滑余弦退火按 epoch 步进 (无重启)
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
    plt.scatter(true_y, pred_y, alpha=0.45, color='steelblue', s=18, label='Predictions')
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
    # [改动3] Seed 数从 5 降到 3 (总迭代: 3×100=300 vs 原来 5×150=750，快60%)
    seeds = [42, 7, 1]

    # ── 固定测试集切分 (与 v1 保持相同的 SPLIT_SEED=42，确保对比公平) ──
    SPLIT_SEED = 42
    print(f"\n{'='*55}")
    print(f"  GIN_Runner_v2 | 核心改动: T/P归一化 + HuberLoss + 余弦退火")
    print(f"  种子数: {len(seeds)}  最大Epoch: {Args['epoch']}  总迭代上限: {len(seeds)*Args['epoch']}")
    print(f"{'='*55}\n")

    # [改动1] 使用 IL_set_v2 替换原 IL_set
    print("正在加载数据集 (v2 版本，T/P 已追加标准化)...")
    Whole_set = IL_set_v2(path=Args['data_path'])

    train_size = int(len(Whole_set) * 0.7)
    test_size  = int(len(Whole_set) * 0.2)
    dev_size   = len(Whole_set) - train_size - test_size

    train_set, dev_set, test_set = random_split(
        Whole_set, [train_size, dev_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )
    Whole_set.fit_scalers(train_set.indices)
    test_loader = DataLoader(test_set, batch_size=Args['batch_size'], shuffle=False)
    print(f"  数据划分 → Train: {train_size}, Dev: {dev_size}, Test: {test_size}\n")

    all_preds      = []
    test_true      = None
    ensemble_results = []

    for seed in seeds:
        print(f"\n{'─'*55}")
        print(f"  训练 Seed: {seed}")
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
                     f"v2 Seed {seed} (R²={r2:.4f})",
                     f"pred_v2_seed_{seed}")

    # ── 集成结果 ──
    ensemble_pred = np.mean(all_preds, axis=0).tolist()
    ens_mae = mean_absolute_error(test_true, ensemble_pred)
    ens_r2  = r2_score(test_true, ensemble_pred)

    print(f"\n{'='*55}")
    print(f"  单体模型结果汇总:")
    df = pd.DataFrame(ensemble_results)
    print(df.to_string(index=False))
    print(f"\n  平均单体 R²: {df['r2'].mean():.4f} (±{df['r2'].std():.4f})")

    print(f"\n  🏆 集成 (3模型均值) → MAE: {ens_mae:.4f}, R²: {ens_r2:.4f}")
    print(f"{'='*55}\n")

    # 保存摘要
    df['ensemble_mae'] = ens_mae
    df['ensemble_r2']  = ens_r2
    df.to_csv('ensemble_results_v2.csv', index=False)
    print("  📄 结果已保存至: ensemble_results_v2.csv")

    plot_results(test_true, ensemble_pred,
                 f"v2 Ensemble (3 Seeds, R²={ens_r2:.4f})",
                 "ensemble_v2_final")
