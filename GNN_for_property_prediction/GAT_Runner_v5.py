"""
GAT_Runner_v5.py — AI for Science Generalization Benchmark Runner
===================================================
【v5 核心升级】
  1. 引入了 PICP (Prediction Interval Coverage Probability) 和 MPIW 来代替分类任务的 ECE。
  2. 支持 Dataset_v5 的 `use_ani_mw` 传参控制。
  3. 支持 Model_v5 的 `pool` 传参 (global/mean/attention) 消融测试。
  4. 保证 Scaler 级别的 Fit-Once-Per-Split（所有 Seed 共享一个 Scaler）。
"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import os
import pandas as pd
import pathlib as pl

# 从 v5 版本导入
from Dataset_v5 import IL_set_v5
from Model_v5 import IL_GAT_v5

current_script_dir = str(pl.Path(__file__).resolve().parent)
ROOT_DIR = str(pl.Path(current_script_dir).parent)

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
# Runner
# ============================================================
class Runner:
    def __init__(self, args, seed=42, save_dir="checkpoints_v5"):
        self.args = args
        self.seed = seed
        self.save_dir = save_dir
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = IL_GAT_v5(args).to(self._device)
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
        early_stopping = EarlyStopping(patience=self.args['patience'])
        best_v_loss = float('inf')

        for epoch in range(1, self.args['epoch'] + 1):
            self._model.train()
            train_loss = 0.0
            
            bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, desc=f"Epoch {epoch:>3d}")
            for graph, cond, label in train_loader:
                graph, cond, label = graph.to(self._device), cond.to(self._device), label.to(self._device)
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
                    graph, cond, label = graph.to(self._device), cond.to(self._device), label.to(self._device)
                    y = self._model(graph, cond)
                    val_loss += self._criterion(y.flatten(), label.flatten()).item()

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(dev_loader)

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
        pred_y, true_y = [], []
        with torch.no_grad():
            for graph, cond, label in test_loader:
                graph, cond, label = graph.to(self._device), cond.to(self._device), label.to(self._device)
                pred = self._model(graph, cond)
                pred_vals = np.clip(pred.flatten().cpu().numpy(), 0.0, 1.0)
                pred_y.extend(pred_vals.tolist())
                true_y.extend(label.numpy().tolist())

        mae = mean_absolute_error(true_y, pred_y)
        rmse = np.sqrt(mean_squared_error(true_y, pred_y))
        r2  = r2_score(true_y, pred_y)
        print(f"  ✅ Seed {self.seed} → MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
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
    os.makedirs('figure_v5', exist_ok=True)
    plt.savefig(f"figure_v5/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# 主程序入口
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generalization Benchmark Runner v5")
    parser.add_argument("--level", type=str, default="L0", help="Split level to run (L0, L1, L2, L3, L4)")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to run for Deep Ensemble")
    parser.add_argument("--epoch", type=int, default=100, help="Max epochs")
    parser.add_argument("--pool", type=str, default="global", help="Pooling method (global, mean, attention)")
    parser.add_argument("--use_ani_mw", action="store_true", help="Use Anion Molecular Weight")
    
    cmd_args = parser.parse_args()
    
    Args = {
        'data_path':     os.path.join(ROOT_DIR, 'processed_tri_data/'),
        'batch_size':    64,
        'lr':            0.001,
        'epoch':         cmd_args.epoch,       
        'weight_decay':  1e-6,
        'emb_dim':       300,
        'dropout_rate':  0.2,
        'patience':      20,
        'pool':          cmd_args.pool,
        'use_ani_mw':    cmd_args.use_ani_mw
    }

    LEVEL = cmd_args.level
    NUM_SEEDS = cmd_args.seeds
    SEEDS = list(range(NUM_SEEDS))
    SAVE_DIR = f"checkpoints_v5/{LEVEL}"

    print(f"\n{'='*60}")
    print(f"  GAT_Runner_v5 | Target: {LEVEL}")
    print(f"  Ensemble Seeds: {NUM_SEEDS}")
    print(f"  Pooling: {Args['pool']} | Use ANI MW: {Args['use_ani_mw']}")
    print(f"{'='*60}\n")

    # 1. 加载切分索引
    split_file = os.path.join(ROOT_DIR, f"split_{LEVEL}_indices.npz")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"找不到 {split_file}，请先运行 step1_generalization_ladder_v2.py")
    
    loaded_idx = np.load(split_file)
    train_indices = loaded_idx['train'].tolist()
    val_indices   = loaded_idx['val'].tolist()
    test_indices  = loaded_idx['test'].tolist()
    
    print("正在加载数据集 (v5)...")
    Whole_set = IL_set_v5(path=Args['data_path'], args=Args)

    train_set = torch.utils.data.Subset(Whole_set, train_indices)
    dev_set   = torch.utils.data.Subset(Whole_set, val_indices)
    test_set  = torch.utils.data.Subset(Whole_set, test_indices)

    # 【重要修复】只 Fit 一次 Scaler，所有 Seed 共享
    Whole_set.fit_scalers(train_indices, save_dir=SAVE_DIR)
    
    test_loader = DataLoader(test_set, batch_size=Args['batch_size'], shuffle=False)
    print(f"  数据集 {LEVEL} 划分 → Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}\n")

    all_preds = []
    test_true = None
    ensemble_results = []

    for seed in SEEDS:
        print(f"\n{'─'*60}")
        print(f"  🚀 训练 GAT_v5 | {LEVEL} | Seed {seed}")
        print(f"{'─'*60}")
        set_seed(seed)

        train_loader = DataLoader(train_set, batch_size=Args['batch_size'], shuffle=True)
        dev_loader   = DataLoader(dev_set,   batch_size=Args['batch_size'], shuffle=False)

        runner = Runner(Args, seed=seed, save_dir=SAVE_DIR)
        runner.train(train_loader, dev_loader)

        test_pred, test_true = runner.test(test_loader)
        all_preds.append(test_pred)

        mae = mean_absolute_error(test_true, test_pred)
        rmse = np.sqrt(mean_squared_error(test_true, test_pred))
        r2  = r2_score(test_true, test_pred)
        ensemble_results.append({'level': LEVEL, 'seed': seed, 'mae': mae, 'rmse': rmse, 'r2': r2})

    # ── 集成计算 (Mean, PICP, MPIW) ──
    all_preds = np.array(all_preds)      # Shape: (NUM_SEEDS, num_test_samples)
    test_true = np.array(test_true)
    
    ensemble_pred = np.mean(all_preds, axis=0)
    ensemble_std  = np.std(all_preds, axis=0)

    # 95% 预测区间 (Assuming Gaussian distribution of predictions)
    lower_bound = ensemble_pred - 1.96 * ensemble_std
    upper_bound = ensemble_pred + 1.96 * ensemble_std

    covered = (test_true >= lower_bound) & (test_true <= upper_bound)
    picp = np.mean(covered) * 100.0  # Percentage
    mpiw = np.mean(upper_bound - lower_bound)

    ens_mae = mean_absolute_error(test_true, ensemble_pred)
    ens_rmse = np.sqrt(mean_squared_error(test_true, ensemble_pred))
    ens_r2  = r2_score(test_true, ensemble_pred)

    print(f"\n{'='*60}")
    print(f"  {LEVEL} 单体模型结果汇总:")
    df = pd.DataFrame(ensemble_results)
    print(df.to_string(index=False))
    print(f"\n  单体均值 → R²: {df['r2'].mean():.4f} (±{df['r2'].std():.4f}) | MAE: {df['mae'].mean():.4f}")

    print(f"\n  🏆 最终集成 (Ensemble of {NUM_SEEDS})")
    print(f"  MAE:  {ens_mae:.4f}")
    print(f"  RMSE: {ens_rmse:.4f}")
    print(f"  R²:   {ens_r2:.4f}")
    print(f"  -----------------------")
    print(f"  [Uncertainty Calibration]")
    print(f"  PICP: {picp:.2f}% (Target: ~95%)")
    print(f"  MPIW: {mpiw:.4f} (Sharpness)")
    print(f"{'='*60}\n")

    os.makedirs('results_v5', exist_ok=True)
    df['ensemble_mae'] = ens_mae
    df['ensemble_rmse'] = ens_rmse
    df['ensemble_r2']  = ens_r2
    df['ensemble_picp'] = picp
    df['ensemble_mpiw'] = mpiw
    df.to_csv(f'results_v5/{LEVEL}_ensemble_results.csv', index=False)
    
    plot_results(test_true, ensemble_pred,
                 f"GAT {LEVEL} Ensemble ({NUM_SEEDS} Seeds, R²={ens_r2:.4f})",
                 f"{LEVEL}_ensemble_final")
