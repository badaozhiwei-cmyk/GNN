"""
GIN_Runner_v4.py
================
【相比 GIN_Runner_v3 的核心改动（与 GAT_Runner_v4 完全对称）】
  [改动1] 数据划分：加载 step1 生成的 .npz 索引，支持 --split A/B 切换
  [改动2] 多种子：默认 5 个种子（42,123,2024,3407,6666），汇报 mean±std
  [改动3] 增加 RMSE 指标
  [改动4] Checkpoint / 结果按 split 分文件夹，与 GAT 不互相覆盖
  [改动5] 每个种子的预测结果存为 CSV，供后续统一画图对比

【运行方法】
  python GIN_Runner_v4.py --split A
  python GIN_Runner_v4.py --split B
  python GIN_Runner_v4.py --split B --seeds 42   # 快速单种子测试
"""

import argparse
import os

# ── 自动定位项目根目录（脚本在 research_pipeline/ 子文件夹中）──────
import pathlib as _pl
ROOT = str(_pl.Path(__file__).resolve().parent.parent)
import os as _os; _os.chdir(ROOT)
# ─────────────────────────────────────────────────────────────────────
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))
from Dataset import IL_set
from Model import GIN


# ============================================================
# 命令行参数
# ============================================================
parser = argparse.ArgumentParser(description='GIN Runner v4 — OOD 划分 + 多种子')
parser.add_argument('--split', type=str, default='B', choices=['A', 'B'])
parser.add_argument('--seeds', type=str, default='42,123,2024,3407,6666')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--data_path', type=str, default='processed_tri_data/')
args_cli = parser.parse_args()

SPLIT_NAME = args_cli.split
SEEDS      = [int(s) for s in args_cli.seeds.split(',')]
CKPT_DIR   = f'checkpoints_gin_split{SPLIT_NAME}'
FIG_DIR    = f'figure_gin_split{SPLIT_NAME}'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

print(f"\n{'='*60}")
print(f"  GIN Runner v4 | Split {SPLIT_NAME} | Seeds: {SEEDS}")
print(f"  Checkpoint 目录: {CKPT_DIR}")
print(f"{'='*60}\n")


# ============================================================
# 超参数（与 v3 保持一致，方便和旧结果对比）
# ============================================================
Args = {
    'data_path':     args_cli.data_path,
    'batch_size':    64,
    'lr':            0.001,
    'epoch':         args_cli.epoch,
    'weight_decay':  1e-6,
    # GIN 模型超参数
    'num_gin_layer': 3,
    'emb_dim':       300,
    'feat_dim':      512,
    'drop_ratio':    0.2,
    'pool':          'mean',
    # Early Stopping
    'patience':      20,
}


# ============================================================
# Dataset_v2：标准化器随划分走
# ============================================================
class IL_set_v2(IL_set):
    def __init__(self, path):
        super().__init__(path)
        self.scalers = None

    def fit_scalers(self, train_indices, save_path):
        self.scalers = [StandardScaler() for _ in range(7)]
        for feat_idx in range(7):
            raw = np.array(
                [self.data[i][feat_idx + 3] for i in train_indices],
                dtype=np.float32
            ).reshape(-1, 1)
            self.scalers[feat_idx].fit(raw)
        joblib.dump(self.scalers, save_path)
        print(f"  [Scaler] 标准化器已拟合并保存至 {save_path}")

    def __getitem__(self, idx):
        graph, cond, label = super().__getitem__(idx)
        if self.scalers is not None:
            for feat_idx in range(7):
                raw = float(self.data[idx][feat_idx + 3])
                scaled = float(self.scalers[feat_idx].transform([[raw]])[0][0])
                cond[feat_idx] = scaled
        return graph, cond, label


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter  = 0
        self.best     = None
        self.stop     = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best is None:
            self.best = score
        elif score < self.best:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best    = score
            self.counter = 0


def compute_metrics(true_y, pred_y):
    mae  = mean_absolute_error(true_y, pred_y)
    r2   = r2_score(true_y, pred_y)
    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    return mae, r2, rmse


def plot_pred(true_y, pred_y, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_y, pred_y, alpha=0.4, s=15, color='steelblue', label='Predictions')
    lims = [min(min(true_y), min(pred_y)) - 0.02,
            max(max(true_y), max(pred_y)) + 0.02]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Ideal (y=x)')
    ax.set_xlabel('Experimental x₁', fontsize=12)
    ax.set_ylabel('Predicted x₁',    fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Runner 类
# ============================================================
class Runner:
    def __init__(self, seed):
        self.seed   = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = GIN(Args).to(self.device)
        self.optim  = torch.optim.Adam(
            self.model.parameters(), lr=Args['lr'], weight_decay=Args['weight_decay']
        )
        self.sched  = CosineAnnealingLR(self.optim, T_max=Args['epoch'], eta_min=1e-5)
        self.crit   = nn.HuberLoss(delta=1.0)
        self.ckpt   = os.path.join(CKPT_DIR, f'best_gin_seed{seed}.pth')

    def _save(self):
        torch.save({'model_state_dict': self.model.state_dict()}, self.ckpt)

    def _load(self):
        if os.path.exists(self.ckpt):
            ck = torch.load(self.ckpt, map_location=self.device)
            self.model.load_state_dict(ck['model_state_dict'])

    def train(self, train_loader, val_loader):
        es       = EarlyStopping(patience=Args['patience'])
        best_val = float('inf')

        for epoch in range(1, Args['epoch'] + 1):
            self.model.train()
            tr_loss = 0.0
            bar = tqdm(train_loader, leave=False, desc=f'Ep{epoch:>3d} Train')
            for graph, cond, label in bar:
                graph, cond, label = (graph.to(self.device),
                                      cond.to(self.device),
                                      label.to(self.device))
                self.optim.zero_grad()
                loss = self.crit(self.model(graph, cond).flatten(), label.flatten())
                loss.backward()
                self.optim.step()
                tr_loss += loss.item()
                bar.set_postfix(loss=f'{tr_loss/len(train_loader):.4f}')
            self.sched.step()

            self.model.eval()
            vl_loss = 0.0
            with torch.no_grad():
                for graph, cond, label in val_loader:
                    graph, cond, label = (graph.to(self.device),
                                          cond.to(self.device),
                                          label.to(self.device))
                    vl_loss += self.crit(
                        self.model(graph, cond).flatten(), label.flatten()
                    ).item()

            avg_val = vl_loss / len(val_loader)
            avg_tr  = tr_loss / len(train_loader)

            if avg_val < best_val:
                best_val = avg_val
                self._save()

            if epoch % 10 == 0 or epoch == 1:
                print(f'    Ep {epoch:>3d} | Train {avg_tr:.4f} | Val {avg_val:.4f} '
                      f'| LR {self.optim.param_groups[0]["lr"]:.2e}')

            es(avg_val)
            if es.stop:
                print(f'    ⏹ Early stop @ epoch {epoch}')
                break

    def test(self, test_loader):
        self._load()
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for graph, cond, label in tqdm(test_loader, desc='Testing', leave=False):
                graph, cond = graph.to(self.device), cond.to(self.device)
                out = self.model(graph, cond).flatten().cpu().numpy()
                preds.extend(out.tolist())
                trues.extend(label.numpy().tolist())
        return np.array(preds), np.array(trues)


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    # ── 1. 加载数据集 ──
    print('正在加载数据集...')
    dataset = IL_set_v2(path=Args['data_path'])

    # ── 2. 加载划分索引 ──
    npz_path = f'split_{SPLIT_NAME}_indices.npz'
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'找不到 {npz_path}，请先运行 step1_anion_family_splitter.py')
    split_data = np.load(npz_path)
    train_idx  = split_data['train']
    val_idx    = split_data['val']
    test_idx   = split_data['test']
    print(f'  Split {SPLIT_NAME} → Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}')

    # ── 3. 拟合标准化器（仅在训练集上）──
    scaler_path = os.path.join(CKPT_DIR, 'scalers.pkl')
    dataset.fit_scalers(train_idx, scaler_path)

    # ── 4. 构建 Subset ──
    from torch.utils.data import Subset
    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)
    test_loader = DataLoader(test_set, batch_size=Args['batch_size'], shuffle=False)

    # ── 5. 多种子训练 ──
    all_preds   = []
    all_results = []
    test_true_y = None

    for seed in SEEDS:
        print(f'\n{"─"*55}')
        print(f'  训练 GIN | Split {SPLIT_NAME} | Seed {seed}')
        print(f'{"─"*55}')
        set_seed(seed)

        train_loader = DataLoader(train_set, batch_size=Args['batch_size'], shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=Args['batch_size'], shuffle=False)

        runner = Runner(seed)
        runner.train(train_loader, val_loader)
        pred_y, true_y = runner.test(test_loader)

        if test_true_y is None:
            test_true_y = true_y

        mae, r2, rmse = compute_metrics(true_y, pred_y)
        print(f'  ✅ Seed {seed} → MAE: {mae:.4f} | R²: {r2:.4f} | RMSE: {rmse:.4f}')
        all_preds.append(pred_y)
        all_results.append({'seed': seed, 'MAE': mae, 'R2': r2, 'RMSE': rmse})

        plot_pred(true_y, pred_y,
                  f'GIN Split-{SPLIT_NAME} Seed {seed} (R²={r2:.4f})',
                  os.path.join(FIG_DIR, f'gin_split{SPLIT_NAME}_seed{seed}.png'))

        pd.DataFrame({'true': true_y, 'pred': pred_y}).to_csv(
            os.path.join(CKPT_DIR, f'pred_seed{seed}.csv'), index=False
        )

    # ── 6. 集成 ──
    ensemble_pred           = np.mean(all_preds, axis=0)
    ens_mae, ens_r2, ens_rmse = compute_metrics(test_true_y, ensemble_pred)

    df_res = pd.DataFrame(all_results)
    print(f'\n{"="*55}')
    print(f'  GIN Split-{SPLIT_NAME} 单体结果：')
    print(df_res[['seed','MAE','R2','RMSE']].to_string(index=False))
    print(f'\n  单体均值 → '
          f'MAE: {df_res["MAE"].mean():.4f}±{df_res["MAE"].std():.4f} | '
          f'R²: {df_res["R2"].mean():.4f}±{df_res["R2"].std():.4f} | '
          f'RMSE: {df_res["RMSE"].mean():.4f}±{df_res["RMSE"].std():.4f}')
    print(f'\n  🏆 集成（{len(SEEDS)}种子均值）→ '
          f'MAE: {ens_mae:.4f} | R²: {ens_r2:.4f} | RMSE: {ens_rmse:.4f}')
    print(f'{"="*55}')

    df_res['split'] = SPLIT_NAME
    df_res.to_csv(f'gin_split{SPLIT_NAME}_results.csv', index=False)
    pd.DataFrame({'true': test_true_y, 'pred_ensemble': ensemble_pred}).to_csv(
        os.path.join(CKPT_DIR, 'pred_ensemble.csv'), index=False
    )
    plot_pred(test_true_y, ensemble_pred,
              f'GIN Split-{SPLIT_NAME} Ensemble ({len(SEEDS)} seeds, R²={ens_r2:.4f})',
              os.path.join(FIG_DIR, f'gin_split{SPLIT_NAME}_ensemble.png'))

    print(f'\n✅ GIN Split-{SPLIT_NAME} 全部完成！')
