"""
MPNN_Runner.py
==============
【MPNN 简介】
  MPNN（Message Passing Neural Network）与 GAT/GIN 的核心区别：
  - GAT：注意力加权聚合邻居节点特征（不显式用边特征）
  - GIN：对邻居节点特征做 sum 聚合（不显式用边特征）
  - MPNN：通过一个可学习的 MLP 把"边特征"转换为"消息权重矩阵"，
           显式利用化学键类型（单键/双键/芳香键）参与消息传递
  → 即 NNConv（PyTorch Geometric 内置）
  → 科学意义：测试"显式利用键特征"是否比注意力机制更有利于化学预测

【架构设计】
  与 GAT/GIN 完全对称：
  - 7维 Embedding 层（原子特征）
  - 3层 NNConv（MPNN 消息传递）
  - 全局节点提取（与 GAT/GIN 相同的"祖宗之法"）
  - 相同的 MLP Head（519→1024→512→1）
  - HuberLoss + CosineAnnealingLR + EarlyStopping

【运行方法】
  python MPNN_Runner.py --split A
  python MPNN_Runner.py --split B
  python MPNN_Runner.py --split B --seeds 42
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
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import DataLoader
from torch_geometric.nn import NNConv
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))
from Dataset import IL_set


# ============================================================
# 命令行参数
# ============================================================
parser = argparse.ArgumentParser(description='MPNN Runner — OOD 划分 + 多种子')
parser.add_argument('--split',     type=str, default='B', choices=['A', 'B'])
parser.add_argument('--seeds',     type=str, default='42,123,2024,3407,6666')
parser.add_argument('--epoch',     type=int, default=100)
parser.add_argument('--data_path', type=str, default='processed_tri_data/')
args_cli = parser.parse_args()

SPLIT_NAME = args_cli.split
SEEDS      = [int(s) for s in args_cli.seeds.split(',')]
CKPT_DIR   = f'checkpoints_mpnn_split{SPLIT_NAME}'
FIG_DIR    = f'figure_mpnn_split{SPLIT_NAME}'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

print(f"\n{'='*60}")
print(f"  MPNN Runner | Split {SPLIT_NAME} | Seeds: {SEEDS}")
print(f"  Checkpoint 目录: {CKPT_DIR}")
print(f"{'='*60}\n")


# ============================================================
# 超参数（与 GAT/GIN 完全对称）
# ============================================================
Args = {
    'data_path':     args_cli.data_path,
    'batch_size':    64,
    'lr':            0.001,
    'epoch':         args_cli.epoch,
    'weight_decay':  1e-6,
    'emb_dim':       300,
    'dropout_rate':  0.2,
    'patience':      20,
    # MPNN 边特征维度：原始 edge_attr 是 3 维（键类型/环/芳香）
    'edge_dim':      3,
}

# ── Embedding 表大小（与 Model.py 一致）──────────────────────
num_atom_type = 119
num_Hbrid     = 8
num_Aro       = 2
num_degree    = 7
num_charge    = 3
num_eneg      = 8
num_radius    = 8


# ============================================================
# MPNN 模型定义（NNConv 版本）
# ============================================================
class IL_MPNN(nn.Module):
    """
    3层 NNConv MPNN，与 GAT/GIN 完全对称的接口：
      forward(data, cond) → scalar prediction
    """
    def __init__(self, args):
        super().__init__()
        self.emb_dim = args['emb_dim']
        edge_dim     = args['edge_dim']   # 3（键类型/环/芳香）

        # ── 节点 Embedding（与 GAT/GIN 完全一致）──────────────
        self.x_emb = nn.ModuleList([
            nn.Embedding(num_atom_type, self.emb_dim),
            nn.Embedding(num_Hbrid,     self.emb_dim),
            nn.Embedding(num_Aro,       self.emb_dim),
            nn.Embedding(num_degree,    self.emb_dim),
            nn.Embedding(num_charge,    self.emb_dim),
            nn.Embedding(num_eneg,      self.emb_dim),
            nn.Embedding(num_radius,    self.emb_dim),
        ])
        for emb in self.x_emb:
            nn.init.xavier_uniform_(emb.weight.data)

        # ── 3 层 NNConv ────────────────────────────────────────
        # NNConv：edge_nn 把边特征(edge_dim)映射成 (in_channels × out_channels) 的矩阵
        # 即对每条边学习一个专属的线性变换 → 显式利用化学键类型
        def make_edge_nn(in_ch, out_ch):
            return nn.Sequential(
                nn.Linear(edge_dim, 64),
                nn.ReLU(),
                nn.Linear(64, in_ch * out_ch),
            )

        self.conv1 = NNConv(self.emb_dim, 512, make_edge_nn(self.emb_dim, 512), aggr='mean')
        self.conv2 = NNConv(512,          1024, make_edge_nn(512, 1024),          aggr='mean')
        self.conv3 = NNConv(1024,         512,  make_edge_nn(1024, 512),          aggr='mean')

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(p=args['dropout_rate'])

        # ── MLP Head（与 GAT 完全相同：512 + 7 cond = 519）─────
        self.mlp = nn.Sequential(
            nn.Linear(519, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1),
        )

    def extract_global(self, x, batch):
        """提取每个图的全局节点（最后一个节点）作为图表示"""
        _, count = torch.unique(batch, return_counts=True)
        ends = count.cumsum(0)
        return torch.cat([x[e - 1].unsqueeze(0) for e in ends], dim=0)

    def forward(self, data, cond):
        # ── 节点初始 Embedding ─────────────────────────────────
        h = sum(self.x_emb[i](data.x[:, i]) for i in range(7))

        # edge_attr 是整数类型，NNConv 需要 float
        edge_attr = data.edge_attr.float()
        edge_index = data.edge_index

        # ── 3 层 MPNN 消息传递 ─────────────────────────────────
        h = F.relu(self.bn1(self.conv1(h, edge_index, edge_attr)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.conv2(h, edge_index, edge_attr)))
        h = self.dropout(h)
        h = F.relu(self.bn3(self.conv3(h, edge_index, edge_attr)))
        h = self.dropout(h)

        # ── 提取全局节点特征 ────────────────────────────────────
        h_g = self.extract_global(h, data.batch)

        # ── 拼接物理条件向量，MLP 预测 ─────────────────────────
        return self.mlp(torch.cat([h_g, cond], dim=1))


# ============================================================
# Dataset_v2
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
        print(f"  [Scaler] 标准化器已保存至 {save_path}")

    def __getitem__(self, idx):
        graph, cond, label = super().__getitem__(idx)
        if self.scalers is not None:
            for feat_idx in range(7):
                raw    = float(self.data[idx][feat_idx + 3])
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
    return (mean_absolute_error(true_y, pred_y),
            r2_score(true_y, pred_y),
            np.sqrt(mean_squared_error(true_y, pred_y)))


def plot_pred(true_y, pred_y, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_y, pred_y, alpha=0.4, s=15, color='mediumseagreen', label='Predictions')
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
        self.model  = IL_MPNN(Args).to(self.device)
        self.optim  = torch.optim.Adam(
            self.model.parameters(), lr=Args['lr'], weight_decay=Args['weight_decay']
        )
        self.sched  = CosineAnnealingLR(self.optim, T_max=Args['epoch'], eta_min=1e-5)
        self.crit   = nn.HuberLoss(delta=1.0)
        self.ckpt   = os.path.join(CKPT_DIR, f'best_mpnn_seed{seed}.pth')

    def _save(self):
        torch.save({'model_state_dict': self.model.state_dict()}, self.ckpt)

    def _load(self):
        if os.path.exists(self.ckpt):
            ck = torch.load(self.ckpt, map_location=self.device)
            self.model.load_state_dict(ck['model_state_dict'])

    def train(self, train_loader, val_loader):
        es = EarlyStopping(patience=Args['patience'])
        best_val = float('inf')
        for epoch in range(1, Args['epoch'] + 1):
            self.model.train()
            tr_loss = 0.0
            bar = tqdm(train_loader, leave=False, desc=f'Ep{epoch:>3d}')
            for graph, cond, label in bar:
                graph, cond, label = (graph.to(self.device),
                                      cond.to(self.device),
                                      label.to(self.device))
                self.optim.zero_grad()
                loss = self.crit(self.model(graph, cond).flatten(), label.flatten())
                loss.backward()
                self.optim.step()
                tr_loss += loss.item()
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
                preds.extend(self.model(graph, cond).flatten().cpu().numpy().tolist())
                trues.extend(label.numpy().tolist())
        return np.array(preds), np.array(trues)


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print('正在加载数据集...')
    dataset = IL_set_v2(path=Args['data_path'])

    npz_path = f'split_{SPLIT_NAME}_indices.npz'
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'找不到 {npz_path}，请先运行 step1_anion_family_splitter.py')
    split_data = np.load(npz_path)
    train_idx, val_idx, test_idx = split_data['train'], split_data['val'], split_data['test']
    print(f'  Split {SPLIT_NAME} → Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}')

    scaler_path = os.path.join(CKPT_DIR, 'scalers.pkl')
    dataset.fit_scalers(train_idx, scaler_path)

    from torch.utils.data import Subset
    train_set   = Subset(dataset, train_idx)
    val_set     = Subset(dataset, val_idx)
    test_set    = Subset(dataset, test_idx)
    test_loader = DataLoader(test_set, batch_size=Args['batch_size'], shuffle=False)

    all_preds   = []
    all_results = []
    test_true_y = None

    for seed in SEEDS:
        print(f'\n{"─"*55}')
        print(f'  训练 MPNN | Split {SPLIT_NAME} | Seed {seed}')
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
                  f'MPNN Split-{SPLIT_NAME} Seed {seed} (R²={r2:.4f})',
                  os.path.join(FIG_DIR, f'mpnn_split{SPLIT_NAME}_seed{seed}.png'))
        pd.DataFrame({'true': true_y, 'pred': pred_y}).to_csv(
            os.path.join(CKPT_DIR, f'pred_seed{seed}.csv'), index=False
        )

    ensemble_pred              = np.mean(all_preds, axis=0)
    ens_mae, ens_r2, ens_rmse  = compute_metrics(test_true_y, ensemble_pred)

    df_res = pd.DataFrame(all_results)
    print(f'\n{"="*55}')
    print(f'  MPNN Split-{SPLIT_NAME} 单体结果：')
    print(df_res[['seed','MAE','R2','RMSE']].to_string(index=False))
    print(f'\n  单体均值 → '
          f'MAE: {df_res["MAE"].mean():.4f}±{df_res["MAE"].std():.4f} | '
          f'R²: {df_res["R2"].mean():.4f}±{df_res["R2"].std():.4f} | '
          f'RMSE: {df_res["RMSE"].mean():.4f}±{df_res["RMSE"].std():.4f}')
    print(f'\n  🏆 集成（{len(SEEDS)}种子均值）→ '
          f'MAE: {ens_mae:.4f} | R²: {ens_r2:.4f} | RMSE: {ens_rmse:.4f}')
    print(f'{"="*55}')

    df_res['split'] = SPLIT_NAME
    df_res.to_csv(f'mpnn_split{SPLIT_NAME}_results.csv', index=False)
    pd.DataFrame({'true': test_true_y, 'pred_ensemble': ensemble_pred}).to_csv(
        os.path.join(CKPT_DIR, 'pred_ensemble.csv'), index=False
    )
    plot_pred(test_true_y, ensemble_pred,
              f'MPNN Split-{SPLIT_NAME} Ensemble ({len(SEEDS)} seeds, R²={ens_r2:.4f})',
              os.path.join(FIG_DIR, f'mpnn_split{SPLIT_NAME}_ensemble.png'))

    print(f'\n✅ MPNN Split-{SPLIT_NAME} 全部完成！')
