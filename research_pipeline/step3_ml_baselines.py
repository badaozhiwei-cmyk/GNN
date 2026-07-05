"""
step3_ml_baselines.py
=====================
【目的】
  用 Morgan 指纹 + T + P 作为输入特征，在相同的 Split A / Split B 测试集上
  评估四个传统/浅层模型：Random Forest / XGBoost / LightGBM / MLP
  确保与 GAT/GIN/MPNN 的比较完全公平（相同测试集，相同指标）

【特征说明】
  - Morgan Fingerprint（半径=2，2048 位）：将分子 SMILES 编码为固定长度二进制向量
    注：这里把阳离子+阴离子的指纹拼接，然后再拼制冷剂指纹，共 2048×3 = 6144 维
    可选：只用阴离子+制冷剂（4096维），通过 --fp_mode 参数切换
  - T（温度，K）和 P（压力，MPa）：直接追加在指纹向量后面
  - 最终特征维度：6144 + 2 = 6146（或 4096+2=4098）

【运行方法】（在 Kaggle 上）
  python step3_ml_baselines.py --split A
  python step3_ml_baselines.py --split B
  python step3_ml_baselines.py --split B --fp_mode anion_refri  # 只用阴离子+制冷剂指纹

【输出文件】
  ml_baselines_split{A|B}_results.csv   → 四个模型的 MAE/R²/RMSE 汇总
  ml_baselines_split{A|B}_predictions/  → 每个模型的 pred/true CSV
  figure_ml_split{A|B}/                 → 每个模型的散点图
"""

import argparse
import os

# ── 自动定位项目根目录（脚本在 research_pipeline/ 子文件夹中）──────
import pathlib as _pl
ROOT = str(_pl.Path(__file__).resolve().parent.parent)
import os as _os; _os.chdir(ROOT)
# ─────────────────────────────────────────────────────────────────────
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# ── RDKit（Kaggle 上有）──────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("⚠️  RDKit 未安装，无法生成 Morgan 指纹。请在 Kaggle 上运行。")

# ── XGBoost / LightGBM（Kaggle 上有）────────────────────────
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("⚠️  xgboost 未安装")

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False
    print("⚠️  lightgbm 未安装")


# ============================================================
# 命令行参数
# ============================================================
parser = argparse.ArgumentParser(description='ML Baselines — RF/XGB/LGBM/MLP')
parser.add_argument('--split',    type=str, default='B', choices=['A', 'B'])
parser.add_argument('--fp_mode', type=str, default='all',
                    choices=['all', 'anion_refri'],
                    help='all=阳离子+阴离子+制冷剂三份指纹; anion_refri=仅阴离子+制冷剂')
parser.add_argument('--fp_radius', type=int, default=2,  help='Morgan 指纹半径')
parser.add_argument('--fp_bits',   type=int, default=2048, help='Morgan 指纹位数')
parser.add_argument('--index_csv', type=str,
                    default='../index_with_anion.csv',
                    help='Step 0 生成的索引映射表路径')
parser.add_argument('--npz_dir',   type=str, default='../',
                    help='split_X_indices.npz 所在目录')
args_cli = parser.parse_args()

SPLIT_NAME = args_cli.split
FP_RADIUS  = args_cli.fp_radius
FP_BITS    = args_cli.fp_bits
FP_MODE    = args_cli.fp_mode

PRED_DIR   = f'ml_baselines_split{SPLIT_NAME}_predictions'
FIG_DIR    = f'figure_ml_split{SPLIT_NAME}'
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

print(f"\n{'='*60}")
print(f"  ML Baselines | Split {SPLIT_NAME} | FP mode: {FP_MODE}")
print(f"  Morgan 指纹: radius={FP_RADIUS}, bits={FP_BITS}")
print(f"{'='*60}\n")


# ============================================================
# 1. 生成 Morgan 指纹特征矩阵
# ============================================================
def smiles_to_fp(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    """将 SMILES 转换为 Morgan 指纹（numpy 数组）"""
    if not RDKIT_OK:
        return np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def build_feature_matrix(index_df: pd.DataFrame,
                          fp_radius: int, fp_bits: int,
                          fp_mode: str) -> np.ndarray:
    """
    为 index_df 中的每一行构建特征向量：
      - 阴离子指纹（fp_bits 维）
      - 制冷剂指纹（fp_bits 维）
      - [fp_mode='all'] 阳离子指纹（fp_bits 维）
      - T（K）
      - P（MPa）
    """
    print(f"  正在生成 Morgan 指纹（共 {len(index_df)} 条）...")
    rows = []
    for i, row in index_df.iterrows():
        ani_fp   = smiles_to_fp(str(row['anion_smiles']),   fp_radius, fp_bits)
        refri_fp = smiles_to_fp(str(row['refri_smiles']),   fp_radius, fp_bits)
        T        = float(row['T_K'])
        P        = float(row['P_MPa'])

        if fp_mode == 'all':
            cat_fp = smiles_to_fp(str(row['cation_smiles']), fp_radius, fp_bits)
            feat   = np.concatenate([cat_fp, ani_fp, refri_fp, [T, P]])
        else:
            feat   = np.concatenate([ani_fp, refri_fp, [T, P]])
        rows.append(feat)

        if (i + 1) % 500 == 0:
            print(f"    已处理 {i+1}/{len(index_df)} 条")

    X = np.vstack(rows)
    print(f"  ✅ 特征矩阵形状：{X.shape}")
    return X


# ============================================================
# 2. 读取数据和划分索引
# ============================================================
index_csv = args_cli.index_csv
if not os.path.exists(index_csv):
    # 尝试当前目录
    if os.path.exists('index_with_anion.csv'):
        index_csv = 'index_with_anion.csv'
    else:
        raise FileNotFoundError(
            "找不到 index_with_anion.csv，请先运行 step0_verify_alignment.py"
        )

index_df = pd.read_csv(index_csv)
print(f"读取索引映射表：{len(index_df)} 条")

npz_path = os.path.join(args_cli.npz_dir, f'split_{SPLIT_NAME}_indices.npz')
if not os.path.exists(npz_path):
    if os.path.exists(f'split_{SPLIT_NAME}_indices.npz'):
        npz_path = f'split_{SPLIT_NAME}_indices.npz'
    else:
        raise FileNotFoundError(
            f"找不到 {npz_path}，请先运行 step1_anion_family_splitter.py"
        )

split_data = np.load(npz_path)
train_idx  = split_data['train']
val_idx    = split_data['val']
test_idx   = split_data['test']
print(f"Split {SPLIT_NAME} → Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

# 确保 index_df 的顺序和 npy_idx 一致
index_df = index_df.sort_values('npy_idx').reset_index(drop=True)

# ── 生成全量特征矩阵 ─────────────────────────────────────────
X_all = build_feature_matrix(index_df, FP_RADIUS, FP_BITS, FP_MODE)
y_all = index_df['x1'].values.astype(np.float32)

# ── 按索引切分 ───────────────────────────────────────────────
X_train = X_all[train_idx]
y_train = y_all[train_idx]
X_val   = X_all[val_idx]
y_val   = y_all[val_idx]
X_test  = X_all[test_idx]
y_test  = y_all[test_idx]

# ── 特征标准化（ML 模型需要，树模型不严格需要但无害）────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)


# ============================================================
# 3. 评估工具
# ============================================================
def compute_metrics(true_y, pred_y):
    mae  = mean_absolute_error(true_y, pred_y)
    r2   = r2_score(true_y, pred_y)
    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    return mae, r2, rmse


def plot_pred(true_y, pred_y, title, save_path, color='royalblue'):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_y, pred_y, alpha=0.4, s=12, color=color, label='Predictions')
    lims = [min(min(true_y), min(pred_y)) - 0.02,
            max(max(true_y), max(pred_y)) + 0.02]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Ideal (y=x)')
    ax.set_xlabel('Experimental x₁', fontsize=12)
    ax.set_ylabel('Predicted x₁',    fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_model(name, model, X_tr, y_tr, X_te, y_te, color):
    """训练并评估一个模型，保存结果"""
    print(f"\n  {'─'*45}")
    print(f"  训练 {name}...")
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae, r2, rmse = compute_metrics(y_te, pred)
    print(f"  ✅ {name} → MAE: {mae:.4f} | R²: {r2:.4f} | RMSE: {rmse:.4f}")

    # 保存预测 CSV
    pd.DataFrame({'true': y_te, 'pred': pred}).to_csv(
        os.path.join(PRED_DIR, f'{name.replace(" ","_")}_pred.csv'), index=False
    )
    # 保存散点图
    plot_pred(y_te, pred,
              f'{name} Split-{SPLIT_NAME} (R²={r2:.4f})',
              os.path.join(FIG_DIR, f'{name.replace(" ","_")}_split{SPLIT_NAME}.png'),
              color=color)
    return {'model': name, 'MAE': mae, 'R2': r2, 'RMSE': rmse}


# ============================================================
# 4. 定义并训练各模型
# ============================================================
results = []

# ── Random Forest ─────────────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
results.append(run_model('Random Forest', rf,
                          X_train, y_train, X_test, y_test,
                          color='#27AE60'))

# ── XGBoost ──────────────────────────────────────────────
if XGB_OK:
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        eval_metric='rmse',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train_sc, y_train,
                  eval_set=[(X_val_sc, y_val)],
                  verbose=False)
    pred_xgb = xgb_model.predict(X_test_sc)
    mae, r2, rmse = compute_metrics(y_test, pred_xgb)
    print(f"\n  {'─'*45}")
    print(f"  ✅ XGBoost → MAE: {mae:.4f} | R²: {r2:.4f} | RMSE: {rmse:.4f}")
    pd.DataFrame({'true': y_test, 'pred': pred_xgb}).to_csv(
        os.path.join(PRED_DIR, 'XGBoost_pred.csv'), index=False
    )
    plot_pred(y_test, pred_xgb,
              f'XGBoost Split-{SPLIT_NAME} (R²={r2:.4f})',
              os.path.join(FIG_DIR, f'XGBoost_split{SPLIT_NAME}.png'),
              color='#E67E22')
    results.append({'model': 'XGBoost', 'MAE': mae, 'R2': r2, 'RMSE': rmse})

# ── LightGBM ─────────────────────────────────────────────
if LGB_OK:
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    lgb_model.fit(X_train_sc, y_train,
                  eval_set=[(X_val_sc, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(period=-1)])
    pred_lgb = lgb_model.predict(X_test_sc)
    mae, r2, rmse = compute_metrics(y_test, pred_lgb)
    print(f"\n  {'─'*45}")
    print(f"  ✅ LightGBM → MAE: {mae:.4f} | R²: {r2:.4f} | RMSE: {rmse:.4f}")
    pd.DataFrame({'true': y_test, 'pred': pred_lgb}).to_csv(
        os.path.join(PRED_DIR, 'LightGBM_pred.csv'), index=False
    )
    plot_pred(y_test, pred_lgb,
              f'LightGBM Split-{SPLIT_NAME} (R²={r2:.4f})',
              os.path.join(FIG_DIR, f'LightGBM_split{SPLIT_NAME}.png'),
              color='#8E44AD')
    results.append({'model': 'LightGBM', 'MAE': mae, 'R2': r2, 'RMSE': rmse})

# ── MLP（sklearn）────────────────────────────────────────
mlp = MLPRegressor(
    hidden_layer_sizes=(1024, 512, 256),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=False,
)
results.append(run_model('MLP', mlp,
                          X_train_sc, y_train, X_test_sc, y_test,
                          color='#E74C3C'))

# ============================================================
# 5. 汇总结果
# ============================================================
df_res = pd.DataFrame(results)
df_res['split'] = SPLIT_NAME

print(f"\n{'='*60}")
print(f"  ML 基线模型汇总（Split {SPLIT_NAME}）：")
print(df_res[['model','MAE','R2','RMSE']].to_string(index=False))
print(f"{'='*60}")

out_csv = f'ml_baselines_split{SPLIT_NAME}_results.csv'
df_res.to_csv(out_csv, index=False)
print(f"\n  📄 汇总已保存：{out_csv}")

# ── 绘制 4 模型对比条形图 ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics    = ['R2', 'MAE', 'RMSE']
colors_bar = ['#27AE60', '#E67E22', '#8E44AD', '#E74C3C']
labels     = df_res['model'].tolist()

for ax, metric in zip(axes, metrics):
    vals = df_res[metric].values
    bars = ax.bar(labels, vals,
                  color=colors_bar[:len(labels)],
                  edgecolor='white', linewidth=0.8, alpha=0.88)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle(f'ML Baseline Comparison — Split {SPLIT_NAME}', fontsize=13, fontweight='bold')
plt.tight_layout()
bar_path = os.path.join(FIG_DIR, f'ml_comparison_split{SPLIT_NAME}.png')
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  📊 对比条形图已保存：{bar_path}")

print(f"\n✅ Step 3（ML 基线）Split-{SPLIT_NAME} 全部完成！")
print("  下一步：运行 GNN 模型（GAT/GIN/MPNN）后，将所有结果合并成论文大对比表")
