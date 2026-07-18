"""
step3_ml_baselines_v2.py
========================
【目的】
  传统 ML Baseline：用 Morgan Fingerprint + T + P 作为输入，
  在 L0-L4 泛化阶梯的相同 split 上评估 Random Forest / XGBoost / MLP。
  确保与 GAT_v5 的比较完全公平（相同测试集，相同指标）。

【运行方法】（在 Kaggle 上）
  python research_pipeline/step3_ml_baselines_v2.py --level L0
  python research_pipeline/step3_ml_baselines_v2.py --level L2
  python research_pipeline/step3_ml_baselines_v2.py --level L0 L1 L2 L3 L4  # 批量

【输出】
  results_v5/ml_baselines_{LEVEL}.csv
  figure_v5/ml_{LEVEL}_{model}.png
"""

import argparse
import os
import sys
import pathlib as pl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# ── 自动定位项目根目录 ──
ROOT = str(pl.Path(__file__).resolve().parent.parent)
os.chdir(ROOT)

# ── RDKit ──
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("⚠️  RDKit 未安装，请在 Kaggle 上运行。")
    sys.exit(1)

# ── XGBoost ──
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("⚠️  xgboost 未安装，将跳过 XGBoost。")


# ============================================================
# 工具函数
# ============================================================
def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """SMILES -> Morgan Fingerprint (numpy array)"""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    为每行构建特征向量:
      Cation FP (2048) + Anion FP (2048) + Refrigerant FP (2048) + T + P
      = 6146 维
    """
    print(f"  正在生成 Morgan 指纹（共 {len(df)} 条）...")
    rows = []
    for i, row in df.iterrows():
        cat_fp   = smiles_to_fp(row['cation_smiles'])
        ani_fp   = smiles_to_fp(row['anion_smiles'])
        refri_fp = smiles_to_fp(row['refri_smiles'])
        T = float(row['T_K'])
        P = float(row['P_MPa'])
        feat = np.concatenate([cat_fp, ani_fp, refri_fp, [T, P]])
        rows.append(feat)
        if (i + 1) % 1000 == 0:
            print(f"    已处理 {i+1}/{len(df)}")
    X = np.vstack(rows)
    print(f"  ✅ 特征矩阵: {X.shape}")
    return X


def compute_metrics(true_y, pred_y):
    mae  = mean_absolute_error(true_y, pred_y)
    r2   = r2_score(true_y, pred_y)
    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    return mae, r2, rmse


def plot_parity(true_y, pred_y, title, save_path, color='royalblue'):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_y, pred_y, alpha=0.4, s=12, color=color)
    lims = [min(min(true_y), min(pred_y)) - 0.02,
            max(max(true_y), max(pred_y)) + 0.02]
    ax.plot(lims, lims, 'r--', lw=1.5)
    ax.set_xlabel('Experimental x₁', fontsize=12)
    ax.set_ylabel('Predicted x₁',    fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# 主程序
# ============================================================
def run_level(level: str):
    print(f"\n{'='*60}")
    print(f"  ML Baselines v2 | Level: {level}")
    print(f"{'='*60}")

    # 1. 读取数据
    df = pd.read_csv('index_with_anion.csv')
    df = df.sort_values('npy_idx').reset_index(drop=True)

    # 2. 读取 split
    npz_path = f'split_{level}_indices.npz'
    if not os.path.exists(npz_path):
        print(f"  ❌ 找不到 {npz_path}，跳过 {level}")
        return None
    loaded = np.load(npz_path)
    train_idx = loaded['train']
    val_idx   = loaded['val']
    test_idx  = loaded['test']
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # 3. 构建特征
    X_all = build_features(df)
    y_all = df['x1'].values.astype(np.float32)

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val,   y_val   = X_all[val_idx],   y_all[val_idx]
    X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

    # 合并 train+val（传统 ML 不需要 early stopping）
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # 标准化
    scaler = StandardScaler()
    X_trainval_sc = scaler.fit_transform(X_trainval)
    X_test_sc     = scaler.transform(X_test)

    # 4. 定义模型
    models = [
        ('Random_Forest', RandomForestRegressor(
            n_estimators=500, max_depth=20, n_jobs=-1, random_state=42
        ), 'forestgreen'),
        ('MLP', MLPRegressor(
            hidden_layer_sizes=(512, 256, 128),
            max_iter=500, early_stopping=True,
            random_state=42
        ), 'darkorange'),
    ]
    if XGB_OK:
        models.insert(1, ('XGBoost', xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbosity=0
        ), 'dodgerblue'))

    # 5. 训练 & 评估
    results = []
    os.makedirs('figure_v5', exist_ok=True)
    os.makedirs('results_v5', exist_ok=True)

    for name, model, color in models:
        print(f"\n  训练 {name}...")
        # RF / XGB 不需要标准化，但统一用标准化后的也无害
        model.fit(X_trainval_sc, y_trainval)
        pred = model.predict(X_test_sc)
        pred = np.clip(pred, 0.0, 1.0)
        mae, r2, rmse = compute_metrics(y_test, pred)
        print(f"  ✅ {name} → MAE: {mae:.4f} | R²: {r2:.4f} | RMSE: {rmse:.4f}")
        results.append({'level': level, 'model': name, 'mae': mae, 'r2': r2, 'rmse': rmse})

        plot_parity(y_test, pred,
                    f'{name} | {level} (R²={r2:.4f})',
                    f'figure_v5/ml_{level}_{name}.png', color)

    # 6. 保存
    res_df = pd.DataFrame(results)
    res_df.to_csv(f'results_v5/ml_baselines_{level}.csv', index=False)
    print(f"\n  📊 结果已保存至 results_v5/ml_baselines_{level}.csv")
    print(res_df.to_string(index=False))
    return res_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Baselines v2 for L0-L4')
    parser.add_argument('--level', nargs='+', default=['L0'],
                        help='Split levels (e.g., L0 L1 L2 L3 L4)')
    cmd = parser.parse_args()

    all_results = []
    for lv in cmd.level:
        r = run_level(lv)
        if r is not None:
            all_results.append(r)

    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv('results_v5/ml_baselines_all_levels.csv', index=False)
        print(f"\n{'='*60}")
        print("  全部 ML Baseline 汇总：")
        print(combined.to_string(index=False))
        print(f"{'='*60}")
