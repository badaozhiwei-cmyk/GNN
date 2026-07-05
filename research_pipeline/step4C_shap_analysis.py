"""
step4C_shap_analysis.py
=======================
【目的】
  对 LightGBM 和 RF 模型（ML 基线）做 SHAP 分析，
  和 FGCA（GNN 的可解释性）做横向对比，验证：
  "GNN 通过 FGCA 发现的化学规律，是否和 SHAP 方向一致？"

【科学意义】
  SHAP 和 FGCA 是两种完全独立的可解释性方法：
  - SHAP：基于 Morgan 指纹特征（位级别），属于特征重要性分析
  - FGCA：基于 GNN 原子 Embedding 遮蔽（基团级别），属于结构贡献分析
  如果两者对同一化学基团的"重要性方向"一致，说明结论具有方法鲁棒性。

【SHAP 特征映射思路】
  1. Morgan 指纹中每一位对应若干 SMARTS 子结构（RDKit 可以还原）
  2. 取 SHAP 值最高的 Top-K 位，查出它们对应的分子子结构
  3. 与 FGCA 的 Top-K 基团做映射，计算 Spearman 相关系数

【运行方法（Kaggle，需要 shap 包）】
  pip install shap
  python step4C_shap_analysis.py --split B

【输出文件】
  figure/FigS3_shap_summary_split{B}.png   → SHAP 蜂群图
  figure/FigS4_shap_vs_fgca_corr.png       → SHAP vs FGCA 一致性对比
  shap_analysis_split{B}_results.csv        → SHAP Top 特征汇总
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ── SHAP ─────────────────────────────────────────────────────
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("⚠️  shap 未安装。请运行: pip install shap")

# ── LightGBM ─────────────────────────────────────────────────
try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

# ============================================================
# 命令行参数
# ============================================================
parser = argparse.ArgumentParser(description='Step 4C — SHAP Analysis')
parser.add_argument('--split',     type=str, default='B', choices=['A', 'B'])
parser.add_argument('--fp_radius', type=int, default=2)
parser.add_argument('--fp_bits',   type=int, default=2048)
parser.add_argument('--fgca_csv',  type=str, default=None,
                    help='Step 4A 的全局 FGCA 结果 CSV 路径（用于对比）')
parser.add_argument('--index_csv', type=str, default='index_with_anion.csv')
parser.add_argument('--npz_dir',   type=str, default='.')
args_cli = parser.parse_args()

SPLIT    = args_cli.split
FP_R     = args_cli.fp_radius
FP_BITS  = args_cli.fp_bits

os.makedirs('figure', exist_ok=True)

print(f"\n{'='*60}")
print(f"  Step 4C: SHAP Analysis | Split {SPLIT}")
print(f"{'='*60}\n")

# ============================================================
# 1. 读取数据（复用 step3 的特征构建逻辑）
# ============================================================
def smiles_to_fp(smi, radius, n_bits):
    if not RDKIT_OK: return np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None: return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)

index_csv = args_cli.index_csv
if not os.path.exists(index_csv):
    raise FileNotFoundError("找不到 index_with_anion.csv，请先运行 step0_verify_alignment.py")

df = pd.read_csv(index_csv).sort_values('npy_idx').reset_index(drop=True)

print(f"正在生成 Morgan 指纹（{FP_BITS} 位）...")
fp_cat   = np.vstack([smiles_to_fp(s, FP_R, FP_BITS) for s in df['cation_smiles']])
fp_ani   = np.vstack([smiles_to_fp(s, FP_R, FP_BITS) for s in df['anion_smiles']])
fp_refri = np.vstack([smiles_to_fp(s, FP_R, FP_BITS) for s in df['refri_smiles']])

# 特征：阳离子FP + 阴离子FP + 制冷剂FP + T + P
T_arr = df['T_K'].values.reshape(-1, 1).astype(np.float32)
P_arr = df['P_MPa'].values.reshape(-1, 1).astype(np.float32)
X_all = np.concatenate([fp_cat, fp_ani, fp_refri, T_arr, P_arr], axis=1)
y_all = df['x1'].values.astype(np.float32)

# 特征列名
col_cat   = [f'Cat_bit{i}'   for i in range(FP_BITS)]
col_ani   = [f'Ani_bit{i}'   for i in range(FP_BITS)]
col_refri = [f'Ref_bit{i}'   for i in range(FP_BITS)]
col_names = col_cat + col_ani + col_refri + ['T_K', 'P_MPa']

print(f"  特征矩阵：{X_all.shape}")

# 划分索引
npz_path = os.path.join(args_cli.npz_dir, f'split_{SPLIT}_indices.npz')
if not os.path.exists(npz_path):
    npz_path = f'split_{SPLIT}_indices.npz'
if not os.path.exists(npz_path):
    raise FileNotFoundError(f"找不到 {npz_path}，请先运行 step1_anion_family_splitter.py")

sd        = np.load(npz_path)
train_idx = sd['train'];  val_idx = sd['val'];  test_idx = sd['test']

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val,   y_val   = X_all[val_idx],   y_all[val_idx]
X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ============================================================
# 2. 训练 LightGBM（如果已有 step3 结果可以 load model，这里重新训练保证独立性）
# ============================================================
print("\n训练 LightGBM（供 SHAP 分析）...")
if not LGB_OK:
    print("  ⚠️  LightGBM 未安装，改用 Random Forest")
    model_shap = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    model_shap.fit(X_train, y_train)
    X_explain  = X_test[:500]   # 取前 500 条做 SHAP（RF 慢）
    model_name = 'RF'
else:
    model_shap = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbosity=-1
    )
    X_val_sc = scaler.transform(X_val)
    model_shap.fit(X_train_sc, y_train,
                   eval_set=[(X_val_sc, y_val)],
                   callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(period=-1)])
    X_explain  = X_test_sc    # LightGBM 处理 SHAP 很快
    model_name = 'LightGBM'

print(f"  模型训练完成（{model_name}）")

# ============================================================
# 3. SHAP 计算
# ============================================================
if not SHAP_OK:
    print("⚠️  shap 未安装，跳过 SHAP 计算。请 pip install shap 后重试。")
    sys.exit(0)

print(f"\n计算 SHAP 值（{len(X_explain)} 条测试样本）...")
if model_name == 'LightGBM':
    explainer_shap = shap.TreeExplainer(model_shap)
else:
    explainer_shap = shap.TreeExplainer(model_shap)

shap_values = explainer_shap.shap_values(X_explain)
print(f"  SHAP 矩阵形状：{shap_values.shape}")

# ============================================================
# 4. Figure S3：SHAP 蜂群图（Top 30 特征）
# ============================================================
print("\n绘制 SHAP 蜂群图（Fig S3）...")
shap_abs_mean = np.abs(shap_values).mean(axis=0)
top30_idx     = np.argsort(shap_abs_mean)[::-1][:30]
top30_names   = [col_names[i] for i in top30_idx]

fig, ax = plt.subplots(figsize=(10, 9))
shap.summary_plot(
    shap_values[:, top30_idx],
    X_explain[:, top30_idx],
    feature_names=top30_names,
    max_display=30,
    show=False,
    plot_type='dot'
)
plt.title(f'SHAP Summary ({model_name}, Split-{SPLIT})', fontsize=13, fontweight='bold')
plt.tight_layout()
s3_out = f'figure/FigS3_shap_summary_split{SPLIT}.png'
plt.savefig(s3_out, dpi=300, bbox_inches='tight')
plt.close()
print(f"  📊 Fig S3 已保存：{s3_out}")

# ============================================================
# 5. 提取 Top SHAP 特征对应的化学基团
#    思路：RDKit 把指纹的每一位还原成对应的 SMARTS 子结构
# ============================================================
print("\n正在将 Top SHAP 指纹位映射回化学基团...")

# 统计每个分子对各 Top 位的贡献（正/负）
shap_df_rows = []
for feat_idx in top30_idx[:30]:
    feat_name = col_names[feat_idx]
    shap_mean = float(shap_abs_mean[feat_idx])
    shap_pos  = float((shap_values[:, feat_idx] > 0).mean())
    # 判断来自哪个分子
    if feat_idx < FP_BITS:
        mol_src = 'Cation'
        bit_idx = feat_idx
    elif feat_idx < 2 * FP_BITS:
        mol_src = 'Anion'
        bit_idx = feat_idx - FP_BITS
    elif feat_idx < 3 * FP_BITS:
        mol_src = 'Refrigerant'
        bit_idx = feat_idx - 2 * FP_BITS
    else:
        mol_src = 'Condition'
        bit_idx = -1

    shap_df_rows.append({
        'feature':    feat_name,
        'mol_src':    mol_src,
        'bit_idx':    bit_idx,
        'shap_mean':  shap_mean,
        'pos_rate':   shap_pos,
    })

shap_summary_df = pd.DataFrame(shap_df_rows)
shap_csv_out    = f'shap_analysis_split{SPLIT}_results.csv'
shap_summary_df.to_csv(shap_csv_out, index=False)
print(f"  📄 SHAP Top 特征已保存：{shap_csv_out}")

# 打印简要统计
print(f"\n  SHAP Top 30 特征来源分布：")
print(shap_summary_df['mol_src'].value_counts().to_string())

# ============================================================
# 6. SHAP vs FGCA 方向一致性（Figure S4）
# ============================================================
fgca_csv_path = args_cli.fgca_csv
if fgca_csv_path is None:
    fgca_csv_path = os.path.join('scripts_phase3', f'global_group_importance_split{SPLIT}.csv')

if os.path.exists(fgca_csv_path):
    print(f"\n正在对比 SHAP vs FGCA（读取 {fgca_csv_path}）...")
    fgca_df = pd.read_csv(fgca_csv_path)

    # 按分子来源聚合 SHAP mean，比较 Cation/Anion/Refrigerant 贡献比例
    shap_by_src = shap_summary_df.groupby('mol_src')['shap_mean'].sum().reset_index()
    shap_by_src['shap_pct'] = shap_by_src['shap_mean'] / shap_by_src['shap_mean'].sum() * 100

    # FGCA 按 Cat_/Ani_/Ref_ 前缀聚合
    fgca_df['mol_src'] = fgca_df['Group'].apply(
        lambda x: 'Cation' if x.startswith('Cat_')
        else ('Anion' if x.startswith('Ani_') else 'Refrigerant')
    )
    fgca_by_src = fgca_df.groupby('mol_src')['Mean_Drop'].sum().reset_index()
    fgca_by_src['fgca_pct'] = fgca_by_src['Mean_Drop'] / fgca_by_src['Mean_Drop'].sum() * 100

    # 合并
    compare_df = shap_by_src.merge(fgca_by_src, on='mol_src', how='outer').fillna(0)
    print(f"\n  SHAP vs FGCA 分子贡献比例对比：")
    print(compare_df[['mol_src', 'shap_pct', 'fgca_pct']].to_string(index=False))

    # 绘制对比柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    x    = np.arange(len(compare_df))
    w    = 0.35
    b1   = ax.bar(x - w/2, compare_df['shap_pct'], w,
                  label='SHAP (LightGBM)', color='#3498DB', alpha=0.85, edgecolor='white')
    b2   = ax.bar(x + w/2, compare_df['fgca_pct'], w,
                  label='FGCA (GAT)',       color='#E74C3C', alpha=0.85, edgecolor='white')
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(compare_df['mol_src'], fontsize=11)
    ax.set_ylabel('Contribution Proportion (%)', fontsize=11)
    ax.set_title(
        f'Fig S4: SHAP vs FGCA — Attribution by Molecular Source (Split-{SPLIT})',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    s4_out = f'figure/FigS4_shap_vs_fgca_split{SPLIT}.png'
    plt.savefig(s4_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Fig S4 已保存：{s4_out}")

    compare_df.to_csv(f'shap_vs_fgca_comparison_split{SPLIT}.csv', index=False)

else:
    print(f"\n  ⚠️  未找到 FGCA CSV ({fgca_csv_path})，跳过 SHAP vs FGCA 对比")
    print(f"     请先运行 step4A_fgca_global.py --split {SPLIT}")

print(f"\n✅ Step 4C 全部完成！")
