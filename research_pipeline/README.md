# Research Pipeline — 运行说明

本文件夹包含论文全部实验脚本，**按编号顺序运行**即可复现所有结果。

---

## 📁 文件夹结构

```
research_pipeline/          ← 本文件夹（所有脚本）
│
├── step0_verify_alignment.py      Step 0：数据对齐验证
├── step0_5_dataset_stats.py       Step 0.5：数据集统计 + Split B 比例预检
├── step1_anion_family_splitter.py Step 1：生成 Split A/B/C 索引文件
│
├── GAT_Runner_v4.py               Step 2a：GAT 训练（本文主模型）
├── GIN_Runner_v4.py               Step 2b：GIN 训练
├── MPNN_Runner.py                 Step 2c：MPNN 训练
├── step3_ml_baselines.py          Step 3：RF / XGBoost / LightGBM / MLP
│
├── step4A_fgca_global.py          Step 4A：FGCA 全局基团重要性排行
├── step4B_fgca_casestudy.py       Step 4B：分子热力图案例分析 + IG 对比
├── step4C_shap_analysis.py        Step 4C：SHAP 分析 + SHAP vs FGCA 对比
│
└── step5_paper_figures.py         Step 5：汇总 Table 2 + Fig 2~4（本地可运行）
```

---

## 🚀 Kaggle 完整运行顺序

```bash
# ── 进入本文件夹 ──
cd research_pipeline

# ── Step 0：数据准备（必须先跑）──
python step0_verify_alignment.py
python step0_5_dataset_stats.py
python step1_anion_family_splitter.py

# ── Step 2：GNN 模型训练（可开多个 Notebook 并行）──
python GAT_Runner_v4.py --split A
python GAT_Runner_v4.py --split B

python GIN_Runner_v4.py --split A
python GIN_Runner_v4.py --split B

python MPNN_Runner.py --split A
python MPNN_Runner.py --split B

# ── Step 3：ML 基线 ──
python step3_ml_baselines.py --split A
python step3_ml_baselines.py --split B

# ── Step 4：可解释性分析（需要先完成 Step 2）──
python step4A_fgca_global.py   --split B --seed 42
python step4B_fgca_casestudy.py --split B --seed 42
python step4C_shap_analysis.py  --split B

# ── Step 5：汇总出图（本地 Windows 也可运行）──
python step5_paper_figures.py
```

> **快速单种子测试（确认脚本无报错）：**
> ```bash
> python GAT_Runner_v4.py --split B --seeds 42 --epoch 5
> ```

---

## 📦 产出文件位置（均在项目根目录）

| 文件/文件夹 | 内容 |
|------------|------|
| `index_with_anion.csv` | 数据对齐索引映射表 |
| `split_A_indices.npz` | Split A 随机划分索引 |
| `split_B_indices.npz` | Split B 阴离子 OOD 划分索引 |
| `split_C_indices.npz` | Split C 制冷剂 OOD 划分索引 |
| `checkpoints_splitA/` | GAT Split A 模型权重 |
| `checkpoints_splitB/` | GAT Split B 模型权重 |
| `checkpoints_gin_split*/` | GIN 模型权重 |
| `checkpoints_mpnn_split*/` | MPNN 模型权重 |
| `gat_split*_results.csv` | GAT 5种子结果（mean±std）|
| `gin_split*_results.csv` | GIN 5种子结果 |
| `mpnn_split*_results.csv` | MPNN 5种子结果 |
| `ml_baselines_split*_results.csv` | ML 基线结果 |
| `table2_model_comparison.csv` | **论文 Table 2** |
| `figure/` | 所有论文图表（Fig 2~6, Fig S1~S4）|
| `scripts_phase3/global_group_importance_splitB.csv` | FGCA 全局排行数据 |

---

## 🔑 关键参数说明

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--split A/B` | 使用哪种数据划分 | `B` |
| `--seeds 42,123,...` | 多种子（逗号分隔）| `42,123,2024,3407,6666` |
| `--epoch 100` | 最大训练轮数 | `100` |
| `--seed 42` | FGCA 用哪个种子的模型 | `42` |

---

## 📋 模型对比方案

| 模型 | 类型 | 核心特点 |
|------|------|---------|
| **GAT（本文）** | GNN | 注意力加权 + 全局节点 |
| GIN | GNN | 最强图同构表达能力（WL test）|
| MPNN | GNN | NNConv 显式利用化学键特征 |
| Random Forest | ML | Morgan 指纹 + 集成树 |
| XGBoost | ML | 梯度提升树 |
| LightGBM | ML | 轻量梯度提升树 |
| MLP | ML | 前馈神经网络 |
