# V7 真实留一冷媒（LORO）实验设计与前置检查报告

## 一、PowerShell 检查指令与关键发现

### 1. 检查指令 1：历史划分文件核验
执行指令：
```powershell
uv run python -c "from pathlib import Path; import numpy as np; d=Path('splits/HFC_loro_all'); fs=sorted(d.glob('*.npz')); print('fold_count=',len(fs)); [(lambda z,p: print('\n'+p.name, {k:{'shape':z[k].shape,'dtype':str(z[k].dtype)} for k in z.files}))(np.load(p,allow_pickle=True),p) for p in fs]"
```
**关键发现与致命隐患排查**：
- `splits/HFC_loro_all` 下现存的 12 个 `.npz` 文件，其 `train + val + test` 样本总和全部等于 **1,403**（索引范围 0 ~ 1,402）。
- 例如 `loro_R125.npz` 中 `test` 样本仅为 170 条；但当前 V7 使用的标准饱和 HFC 全集（`processed_tri_data_hfc2739`）中 R125 实际有 **260 条**。
- **结论**：旧的 `.npz` 文件是基于 V5/V6 早期的 1,403 条旧数据集编制的！**若直接传给 V7 的 2,739 维度加载器，会发生致命的行索引错位与物种串扰！**

### 2. 检查指令 2：V7 训练入口传参核验
执行指令：
```powershell
uv run python "v7_shadow_experiment/run_v7_pilot.py" --help
```
输出参数：
- `--model {V7-A, V7-B}`
- `--seeds SEEDS`
- `--split-file SPLIT_FILE`
- `--epoch`, `--batch_size`, `--lr`, `--weight_decay`, `--patience`, `--delta`
- `--desc_dropout_p` (V7-B 模态随机丢弃率，默认 0.3)

---

## 二、严格无泄漏的 2,739 LORO 新划分生成

为杜绝索引错位，我们在 `processed_tri_data_hfc2739`（N=2,739）上重新生成了 100% 对齐的 12 折干净 LORO 划分：
保存路径：`splits/HFC_loro_v7_2739/loro_{ref}.npz`

### 12 种纯 HFC 冷媒样本统计与划分构成

| 目标冷媒 (Held-Out) | 训练集 (Train) | 验证集 (Val) | 测试集 (Test, 纯留出) | 样本总数 | 泄漏断言 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **R32** | 1,860 | 207 | **672** | 2,739 | Train/Val 严格无 R32 |
| **R134a** | 1,880 | 210 | **649** | 2,739 | Train/Val 严格无 R134a |
| **R125** | 2,230 | 249 | **260** | 2,739 | Train/Val 严格无 R125 |
| **R152a** | 2,268 | 254 | **217** | 2,739 | Train/Val 严格无 R152a |
| **R134** | 2,296 | 258 | **185** | 2,739 | Train/Val 严格无 R134 |
| **R23** | 2,316 | 259 | **164** | 2,739 | Train/Val 严格无 R23 |
| **R161** | 2,327 | 257 | **155** | 2,739 | Train/Val 严格无 R161 |
| **R143a** | 2,358 | 263 | **118** | 2,739 | Train/Val 严格无 R143a |
| **R245fa** | 2,387 | 267 | **85** | 2,739 | Train/Val 严格无 R245fa |
| **R236fa** | 2,390 | 264 | **85** | 2,739 | Train/Val 严格无 R236fa |
| **R227ea** | 2,395 | 264 | **80** | 2,739 | Train/Val 严格无 R227ea |
| **R41** | 2,406 | 264 | **69** | 2,739 | Train/Val 严格无 R41 |

### 严格的物理与数学断言：
1. `set(train).isdisjoint(set(test)) == True`（目标冷媒物理隔绝）
2. `set(val).isdisjoint(set(test)) == True`（目标冷媒不参与验证与早停）
3. `set(train).isdisjoint(set(val)) == True`（训练集与验证集严格分组状态无重叠）
4. 归一化 Scaler 严格在 `train` 索引上计算均值和方差，目标冷媒 0 接触。

---

## 三、Kaggle 执行脚本设计（`run_v7_loro.py`）

新编写的专用执行脚本 `run_v7_loro.py`：
1. **支持单折调试**：`--target-ref R125` 或 `--split-file splits/HFC_loro_v7_2739/loro_R125.npz`；
2. **支持全量遍历**：留空 `--target-ref` 则自动循环执行 12 折并自动生成汇总 CSV；
3. **输出材料完备**：每折保存最优权重 `best_model.pth`、逐样本真实预测对比 `predictions_{target_ref}.csv`，以及 `loro_metrics_summary.csv`。

---

## 四、与 V6 历史基准的比较边界（需要另行核验）

此前整理中曾引用下列 V6 数值：R32 N=672、MAE=0.0715、R²=0.6988；R134a N=649、MAE=0.1182、R²=0.1816；R125 N=260、MAE=0.0805、R²=0.5957。**这些数字在本压缩包内没有附带原始 `loro_eval_results.csv`、对应 V6 训练脚本、每折训练日志或预测明细，因此本包审查无法独立确认这些数字的来源和可比性。**

特别地，旧的 `splits/HFC_loro_all/*.npz` 被核查为 1,403 行索引，而当前新划分为 2,739 行。该事实本身不能证明历史 V6 结果一定使用 1,403 行，但也不能用当前样本计数倒推历史结果一定基于 2,739 行。必须查看 V6 原始 CSV、数据版本、每折训练/测试身份、预处理方式以及指标是否使用 clip，才能决定是否可作 apples-to-apples 比较。

所以，当前这份包只负责 V7 的 LORO 数据和执行；V6 对照结论应在补查历史证据前保持为“待核实”，不得称已完成硬碰硬对比。
