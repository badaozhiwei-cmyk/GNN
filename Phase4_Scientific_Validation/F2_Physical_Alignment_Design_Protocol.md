# F2 独立物理对齐系统架构与严谨统计协议 (F2 Physics Alignment Protocol - v2.4 最终审计定稿版)

> **核心科学定位**：F2 是归因敏感度与独立微观结合强度的 **Pilot / Proof-of-Consistency 外部物理验证**。
> **六项核心铁律**：
> 1. **主证据层级原则**：以 10 体系配对差值对比（Paired Contrast）为第一主分析，压制阴阳离子间天然组间均值分离（Between-group separation）的抬高偏差；
> 2. **模型分轨与 Pooled 约束**：V7-A 与 V7-B 完全分轨独立计算推断证据；Pooled 严格限定为探索性系综参考（Exploratory only），不作为主要统计依据；
> 3. **配对身份聚合严谨定性**：16 个唯一物理对分析消除了物理能量轴上的重复计数，归纳的是当前上下文条件下的典型归因；
> 4. **单纯形归一化掩盖效应实证**：在 Layer 2 对比归一化份额 $P_{\text{ref}}$、原始未归一化归因 $A_{\text{ref}}$ 与对数比 $\text{logit}(P_{\text{ref}})$，客观解释相关性衰减成因；
> 5. **严禁过度断言（Strict Epistemic Caution）**：
>    - 负值 $\Delta E_{\text{assoc}}$ 仅代表相对于孤立单体在电子能量上具有相互吸引结合（$\Delta E_{\text{assoc}} < 0$），绝不等于自由能变 $\Delta G < 0$ 或相平衡稳定性；
>    - $d_{\min} < 1.50$ Å 的极端配对（`[Ac]–R1234yf` 与 `[SCN]–R1234yf`，其底层制冷剂为六氟丙烯 HFP）定性为气相强短程吸引离群点，在缺乏电子结构波函数证据前不妄称“证明了准共价/电荷转移”；
> 6. **有限置换检验规范**：全量置换检验严格采用 Phipson & Smyth (2010) 公式 $p = \frac{n_{\text{extreme}} + 1}{n_{\text{perm}} + 1}$，完整公开 $n_{\text{perm}}$ 与 $n_{\text{extreme}}$。

---

## 一、有效物理样本量层级解耦 (Sample Hierarchy & Epistemic Boundary)

当前 Phase 2 的 430 次 Graph-IG 归因评估在物理上绝非 430 个独立样本。其数据、体系上下文与微观物理实体之间的层级映射关系严格界定如下：

$$\boxed{\begin{aligned}
430\text{ 次随机模型评估} &\;\longrightarrow\; 43\text{ 个实验工况点样本} \\
&\;\longrightarrow\; \mathbf{N_{\mathrm{system}} = 10\text{ 个唯一 IL–制冷剂三元体系上下文 (Unique System Contexts)}} \\
&\;\longrightarrow\; \mathbf{N_{\mathrm{unique\ pairs}} = 16\text{ 个去重后的唯一离子–制冷剂物理身份 (7 C-R + 9 A-R)}} \\
&\;\longrightarrow\; \mathbf{N_{\mathrm{link-instance}} = 20\text{ 个体系关联链接实例 (作为辅助观察，附组间效应警示)}}
\end{aligned}}$$

全量 xTB 配对优化生产池：
$$\mathbf{N_{\mathrm{xTB}} = 218\text{ 对离子–制冷剂复合物 (125 阴离子–制冷剂 + 93 阳离子–制冷剂)}}$$

---

## 二、证据分级与统计分析矩阵 (Evidence Tiers)

| 证据层级 (Evidence Tier) | 统计单元与样本量 | 物理描述符与归因指标 | 核心方法与统计推断 | 审稿人视角的科学定论与边界声明 |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 第一主证据 (Main Evidence)** | $N_{\text{system}} = 10$ 核心三元体系上下文 | $D_E = \|\Delta E_{\text{ani}}\| - \|\Delta E_{\text{cat}}\|$ 对比 $D_P = P_{\text{ani}} - P_{\text{cat}}$ | Spearman $\rho$ + Phipson-Smyth 置换检验 + Cluster Bootstrap (N=10) | paired contrast removes the simplest additive component-type separation from the analysis, while residual component-specific and system-level dependence may remain. 呈现中等正向趋势（$\rho \approx 0.46 \sim 0.54$），因样本量有限未达 0.05 阈值，构成 suggestive positive association。 |
| **Tier 2: 配对身份级 (Pair Identity)** | $N_{\text{unique}} = 16$ (7 C-R, 9 A-R) 唯一物理对 | 跨体系唯一对中位数归因与能量 | 聚合跨体系重复物理对中位数，分别检验全量与组内（Within-anion / Within-cation） | removes duplicated pair identities on the physical-energy axis and summarizes context-conditioned attribution at the unique-pair level. 阴离子组内显示中强单调性（Pooled $\rho=0.650$），阳离子单调性较弱（$\rho=0.214$）。 |
| **Tier 3: 辅助证据 (Supporting Instances)** | $N_{\text{instance}} = 20$ 体系关联链接实例 | 展开实例的绝对归因份额与能量 | Spearman $\rho$ + Cluster Bootstrap ($N_{\text{cluster}}=10$) | 降级为辅助观察，明确警示高相关性（$\rho \approx 0.82$）主要由阴阳离子天然组间两极分离驱动，不可直接解释为微观单调物理律。 |
| **Tier 4: 边界证据与敏感性检验 (Boundary Evidence)** | $N_{\text{system}} = 10$ 核心三元体系上下文 | 体系总结合强度 vs 归一化 $P_{\text{ref}}$、原始归因 $A_{\text{ref}}$ 与 $\text{logit}(P_{\text{ref}})$ | 三重归因形式敏感性对比检验 | 实证证明 $P_{\text{ref}}$ 的相关性衰减源自单纯形和为 1 归一化压缩（Compositionality）；去除分母后未归一化的 $A_{\text{ref}}$ 与能量存在显著正相关（V7-B $\rho = 0.723, p=0.022$）。 |
| **Tier 5: 单分子辅线 (Species Alignment)** | 全部 6 种制冷剂分子 | 偶极矩 $\mu$、极化率 $\alpha$、分子体积与 $P_{\text{ref}}$ | 区分非极性 R134 与强极性 R134a；顺反异构体对照 | 作为控制对照案例（Controlled Case Study），体现模型对异构体极性与体积的定性敏感度。 |

---

## 三、异常高能配对量子几何审计结论 (`[Ac]–R1234yf` & `[SCN]–R1234yf`)

在 218 对全量 xTB 优化中，`[Ac]–R1234yf`（$-47.59$ kcal/mol）与 `[SCN]–R1234yf`（$-47.61$ kcal/mol）的最短接触间距收敛至 $d_{\min} = 1.43$ Å，跌破常规范德华/弱氢键接触区间（$1.5 - 3.5$ Å）。

1. **底层分子真相**：
   在原始实验数据集 `index_with_anion.csv` 中，标签为 `R1234yf` 的分子 SMILES 为 `C(=C(F)F)(C(F)(F)F)F`（实际为全氟丙烯/六氟丙烯 HFP，全重原子 3 个 C、6 个 F，无氢原子）。
2. **物理作用机制**：
   全氟丙烯是极度缺电子的亲电烯烃。在气相/真空无溶剂化介电屏蔽条件下，富电子的小尺寸强亲核阴离子（乙酸根羧基氧与硫氰酸根硫/氮）向缺电子的不饱和碳中心发生强短程亲核靠拢。
3. **审稿人安全定论**：
   本协议严格限定措辞为：**“consistent with a very strong short-range attractive interaction between compact nucleophilic anions and electron-deficient fluoroalkene carbon centres under vacuum conditions”**。严禁在无电子结构波函数证据前使用“准共价络合”或“电荷转移态”等过度断言。

---

## 四、生产完整性与多构象采样客观描述规范

1. **二聚体级收敛完备度**：
   全量 218 个离子–制冷剂配对体系中，**218/218 (100%)** 均获得至少一个严格几何优化收敛的物理复合物结构。
2. **构向采样级收敛率**：
   在全量 872 个初始采样构向中，**812/872 (93.12%)** 严格几何收敛。
3. **冗余采样客观定性**：
   (170+37)/218 = **95.0% 的配对体系拥有 $\ge 3$ 个几何收敛构象**，规范表述为：
   *“providing redundant converged starting-orientation outcomes for 95.0% of pair systems, reducing the sensitivity to initial spatial orientation without claiming global minimum completeness.”*
