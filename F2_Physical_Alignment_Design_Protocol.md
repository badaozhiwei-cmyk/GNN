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
| **Tier 1: 第一主推论证据 (Primary Inferential)** | $N_{\text{system}} = 10$ 核心三元体系上下文 | $D_E = \|\Delta E_{\text{ani}}\| - \|\Delta E_{\text{cat}}\|$ 对比 $D_P = P_{\text{ani}} - P_{\text{cat}}$ | Spearman $\rho$ + Phipson-Smyth ($p_{\text{mc}}$) + Cluster Bootstrap (N=10) | paired contrast 从根本上剔除加性组分分离效应（Additive component-type separation）。V7-A $\rho=0.552, p_{\text{mc}}=0.1091$; V7-B $\rho=0.358, p_{\text{mc}}=0.3111$; Pooled $\rho=0.527, p_{\text{mc}}=0.1201$。因样本量有限未达 0.05 阈值，定性为 suggestive positive association，不作过度断言。 |
| **Tier 2: 配对身份级描述/敏感性 (Pair Identity)** | $N_{\text{unique}} = 16$ (7 C-R, 9 A-R) 唯一物理对 | 跨体系唯一对中位数归因与能量 | 聚合跨体系重复物理对中位数，分别检验全量与组内（Within-anion / Within-cation） | removes duplicated pair identities on the physical-energy axis and summarizes context-conditioned attribution at the unique-pair level. Pooled-16 相关性虽高（V7-A $\rho=0.768, p_{\text{mc}}=0.0008$; V7-B $\rho=0.700, p_{\text{mc}}=0.0022$），但主要受阴阳离子两极分离驱动；组内单调性较弱（阴离子组内 V7-A $\rho=0.517, p_{\text{mc}}=0.1600$; 阳离子组内 $\rho=-0.036$）；CI 标注为 N/A（杜绝在 20 link instances 上的错位 Bootstrap）。 |
| **Tier 3: 辅助描述性证据 (Supporting Instances)** | $N_{\text{instance}} = 20$ 体系关联链接实例 | 展开实例的绝对归因份额与能量 | Spearman $\rho$ + Cluster Bootstrap ($N_{\text{cluster}}=10$) | 降级为纯描述性辅助观察，明确警示高相关性（V7-A $\rho=0.755, 95\%\text{ CI: } [0.597, 0.895]$; V7-B $\rho=0.697, 95\%\text{ CI: } [0.397, 0.911]$）主要由组间两极分离驱动，因聚类结构打破独立可置换性，不报告假设独立的置换检验 $p$ 值。 |
| **Tier 4: 边界证据与敏感性检验 (Boundary Evidence)** | $N_{\text{system}} = 10$ 核心三元体系上下文 | 体系总结合强度 vs 归一化 $P_{\text{ref}}$、原始归因 $A_{\text{ref}}$ 与 $\text{logit}(P_{\text{ref}})$ | 三重归因形式敏感性对比 + 配对 Bootstrap $\Delta\rho$ 检验 | 实证检验 $P_{\text{ref}}$ 的相关性衰减是否源自单纯形归一化掩盖；配对 Bootstrap 直接检验 $\Delta\rho = \rho(\text{Raw } A_{\text{ref}}) - \rho(\text{Norm } P_{\text{ref}})$：V7-A $\Delta\rho = +0.376, 95\%\text{ CI: } [-0.113, 1.019], P(\Delta\le 0)=0.1354$; V7-B $\Delta\rho = +0.303, 95\%\text{ CI: } [-0.200, 0.957], P(\Delta\le 0)=0.1367$。客观记录衰减趋势，不妄称证明因果。 |
| **Tier 5: 单分子辅线 (Species Alignment)** | 全部 6 种制冷剂分子 | 偶极矩 $\mu$、极化率 $\alpha$、分子体积与 $P_{\text{ref}}$ | 区分非极性 R134 与强极性 R134a；顺反异构体对照 | 作为控制对照案例（Controlled Case Study），体现模型对异构体极性与体积的定性敏感度。 |

---

## 三、化学身份纠正与物理产物池解耦 (Chemical Identity Correction & Artifact Decoupling)

1. **历史身份污染事件 (Historical Identity-Contamination Incident)**：
   在原始历史数据集中，标签为 `R1234yf` 的分子 SMILES 曾误标为六氟丙烯 HFP（`C(=C(F)F)(C(F)(F)F)F`），导致历史计算中 `[Ac]–R1234yf` 产生了高达 $-47.59\text{ kcal/mol}$ 的异常亲核结合能。
2. **纠正与洁净重算行动 (Corrective Action)**：
   使用官方认证的真实 R1234yf 分子身份（SMILES: `C=C(F)C(F)(F)F`, 分子式 $C_3H_2F_4$, 7 重原子），在 GFN2-xTB 6.6.1 环境下对全部 12 对 R1234yf 离子–制冷剂配对执行了完整洁净重算。真实结合能已恢复至正常物理范畴：
   - `[Ac]–R1234yf` 纠正为 **$-21.900483\text{ kcal/mol}$**（接触间距 $1.72\text{ \AA}$）；
   - `[SCN]–R1234yf` 纠正为 **$-11.283542\text{ kcal/mol}$**（接触间距 $2.31\text{ \AA}$）；
   - `[BF4]–R1234yf` 为 **$-11.776515\text{ kcal/mol}$**；`[PF6]–R1234yf` 为 **$-10.638888\text{ kcal/mol}$**。
3. **当前 F2 物理产物池 (Current F2 Physical Pool)**：
   F2 的 Core-10 体系直接基于版本受控的 `full_pair_interaction_results_v7_clean.csv` 自动派生，绝不使用历史未纠正的 Core-10 表。
4. **独立哨兵接缝诊断范围声明 (Independent Seam Diagnostics Scope)**：
   在 16 对哨兵接缝审计中发现的两个离群点（`[Tf2N]–R32` 与 `[SCN]–R134a`）属于全局接缝连续性诊断记录，**这两对配对均不在 F2 的 10 个核心三元体系及 16 个唯一物理对之内**，未进入 F2 的任何统计检验推断。
5. **严谨定论表述规范 (Standard Epistemic Claim)**：
   *“F2 uses the Core-10 physical interaction values resolved from the version-controlled v7 pair artifact; the artifact comprises 12 newly recomputed R1234yf pairs and 206 retained historical pair records. Independent seam anomalies identified in two sentinel pairs were outside the Core-10 analysis scope.”*

---

## 四、生产完整性与多构象采样客观描述规范

1. **二聚体级收敛完备度**：
   全量 218 个离子–制冷剂配对体系中，**218/218 (100%)** 均获得至少一个严格几何优化收敛的物理复合物结构。
2. **构向采样级收敛率**：
   在全量 872 个初始采样构向中，**812/872 (93.12%)** 严格几何收敛。而在 16 组哨兵接缝审计重算中，**63/64 (98.44%)** 构向优化严格收敛（Sentinel 4 `[Tf2N]+R134` 的 Orientation 2 未收敛，依照规则在 3 个收敛取向中选取 argmin）。
3. **冗余采样客观定性**：
   (170+37)/218 = **95.0% 的配对体系拥有 $\ge 3$ 个几何收敛构象**，规范表述为：
   *“providing redundant converged starting-orientation outcomes for 95.0% of pair systems, reducing the sensitivity to initial spatial orientation without claiming global minimum completeness.”*
