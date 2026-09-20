# Supplementary Information: Cross-Axis Generalization Boundaries of Molecular Graph Neural Networks for Multicomponent Refrigerant Solubility

---

## Section S1. Data Provenance, Curation, and Strict Split Definitions

### S1.1 Dataset Compilation and Sourcing
The experimental data utilized in this study comprise two authoritative collections of vapor–liquid equilibrium solubility measurements ($x_1$, mole fraction of refrigerant in ionic liquid):
1. **Saturated Hydrofluorocarbon (HFC) Dataset ($N=2739$)**: Collected from published literature across 12 distinct HFC refrigerant solutes (R32, R125, R134a, R152a, R23, R134, R161, R143a, R245fa, R236fa, R227ea, R41) paired with 86 unique room-temperature ionic liquids comprising diverse imidazolium, pyridinium, pyrrolidinium, phosphonium, and ammonium cations with fluorinated and non-fluorinated anions.
2. **Unsaturated Hydrofluoroolefin/Hydrochlorofluoroolefin (HFO/HCFO) Dataset ($N=1106$)**: Compiled from recent experimental literature for fourth-generation low-global-warming-potential (GWP) working fluids (R1234yf, R1234ze(E), R1233zd(E), R1336mzz(E), R1336mzz(Z)) across an overlapping suite of ionic liquids.

### S1.2 Authoritative Sample Size and Metric Integrity Audit
To ensure absolute mathematical consistency and preclude ambiguities in sample counts, all test point counts were formally audited and frozen via `step14_final_cross_axis_freeze_audit.py`:

| OOD Benchmark Axis | Primary Shift Target | Authoritative Test $N$ | Aggregation Basis | Outlawed Legacy Provisional Counts |
| :--- | :--- | :---: | :--- | :--- |
| **M1 LORO** | Refrigerant Identity | **1403** | 5-seed mean; macro across 12 refrigerants | ~260 (single species R125) |
| **B1 Anion OOD** | Anion Matrix (Fam-2) | **513** | 5-seed mean; pooled test points | ~584 (preliminary cluster) |
| **B2 Anion OOD** | Anion Matrix (Fam-3) | **598** | 5-seed mean; pooled test points | ~162 (preliminary cluster) |
| **L2 Compositional**| Unseen Cation–Anion Pairs | **374** | 5-seed mean; pooled test points | — |
| **HFO/HCFO Zero-Shot**| Saturation / Cross-Family | **1106** | 5-seed mean; pooled test points | — |

### S1.3 Thermodynamic Consistency and Sample Integrity
Every entry in both datasets underwent automated physical validation:
- **Thermodynamic State Duplicates and Literature Provenance**: The comprehensive data lineage audit (Step 21) identified 13 identical thermodynamic state combinations ($(T, P, \text{solute}, \text{IL})$) originating from independent published experimental literature sources. In accordance with physical data retention best practices, these measurements were verified to have consistent, non-conflicting solubility targets and were deliberately retained as distinct experimental records reflecting inter-laboratory measurement reproducibility rather than being silently discarded or deduplicated; all sample-level identifiers remained strictly unique, with exactly 0 cross-split leakage.
- **Algebraic Verification**: Reduced state variables were verified to satisfy $T_r = T / T_c$ and $P_r = P / P_c$ using the compiled critical-property data ($T_c, P_c, \omega$) employed for the reduced coordinate calculations. Operating ranges span $T \in [283.15, 363.15]\text{ K}$ and $P \in [0.01, 2.50]\text{ MPa}$.
- **Absence of Data Leakage**: In all five OOD split configurations, training, validation, and test subsets were strictly partitioned at the molecular entity level before data standardization scalers were fitted.

### S1.4 Critical Thermodynamic Property Provenance for Unsaturated Solutes
The macroscopic thermodynamic parameters ($T_c, P_c, \omega$) utilized in the reduced coordinate calculations ($M_{\rm reduced}$) for the external unsaturated (HFO/HCFO) evaluation benchmark were obtained from internationally recognized reference equations of state (NIST REFPROP database and literature formulations), summarized in **Table S2**.

| Refrigerant | Chemical Name | CAS Registry No. | Formula | $T_c$ (K) | $P_c$ (MPa) | $\omega$ | Source / Reference Equation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **R1234yf** | 2,3,3,3-Tetrafluoropropene | 754-12-1 | $\text{C}_3\text{H}_2\text{F}_4$ | 367.85 | 3.3820 | 0.2760 | Lemmon & Akasaka, *J. Chem. Eng. Data* (2014) |
| **R1234ze(E)** | *trans*-1,3,3,3-Tetrafluoropropene | 29118-24-9 | $\text{C}_3\text{H}_2\text{F}_4$ | 382.51 | 3.6350 | 0.3130 | Akasaka, *Int. J. Refrig.* (2011) |
| **R1233zd(E)** | *trans*-1-Chloro-3,3,3-trifluoropropene | 102687-65-0 | $\text{C}_3\text{H}_2\text{ClF}_3$ | 438.86 | 3.5828 | 0.3040 | Mondéjar et al., *J. Chem. Eng. Data* (2015) |
| **R1336mzz(E)** | *trans*-1,1,1,4,4,4-Hexafluoro-2-butene | 66711-86-2 | $\text{C}_4\text{H}_2\text{F}_6$ | 403.53 | 2.7790 | 0.4128 | Akasaka et al., *J. Chem. Eng. Data* (2018) |
| **R1336mzz(Z)** | *cis*-1,1,1,4,4,4-Hexafluoro-2-butene | 692-49-9 | $\text{C}_4\text{H}_2\text{F}_6$ | 444.50 | 2.9037 | 0.3860 | Tanaka et al., *Int. J. Refrig.* (2016) |

---

## Section S2. Molecular Featurization and Neural Network Architecture

### S2.1 Tri-Graph Representation
Each absorption mixture is represented as three distinct molecular graphs:
$$\mathcal{M} = \{\mathcal{G}_{\rm cation}, \mathcal{G}_{\rm anion}, \mathcal{G}_{\rm refrigerant}\}$$
For each graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, atom node features are encoded via $7$ discrete categorical lookup tables into a shared embedding space ($d_{\rm emb} = 300$):
1. Atomic number ($119$ bins)
2. Hybridization state ($\text{sp}, \text{sp}^2, \text{sp}^3, \text{sp}^3\text{d}, \text{sp}^3\text{d}^2$; $8$ bins)
3. Aromaticity indicator ($2$ bins)
4. Degree of connectivity ($7$ bins)
5. Formal charge ($3$ bins)
6. Pauling electronegativity bucket ($8$ bins)
7. Covalent radius bucket ($8$ bins)

In addition, each atom receives a component identity embedding ($3$ classes: cation, anion, solute). The three component subgraphs are assembled into a combined tri-graph representation, to which a single virtual Global Token node is appended and bidirectionally connected to all constituent atoms across the ternary mixture, initialized with an independent learnable parameter vector $\mathbf{h}_{\rm global} \in \mathbb{R}^{300}$.

Chemical bond edge features are mapped via $3$ categorical lookup tables into $d_{\rm emb} = 300$:
1. Bond type (global connection, single, double, triple, aromatic; $5$ categories)
2. Ring membership flag ($2$ categories)
3. Aromaticity flag ($2$ categories)

In the diagnostic stereochemical ablation (Section 2.7), edge features are expanded to $4$ dimensions by incorporating a $6$-category directional stereochemistry tag (None, Any, Z, E, Cis, Trans).

### S2.2 Scalar Condition Feature Schemas
The scalar conditioning branch feeds into the fusion MLP alongside the pooled graph representations:
- **Baseline $M_0$ (9-dimensional scalar vector)**:
  1. Operating temperature $T$ (K)
  2. Operating pressure $P$ (MPa)
  3. Refrigerant maximum absolute partial charge $q_{\rm max}^{\rm ref}$
  4. Refrigerant octanol-water partition coefficient $\text{MolLogP}^{\rm ref}$
  5. Anion molecular weight $\text{MW}^{\rm ani}$
  6. Cation maximum absolute partial charge $q_{\rm max}^{\rm cat}$
  7. Cation topological polar surface area $\text{TPSA}^{\rm cat}$
  8. Refrigerant molecular weight $\text{MW}^{\rm ref}$
  9. Cation molecular weight $\text{MW}^{\rm cat}$
- **Intervention $M_{\rm reduced}$ (12-dimensional scalar vector)**:
  Augments $M_0$ with three corresponding-states scaling coordinates:
  10. Reduced temperature $T_r = T / T_c$
  11. Reduced pressure $P_r = P / P_c$
  12. Pitzer acentric factor $\omega$

### S2.3 Architecture and Hyperparameters
- **Graph Encoder**: 3-layer relational Graph Attention Network v2 (GATv2Conv) with edge-attribute conditioning ($\text{edge\_dim} = 300$). Layer channel progression is $300 \to 512 \to 1024 \to 512$, with $K = 4$ attention heads, `concat=False`, ReLU activation, and dropout probability $p_{\rm drop} = 0.2$.
- **Readout Pooling**: The topological representation $\mathbf{h}_{\rm topo} \in \mathbb{R}^{512}$ is extracted directly from the updated virtual Global Token node at the final layer.
- **Fusion MLP**: Combines $\mathbf{h}_{\rm topo}$ with standardized scalar condition vector $\mathbf{z} \in \mathbb{R}^{d_{\rm cond}}$ ($d_{\rm cond} = 9$ for $M_0$, $12$ for $M_{\rm reduced}$):
  $$\text{MLP}: \mathbb{R}^{512 + d_{\rm cond}} \xrightarrow{\text{Linear}(1024)} \text{Norm} \xrightarrow{\text{ReLU}} \text{Dropout}(0.4) \xrightarrow{\text{Linear}(512)} \text{Norm} \xrightarrow{\text{ReLU}} \text{Dropout}(0.3) \xrightarrow{\text{Linear}(1)} \hat{x}_1$$
- **Training Protocol**: Adam optimizer, initial learning rate $\eta_0 = 1.0 \times 10^{-3}$ with Cosine Annealing decay down to $\eta_{\rm min} = 1.0 \times 10^{-5}$, weight decay $\lambda = 1.0 \times 10^{-6}$, mini-batch size of $32$, and robust Huber loss with threshold $\delta = 0.05$. Models were trained for $100$ epochs with early stopping patience of $15$ epochs on validation loss. Five independent seeds ($\mathcal{S} = \{42, 43, 44, 45, 46\}$) were executed for each benchmark configuration. Scalers ($\boldsymbol{\mu}_{\rm train}, \boldsymbol{\sigma}_{\rm train}$) were fit strictly on the training partition of each fold.

---

## Section S3. Statistical Robustness and Sensitivity Analyses

### S3.1 L2 Paired Bootstrap and Permutation Testing
To verify the hypothesis that $M_{\rm reduced}$ provides a statistically significant error reduction on unseen cation–anion mixtures (L2 benchmark, $N=374$), we conducted non-parametric resampling:
- **Pointwise Paired Difference**: $\Delta e_i = |y_i - \hat{y}_i^{(M_0)}| - |y_i - \hat{y}_i^{(M_{\rm red})}|$
- **Bootstrap Protocol**: $B = 10,000$ resamples with replacement over the 374 points.
- **Empirical Distribution**:
  - Mean: $0.002724$
  - Standard error: $0.000612$
  - 95% Studentized Confidence Interval: $[0.001515, 0.003915]$
  - Wilcoxon signed-rank test (one-sided): $W = 44102.0$, $p = 5.18 \times 10^{-6}$
  - Pairwise win rate: $59.89\%$ of instances favored $M_{\rm reduced}$.

### S3.2 Sensitivity to HCFO Exclusion in Cross-Family Transfer
The external unsaturated dataset includes 90 points for the hydrochlorofluoroolefin R1233zd(E). Because chlorine is absent from all training HFCs, we tested whether the observed cross-family gain was driven by R1233zd(E):

| Dataset Scope | $N$ | $M_0$ Ensemble MAE | $M_{\rm red}$ Ensemble MAE | $\Delta\text{MAE}$ (%) | $M_0$ $R^2$ | $M_{\rm red}$ $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Unsaturated** (HFO + HCFO) | 1106 | 0.0367 | 0.0302 | −17.6% | 0.6045 | 0.7384 |
| **Pure HFO Only** (Excl. R1233zd(E)) | 1016 | 0.0374 | 0.0302 | \textbf{−19.4\%} | 0.6028 | 0.7410 |

The performance advantage of $M_{\rm reduced}$ remains completely robust on the pure-HFO subset ($-19.4\%$ error reduction, $\Delta R^2 = +0.138$), supporting the robustness of thermodynamic coordinate regularization across purely fluorinated olefins.

---

## Section S4. Comprehensive Benchmark Compendium

### S4.1 Leave-One-Refrigerant-Out (M1 LORO) Detailed Species Performance
Performance across all 12 held-out HFC refrigerants under baseline $M_0$ and intervention $M_{\rm reduced}$ (5-seed ensemble):

| Refrigerant | Formula | $N$ | $T_c$ (K) | $P_c$ (MPa) | $\omega$ | $M_0$ MAE | $M_{\rm red}$ MAE | $\Delta\text{MAE}$ (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R134a** | $\text{CH}_2\text{FCF}_3$ | 288 | 374.21 | 4.06 | 0.327 | 0.1525 | 0.0769 | \textbf{−49.6\%} |
| **R32** | $\text{CH}_2\text{F}_2$ | 287 | 351.26 | 5.78 | 0.277 | 0.0952 | 0.1036 | +8.8\% |
| **R125** | $\text{CHF}_2\text{CF}_3$ | 170 | 339.17 | 3.62 | 0.305 | 0.1094 | 0.0369 | \textbf{−66.3\%} |
| **R152a** | $\text{CH}_3\text{CHF}_2$ | 98 | 386.41 | 4.52 | 0.279 | 0.1006 | 0.0561 | \textbf{−44.2\%} |
| **R23** | $\text{CHF}_3$ | 97 | 299.29 | 4.83 | 0.264 | 0.0928 | 0.0434 | \textbf{−53.2\%} |
| **R134** | $\text{CHF}_2\text{CHF}_2$| 92 | 391.73 | 4.60 | 0.297 | 0.2086 | 0.1414 | \textbf{−32.2\%} |
| **R161** | $\text{CH}_3\text{CH}_2\text{F}$ | 84 | 375.25 | 5.01 | 0.218 | 0.0850 | 0.0681 | −19.9\% |
| **R143a** | $\text{CH}_3\text{CF}_3$ | 83 | 345.86 | 3.76 | 0.261 | 0.1156 | 0.0646 | \textbf{−44.1\%} |
| **R245fa** | $\text{CHF}_2\text{CH}_2\text{CF}_3$ | 55 | 427.16 | 3.65 | 0.378 | 0.0764 | 0.0852 | +11.5\% |
| **R236fa** | $\text{CF}_3\text{CH}_2\text{CF}_3$ | 55 | 398.07 | 3.20 | 0.376 | 0.1173 | 0.0304 | \textbf{−74.1\%} |
| **R227ea** | $\text{CF}_3\text{CHFCF}_3$| 55 | 374.90 | 2.93 | 0.357 | 0.0602 | 0.0267 | \textbf{−55.6\%} |
| **R41** | $\text{CH}_3\text{F}$ | 39 | 317.28 | 5.90 | 0.200 | 0.0403 | 0.0528 | +31.0\% |
| \textbf{Macro Average} | — | \textbf{1403} | — | — | — | \textbf{0.1049} | \textbf{0.0666} | \textbf{−36.5\%} |

### S4.2 Risk–Coverage Abatement Trajectories Across All Five Axes
Selective prediction risk (MAE) as a function of sample coverage when retaining points with lowest seed-ensemble disagreement $\sigma_i$:

| Axis | 100% Coverage (Full) | 90% Coverage | 80% Coverage | 70% Coverage | 60% Coverage | 50% Coverage (Pruned) | Total Risk Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1 LORO** | 0.0720 | 0.0663 | 0.0634 | 0.0618 | 0.0579 | \textbf{0.0524} | \textbf{−27.2\%} |
| **B1 Anion** | 0.0292 | 0.0255 | 0.0219 | 0.0185 | 0.0163 | \textbf{0.0139} | \textbf{−52.5\%} |
| **B2 Anion** | 0.0472 | 0.0416 | 0.0362 | 0.0334 | 0.0282 | \textbf{0.0230} | \textbf{−51.2\%} |
| **L2 Composition** | 0.0271 | 0.0256 | 0.0243 | 0.0246 | 0.0230 | \textbf{0.0201} | \textbf{−25.8\%} |
| **HFO/HCFO** | 0.0302 | 0.0230 | 0.0194 | 0.0177 | 0.0168 | \textbf{0.0161} | \textbf{−46.7\%} |
