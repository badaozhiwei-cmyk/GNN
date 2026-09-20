# 2. Computational and Theoretical Methodology

## 2.1 Dataset Universe, Quality Control, and Lineage Tracking

To systematically evaluate the out-of-distribution (OOD) generalization boundaries of deep graph representations in multicomponent gas–liquid equilibria, we assembled a comprehensive database of vapor–liquid equilibrium (VLE) solubility points for fluorinated refrigerants across diverse ionic liquid (IL) matrices. The raw experimental database comprises $4,444$ high-precision solubility measurements across $1,029$ unique cation–anion–refrigerant chemical triplets compiled from NIST and peer-reviewed experimental literature [Sarmiento et al., *J. Mol. Liq.*, 2021].

#### Benchmark Data Universes and Physical Isolation
To enforce rigorous provenance and eliminate data contamination, the data universe is partitioned into two physically isolated regimes:

1. **Saturated Hydrofluorocarbon (HFC) Benchmark Universe ($N = 2,739$, Table S3)**:
   This dataset comprises $2,739$ experimental data points spanning $12$ structurally diverse pure saturated HFC solutes: $\text{R23}$ ($\text{CHF}_3$), $\text{R32}$ ($\text{CH}_2\text{F}_2$), $\text{R41}$ ($\text{CH}_3\text{F}$), $\text{R125}$ ($\text{CHF}_2\text{CF}_3$), $\text{R134}$ ($\text{CHF}_2\text{CHF}_2$), $\text{R134a}$ ($\text{CF}_3\text{CH}_2\text{F}$), $\text{R143a}$ ($\text{CH}_3\text{CF}_3$), $\text{R152a}$ ($\text{CH}_3\text{CHF}_2$), $\text{R161}$ ($\text{CH}_3\text{CH}_2\text{F}$), $\text{R227ea}$ ($\text{CF}_3\text{CHFCF}_3$), $\text{R236fa}$ ($\text{CF}_3\text{CH}_2\text{CF}_3$), and $\text{R245fa}$ ($\text{CF}_3\text{CH}_2\text{CHF}_2$). The matrices span $67$ distinct ionic liquid cations and $41$ anions across broad temperature ($273.15\text{--}368.15\text{ K}$) and pressure ($0.0095\text{--}4.999\text{ MPa}$) regimes, corresponding to liquid-phase refrigerant mole fractions $x_1 \in [0.0010, 0.8900]$.
   
   To ensure immutable reproducibility across scientific pipelines, the row order and floating-point contents of this benchmark universe are locked via cryptographic SHA-256 digests:
   $$\text{Hash}_{\rm row\_order} = \texttt{33719d8f5e1a0b66b18165dab5f613282c4935926cb9adecd277cbd0ad92bbc5}$$
   $$\text{Hash}_{\rm data.npy} = \texttt{a408c2832352d69799d1c0a8b7151dd7535fe2edf34d171992b9385b1bd3e6bb}$$

2. **M1 Benchmark Evaluation Subset ($N = 1,403$)**:
   Within the saturated HFC domain, an authoritative multi-condition benchmark subset of $N = 1,403$ points across the identical $12$ HFC species is maintained for standardized Leave-One-Refrigerant-Out (LORO) comparative benchmarking, exactly matching all historical baseline checkpoints and uncertainty quantification (UQ) distributions.

3. **Unsaturated Hydrofluoroolefin (HFO/HCFO) Zero-Shot Universe ($N = 1,106$, Table S4)**:
   This evaluation domain comprises $1,106$ experimental points encompassing five commercial and next-generation low-global-warming-potential (GWP) unsaturated haloolefins: $\text{R1234yf}$ ($N=685$), $\text{R1234ze(E)}$ ($N=308$), $\text{R1233zd(E)}$ ($N=90$), $\text{R1336mzz(E)}$ ($N=11$), and $\text{R1336mzz(Z)}$ ($N=12$). This universe serves strictly as an external, zero-shot downstream test bed; **it is physically isolated during all model training, hyperparameter selection, and feature scaler fitting** [Li et al., *Nat. Commun.*, 2023].

4. **Data Accounting and Curation Scope**:
   Across the complete experimental corpus of $4,444$ collected solubility records, benchmark analyses rigorously focus on the $3,845$ validated points spanning the $2,739$ saturated HFC points and $1,106$ unsaturated haloolefin points. The remaining $599$ auxiliary records (Table S5) comprise legacy chlorofluorocarbons (CFCs), hydrochlorofluorocarbons (HCFCs), and non-target refrigerants outside the defined chemical evaluation ladder.

---

## 2.2 Tri-Graph Molecular Representation and GNN Architecture

### Disjoint Tri-Graph Representation
Unlike single-molecule property prediction pipelines, gas solubility in ionic liquid solvents involves a ternary mixture consisting of an organic cation, an inorganic or organic counter-anion, and a gaseous solute molecule. Because the liquid phase comprises discrete dissociated ions in dynamic solvation equilibrium with dissolved gas, concatenating molecular structures into a single artificial graph would introduce non-physical covalent edges between non-bonded species [Chu et al., *Green Chem. Eng.*, 2024; Wu et al., *Chem. Sci.*, 2018; Schwaller et al., *J. Cheminform.*, 2024].

We employ a **disjoint tri-graph representation** $\mathcal{T} = (\mathcal{G}_{\rm cat}, \mathcal{G}_{\rm ani}, \mathcal{G}_{\rm sol})$. Each molecular entity $m \in \{\text{cat}, \text{ani}, \text{sol}\}$ is independently translated from Canonical SMILES into an attributed molecular graph $\mathcal{G}_m = (\mathcal{V}_m, \mathcal{E}_m)$ using RDKit:
- **Node Feature Vector $\mathbf{v}_i \in \mathbb{Z}^{7}$**: Each atom $i \in \mathcal{V}_m$ is parameterized by $7$ categorical indices mapped into a shared embedding space ($d_{\rm emb} = 300$): (1) atomic number ($119$ bins), (2) hybridization state ($\text{sp}, \text{sp}^2, \text{sp}^3, \text{sp}^3\text{d}, \text{sp}^3\text{d}^2$, $8$ bins), (3) aromaticity indicator ($2$ bins), (4) degree of connectivity ($7$ bins), (5) formal charge ($3$ bins), (6) Pauling electronegativity bucket ($8$ bins), and (7) covalent radius bucket ($8$ bins). In addition, an entity identity embedding ($3$ classes: cation, anion, solute) is added to atom representations.
- **Edge Feature Vector $\mathbf{e}_{ij} \in \mathbb{Z}^{3}$**: Each chemical bond $(i, j) \in \mathcal{E}_m$ is parameterized by $3$ discrete categorical features mapped to $d_{\rm emb} = 300$: (1) bond type (global connection, single, double, triple, aromatic; $5$ categories), (2) ring membership flag ($2$ categories), and (3) aromaticity flag ($2$ categories). (In the stereochemical intervention protocol in Section 2.7, this is systematically expanded to $\mathbb{Z}^4$ by adding a $6$-category directional stereochemistry feature).
- **Virtual Global Token**: The three subgraphs are assembled into a combined tri-graph representation, to which a single virtual Global Token node is appended and bidirectionally connected to all constituent atoms across the ternary mixture. The Global Token bypasses atom-level categorical embeddings and is initialized with an independent learnable parameter vector $\mathbf{h}_{\rm global} \in \mathbb{R}^{300}$.

### Neural Network Architecture
The computational architecture consists of two cooperative streams: a relational Graph Attention Network v2 (GATv2) encoder with edge feature conditioning for topological representation learning, and a Multi-Layer Perceptron (MLP) conditioning network for macroscopic thermodynamic state variables.

```
       [ Cation Graph G_cat ]   ---\
       [ Anion Graph G_ani   ]   ---+---> [ 3-Layer Relational GATv2 ] ---> [ Global Token Readout ] ---\
       [ Solute Graph G_sol  ]   ---/     (300 -> 512 -> 1024 -> 512)       (h_topo: 512-dim)           |
                                                                                                        +---> [ Regression MLP ] ---> x_1
       [ Operating Conditions &  ]                                                                      |     (512+d_cond -> 1024 -> 512 -> 1)
       [ Physical Descriptors z  ] ---> [ Thermodynamic Conditioning (d_cond) ] -----------------------/
```

1. **Topological Message Passing**:
   Message passing is performed across $L = 3$ relational GATv2Conv layers with hidden channel dimensions $300 \to 512 \to 1024 \to 512$. Each layer applies $K = 4$ multi-head attention mechanisms with edge-attribute conditioning ($\text{edge\_dim} = 300$):
   $$\alpha_{ij}^{(k)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}_k^T \left[\mathbf{W}_k \mathbf{h}_i^{(l-1)} \,\|\, \mathbf{W}_k \mathbf{h}_j^{(l-1)} \,\|\, \mathbf{W}_e \mathbf{e}_{ij}\right]\right)\right)}{\sum_{u \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}_k^T \left[\mathbf{W}_k \mathbf{h}_i^{(l-1)} \,\|\, \mathbf{W}_k \mathbf{h}_u^{(l-1)} \,\|\, \mathbf{W}_e \mathbf{e}_{iu}\right]\right)\right)}$$
   where $\mathbf{W}_k$ projects node representations, $\mathbf{W}_e$ projects edge embeddings, and $\|$ denotes vector concatenation. Head outputs are averaged (`concat=False`), followed by ReLU activation and dropout ($p_{\rm drop} = 0.2$).

2. **Global Token Pooling**:
   Rather than performing unweighted spatial mean pooling across divergent ionic sizes, the ternary mixture topological state $\mathbf{h}_{\rm topo} \in \mathbb{R}^{512}$ is extracted directly from the updated virtual Global Token node representation at the output of the final GATv2 layer ($L=3$).

3. **Thermodynamic Fusion and Readout**:
   The extracted topological embedding $\mathbf{h}_{\rm topo} \in \mathbb{R}^{512}$ is concatenated with the standardized thermodynamic conditioning vector $\mathbf{z} \in \mathbb{R}^{d_{\rm cond}}$ (where $d_{\rm cond} = 9$ for $M_0$ and $12$ for $M_{\rm reduced}$). The combined representation is fed into a 3-layer regression MLP:
   $$\text{MLP}: \mathbb{R}^{512 + d_{\rm cond}} \xrightarrow{\text{Linear}(1024)} \text{Norm} \xrightarrow{\text{ReLU}} \text{Dropout}(0.4) \xrightarrow{\text{Linear}(512)} \text{Norm} \xrightarrow{\text{ReLU}} \text{Dropout}(0.3) \xrightarrow{\text{Linear}(1)} \hat{x}_1$$
   predicting the liquid-phase refrigerant equilibrium mole fraction $\hat{x}_1$.

---

## 2.3 Thermodynamic Coordinate Grounding and Feature Hierarchy

To isolate the inductive effect of physical inductive biases from topological graph learning, we formulate two distinct thermodynamic feature conditioning regimes:

### 1. Pure Topological Baseline ($M_0$, Unregularized)
In the baseline model $M_0$, the conditioning vector $\mathbf{z}_{M_0} \in \mathbb{R}^{9}$ contains only ambient experimental operating variables and basic molecular summary statistics:
$$\mathbf{z}_{M_0} = \left[T, P, q_{\rm ref}, \text{LogP}_{\rm ref}, \text{MW}_{\rm ani}, q_{\rm cat}, \text{TPSA}_{\rm cat}, \text{MW}_{\rm ref}, \text{MW}_{\rm cat}\right]$$
where $T$ is temperature in Kelvin, $P$ is pressure in MPa, $q$ is formal net charge, $\text{MW}$ is molecular weight, and $\text{TPSA}$ is topological polar surface area. Here, the network must learn phase-equilibrium scaling entirely from unscaled operating variables.

### 2. Reduced Thermodynamic Model ($M_{\rm reduced}$, Corresponding States Grounding)
The central methodological innovation of this study is regularizing the continuous state function via the **van der Waals Theorem of Corresponding States** [Pitzer, *J. Am. Chem. Soc.*, 1955; Prausnitz et al., *Molecular Thermodynamics of Fluid-Phase Equilibria*, 1998; Kocis et al., *Chem. Eng. Sci.*, 2020]. According to macroscopic thermodynamic principles, the volumetric and phase behavior of distinct fluids collapses onto universal dimensionless equations of state when scaled by their vapor–liquid critical coordinates ($T_c, P_c$) and the Pitzer acentric factor ($\omega$):
$$T_r = \frac{T}{T_c}, \quad P_r = \frac{P}{P_c}$$
$$\omega = -\log_{10}\left(\frac{P^{\rm sat}(T_r = 0.7)}{P_c}\right) - 1.0$$
The feature vector $\mathbf{z}_{M_{\rm red}} \in \mathbb{R}^{12}$ directly substitutes raw critical boundaries with normalized dimensionless coordinates:
$$\mathbf{z}_{M_{\rm red}} = \left[\mathbf{z}_{M_0} \setminus \{T, P\}, T_r, P_r, \omega\right]$$
These critical properties ($T_c, P_c, \omega$) are derived exclusively from high-precision NIST Chemistry WebBook experimental standards. By providing $(T_r, P_r, \omega)$, the network is relieved of the mathematically ill-posed burden of inferring critical scaling limits from ambient $T, P$ in the presence of chemical distribution shift.

---

## 2.4 Chemical Generalization Ladder and Evaluation Protocols

To rigorously benchmark generalizability without confounding multiple distribution shifts, we designed a five-tier **Chemical Generalization Ladder** probing isolated orthogonal axes of chemical and thermodynamic extrapolation [Choudhary et al., *npj Comput. Mater.*, 2024; Wu et al., *Chem. Sci.*, 2018]:

1. **Axis M1: Solute Identity Extrapolation (Leave-One-Refrigerant-Out, LORO)**:
   Probes the model's ability to extrapolate to an entirely unseen gas solute within the same chemical family. In each fold $k \in \{1, \dots, 12\}$, all data points containing the $k$-th pure HFC refrigerant are completely removed from the training set and reserved exclusively for testing ($N_{\rm test} = 1,403$ cumulative points; $N=2,739$ in the full orthogonal partition). Zero solute cross-contamination is enforced by design across all folds.
2. **Axis B1: High-Redundancy Anion Matrix Shift (Fam-2 Fluorosulfonates)**:
   Evaluates extrapolation across the ionic liquid matrix when the test anion belongs to the bulky fluorosulfonate family ($\text{OTf}^-$, $\text{TFES}^-$, $\text{HFPS}^-$, $\text{TPES}^-$, $\text{PFBS}^-$, $\text{TTES}^-$, $\text{FS}^-$). Test instances ($N=513$) are completely held out from training ($N=2,003$), probing structural interpolation across fluorinated alkyl side-chains.
3. **Axis B2: Low-Redundancy Inorganic Fluoride Shift (Fam-3 Inorganic Fluorides)**:
   Evaluates extrapolation to compact, highly symmetric inorganic anions ($\text{BF}_4^-$, $\text{PF}_6^-$) characterized by spherical coordination geometry and low polarizability. Test instances ($N=598$) are isolated from training ($N=1,927$), probing representation transfer across divergent charge-delocalization geometries.
4. **Axis L2: Compositional Recombination Shift (Unseen Cation–Anion Pairs)**:
   Evaluates true compositional learning in ionic mixtures. Four common ionic pairs are held out entirely from training: $[\text{emim}][\text{BF}_4]$ ($N=93$), $[\text{emim}][\text{OTf}]$ ($N=114$), $[\text{bmim}][\text{OTf}]$ ($N=50$), and $[\text{hmim}][\text{BF}_4]$ ($N=117$). Both the cations ($[\text{emim}]^+$, $[\text{bmim}]^+$, $[\text{hmim}]^+$) and the anions ($[\text{BF}_4]^-$, $[\text{OTf}]^-$) appear extensively in the training set paired with other counter-ions, but their specific pairwise combinations are never observed during training ($N_{\rm test} = 374, N_{\rm train} = 2,129$).
5. **Axis HFO/HCFO: Cross-Family Zero-Shot Transfer**:
   Evaluates transfer from saturated hydrofluorocarbons (training corpus) to unsaturated hydrofluoroolefins (HFOs) and hydrochlorofluoroolefins (HCFOs) containing carbon–carbon double bonds ($\text{C}=\text{C}$) and chlorine substituents ($N_{\rm test} = 1,106$).

---

## 2.5 Model Training, Optimization, and Anti-Leakage Guardrails

### Loss Function
Experimental VLE solubility measurements are subject to non-negligible experimental measurement uncertainties across different laboratories, apparatuses (isochoric vs. gravimetric), and purity levels [Sarmiento et al., *J. Mol. Liq.*, 2021]. To prevent regression models from overfitting to heavy-tailed measurement outliers, models were trained using the **Huber loss** with delta threshold $\delta = 0.05$:
$$\mathcal{L}_{\rm Huber}(y, \hat{y}) = \begin{cases} \frac{1}{2} (y - \hat{y})^2, & \text{for } |y - \hat{y}| \le \delta \\ \delta \cdot \left(|y - \hat{y}| - \frac{1}{2}\delta\right), & \text{otherwise} \end{cases}$$
The Huber loss behaves as an $\ell_2$ penalty for small residual errors while transitioning to a robust $\ell_1$ linear penalty for deviations exceeding $0.05$ mole fraction.

### Optimization Protocol
- **Optimizer**: Adam with initial learning rate $\eta_0 = 1.0 \times 10^{-3}$ and weight decay $\lambda = 1.0 \times 10^{-6}$.
- **Learning Rate Schedule**: Cosine Annealing decay without restarts over $100$ epochs down to $\eta_{\rm min} = 1.0 \times 10^{-5}$.
- **Batch Size and Regularization**: Mini-batch size of $32$; dropout of $0.2$ across all GNN message-passing and MLP layers.
- **Early Stopping**: Monitored on validation loss with patience of $15$ epochs, restoring model weights from the best historical checkpoint.
- **Multi-Seed Replication**: All benchmark axes were trained and evaluated across $5$ independent, pre-registered random seeds: $\mathcal{S} = \{42, 43, 44, 45, 46\}$.
- **Anti-Leakage Scaler Protocols**: All feature normalizations were executed under strict **Train-Only Fitting** ($\mathbf{z}_{\rm norm} = (\mathbf{z} - \boldsymbol{\mu}_{\rm train})/(\boldsymbol{\sigma}_{\rm train} + \epsilon)$), serialized to `scalers.pkl`, eliminating data leakage.

---

## 2.6 Epistemic Uncertainty Quantification and Selective Prediction Framework

In safety-critical chemical engineering processes and high-throughput solvent screening, automated systems must provide actionable epistemic uncertainty signals to flag low-confidence predictions.

### Deep Ensemble Disagreement Proxy
We employ deep ensembling across the $M = 5$ independently trained seeds as an epistemic uncertainty estimator [Lakshminarayanan et al., *NeurIPS*, 2017]. For a test query $\mathbf{x}_i$, the ensemble mean prediction $\bar{y}_i$ and epistemic disagreement $\sigma_i$ are:
$$\bar{y}_i = \frac{1}{M} \sum_{m=1}^M \hat{y}_i^{(m)}, \quad \sigma_i = \sqrt{\frac{1}{M-1} \sum_{m=1}^M \left(\hat{y}_i^{(m)} - \bar{y}_i\right)^2}$$
In accordance with strict statistical rigor, $\sigma_i$ is treated strictly as an **epistemic disagreement proxy**, not an empirical error probability.

### Rank Association and Selective Prediction
The validity of $\sigma_i$ as an actionable uncertainty estimator is evaluated via the Spearman rank correlation coefficient $\rho(\sigma_i, |y_i - \bar{y}_i|)$. Selective prediction profiles are constructed by ranking queries in ascending order of $\sigma_i$ and computing the retained risk $\mathcal{R}(\kappa)$ at coverage retention rates $\kappa \in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]$ following the risk-coverage framework of Geifman & El-Yaniv (*NeurIPS*, 2017).

---

## 2.7 Representation-Support Boundary and Stereochemical Diagnostic Intervention

### Mathematical Formulation of Stereochemical Degeneracy
A foundational limitation of standard 2D molecular graph featurization in chemical engineering is the omission of geometric isomerism. For $(Z)\text{-1,1,1,4,4,4-hexafluoro-2-butene}$ ($\text{R1336mzz(Z)}$) and $(E)\text{-1,1,1,4,4,4-hexafluoro-2-butene}$ ($\text{R1336mzz(E)}$):
$$\mathcal{G}_{\text{R1336mzz(E)}} \equiv \mathcal{G}_{\text{R1336mzz(Z)}} \implies \mathbf{h}_{\rm topo}^{(E)} = \mathbf{h}_{\rm topo}^{(Z)}$$
Consequently, topological graph encoders cannot distinguish between the stereoisomers [Morehead & Cheng, *Bioinformatics*, 2024; Adams et al., *Commun. Chem.*, 2024].

### Mechanistic Intervention vs. Distribution Support Boundary
To determine whether this bottleneck can be resolved via targeted inductive intervention, edge features were expanded to $\mathbb{R}^4$ with directional stereochemistry tags and a dedicated embedding layer $\mathbf{E}_{\rm stereo}$. 

However, because the training corpus consists exclusively of saturated HFC refrigerants and contains no chemically encoded non-zero stereochemical bond states ($N_{\text{stereo state 1-5}}^{\text{chemical bonds}} = 0$), the task gradient for non-zero stereochemical categories is identically zero during training:
$$\nabla_{\mathbf{E}[k]} \mathcal{L}_{\rm train} \equiv \mathbf{0} \quad \forall k \ge 1$$
Consequently, while stereochemical edge featurization mathematically resolves graph equivalence ($\mathcal{G}_E \neq \mathcal{G}_Z$), **the model cannot learn meaningful stereochemical inductive biases without training-set support**. This establishes a clear theoretical and practical conclusion: *architectural expressivity is bounded by data distribution support; introducing unconstrained geometric embeddings without corresponding training examples creates degenerate inductive transfer.*
