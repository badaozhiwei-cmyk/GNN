# Claim-to-Evidence-to-Reference Mapping Framework (权威论断-证据-文献对照图谱)

> **Purpose**: This authoritative document establishes an unbroken, bidirectional correspondence between every major scientific claim in the manuscript (Sections 1–4) and its primary literature citations, evidence types, and defensive rebuttal roles. 
> 
> All claims are calibrated to adhere strictly to non-overclaiming scientific principles:
> 1. *Distance Metrics*: Framed as "no single scalar distance provides a sufficient description of prediction difficulty across multi-dimensional chemical distribution shifts" (NOT "distance metrics are universally invalid").
> 2. *Thermodynamic Regularization*: Framed as "selective thermodynamic grounding" (NOT an all-encompassing sole mechanism).
> 3. *Uncertainty Quantification*: Formulated as "seed-ensemble disagreement as an actionable epistemic-uncertainty proxy" (NOT Bayesian variance or calibrated probability).
> 4. *Stereochemical Featurization*: Framed as "architectural expressivity is bounded by training distribution support" (NOT that edge stereochemistry itself is flawed).

---

## Master Claim-to-Reference Table

| Claim ID | Scientific Claim Statement | Primary Citations | Evidence Type | Manuscript Section | Reviewer Defense / Rebuttal Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CLM-01** | Standard random train/test splits overestimate generalization in multi-component chemical systems due to structural redundancy across conditions. | `Choudhary2024_OODBenchmark`<br>`Li2023_NatCommun`<br>`Wu2018_MoleculeNet` | Benchmark Study / Empirical Data | Introduction §1.2<br>Methods §2.1 & §2.4 | Defends against: *"Why not simply report cross-validation on random 80/20 splits like earlier literature?"* Preemptively demonstrates that random splitting permits data leakage across temperatures and counter-ions. |
| **CLM-02** | Deep GNNs can achieve high interpolation accuracy on refrigerant–IL mixtures under ambient operating conditions, but their real-world deployment requires systematic OOD evaluation. | `Chu2024_AEGNN`<br>`Qin2024_CAILD` | Direct Baseline / State-of-the-Art | Introduction §1.1 & §1.2<br>Discussion §4.1 | Establishes respectful dialogue with direct neighbors: directly acknowledges Chu et al. (AEGNN, 2024) as a strong precedent, while highlighting that our work addresses the previously unexamined OOD generalization ladder. |
| **CLM-03** | Group-contribution models (UNIFAC) and quantum screening models (COSMO-RS) face parameter scarcity for novel fluorinated groups and prohibitive computational costs for large screening libraries. | `Fredenslund1975_UNIFAC`<br>`Lei2014_UNIFAC_IL`<br>`Klamt2000_COSMORS` | Thermodynamic Theory & Review | Introduction §1.1 | Defends against: *"Why not use traditional chemical engineering thermodynamic models instead of GNNs?"* Documents that UNIFAC lacks parameters for specialized fluorinated structures, while COSMO-RS requires expensive DFT. |
| **CLM-04** | Experimental VLE solubility data can exhibit laboratory-dependent measurement noise, requiring robust training objectives and thermodynamic consistency checks. | `Sarmiento2021_Consistency` | Thermodynamic Validation | Methods §2.1 & §2.5 | Defends against: *"Why use Huber loss instead of standard MSE?"* Shows that experimental gas solubility datasets contain heavy-tailed outliers, making Huber loss ($\delta=0.05$) mathematically robust. |
| **CLM-05** | Disjoint tri-graph representations preserve physical solvation reality by avoiding non-physical covalent edges across unbonded ionic and gaseous species. | `Chu2024_AEGNN`<br>`Wu2018_MoleculeNet` | Methodological Design | Methods §2.2 | Defends against: *"Why not concatenate the entire mixture into a single molecular graph?"* Explains that dissociated ions and gas molecules lack covalent bonds; tri-graphs reflect true ternary solution topology. |
| **CLM-06** | Under the Theorem of Corresponding States, macroscopic phase behavior collapses onto universal dimensionless curves parameterized by reduced variables ($T_r, P_r$) and the acentric factor ($\omega$). | `Pitzer1955_Acentric`<br>`Prausnitz1998_FluidPhase`<br>`Kocis2020_PINN_Thermo` | Fundamental Thermodynamic Law | Methods §2.3<br>Discussion §4.2 | Defends against: *"Are $(T_r, P_r, \omega)$ arbitrary hand-picked features?"* Demonstrates they are the canonical universal coordinates of corresponding-states thermodynamics dating to van der Waals and Pitzer. |
| **CLM-07** | Thermodynamic feature regularization does not act as a universal panacea; its benefit is strictly selective, regularizing solute extrapolation while having negligible effect under dense matrix redundancy. | `Karpatne2017_PGML`<br>`Karniadakis2021_PINN_Review` | Physics-ML Theory / Empirical Finding | Introduction §1.3<br>Results §3.1<br>Discussion §4.2 | Defends against: *"Why did $M_{\rm reduced}$ not improve performance on the B1 fluorosulfonate benchmark?"* Formulates the selective grounding principle: physical priors aid where extrapolation bottlenecks exist, but yield marginal utility where topological redundancy suffices. |
| **CLM-08** | In multicomponent gas–IL absorption, no single scalar distance metric (topological $D_{\rm FP}$ or physicochemical $D_{\rm phys}$) provides a sufficient description of prediction difficulty across all distribution shifts. | `Tropsha2003_QsarValidation`<br>`Netzeva2005_ApplicabilityDomain` | QSAR Distance Theory & Empirical Disproof | Introduction §1.2<br>Results §3.3<br>Discussion §4.1 | Defends against: *"Why did the model fail on R134 while succeeding on R134a despite near-zero chemical distance?"* Explains that isotropic scalar distance cannot capture discrete quantum dipole cancellations ($\mu = 0.003\text{ D}$ vs $2.719\text{ D}$). |
| **CLM-09** | Deep ensemble disagreement across independently initialized seeds serves as an actionable epistemic-uncertainty proxy for automated selective prediction and risk mitigation. | `Lakshminarayanan2017_DeepEnsembles`<br>`Geifman2017_SelectiveNet` | Uncertainty Quantification Theory | Methods §2.6<br>Results §3.4<br>Discussion §4.3 | Defends against: *"Can 5-seed disagreement be claimed as true Bayesian posterior variance?"* Clarifies that $\sigma_i$ is treated strictly as an epistemic disagreement proxy and ranking signal for selective prediction, cutting risk by $>50\%$. |
| **CLM-10** | Standard 2D molecular graph representations do not uniquely encode geometric stereochemistry ($\mathcal{G}_E \equiv \mathcal{G}_Z$); resolving this requires distinguishing mathematical expressivity from data distribution support. | `Morehead2024_GCPNet`<br>`Adams2024_GeomComplete`<br>`Schwaller2024_3DGeom` | Geometric Deep Learning Theory | Methods §2.7<br>Results §3.3<br>Discussion §4.3 | Defends against: *"Why not simply add 3D coordinates or stereochemical edge tags to fix the R1336mzz(E/Z) failure?"* Demonstrates that without training data support, adding unconstrained stereo embeddings yields zero task gradient ($\nabla_{\mathbf{E}} \mathcal{L} \equiv 0$) and fails to generalize. |

---

## Detailed Section-by-Section Integration Guide

### Section 1: Introduction

- **§1.1 Industrial Context & Thermodynamic Modeling Challenges**:
  - *Context*: Kigali Amendment, low-GWP HFOs, IL green separation.
  - *Citations*: `Qin2024_CAILD` (refrigerant azeotropic separation via ILs); `Fredenslund1975_UNIFAC`, `Lei2014_UNIFAC_IL` (UNIFAC limitations for fluorinated IL systems); `Klamt2000_COSMORS` (COSMO-RS computational cost).
  - *Textual Anchor*: *"While ionic liquids present immense synthetic flexibility for refrigerant absorption cycles, classical group-contribution methods struggle with parameter scarcity for newly synthesized fluorinated groups [Lei2014_UNIFAC_IL], and quantum-chemical COSMO-RS calculations remain computationally intensive for large-scale solvent screening [Klamt2000_COSMORS, Qin2024_CAILD]."*

- **§1.2 Molecular AI, Random Splitting, and Chemical Generalization**:
  - *Context*: Transition from random interpolation to genuine chemical OOD.
  - *Citations*: `Chu2024_AEGNN` (direct neighbor baseline); `Choudhary2024_OODBenchmark` (materials OOD benchmark); `Li2023_NatCommun` (temperature data leakage in ILs); `Wu2018_MoleculeNet` (scaffold splitting).
  - *Textual Anchor*: *"Recent pioneering work has demonstrated the viability of molecular graph neural networks for HFC/HFO solubility prediction in ionic liquids under random train/test splits [Chu2024_AEGNN]. However, as recently underscored in computational materials and ionic polymer benchmarks [Choudhary2024_OODBenchmark, Li2023_NatCommun], random splitting masks severe failure under real-world chemical distribution shifts."*

- **§1.3 Distance Metric Limits & Physics Grounding**:
  - *Context*: Non-monotonicity of scalar chemical distance and corresponding states.
  - *Citations*: `Tropsha2003_QsarValidation`, `Netzeva2005_ApplicabilityDomain` (applicability domain theory); `Pitzer1955_Acentric`, `Prausnitz1998_FluidPhase` (corresponding states); `Karpatne2017_PGML` (physics-guided ML).

---

### Section 2: Methods

- **§2.1 Data Universes & Quality Control**:
  - *Citations*: `Sarmiento2021_Consistency` (thermodynamic consistency of gas–IL data); `Li2023_NatCommun` (isolation of repeated multi-temperature measurements).
- **§2.2 Disjoint Tri-Graph Representation & Architecture**:
  - *Citations*: `Chu2024_AEGNN` (tri-component input formulation); `Schwaller2024_3DGeom` (2D topology vs 3D conformer ambiguity).
- **§2.3 Thermodynamic Coordinate Grounding**:
  - *Citations*: `Pitzer1955_Acentric` (acentric factor definition); `Prausnitz1998_FluidPhase` (corresponding states theorem); `Kocis2020_PINN_Thermo` (dimensionless coordinates in ML).
- **§2.4 Chemical Generalization Ladder**:
  - *Citations*: `Choudhary2024_OODBenchmark` (multi-category OOD benchmark design); `Wu2018_MoleculeNet` (out-of-distribution partitioning).
- **§2.5 Optimization & Anti-Leakage Scaler Protocols**:
  - *Citations*: `Sarmiento2021_Consistency` (outlier resilience in phase equilibria).
- **§2.6 Epistemic Uncertainty & Selective Prediction**:
  - *Citations*: `Lakshminarayanan2017_DeepEnsembles` (deep ensemble formulation); `Geifman2017_SelectiveNet` (risk-coverage selective prediction framework).
- **§2.7 Representation-Support Boundary & Stereochemical Intervention**:
  - *Citations*: `Morehead2024_GCPNet`, `Adams2024_GeomComplete` (chirality and stereochemical representation limits in graph neural networks).

---

### Section 3 & 4: Results & Discussion

- **§4.1 Representation Bottlenecks vs. Scalar Distance Metrics**:
  - *Citations*: `Tropsha2003_QsarValidation`, `Netzeva2005_ApplicabilityDomain`.
  - *Key Synthesis*: Contrasts classical QSAR distance assumptions with empirical evidence from R134/R134a and R245fa/R236fa.
- **§4.2 Selective Thermodynamic Grounding**:
  - *Citations*: `Pitzer1955_Acentric`, `Karpatne2017_PGML`, `Karniadakis2021_PINN_Review`.
  - *Key Synthesis*: Explains why corresponding-states variables anchor solute extrapolation while exhibiting marginal utility under matrix redundancy.
- **§4.3 Stereochemical Degeneracy & Epistemic Guardrails**:
  - *Citations*: `Morehead2024_GCPNet`, `Lakshminarayanan2017_DeepEnsembles`, `Geifman2017_SelectiveNet`.
  - *Key Synthesis*: Pairs the hard representation ceiling of R1336mzz(E/Z) with actionable deep ensemble uncertainty guardrails.
