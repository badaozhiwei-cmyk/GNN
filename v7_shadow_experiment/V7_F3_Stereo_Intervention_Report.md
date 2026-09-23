# F3: Controlled Stereochemical Intervention & Unseen-State Sensitivity Report

**Status**: Formally Evaluated & Audited  
**Date**: 2026-09-23 16:45:00  
**Evaluation Universe**: Table S4 VLE HFOs ($N=1106$ evaluation points; $N=23$ geometric isomer points for R1336mzz)  
**Models Evaluated**:
- **V6 Reference**: `results_hfc_all_stereo/HFC_all_stereo_M0/best_seed_42.pth` (Standalone 4D-edge baseline)
- **V7-A**: 5 individual seeds (42..46) + 5-member A-Ensemble
- **V7-B**: 5 individual seeds (42..46) + 5-member B-Ensemble

---

## 1. Executive Epistemological Summary & Grounded Boundaries

> [!IMPORTANT]
> **Defensible Physical Boundary**:
> As audited in Step 24D-0, the saturated HFC training universe contains **strictly 0/143,610 covalent edges with stereo parity states 1..5**. Pure task loss gradients on `edge_embedding4` rows 1..5 were identically zero.
> Therefore, any observed output shift under stereochemical intervention ($\Delta y_s = |\hat{y}(s) - \hat{y}(0)|$) measures **architectural responsiveness to previously unseen topological edge tokens**, NOT that the neural network 'learned' quantum stereochemical principles without supervision.

### Primary Experimental Findings

1. **Unseen-State Input Sensitivity**:
   - **V6 Reference (Seed 42)**: Exhibits near-zero output perturbation across the entire evaluation universe:
     - Mean Perturbation (True Stereo): **$8.68 \times 10^{-9}$**
     - Max Perturbation: **$8.94 \times 10^{-8}$**
     - Confirms that V6's prediction head was functionally decoupled from graph edge perturbations (consistent with scalar bypass).
   - **V7-A Ensemble**:
     - Mean Perturbation (True Stereo, Full HFO): **$1.62 \times 10^{-6}$**
     - R1336mzz Geometric Isomers: Mean Perturbation = **$3.30 \times 10^{-6}$** (Max: **$5.51 \times 10^{-6}$**)
     - *Sensitivity Shift*: $\sim 100\times$ to $300\times$ higher than V6 Reference, demonstrating active propagation through SolvGNN interaction graph.
   - **V7-B Ensemble**:
     - Mean Perturbation (True Stereo, Full HFO): **$9.09 \times 10^{-7}$**
     - R1336mzz Geometric Isomers: Mean Perturbation = **$1.92 \times 10^{-6}$** (Max: **$3.01 \times 10^{-6}$**)
     - *Sensitivity Shift*: $\sim 100\times$ to $200\times$ higher than V6 Reference.

2. **Null-Control Specificity Test ($s \in \{1, 2, 3, 4, 5\}$)**:
   - We compared true E/Z perturbation against 3 unseen null-control tokens: $s=1$ (Any), $s=4$ (Cis), $s=5$ (Trans).
   - On R1336mzz geometric isomers:
     - V7-A Ensemble Specificity Ratio: **0.961**
     - V7-B Ensemble Specificity Ratio: **0.941**
     - Seed-by-seed ranges tightly within $[0.62, 1.11]$, centered at $\approx 1.0$.
   - *Epistemological Implication*: Because the specificity ratio is essentially unity ($0.95 \approx 1.0$), **the models exhibit general architectural input sensitivity to out-of-distribution embedding rows rather than an isolated physical E/Z comprehension**.

3. **Diastereomer Disparity Gap ($|\text{MAE}(Z) - \text{MAE}(E)|$)**:
   - Two-sample independent stratified bootstrap (1000 iterations):
     - **V6 Reference**: Control Gap = `0.0607` -> Stereo Gap = `0.0607` ($\Delta\text{Gap} = -1.49 \times 10^{-8}$, 95% CI: `[-5.96e-08, +5.96e-08]`)
     - **V7-A Ensemble**: Control Gap = `0.0111` -> Stereo Gap = `0.0111` ($\Delta\text{Gap} = +3.31 \times 10^{-6}$, 95% CI: `[-4.71e-06, +5.25e-06]`)
     - **V7-B Ensemble**: Control Gap = `0.0497` -> Stereo Gap = `0.0497` ($\Delta\text{Gap} = -3.87 \times 10^{-6}$, 95% CI: `[-4.11e-06, +4.04e-06]`)
   - *Epistemological Implication*: The stereochemical intervention does not narrow the E-Z gap without explicit stereochemical supervision during training.

---

## 2. Seed-by-Seed Dissection & Cross-Seed Stability

### V7-A Cross-Seed Perturbations on R1336mzz(Combined):
| Model | Seed | Control MAE | Stereo MAE | $\Delta$MAE (95% Paired CI) | Mean Pert (Stereo) | Mean Pert (Null) | Specificity Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| V7-A | 42 | 0.2570 | 0.2570 | $+1.07 \times 10^{-6}$ (`[+3.26e-07, +1.96e-06]`) | $1.94 \times 10^{-6}$ | $2.12 \times 10^{-6}$ | 0.916 |
| V7-A | 43 | 0.2589 | 0.2589 | $-9.54 \times 10^{-7}$ (`[-2.33e-06, +4.47e-07]`) | $3.12 \times 10^{-6}$ | $4.63 \times 10^{-6}$ | 0.675 |
| V7-A | 44 | 0.2699 | 0.2700 | $+8.88 \times 10^{-6}$ (`[+4.44e-06, +1.37e-05]`) | $1.12 \times 10^{-5}$ | $1.01 \times 10^{-5}$ | 1.106 |
| V7-A | 45 | 0.2747 | 0.2747 | $+5.96 \times 10^{-8}$ (`[-3.31e-07, +4.32e-07]`) | $8.16 \times 10^{-7}$ | $8.25 \times 10^{-7}$ | 0.989 |
| V7-A | 46 | 0.2614 | 0.2614 | $+8.34 \times 10^{-7}$ (`[-1.10e-06, +3.17e-06]`) | $5.09 \times 10^{-6}$ | $5.15 \times 10^{-6}$ | 0.989 |
| **V7-A** | **Ensemble** | **0.2644** | **0.2644** | **$+1.97 \times 10^{-6}$ (`[+8.00e-07, +3.19e-06]`)** | **$3.30 \times 10^{-6}$** | **$3.44 \times 10^{-6}$** | **0.961** |

### V7-B Cross-Seed Perturbations on R1336mzz(Combined):
| Model | Seed | Control MAE | Stereo MAE | $\Delta$MAE (95% Paired CI) | Mean Pert (Stereo) | Mean Pert (Null) | Specificity Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| V7-B | 42 | 0.2520 | 0.2520 | $+2.38 \times 10^{-7}$ (`[-2.73e-07, +7.68e-07]`) | $1.22 \times 10^{-6}$ | $1.38 \times 10^{-6}$ | 0.881 |
| V7-B | 43 | 0.2441 | 0.2441 | $+8.79 \times 10^{-7}$ (`[+1.88e-07, +1.46e-06]`) | $1.47 \times 10^{-6}$ | $2.38 \times 10^{-6}$ | 0.618 |
| V7-B | 44 | 0.2654 | 0.2654 | $+5.96 \times 10^{-8}$ (`[-4.54e-08, +1.28e-07]`) | $2.00 \times 10^{-7}$ | $1.95 \times 10^{-7}$ | 1.028 |
| V7-B | 45 | 0.2553 | 0.2553 | $+8.34 \times 10^{-7}$ (`[-1.30e-06, +2.69e-06]`) | $4.65 \times 10^{-6}$ | $4.20 \times 10^{-6}$ | 1.106 |
| V7-B | 46 | 0.2554 | 0.2554 | $+4.47 \times 10^{-7}$ (`[-4.62e-07, +1.27e-06]`) | $2.06 \times 10^{-6}$ | $2.05 \times 10^{-6}$ | 1.007 |
| **V7-B** | **Ensemble** | **0.2544** | **0.2544** | **$+5.36 \times 10^{-7}$ (`[-3.74e-07, +1.25e-06]`)** | **$1.92 \times 10^{-6}$** | **$2.04 \times 10^{-6}$** | **0.941** |

---

## 3. Macro Performance Across Full HFO Universe ($N=1106$)

| Model Family | Configuration | Control MAE | Stereo MAE | $\Delta$MAE (95% Paired CI) | Mean Output Perturbation | Max Output Perturbation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **V6 Reference** | Seed 42 | 0.0595 | 0.0595 | $+0.00 \times 10^0$ (`[-2.89e-09, -5.86e-10]`) | $8.68 \times 10^{-9}$ | $8.94 \times 10^{-8}$ |
| **V7-A** | 5-Seed Ensemble | 0.0715 | 0.0715 | $-6.71 \times 10^{-7}$ (`[-8.60e-07, -5.18e-07]`) | $1.62 \times 10^{-6}$ | $1.51 \times 10^{-5}$ |
| **V7-B** | 5-Seed Ensemble | 0.1014 | 0.1014 | $-5.89 \times 10^{-7}$ (`[-6.98e-07, -4.89e-07]`) | $9.09 \times 10^{-7}$ | $8.43 \times 10^{-6}$ |

---

## 4. Scientific Conclusion & Peer-Review Framing

1. **Closure of the Mechanistic Chain**:
   - V6's near-zero perturbation ($\sim 10^{-9}$) reflected complete readout head insensitivity to graph edge perturbations.
   - V7-A and V7-B demonstrate that structural interaction graphs restore **topological edge feature transmission to the readout head**, amplifying output perturbation by $\sim 100\times$ to $300\times$ ($10^{-6} \sim 10^{-5}$).
2. **The Supervision Deficit Boundary**:
   - Because the saturated training universe contained zero stereo supervision, the network's perturbation under E/Z tokens perfectly tracks its perturbation under null-control tokens ($s=1, 4, 5$), yielding a specificity ratio of $\approx 0.95$.
   - This provides definitive empirical evidence that **architectural capacity alone does not substitute for training exposure**: resolving geometric isomer degeneracy requires coupling stereochemical architectures with explicit stereochemical pretraining or supervision.
