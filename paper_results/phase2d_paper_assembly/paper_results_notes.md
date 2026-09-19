# Phase II-D paper assembly notes

## Core cross-axis result
- M1: ensemble macro MAE 0.0940 -> 0.0613; ensemble macro R² -0.2399 -> 0.5005.
- B1: ensemble MAE 0.0287 -> 0.0292; seed-mean R² 0.9265 -> 0.9285.
- B2: ensemble MAE 0.0518 -> 0.0476; seed-mean R² 0.7931 -> 0.8333.
- L2: ensemble MAE 0.0298 -> 0.0271 (-9.14%); ensemble R² 0.9108 -> 0.9218; mean sigma 0.01864 -> 0.01425.
- HFO/HCFO: ensemble MAE 0.03742 -> 0.03002 (-19.77%); ensemble R² 0.62269 -> 0.71070; mean sigma 0.03450 -> 0.01440; Spearman sigma-error rho 0.34068 -> 0.49259.

## L2 paired evidence
Mean paired ΔMAE (M0 - Mred) = 0.002724
Bootstrap 95% CI = [0.001515, 0.003915]
Pointwise win rate = 59.9%
One-sided Wilcoxon p = 5.18e-06

## HFO/HCFO interpretation guardrails
- Treat the 1106-point task as Cross-Family Unsaturated Refrigerant Stress Test (HFO/HCFO), not independent external validation.
- Strict HFO-only sensitivity: N=1016; M0 MAE 0.03810, Mred MAE 0.03098; M0 R² 0.62241, Mred R² 0.70845.
- R1233zd(E) MAE=0.0193 is an absolute mole-fraction error, not a <2% relative error.
- R1336mzz(E/Z) share the same graph tensor under the current graph featurization because bond stereochemistry is not encoded; Mred may distinguish them indirectly through species-dependent Tc/Pc/omega.

## UQ language
Use “Seed-Ensemble Epistemic Uncertainty Proxy” or “ensemble disagreement”.
Do not call Spearman rho calibration. Use “error-uncertainty rank association”.
Do not call reliability curves probabilistic calibration.

## Causal/physical language
For R1336mzz failure, cavity/steric penalty is a possible explanation/hypothesis, not a demonstrated mechanism.
