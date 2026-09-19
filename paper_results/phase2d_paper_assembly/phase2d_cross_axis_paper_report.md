# Phase II-D Cross-Axis Diagnostic Assembly

## Main interpretation
These OOD axes probe different distribution shifts and are compared descriptively rather than ranked by raw error.

### Table 1. Five-seed mean performance
| axis     | held_out                             |    N |   M0_MAE |   Mred_MAE | delta_MAE_pct   |   M0_R2 |   Mred_R2 | metric_basis                                       |
|:---------|:-------------------------------------|-----:|---------:|-----------:|:----------------|--------:|----------:|:---------------------------------------------------|
| M1       | 12 HFC refrigerants (LORO)           | 1403 |   0.1049 |     0.0666 | -36.5%          | -0.4158 |    0.3824 | 5-seed mean; macro across 12 held-out refrigerants |
| B1       | Fam-2 fluorosulfonate anions         |  513 |   0.0304 |     0.0302 | -0.7%           |  0.9265 |    0.9285 | 5-seed mean; pooled test points                    |
| B2       | Fam-3 BF4/PF6 anions                 |  598 |   0.0534 |     0.0483 | -9.6%           |  0.7931 |    0.8333 | 5-seed mean; pooled test points                    |
| L2       | 4 unseen cation–anion recombinations |  374 |   0.0334 |     0.0293 | -12.2%          |  0.8878 |    0.9126 | 5-seed mean; pooled test points                    |
| HFO/HCFO | 1106 unsaturated refrigerant points  | 1106 |   0.0468 |     0.0326 | -30.4%          |  0.492  |    0.6926 | 5-seed mean; pooled test points                    |

### Table 2. Ensemble/UQ diagnostics
| axis     |   N_test |   M0_ensemble_MAE |   Mred_ensemble_MAE | M0_mean_sigma   | Mred_mean_sigma   |   M0_rho_sigma_error |   Mred_rho_sigma_error | risk50_Mred   |
|:---------|---------:|------------------:|--------------------:|:----------------|:------------------|---------------------:|-----------------------:|:--------------|
| M1       |     1403 |            0.094  |              0.0613 | --              | --                |               0.1028 |                 0.3058 | 27.2%         |
| B1       |      513 |            0.0287 |              0.0292 | 0.0116          | 0.0092            |               0.2891 |                 0.5062 | 52.5%         |
| B2       |      598 |            0.0518 |              0.0476 | 0.0183          | 0.0110            |               0.4761 |                 0.6101 | 51.2%         |
| L2       |      374 |            0.0298 |              0.0271 | 0.0186          | 0.0142            |               0.3506 |                 0.3084 | 25.8%         |
| HFO/HCFO |     1106 |            0.0374 |              0.03   | 0.0345          | 0.0144            |               0.3407 |                 0.4926 | 34.4%         |

### Table 3. L2 pair-level results
| pair      |   M0_N |   M0_MAE |   Mred_MAE | delta_MAE_pct   |   M0_R2 |   Mred_R2 |   M0_mean_sigma |   Mred_mean_sigma |
|:----------|-------:|---------:|-----------:|:----------------|--------:|----------:|----------------:|------------------:|
| BMIM__OTF |     50 |   0.0437 |     0.0407 | 7.0%            |  0.8667 |    0.8768 |          0.0117 |            0.0121 |
| EMIM__BF4 |     93 |   0.0264 |     0.0207 | 21.6%           |  0.9436 |    0.9652 |          0.0254 |            0.0153 |
| EMIM__OTF |    114 |   0.0294 |     0.0274 | 6.7%            |  0.8657 |    0.8838 |          0.0171 |            0.0129 |
| HMIM__BF4 |    117 |   0.027  |     0.026  | 3.5%            |  0.8947 |    0.896  |          0.0177 |            0.0157 |

### Table 4. Unsaturated species zero-shot results
| species     | chemical_class   |   N |   M0_MAE |   Mred_MAE |   MAE_change | MAE_change_pct   |   M0_R2 |   Mred_R2 |   M0_sigma |   Mred_sigma |   Mred_rho_sigma_error |
|:------------|:-----------------|----:|---------:|-----------:|-------------:|:-----------------|--------:|----------:|-----------:|-------------:|-----------------------:|
| R1234yf     | HFO              | 685 |   0.032  |     0.025  |   0.00701812 | 21.9%            |  0.724  |    0.8017 |     0.0301 |       0.0134 |                 0.4849 |
| R1234ze(E)  | HFO              | 308 |   0.0391 |     0.0303 |   0.00879885 | 22.5%            |  0.5433 |    0.7614 |     0.048  |       0.0158 |                 0.5332 |
| R1233zd(E)  | HCFO             |  90 |   0.0297 |     0.0193 |   0.0104704  | 35.2%            | -0.4074 |    0.2307 |     0.0172 |       0.0165 |                 0.6303 |
| R1336mzz(E) | HFO              |  11 |   0.1196 |     0.2885 |  -0.168836   | -141.1%          | -9.0666 |  -53.4281 |     0.0774 |       0.0213 |                 0.0818 |
| R1336mzz(Z) | HFO              |  12 |   0.2854 |     0.1536 |   0.131761   | 46.2%            | -3.3085 |   -0.6739 |     0.0277 |       0.0116 |                -0.1119 |

### L2 paired effect
Mean paired ΔMAE (M0 − Mreduced) = 0.002724; bootstrap 95% CI [0.001515, 0.003915]; win rate = 59.9%; one-sided Wilcoxon p = 5.18e-06.

### Figure captions
1. **Cross-axis diagnostic MAE change.** Five-seed mean test MAE change from M0 to Mreduced. M1 uses the macro-average across 12 held-out refrigerants; B1, B2, L2, and HFO/HCFO use pooled test-point metrics within each seed.
2. **Error–uncertainty landscape.** Ensemble MAE versus mean seed-ensemble disagreement for OOD axes with directly verified global σ. M1 is omitted because the frozen UQ summary does not expose a directly verified global mean σ.
3. **Risk–coverage profiles.** Mreduced risk after rejecting samples with the highest seed-ensemble disagreement. At 50% coverage, risk reductions are 27.2% (M1), 52.5% (B1), 51.2% (B2), 25.8% (L2), and 34.4% (HFO/HCFO).
4. **Unsaturated-refrigerant zero-shot transfer.** Species-level ensemble MAE for the All-HFC model on HFO/HCFO points. R1233zd(E) is HCFO; R1336mzz(E/Z) are small-sample boundary anchors.

### Language guardrails
- Use “error–uncertainty rank association” rather than “calibration improvement” for Spearman ρ.
- Treat σ as a seed-ensemble epistemic uncertainty proxy / ensemble disagreement, not full Bayesian epistemic uncertainty.
- Treat the R1336mzz cavity/steric explanation as a hypothesis.
- Do not state that the model is unable to distinguish E/Z in its entirety; the graph branch is identical under the current featurization, while Mreduced can distinguish species indirectly through Tc/Pc/ω.