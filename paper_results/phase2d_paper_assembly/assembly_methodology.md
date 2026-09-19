# Phase II-D cross-axis assembly methodology

## Main performance table
Uses five-seed mean metrics for a consistent within-axis aggregation. M1 uses the frozen macro-average over 12 held-out refrigerants; B1/B2/L2/HFO use pooled test-point metrics computed for each seed and then averaged.

## Ensemble/UQ table
Uses the frozen/recomputed five-seed ensemble prediction (mean across clipped seed predictions) and sample standard deviation (ddof=1). This table is used for uncertainty and risk-coverage diagnostics.

## Risk-coverage
Samples are ordered by ensemble disagreement sigma; the highest-sigma samples are rejected first. The reported 50% values are pooled test-point MAE at 50% coverage. M1/B1/B2 curves are from frozen Phase II-C UQ outputs; L2/HFO curves are recomputed from Phase II-D source-of-truth predictions.

## Interpretation constraints
The axes are different distribution shifts and should be compared descriptively, not ranked by raw MAE. M1 and the other axes also differ in aggregation basis and test composition.
