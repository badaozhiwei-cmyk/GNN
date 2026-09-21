# Supplementary Note: Computational Efficiency & Resource Quantification (F4)

This supplementary section provides full technical transparency on the model's architectural parameterization, computational complexity, inference latency, hardware footprint, and 5-seed validation convergence.

## 1. Quantitative Efficiency Summary

           Category                                            Metric                                          Value            Unit
       Architecture                                  Total Parameters                                     13,214,945           count
       Architecture                      Embedding & Token Parameters                                         50,400           count
       Architecture                  GATv2 Message Passing Parameters                                     12,101,632           count
       Architecture                       MLP Readout Head Parameters                                      1,062,913           count
         Complexity Multiply-Accumulate Operations (MACs, Analytical)                                         566.79  MMACs / sample
         Complexity     Floating Point Operations (FLOPs, Analytical)                                        1133.58 MFLOPs / sample
         Complexity                        Batched Compute (Batch=32)                                         36.275  GFLOPs / batch
            Latency                         Single-Sample CPU Latency                                   13.49 ± 1.69              ms
            Latency                         P95 Single-Sample Latency                                          15.71              ms
            Latency                    Batched CPU Latency (Batch=32)                                         339.54              ms
         Throughput                            Batched CPU Throughput                                           94.2   samples / sec
             Memory                          Process Memory Footprint                         931.3 MB (Process RSS)          memory
Training & Ensemble                      5-Seed Validation Huber Loss                            0.000581 ± 0.000194            loss
Training & Ensemble                             5-Seed Validation MAE                              0.02475 ± 0.00568   mole fraction
Training & Ensemble                              5-Seed Validation R2                                0.9504 ± 0.0178              R2
Training & Ensemble                             Optimization Protocol Max 100 epochs, Patience 15, CosineAnnealingLR        schedule

---

## 2. Reviewer Pre-emption Discussion

1. **Architectural Capacity**: The complete architecture comprises exactly **13,214,945 trainable parameters (~13.2M)**. The capacity is predominantly allocated to the 3-layer GATv2 message-passing backbone (12,101,632 params, 91.58%) with multi-head attention (heads=4) and 300-dimensional edge featurization, followed by the non-linear readout MLP (1,062,913 params, 8.04%) and molecular token embeddings (50,400 params, 0.38%).
2. **Computational Complexity**: An analytical evaluation of the matrix multiplication operations across the 3 GATv2 layers, edge projections, attention mechanisms, and readout MLP yields approximately **1133.58 MFLOPs (566.79 MMACs) per ternary evaluation** (equivalent to ~36.27 GFLOPs per batch of 32). This modest operational load allows high-throughput evaluation without specialized hardware accelerators.
3. **Empirical Latency & Throughput**: Benchmarked on a commodity single-core CPU, single-sample evaluation incurs **13.49 ± 1.69 ms** (P95: 15.71 ms). Under batched execution (batch size = 32), throughput reaches **94.2 samples / second** (10.61 ms per sample). This enables screening large thermodynamic candidate spaces (e.g., >100,000 ternary combinations in under 25 minutes on standard CPU workstations).
4. **5-Seed Ensemble Validation Robustness**: Under the standardized training protocol (maximum 100 epochs, early stopping patience of 15 epochs on validation loss, CosineAnnealingLR schedule with $\eta_{min}=10^{-5}$, and Huber loss with $\delta=0.05$), the 5 production random seeds achieve highly consistent convergence: Validation Loss = **0.000581 ± 0.000194**, Validation MAE = **0.02475 ± 0.00568**, and Validation $R^2$ = **0.9504 ± 0.0178**.
