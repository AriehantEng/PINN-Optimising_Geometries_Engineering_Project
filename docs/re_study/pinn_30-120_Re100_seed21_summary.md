## pinn_30-120_Re100_seed21

Layout 30/120, share q = 0.5, Re_h = 100 (inertial pressure), seed 21.

- PINN optimum s* = 0.3998 (scan, then Brent), J = 0.428008; on the range's bound: False
- finite elements at the PINN's s*: J = 0.426095 (reference mesh)
- loss at s*: 1.654e-04 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.3859 (J = 0.425753); argmin error +0.0139; regret +0.0803%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 1.27e-02 | 1.63e-02 | 7.49e-02 | +0.574% |
| 0.40 | 8.74e-03 | 1.22e-02 | 1.19e-02 | +0.450% |
| 0.45 | 7.74e-03 | 1.17e-02 | 1.92e-02 | -0.018% |
| 0.50 | 8.14e-03 | 1.34e-02 | 1.19e-02 | +0.661% |
| 0.55 | 9.48e-03 | 1.65e-02 | 3.48e-02 | +2.458% |
| 0.60 | 1.24e-02 | 2.15e-02 | 2.95e-02 | +2.675% |
| 0.65 | 1.94e-02 | 3.08e-02 | 3.63e-02 | -3.240% |
| 0.3998 (s*) | 8.74e-03 | 1.22e-02 | 1.19e-02 | +0.449% |

Training: 184 min, final training loss 7.754e-04, selected lm_average:141 with held-out loss 2.030e-04; plateaued: no: the held-out loss was still falling when the fixed schedule ended (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
