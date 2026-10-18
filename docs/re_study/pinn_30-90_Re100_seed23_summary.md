## pinn_30-90_Re100_seed23

Layout 30/90, Re_h = 100 (inertial pressure), seed 23.

- PINN optimum s* = 0.4895 (scan, then Brent), J = 0.449906; on the range's bound: False
- finite elements at the PINN's s*: J = 0.454613 (reference mesh)
- loss at s*: 1.335e-04 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4889 (J = 0.454608); argmin error +0.0006; regret +0.0011%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 1.71e-02 | 2.32e-02 | 3.84e-02 | +1.890% |
| 0.40 | 1.23e-02 | 1.74e-02 | 3.51e-02 | +1.083% |
| 0.45 | 1.02e-02 | 1.43e-02 | 3.97e-02 | -0.489% |
| 0.50 | 9.40e-03 | 1.21e-02 | 1.56e-02 | -1.011% |
| 0.55 | 9.06e-03 | 1.04e-02 | 1.48e-02 | -0.031% |
| 0.60 | 9.56e-03 | 1.03e-02 | 1.79e-02 | +0.642% |
| 0.65 | 1.36e-02 | 1.41e-02 | 2.92e-02 | -3.997% |
| 0.4895 (s*) | 9.51e-03 | 1.25e-02 | 2.18e-02 | -1.035% |

Training: 173 min, final training loss 1.408e-04, selected lm_average:131 with held-out loss 2.423e-04; converged: yes (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
