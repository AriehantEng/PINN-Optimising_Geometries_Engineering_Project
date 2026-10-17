## pinn_30-90_Re0.1_seed21

Layout 30/90, Re_h = 0.1 (viscous pressure), seed 21.

- PINN optimum s* = 0.4870 (scan, then Brent), J = 40.1926; on the range's bound: False
- loss at s*: 9.663e-02 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4843; argmin error +0.0026; regret +0.0115%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 3.37e-03 | 6.31e-03 | 8.39e-03 | -0.477% |
| 0.40 | 1.73e-03 | 3.30e-03 | 3.25e-03 | +0.229% |
| 0.45 | 1.91e-03 | 3.08e-03 | 4.14e-03 | +0.414% |
| 0.50 | 2.54e-03 | 3.06e-03 | 2.79e-03 | +0.107% |
| 0.55 | 3.50e-03 | 3.51e-03 | 3.30e-03 | -0.292% |
| 0.60 | 4.15e-03 | 4.18e-03 | 6.88e-03 | +0.733% |
| 0.65 | 5.54e-03 | 5.30e-03 | 2.67e-02 | -2.643% |
| 0.4870 (s*) | 2.34e-03 | 3.10e-03 | 1.67e-03 | +0.241% |

Training: 160 min, final training loss 7.222e-02, selected lm_average:144 with held-out loss 1.186e-01; converged: no: the held-out loss was still falling when the fixed schedule ended (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
