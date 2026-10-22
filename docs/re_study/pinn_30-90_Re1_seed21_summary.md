## pinn_30-90_Re1_seed21

Layout 30/90, share q = 0.5, Re_h = 1 (viscous pressure), seed 21.

- PINN optimum s* = 0.4830 (scan, then Brent), J = 40.3011; on the range's bound: False
- finite elements at the PINN's s*: J = 40.0962 (reference mesh)
- loss at s*: 1.158e-01 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4843 (J = 40.095); argmin error -0.0013; regret +0.0032%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 5.40e-03 | 7.31e-03 | 1.26e-02 | +0.218% |
| 0.40 | 2.29e-03 | 4.26e-03 | 5.42e-03 | -0.041% |
| 0.45 | 2.46e-03 | 3.14e-03 | 1.48e-03 | +0.194% |
| 0.50 | 3.32e-03 | 3.15e-03 | 4.06e-03 | +0.535% |
| 0.55 | 4.46e-03 | 4.09e-03 | 1.34e-03 | -0.193% |
| 0.60 | 4.47e-03 | 3.94e-03 | 2.90e-03 | +0.288% |
| 0.65 | 6.81e-03 | 5.62e-03 | 3.04e-02 | -3.140% |
| 0.4830 (s*) | 3.03e-03 | 3.01e-03 | 3.86e-03 | +0.511% |

Training: 176 min, final training loss 1.748e-01, selected lm:147 with held-out loss 1.749e-01; plateaued: no: the held-out loss was still falling when the fixed schedule ended (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
