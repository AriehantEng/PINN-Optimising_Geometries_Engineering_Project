## pinn_30-90_Re10_seed22

Layout 30/90, Re_h = 10 (inertial pressure), seed 22.

- PINN optimum s* = 0.4852 (scan, then Brent), J = 4.03636; on the range's bound: False
- finite elements at the PINN's s*: J = 4.03584 (reference mesh)
- loss at s*: 5.969e-04 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4848 (J = 4.03583); argmin error +0.0004; regret +0.0002%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 3.34e-03 | 3.16e-03 | 7.62e-03 | -0.403% |
| 0.40 | 2.27e-03 | 2.05e-03 | 3.90e-03 | +0.109% |
| 0.45 | 2.40e-03 | 2.15e-03 | 2.16e-03 | +0.029% |
| 0.50 | 2.41e-03 | 2.13e-03 | 3.29e-03 | +0.004% |
| 0.55 | 2.33e-03 | 2.04e-03 | 1.37e-03 | +0.016% |
| 0.60 | 2.54e-03 | 2.23e-03 | 4.14e-03 | +0.281% |
| 0.65 | 4.24e-03 | 4.27e-03 | 1.29e-02 | -0.760% |
| 0.4852 (s*) | 2.45e-03 | 2.19e-03 | 2.34e-03 | +0.013% |

Training: 163 min, final training loss 6.918e-04, selected lm_average:139 with held-out loss 9.310e-04; converged: no: the held-out loss was still falling when the fixed schedule ended (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
