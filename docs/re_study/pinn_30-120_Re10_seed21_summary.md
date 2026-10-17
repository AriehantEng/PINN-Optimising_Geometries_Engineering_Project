## pinn_30-120_Re10_seed21

Layout 30/120, Re_h = 10 (inertial pressure), seed 21.

- PINN optimum s* = 0.3500 (scan, then Brent), J = 3.15109; on the range's bound: True
- loss at s*: 3.171e-03 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.3500; argmin error +0.0000; regret +0.0000%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 1.05e-02 | 1.05e-02 | 1.11e-02 | -0.175% |
| 0.40 | 7.06e-03 | 6.76e-03 | 1.57e-03 | +0.046% |
| 0.45 | 7.14e-03 | 6.18e-03 | 3.38e-03 | -0.065% |
| 0.50 | 8.01e-03 | 6.43e-03 | 2.16e-03 | -0.108% |
| 0.55 | 8.74e-03 | 6.85e-03 | 2.83e-03 | +0.095% |
| 0.60 | 8.79e-03 | 7.11e-03 | 5.27e-03 | +0.364% |
| 0.65 | 8.58e-03 | 8.13e-03 | 1.35e-02 | -1.276% |
| 0.3500 (s*) | 1.05e-02 | 1.05e-02 | 1.11e-02 | -0.175% |

Training: 167 min, final training loss 1.361e-03, selected lm_average:132 with held-out loss 2.008e-03; converged: no: the held-out loss was still falling when the fixed schedule ended (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
