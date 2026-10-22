## pinn_30-90_Re10_seed21

Layout 30/90, share q = 0.5, Re_h = 10 (inertial pressure), seed 21.

- PINN optimum s* = 0.4862 (scan, then Brent), J = 4.03557; on the range's bound: False
- finite elements at the PINN's s*: J = 4.03597 (reference mesh)
- loss at s*: 7.959e-04 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4848 (J = 4.03583); argmin error +0.0014; regret +0.0035%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 2.73e-03 | 3.34e-03 | 1.17e-02 | -0.578% |
| 0.40 | 1.48e-03 | 2.16e-03 | 5.56e-03 | +0.302% |
| 0.45 | 1.46e-03 | 1.58e-03 | 3.48e-03 | +0.180% |
| 0.50 | 1.59e-03 | 1.43e-03 | 3.85e-03 | -0.076% |
| 0.55 | 1.90e-03 | 1.74e-03 | 2.04e-03 | -0.076% |
| 0.60 | 2.13e-03 | 1.99e-03 | 4.69e-03 | +0.339% |
| 0.65 | 4.01e-03 | 3.90e-03 | 1.21e-02 | -1.149% |
| 0.4862 (s*) | 1.52e-03 | 1.42e-03 | 2.55e-03 | -0.010% |

Training: 176 min, final training loss 8.170e-04, selected lm_average:147 with held-out loss 9.154e-04; plateaued: no: the held-out loss was still falling when the fixed schedule ended (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
