## pinn_30-90_Re10_seed23

Layout 30/90, share q = 0.5, Re_h = 10 (inertial pressure), seed 23.

- PINN optimum s* = 0.4855 (scan, then Brent), J = 4.03543; on the range's bound: False
- finite elements at the PINN's s*: J = 4.03588 (reference mesh)
- loss at s*: 7.026e-04 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4848 (J = 4.03583); argmin error +0.0008; regret +0.0011%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 2.87e-03 | 4.03e-03 | 9.65e-03 | -0.746% |
| 0.40 | 1.78e-03 | 2.88e-03 | 3.33e-03 | +0.103% |
| 0.45 | 1.67e-03 | 2.98e-03 | 2.94e-03 | +0.081% |
| 0.50 | 1.91e-03 | 3.08e-03 | 3.41e-03 | -0.057% |
| 0.55 | 2.17e-03 | 3.09e-03 | 2.89e-03 | -0.068% |
| 0.60 | 2.77e-03 | 3.30e-03 | 4.15e-03 | +0.436% |
| 0.65 | 3.80e-03 | 4.14e-03 | 1.13e-02 | -0.931% |
| 0.4855 (s*) | 1.84e-03 | 3.06e-03 | 2.02e-03 | -0.011% |

Training: 175 min, final training loss 7.259e-04, selected lm_average:150 with held-out loss 9.154e-04; plateaued: no: the held-out loss was still falling when the fixed schedule ended (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
