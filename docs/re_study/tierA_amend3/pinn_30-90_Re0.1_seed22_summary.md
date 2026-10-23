## pinn_30-90_Re0.1_seed22

Layout 30/90, share q = 0.5, Re_h = 0.1 (viscous pressure), seed 22; trained over s = 0.3343 to 0.6343.

- PINN optimum s* = 0.4855 (scan, then Brent), J = 40.0101; on the range's bound: False
- finite elements at the PINN's s*: J = 40.0924 (reference mesh)
- loss at s*: 2.263e-01 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4844 (J = 40.0916); argmin error +0.0011; regret +0.0021%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.33 | 1.47e-02 | 1.88e-02 | 1.17e-01 | -0.992% |
| 0.38 | 1.09e-02 | 1.40e-02 | 2.28e-02 | +0.497% |
| 0.43 | 9.34e-03 | 1.29e-02 | 2.56e-02 | +0.460% |
| 0.48 | 8.62e-03 | 1.24e-02 | 1.28e-02 | -0.200% |
| 0.53 | 8.30e-03 | 1.16e-02 | 2.49e-02 | +0.621% |
| 0.58 | 8.18e-03 | 1.05e-02 | 1.85e-02 | +1.439% |
| 0.63 | 1.04e-02 | 1.28e-02 | 9.73e-02 | -5.271% |
| 0.4855 (s*) | 8.61e-03 | 1.24e-02 | 1.20e-02 | -0.205% |

Training: 183 min, final training loss 1.479e+00, selected lm_average:199 with held-out loss 6.504e-01; plateaued: yes (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
