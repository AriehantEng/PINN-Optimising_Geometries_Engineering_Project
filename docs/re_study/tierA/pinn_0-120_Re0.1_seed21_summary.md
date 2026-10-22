## pinn_0-120_Re0.1_seed21

Layout 0/120, share q = 0.5, Re_h = 0.1 (viscous pressure), seed 21; trained over s = 0.3209 to 0.6209.

- PINN optimum s* = 0.4527 (scan, then Brent), J = 35.7921; on the range's bound: False
- finite elements at the PINN's s*: J = 35.7737 (reference mesh)
- loss at s*: 3.503e-01 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4526 (J = 35.7737); argmin error +0.0001; regret -0.0000%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.32 | 1.88e-02 | 3.13e-02 | 3.38e-01 | -5.176% |
| 0.37 | 1.43e-02 | 2.65e-02 | 9.77e-02 | +1.881% |
| 0.42 | 1.38e-02 | 2.62e-02 | 1.10e-02 | +0.803% |
| 0.47 | 1.18e-02 | 2.27e-02 | 1.17e-02 | +0.310% |
| 0.52 | 1.08e-02 | 2.02e-02 | 3.97e-02 | +1.814% |
| 0.57 | 1.10e-02 | 2.04e-02 | 3.21e-02 | -0.567% |
| 0.62 | 1.71e-02 | 2.93e-02 | 2.06e-01 | -8.119% |
| 0.4527 (s*) | 1.25e-02 | 2.40e-02 | 6.05e-03 | +0.052% |

Training: 170 min, final training loss 1.338e+00, selected lm_average:131 with held-out loss 1.114e+00; plateaued: yes (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
