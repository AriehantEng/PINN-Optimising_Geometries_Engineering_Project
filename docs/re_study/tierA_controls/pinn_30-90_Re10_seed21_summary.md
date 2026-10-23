## pinn_30-90_Re10_seed21

Layout 30/90, share q = 0.5, Re_h = 10 (inertial pressure), seed 21; trained over s = 0.3343 to 0.6343.

- PINN optimum s* = 0.4838 (scan, then Brent), J = 4.06263; on the range's bound: False
- finite elements at the PINN's s*: J = 4.03589 (reference mesh)
- loss at s*: 4.970e-03 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4848 (J = 4.03584); argmin error -0.0010; regret +0.0013%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.33 | 1.89e-02 | 2.41e-02 | 1.54e-02 | +1.159% |
| 0.38 | 1.58e-02 | 1.96e-02 | 1.21e-02 | +1.021% |
| 0.43 | 1.42e-02 | 1.82e-02 | 1.00e-02 | +0.713% |
| 0.48 | 1.28e-02 | 1.76e-02 | 6.67e-03 | +0.664% |
| 0.53 | 1.22e-02 | 1.73e-02 | 1.29e-02 | +1.172% |
| 0.58 | 1.31e-02 | 1.79e-02 | 2.02e-02 | +1.637% |
| 0.63 | 1.84e-02 | 2.28e-02 | 2.34e-02 | -1.422% |
| 0.4838 (s*) | 1.29e-02 | 1.76e-02 | 6.75e-03 | +0.662% |

Training: 172 min, final training loss 8.861e-03, selected lm_average:116 with held-out loss 1.732e-02; plateaued: yes (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
