## pinn_30-90_Re0.1_seed21

Layout 30/90, share q = 0.5, Re_h = 0.1 (viscous pressure), seed 21; trained over s = 0.3343 to 0.6343.

- PINN optimum s* = 0.4831 (scan, then Brent), J = 40.0399; on the range's bound: False
- finite elements at the PINN's s*: J = 40.0927 (reference mesh)
- loss at s*: 3.748e-01 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4844 (J = 40.0916); argmin error -0.0012; regret +0.0029%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.33 | 1.60e-02 | 2.54e-02 | 3.38e-01 | -4.098% |
| 0.38 | 1.08e-02 | 1.43e-02 | 5.19e-02 | +1.344% |
| 0.43 | 1.06e-02 | 1.35e-02 | 5.15e-02 | +0.794% |
| 0.48 | 1.07e-02 | 1.33e-02 | 7.56e-03 | -0.125% |
| 0.53 | 9.83e-03 | 1.25e-02 | 4.46e-02 | +1.295% |
| 0.58 | 9.00e-03 | 1.13e-02 | 7.00e-03 | +0.878% |
| 0.63 | 1.59e-02 | 1.70e-02 | 1.77e-01 | -7.304% |
| 0.4831 (s*) | 1.07e-02 | 1.34e-02 | 9.09e-03 | -0.132% |

Training: 186 min, final training loss 6.593e-01, selected lm_average:150 with held-out loss 2.042e+00; plateaued: yes (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
