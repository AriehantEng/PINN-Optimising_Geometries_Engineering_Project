## pinn_30-90_Re100_seed22

Layout 30/90, Re_h = 100 (inertial pressure), seed 22.

- PINN optimum s* = 0.4876 (scan, then Brent), J = 0.44931; on the range's bound: False
- finite elements at the PINN's s*: J = 0.454621 (reference mesh)
- loss at s*: 8.757e-05 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4889 (J = 0.454608); argmin error -0.0012; regret +0.0028%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 5.65e-03 | 7.43e-03 | 1.56e-02 | -1.186% |
| 0.40 | 4.93e-03 | 5.17e-03 | 1.35e-02 | -0.236% |
| 0.45 | 5.00e-03 | 4.85e-03 | 2.17e-02 | -0.986% |
| 0.50 | 4.56e-03 | 3.99e-03 | 8.73e-03 | -1.059% |
| 0.55 | 5.36e-03 | 4.08e-03 | 1.13e-02 | +0.205% |
| 0.60 | 6.82e-03 | 6.06e-03 | 1.28e-02 | +1.167% |
| 0.65 | 1.03e-02 | 1.17e-02 | 2.72e-02 | -2.920% |
| 0.4876 (s*) | 4.64e-03 | 4.21e-03 | 1.32e-02 | -1.168% |

Training: 166 min, final training loss 1.429e-04, selected lm_average:150 with held-out loss 1.464e-04; converged: no: the held-out loss was still falling when the fixed schedule ended (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
