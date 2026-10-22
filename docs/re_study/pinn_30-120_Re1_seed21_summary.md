## pinn_30-120_Re1_seed21

Layout 30/120, share q = 0.5, Re_h = 1 (viscous pressure), seed 21.

- PINN optimum s* = 0.3500 (scan, then Brent), J = 30.6225; on the range's bound: True
- finite elements at the PINN's s*: J = 31.1968 (reference mesh)
- loss at s*: 4.697e-01 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.3500 (J = 31.1968); argmin error +0.0000; regret +0.0000%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 1.34e-02 | 1.58e-02 | 2.55e-02 | -1.841% |
| 0.40 | 9.20e-03 | 9.78e-03 | 1.86e-02 | +0.525% |
| 0.45 | 8.51e-03 | 1.01e-02 | 7.27e-03 | +1.555% |
| 0.50 | 8.24e-03 | 1.01e-02 | 2.22e-03 | -0.083% |
| 0.55 | 7.86e-03 | 9.18e-03 | 2.02e-02 | -0.103% |
| 0.60 | 7.25e-03 | 8.13e-03 | 1.09e-02 | +1.405% |
| 0.65 | 9.37e-03 | 1.26e-02 | 1.24e-01 | -7.581% |
| 0.3500 (s*) | 1.34e-02 | 1.58e-02 | 2.55e-02 | -1.841% |

Training: 180 min, final training loss 3.632e-01, selected lm_average:150 with held-out loss 4.113e-01; plateaued: no: the held-out loss was still falling when the fixed schedule ended (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
