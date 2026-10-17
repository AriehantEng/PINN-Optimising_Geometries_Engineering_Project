## pinn_30-90_Re100_seed21

Layout 30/90, Re_h = 100 (inertial pressure), seed 21.

- PINN optimum s* = 0.4838 (scan, then Brent), J = 0.452993; on the range's bound: False
- loss at s*: 1.452e-04 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4889; argmin error -0.0050; regret +0.0421%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 7.69e-03 | 1.05e-02 | 3.39e-02 | +0.197% |
| 0.40 | 5.17e-03 | 4.97e-03 | 1.65e-02 | +0.226% |
| 0.45 | 4.75e-03 | 4.64e-03 | 3.54e-02 | -0.601% |
| 0.50 | 3.69e-03 | 3.73e-03 | 9.17e-03 | -0.068% |
| 0.55 | 4.75e-03 | 4.38e-03 | 2.15e-02 | +1.344% |
| 0.60 | 6.17e-03 | 5.32e-03 | 1.63e-02 | +1.410% |
| 0.65 | 1.34e-02 | 1.12e-02 | 4.66e-02 | -3.614% |
| 0.4838 (s*) | 3.88e-03 | 3.90e-03 | 2.04e-02 | -0.397% |

Training: 160 min, final training loss 1.286e-04, selected lm:144 with held-out loss 2.122e-04; converged: no: the held-out loss was still falling when the fixed schedule ended (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
