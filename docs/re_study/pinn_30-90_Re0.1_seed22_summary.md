## pinn_30-90_Re0.1_seed22

Layout 30/90, Re_h = 0.1 (viscous pressure), seed 22.

- PINN optimum s* = 0.4902 (scan, then Brent), J = 40.1215; on the range's bound: False
- finite elements at the PINN's s*: J = 40.1141 (reference mesh)
- loss at s*: 1.202e-01 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4843 (J = 40.0915); argmin error +0.0058; regret +0.0563%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 5.68e-03 | 6.15e-03 | 3.43e-02 | -0.928% |
| 0.40 | 2.70e-03 | 3.88e-03 | 1.26e-02 | -0.217% |
| 0.45 | 2.95e-03 | 3.19e-03 | 8.95e-03 | +0.371% |
| 0.50 | 2.63e-03 | 3.04e-03 | 8.61e-03 | -0.178% |
| 0.55 | 2.47e-03 | 2.90e-03 | 3.59e-03 | -0.633% |
| 0.60 | 2.40e-03 | 2.58e-03 | 1.25e-02 | +0.803% |
| 0.65 | 6.49e-03 | 5.90e-03 | 4.52e-02 | -4.575% |
| 0.4902 (s*) | 2.73e-03 | 3.04e-03 | 6.42e-03 | +0.018% |

Training: 170 min, final training loss 9.121e-02, selected lm:150 with held-out loss 1.619e-01; converged: no: the held-out loss was still falling when the fixed schedule ended (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
