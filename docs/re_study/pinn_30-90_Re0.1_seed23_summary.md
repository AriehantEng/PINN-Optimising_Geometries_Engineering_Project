## pinn_30-90_Re0.1_seed23

Layout 30/90, share q = 0.5, Re_h = 0.1 (viscous pressure), seed 23.

- PINN optimum s* = 0.4862 (scan, then Brent), J = 40.2507; on the range's bound: False
- finite elements at the PINN's s*: J = 40.0938 (reference mesh)
- loss at s*: 1.052e-01 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4843 (J = 40.0915); argmin error +0.0018; regret +0.0058%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.35 | 4.90e-03 | 5.78e-03 | 1.76e-02 | -0.118% |
| 0.40 | 3.47e-03 | 3.13e-03 | 2.47e-03 | -0.062% |
| 0.45 | 3.53e-03 | 2.96e-03 | 2.36e-03 | +0.331% |
| 0.50 | 3.12e-03 | 2.58e-03 | 1.70e-03 | +0.255% |
| 0.55 | 3.75e-03 | 2.87e-03 | 2.49e-03 | -0.538% |
| 0.60 | 4.60e-03 | 3.59e-03 | 8.05e-03 | +0.546% |
| 0.65 | 6.82e-03 | 5.96e-03 | 4.33e-02 | -3.884% |
| 0.4862 (s*) | 3.18e-03 | 2.61e-03 | 2.07e-03 | +0.391% |

Training: 162 min, final training loss 1.108e-01, selected lm:144 with held-out loss 1.504e-01; plateaued: no: the held-out loss was still falling when the fixed schedule ended (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
