## pinn_0-120_Re10_seed21

Layout 0/120, share q = 0.5, Re_h = 10 (inertial pressure), seed 21; trained over s = 0.3209 to 0.6209.

- PINN optimum s* = 0.4527 (scan, then Brent), J = 3.6127; on the range's bound: False
- finite elements at the PINN's s*: J = 3.62273 (reference mesh)
- loss at s*: 3.241e-03 (4 x 1024 interior points at s*, stream 2026)
- reference-mesh optimum (Brent from the evaluation splits): 0.4527 (J = 3.62273); argmin error +0.0000; regret -0.0000%

| s | u | v | p | drop error |
|---|---|---|---|---|
| 0.32 | 1.81e-02 | 2.99e-02 | 2.48e-02 | +1.136% |
| 0.37 | 1.32e-02 | 2.40e-02 | 8.44e-03 | +0.620% |
| 0.42 | 1.20e-02 | 2.23e-02 | 1.09e-02 | -0.092% |
| 0.47 | 1.17e-02 | 2.12e-02 | 5.83e-03 | -0.184% |
| 0.52 | 1.16e-02 | 2.08e-02 | 1.53e-02 | +0.940% |
| 0.57 | 1.20e-02 | 2.26e-02 | 2.04e-02 | +1.643% |
| 0.62 | 1.50e-02 | 3.03e-02 | 3.24e-02 | -3.193% |
| 0.4527 (s*) | 1.17e-02 | 2.15e-02 | 9.53e-03 | -0.277% |

Training: 153 min, final training loss 2.777e-02, selected lm_average:119 with held-out loss 5.355e-03; converged: yes (definition: finished, every loss finite, and the best held-out loss of the last 25 LM steps not more than 10% below the best before them)
