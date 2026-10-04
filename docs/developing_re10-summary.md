## Every setup against the finite-element reference, most accurate first

Relative L2 errors (entrance length: relative error). Several seeds: mean (range). The reference's own error is the last row; values near it cannot be ranked against each other.

| Setup | Seeds | u, whole channel | u, entrance region | u, developed region | v, whole channel | p, whole channel | entrance length | total-pressure drop (the design objective) | viscous dissipation | energy budget, PINN alone (no reference) | Training time (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P3 G3 + 100 steps, mean of 20, 32k held-out | 1 | 4.62e-06 | 6.58e-06 | 2.91e-06 | 1.36e-04 | 2.01e-05 | 9.25e-05 | 1.97e-06 | 4.00e-06 | 3.38e-06 | 23 |
| G3 LM, fresh points each step | 1 | 5.42e-06 | 5.56e-06 | 5.33e-06 | 1.57e-04 | 2.22e-05 | 1.32e-05 | 1.10e-05 | 5.29e-06 | 1.23e-05 | 206 |
| P2 P1 with 16k points per step | 1 | 6.08e-06 | 5.49e-06 | 6.40e-06 | 1.32e-04 | 2.03e-05 | 1.12e-05 | 5.97e-06 | 6.90e-06 | 1.06e-06 | 76 |
| G1 width 40 + LM | 1 | 7.32e-06 | 8.58e-06 | 6.47e-06 | 1.68e-04 | 2.20e-05 | 1.93e-04 | 1.27e-05 | 3.86e-06 | 1.22e-05 | 123 |
| P4 P1 with focus fraction 0.75 | 1 | 7.72e-06 | 5.91e-06 | 8.60e-06 | 1.98e-04 | 2.31e-05 | 4.53e-05 | 1.11e-05 | 6.43e-06 | 3.34e-06 | 25 |
| P1 G4 + 100 steps, mean of 20, 32k held-out | 3 | 7.75e-06 (7.4e-06 to 8.2e-06) | 7.13e-06 (6.7e-06 to 7.5e-06) | 8.08e-06 (7.3e-06 to 8.7e-06) | 1.48e-04 (1.4e-04 to 1.7e-04) | 2.01e-05 (2.0e-05 to 2.0e-05) | 3.49e-05 (1.3e-05 to 5.9e-05) | 5.30e-06 (4.8e-06 to 6.2e-06) | 1.06e-05 (1.0e-05 to 1.1e-05) | 3.04e-06 (1.4e-06 to 5.6e-06) | 26 |
| G4 G3 + dense corner inlet grid | 1 | 8.31e-06 | 6.17e-06 | 9.34e-06 | 1.64e-04 | 1.93e-05 | 1.14e-05 | 2.43e-06 | 9.98e-06 | 4.15e-06 | 229 |
| P0 G4 + 100 LM steps | 1 | 1.18e-05 | 1.28e-05 | 1.12e-05 | 2.12e-04 | 2.58e-05 | 1.58e-05 | 1.43e-05 | 1.90e-05 | 5.30e-06 | 26 |
| E8 exponential input map | 1 | 1.98e-04 | 2.34e-04 | 1.73e-04 | 3.21e-03 | 2.80e-04 | 2.75e-03 | 2.37e-05 | 8.99e-05 | 9.40e-05 | 51 |
| G2 stream, exact BCs + LM | 1 | 2.06e-04 | 1.96e-04 | 2.12e-04 | 6.06e-03 | 4.34e-04 | 3.00e-03 | 6.82e-05 | 1.06e-05 | 7.86e-05 | 112 |
| E1 24 L-BFGS cycles | 3 | 2.30e-04 (1.8e-04 to 3.1e-04) | 1.53e-04 (1.4e-04 to 1.6e-04) | 2.62e-04 (1.9e-04 to 3.7e-04) | 3.86e-03 (3.7e-03 to 4.1e-03) | 4.06e-04 (3.6e-04 to 4.8e-04) | 5.38e-03 (9.2e-04 to 9.2e-03) | 5.61e-05 (7.3e-06 to 9.6e-05) | 4.11e-04 (2.8e-04 to 6.0e-04) | 3.34e-04 (1.7e-04 to 5.7e-04) | 81 |
| baseline (reference setup) | 3 | 2.83e-04 (2.4e-04 to 3.3e-04) | 2.38e-04 (2.2e-04 to 2.5e-04) | 3.06e-04 (2.4e-04 to 3.7e-04) | 5.62e-03 (5.4e-03 to 6.0e-03) | 4.86e-04 (3.8e-04 to 6.3e-04) | 4.37e-03 (2.8e-03 to 6.5e-03) | 1.36e-04 (2.7e-05 to 2.1e-04) | 1.71e-04 (1.5e-05 to 3.0e-04) | 1.68e-04 (3.0e-05 to 3.2e-04) | 39 |
| E4 proportional RAR | 1 | 2.86e-04 | 1.62e-04 | 3.38e-04 | 4.58e-03 | 6.71e-04 | 7.64e-03 | 1.66e-04 | 4.57e-04 | 2.66e-04 | 44 |
| E7 stream function, soft inlet | 3 | 3.12e-04 (2.6e-04 to 3.4e-04) | 3.95e-04 (3.1e-04 to 4.7e-04) | 2.47e-04 (2.2e-04 to 3.0e-04) | 7.24e-03 (7.0e-03 to 7.6e-03) | 7.40e-04 (5.7e-04 to 8.5e-04) | 5.06e-03 (4.1e-03 to 5.8e-03) | 1.47e-04 (3.1e-06 to 2.3e-04) | 4.40e-04 (3.2e-04 to 5.2e-04) | 5.85e-04 (3.1e-04 to 7.6e-04) | 93 |
| E3d inlet grid, weight 1 | 1 | 3.27e-04 | 2.08e-04 | 3.79e-04 | 6.04e-03 | 3.93e-04 | 4.68e-03 | 4.99e-05 | 7.90e-06 | 1.91e-05 | 52 |
| E3c inlet grid, weight 10 | 1 | 3.59e-04 | 2.71e-04 | 4.02e-04 | 7.77e-03 | 5.87e-04 | 4.12e-03 | 3.79e-04 | 2.96e-04 | 8.50e-05 | 39 |
| E2 10k Adam, cosine | 1 | 3.69e-04 | 1.96e-04 | 4.40e-04 | 4.05e-03 | 5.85e-04 | 1.05e-02 | 3.60e-05 | 7.02e-04 | 6.13e-04 | 59 |
| greedy RAR | 1 | 3.77e-04 | 2.86e-04 | 4.21e-04 | 7.28e-03 | 6.28e-04 | 2.78e-03 | 2.90e-04 | 2.80e-05 | 3.14e-04 | 47 |
| G0 width 40, no LM | 1 | 5.02e-04 | 3.75e-04 | 5.64e-04 | 1.15e-02 | 1.02e-03 | 1.25e-02 | 2.64e-04 | 2.24e-04 | 2.00e-05 | 16 |
| E6b SiLU, init gain 2 | 1 | 7.42e-04 | 2.52e-04 | 9.15e-04 | 9.31e-03 | 1.05e-03 | 1.10e-03 | 6.74e-04 | 5.56e-04 | 7.18e-05 | 56 |
| E6 SiLU | 1 | 8.30e-04 | 3.52e-04 | 1.01e-03 | 9.78e-03 | 1.08e-03 | 8.55e-04 | 7.38e-04 | 5.18e-04 | 1.84e-04 | 41 |
| E5c exact inlet, exp ramp, blended lift | 1 | 1.12e-03 | 4.78e-04 | 1.36e-03 | 2.63e-02 | 1.03e-03 | 8.03e-04 | 3.48e-05 | 6.95e-04 | 7.29e-04 | 49 |
| E5b exact inlet, exp ramp | 1 | 1.41e-03 | 6.55e-04 | 1.70e-03 | 2.50e-02 | 3.11e-03 | 4.16e-02 | 6.25e-04 | 9.31e-04 | 3.07e-04 | 51 |
| E7b stream, exact inlet | 1 | 1.45e-03 | 1.22e-03 | 1.56e-03 | 4.58e-02 | 4.58e-03 | 1.21e-02 | 4.51e-04 | 4.28e-05 | 4.09e-04 | 105 |
| E9 stream, exact inlet, blended lift | 1 | 1.76e-03 | 1.05e-03 | 2.06e-03 | 4.01e-02 | 3.58e-03 | 9.92e-03 | 2.12e-03 | 4.03e-05 | 2.08e-03 | 104 |
| E5 exact inlet, linear ramp | 1 | 9.22e-03 | 4.23e-03 | 1.12e-02 | 1.04e-01 | 1.66e-02 | 1.54e-01 | 1.39e-03 | 1.50e-02 | 1.38e-02 | 37 |
| *finite-element reference's own error* | | 8.7e-07 | 8.7e-07 | 8.7e-07 | 9.8e-06 | 1.5e-05 | 1.0e-06 | 6.7e-08 | 1.8e-07 | 1.5e-07 | |
