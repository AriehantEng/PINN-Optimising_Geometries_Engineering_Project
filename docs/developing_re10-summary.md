## Every setup against the finite-element reference, most accurate first

Relative L2 errors (entrance length: relative error). Several seeds: mean (range). The reference's own error is the last row; values near it cannot be ranked against each other.

| Setup | Seeds | u, whole channel | u, entrance region | u, developed region | v, whole channel | p, whole channel | entrance length | Training time (min) |
|---|---|---|---|---|---|---|---|---|
| G3 LM, fresh points each step | 1 | 5.50e-06 | 5.77e-06 | 5.33e-06 | 1.59e-04 | 3.37e-05 | 9.12e-06 | 206 |
| G1 width 40 + LM | 1 | 7.42e-06 | 8.81e-06 | 6.47e-06 | 1.72e-04 | 3.44e-05 | 1.87e-04 | 123 |
| E8 exponential input map | 1 | 1.98e-04 | 2.35e-04 | 1.73e-04 | 3.34e-03 | 3.17e-04 | 2.77e-03 | 51 |
| G2 stream, exact BCs + LM | 1 | 2.06e-04 | 1.96e-04 | 2.12e-04 | 6.06e-03 | 4.34e-04 | 2.98e-03 | 112 |
| E1 24 L-BFGS cycles | 3 | 2.30e-04 (1.8e-04 to 3.1e-04) | 1.54e-04 (1.4e-04 to 1.6e-04) | 2.62e-04 (1.9e-04 to 3.7e-04) | 3.99e-03 (3.8e-03 to 4.1e-03) | 4.37e-04 (4.0e-04 to 5.1e-04) | 5.37e-03 (9.0e-04 to 9.2e-03) | 81 |
| baseline (reference setup) | 3 | 2.84e-04 (2.4e-04 to 3.3e-04) | 2.39e-04 (2.2e-04 to 2.5e-04) | 3.06e-04 (2.4e-04 to 3.7e-04) | 5.77e-03 (5.6e-03 to 6.1e-03) | 5.31e-04 (4.2e-04 to 6.7e-04) | 4.37e-03 (2.9e-03 to 6.5e-03) | 39 |
| E4 proportional RAR | 1 | 2.86e-04 | 1.63e-04 | 3.39e-04 | 4.69e-03 | 6.90e-04 | 7.63e-03 | 44 |
| E7 stream function, soft inlet | 3 | 3.15e-04 (2.6e-04 to 3.4e-04) | 4.02e-04 (3.2e-04 to 4.7e-04) | 2.47e-04 (2.2e-04 to 3.0e-04) | 7.59e-03 (7.3e-03 to 8.0e-03) | 8.23e-04 (6.3e-04 to 9.4e-04) | 5.06e-03 (4.1e-03 to 5.8e-03) | 93 |
| E3d inlet grid, weight 1 | 1 | 3.27e-04 | 2.10e-04 | 3.79e-04 | 6.16e-03 | 4.29e-04 | 4.69e-03 | 52 |
| E3c inlet grid, weight 10 | 1 | 3.59e-04 | 2.71e-04 | 4.02e-04 | 7.79e-03 | 5.94e-04 | 4.14e-03 | 39 |
| E2 10k Adam, cosine | 1 | 3.69e-04 | 1.97e-04 | 4.40e-04 | 4.18e-03 | 6.16e-04 | 1.05e-02 | 59 |
| greedy RAR | 1 | 3.77e-04 | 2.88e-04 | 4.21e-04 | 7.39e-03 | 6.57e-04 | 2.80e-03 | 47 |
| G0 width 40, no LM | 1 | 5.03e-04 | 3.77e-04 | 5.64e-04 | 1.16e-02 | 1.04e-03 | 1.26e-02 | 16 |
| E6b SiLU, init gain 2 | 1 | 7.42e-04 | 2.54e-04 | 9.15e-04 | 9.42e-03 | 1.08e-03 | 1.09e-03 | 56 |
| E6 SiLU | 1 | 8.30e-04 | 3.53e-04 | 1.01e-03 | 9.99e-03 | 1.13e-03 | 8.43e-04 | 41 |
| E5c exact inlet, exp ramp, blended lift | 1 | 1.12e-03 | 4.77e-04 | 1.36e-03 | 2.63e-02 | 1.04e-03 | 7.92e-04 | 49 |
| E5b exact inlet, exp ramp | 1 | 1.41e-03 | 6.55e-04 | 1.70e-03 | 2.51e-02 | 3.11e-03 | 4.16e-02 | 51 |
| E7b stream, exact inlet | 1 | 1.44e-03 | 1.22e-03 | 1.56e-03 | 4.58e-02 | 4.58e-03 | 1.21e-02 | 105 |
| E9 stream, exact inlet, blended lift | 1 | 1.76e-03 | 1.05e-03 | 2.06e-03 | 4.01e-02 | 3.58e-03 | 9.93e-03 | 104 |
| E5 exact inlet, linear ramp | 1 | 9.22e-03 | 4.23e-03 | 1.12e-02 | 1.04e-01 | 1.67e-02 | 1.54e-01 | 37 |
| *finite-element reference's own error* | | 9.1e-07 | 9.1e-07 | 9.1e-07 | 1.0e-05 | 1.5e-05 | 1.0e-06 | |
