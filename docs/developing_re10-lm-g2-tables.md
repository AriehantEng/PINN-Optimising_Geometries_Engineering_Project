## Table 1. Whole-channel accuracy against the finite-element reference

The reference's own error is about 1e-6 in u (mesh convergence, docs/developing_re10-reference.json), well below every number here.

| Run | u, relative L2, whole channel | u, entrance region x* < 3.77 | u, developed region | largest error in u | v, relative L2 | p, relative L2 | entrance length (reference 1.7794) | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|
| G1 velocity, soft inlet + LM | 7.42e-06 | 8.81e-06 | 6.47e-06 | 2.77e-04 | 1.72e-04 | 3.44e-05 | 1.7791 (-0.02%) | 3.08e-08 | 122.8 |
| G2 stream, all BCs exact + LM | 2.06e-04 | 1.96e-04 | 2.12e-04 | 4.57e-04 | 6.06e-03 | 4.34e-04 | 1.7741 (-0.30%) | 1.85e-04 | 112.2 |
| E7 stream, soft inlet (seed 0) | 3.45e-04 | 4.15e-04 | 2.96e-04 | 3.25e-03 | 7.34e-03 | 6.26e-04 | 1.7701 (-0.52%) | 1.95e-05 | 72.3 |

## Table 2. u across the channel at each station, against the reference

| Station x* | Run | u, relative L2 | largest error in u | where (y*) | v, largest error | v, largest reference value |
|---|---|---|---|---|---|---|
| 0 | G1 velocity, soft inlet + LM | 5.61e-05 | 2.95e-04 | +0.994 | 1.62e-04 | 0.00e+00 |
| 0 | G2 stream, all BCs exact + LM | 5.48e-06 | 1.43e-05 | +0.908 | 0.00e+00 | 0.00e+00 |
| 0 | E7 stream, soft inlet (seed 0) | 1.95e-03 | 3.25e-03 | -0.826 | 7.60e-03 | 0.00e+00 |
| 0.5 | G1 velocity, soft inlet + LM | 5.55e-06 | 8.94e-06 | +0.522 | 5.59e-06 | 1.14e-01 |
| 0.5 | G2 stream, all BCs exact + LM | 1.07e-04 | 2.27e-04 | -0.816 | 2.08e-04 | 1.14e-01 |
| 0.5 | E7 stream, soft inlet (seed 0) | 3.77e-04 | 5.71e-04 | +0.198 | 1.72e-04 | 1.14e-01 |
| 1 | G1 velocity, soft inlet + LM | 4.46e-06 | 8.26e-06 | -0.188 | 7.34e-06 | 4.73e-02 |
| 1 | G2 stream, all BCs exact + LM | 1.81e-04 | 3.57e-04 | -0.334 | 1.84e-04 | 4.73e-02 |
| 1 | E7 stream, soft inlet (seed 0) | 3.33e-04 | 4.68e-04 | +0.390 | 2.75e-04 | 4.73e-02 |
| 2 | G1 velocity, soft inlet + LM | 5.34e-06 | 9.01e-06 | -0.188 | 5.24e-06 | 3.70e-03 |
| 2 | G2 stream, all BCs exact + LM | 2.89e-04 | 4.30e-04 | -0.730 | 1.81e-04 | 3.70e-03 |
| 2 | E7 stream, soft inlet (seed 0) | 3.89e-04 | 5.17e-04 | -0.094 | 9.85e-05 | 3.70e-03 |
| 4 | G1 velocity, soft inlet + LM | 7.06e-06 | 1.04e-05 | -0.748 | 7.58e-06 | 3.76e-06 |
| 4 | G2 stream, all BCs exact + LM | 1.55e-04 | 2.75e-04 | +0.702 | 2.61e-04 | 3.76e-06 |
| 4 | E7 stream, soft inlet (seed 0) | 1.66e-04 | 2.38e-04 | -0.658 | 3.86e-04 | 3.76e-06 |
| 10 | G1 velocity, soft inlet + LM | 8.60e-06 | 1.55e-05 | +0.262 | 6.91e-06 | 2.61e-12 |
| 10 | G2 stream, all BCs exact + LM | 2.32e-04 | 4.32e-04 | +0.836 | 7.06e-05 | 2.61e-12 |
| 10 | E7 stream, soft inlet (seed 0) | 1.60e-04 | 2.56e-04 | +0.190 | 7.20e-05 | 2.61e-12 |

## Table 3. u along the channel at each height, against the reference

| Height y* | Run | u at x* = 1, PINN | reference | u at x* = 2, PINN | reference | outlet u, PINN | reference | largest error along x | where (x*) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | G1 velocity, soft inlet + LM | 0.92323 | 0.92323 | 0.99481 | 0.99480 | 0.99999 | 1.00000 | 2.45e-05 | 0.065 |
| 0 | G2 stream, all BCs exact + LM | 0.92315 | 0.92323 | 0.99500 | 0.99480 | 1.00000 | 1.00000 | 2.08e-04 | 8.565 |
| 0 | E7 stream, soft inlet (seed 0) | 0.92305 | 0.92323 | 0.99525 | 0.99480 | 1.00006 | 1.00000 | 7.69e-04 | 0.000 |
| 0.5 | G1 velocity, soft inlet + LM | 0.77606 | 0.77606 | 0.75208 | 0.75208 | 0.74999 | 0.75000 | 2.29e-05 | 0.025 |
| 0.5 | G2 stream, all BCs exact + LM | 0.77617 | 0.77606 | 0.75225 | 0.75208 | 0.75014 | 0.75000 | 3.09e-04 | 5.255 |
| 0.5 | E7 stream, soft inlet (seed 0) | 0.77577 | 0.77606 | 0.75202 | 0.75208 | 0.74997 | 0.75000 | 8.42e-04 | 0.000 |
| 0.9 | G1 velocity, soft inlet + LM | 0.21508 | 0.21508 | 0.19117 | 0.19117 | 0.19000 | 0.19000 | 2.66e-05 | 0.000 |
| 0.9 | G2 stream, all BCs exact + LM | 0.21510 | 0.21508 | 0.19126 | 0.19117 | 0.18963 | 0.19000 | 4.37e-04 | 9.790 |
| 0.9 | E7 stream, soft inlet (seed 0) | 0.21539 | 0.21508 | 0.19093 | 0.19117 | 0.18997 | 0.19000 | 1.42e-03 | 0.000 |
