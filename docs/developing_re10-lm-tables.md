## Table 1. Whole-channel accuracy against the finite-element reference

The reference's own error is about 1e-6 in u (mesh convergence, docs/developing_re10-reference.json), well below every number here.

| Run | u, relative L2, whole channel | u, entrance region x* < 3.77 | u, developed region | largest error in u | v, relative L2 | p, relative L2 | entrance length (reference 1.7794) | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|
| G1 Levenberg-Marquardt | 7.42e-06 | 8.81e-06 | 6.47e-06 | 2.77e-04 | 1.72e-04 | 3.44e-05 | 1.7791 (-0.02%) | 3.08e-08 | 122.8 |
| E1 seed 1 (best so far) | 1.78e-04 | 1.61e-04 | 1.88e-04 | 1.27e-03 | 3.82e-03 | 4.00e-04 | 1.7778 (-0.09%) | 6.01e-06 | 85.7 |
| baseline seed 0 | 2.45e-04 | 2.48e-04 | 2.43e-04 | 1.60e-03 | 6.07e-03 | 4.22e-04 | 1.7678 (-0.65%) | 7.67e-06 | 40.3 |

## Table 2. u across the channel at each station, against the reference

| Station x* | Run | u, relative L2 | largest error in u | where (y*) | v, largest error | v, largest reference value |
|---|---|---|---|---|---|---|
| 0 | G1 Levenberg-Marquardt | 5.61e-05 | 2.95e-04 | +0.994 | 1.62e-04 | 0.00e+00 |
| 0 | E1 seed 1 (best so far) | 5.44e-04 | 1.04e-03 | -0.928 | 4.24e-03 | 0.00e+00 |
| 0 | baseline seed 0 | 6.36e-04 | 1.27e-03 | -0.862 | 4.10e-03 | 0.00e+00 |
| 0.5 | G1 Levenberg-Marquardt | 5.55e-06 | 8.94e-06 | +0.522 | 5.59e-06 | 1.14e-01 |
| 0.5 | E1 seed 1 (best so far) | 9.93e-05 | 1.44e-04 | -0.800 | 7.29e-05 | 1.14e-01 |
| 0.5 | baseline seed 0 | 1.51e-04 | 2.92e-04 | +0.346 | 1.36e-04 | 1.14e-01 |
| 1 | G1 Levenberg-Marquardt | 4.46e-06 | 8.26e-06 | -0.188 | 7.34e-06 | 4.73e-02 |
| 1 | E1 seed 1 (best so far) | 1.23e-04 | 1.75e-04 | +0.838 | 1.87e-04 | 4.73e-02 |
| 1 | baseline seed 0 | 9.17e-05 | 1.31e-04 | +0.754 | 2.72e-04 | 4.73e-02 |
| 2 | G1 Levenberg-Marquardt | 5.34e-06 | 9.01e-06 | -0.188 | 5.24e-06 | 3.70e-03 |
| 2 | E1 seed 1 (best so far) | 1.31e-04 | 1.87e-04 | +0.466 | 1.31e-04 | 3.70e-03 |
| 2 | baseline seed 0 | 2.22e-04 | 3.27e-04 | -0.616 | 1.48e-04 | 3.70e-03 |
| 4 | G1 Levenberg-Marquardt | 7.06e-06 | 1.04e-05 | -0.748 | 7.58e-06 | 3.76e-06 |
| 4 | E1 seed 1 (best so far) | 1.12e-04 | 1.75e-04 | +0.682 | 1.38e-04 | 3.76e-06 |
| 4 | baseline seed 0 | 8.84e-05 | 1.21e-04 | -0.518 | 1.08e-04 | 3.76e-06 |
| 10 | G1 Levenberg-Marquardt | 8.60e-06 | 1.55e-05 | +0.262 | 6.91e-06 | 2.61e-12 |
| 10 | E1 seed 1 (best so far) | 2.46e-04 | 3.54e-04 | +0.100 | 3.31e-04 | 2.61e-12 |
| 10 | baseline seed 0 | 3.38e-04 | 5.71e-04 | -0.444 | 4.45e-04 | 2.61e-12 |

## Table 3. u along the channel at each height, against the reference

| Height y* | Run | u at x* = 1, PINN | reference | u at x* = 2, PINN | reference | outlet u, PINN | reference | largest error along x | where (x*) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | G1 Levenberg-Marquardt | 0.92323 | 0.92323 | 0.99481 | 0.99480 | 0.99999 | 1.00000 | 2.45e-05 | 0.065 |
| 0 | E1 seed 1 (best so far) | 0.92333 | 0.92323 | 0.99464 | 0.99480 | 0.99972 | 1.00000 | 3.16e-04 | 8.550 |
| 0 | baseline seed 0 | 0.92318 | 0.92323 | 0.99493 | 0.99480 | 0.99994 | 1.00000 | 3.87e-04 | 0.055 |
| 0.5 | G1 Levenberg-Marquardt | 0.77606 | 0.77606 | 0.75208 | 0.75208 | 0.74999 | 0.75000 | 2.29e-05 | 0.025 |
| 0.5 | E1 seed 1 (best so far) | 0.77611 | 0.77606 | 0.75226 | 0.75208 | 0.74973 | 0.75000 | 2.71e-04 | 9.995 |
| 0.5 | baseline seed 0 | 0.77610 | 0.77606 | 0.75208 | 0.75208 | 0.74997 | 0.75000 | 4.66e-04 | 0.115 |
| 0.9 | G1 Levenberg-Marquardt | 0.21508 | 0.21508 | 0.19117 | 0.19117 | 0.19000 | 0.19000 | 2.66e-05 | 0.000 |
| 0.9 | E1 seed 1 (best so far) | 0.21522 | 0.21508 | 0.19113 | 0.19117 | 0.18996 | 0.19000 | 1.06e-03 | 0.035 |
| 0.9 | baseline seed 0 | 0.21513 | 0.21508 | 0.19124 | 0.19117 | 0.18989 | 0.19000 | 1.15e-03 | 0.030 |
