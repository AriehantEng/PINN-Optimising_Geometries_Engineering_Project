## Table 1. Whole-channel accuracy against the finite-element reference

The reference's own error is about 1e-6 in u (mesh convergence, docs/developing_re10-reference.json), well below every number here.

| Run | u, relative L2, whole channel | u, entrance region x* < 3.77 | u, developed region | largest error in u | v, relative L2 | p, relative L2 | entrance length (reference 1.7794) | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|
| E8 exponential input map | 1.98e-04 | 2.35e-04 | 1.73e-04 | 1.41e-03 | 3.34e-03 | 3.17e-04 | 1.7843 (+0.28%) | 4.81e-06 | 51.4 |
| baseline seed 0 | 2.45e-04 | 2.48e-04 | 2.43e-04 | 1.60e-03 | 6.07e-03 | 4.22e-04 | 1.7678 (-0.65%) | 7.67e-06 | 40.3 |
| baseline seed 1 | 2.78e-04 | 2.20e-04 | 3.06e-04 | 1.73e-03 | 5.58e-03 | 5.00e-04 | 1.7728 (-0.37%) | 1.28e-05 | 38.8 |
| baseline seed 2 | 3.28e-04 | 2.48e-04 | 3.68e-04 | 1.85e-03 | 5.67e-03 | 6.71e-04 | 1.7845 (+0.29%) | 1.06e-05 | 39.4 |

## Table 2. u across the channel at each station, against the reference

| Station x* | Run | u, relative L2 | largest error in u | where (y*) | v, largest error | v, largest reference value |
|---|---|---|---|---|---|---|
| 0 | E8 exponential input map | 5.97e-04 | 1.11e-03 | -0.862 | 3.01e-03 | 0.00e+00 |
| 0 | baseline seed 0 | 6.36e-04 | 1.27e-03 | -0.862 | 4.10e-03 | 0.00e+00 |
| 0 | baseline seed 1 | 9.24e-04 | 1.46e-03 | -0.810 | 5.06e-03 | 0.00e+00 |
| 0 | baseline seed 2 | 5.55e-04 | 9.95e-04 | +0.842 | 5.03e-03 | 0.00e+00 |
| 0.5 | E8 exponential input map | 1.79e-04 | 2.62e-04 | -0.008 | 1.27e-04 | 1.14e-01 |
| 0.5 | baseline seed 0 | 1.51e-04 | 2.92e-04 | +0.346 | 1.36e-04 | 1.14e-01 |
| 0.5 | baseline seed 1 | 1.30e-04 | 1.96e-04 | +0.640 | 2.86e-04 | 1.14e-01 |
| 0.5 | baseline seed 2 | 1.64e-04 | 2.43e-04 | -0.630 | 1.75e-04 | 1.14e-01 |
| 1 | E8 exponential input map | 2.05e-04 | 2.94e-04 | -0.560 | 9.01e-05 | 4.73e-02 |
| 1 | baseline seed 0 | 9.17e-05 | 1.31e-04 | +0.754 | 2.72e-04 | 4.73e-02 |
| 1 | baseline seed 1 | 1.86e-04 | 2.26e-04 | +0.816 | 2.62e-04 | 4.73e-02 |
| 1 | baseline seed 2 | 2.16e-04 | 2.65e-04 | -0.218 | 9.04e-05 | 4.73e-02 |
| 2 | E8 exponential input map | 1.46e-04 | 1.98e-04 | -0.286 | 6.66e-05 | 3.70e-03 |
| 2 | baseline seed 0 | 2.22e-04 | 3.27e-04 | -0.616 | 1.48e-04 | 3.70e-03 |
| 2 | baseline seed 1 | 1.13e-04 | 2.13e-04 | +0.724 | 9.08e-05 | 3.70e-03 |
| 2 | baseline seed 2 | 2.72e-04 | 4.05e-04 | -0.070 | 1.87e-04 | 3.70e-03 |
| 4 | E8 exponential input map | 1.02e-04 | 1.14e-04 | +0.050 | 5.54e-05 | 3.76e-06 |
| 4 | baseline seed 0 | 8.84e-05 | 1.21e-04 | -0.518 | 1.08e-04 | 3.76e-06 |
| 4 | baseline seed 1 | 1.87e-04 | 3.12e-04 | +0.308 | 1.35e-04 | 3.76e-06 |
| 4 | baseline seed 2 | 3.11e-04 | 3.89e-04 | +0.580 | 2.43e-04 | 3.76e-06 |
| 10 | E8 exponential input map | 1.30e-04 | 1.89e-04 | -0.542 | 1.30e-04 | 2.61e-12 |
| 10 | baseline seed 0 | 3.38e-04 | 5.71e-04 | -0.444 | 4.45e-04 | 2.61e-12 |
| 10 | baseline seed 1 | 2.20e-04 | 3.32e-04 | +0.136 | 4.44e-04 | 2.61e-12 |
| 10 | baseline seed 2 | 4.58e-04 | 5.78e-04 | -0.156 | 3.10e-04 | 2.61e-12 |

## Table 3. u along the channel at each height, against the reference

| Height y* | Run | u at x* = 1, PINN | reference | u at x* = 2, PINN | reference | outlet u, PINN | reference | largest error along x | where (x*) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | E8 exponential input map | 0.92318 | 0.92323 | 0.99468 | 0.99480 | 1.00012 | 1.00000 | 4.00e-04 | 0.145 |
| 0 | baseline seed 0 | 0.92318 | 0.92323 | 0.99493 | 0.99480 | 0.99994 | 1.00000 | 3.87e-04 | 0.055 |
| 0 | baseline seed 1 | 0.92332 | 0.92323 | 0.99483 | 0.99480 | 0.99981 | 1.00000 | 4.28e-04 | 0.000 |
| 0 | baseline seed 2 | 0.92336 | 0.92323 | 0.99519 | 0.99480 | 1.00047 | 1.00000 | 5.23e-04 | 7.815 |
| 0.5 | E8 exponential input map | 0.77610 | 0.77606 | 0.75202 | 0.75208 | 0.75005 | 0.75000 | 4.56e-04 | 0.095 |
| 0.5 | baseline seed 0 | 0.77610 | 0.77606 | 0.75208 | 0.75208 | 0.74997 | 0.75000 | 4.66e-04 | 0.115 |
| 0.5 | baseline seed 1 | 0.77609 | 0.77606 | 0.75209 | 0.75208 | 0.74976 | 0.75000 | 4.96e-04 | 5.340 |
| 0.5 | baseline seed 2 | 0.77617 | 0.77606 | 0.75207 | 0.75208 | 0.75020 | 0.75000 | 6.25e-04 | 9.265 |
| 0.9 | E8 exponential input map | 0.21511 | 0.21508 | 0.19111 | 0.19117 | 0.19001 | 0.19000 | 1.11e-03 | 0.025 |
| 0.9 | baseline seed 0 | 0.21513 | 0.21508 | 0.19124 | 0.19117 | 0.18989 | 0.19000 | 1.15e-03 | 0.030 |
| 0.9 | baseline seed 1 | 0.21525 | 0.21508 | 0.19104 | 0.19117 | 0.19004 | 0.19000 | 1.30e-03 | 0.040 |
| 0.9 | baseline seed 2 | 0.21515 | 0.21508 | 0.19105 | 0.19117 | 0.19006 | 0.19000 | 1.07e-03 | 0.035 |
