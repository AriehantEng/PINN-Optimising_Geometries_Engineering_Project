## Table 1. Whole-channel accuracy against the finite-element reference

The reference's own error is about 1e-6 in u (mesh convergence, docs/developing_re10-reference.json), well below every number here.

| Run | u, relative L2, whole channel | u, entrance region x* < 3.77 | u, developed region | largest error in u | v, relative L2 | p, relative L2 | entrance length (reference 1.7794) | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|
| E3d fixed dense inlet grid, weight 1 | 3.27e-04 | 2.10e-04 | 3.79e-04 | 1.30e-03 | 6.16e-03 | 4.29e-04 | 1.7878 (+0.47%) | 8.43e-06 | 52.2 |
| E3c same grid, weight 10 | 3.59e-04 | 2.71e-04 | 4.02e-04 | 7.95e-04 | 7.79e-03 | 5.94e-04 | 1.7868 (+0.41%) | 1.96e-05 | 39.4 |
| baseline seed 0 | 2.45e-04 | 2.48e-04 | 2.43e-04 | 1.60e-03 | 6.07e-03 | 4.22e-04 | 1.7678 (-0.65%) | 7.67e-06 | 40.3 |
| baseline seed 1 | 2.78e-04 | 2.20e-04 | 3.06e-04 | 1.73e-03 | 5.58e-03 | 5.00e-04 | 1.7728 (-0.37%) | 1.28e-05 | 38.8 |
| baseline seed 2 | 3.28e-04 | 2.48e-04 | 3.68e-04 | 1.85e-03 | 5.67e-03 | 6.71e-04 | 1.7845 (+0.29%) | 1.06e-05 | 39.4 |

## Table 2. u across the channel at each station, against the reference

| Station x* | Run | u, relative L2 | largest error in u | where (y*) | v, largest error | v, largest reference value |
|---|---|---|---|---|---|---|
| 0 | E3d fixed dense inlet grid, weight 1 | 7.64e-04 | 1.00e-03 | -0.844 | 4.39e-03 | 0.00e+00 |
| 0 | E3c same grid, weight 10 | 2.57e-04 | 5.79e-04 | -0.878 | 1.80e-03 | 0.00e+00 |
| 0 | baseline seed 0 | 6.36e-04 | 1.27e-03 | -0.862 | 4.10e-03 | 0.00e+00 |
| 0 | baseline seed 1 | 9.24e-04 | 1.46e-03 | -0.810 | 5.06e-03 | 0.00e+00 |
| 0 | baseline seed 2 | 5.55e-04 | 9.95e-04 | +0.842 | 5.03e-03 | 0.00e+00 |
| 0.5 | E3d fixed dense inlet grid, weight 1 | 1.49e-04 | 2.64e-04 | +0.286 | 1.92e-04 | 1.14e-01 |
| 0.5 | E3c same grid, weight 10 | 2.03e-04 | 2.42e-04 | +0.666 | 6.36e-04 | 1.14e-01 |
| 0.5 | baseline seed 0 | 1.51e-04 | 2.92e-04 | +0.346 | 1.36e-04 | 1.14e-01 |
| 0.5 | baseline seed 1 | 1.30e-04 | 1.96e-04 | +0.640 | 2.86e-04 | 1.14e-01 |
| 0.5 | baseline seed 2 | 1.64e-04 | 2.43e-04 | -0.630 | 1.75e-04 | 1.14e-01 |
| 1 | E3d fixed dense inlet grid, weight 1 | 1.62e-04 | 1.93e-04 | +0.118 | 2.44e-04 | 4.73e-02 |
| 1 | E3c same grid, weight 10 | 3.28e-04 | 4.53e-04 | +0.496 | 2.98e-04 | 4.73e-02 |
| 1 | baseline seed 0 | 9.17e-05 | 1.31e-04 | +0.754 | 2.72e-04 | 4.73e-02 |
| 1 | baseline seed 1 | 1.86e-04 | 2.26e-04 | +0.816 | 2.62e-04 | 4.73e-02 |
| 1 | baseline seed 2 | 2.16e-04 | 2.65e-04 | -0.218 | 9.04e-05 | 4.73e-02 |
| 2 | E3d fixed dense inlet grid, weight 1 | 1.23e-04 | 1.76e-04 | -0.390 | 2.60e-04 | 3.70e-03 |
| 2 | E3c same grid, weight 10 | 3.92e-04 | 5.78e-04 | +0.494 | 3.99e-04 | 3.70e-03 |
| 2 | baseline seed 0 | 2.22e-04 | 3.27e-04 | -0.616 | 1.48e-04 | 3.70e-03 |
| 2 | baseline seed 1 | 1.13e-04 | 2.13e-04 | +0.724 | 9.08e-05 | 3.70e-03 |
| 2 | baseline seed 2 | 2.72e-04 | 4.05e-04 | -0.070 | 1.87e-04 | 3.70e-03 |
| 4 | E3d fixed dense inlet grid, weight 1 | 2.85e-04 | 3.27e-04 | +0.638 | 3.34e-04 | 3.76e-06 |
| 4 | E3c same grid, weight 10 | 2.95e-04 | 3.65e-04 | +0.534 | 2.44e-04 | 3.76e-06 |
| 4 | baseline seed 0 | 8.84e-05 | 1.21e-04 | -0.518 | 1.08e-04 | 3.76e-06 |
| 4 | baseline seed 1 | 1.87e-04 | 3.12e-04 | +0.308 | 1.35e-04 | 3.76e-06 |
| 4 | baseline seed 2 | 3.11e-04 | 3.89e-04 | +0.580 | 2.43e-04 | 3.76e-06 |
| 10 | E3d fixed dense inlet grid, weight 1 | 3.45e-04 | 5.32e-04 | -0.598 | 2.54e-04 | 2.61e-12 |
| 10 | E3c same grid, weight 10 | 4.22e-04 | 6.62e-04 | -0.806 | 6.31e-04 | 2.61e-12 |
| 10 | baseline seed 0 | 3.38e-04 | 5.71e-04 | -0.444 | 4.45e-04 | 2.61e-12 |
| 10 | baseline seed 1 | 2.20e-04 | 3.32e-04 | +0.136 | 4.44e-04 | 2.61e-12 |
| 10 | baseline seed 2 | 4.58e-04 | 5.78e-04 | -0.156 | 3.10e-04 | 2.61e-12 |

## Table 3. u along the channel at each height, against the reference

| Height y* | Run | u at x* = 1, PINN | reference | u at x* = 2, PINN | reference | outlet u, PINN | reference | largest error along x | where (x*) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | E3d fixed dense inlet grid, weight 1 | 0.92337 | 0.92323 | 0.99489 | 0.99480 | 1.00039 | 1.00000 | 4.30e-04 | 6.265 |
| 0 | E3c same grid, weight 10 | 0.92340 | 0.92323 | 0.99476 | 0.99480 | 0.99976 | 1.00000 | 6.38e-04 | 9.085 |
| 0 | baseline seed 0 | 0.92318 | 0.92323 | 0.99493 | 0.99480 | 0.99994 | 1.00000 | 3.87e-04 | 0.055 |
| 0 | baseline seed 1 | 0.92332 | 0.92323 | 0.99483 | 0.99480 | 0.99981 | 1.00000 | 4.28e-04 | 0.000 |
| 0 | baseline seed 2 | 0.92336 | 0.92323 | 0.99519 | 0.99480 | 1.00047 | 1.00000 | 5.23e-04 | 7.815 |
| 0.5 | E3d fixed dense inlet grid, weight 1 | 0.77622 | 0.77606 | 0.75221 | 0.75208 | 0.74998 | 0.75000 | 8.91e-04 | 7.315 |
| 0.5 | E3c same grid, weight 10 | 0.77651 | 0.77606 | 0.75151 | 0.75208 | 0.74971 | 0.75000 | 5.79e-04 | 2.020 |
| 0.5 | baseline seed 0 | 0.77610 | 0.77606 | 0.75208 | 0.75208 | 0.74997 | 0.75000 | 4.66e-04 | 0.115 |
| 0.5 | baseline seed 1 | 0.77609 | 0.77606 | 0.75209 | 0.75208 | 0.74976 | 0.75000 | 4.96e-04 | 5.340 |
| 0.5 | baseline seed 2 | 0.77617 | 0.77606 | 0.75207 | 0.75208 | 0.75020 | 0.75000 | 6.25e-04 | 9.265 |
| 0.9 | E3d fixed dense inlet grid, weight 1 | 0.21511 | 0.21508 | 0.19111 | 0.19117 | 0.19008 | 0.19000 | 8.80e-04 | 0.035 |
| 0.9 | E3c same grid, weight 10 | 0.21533 | 0.21508 | 0.19098 | 0.19117 | 0.19011 | 0.19000 | 4.26e-04 | 0.040 |
| 0.9 | baseline seed 0 | 0.21513 | 0.21508 | 0.19124 | 0.19117 | 0.18989 | 0.19000 | 1.15e-03 | 0.030 |
| 0.9 | baseline seed 1 | 0.21525 | 0.21508 | 0.19104 | 0.19117 | 0.19004 | 0.19000 | 1.30e-03 | 0.040 |
| 0.9 | baseline seed 2 | 0.21515 | 0.21508 | 0.19105 | 0.19117 | 0.19006 | 0.19000 | 1.07e-03 | 0.035 |
