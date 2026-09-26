## Table 1. Whole-channel accuracy against the finite-element reference

The reference's own error is about 1e-6 in u (mesh convergence, docs/developing_re10-reference.json), well below every number here.

| Run | u, relative L2, whole channel | u, entrance region x* < 3.77 | u, developed region | largest error in u | v, relative L2 | p, relative L2 | entrance length (reference 1.7794) | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|
| E7 seed 0 | 3.45e-04 | 4.15e-04 | 2.96e-04 | 3.25e-03 | 7.34e-03 | 6.26e-04 | 1.7701 (-0.52%) | 1.95e-05 | 72.3 |
| E7 seed 1 | 3.36e-04 | 4.74e-04 | 2.18e-04 | 4.88e-03 | 7.45e-03 | 9.41e-04 | 1.7721 (-0.41%) | 2.86e-05 | 107.9 |

## Table 2. u across the channel at each station, against the reference

| Station x* | Run | u, relative L2 | largest error in u | where (y*) | v, largest error | v, largest reference value |
|---|---|---|---|---|---|---|
| 0 | E7 seed 0 | 1.95e-03 | 3.25e-03 | -0.826 | 7.60e-03 | 0.00e+00 |
| 0 | E7 seed 1 | 2.61e-03 | 4.29e-03 | +0.808 | 9.39e-03 | 0.00e+00 |
| 0.5 | E7 seed 0 | 3.77e-04 | 5.71e-04 | +0.198 | 1.72e-04 | 1.14e-01 |
| 0.5 | E7 seed 1 | 3.01e-04 | 5.28e-04 | +0.620 | 2.79e-04 | 1.14e-01 |
| 1 | E7 seed 0 | 3.33e-04 | 4.68e-04 | +0.390 | 2.75e-04 | 4.73e-02 |
| 1 | E7 seed 1 | 2.29e-04 | 3.84e-04 | +0.326 | 1.96e-04 | 4.73e-02 |
| 2 | E7 seed 0 | 3.89e-04 | 5.17e-04 | -0.094 | 9.85e-05 | 3.70e-03 |
| 2 | E7 seed 1 | 1.23e-04 | 1.95e-04 | -0.160 | 3.36e-04 | 3.70e-03 |
| 4 | E7 seed 0 | 1.66e-04 | 2.38e-04 | -0.658 | 3.86e-04 | 3.76e-06 |
| 4 | E7 seed 1 | 2.14e-04 | 2.92e-04 | -0.366 | 1.57e-04 | 3.76e-06 |
| 10 | E7 seed 0 | 1.60e-04 | 2.56e-04 | +0.190 | 7.20e-05 | 2.61e-12 |
| 10 | E7 seed 1 | 3.55e-04 | 5.38e-04 | -0.390 | 6.70e-05 | 2.61e-12 |

## Table 3. u along the channel at each height, against the reference

| Height y* | Run | u at x* = 1, PINN | reference | u at x* = 2, PINN | reference | outlet u, PINN | reference | largest error along x | where (x*) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | E7 seed 0 | 0.92305 | 0.92323 | 0.99525 | 0.99480 | 1.00006 | 1.00000 | 7.69e-04 | 0.000 |
| 0 | E7 seed 1 | 0.92322 | 0.92323 | 0.99483 | 0.99480 | 0.99998 | 1.00000 | 6.11e-04 | 0.000 |
| 0.5 | E7 seed 0 | 0.77577 | 0.77606 | 0.75202 | 0.75208 | 0.74997 | 0.75000 | 8.42e-04 | 0.000 |
| 0.5 | E7 seed 1 | 0.77613 | 0.77606 | 0.75207 | 0.75208 | 0.74966 | 0.75000 | 1.91e-03 | 0.000 |
| 0.9 | E7 seed 0 | 0.21539 | 0.21508 | 0.19093 | 0.19117 | 0.18997 | 0.19000 | 1.42e-03 | 0.000 |
| 0.9 | E7 seed 1 | 0.21523 | 0.21508 | 0.19118 | 0.19117 | 0.18984 | 0.19000 | 1.75e-03 | 0.000 |
