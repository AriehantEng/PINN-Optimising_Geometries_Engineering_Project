## Table 1. Whole-channel accuracy against the finite-element reference

The reference's own error is about 1e-6 in u (mesh convergence, docs/developing_re10-reference.json), well below every number here.

| Run | u, relative L2, whole channel | u, entrance region x* < 3.77 | u, developed region | largest error in u | v, relative L2 | p, relative L2 | entrance length (reference 1.7794) | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|
| E9 stream, all BCs exact, blended lift | 1.76e-03 | 1.05e-03 | 2.06e-03 | 4.63e-03 | 4.01e-02 | 3.58e-03 | 1.7971 (+0.99%) | 2.41e-03 | 103.9 |
| E7 stream, soft inlet | 3.45e-04 | 4.15e-04 | 2.96e-04 | 3.25e-03 | 7.34e-03 | 6.26e-04 | 1.7701 (-0.52%) | 1.95e-05 | 72.3 |
| baseline | 2.45e-04 | 2.48e-04 | 2.43e-04 | 1.60e-03 | 6.07e-03 | 4.22e-04 | 1.7678 (-0.65%) | 7.67e-06 | 40.3 |

## Table 2. u across the channel at each station, against the reference

| Station x* | Run | u, relative L2 | largest error in u | where (y*) | v, largest error | v, largest reference value |
|---|---|---|---|---|---|---|
| 0 | E9 stream, all BCs exact, blended lift | 5.48e-06 | 1.43e-05 | +0.908 | 0.00e+00 | 0.00e+00 |
| 0 | E7 stream, soft inlet | 1.95e-03 | 3.25e-03 | -0.826 | 7.60e-03 | 0.00e+00 |
| 0 | baseline | 6.36e-04 | 1.27e-03 | -0.862 | 4.10e-03 | 0.00e+00 |
| 0.5 | E9 stream, all BCs exact, blended lift | 7.64e-04 | 1.05e-03 | +0.146 | 7.46e-04 | 1.14e-01 |
| 0.5 | E7 stream, soft inlet | 3.77e-04 | 5.71e-04 | +0.198 | 1.72e-04 | 1.14e-01 |
| 0.5 | baseline | 1.51e-04 | 2.92e-04 | +0.346 | 1.36e-04 | 1.14e-01 |
| 1 | E9 stream, all BCs exact, blended lift | 1.20e-03 | 1.84e-03 | +0.428 | 2.18e-03 | 4.73e-02 |
| 1 | E7 stream, soft inlet | 3.33e-04 | 4.68e-04 | +0.390 | 2.75e-04 | 4.73e-02 |
| 1 | baseline | 9.17e-05 | 1.31e-04 | +0.754 | 2.72e-04 | 4.73e-02 |
| 2 | E9 stream, all BCs exact, blended lift | 5.13e-04 | 7.38e-04 | -0.468 | 1.44e-03 | 3.70e-03 |
| 2 | E7 stream, soft inlet | 3.89e-04 | 5.17e-04 | -0.094 | 9.85e-05 | 3.70e-03 |
| 2 | baseline | 2.22e-04 | 3.27e-04 | -0.616 | 1.48e-04 | 3.70e-03 |
| 4 | E9 stream, all BCs exact, blended lift | 1.70e-03 | 2.27e-03 | +0.160 | 6.68e-04 | 3.76e-06 |
| 4 | E7 stream, soft inlet | 1.66e-04 | 2.38e-04 | -0.658 | 3.86e-04 | 3.76e-06 |
| 4 | baseline | 8.84e-05 | 1.21e-04 | -0.518 | 1.08e-04 | 3.76e-06 |
| 10 | E9 stream, all BCs exact, blended lift | 1.34e-03 | 2.40e-03 | -0.244 | 4.32e-04 | 2.61e-12 |
| 10 | E7 stream, soft inlet | 1.60e-04 | 2.56e-04 | +0.190 | 7.20e-05 | 2.61e-12 |
| 10 | baseline | 3.38e-04 | 5.71e-04 | -0.444 | 4.45e-04 | 2.61e-12 |

## Table 3. u along the channel at each height, against the reference

| Height y* | Run | u at x* = 1, PINN | reference | u at x* = 2, PINN | reference | outlet u, PINN | reference | largest error along x | where (x*) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | E9 stream, all BCs exact, blended lift | 0.92305 | 0.92323 | 0.99424 | 0.99480 | 0.99920 | 1.00000 | 4.24e-03 | 8.905 |
| 0 | E7 stream, soft inlet | 0.92305 | 0.92323 | 0.99525 | 0.99480 | 1.00006 | 1.00000 | 7.69e-04 | 0.000 |
| 0 | baseline | 0.92318 | 0.92323 | 0.99493 | 0.99480 | 0.99994 | 1.00000 | 3.87e-04 | 0.055 |
| 0.5 | E9 stream, all BCs exact, blended lift | 0.77444 | 0.77606 | 0.75270 | 0.75208 | 0.75102 | 0.75000 | 3.02e-03 | 7.310 |
| 0.5 | E7 stream, soft inlet | 0.77577 | 0.77606 | 0.75202 | 0.75208 | 0.74997 | 0.75000 | 8.42e-04 | 0.000 |
| 0.5 | baseline | 0.77610 | 0.77606 | 0.75208 | 0.75208 | 0.74997 | 0.75000 | 4.66e-04 | 0.115 |
| 0.9 | E9 stream, all BCs exact, blended lift | 0.21502 | 0.21508 | 0.19110 | 0.19117 | 0.18950 | 0.19000 | 1.50e-03 | 8.015 |
| 0.9 | E7 stream, soft inlet | 0.21539 | 0.21508 | 0.19093 | 0.19117 | 0.18997 | 0.19000 | 1.42e-03 | 0.000 |
| 0.9 | baseline | 0.21513 | 0.21508 | 0.19124 | 0.19117 | 0.18989 | 0.19000 | 1.15e-03 | 0.030 |
