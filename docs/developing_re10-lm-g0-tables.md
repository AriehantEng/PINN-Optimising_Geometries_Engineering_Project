## Table 1. Whole-channel accuracy against the finite-element reference

The reference's own error is about 1e-6 in u (mesh convergence, docs/developing_re10-reference.json), well below every number here.

| Run | u, relative L2, whole channel | u, entrance region x* < 3.77 | u, developed region | largest error in u | v, relative L2 | p, relative L2 | entrance length (reference 1.7794) | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|
| G1 width 40 + LM | 7.42e-06 | 8.81e-06 | 6.47e-06 | 2.77e-04 | 1.72e-04 | 3.44e-05 | 1.7791 (-0.02%) | 3.08e-08 | 122.8 |
| G0 width 40, no LM | 5.03e-04 | 3.77e-04 | 5.64e-04 | 2.11e-03 | 1.16e-02 | 1.04e-03 | 1.8018 (+1.26%) | 1.50e-05 | 16.5 |
| baseline width 128, no LM | 2.45e-04 | 2.48e-04 | 2.43e-04 | 1.60e-03 | 6.07e-03 | 4.22e-04 | 1.7678 (-0.65%) | 7.67e-06 | 40.3 |

## Table 2. u across the channel at each station, against the reference

| Station x* | Run | u, relative L2 | largest error in u | where (y*) | v, largest error | v, largest reference value |
|---|---|---|---|---|---|---|
| 0 | G1 width 40 + LM | 5.61e-05 | 2.95e-04 | +0.994 | 1.62e-04 | 0.00e+00 |
| 0 | G0 width 40, no LM | 9.40e-04 | 1.15e-03 | +0.312 | 5.14e-03 | 0.00e+00 |
| 0 | baseline width 128, no LM | 6.36e-04 | 1.27e-03 | -0.862 | 4.10e-03 | 0.00e+00 |
| 0.5 | G1 width 40 + LM | 5.55e-06 | 8.94e-06 | +0.522 | 5.59e-06 | 1.14e-01 |
| 0.5 | G0 width 40, no LM | 2.40e-04 | 3.20e-04 | +0.282 | 2.46e-04 | 1.14e-01 |
| 0.5 | baseline width 128, no LM | 1.51e-04 | 2.92e-04 | +0.346 | 1.36e-04 | 1.14e-01 |
| 1 | G1 width 40 + LM | 4.46e-06 | 8.26e-06 | -0.188 | 7.34e-06 | 4.73e-02 |
| 1 | G0 width 40, no LM | 2.73e-04 | 2.98e-04 | +0.782 | 2.33e-04 | 4.73e-02 |
| 1 | baseline width 128, no LM | 9.17e-05 | 1.31e-04 | +0.754 | 2.72e-04 | 4.73e-02 |
| 2 | G1 width 40 + LM | 5.34e-06 | 9.01e-06 | -0.188 | 5.24e-06 | 3.70e-03 |
| 2 | G0 width 40, no LM | 3.21e-04 | 4.58e-04 | +0.140 | 4.21e-04 | 3.70e-03 |
| 2 | baseline width 128, no LM | 2.22e-04 | 3.27e-04 | -0.616 | 1.48e-04 | 3.70e-03 |
| 4 | G1 width 40 + LM | 7.06e-06 | 1.04e-05 | -0.748 | 7.58e-06 | 3.76e-06 |
| 4 | G0 width 40, no LM | 1.86e-04 | 2.84e-04 | +0.788 | 4.62e-04 | 3.76e-06 |
| 4 | baseline width 128, no LM | 8.84e-05 | 1.21e-04 | -0.518 | 1.08e-04 | 3.76e-06 |
| 10 | G1 width 40 + LM | 8.60e-06 | 1.55e-05 | +0.262 | 6.91e-06 | 2.61e-12 |
| 10 | G0 width 40, no LM | 8.29e-04 | 1.15e-03 | +0.198 | 5.98e-04 | 2.61e-12 |
| 10 | baseline width 128, no LM | 3.38e-04 | 5.71e-04 | -0.444 | 4.45e-04 | 2.61e-12 |

## Table 3. u along the channel at each height, against the reference

| Height y* | Run | u at x* = 1, PINN | reference | u at x* = 2, PINN | reference | outlet u, PINN | reference | largest error along x | where (x*) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | G1 width 40 + LM | 0.92323 | 0.92323 | 0.99481 | 0.99480 | 0.99999 | 1.00000 | 2.45e-05 | 0.065 |
| 0 | G0 width 40, no LM | 0.92341 | 0.92323 | 0.99514 | 0.99480 | 1.00097 | 1.00000 | 9.69e-04 | 9.995 |
| 0 | baseline width 128, no LM | 0.92318 | 0.92323 | 0.99493 | 0.99480 | 0.99994 | 1.00000 | 3.87e-04 | 0.055 |
| 0.5 | G1 width 40 + LM | 0.77606 | 0.77606 | 0.75208 | 0.75208 | 0.74999 | 0.75000 | 2.29e-05 | 0.025 |
| 0.5 | G0 width 40, no LM | 0.77633 | 0.77606 | 0.75237 | 0.75208 | 0.75039 | 0.75000 | 6.04e-04 | 8.000 |
| 0.5 | baseline width 128, no LM | 0.77610 | 0.77606 | 0.75208 | 0.75208 | 0.74997 | 0.75000 | 4.66e-04 | 0.115 |
| 0.9 | G1 width 40 + LM | 0.21508 | 0.21508 | 0.19117 | 0.19117 | 0.19000 | 0.19000 | 2.66e-05 | 0.000 |
| 0.9 | G0 width 40, no LM | 0.21529 | 0.21508 | 0.19108 | 0.19117 | 0.19012 | 0.19000 | 1.12e-03 | 0.035 |
| 0.9 | baseline width 128, no LM | 0.21513 | 0.21508 | 0.19124 | 0.19117 | 0.18989 | 0.19000 | 1.15e-03 | 0.030 |
