## Table 1. Whole-channel accuracy against the finite-element reference

The reference's own error is about 1e-6 in u (mesh convergence, docs/developing_re10-reference.json), well below every number here.

| Run | u, relative L2, whole channel | u, entrance region x* < 3.77 | u, developed region | largest error in u | v, relative L2 | p, relative L2 | entrance length (reference 1.7794) | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|
| G4 | 8.37e-06 | 6.40e-06 | 9.34e-06 | 2.23e-04 | 1.67e-04 | 3.02e-05 | 1.7794 (+0.00%) | 1.38e-08 | 229.2 |
| G3 | 5.50e-06 | 5.77e-06 | 5.33e-06 | 1.98e-04 | 1.59e-04 | 3.37e-05 | 1.7794 (-0.00%) | 1.35e-08 | 205.6 |
| control: G4 + 100 LM steps | 1.19e-05 | 1.29e-05 | 1.12e-05 | 2.10e-04 | 2.13e-04 | 3.46e-05 | 1.7794 (-0.00%) | 1.94e-08 | 26.3 |
| averaged: same iterates, mean of 20 | 7.46e-06 | 7.71e-06 | 7.31e-06 | 2.08e-04 | 1.38e-04 | 3.03e-05 | 1.7793 (-0.01%) | 1.46e-08 | 26.2 |

## Table 2. u across the channel at each station, against the reference

| Station x* | Run | u, relative L2 | largest error in u | where (y*) | v, largest error | v, largest reference value |
|---|---|---|---|---|---|---|
| 0 | G4 | 4.83e-05 | 2.38e-04 | -0.994 | 1.13e-04 | 0.00e+00 |
| 0 | G3 | 4.41e-05 | 2.03e-04 | +0.994 | 8.13e-05 | 0.00e+00 |
| 0 | control: G4 + 100 LM steps | 4.75e-05 | 2.22e-04 | -0.994 | 9.61e-05 | 0.00e+00 |
| 0 | averaged: same iterates, mean of 20 | 4.61e-05 | 2.19e-04 | -0.994 | 9.28e-05 | 0.00e+00 |
| 0.5 | G4 | 4.38e-06 | 6.96e-06 | -0.376 | 6.31e-06 | 1.14e-01 |
| 0.5 | G3 | 4.76e-06 | 8.20e-06 | +0.522 | 5.56e-06 | 1.14e-01 |
| 0.5 | control: G4 + 100 LM steps | 1.33e-05 | 1.49e-05 | -0.346 | 4.13e-06 | 1.14e-01 |
| 0.5 | averaged: same iterates, mean of 20 | 7.10e-06 | 1.01e-05 | -0.376 | 2.97e-06 | 1.14e-01 |
| 1 | G4 | 4.44e-06 | 6.60e-06 | -0.124 | 9.24e-06 | 4.73e-02 |
| 1 | G3 | 3.01e-06 | 5.89e-06 | +0.406 | 6.93e-06 | 4.73e-02 |
| 1 | control: G4 + 100 LM steps | 1.50e-05 | 1.83e-05 | -0.124 | 7.19e-06 | 4.73e-02 |
| 1 | averaged: same iterates, mean of 20 | 6.94e-06 | 1.02e-05 | -0.124 | 3.65e-06 | 4.73e-02 |
| 2 | G4 | 4.53e-06 | 7.72e-06 | -0.090 | 6.19e-06 | 3.70e-03 |
| 2 | G3 | 1.92e-06 | 2.64e-06 | -0.448 | 7.32e-06 | 3.70e-03 |
| 2 | control: G4 + 100 LM steps | 1.22e-05 | 1.59e-05 | -0.124 | 7.64e-06 | 3.70e-03 |
| 2 | averaged: same iterates, mean of 20 | 5.86e-06 | 9.86e-06 | -0.092 | 4.41e-06 | 3.70e-03 |
| 4 | G4 | 3.42e-06 | 5.50e-06 | +0.076 | 6.75e-06 | 3.76e-06 |
| 4 | G3 | 2.97e-06 | 4.07e-06 | +0.114 | 9.15e-06 | 3.76e-06 |
| 4 | control: G4 + 100 LM steps | 5.60e-06 | 8.33e-06 | +0.054 | 7.95e-06 | 3.76e-06 |
| 4 | averaged: same iterates, mean of 20 | 3.29e-06 | 5.52e-06 | +0.064 | 5.72e-06 | 3.76e-06 |
| 10 | G4 | 7.75e-06 | 9.72e-06 | -0.586 | 8.32e-06 | 2.61e-12 |
| 10 | G3 | 5.86e-06 | 7.88e-06 | +0.312 | 5.68e-06 | 2.61e-12 |
| 10 | control: G4 + 100 LM steps | 1.57e-05 | 2.06e-05 | -0.586 | 1.15e-05 | 2.61e-12 |
| 10 | averaged: same iterates, mean of 20 | 6.83e-06 | 8.54e-06 | -0.660 | 5.60e-06 | 2.61e-12 |

## Table 3. u along the channel at each height, against the reference

| Height y* | Run | u at x* = 1, PINN | reference | u at x* = 2, PINN | reference | outlet u, PINN | reference | largest error along x | where (x*) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | G4 | 0.92323 | 0.92323 | 0.99481 | 0.99480 | 1.00000 | 1.00000 | 2.03e-05 | 0.050 |
| 0 | G3 | 0.92323 | 0.92323 | 0.99481 | 0.99480 | 1.00000 | 1.00000 | 1.19e-05 | 0.030 |
| 0 | control: G4 + 100 LM steps | 0.92324 | 0.92323 | 0.99482 | 0.99480 | 1.00001 | 1.00000 | 1.67e-05 | 1.080 |
| 0 | averaged: same iterates, mean of 20 | 0.92324 | 0.92323 | 0.99481 | 0.99480 | 1.00000 | 1.00000 | 1.93e-05 | 0.050 |
| 0.5 | G4 | 0.77606 | 0.77606 | 0.75209 | 0.75208 | 0.75000 | 0.75000 | 1.25e-05 | 0.030 |
| 0.5 | G3 | 0.77607 | 0.77606 | 0.75208 | 0.75208 | 0.74999 | 0.75000 | 1.35e-05 | 0.040 |
| 0.5 | control: G4 + 100 LM steps | 0.77607 | 0.77606 | 0.75209 | 0.75208 | 0.75000 | 0.75000 | 1.37e-05 | 0.025 |
| 0.5 | averaged: same iterates, mean of 20 | 0.77607 | 0.77606 | 0.75209 | 0.75208 | 0.75000 | 0.75000 | 1.64e-05 | 0.025 |
| 0.9 | G4 | 0.21508 | 0.21508 | 0.19117 | 0.19117 | 0.18999 | 0.19000 | 2.65e-05 | 0.000 |
| 0.9 | G3 | 0.21508 | 0.21508 | 0.19117 | 0.19117 | 0.19000 | 0.19000 | 2.76e-05 | 0.000 |
| 0.9 | control: G4 + 100 LM steps | 0.21508 | 0.21508 | 0.19117 | 0.19117 | 0.19000 | 0.19000 | 2.68e-05 | 0.000 |
| 0.9 | averaged: same iterates, mean of 20 | 0.21508 | 0.21508 | 0.19117 | 0.19117 | 0.18999 | 0.19000 | 2.89e-05 | 0.000 |
