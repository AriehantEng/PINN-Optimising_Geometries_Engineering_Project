## Table 1. Whole-channel accuracy against the finite-element reference

The reference's own error is about 1e-6 in u (mesh convergence, docs/developing_re10-reference.json), well below every number here.

| Run | u, relative L2, whole channel | u, entrance region x* < 3.77 | u, developed region | largest error in u | v, relative L2 | p, relative L2 | entrance length (reference 1.7794) | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|
| G4 LM fresh + dense corner inlet grid | 8.37e-06 | 6.40e-06 | 9.34e-06 | 2.23e-04 | 1.67e-04 | 3.02e-05 | 1.7794 (+0.00%) | 1.38e-08 | 229.2 |
| G3 LM fresh points each step | 5.50e-06 | 5.77e-06 | 5.33e-06 | 1.98e-04 | 1.59e-04 | 3.37e-05 | 1.7794 (-0.00%) | 1.35e-08 | 205.6 |
| G1 LM | 7.42e-06 | 8.81e-06 | 6.47e-06 | 2.77e-04 | 1.72e-04 | 3.44e-05 | 1.7791 (-0.02%) | 3.08e-08 | 122.8 |

## Table 2. u across the channel at each station, against the reference

| Station x* | Run | u, relative L2 | largest error in u | where (y*) | v, largest error | v, largest reference value |
|---|---|---|---|---|---|---|
| 0 | G4 LM fresh + dense corner inlet grid | 4.83e-05 | 2.38e-04 | -0.994 | 1.13e-04 | 0.00e+00 |
| 0 | G3 LM fresh points each step | 4.41e-05 | 2.03e-04 | +0.994 | 8.13e-05 | 0.00e+00 |
| 0 | G1 LM | 5.61e-05 | 2.95e-04 | +0.994 | 1.62e-04 | 0.00e+00 |
| 0.5 | G4 LM fresh + dense corner inlet grid | 4.38e-06 | 6.96e-06 | -0.376 | 6.31e-06 | 1.14e-01 |
| 0.5 | G3 LM fresh points each step | 4.76e-06 | 8.20e-06 | +0.522 | 5.56e-06 | 1.14e-01 |
| 0.5 | G1 LM | 5.55e-06 | 8.94e-06 | +0.522 | 5.59e-06 | 1.14e-01 |
| 1 | G4 LM fresh + dense corner inlet grid | 4.44e-06 | 6.60e-06 | -0.124 | 9.24e-06 | 4.73e-02 |
| 1 | G3 LM fresh points each step | 3.01e-06 | 5.89e-06 | +0.406 | 6.93e-06 | 4.73e-02 |
| 1 | G1 LM | 4.46e-06 | 8.26e-06 | -0.188 | 7.34e-06 | 4.73e-02 |
| 2 | G4 LM fresh + dense corner inlet grid | 4.53e-06 | 7.72e-06 | -0.090 | 6.19e-06 | 3.70e-03 |
| 2 | G3 LM fresh points each step | 1.92e-06 | 2.64e-06 | -0.448 | 7.32e-06 | 3.70e-03 |
| 2 | G1 LM | 5.34e-06 | 9.01e-06 | -0.188 | 5.24e-06 | 3.70e-03 |
| 4 | G4 LM fresh + dense corner inlet grid | 3.42e-06 | 5.50e-06 | +0.076 | 6.75e-06 | 3.76e-06 |
| 4 | G3 LM fresh points each step | 2.97e-06 | 4.07e-06 | +0.114 | 9.15e-06 | 3.76e-06 |
| 4 | G1 LM | 7.06e-06 | 1.04e-05 | -0.748 | 7.58e-06 | 3.76e-06 |
| 10 | G4 LM fresh + dense corner inlet grid | 7.75e-06 | 9.72e-06 | -0.586 | 8.32e-06 | 2.61e-12 |
| 10 | G3 LM fresh points each step | 5.86e-06 | 7.88e-06 | +0.312 | 5.68e-06 | 2.61e-12 |
| 10 | G1 LM | 8.60e-06 | 1.55e-05 | +0.262 | 6.91e-06 | 2.61e-12 |

## Table 3. u along the channel at each height, against the reference

| Height y* | Run | u at x* = 1, PINN | reference | u at x* = 2, PINN | reference | outlet u, PINN | reference | largest error along x | where (x*) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | G4 LM fresh + dense corner inlet grid | 0.92323 | 0.92323 | 0.99481 | 0.99480 | 1.00000 | 1.00000 | 2.03e-05 | 0.050 |
| 0 | G3 LM fresh points each step | 0.92323 | 0.92323 | 0.99481 | 0.99480 | 1.00000 | 1.00000 | 1.19e-05 | 0.030 |
| 0 | G1 LM | 0.92323 | 0.92323 | 0.99481 | 0.99480 | 0.99999 | 1.00000 | 2.45e-05 | 0.065 |
| 0.5 | G4 LM fresh + dense corner inlet grid | 0.77606 | 0.77606 | 0.75209 | 0.75208 | 0.75000 | 0.75000 | 1.25e-05 | 0.030 |
| 0.5 | G3 LM fresh points each step | 0.77607 | 0.77606 | 0.75208 | 0.75208 | 0.74999 | 0.75000 | 1.35e-05 | 0.040 |
| 0.5 | G1 LM | 0.77606 | 0.77606 | 0.75208 | 0.75208 | 0.74999 | 0.75000 | 2.29e-05 | 0.025 |
| 0.9 | G4 LM fresh + dense corner inlet grid | 0.21508 | 0.21508 | 0.19117 | 0.19117 | 0.18999 | 0.19000 | 2.65e-05 | 0.000 |
| 0.9 | G3 LM fresh points each step | 0.21508 | 0.21508 | 0.19117 | 0.19117 | 0.19000 | 0.19000 | 2.76e-05 | 0.000 |
| 0.9 | G1 LM | 0.21508 | 0.21508 | 0.19117 | 0.19117 | 0.19000 | 0.19000 | 2.66e-05 | 0.000 |
