## Table 1. Whole-channel accuracy against the finite-element reference

The reference's own error is about 1e-6 in u (mesh convergence, docs/developing_re10-reference.json), well below every number here.

| Run | u, relative L2, whole channel | u, entrance region x* < 3.77 | u, developed region | largest error in u | v, relative L2 | p, relative L2 | entrance length (reference 1.7794) | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|
| E6b SiLU, init gain 2 | 7.42e-04 | 2.54e-04 | 9.15e-04 | 2.33e-03 | 9.42e-03 | 1.08e-03 | 1.7775 (-0.11%) | 1.20e-05 | 55.6 |
| E6 SiLU, init gain 1 | 8.30e-04 | 3.53e-04 | 1.01e-03 | 3.44e-03 | 9.99e-03 | 1.13e-03 | 1.7779 (-0.08%) | 2.10e-05 | 41.4 |
| baseline tanh | 2.45e-04 | 2.48e-04 | 2.43e-04 | 1.60e-03 | 6.07e-03 | 4.22e-04 | 1.7678 (-0.65%) | 7.67e-06 | 40.3 |

## Table 2. u across the channel at each station, against the reference

| Station x* | Run | u, relative L2 | largest error in u | where (y*) | v, largest error | v, largest reference value |
|---|---|---|---|---|---|---|
| 0 | E6b SiLU, init gain 2 | 8.94e-04 | 1.60e-03 | +0.844 | 5.87e-03 | 0.00e+00 |
| 0 | E6 SiLU, init gain 1 | 1.04e-03 | 2.09e-03 | -0.976 | 6.47e-03 | 0.00e+00 |
| 0 | baseline tanh | 6.36e-04 | 1.27e-03 | -0.862 | 4.10e-03 | 0.00e+00 |
| 0.5 | E6b SiLU, init gain 2 | 2.89e-04 | 3.80e-04 | -0.298 | 1.82e-04 | 1.14e-01 |
| 0.5 | E6 SiLU, init gain 1 | 2.23e-04 | 3.73e-04 | +0.640 | 2.53e-04 | 1.14e-01 |
| 0.5 | baseline tanh | 1.51e-04 | 2.92e-04 | +0.346 | 1.36e-04 | 1.14e-01 |
| 1 | E6b SiLU, init gain 2 | 1.52e-04 | 2.67e-04 | -0.660 | 3.25e-04 | 4.73e-02 |
| 1 | E6 SiLU, init gain 1 | 2.13e-04 | 3.04e-04 | -0.158 | 3.41e-04 | 4.73e-02 |
| 1 | baseline tanh | 9.17e-05 | 1.31e-04 | +0.754 | 2.72e-04 | 4.73e-02 |
| 2 | E6b SiLU, init gain 2 | 1.76e-04 | 2.89e-04 | -0.054 | 3.56e-04 | 3.70e-03 |
| 2 | E6 SiLU, init gain 1 | 2.46e-04 | 3.55e-04 | -0.466 | 2.74e-04 | 3.70e-03 |
| 2 | baseline tanh | 2.22e-04 | 3.27e-04 | -0.616 | 1.48e-04 | 3.70e-03 |
| 4 | E6b SiLU, init gain 2 | 1.90e-04 | 2.47e-04 | +0.426 | 4.80e-04 | 3.76e-06 |
| 4 | E6 SiLU, init gain 1 | 3.97e-04 | 6.22e-04 | +0.410 | 3.78e-04 | 3.76e-06 |
| 4 | baseline tanh | 8.84e-05 | 1.21e-04 | -0.518 | 1.08e-04 | 3.76e-06 |
| 10 | E6b SiLU, init gain 2 | 2.50e-04 | 3.42e-04 | +0.676 | 3.58e-04 | 2.61e-12 |
| 10 | E6 SiLU, init gain 1 | 4.88e-04 | 5.08e-04 | +0.362 | 6.26e-04 | 2.61e-12 |
| 10 | baseline tanh | 3.38e-04 | 5.71e-04 | -0.444 | 4.45e-04 | 2.61e-12 |

## Table 3. u along the channel at each height, against the reference

| Height y* | Run | u at x* = 1, PINN | reference | u at x* = 2, PINN | reference | outlet u, PINN | reference | largest error along x | where (x*) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | E6b SiLU, init gain 2 | 0.92325 | 0.92323 | 0.99507 | 0.99480 | 1.00022 | 1.00000 | 1.55e-03 | 6.780 |
| 0 | E6 SiLU, init gain 1 | 0.92340 | 0.92323 | 0.99481 | 0.99480 | 0.99951 | 1.00000 | 2.06e-03 | 7.165 |
| 0 | baseline tanh | 0.92318 | 0.92323 | 0.99493 | 0.99480 | 0.99994 | 1.00000 | 3.87e-04 | 0.055 |
| 0.5 | E6b SiLU, init gain 2 | 0.77601 | 0.77606 | 0.75211 | 0.75208 | 0.74978 | 0.75000 | 1.47e-03 | 7.270 |
| 0.5 | E6 SiLU, init gain 1 | 0.77597 | 0.77606 | 0.75213 | 0.75208 | 0.74960 | 0.75000 | 1.13e-03 | 7.235 |
| 0.5 | baseline tanh | 0.77610 | 0.77606 | 0.75208 | 0.75208 | 0.74997 | 0.75000 | 4.66e-04 | 0.115 |
| 0.9 | E6b SiLU, init gain 2 | 0.21502 | 0.21508 | 0.19126 | 0.19117 | 0.18986 | 0.19000 | 1.38e-03 | 0.035 |
| 0.9 | E6 SiLU, init gain 1 | 0.21504 | 0.21508 | 0.19134 | 0.19117 | 0.19013 | 0.19000 | 8.76e-04 | 0.040 |
| 0.9 | baseline tanh | 0.21513 | 0.21508 | 0.19124 | 0.19117 | 0.18989 | 0.19000 | 1.15e-03 | 0.030 |
