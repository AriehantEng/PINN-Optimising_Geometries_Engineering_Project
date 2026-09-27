## Table 1. Whole-channel accuracy against the finite-element reference

The reference's own error is about 1e-6 in u (mesh convergence, docs/developing_re10-reference.json), well below every number here.

| Run | u, relative L2, whole channel | u, entrance region x* < 3.77 | u, developed region | largest error in u | v, relative L2 | p, relative L2 | entrance length (reference 1.7794) | held-out loss | training time (min) |
|---|---|---|---|---|---|---|---|---|---|
| E5b exact inlet, exp ramp, inlet lift | 1.41e-03 | 6.55e-04 | 1.70e-03 | 3.65e-03 | 2.51e-02 | 3.11e-03 | 1.7054 (-4.16%) | 1.85e-04 | 50.7 |
| E5c exact inlet, exp ramp, blended lift | 1.12e-03 | 4.77e-04 | 1.36e-03 | 3.14e-03 | 2.63e-02 | 1.04e-03 | 1.7780 (-0.08%) | 1.55e-04 | 49.1 |
| baseline (soft inlet) | 2.45e-04 | 2.48e-04 | 2.43e-04 | 1.60e-03 | 6.07e-03 | 4.22e-04 | 1.7678 (-0.65%) | 7.67e-06 | 40.3 |

## Table 2. u across the channel at each station, against the reference

| Station x* | Run | u, relative L2 | largest error in u | where (y*) | v, largest error | v, largest reference value |
|---|---|---|---|---|---|---|
| 0 | E5b exact inlet, exp ramp, inlet lift | 5.48e-06 | 1.43e-05 | +0.908 | 0.00e+00 | 0.00e+00 |
| 0 | E5c exact inlet, exp ramp, blended lift | 5.48e-06 | 1.43e-05 | +0.908 | 0.00e+00 | 0.00e+00 |
| 0 | baseline (soft inlet) | 6.36e-04 | 1.27e-03 | -0.862 | 4.10e-03 | 0.00e+00 |
| 0.5 | E5b exact inlet, exp ramp, inlet lift | 6.07e-04 | 8.61e-04 | +0.822 | 9.32e-04 | 1.14e-01 |
| 0.5 | E5c exact inlet, exp ramp, blended lift | 4.34e-04 | 5.04e-04 | -0.066 | 8.49e-04 | 1.14e-01 |
| 0.5 | baseline (soft inlet) | 1.51e-04 | 2.92e-04 | +0.346 | 1.36e-04 | 1.14e-01 |
| 1 | E5b exact inlet, exp ramp, inlet lift | 8.51e-04 | 1.11e-03 | -0.552 | 4.75e-04 | 4.73e-02 |
| 1 | E5c exact inlet, exp ramp, blended lift | 4.47e-04 | 6.36e-04 | +0.676 | 9.92e-04 | 4.73e-02 |
| 1 | baseline (soft inlet) | 9.17e-05 | 1.31e-04 | +0.754 | 2.72e-04 | 4.73e-02 |
| 2 | E5b exact inlet, exp ramp, inlet lift | 8.93e-04 | 1.09e-03 | -0.348 | 4.53e-04 | 3.70e-03 |
| 2 | E5c exact inlet, exp ramp, blended lift | 4.22e-04 | 7.11e-04 | +0.444 | 4.93e-04 | 3.70e-03 |
| 2 | baseline (soft inlet) | 2.22e-04 | 3.27e-04 | -0.616 | 1.48e-04 | 3.70e-03 |
| 4 | E5b exact inlet, exp ramp, inlet lift | 7.95e-04 | 1.16e-03 | +0.340 | 7.39e-04 | 3.76e-06 |
| 4 | E5c exact inlet, exp ramp, blended lift | 5.89e-04 | 9.37e-04 | -0.684 | 4.79e-04 | 3.76e-06 |
| 4 | baseline (soft inlet) | 8.84e-05 | 1.21e-04 | -0.518 | 1.08e-04 | 3.76e-06 |
| 10 | E5b exact inlet, exp ramp, inlet lift | 2.90e-03 | 3.65e-03 | +0.066 | 1.64e-03 | 2.61e-12 |
| 10 | E5c exact inlet, exp ramp, blended lift | 1.28e-03 | 1.43e-03 | -0.444 | 1.29e-03 | 2.61e-12 |
| 10 | baseline (soft inlet) | 3.38e-04 | 5.71e-04 | -0.444 | 4.45e-04 | 2.61e-12 |

## Table 3. u along the channel at each height, against the reference

| Height y* | Run | u at x* = 1, PINN | reference | u at x* = 2, PINN | reference | outlet u, PINN | reference | largest error along x | where (x*) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | E5b exact inlet, exp ramp, inlet lift | 0.92384 | 0.92323 | 0.99471 | 0.99480 | 0.99652 | 1.00000 | 3.48e-03 | 10.000 |
| 0 | E5c exact inlet, exp ramp, blended lift | 0.92361 | 0.92323 | 0.99497 | 0.99480 | 0.99949 | 1.00000 | 2.55e-03 | 6.530 |
| 0 | baseline (soft inlet) | 0.92318 | 0.92323 | 0.99493 | 0.99480 | 0.99994 | 1.00000 | 3.87e-04 | 0.055 |
| 0.5 | E5b exact inlet, exp ramp, inlet lift | 0.77624 | 0.77606 | 0.75115 | 0.75208 | 0.74752 | 0.75000 | 2.52e-03 | 9.690 |
| 0.5 | E5c exact inlet, exp ramp, blended lift | 0.77586 | 0.77606 | 0.75148 | 0.75208 | 0.74910 | 0.75000 | 1.21e-03 | 7.570 |
| 0.5 | baseline (soft inlet) | 0.77610 | 0.77606 | 0.75208 | 0.75208 | 0.74997 | 0.75000 | 4.66e-04 | 0.115 |
| 0.9 | E5b exact inlet, exp ramp, inlet lift | 0.21576 | 0.21508 | 0.19050 | 0.19117 | 0.19044 | 0.19000 | 1.44e-03 | 8.695 |
| 0.9 | E5c exact inlet, exp ramp, blended lift | 0.21503 | 0.21508 | 0.19091 | 0.19117 | 0.18969 | 0.19000 | 8.74e-04 | 0.105 |
| 0.9 | baseline (soft inlet) | 0.21513 | 0.21508 | 0.19124 | 0.19117 | 0.18989 | 0.19000 | 1.15e-03 | 0.030 |
