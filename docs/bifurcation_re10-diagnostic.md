## Where the parametric PINN's error lives

Against the finite-element solution, every integral on the reference's mesh. Annuli by distance r to the nearest singular corner (B, C, the septum tip).

### Training

| run | parameters | minutes | selected | held-out loss | final held-out / train |
|---|---|---|---|---|---|
| original | 5487 | 163 | lm:62 | 1.478e-02 | 2.85 |
| corner modes | 5507 | 176 | lm_average:147 | 9.154e-04 | 1.63 |

### Loss terms and their gradients at the trained parameters

On the held-out design nearest each split. Ratio: gradient norm of the interior (momentum) term over that of the boundary terms together.

| run | split | interior loss | inlet loss | outlet 1 loss | outlet 2 loss | interior / boundary gradient |
|---|---|---|---|---|---|---|
| original | 0.3521 | 1.17e-01 | 1.58e-05 | 6.62e-04 | 1.43e-05 | 6.47 |
| original | 0.5031 | 1.16e-02 | 5.64e-06 | 2.07e-05 | 2.21e-06 | 3.86 |
| original | 0.6451 | 2.49e-02 | 1.27e-05 | 2.03e-04 | 1.73e-05 | 2.97 |
| corner modes | 0.3521 | 2.39e-03 | 4.57e-06 | 1.40e-05 | 6.16e-05 | 1.26 |
| corner modes | 0.5031 | 5.91e-04 | 4.29e-06 | 1.35e-06 | 6.18e-05 | 0.44 |
| corner modes | 0.6451 | 2.64e-03 | 2.05e-06 | 5.86e-06 | 8.66e-06 | 3.94 |

### Split 0.35

| run | u | v | p | drop error | rank corr. residual-error |
|---|---|---|---|---|---|
| original | 1.91e-02 | 2.36e-02 | 1.99e-02 | +0.206% | 0.75 |
| corner modes | 2.73e-03 | 3.34e-03 | 1.17e-02 | -0.578% | 0.44 |

| r | area | original: vel. err. share, local rel. err., residual RMS | corner modes: vel. err. share, local rel. err., residual RMS |
|---|---|---|---|
| 0–0.02 | 0.02% | 0.7%, 6.99e-01, 1.70e+02 | 0.1%, 2.77e-02, 8.21e+00 |
| 0.02–0.05 | 0.08% | 3.6%, 3.66e-01, 3.08e+01 | 0.5%, 1.87e-02, 1.72e+00 |
| 0.05–0.1 | 0.29% | 7.3%, 1.84e-01, 6.06e+00 | 1.5%, 1.18e-02, 6.05e-01 |
| 0.1–0.2 | 1.16% | 12.7%, 8.72e-02, 1.23e+00 | 5.1%, 7.87e-03, 1.87e-01 |
| 0.2–0.5 | 8.16% | 27.1%, 3.75e-02, 2.21e-01 | 19.6%, 4.53e-03, 6.55e-02 |
| 0.5–1 | 23.45% | 32.0%, 2.39e-02, 9.16e-02 | 45.7%, 4.06e-03, 4.43e-02 |
| > 1 | 66.84% | 16.6%, 1.03e-02, 4.29e-02 | 27.5%, 1.89e-03, 1.77e-02 |

Drop error by opening (each term divided by the inflow; they sum to J_pinn − J_fe):

| run | inlet pressure | inlet kinetic | outlet1 pressure | outlet1 kinetic | outlet2 pressure | outlet2 kinetic |
|---|---|---|---|---|---|---|
| original | +5.19e-02 | +1.95e-03 | -3.91e-02 | -3.26e-03 | -8.48e-04 | +6.25e-04 |
| corner modes | -3.16e-02 | +6.39e-05 | -3.91e-03 | -2.48e-04 | +4.17e-03 | -9.72e-05 |

### Split 0.5

| run | u | v | p | drop error | rank corr. residual-error |
|---|---|---|---|---|---|
| original | 1.29e-02 | 1.79e-02 | 4.55e-03 | +0.111% | 0.81 |
| corner modes | 1.59e-03 | 1.43e-03 | 3.85e-03 | -0.076% | 0.17 |

| r | area | original: vel. err. share, local rel. err., residual RMS | corner modes: vel. err. share, local rel. err., residual RMS |
|---|---|---|---|
| 0–0.02 | 0.02% | 0.9%, 6.59e-01, 1.76e+02 | 0.0%, 1.52e-02, 6.06e+00 |
| 0.02–0.05 | 0.09% | 4.5%, 3.64e-01, 2.68e+01 | 0.2%, 7.68e-03, 1.47e+00 |
| 0.05–0.1 | 0.31% | 9.0%, 1.82e-01, 4.17e+00 | 0.3%, 3.17e-03, 4.62e-01 |
| 0.1–0.2 | 1.22% | 14.9%, 8.23e-02, 6.87e-01 | 0.5%, 1.49e-03, 1.32e-01 |
| 0.2–0.5 | 8.55% | 25.8%, 2.92e-02, 8.54e-02 | 7.9%, 1.66e-03, 3.74e-02 |
| 0.5–1 | 23.16% | 42.2%, 2.04e-02, 3.96e-02 | 15.8%, 1.29e-03, 2.73e-02 |
| > 1 | 66.66% | 2.7%, 2.94e-03, 1.64e-02 | 75.3%, 1.59e-03, 1.10e-02 |

Drop error by opening (each term divided by the inflow; they sum to J_pinn − J_fe):

| run | inlet pressure | inlet kinetic | outlet1 pressure | outlet1 kinetic | outlet2 pressure | outlet2 kinetic |
|---|---|---|---|---|---|---|
| original | -1.89e-03 | -3.55e-04 | +6.85e-03 | -4.28e-04 | -2.41e-04 | +5.47e-04 |
| corner modes | -1.42e-02 | -9.49e-05 | +7.87e-03 | -4.74e-04 | +4.12e-03 | -3.30e-04 |

### Split 0.65

| run | u | v | p | drop error | rank corr. residual-error |
|---|---|---|---|---|---|
| original | 1.79e-02 | 2.05e-02 | 2.68e-02 | -2.052% | 0.71 |
| corner modes | 4.01e-03 | 3.90e-03 | 1.21e-02 | -1.149% | 0.43 |

| r | area | original: vel. err. share, local rel. err., residual RMS | corner modes: vel. err. share, local rel. err., residual RMS |
|---|---|---|---|
| 0–0.02 | 0.02% | 1.0%, 7.24e-01, 1.52e+02 | 0.1%, 5.46e-02, 1.93e+01 |
| 0.02–0.05 | 0.09% | 5.6%, 4.35e-01, 3.39e+01 | 1.1%, 3.99e-02, 4.62e+00 |
| 0.05–0.1 | 0.31% | 11.6%, 2.27e-01, 5.51e+00 | 2.8%, 2.30e-02, 1.22e+00 |
| 0.1–0.2 | 1.26% | 16.8%, 9.93e-02, 8.81e-01 | 7.2%, 1.35e-02, 2.69e-01 |
| 0.2–0.5 | 8.81% | 36.1%, 4.11e-02, 1.73e-01 | 35.7%, 8.47e-03, 8.18e-02 |
| 0.5–1 | 19.13% | 22.3%, 2.08e-02, 9.42e-02 | 33.9%, 5.31e-03, 5.04e-02 |
| > 1 | 70.38% | 6.7%, 5.79e-03, 4.80e-02 | 19.2%, 2.03e-03, 1.90e-02 |

Drop error by opening (each term divided by the inflow; they sum to J_pinn − J_fe):

| run | inlet pressure | inlet kinetic | outlet1 pressure | outlet1 kinetic | outlet2 pressure | outlet2 kinetic |
|---|---|---|---|---|---|---|
| original | -2.71e-01 | +8.61e-04 | +1.35e-01 | +8.62e-04 | +1.70e-03 | -1.54e-03 |
| corner modes | -1.36e-01 | +3.03e-04 | +6.29e-02 | -3.09e-04 | -1.52e-03 | -5.14e-04 |
