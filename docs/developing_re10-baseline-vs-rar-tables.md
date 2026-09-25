## Table 0. The runs being compared

| Run | Adaptive refinement (RAR) | Interior points per draw | Training time (min) | Loss on held-out points | Held-out loss / training loss, last check |
|---|---|---|---|---|---|
| baseline | off | 4,096 | 40.3 | 7.667e-06 | 1.04 |
| with RAR | 1024 worst of 32768 | 5,120 | 47.0 | 9.951e-06 | 1.04 |

## Table 1. u against y: velocity across the channel at each station

| Station x* | Run | Is there an exact solution here? | Relative L2 error of u vs exact | Largest pointwise error in u | Flux error vs 4/3 (%) | Symmetry error, max abs u(y) - u(-y) (exact 0) | Largest abs v (developed: 0) |
|---|---|---|---|---|---|---|---|
| 0 | baseline | yes: prescribed inlet U0(1 - y^16) | 6.36e-04 | 1.25e-03 | -0.0010 | 6.5e-04 | 4.09e-03 |
| 0 | with RAR | yes: prescribed inlet U0(1 - y^16) | 9.43e-04 | 1.29e-03 | -0.0004 | 8.3e-04 | 4.70e-03 |
| 0.5 | baseline | no: flow still developing | n/a | n/a | +0.0037 | 2.7e-04 | 1.14e-01 |
| 0.5 | with RAR | no: flow still developing | n/a | n/a | +0.0006 | 3.8e-04 | 1.14e-01 |
| 1 | baseline | no: flow still developing | n/a | n/a | +0.0019 | 1.9e-04 | 4.73e-02 |
| 1 | with RAR | no: flow still developing | n/a | n/a | -0.0060 | 4.1e-04 | 4.74e-02 |
| 2 | baseline | no: flow still developing | n/a | n/a | -0.0111 | 3.8e-04 | 3.73e-03 |
| 2 | with RAR | no: flow still developing | n/a | n/a | -0.0081 | 1.8e-04 | 3.81e-03 |
| 4 | baseline | yes: Poiseuille parabola 1 - y^2 | 8.80e-05 | 1.24e-04 | -0.0008 | 2.0e-04 | 1.04e-04 |
| 4 | with RAR | yes: Poiseuille parabola 1 - y^2 | 3.42e-04 | 4.81e-04 | -0.0089 | 6.1e-04 | 2.26e-04 |
| 10 | baseline | yes: Poiseuille parabola 1 - y^2 | 3.38e-04 | 5.71e-04 | -0.0254 | 6.8e-04 | 4.45e-04 |
| 10 | with RAR | yes: Poiseuille parabola 1 - y^2 | 1.66e-04 | 2.41e-04 | -0.0000 | 2.8e-04 | 3.74e-04 |

Exact solutions exist only at the inlet (prescribed) and where the flow is developed (x* >= 3.77). Between them there is no closed form, so no error is quoted.

## Table 2. u against x: velocity along the channel at fixed heights

| Height y* | Run | Inlet u, exact | Inlet u, PINN | Developed u, exact 1 - y*^2 | Outlet u, PINN | Outlet relative error vs exact | Distance x* after which u stays within 1% of exact |
|---|---|---|---|---|---|---|---|
| 0 | baseline | 0.70833 | 0.70801 | 1.00000 | 0.99994 | 6.34e-05 | 1.780 |
| 0 | with RAR | 0.70833 | 0.70769 | 1.00000 | 1.00016 | 1.57e-04 | 1.790 |
| 0.5 | baseline | 0.70832 | 0.70850 | 0.75000 | 0.74997 | 3.71e-05 | 1.550 |
| 0.5 | with RAR | 0.70832 | 0.70854 | 0.75000 | 0.74990 | 1.34e-04 | 1.540 |
| 0.9 | baseline | 0.57708 | 0.57726 | 0.19000 | 0.18989 | 5.97e-04 | 1.860 |
| 0.9 | with RAR | 0.57708 | 0.57681 | 0.19000 | 0.19003 | 1.66e-04 | 1.870 |

## Table 3. The original equations (exact residual 0), on 80,601 grid points

| Equation | Run | RMS residual, whole channel | RMS residual, excluding x* < 0.5 | Largest residual | Where the largest residual is (x*, y*) | RMS on the centreline, developed region |
|---|---|---|---|---|---|---|
| x-momentum | baseline | 1.10e-03 | 1.08e-03 | 1.29e-02 | (0.025, +0.920) | 9.30e-04 |
| x-momentum | with RAR | 1.10e-03 | 1.07e-03 | 1.09e-02 | (0.000, +0.970) | 1.29e-03 |
| y-momentum | baseline | 8.76e-04 | 8.31e-04 | 1.09e-02 | (0.025, +1.000) | 3.92e-04 |
| y-momentum | with RAR | 9.48e-04 | 8.72e-04 | 2.10e-02 | (0.025, -0.990) | 9.25e-04 |
| continuity | baseline | 7.65e-04 | 6.26e-04 | 3.94e-02 | (0.000, -1.000) | 5.36e-04 |
| continuity | with RAR | 9.02e-04 | 7.52e-04 | 4.56e-02 | (0.000, -1.000) | 9.49e-04 |

## Table 4. The original equations on the dense unseen sweep

| Equation | Run | Points checked | RMS residual | 99.9th percentile of abs residual | Largest residual | Share of points above 1e-2 |
|---|---|---|---|---|---|---|
| x-momentum | baseline | 2,003,001 | 1.08e-03 | 5.53e-03 | 2.03e-02 | 0.003% |
| x-momentum | with RAR | 2,003,001 | 1.09e-03 | 4.17e-03 | 2.42e-02 | 0.003% |
| y-momentum | baseline | 2,003,001 | 8.64e-04 | 4.27e-03 | 1.35e-02 | 0.000% |
| y-momentum | with RAR | 2,003,001 | 9.32e-04 | 4.51e-03 | 2.11e-02 | 0.004% |
| continuity | baseline | 2,003,001 | 7.16e-04 | 4.30e-03 | 3.94e-02 | 0.025% |
| continuity | with RAR | 2,003,001 | 8.33e-04 | 5.06e-03 | 4.56e-02 | 0.025% |
