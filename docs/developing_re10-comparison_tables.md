## Table 1. u against y: velocity across the channel at each station (Re = 10)

| Station x* | Is there an exact solution here? | Relative L2 error of u vs exact | Largest pointwise error in u | Flux = integral of u dy (exact 1.3333) | Flux error (%) | Symmetry error, max abs u(y) - u(-y) (exact 0) | Largest abs v (developed: 0) | Departure from final parabola, relative L2 |
|---|---|---|---|---|---|---|---|---|
| 0 | yes: prescribed inlet U0(1 - y^16) | 6.36e-04 | 1.25e-03 | 1.33332 | -0.0010 | 6.5e-04 | 4.09e-03 | 3.23e-01 |
| 0.5 | no: flow still developing | n/a | n/a | 1.33338 | +0.0037 | 2.7e-04 | 1.14e-01 | 1.77e-01 |
| 1 | no: flow still developing | n/a | n/a | 1.33336 | +0.0019 | 1.9e-04 | 4.73e-02 | 5.99e-02 |
| 2 | no: flow still developing | n/a | n/a | 1.33319 | -0.0111 | 3.8e-04 | 3.73e-03 | 3.70e-03 |
| 4 | yes: Poiseuille parabola 1 - y^2 | 8.80e-05 | 1.24e-04 | 1.33332 | -0.0008 | 2.0e-04 | 1.04e-04 | 8.80e-05 |
| 10 | yes: Poiseuille parabola 1 - y^2 | 3.38e-04 | 5.71e-04 | 1.33299 | -0.0254 | 6.8e-04 | 4.45e-04 | 3.38e-04 |

Exact solutions exist only at the inlet (prescribed) and in the developed region (taken here as x* >= 3.77, twice the correlation entrance length). In between, the departure from the parabola is the flow still developing, not an error.

## Table 2. u against x: velocity along the channel at fixed heights

| Height y* | Inlet u, exact (prescribed) | Inlet u, PINN | Developed u, exact 1 - y*^2 | Outlet u, PINN | Outlet relative error vs exact | Distance x* after which u stays within 1% of exact |
|---|---|---|---|---|---|---|
| 0 | 0.70833 | 0.70801 | 1.00000 | 0.99994 | 6.34e-05 | 1.780 |
| 0.5 | 0.70832 | 0.70850 | 0.75000 | 0.74997 | 3.71e-05 | 1.550 |
| 0.9 | 0.57708 | 0.57726 | 0.19000 | 0.18989 | 5.97e-04 | 1.860 |

## Table 3. The original equations: how close each is to being satisfied (exact value 0)

| Equation | What it enforces | RMS residual, whole channel | RMS residual, excluding x* < 0.5 | Largest residual | Where the largest residual is (x*, y*) | RMS residual on the centreline, developed region |
|---|---|---|---|---|---|---|
| x-momentum | forces along the channel balance (inertia + pressure + viscosity) | 1.10e-03 | 1.08e-03 | 1.29e-02 | (0.025, +0.920) | 9.30e-04 |
| y-momentum | forces across the channel balance | 8.76e-04 | 8.31e-04 | 1.09e-02 | (0.025, +1.000) | 3.92e-04 |
| continuity | no fluid created or destroyed (du/dx + dv/dy = 0) | 7.65e-04 | 6.26e-04 | 3.94e-02 | (0.000, -1.000) | 5.36e-04 |
