## Every setup against the finite-element reference, most accurate first

Relative L2 errors (entrance length: relative error). Several seeds: mean (range). The reference's own error is the last row; values near it cannot be ranked against each other.

| Setup | Seeds | u, whole channel | u, entrance region | u, developed region | v, whole channel | p, whole channel | entrance length | total-pressure drop (the design objective) | viscous dissipation | resistance: dissipation / flux^2 (flux error cancelled) | energy budget, PINN alone (no reference) | Training time (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P7 | 3 | 3.26e-06 (2.7e-06 to 4.0e-06) | 3.45e-06 (3.0e-06 to 4.2e-06) | 3.13e-06 (2.4e-06 to 3.9e-06) | 7.08e-05 (6.8e-05 to 7.2e-05) | 1.73e-05 (1.7e-05 to 1.8e-05) | 3.45e-05 (1.9e-05 to 4.2e-05) | 3.87e-06 (5.5e-07 to 7.2e-06) | 5.02e-06 (3.1e-06 to 7.8e-06) | 1.01e-06 (3.4e-07 to 1.9e-06) | 2.57e-06 (1.4e-06 to 4.4e-06) | 80 |
| G5 (reference setup) | 1 | 3.63e-06 | 4.45e-06 | 3.05e-06 | 1.08e-04 | 1.71e-05 | 1.33e-04 | 1.07e-06 | 3.23e-06 | 2.50e-06 | 1.62e-06 | 206 |
| *finite-element reference's own error* | | 8.7e-07 | 8.7e-07 | 8.7e-07 | 9.8e-06 | 1.5e-05 | 1.0e-06 | 6.7e-08 | 1.8e-07 | 1.8e-07 | 1.5e-07 | |
