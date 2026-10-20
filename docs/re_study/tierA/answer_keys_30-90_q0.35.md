## Finite-element answer keys for the parametric PINNs

Geometry: parent 3 long, outer walls 4, share q = 0.35. Search mesh 0.02/3, check mesh 0.005/5, reference mesh 0.002/6. J in the case's pressure units (inertial: rho U_c^2; viscous: mu U_c / h).

| case | s_FE | argmin bound ε_s | J_FE (search) | J_FE (check) | J_FE (reference, at s_FE) | L1, L2 (centreline at s_FE) | s_M | s_L | survives refinement |
|---|---|---|---|---|---|---|---|---|---|
| 30-90_Re0.1 | 0.4017 | 0.0010 | 38.3383 | 38.3483 | 38.3489 | 3.5410, 4.8817 | 0.4232 | 0.4039 | yes |
| 30-90_Re10 | 0.4011 | 0.0010 | 3.87143 | 3.87246 | 3.87257 | 3.5400, 4.8826 | 0.4232 | 0.4039 | yes |

### J on the reference mesh

| case | s = 0.25 | s = 0.30 | s = 0.35 | s = 0.40 | s = 0.45 | s = 0.50 | s = 0.55 |
|---|---|---|---|---|---|---|---|
| 30-90_Re0.1 | 60.1216 | 46.0204 | 39.9283 | 38.3538 | 40.1463 | 45.3275 | 54.9009 |
| 30-90_Re10 | 6.05148 | 4.64014 | 4.03048 | 3.8731 | 4.05313 | 4.57329 | 5.53472 |
