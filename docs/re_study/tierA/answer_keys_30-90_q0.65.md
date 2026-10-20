## Finite-element answer keys for the parametric PINNs

Geometry: parent 3 long, outer walls 4, share q = 0.65. Search mesh 0.02/3, check mesh 0.005/5, reference mesh 0.002/6. J in the case's pressure units (inertial: rho U_c^2; viscous: mu U_c / h).

| case | s_FE | argmin bound ε_s | J_FE (search) | J_FE (check) | J_FE (reference, at s_FE) | L1, L2 (centreline at s_FE) | s_M | s_L | survives refinement |
|---|---|---|---|---|---|---|---|---|---|
| 30-90_Re0.1 | 0.5681 | 0.0010 | 38.8304 | 38.837 | 38.8375 | 3.8293, 4.5933 | 0.5768 | 0.5655 | yes |
| 30-90_Re10 | 0.5695 | 0.0010 | 3.9012 | 3.90191 | 3.90197 | 3.8316, 4.5910 | 0.5768 | 0.5655 | yes |

### J on the reference mesh

| case | s = 0.42 | s = 0.47 | s = 0.52 | s = 0.57 | s = 0.62 | s = 0.67 | s = 0.72 |
|---|---|---|---|---|---|---|---|
| 30-90_Re0.1 | 54.9487 | 45.7627 | 40.6529 | 38.8424 | 40.5076 | 47.0044 | 62.0122 |
| 30-90_Re10 | 5.53697 | 4.60865 | 4.08959 | 3.90271 | 4.06586 | 4.71439 | 6.21589 |
