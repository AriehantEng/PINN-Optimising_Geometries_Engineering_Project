## What a design evaluation costs: finite elements against the parametric PINN

Layout of the parametric runs at split 0.5; one process, nothing else running; times are medians after a warm-up call.

### Finite elements

| max area | corner levels | triangles | unknowns | solve (s) | peak memory (MB) | drop | drop error vs finest |
|---|---|---|---|---|---|---|---|
| 0.08 | 2 | 520 | 2569 | 0.47 | 236 | 4.048909 | -0.0664% |
| 0.04 | 3 | 1127 | 5383 | 0.46 | 252 | 4.049837 | -0.0435% |
| 0.02 | 3 | 2176 | 10236 | 1.00 | 284 | 4.050593 | -0.0249% |
| 0.01 | 4 | 4844 | 22517 | 2.91 | 368 | 4.051389 | -0.0052% |
| 0.005 | 5 | 11181 | 51326 | 9.83 | 599 | 4.051519 | -0.0020% |
| 0.002 | 6 | 28399 | 129562 | 34.51 | 1414 | 4.051576 | -0.0006% |
| 0.001 | 7 | 61770 | 280609 | 115.37 | 2935 | 4.051601 | +0.0000% |

Solve time grows like (unknowns)^1.45 over the meshes with at least 20000 unknowns.

### Parametric PINN

| run | parameters | one drop query (s) | first query, with JIT (s) | field at reference Gauss points | one LM step (s) | LM step peak memory (MB) | training (min) |
|---|---|---|---|---|---|---|---|
| original | 5487 | 0.0235 | 1.5 | 340788 points in 0.89 s | 36.2 | 2977 | 163 |
| corner modes | 5507 | 0.0233 | 1.9 | 340788 points in 0.97 s | 37.6 | 3891 | 176 |

### One Levenberg-Marquardt step against the network's width

| width | parameters | rows | step (s) | peak memory (MB) |
|---|---|---|---|---|
| 40 | 5487 | 17920 | 36.2 | 2602 |
| 56 | 10367 | 17920 | 122.4 | 4168 |
| 72 | 16783 | 17920 | 382.1 | 8398 |
