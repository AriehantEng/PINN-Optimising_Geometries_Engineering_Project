## Network 1's reference on four meshes (docs/developing_re10-reference.json)

Q2/Q1 channel solution at Re_h = 10, 10 half-widths long; each mesh twice as fine as the last in x and y. Values per level in `network1_reference_four_meshes.csv`.

| quantity | observed order (finest three) | Richardson extrapolation | GCI, finest |
|---|---|---|---|
| dissipation | 3.98 | 2.8464579335 | 1.8e-07 |
| total pressure drop | 4.39 | 2.1348436729 | 6.7e-08 |
| static pressure drop | 4.31 | 2.2393994328 | 8.6e-08 |

Field changes between successive meshes (relative L2 on the finest mesh's Gauss points):

| meshes | u | v | p | entrance length |
|---|---|---|---|---|
| 50x12 → 100x24 | 4.2e-04 | 2.8e-04 | 9.9e-04 | 1.6e-03 |
| 100x24 → 200x48 | 5.0e-05 | 2.4e-05 | 2.1e-04 | 8.2e-05 |
| 200x48 → 400x96 | 6.2e-06 | 2.5e-06 | 4.9e-05 | 1.8e-06 |

Departure of the finest reference's u from the Poiseuille parabola, by station x (relative):

| x | 0.5 | 1 | 2 | 3 | 3.77 | 4 | 5 | 6 | 7.5 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| departure | 2.0e-01 | 7.7e-02 | 5.2e-03 | 1.4e-04 | 1.1e-05 | 1.1e-05 | 1.7e-06 | 1.0e-07 | 4.4e-08 | 4.5e-08 | 4.5e-08 |
