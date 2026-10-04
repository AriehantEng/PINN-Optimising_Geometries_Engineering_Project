## Network 2: the pressure-drop PINN on one bifurcation, against finite elements

One fixed geometry, solved three times as the method was corrected: 30/90
degrees (daughter 1 at 30 degrees up, daughter 2 at 90 degrees down), split
0.5 (widths 1 and 1), parent 3 long, daughters 4 long, Re_h = 10, parabolic
inflow, each daughter required to carry half the flow (the flux-defective
outlet: outlet 1's pressure sits above outlet 2's by whatever that takes).
Each run is scored against the finite-element solution of the same problem
(P2/P1, 28,399 triangles, 36 s), errors integrated on the reference's mesh.

| | 1: velocity-pressure | 2: stream function | 3: + walls extended |
|---|---|---|---|
| u, relative L2 | 2.46e-2 | 1.86e-2 | **9.0e-3** |
| v, relative L2 | 1.08e-1 | 2.59e-2 | **1.33e-2** |
| p, relative L2 | 1.48e-1 | **5.2e-3** | 6.3e-3 |
| total-pressure drop (reference 4.0516) | 3.8678 (-4.5%) | 4.0356 (-0.40%) | 4.0684 (+0.42%) |
| flow split (reference 0.5) | 0.5011 | 0.5 exact | 0.5 exact |
| mass imbalance | -6.3% | 0 (exact) | 8e-16 |
| outlet 1's extra pressure (reference +0.5421) | -0.023 | +0.555 | +0.556 |
| inlet static pressure (reference 4.3226) | 3.798 | 4.308 | 4.347 |
| training time | 70 min | 101 min | 90 min |

Errors outside discs of radius 0.02-0.05 around the reflex corners are the
same to 1-6% (u, v) and 5% (p): the error is spread through the domain,
not concentrated at the singular corners.

### What each change fixed

1. **Velocity-pressure** (u = D N_u, v = D N_v, continuity and the share as
   loss terms) lost 6.3% of the mass: daughter 2 delivered 0.580 of its
   0.667. Continuity as a penalty lets mass drift, exactly as Phase 1's
   Case B did. Half the loss sat on 0.5% of the points, 0.05-0.1 from the
   reflex corners, where the residual of u = D N grows like 1/r for any
   network.
2. **Stream function** (psi = G + D^2 N, G blending the wall chains' values
   with zero normal derivative) makes mass and the share exact by
   construction; residuals near the reflex corners weighted by min(1, r/0.3).
   p improved 29x and the objective 11x, and the outlet pressure offset
   became right. One point near daughter 1's outlet corner then held 86% of
   a held-out loss: the stream function's third derivatives of the wall
   distances are large where a wall segment ends, and the walls ended at the
   outlets.
3. **Walls extended 0.5 past the openings**, so no wall ends on the domain's
   boundary except at the reflex corners. Held-out loss 4x lower, u and v
   errors halved.

### Where it stands

The total-pressure drop, the design objective, is within 0.4% in both
stream-function runs, of opposite sign (-0.40%, +0.42%): that is the run-to-
run band, and the third run's halved field errors did not narrow it. The
finite-element solve of this geometry takes 36 s; a network takes 70-100
min for 0.4%. One geometry at a time, the network does not compete with
finite elements; it can only as a parametric network (geometry as an
input), trained once and then queried, which is the next stage, and only
if it holds this accuracy across the parameter range.
