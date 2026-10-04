# PINN for branching channel flow

## What this project is

A physics-informed neural network solving the 2D steady incompressible
Navier–Stokes equations, building toward geometry optimisation of a 1→2 channel
bifurcation. Phase 1 (current) is straight-channel Poiseuille flow.
Read `SPEC-phase1-poiseuille.md` before starting work.

## Commands

- Run an experiment: `python -m src.run --config experiments/<name>.yaml`
- Full test suite: `pytest tests/ -q`
- Fast subset (skips training-based tests): `pytest tests/ -q -m "not slow"`
- Accuracy report for a finished run: `python -m src.report --run runs/<id>`

## Non-negotiable numerical rules

- **IMPORTANT: float64 everywhere.** Enable double precision before any array is
  created. float32 silently caps achievable accuracy near 1e-4 and the symptom
  looks exactly like an optimiser failure, which wastes days.
- All physics is expressed in non-dimensional variables. Never put an SI-valued
  viscosity, density or channel height into a loss term.
- **Never widen a tolerance or lower a target error to make a test pass.** If a
  test fails, report the failure and stop. This rule exists because a PINN that
  reaches low loss with a wrong solution is the expected failure mode here.
- Accuracy is reported as relative L2 error against the analytic solution.
  Loss value is a diagnostic, never an accuracy claim.
- Every run is fully determined by its config file plus its seed. No
  hyperparameters hard-coded in source, no defaults that silently apply.

## Code style

- Residual functions return unweighted residuals. All loss weighting happens in
  one place (`src/training/weights.py`).
- No silent fallbacks. A missing config key raises rather than defaulting.
- Physics functions are pure and take non-dimensional inputs. No module globals.
- Prefer one small function per governing equation term over one large residual
  function, so terms can be unit-tested independently.

## Making a config key required

**IMPORTANT: when a config key becomes required, migrate the stored runs in the
SAME change.** `load_config` refuses a missing key rather than defaulting, which
is right for launching a run -- a run's equations must be readable off its
config. But every finished run under `runs/` has a config written before the key
existed, so adding one locks them all out of `load_run` and no figure can be
regenerated from any completed work.

This has now happened twice: `physics.convective_coefficient` bricked all six
runs, and `physics.inlet_profile` bricked both Case B runs. Both times it was
invisible until something tried to load a run.

The procedure:

1. add the key and its validation
2. extend `src/migrate_configs.py` with the value that describes what the
   stored runs ACTUALLY did -- not a convenient default; if the honest value
   cannot be expressed in the schema, record it as inert and say so
3. run `python -m src.migrate_configs --write`
4. `pytest tests/test_predict.py` -- `test_every_stored_run_still_loads` is the
   guard, and it is not in the "directly affected" set you will think to run

## Gotchas found so far

<!-- Append entries here as we hit them. Format: symptom → cause → fix. -->

**Central differences cannot check second derivatives to 1e-7 on O(5) fields.**
SPEC test 5 failed at ~4e-7 with a clean autodiff implementation → the
three-point central second difference has a roundoff floor of ~4ε|f|/h², and
O(h²) truncation caps h at ~1e-4, so its floor is ~1e-7|f| — above the target
for fields of magnitude 5, which an untrained network produces → use the
five-point (fourth-order) central stencil at h = 1e-3. Measured floor ~2e-9.
The tolerance was not touched. Same trap will appear in Phase 2.

**`np.asarray` on a JAX array gives a read-only view.**
`ValueError: assignment destination is read-only` when clipping sampled points
in a test → JAX buffers are immutable and `asarray` does not copy → use
`np.array(...)` when you intend to mutate.

**vmapping per-point nested `jacfwd` is 2.4× slower than it needs to be.**
`value_and_grad` cost 6.5× the forward derivative bundle, when reverse mode
should cost 2–3× → the reverse pass over a vmapped nested `jacfwd` stores a
tangent copy of every activation for every collocation point, which goes
memory-bandwidth bound → differentiate the *already-batched* field function
along constant tangent directions instead (`_directional` in
`physics/derivatives.py`). 491 ms → 206 ms at the baseline size, values
identical to 5.6e-17. Keep the single-point version as the tested reference.

**A stream function of the form ψ = φ²N cannot represent channel flow.**
Any such ψ is constant on both walls, but ψ(+1) − ψ(−1) *is* the flux, so the
ansatz can only express zero net throughflow → add a base flow carrying the
flux: ψ = ψ_base(y) + φ(y)²N. Note ψ_base = (Q/4)(3y − y³) *is* the exact
Case A solution, so a comparison against the velocity–pressure formulation
using it measures the ansatz, not the method. Use the sine base for any
reported comparison.

**A uniform inlet is incompatible with hard no-slip at the corners.**
Case B's u* = 1 inlet needs N_u = 1/(1 − y²) at x* = 0, which is unbounded →
the network under-delivers flux near the walls → this is an inlet artefact, not
a solver error. Report the flux deficit separately and test the developed
profile shape against the *measured* flux; mass conservation is checked
independently. Phase 2's bifurcation inlet will have the same issue at the
junction corners.

**Loss values are not comparable across configs with different term sets.**
The lifting-inlet variant reported a LOWER final loss than the soft-inlet
baseline (1.63e-9 vs 2.86e-9) and a WORSE relative L2 error on u (5.65e-5 vs
3.48e-5) → lifting removes the inlet loss term, so the total is a sum over five
terms instead of six; part of the "improvement" is a deleted term → only ever
rank configurations by relative L2 against the oracle. This is a sharper form
of "loss is a diagnostic, never an accuracy claim": it bites even when
comparing two runs of the same code. Phase 2 will vary configs far more, so
this trap gets worse, not better.

**Gradient-norm loss weighting explodes the weight of a satisfied term.**
The grad_norm sweep variant was worse than fixed weights (4.95e-5 vs 3.48e-5
on u) → the scheme sets w_i <- (sum_j ||grad L_j||) / ||grad L_i||, so a term
the optimiser has already nailed has a vanishing gradient norm and therefore an
exploding weight. outlet_velocity (du/dx = 0, satisfied identically by the
exact solution) reached 6.3e3 against ~10 for the momentum terms, and the
optimiser then spent its budget on an already-solved constraint → keep
weights.scheme: fixed unless a specific imbalance is demonstrated, and always
read runs/<id>/loss_history.png, whose right-hand panel is the weight trace.
Logging the weights per step is what made this diagnosable at all.

**Soft continuity lets mass drift downstream, silently.**
Case B (velocity-pressure) reproduced the entrance physics correctly but gained
6% mass along the channel (station flux 2.004 -> 2.132) → continuity is only a
penalty term there, so nothing forbids a slow drift, and the loss looks
unremarkable while it happens → the drift then corrupted the entrance-length
measurement too, because that is taken at 99% of the developed centreline and
the centreline was still creeping upward (threshold crossed at x* = 9.3 versus
a correlation value of 2.4). Always plot station flux against x before
believing an entrance length. The stream-function formulation removes the
failure mode by construction, at the cost of making SPEC test 3 trivially true
and therefore no longer an independent check. Phase 2 splits flux between two
daughter branches, so this is the single most important Phase 1 finding for it.

**A top-hat inlet cannot be imposed on hard no-slip walls.**
Case B floored at loss 2.45e-3 with a momentum-x residual of 525 at the inlet
corner, against 8.35e-13 and 3.8e-6 for Case A -> u = phi(y) N_u with
phi = 1 - y^2 means a uniform inlet needs N_u = U/(1 - y^2), unbounded at
y = +-1; the network is being asked for infinity at two points -> use the
regularised family u*(0,y) = U0 (1 - y^(2m)) with U0 = (2/3)(2m+1)/(2m)
(`physics/inlets.py`). It carries flux 4/3 for every m, satisfies no-slip
exactly, and because 1 - y^(2m) = (1 - y^2)(1 + y^2 + ... ) the demand on the
network is a BOUNDED polynomial peaking at U0*m. Measured at y = 0.9999:
3333 for the top hat against 5.66 at m = 8. The top hat is the m -> infinity
limit, so it is recovered as a limit rather than imposed. NOTE m = 1 is the
developed parabola and makes Case B vacuous; config refuses it. Phase 2's
junction inlet has the same corner problem.

**`curve_fit` reporting success is not evidence that a fit happened.**
The 30000-iteration run reported `D_inf = 1.885e-07 +- 9.1e-07, consistent
with zero` from a fit whose every parameter was still the initial guess -> the
unweighted least-squares cost over data spanning 2.7e-4 to 1.9e-7 was ~1e-9 at
`p0`, already under `curve_fit`'s default `ftol`, so it returned immediately
and reported convergence; the decay-range guard then passed it because that
guard is computed FROM the fitted exponent, which was never fitted -> weight
residuals relatively (`sigma=drifts, absolute_sigma=False`) and check
`max|model - D|/D` directly. Never let a guard derived from a fitted parameter
be the only thing standing between you and a verdict. Phase 2 fits loss
coefficients against geometry, so this recurs.

**4 CPU cores, so do not train and run pytest at the same time.**
Both slow down several-fold and timings become meaningless. Run the sweep, then
the tests.


**A small held-out set cannot rank networks on a Levenberg-Marquardt plateau.**
`developing_re10_polish_control` (G4 + 100 LM steps) selected an iterate whose
held-out loss beat G4's (1.94e-8 vs 2.64e-8) and whose error against the
finite-element reference was WORSE in u, v, p and the objective (u 1.19e-5 vs
8.4e-6, total-pressure drop 1.4e-5 vs 2.4e-6) → with fresh points every step
the iterates scatter by one draw's sampling noise, and 4,096 held-out points
measure the loss with more noise than the iterates differ by (G4 alone scores
1.38e-8 on one such set, 2.64e-8 on another) → select on many more held-out
points, average iterates, and rank by the reference. Corollary: the G1/G3/G4
differences are partly selection luck. A warm start must also not reuse an
ancestor's seed: its held-out points chose the parent, so they flatter it.

**Migrating stored runs can break a queue running on older frozen code.**
`re10_polish_points` died at start-up with `unknown key(s) under training.lm:
['average']` → the migration for a new required key had rewritten G4's stored
config, and the queue's frozen worktree predated the key, so it refused G4 as
a warm-start parent → since warm starts, a run reads its siblings' configs, so
migrate only when no queue on older code will read them, or move the queue's
worktree to the migrating commit first. A checkpoint from the older code then
refuses to resume (code fingerprint), by design.

**A uniform grid over-weights the inlet corners in an L2 norm.** The PINN's
pressure error "converged" with the scoring grid (7.0e-5, 4.2e-5, 3.0e-5,
2.5e-5 from 251 to 2001 columns) while u and v did not move → the inlet
corners carry a pressure spike narrower than the grid spacing, and a grid sum
weights the x = 0 column like any interior one: that column held 53.5% of the
summed p error → integrate norms on the reference's own mesh
(`reference.relative_errors`, Gauss points in every element; corner elements
are 0.00085 x 0.00054). Corrected p for G3/G4 is 2.2e-5/1.9e-5, within 1.5x
of the reference's own 1.5e-5. Phase 2's junction corners will do the same.

**Linear interpolation of a threshold crossing biases the entrance length.**
At 1,001 centreline points the bias was 4.8e-6 relative, half of G3's and
G4's reported entrance-length error, and at 10,001 still 1.4e-7 (a new test
caught this at 1e-7; the tolerance was kept) → bracket on a grid, then root-
find on the field itself (`pinn.entrance_length`, Brent). Use one measuring
function for the PINN and the reference, never two that look alike.

**Equal outlet pressures make the pressure-drop optimum a width bound.**
A finite-element split sweep at 60/60, Re 10 fell monotonically from 7.37 at
split 0.5 to 3.05 at 0.1 and 0.9, where the narrow daughter carried 0.2% of
the flow → with do-nothing outlets the daughters are parallel resistances
conducting like w^3, and w1^3 + w2^3 is convex under w1 + w2 = w0, so "lowest
pressure drop" means "close one branch" and min_width alone sets the answer →
a design question needs the flow each daughter must deliver: the flux-defective
outlet (`outlet1_share`, a Lagrange multiplier = outlet 1's extra pressure)
gives an interior optimum (split 0.50 at 60/60, 30/90, 0/120, 90/90). Which
formulation is wanted is the user's call; both are in the code and the tests.

**The inlet is a segment, not the line x* = 0.** At 0/120 the reference lost
152% of its mass through a wall → the Dirichlet data put the inlet profile on
every node with x* = 0, and daughter 2's outer wall crosses x* = 0 at
y* = -6.2, so that wall node got u = 1 - 6.2^2 = -37 → identify boundaries by
distance to their own segment. `solve_bifurcation` now refuses any solution
whose wall velocity is not exactly zero. Any geometry that turns back past the
inlet plane hits this; Phase 2 sweeps will.

**Parallel inner walls have no septum tip.** At a1 + a2 = 180 (90/90, 60/120)
the tip was put at start1, which bent daughter 2's inner wall by 2.3 degrees
at (90, 90) split 0.4 and 5.9 degrees at (60, 120) split 0.5, silently; 30/150
was refused as self-intersecting → 180 degrees is a different shape: a flat far
wall from start2 to start1 (`septum = "face"`, 10 vertices). Tests on strip
widths now run on both shapes. Near 180 with unequal widths the walls meet
past an outlet and the geometry refuses; one daughter at 180 folds onto the
parent's wall and closes, and the entrance gap obeys min_width like the widths.

**At a reflex corner the residual of u = D N grows like 1/r, for any network.**
Network 2's first tiny runs looked like overfitting (held-out 20-50x the
training loss; one draw scored 1.4e3 against a held-out 21) → near B, C and
the septum tip the wall distance D is ~ the distance to a point, whose
Laplacian is ~ 1/r, and the true u ~ r^0.54 means N = u/D cannot be smooth:
r * residual was constant from r = 1e-2 to 1e-4 at an UNTRAINED network. Its
square is not integrable against a 1/r point density, so each draw's loss was
its closest corner point → collocate outside a disc (`sampling.corner_exclusion`)
and add corner points uniform in area. Held-out then tracked training (ratio
~1) and fell 67x. Merging collinear wall edges removed a false 1/r at a
straight daughter's 180-degree joint, an artefact of the R-function
composition. The parametric network and every Phase 2 geometry with a septum
inherit this.
