# Phase 1 — Poiseuille flow PINN

## Purpose

Build and validate the solver harness on a problem with an exact analytic
solution, so that later phases (bifurcation geometry, design optimisation) are
built on machinery that is known to be correct.

Phase 1 is **not** a research result. Plane Poiseuille flow was solved
analytically in the 19th century. The deliverable is a trustworthy harness and a
test suite that will catch silent errors in Phase 2, where no analytic solution
exists.

---

## 1. Governing equations

2D steady incompressible Navier–Stokes, velocity–pressure form, non-dimensional:

```
(u · ∇)u = -∇p + (1/Re) ∇²u
∇ · u    = 0
```

with `u = (u, v)`, and `Re = ρ U_max h / μ` where `h` is the channel half-height
and `U_max` the centreline velocity of the fully developed profile.

### Non-dimensionalisation

- Lengths by `h`: `x* = x/h`, `y* = y/h`
- Velocities by `U_max`
- Pressure: use **viscous scaling** `p* = p h / (μ U_max)` for `Re ≤ 1`, and
  **inertial scaling** `p* = p / (ρ U_max²)` for `Re > 1`. Make this a config
  option, not a hard-coded choice — the conditioning of the loss depends on it.

Write the non-dimensionalisation as an explicit, tested transform in
`src/physics/scaling.py`. A unit test must confirm round-tripping SI → non-dim →
SI is exact to machine precision.

### Analytic solution (the oracle)

Channel `y ∈ [-h, h]`, pressure gradient `G = -dp/dx > 0`:

```
u(y)  = (G / 2μ)(h² - y²)
U_max = G h² / (2μ)
v(y)  = 0
Q     = 2 G h³ / (3 μ)   (per unit depth)
```

In non-dimensional form: `u* = 1 - y*²`, `v* = 0`, `dp*/dx* = const`.

Note the exponent: `Q ∝ h³` in 2D, not `r⁴` as for a circular pipe. This matters
in Phase 3 and must not be silently mixed with circular-pipe formulas.

---

## 2. Domain, boundary conditions, and cases

Domain: `x* ∈ [0, L*]`, `y* ∈ [-1, 1]`. Use `L* = 20` unless the config says
otherwise.

### Case A — fully developed channel (harness validation)

- Inlet (`x* = 0`): parabolic profile `u* = 1 - y*²`, `v* = 0`
- Outlet (`x* = L*`): zero pressure, zero streamwise gradient of velocity
- Walls (`y* = ±1`): no-slip, `u* = v* = 0`

Exact solution known everywhere. This case exists to prove the harness works.

### Case B — developing entrance flow (capability test)

- Inlet: **uniform** `u* = 1`, `v* = 0`
- Same wall and outlet conditions

No closed-form solution, but two checks apply: the downstream profile must
converge to the Case A analytic profile, and the measured entrance length must
agree with published low-Reynolds microchannel/channel correlations to within a
stated tolerance.

Case B is the one that actually predicts Phase 2 success, because the
bifurcation problem is a redeveloping-flow problem.

### Reynolds numbers to run

Run the full test matrix at `Re ∈ {1, 10, 50, 100}`. Do not attempt `Re > 200`
in Phase 1. Record accuracy versus `Re` — degradation with increasing `Re` is
expected and is a result worth reporting in the paper.

---

## 3. Model architecture

Implement these as swappable components selected by config, in rough order of
implementation priority:

1. **Baseline MLP** — tanh activation, width 128–256, depth 4. This is the
   published recommendation range (width 128–512, depth 3–6) and is a deliberate
   starting point, not a placeholder.
2. **Random Fourier feature embedding** on the input coordinates, with a
   configurable bandwidth. Addresses spectral bias.
3. **Modified MLP** (the two-encoder variant with multiplicative mixing).
4. **Random weight factorisation** on all dense layers.
5. **PirateNet-style adaptive residual blocks** — only if 1–4 leave you short of
   the accuracy target. Do not start here.

### Hard boundary conditions (required, not optional)

Do not enforce no-slip through a penalty term. Construct the ansatz so no-slip
holds exactly:

```
u*(x*, y*) = φ(y*) · N_u(x*, y*)
v*(x*, y*) = φ(y*) · N_v(x*, y*)
```

with `φ(y*) = 1 - y*²`, which vanishes on both walls. This removes the wall
boundary loss entirely and is the single largest accuracy lever in the project.

Handle the inlet condition by an additional lifting term or by a soft loss;
document which was chosen and why. Pressure is defined up to a constant unless
pinned — gauge-fix it explicitly at the outlet and note this in the report,
because an ungauged pressure will make the pressure L2 error meaningless.

### Continuity

Implement two options and compare:

- **Velocity–pressure**: continuity enters as a residual loss term.
- **Stream function**: `u = ∂ψ/∂y`, `v = -∂ψ/∂x`, which satisfies continuity
  identically and removes one loss term at the cost of second derivatives in the
  momentum residual.

Report which performs better. This is a legitimate, publishable comparison at
this level.

---

## 4. Training

- **Optimiser schedule**: Adam with a decaying learning rate, then L-BFGS. This
  is the baseline. If accuracy stalls above target, try a quasi-second-order
  method (SOAP or a Gauss-Newton / natural-gradient variant), which is where the
  recent accuracy gains in the literature have come from.
- **Loss weighting**: start with fixed equal weights. If the residual terms are
  imbalanced, add gradient-norm-based annealing or an NTK-based scheme. Keep all
  weighting logic in one module.
- **Collocation sampling**: uniform random resampling each N steps, plus optional
  residual-based adaptive refinement. Log the sampling seed.
- **Precision**: float64 throughout, enabled before any array creation.

---

## 5. Test suite — this is the core deliverable

Every test below must exist and pass before Phase 2 begins.

| # | Test | Assertion |
|---|------|-----------|
| 1 | Analytic accuracy, Case A | Relative L2 error on `u` ≤ **1e-5**, target 1e-6 |
| 2 | Transverse velocity, Case A | `max|v*|` ≤ 1e-5 (the PINN must *discover* this) |
| 3 | Mass conservation | `∫u* dy*` at any 10 streamwise stations agrees to within 1e-6 relative |
| 4 | Parent/daughter flux (stub for Phase 2) | Placeholder that fails loudly until implemented |
| 5 | Autodiff gradient check | Residual derivatives match central finite differences to 1e-7 |
| 6 | Method of manufactured solutions | Residual operator applied to a known smooth field reproduces the analytically derived forcing |
| 7 | Non-dimensionalisation round trip | SI → non-dim → SI exact to machine precision |
| 8 | Determinism | Same config and seed reproduce the same final error to 1e-12 |
| 9 | Case B convergence | Downstream profile matches analytic to 1e-4; entrance length within 15% of correlation |

Tests 1–3 and 9 are slow (require training) — mark them `@pytest.mark.slow` and
run a short-schedule version in CI.

**Test 1 is the gate.** Published work using a plain DeepXDE setup on this exact
problem reached a total test loss of 8.7e-5 but a relative L2 error of 8.6%.
That is the failure mode this test exists to catch. If loss is low and test 1
fails, the bug is in the formulation or the boundary handling, not the optimiser.

---

## 6. Module layout

```
src/
  physics/      residuals, scaling, analytic solutions
  geometry/     domain definitions, distance functions φ, collocation sampling
  models/       MLP, Fourier features, modified MLP, RWF, hard-BC wrapper
  training/     optimisers, weights.py, schedules, training loop
  evaluation/   L2 metrics, residual maps, conservation checks, plots
  run.py        config → experiment
  report.py     run directory → accuracy table
experiments/    one YAML per run; no logic
tests/
runs/           outputs, git-ignored
```

Rules: `experiments/` holds configuration only. `runs/` is written by the code
and never edited by hand or by an agent. `physics/` has no dependency on
`models/` or `training/`.

---

## 7. Milestones

| Day | Deliverable | Done when |
|-----|-------------|-----------|
| 1 | Repo skeleton, scaling module, analytic solution module | Tests 6, 7 pass |
| 2 | Geometry, sampling, hard-BC wrapper, baseline MLP | Test 5 passes |
| 3 | Training loop, Case A end to end | Test 1 passes at 1e-3 or better |
| 4 | Fourier features, loss weighting, optimiser tuning | Tests 1, 2, 3 pass at target |
| 5 | Case B developing flow | Test 9 passes |
| 6 | Re sweep, stream-function comparison, plots | Accuracy vs Re table produced |
| 7 | Written summary of Phase 1 findings | Report generated from `runs/` |

---

## 8. Out of scope for Phase 1

3D, turbulence, non-Newtonian fluids, compliant walls, heat transfer, the
bifurcation geometry, and any design optimisation. Do not add these
speculatively. If a design choice would need revisiting for Phase 2, note it in
the gotchas section of `CLAUDE.md` rather than building for it now.

---

## 9. Known traps

- **Low loss, wrong answer.** The defining failure mode. Always check test 1.
- **float32.** Caps accuracy near 1e-4 and mimics an optimiser failure.
- **Ungauged pressure.** Pressure is unique only up to a constant; pin it.
- **Second derivatives through tanh** lose precision; if the stream-function form
  stalls, this is a likely cause.
- **Adam → L-BFGS handoff** frequently resets or mishandles state. Verify the
  loss is continuous across the switch.
- **Resampling collocation points** changes the effective objective between
  epochs; confirm the reported error is computed on a fixed evaluation grid, not
  on the training points.
- **Convective term at higher Re** destabilises training. If `Re = 100` fails
  while `Re = 10` succeeds, that is expected behaviour, not a bug — record it.
