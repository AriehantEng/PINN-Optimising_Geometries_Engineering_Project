# SPEC v2 — Solutions S1–S9

Supersedes the corresponding sections of `SPEC-phase1-poiseuille.md`. Where the
two conflict, this document wins. Phase 1's test numbering is preserved.

Implement in the order S1 → S2 → S3 → (re-run Case B) → S4 → S5 → S7, then
S6, S8, S9 as Phase 2/3 gates. **S3 sizes a domain using a correlation that S1
defines. Running S3 first bakes a convention error into every subsequent run.**

---

## S1 — Unified Reynolds and length-scale conventions

### Problem

Test 9 missed the entrance-length correlation by 289% while every other quantity
derived from the same solution sat within a few percent. That signature is
characteristic of a convention mismatch, not a solver error.

### Mathematics

Two Reynolds numbers appear in this project and they are not equal.

**Solver convention**, based on half-height and centreline velocity:

$$\mathrm{Re}_h = \frac{\rho\, U_{\max}\, h}{\mu}$$

**Correlation convention**, based on hydraulic diameter and mean velocity:

$$\mathrm{Re}_{D_h} = \frac{\rho\, U_{\text{mean}}\, D_h}{\mu}$$

For a wide plane channel of half-height `h` and width `w ≫ h`, the hydraulic
diameter is

$$D_h = \frac{4A}{P} = \frac{4(2hw)}{2w} = 4h$$

and for the parabolic profile the mean velocity is

$$U_{\text{mean}} = \frac{1}{2h}\int_{-h}^{h} U_{\max}\left(1 - \frac{y^2}{h^2}\right)dy = \frac{2}{3}U_{\max}$$

Therefore

$$\boxed{\ \mathrm{Re}_{D_h} = \frac{8}{3}\,\mathrm{Re}_h\ }$$

and the reference lengths differ by a factor of four. Chained through a
correlation linear in Re, that is roughly an order of magnitude.

### Entrance-length correlation

Use the Durst-type blended form, which is valid at both low and moderate Re
where the pure boundary-layer scaling `L_e/D_h = 0.05·Re` is not:

$$\frac{L_e}{D_h} = \left[ C_0^{\,n} + \left(C_1 \mathrm{Re}_{D_h}\right)^{n} \right]^{1/n}$$

**The constants `C_0`, `C_1`, `n` differ between pipe and plane-channel geometry
and must be read from the source, not assumed.** Put them in the config with a
mandatory `citation` string, and have the loader raise if it is absent. Do not
hard-code values into `src/`. A wrong constant here is invisible; a missing
citation is not.

### Implementation

`src/physics/conventions.py` exposes:

```
reynolds_h(rho, u_max, h, mu)          -> Re_h
reynolds_dh(rho, u_mean, d_h, mu)      -> Re_Dh
re_h_to_re_dh(re_h)                    -> (8/3) * re_h
entrance_length_over_h(re_h, corr_cfg) -> L_e / h
```

`entrance_length_over_h` must convert internally and return in units of `h`,
since `L_e/h = 4·(L_e/D_h)`. Every correlation call site declares which Reynolds
number it consumes; a function taking `Re` without a suffix is a bug.

### Tests

- **T-S1.1** `re_h_to_re_dh(1.0) == 8/3` exactly.
- **T-S1.2** Round trip `Re_h → Re_Dh → Re_h` exact to machine precision.
- **T-S1.3** Recomputing `U_mean` by quadrature of the analytic profile gives
  `2/3 · U_max` to 1e-14.
- **T-S1.4** Config loader raises when `citation` is absent from the
  correlation block.

### What this solves

Test 9's entrance-length failure, if the diagnosis holds. Independently, it
pre-empts the same bug in Phase 3, where PINN resistance is compared against 1D
theory and the two literatures use different reference velocities.

---

## S2 — Case B inlet amplitude

### Problem

A uniform inlet `u* = 1` cannot develop into `1 − y*²`. The two carry different
flux, so mass conservation forbids it. This is a reference-velocity mismatch in
the spec, not a contradiction in the physics.

### Mathematics

Non-dimensional flux per unit depth for the developed profile:

$$Q^* = \int_{-1}^{1}\left(1 - y^{*2}\right)dy^* = \frac{4}{3}$$

A uniform inlet `u* = c` carries `2c`. Matching gives `c = 2/3`.

Equivalently: the developed profile reached from a uniform inlet of amplitude 1
is `1.5(1 − y*²)`, whose centreline exceeds the reference velocity by 50%.

### Implementation

Set the Case B inlet to `u* = 2/3`, `v* = 0`, keeping `U_max` of the *developed*
profile as the single reference velocity across both cases. Do not instead
change test 9's target — that would leave two scaling conventions live in one
codebase, which is the class of bug S1 exists to eliminate.

### Diagnostic to run first

Case B's reported profile error was 3.53e-2 while its mass error was 3.62e-2.
An uncorrected 50% amplitude error cannot produce 3.5e-2. Before implementing,
check whether the run already normalised by mean velocity somewhere. If the two
numbers are tracking each other, the profile shape is essentially correct and
the entire profile error is mass drift appearing as an amplitude offset — in
which case the S4 re-run collapses test 9's profile failure without S2 doing
anything. Record which of these is true; it changes what test 9 is measuring.

### Tests

- **T-S2.1** Inlet flux equals developed flux to 1e-12.
- **T-S2.2** Under Case A initialisation, the ratio of profile error to mass
  error is logged for every run, so the coupling above stays visible.

---

## S3 — Reynolds-dependent domain length

### Problem

`L* = 20` was specified with no Re dependence. Entrance length grows linearly in
Re at moderate Re, so no constant can be correct across the sweep.

### Mathematics

Required streamwise extent, in units of half-height:

$$L^*_{\min} = \alpha\,\frac{L_e(\mathrm{Re}_h)}{h} + L^*_{\text{out}}$$

with development margin `α = 1.5` and outlet margin `L*_out ≥ 5`. The outlet
margin exists so that the zero-gradient outflow condition is applied where the
flow is already developed; without it the outlet BC contaminates the region
being measured.

### Implementation

`L*` is **derived**, not configured. The config supplies `alpha`,
`outlet_margin` and the correlation block; `L*` is computed at load. Add a
load-time assertion that fails loudly:

```
assert L_star >= alpha * L_e_over_h(Re_h) + outlet_margin
```

A truncated domain produces a solution that looks converged and is wrong, which
is exactly the failure class this project is built to catch. The assertion must
fire before training, not after.

### Tests

- **T-S3.1** `L*` scales linearly with Re in the moderate-Re limit.
- **T-S3.2** Constructing a config with `L*` below the bound raises at load.
- **T-S3.3** Measured entrance length is at most `L*/α`, i.e. development
  completes inside the domain with margin to spare.

---

## S4 — Stream-function formulation

### Governing system

Define `ψ` by

$$u = \frac{\partial \psi}{\partial y}, \qquad v = -\frac{\partial \psi}{\partial x}$$

Continuity `∇·u = 0` is then satisfied identically. Vorticity is

$$\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y} = -\nabla^2\psi$$

Taking the curl of the momentum equation eliminates pressure and gives steady
vorticity transport. **Do not substitute `ω = −∇²ψ` into it.** That yields a
fourth-order equation in `ψ`, and fourth-order autodiff through `tanh` will be
your precision floor. Solve the coupled second-order system with two network
outputs `(ψ, ω)`:

$$\begin{aligned}
r_1 &\equiv \omega + \nabla^2\psi = 0 \\[2pt]
r_2 &\equiv \frac{\partial \psi}{\partial y}\frac{\partial \omega}{\partial x} - \frac{\partial \psi}{\partial x}\frac{\partial \omega}{\partial y} - \frac{1}{\mathrm{Re}}\nabla^2\omega = 0
\end{aligned}$$

`ω` carries no hard constraint and is coupled only through `r_1`, which
partially restores the falsifiability that the pure-`ψ` form destroys. This is a
feature; do not hard-constrain `ω`.

### Why the stream function kills test 3

Global flux conservation does **not** follow from the identity alone. It follows
from the identity *plus* the walls being streamlines. No-slip forces
`v = −∂ψ/∂x = 0` on each wall, so `ψ` is constant along it, and

$$\int_{-1}^{1} u^*\,dy^* = \psi\big|_{y^*=1} - \psi\big|_{y^*=-1} = \text{const in } x^*$$

Confirm which of these your ansatz enforces exactly. A factor of `φ` gives
zero velocity at the wall; a factor of `φ²` additionally pins `ψ` there. Only
the second makes test 3 fully vacuous. If you are on `φ`, test 3 is weakened
rather than dead and you should know by how much.

### Ansatz

With `φ(y*) = 1 − y*²`:

$$\psi(x^*,y^*) = \psi_{\text{lift}}(y^*) + \varphi(y^*)^2\, N_\psi(x^*,y^*)$$

At the walls `φ = 0`, so `ψ = ψ_lift`; and
`∂_y[φ²N] = 2φφ'N + φ²∂_yN = 0` since `φ = 0`. Both conditions hold exactly.
`ψ_lift` must satisfy `ψ_lift(±1) = ±Q*/2` and `ψ'_lift(±1) = 0`.

### The Case A degeneracy — second disarming instance

The minimal cubic satisfying those two conditions is

$$\psi_{\text{lift}}^{\text{cubic}}(y^*) = \frac{3Q^*}{4}\left(y^* - \frac{y^{*3}}{3}\right)$$

which for `Q* = 4/3` is `y* − y*³/3`. Its derivative is `1 − y*²`. **That is the
exact Case A solution.** With this lift, Case A's correct answer is `N_ψ ≡ 0`,
so Case A no longer tests the network's expressivity — only that the lift and
training machinery are wired up. The hard constraint has disarmed the gate test
the same way it disarmed test 3.

Fix: provide a second, non-degenerate lift satisfying the same boundary
conditions,

$$\psi_{\text{lift}}^{\text{sin}}(y^*) = \frac{Q^*}{2}\sin\!\left(\frac{\pi y^*}{2}\right)$$

for which `ψ(±1) = ±Q*/2` and `ψ' = (Q*π/4)\cos(πy*/2) = 0` at `y* = ±1`, but
whose derivative is not the Poiseuille profile. `N_ψ` must then learn a nonzero
correction.

**Run Case A under both lifts.** Cubic is the machinery check; sinusoidal is the
expressivity check. Test 1 must pass under both, and only the sinusoidal result
may be quoted as an accuracy claim.

### Pressure recovery

`p` is absent from the ψ–ω system and is recovered post hoc by integrating

$$\nabla p = -(\mathbf{u}\cdot\nabla)\mathbf{u} + \frac{1}{\mathrm{Re}}\nabla^2\mathbf{u}$$

along a path from the gauge point at the outlet. Path independence of the result
is itself a consistency check worth asserting: integrating along two distinct
paths must agree to tolerance.

### Retain velocity–pressure as a diagnostic probe

Keep the velocity–pressure implementation as a selectable configuration. It is
now the **falsifiability probe**: the formulation in which mass conservation
remains a real test. Run it on every new geometry before trusting the ψ result
there. This converts the Phase 1 finding from a warning into a standing
procedure.

### Tests

- **T-S4.1** `r_1` and `r_2` reduce to the analytic Case A solution when
  evaluated on it, to 1e-12.
- **T-S4.2** Pressure recovered along two distinct paths agrees to 1e-8.
- **T-S4.3** Test 1 passes under both lifts; both errors are logged separately.
- **T-S4.4** `ω` is not hard-constrained anywhere (structural assertion on the
  model definition).

---

## S5 — Falsifiability matrix and sentinel tests

### Problem

Test 3 became vacuous under the stream function and nothing in the process
reported it. Hard no-slip had already done this to the wall tests silently. The
defect is not the disarming, it is the absence of detection.

### Definition

A test `T` with assertion `A(θ)` over network parameters `θ` is **vacuous under
ansatz `F`** if `A(F(θ))` holds for all `θ` in parameter space. Then `T` reports
a property of `F`, not of the trained solution.

### Executable form

Vacuity is decidable by sampling. If a randomly initialised, untrained network
passes a test, that test cannot distinguish a trained solution from noise.

Implement `tests/test_sentinels.py`. For every test in the suite, instantiate
the model at random initialisation with several seeds and assert the test
**fails**. A sentinel that passes is a defect report:

```
For each (test, formulation) pair:
    model = build_model(formulation, seed=k)   # untrained
    assert test(model) fails, else mark VACUOUS(test, formulation)
```

Emit the results as a matrix, tests down the side, formulations across the top,
each cell one of `LIVE`, `WEAKENED`, `VACUOUS`. Regenerate on every ansatz
change. **A row that is `VACUOUS` in all columns means the property is untested
by this suite** and must be flagged in the report, not silently accepted.

### Expected initial output

Wall no-slip: `VACUOUS` in every column (hard constraint).
Test 3: `LIVE` under velocity–pressure, `VACUOUS` under stream function with
`φ²`, `WEAKENED` under `φ`.
Test 1 under the cubic lift: likely `WEAKENED`, since `N_ψ ≡ 0` is close to
random-initialisation behaviour at small weight scale. Check this.

### Tests

- **T-S5.1** The sentinel suite runs on every formulation and emits a matrix.
- **T-S5.2** A deliberately vacuous test (`assert True`) is correctly reported
  as `VACUOUS`, proving the detector itself is not vacuous.

---

## S6 — Test 4 rewritten around the septum

### Geometry

The bifurcated domain has three wall segments: outer wall A, the septum S
between the daughters, and outer wall B. Each is a streamline with its own
constant `ψ_A`, `ψ_S`, `ψ_B`. The daughter fluxes are

$$Q_1 = \psi_A - \psi_S, \qquad Q_2 = \psi_S - \psi_B, \qquad Q_{\text{tot}} = \psi_A - \psi_B$$

**The split ratio is `ψ_S`.** If `ψ_S` is hard-constrained the way `ψ_A` and
`ψ_B` are, the flow split has been prescribed as a boundary condition and the
network cannot discover it. Test 4 would pass while reporting a number that was
typed into the config.

### Requirements

1. `ψ_A` and `ψ_B` are fixed by the total inlet flux and the gauge choice.
2. **`ψ_S` is a free trainable scalar**, initialised deliberately away from the
   expected value (initialise at a 70/30 split when 50/50 is expected).
3. The ansatz on the branched domain uses an approximate distance function per
   wall segment, combined by R-equivalence:

$$\varphi = \left(\sum_i \varphi_i^{-m}\right)^{-1/m}, \qquad m \geq 2$$

   with `φ_i` the individual segment distance functions. The `φ²` structure is
   retained so each segment pins its own `ψ` constant.

### Reference prediction

In the long-channel limit, junction losses are negligible and the split follows
1D parallel-resistance theory. For plane channels `R_i = 3μL_i/(2h_i^3)`, so

$$\frac{Q_1}{Q_2} = \frac{h_1^3/L_1}{h_2^3/L_2}$$

### Assertions

- **T4.1** Split ratio matches the 1D prediction to within tolerance in the
  long-channel limit.
- **T4.2** `|ψ_S,final − ψ_S,init|` exceeds a threshold, proving `ψ_S` was
  learned rather than imposed.
- **T4.3** At least one **asymmetric** case (`h_1/h_2 = 1.5`, equal lengths).
  A symmetric junction splits 50/50 by symmetry, so a symmetric case cannot
  detect a wrongly-imposed `ψ_S` regardless of the other assertions.
- **T4.4** Sentinel: an untrained network fails T4.1.

Note that T4.3 is required even though the production geometry is symmetric. It
is the only construction that distinguishes learned from imposed.

---

## S7 — Characterise the mass drift, do not relax the tolerance

### Metric

With `M(x*) = ∫u* dy*` evaluated on a fixed grid at `K ≥ 10` stations:

$$D = \frac{\max_{x^*} \left| M(x^*) - M(0) \right|}{M(0)}$$

Test 3 asserts `D ≤ 1e-6`. Case A currently sits at 2.23e-6 with `u` itself
accurate to 1.22e-6.

### Why the tolerance stays

Test 1 measures a pointwise field error; test 3 measures the variation of an
integral. Integration normally averages errors down, so a tighter bound on the
integral is the correct default. `D` exceeding the pointwise error means the
error is *systematic and accumulating downstream* rather than cancelling —
which is the finding, and relaxing to 5e-6 would erase it while leaving Case B
red anyway.

### Discriminating test

Log `D` against L-BFGS iteration count and fit

$$D(n) = D_\infty + a\,n^{-b}$$

- `D_∞` significantly above 1e-6 → formulation bias. Report `D_∞` with a
  confidence interval and close the test as a characterised limitation.
- `D_∞` consistent with zero → under-convergence. The extended run clears it.

Either outcome is a result. Record which, with the fit, in the findings
document.

### Tests

- **T-S7.1** `D` is computed on a fixed evaluation grid, never on training
  collocation points.
- **T-S7.2** The `D(n)` trace is written to the run directory for every run.

---

## S8 — Phase 3 objective and its analytic optimum

### Objective

Minimise total network hydraulic resistance at fixed total channel area. This
needs no metabolic-cost coefficient, which is why it is preferable to a literal
Murray formulation here.

### Derivation

Per unit depth, a plane channel of half-height `h`, length `L` has

$$Q = \frac{2Gh^3}{3\mu} \implies R = \frac{\Delta p}{Q} = \frac{3\mu L}{2h^3}$$

Parent `(h_0, L_0)` feeding two identical daughters `(h_1, L_1)` in parallel:

$$R_{\text{tot}} = \frac{3\mu}{2}\left[\frac{L_0}{h_0^3} + \frac{L_1}{2h_1^3}\right], \qquad A = 2h_0L_0 + 4h_1L_1$$

Lagrangian `ℒ = R_tot + λ(A − A_0)`. Stationarity:

$$\frac{\partial \mathcal{L}}{\partial h_0} = -\frac{9\mu L_0}{2h_0^4} + 2\lambda L_0 = 0 \implies h_0^4 = \frac{9\mu}{4\lambda}$$

$$\frac{\partial \mathcal{L}}{\partial h_1} = -\frac{9\mu L_1}{4h_1^4} + 4\lambda L_1 = 0 \implies h_1^4 = \frac{9\mu}{16\lambda}$$

Dividing, `h_0^4/h_1^4 = 4`, so

$$\boxed{\ \frac{h_0}{h_1} = 4^{1/4} = \sqrt{2} \approx 1.4142, \qquad h_0^2 = h_1^2 + h_2^2\ }$$

This is the 2D analogue of Murray's law. Note the exponent is **2, not 3** — the
cube law is specific to circular cross-sections, where `R ∝ r^{-4}` and volume
`∝ r^2` rather than `R ∝ h^{-3}` and area `∝ h`.

Cross-check via the flow-rate form: minimising `AQ^2h^{-n} + Bh^m` gives
`Q ∝ h^{(n+m)/2}`. Plane channel `(n,m) = (3,1)` gives `Q ∝ h^2`; circular pipe
`(n,m) = (4,2)` gives `Q ∝ r^3`, recovering Murray. The two routes agreeing is
worth asserting as a test.

### What this solves

Phase 3 had no analytic oracle. It now has one, valid in the fully-developed
limit. **The deviation of the PINN optimum from √2 is the result** — the law
itself is not.

### Tests

- **T-S8.1** Both derivation routes give `h_0/h_1 = √2` to 1e-12.
- **T-S8.2** The circular-pipe specialisation recovers `Q ∝ r^3`, guarding
  against silently mixing 2D and 3D formulas.
- **T-S8.3** As `L_1/h_1 → ∞` the PINN optimum approaches √2 within tolerance.

---

## S9 — Junction geometry, fillet, and the angle sweep

### Parameterisation

Junction apex at the origin. Parent occupies `x* < 0` along the axis, half-height
`h_0`. Daughters emanate at `±θ/2`, half-height `h_1`, length `L_1` measured
along their own centrelines.

**`L_1` is held fixed across the angle sweep and outlet positions float** at

$$\left(L_1\cos(\theta/2),\ \pm L_1\sin(\theta/2)\right)$$

If outlet positions were held fixed instead, `L_1 = f(θ)`, and since `R ∝ L/h^3`
the analytic optimum would shift with `θ` for purely geometric reasons unrelated
to the junction. Fixing `L_1` makes the fully-developed baseline `√2` for every
`θ`, so any measured deviation is junction physics by construction.

### Fillet

The apex is a re-entrant corner where velocity and pressure gradients are
unbounded. A smooth network cannot represent it, and the residual loss will
concentrate there and distort the global solution.

Fillet with radius `r_f = 0.1\,h_1`, tangent to both daughter inner walls, held
**fixed** — not a design variable, since it would compete with `θ`. Run the
optimum at `r_f/h_1 ∈ {0.05, 0.1, 0.2}` and report the sensitivity of the
recovered height ratio. If the ratio moves by more than the deviation you are
measuring, the result is a discretisation artifact and must be reported as such.

Measure the precision floor at the fillet specifically. It is where the
second-derivative terms in `r_1` and `r_2` are largest.

### Angle sweep

`θ ∈ {15°, 30°, 45°, 60°}`, each an independent forward problem. Deliverable is
the design curve `h_1/h_0` versus `θ`, so the user's later angle choice becomes
a lookup rather than a re-run.

### Separation check

Not optimising `θ` does not stop the solver from being asked to run at 60°. The
risk moves from the optimiser wandering into an invalid region to the user later
selecting an unvalidated angle.

Assert in every Phase 2 run: no point downstream of the apex has streamwise
velocity (projected onto the local daughter centreline) below zero. Record the
`(θ, Re)` pairs that pass. **The validated range is a reported result**; a
documented ceiling is a legitimate finding, a silent extrapolation is not.

### Design variable handling

`h_1/h_0` enters as a network input rather than being swept discretely, giving a
continuous `R(h_1/h_0)` from one training run per angle. The minimum is then
located by differentiating the surrogate.

Be accurate about the justification in writing. With a one-dimensional design
space, the strong argument for a parametric PINN — cheap gradients over a
high-dimensional space — does not apply. The claims that survive are mesh-free
operation (no remeshing per geometry) and continuity in the design parameter.
Both are true and sufficient. Do not overclaim beyond them.

### Tests

- **T-S9.1** `L_1` is invariant across the angle sweep (structural assertion on
  the geometry builder).
- **T-S9.2** Fillet sensitivity: recovered ratio varies by less than the
  measured deviation from √2 across `r_f/h_1 ∈ {0.05, 0.2}`.
- **T-S9.3** Separation check runs on every Phase 2 solution and writes its
  verdict to the run directory.
- **T-S9.4** `R(h_1/h_0)` from the surrogate matches discrete forward solves at
  three sampled ratios to within tolerance.

---

## Module additions

```
src/physics/conventions.py     S1
src/physics/streamfunction.py  S4: r_1, r_2, pressure recovery
src/physics/optimum.py         S8: both analytic routes
src/geometry/lifts.py          S4: cubic and sinusoidal
src/geometry/branched.py       S6/S9: R-equivalence ADF, fillet
src/evaluation/separation.py   S9
src/evaluation/drift.py        S7: D(n) fit
tests/test_sentinels.py        S5
```

## Milestones

| Block | Contents | Gate |
|-------|----------|------|
| A | S1, S2, S3 | T-S1.*, T-S2.*, T-S3.* pass; Case B re-run under both formulations |
| B | S4, S5 | Test 1 passes under both lifts; falsifiability matrix emitted |
| C | S7 | `D_∞` reported with fit; test 3 closed as pass or characterised |
| D | S8 | T-S8.* pass (pure analysis, no training) |
| E | S6, S9 | Phase 2 opens |

Block D is independent of A–C and can run in parallel; it requires no compute.

## Standing rules

- No tolerance may be widened to make a test pass.
- The falsifiability matrix is regenerated on every ansatz change; an all-vacuous
  row is a defect report.
- Every new geometry is validated under velocity–pressure before its
  stream-function result is trusted.
- Every conservation test includes at least one asymmetric case.
