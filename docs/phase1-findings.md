# Phase 1 findings

*Status: methodology and harness complete; accuracy results are filled in from
`runs/` by `python -m src.report --all --table`. Numbers in this document are
regenerated from run directories and are not typed in by hand.*

---

## 1. What Phase 1 was for

Not a research result — plane Poiseuille flow was solved analytically in the
19th century. The deliverable is a harness whose correctness is established
independently of the answer it produces, so that Phase 2 (the 1→2 bifurcation,
where no analytic solution exists) rests on machinery that is known to work.

Everything below is therefore reported as *evidence the harness is sound*, not
as a fluid-dynamics finding.

## 2. Formulation

**Governing equations.** 2D steady incompressible Navier–Stokes in
velocity–pressure form, non-dimensionalised on the channel half-height `h` and
the developed centreline velocity `U_max`, with `Re = ρ U_max h / μ`.

**Pressure scaling is a config option, not a hard-coded choice**, because the
conditioning of the loss depends on it:

| scaling | momentum residual | used for |
|---------|-------------------|----------|
| viscous | `Re (u·∇)u + ∇p − ∇²u` | `Re ≤ 1` |
| inertial | `(u·∇)u + ∇p − ∇²u/Re` | `Re > 1` |

The two are the same equation multiplied by `Re`, since `p*_viscous =
Re · p*_inertial`. They have identical solutions and different conditioning.
Both are tested against hand-derived manufactured forcings (SPEC test 6).

**Boundary conditions.**

* Walls: **hard**, by construction. `u = φ(y) N_u`, `v = φ(y) N_v`,
  `φ(y) = 1 − y²`. There is no wall loss term at all. Verified bitwise zero on
  both walls for arbitrary parameters, on every architecture.
* Inlet: **soft by default**, with a lifting construction available for Case A.
  Rationale below.
* Outlet: soft `∂u/∂x = ∂v/∂x = 0`.
* Pressure gauge: soft `p(L*, y*) = 0`, or hard via `p = (1 − x*/L*) N_p`.

### Why the inlet is soft by default

With lifting, `u(0, y) = φ(y) = 1 − y²` is *exactly* the Case A inlet
condition, so the inlet stops being something the network has to solve for.
That makes the run faster and more accurate, and less informative: Case A
exists to prove the harness can discover the profile from the PDE. The soft
inlet is therefore the default, and lifting is available and tested for the
runs where the question is optimiser behaviour rather than harness validity.

Lifting is *unavailable* for Case B, and this is enforced in config validation
rather than left as a footnote: a uniform inlet is discontinuous against hard
no-slip walls at the corners, so no lifting construction can satisfy both.

## 3. What each test actually establishes

| # | Test | What a pass rules out |
|---|------|-----------------------|
| 5 | autodiff vs 4th-order central differences, 1e-7 | a wrong derivative in the residual operator |
| 6 | manufactured solutions, both scalings | a wrong sign or coefficient in any equation term |
| 7 | SI ↔ non-dimensional round trip, 4ε | a dropped factor in the scaling transform |
| 1 | Case A relative L2 on `u` ≤ 1e-5 | **a low loss with a wrong solution** |
| 2 | `max\|v*\| ≤ 1e-5` | a solution that fits `u` but not the physics |
| 3 | mass conservation to 1e-6 across 10 stations | a locally plausible, globally wrong field |
| 8 | determinism to 1e-12 | irreproducible results |
| 9 | Case B profile and entrance length | inability to handle redeveloping flow |

Test 1 is the gate. Published work using a plain DeepXDE setup on this exact
problem reached a total test loss of 8.7e-5 with a relative L2 error of 8.6%.
That is the whole reason accuracy in this project is *never* reported as a loss
value: `src/report.py` prints accuracy and diagnostics under separate headings,
and the loss is always under "diagnostics".

## 4. What limits Case A accuracy (and why it is not the optimiser)

Worth writing down because it decides which lever actually matters.

Substituting the exact solution `u* = 1 − y*²`, `v* = 0` into the x-momentum
residual leaves

```
R_x = u u_x + v u_y + p_x − (1/Re)(u_xx + u_yy) = p_x + 2/Re
```

so the residual demands `p_x ≡ −2/Re` **exactly**, everywhere. The velocity is
already exactly representable: with `u = φ(y) N_u`, the exact answer is
`N_u ≡ 1`, which a network reaches by zeroing the `u` row of its output layer
and setting that bias to 1. Same for `N_v ≡ 0`. So Case A's accuracy is not
limited by whether the network can represent the velocity — it is limited by
how well it can represent a **linear pressure field**, because any error there
pushes back on `u` through the momentum residual.

This makes the pressure gauge a structural lever, not a detail:

| gauge | what `N_p` must represent | exactly representable? |
|-------|---------------------------|------------------------|
| soft, `p = N_p` | affine in `x`: `(2/Re)(L* − x*)` | no — tanh cannot be exactly affine |
| hard, `p = (1 − x*/L*) N_p` | the constant `2L*/Re` | **yes** |

With the hard gauge the entire Case A solution collapses to three constants
(`N_u ≡ 1`, `N_v ≡ 0`, `N_p ≡ 2L*/Re`) and the loss can reach exactly zero.

**This is a real caveat, not a win to be quoted uncritically.** The hard gauge
is explicitly endorsed by the SPEC ("gauge-fix it explicitly at the outlet"),
but it makes Case A an unusually easy optimisation problem, so a good Case A
number obtained with it validates less of the harness than the same number
obtained with the soft gauge. Both are run and both are reported. Case B, whose
solution is not a set of constants under any gauge, remains the case that
actually predicts Phase 2.

## 5. Deliberately not built

The SPEC gates several items on need. They are listed here so that "absent" is
never mistaken for "forgotten".

| SPEC item | Status | Why |
|-----------|--------|-----|
| PirateNet adaptive residual blocks (§3 item 5) | not built | "only if 1–4 leave you short of the accuracy target. Do not start here." |
| SOAP / Gauss-Newton / natural gradient (§4) | not built | gated on "if accuracy stalls above target" |
| NTK-based loss weighting (§4) | not built | the SPEC asks for gradient-norm **or** NTK; gradient-norm annealing is implemented |
| Residual-based adaptive refinement (§4) | not built | explicitly "optional" in the SPEC; uniform resampling every N steps is implemented |

Where an item is not built, no dead configuration key exists for it either —
the config loader rejects keys that would have no effect, so a config file can
never imply a capability the code does not have.

## 6. Results

### 6.1 What helped, what hurt

Ten formulations at an identical reduced budget (Adam 1500 + L-BFGS 3000,
width 64, 2048 interior points, Case A, Re = 10). Ranked by relative L2 error
on `u` against the analytic solution; the SPEC target is 1e-5.

| variant | u relL2 | vs baseline | final loss |
|---------|---------|-------------|------------|
| hard pressure gauge | **4.32e-6** | 8.0× better | 8.84e-11 |
| modified MLP | **6.96e-6** | 5.0× better | 1.11e-10 |
| lifting + hard gauge | **8.08e-6** | 4.3× better | 3.83e-11 |
| random weight factorisation | 1.80e-5 | 1.9× better | 6.94e-10 |
| *baseline* (plain MLP, soft gauge) | 3.48e-5 | — | 2.86e-09 |
| width 128 instead of 64 | 4.42e-5 | 1.3× worse | 1.26e-09 |
| gradient-norm weighting | 4.95e-5 | 1.4× worse | 7.65e-09 |
| inlet lifting | 5.65e-5 | 1.6× worse | 1.63e-09 |
| Fourier features | 8.46e-5 | 2.4× worse | 2.31e-08 |
| viscous scaling (wrong regime) | 6.11e-4 | 18× worse | 4.15e-07 |

Every run terminated on its L-BFGS iteration cap rather than on `gtol`, so all
ten are budget-limited and the ranking, not the absolute values, is the result.

**Four of the SPEC's suggested enhancements measured harmful on this problem**
(Fourier features, gradient-norm weighting, inlet lifting, extra width). They
are disabled in the committed configs with the measurement recorded beside
each disabled option, so they are not re-enabled speculatively. None of this
generalises to Phase 2 — see section 7.

### 6.2 Three variants lowered the loss while worsening the answer

| variant | final loss | u relL2 |
|---------|-----------|---------|
| baseline | 2.86e-9 | 3.48e-5 |
| inlet lifting | **1.63e-9** (lower) | **5.65e-5** (worse) |
| lifting + hard gauge | **3.83e-11** (lowest) | 8.08e-6 (worse than hard gauge alone at 8.84e-11 / 4.32e-6) |

The mechanism is mundane and worth stating because it is easy to trip over:
lifting removes the inlet loss term, so the total is a sum over five terms
instead of six. Part of the "improvement" is a deleted term.

**Loss values are not comparable across configurations with different active
term sets.** This is a sharper statement than "loss is a diagnostic, never an
accuracy claim" — it bites when comparing two runs of the same code on the same
problem. Phase 2 will vary configurations far more, so this gets worse.

### 6.3 Gradient-norm weighting fails by construction here

The scheme sets `w_i ← (Σ_j ‖∇L_j‖) / ‖∇L_i‖`, so a term the optimiser has
*already satisfied* has a vanishing gradient norm and therefore an exploding
weight. Final weights on the Case A run:

```
outlet_velocity  6325.8      ← du/dx = 0, satisfied identically by the exact solution
momentum_y         24.7
momentum_x         10.8
pressure_gauge      6.0
continuity          5.0
inlet               2.2
```

The optimiser then spent its budget on a solved constraint. Fixed weights are
the default everywhere as a result. The implementation stays, tested, because
the SPEC offers the scheme conditionally and Phase 2 may have a genuine
imbalance; the weight trace is logged per step and plotted in every run's
`loss_history.png`, which is what made this diagnosable at all.

### 6.4 Pressure scaling costs 18× at Re = 10

Viscous and inertial scaling are the same equation multiplied by `Re`, with
identical solutions, so the entire 6.11e-4 vs 3.48e-5 gap is conditioning.
This is the largest single effect in the sweep apart from the gauge, and it
confirms the SPEC's insistence that the choice be a config option rather than
hard-coded.

### 6.5 Case A at the selected configuration

Modified MLP, hard pressure gauge, Re = 10, width 128, Adam 2000 + L-BFGS 6000:

| SPEC test | measured | target | |
|-----------|----------|--------|-|
| 1 — `u` relative L2 | **1.218e-06** | ≤ 1e-5 | **PASS** (stretch target 1e-6) |
| 2 — `max\|v*\|` | **7.929e-07** | ≤ 1e-5 | **PASS** |
| 3 — flux conservation | 2.227e-06 | ≤ 1e-6 | **FAIL** |

Also: `p` relative L2 1.080e-06, and the Adam→L-BFGS loss discontinuity is
1.6e-15, so the handoff trap in SPEC section 9 is measured absent rather than
assumed.

**On the test 3 failure.** L-BFGS terminated on its iteration cap while still
descending, so the first thing to try is more budget — that is not widening a
tolerance and the tolerance has not been touched. Beyond that, it is worth
noticing that test 3 as specified is the strictest test in the suite: it asks
for station-to-station flux agreement of 1e-6 while test 1 asks for 1e-5 on
`u` itself. Here `u` is accurate to 1.2e-6 and the flux spread is 2.2e-6 — the
same order — so test 3 is not detecting a mass-conservation defect so much as
demanding roughly an order of magnitude more accuracy than test 1. That is a
question for the SPEC author, not a licence to move the number.

### 6.6 Accuracy versus Reynolds number

Case A, selected configuration, identical budget at every Re. SPEC section 2
asks for this table and predicts degradation with increasing Re.

| Re | scaling | u relL2 | max\|v*\| | flux cons | p relL2 | required `N_p` |
|----|---------|---------|-----------|-----------|---------|----------------|
| 1 | viscous | 1.65e-4 | 5.02e-5 | 2.26e-4 | 1.28e-4 | 40 |
| 10 | inertial | **1.22e-6** | 7.93e-7 | 2.23e-6 | 1.08e-6 | 4 |
| 50 | inertial | 4.46e-6 | 2.88e-6 | 4.78e-6 | 1.33e-5 | 0.8 |
| 100 | inertial | 1.02e-5 | 3.94e-6 | 4.55e-6 | 2.11e-5 | 0.4 |

The predicted degradation is there from Re = 10 upward: 1.22e-6 → 4.46e-6 →
1.02e-5, with Re = 100 marginally missing test 1's 1e-5. That is the expected
convective stiffening and is a reportable result, not a bug.

**But the trend is not monotonic — Re = 1 is 135× worse than Re = 10 — and the
reason is not the one the SPEC anticipates.**

Re = 1 is the only row the SPEC assigns to viscous scaling, so the obvious
suspicion is a scaling artefact, especially since the sweep measured viscous
scaling as 18× harmful at Re = 10. That suspicion is wrong, and provably so:
`p*_viscous = Re · p*_inertial`, and the momentum coefficients are `(Re, 1)`
for viscous against `(1, 1/Re)` for inertial, so **at Re = 1 the two forms are
the same function**. There is no difference to blame. This is asserted as a
test (`test_the_two_scalings_coincide_exactly_at_reynolds_one`) with a negative
control, rather than argued.

The actual mechanism is in the last column. Under the hard gauge the network
must emit the constant `N_p = |dp*/dx*| · L* = 2L*/Re`, which is **40 at
Re = 1** and 0.4 at Re = 100, while a Glorot-initialised tanh network emits
O(1). Low Re therefore demands a large-magnitude constant output, which is
badly conditioned; high Re demands a stiff convective term. Error tracks the
pressure error at every Re, consistent with section 4's analysis.

**Two competing mechanisms, with an optimum near Re = 10 for this
configuration.** Anyone extrapolating "PINNs get worse as Re rises" from the
SPEC alone would predict Re = 1 to be the easiest case. It is the hardest.

### 6.7 Case B, and the stream-function comparison (SPEC section 3)

Same case, same budget, same architecture; only the continuity treatment
differs. Re = 10, modified MLP, width 128, Adam 2000 + L-BFGS 6000.

| | velocity-pressure | stream function |
|---|---|---|
| flux conservation (test 3) | 3.62e-2 **FAIL** | 8.88e-16 **PASS** |
| entrance length (test 9) | 9.33, 289% error **FAIL** | 2.659, 12.9% error **PASS** |
| outlet profile L2 (test 9) | 3.53e-2 **FAIL** | 4.02e-3 **FAIL** |
| max continuity residual | 2.27e+1 | 2.31e-14 |
| wall time | 2180 s | 5894 s |

**What went wrong in the velocity-pressure run.** The entrance physics is
right: centreline 0.995 at the inlet, 1.472 by `x* = 2`, developed by `x* = 4`.
But the flux grows monotonically down the channel, 2.004 at the first station
to 2.132 at the last -- **6% of mass created from nothing**. Continuity is only
a penalty term there, so nothing forbids it, and the loss gives no hint.

That drift is also what broke the entrance-length measurement. The measurement
is taken where the centreline first reaches 99% of its developed value; with
the centreline still creeping upward, the threshold was not crossed until
`x* = 9.3` against a correlation value of 2.4. **The entrance length was never
an entrance-length problem.** Removing the drift moved it from 289% error to
12.9%, inside the SPEC's 15% tolerance.

**Two honest caveats.**

1. The stream function's 8.88e-16 on test 3 is **not an achievement, it is a
   tautology**: the `ψ_base` construction pins the flux exactly, so mass
   conservation holds by construction and test 3 stops being an independent
   check for this formulation. It remains a real check for velocity-pressure,
   which is precisely where it caught something.
2. Test 9 still **fails** on the outlet profile (4.02e-3 against 1e-4), 8.8×
   better than velocity-pressure but 40× short. L-BFGS terminated on its
   iteration cap with the loss still descending, and the momentum residual
   peaks at 828 localised at the inlet corners, where the uniform-inlet /
   no-slip discontinuity lives. This looks like under-convergence against an
   irreducible corner singularity rather than a formulation defect, but it is
   reported as a failure, not as a near-miss.

**Cost.** The stream function trades one loss term for an extra order of
differentiation (`u_xx` becomes a third derivative of ψ) and ran 2.7× slower.
On this problem the differentiation costs more than the deleted term saves.

## 7. Carried into Phase 2

1. **Use the stream-function formulation for the bifurcation.** Phase 2's
   deliverable is flux splitting between two daughter branches, and the
   velocity-pressure formulation was measured creating 6% of mass out of
   nothing on a straight channel, silently. Whatever a soft continuity penalty
   does at a junction will be worse and harder to see. Budget for the 2.7×
   cost.
2. **SPEC test 4 must not rely on the ansatz.** A `ψ_base` construction that
   pins the parent flux would make parent/daughter conservation true by
   construction, exactly as it made test 3 vacuous here. The Phase 2 flux test
   has to check the *split ratio* between branches, which no ansatz fixes for
   free.
3. **Gauge-fixing is a first-class accuracy lever, not bookkeeping.** The hard
   outlet gauge bought 8× on Case A. It works because it makes the required
   network output a constant; whether an equivalent trick exists for the
   bifurcation depends on the pressure field there and is worth deriving
   before training anything.
4. **Corner singularities dominate the residual.** Both Case B runs peak at
   residuals of several hundred at the inlet corners while the field elsewhere
   is smooth. The bifurcation junction has the same geometry. Expect the loss
   to be dominated by a measure-zero region and plan for it -- residual-based
   adaptive refinement, which Phase 1 did not need, becomes relevant.
5. **Do not re-enable Fourier features, gradient-norm weighting or inlet
   lifting without re-measuring.** All three were harmful here. But Phase 1's
   solution is a low-order polynomial and Phase 2's is not, so these are
   verdicts about Poiseuille flow, not about the methods.
