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

### 6.0 Official test-suite run

`pytest tests/ -v` on the full schedule, 63 minutes, retraining both gate
configurations from scratch:

```
4 failed, 120 passed, 1 xfailed
```

| SPEC test | result | |
|-----------|--------|-|
| 1 — Case A `u` relative L2 ≤ 1e-5 | **PASS** | 1.22e-6 |
| 2 — Case A `max\|v*\|` ≤ 1e-5 | **PASS** | 7.93e-7 |
| 3 — mass conservation ≤ 1e-6 | **FAIL** | 2.23e-6 Case A, 3.62e-2 Case B |
| 4 — parent/daughter flux | xfail (strict) | Phase 2 placeholder, by design |
| 5 — autodiff vs central differences ≤ 1e-7 | **PASS** | ~2e-9 measured |
| 6 — method of manufactured solutions | **PASS** | |
| 7 — non-dimensionalisation round trip | **PASS** | 4ε |
| 8 — determinism ≤ 1e-12 | **PASS** | with a negative control |
| 9 — Case B profile and entrance length | **FAIL** | 3.53e-2 and 289% |

Six of nine numbered tests pass, one is the Phase 2 placeholder, and two fail.
**Both failures are the same defect**: mass is not conserved in the
velocity–pressure formulation, because continuity there is only a penalty term.
Section 6.7 shows the stream-function formulation removes it entirely.

The failures are reported as failures. No tolerance in this project was
widened, and the two SPEC targets involved (1e-6 and 1e-4) are unchanged from
the specification.

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

### 6.8 Which formulation wins (the SPEC section 3 question, answered)

Case A, Re = 10, identical architecture, sampling, budget and evaluation grid;
the stream function uses the SINE base flow, so the network is not handed the
answer.

| | velocity-pressure | stream function |
|---|---|---|
| test 1 — `u` relative L2 | 1.22e-6 **PASS** | **1.49e-7 PASS** |
| test 2 — `max\|v*\|` | 7.93e-7 **PASS** | **1.07e-7 PASS** |
| test 3 — flux conservation | 2.23e-6 **FAIL** | 9.99e-16 **PASS**, by construction |
| `p` relative L2 | 1.08e-6 | 1.29e-7 |
| max continuity residual | 9.22e-6 | 4.08e-17 |
| wall time | 2016 s | 5492 s |

**The stream function wins by roughly 8× on every genuine accuracy metric, on
both cases, at 2.7× the cost.** On Case A it beats not just the 1e-5 target but
the SPEC's 1e-6 stretch target, by a factor of 6.7. On Case B it is the
difference between an entrance length that passes and one that fails by 289%.

**The gate configuration is nevertheless the velocity–pressure one.** The
stream function passes more tests, but its test 3 pass is vacuous: `ψ_base`
pins the flux, so mass conservation cannot fail there no matter what the
network does. Making that the headline result would be precisely the hollow
green tick this harness exists to prevent — and it was the velocity–pressure
run, where test 3 is a real check, that caught the 6% mass drift in Case B.
A test suite is worth what its hardest honest configuration is worth.

For Phase 2 the recommendation is still to use the stream function (section 7
item 1), with the accompanying warning that SPEC test 4 must then be written to
check the daughter *split ratio* rather than a total flux the ansatz pins for
free.

## 6.9 SPEC v2 Block A: the entrance-length failure was a convention error

Case B, Re_h = 10, re-run under S1 (unified Reynolds conventions), S2 (inlet
amplitude 2/3) and S3 (derived domain length). Same formulation, same budget.

| | SPEC v1 | SPEC v2 A |
|---|---|---|
| entrance length | 9.328 vs 2.356, 289% **FAIL** | **1.738 vs 1.883, 7.7% PASS** |
| flux conservation `D` | 3.62e-2 | 6.66e-3 |
| outlet profile L2 | 3.53e-2 | 7.89e-3 |
| mass behaviour | **gained** 6% (2.004 -> 2.132) | loses 1.2% (1.310 -> 1.295) |
| `L*` | 20 (flat) | 7.82 (derived from Re) |

**The 289% entrance-length miss was a convention mismatch, not a solver error**,
exactly as S1's diagnosis predicted from the signature: one quantity out by
roughly an order of magnitude while everything else derived from the same
solution sat within a few percent. My correlation formula was consistent with
the (8/3) conversion, but the Case B call site passed a bulk velocity of 1.0
where the parabolic mean 2/3 was required -- the unsuffixed-Reynolds bug S1
exists to eliminate. Correcting it moves the target from 2.356 to 1.883.

An earlier result in this document was wrong because of it: the SPEC v1
stream-function Case B run was reported as passing test 9's entrance length at
12.9%. Against the corrected correlation the same measurement is 41.2%, a
FAIL. That claim is retracted.

The sign of the mass error also flipped. Under SPEC v1 the flow CREATED 6% of
mass; it now loses 2.26% relative to nominal, which is the corner-singularity
inlet deficit and a physically sensible direction. The profile-to-mass error
ratio is 1.19, so by S2's own diagnostic the residual profile error is still
essentially the mass drift -- which is what S4 addresses.

Test 3 and test 9's profile criterion still fail, at 6.66e-3 and 7.89e-3.

## 6.10 SPEC v2 Blocks B and C: Case A under both formulations

Re_h = 10, identical architecture, sampling, budget, derived domain (L* = 7.82)
and evaluation grid. The psi-omega column uses the SINUSOIDAL lift, so the
network is not handed the answer.

| | velocity-pressure | psi-omega (sinusoidal) |
|---|---|---|
| test 1 — `u` relative L2 | 8.638e-7 **PASS** | 7.376e-7 **PASS** |
| test 2 — `max\|v*\|` | 1.709e-6 **PASS** | 1.436e-6 **PASS** |
| test 3 — mass drift `D` | 1.622e-6 **FAIL** | 8.327e-16 **PASS**, vacuous |
| `p` relative L2 | 1.429e-6 | recovered post hoc |
| wall time | 1589 s | 1684 s |

Both formulations now beat not just the 1e-5 target but the SPEC's 1e-6 stretch
target on test 1.

### Two earlier claims in this document do not survive SPEC v2

**"The stream function wins by roughly 8x on every genuine accuracy metric."**
Retracted. Under the corrected conventions and the derived domain the two
formulations are within 17% of each other on test 1 (7.38e-7 against 8.64e-7).
The v1 gap was inflated by a domain 2.6x longer than S3 requires and by the
convention error in S1, and it compared a different, pure-psi formulation.

**"The stream function costs 2.7x in wall time."** Retracted, and the reason is
a design instruction in S4 that I had not followed. My v1 stream function
substituted `omega = -lap psi` and solved a fourth-order equation in psi, so
`u_xx` was a third derivative of the network. S4's coupled second-order system
in `(psi, omega)` avoids that entirely: **1684 s against 1589 s, 6% slower, not
170%.** The cost argument against the stream function was an artefact of my
formulation, not a property of stream functions.

What survives is narrower and cleaner: at equal accuracy and equal cost, the
psi-omega system conserves mass exactly, and the price is that test 3 stops
being able to fail.

### Block C: the test 3 miss is under-convergence, not formulation bias

Fitting `D(n) = D_inf + a n^-b` to the 41-point L-BFGS trace:

```
D_inf    = 1.466e-06  +-  7.917e-06
exponent = 0.424
D(n)     : 2.73e-4 (n=1) -> 4.56e-6 (1800) -> 1.78e-6 (5400) -> 1.62e-6 (6000)
```

**The correct verdict is that this trace settles nothing.** The standard error
is 5.4x the estimate and, more to the point, `+-7.9e-6` is larger than the 1e-6
tolerance the test applies -- so the fit cannot distinguish `D_inf = 0` from a
real bias an order of magnitude above the target. `fit_drift` now refuses to
rule either way in that regime:

> the trace cannot resolve D_inf at the 1e-06 scale: the fitted uncertainty is
> +-7.9e-06, larger than the tolerance being tested. Any verdict here would be
> an over-claim -- run longer before concluding anything

An earlier version of this section reported the verdict as "consistent with
zero, i.e. under-convergence". That was an over-claim: it is what the
two-sigma test says, but a two-sigma test against zero is not informative when
the uncertainty exceeds the quantity being tested. The qualitative evidence for
under-convergence is real -- `D` is still descending monotonically at the
iteration cap -- but it is qualitative, and the fit was not entitled to
promote it to a verdict.

`caseA_re10_extended` at 5x the L-BFGS budget is the discriminator S7
prescribes, and a longer trace is exactly what would resolve `D_inf`.

## 6.11 SPEC v2 Block B gate: test 1 under both lifts

S4 requires Case A to pass test 1 under both the cubic and the sinusoidal lift,
with the errors logged separately.

| configuration | test 1 `u` relL2 | test 2 `max\|v*\|` | test 3 drift `D` | wall s |
|---|---|---|---|---|
| velocity-pressure | 8.638e-7 **PASS** | 1.709e-6 | 1.622e-6 **FAIL** | 1589 |
| psi-omega, sinusoidal lift | **7.376e-7 PASS** | 1.436e-6 | 8.33e-16 vacuous | 1684 |
| psi-omega, cubic lift | 1.095e-6 **PASS** | 1.411e-6 | 1.50e-15 vacuous | 1918 |

Gate met: test 1 passes under both lifts. Only the sinusoidal number is quoted
as an accuracy result; the cubic run is the machinery check.

**The cubic lift is slightly WORSE than the sinusoidal one (1.10e-6 against
7.38e-7), and that is the point.** Under the cubic lift the correct answer is
`N_psi = 0`, exactly representable, and the degenerate-target probe shows the
whole Case A suite passes there with the network removed entirely. One might
expect that to show up as a suspiciously good number. It does not — it shows up
as a marginally worse one, because driving a deep tanh network's output to
exactly zero is not easier than fitting a smooth correction, and `omega` still
has to learn `2y` either way through `r_1`.

So **a vacuous configuration is not detectable from its accuracy figure.**
Nothing in the reported numbers distinguishes the run that measures the network
from the run that measures the lift. That is the entire justification for S5's
matrix and for the degenerate-target probe: without them, the cubic result
would sit in a table looking like an ordinary, slightly unlucky data point.

## 6.12 Block A gate complete: Case B under both formulations

Re_h = 10, S1/S2/S3 conventions, identical budget and derived domain.

| | velocity-pressure | psi-omega (sinusoidal) |
|---|---|---|
| test 9 — entrance length | 1.738 vs 1.883, **7.7% PASS** | 1.766 vs 1.883, **6.2% PASS** |
| test 3 — flux conservation | 6.657e-3 **FAIL** | 1.166e-15 **PASS**, vacuous |
| test 9 — outlet profile L2 | 7.893e-3 **FAIL** | 6.392e-3 **FAIL** |
| inlet flux deficit | 2.26e-2 | 5.8e-15 (none) |

**Both formulations pass the entrance-length criterion.** That criterion was a
289% failure under SPEC v1 and it is met either way once S1's convention error
is corrected, so the formulation was never the issue there.

### The S2 diagnostic's hypothesis is refuted

S2 asked for this to be recorded because "it changes what test 9 is measuring".
The reasoning was: if the profile error and the mass error track each other,
then the profile error is mass drift wearing an amplitude disguise, and the S4
re-run collapses test 9's profile failure without S2 doing anything.

The two DID track — ratio 1.19 for velocity-pressure. But driving the mass
drift to **machine zero** (1.166e-15) moved the profile error only from
7.893e-3 to 6.392e-3, a factor of **1.2. Not a collapse.**

So the tracking was coincidental, not causal. The residual outlet-profile error
is an independent error source that survives exact mass conservation, and the
obvious candidate is the inlet corner singularity: a uniform inlet against hard
no-slip walls needs `N_u = 1/(1-y^2)` at `x* = 0`, which is unbounded, and both
Case B runs peak their residuals in the hundreds at those corners.

**Consequence for the SPEC:** test 9's 1e-4 profile criterion cannot be reached
by fixing conservation, and probably cannot be reached at all while the inlet
is a discontinuous uniform profile imposed on a smooth network. Either the
criterion or the inlet condition needs revisiting — that is a decision for the
SPEC author, and the tolerance has not been touched.

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

## `curve_fit` reporting success is not evidence that a fit happened

The 30000-iteration Case A run recorded `D_inf = 1.885e-07 +- 9.1e-07,
consistent with zero: under-convergence`. Every parameter in that fit was
exactly the initial guess: `d_inf` was `drifts[-1]`, `a` was `drifts[0]`, and
the exponent was the literal `0.5` from `p0`. The fitted curve missed its own
final data point by a factor of nine and the trace as a whole by 14.5x.

`fit_drift` posed an unweighted least-squares problem over data spanning three
and a half decades, from 2.7e-4 down to 1.9e-7. The cost at the initial guess
was therefore around 1e-9, already below `curve_fit`'s default `ftol` of 1e-8,
so the optimiser terminated immediately and reported convergence. Both guards
downstream passed it: `identifiable` computes its decay range as
`(last/first)**exponent`, and with the exponent still at its initial 0.5 that
came to 173, sailing past the 10x cut. A guard derived from a fitted parameter
cannot detect that the parameter was never fitted.

Two changes. `curve_fit` now weights each residual by its own `D`
(`sigma=drifts, absolute_sigma=False`), which makes the residuals O(1) and is
also the right statistical choice, since the trace is meaningful to constant
relative rather than constant absolute precision. And `DriftFit` now measures
`max |model - D| / D` across the trace and refuses any verdict above 2.0,
which catches a non-tracking fit however it arose.

Relative weighting turned out to change more than the degenerate case. The
identifiability guard had been calibrated on noiseless synthetic traces, where
the unweighted fit was 33% out at b = 0.4 and a factor of 25 out at b = 0.3.
Weighted, those same noiseless traces come back exact to nine significant
figures, so that calibration was measuring the fitter's bug rather than any
property of the data. Identifiability is a property of NOISY data, and
recalibrated on 10% multiplicative scatter the guard is still earning its
place: median relative error in `D_inf` is 801% at decay 4.5 and 218% at 7.1,
against 33% at 29 and 1.6% at 197. Across 240 noisy synthetic traces spanning
four exponents and three true asymptotes, the guard chain drew zero false
verdicts -- declining as unidentifiable or unresolved in every case it could
not measure.

Note the guards catch different things and both are needed: at decay 29 with a
true `D_inf` of 1e-7 the fitted value is still 1040% out, and `identifiable`
admits it. `resolves` is what declines it, because the standard error there is
4.4e-6 against the 1e-6 tolerance.

The conclusion of Block C is unchanged, and now rests on something better than
the fit. Refitted properly, the 6000-iteration run gives `D_inf = 2.3e-22 +-
8.3e-07` and the 30000-iteration run `D_inf = 2.5e-23 +- 1.25e-07` -- both
asymptotes indistinguishable from zero, with the longer trace tightening the
bound 6.6x. The direct measurement is independent of any fit at all: D fell
from 1.622e-6 at 6000 iterations to 1.885e-7 at 30000, monotonically in
budget. The residual drift is under-convergence, not formulation bias.

Refitting the stored traces also corrected two runs that predate the
`at_machine_precision` guard. Both psi-omega runs had recorded `D_inf =
1.000e-10 +- 4.9e-11 lies below 1e-06: more iterations should reach the
target` -- a convergence claim about a formulation that conserves mass by
construction and whose D is exactly zero. They now read VACUOUS, which is what
the falsifiability matrix requires. `src/refit_drift.py` recomputes
`diag_drift_*` from a stored `drift_trace.csv`; the traced D(n) is measured
rather than fitted, so no retraining was needed.
