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

<!-- FILLED FROM runs/ -->

## 7. Carried into Phase 2

<!-- FILLED FROM runs/ -->
