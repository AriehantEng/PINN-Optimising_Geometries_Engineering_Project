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

## 4. Results

<!-- FILLED FROM runs/ -->

## 5. Carried into Phase 2

<!-- FILLED FROM runs/ -->
