# Conventions

Written down because a Reynolds-number convention mismatch produced a wrong
entrance-length estimate once, and that class of error is invisible until the
conventions are on paper.

## "Inertial" means two different things in this code

Both of these are set in the `physics:` block and both get called "inertial" in
conversation. They are unrelated.

| Key | Values | What it actually changes |
|---|---|---|
| `pressure_scaling` | `inertial` / `viscous` | **Non-dimensionalisation only.** `inertial` uses p_ref = rho U_max^2, `viscous` uses p_ref = mu U_max / h. The physical solution is identical; only the conditioning of the loss differs. Related by p*_viscous = Re . p*_inertial. Choosing `inertial` does NOT switch inertia on. |
| `convective_coefficient` | `1.0` / `0.0` | **The inertial terms themselves.** Multiplies u.grad(u) in both momentum residuals. `1.0` solves Navier-Stokes; `0.0` solves Stokes. This is the only switch that changes the equations. |

So to make the solver inertial:

```yaml
physics:
  convective_coefficient: 1.0    # Navier-Stokes
  # convective_coefficient: 0.0  # Stokes
```

## Why Case A cannot show you inertia

Every Case A run already has `convective_coefficient: 1.0`, and every Case A
run before the switch existed included the convective term unconditionally.
The terms are there. They contribute exactly nothing, because the Case A
solution annihilates them:

    u* = 1 - y*^2,  v* = 0
    u . du/dx = u* . 0 = 0        (u* has no x-dependence)
    v . du/dy = 0 . du*/dy* = 0   (v* is identically zero)

So u.grad(u) = 0 pointwise, and the Navier-Stokes and Stokes residuals are the
same function on this solution. `test_the_switch_is_a_no_op_on_the_case_a_
solution` in `tests/test_autodiff.py` asserts exactly this, and it is the
cheapest check that the coefficient is wired to the right term.

Demonstrating inertia needs a flow where v* != 0 and du*/dx* != 0 -- that is
Case B, the developing entrance flow. The measurement is an ablation: the same
config at 1.0 and at 0.0, compared on entrance length. The creeping-flow limit
of the entrance-length correlation is the c0 term and the inertial part is the
c1.Re term, so a Stokes run should flatten to a constant as Re rises while a
Navier-Stokes run grows linearly.

## Reynolds number

| Symbol | Definition | Used by |
|---|---|---|
| Re_h | U_max . h / nu | the solver, every config |
| Re_Dh | U_mean . D_h / nu | published correlations |

    Re_Dh = (8/3) Re_h        D_h = 4h        U_mean = (2/3) U_max

D_h = 4h because D_h = 2H for a wide plane channel and H = 2h is the FULL
height. Getting this factor of 2.67 wrong is small enough to look like a
modelling discrepancy and large enough to invalidate a comparison.

## Lengths, velocities, pressure

| Quantity | Scaled by | Consequence |
|---|---|---|
| x*, y* | the HALF-height h | walls at y* = +-1, half-height 1, full height 2 |
| u* | U_max | developed centreline is exactly 1 |
| flux* | -- | 4/3 for the developed profile; a UNIFORM profile of the same flux sits at u* = 2/3 |
| L* | derived | from the entrance-length correlation, never set by hand |

## The distance function

Analytic, not numerical:

    phi(y*) = 1 - y*^2              velocity-pressure
    phi(y*)^2 = (1 - y*^2)^2        psi-omega (needs zero slope too)

`u = phi . N_u` makes no-slip exact for any network output, so it is never a
loss term. The half-height is 1 **by construction of the non-dimensionalisation**
and is not configurable -- which is why Phase 2's (x, y, theta) input needs the
scaling reworked, not just an extra input added.

## Entrance-length correlation

    L_e/D_h = [c0^n + (c1 Re_Dh)^n]^(1/n)      c0 = 0.3155, c1 = 0.01105, n = 1.6

Converted from the published plane-channel fit C0 = 0.631, C1 = 0.0442 per FULL
height H with Re_H on H and mean velocity. The two divisors differ (C0/2 and
C1/4) and that is **correct**, not a unit error: one factor of 1/2 comes from
the length scale (D_h = 2H) and another from the Reynolds definition
(Re_H = Re_Dh/2), and pushing the outer 1/2 inside the bracket gives

    (1/2)[C0^n + (C1 Re_H)^n]^(1/n) = [(C0/2)^n + (C1 Re_Dh/4)^n]^(1/n)

Verified to machine precision across seven decades of Re. Cross-check: the
high-Re asymptote L_e/D_h = 0.01105 Re_Dh reproduces the classical plane-channel
value 0.011 Re_Dh.
