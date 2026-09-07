# Deviations from SPEC-phase1-poiseuille.md, and why

Every entry here is a place where the SPEC as written could not be implemented
literally, or where implementing it literally would have produced a test that
does not test what it claims to. Nothing here relaxes an accuracy target.

---

## 1. Case B's inlet value and its developed profile are inconsistent

**SPEC section 2, Case B** says the inlet is `u* = 1` uniform, and that "the
downstream profile must converge to the Case A analytic profile" (`u* = 1 - y*²`).

These cannot both hold. Mass is conserved, so the flux at the inlet equals the
flux downstream:

```
inlet (uniform, full height 2):   Q* = 2 * 1        = 2
Case A parabola:                  Q* = ∫(1-y²)dy    = 4/3
```

A uniform inlet of magnitude 1 develops into `u* = (3/2)(1 - y*²)`, whose
centreline is 1.5, not 1. The two statements differ by exactly the factor 3/2.

The underlying tension is in the non-dimensionalisation: SPEC section 1 defines
`U_max` as "the centreline velocity of the fully developed profile", so scaling
by `U_max` forces the *developed* centreline to be 1 and the uniform inlet to be
`2/3`. Section 2 instead scales by the inlet velocity.

**What was implemented.** The literal reading of section 2: the inlet value is
config (`physics.inlet_uniform_value`, set to 1.0), and the downstream target is
derived from flux conservation rather than assumed:

```python
developed_profile_nd(y, flux) = (3*flux/4) * (1 - y**2)
```

For Case A (`flux = 4/3`) this reduces exactly to `1 - y*²`, so nothing changes
there. For Case B it gives the correct `(3/2)(1 - y*²)`. Test 9 compares against
this flux-consistent profile.

**If you prefer the other reading**, set `physics.inlet_uniform_value: 0.6667`
and everything else follows automatically — the harness derives the target from
the configured inlet, so no code changes.

**This needs your decision** before the Phase 1 write-up, because it changes
what the reported Case B numbers mean.

---

## 2. Case B at Re = 100 does not fit in the specified domain

**SPEC section 2** fixes `L* = 20` and asks for Case B at `Re ∈ {1, 10, 50, 100}`.

The Durst et al. (2005) channel correlation gives, in units of the half-height:

| Re  | entrance length L_e* |
|-----|----------------------|
| 1   | 1.30                 |
| 10  | 2.36                 |
| 50  | 9.08                 |
| 100 | 17.84                |

At Re = 100 the flow is still developing when it reaches `x* = 20`. The outlet
condition (`∂u/∂x = 0`) is then *physically wrong*, not merely inaccurate, and
it will contaminate the entrance-length measurement it is supposed to validate.

**What was implemented.** `domain.length` is config, and the Case B experiment
files use `L* = 20` for Re ≤ 50 and `L* = 60` for Re = 100 (about 3.4 entrance
lengths). Case A is unaffected and keeps `L* = 20` throughout.

---

## 3. Test 4 (parent/daughter flux) is an `xfail`, not a hard failure

**SPEC section 5** asks for "a placeholder that fails loudly until implemented",
but the same section also says "every test below must exist and pass before
Phase 2 begins". A test that always fails makes `pytest tests/ -q` permanently
red, which destroys the signal from the tests that *do* mean something — and
this project's whole thesis is that a silent wrong answer is the enemy.

**What was implemented.** `@pytest.mark.xfail(strict=True)`. This is loud in the
report (`XFAIL`, listed by name with its reason), it cannot be forgotten, and
because it is *strict* it turns into a hard failure the moment someone implements
the bifurcation flux check without removing the marker. The test body raises
`NotImplementedError` with the Phase 2 requirements spelled out.

If you want it to be a hard red failure instead, delete the marker — the test
body already fails on its own.

---

## 4. Test 5 uses the five-point central stencil

**SPEC section 5** asks that residual derivatives match "central finite
differences to 1e-7".

The three-point central second difference has a roundoff floor of about
`4 ε |f| / h²`, and its `O(h²)` truncation error forces `h ≲ 1e-4`. Its best
achievable accuracy is therefore about `1e-7 |f|`. On fields of magnitude ~5 —
which is what an untrained network produces — that floor sits *above* the 1e-7
target, so the test measures the finite-difference method rather than the
autodiff it exists to check. That is how it first failed here.

**What was implemented.** The five-point (fourth-order) central stencil, still a
central difference, at `h = 1e-3`. Measured agreement is ~2e-9 across every
architecture, about 50× below target, so a real derivative error is caught with
margin. **The 1e-7 tolerance is unchanged.**

---

## 5. "Exact to machine precision" for the round trip means a few ulp

**SPEC section 1** asks that SI → non-dim → SI be "exact to machine precision".

A round trip is one division then one multiplication. Each is correctly rounded,
so the result is within ~2 ulp of the input, but it is *not* bitwise identical in
general (`(x/h)*h != x` for most `x`, `h`).

**What was implemented.** `ROUNDTRIP_RTOL = 4 * eps ≈ 8.9e-16`, derived from that
bound rather than fitted, and named so it cannot be quietly widened.

---

## 6. The stream-function ansatz needs a base flow

**SPEC section 3** gives `φ(y*) = 1 - y*²` for hard no-slip and asks for a stream
function variant.

`ψ = φ² N` satisfies `ψ_x = ψ_y = 0` on both walls, but it also forces
`ψ(+1) = ψ(-1)`, and `ψ(+1) - ψ(-1)` *is* the flux. Such an ansatz can only
represent a channel with zero net throughflow — it cannot represent Poiseuille
flow at all.

**What was implemented.**

```
ψ = ψ_base(y*) + φ(y*)² N_ψ(x*, y*)
```

with `ψ_base` carrying the flux and having zero slope at both walls. Two options:

* `cubic`: `(Q/4)(3y - y³)` — this *is* the exact Case A solution, so the network
  only has to learn `N_ψ = 0`.
* `sine`: `(Q/2) sin(πy/2)` — same flux, different shape, so the network still
  has to work.

**The stream-function vs velocity-pressure comparison in SPEC section 3 is only
meaningful with `sine`.** Reporting the `cubic` result as a win for the stream
function would be reporting the ansatz, not the method. The default is `sine`.

A side effect worth carrying into Phase 2: this ansatz conserves mass *exactly*
by construction, so SPEC test 3 is trivially satisfied for it and is only a real
test of the velocity-pressure formulation.

---

## 7. Input normalisation is automatic, not configurable

CLAUDE.md forbids hyperparameters that are not in the config. Coordinates are
mapped from `[0, L*] × [-1, 1]` to `[-1, 1]²` before entering the network.

This is treated as a structural property of the domain rather than a
hyperparameter: it is fully determined by `domain.length`, it has no tunable
degrees of freedom, and without it the two inputs differ in scale by a factor
`L*`. It is implemented as a named, tested function (`geometry.domain.normalise`)
rather than an inline expression.
