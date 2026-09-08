# What this program does, and why each part exists

A guide to all 4,644 lines of `src/` and 2,336 lines of `tests/`, for someone
who needs to understand and defend the whole thing.

---

## 1. The physical problem

**Plane Poiseuille flow.** Steady, incompressible, viscous fluid between two
parallel walls, driven along the channel by a pressure gradient. The walls are
at `y = ±h`; the flow is along `x`.

The governing equations are the 2D steady incompressible Navier–Stokes
equations:

```
(u · ∇)u = −∇p + (1/Re) ∇²u        momentum
∇ · u    = 0                        continuity (mass conservation)
```

with no-slip at the walls (`u = v = 0` at `y = ±h`).

**This problem has an exact analytic solution**, known since the 19th century:

```
u(y) = (G/2μ)(h² − y²)        a parabola
v    = 0                       no transverse motion at all
Q    = 2Gh³/(3μ)               flux per unit depth
```

Non-dimensionalised on the half-height `h` and the centreline velocity `U_max`,
that becomes simply `u* = 1 − y*²`, `v* = 0`.

**That exactness is the entire point of Phase 1.** The project's real target is
Phase 2: flow through a 1→2 channel bifurcation, where no analytic solution
exists and there is nothing to check an answer against. So Phase 1 builds the
solver on a problem where the answer IS known, and proves the machinery
correct there first.

---

## 2. What a PINN is, and the one thing that makes it dangerous

A **physics-informed neural network** represents the solution as a neural
network `u(x, y; θ)` and trains it by minimising the *residual of the governing
equations themselves*, evaluated at scattered "collocation" points:

```
loss = mean[ (momentum residual)² ] + mean[ (continuity residual)² ] + boundary terms
```

There is no training data. Nothing is fitted to measurements. The derivatives
in the residual come from automatic differentiation of the network with respect
to its *inputs*, which is exact to machine precision — not from a mesh or a
finite-difference stencil. This is why a PINN is mesh-free, which is the
property that makes it attractive for geometry optimisation: change the shape,
and there is nothing to re-mesh.

**The failure mode.** A low loss does not imply a correct answer. The loss
measures how well the equations are satisfied *at the sampled points*, which is
not the same as how close the solution is to the truth. Published work using a
standard PINN toolkit on this exact problem reached a total test loss of 8.7e-5
while its relative L2 error was **8.6%** — a number that looks like success and
is off by four orders of magnitude.

Almost every design decision in this codebase follows from taking that
seriously:

* accuracy is **always** reported as relative L2 error against the analytic
  solution, never as a loss value;
* `src/report.py` prints accuracy and diagnostics under separate headings;
* the training display shows loss and error side by side, so their divergence
  is visible as it happens;
* `docs/falsifiability-matrix.md` records which tests can still fail.

---

## 3. Module walkthrough

### `src/physics/` — the equations (no dependency on networks)

| file | what it does |
|---|---|
| `scaling.py` | SI ↔ non-dimensional transform. Two pressure scalings: viscous (`p* = ph/μU`, for Re ≲ 1) and inertial (`p* = p/ρU²`, Re > 1). They are the same equation multiplied by `Re`, so they have identical solutions and different conditioning. |
| `analytic.py` | The oracle: the exact solution in both SI and non-dimensional form, and the `Q ∝ h³` flux law (2D slot — **not** the `r⁴` of a circular pipe). |
| `residuals.py` | One small function per equation term (`convective_x`, `laplacian`, `divergence`, …), each returning an **unweighted** residual, so each can be unit-tested alone. |
| `derivatives.py` | Assembles the derivative bundle by forward-mode autodiff. |
| `conventions.py` | Reynolds-number bookkeeping: `Re_h` (solver: half-height, centreline velocity) versus `Re_Dh` (correlations: hydraulic diameter, mean velocity). `Re_Dh = (8/3) Re_h`. |
| `streamfunction.py` | The ψ–ω formulation and post-hoc pressure recovery. |
| `optimum.py` | The Phase 3 analytic optimum, `h₀/h₁ = √2`. |

### `src/geometry/` — where the equations are enforced

`domain.py` defines the channel and the wall-distance function `φ(y) = 1 − y²`.
`sampling.py` draws collocation points; `lifts.py` holds the two stream-function
base flows.

### `src/models/` — what the network is allowed to represent

`dense.py`, `mlp.py`, `fourier.py` build the networks. `hard_bc.py` is the
important one: it constructs the **ansatz**.

### `src/training/` — how it is fitted

`losses.py` (one mean-squared term per equation and boundary condition),
`weights.py` (**the only place weighting happens**), `optimisers.py` (Adam then
L-BFGS), `loop.py` (the loop, live progress, held-out monitoring, checkpoints).

### `src/evaluation/` — whether to believe it

`metrics.py` (relative L2, Gauss–Legendre flux), `accuracy.py` (per-case
metrics), `drift.py` (mass-drift characterisation), `falsifiability.py` (which
tests can still fail), `plots.py`.

### Top level

`config.py` (strict loading — a missing key raises, an unknown key raises, and
a key that would have no effect raises), `run.py` (config → run directory),
`report.py` (run → accuracy table), `predict.py` (use a trained model).

---

## 4. The design decisions that matter

### Hard boundary conditions

No-slip is **not** a penalty term. The velocity is constructed so it cannot
violate the wall condition:

```
u*(x*, y*) = φ(y*) · N_u(x*, y*),    φ(y*) = 1 − y*²
```

`φ` vanishes at both walls, so `u = 0` there for *any* network output. Verified
bitwise zero. This removes a loss term entirely and is the single largest
accuracy lever available.

**The cost, which must be reported:** a constraint imposed by construction
cannot be tested. The wall tests became vacuous the moment this was adopted.
That is what `docs/falsifiability-matrix.md` exists to record.

### The pressure gauge — worth 8× on its own

Pressure is only defined up to a constant, so it must be pinned. Substituting
the exact solution into the x-momentum residual leaves

```
R_x = p_x + 2/Re
```

so the residual demands an **exactly linear** pressure field. Under a soft
gauge, `N_p` must be affine in `x`, which a tanh network cannot be exactly.
Under the hard gauge `p = (1 − x*/L*) N_p`, the required `N_p` is the *constant*
`2L*/Re`, which is exactly representable. Measured: **3.48e-5 → 4.32e-6.**

### Two continuity formulations

* **velocity–pressure**: continuity is a penalty term. It can fail — and it
  caught Case B silently creating 6% of mass out of nothing.
* **ψ–ω**: `u = ∂ψ/∂y`, `v = −∂ψ/∂x` satisfies continuity identically, solved
  as two coupled *second-order* equations in `(ψ, ω)` rather than one
  fourth-order equation in `ψ` — fourth-order autodiff through `tanh` would set
  the precision floor.

Mass conservation then holds by construction, which makes the mass test
**vacuous** for that formulation. Both are kept; velocity–pressure is the
falsifiability probe.

### Adam then L-BFGS

Adam (first-order, robust, handles the early chaotic phase) followed by L-BFGS
(quasi-Newton, converges hard near the optimum). The handoff is a known trap;
this code measures the loss on both sides of it. Measured discontinuity:
**~1e-16**, i.e. none.

Collocation points are **frozen** during L-BFGS: resampling underneath a
quasi-Newton method invalidates its curvature estimate.

### float64 everywhere

Enabled at package import, before any array exists. float32 caps achievable
accuracy near 1e-4 — worse than every target here — and the symptom looks
exactly like an optimiser failure.

---

## 5. The test suite: what each test rules out

| # | test | what a pass rules out |
|---|---|---|
| 5 | autodiff vs 4th-order central differences | a wrong derivative in the residual operator |
| 6 | method of manufactured solutions | a wrong sign or coefficient in any term |
| 7 | SI ↔ non-dimensional round trip | a dropped factor in the scaling |
| 1 | Case A relative L2 on `u` ≤ 1e-5 | **a low loss with a wrong solution** |
| 2 | `max\|v*\| ≤ 1e-5` | a solution that fits `u` but not the physics |
| 3 | mass conservation ≤ 1e-6 | a locally plausible, globally wrong field |
| 8 | determinism ≤ 1e-12 | irreproducible results |
| 9 | Case B profile and entrance length | inability to handle redeveloping flow |

**Test 6 (manufactured solutions)** is worth understanding: pick an arbitrary
smooth field, substitute it into the equations by hand, and check the code
reproduces that forcing. The code reaches the answer by autodiff; the test
reaches it by pencil-and-paper differentiation. Two independent routes.

**The sentinel suite** (`tests/test_sentinels.py`) asks a different question of
every test: *does an untrained, randomly initialised network fail it?* If not,
the test measures the ansatz rather than the solution.

---

## 6. Results

| SPEC test | target | measured |
|---|---|---|
| 1 — `u` relative L2 | ≤ 1e-5 | **8.64e-7** (beats the 1e-6 stretch goal) |
| 2 — `max\|v*\|` | ≤ 1e-5 | **1.71e-6** |
| 3 — mass drift | ≤ 1e-6 | 1.62e-6 — **fails** |
| 9 — entrance length | ≤ 15% | **7.7%** |
| 9 — outlet profile | ≤ 1e-4 | 7.89e-3 — **fails** |

`v* = 0` is *discovered*, never imposed: nothing in the loss says the transverse
velocity should vanish. It follows from the equations, and comes out at 1.7e-6.

---

## 7. Sources

### Directly implemented

**Physics-informed neural networks.** Raissi, Perdikaris & Karniadakis (2019),
"Physics-informed neural networks: A deep learning framework for solving forward
and inverse problems involving nonlinear partial differential equations",
*Journal of Computational Physics* 378:686–707. The originating method.

**Entrance-length correlation** — `physics/conventions.py`, and the `citation`
field every config must carry. Durst, Ray, Ünsal & Bayoumi (2005), "The
development lengths of laminar pipe and channel flows", *Journal of Fluids
Engineering* 127(6):1154–1160.

> ⚠️ **Verify this one against the paper before publishing.** The code uses
> `c0 = 0.3155`, `c1 = 0.01105`, `n = 1.6` in *hydraulic-diameter* units. Those
> are **my conversion** of the published plane-channel constants (0.631, 0.0442,
> 1.6), which are stated per full channel height, via `c0 = C0/2`, `c1 = C1/4`.
> The conversion is cross-checked against the classical high-Re asymptote
> `L_e/D_h ≈ 0.011 Re_Dh` and agrees, but the arithmetic is mine and a wrong
> constant here is invisible in the output.

**Adam.** Kingma & Ba (2015), "Adam: A Method for Stochastic Optimization",
ICLR. Used via `optax`.

**L-BFGS-B.** Byrd, Lu, Nocedal & Zhu (1995), "A limited memory algorithm for
bound constrained optimization", *SIAM Journal on Scientific Computing*
16(5):1190–1208. Used via `scipy.optimize.minimize`.

**Glorot initialisation.** Glorot & Bengio (2010), "Understanding the difficulty
of training deep feedforward neural networks", AISTATS.

**Random Fourier features.** Tancik et al. (2020), "Fourier Features Let
Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS.
Implemented, and **measured harmful here** (2.4× worse) because this solution is
a low-order polynomial.

**Gradient-norm loss weighting.** Wang, Teng & Perdikaris (2021), "Understanding
and mitigating gradient flow pathologies in physics-informed neural networks",
*SIAM Journal on Scientific Computing* 43(5):A3055–A3081. Implemented, and
**measured harmful here**.

**Modified MLP and random weight factorisation.** Wang, Sankaran, Wang &
Perdikaris (2023), "An Expert's Guide to Training Physics-Informed Neural
Networks", arXiv:2308.08468, and the associated JAX-PI codebase. The modified
MLP is the second-largest accuracy lever found (5×).

**JAX.** Bradbury et al. (2018), "JAX: composable transformations of
Python+NumPy programs".

### Classical fluid mechanics

**Poiseuille flow and the `h³` flux law.** Any standard viscous-flow text; White,
*Viscous Fluid Flow*, and Batchelor, *An Introduction to Fluid Dynamics*.

**Entrance length, classical value.** Shah & London (1978), *Laminar Flow Forced
Convection in Ducts*, is the standard compilation; the `L_e/D_h ≈ 0.011 Re_Dh`
asymptote for parallel plates comes from that literature and is used here as the
cross-check on the converted constants.

**Murray's law** (referenced by `physics/optimum.py` for contrast). Murray
(1926), "The physiological principle of minimum work: I. The vascular system and
the cost of blood volume", *PNAS* 12(3):207–214. Murray's cube law is specific
to circular cross-sections; the 2D slot analogue derived here is
`h₀² = h₁² + h₂²`, exponent **2, not 3**.

### Mentioned, deliberately not built

**PirateNets.** Wang, Li, ... & Perdikaris (2024), "PirateNets: Physics-informed
Deep Learning with Residual Adaptive Networks", arXiv:2402.00326. The SPEC gates
this on "only if 1–4 leave you short of the accuracy target". They did not.

### On confidence

Author lists, years, journals and volume/page numbers above are given from
memory and should be checked against the actual papers before they go into a
report — particularly the Wang et al. entries, where that group has several
closely-related papers and it is easy to attribute a technique to the wrong one.
The two things I am confident about because they are load-bearing in the code
and were verified numerically are the Durst *conversion arithmetic* (checked
against the classical asymptote) and the `Q ∝ h³` / `h₀² = h₁² + h₂²` exponents
(derived two independent ways, and asserted as tests).
