# How to run this, end to end

Three stages, in order: **verify the code is correct**, **train a model**, then
**use the trained model**. Each has one command.

---

## 0. Where to run it

| Where | Verdict |
|---|---|
| **Any laptop, CPU** | Works. A full Case A run is ~26 min on 4 cores. This is how every result in `docs/phase1-findings.md` was produced. |
| **Google Colab, free (T4)** | Works, but **may be no faster than a laptop.** Everything here is float64, and a T4's FP64 throughput is 1/32 of its FP32. Measure before believing. |
| **Colab Pro (A100/V100)** | Genuinely faster: those have real FP64 units. |
| **A GPU box** | Best. Install `jax[cuda12]` instead of plain `jax`. |

Nothing in the code assumes a GPU. float64 is non-negotiable — float32 caps
achievable error near 1e-4, which is above every target in the SPEC.

### Colab setup

```python
!git clone -b claude/session-start-9vssgw \
  https://github.com/AriehantEng/PINN-Optimising_Geometries_Engineering_Project.git
%cd PINN-Optimising_Geometries_Engineering_Project
!pip install -q optax pyyaml
```

Note the `-b` flag: **all the work is on that branch**, `main` is empty.
The `%cd` matters — every command below assumes the repo root.

Results live in `runs/`, which is git-ignored and dies with the Colab VM, so
mount Drive first if you want to keep them:

```python
from google.colab import drive; drive.mount('/content/drive')
```

---

## 1. Verify the code is correct (~90 seconds)

```bash
python -m pytest tests/ -q -m "not slow"
```

234 tests, no training required. These check the physics operators against
hand-derived forcings, the autodiff against finite differences, the scaling
round trip, the boundary-condition ansatz, and the falsifiability matrix.

If this fails, do not train anything — the result would be meaningless.

Optional, slower:

```bash
python -m pytest tests/ -q -m slow --schedule short   # ~3 min, smoke bounds
python -m pytest tests/ -q                            # ~60 min, the real SPEC targets
```

---

## 2. Train a model (~26 min for the default)

```bash
python -m src.run --config experiments/caseA_re10.yaml
```

You will see live progress:

```
  [adam] 2000 steps, resampling every 500 (one 'epoch' = one resampling interval, 4 of them)
      step  epoch         loss  rel L2 vs exact        lr
         0      0   6.6492e-01       1.0070e+00  1.00e-03
       300      0   7.4260e-03       6.2084e-01  8.54e-04
       ...
  [lbfgs] up to 6000 iterations
      iter                loss  rel L2 vs exact   drift D
         1          4.0066e-03       4.5261e-01  6.00e-01
       ...
```

### Reading the columns

* **step / iter** — optimiser steps. Adam first, then L-BFGS.
* **epoch** — PINNs have no epochs in the usual sense: there is no training set
  to pass through. The nearest equivalent is a *resampling interval*, i.e. how
  many times the collocation points have been redrawn, so that is what this
  column counts.
* **loss** — the weighted sum of squared residuals. **This is a diagnostic, not
  an accuracy claim.** It can fall while the answer gets worse.
* **rel L2 vs exact** — relative L2 error against the analytic Poiseuille
  solution. **This is the accuracy number.** Watch this one.
* **drift D** — mass created or destroyed along the channel (SPEC v2 S7).

Watching those two middle columns diverge is the whole point. In a real run the
loss dropped 2.6x between L-BFGS iterations 80 and 120 while the relative L2
error got 2.3x *worse*.

### Choosing a configuration

```bash
ls experiments/
```

| Config | What it is |
|---|---|
| `caseA_re10.yaml` | Fully developed channel. Has an exact solution, so accuracy is measurable. **Start here.** |
| `caseA_re{1,50,100}.yaml` | Same, across Reynolds number. |
| `caseB_re10.yaml` | Developing entrance flow. The case that predicts Phase 2. |
| `caseA_re10_psiomega_sinusoidal.yaml` | Stream-function/vorticity formulation. |
| `caseA_re10_psiomega_cubic.yaml` | **Machinery check only** — its lift *is* the exact answer, so never quote its accuracy. |

To go faster while experimenting, cut the budget:

```yaml
training:
  adam:  {steps: 500, ...}
  lbfgs: {max_iterations: 1000, ...}
model: {width: 64, depth: 3, ...}
```

### What a finished run leaves you

```
runs/caseA_re10/
  metrics.json      every accuracy number and diagnostic
  params.npz        the trained weights
  config.yaml       exactly what produced them
  history.csv       loss + accuracy per logged Adam step
  drift_trace.csv   mass drift vs L-BFGS iteration
  profiles.png      predicted vs analytic profiles, and the error map
  residuals.png     where the PDE residual lives
  loss_history.png  per-term losses and weights
  environment.json  Python, JAX, platform, float64 flag
```

Read the headline table:

```bash
python -m src.report --run runs/caseA_re10     # one run, with PASS/FAIL per SPEC test
python -m src.report --all --table             # every run, one line each
```

---

## 3. Use the trained model

This is what makes it a model rather than an experiment.

```bash
python -m src.predict --run runs/caseA_re10 --at 3.0,0.5
python -m src.predict --run runs/caseA_re10 --profile-at 3.0 --out profile.csv
python -m src.predict --run runs/caseA_re10 --grid 201 101 --out field.csv
python -m src.predict --run runs/caseA_re10 --flux-at 3.0
```

From Python or a Colab cell:

```python
from src.predict import load_run

solution = load_run("runs/caseA_re10")
print(solution.summary())

u, v = solution.velocity(3.0, 0.5)      # velocity at one point
p     = solution.pressure(3.0, 0.5)     # pressure, gauged to 0 at the outlet
y, u, v = solution.profile(x=3.0)       # a cross-channel profile
q     = solution.flux(3.0)              # flux through a station
x, y, u, v = solution.grid(201, 101)    # the whole field
```

Plot it:

```python
import matplotlib.pyplot as plt
from src.physics import analytic

y, u, _ = solution.profile(x=3.0)
plt.plot(u, y, label="PINN")
plt.plot(analytic.u_nd(y), y, "k--", label="analytic")
plt.xlabel("u*"); plt.ylabel("y*"); plt.legend()
```

### Two things `predict` refuses to do quietly

* **Evaluate outside the trained domain.** A PINN is fitted, not solved; outside
  `x* in [0, L*]`, `y* in [-1, 1]` it is extrapolating and its output solves
  nothing. Pass `allow_extrapolation=True` if you mean it.
* **Report a pressure the model does not have.** The psi-omega system has no
  pressure in it, so `pressure()` recovers one by path integration rather than
  returning the placeholder zero in the field's third slot.

---

## 4. Is it accurate? Compared to what?

"The original model" here is the **analytic solution**, not another network.
Plane Poiseuille flow was solved in the 19th century; that closed form is the
oracle, and accuracy is always relative L2 against it.

Current results (`docs/phase1-findings.md` has the full table):

| SPEC test | target | measured |
|---|---|---|
| 1 — `u` relative L2 | ≤ 1e-5 | **8.64e-7** (beats the 1e-6 stretch target) |
| 2 — `max\|v*\|` | ≤ 1e-5 | **1.71e-6** |
| 3 — mass drift `D` | ≤ 1e-6 | 1.62e-6 — **fails**, characterised as under-convergence |
| 9 — Case B entrance length | ≤ 15% | **7.7%** |

Do not read the loss as accuracy. `src/report.py` prints the two under separate
headings for exactly that reason.
