# PINN for branching channel flow — Phase 1

A physics-informed neural network for the 2D steady incompressible
Navier–Stokes equations, building toward geometry optimisation of a 1→2 channel
bifurcation.

**Phase 1 is not a research result.** Plane Poiseuille flow was solved
analytically in the 19th century. The deliverable is a trustworthy harness and
a test suite that will catch silent errors in Phase 2, where no analytic
solution exists.

Read `SPEC-phase1-poiseuille.md` for the specification, `CLAUDE.md` for the
working rules, and `docs/spec-deviations.md` for every place the two diverge
and why.

## Install

```bash
pip install -r requirements.txt
```

JAX runs on CPU here. Nothing in the project assumes a GPU.

**Just want to run it?** Open `PINN_Poiseuille.ipynb` in Google Colab and hit
Runtime → Run all. It is fully self-contained: no clone, no setup, the whole
project is embedded in the first cell. Regenerate it with
`python tools/make_notebook.py` after changing any source file.

**Want to understand the code?** `docs/code-guide.md` walks through every
module, the design decisions and why they were made, what each test rules out,
and the sources behind each implemented technique.

**New here? Read `docs/RUNNING.md`** — it walks through verifying the code,
training a model, and using the trained model, with the exact commands and
where to run them.

## Commands

```bash
python -m src.run --config experiments/<name>.yaml   # run an experiment
pytest tests/ -q                                     # full suite, SPEC targets
pytest tests/ -q -m "not slow"                       # skip training-based tests
pytest tests/ -q --schedule short                    # CI smoke, NOT the gate
python -m src.report --run runs/<id>                 # accuracy table for a run
python -m src.report --all --table                   # every run, one line each
```

`--schedule short` trains a much smaller network for far fewer steps and
asserts documented smoke bounds. It is not the default, so a plain
`pytest tests/ -q` always runs the real SPEC targets.

## Reading the output

Accuracy is **always** the relative L2 error against the analytic solution,
measured on a fixed evaluation grid that is disjoint from the training points.
The loss value is a diagnostic and is never an accuracy claim — the failure
mode this project exists to catch is a low loss with a wrong solution.
`src/report.py` prints the two under separate headings for that reason.

## Layout

```
src/
  physics/      residuals, scaling, analytic solutions, autodiff derivatives
  geometry/     domain, wall distance function phi, collocation sampling
  models/       MLP, Fourier features, modified MLP, RWF, hard-BC ansatz
  training/     losses, weights.py, optimisers, training loop
  evaluation/   L2 metrics, conservation checks, residual maps, plots
  config.py     strict config loading
  run.py        config -> experiment
  report.py     run directory -> accuracy table
experiments/    one YAML per run; no logic
tests/
runs/           written by the code, git-ignored, never edited by hand
docs/
```

`physics/` does not import from `models/` or `training/`.

## Rules that are enforced, not just documented

* **float64 everywhere**, enabled in `src/__init__.py` before any array can be
  created, and asserted at session start in the test suite.
* **No silent config fallbacks.** A missing key raises. An unknown key raises.
  A key that would have no effect under the active choices raises, so a config
  always describes exactly what was run.
* **No-slip is exact**, not a penalty: `u = φ(y) N_u` with `φ = 1 - y²`. Tested
  to be bitwise zero on both walls for arbitrary parameters.
* **All loss weighting lives in `src/training/weights.py`.** Residual functions
  return unweighted residuals.
* **Never widen a tolerance to make a test pass.** If a test fails, it is
  reported as a failure.

## Test suite

| # | Test | File |
|---|------|------|
| 1 | Case A relative L2 on `u` ≤ 1e-5 (**the gate**) | `tests/test_accuracy.py` |
| 2 | Case A `max\|v*\|` ≤ 1e-5 | `tests/test_accuracy.py` |
| 3 | Mass conservation across 10 stations, 1e-6 | `tests/test_accuracy.py` |
| 4 | Parent/daughter flux — Phase 2 placeholder | `tests/test_accuracy.py` |
| 5 | Autodiff vs central finite differences, 1e-7 | `tests/test_autodiff.py` |
| 6 | Method of manufactured solutions | `tests/test_mms.py` |
| 7 | Non-dimensionalisation round trip | `tests/test_scaling.py` |
| 8 | Determinism, 1e-12 | `tests/test_accuracy.py` |
| 9 | Case B profile + entrance length | `tests/test_accuracy.py` |

Test 4 is an `xfail(strict=True)`: it is a Phase 2 placeholder that becomes a
hard failure the moment it starts passing. See `docs/spec-deviations.md` item 3.
