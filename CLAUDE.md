# PINN for branching channel flow

## What this project is

A physics-informed neural network solving the 2D steady incompressible
Navier–Stokes equations, building toward geometry optimisation of a 1→2 channel
bifurcation. Phase 1 (current) is straight-channel Poiseuille flow.
Read `SPEC-phase1-poiseuille.md` before starting work.

## Commands

- Run an experiment: `python -m src.run --config experiments/<name>.yaml`
- Full test suite: `pytest tests/ -q`
- Fast subset (skips training-based tests): `pytest tests/ -q -m "not slow"`
- Accuracy report for a finished run: `python -m src.report --run runs/<id>`

## Non-negotiable numerical rules

- **IMPORTANT: float64 everywhere.** Enable double precision before any array is
  created. float32 silently caps achievable accuracy near 1e-4 and the symptom
  looks exactly like an optimiser failure, which wastes days.
- All physics is expressed in non-dimensional variables. Never put an SI-valued
  viscosity, density or channel height into a loss term.
- **Never widen a tolerance or lower a target error to make a test pass.** If a
  test fails, report the failure and stop. This rule exists because a PINN that
  reaches low loss with a wrong solution is the expected failure mode here.
- Accuracy is reported as relative L2 error against the analytic solution.
  Loss value is a diagnostic, never an accuracy claim.
- Every run is fully determined by its config file plus its seed. No
  hyperparameters hard-coded in source, no defaults that silently apply.

## Code style

- Residual functions return unweighted residuals. All loss weighting happens in
  one place (`src/training/weights.py`).
- No silent fallbacks. A missing config key raises rather than defaulting.
- Physics functions are pure and take non-dimensional inputs. No module globals.
- Prefer one small function per governing equation term over one large residual
  function, so terms can be unit-tested independently.

## Gotchas found so far

<!-- Append entries here as we hit them. Format: symptom → cause → fix. -->
