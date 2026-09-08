"""Build a single self-contained Colab notebook from the tested source.

    python tools/make_notebook.py

Writes `PINN_Poiseuille.ipynb`: one file, no clone, no repo layout to arrange.
Open it in Colab, Run All, and it trains a model and uses it.

The notebook does NOT contain a hand-written copy of the physics. Its first
cell carries the source files VERBATIM as a dict and writes them to disk, so
what runs in the notebook is byte-for-byte what the test suite covers. A stale
copy is impossible to produce by accident, and `tests/test_notebook.py` fails
if the committed notebook drifts from the sources.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "PINN_Poiseuille.ipynb"

#: Source files embedded in the notebook, in no particular order (they are
#: written to disk, then imported as a normal package).
SOURCE_GLOBS = ("src/**/*.py",)


def collect_sources() -> dict[str, str]:
    files: dict[str, str] = {}
    for pattern in SOURCE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            files[str(path.relative_to(ROOT))] = path.read_text(encoding="utf-8")
    if not files:
        raise SystemExit("no source files found; run this from the repo root")
    return files


def markdown(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def build() -> dict:
    sources = collect_sources()
    payload = json.dumps(sources, indent=0)

    cells = [
        markdown("""
# PINN for plane Poiseuille flow — self-contained

Everything is in this one notebook. **Runtime → Run all.**

Three stages: verify the code, train a model, use the trained model.

* Accuracy is the **relative L2 error against the analytic solution** — the
  19th-century closed form for this flow. That is the `rel L2 vs exact` column.
* The **loss is a diagnostic, never an accuracy claim.** It can fall while the
  answer gets worse, which is why both are shown side by side.
* Everything runs in **float64**. On a free Colab T4 that may be no faster than
  a laptop (a T4's FP64 rate is 1/32 of its FP32); an A100 is genuinely faster.
"""),
        markdown("## 1. Unpack the code and install dependencies"),
        code(f"""
import json, pathlib

# The project's source files, verbatim from the tested repository.
SOURCES = json.loads(r'''{payload}''')

for relative, text in SOURCES.items():
    path = pathlib.Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

print(f"wrote {{len(SOURCES)}} source files")
"""),
        code("""
!pip install -q optax pyyaml

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
print("jax", jax.__version__, "| devices", jax.devices(),
      "| float64 active:", jnp.zeros(1).dtype == jnp.float64)
"""),
        markdown("""
## 2. Verify the physics is correct

These check the residual operators against hand-derived forcings, the autodiff
against fourth-order finite differences, and the scaling round trip. **No
training.** If any of these fail, do not trust anything below.
"""),
        code("""
import numpy as np, jax.numpy as jnp
from src.physics import analytic, residuals
from src.physics.derivatives import batched_field_derivatives
from src.physics.conventions import re_h_to_re_dh, entrance_length_over_h, EntranceLengthCorrelation

# The analytic solution must satisfy the residual operator exactly.
Re, L = 10.0, 7.824
field = lambda xy: jnp.array([1.0 - xy[1]**2, 0.0, -2.0/Re*(xy[0] - L)])
pts = jnp.asarray(np.random.default_rng(0).uniform([0,-1],[L,1], size=(512,2)))
r_x, r_y, r_c = residuals.navier_stokes(batched_field_derivatives(field, pts), Re, "inertial")
print(f"analytic solution residual:  max|R| = {float(jnp.max(jnp.abs(jnp.stack([r_x,r_y,r_c])))):.2e}   (want ~0)")

# Reynolds conventions: Re_Dh = (8/3) Re_h exactly.
print(f"Re_Dh / Re_h = {re_h_to_re_dh(1.0)!r}   (want {8/3!r})")

# Flux of the reference profile is exactly 4/3.
nodes, weights = np.polynomial.legendre.leggauss(200)
print(f"flux of 1-y^2  = {float(analytic.u_nd(nodes) @ weights):.15f}   (want {4/3:.15f})")
"""),
        markdown("""
## 3. Configure

`L*` is **derived**, not chosen: it comes from the entrance-length correlation
so the domain is always long enough for the Reynolds number.

Reduce `steps` / `max_iterations` for a quick look; raise them for accuracy.
"""),
        code("""
import yaml
from src.config import parse_config

CONFIG = yaml.safe_load(r'''
name: colab_caseA_re10
seed: 0
case: A

domain: {alpha: 1.5, outlet_margin: 5.0}

correlation:
  c0: 0.3155
  c1: 0.01105
  n: 1.6
  citation: >
    Durst, Ray, Unsal & Bayoumi (2005), J. Fluids Eng. 127:1154-1160,
    plane-channel fit C0=0.631, C1=0.0442, n=1.6 in full-height units;
    converted to hydraulic diameter D_h = 2H via c0 = C0/2, c1 = C1/4.

physics:
  reynolds: 10.0
  pressure_scaling: inertial
  formulation: velocity_pressure

model:
  arch: modified_mlp
  width: 128
  depth: 4
  fourier: {enabled: false}
  rwf: {enabled: false}

bc: {inlet: soft, outlet_velocity: soft, pressure_gauge: hard}

sampling:
  n_interior: 2048
  n_validation: 2048
  n_inlet: 128
  n_outlet: 128
  resample_every: 500
  seed: 1

weights:
  scheme: fixed
  values: {momentum_x: 1.0, momentum_y: 1.0, continuity: 1.0, inlet: 1.0, outlet_velocity: 1.0}

training:
  select: best_validation
  adam:  {steps: 2000, lr_init: 1.0e-3, decay_rate: 0.9, decay_steps: 1000}
  lbfgs: {enabled: true, max_iterations: 6000, gtol: 1.0e-14}

evaluation: {nx: 401, ny: 201}
''')

config = parse_config(CONFIG)
print(f"{config.name}: case {config.case}, Re_h = {config.physics.reynolds:g}")
print(f"  entrance length L_e/h = {config.domain.entrance_length_over_h:.3f}")
print(f"  derived domain L*     = {config.domain.length:.3f}")
"""),
        markdown("""
## 4. Train

Watch **`train loss` against `held-out` and `rel L2 vs exact`**. The training
loss is measured on the points being optimised; the held-out loss is measured
on points the optimiser never sees. When they separate, the model is fitting
its collocation sample rather than the equations — and `select:
best_validation` keeps the best held-out iterate rather than the last one.
"""),
        code("""
from src.training.loop import train
from src.evaluation.accuracy import evaluate

result = train(config, verbose=True)
metrics = evaluate(config, result.model, result.params)
"""),
        markdown("## 5. Results against the analytic solution"),
        code("""
print(f"test 1  u relative L2 : {metrics['u_relative_l2']:.4e}   target <= 1e-5 (stretch 1e-6)")
print(f"test 2  max |v*|      : {metrics['v_max_abs']:.4e}   target <= 1e-5")
print(f"test 3  mass drift D  : {metrics['flux_conservation_relative_error']:.4e}   target <= 1e-6")
if "p_relative_l2" in metrics:
    print(f"        p relative L2 : {metrics['p_relative_l2']:.4e}")
print()
print("DIAGNOSTICS (not accuracy claims)")
print(f"  final loss          : {result.diagnostics.get('loss_after_lbfgs'):.4e}")
print(f"  held-out loss best  : {result.diagnostics.get('validation_loss_best'):.4e}")
print(f"  parameters selected : {result.diagnostics.get('selected_from')}")
"""),
        code("""
import matplotlib.pyplot as plt
import numpy as np
from src.geometry.domain import Channel
from src.geometry.sampling import evaluation_grid
from src.evaluation.metrics import evaluate_fields

points, x, y = evaluation_grid(Channel(config.domain.length), 201, 101)
u, v, p = evaluate_fields(result.model.field, result.params, points)
u = u.reshape(201, 101)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for i in np.linspace(0, 200, 5).astype(int):
    ax[0].plot(u[i], y, alpha=0.8)
ax[0].plot(analytic.u_nd(np.asarray(y)), y, "k--", lw=2, label="analytic 1 - y*²")
ax[0].set_xlabel("u*"); ax[0].set_ylabel("y*"); ax[0].legend(); ax[0].set_title("profiles")

err = np.abs(u - analytic.u_nd(np.asarray(y))[None, :])
im = ax[1].pcolormesh(x, y, err.T, shading="auto")
fig.colorbar(im, ax=ax[1]); ax[1].set_title("|u* - exact|"); ax[1].set_xlabel("x*")

rows = [r for r in result.history]
ax[2].semilogy([r["step"] for r in rows], [r["loss"] for r in rows], label="train loss")
ax[2].semilogy([r["step"] for r in rows], [r["validation_loss"] for r in rows], label="held-out loss")
ax[2].semilogy([r["step"] for r in rows], [r["relative_l2_vs_exact"] for r in rows], label="rel L2 vs exact")
ax[2].set_xlabel("Adam step"); ax[2].legend(); ax[2].set_title("loss is not accuracy")
plt.tight_layout(); plt.show()
"""),
        markdown("""
## 6. Use the trained model

`solution` evaluates the trained network anywhere in the channel. It refuses to
evaluate outside the trained domain: a PINN is fitted, not solved, and outside
the region where its residual was enforced it is not solving anything.
"""),
        code("""
from src.predict import TrainedModel

solution = TrainedModel(config, result.params, result.model, run_directory=None)
print(solution.summary())

u_pt, v_pt = solution.velocity(3.0, 0.5)
print(f"  u*(3.0, 0.5) = {u_pt:+.9f}   exact {1 - 0.5**2:+.9f}")
print(f"  v*(3.0, 0.5) = {v_pt:+.9f}   exact +0.000000000")
print(f"  flux at x*=3 = {solution.flux(3.0):.12f}   exact {4/3:.12f}")

try:
    solution.velocity(config.domain.length * 2, 0.0)
except ValueError as error:
    print(f"\\n  refused, correctly: {error}")
"""),
        markdown("""
## 7. Save the trained model

Colab discards the VM when it disconnects. Mount Drive to keep the weights.
"""),
        code("""
import numpy as np
from jax.flatten_util import ravel_pytree

flat, _ = ravel_pytree(result.params)
np.savez_compressed("trained_params.npz", flat=np.asarray(flat, dtype=np.float64))
print(f"saved {flat.size} parameters to trained_params.npz")

# from google.colab import drive; drive.mount('/content/drive')
# !cp trained_params.npz /content/drive/MyDrive/
"""),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    NOTEBOOK.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {NOTEBOOK} ({NOTEBOOK.stat().st_size / 1024:.0f} KB, "
          f"{len(collect_sources())} source files embedded)")
