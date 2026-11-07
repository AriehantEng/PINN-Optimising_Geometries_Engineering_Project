"""Build one self-contained Python file containing the whole project.

    python tools/make_single_file.py

Produces `pinn_poiseuille_all_in_one.py`: a single text file with every source
file embedded verbatim. Run it and it unpacks itself, trains, shows live
progress, reports accuracy against the analytic solution, and demonstrates
using the trained model.

The embedded code is byte-for-byte the tested source, not a hand-written copy,
so it cannot silently drift from what the test suite covers.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "pinn_poiseuille_all_in_one.py"


def collect_sources() -> dict[str, str]:
    files = {
        str(p.relative_to(ROOT)): p.read_text(encoding="utf-8")
        for p in sorted(ROOT.glob("src/**/*.py"))
    }
    for name, text in files.items():
        if "'''" in text:
            raise SystemExit(f"{name} contains ''' and cannot be embedded verbatim")
    return files


HEADER = '''#!/usr/bin/env python3
"""
PINN for plane Poiseuille flow — the entire project in one file.

============================================================================
HOW TO RUN
============================================================================

  1. Install the two dependencies that are not already standard:

         pip install jax optax pyyaml matplotlib

  2. Run it:

         python pinn_poiseuille_all_in_one.py            quick demo, ~3 minutes
         python pinn_poiseuille_all_in_one.py --full     full accuracy, ~30 min
         python pinn_poiseuille_all_in_one.py --check    verify only, no training

Running it creates a `src/` folder next to this file (that is the project's
code, written out so Python can import it) and a `trained_model.npz` when
training finishes.

============================================================================
WHAT YOU WILL SEE
============================================================================

During training, two numbers side by side:

    step  epoch   train loss     held-out   rel L2 vs exact      lr
     300      0   7.4260e-03   7.9012e-03       6.2084e-01  8.54e-04

  * `train loss` is the residual of the Navier-Stokes equations on the points
    being optimised. It is a DIAGNOSTIC. It can fall while the answer gets
    worse, and in this project it demonstrably does.

  * `rel L2 vs exact` is the relative L2 error against the ANALYTIC solution
    of this flow, which has been known since the 19th century. THIS is the
    accuracy number. Watch this one.

  * `held-out` is the same residual measured on points the optimiser never
    sees, which is how overfitting would show up without needing the exact
    answer (Phase 2 will not have one).

At the end it prints the accuracy against the analytic solution, then
demonstrates using the trained model to predict velocity and pressure at
arbitrary points.

============================================================================
WHAT THE NUMBERS MEAN
============================================================================

Targets from the project specification:

    relative L2 error on u    <= 1e-5   (stretch goal 1e-6)
    max |v*|                  <= 1e-5
    mass conservation drift   <= 1e-6

Best measured so far on the full schedule: 8.6e-7 on u, which beats the
stretch goal.

Everything runs in float64. That is not optional here: float32 caps the
achievable error near 1e-4, which is worse than every target above.
"""

import argparse
import json
import pathlib
import sys
import textwrap

# ===========================================================================
# THE PROJECT SOURCE, EMBEDDED VERBATIM
# ===========================================================================
#
#   *** IF YOUR EDITOR SHOWS EVERYTHING BELOW AS ONE GREY BLOCK, THAT IS  ***
#   *** SYNTAX HIGHLIGHTING, NOT A PROBLEM. The code is NOT commented out.***
#
# Each entry below is one file of the project, stored as string DATA inside
# a raw triple-quoted literal. Python reads them as data; unpack() writes
# them to disk as real .py files, which are then imported normally.
#
# Most editors colour everything between the triple quotes as a string, so
# this file LOOKS inert. It is not: it compiles, runs, trains and predicts.
# The quotes are balanced - exactly one opening and one closing delimiter
# per embedded file.
#
# To read the code as ordinary syntax-highlighted Python instead, run:
#
#       python pinn_poiseuille_all_in_one.py --extract
#
# which writes the 34 files into ./src/ and stops, without training.

FILES = {}

'''

FOOTER = '''

# ===========================================================================
# RUNNER
# ===========================================================================


def unpack(destination="."):
    """Write the embedded source files to disk."""
    root = pathlib.Path(destination)
    for relative, text in FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return len(FILES)


def require_dependencies():
    missing = []
    for module, install in (("jax", "jax"), ("optax", "optax"),
                            ("yaml", "pyyaml"), ("numpy", "numpy"),
                            ("scipy", "scipy")):
        try:
            __import__(module)
        except ImportError:
            missing.append(install)
    if missing:
        print("Missing dependencies. Install them with:\\n")
        print(f"    pip install {' '.join(sorted(set(missing)))}\\n")
        sys.exit(1)


CONFIG = """
name: all_in_one
seed: 0
case: A

domain: {alpha: 1.5, outlet_margin: 5.0}

correlation:
  c0: 0.3155
  c1: 0.01105
  n: 1.6
  citation: >
    Durst, Ray, Unsal & Bayoumi (2005), J. Fluids Eng. 127:1154-1160,
    plane-channel fit, converted to hydraulic-diameter units.

physics:
  reynolds: 10.0
  pressure_scaling: inertial
  formulation: velocity_pressure

model:
  arch: modified_mlp
  width: WIDTH
  depth: DEPTH
  fourier: {enabled: false}
  rwf: {enabled: false}

bc: {inlet: soft, outlet_velocity: soft, pressure_gauge: hard}

sampling:
  n_interior: NINT
  n_validation: NINT
  n_inlet: 128
  n_outlet: 128
  resample_every: 500
  seed: 1

weights:
  scheme: fixed
  values: {momentum_x: 1.0, momentum_y: 1.0, continuity: 1.0, inlet: 1.0, outlet_velocity: 1.0}

training:
  select: best_validation
  adam:  {steps: ADAM, lr_init: 1.0e-3, decay_rate: 0.9, decay_steps: DECAY}
  lbfgs: {enabled: true, max_iterations: LBFGS, gtol: 1.0e-14}

evaluation: {nx: NX, ny: NY}
"""


def build_config(full):
    substitutions = (
        {"WIDTH": 128, "DEPTH": 4, "NINT": 2048, "ADAM": 2000,
         "DECAY": 1000, "LBFGS": 6000, "NX": 401, "NY": 201}
        if full else
        {"WIDTH": 48, "DEPTH": 3, "NINT": 768, "ADAM": 500,
         "DECAY": 250, "LBFGS": 800, "NX": 121, "NY": 61}
    )
    text = CONFIG
    for key, value in substitutions.items():
        text = text.replace(key, str(value))
    return text


def check_physics():
    """Verify the equations before trusting anything trained from them."""
    import numpy as np, jax.numpy as jnp
    from src.physics import analytic, residuals
    from src.physics.derivatives import batched_field_derivatives
    from src.physics.conventions import re_h_to_re_dh

    print("\\n--- VERIFYING THE PHYSICS (no training) ---")

    Re, L = 10.0, 7.824
    field = lambda xy: jnp.array([1.0 - xy[1] ** 2, 0.0, -2.0 / Re * (xy[0] - L)])
    pts = jnp.asarray(np.random.default_rng(0).uniform([0, -1], [L, 1], size=(512, 2)))
    r = residuals.navier_stokes(batched_field_derivatives(field, pts), Re, "inertial")
    worst = float(jnp.max(jnp.abs(jnp.stack(r))))
    print(f"  analytic solution satisfies the equations : max|residual| = {worst:.2e}   (want ~0)")

    nodes, weights = np.polynomial.legendre.leggauss(200)
    flux = float(analytic.u_nd(nodes) @ weights)
    print(f"  flux of the reference profile             : {flux:.15f}")
    print(f"                                       exact: {4/3:.15f}")
    print(f"  Reynolds convention Re_Dh / Re_h          : {re_h_to_re_dh(1.0):.15f}  (want {8/3:.15f})")
    assert worst < 1e-12, "the analytic solution does not satisfy the residual operator"
    print("  OK")


def in_notebook():
    """True inside Jupyter, Colab, or any IPython kernel."""
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


def main(argv=None):
    """Entry point.

    `argv` defaults to the command line, EXCEPT in a notebook. Colab and
    Jupyter inject their own arguments (-f /path/to/kernel.json) into sys.argv,
    which argparse rejects with "unrecognized arguments" -- so pasting this
    file into a Colab cell and running it would fail at the first line of
    main(). In a notebook the command line is ignored and the defaults are
    used; pass options explicitly instead:

        main()                # quick demo
        main(["--full"])      # full accuracy schedule
        main(["--check"])     # verify the physics only
        main(["--extract"])   # write readable source files and stop
    """
    if argv is None:
        argv = [] if in_notebook() else sys.argv[1:]

    parser = argparse.ArgumentParser(description="PINN for plane Poiseuille flow")
    parser.add_argument("--full", action="store_true",
                        help="full accuracy schedule (~30 min) instead of the quick demo")
    parser.add_argument("--check", action="store_true",
                        help="verify the physics and exit, without training")
    parser.add_argument("--extract", action="store_true",
                        help="write the source files to ./src/ and exit, so they "
                             "can be read as ordinary Python")
    args = parser.parse_args(argv)

    count = unpack()
    if args.extract:
        print(f"wrote {count} source files into ./src/\\n")
        for relative in sorted(FILES):
            print(f"    {relative}")
        print("\\nThese are ordinary Python files. Open them in any editor.")
        return 0

    print(f"unpacking {count} source files into ./src/")
    require_dependencies()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import yaml

    print(f"jax {jax.__version__} | devices {jax.devices()} | "
          f"float64 active: {jnp.zeros(1).dtype == jnp.float64}")

    check_physics()
    if args.check:
        return 0

    from src.config import parse_config
    from src.training.loop import train
    from src.evaluation.accuracy import evaluate
    from src.physics import analytic
    from src.predict import TrainedModel
    from src.report import format_gate, gate_results

    config = parse_config(yaml.safe_load(build_config(args.full)))
    print(f"\\n--- TRAINING: {config.name} ---")
    print(f"  case {config.case}, Re_h = {config.physics.reynolds:g}, "
          f"{config.physics.formulation}")
    print(f"  entrance length L_e/h = {config.domain.entrance_length_over_h:.3f} "
          f"-> derived domain L* = {config.domain.length:.3f}")
    if not args.full:
        print("  QUICK DEMO schedule. Use --full for the accuracy targets.")

    result = train(config, verbose=True)
    metrics = evaluate(config, result.model, result.params)

    print("\\n--- ACCURACY vs THE ANALYTIC SOLUTION ---")
    def row(label, key, target):
        value = metrics.get(key)
        if value is None:
            return
        verdict = "PASS" if value <= target else "FAIL"
        print(f"  {label:<34}{value:.4e}   target <= {target:.0e}   {verdict}")
    row("relative L2 error on u", "u_relative_l2", 1e-5)
    row("max |v*|", "v_max_abs", 1e-5)
    row("mass conservation drift D", "flux_conservation_relative_error", 1e-6)
    if "p_relative_l2" in metrics:
        print(f"  {'relative L2 error on p':<34}{metrics['p_relative_l2']:.4e}")

    # Decided here, reported at the very end so it is the last thing on screen.
    # Only --full is gated: the quick schedule is not meant to reach the
    # targets and says so itself.
    gate_failures, gate_vacuous = gate_results(metrics) if args.full else ([], [])

    print("\\n  DIAGNOSTICS (not accuracy claims)")
    final = result.diagnostics.get("loss_after_lbfgs",
                                   result.diagnostics.get("loss_after_adam"))
    print(f"    final loss                      {final:.4e}")
    print(f"    parameters selected from        {result.diagnostics.get('selected_from')}")
    if not args.full:
        print("\\n  NOTE: the quick schedule does not reach the targets. That is expected.")
        print("        Re-run with --full for the real numbers.")

    print("\\n--- USING THE TRAINED MODEL ---")
    solution = TrainedModel(config, result.params, result.model, run_directory=None)
    print(f"  {solution.summary()}")
    for x, y in ((3.0, 0.5), (1.0, 0.0), (5.0, -0.75)):
        if x > config.domain.length:
            continue
        u, v = solution.velocity(x, y)
        p = solution.pressure(x, y)
        print(f"    at (x*={x:g}, y*={y:+g}):  u* = {u:+.9f}  (exact {1 - y**2:+.9f})"
              f"   v* = {v:+.9f}   p* = {p:+.9f}")
    print(f"    flux at x* = 3            :  {solution.flux(3.0):.12f}   "
          f"(exact {4/3:.12f})")

    try:
        solution.velocity(config.domain.length * 2, 0.0)
    except ValueError:
        print("    evaluating outside the trained domain is refused, correctly")

    from jax.flatten_util import ravel_pytree
    flat, _ = ravel_pytree(result.params)
    np.savez_compressed("trained_model.npz", flat=np.asarray(flat, dtype=np.float64))
    print(f"\\n  saved {flat.size} trained parameters to trained_model.npz")

    print("\\n--- TO USE IT YOURSELF ---")
    print(textwrap.dedent("""
        from src.predict import TrainedModel
        # (re-train, or load runs/<id> with src.predict.load_run)
        u, v = solution.velocity(3.0, 0.5)     # velocity anywhere in the channel
        y, u, v = solution.profile(x=3.0)      # a cross-channel profile
        x, y, u, v = solution.grid(201, 101)   # the whole field
    """).strip())

    report = format_gate(gate_failures, gate_vacuous)
    if report:
        print(f"\\n{report}")
    if gate_failures:
        print("\\n  A FAIL above is a real miss against the SPEC targets, so this "
              "exits non-zero.")
        print("  For the drift target specifically, raise "
              "training.lbfgs.max_iterations: it is")
        print("  under-convergence, not a formulation bias (see "
              "docs/phase1-findings.md).")
        return 1
    return 0


if __name__ == "__main__":
    _exit_code = main()
    # In a notebook, `raise SystemExit` prints an ugly traceback and can look
    # like a crash, so only use it from a real command line.
    if not in_notebook() and _exit_code:
        raise SystemExit(_exit_code)
'''


def build() -> str:
    parts = [HEADER]
    for name, text in collect_sources().items():
        parts.append("# ---------------------------------------------------------------------------\n")
        parts.append(f"# {name}\n")
        parts.append("# ---------------------------------------------------------------------------\n")
        parts.append(f"FILES[{name!r}] = r'''{text}'''\n\n")
    parts.append(FOOTER)
    return "".join(parts)


if __name__ == "__main__":
    OUTPUT.write_text(build(), encoding="utf-8")
    size = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT.name}  ({size:.0f} KB, {len(collect_sources())} files embedded, "
          f"{len(OUTPUT.read_text().splitlines())} lines)")
