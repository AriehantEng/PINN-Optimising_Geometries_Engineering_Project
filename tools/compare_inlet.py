"""Compare two runs' residuals, globally and where the inlet condition lives.

Step 2's question is whether regularising the inlet removes the corner
singularity. That cannot be read off `metrics.json` alone for the baseline,
because the near-inlet metrics did not exist when it ran -- and it must not be
read off a different script than the comparison, or the two numbers are not
commensurable. So both runs are evaluated here, by one piece of code, on one
grid.

    python tools/compare_inlet.py caseB_re10 caseB_re10_smooth8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.accuracy import INLET_BAND_FRACTION  # noqa: E402
from src.geometry.domain import Channel  # noqa: E402
from src.geometry.sampling import evaluation_grid  # noqa: E402
from src.physics import residuals, streamfunction  # noqa: E402
from src.physics.derivatives import (  # noqa: E402
    batched_field_derivatives,
    batched_stream_derivatives,
)
from src.predict import load_run  # noqa: E402

X_BANDS = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, None)]
Y_BANDS = [(0.98, 1.0), (0.9, 0.98), (0.7, 0.9), (0.3, 0.7), (0.0, 0.3)]


def measure(name, nx=401, ny=201):
    """Residual fields plus the coordinates they sit on, for one run."""
    directory = ROOT / "runs" / name
    model = load_run(str(directory))
    config = model.config
    points, x, y = evaluation_grid(Channel(config.domain.length), nx, ny)

    if config.physics.formulation == "psi_omega":
        derivatives = batched_stream_derivatives(
            lambda xy: model.model.stream(model.params, xy), points
        )
        names = ("vorticity", "vorticity_transport")
        values = streamfunction.residuals(derivatives, config.physics.reynolds)
    else:
        derivatives = batched_field_derivatives(
            lambda xy: model.model.field(model.params, xy), points
        )
        names = ("momentum_x", "momentum_y", "continuity")
        values = residuals.navier_stokes(
            derivatives, config.physics.reynolds, config.physics.pressure_scaling,
            config.physics.convective_coefficient,
        )

    coordinates = np.asarray(points)
    lbfgs = json.loads((directory / "lbfgs.json").read_text() or "[]")
    metrics = json.loads((directory / "metrics.json").read_text())
    return {
        "name": name,
        "config": config,
        "length": float(config.domain.length),
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
        "residuals": {n: np.abs(np.asarray(v)) for n, v in zip(names, values)},
        "lbfgs": lbfgs[-1] if isinstance(lbfgs, list) and lbfgs else {},
        "metrics": metrics,
    }


def _describe(run):
    config = run["config"]
    inlet = config.inlet_profile or "parabolic (case A)"
    detail = (f"m = {config.inlet_sharpness}, peak {config.inlet_peak:.6f}"
              if config.inlet_profile == "smoothed"
              else f"U = {config.inlet_peak:.6f}")
    return f"{run['name']}   inlet: {inlet}   {detail}"


def report(first, second) -> None:
    for run in (first, second):
        print(_describe(run))
    print()

    shared = [n for n in first["residuals"] if n in second["residuals"]]
    if not shared:
        print("the two runs use different formulations; no residual is comparable")
        return

    width = max(len(n) for n in shared) + 2
    print(f"GLOBAL          {'':<{width}}{first['name']:>16}{second['name']:>20}{'ratio':>11}")
    for name in shared:
        for statistic, fn in (("max", np.max), ("rms", lambda v: np.sqrt(np.mean(v**2)))):
            a, b = fn(first["residuals"][name]), fn(second["residuals"][name])
            print(f"  {name+' '+statistic:<{width+14}}{a:>16.4e}{b:>20.4e}{b/a:>11.3g}")

    print()
    band = INLET_BAND_FRACTION
    print(f"INLET BAND  x* < {band}*L*")
    for name in shared:
        a = float(np.max(first["residuals"][name][first["x"] < band * first["length"]]))
        b = float(np.max(second["residuals"][name][second["x"] < band * second["length"]]))
        print(f"  {'max '+name:<{width+14}}{a:>16.4e}{b:>20.4e}{b/a:>11.3g}")

    for label, bands, key in (("BY x-BAND", X_BANDS, "x"), ("BY |y|-BAND", Y_BANDS, "y")):
        print(f"\n{label}  (max |momentum-x| or first residual)")
        name = "momentum_x" if "momentum_x" in shared else shared[0]
        print(f"  {'band':<18}{first['name']:>16}{second['name']:>20}{'ratio':>11}")
        for low, high in bands:
            row = []
            for run in (first, second):
                coordinate = run[key] if key == "x" else np.abs(run[key])
                scale = run["length"] if key == "x" else 1.0
                upper = (high if high is not None else 1e9) * scale
                mask = (coordinate >= low * scale) & (coordinate < upper)
                row.append(float(np.max(run["residuals"][name][mask])) if mask.any() else float("nan"))
            span = f"[{low:.2f}, {high:.2f})" if high is not None else f"[{low:.2f}, end)"
            print(f"  {span:<18}{row[0]:>16.4e}{row[1]:>20.4e}{row[1]/row[0]:>11.3g}")

    print("\nTRAINING")
    for run in (first, second):
        lbfgs, metrics = run["lbfgs"], run["metrics"]
        print(f"  {run['name']}")
        print(f"    final loss      {metrics.get('diag_loss_after_lbfgs', float('nan')):.4e}")
        print(f"    iterations      {lbfgs.get('iterations')}")
        print(f"    termination     {lbfgs.get('message')}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args(argv)
    report(measure(args.baseline), measure(args.candidate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
