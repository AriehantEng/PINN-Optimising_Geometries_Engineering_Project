"""Network 1's outlet profile against the analytical Poiseuille profile.

    python -m developing_flow.outlet_error --solution developing_flow/runs/reference_re10_m8 \\
        --out docs/developing_re10-outlet \\
        developing_flow/runs/developing_re10_polish_g5_points \\
        developing_flow/runs/developing_re10_polish_g5_points_seed2 ...

At the outlet, x* = L, the flow should be the developed parabola. Three
comparisons of u there, each a relative L2 norm across the channel on the
reference's own heights (uniform, Simpson weights):

    analytical   u = 1 - y^2, the parabola at the PRESCRIBED flux 4/3
    shape        (3Q/4)(1 - y^2) at the network's OWN outlet flux Q, so a flux
                 error is left out (pinn.py's outlet_profile_relative_l2, there
                 on 201 points)
    reference    the finite-element solution at x* = L

The reference's own departure from the analytical profile is reported beside
them, so a network below it cannot be told apart from the reference there.
Writes <out>.md and <out>.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from developing_flow import pinn, reference


def simpson_weights(y):
    """Composite Simpson weights on uniform heights: exact for the parabola, so
    the reference's own flux and shape come out exact, where the trapezoidal
    rule's 1e-6 error on 1001 heights would sit at the networks' error level."""
    n = len(y)
    h = (y[-1] - y[0]) / (n - 1)
    if n % 2 == 0 or not np.allclose(np.diff(y), h, rtol=0, atol=1e-12):
        raise ValueError("Simpson's rule needs an odd number of uniformly spaced heights")
    w = np.full(n, 2.0)
    w[1::2] = 4.0
    w[0] = w[-1] = 1.0
    return w * h / 3.0


def relative_l2(u, exact, w):
    return float(np.sqrt(np.sum(w * (u - exact) ** 2) / np.sum(w * exact**2)))


def outlet_profile(u, y, w, analytical, ref_u):
    """The comparisons for one outlet profile u(y)."""
    flux = float(np.sum(w * u))
    return {
        "analytical": relative_l2(u, analytical, w),
        "analytical_max_abs": float(np.max(np.abs(u - analytical))),
        "shape": relative_l2(u, pinn.developed_profile(y, flux), w),
        "reference": relative_l2(u, ref_u, w),
        "flux_error": flux / pinn.INLET_FLUX - 1.0,
    }


def measure(solution, runs):
    x, y, ref_u, _, _ = reference.load(solution)
    i = int(np.argmin(np.abs(x - x[-1])))
    length = float(x[i])
    w = simpson_weights(y)
    analytical = pinn.developed_profile(y, pinn.INLET_FLUX)
    out = {"length": length, "heights": len(y),
           "reference_against_analytical": outlet_profile(ref_u[i], y, w, analytical, ref_u[i]),
           "runs": []}
    for directory in runs:
        model = pinn.load_run(directory)
        if abs(model.config.length - length) > 1e-12:
            raise ValueError(f"{directory}: channel length {model.config.length}, reference {length}")
        u, v, _ = model(np.full_like(y, length), y)
        row = outlet_profile(u, y, w, analytical, ref_u[i])
        row.update({"run": str(directory), "seed": model.config.seed,
                    "outlet_max_abs_v": float(np.max(np.abs(v)))})
        out["runs"].append(row)
    return out


def markdown(result):
    keys = ("analytical", "analytical_max_abs", "shape", "reference", "flux_error", "outlet_max_abs_v")
    head = ("relative L2 against 1 − y²", "max \\|u − (1 − y²)\\|", "against the parabola at its own flux",
            "against the reference at the outlet", "outlet flux error", "max \\|v\\|")
    lines = [f"## Network 1's outlet profile at x* = {result['length']:g} against the Poiseuille parabola",
             "", f"Relative L2 norms across the channel on the reference's {result['heights']} heights "
             "(Simpson weights). Analytical profile: u = 1 − y², v = 0 (flux 4/3).", "",
             "| | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
    for r in result["runs"]:
        lines.append(f"| seed {r['seed']} | " + " | ".join(f"{r[k]:.2e}" for k in keys) + " |")
    if len(result["runs"]) > 1:
        cells = []
        for k in keys:
            vals = [r[k] for r in result["runs"]]
            cells.append(f"{np.mean(vals):.2e} ({min(vals):.2e} to {max(vals):.2e})")
        lines.append("| mean (range) | " + " | ".join(cells) + " |")
    ref = result["reference_against_analytical"]
    lines.append(f"| reference solution | {ref['analytical']:.2e} | {ref['analytical_max_abs']:.2e} | "
                 f"{ref['shape']:.2e} | — | {ref['flux_error']:.2e} | — |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="finished developing-flow run directories")
    parser.add_argument("--solution", required=True, help="a reference from developing_flow.reference")
    parser.add_argument("--out", required=True, help="output path without extension")
    args = parser.parse_args(argv)
    result = measure(args.solution, args.runs)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    out.with_suffix(".md").write_text(markdown(result))
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
