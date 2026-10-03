"""The mechanical-energy budget of a trained developing-flow PINN.

    python -m developing_flow.energy developing_flow/runs/developing_re10_lm_fresh

The bifurcation study's objective is the flux-weighted total-pressure drop,
which for steady incompressible flow with no-slip walls equals the viscous
dissipation divided by the flow rate (literature review, Sec. 5; the same
identity as reference.energy_budget):

    inflow - outflow = dissipation,
    inflow  = int_inlet  (p + a |u|^2 / 2) u dy,   outflow likewise at x* = L,
    dissipation = 2 nu int e:e dA.

Both sides come from the PINN alone, so |inflow - outflow - dissipation| /
dissipation is a LABEL-FREE accuracy check: it needs no reference solution.
Against the finite-element reference the same quantities give the error of
the objective itself, which is what a design optimum depends on.

Quadrature: Gauss-Legendre in y (128 nodes, clustered at the walls), and
composite Gauss-Legendre in x on segments crowded toward the inlet, where the
flow develops. Checked on the exact Poiseuille field in the tests.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import jax
import numpy as np

from developing_flow import pinn
from developing_flow.compare import load

#: x segments for the dissipation integral, crowded toward the inlet.
X_BREAKS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 7.0)
NODES_PER_SEGMENT = 32
Y_NODES = 128


def quadrature(length):
    """(x, y, weight) of the composite Gauss-Legendre rule over [0, L] x [-1, 1]."""
    breaks = [b for b in X_BREAKS if b < length] + [length]
    gx, wx = np.polynomial.legendre.leggauss(NODES_PER_SEGMENT)
    xs, ws = [], []
    for left, right in zip(breaks[:-1], breaks[1:]):
        xs.append(0.5 * (right - left) * gx + 0.5 * (right + left))
        ws.append(0.5 * (right - left) * wx)
    x, wxs = np.concatenate(xs), np.concatenate(ws)
    y, wy = np.polynomial.legendre.leggauss(Y_NODES)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X.ravel(), Y.ravel(), np.outer(wxs, wy).ravel(), y, wy


def budget(config, field, params, chunk=8192):
    """The energy budget of `field` (any field(params, xy) -> (u, v, p))."""
    a, nu = pinn.scaling_coefficients(config.reynolds, config.pressure_scaling)
    L = config.length
    x, y2, w2, y, wy = quadrature(L)
    points = np.stack([x, y2], axis=-1)

    @jax.jit
    def strain_squared(p, pts):
        d = pinn.batched_derivatives(lambda xy: field(p, xy), pts)
        shear = 0.5 * (d.u_y + d.v_x)
        return d.u_x**2 + d.v_y**2 + 2.0 * shear**2      # e:e

    ee = np.concatenate([np.asarray(strain_squared(params, points[i:i + chunk]))
                         for i in range(0, len(points), chunk)])
    dissipation = float(2.0 * nu * (ee @ w2))

    def through(xv):
        u, v, p = pinn.predict(field, params, np.stack([np.full_like(y, xv), y], axis=-1))
        return (float(u @ wy), float(((p + 0.5 * a * (u**2 + v**2)) * u) @ wy),
                float((p * u) @ wy))

    q_in, total_in, static_in = through(0.0)
    q_out, total_out, static_out = through(L)
    return {"flux_in": q_in, "flux_out": q_out, "energy_in": total_in, "energy_out": total_out,
            "dissipation": dissipation,
            "total_pressure_drop": (total_in - total_out) / q_in,
            "static_pressure_drop": (static_in - static_out) / q_in,
            "energy_consistency": abs(total_in - total_out - dissipation) / dissipation}


def run_budget(directory):
    config, params = load(pathlib.Path(directory))
    return budget(config, pinn.make_field(config), params)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--reference", default=None,
                        help="a reference directory: also print the errors against its budget")
    args = parser.parse_args(argv)
    result = run_budget(args.run)
    if args.reference:
        ref = json.loads((pathlib.Path(args.reference) / "reference.json").read_text())["energy"]
        for key in ("dissipation", "total_pressure_drop", "static_pressure_drop"):
            result[f"{key}_error"] = abs(result[key] / ref[key] - 1.0)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
