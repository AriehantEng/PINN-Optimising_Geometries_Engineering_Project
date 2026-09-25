"""Check a trained developing-flow PINN on a dense grid it never trained on.

    python -m developing_flow.verify developing_flow/runs/developing_re10 --nx 2001 --ny 1001

A PINN can drive its residuals down at its collocation points while breaking
the equations in the gaps between them. This evaluates, on nx * ny evenly
spaced points (about 2 million at the sizes above, against 4,096 per training
draw):

* the Navier-Stokes residuals everywhere: RMS, maximum and upper percentiles;
* u against the exact Poiseuille parabola wherever the flow is developed
  (x* >= twice the correlation entrance length);
* the symmetry u(x, y) = u(x, -y), which the exact solution obeys everywhere;
* the flux through 201 stations against the prescribed 4/3.

Writes verify.json into the run directory.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from developing_flow import pinn
from developing_flow.compare import load, make_evaluator


def sweep(directory, nx: int, ny: int, verbose: bool = True) -> dict:
    directory = pathlib.Path(directory)
    config, params = load(directory)
    evaluate = make_evaluator(config)
    L = config.length

    x = np.linspace(0.0, L, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    u, v, _, rx, ry, rc = evaluate(params, points, chunk=8192)
    U = u.reshape(nx, ny)
    V = v.reshape(nx, ny)

    developed_from = 2.0 * pinn.correlation_entrance_length(config.reynolds)
    region = x >= developed_from
    exact = pinn.developed_profile(y, pinn.INLET_FLUX)[None, :]
    error = U[region] - exact

    nodes, weights = np.polynomial.legendre.leggauss(200)
    stations = np.linspace(0.0, L, 201)
    gx, gy = np.meshgrid(stations, nodes, indexing="ij")
    flux = evaluate(params, np.stack([gx.ravel(), gy.ravel()], axis=-1))[0]
    flux = flux.reshape(len(stations), -1) @ weights

    away = points[:, 0] >= 0.5

    def stats(values):
        magnitude = np.abs(values)
        return {
            "rms": float(np.sqrt(np.mean(values**2))),
            "rms_excluding_x_below_0.5": float(np.sqrt(np.mean(values[away] ** 2))),
            "p99": float(np.percentile(magnitude, 99)),
            "p99.9": float(np.percentile(magnitude, 99.9)),
            "max": float(magnitude.max()),
            "fraction_above_1e-2": float(np.mean(magnitude > 1e-2)),
        }

    result = {
        "points": int(len(points)),
        "grid": [nx, ny],
        "training_points_per_draw": config.n_interior,
        "residual_momentum_x": stats(rx),
        "residual_momentum_y": stats(ry),
        "residual_continuity": stats(rc),
        "developed_region_from_x": developed_from,
        "developed_u_relative_l2": float(np.linalg.norm(error) / np.linalg.norm(
            np.broadcast_to(exact, error.shape))),
        "developed_u_max_abs_error": float(np.abs(error).max()),
        "developed_v_max_abs": float(np.abs(V[region]).max()),
        "symmetry_max_abs": float(np.abs(U - U[:, ::-1]).max()),
        "flux_max_relative_error": float(np.abs(flux / pinn.INLET_FLUX - 1.0).max()),
    }
    with (directory / "verify.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if verbose:
        print(format_result(result))
    return result


def format_result(r) -> str:
    lines = [
        f"Dense sweep: {r['points']:,} unseen points ({r['grid'][0]} x {r['grid'][1]}); "
        f"training used {r['training_points_per_draw']:,} per draw",
        "",
        "| Equation (exact residual 0) | RMS | RMS excluding x* < 0.5 | 99th percentile | "
        "99.9th percentile | Maximum | Share of points above 1e-2 |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, name in (("residual_momentum_x", "x-momentum"),
                      ("residual_momentum_y", "y-momentum"),
                      ("residual_continuity", "continuity")):
        s = r[key]
        lines.append(f"| {name} | {s['rms']:.2e} | {s['rms_excluding_x_below_0.5']:.2e} | "
                     f"{s['p99']:.2e} | {s['p99.9']:.2e} | {s['max']:.2e} | "
                     f"{100 * s['fraction_above_1e-2']:.3f}% |")
    lines += [
        "",
        f"| Check against an exact property | Value |",
        "|---|---|",
        f"| u vs Poiseuille parabola, relative L2, x* >= {r['developed_region_from_x']:.2f} | "
        f"{r['developed_u_relative_l2']:.2e} |",
        f"| u vs Poiseuille parabola, largest pointwise error | {r['developed_u_max_abs_error']:.2e} |",
        f"| largest abs v in the developed region (exact 0) | {r['developed_v_max_abs']:.2e} |",
        f"| symmetry, max abs u(x, y) - u(x, -y) (exact 0) | {r['symmetry_max_abs']:.2e} |",
        f"| flux at 201 stations, largest relative error vs 4/3 | {r['flux_max_relative_error']:.2e} |",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="a developing_flow run directory")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    args = parser.parse_args(argv)
    sweep(args.run, args.nx, args.ny)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
