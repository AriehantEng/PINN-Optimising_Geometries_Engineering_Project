"""The mechanical-energy budget of a trained bifurcation network: a check that
needs no reference solution.

For steady incompressible flow with no-slip walls,

    E_in - E_out1 - E_out2 = Phi + (viscous flux through the openings),
    E = int (p + a |u|^2 / 2) u . n,   Phi = 2 nu int e:e dA,

and the opening term vanishes where the flow there is developed (paper,
Appendix A.6). Both sides come from the network's own field, so

    inconsistency = |Q_in J - Phi| / Phi,   J the total-pressure drop per unit flow,

is a label-free accuracy check. For a network the left side is measured on
the openings and the right in the domain, and any momentum residual R adds
int u . R to their difference. Network 2 conserves mass and its shares
exactly, so this is the one internal check its errors can still show up in.

The finite-element reference is solved for the same problem (the run's own
evaluation settings), and supplies (i) the quadrature for Phi: its own mesh,
refined at the corners, where e:e ~ r^(2 lambda - 2) is integrable but
singular; and (ii) the truth: the reference's J and Phi, and its own
inconsistency, which bounds what the check can resolve.

    python -m bifurcation.energy bifurcation/runs/bifurcation_re10_3090_equal_stream_ext \\
        bifurcation/runs/parametric_re10_3090_equal --out docs/bifurcation_re10-energy
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from bifurcation import flow, parametric, reference  # noqa: E402
from developing_flow import pinn  # noqa: E402

CHUNK = 8192


def dissipation(field_xy, solution, nu):
    """Phi = 2 nu int e:e dA of `field_xy` (xy -> (u, v, p)), on the reference's
    own quadrature: every element's Gauss points."""
    x, y = solution.basis_u.global_coordinates().value
    weights = np.asarray(solution.basis_u.dx).ravel()
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    gradient = jax.jit(jax.vmap(jax.jacfwd(field_xy)))
    total = 0.0
    for i in range(0, len(points), CHUNK):
        g = np.asarray(gradient(jnp.asarray(points[i:i + CHUNK])))[:, :2, :]   # du_i/dx_j
        e = 0.5 * (g + np.transpose(g, (0, 2, 1)))
        total += float(weights[i:i + CHUNK] @ np.sum(e * e, axis=(1, 2)))
    return 2.0 * nu * total


def budget(field_xy, model_xy, geometry, solution, reynolds, pressure_scaling, n_line):
    """One geometry: the network's and the reference's energy budgets."""
    a, nu = pinn.scaling_coefficients(reynolds, pressure_scaling)
    mine = flow.opening_measures(model_xy, geometry, a, n_line)
    theirs = flow.opening_measures(solution, geometry, a, n_line)
    phi_net = dissipation(field_xy, solution, nu)
    phi_ref = solution.measures()["dissipation"]
    left_net = mine["flux_in"] * mine["total_pressure_drop"]
    left_ref = theirs["flux_in"] * theirs["total_pressure_drop"]
    return {
        "drop_net": mine["total_pressure_drop"], "drop_ref": theirs["total_pressure_drop"],
        "drop_error": mine["total_pressure_drop"] / theirs["total_pressure_drop"] - 1.0,
        "dissipation_net": phi_net, "dissipation_ref": phi_ref,
        "dissipation_error": phi_net / phi_ref - 1.0,
        "inconsistency_net": abs(left_net - phi_net) / phi_net,
        "inconsistency_ref": abs(left_ref - phi_ref) / phi_ref,
    }


def fixed_run(directory):
    """[(label, budget)] for a fixed-geometry Network 2 run."""
    model, config, params = flow.load_run(directory)
    geometry = config.geometry()
    inlet = flow.inlet_function(config)
    q_in = flow.inlet_flux(inlet, config.n_flux) if config.n_flux is not None else None
    field = flow.make_field(config, geometry, q_in)
    solution = reference.solve_bifurcation(
        geometry, config.reynolds, config.pressure_scaling, inlet,
        config.outlet1_share if config.prescribed else None, config.fe_max_area,
        config.fe_corner_levels, config.fe_corner_radius, continuation=())
    row = budget(lambda xy: field(params, xy), model, geometry, solution, config.reynolds,
                 config.pressure_scaling, config.n_line)
    return [({"run": pathlib.Path(directory).name, "split": config.split}, row)]


def parametric_run(directory):
    """[(label, budget)] for a parametric run, at each of its evaluation splits."""
    model, config, params = parametric.load_run(directory)
    _, inlet, _, field = parametric.build(config)
    out = []
    for s in config.eval_splits:
        geometry = config.geometry(s)
        solution = reference.solve_bifurcation(
            geometry, config.reynolds, config.pressure_scaling, inlet, config.outlet1_share,
            config.fe_max_area, config.fe_corner_levels, config.fe_corner_radius, continuation=())
        row = budget(lambda xy, s=s: field(params, xy, s), parametric.at_split(model, s),
                     geometry, solution, config.reynolds, config.pressure_scaling, config.n_line)
        out.append(({"run": pathlib.Path(directory).name, "split": s}, row))
    return out


def run(directories, out):
    rows = []
    for directory in directories:
        document = yaml.safe_load((pathlib.Path(directory) / "config.yaml").read_text())
        rows += parametric_run(directory) if "design" in document else fixed_run(directory)
        for label, row in rows[-1:]:
            print(f"  {label['run']} split {label['split']}: inconsistency "
                  f"{row['inconsistency_net']:.2e} (reference {row['inconsistency_ref']:.1e}), "
                  f"drop error {row['drop_error']:+.2e}", flush=True)
    records = [{**label, **row} for label, row in rows]
    out = pathlib.Path(out)
    out.with_suffix(".json").write_text(json.dumps(records, indent=1))
    lines = ["## Energy budget of the bifurcation networks: a check without reference data", "",
             "inconsistency = |Q_in J - Phi| / Phi from the network's own field (J on the openings, "
             "Phi = 2 nu int e:e on the reference's mesh). The reference's own inconsistency is the "
             "floor of the check. drop error and dissipation error are against the reference.", "",
             "| run | split | inconsistency (network) | inconsistency (reference) | drop error | "
             "dissipation error |", "|---|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['run']} | {r['split']:.2f} | {r['inconsistency_net']:.2e} | "
                     f"{r['inconsistency_ref']:.1e} | {r['drop_error']:+.2e} | "
                     f"{r['dissipation_error']:+.2e} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True, help="path stem: writes .md and .json")
    args = parser.parse_args(argv)
    run(args.runs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
