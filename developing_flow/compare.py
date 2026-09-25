"""Compare a trained developing-flow PINN with the original equations.

    python -m developing_flow.compare developing_flow/runs/developing_re10

Two kinds of reference, kept apart because they mean different things:

* EXACT SOLUTIONS, where the equations have one:
    inlet, x* = 0          u = U0 (1 - y^(2m))     (prescribed)
    developed region       u = (3Q/4)(1 - y^2)     (Hagen-Poiseuille, Q = 4/3)
    everywhere             u(x, y) = u(x, -y)      (symmetric channel and inlet)
    everywhere             int u dy = 4/3          (continuity, integrated)
  Inside the entrance region developing flow has no closed-form solution, so
  no "error" is claimed there; the departure from the parabola is reported as
  development, not as error.

* THE ORIGINAL EQUATIONS THEMSELVES: the residuals of x-momentum, y-momentum
  and continuity, which must be zero everywhere, entrance region included.

Writes u_vs_y.png, u_vs_x.png, residuals_vs_y.png and comparison_tables.md
into the run directory and prints the tables.
"""

from __future__ import annotations

import math
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from developing_flow import pinn

#: Heights at which u is followed along the channel.
HEIGHTS = (0.0, 0.5, 0.9)
#: Tolerance defining "developed" at a height: within 1% of the parabola.
DEVELOPED_WITHIN = 0.01


def load(directory):
    config = pinn.load_config(directory / "config.yaml")
    template = pinn.init_params(config, pinn.root_keys(config.seed)[0])
    _, unflatten = ravel_pytree(template)
    params = unflatten(jnp.asarray(np.load(directory / "params.npz")["flat"]))
    return config, params


def make_evaluator(config):
    field = pinn.make_field(config)

    def bundle(params, points):
        d = pinn.batched_derivatives(lambda xy: field(params, xy), points)
        return (d.u, d.v, d.p,
                pinn.momentum_x(d, config.reynolds, config.pressure_scaling),
                pinn.momentum_y(d, config.reynolds, config.pressure_scaling),
                pinn.continuity(d))

    compiled = jax.jit(bundle)

    def evaluate(params, points, chunk=4096):
        parts = [compiled(params, jnp.asarray(points[i:i + chunk]))
                 for i in range(0, len(points), chunk)]
        return tuple(np.concatenate([np.asarray(part[k]) for part in parts]) for k in range(6))

    return evaluate


def developed(y):
    return pinn.developed_profile(y, pinn.INLET_FLUX)


def relative_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def measure(directory):
    """Every line quantity the figures and tables use, for one run."""
    directory = pathlib.Path(directory)
    config, params = load(directory)
    evaluate = make_evaluator(config)
    L, m = config.length, config.inlet_sharpness
    developed_from = 2.0 * pinn.correlation_entrance_length(config.reynolds)

    # ---------------- u against y, at each station ----------------------- #
    y = np.linspace(-1.0, 1.0, 401)
    nodes, weights = np.polynomial.legendre.leggauss(200)
    station_rows, profiles = [], {}
    for x in config.profile_stations:
        u, v, _, rx, ry, rc = evaluate(params, np.stack([np.full_like(y, x), y], axis=-1))
        u_nodes = evaluate(params, np.stack([np.full_like(nodes, x), nodes], axis=-1))[0]
        flux = float(u_nodes @ weights)
        if x == 0.0:
            exact, label = pinn.inlet_profile(y, m), "yes: prescribed inlet U0(1 - y^16)"
        elif x >= developed_from:
            exact, label = developed(y), "yes: Poiseuille parabola 1 - y^2"
        else:
            exact, label = None, "no: flow still developing"
        profiles[x] = (u, v, rx, ry, rc, exact)
        station_rows.append({
            "x": x, "label": label,
            "l2": relative_l2(u, exact) if exact is not None else None,
            "max": float(np.max(np.abs(u - exact))) if exact is not None else None,
            "flux": flux, "flux_err": 100.0 * (flux / pinn.INLET_FLUX - 1.0),
            "symmetry": float(np.max(np.abs(u - u[::-1]))),
            "departure": relative_l2(u, developed(y)),
            "v_max": float(np.max(np.abs(v))),
        })

    # ---------------- u against x, at fixed heights ----------------------- #
    x = np.linspace(0.0, L, 1001)
    along = {}
    height_rows = []
    for h in HEIGHTS:
        u, _, _, _, _, _ = evaluate(params, np.stack([x, np.full_like(x, h)], axis=-1))
        target = float(developed(np.array(h)))
        along[h] = u
        outside = np.abs(u - target) > DEVELOPED_WITHIN * target
        settle = float(x[np.nonzero(outside)[0][-1] + 1]) if outside.any() else 0.0
        height_rows.append({
            "y": h, "inlet_exact": float(pinn.inlet_profile(np.array(h), m)),
            "inlet_pinn": float(u[0]), "developed_exact": target, "outlet_pinn": float(u[-1]),
            "outlet_err": abs(float(u[-1]) - target) / target, "settle": settle,
        })
    flux_x = np.linspace(0.0, L, 81)
    gx, gy = np.meshgrid(flux_x, nodes, indexing="ij")
    u_flux = evaluate(params, np.stack([gx.ravel(), gy.ravel()], axis=-1))[0]
    flux_along = u_flux.reshape(len(flux_x), -1) @ weights
    _, _, pc, rxc, ryc, rcc = evaluate(params, np.stack([x, np.zeros_like(x)], axis=-1))

    # ---------------- the equations over the whole channel ---------------- #
    xg, yg = np.linspace(0.0, L, 401), np.linspace(-1.0, 1.0, 201)
    GX, GY = np.meshgrid(xg, yg, indexing="ij")
    grid = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    _, _, _, RX, RY, RC = evaluate(params, grid)
    equation_rows = []
    for name, meaning, values, centre in (
        ("x-momentum", "forces along the channel balance (inertia + pressure + viscosity)", RX, rxc),
        ("y-momentum", "forces across the channel balance", RY, ryc),
        ("continuity", "no fluid created or destroyed (du/dx + dv/dy = 0)", RC, rcc),
    ):
        worst = int(np.argmax(np.abs(values)))
        developed_region = x > developed_from
        entrance = (grid[:, 0] < 0.5)
        equation_rows.append({
            "name": name, "meaning": meaning,
            "rms": float(np.sqrt(np.mean(values**2))),
            "rms_away": float(np.sqrt(np.mean(values[~entrance] ** 2))),
            "max": float(np.abs(values[worst])),
            "where": (float(grid[worst, 0]), float(grid[worst, 1])),
            "centre_developed": float(np.sqrt(np.mean(centre[developed_region] ** 2))),
        })

    return {
        "config": config, "y": y, "profiles": profiles, "station_rows": station_rows,
        "x": x, "along": along, "height_rows": height_rows,
        "flux_x": flux_x, "flux_along": flux_along, "rxc": rxc, "ryc": ryc, "rcc": rcc,
        "equation_rows": equation_rows, "developed_from": developed_from,
    }


def main(directory) -> int:
    directory = pathlib.Path(directory)
    m = measure(directory)
    config, developed_from = m["config"], m["developed_from"]
    write_figures(config, directory, m["y"], m["profiles"], m["x"], m["along"],
                  m["flux_x"], m["flux_along"], m["rxc"], m["ryc"], m["rcc"], developed_from)
    tables = format_tables(config, m["station_rows"], m["height_rows"], m["equation_rows"],
                           developed_from)
    (directory / "comparison_tables.md").write_text(tables, encoding="utf-8")
    print(tables)
    return 0


def write_figures(config, directory, y, profiles, x, along, flux_x, flux_along,
                  rxc, ryc, rcc, developed_from):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series, ref = pinn.SERIES, pinn.REFERENCE

    # u against y: one panel per station.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for ax, (xs, (u, _, _, _, _, exact)) in zip(axes.ravel(), profiles.items()):
        ax.plot(u, y, color=series[0], linewidth=2, label="PINN")
        if exact is not None:
            ax.plot(exact, y, color=ref, linewidth=1.5, linestyle="--", label="exact solution")
            ax.set_title(f"x* = {xs:g}   error {relative_l2(u, exact):.1e}", loc="left", fontsize=11)
        else:
            ax.plot(developed(y), y, color=pinn.MUTED, linewidth=1, linestyle=":",
                    label="parabola it develops into")
            ax.set_title(f"x* = {xs:g}   developing: no exact solution", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        pinn._style(ax)
    for ax in axes[1]:
        ax.set_xlabel("u*  (velocity along the channel)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y*  (height across the channel)")
    fig.suptitle(f"u against y at six stations, Re = {config.reynolds:g}", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(directory / "u_vs_y.png", dpi=150)
    plt.close(fig)

    # u against x: heights, their approach to the exact value, flux, residuals.
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(13, 8.5))
    for colour, h in zip(series, HEIGHTS):
        target = float(developed(np.array(h)))
        a.plot(x, along[h], color=colour, linewidth=2, label=f"PINN, y* = {h:g}")
        a.axhline(target, color=colour, linewidth=1, linestyle="--")
        b.semilogy(x, np.maximum(np.abs(along[h] - target) / target, 1e-12), color=colour,
                   linewidth=2, label=f"y* = {h:g}")
    a.plot([], [], color=ref, linestyle="--", linewidth=1, label="exact developed value 1 - y*^2")
    a.set_xlabel("x*  (distance along the channel)")
    a.set_ylabel("u*")
    a.set_title("u against x at three heights", loc="left")
    a.legend(frameon=False, fontsize=8, loc="center right")
    pinn._style(a)
    b.axhline(DEVELOPED_WITHIN, color=pinn.MUTED, linewidth=1, linestyle=":", label="1% of exact")
    b.set_xlabel("x*")
    b.set_ylabel("|u - exact developed| / exact")
    b.set_title("Approach to the exact developed solution", loc="left")
    b.legend(frameon=False, fontsize=8)
    pinn._style(b)
    c.plot(flux_x, 100.0 * (flux_along / pinn.INLET_FLUX - 1.0), color=series[2], linewidth=2)
    c.axhline(0.0, color=ref, linewidth=1, linestyle="--")
    c.set_xlabel("x*")
    c.set_ylabel("error in flux (%)")
    c.set_title("Flux, integral of u across y, against exact 4/3", loc="left")
    pinn._style(c)
    for colour, values, name in ((series[0], rxc, "x-momentum"), (series[1], ryc, "y-momentum"),
                                 (series[2], rcc, "continuity")):
        d.plot(x, values, color=colour, linewidth=1.5, label=name)
    d.axhline(0.0, color=ref, linewidth=1, linestyle="--")
    d.axvline(developed_from, color=pinn.MUTED, linewidth=1, linestyle=":")
    d.set_xlabel("x*  (along the centreline)")
    d.set_ylabel("residual (exact solution: 0)")
    d.set_title("Navier-Stokes residuals along the centreline", loc="left")
    d.legend(frameon=False, fontsize=8)
    pinn._style(d)
    fig.tight_layout()
    fig.savefig(directory / "u_vs_x.png", dpi=150)
    plt.close(fig)

    # Residuals of the original equations across y at each station.
    # Three representative stations (inlet, developing, developed), signed on
    # a linear axis: on a log axis every zero crossing becomes a spike.
    shown = [s for s in (0.0, 1.0, config.length) if s in profiles]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, index, name in zip(axes, (2, 3, 4), ("x-momentum", "y-momentum", "continuity")):
        for colour, xs in zip(series, shown):
            ax.plot(profiles[xs][index], y, color=colour, linewidth=1.5, label=f"x* = {xs:g}")
        ax.axvline(0.0, color=ref, linewidth=1, linestyle="--")
        ax.set_xlabel(f"{name} residual  (exact: 0)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_title(name, loc="left")
        pinn._style(ax)
    axes[0].set_ylabel("y*")
    axes[-1].legend(frameon=False, fontsize=8, loc="center left")
    fig.suptitle("How well the original equations hold across the channel", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(directory / "residuals_vs_y.png", dpi=150)
    plt.close(fig)


def _e(value):
    return "n/a" if value is None else f"{value:.2e}"


def format_tables(config, station_rows, height_rows, equation_rows, developed_from):
    lines = [
        f"## Table 1. u against y: velocity across the channel at each station (Re = {config.reynolds:g})",
        "",
        "| Station x* | Is there an exact solution here? | Relative L2 error of u vs exact | "
        "Largest pointwise error in u | Flux = integral of u dy (exact 1.3333) | Flux error (%) | "
        "Symmetry error, max abs u(y) - u(-y) (exact 0) | Largest abs v (developed: 0) | "
        "Departure from final parabola, relative L2 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in station_rows:
        lines.append(
            f"| {r['x']:g} | {r['label']} | {_e(r['l2'])} | {_e(r['max'])} | {r['flux']:.5f} | "
            f"{r['flux_err']:+.4f} | {r['symmetry']:.1e} | {r['v_max']:.2e} | {r['departure']:.2e} |")
    lines += [
        "",
        f"Exact solutions exist only at the inlet (prescribed) and in the developed region "
        f"(taken here as x* >= {developed_from:.2f}, twice the correlation entrance length). "
        "In between, the departure from the parabola is the flow still developing, not an error.",
        "",
        "## Table 2. u against x: velocity along the channel at fixed heights",
        "",
        "| Height y* | Inlet u, exact (prescribed) | Inlet u, PINN | Developed u, exact 1 - y*^2 | "
        "Outlet u, PINN | Outlet relative error vs exact | Distance x* after which u stays within 1% of exact |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in height_rows:
        lines.append(
            f"| {r['y']:g} | {r['inlet_exact']:.5f} | {r['inlet_pinn']:.5f} | {r['developed_exact']:.5f} | "
            f"{r['outlet_pinn']:.5f} | {r['outlet_err']:.2e} | {r['settle']:.3f} |")
    lines += [
        "",
        "## Table 3. The original equations: how close each is to being satisfied (exact value 0)",
        "",
        "| Equation | What it enforces | RMS residual, whole channel | RMS residual, excluding x* < 0.5 | "
        "Largest residual | Where the largest residual is (x*, y*) | RMS residual on the centreline, developed region |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in equation_rows:
        lines.append(
            f"| {r['name']} | {r['meaning']} | {r['rms']:.2e} | {r['rms_away']:.2e} | {r['max']:.2e} | "
            f"({r['where'][0]:.3f}, {r['where'][1]:+.3f}) | {r['centre_developed']:.2e} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "developing_flow/runs/developing_re10"))
