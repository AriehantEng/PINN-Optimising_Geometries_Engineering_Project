"""Compare several developing-flow runs with the exact solutions, side by side.

    python -m developing_flow.compare_runs \\
        developing_flow/runs/developing_re10 developing_flow/runs/developing_re10_rar \\
        --labels baseline "with RAR" --out developing_flow/runs/comparison_re10

Uses exactly the measurements of `compare.measure` for each run, so every
number here is the one the single-run comparison would report. Where a run has
a verify.json (the dense sweep of `verify.py`), its figures are tabulated too.

Writes u_vs_y.png, u_vs_x.png and tables.md into --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

from developing_flow import pinn
from developing_flow.compare import DEVELOPED_WITHIN, HEIGHTS, developed, measure, relative_l2


def _e(value):
    return "n/a" if value is None else f"{value:.2e}"


def write_figures(runs, labels, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = pinn.SERIES
    first = runs[0]
    y, stations = first["y"], list(first["profiles"])
    exact_stations = [s for s in stations if first["profiles"][s][5] is not None]

    # u against y: profiles at six stations, then signed error where exact exists.
    fig, axes = plt.subplots(3, 3, figsize=(14, 12.5))
    for ax, station in zip(axes[:2].ravel(), stations):
        exact = first["profiles"][station][5]
        for colour, run, label in zip(colours, runs, labels):
            ax.plot(run["profiles"][station][0], y, color=colour, linewidth=2, label=label)
        if exact is not None:
            ax.plot(exact, y, color=pinn.REFERENCE, linewidth=1.5, linestyle="--", label="exact solution")
            errors = ", ".join(f"{label} {relative_l2(run['profiles'][station][0], exact):.1e}"
                               for run, label in zip(runs, labels))
            ax.set_title(f"x* = {station:g}   error: {errors}", loc="left", fontsize=10)
        else:
            ax.plot(developed(y), y, color=pinn.MUTED, linewidth=1, linestyle=":",
                    label="parabola it develops into")
            ax.set_title(f"x* = {station:g}   developing: no exact solution", loc="left", fontsize=10)
        ax.set_xlabel("u*")
        ax.set_ylabel("y*")
        # Centre-left is empty for every profile shape here: the inlet sits at
        # u ~ 0.7 and the parabolas at u ~ 0.8-1 across the middle.
        ax.legend(frameon=False, fontsize=8, loc="center left")
        pinn._style(ax)
    for ax, station in zip(axes[2], exact_stations[:3]):
        exact = first["profiles"][station][5]
        for colour, run, label in zip(colours, runs, labels):
            ax.plot(run["profiles"][station][0] - exact, y, color=colour, linewidth=1.5, label=label)
        ax.axvline(0.0, color=pinn.REFERENCE, linewidth=1, linestyle="--")
        ax.set_xlabel("u* (PINN) - u* (exact)")
        ax.set_ylabel("y*")
        ax.set_title(f"Signed error at x* = {station:g}", loc="left", fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        pinn._style(ax)
    for ax in axes[2][len(exact_stations):]:
        ax.axis("off")
    fig.suptitle(f"u against y, Re = {first['config'].reynolds:g}: " + " vs ".join(labels)
                 + " vs exact", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "u_vs_y.png", dpi=150)
    plt.close(fig)

    # u against x at three heights, the approach to the exact value, and flux.
    x = first["x"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for column, h in enumerate(HEIGHTS):
        target = float(developed(np.array(h)))
        top, middle = axes[0, column], axes[1, column]
        for colour, run, label in zip(colours, runs, labels):
            top.plot(x, run["along"][h], color=colour, linewidth=2, label=label)
            middle.semilogy(x, np.maximum(np.abs(run["along"][h] - target) / target, 1e-12),
                            color=colour, linewidth=1.5, label=label)
        top.axhline(target, color=pinn.REFERENCE, linewidth=1.2, linestyle="--",
                    label=f"exact developed {target:g}")
        top.set_title(f"u against x at y* = {h:g}", loc="left", fontsize=10)
        top.set_xlabel("x*")
        top.set_ylabel("u*")
        top.legend(frameon=False, fontsize=8, loc="upper right" if h > 0.75 else "lower right")
        middle.axhline(DEVELOPED_WITHIN, color=pinn.MUTED, linewidth=1, linestyle=":", label="1%")
        middle.set_title(f"|u - exact developed| / exact, y* = {h:g}", loc="left", fontsize=10)
        middle.set_xlabel("x*")
        middle.legend(frameon=False, fontsize=8, loc="upper right")
        pinn._style(top)
        pinn._style(middle)
    flux_ax, mom_ax, con_ax = axes[2]
    for colour, run, label in zip(colours, runs, labels):
        flux_ax.plot(run["flux_x"], 100.0 * (run["flux_along"] / pinn.INLET_FLUX - 1.0),
                     color=colour, linewidth=1.5, label=label)
        mom_ax.plot(x, run["rxc"], color=colour, linewidth=1.2, label=label)
        con_ax.plot(x, run["rcc"], color=colour, linewidth=1.2, label=label)
    for ax, title, ylabel in (
        (flux_ax, "Flux, integral of u dy, vs exact 4/3", "flux error (%)"),
        (mom_ax, "x-momentum residual on the centreline", "residual (exact 0)"),
        (con_ax, "Continuity residual on the centreline", "residual (exact 0)"),
    ):
        ax.axhline(0.0, color=pinn.REFERENCE, linewidth=1, linestyle="--")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("x*")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        pinn._style(ax)
    fig.suptitle(f"u against x, Re = {first['config'].reynolds:g}: " + " vs ".join(labels)
                 + " vs exact", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "u_vs_x.png", dpi=150)
    plt.close(fig)


def training_summary(directory):
    metrics = json.loads((directory / "metrics.json").read_text())
    with (directory / "history.csv").open() as handle:
        last = list(csv.DictReader(handle))[-1]
    config = pinn.load_config(directory / "config.yaml")
    return {
        "points": config.n_interior + config.rar_points,
        "rar": f"{config.rar_points} worst of {config.rar_pool}" if config.rar_points else "off",
        "minutes": metrics["info_total_seconds"] / 60.0,
        "heldout": metrics["loss_heldout"],
        "ratio": float(last["heldout_over_train"]),
    }


def format_tables(runs, labels, directories):
    first = runs[0]
    lines = [
        "## Table 0. The runs being compared",
        "",
        "| Run | Adaptive refinement (RAR) | Interior points per draw | Training time (min) | "
        "Loss on held-out points | Held-out loss / training loss, last check |",
        "|---|---|---|---|---|---|",
    ]
    for label, directory in zip(labels, directories):
        t = training_summary(directory)
        lines.append(f"| {label} | {t['rar']} | {t['points']:,} | {t['minutes']:.1f} | "
                     f"{t['heldout']:.3e} | {t['ratio']:.2f} |")
    lines += [
        "",
        "## Table 1. u against y: velocity across the channel at each station",
        "",
        "| Station x* | Run | Is there an exact solution here? | Relative L2 error of u vs exact | "
        "Largest pointwise error in u | Flux error vs 4/3 (%) | "
        "Symmetry error, max abs u(y) - u(-y) (exact 0) | Largest abs v (developed: 0) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(first["station_rows"]):
        for run, label in zip(runs, labels):
            r = run["station_rows"][i]
            lines.append(f"| {r['x']:g} | {label} | {r['label']} | {_e(r['l2'])} | {_e(r['max'])} | "
                         f"{r['flux_err']:+.4f} | {r['symmetry']:.1e} | {r['v_max']:.2e} |")
    lines += [
        "",
        f"Exact solutions exist only at the inlet (prescribed) and where the flow is developed "
        f"(x* >= {first['developed_from']:.2f}). Between them there is no closed form, so no error is quoted.",
        "",
        "## Table 2. u against x: velocity along the channel at fixed heights",
        "",
        "| Height y* | Run | Inlet u, exact | Inlet u, PINN | Developed u, exact 1 - y*^2 | "
        "Outlet u, PINN | Outlet relative error vs exact | Distance x* after which u stays within 1% of exact |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(first["height_rows"]):
        for run, label in zip(runs, labels):
            r = run["height_rows"][i]
            lines.append(f"| {r['y']:g} | {label} | {r['inlet_exact']:.5f} | {r['inlet_pinn']:.5f} | "
                         f"{r['developed_exact']:.5f} | {r['outlet_pinn']:.5f} | {r['outlet_err']:.2e} | "
                         f"{r['settle']:.3f} |")
    lines += [
        "",
        "## Table 3. The original equations (exact residual 0), on 80,601 grid points",
        "",
        "| Equation | Run | RMS residual, whole channel | RMS residual, excluding x* < 0.5 | "
        "Largest residual | Where the largest residual is (x*, y*) | RMS on the centreline, developed region |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(first["equation_rows"]):
        for run, label in zip(runs, labels):
            r = run["equation_rows"][i]
            lines.append(f"| {r['name']} | {label} | {r['rms']:.2e} | {r['rms_away']:.2e} | {r['max']:.2e} | "
                         f"({r['where'][0]:.3f}, {r['where'][1]:+.3f}) | {r['centre_developed']:.2e} |")
    sweeps = [(label, directory / "verify.json") for label, directory in zip(labels, directories)]
    if all(path.is_file() for _, path in sweeps):
        lines += [
            "",
            "## Table 4. The original equations on the dense unseen sweep",
            "",
            "| Equation | Run | Points checked | RMS residual | 99.9th percentile of abs residual | "
            "Largest residual | Share of points above 1e-2 |",
            "|---|---|---|---|---|---|---|",
        ]
        data = [(label, json.loads(path.read_text())) for label, path in sweeps]
        for key, name in (("residual_momentum_x", "x-momentum"), ("residual_momentum_y", "y-momentum"),
                          ("residual_continuity", "continuity")):
            for label, d in data:
                s = d[key]
                lines.append(f"| {name} | {label} | {d['points']:,} | {s['rms']:.2e} | {s['p99.9']:.2e} | "
                             f"{s['max']:.2e} | {100 * s['fraction_above_1e-2']:.3f}% |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if len(args.labels) != len(args.runs):
        parser.error("give one label per run")
    if len(args.runs) > len(pinn.SERIES):
        parser.error(f"at most {len(pinn.SERIES)} runs")
    directories = [pathlib.Path(r) for r in args.runs]
    runs = [measure(d) for d in directories]
    stations = [tuple(r["profiles"]) for r in runs]
    if len(set(stations)) != 1 or len({r["config"].length for r in runs}) != 1:
        parser.error("runs must share their channel length and profile stations to be compared")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_figures(runs, args.labels, out)
    tables = format_tables(runs, args.labels, directories)
    (out / "tables.md").write_text(tables, encoding="utf-8")
    print(tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
