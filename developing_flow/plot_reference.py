"""Figures for the finite-element reference and for PINN errors against it.

    python -m developing_flow.plot_reference solution developing_flow/runs/reference_re10_m8 \\
        --out docs/developing_re10-reference.png
    python -m developing_flow.plot_reference errors developing_flow/runs/reference_re10_m8 \\
        --out docs/developing_re10-error-maps.png \\
        "baseline (seed 0)=developing_flow/runs/developing_re10" ...

`solution` draws what the reference is and how far to trust it: the u field,
how far the flow still is from the parabola along the channel, the mesh
convergence, and the centreline. `errors` maps |u_PINN - u_ref| over the
channel for each run, on one shared logarithmic colour scale, so where each
run is wrong can be compared at a glance.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from developing_flow import pinn, reference

#: Sequential blue ramp, light to dark (steps 100-700 of the project palette).
BLUES = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    return plt, LinearSegmentedColormap.from_list("blues", BLUES)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=pinn.MUTED)
    ax.grid(True, color=pinn.GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_solution(solution_dir, out):
    plt, cmap = _pyplot()
    solution_dir = pathlib.Path(solution_dir)
    x, y, u, _, _ = reference.load(solution_dir)
    described = json.loads((solution_dir / "reference.json").read_text())
    parabola = pinn.developed_profile(y, pinn.INLET_FLUX)
    departure = np.max(np.abs(u - parabola[None, :]), axis=1)
    centre = int(np.argmin(np.abs(y)))
    entrance = described["entrance_length"]
    developed_from = 2.0 * pinn.correlation_entrance_length(described["reynolds"])

    fig = plt.figure(figsize=(13, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15), hspace=0.42, wspace=0.25)

    ax = fig.add_subplot(grid[0, :])
    image = ax.pcolormesh(x[::4], y[::4], u[::4, ::4].T, cmap=cmap, shading="auto")
    ax.axvline(entrance, color=pinn.SERIES[1], linewidth=1.5)
    ax.text(entrance + 0.08, 0.72, f"entrance length {entrance:.4f}", color=pinn.REFERENCE, fontsize=9)
    ax.set_xlabel("x*")
    ax.set_ylabel("y*")
    ax.set_title(f"(a) u*, finite-element reference, Re = {described['reynolds']:g}, "
                 f"inlet sharpness m = {described['inlet_sharpness']}", loc="left", fontsize=11)
    fig.colorbar(image, ax=ax, pad=0.01, label="u*")

    ax = fig.add_subplot(grid[1, 0])
    ax.semilogy(x, departure, color=pinn.SERIES[0], linewidth=2)
    ax.axvline(developed_from, color=pinn.MUTED, linewidth=1, linestyle="--")
    ax.text(developed_from + 0.1, 3e-1, f"'developed' from x* = {developed_from:.2f}",
            color=pinn.REFERENCE, fontsize=9)
    ax.axhline(3e-4, color=pinn.SERIES[1], linewidth=1)
    ax.text(6.0, 4.2e-4, "typical PINN error, 3e-4", color=pinn.REFERENCE, fontsize=9)
    ax.set_xlabel("x*")
    ax.set_ylabel("largest |u* - (1 - y*^2)| across the channel")
    ax.set_title("(b) how far the true flow still is from the parabola", loc="left", fontsize=11)
    ax.text(4.6, 1.5e-8, "floor: the reference's own error, ~5e-8", color=pinn.REFERENCE, fontsize=9)
    ax.set_ylim(1e-8, 1.0)
    _style(ax)

    ax = fig.add_subplot(grid[1, 1])
    levels = described["levels"]
    finer = [row["fine"][0] for row in described["convergence"]]
    for i, name in enumerate(("u", "v", "p")):
        change = [row[f"{name}_l2_change"] for row in described["convergence"]]
        order = described["convergence"][-1][f"{name}_observed_order"]
        ax.loglog(finer, change, "o-", color=pinn.SERIES[i], linewidth=2, markersize=8,
                  label=f"{name}, observed order {order:.2f}")
    ax.set_xticks(finer)
    ax.set_xticklabels([f"{row['fine'][0]} x {row['fine'][1]}" for row in described["convergence"]])
    ax.minorticks_off()
    ax.set_xlabel("finer mesh (elements)")
    ax.set_ylabel("relative L2 change from the previous mesh")
    ax.set_title(f"(c) mesh convergence, {levels[0]['nx']}x{levels[0]['ny']} "
                 f"to {levels[-1]['nx']}x{levels[-1]['ny']} (Q2/Q1 theory: 3, 3, 2)",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    _style(ax)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def plot_errors(solution_dir, runs, out, stride=2):
    plt, cmap = _pyplot()
    from matplotlib.colors import LogNorm

    x, y, ru, _, _ = reference.load(solution_dir)
    x, y, ru = x[::stride], y[::stride], ru[::stride, ::stride]
    X, Y = np.meshgrid(x, y, indexing="ij")
    errors = []
    for label, directory in runs:
        model = pinn.load_run(directory)
        u, _, _ = model(X, Y)
        errors.append((label, np.abs(u - ru)))
    norm = LogNorm(vmin=1e-6, vmax=1e-2)
    fig, axes = plt.subplots(len(errors), 1, figsize=(12, 1.9 * len(errors) + 0.6),
                             sharex=True, squeeze=False)
    for ax, (label, error) in zip(axes[:, 0], errors):
        image = ax.pcolormesh(x, y, np.clip(error, 1e-6, None).T, cmap=cmap, norm=norm,
                              shading="auto")
        relative = np.linalg.norm(error) / np.linalg.norm(ru)
        ax.set_title(f"{label}: |u* - u*_ref|, relative L2 {relative:.2e}", loc="left", fontsize=10)
        ax.set_ylabel("y*")
    axes[-1, 0].set_xlabel("x*")
    fig.colorbar(image, ax=axes[:, 0], pad=0.01, label="|u* - u*_ref|")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("solution")
    one.add_argument("solution")
    one.add_argument("--out", required=True)
    two = sub.add_parser("errors")
    two.add_argument("solution")
    two.add_argument("runs", nargs="+", help="label=path")
    two.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "solution":
        plot_solution(args.solution, args.out)
    else:
        runs = []
        for item in args.runs:
            label, _, path = item.partition("=")
            if not path:
                parser.error(f"expected label=path, got {item!r}")
            runs.append((label, pathlib.Path(path)))
        plot_errors(args.solution, runs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
