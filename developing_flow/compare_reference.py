"""Compare developing-flow runs with the finite-element reference, everywhere.

    python -m developing_flow.compare_reference developing_flow/runs/reference_re10_m8 \\
        "E7 seed 0=developing_flow/runs/developing_re10_stream" \\
        "E7 seed 1=developing_flow/runs/developing_re10_stream_seed1" \\
        --out developing_flow/runs/comparison_stream_seeds

compare_runs.py can only quote an error where a closed-form solution exists:
at the inlet and where the flow is developed. The finite-element reference
(developing_flow/reference.py; its own error is ~1e-6 in u) exists
everywhere, so this compares u, v and p at every station and height,
entrance region included. Reference values are read straight off its stored
grid, so every station and height used here must lie on that grid.

Writes into --out:
    u_vs_y.png       u across the channel at six stations, and the signed error there
    u_vs_x.png       u along the channel at three heights, its error, and v and p
    error_maps.png   |u|, |v| and |p| errors over the whole channel, per run
    tables.md        the numbers
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from developing_flow import pinn, reference

STATIONS = (0.0, 0.5, 1.0, 2.0, 4.0, 10.0)
HEIGHTS = (0.0, 0.5, 0.9)


def _index(grid, value):
    i = int(np.argmin(np.abs(grid - value)))
    if abs(grid[i] - value) > 1e-12:
        raise ValueError(f"{value} is not on the reference grid")
    return i


def _relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def measure(solution_dir, directory, stride=2):
    """Everything the figures and tables need for one run."""
    rx, ry, ru, rv, rp = reference.load(solution_dir)
    described = json.loads((pathlib.Path(solution_dir) / "reference.json").read_text())
    model = pinn.load_run(directory)
    config = model.config
    for key, value in (("reynolds", config.reynolds), ("inlet_sharpness", config.inlet_sharpness),
                       ("pressure_scaling", config.pressure_scaling),
                       ("length_sampled", config.length)):
        if described[key] != value:
            raise ValueError(f"{directory} has {key} = {value}, the reference {described[key]}")
    developed_from = 2.0 * pinn.correlation_entrance_length(config.reynolds)

    out = {"config": config, "y": ry, "x": rx, "stations": {}, "heights": {}}
    for s in STATIONS:
        i = _index(rx, s)
        u, v, p = model(np.full_like(ry, s), ry)
        out["stations"][s] = {"u": u, "v": v, "u_ref": ru[i], "v_ref": rv[i]}
    for h in HEIGHTS:
        j = _index(ry, h)
        u, v, p = model(rx, np.full_like(rx, h))
        out["heights"][h] = {"u": u, "v": v, "p": p, "u_ref": ru[:, j], "v_ref": rv[:, j],
                             "p_ref": rp[:, j]}

    xs, ys = rx[::stride], ry[::stride]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    u, v, p = model(X, Y)
    su, sv, sp = ru[::stride, ::stride], rv[::stride, ::stride], rp[::stride, ::stride]
    out["maps"] = {"x": xs, "y": ys, "u": np.abs(u - su), "v": np.abs(v - sv), "p": np.abs(p - sp)}
    metrics = json.loads((pathlib.Path(directory) / "metrics.json").read_text())
    # The norms are integrated on the reference's mesh, not summed over the
    # grid the maps use: the grid over-weights the inlet column, where the
    # corner pressure spike is narrower than its spacing.
    errors = reference.relative_errors(model, reference.load_solution(solution_dir),
                                       config.length, developed_from)
    out["summary"] = {
        **{key: errors[key] for key in ("u", "u_entrance", "u_developed", "v", "p")},
        "u_max": float(np.max(np.abs(u - su))),
        "entrance_length": pinn.entrance_length(model, config.length),
        "entrance_length_ref": described["entrance_length"],
        "minutes": metrics["info_total_seconds"] / 60.0,
        "heldout": metrics["loss_heldout"],
    }
    out["developed_from"] = developed_from
    return out


def _style(ax):
    pinn._style(ax)


def write_figures(runs, labels, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from developing_flow.plot_reference import _pyplot

    _, cmap = _pyplot()
    colours, ref = pinn.SERIES, pinn.REFERENCE
    first = runs[0]
    y, x = first["y"], first["x"]
    names = " vs ".join(labels)

    # u against y: profiles on top, the signed error below.
    fig, axes = plt.subplots(4, 3, figsize=(14, 16))
    for k, s in enumerate(STATIONS):
        top, bottom = axes[k // 3, k % 3], axes[2 + k // 3, k % 3]
        u_ref = first["stations"][s]["u_ref"]
        for colour, run, label in zip(colours, runs, labels):
            top.plot(run["stations"][s]["u"], y, color=colour, linewidth=2, label=label)
            bottom.plot(run["stations"][s]["u"] - u_ref, y, color=colour, linewidth=1.5,
                        label=f"{label}: {_relative(run['stations'][s]['u'], u_ref):.1e}")
        top.plot(u_ref, y, color=ref, linewidth=1.5, linestyle="--", label="reference")
        top.set_title(f"u* across the channel at x* = {s:g}", loc="left", fontsize=10)
        top.set_xlabel("u*")
        top.set_ylabel("y*")
        top.legend(frameon=False, fontsize=8, loc="center left")
        bottom.axvline(0.0, color=ref, linewidth=1, linestyle="--")
        bottom.set_title(f"u* (PINN) - u* (reference) at x* = {s:g}", loc="left", fontsize=10)
        bottom.set_xlabel("signed error in u*")
        bottom.set_ylabel("y*")
        bottom.xaxis.set_major_locator(plt.MaxNLocator(4))
        bottom.legend(frameon=False, fontsize=8, loc="upper right", title="relative L2",
                      title_fontsize=8)
        _style(top)
        _style(bottom)
    fig.suptitle(f"u against y: {names} vs the finite-element reference", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / "u_vs_y.png", dpi=150)
    plt.close(fig)

    # u against x at three heights, its error, then v and p.
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for column, h in enumerate(HEIGHTS):
        top, middle = axes[0, column], axes[1, column]
        u_ref = first["heights"][h]["u_ref"]
        for colour, run, label in zip(colours, runs, labels):
            top.plot(x, run["heights"][h]["u"], color=colour, linewidth=2, label=label)
            middle.plot(x, run["heights"][h]["u"] - u_ref, color=colour, linewidth=1.2, label=label)
        top.plot(x, u_ref, color=ref, linewidth=1.5, linestyle="--", label="reference")
        top.set_title(f"u* along the channel at y* = {h:g}", loc="left", fontsize=10)
        top.set_xlabel("x*")
        top.set_ylabel("u*")
        top.legend(frameon=False, fontsize=8, loc="lower right" if h < 0.75 else "upper right")
        middle.axhline(0.0, color=ref, linewidth=1, linestyle="--")
        middle.set_title(f"u* (PINN) - u* (reference), y* = {h:g}", loc="left", fontsize=10)
        middle.set_xlabel("x*")
        middle.set_ylabel("signed error in u*")
        middle.legend(frameon=False, fontsize=8, loc="upper right")
        _style(top)
        _style(middle)
    v_ax, dv_ax, dp_ax = axes[2]
    h = 0.5
    for colour, run, label in zip(colours, runs, labels):
        row = run["heights"][h]
        v_ax.plot(x, row["v"], color=colour, linewidth=2, label=label)
        dv_ax.plot(x, row["v"] - row["v_ref"], color=colour, linewidth=1.2, label=label)
        centre = run["heights"][0.0]
        dp_ax.plot(x, centre["p"] - centre["p_ref"], color=colour, linewidth=1.2, label=label)
    v_ax.plot(x, first["heights"][h]["v_ref"], color=ref, linewidth=1.5, linestyle="--",
              label="reference")
    v_ax.set_xlim(0.0, 4.0)
    for ax, title, ylabel in (
        (v_ax, "v* (sideways velocity) at y* = 0.5, first 4 half-heights", "v*"),
        (dv_ax, "v* (PINN) - v* (reference), y* = 0.5", "signed error in v*"),
        (dp_ax, "p* (PINN) - p* (reference) on the centreline", "signed error in p*"),
    ):
        if ax is not v_ax:
            ax.axhline(0.0, color=ref, linewidth=1, linestyle="--")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("x*")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8, loc="lower right" if ax is v_ax else "upper right")
        _style(ax)
    fig.suptitle(f"u against x: {names} vs the finite-element reference", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / "u_vs_x.png", dpi=150)
    plt.close(fig)

    # |error| maps: rows u, v, p; one column per run; one scale per row.
    fig, axes = plt.subplots(3, len(runs), figsize=(6.5 * len(runs), 7.5), squeeze=False,
                             sharex=True, sharey=True)
    for row, name in enumerate(("u", "v", "p")):
        top = max(float(np.percentile(run["maps"][name], 99.9)) for run in runs)
        norm = LogNorm(vmin=top / 1e4, vmax=top)
        for col, (run, label) in enumerate(zip(runs, labels)):
            ax = axes[row, col]
            m = run["maps"]
            image = ax.pcolormesh(m["x"], m["y"], np.clip(m[name], top / 1e4, None).T,
                                  cmap=cmap, norm=norm, shading="auto")
            ax.set_title(f"{label}: |{name}* - {name}*_ref|, relative L2 "
                         f"{run['summary'][name]:.2e}", loc="left", fontsize=10)
            if col == 0:
                ax.set_ylabel("y*")
            if row == 2:
                ax.set_xlabel("x*")
        fig.colorbar(image, ax=axes[row, :].tolist(), pad=0.01, label=f"|{name}* error|")
    fig.suptitle(f"Where each run is wrong: {names}", x=0.01, ha="left")
    fig.savefig(out / "error_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_tables(runs, labels):
    first = runs[0]
    s0 = first["summary"]
    lines = [
        "## Table 1. Whole-channel accuracy against the finite-element reference",
        "",
        "Relative L2 norms integrated on the reference's finite-element mesh "
        f"({reference.NORM_GAUSS} x {reference.NORM_GAUSS} Gauss points in every element); the "
        "largest error in u is over the sampling grid. The reference's own error is in "
        "docs/developing_re10-reference.json.",
        "",
        "| Run | u, relative L2, whole channel | u, entrance region x* < "
        f"{first['developed_from']:.2f} | u, developed region | largest error in u | "
        "v, relative L2 | p, relative L2 | entrance length (reference "
        f"{s0['entrance_length_ref']:.4f}) | held-out loss | training time (min) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run, label in zip(runs, labels):
        s = run["summary"]
        lines.append(
            f"| {label} | {s['u']:.2e} | {s['u_entrance']:.2e} | {s['u_developed']:.2e} | "
            f"{s['u_max']:.2e} | {s['v']:.2e} | {s['p']:.2e} | {s['entrance_length']:.4f} "
            f"({100 * (s['entrance_length'] / s['entrance_length_ref'] - 1):+.2f}%) | "
            f"{s['heldout']:.2e} | {s['minutes']:.1f} |")
    lines += [
        "",
        "## Table 2. u across the channel at each station, against the reference",
        "",
        "| Station x* | Run | u, relative L2 | largest error in u | where (y*) | "
        "v, largest error | v, largest reference value |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in STATIONS:
        for run, label in zip(runs, labels):
            row = run["stations"][s]
            err = row["u"] - row["u_ref"]
            k = int(np.argmax(np.abs(err)))
            lines.append(
                f"| {s:g} | {label} | {_relative(row['u'], row['u_ref']):.2e} | "
                f"{abs(err[k]):.2e} | {run['y'][k]:+.3f} | "
                f"{np.max(np.abs(row['v'] - row['v_ref'])):.2e} | {np.max(np.abs(row['v_ref'])):.2e} |")
    lines += [
        "",
        "## Table 3. u along the channel at each height, against the reference",
        "",
        "| Height y* | Run | u at x* = 1, PINN | reference | u at x* = 2, PINN | reference | "
        "outlet u, PINN | reference | largest error along x | where (x*) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    x = first["x"]
    i1, i2 = _index(x, 1.0), _index(x, 2.0)
    for h in HEIGHTS:
        for run, label in zip(runs, labels):
            row = run["heights"][h]
            err = row["u"] - row["u_ref"]
            k = int(np.argmax(np.abs(err)))
            lines.append(
                f"| {h:g} | {label} | {row['u'][i1]:.5f} | {row['u_ref'][i1]:.5f} | "
                f"{row['u'][i2]:.5f} | {row['u_ref'][i2]:.5f} | {row['u'][-1]:.5f} | "
                f"{row['u_ref'][-1]:.5f} | {abs(err[k]):.2e} | {x[k]:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("solution", help="a reference directory from developing_flow.reference")
    parser.add_argument("runs", nargs="+", help="label=path, one per run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if len(args.runs) > len(pinn.SERIES):
        parser.error(f"at most {len(pinn.SERIES)} runs")
    labels, directories = [], []
    for item in args.runs:
        label, _, path = item.partition("=")
        if not path:
            parser.error(f"expected label=path, got {item!r}")
        labels.append(label)
        directories.append(pathlib.Path(path))
    runs = [measure(args.solution, d) for d in directories]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_figures(runs, labels, out)
    tables = format_tables(runs, labels)
    (out / "tables.md").write_text(tables, encoding="utf-8")
    print(tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
