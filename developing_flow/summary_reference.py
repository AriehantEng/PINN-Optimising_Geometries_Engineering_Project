"""Rank every developing-flow run against the finite-element reference, in one table and graph.

    python -m developing_flow.summary_reference --out docs/developing_re10-summary \\
        --solutions developing_flow/runs/reference_re10_m8 \\
        --reference-setup baseline \\
        baseline=developing_flow/runs/developing_re10 \\
        baseline=developing_flow/runs/developing_re10_seed1 ...

Each run is measured against the reference for its own problem (picked from
--solutions by Reynolds number, scaling, inlet sharpness and length), on every
second point of the reference's stored grid. Runs sharing a label are seeds of
one setup and are shown as a range. The reference's own error, estimated from
its mesh convergence, is drawn on each panel: a run below that line cannot be
ranked against another run below it.

Writes summary.md, summary.json and summary.png into --out.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from developing_flow import pinn, reference
from developing_flow.scorecard import pick_solution

METRICS = (
    ("u", "u, whole channel"),
    ("u_entrance", "u, entrance region"),
    ("u_developed", "u, developed region"),
    ("v", "v, whole channel"),
    ("p", "p, whole channel"),
    ("entrance_length", "entrance length"),
)


def _relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def reference_floor(solution_dir):
    """The reference's own relative L2 errors, from its two finest meshes:
    change / (2^order - 1) at the observed order; entrance length likewise."""
    described = json.loads((pathlib.Path(solution_dir) / "reference.json").read_text())
    last = described["convergence"][-1]
    _, _, ru, rv, _ = reference.load(solution_dir)
    floor = {}
    for name in ("u", "v", "p"):
        order = last[f"{name}_observed_order"]
        change = last[f"{name}_l2_change"]
        if name == "v":  # the convergence study scales v's change by |u|
            change *= np.linalg.norm(ru) / np.linalg.norm(rv)
        floor[name] = change / (2.0**order - 1.0)
    floor["u_entrance"] = floor["u_developed"] = floor["u"]
    floor["entrance_length"] = abs(last["entrance_length_change"]) / described["entrance_length"]
    return floor


def measure(directory, solutions, stride=2):
    model = pinn.load_run(directory)
    config = model.config
    solution = pick_solution(config, solutions)
    x, y, ru, rv, rp = reference.load(solution)
    x, y = x[::stride], y[::stride]
    ru, rv, rp = ru[::stride, ::stride], rv[::stride, ::stride], rp[::stride, ::stride]
    X, Y = np.meshgrid(x, y, indexing="ij")
    u, v, p = model(X, Y)
    entrance = x < 2.0 * pinn.correlation_entrance_length(config.reynolds)
    described = json.loads((pathlib.Path(solution) / "reference.json").read_text())
    metrics = json.loads((pathlib.Path(directory) / "metrics.json").read_text())
    return {
        "u": _relative(u, ru),
        "u_entrance": _relative(u[entrance], ru[entrance]),
        "u_developed": _relative(u[~entrance], ru[~entrance]),
        "v": _relative(v, rv),
        "p": _relative(p, rp),
        "entrance_length": abs(metrics["entrance_length"] / described["entrance_length"] - 1.0)
        if np.isfinite(metrics["entrance_length"]) else float("inf"),
        "minutes": metrics["info_total_seconds"] / 60.0,
        "solution": str(solution),
    }


def summarise(groups):
    out = {}
    for label, rows in groups.items():
        out[label] = {}
        for key, _ in METRICS + (("minutes", ""),):
            values = np.array([r[key] for r in rows])
            out[label][key] = {"mean": float(values.mean()), "min": float(values.min()),
                               "max": float(values.max()), "n": len(values)}
    return out


def format_markdown(summary, floor, reference_setup):
    order = sorted(summary, key=lambda s: summary[s]["u"]["mean"])
    names = [name for _, name in METRICS]
    lines = [
        "## Every setup against the finite-element reference, most accurate first",
        "",
        "Relative L2 errors (entrance length: relative error). Several seeds: mean "
        "(range). The reference's own error is the last row; values near it cannot be "
        "ranked against each other.",
        "",
        "| Setup | Seeds | " + " | ".join(names) + " | Training time (min) |",
        "|---|---|" + "---|" * (len(names) + 1),
    ]
    for label in order:
        s = summary[label]
        n = s["u"]["n"]
        cells = [f"{s[k]['mean']:.2e}" if n == 1 else
                 f"{s[k]['mean']:.2e} ({s[k]['min']:.1e} to {s[k]['max']:.1e})"
                 for k, _ in METRICS]
        mark = " (reference setup)" if label == reference_setup else ""
        lines.append(f"| {label}{mark} | {n} | " + " | ".join(cells)
                     + f" | {s['minutes']['mean']:.0f} |")
    lines.append("| *finite-element reference's own error* | | "
                 + " | ".join(f"{floor[k]:.1e}" for k, _ in METRICS) + " | |")
    return "\n".join(lines) + "\n"


def plot(summary, rows, floor, reference_setup, out, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    order = sorted(summary, key=lambda s: summary[s]["u"]["mean"])[::-1]
    y = np.arange(len(order))
    fig, axes = plt.subplots(2, 3, figsize=(16, 2.2 + 0.42 * len(order) * 2), sharey=True)
    ref = summary[reference_setup]
    for ax, (key, name) in zip(axes.ravel(), METRICS):
        ax.axvspan(ref[key]["min"], ref[key]["max"], color=pinn.GRID, zorder=0)
        ax.axvline(floor[key], color=pinn.SERIES[1], linewidth=1.5, linestyle="--", zorder=1)
        for yi, label in zip(y, order):
            values = [r[key] for r in rows[label]]
            ax.plot(values, [yi] * len(values), "o", markersize=4, color=pinn.MUTED, zorder=2)
            ax.plot(summary[label][key]["mean"], yi, "o", markersize=9, color=pinn.SERIES[0],
                    zorder=3)
        ax.set_xscale("log")
        finite = [v for label in order for v in (r[key] for r in rows[label]) if np.isfinite(v)]
        ax.set_xlim(min(min(finite), floor[key]) / 2.0, max(finite) * 2.0)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(name, loc="left", fontsize=11)
        ax.grid(True, axis="x", color=pinn.GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=pinn.MUTED)
    for ax in axes[:, 0]:
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=9)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=pinn.SERIES[0],
                   label="setup mean"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=pinn.MUTED,
                   label="each seed"),
        plt.Rectangle((0, 0), 1, 1, color=pinn.GRID, label=f"{reference_setup} seed range"),
        plt.Line2D([], [], linestyle="--", color=pinn.SERIES[1],
                   label="the reference's own error"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 0.98))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="label=path; a repeated label means seeds")
    parser.add_argument("--solutions", nargs="+", required=True)
    parser.add_argument("--reference-setup", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    groups = {}
    for item in args.runs:
        label, _, path = item.partition("=")
        if not path:
            parser.error(f"expected label=path, got {item!r}")
        groups.setdefault(label, []).append(measure(pathlib.Path(path), args.solutions))
    if args.reference_setup not in groups:
        parser.error(f"no run is labelled {args.reference_setup!r}")
    floors = {}
    for rows in groups.values():
        for row in rows:
            floors.setdefault(row["solution"], reference_floor(row["solution"]))
    if len(floors) != 1:
        parser.error("the graph draws one reference floor; score runs of one problem at a time")
    floor = next(iter(floors.values()))
    summary = summarise(groups)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    markdown = format_markdown(summary, floor, args.reference_setup)
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "runs": groups, "reference_floor": floor,
                   "reference_setup": args.reference_setup}, handle, indent=2)
    plot(summary, groups, floor, args.reference_setup, out / "summary.png", args.title)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
