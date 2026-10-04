"""Rank every developing-flow run against the finite-element reference, in one table and graph.

    python -m developing_flow.summary_reference --out docs/developing_re10-summary \\
        --solutions developing_flow/runs/reference_re10_m8 \\
        --reference-setup baseline \\
        baseline=developing_flow/runs/developing_re10 \\
        baseline=developing_flow/runs/developing_re10_seed1 ...

Each run is measured against the reference for its own problem (picked from
--solutions by Reynolds number, scaling, inlet sharpness and length), by
relative L2 norms integrated on the reference's own finite-element mesh
(reference.relative_errors; a uniform grid over-weights the inlet corners,
where a pressure spike narrower than its spacing sits), and by its energy budget
(developing_flow/energy.py): the flux-weighted total-pressure drop, which is
the bifurcation study's objective, the viscous dissipation, and how well the
two balance with no reference at all. Runs sharing a label are seeds of
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

from developing_flow import energy, pinn, reference
from developing_flow.scorecard import pick_solution

METRICS = (
    ("u", "u, whole channel"),
    ("u_entrance", "u, entrance region"),
    ("u_developed", "u, developed region"),
    ("v", "v, whole channel"),
    ("p", "p, whole channel"),
    ("entrance_length", "entrance length"),
    ("total_pressure_drop", "total-pressure drop (the design objective)"),
    ("dissipation", "viscous dissipation"),
    ("resistance", "resistance: dissipation / flux^2 (flux error cancelled)"),
    ("energy_consistency", "energy budget, PINN alone (no reference)"),
)


def reference_floor(solution_dir):
    """The reference's own relative L2 errors, from its two finest meshes:
    change / (2^order - 1) at the observed order; entrance length likewise.
    The changes must be mesh-integrated norms, the same norm the runs are
    scored in; a reference whose convergence study used a sampling grid is
    refused rather than mixed in."""
    described = json.loads((pathlib.Path(solution_dir) / "reference.json").read_text())
    if "convergence_norm" not in described:
        raise ValueError(f"{solution_dir}: its convergence study predates mesh-integrated norms "
                         "(it over-weights the inlet column); rerun developing_flow.reference")
    last = described["convergence"][-1]
    solution = reference.load_solution(solution_dir)
    X, Y, W = reference.norm_quadrature(solution.x_nodes, solution.y_nodes,
                                        described["length_sampled"])
    ru, rv, _ = solution(X, Y)
    floor = {}
    for name in ("u", "v", "p"):
        order = last[f"{name}_observed_order"]
        change = last[f"{name}_l2_change"]
        if name == "v":  # the convergence study scales v's change by |u|
            change *= np.sqrt(np.sum(W * ru**2) / np.sum(W * rv**2))
        floor[name] = change / (2.0**order - 1.0)
    floor["u_entrance"] = floor["u_developed"] = floor["u"]
    floor["entrance_length"] = abs(last["entrance_length_change"]) / described["entrance_length"]
    gci = described["grid_convergence_index"]
    floor["total_pressure_drop"] = gci["total_pressure_drop"]["gci_fine"]
    floor["dissipation"] = gci["dissipation"]["gci_fine"]
    # The reference's flux is exact to 5e-8 (|flux / (4/3) - 1|), so its
    # resistance is as good as its dissipation: the same GCI bounds it.
    floor["resistance"] = gci["dissipation"]["gci_fine"]
    floor["energy_consistency"] = described["energy"]["energy_consistency"]
    return floor


def reference_resistance(described):
    """dissipation / (mean flux)^2 of the reference, as energy.budget defines it."""
    energy_ = described["energy"]
    return energy_["dissipation"] / (0.5 * (energy_["flux_in"] + energy_["flux_out"])) ** 2


def measure(directory, solutions):
    model = pinn.load_run(directory)
    config = model.config
    solution = pick_solution(config, solutions)
    errors = reference.relative_errors(
        model, reference.load_solution(solution), config.length,
        2.0 * pinn.correlation_entrance_length(config.reynolds))
    described = json.loads((pathlib.Path(solution) / "reference.json").read_text())
    entrance = pinn.entrance_length(model, config.length)
    metrics = json.loads((pathlib.Path(directory) / "metrics.json").read_text())
    budget = energy.run_budget(directory)
    return {
        "total_pressure_drop": abs(budget["total_pressure_drop"]
                                   / described["energy"]["total_pressure_drop"] - 1.0),
        "dissipation": abs(budget["dissipation"] / described["energy"]["dissipation"] - 1.0),
        "resistance": abs(budget["resistance"] / reference_resistance(described) - 1.0),
        "energy_consistency": budget["energy_consistency"],
        **{key: errors[key] for key in ("u", "u_entrance", "u_developed", "v", "p")},
        # Recomputed, not read from metrics.json: runs before CENTRELINE_POINTS
        # measured it with a 1,001-point centreline and its interpolation bias.
        "entrance_length": abs(entrance / described["entrance_length"] - 1.0)
        if np.isfinite(entrance) else float("inf"),
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
    fig, axes = plt.subplots(3, 3, figsize=(16, 2.6 + 0.42 * len(order) * 3), sharey=True)
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
