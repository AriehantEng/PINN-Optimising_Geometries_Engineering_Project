"""Plot a scorecard: each setup against the reference setup's seed range.

    python -m developing_flow.plot_scorecard developing_flow/runs/scorecard_step1/scorecard.json \\
        --out developing_flow/runs/scorecard_step1/scorecard.png

One panel per metric, setups down the side, value along a log axis, and the
reference seeds' range as a grey band. A point inside the band is not an
effect of the change. Values below 1e-12 are exact by construction (the
ansatz cannot violate them) and are written as text, not plotted.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from developing_flow import pinn

PANELS = (
    ("developed_u_error", "u error, developed region"),
    ("inlet_u_error", "u error at the inlet"),
    ("wiggle_x10", "wiggle at the outlet (x* = 10)"),
    ("flux_error", "worst flux error"),
    ("rms_x", "RMS x-momentum residual"),
    ("max_residual", "largest residual"),
)
EXACT = 1e-12


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scorecard")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    data = json.loads(pathlib.Path(args.scorecard).read_text())
    summary, reference = data["summary"], data["reference"]
    setups = [label for label in summary if label != reference]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    y = np.arange(len(setups))[::-1]
    for ax, (key, title) in zip(axes.ravel(), PANELS):
        ref = summary[reference][key]
        ax.axvspan(ref["min"], ref["max"], color=pinn.GRID, zorder=0,
                   label=f"{reference}: {ref['n']} seeds")
        ax.axvline(ref["mean"], color=pinn.MUTED, linewidth=1, zorder=1)
        values = np.array([summary[s][key]["mean"] for s in setups])
        finite = values[values > EXACT]
        lo, hi = finite.min() / 2.0, finite.max() * 2.0
        for yi, setup, value in zip(y, setups, values):
            if value <= EXACT:
                ax.text(lo * 1.1, yi, "exact by construction", va="center",
                        fontsize=8, color=pinn.MUTED)
                continue
            if value < ref["min"]:
                colour = pinn.SERIES[2]
            elif value > ref["max"]:
                colour = pinn.SERIES[1]
            else:
                colour = pinn.MUTED
            ax.plot(value, yi, "o", markersize=8, color=colour, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        # A panel spanning less than a decade would otherwise label its minor
        # ticks too, and they collide.
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, axis="x", color=pinn.GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=pinn.MUTED)
    for ax in axes[:, 0]:
        ax.set_yticks(y)
        ax.set_yticklabels(setups, fontsize=9)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=pinn.SERIES[2], label="better than every seed"),
        plt.Line2D([], [], marker="o", linestyle="", color=pinn.MUTED, label="within the seed range"),
        plt.Line2D([], [], marker="o", linestyle="", color=pinn.SERIES[1], label="worse than every seed"),
        plt.Rectangle((0, 0), 1, 1, color=pinn.GRID, label=f"{reference} seed range"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Step 1: each change against the baseline's seed-to-seed range (lower is better)",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
