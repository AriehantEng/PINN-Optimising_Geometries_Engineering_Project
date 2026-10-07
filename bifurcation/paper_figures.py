"""Figures for the paper, from the stored results.

    python -m bifurcation.paper_figures --out docs/paper

Figure 5.1: the departure of the resolved optimum from the Murray split,
s* - s_M, against daughter 1's share, one panel per bifurcation angle. Layouts
are coloured by role (symmetric, mildly asymmetric, strongly asymmetric), each
line is direct-labelled and has its own marker, and the grey band is the
search tolerance (+-0.001), inside which a departure is not resolved.
"""

from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: categorical slots 1-3 of the validated default palette (dataviz skill), by role
ROLE_COLOUR = {"symmetric": "#2a78d6", "mild": "#eb6834", "strong": "#1baf7a"}
ROLE_MARKER = {"symmetric": "o", "mild": "s", "strong": "^"}
ROLE_NAME = {"symmetric": "symmetric", "mild": "mildly asymmetric", "strong": "strongly asymmetric"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e6e3"


def role(a1, a2, layouts_at_angle):
    """symmetric, or by asymmetry rank among the layouts sharing a bifurcation angle."""
    if a1 == a2:
        return "symmetric"
    asym = sorted(abs(b - a) for a, b in layouts_at_angle if a != b)
    return "strong" if abs(a2 - a1) == asym[-1] else "mild"


def fig_departure(study_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    study = json.loads(pathlib.Path(study_path).read_text())
    rows = [r for r in study["rows"] if r["share"] is not None and r["feasible"]]
    layouts = sorted({tuple(r["angles"]) for r in rows}, key=lambda a: (a[0] + a[1], a[1] - a[0]))
    angles = sorted({a + b for a, b in layouts})
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    for ax, total in zip(axes.ravel(), angles):
        here = [a for a in layouts if a[0] + a[1] == total]
        ax.axhspan(-0.001, 0.001, color=GRID, zorder=0, lw=0)
        ax.axhline(0.0, color=MUTED, lw=0.8, zorder=1)
        for a1, a2 in here:
            rs = sorted((r for r in rows if tuple(r["angles"]) == (a1, a2)), key=lambda r: r["share"])
            q = [r["share"] for r in rs]
            gap = [r["best_split"] - r["friction_law_split"] for r in rs]
            k = role(a1, a2, here)
            ax.plot(q, gap, color=ROLE_COLOUR[k], lw=2, marker=ROLE_MARKER[k], ms=5.5,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            end = max(range(len(gap)), key=lambda i: abs(gap[i]))
            ax.annotate(f"{a1:g}/{a2:g}", (q[end], gap[end]), textcoords="offset points",
                        xytext=(4, 6 if gap[end] >= 0 else -11), fontsize=8, color=INK, zorder=4)
        ax.set_title(f"bifurcation angle {total:g}°", fontsize=9.5, loc="left", color=INK)
        ax.set_xlim(0, 1)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        lim = max(0.003, 1.25 * max(abs(r["best_split"] - r["friction_law_split"])
                                    for r in rows if r["angles"][0] + r["angles"][1] == total))
        ax.set_ylim(-lim, lim)
    for ax in axes[1]:
        ax.set_xlabel("daughter 1's share of the flow, q")
    for ax in axes[:, 0]:
        ax.set_ylabel("s* − s_M  (width split)")
    handles = [plt.Line2D([], [], color=ROLE_COLOUR[k], marker=ROLE_MARKER[k], lw=2, ms=5.5,
                          markeredgecolor="white", label=ROLE_NAME[k]) for k in ROLE_COLOUR]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=GRID, label="search tolerance ±0.001"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, "Each panel has its own vertical scale. Negative: daughter 1 narrower "
             "than the Murray split (daughter 2, the more-turned one, wider).",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="directory for the figure files")
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    fig_departure(ROOT / "docs/bifurcation_re10-shares-all.json", out / "fig5_1_departure.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
