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


def fig_geometry(out_path):
    """Figure 1: the bifurcation polygon with its named points, angles and widths
    (a tip-septum layout), and the face septum at a1 + a2 = 180 as an inset."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Arc, Polygon
    from bifurcation.geometry import Geometry
    plt.rcParams.update({"font.size": 9, "text.color": INK})
    fig, (ax, inset) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [2.3, 1]})
    for axis, g, names in (
            (ax, Geometry.from_split(0.4, 30.0, 70.0, 3.0, 4.0, 0.2),
             ["A", "B", "E₂ₒ", "E₂ᵢ", "T", "E₁ᵢ", "E₁ₒ", "C", "D"]),
            (inset, Geometry.from_split(0.4, 60.0, 120.0, 3.0, 4.0, 0.2),
             ["A", "B", "E₂ₒ", "E₂ᵢ", "", "", "E₁ᵢ", "E₁ₒ", "C", "D"])):
        v = g.vertices()
        axis.add_patch(Polygon(v, closed=True, facecolor="#dde8f4", edgecolor=INK, lw=1.2))
        kinds = [k for _, _, k in g.edges()]
        for i, kind in enumerate(kinds):
            if kind != "wall":
                a, b = v[i], v[(i + 1) % len(v)]
                axis.plot([a[0], b[0]], [a[1], b[1]], color=ROLE_COLOUR["mild"], lw=2.6)
        centre = v.mean(axis=0)
        for p, name in zip(v, names):
            if name == "T":                     # the reflex tip: label in the notch, outside the fluid
                axis.annotate(name, p, xytext=(p[0] + 0.3, p[1]), ha="left", va="center",
                              fontsize=9, color=INK)
            elif name:
                d = p - centre
                d = d / max(np.linalg.norm(d), 1e-9)
                axis.annotate(name, p, xytext=(p[0] + 0.35 * d[0], p[1] + 0.35 * d[1]),
                              ha="center", va="center", fontsize=9, color=INK)
        axis.set_aspect("equal")
        axis.axis("off")
        lo, hi = v.min(axis=0), v.max(axis=0)
        axis.set_xlim(lo[0] - 0.8, hi[0] + 0.8)
        axis.set_ylim(lo[1] - 0.8, hi[1] + 0.8)
    # angles and dimensions on the main panel
    g = Geometry.from_split(0.4, 30.0, 70.0, 3.0, 4.0, 0.2)
    v = g.vertices()
    B, C = v[1], v[7]
    for corner, start, extent, label, side in ((C, 0.0, 30.0, "α₁", 1), (B, -70.0, 70.0, "α₂", -1)):
        ax.plot([corner[0], corner[0] + 1.6], [corner[1], corner[1]], color=MUTED, lw=0.8, ls="--")
        ax.add_patch(Arc(corner, 2.2, 2.2, theta1=start, theta2=start + extent, color=MUTED, lw=0.9))
        mid = math.radians(start + extent / 2)
        ax.text(corner[0] + 1.35 * math.cos(mid), corner[1] + 1.35 * math.sin(mid), label,
                ha="center", va="center", fontsize=10, color=INK)
    ax.annotate("", xy=(0.0, -1.55), xytext=(3.0, -1.55),
                arrowprops={"arrowstyle": "<->", "color": MUTED, "lw": 0.8})
    ax.text(1.5, -1.8, "L_p", ha="center", va="top", fontsize=9, color=INK)
    ax.text(-0.25, 0.0, "inlet\nw₀", ha="right", va="center", fontsize=8.5, color=INK)
    ax.text(v[5][0] + 0.4 * (v[6][0] - v[5][0]) + 0.25, 0.5 * (v[5][1] + v[6][1]) + 0.3,
            "outlet 1, w₁", fontsize=8.5, color=INK)
    ax.text(v[3][0] + 0.55, v[3][1] + 0.35, "outlet 2, w₂", fontsize=8.5, color=INK)
    ax.set_title("Septum tip (here 30°/70°, s = 0.4); daughters L_d long along their outer walls",
                 fontsize=9, loc="left", color=INK)
    inset.set_title("Face septum, α₁ + α₂ = 180° (60°/120°)", fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="directory for the figure files")
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    fig_departure(ROOT / "docs/bifurcation_re10-shares-all.json", out / "fig5_1_departure.png")
    fig_geometry(out / "fig1_geometry.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
