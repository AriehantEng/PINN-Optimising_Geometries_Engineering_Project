"""Figures 1-6 and 9 of the parametric-PINN sections (docs/bifurcation_re10-advanced.md).

    python docs/figures/make_figures.py

Reads the trained runs (bifurcation/runs/...), their metrics and histories, the
error diagnostic (docs/bifurcation_re10-diagnostic.json) and the finite-element
fields sampled on a grid by `python -m bifurcation.fields --config
bifurcation/fields_re10.yaml fe` (cached in runs/fields_re10). Figures 7 and 8
are fig_a_time_scaling.py and fig_b_break_even.py, which stand alone.

Colours follow the entity in every figure: finite elements ink, the original
network orange, the corner-mode network blue, the four continuations aqua,
yellow, magenta and green (validated categorical slots).
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = pathlib.Path(__file__).resolve().parent

from bifurcation import corners as K  # noqa: E402
from bifurcation import parametric as P  # noqa: E402
from bifurcation.geometry import Geometry  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RUNS = {  # label: (run directory, colour)
    "original": ("parametric_re10_3090_equal", "#eb6834"),
    "corner modes": ("parametric_re10_3090_equal_corners", "#2a78d6"),
    "1: continue": ("parametric_re10_3090_corners_more", "#1baf7a"),
    "2: ease corner weight": ("parametric_re10_3090_corners_ease", "#eda100"),
    "3: next corner terms": ("parametric_re10_3090_corners_next", "#e87ba4"),
    "4: wider (56)": ("parametric_re10_3090_corners_wide", "#008300"),
}
SPLITS = (0.35, 0.4848, 0.65)        # the cached finite-element fields
FE_OPTIMUM = 0.48475260310208196


def metrics(label):
    return json.loads((ROOT / "bifurcation/runs" / RUNS[label][0] / "metrics.json").read_text())


def history(label):
    return json.loads((ROOT / "bifurcation/runs" / RUNS[label][0] / "history.json").read_text())


def style(ax, grid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    if grid:
        ax.grid(color=GRID, lw=0.8)


def fe_field(s):
    path = ROOT / "runs/fields_re10" / f"fe_parametric_s{s:.4f}.npz"
    if not path.is_file():
        raise SystemExit(f"{path} missing: run python -m bifurcation.fields "
                         "--config bifurcation/fields_re10.yaml fe")
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in ("X", "Y", "inside", "u", "v", "p", "vertices")}, \
        json.loads(str(z["meta"]))


def outline(ax, vertices, colour=INK, lw=0.8):
    v = np.vstack([vertices, vertices[:1]])
    ax.plot(v[:, 0], v[:, 1], color=colour, lw=lw)


# --------------------------------------------------------------------------- #

def figure1():
    """Geometry: outlines at three splits, the reflex corners, their exponents."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    greys = ("#b9b8b2", INK, "#7a7974")
    for s, grey in zip(SPLITS, greys):
        g = Geometry.from_split(s, 30.0, 90.0, 3.0, 4.0, 0.2)
        for a, b, kind in g.edges():
            colour = "#eb6834" if kind != "wall" else grey
            ax.plot([a[0], b[0]], [a[1], b[1]], color=colour,
                    lw=2.0 if s == SPLITS[1] else 1.1, solid_capstyle="round")
    g = Geometry.from_split(SPLITS[1], 30.0, 90.0, 3.0, 4.0, 0.2)
    v = g.vertices()
    names = {}
    for corner in g.singular_corners():
        i = int(np.argmin(np.linalg.norm(v - corner, axis=1)))
        c = K.corner(v, i, 2.0)
        real = [m.z for m in c.modes if not isinstance(m.z, complex) and m.z < 1.0]
        name = {210: "B", 270: "C", 240: "T"}[round(np.degrees(c.angle))]
        names[name] = (corner, np.degrees(c.angle), real)
        ax.add_patch(plt.Circle(corner, 0.5, fill=False, ls="--", lw=0.8, color="#2a78d6"))
        ax.plot(*corner, "o", color="#2a78d6", ms=5, zorder=4)
    places = {"B": (0.2, 2.6), "C": (0.2, -2.6), "T": (5.0, -0.9)}
    for name, (corner, angle, real) in names.items():
        zs = ", ".join(f"{z:.4f}" for z in real)
        ax.annotate(f"{name}: {angle:.0f}° of fluid\nz = {zs}", corner, xytext=places[name],
                    fontsize=8.5, color=INK, va="center",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    ax.set_aspect("equal")
    ax.set_xlabel("x*", fontsize=9, color=INK)
    ax.set_ylabel("y*", fontsize=9, color=INK)
    style(ax, grid=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_geometry.png", dpi=200)
    plt.close(fig)


def figure2():
    """Reference flow at three splits: speed with streamlines, and pressure."""
    fields = [fe_field(s) for s in SPLITS]
    speed_max = max(np.nanmax(np.hypot(f["u"], f["v"])[f["inside"]]) for f, _ in fields)
    p_max = max(np.nanmax(np.abs(f["p"][f["inside"]])) for f, _ in fields)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
    levels = np.linspace(0.0, speed_max, 17)
    for col, (f, meta) in enumerate(fields):
        inside = f["inside"]
        X, Y = f["X"], f["Y"]
        speed = np.ma.masked_where(~inside, np.hypot(f["u"], f["v"]))
        a = axes[0, col]
        cs = a.contourf(X, Y, speed, levels=levels, cmap="viridis")
        a.streamplot(X[0, :] if X[0, 0] != X[0, -1] else X[:, 0],
                     Y[:, 0] if Y[0, 0] != Y[-1, 0] else Y[0, :],
                     np.ma.masked_where(~inside, f["u"]), np.ma.masked_where(~inside, f["v"]),
                     color="white", linewidth=0.5, density=1.1, arrowsize=0.6)
        outline(a, f["vertices"])
        a.set_title(f"s = {meta['split']:g}:  J = {meta['total_pressure_drop']:.4f},  "
                    f"λ = {meta['outlet1_pressure_offset']:+.3f}", fontsize=9, color=INK)
        b = axes[1, col]
        pc = b.contourf(X, Y, np.ma.masked_where(~inside, f["p"]),
                        levels=np.linspace(-p_max, p_max, 25), cmap="coolwarm",
                        norm=TwoSlopeNorm(0.0, -p_max, p_max))
        outline(b, f["vertices"])
        for ax in (a, b):
            ax.set_aspect("equal")
            style(ax, grid=False)
    fig.colorbar(cs, ax=axes[0, :], shrink=0.85, label="speed |u|")
    fig.colorbar(pc, ax=axes[1, :], shrink=0.85, label="pressure p (outlet 2 at 0)")
    fig.savefig(OUT / "fig2_fe_fields.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def figure3():
    """Pointwise velocity error of three networks against the reference."""
    shown = ("original", "corner modes", "4: wider (56)")
    models = {label: P.load_run(ROOT / "bifurcation/runs" / RUNS[label][0])[0] for label in shown}
    fig, axes = plt.subplots(len(shown), 3, figsize=(13.5, 4.0 * len(shown)))
    norm = LogNorm(1e-5, 1e-1)
    for col, s in enumerate(SPLITS):
        f, meta = fe_field(s)
        inside = f["inside"]
        g = Geometry.from_split(s, 30.0, 90.0, 3.0, 4.0, 0.2)
        for row, label in enumerate(shown):
            u, v, _ = models[label](f["X"][inside], f["Y"][inside], np.full(inside.sum(), s))
            err = np.full(f["X"].shape, np.nan)
            err[inside] = np.hypot(u - f["u"][inside], v - f["v"][inside])
            ax = axes[row, col]
            im = ax.pcolormesh(f["X"], f["Y"], np.ma.masked_invalid(err), cmap="magma",
                               norm=norm, shading="auto")
            outline(ax, f["vertices"], colour=MUTED)
            for c in g.singular_corners():
                ax.plot(*c, "x", color="white", ms=6, mew=1.2)
            m = min(metrics(label)["splits"], key=lambda r: abs(r["split"] - s))
            if row == 0:
                ax.set_title(f"s = {s:g}", fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel(label, fontsize=10, color=INK)
            ax.text(0.97, 0.05, f"relative L2 of u\n{m['relative_l2']['u']:.1e}\n(at s = {m['split']:g})",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
            ax.set_aspect("equal")
            style(ax, grid=False)
    fig.colorbar(im, ax=axes, shrink=0.6, label="|u_PINN − u_FE| (log scale)")
    fig.savefig(OUT / "fig3_error_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure4():
    """Velocity error against distance to the nearest corner (diagnostic)."""
    d = json.loads((ROOT / "docs/bifurcation_re10-diagnostic.json").read_text())
    edges = d["config"]["corner_bins"]
    centres = [np.sqrt(max(a, 0.01) * b) if b < 50 else 2.0 * a for a, b in zip(edges, edges[1:])]
    fig, axes = plt.subplots(1, len(d["splits"]), figsize=(12.5, 3.9), sharey=True)
    for ax, entry in zip(axes, d["splits"]):
        for r in entry["runs"]:
            ax.loglog(centres, r["velocity_error_local"], color=RUNS[r["label"]][1], lw=2,
                      marker="o", ms=6, label=r["label"])
        ax.set_title(f"s = {entry['split']:g}", fontsize=10, color=INK)
        ax.set_xlabel("distance r to the nearest reflex corner", fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("local relative velocity error in the annulus", fontsize=9, color=INK)
    axes[0].legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_error_vs_corner_distance.png", dpi=200)
    plt.close(fig)


def figure5():
    """Training: held-out loss against time, and the four continuations."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for label in ("original", "corner modes"):
        h = history(label)
        a.semilogy([r["seconds"] / 60 for r in h], [r["heldout_loss"] for r in h],
                   color=RUNS[label][1], lw=1.6, label=label)
        best = metrics(label)["training"]
        where = best["selected"]
        a.plot([], [])
        sel = [r for r in h if f"{r['phase']}:{r['iteration']}" == where.replace("_average", "")]
        if sel:
            a.plot(sel[0]["seconds"] / 60, best["selected_heldout_loss"], "o", mfc="white",
                   mec=RUNS[label][1], ms=8, mew=1.6, zorder=4)
            a.annotate(f"{best['selected_heldout_loss']:.2e}", (sel[0]["seconds"] / 60,
                       best["selected_heldout_loss"]), xytext=(-8, -4), ha="right",
                       textcoords="offset points", fontsize=8, color=INK)
    h = history("corner modes")
    ends = {}
    for r in h:
        ends[r["phase"]] = r["seconds"] / 60
    start = 0.0
    for phase, shade in (("adam", "#f3f2ee"), ("lbfgs", "#e9e8e3"), ("lm", "#f3f2ee")):
        a.axvspan(start, ends[phase], color=shade, zorder=0, lw=0)
        a.text((start + ends[phase]) / 2, 0.97, {"adam": "Adam", "lbfgs": "L-BFGS",
               "lm": "Levenberg–Marquardt"}[phase], transform=a.get_xaxis_transform(),
               ha="center", va="top", fontsize=8, color=MUTED)
        start = ends[phase]
    a.set_xlabel("training time (min)", fontsize=9, color=INK)
    a.set_ylabel("held-out loss (a diagnostic, not accuracy)", fontsize=9, color=INK)
    a.legend(fontsize=8.5, frameon=False, loc="lower left")
    style(a)
    for label in ("1: continue", "2: ease corner weight", "3: next corner terms", "4: wider (56)"):
        h = [r for r in history(label) if r["phase"] == "lm"]
        b.semilogy([r["iteration"] for r in h], [r["heldout_loss"] for r in h],
                   color=RUNS[label][1], lw=1.4, ls="--" if label.startswith("2") else "-",
                   label=label + (" (other weights:\nloss not comparable)"
                                  if label.startswith("2") else ""))
    b.axhline(metrics("corner modes")["training"]["selected_heldout_loss"], color=RUNS["corner modes"][1],
              lw=1, ls=":", label="corner modes, selected (start)")
    b.set_xlabel("Levenberg–Marquardt step of the continuation", fontsize=9, color=INK)
    b.set_ylabel("held-out loss (seed 22's held-out points)", fontsize=9, color=INK)
    b.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    style(b)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_training.png", dpi=200)
    plt.close(fig)


def figure6():
    """The design objective J(s), a zoom on its optimum, and the drop error."""
    fig, (a, z, b) = plt.subplots(1, 3, figsize=(15.5, 4.6),
                                  gridspec_kw={"width_ratios": [1.1, 1.0, 1.1]})
    ref = metrics("corner modes")["splits"]
    shown = ("original", "corner modes", "4: wider (56)")
    for ax in (a, z):
        for label in shown:
            m = metrics(label)["optimum"]
            ax.plot(m["curve"]["split"], m["curve"]["pinn_drop"], color=RUNS[label][1], lw=1.8,
                    label=f"{label}: s* = {m['pinn_split']:.4f}")
        ax.plot([r["split"] for r in ref], [r["reference_drop"] for r in ref], "o", color=INK,
                ms=6, zorder=4, label=f"finite elements: s* = {FE_OPTIMUM:.4f}")
    fe_min = metrics("corner modes")["optimum"]["reference_min_drop"]
    z.plot([FE_OPTIMUM], [fe_min], "*", color=INK, ms=11, zorder=5)
    for label in shown:
        m = metrics(label)["optimum"]
        z.plot([m["pinn_split"]], [m["pinn_min_drop"]], "D", color=RUNS[label][1], ms=6, zorder=5)
    z.axvline(0.5, color=MUTED, lw=0.9, ls="-.")
    z.annotate("Murray split", (0.5, 4.027), xytext=(4, 0), textcoords="offset points",
               fontsize=8, color=MUTED)
    z.set_xlim(0.465, 0.51)
    z.set_ylim(4.025, 4.062)
    z.set_title("zoom on the optimum: star finite elements, diamonds PINNs", fontsize=9,
                color=INK)
    a.set_ylabel("total-pressure drop J", fontsize=9, color=INK)
    a.legend(fontsize=7.5, frameon=False, loc="upper center")
    for label in RUNS:
        rows = metrics(label)["splits"]
        b.plot([r["split"] for r in rows], [100 * r["drop_error"] for r in rows],
               color=RUNS[label][1], lw=1.6, marker="o", ms=4, label=label)
    b.axhline(0.0, color=MUTED, lw=0.8)
    b.set_ylabel("drop error against finite elements (%)", fontsize=9, color=INK)
    b.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower left")
    for ax in (a, z, b):
        ax.set_xlabel("width split s = w₁/w₀", fontsize=9, color=INK)
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_objective.png", dpi=200)
    plt.close(fig)


def figure9():
    """The four accuracy options against their starting network."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for label in list(RUNS)[1:]:
        rows = metrics(label)["splits"]
        s = [r["split"] for r in rows]
        a.semilogy(s, [r["relative_l2"]["u"] for r in rows], color=RUNS[label][1], lw=1.8,
                   marker="o", ms=5, label=label, ls="--" if label == "corner modes" else "-")
        b.plot(s, [100 * abs(r["drop_error"]) for r in rows], color=RUNS[label][1], lw=1.8,
               marker="o", ms=5, label=label, ls="--" if label == "corner modes" else "-")
    a.set_ylabel("u, relative L2 error against finite elements", fontsize=9, color=INK)
    b.set_ylabel("|drop error| (%)", fontsize=9, color=INK)
    for ax in (a, b):
        ax.set_xlabel("width split s = w₁/w₀", fontsize=9, color=INK)
        style(ax)
    a.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_options.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    chosen = sys.argv[1:] or ["figure1", "figure4", "figure5", "figure6", "figure9", "figure2",
                               "figure3"]
    for make in (globals()[name] for name in chosen):
        make()
        print("done", make.__name__, flush=True)
