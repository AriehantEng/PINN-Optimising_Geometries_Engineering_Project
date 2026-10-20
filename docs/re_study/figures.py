"""Figures of the Reynolds-number study, from the exported runs in docs/re_study.

    python docs/re_study/figures.py

    fig_re_objective.png   J(s) from each finished PINN against the reference solution
    fig_re_drop_error.png  the PINNs' signed pressure-drop error against s, a panel per Re_h
    fig_re_training.png    held-out loss against training time, a panel per Re_h
Redraw as runs finish: every finished run's files are picked up, and a figure
that has not drawn every one of them refuses to save.
"""
import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                               "#e87ba4", "#008300")
COLOURS = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN)
# one look per network, the same in every panel: colour by seed, and 30/120
# dashed with square markers so it never rests on colour alone
LOOK = {("30/90", 21): (BLUE, "-", "o"), ("30/90", 22): (ORANGE, "-", "o"),
        ("30/90", 23): (AQUA, "-", "o"), ("30/120", 21): (MAGENTA, "--", "s")}


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.8)


def runs():
    out = []
    for path in sorted(HERE.glob("pinn_*_summary.json")):
        s = json.loads(path.read_text())
        if "failed" in s:
            continue
        stem = path.name[: -len("_summary.json")]
        rows = list(csv.DictReader((HERE / f"{stem}.csv").open()))
        loss = list(csv.DictReader((HERE / f"{stem}_loss.csv").open()))
        layout = f"{s['layout'][0]:g}/{s['layout'][1]:g}"
        label = f"{layout}, Re_h = {s['reynolds']:g}, seed {s['seed']}"
        out.append({"summary": s, "rows": rows, "loss": loss, "label": label,
                    "network": (layout, s["seed"])})
    out.sort(key=lambda r: (r["summary"]["layout"][1], r["summary"]["reynolds"],
                            r["summary"]["seed"]))
    return out


def objective(all_runs):
    n = len(all_runs)
    cols = 3
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(14.5, 3.5 * ((n + cols - 1) // cols)),
                             squeeze=False)
    for ax, r in zip(axes.flat, all_runs):
        s = r["summary"]
        x = [float(row["s"]) for row in r["rows"]]
        ax.plot(x, [float(row["J_pinn"]) for row in r["rows"]], color=BLUE, lw=2,
                label="PINN, J(s) over 61 splits")
        fe = [(float(row["s"]), float(row["J_fe"])) for row in r["rows"] if row["J_fe"]]
        ax.plot(*zip(*fe), "o", color=INK, ms=6, label="reference solution")
        ax.axvline(s["s_star_pinn"], color=BLUE, ls="--", lw=1.2,
                   label=f"PINN optimum {s['s_star_pinn']:.4f}")
        ax.axvline(s["s_star_reference_mesh_brent"], color=INK, ls=":", lw=1.4,
                   label=f"reference optimum {s['s_star_reference_mesh_brent']:.4f}")
        units = "μU_c/h" if s["pressure_scaling"] == "viscous" else "ρU_c²"
        ax.set_title(r["label"], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("width split s = w₁/w₀", fontsize=9, color=INK)
        ax.set_ylabel(f"total-pressure drop J ({units})", fontsize=9, color=INK)
        ax.legend(fontsize=7.5, loc="upper center", framealpha=0.92, edgecolor="none")
        style(ax)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "fig_re_objective.png", dpi=180)
    plt.close(fig)


def by_reynolds(all_runs):
    """The runs grouped by Reynolds number, in increasing order."""
    groups = {}
    for r in all_runs:
        groups.setdefault(r["summary"]["reynolds"], []).append(r)
    return sorted(groups.items())


def per_reynolds(all_runs, draw, ylabel, path, log=False, markers=True, units=True, extra=()):
    """A panel per Re_h, each network a line in the same look in every panel."""
    groups = by_reynolds(all_runs)
    rows = (len(groups) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11.5, 3.9 * rows + 0.6), squeeze=False)
    drawn = 0
    for ax, (re_, group) in zip(axes.flat, groups):
        for r in group:
            colour, ls, marker = LOOK[r["network"]]
            layout, seed = r["network"]
            drawn += draw(ax, r, dict(color=colour, ls=ls, marker=marker,
                                      label=f"{layout}, seed {seed}"))
        title = f"Re_h = {re_:g}"
        if units:
            title += ("  (J in μU_c/h)" if group[0]["summary"]["pressure_scaling"] == "viscous"
                      else "  (J in ρU_c²)")
        if not any(r["network"][0] == "30/120" for r in group):
            title += "\n30/120 not trained here: its optimum lies outside the range"
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
        if log:
            ax.set_yscale("log")
        style(ax)
    for ax in list(axes.flat)[len(groups):]:
        ax.axis("off")
    if drawn != len(all_runs):
        raise RuntimeError(f"{path.name}: drew {drawn} of {len(all_runs)} runs")
    # one legend for the figure: a network looks the same in every panel
    handles = [plt.Line2D([], [], color=c, ls=ls, marker=m if markers else None, lw=2, ms=6,
                          label=f"{lay}, seed {sd}")
               for (lay, sd), (c, ls, m) in LOOK.items()
               if any(r["network"] == (lay, sd) for r in all_runs)] + list(extra)
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), fontsize=8.5,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.15 + 0.22 * ((len(handles) + 3) // 4)) / fig.get_figheight()))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return drawn


def drop_error(all_runs):
    def draw(ax, r, look):
        fe = [(float(row["s"]), 100 * (float(row["J_pinn"]) / float(row["J_fe"]) - 1.0))
              for row in r["rows"] if row["J_fe"]]
        ax.axhline(0.0, color=MUTED, lw=1)
        ax.plot(*zip(*fe), lw=2, ms=6, **look)
        ax.set_xlabel("width split s = w₁/w₀", fontsize=9, color=INK)
        return 1
    return per_reynolds(all_runs, draw,
                        "pressure-drop error of the PINN\nagainst the reference solution (%)",
                        HERE / "fig_re_drop_error.png")


def training(all_runs):
    def draw(ax, r, look):
        minutes = [float(row["seconds"]) / 60.0 for row in r["loss"]]
        held = [float(row["heldout_loss"]) for row in r["loss"]]
        marker = look.pop("marker")
        ax.plot(minutes, held, lw=1.6, **look)
        # where each phase begins: L-BFGS, then Levenberg-Marquardt
        for phase, m in (("lbfgs", "s"), ("lm", "^")):
            first = next(k for k, row in enumerate(r["loss"]) if row["phase"] == phase)
            ax.plot(minutes[first], held[first], m, color=look["color"], ms=7, mec="white")
        ax.set_xlabel("training time (min)", fontsize=9, color=INK)
        return 1
    starts = [plt.Line2D([], [], ls="none", marker=m, color=MUTED, ms=7, label=label)
              for m, label in (("s", "start of L-BFGS"), ("^", "start of Levenberg–Marquardt"))]
    drawn = per_reynolds(all_runs, draw, "held-out loss (diagnostic only)",
                         HERE / "fig_re_training.png", log=True, markers=False, units=False,
                         extra=starts)
    return drawn


if __name__ == "__main__":
    finished = runs()
    objective(finished)
    drop_error(finished)
    training(finished)
    print(f"{len(finished)} runs drawn")
