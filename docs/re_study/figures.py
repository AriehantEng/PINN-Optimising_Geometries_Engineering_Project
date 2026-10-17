"""Figures of the Reynolds-number study, from the exported runs in docs/re_study.

    python docs/re_study/figures.py

    fig_re_objective.png   J(s) from each finished PINN against the reference solution
    fig_re_drop_error.png  the PINNs' signed pressure-drop error against s
    fig_re_training.png    held-out loss against training time
Redraw as runs finish: every finished run's files are picked up.
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
        label = (f"{s['layout'][0]:g}/{s['layout'][1]:g}, Re_h = {s['reynolds']:g}"
                 + ("" if s["seed"] == 21 else f", seed {s['seed']}"))
        out.append({"summary": s, "rows": rows, "loss": loss, "label": label})
    out.sort(key=lambda r: (r["summary"]["layout"][1], r["summary"]["reynolds"],
                            r["summary"]["seed"]))
    return out


def objective(all_runs):
    n = len(all_runs)
    cols = 2
    fig, axes = plt.subplots((n + 1) // cols, cols, figsize=(11, 3.6 * ((n + 1) // cols)),
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


def drop_error(all_runs):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axhline(0.0, color=MUTED, lw=1)
    for colour, r in zip(COLOURS, all_runs):
        fe = [(float(row["s"]), 100 * (float(row["J_pinn"]) / float(row["J_fe"]) - 1.0))
              for row in r["rows"] if row["J_fe"]]
        ax.plot(*zip(*fe), "o-", color=colour, lw=2, ms=6, label=r["label"])
    ax.set_xlabel("width split s = w₁/w₀", fontsize=9, color=INK)
    ax.set_ylabel("pressure-drop error of the PINN against the reference solution (%)", fontsize=9, color=INK)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.92, edgecolor="none")
    style(ax)
    fig.tight_layout()
    fig.savefig(HERE / "fig_re_drop_error.png", dpi=180)
    plt.close(fig)


def training(all_runs):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for colour, r in zip(COLOURS, all_runs):
        minutes = [float(row["seconds"]) / 60.0 for row in r["loss"]]
        held = [float(row["heldout_loss"]) for row in r["loss"]]
        ax.semilogy(minutes, held, color=colour, lw=1.6, label=r["label"])
        # where each phase begins: L-BFGS, then Levenberg-Marquardt
        for phase, marker in (("lbfgs", "s"), ("lm", "^")):
            first = next(k for k, row in enumerate(r["loss"]) if row["phase"] == phase)
            ax.plot(minutes[first], held[first], marker, color=colour, ms=7, mec="white")
    ax.plot([], [], "s", color=MUTED, label="start of L-BFGS")
    ax.plot([], [], "^", color=MUTED, label="start of Levenberg-Marquardt")
    ax.set_xlabel("training time (min)", fontsize=9, color=INK)
    ax.set_ylabel("held-out loss (diagnostic only)", fontsize=9, color=INK)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92, edgecolor="none")
    style(ax)
    fig.tight_layout()
    fig.savefig(HERE / "fig_re_training.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    finished = runs()
    objective(finished)
    drop_error(finished)
    training(finished)
    print(f"{len(finished)} runs drawn")
