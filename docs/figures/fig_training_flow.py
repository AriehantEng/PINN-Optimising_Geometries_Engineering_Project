"""Diagram: how the parametric PINN is trained, checked and scored.

Training uses the physics only; the finite-element solver enters after
training, to score the network, and never feeds back into it.

    python docs/figures/fig_training_flow.py   ->   docs/figures/fig_training_flow.png
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
PALE = {BLUE: "#e6effb", AQUA: "#e3f5ee", ORANGE: "#fdebe4", MUTED: "#f1f0ec"}


def box(ax, x, y, w, h, title, body, colour):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=PALE[colour], ec=colour, lw=1.6))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=8.6, color=INK,
            linespacing=1.45)


def arrow(ax, a, b, colour=INK, style="-|>", ls="-", rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, color=colour,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def band(ax, y, h, label, note, colour):
    ax.add_patch(FancyBboxPatch((0.15, y), 15.7, h, boxstyle="round,pad=0,rounding_size=0.2",
                                fc="none", ec=colour, lw=1.0, ls=(0, (4, 3))))
    ax.text(0.4, y + h - 0.18, label, ha="left", va="top", fontsize=11.5, color=colour,
            fontweight="bold")
    ax.text(15.6, y + h - 0.2, note, ha="right", va="top", fontsize=9.5, color=colour)


def main():
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    # -- 1. training: physics only ---------------------------------------------
    band(ax, 6.4, 3.1, "1  TRAINING", "physics only, no solver data  ·  one run, 163-176 min", BLUE)
    box(ax, 0.5, 6.85, 3.3, 2.15, "Random points",
        "(x, y, s) inside the channel,\n16 width splits s per draw;\nfresh draw every step:\n"
        "about 2.7 million in all", BLUE)
    box(ax, 4.5, 6.85, 3.3, 2.15, "The network",
        "ψ = G + D²N + corner modes;\nexact for any weights:\nno-slip, mass and\n"
        "the 50/50 flow share", BLUE)
    box(ax, 8.5, 6.85, 3.3, 2.15, "Physics loss",
        "Navier-Stokes momentum\nresidual at every point,\ninlet profile, outlet\n"
        "pressure conditions", BLUE)
    box(ax, 12.5, 6.85, 3.1, 2.15, "Optimiser",
        "Adam 3,000 steps,\nL-BFGS 4 x 500,\nLevenberg-Marquardt 150;\nadjusts the weights", BLUE)
    for x0, x1 in ((3.8, 4.5), (7.8, 8.5), (11.8, 12.5)):
        arrow(ax, (x0, 7.9), (x1, 7.9), BLUE)
    arrow(ax, (14.05, 6.85), (6.15, 6.85), BLUE, rad=-0.06)
    ax.text(10.1, 6.75, "new weights: repeat", ha="center", va="center", fontsize=8.8,
            color=BLUE, style="italic")

    # -- 2. self-check: held-out points ----------------------------------------
    band(ax, 3.4, 2.65, "2  SELF-CHECK", "during training, still no solver data", AQUA)
    box(ax, 0.5, 3.6, 7.3, 1.75, "Held-out points, never trained on",
        "32 splits x 1,024 points: after every step the same\nphysics loss is measured here; "
        "the best iterate is kept", AQUA)
    box(ax, 8.5, 3.6, 7.1, 1.75, "What it can and cannot tell",
        "catches over-fitting to the training points;\nlow loss is NOT proof of accuracy "
        "(a diagnostic only)", AQUA)
    arrow(ax, (6.15, 6.85), (6.15, 5.35), AQUA)
    ax.text(6.3, 6.22, "evaluate", ha="left", va="center", fontsize=8.8, color=AQUA,
            style="italic")

    # -- 3. scoring: finite elements, after training ---------------------------
    band(ax, 0.1, 2.9, "3  SCORING", "after training, against the solver", ORANGE)
    box(ax, 0.5, 0.3, 4.6, 1.75, "Trained PINN",
        "J(s) at any split in 0.023 s;\nthe whole flow field", BLUE)
    box(ax, 5.8, 0.3, 4.4, 1.75, "Compare",
        "relative L2 error of u, v, p;\npressure-drop error; optimum s*", MUTED)
    box(ax, 10.9, 0.3, 4.7, 1.75, "Finite-element reference",
        "scikit-fem, Taylor-Hood P2/P1,\n129,562 unknowns, 34.5 s per split", ORANGE)
    arrow(ax, (5.1, 1.18), (5.8, 1.18), INK)
    arrow(ax, (10.9, 1.18), (10.2, 1.18), ORANGE)
    arrow(ax, (4.2, 3.6), (4.2, 2.05), BLUE)
    ax.text(4.35, 3.2, "kept iterate", ha="left", va="center", fontsize=8.8, color=BLUE,
            style="italic")

    # the solver never feeds back
    arrow(ax, (11.6, 2.05), (11.6, 3.6), ORANGE, style="-", ls=(0, (2, 2)), lw=1.2)
    ax.text(11.6, 3.2, "✕", ha="center", va="center", fontsize=15, color=ORANGE,
            fontweight="bold")
    ax.text(11.4, 3.2, "never feeds back into training", ha="right", va="center",
            fontsize=8.8, color=ORANGE, style="italic")

    fig.tight_layout()
    out = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
