"""Figure A: simulation time scaling, finite elements against the parametric PINN.

Standalone: numpy and matplotlib only. Every number below was measured by
`python -m bifurcation.cost --config bifurcation/cost_re10.yaml` on one 4-core
machine, one job at a time (docs/bifurcation_re10-cost.json): the 30/90
layout at split 0.5, Re_h = 10, half the flow through each daughter.

    (a) one finite-element solve against the number of unknowns, with the
        fitted exponent and the textbook rates for comparison;
    (b) cumulative machine time for N design evaluations, 1 to 10^4.

    python docs/figures/fig_a_time_scaling.py   ->   docs/figures/fig_a_time_scaling.png
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# -- measured: finite elements (P2/P1 Taylor-Hood, Newton, sparse direct) ------
UNKNOWNS = np.array([2569, 5383, 10236, 22517, 51326, 129562, 280609])
SOLVE_S = np.array([0.468, 0.458, 0.997, 2.912, 9.833, 34.512, 115.370])
DROP_ERROR = np.array([-6.64e-4, -4.35e-4, -2.49e-4, -5.24e-5, -2.02e-5, -6.29e-6, 0.0])  # vs finest
PINN_DROP_ERROR = -7.6e-4  # the corner-mode PINN at split 0.5, against the reference mesh
# -- measured: the corner-mode parametric PINN ------------------------------
TRAIN_S = 10537.0          # Adam 3,000 + L-BFGS 4 x 500 + LM 150, all four cores
QUERY_S = 0.0233           # one total-pressure drop J(s): three 256-point line integrals
FIT_FROM = 20000           # the exponent is fitted where the solve dominates the setup

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.8, which="major")


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # (a) one solve against its size
    big = UNKNOWNS >= FIT_FROM
    slope, intercept = np.polyfit(np.log(UNKNOWNS[big]), np.log(SOLVE_S[big]), 1)
    n = np.logspace(np.log10(2e3), np.log10(4e5), 100)
    a.loglog(UNKNOWNS, SOLVE_S, "o", color=BLUE, ms=7, zorder=3, label="finite elements, measured")
    a.loglog(n, np.exp(intercept) * n**slope, color=BLUE, lw=2,
             label=f"fit: time ~ N^{slope:.2f}")
    anchor_n, anchor_t = UNKNOWNS[-2], SOLVE_S[-2]
    for power, colour, name in ((1.5, MUTED, "N^1.5 (2-D nested dissection)"),
                                (3.0, ORANGE, "N^3 (dense LU)")):
        a.loglog(n, anchor_t * (n / anchor_n) ** power, color=colour, lw=1.2, ls="--", label=name)
    a.axhline(QUERY_S, color=AQUA, lw=2, label="PINN: one drop query, any split in range")
    a.set_ylim(1e-2, 1e3)
    a.set_xlabel("finite-element unknowns N", fontsize=9, color=INK)
    a.set_ylabel("wall time of one evaluation (s)", fontsize=9, color=INK)
    a.set_title("(a) one design evaluation", fontsize=10, color=INK, loc="left")
    a.legend(fontsize=7.5, frameon=False, loc="upper left")
    style(a)

    # (b) cumulative machine time for N designs
    designs = np.logspace(0, 4, 200)
    lines = ((SOLVE_S[5], BLUE, "FE, the PINN's reference mesh (34.5 s)"),
             (SOLVE_S[2], YELLOW, "FE, the design study's search mesh (1.0 s)"),
             (SOLVE_S[0], ORANGE, f"FE at the PINN's accuracy (0.47 s; drop error "
                                  f"{100 * abs(DROP_ERROR[0]):.3f}% vs PINN {100 * abs(PINN_DROP_ERROR):.3f}%)"))
    for each, colour, name in lines:
        b.loglog(designs, designs * each / 3600.0, color=colour, lw=2, label=name)
        crossing = TRAIN_S / (each - QUERY_S)
        if crossing <= designs[-1]:
            b.plot([crossing], [crossing * each / 3600.0], "o", color=colour, ms=7, zorder=4)
            b.annotate(f"break-even\nN* = {crossing:,.0f}", (crossing, crossing * each / 3600.0),
                       xytext=(8, -18), textcoords="offset points", fontsize=8, color=INK)
        else:
            b.annotate(f"N* = {crossing:,.0f}, off this axis", (designs[-1], designs[-1] * each / 3600.0),
                       xytext=(-6, -14), textcoords="offset points", fontsize=8, color=INK,
                       ha="right")
    b.loglog(designs, (TRAIN_S + designs * QUERY_S) / 3600.0, color=AQUA, lw=2,
             label="parametric PINN: 2.9 h training, then 0.023 s each")
    b.set_xlabel("number of design evaluations", fontsize=9, color=INK)
    b.set_ylabel("cumulative machine time (h)", fontsize=9, color=INK)
    b.set_title("(b) many design evaluations", fontsize=10, color=INK, loc="left")
    b.legend(fontsize=7.5, frameon=False, loc="upper left")
    style(b)

    fig.tight_layout()
    out = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=200)
    print(f"fitted exponent {slope:.3f}; wrote {out}")


if __name__ == "__main__":
    main()
