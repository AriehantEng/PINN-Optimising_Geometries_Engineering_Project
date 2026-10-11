"""Figure B: amortisation and the economic break-even of the parametric PINN.

Standalone: numpy and matplotlib only. Times measured by
`python -m bifurcation.cost --config bifurcation/cost_re10.yaml`
(docs/bifurcation_re10-cost.json), each job alone on one 4-core machine, so a
measured wall-clock second is a machine-second.

Dollars are machine-hours times a price PRICE_PER_MACHINE_HOUR. The price is an
assumption, not a measurement: set it to your own rate. It scales every curve
by the same factor, so the break-even counts N* = T_train / (t_FE - t_query)
do not depend on it.

The finite-element baselines are counted one solve at a time. Design
evaluations are independent, so four single-process solves can run at once
on the same machine; that divides every FE curve by up to four and multiplies
each N* by up to four, against the PINN. The default here favours the PINN.

    python docs/figures/fig_b_break_even.py   ->   docs/figures/fig_b_break_even.png
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PRICE_PER_MACHINE_HOUR = 0.20     # USD, ASSUMED for illustration (4-vCPU machine)
TRAIN_S = {"original": 9778.0, "corner modes": 10537.0}   # recorded training time
QUERY_S = 0.0233                                          # one drop query, measured
FE = (("FE, the PINN's reference mesh", 34.512, "#2a78d6", (6, -16), "left"),
      ("FE, the design study's search mesh", 0.997, "#eda100", (-6, 10), "right"),
      ("FE at the PINN's accuracy", 0.468, "#eb6834", (6, -16), "left"))
PINN_COLOUR = "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def dollars(seconds):
    return seconds / 3600.0 * PRICE_PER_MACHINE_HOUR


def main():
    designs = np.logspace(0, 5, 300)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    train = TRAIN_S["corner modes"]
    for name, each, colour, offset, align in FE:
        ax.loglog(designs, dollars(designs * each), color=colour, lw=2, label=f"{name} ({each:g} s)")
        n_star = train / (each - QUERY_S)
        ax.plot([n_star], [dollars(n_star * each)], "o", color=colour, ms=8, zorder=4)
        ax.axvline(n_star, color=colour, lw=1, ls=":", zorder=1)
        ax.annotate(f"N* = {n_star:,.0f}", (n_star, dollars(n_star * each)), xytext=offset,
                    textcoords="offset points", fontsize=8.5, color=INK, ha=align)
    ax.loglog(designs, dollars(train + designs * QUERY_S), color=PINN_COLOUR, lw=2.4,
              label=f"parametric PINN: {train / 3600:.1f} h training + {QUERY_S} s per query")
    ax.set_xlabel("number of design evaluations N", fontsize=10, color=INK)
    ax.set_ylabel(f"total cost (USD at ${PRICE_PER_MACHINE_HOUR:.2f} per machine-hour, assumed)",
                  fontsize=10, color=INK)
    ax.set_title("Where the PINN's training pays for itself", fontsize=11, color=INK, loc="left")
    ax.text(0.99, 0.02, "N* does not depend on the price: it scales every curve alike",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    out = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=200)
    for name, each, *_ in FE:
        print(f"{name}: N* = {train / (each - QUERY_S):,.0f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
