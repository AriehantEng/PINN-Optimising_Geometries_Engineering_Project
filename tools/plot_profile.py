"""u*(y*): the PINN prediction against the exact solution.

    python tools/plot_profile.py caseA_re10_extended
    python tools/plot_profile.py caseA_re10_extended --x 3.0 --out docs/profile.png

Two panels, because one is not enough. At ~1e-7 agreement an overlay of
prediction and exact is a single curve: it proves the match and shows nothing.
The right panel carries the signed difference on its own axis, which is where
the agreement is actually legible.

Colours are the validated categorical palette (dataviz reference, light mode).
The exact solution is drawn as a neutral reference line rather than a second
series colour -- it is not a peer of the prediction, it is the thing being
compared against.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.physics import analytic  # noqa: E402
from src.predict import load_run  # noqa: E402

PINN = "#2a78d6"      # categorical slot 1
REFERENCE = "#333333"  # neutral: the exact solution is a reference, not a series


def build(run_name: str, x: float | None, out: pathlib.Path, points: int = 201):
    model = load_run(str(ROOT / "runs" / run_name))
    config = model.config
    if x is None:
        x = 0.5 * model.domain_length

    y, u, _ = model.profile(x=x, n=points)
    y, u = np.asarray(y), np.asarray(u)

    if config.case == "A":
        exact = 1.0 - y**2
        label = "exact:  u* = 1 - y*²"
    else:
        exact = np.asarray(analytic.developed_profile_nd(y, config.inlet_flux))
        label = f"developed profile (flux {config.inlet_flux:.4f})"

    difference = u - exact
    relative = float(np.linalg.norm(difference) / np.linalg.norm(exact))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # -- the requested graph: u against y ---------------------------------- #
    axis = axes[0]
    axis.plot(y, exact, "-", color=REFERENCE, linewidth=2.2, label=label, zorder=1)
    every = max(1, len(y) // 26)
    axis.plot(y[::every], u[::every], "o", markersize=7, markerfacecolor="none",
              markeredgewidth=1.6, color=PINN, label="PINN prediction", zorder=2)
    axis.set_xlabel("y*   (wall-normal, walls at ±1)")
    axis.set_ylabel("u*   (streamwise velocity)")
    axis.set_title(f"{run_name}:  u* against y*   at x* = {x:.2f}")
    axis.legend(fontsize=9, loc="lower center")
    axis.grid(alpha=0.3)
    axis.set_xlim(-1.05, 1.05)

    # -- where the agreement is actually visible --------------------------- #
    axis = axes[1]
    axis.plot(y, difference, "-", color=PINN, linewidth=1.6)
    axis.axhline(0.0, color=REFERENCE, linewidth=1.0)
    axis.set_xlabel("y*")
    axis.set_ylabel("u*  -  exact")
    axis.set_title("difference from exact (note the axis scale)")
    axis.grid(alpha=0.3)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axis.set_xlim(-1.05, 1.05)
    axis.annotate(
        f"relative L2 = {relative:.3e}\nmax |difference| = {np.abs(difference).max():.3e}",
        xy=(0.03, 0.03), xycoords="axes fraction", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"  station x* = {x:.4f}   {points} points")
    print(f"  relative L2      {relative:.4e}")
    print(f"  max |difference| {np.abs(difference).max():.4e}")
    print(f"  u* at the walls  {u[0]:.3e}, {u[-1]:.3e}   (no-slip is exact by construction)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", default="caseA_re10_extended")
    parser.add_argument("--x", type=float, default=None,
                        help="streamwise station; default is mid-channel")
    parser.add_argument("--out", default=None)
    parser.add_argument("--points", type=int, default=201)
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out) if args.out else ROOT / "docs" / f"{args.run}-profile.png"
    return build(args.run, args.x, out, args.points)


if __name__ == "__main__":
    raise SystemExit(main())
