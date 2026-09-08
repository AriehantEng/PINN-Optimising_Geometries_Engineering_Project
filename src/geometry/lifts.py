"""Stream-function lifts (SPEC v2, S4).

A lift `psi_lift(y*)` must satisfy

    psi_lift(+-1) = +- Q*/2      (the wall values differ by the flux)
    psi_lift'(+-1) = 0           (no-slip: u = dpsi/dy vanishes at the wall)

so that with `psi = psi_lift + phi^2 N_psi` and `phi = 1 - y*^2` both wall
conditions hold exactly for any `N_psi`.

Two lifts are provided because ONE OF THEM DISARMS THE GATE TEST.
"""

from __future__ import annotations

import jax.numpy as jnp


def cubic(y, flux):
    """psi = (3Q/4)(y - y^3/3).

    Check: psi(1) = (3Q/4)(2/3) = Q/2, and psi' = (3Q/4)(1 - y^2) = 0 at y = +-1.

    WARNING -- this lift IS the exact Case A solution. Its derivative is
    (3Q/4)(1 - y^2), which for Q = 4/3 is exactly 1 - y^2. Under this lift the
    correct Case A answer is `N_psi = 0`, so Case A stops testing the network's
    expressivity and tests only that the lift and training machinery are wired
    up. Use it as the MACHINERY check; never quote its accuracy as a result.
    """
    return 0.75 * flux * (y - y**3 / 3.0)


def sinusoidal(y, flux):
    """psi = (Q/2) sin(pi y / 2).

    Check: psi(+-1) = +-Q/2, and psi' = (Q pi/4) cos(pi y/2) = 0 at y = +-1.

    Satisfies the same boundary conditions as `cubic` but its derivative is NOT
    the Poiseuille profile, so `N_psi` must learn a non-zero correction. This is
    the EXPRESSIVITY check and the only one whose accuracy may be quoted.
    """
    return 0.5 * flux * jnp.sin(0.5 * jnp.pi * y)


LIFTS = {"cubic": cubic, "sinusoidal": sinusoidal}

#: Lifts whose derivative is the exact fully developed profile, so that the
#: correct answer is N_psi = 0. Reported by the sentinel matrix (S5).
DEGENERATE_LIFTS = frozenset({"cubic"})


def get(name):
    if name not in LIFTS:
        raise ValueError(f"unknown lift {name!r}; expected one of {sorted(LIFTS)}")
    return LIFTS[name]
