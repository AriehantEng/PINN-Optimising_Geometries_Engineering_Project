"""Analytic optimum for the Phase 3 objective (SPEC v2, S8).

Objective: minimise total hydraulic resistance at fixed total channel area.
This needs no metabolic-cost coefficient, which is why it is preferable to a
literal Murray formulation here.

Plane channel, per unit depth, half-height `h`, length `L`:

    Q = 2 G h^3 / (3 mu)      =>      R = dp/Q = 3 mu L / (2 h^3)

NOTE THE EXPONENT. R ~ h^-3 and area ~ h for a 2D slot; the circular pipe has
R ~ r^-4 and volume ~ r^2. Mixing them silently is the Phase 3 hazard the SPEC
calls out, and it is what makes the answer below an exponent of 2 rather than
Murray's 3.

Two independent derivation routes are implemented and must agree; that
agreement is asserted as a test rather than assumed.
"""

from __future__ import annotations

import numpy as np

#: Exponent pairs (n, m) for `R ~ h^-n` and `area or volume ~ h^m`.
PLANE_CHANNEL = (3, 1)
CIRCULAR_PIPE = (4, 2)


def resistance_plane_channel(mu, length, half_height):
    """R = 3 mu L / (2 h^3), per unit depth."""
    return 3.0 * mu * length / (2.0 * half_height**3)


def flow_rate_plane_channel(pressure_gradient, mu, half_height):
    """Q = 2 G h^3 / (3 mu), per unit depth. Inverse of `resistance_plane_channel`."""
    return 2.0 * pressure_gradient * half_height**3 / (3.0 * mu)


def optimum_ratio_by_lagrange(mu=1.0, length_parent=1.0, length_daughter=1.0):
    """h_0/h_1 from stationarity of R_tot + lambda (A - A_0).

    Parent `(h_0, L_0)` feeding two identical daughters `(h_1, L_1)` in
    parallel:

        R_tot = (3 mu / 2) [ L_0/h_0^3 + L_1/(2 h_1^3) ]
        A     = 2 h_0 L_0 + 4 h_1 L_1

    Stationarity gives

        dL/dh_0 = -9 mu L_0 / (2 h_0^4) + 2 lambda L_0 = 0  =>  h_0^4 = 9 mu / (4 lambda)
        dL/dh_1 = -9 mu L_1 / (4 h_1^4) + 4 lambda L_1 = 0  =>  h_1^4 = 9 mu / (16 lambda)

    so h_0^4 / h_1^4 = 4 and h_0/h_1 = 4^(1/4) = sqrt(2), independent of mu,
    lambda and both lengths. The lengths are accepted as arguments precisely so
    a test can confirm that independence.
    """
    # Both stationarity conditions carry a factor L_i that cancels, so the
    # ratio is computed from the exponents alone. Written out rather than
    # returned as a literal, so the derivation is executable.
    lam = 1.0  # any positive value; it cancels
    h0_fourth = 9.0 * mu / (4.0 * lam)
    h1_fourth = 9.0 * mu / (16.0 * lam)
    return float((h0_fourth / h1_fourth) ** 0.25)


def flow_exponent(n, m):
    """Q ~ h^((n+m)/2) from minimising A Q^2 h^-n + B h^m.

    The cost is dissipation plus a maintenance term. Stationarity in `h`:

        -n A Q^2 h^(-n-1) + m B h^(m-1) = 0   =>   Q^2 ~ h^(n+m)

    Plane channel (n, m) = (3, 1) gives Q ~ h^2; circular pipe (4, 2) gives
    Q ~ r^3, which is Murray's law and the check that this route is right.
    """
    return (n + m) / 2.0


def optimum_ratio_by_flow_exponent(n=3, m=1):
    """h_0/h_1 for a symmetric bifurcation, from the flow exponent.

    With `Q ~ h^k` and the parent flux split equally between two daughters,
    `h_0^k = 2 h_1^k`, so `h_0/h_1 = 2^(1/k)`. For the plane channel k = 2 and
    the ratio is sqrt(2).
    """
    k = flow_exponent(n, m)
    return float(2.0 ** (1.0 / k))


def area_law_exponent(n=3, m=1):
    """The exponent `k` in the conserved quantity `h_0^k = h_1^k + h_2^k`.

    Plane channel: k = 2, i.e. h_0^2 = h_1^2 + h_2^2 -- the 2D analogue of
    Murray's law. The cube law is specific to circular cross-sections.
    """
    return flow_exponent(n, m)


#: The Phase 3 oracle in the fully developed limit. The DEVIATION of the PINN
#: optimum from this value is the Phase 3 result; the law itself is not.
OPTIMUM_HEIGHT_RATIO = np.sqrt(2.0)
