"""Analytic solutions for plane Poiseuille flow — the oracle for Phase 1.

Two coordinate worlds are kept strictly apart:

* `*_si` functions take SI arguments and are used only by the scaling tests.
* everything else is non-dimensional (lengths in units of the half-height h,
  velocities in units of U_max, pressure per `Scales.pressure_scale`).

Sign convention: the flow is in +x, driven by a favourable pressure gradient
G = -dp/dx > 0.
"""

from __future__ import annotations

import numpy as np

from .scaling import PressureScaling

#: Non-dimensional flux of the reference profile u* = 1 - y*^2 over y* in
#: [-1, 1]:  int_{-1}^{1} (1 - y^2) dy = 4/3.
FLUX_ND = 4.0 / 3.0

#: Non-dimensional full channel height (walls at y* = +-1).
CHANNEL_HEIGHT_ND = 2.0


# --------------------------------------------------------------------------- #
# SI forms
# --------------------------------------------------------------------------- #


def u_si(y, pressure_gradient, h, mu):
    """u(y) = (G / 2 mu) (h^2 - y^2) for G = -dp/dx > 0."""
    y = np.asarray(y, dtype=np.float64)
    return (pressure_gradient / (2.0 * mu)) * (h**2 - y**2)


def u_max_si(pressure_gradient, h, mu):
    """U_max = G h^2 / (2 mu)."""
    return pressure_gradient * h**2 / (2.0 * mu)


def flow_rate_si(pressure_gradient, h, mu):
    """Q = 2 G h^3 / (3 mu) per unit depth.

    NOTE THE EXPONENT. In 2D the flux scales as h^3, not as r^4 (Hagen-
    Poiseuille, circular pipe). Phase 3 geometry optimisation depends on this
    exponent and the two must never be mixed. See SPEC section 1.
    """
    return 2.0 * pressure_gradient * h**3 / (3.0 * mu)


def pressure_gradient_si(u_max, h, mu):
    """Inverse of `u_max_si`: G = 2 mu U_max / h^2."""
    return 2.0 * mu * u_max / h**2


# --------------------------------------------------------------------------- #
# Non-dimensional forms
# --------------------------------------------------------------------------- #


def u_nd(y):
    """u*(y*) = 1 - y*^2."""
    y = np.asarray(y, dtype=np.float64)
    return 1.0 - y**2


def v_nd(y):
    """v* = 0 everywhere. The PINN has to discover this, not be told it."""
    y = np.asarray(y, dtype=np.float64)
    return np.zeros_like(y)


def developed_profile_nd(y, flux):
    """Fully developed parabolic profile carrying a prescribed flux.

    Mass conservation fixes the amplitude:  int_{-1}^{1} u* dy* = flux, and for
    u* = A (1 - y*^2) that integral is 4A/3, so A = 3*flux/4.

    For Case A (flux = 4/3) this is exactly `u_nd`. For Case B, whose uniform
    inlet u* = 1 carries flux 2, the developed centreline velocity is 3/2 --
    NOT 1. See docs/spec-deviations.md.
    """
    y = np.asarray(y, dtype=np.float64)
    return 0.75 * flux * (1.0 - y**2)


def pressure_gradient_nd(reynolds: float, pressure_scaling: PressureScaling):
    """Constant dp*/dx* of the fully developed non-dimensional solution.

    Substituting u* = 1 - y*^2, v* = 0 into the momentum equation:

    viscous scaling   Re (u.grad)u = -grad p* + lap u*  ->  dp*/dx* = -2
    inertial scaling  (u.grad)u = -grad p* + lap(u*)/Re ->  dp*/dx* = -2/Re

    The ratio Re between the two is exactly the ratio of the pressure scales,
    which is checked in the scaling tests.
    """
    if pressure_scaling == "viscous":
        return -2.0
    if pressure_scaling == "inertial":
        return -2.0 / reynolds
    raise ValueError(f"unknown pressure_scaling {pressure_scaling!r}")


def p_nd(x, domain_length: float, reynolds: float, pressure_scaling: PressureScaling):
    """p*(x*) with the gauge p*(x* = L*) = 0.

    Pressure is unique only up to a constant. Pinning it at the outlet is the
    gauge fix used throughout Phase 1; without it the pressure L2 error is
    meaningless (SPEC section 9).
    """
    x = np.asarray(x, dtype=np.float64)
    dpdx = pressure_gradient_nd(reynolds, pressure_scaling)
    return dpdx * (x - domain_length)


def stream_function_nd(y, flux=FLUX_ND):
    """psi*(y*) with u* = dpsi/dy, for the fully developed profile.

    psi = A (y - y^3/3) with A = 3*flux/4, gauged so psi(0) = 0. The wall values
    differ by exactly `flux`, which is what makes a stream-function ansatz that
    is constant on both walls unable to represent a channel with net throughflow
    (see src/models/hard_bc.py).
    """
    y = np.asarray(y, dtype=np.float64)
    return 0.75 * flux * (y - y**3 / 3.0)


# --------------------------------------------------------------------------- #
# Entrance length (Case B)
# --------------------------------------------------------------------------- #


def entrance_length_correlation_nd(reynolds: float, bulk_velocity_nd: float = 1.0):
    """Developing length for a plane channel, in units of the half-height h.

    Durst, Ray, Uensal & Bayoumi (2005), "The development lengths of laminar
    pipe and channel flows", J. Fluids Eng. 127:1154-1160, plane-channel fit:

        L_e / H = [ C1^n + (C2 * Re_H)^n ]^(1/n),
        C1 = 0.631, C2 = 0.0442, n = 1.6

    where H is the FULL channel height and Re_H = rho * U_bulk * H / mu. The
    high-Re asymptote L_e/H = 0.0442 Re_H is the classical L_e/D_h = 0.011
    Re_Dh with D_h = 2H, which is the cross-check that fixes these definitions.

    Converting to this project's variables: H* = 2, and this project's Reynolds
    number uses U_max and h, so Re_H = 2 * Re * bulk_velocity_nd.

    Returns L_e in units of h (i.e. directly comparable to x*).
    """
    c1, c2, n = 0.631, 0.0442, 1.6
    re_h = 2.0 * reynolds * bulk_velocity_nd
    le_over_h_full = (c1**n + (c2 * re_h) ** n) ** (1.0 / n)
    return CHANNEL_HEIGHT_ND * le_over_h_full
