"""Unweighted residuals of the 2D steady incompressible Navier-Stokes system.

One small function per governing-equation term, so each term can be unit-tested
on its own (CLAUDE.md, "Code style"). Every function here is pure, takes
non-dimensional inputs, and returns an UNWEIGHTED residual. All loss weighting
lives in src/training/weights.py and nowhere else.

Non-dimensional forms
---------------------
inertial pressure scaling (p* = p / (rho U^2)):

    R_x = u u_x + v u_y + p_x - (1/Re) (u_xx + u_yy)
    R_y = u v_x + v v_y + p_y - (1/Re) (v_xx + v_yy)

viscous pressure scaling (p* = p h / (mu U)):

    R_x = Re (u u_x + v u_y) + p_x - (u_xx + u_yy)
    R_y = Re (u v_x + v v_y) + p_y - (v_xx + v_yy)

continuity (both):

    R_c = u_x + v_y

The two momentum forms are the same equation multiplied by Re (because
p*_viscous = Re p*_inertial), so they have identical solutions and different
conditioning. That is exactly why the SPEC makes the choice a config option.
"""

from __future__ import annotations

from typing import NamedTuple

from .scaling import PressureScaling


class Derivatives(NamedTuple):
    """Field values and the derivatives the residual operator needs.

    A NamedTuple so it is a JAX pytree and can be vmapped/jitted directly.
    All entries are non-dimensional.
    """

    u: object
    v: object
    p: object
    u_x: object
    u_y: object
    v_x: object
    v_y: object
    p_x: object
    p_y: object
    u_xx: object
    u_yy: object
    v_xx: object
    v_yy: object


# --------------------------------------------------------------------------- #
# Individual terms
# --------------------------------------------------------------------------- #


def convective_x(u, v, u_x, u_y):
    """(u . grad) u, x-component."""
    return u * u_x + v * u_y


def convective_y(u, v, v_x, v_y):
    """(u . grad) u, y-component."""
    return u * v_x + v * v_y


def laplacian(f_xx, f_yy):
    """grad^2 f."""
    return f_xx + f_yy


def divergence(u_x, v_y):
    """div u = u_x + v_y. This is the continuity residual, unweighted."""
    return u_x + v_y


# --------------------------------------------------------------------------- #
# Assembled momentum residuals
# --------------------------------------------------------------------------- #


def _scaling_coefficients(
    reynolds: float,
    pressure_scaling: PressureScaling,
    convective_coefficient: float = 1.0,
):
    """(convective coefficient, viscous coefficient) for the chosen scaling.

    `convective_coefficient` scales u.grad(u): 1.0 gives Navier-Stokes, 0.0
    gives Stokes. It defaults to 1.0 here so the pure-physics helpers stay
    callable on their own (the MMS and autodiff tests use them directly), but
    every run takes it from `physics.convective_coefficient`, which is a
    REQUIRED config key with no default -- a run's equations must be readable
    off its config.
    """
    if pressure_scaling == "inertial":
        return convective_coefficient * 1.0, 1.0 / reynolds
    if pressure_scaling == "viscous":
        return convective_coefficient * reynolds, 1.0
    raise ValueError(
        f"unknown pressure_scaling {pressure_scaling!r}; "
        "expected 'inertial' or 'viscous'"
    )


def momentum_x(
    d: Derivatives,
    reynolds: float,
    pressure_scaling: PressureScaling,
    convective_coefficient: float = 1.0,
):
    """x-momentum residual, unweighted."""
    a_conv, a_visc = _scaling_coefficients(
        reynolds, pressure_scaling, convective_coefficient
    )
    return (
        a_conv * convective_x(d.u, d.v, d.u_x, d.u_y)
        + d.p_x
        - a_visc * laplacian(d.u_xx, d.u_yy)
    )


def momentum_y(
    d: Derivatives,
    reynolds: float,
    pressure_scaling: PressureScaling,
    convective_coefficient: float = 1.0,
):
    """y-momentum residual, unweighted."""
    a_conv, a_visc = _scaling_coefficients(
        reynolds, pressure_scaling, convective_coefficient
    )
    return (
        a_conv * convective_y(d.u, d.v, d.v_x, d.v_y)
        + d.p_y
        - a_visc * laplacian(d.v_xx, d.v_yy)
    )


def continuity(d: Derivatives):
    """Continuity residual, unweighted."""
    return divergence(d.u_x, d.v_y)


def navier_stokes(
    d: Derivatives,
    reynolds: float,
    pressure_scaling: PressureScaling,
    convective_coefficient: float = 1.0,
):
    """All three residuals as a tuple (R_x, R_y, R_c), unweighted."""
    return (
        momentum_x(d, reynolds, pressure_scaling, convective_coefficient),
        momentum_y(d, reynolds, pressure_scaling, convective_coefficient),
        continuity(d),
    )
