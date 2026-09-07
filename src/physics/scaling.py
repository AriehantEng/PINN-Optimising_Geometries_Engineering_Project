"""Explicit, invertible SI <-> non-dimensional transform.

Every physics quantity in this project lives in one of two worlds and the
transform between them is written down here once. Nothing else in the codebase
is allowed to hold an SI-valued viscosity, density or channel height.

Reference quantities
--------------------
length    h        channel half-height
velocity  U_max    centreline velocity of the fully developed profile
pressure  depends on the chosen scaling:
            viscous   p_ref = mu * U_max / h        (use for Re <~ 1)
            inertial  p_ref = rho * U_max**2        (use for Re > 1)

Reynolds number
---------------
    Re = rho * U_max * h / mu

The two pressure scalings are related by exactly a factor Re:

    p*_viscous = p*_inertial * Re

which is a useful consistency check and is tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

PressureScaling = Literal["viscous", "inertial"]

_VALID_PRESSURE_SCALINGS = ("viscous", "inertial")

#: Relative tolerance for a "machine precision" SI -> non-dim -> SI round trip.
#: A round trip is one division followed by one multiplication, each correctly
#: rounded, so the relative error is bounded by ~2 eps. 4 eps is that bound with
#: one bit of headroom. This is a derived bound, not a fitted tolerance: it must
#: never be widened to make a test pass.
ROUNDTRIP_RTOL = 4.0 * float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class Scales:
    """Reference quantities defining the non-dimensionalisation.

    Parameters are SI. This is the only object in the codebase that is allowed
    to carry SI values; it converts them and hands non-dimensional numbers to
    everything else.

    Attributes
    ----------
    h : channel half-height [m]
    u_max : centreline velocity of the fully developed profile [m/s]
    rho : density [kg/m^3]
    mu : dynamic viscosity [Pa s]
    pressure_scaling : "viscous" or "inertial"
    """

    h: float
    u_max: float
    rho: float
    mu: float
    pressure_scaling: PressureScaling

    def __post_init__(self) -> None:
        for name in ("h", "u_max", "rho", "mu"):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"Scales.{name} must be finite, got {value!r}")
            if value <= 0.0:
                raise ValueError(f"Scales.{name} must be positive, got {value!r}")
        if self.pressure_scaling not in _VALID_PRESSURE_SCALINGS:
            raise ValueError(
                f"pressure_scaling must be one of {_VALID_PRESSURE_SCALINGS}, "
                f"got {self.pressure_scaling!r}"
            )

    # -- reference quantities -------------------------------------------------

    @property
    def reynolds(self) -> float:
        """Re = rho * U_max * h / mu."""
        return self.rho * self.u_max * self.h / self.mu

    @property
    def length_scale(self) -> float:
        return self.h

    @property
    def velocity_scale(self) -> float:
        return self.u_max

    @property
    def pressure_scale(self) -> float:
        """Pressure reference p_ref such that p* = p / p_ref."""
        if self.pressure_scaling == "viscous":
            return self.mu * self.u_max / self.h
        return self.rho * self.u_max**2

    # -- SI -> non-dimensional ------------------------------------------------

    def length_to_nondim(self, x):
        return np.asarray(x, dtype=np.float64) / self.length_scale

    def velocity_to_nondim(self, u):
        return np.asarray(u, dtype=np.float64) / self.velocity_scale

    def pressure_to_nondim(self, p):
        return np.asarray(p, dtype=np.float64) / self.pressure_scale

    def pressure_gradient_to_nondim(self, dpdx):
        """dp*/dx* = (dp/dx) * h / p_ref."""
        return (
            np.asarray(dpdx, dtype=np.float64)
            * self.length_scale
            / self.pressure_scale
        )

    # -- non-dimensional -> SI ------------------------------------------------

    def length_to_si(self, x_nd):
        return np.asarray(x_nd, dtype=np.float64) * self.length_scale

    def velocity_to_si(self, u_nd):
        return np.asarray(u_nd, dtype=np.float64) * self.velocity_scale

    def pressure_to_si(self, p_nd):
        return np.asarray(p_nd, dtype=np.float64) * self.pressure_scale

    def pressure_gradient_to_si(self, dpdx_nd):
        return (
            np.asarray(dpdx_nd, dtype=np.float64)
            * self.pressure_scale
            / self.length_scale
        )


def scales_from_reynolds(
    reynolds: float,
    pressure_scaling: PressureScaling,
    *,
    h: float = 1.0,
    u_max: float = 1.0,
    rho: float = 1.0,
) -> Scales:
    """Build `Scales` with a prescribed Reynolds number.

    mu is solved for rather than defaulted, so the returned object satisfies
    `scales.reynolds == reynolds` up to rounding. Used by configs that specify
    Re directly instead of four SI quantities.
    """
    if not np.isfinite(reynolds) or reynolds <= 0.0:
        raise ValueError(f"reynolds must be finite and positive, got {reynolds!r}")
    mu = rho * u_max * h / reynolds
    return Scales(
        h=h, u_max=u_max, rho=rho, mu=mu, pressure_scaling=pressure_scaling
    )


def recommended_pressure_scaling(reynolds: float) -> PressureScaling:
    """The SPEC's recommendation: viscous for Re <= 1, inertial above.

    Advisory only. The active scaling always comes from the config; this
    function exists so a run can *report* that it deviates from the
    recommendation, never so it can silently override the config.
    """
    return "viscous" if reynolds <= 1.0 else "inertial"
