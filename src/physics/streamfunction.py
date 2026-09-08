"""Coupled psi-omega system and post-hoc pressure recovery (SPEC v2, S4).

    u = dpsi/dy,   v = -dpsi/dx      (continuity satisfied identically)
    omega = -lap psi                 (vorticity)

Taking the curl of the momentum equation eliminates pressure. The system is
solved as TWO COUPLED SECOND-ORDER equations in two network outputs
`(psi, omega)`:

    r_1 = omega + lap psi                                             = 0
    r_2 = psi_y omega_x - psi_x omega_y - lap(omega)/Re               = 0

rather than by substituting `omega = -lap psi` into r_2, which would give a
FOURTH-ORDER equation in psi. Fourth-order autodiff through tanh would then set
the precision floor -- and Phase 1 already measured second derivatives through
tanh costing an order of magnitude of achievable accuracy.

`omega` carries no hard constraint and is coupled only through r_1. That is
deliberate: it partially restores the falsifiability that the pure-psi form
destroys, because r_1 is a real residual the network can violate.
"""

from __future__ import annotations

from typing import NamedTuple


class StreamDerivatives(NamedTuple):
    """Derivative bundle for the psi-omega system. All non-dimensional."""

    psi_x: object
    psi_y: object
    psi_xx: object
    psi_yy: object
    omega: object
    omega_x: object
    omega_y: object
    omega_xx: object
    omega_yy: object


def laplacian(f_xx, f_yy):
    return f_xx + f_yy


def vorticity_residual(d: StreamDerivatives):
    """r_1 = omega + lap psi. Unweighted."""
    return d.omega + laplacian(d.psi_xx, d.psi_yy)


def vorticity_transport_residual(d: StreamDerivatives, reynolds: float):
    """r_2 = psi_y omega_x - psi_x omega_y - lap(omega)/Re. Unweighted.

    The convective part is (u . grad) omega with u = psi_y, v = -psi_x:

        u omega_x + v omega_y = psi_y omega_x - psi_x omega_y
    """
    return (
        d.psi_y * d.omega_x
        - d.psi_x * d.omega_y
        - laplacian(d.omega_xx, d.omega_yy) / reynolds
    )


def residuals(d: StreamDerivatives, reynolds: float):
    """(r_1, r_2), unweighted."""
    return vorticity_residual(d), vorticity_transport_residual(d, reynolds)


def velocity_from_stream(psi_x, psi_y):
    """(u, v) = (psi_y, -psi_x)."""
    return psi_y, -psi_x


# --------------------------------------------------------------------------- #
# Post-hoc pressure recovery (SPEC v2, S4)
# --------------------------------------------------------------------------- #

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from .derivatives import batched_directional_derivatives  # noqa: E402
from .residuals import _scaling_coefficients  # noqa: E402


def pressure_gradient(velocity_fn, points, reynolds, pressure_scaling):
    """(p_x, p_y) from the momentum equation, at an (N, 2) array of points.

    Rearranging the momentum residual to zero:

        p_x = a_visc (u_xx + u_yy) - a_conv (u u_x + v u_y)
        p_y = a_visc (v_xx + v_yy) - a_conv (u v_x + v v_y)

    with (a_conv, a_visc) the scaling coefficients, so this works unchanged
    under either pressure scaling. `velocity_fn(xy) -> [u, v]`.
    """
    a_conv, a_visc = _scaling_coefficients(reynolds, pressure_scaling)
    value, d_dx, d_dy, d2_dx2, d2_dy2 = batched_directional_derivatives(
        velocity_fn, points
    )
    u, v = value[..., 0], value[..., 1]
    u_x, v_x = d_dx[..., 0], d_dx[..., 1]
    u_y, v_y = d_dy[..., 0], d_dy[..., 1]
    lap_u = d2_dx2[..., 0] + d2_dy2[..., 0]
    lap_v = d2_dx2[..., 1] + d2_dy2[..., 1]

    p_x = a_visc * lap_u - a_conv * (u * u_x + v * u_y)
    p_y = a_visc * lap_v - a_conv * (u * v_x + v * v_y)
    return p_x, p_y


def _simpson(values, spacing):
    """Composite Simpson on an odd number of equally spaced samples."""
    n = len(values) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's rule needs an even number of intervals")
    weights = np.ones(n + 1)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return float(spacing / 3.0 * np.dot(weights, values))


def _segment(velocity_fn, start, end, reynolds, pressure_scaling, nodes):
    """Integral of grad p . dl along the straight segment start -> end."""
    if nodes % 2 == 0:
        nodes += 1
    t = np.linspace(0.0, 1.0, nodes)
    path = np.outer(1.0 - t, np.asarray(start)) + np.outer(t, np.asarray(end))
    p_x, p_y = pressure_gradient(
        velocity_fn, jnp.asarray(path), reynolds, pressure_scaling
    )
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    integrand = np.asarray(p_x) * delta[0] + np.asarray(p_y) * delta[1]
    return _simpson(integrand, 1.0 / (nodes - 1))


def recover_pressure(
    velocity_fn, target, gauge_point, reynolds, pressure_scaling, *, path="x_first",
    nodes=201,
):
    """p at `target` (a single (2,) point), gauged to zero at `gauge_point`.

    Integrates grad p along an L-shaped path. Two orderings are available and
    they must agree: path independence is a real consistency check on the
    recovered field, since grad p is only a gradient if the flow actually
    satisfies momentum (SPEC v2 S4, T-S4.2).

    "x_first": gauge -> (target_x, gauge_y) -> target
    "y_first": gauge -> (gauge_x, target_y) -> target
    """
    gauge = np.asarray(gauge_point, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if path == "x_first":
        corner = np.array([target[0], gauge[1]])
    elif path == "y_first":
        corner = np.array([gauge[0], target[1]])
    else:
        raise ValueError(f"unknown path {path!r}")

    total = 0.0
    for start, end in ((gauge, corner), (corner, target)):
        if not np.allclose(start, end):
            total += _segment(
                velocity_fn, start, end, reynolds, pressure_scaling, nodes
            )
    return total
