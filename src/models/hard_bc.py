"""The boundary-condition ansatz: what the network is allowed to represent.

No-slip is NEVER a penalty term in this project (SPEC section 3). The velocity
is constructed so that it vanishes on both walls for any parameter values:

    u*(x*, y*) = phi(y*) N_u(x*, y*)
    v*(x*, y*) = phi(y*) N_v(x*, y*)      phi(y*) = 1 - y*^2

Inlet handling
--------------
Two options, both implemented, chosen by config (`bc.inlet`):

* "soft"    -- an inlet loss term. The PINN has to DISCOVER the parabolic
               profile from the PDE plus a soft nudge. This is the default for
               Case A because it is the honest test of the harness: with
               lifting, u(0, y) = phi(y) = 1 - y^2 is already the exact Case A
               inlet condition and the inlet is no longer being solved for.
* "lifting" -- u = phi(y) [1 + lambda(x) N_u], v = phi(y) lambda(x) N_v with
               lambda(x) = 1 - exp(-x / l), lambda(0) = 0. This makes the
               Case A inlet exact by construction and removes the inlet loss
               term entirely. Only available for Case A: a uniform Case B inlet
               is incompatible with hard no-slip at the corners.

Pressure gauge
--------------
Pressure is unique only up to a constant. `bc.pressure_gauge`:

* "soft" -- a loss term pinning p*(L*, y*) = 0.
* "hard" -- p* = (1 - x*/L*) N_p, which is exactly zero on the outlet plane.

Without one of these the pressure relative L2 error is meaningless (SPEC
section 9).

Stream function
---------------
u = dpsi/dy, v = -dpsi/dx satisfies continuity identically. The wall condition
needs psi_y = psi_x = 0 at y* = +-1, which phi^2 gives:

    psi = psi_base(y*) + phi(y*)^2 N_psi(x*, y*)

phi^2 alone is NOT enough, though: psi(+1) - psi(-1) equals the flux, so a psi
that is constant on both walls can only represent zero net throughflow. Hence
`psi_base`, which carries the flux and has zero slope at both walls:

    "cubic"  psi_base = (Q/4)(3y - y^3)   -> psi_base' = (3Q/4)(1 - y^2)
    "sine"   psi_base = (Q/2) sin(pi y/2) -> psi_base' = (Q pi/4) cos(pi y/2)

The cubic base IS the exact Case A solution, so with it the network only has to
learn N_psi = 0 and the Case A comparison against the velocity-pressure
formulation would be meaningless. "sine" carries the same flux with the wrong
shape, so the network still has to do work. Use "sine" for any reported
comparison; "cubic" exists to isolate optimiser behaviour.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..geometry import lifts
from ..geometry.domain import normalise, phi


def _lifting(x, lifting_length):
    """lambda(x) = 1 - exp(-x/l): zero at the inlet, ~1 within a few l."""
    return 1.0 - jnp.exp(-x / lifting_length)


def _pressure_gauge_factor(x, length, gauge_mode):
    if gauge_mode == "soft":
        return 1.0
    if gauge_mode == "hard":
        return 1.0 - x / length
    raise ValueError(f"unknown pressure gauge mode {gauge_mode!r}")


def psi_base(y, flux, base):
    """Stream function carrying `flux` with zero slope at both walls."""
    if base == "cubic":
        return 0.25 * flux * (3.0 * y - y**3)
    if base == "sine":
        return 0.5 * flux * jnp.sin(0.5 * jnp.pi * y)
    raise ValueError(f"unknown stream_base {base!r}")


def make_velocity_pressure_field(
    net_apply, *, length, inlet_mode, lifting_length, gauge_mode
):
    """Build field_fn(params, xy) -> [u, v, p] for the u-v-p formulation."""

    def field_fn(params, xy):
        x, y = xy[0], xy[1]
        raw = net_apply(params, normalise(xy, length))
        wall = phi(y)

        if inlet_mode == "soft":
            u = wall * raw[0]
            v = wall * raw[1]
        elif inlet_mode == "lifting":
            lift = _lifting(x, lifting_length)
            # u(0, y) = phi(y) = 1 - y^2 exactly; v(0, y) = 0 exactly.
            u = wall * (1.0 + lift * raw[0])
            v = wall * lift * raw[1]
        else:
            raise ValueError(f"unknown inlet mode {inlet_mode!r}")

        p = _pressure_gauge_factor(x, length, gauge_mode) * raw[2]
        return jnp.array([u, v, p])

    return field_fn


def make_stream_function_field(net_apply, *, length, flux, base, gauge_mode):
    """Build field_fn(params, xy) -> [u, v, p] for the stream-function form.

    Continuity is satisfied identically, so there is no continuity loss term.
    The cost is one extra order of differentiation in the momentum residual:
    u_xx here is a third derivative of the network.
    """

    def stream(params, xy):
        y = xy[1]
        raw = net_apply(params, normalise(xy, length))
        return psi_base(y, flux, base) + phi(y) ** 2 * raw[0]

    def field_fn(params, xy):
        gradient = jax.grad(stream, argnums=1)(params, xy)
        u = gradient[1]
        v = -gradient[0]
        raw = net_apply(params, normalise(xy, length))
        p = _pressure_gauge_factor(xy[0], length, gauge_mode) * raw[1]
        return jnp.array([u, v, p])

    return field_fn


def make_psi_omega_field(net_apply, *, length, flux, lift):
    """Build the psi-omega pair for SPEC v2 S4.

        psi   = psi_lift(y*) + phi(y*)^2 N_psi(x*, y*)
        omega = N_omega(x*, y*)                          -- NO hard constraint

    At the walls phi = 0, so psi = psi_lift there and
    d_y[phi^2 N] = 2 phi phi' N + phi^2 d_y N = 0. Both wall conditions hold
    exactly for any parameters, which is what pins each wall to its own
    streamline constant and hence fixes the flux.

    `omega` is deliberately left free. It is coupled only through
    r_1 = omega + lap psi, which the network can violate -- that residual is
    what partially restores the falsifiability the pure-psi form destroys.
    Hard-constraining omega would remove it (SPEC v2 S4, and see S5).
    """
    lift_fn = lifts.get(lift)

    def stream_fn(params, xy):
        y = xy[1]
        raw = net_apply(params, normalise(xy, length))
        psi = lift_fn(y, flux) + phi(y) ** 2 * raw[0]
        omega = raw[1]
        return jnp.array([psi, omega])

    return stream_fn


def make_velocity_from_stream(stream_fn):
    """(u, v) = (dpsi/dy, -dpsi/dx) from a psi-omega stream function."""

    def velocity_fn(params, xy):
        gradient = jax.grad(lambda p, z: stream_fn(p, z)[0], argnums=1)(params, xy)
        return jnp.array([gradient[1], -gradient[0]])

    return velocity_fn
