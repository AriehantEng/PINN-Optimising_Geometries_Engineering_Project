"""Autodiff assembly of the derivative bundle the residual operator consumes.

Kept separate from `residuals.py` so that the residual algebra can be unit
tested against hand-derived expressions with no autodiff in the loop, and the
autodiff itself can be checked against finite differences independently
(SPEC test 5).

A "field function" here is any callable `f(xy) -> array of shape (3,)` giving
(u, v, p) at the single point `xy = [x, y]`. It may internally be a network, a
hard-BC-wrapped network, an analytic expression, or a stream-function
construction that differentiates a scalar potential -- this module does not
care, and takes second derivatives of whatever it is handed.

Two implementations of the same mathematics live here:

`field_derivatives`
    One point, nested `jacfwd`. Short, obviously correct, used as the reference.

`batched_field_derivatives`
    A whole (N, 2) array at once. Instead of vmapping the single-point version,
    it differentiates the ALREADY-BATCHED field function along a constant
    tangent direction, twice. Each output row depends only on its own input
    row, so a batch-wide directional derivative gives every point's derivative
    in that direction in one pass -- and every pass stays a full-size matrix
    product rather than a vmapped matrix-vector product.

    Measured on the baseline config (width 128, depth 4, 4096 points): the
    batched form costs 195 ms per loss-and-gradient against 460 ms for the
    vmapped one, for bitwise-comparable values. The reverse-mode pass over the
    vmapped nested `jacfwd` is the expensive part -- it stores a tangent copy
    of every activation for every point -- and the batched form cuts the number
    of those copies. `test_batched_derivatives_match_single_point` pins the two
    together so the fast path cannot drift from the reference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .residuals import Derivatives

#: Unit tangent directions in (x, y).
_EX = jnp.array([1.0, 0.0])
_EY = jnp.array([0.0, 1.0])


def field_derivatives(field_fn, xy) -> Derivatives:
    """Value, first and second derivatives of (u, v, p) at one point.

    Forward-mode throughout: the input is 2-dimensional and the output is
    3-dimensional, so forward mode is the cheaper direction for both the
    Jacobian and the Hessian.
    """

    def value_and_jacobian(point):
        value = field_fn(point)
        jacobian = jax.jacfwd(field_fn)(point)
        # `has_aux` lets the outer jacfwd differentiate the Jacobian while
        # handing back the value and Jacobian it already computed, instead of
        # recomputing them in a separate pass.
        return jacobian, (value, jacobian)

    hessian, (value, jacobian) = jax.jacfwd(value_and_jacobian, has_aux=True)(xy)

    return _assemble(value, jacobian[:, 0], jacobian[:, 1], hessian[:, 0, 0], hessian[:, 1, 1])


def _assemble(value, d_dx, d_dy, d2_dx2, d2_dy2) -> Derivatives:
    """Pack directional derivatives into the bundle. Index 0=u, 1=v, 2=p."""
    return Derivatives(
        u=value[..., 0], v=value[..., 1], p=value[..., 2],
        u_x=d_dx[..., 0], v_x=d_dx[..., 1], p_x=d_dx[..., 2],
        u_y=d_dy[..., 0], v_y=d_dy[..., 1], p_y=d_dy[..., 2],
        u_xx=d2_dx2[..., 0], v_xx=d2_dx2[..., 1],
        u_yy=d2_dy2[..., 0], v_yy=d2_dy2[..., 1],
    )


def _directional(batched_fn, points, direction):
    """(value, first, second) derivatives along a constant `direction`.

    The inner `jvp` returns the value alongside the first derivative, and the
    outer one differentiates that pair, so a single call yields all three
    without a separate primal evaluation.
    """
    tangents = jnp.broadcast_to(direction, points.shape)

    def value_and_first(pts):
        return jax.jvp(batched_fn, (pts,), (tangents,))

    (value, first), (_, second) = jax.jvp(value_and_first, (points,), (tangents,))
    return value, first, second


def batched_field_derivatives(field_fn, points) -> Derivatives:
    """Derivative bundle for an (N, 2) array of points."""
    batched_fn = jax.vmap(field_fn)
    value, d_dx, d2_dx2 = _directional(batched_fn, points, _EX)
    _, d_dy, d2_dy2 = _directional(batched_fn, points, _EY)
    return _assemble(value, d_dx, d_dy, d2_dx2, d2_dy2)


def field_and_jacobian(field_fn, xy):
    """(value, jacobian) at one point, without the Hessian.

    Boundary terms (outlet du/dx, dv/dx) need only first derivatives; computing
    the full bundle there would triple the cost for nothing.
    """
    return field_fn(xy), jax.jacfwd(field_fn)(xy)


def batched_field_and_jacobian(field_fn, points):
    """(values (N, 3), jacobians (N, 3, 2)) for an (N, 2) array of points."""
    batched_fn = jax.vmap(field_fn)
    value, d_dx = jax.jvp(batched_fn, (points,), (jnp.broadcast_to(_EX, points.shape),))
    _, d_dy = jax.jvp(batched_fn, (points,), (jnp.broadcast_to(_EY, points.shape),))
    return value, jnp.stack([d_dx, d_dy], axis=-1)
