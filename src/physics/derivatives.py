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
"""

from __future__ import annotations

import jax

from .residuals import Derivatives


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

    u, v, p = value[0], value[1], value[2]
    return Derivatives(
        u=u,
        v=v,
        p=p,
        u_x=jacobian[0, 0],
        u_y=jacobian[0, 1],
        v_x=jacobian[1, 0],
        v_y=jacobian[1, 1],
        p_x=jacobian[2, 0],
        p_y=jacobian[2, 1],
        u_xx=hessian[0, 0, 0],
        u_yy=hessian[0, 1, 1],
        v_xx=hessian[1, 0, 0],
        v_yy=hessian[1, 1, 1],
    )


def batched_field_derivatives(field_fn, points) -> Derivatives:
    """`field_derivatives` vmapped over an (N, 2) array of points."""
    return jax.vmap(lambda xy: field_derivatives(field_fn, xy))(points)
