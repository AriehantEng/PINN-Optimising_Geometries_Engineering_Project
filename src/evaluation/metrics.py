"""Accuracy metrics.

Accuracy is ALWAYS the relative L2 error against the analytic solution,
computed on the fixed evaluation grid. The loss value is a diagnostic and is
never reported as an accuracy claim (CLAUDE.md).

The published failure mode this guards against: a plain DeepXDE setup on this
exact problem reaching a total test loss of 8.7e-5 while its relative L2 error
was 8.6% (SPEC section 5).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry.domain import WALL


def relative_l2(predicted, exact):
    """||predicted - exact||_2 / ||exact||_2.

    Raises if the reference field is identically zero: the relative L2 error of
    a zero field is not defined, and silently returning an absolute error there
    would be exactly the kind of quiet lie this project is built to avoid. Use
    `max_abs` for fields whose exact value is zero (e.g. v*).
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    denominator = np.linalg.norm(exact)
    if denominator == 0.0:
        raise ValueError(
            "relative L2 error is undefined against an identically zero "
            "reference field; use max_abs instead"
        )
    return float(np.linalg.norm(predicted - exact) / denominator)


def max_abs(values):
    return float(np.max(np.abs(np.asarray(values))))


def gauss_legendre_nodes(n: int):
    """Gauss-Legendre nodes and weights mapped onto y* in [-1, 1].

    Used for every cross-channel integral. The channel half-height is exactly
    the Legendre interval, so no rescaling is needed, and GL is exact for the
    analytic parabola to machine precision -- which matters for SPEC test 3,
    whose tolerance is 1e-6 relative.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes * WALL, weights * WALL


def evaluate_fields(field_fn, params, points, batch_size: int = 8192):
    """(u, v, p) at an (N, 2) array of points, batched to bound peak memory."""
    apply = jax.jit(jax.vmap(lambda xy: field_fn(params, xy)))
    outputs = []
    for start in range(0, points.shape[0], batch_size):
        outputs.append(np.asarray(apply(points[start : start + batch_size])))
    stacked = np.concatenate(outputs, axis=0)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


def cross_channel_flux(field_fn, params, x_stations, n_nodes: int = 200):
    """int_{-1}^{1} u*(x, y) dy at each x station, by Gauss-Legendre quadrature."""
    nodes, weights = gauss_legendre_nodes(n_nodes)
    x_stations = np.asarray(x_stations, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_stations, nodes, indexing="ij")
    points = jnp.asarray(np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1))
    u, _, _ = evaluate_fields(field_fn, params, points)
    u = u.reshape(len(x_stations), n_nodes)
    return u @ weights


def flux_conservation_error(fluxes):
    """Relative spread of the flux across stations.

    Station-to-station AGREEMENT, not agreement with the analytic flux: this is
    the mass-conservation check (SPEC test 3), and a solution that conserves
    mass at the wrong flux is a different failure from one that does not
    conserve it at all. Both are reported.
    """
    fluxes = np.asarray(fluxes, dtype=np.float64)
    mean = float(np.mean(fluxes))
    if mean == 0.0:
        raise ValueError("mean flux is zero; conservation error is undefined")
    return float(np.max(np.abs(fluxes - mean)) / abs(mean))


def centreline_velocity(field_fn, params, x_stations):
    """u*(x, 0) at each station -- the entrance-length diagnostic."""
    x_stations = np.asarray(x_stations, dtype=np.float64)
    points = jnp.asarray(np.stack([x_stations, np.zeros_like(x_stations)], axis=-1))
    u, _, _ = evaluate_fields(field_fn, params, points)
    return u


def entrance_length(x_stations, centreline, developed_value, threshold=0.99):
    """First x* at which the centreline reaches `threshold` of its developed value.

    Linear interpolation between the two bracketing stations. Returns NaN if the
    threshold is never reached inside the domain, which is a real result (the
    domain is too short) and must not be silently clipped to L*.
    """
    x_stations = np.asarray(x_stations, dtype=np.float64)
    centreline = np.asarray(centreline, dtype=np.float64)
    target = threshold * developed_value
    reached = centreline >= target
    if not np.any(reached):
        return float("nan")
    index = int(np.argmax(reached))
    if index == 0:
        return float(x_stations[0])
    x0, x1 = x_stations[index - 1], x_stations[index]
    c0, c1 = centreline[index - 1], centreline[index]
    if c1 == c0:
        return float(x1)
    return float(x0 + (target - c0) * (x1 - x0) / (c1 - c0))
