"""Collocation sampling.

Two rules from the SPEC drive the design here:

* the sampling seed is logged and every draw is a pure function of an explicit
  PRNG key, so a run is reproducible from (config, seed) alone;
* the fixed EVALUATION grid is built by a different function from the training
  points, because reported error must never be measured on training points
  (SPEC section 9, "Resampling collocation points").
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .domain import WALL, Channel


class CollocationPoints(NamedTuple):
    """One draw of training points. All arrays are (N, 2) of (x*, y*)."""

    interior: jnp.ndarray
    inlet: jnp.ndarray
    outlet: jnp.ndarray


def sample_interior(key, channel: Channel, n: int):
    """Uniform random points strictly inside the channel.

    Wall points are excluded: no-slip is exact by construction, so a
    collocation point on the wall contributes nothing but a degenerate residual
    (phi = 0 makes u and v identically zero there for any parameters).
    """
    return jax.random.uniform(
        key,
        shape=(n, 2),
        minval=channel.bounds_lower,
        maxval=channel.bounds_upper,
        dtype=jnp.float64,
    )


def sample_inlet(key, channel: Channel, n: int):
    """Points on x* = 0, y* uniform in (-1, 1)."""
    y = jax.random.uniform(key, shape=(n,), minval=-WALL, maxval=WALL, dtype=jnp.float64)
    return jnp.stack([jnp.zeros_like(y), y], axis=-1)


def sample_outlet(key, channel: Channel, n: int):
    """Points on x* = L*, y* uniform in (-1, 1)."""
    y = jax.random.uniform(key, shape=(n,), minval=-WALL, maxval=WALL, dtype=jnp.float64)
    return jnp.stack([jnp.full_like(y, channel.length), y], axis=-1)


def sample_all(key, channel: Channel, n_interior: int, n_inlet: int, n_outlet: int):
    """One complete draw. Pure function of `key`."""
    k_int, k_in, k_out = jax.random.split(key, 3)
    return CollocationPoints(
        interior=sample_interior(k_int, channel, n_interior),
        inlet=sample_inlet(k_in, channel, n_inlet),
        outlet=sample_outlet(k_out, channel, n_outlet),
    )


def evaluation_grid(channel: Channel, nx: int, ny: int):
    """Fixed tensor-product grid for reporting error. Never used for training.

    Endpoints are included: the walls are where the hard BC is exact and the
    inlet/outlet are where the boundary conditions live, so an error metric
    that skipped them would flatter the model.

    Returns (points (nx*ny, 2), x (nx,), y (ny,)).
    """
    x = jnp.linspace(0.0, channel.length, nx, dtype=jnp.float64)
    y = jnp.linspace(-WALL, WALL, ny, dtype=jnp.float64)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    points = jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    return points, x, y


def station_grid(channel: Channel, n_stations: int, ny: int, *, margin: float = 0.0):
    """Streamwise stations for the mass-conservation check (SPEC test 3).

    `margin` trims the stations away from the inlet and outlet planes; the
    conservation check is about the interior, and a station sitting exactly on
    a boundary would fold a BC error into a conservation error.

    Returns (points (n_stations, ny, 2), x_stations (n_stations,), y (ny,)).
    """
    x = jnp.linspace(
        margin * channel.length,
        (1.0 - margin) * channel.length,
        n_stations,
        dtype=jnp.float64,
    )
    y = jnp.linspace(-WALL, WALL, ny, dtype=jnp.float64)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    return jnp.stack([xx, yy], axis=-1), x, y
