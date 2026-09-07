"""The Phase 1 domain: a straight channel x* in [0, L*], y* in [-1, 1].

Everything is non-dimensional. The half-height is 1 by construction -- that is
what "lengths are scaled by h" means -- so it is not a free parameter and is
not configurable.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

#: Non-dimensional wall position. Walls are at y* = -WALL and y* = +WALL.
WALL = 1.0


@dataclass(frozen=True)
class Channel:
    """Straight channel of non-dimensional length `length`."""

    length: float

    def __post_init__(self):
        if not (self.length > 0):
            raise ValueError(f"channel length must be positive, got {self.length!r}")

    @property
    def bounds_lower(self):
        return jnp.array([0.0, -WALL])

    @property
    def bounds_upper(self):
        return jnp.array([self.length, WALL])


def phi(y):
    """Wall distance function: phi(y*) = 1 - y*^2.

    Vanishes on both walls and is strictly positive between them, so
    multiplying a network output by it enforces no-slip EXACTLY rather than
    through a penalty. This is the single largest accuracy lever in the project
    (SPEC section 3), which is why it is a named function with its own tests
    rather than an inline expression.
    """
    return 1.0 - y**2


def dphi_dy(y):
    """d(phi)/dy = -2y. Written out so tests can check phi against it."""
    return -2.0 * y


def normalise(xy, length):
    """Map (x*, y*) from [0, L*] x [-1, 1] onto [-1, 1]^2.

    A tanh network fed x* in [0, 20] and y* in [-1, 1] sees two inputs whose
    scales differ by a factor 20; the first hidden layer then has to undo that
    with its weights. This is a structural property of the domain, not a tuned
    hyperparameter, so it is derived from the config's domain length rather
    than being configurable.
    """
    x, y = xy[..., 0], xy[..., 1]
    return jnp.stack([2.0 * x / length - 1.0, y / WALL], axis=-1)
