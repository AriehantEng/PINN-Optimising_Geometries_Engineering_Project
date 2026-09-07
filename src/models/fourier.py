"""Random Fourier feature embedding of the input coordinates.

    gamma(xi) = [cos(B xi), sin(B xi)],   B ~ N(0, bandwidth^2)

with `xi` the coordinates already normalised to [-1, 1]^2. Addresses the
spectral bias of a plain MLP (SPEC section 3 item 2).

B is drawn once from the run's PRNG key and then held FIXED -- it is not a
trainable parameter, so it lives outside the parameter pytree and is never
touched by the optimiser. That also keeps the L-BFGS parameter vector free of
constants.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def init_fourier_matrix(key, num_features: int, bandwidth: float, n_inputs: int = 2):
    """Draw the fixed frequency matrix B of shape (num_features, n_inputs)."""
    return bandwidth * jax.random.normal(
        key, (num_features, n_inputs), dtype=jnp.float64
    )


def apply_fourier(matrix, xi):
    """Embed a single normalised coordinate vector. Output dim = 2 * num_features."""
    projection = matrix @ xi
    return jnp.concatenate([jnp.cos(projection), jnp.sin(projection)])
