"""Dense layers, with optional random weight factorisation (RWF).

RWF (SPEC section 3 item 4) reparameterises each weight matrix as

    W = diag(exp(s)) V,      s ~ N(mean, stddev)

so every output neuron carries its own learnable scale. Initialisation draws W
by Glorot normal first and then factors it, so switching RWF on does not change
the initial function the network represents -- only how the optimiser moves
through parameter space.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def glorot_normal(key, n_out: int, n_in: int):
    """Glorot (Xavier) normal initialisation, float64."""
    stddev = jnp.sqrt(2.0 / (n_in + n_out))
    return stddev * jax.random.normal(key, (n_out, n_in), dtype=jnp.float64)


def init_dense(key, n_in: int, n_out: int, *, rwf_enabled: bool, rwf_mean=None, rwf_stddev=None):
    """Initialise one dense layer as a parameter pytree."""
    w_key, s_key = jax.random.split(key)
    weight = glorot_normal(w_key, n_out, n_in)
    bias = jnp.zeros((n_out,), dtype=jnp.float64)

    if not rwf_enabled:
        return {"W": weight, "b": bias}

    if rwf_mean is None or rwf_stddev is None:
        raise ValueError("RWF requires both rwf_mean and rwf_stddev")
    scale = rwf_mean + rwf_stddev * jax.random.normal(s_key, (n_out,), dtype=jnp.float64)
    # V = W / exp(s), so that diag(exp(s)) V == W exactly at initialisation.
    factor = weight / jnp.exp(scale)[:, None]
    return {"s": scale, "V": factor, "b": bias}


def apply_dense(layer, x):
    """Apply one dense layer to a single input vector."""
    if "W" in layer:
        return layer["W"] @ x + layer["b"]
    return (jnp.exp(layer["s"])[:, None] * layer["V"]) @ x + layer["b"]


def layer_weight(layer):
    """The effective weight matrix of a layer, factorised or not."""
    if "W" in layer:
        return layer["W"]
    return jnp.exp(layer["s"])[:, None] * layer["V"]
