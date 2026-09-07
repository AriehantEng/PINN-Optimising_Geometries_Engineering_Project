"""Baseline tanh MLP and the modified (two-encoder) MLP.

Both take an already-embedded input vector and return `n_out` raw outputs; the
hard boundary-condition ansatz is applied outside, in `hard_bc.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .dense import apply_dense, init_dense


def init_mlp(key, n_in: int, n_out: int, width: int, depth: int, **rwf):
    """Depth `depth` counts HIDDEN layers, so there are depth+1 dense layers."""
    sizes = [n_in] + [width] * depth + [n_out]
    keys = jax.random.split(key, len(sizes) - 1)
    return {
        "layers": [
            init_dense(k, a, b, **rwf) for k, a, b in zip(keys, sizes[:-1], sizes[1:])
        ]
    }


def apply_mlp(params, x):
    *hidden, last = params["layers"]
    for layer in hidden:
        x = jnp.tanh(apply_dense(layer, x))
    return apply_dense(last, x)


def init_modified_mlp(key, n_in: int, n_out: int, width: int, depth: int, **rwf):
    """Wang et al.'s modified MLP: two input encoders mixed multiplicatively.

        U = tanh(W_u xi + b_u),  V = tanh(W_v xi + b_v)
        H_1 = tanh(W_1 xi + b_1)
        Z_k = tanh(W_k H_k + b_k)
        H_{k+1} = (1 - Z_k) * U + Z_k * V

    The encoders let every hidden layer see the input again, which is what
    fixes the vanishing input sensitivity of a deep plain MLP.
    """
    keys = jax.random.split(key, depth + 3)
    encoder_u = init_dense(keys[0], n_in, width, **rwf)
    encoder_v = init_dense(keys[1], n_in, width, **rwf)
    hidden = [init_dense(keys[2], n_in, width, **rwf)]
    hidden += [init_dense(keys[3 + i], width, width, **rwf) for i in range(depth - 1)]
    output = init_dense(keys[depth + 2], width, n_out, **rwf)
    return {"encoder_u": encoder_u, "encoder_v": encoder_v, "hidden": hidden, "output": output}


def apply_modified_mlp(params, x):
    encoded_u = jnp.tanh(apply_dense(params["encoder_u"], x))
    encoded_v = jnp.tanh(apply_dense(params["encoder_v"], x))
    h = x
    for layer in params["hidden"]:
        z = jnp.tanh(apply_dense(layer, h))
        h = (1.0 - z) * encoded_u + z * encoded_v
    return apply_dense(params["output"], h)
