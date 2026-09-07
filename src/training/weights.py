"""The ONLY place loss weighting happens.

CLAUDE.md: "Residual functions return unweighted residuals. All loss weighting
happens in one place (src/training/weights.py)." If you find yourself wanting
to scale a residual anywhere else, the scaling belongs here instead.

Two schemes:

fixed
    The weights from the config, held constant. Start here.

grad_norm
    Learning-rate annealing of the weights (Wang, Teng & Perdikaris 2021). Every
    `update_every` steps:

        w_hat_i = (sum_j ||grad_theta L_j||) / ||grad_theta L_i||
        w_i     <- alpha * w_i + (1 - alpha) * w_hat_i

    which equalises the gradient contribution of each term. The config weights
    are the starting values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def initial_weights(config):
    """Starting weights, ordered exactly as `config.loss_terms`."""
    return {term: jnp.asarray(config.weights.values[term], dtype=jnp.float64)
            for term in config.loss_terms}


def total_loss(terms, weights):
    """Weighted sum. The single point where weights meet residuals."""
    missing = set(weights) ^ set(terms)
    if missing:
        raise KeyError(
            f"loss terms and weights disagree on {sorted(missing)}; "
            "this means the config's active-term list and the loss builder "
            "have drifted apart"
        )
    return sum(weights[name] * terms[name] for name in sorted(terms))


def grad_norm_weights(term_grad_norms, previous, alpha):
    """One grad-norm annealing update.

    `term_grad_norms` maps term name -> ||grad_theta L_term||_2 (the norm of the
    gradient of the UNWEIGHTED term). A term whose gradient norm is zero (e.g. a
    boundary condition already satisfied exactly) keeps its previous weight
    rather than producing an infinity.
    """
    total = sum(term_grad_norms.values())
    updated = {}
    for name, norm in term_grad_norms.items():
        target = jnp.where(norm > 0.0, total / jnp.where(norm > 0.0, norm, 1.0),
                           previous[name])
        updated[name] = alpha * previous[name] + (1.0 - alpha) * target
    return updated


def make_grad_norm_update(loss_terms_fn):
    """Build a jittable function computing per-term gradient norms."""

    def term_grad_norms(params, points):
        norms = {}
        for name in loss_terms_fn(params, points):
            grad = jax.grad(lambda p, n=name: loss_terms_fn(p, points)[n])(params)
            leaves = jax.tree.leaves(grad)
            norms[name] = jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))
        return norms

    return term_grad_norms
