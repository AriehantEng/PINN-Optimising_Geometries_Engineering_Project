"""PINN for branching channel flow — Phase 1 (plane Poiseuille).

IMPORTANT: double precision is enabled here, at package import, *before* any
JAX array can be created by a submodule. float32 silently caps achievable
relative L2 error near 1e-4 and the symptom is indistinguishable from an
optimiser failure. Do not move, guard or make this conditional.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

import jax.numpy as _jnp  # noqa: E402  (must follow the x64 switch)

if _jnp.zeros(1).dtype != _jnp.float64:
    raise RuntimeError(
        "float64 is not active: jnp.zeros(1).dtype == "
        f"{_jnp.zeros(1).dtype}. Something imported jax and created an array "
        "before src/__init__.py ran, or JAX_ENABLE_X64 was overridden."
    )

__all__ = ["assert_float64"]


def assert_float64(*arrays, where=""):
    """Raise unless every argument is float64.

    Used at module boundaries where a float32 leak would be silent. `where` is
    a free-text label included in the error message.
    """
    for i, a in enumerate(arrays):
        dtype = _jnp.asarray(a).dtype
        if dtype != _jnp.float64:
            raise TypeError(
                f"float64 required{' in ' + where if where else ''}: "
                f"argument {i} has dtype {dtype}. See CLAUDE.md, "
                "'Non-negotiable numerical rules'."
            )
