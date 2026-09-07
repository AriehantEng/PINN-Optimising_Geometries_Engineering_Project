"""Shared test fixtures.

Importing `src` enables float64 before any array is created; every test module
relies on that having happened, so it is asserted here once for the session.
"""

import jax.numpy as jnp
import pytest

import src  # noqa: F401  (import for its float64 side effect)


@pytest.fixture(scope="session", autouse=True)
def _require_float64():
    assert jnp.zeros(1).dtype == jnp.float64, (
        "float64 is not active. Every accuracy target in this project is "
        "unreachable in float32 (CLAUDE.md, 'Non-negotiable numerical rules')."
    )
