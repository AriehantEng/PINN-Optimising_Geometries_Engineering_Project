"""The regularised Case B inlet family (src/physics/inlets.py).

The top-hat inlet is unsolvable under exact no-slip: it needs an unbounded
network output at the corners. This family approaches it through profiles that
are solvable, so the idealisation is recovered as a LIMIT rather than imposed.
These tests pin the three properties that make that argument work.
"""

import copy

import jax.numpy as jnp
import numpy as np
import pytest

from src.config import ConfigError, load_config, parse_config
from src.physics import inlets

SHARPNESS = [2, 4, 8, 16, 32, 64]


@pytest.mark.parametrize("m", [1] + SHARPNESS)
def test_every_profile_carries_exactly_the_developed_flux(m):
    """Flux 4/3 for every m, so a sweep changes the SHAPE and nothing else.

    If the flux moved with m, a sharpness sweep would confound the inlet shape
    with the amount of fluid entering, and no comparison across m would mean
    anything.
    """
    y = np.linspace(-1.0, 1.0, 400001)
    flux = float(np.trapezoid(np.asarray(inlets.profile(jnp.asarray(y), m)), y))
    assert flux == pytest.approx(4.0 / 3.0, rel=1e-9)
    assert inlets.FLUX == 4.0 / 3.0


@pytest.mark.parametrize("m", [1] + SHARPNESS)
def test_the_profile_satisfies_no_slip_exactly(m):
    """u = 0 at both walls to the last bit -- not approximately."""
    walls = inlets.profile(jnp.array([-1.0, 1.0]), m)
    assert np.asarray(walls).tolist() == [0.0, 0.0]


@pytest.mark.parametrize("m", SHARPNESS)
def test_the_network_demand_is_bounded_and_grows_linearly(m):
    """N_u = u/(1 - y^2) stays finite, which is the whole point.

    1 - y^(2m) = (1 - y^2)(1 + y^2 + ... + y^(2m-2)) exactly, so N_u is a
    polynomial with maximum U0*m at the wall. The top hat is the m -> infinity
    limit and diverges linearly in m; every finite m is representable.
    """
    y = jnp.linspace(-0.999999, 0.999999, 20001)
    demand = np.asarray(inlets.profile(y, m) / (1.0 - y**2))
    assert np.all(np.isfinite(demand))
    predicted = inlets.network_value_at_wall(m)
    assert np.max(np.abs(demand)) <= predicted * (1 + 1e-6)
    assert predicted == pytest.approx(inlets.peak(m) * m)


@pytest.mark.parametrize("m", SHARPNESS)
def test_the_factorisation_the_argument_rests_on(m):
    y = np.linspace(-0.999, 0.999, 9999)
    left = 1.0 - y ** (2 * m)
    right = (1.0 - y**2) * sum(y ** (2 * k) for k in range(m))
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-14)


def test_m_equals_one_is_the_developed_profile():
    """The degeneracy this family has to be guarded against."""
    y = jnp.linspace(-1.0, 1.0, 1001)
    np.testing.assert_allclose(
        np.asarray(inlets.profile(y, 1)), np.asarray(1.0 - y**2),
        rtol=0, atol=1e-15,
    )
    assert inlets.peak(1) == pytest.approx(1.0)
    assert 1 in inlets.DEGENERATE_SHARPNESS


def test_the_family_sharpens_monotonically_toward_the_top_hat():
    """Peak falls toward 2/3 and the wall layer thins as m grows."""
    peaks = [inlets.peak(m) for m in SHARPNESS]
    assert peaks == sorted(peaks, reverse=True)
    assert peaks[-1] == pytest.approx(2.0 / 3.0, rel=2e-2)
    layers = [inlets.wall_layer_thickness(m) for m in SHARPNESS]
    assert layers == sorted(layers, reverse=True)


@pytest.mark.parametrize("bad", [0, -1])
def test_sharpness_below_one_is_refused(bad):
    with pytest.raises(ValueError, match="at least 1"):
        inlets.profile(jnp.array([0.0]), bad)


@pytest.mark.parametrize("bad", [2.5, True])
def test_non_integer_sharpness_is_refused(bad):
    """A non-integer m is not a polynomial, so N_u would not be bounded."""
    with pytest.raises(TypeError, match="integer"):
        inlets.profile(jnp.array([0.0]), bad)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def _case_b_document():
    config = load_config("experiments/caseB_re10_smooth8.yaml")
    return copy.deepcopy(config.raw)


def test_sharpness_one_is_refused_by_config_as_vacuous():
    """A vacuous Case B must not be runnable, or it could be reported."""
    document = _case_b_document()
    document["physics"]["inlet_sharpness"] = 1
    with pytest.raises(ConfigError, match="vacuous"):
        parse_config(document)


def test_the_two_inlet_parameterisations_cannot_be_mixed():
    document = _case_b_document()
    document["physics"]["inlet_uniform_value"] = 0.6666666666666666
    with pytest.raises(ConfigError, match="inlet_uniform_value"):
        parse_config(document)


def test_case_a_rejects_every_inlet_key():
    config = load_config("experiments/caseA_re10.yaml")
    for key, value in (("inlet_profile", "smoothed"), ("inlet_sharpness", 8),
                       ("inlet_uniform_value", 0.667)):
        document = copy.deepcopy(config.raw)
        document["physics"][key] = value
        with pytest.raises(ConfigError, match="case A"):
            parse_config(document)


def test_smoothed_and_uniform_carry_the_same_flux_and_domain():
    """The comparison is only meaningful if nothing else moved."""
    uniform = load_config("experiments/caseB_re10.yaml")
    smoothed = load_config("experiments/caseB_re10_smooth8.yaml")
    assert smoothed.inlet_flux == pytest.approx(uniform.inlet_flux, rel=1e-12)
    assert smoothed.domain.length == pytest.approx(uniform.domain.length, rel=1e-12)
    assert smoothed.physics.reynolds == uniform.physics.reynolds
    assert smoothed.inlet_profile == "smoothed" and smoothed.inlet_sharpness == 8
    assert uniform.inlet_profile == "uniform"
