"""bifurcation/cost_checks.py: its config is complete or refused, and its
curvature estimate removes the h^2 error of the central second difference."""

import copy

import pytest
import yaml

from bifurcation import cost_checks as K

CONFIG = yaml.safe_load(open(K.ROOT / "bifurcation/cost_checks_re10.yaml"))


def test_the_stored_config_parses():
    K.parse_config(CONFIG)


@pytest.mark.parametrize("key", sorted(K.SCHEMA))
def test_a_missing_key_is_refused(key):
    document = copy.deepcopy(CONFIG)
    del document[key]
    with pytest.raises(K.ConfigError, match=key):
        K.parse_config(document)


@pytest.mark.parametrize("steps", [[0.02, 0.01], [0.01], [0.0, 0.01]])
def test_curvature_steps_must_be_two_increasing_positive_steps(steps):
    document = copy.deepcopy(CONFIG)
    document["curvature_steps"] = steps
    with pytest.raises(K.ConfigError, match="curvature_steps"):
        K.parse_config(document)


def test_richardson_is_exact_for_a_quartic():
    """f = 3 + 2 s^2 + 5 s^4 at s = 0: D(h) = 4 + 10 h^2 exactly, f'' = 4."""
    f = lambda s: 3.0 + 2.0 * s**2 + 5.0 * s**4                    # noqa: E731
    steps = (0.01, 0.02)
    second = [(f(-h) - 2.0 * f(0.0) + f(h)) / h**2 for h in steps]
    assert abs(second[0] - 4.0) > 1e-4                              # the plain difference is not
    assert abs(K.richardson(second, steps) - 4.0) < 1e-9
