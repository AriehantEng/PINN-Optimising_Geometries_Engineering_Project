"""The width rules compared in bifurcation/width_rules.py."""

import math

import pytest

from bifurcation import width_rules as wr
from developing_flow import pinn

SHARES = (0.05, 0.2, 0.5, 0.8, 0.95)


@pytest.mark.parametrize("q", SHARES)
def test_each_rule_satisfies_its_defining_law(q):
    w0 = 2.0
    w1, w2 = wr.rule_widths("equal", q)
    assert w1 == w2 == 1.0
    w1, w2 = wr.rule_widths("shared_murray", q)
    assert abs(w1 + w2 - w0) < 1e-15
    assert abs((w1 / w2) - math.sqrt(q / (1 - q))) < 1e-12           # w ~ Q^(1/2)
    w1, w2 = wr.rule_widths("square", q)
    assert abs(w1**2 + w2**2 - w0**2) < 1e-14
    w1, w2 = wr.rule_widths("cube", q)
    assert abs(w1**3 + w2**3 - w0**3) < 1e-13
    w1, w2 = wr.rule_widths("shared_optimum", q, optimum_split=0.42)
    assert (w1, w2) == (2.0 * 0.42, 2.0 * (1.0 - 0.42))
    with pytest.raises(pinn.ConfigError):
        wr.rule_widths("shared_optimum", q)


def test_the_square_law_is_the_shared_rule_scaled_by_sqrt_q_plus_sqrt_1_minus_q():
    """The friction law's optimum total width under Murray's cost:
    k = sqrt(q) + sqrt(1 - q), at the same proportions as the shared rule."""
    for q in SHARES:
        k = math.sqrt(q) + math.sqrt(1 - q)
        shared, square = wr.rule_widths("shared_murray", q), wr.rule_widths("square", q)
        assert abs(square[0] - k * shared[0]) < 1e-14 and abs(square[1] - k * shared[1]) < 1e-14


def test_the_murray_price_makes_the_parent_width_optimal():
    """d/dw [12 nu Q0^2 / w^3 + b w] = 0 at w = w0 = 2."""
    b = wr.murray_price(10.0, "inertial")
    nu, q0 = 0.1, 4.0 / 3.0

    def cost(w):
        return 12.0 * nu * q0**2 / w**3 + b * w

    h = 1e-5
    assert abs((cost(2.0 + h) - cost(2.0 - h)) / (2 * h)) < 1e-9
    assert cost(2.0) < cost(1.9) and cost(2.0) < cost(2.1)


def test_the_config_is_checked():
    import copy
    import yaml
    from pathlib import Path
    document = yaml.safe_load((Path(wr.__file__).parent / "width_rules_re10.yaml").read_text())
    wr.parse_config(document)
    bad = copy.deepcopy(document)
    bad["rules"] = ["equal", "golden"]
    with pytest.raises(pinn.ConfigError, match="golden"):
        wr.parse_config(bad)
    bad = copy.deepcopy(document)
    del bad["scan"]["scales"]
    with pytest.raises(pinn.ConfigError, match="scales"):
        wr.parse_config(bad)
    bad = copy.deepcopy(document)
    bad["shares"] = [0.5, 1.0]
    with pytest.raises(pinn.ConfigError, match="share"):
        wr.parse_config(bad)
