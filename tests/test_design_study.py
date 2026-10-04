"""The flow-share design study (bifurcation/design_study.py): its pieces
that need no finite-element solve. The study itself is the evidence it writes."""

import copy
import math

import pytest
import yaml

from bifurcation import design_study as d
from developing_flow import pinn


def base():
    with open("bifurcation/study_re10_shares.yaml") as handle:
        return yaml.safe_load(handle)


def test_the_friction_law_split():
    """w_i ~ Q_i^(1/2) minimises sum Q_i^2 / w_i^3 under w1 + w2 = w0."""
    assert d.friction_split(0.5, 0.2) == 0.5
    q = 0.3
    s = d.friction_split(q, 0.2)
    cost = lambda x: q**2 / x**3 + (1 - q) ** 2 / (1 - x) ** 3      # noqa: E731
    h = 1e-6
    assert abs((cost(s + h) - cost(s - h)) / (2 * h)) < 1e-6 * cost(s)
    assert d.friction_split(0.001, 0.2) == 0.1                        # clipped to the bound
    assert d.friction_split(0.999, 0.2) == 0.9


def test_the_study_config_is_checked():
    d.parse_config(base())
    for change, match in (({"shares": [0.0, 0.5]}, "share"), ({"angles": [[60.0]]}, "pairs"),
                          ({"search": {"grid": 2, "xatol": 1e-3}}, "grid")):
        doc = copy.deepcopy(base())
        doc.update(change)
        with pytest.raises(pinn.ConfigError, match=match):
            d.parse_config(doc)
    doc = base()
    del doc["workers"]
    with pytest.raises(pinn.ConfigError, match="workers"):
        d.parse_config(doc)
