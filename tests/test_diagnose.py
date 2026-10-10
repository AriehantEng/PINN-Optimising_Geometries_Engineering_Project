"""The PINN error diagnostic (bifurcation/diagnose.py)."""

import copy

import numpy as np
import pytest
import yaml

from bifurcation import diagnose as D
from bifurcation import flow
from bifurcation.geometry import Geometry


def document():
    return yaml.safe_load(open(D.ROOT / "bifurcation/diagnose_re10.yaml"))


def test_the_config_is_checked():
    D.parse_config(document())
    for change, match in ((lambda d: d.pop("runs"), "runs"),
                          (lambda d: d.update(corner_bins=[0.1, 0.5]), "corner_bins"),
                          (lambda d: d.update(corner_bins=[0.0, 0.5, 0.2]), "corner_bins"),
                          (lambda d: d["runs"][0].update(extra=1), "runs\\[0\\]")):
        d = copy.deepcopy(document())
        change(d)
        with pytest.raises(D.ConfigError, match=match):
            D.parse_config(d)


def test_the_annuli_partition_the_integral():
    rng = np.random.default_rng(0)
    r, w, f = rng.uniform(0, 3, 500), rng.uniform(0, 1, 500), rng.normal(size=500)
    edges = [0.0, 0.1, 0.5, 1.0, 100.0]
    whole, part = D.binned(edges, r, w, np.ones_like(r), f)
    assert abs(whole.sum() - w.sum()) < 1e-12
    assert abs(part.sum() - (w * f).sum()) < 1e-12
    assert abs(part[1] - (w * f)[(r >= 0.1) & (r < 0.5)].sum()) < 1e-12


def _field(scale):
    """A smooth (u, v, p) defined everywhere: the parts must reproduce
    flow.opening_measures' drop whatever the field."""
    def model(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return (scale * (1 - 0.3 * y**2) + 0.1 * x, 0.2 * np.sin(x) * y, 3.0 - 0.4 * x + 0.1 * y)
    return model


def test_the_energy_parts_give_opening_measures_drop():
    g = Geometry.from_split(0.45, 30.0, 70.0, 3.0, 4.0, 0.2)
    for scale in (1.0, 1.3):
        parts = D.energy_parts(_field(scale), g, 1.0, 64)
        expected = flow.opening_measures(_field(scale), g, 1.0, 64)["total_pressure_drop"]
        assert abs(D.drop_from_parts(parts) - expected) < 1e-13 * abs(expected)


def test_the_drop_error_parts_sum_to_the_drop_error_at_equal_fluxes():
    """Same field, pressure shifted at the openings: fluxes equal, so the parts
    sum to the difference of the drops exactly."""
    g = Geometry.from_split(0.45, 30.0, 70.0, 3.0, 4.0, 0.2)
    base = _field(1.0)

    def shifted(x, y):
        u, v, p = base(x, y)
        return u, v, p + 0.05 * np.asarray(x)

    a, b = D.energy_parts(shifted, g, 1.0, 64), D.energy_parts(base, g, 1.0, 64)
    parts = D.drop_error_parts(a, b)
    assert abs(sum(parts.values()) - (D.drop_from_parts(a) - D.drop_from_parts(b))) < 1e-13
    assert all(abs(v) < 1e-15 for k, v in parts.items() if k.endswith("kinetic"))


def test_corner_distance():
    corners = np.array([[0.0, 0.0], [3.0, 4.0]])
    points = np.array([[1.0, 0.0], [3.0, 3.5], [10.0, 10.0]])
    assert np.allclose(D.corner_distance(points, corners), [1.0, 0.5, np.hypot(7.0, 6.0)])
