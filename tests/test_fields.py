"""The results-report field sampler (bifurcation/fields.py)."""

import numpy as np
import pytest
import yaml
from matplotlib.path import Path

from bifurcation import fields as F
from bifurcation.geometry import Geometry


@pytest.mark.parametrize("angles", [[30.0, 90.0], [0.0, 150.0], [60.0, 120.0]])
def test_sampled_points_lie_inside_the_polygon(angles):
    g = Geometry.from_split(0.45, angles[0], angles[1], 3.0, 8.0, 0.2)
    X, Y, inside = F.sample_grid(g, 120)
    pts = np.column_stack([X[inside], Y[inside]])
    assert len(pts) > 1000
    assert Path(g.vertices()).contains_points(pts).all()
    # the grid spans the polygon: points on both sides of every daughter
    assert inside.any(axis=0).sum() > 0.5 * X.shape[1]


def test_values_land_inside_and_nan_outside():
    g = Geometry.from_split(0.5, 30.0, 90.0, 3.0, 4.0, 0.2)
    X, Y, inside = F.sample_grid(g, 80)
    u, v, p = F.sample(lambda x, y: (x + 2 * y, -y, x * y), X, Y, inside)
    assert np.allclose(u[inside], X[inside] + 2 * Y[inside])
    assert np.allclose(p[inside], X[inside] * Y[inside])
    assert np.isnan(u[~inside]).all() and np.isnan(v[~inside]).all()


def test_the_config_is_checked():
    document = yaml.safe_load(open(F.ROOT / "bifurcation/fields_re10.yaml"))
    F.parse_config(document)
    del document["gallery"]["grid"]
    with pytest.raises(F.ConfigError):
        F.parse_config(document)
