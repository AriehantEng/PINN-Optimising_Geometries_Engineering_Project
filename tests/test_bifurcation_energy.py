"""The energy budget check of bifurcation networks (bifurcation/energy.py)."""

import jax.numpy as jnp
import numpy as np

from bifurcation import energy
from bifurcation import reference as br
from bifurcation.geometry import Geometry


def coarse(split=0.4, angle1=30.0, angle2=70.0):
    g = Geometry.from_split(split, angle1, angle2, 3.0, 8.0, 0.2)
    return g, br.solve_bifurcation(g, 10.0, "inertial", br.parabola_inlet, None, max_area=0.04,
                                   corner_levels=3, corner_radius=0.4)


def test_the_dissipation_quadrature_integrates_a_known_field_exactly():
    """u = (a y, b x): e:e = (a + b)^2 / 2 everywhere, so Phi = nu (a + b)^2 A,
    with A the polygon's area: the quadrature covers the domain exactly once."""
    g, solution = coarse()
    nu, a, b = 0.1, 0.7, -0.2

    def field(xy):
        return jnp.stack([a * xy[1], b * xy[0], 0.0 * xy[0]])

    phi = energy.dissipation(field, solution, nu)
    assert abs(phi / (nu * (a + b) ** 2 * g.area()) - 1.0) < 1e-12


def test_the_reference_passes_its_own_check():
    """The finite-element solution measured by the same budget: its dissipation
    on its own quadrature is its own, and its inconsistency is small."""
    g, solution = coarse()

    def field(xy):                                  # any field with nonzero dissipation
        return jnp.stack([xy[1], 0.0 * xy[0], 0.0 * xy[0]])

    row = energy.budget(field, solution, g, solution, 10.0, "inertial", 64)
    assert row["inconsistency_ref"] < 1e-3
    assert abs(row["drop_error"]) < 1e-12           # the same model on both sides
    assert np.isfinite(row["dissipation_ref"]) and row["dissipation_ref"] > 0.0
