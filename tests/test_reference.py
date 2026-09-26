"""The finite-element reference solution (developing_flow/reference.py).

A reference is only worth measuring the PINN against once it has been checked
itself: exact on Poiseuille flow, which Q2/Q1 elements contain; third-order
velocity convergence on Kovasznay flow, an exact Navier-Stokes solution whose
convective term is nonzero; and a fast evaluator that agrees with scikit-fem's
own point evaluation.
"""

import math

import numpy as np
import pytest

from developing_flow import pinn, reference


@pytest.mark.parametrize("scaling", ["inertial", "viscous"])
def test_poiseuille_flow_is_reproduced_to_round_off(scaling):
    """With the m = 1 inlet the exact solution is u = 1 - y^2, v = 0 and p
    linear in x, all inside Q2/Q1, so only round-off may separate them. This
    also checks the natural outlet condition: it must give p(L) = 0."""
    reynolds, length = 10.0, 10.0
    solution = reference.solve_channel(reynolds, scaling, 1, length, 20, 6, 3.0)
    x, y = np.meshgrid(np.linspace(0.0, length, 57), np.linspace(-1.0, 1.0, 31), indexing="ij")
    u, v, p = solution(x, y)
    _, viscous = pinn.scaling_coefficients(reynolds, scaling)
    assert np.max(np.abs(u - (1.0 - y**2))) < 1e-12
    assert np.max(np.abs(v)) < 1e-12
    assert np.max(np.abs(p - 2.0 * viscous * (length - x))) < 1e-12


def test_kovasznay_flow_converges_at_the_rates_q2_q1_theory_predicts():
    """Velocity error falls 8x per halving of h (third order), pressure at
    least 4x (second order). Kovasznay flow has a nonzero convective term, so
    the Newton linearisation is exercised, and a traction outlet."""
    reynolds = 40.0
    x, y = np.meshgrid(np.linspace(-0.5, 1.0, 151), np.linspace(-0.5, 1.5, 201), indexing="ij")
    velocity, pressure = [], []
    for n in (8, 16, 32):
        solution, (ue, ve, pe) = reference.solve_kovasznay(reynolds, n)
        u, v, p = solution(x, y)
        velocity.append(math.sqrt(np.mean((u - ue(x, y)) ** 2 + (v - ve(x, y)) ** 2)))
        pressure.append(math.sqrt(np.mean((p - pe(x, y)) ** 2)))
    for coarse, fine in zip(velocity, velocity[1:]):
        assert math.log2(coarse / fine) > 2.8
    for coarse, fine in zip(pressure, pressure[1:]):
        assert math.log2(coarse / fine) > 1.8
    assert velocity[-1] < 5e-4


def test_the_tensor_evaluation_equals_scikit_fems_point_evaluation():
    solution = reference.solve_channel(10.0, "inertial", 8, 10.0, 16, 6, 4.0)
    rng = np.random.default_rng(0)
    points = np.stack([rng.uniform(0.0, 10.0, 400), rng.uniform(-1.0, 1.0, 400)])
    points[:, :4] = [[0.0, 10.0, 0.0, 10.0], [-1.0, 1.0, 1.0, -1.0]]
    u, v, p = solution(points[0], points[1])
    (u_dofs, scalar), (v_dofs, _) = solution.basis_u.split(solution.U)
    probe = scalar.probes(points)
    assert np.max(np.abs(u - probe @ u_dofs)) < 1e-13
    assert np.max(np.abs(v - probe @ v_dofs)) < 1e-13
    assert np.max(np.abs(p - solution.basis_p.probes(points) @ solution.P)) < 1e-13


def test_the_boundary_data_hold_on_the_mesh():
    solution = reference.solve_channel(10.0, "inertial", 8, 10.0, 16, 6, 4.0)
    y = np.linspace(-1.0, 1.0, 13)  # only mesh-independent points: Q2 interpolates the inlet
    x = np.linspace(0.0, 10.0, 11)
    for wall in (-1.0, 1.0):
        u, v, _ = solution(x, np.full_like(x, wall))
        assert np.all(u == 0.0) and np.all(v == 0.0)
    _, v, _ = solution(np.zeros_like(y), y)
    assert np.all(v == 0.0)
    nodes = reference.y_nodes(6)
    u, _, _ = solution(np.zeros_like(nodes), nodes)
    assert np.max(np.abs(u - pinn.inlet_profile(nodes, 8))) < 1e-14


def test_mesh_nodes_span_the_domain_and_crowd_the_inlet_and_walls():
    x = reference.x_nodes(50, 10.0, 5.0)
    assert x[0] == 0.0 and abs(x[-1] - 10.0) < 1e-14 and np.all(np.diff(x) > 0)
    assert np.diff(x)[0] < np.diff(x)[-1] / 50
    assert np.array_equal(reference.x_nodes(4, 8.0, 0.0), np.linspace(0.0, 8.0, 5))
    y = reference.y_nodes(12)
    assert y[0] == -1.0 and y[-1] == 1.0 and np.all(np.diff(y) > 0)
    assert np.diff(y)[0] < np.diff(y)[6] / 5
