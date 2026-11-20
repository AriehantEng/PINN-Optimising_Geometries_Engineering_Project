"""The tight re-search's measurements (bifurcation/tight_search.py), on
synthetic data: no finite-element solve and no network."""

import math

import numpy as np
import pytest

from bifurcation import tight_search as T


def test_the_parabola_finds_the_vertex_curvature_and_jitter_it_was_built_with():
    s = 0.48 + 0.0005 * np.arange(-4, 5)
    jitter = 1e-6 * np.array([1, -1, 1, -1, 1, -1, 1, -1, 1])   # rms 1e-6 exactly
    j = 4.0 * (1.0 + 17.0 * (s - 0.4815) ** 2) + 4.0 * jitter
    fit = T.parabola(list(zip(s, j)), 0.48)
    assert abs(fit["vertex"] - 0.4815) < 2e-5
    assert abs(fit["curvature"] - 34.0) / 34.0 < 0.02
    assert 0.5e-6 < fit["jitter"] < 1.1e-6                       # the fit absorbs a little
    c2 = 0.5 * fit["curvature"] * 4.0
    assert fit["noise_half_width"] == pytest.approx(
        math.sqrt(2.0 * fit["jitter"] * 4.0 * (1.0 + 17.0 * 0.0015 ** 2) / c2), rel=0.01)
    assert fit["points"] == 9 and fit["vertex_spread"] > 0.0


def test_a_parabola_without_jitter_has_no_noise_and_no_spread():
    s = 0.3 + 0.001 * np.arange(-3, 4)
    fit = T.parabola(list(zip(s, 2.0 + 5.0 * (s - 0.3012) ** 2)), 0.3)
    assert abs(fit["vertex"] - 0.3012) < 1e-12
    assert fit["noise_half_width"] < 1e-6 and fit["vertex_spread"] < 1e-10


def test_the_mesh_shift_is_minus_the_error_slope_over_the_curvature():
    xs = [0.47, 0.48, 0.49]
    check = [4.0, 3.9, 4.1]
    ref = [c * (1.0 + 2e-4 + 3e-4 * (x - 0.48)) for c, x in zip(check, xs)]
    out = T.mesh_shift(ref, check, xs, 34.0)
    assert out["error_slope"] == pytest.approx(3e-4)
    assert out["shift"] == pytest.approx(-3e-4 / 34.0)


@pytest.mark.parametrize("error, eps, expected", [
    (0.0004, 0.0003, "pass"), (-0.0004, 0.0003, "pass"),
    (0.0018, 0.0003, "borderline"), (-0.0022, 0.0003, "borderline"),
    (0.0026, 0.0003, "miss"), (-0.0050, 0.0003, "miss"),
    (0.0019, 0.0, "pass"), (0.0021, 0.0, "miss"),
])
def test_the_verdict_marks_only_what_lies_within_eps_of_the_boundary(error, eps, expected):
    assert T.verdict(error, eps) == expected


@pytest.mark.parametrize("name, label", [
    ("parametric_re10_3090_corners_s22", "q = 0.5 (earlier)"),
    ("parametric_re10_3090_equal_corners", "q = 0.5 (earlier)"),
    ("parametric_re100_30120_corners_s21", "q = 0.5 (earlier)"),
    ("parametric_re10_3090_q0.65_corners_s21", "Tier A"),
    ("parametric_re0.1_0120_q0.5_corners_s21", "Tier A"),
    ("parametric_re10_1515_q0.2_corners_s21", "Tier A"),
    ("parametric_re0.1_3090_q0.5_cut0.15_corners_s21", "controls"),
    ("parametric_re10_3090_q0.35_cut0.4_corners_s21", "controls"),
    ("parametric_re10_3090_q0.2_corners_s22", "controls"),
    ("parametric_re10_3090_q0.2_width_corners_s21", "amendment 3"),
    ("parametric_re0.1_3090_q0.5_cut0.15_lm400_corners_s22", "amendment 3"),
    ("parametric_re10_3090_corners_wide", "earlier experiment"),
    ("parametric_re10_3090_equal", "earlier experiment"),
])
def test_each_run_lands_in_its_table(name, label):
    assert T.family(name) == label


def test_the_bracket_is_the_best_evaluation_split_and_its_neighbours():
    rows = [{"split": s, "reference_drop": j} for s, j in
            ((0.35, 5.0), (0.40, 4.5), (0.45, 4.1), (0.50, 4.2), (0.55, 4.8))]
    assert T.bracket_of({"splits": rows[::-1]}) == (0.40, 0.50)
    rows[0]["reference_drop"] = 1.0
    assert T.bracket_of({"splits": rows}) is None                # best at the grid's end
