"""Review checks on the design study (bifurcation/sensitivity.py)."""

import json
import math
import pathlib

import pytest

from bifurcation import sensitivity as S

ROOT = pathlib.Path(S.__file__).resolve().parent.parent
STUDY = json.loads((ROOT / "docs/bifurcation_re10-shares-all.json").read_text())["config"]


def test_a_symmetric_layout_at_equal_widths_has_equal_daughters():
    l1, l2 = S.centreline_lengths(S.make_geometry(STUDY, [60.0, 60.0], 0.5))
    assert abs(l1 - l2) < 1e-12


@pytest.mark.parametrize("angles", [[90.0, 90.0], [60.0, 120.0], [30.0, 150.0]])
def test_the_face_septum_leaves_both_daughters_their_outer_wall_length(angles):
    """At 180 degrees each daughter is a full strip from the cross-section
    through its outer corner, so its centreline is exactly L_d."""
    l1, l2 = S.centreline_lengths(S.make_geometry(STUDY, angles, 0.4))
    assert abs(l1 - STUDY["geometry"]["daughter_length"]) < 1e-12
    assert abs(l2 - STUDY["geometry"]["daughter_length"]) < 1e-12


def test_equal_lengths_give_the_murray_split():
    for share in (0.05, 0.3, 0.5, 0.8):
        s_l, _, _ = S.length_corrected_split(STUDY, [90.0, 90.0], share, 1e-12)
        assert abs(s_l - S.murray_split(share)) < 1e-12


def test_the_length_corrected_split_solves_its_own_relation():
    """s_L / (1 - s_L) = (L1/L2)^(1/4) sqrt(q / (1 - q)), with the lengths of the
    geometry at s_L itself."""
    s_l, l1, l2 = S.length_corrected_split(STUDY, [30.0, 90.0], 0.3, 1e-13)
    at_s_l = S.centreline_lengths(S.make_geometry(STUDY, [30.0, 90.0], s_l))
    assert max(abs(l1 - at_s_l[0]), abs(l2 - at_s_l[1])) < 1e-9
    ratio = (l1 / l2) ** 0.25 * math.sqrt(0.3 / 0.7)
    assert abs(s_l / (1 - s_l) - ratio) < 1e-9


def test_a_turned_daughter_has_the_longer_centreline():
    """The study fixes each daughter's outer wall: the more-turned daughter's
    entrance sits further back, so its centreline is longer."""
    l1, l2 = S.centreline_lengths(S.make_geometry(STUDY, [0.0, 150.0], 0.5))
    assert l2 > l1 + 1.0
    s_l, _, _ = S.length_corrected_split(STUDY, [0.0, 150.0], 0.5, 1e-12)
    assert s_l < 0.5 - 0.01                      # the longer daughter is widened


def test_with_equal_lengths_and_no_junction_loss_the_model_is_the_murray_split():
    s = S.model_split(STUDY, [90.0, 90.0], 0.3, 0.0, 0.0, 1e-9)
    assert abs(s - S.murray_split(0.3)) < 1e-6


def test_a_larger_loss_on_daughter_two_widens_daughter_two():
    """A.5: K2 > K1 at equal shares moves the optimum below the symmetric one."""
    base = S.model_split(STUDY, [90.0, 90.0], 0.5, 1.0, 1.0, 1e-9)
    turned = S.model_split(STUDY, [90.0, 90.0], 0.5, 1.0, 5.0, 1e-9)
    assert turned < base - 1e-4


def test_the_mesh_bound_recovers_a_known_argmin_shift():
    """J = (s - 0.5)^2 + 1 on the coarse mesh, and a fine-mesh error that grows
    linearly, e = c (s - 0.5): the fine argmin moves by -c / (J''/J) to first order."""
    c = 0.01

    def coarse(s):
        return (s - 0.5) ** 2 + 1.0

    row = {"grid": [{"split": s, "total_pressure_drop": coarse(s)} for s in (0.45, 0.5, 0.55)],
           "best_split": 0.5, "total_pressure_drop": coarse(0.5),
           "total_pressure_drop_fine": coarse(0.5) * (1 + c * 0.0)}
    out = S.mesh_bound(row, coarse(0.45) * (1 + c * -0.05), coarse(0.55) * (1 + c * 0.05), 0.001)
    assert abs(out["argmin_shift"] - (-c / 2.0)) < 1e-9
    assert abs(out["bound"] - (c / 2.0 + 0.001)) < 1e-9


def test_the_config_is_checked():
    import yaml
    document = yaml.safe_load((ROOT / "bifurcation/sensitivity_re10.yaml").read_text())
    S.parse_config(document)
    document["junction"]["fit_layouts"] = [[30.0, 90.0]]
    with pytest.raises(S.ConfigError, match="symmetric"):
        S.parse_config(document)


def test_a_path_with_only_friction_has_no_loss_coefficient():
    angles, share, split = [30.0, 90.0], 0.3, 0.45
    l1, l2 = S.centreline_lengths(S.make_geometry(STUDY, angles, split))
    nu, q0, w0 = 0.1, 4.0 / 3.0, 2.0
    parent = 12 * nu * q0 * STUDY["geometry"]["parent_length"] / w0**3
    measures = {"total_pressure_in": 10.0,
                "total_pressure_out1": 10.0 - parent - 12 * nu * share * q0 * l1 / (w0 * split) ** 3,
                "total_pressure_out2": 10.0 - parent
                - 12 * nu * (1 - share) * q0 * l2 / (w0 * (1 - split)) ** 3}
    k1, k2 = S.loss_coefficients(STUDY, angles, share, split, measures)
    assert abs(k1) < 1e-12 and abs(k2) < 1e-12
    measures["total_pressure_out2"] -= 0.5 * 2.0 * ((1 - share) * q0 / (w0 * (1 - split))) ** 2
    assert abs(S.loss_coefficients(STUDY, angles, share, split, measures)[1] - 2.0) < 1e-12


def test_the_reversal_share_is_bracketed_by_the_sampled_shares():
    assert S.reversal_share([0.1, 0.5, 0.9], [-0.01, -0.001, 0.002]) == [0.5, 0.9]
    assert S.reversal_share([0.1, 0.5, 0.9], [-0.01, -0.001, -0.002]) is None
    assert S.reversal_share([0.1, 0.5, 0.9], [0.01, -0.001, -0.002]) is None


def test_the_reversal_table_leaves_out_the_face_septum():
    """At 180 degrees s_L = s_M up to round-off, whose sign is noise."""
    def rec(angles, share, dm, dl, dk):
        return {"angles": angles, "share": share, "departure_murray": dm,
                "s_murray": 0.5, "s_length": 0.5 + dl, "s_model": 0.5 + dk}
    records = [rec([90.0, 90.0], 0.1, -1e-3, -1e-17, 0.0), rec([90.0, 90.0], 0.9, 1e-3, 1e-17, 0.0),
               rec([20.0, 40.0], 0.1, -2e-3, -2e-3, -2e-3), rec([20.0, 40.0], 0.9, 1e-3, -1e-4, 1e-4)]
    table = S.reversal_table(records)
    assert list(table) == ["20/40"]
    assert table["20/40"] == {"observed": [0.1, 0.9], "lengths": None, "model": [0.1, 0.9]}


def _strip_area(q, outer, inner):
    """Shoelace area of daughter 1's strip: outer corner, outlet, inner wall start."""
    pts = [q[outer], q["E1o"], q["E1i"], q[inner]]
    return 0.5 * abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1])))


@pytest.mark.parametrize("angles", [[30.0, 90.0], [0.0, 150.0], [45.0, 45.0]])
def test_the_centreline_is_the_mean_wall_length_and_the_area_over_the_width(angles):
    """The midline of a trapezoid: neither identity is an independent length."""
    g = S.make_geometry(STUDY, angles, 0.45)
    centre, inner = S.centreline_lengths(g), S.inner_wall_lengths(g)
    outer = STUDY["geometry"]["daughter_length"]
    for c, i in zip(centre, inner):
        assert abs(c - 0.5 * (outer + i)) < 1e-12
    q = S.corner_points(g.w1, g.w2, math.radians(g.angle1), math.radians(g.angle2),
                        g.parent_length, g.daughter_length)
    assert abs(_strip_area(q, "C", "T") / g.w1 - centre[0]) < 1e-12


def test_the_closed_length_is_the_shorter_wall():
    """0/150: the straight daughter is closed only downstream of the septum tip,
    the turned one over its whole outer wall."""
    g = S.make_geometry(STUDY, [0.0, 150.0], 0.5)
    closed, inner = S.closed_lengths(g), S.inner_wall_lengths(g)
    assert abs(closed[0] - inner[0]) < 1e-12 and inner[0] < 4.5
    assert closed[1] == STUDY["geometry"]["daughter_length"] < inner[1]
