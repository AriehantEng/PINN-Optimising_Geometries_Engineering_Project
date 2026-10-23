"""The Reynolds-number study's tools: the answer-key config, and the exporter's
history steps and convergence verdict (bifurcation/answer_key.py, re_export.py)."""

import copy

import pytest
import yaml

from bifurcation import answer_key as A, parametric as P, re_export as R

CONFIG = yaml.safe_load(open(A.__file__.replace("answer_key.py", "answer_keys_re.yaml")))


def test_the_answer_key_config_parses():
    A.parse_config(CONFIG)


@pytest.mark.parametrize("key", sorted(A.SCHEMA))
def test_a_missing_answer_key_key_is_refused(key):
    document = copy.deepcopy(CONFIG)
    del document[key]
    with pytest.raises(A.ConfigError, match=key):
        A.parse_config(document)


def test_a_continuation_above_its_reynolds_number_is_refused():
    document = copy.deepcopy(CONFIG)
    document["flows"][0]["continuation"] = [5.0]            # Re 0.1
    with pytest.raises(A.ConfigError, match="continuation"):
        A.parse_config(document)


def test_every_study_config_parses_and_names_its_run():
    import glob
    for path in sorted(glob.glob(str(P.HERE / "p_re*_corners_s*.yaml"))):
        c = P.load_config(path)
        tag = path.rsplit("/p_", 1)[1][:-len(".yaml")]
        assert c.name == f"parametric_{tag}"
        assert c.pressure_scaling == ("viscous" if c.reynolds <= 1.0 else "inertial")
        assert (c.fe_continuation == (25.0, 50.0)) == (c.reynolds == 100.0)


def _history(lm_heldout):
    rows = [{"phase": "adam", "iteration": 250, "train_loss": 1.0, "heldout_loss": 1.0,
             "seconds": 1.0},
            {"phase": "lbfgs", "iteration": 2, "train_loss": 0.5, "heldout_loss": 0.5,
             "seconds": 2.0}]
    rows += [{"phase": "lm", "iteration": k + 1, "train_loss": h, "heldout_loss": h,
              "heldout_average": None, "seconds": 3.0 + k} for k, h in enumerate(lm_heldout)]
    return rows


def test_history_steps_follow_the_schedule():
    c = P.load_config(P.HERE / "p_re0.1_3090_corners_s21.yaml")
    steps = [r["step"] for r in R.history_rows(_history([0.1]), c)]
    assert steps == [250, 3000 + 2 * 500, 3000 + 4 * 500 + 1]


def test_a_plateau_counts_as_plateaued_and_a_late_fall_does_not():
    flat = [1e-3] * 100 + [0.95e-3] * 50
    falling = [1e-3 * 0.99**k for k in range(150)]           # 22% lower over the last 25
    assert R.plateaued(_history(flat), {})["plateaued"]
    verdict = R.plateaued(_history(falling), {})
    assert not verdict["plateaued"] and verdict["still_falling"]
    assert not R.plateaued(_history([1e-3, float("nan")] * 20), {})["plateaued"]


def test_the_study_figures_draw_every_finished_run(tmp_path, monkeypatch):
    """The drop-error and training figures once cycled a six-colour list with
    zip and silently drew 6 of 13 runs; now each draws every run or refuses."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "docs" / "re_study" / "figures.py"
    spec = importlib.util.spec_from_file_location("re_study_figures", path)
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)
    finished = F.runs()
    assert len(finished) == len(list(F.HERE.glob("pinn_*.csv"))) - len(list(F.HERE.glob("pinn_*_loss.csv")))
    monkeypatch.setattr(F, "HERE", tmp_path)
    assert F.drop_error(finished) == len(finished)
    assert F.training(finished) == len(finished)


def _optimum(best, bracket, refused):
    return {"best_split": best, "refused": [{"split": s, "reason": "walls"} for s in refused],
            "fine_bracket": [{"split": s, "total_pressure_drop": j} for s, j in bracket]}


def test_an_optimum_against_a_refused_split_is_a_geometry_limit():
    # 30/120: J still falls at 0.2680 and 0.25, the bracket's left point, is refused
    o = _optimum(0.26803, [(0.25, None), (0.26803, 2.79), (0.35, 3.16)],
                 [0.1, 0.25, 0.26459, 0.26753])
    assert A.feasibility_limit(o) == 0.26753
    o = _optimum(0.7, [(0.65, 3.0), (0.7, 2.9), (0.75, None)], [0.75, 0.8])
    assert A.feasibility_limit(o) == 0.75


def test_an_interior_minimum_is_not_a_geometry_limit():
    assert A.feasibility_limit(_optimum(0.4848, [(0.45, 4.1), (0.4848, 4.03), (0.5, 4.05)],
                                        [])) is None
    # a refused split elsewhere does not make an interior minimum a limit
    assert A.feasibility_limit(_optimum(0.4848, [(0.45, 4.1), (0.4848, 4.03), (0.5, 4.05)],
                                        [0.1])) is None


# --------------------------------------------------------------------------- #
# Tier A (bifurcation/tier_a.py; docs/re_study/tierA/preregistration.md)
# --------------------------------------------------------------------------- #

def test_the_trained_range_is_s_L_plus_minus_015_inside_the_limits():
    from bifurcation import tier_a as T
    assert T.split_range(0.4843) == (0.3343, 0.6343)
    assert T.split_range(0.2) == (0.1, 0.35)              # clipped below
    assert T.split_range(0.82) == (0.67, 0.9)             # and above
    splits = T.evaluation_splits(0.1616, 0.4616)
    assert len(splits) == 7 and splits[0] == 0.1616 and splits[-1] == 0.4616


def test_s_L_needs_no_flow_solve_and_matches_the_answer_key():
    from bifurcation import tier_a as T
    s_l, l1, l2 = T.length_split((30.0, 90.0), 0.5)
    assert abs(s_l - 0.4843) < 5e-5                       # the answer key's s_L for 30/90
    assert s_l / (1 - s_l) == pytest.approx((l1 / l2) ** 0.25, rel=1e-9)


def test_a_share_away_from_half_joins_the_run_name_only_there():
    import types
    config = types.SimpleNamespace(angle1=30.0, angle2=90.0, reynolds=10.0, seed=21,
                                   outlet1_share=0.5)
    assert R.stem(config) == "pinn_30-90_Re10_seed21"     # the earlier runs keep their names
    config.outlet1_share = 0.2
    assert R.stem(config) == "pinn_30-90_q0.2_Re10_seed21"


def test_a_case_at_the_geometry_limit_or_outside_the_range_fails_screening(monkeypatch):
    from bifurcation import tier_a as T, design_study

    def fake(row):
        monkeypatch.setattr(design_study, "optimise", lambda task: row)
        return T.screen_case(((30.0, 90.0), 0.5, T.FLOWS[1]))

    interior = {"feasible": True, "best_split": 0.4848, "on_bound": False, "refused": [],
                "minimum_survives_refinement": True, "steady_bracket": True,
                "fine_bracket": [{"split": 0.45, "total_pressure_drop": 4.1},
                                 {"split": 0.4848, "total_pressure_drop": 4.03},
                                 {"split": 0.5, "total_pressure_drop": 4.05}]}
    assert fake(interior)["passes"]
    near_edge = {**interior, "best_split": 0.36}           # range [0.3343, 0.6343]
    assert not fake(near_edge)["passes"]
    limit = {**interior, "refused": [{"split": 0.45, "reason": "walls"}],
             "fine_bracket": [{"split": 0.45, "total_pressure_drop": None}] + interior["fine_bracket"][1:]}
    r = fake(limit)
    assert not r["passes"] and "geometry's limit" in r["reasons"][0]


def test_answer_key_files_name_the_share_away_from_half(tmp_path):
    """Four 30/90 shares in one directory once overwrote each other's files: the
    share was missing from the CSV names, and with_suffix cut 'q0.2' at its dot."""
    rows = [{"split": 0.3, "J": 1.0}]
    result = {"cases": [{"key": "30-90_Re10", "share": q, "reference": rows} for q in (0.2, 0.8, 0.5)]}
    A.write_csvs(result, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "fe_30-90_Re10_q0.2_reference.csv", "fe_30-90_Re10_q0.8_reference.csv",
        "fe_30-90_Re10_reference.csv"]


def test_width_scaled_sampling_puts_the_same_points_across_every_channel():
    """sampling.density = width: density ~ 1/w^2, so a narrow daughter gets as
    many points across as the parent; the total is unchanged."""
    import math
    import numpy as np
    from bifurcation import flow
    from bifurcation.geometry import Geometry
    g = Geometry.from_split(0.1616, 30.0, 90.0, 3.0, 4.0, 0.2)
    rng = np.random.default_rng(3)
    pts = flow.width_scaled_points(rng, g, 922, 0.02)
    assert len(pts) == 922
    across = []
    for path, w in flow.channel_regions(g):
        v = path.vertices
        area = 0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], -1)) - np.dot(v[:, 1], np.roll(v[:, 0], -1)))
        across.append(w / math.sqrt(area / path.contains_points(pts).sum()))
    assert min(across) > 5.8 and max(across) / min(across) < 1.1
    uniform = flow.uniform_points(np.random.default_rng(3), g, 922, 0.02)
    path, w = flow.channel_regions(g)[0]
    assert path.contains_points(uniform).sum() < path.contains_points(pts).sum() / 3
