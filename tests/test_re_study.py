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


def test_a_plateau_counts_as_converged_and_a_late_fall_does_not():
    flat = [1e-3] * 100 + [0.95e-3] * 50
    falling = [1e-3 * 0.99**k for k in range(150)]           # 22% lower over the last 25
    assert R.converged(_history(flat), {})["converged"]
    verdict = R.converged(_history(falling), {})
    assert not verdict["converged"] and verdict["still_falling"]
    assert not R.converged(_history([1e-3, float("nan")] * 20), {})["converged"]
