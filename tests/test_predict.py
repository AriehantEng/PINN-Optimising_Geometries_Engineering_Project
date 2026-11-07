"""Using a trained model (src/predict.py).

The failure this guards against is the project's defining one in a new place:
`load_run` rebuilds the parameter pytree's STRUCTURE from the config and pours
the saved flat vector into it. If that structure differed at all from the one
that was saved -- a renamed key, a reordered layer list, a changed default --
`unflatten` would still succeed and return a model that produces smooth,
plausible, completely wrong fields. Nothing would raise.

So the round-trip is asserted to be BITWISE identical, not merely close.
"""

import json

import numpy as np
import pytest
import yaml

from src.evaluation.accuracy import evaluate
from src.physics import analytic
from src.predict import load_run
from src.run import execute
from tests.test_accuracy import SHORT_SCHEDULE, _apply


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """Train a deliberately tiny model and save it, once for this module."""
    document = _apply(
        yaml.safe_load(open("experiments/caseA_re10.yaml", encoding="utf-8")),
        SHORT_SCHEDULE,
    )
    document["name"] = "predict_roundtrip"
    directory = tmp_path_factory.mktemp("runs") / "predict_roundtrip"
    config_path = directory.parent / "predict_roundtrip.yaml"
    directory.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    execute(str(config_path), out=str(directory), verbose=False)
    return directory


def test_round_trip_is_bitwise_identical(tiny_run):
    """The loaded model must reproduce the saved one exactly, not approximately."""
    solution = load_run(tiny_run)

    # Rebuild what the run itself evaluated, and compare on the same grid.
    stored = json.loads((tiny_run / "metrics.json").read_text())
    metrics = evaluate(solution.config, solution.model, solution.params)

    for key in ("u_relative_l2", "v_max_abs", "flux_conservation_relative_error",
                "p_relative_l2"):
        if key in stored:
            assert metrics[key] == stored[key], (
                f"{key} differs after a save/load round trip: "
                f"{metrics[key]!r} vs {stored[key]!r}. The parameter pytree was "
                "not reconstructed identically."
            )


def test_loaded_model_matches_the_analytic_solution(tiny_run):
    """A weaker but independent check: the loaded field is physically right."""
    solution = load_run(tiny_run)
    y, u, v = solution.profile(x=0.5 * solution.domain_length)
    np.testing.assert_allclose(u, analytic.u_nd(y), rtol=0.0, atol=5e-2)
    assert np.max(np.abs(v)) < 5e-2


def test_flux_is_recovered_from_the_loaded_model(tiny_run):
    solution = load_run(tiny_run)
    flux = solution.flux(0.5 * solution.domain_length)
    np.testing.assert_allclose(flux, analytic.FLUX_ND, rtol=5e-2)


def test_evaluating_outside_the_trained_domain_raises(tiny_run):
    """A PINN is fitted, not solved: outside its domain it means nothing."""
    solution = load_run(tiny_run)
    length = solution.domain_length

    for point in ((length * 1.5, 0.0), (-0.5, 0.0), (1.0, 1.5), (1.0, -1.2)):
        with pytest.raises(ValueError, match="outside the trained domain"):
            solution.velocity(*point)

    # ... unless you say you mean it.
    u, v = solution.velocity(length * 1.5, 0.0, allow_extrapolation=True)
    assert np.isfinite(u) and np.isfinite(v)


def test_domain_boundaries_themselves_are_allowed(tiny_run):
    """The walls and the inlet/outlet planes are in the domain, not outside it."""
    solution = load_run(tiny_run)
    for point in ((0.0, 0.0), (solution.domain_length, 0.0), (1.0, 1.0), (1.0, -1.0)):
        u, v = solution.velocity(*point)
        assert np.isfinite(u) and np.isfinite(v)
    # No-slip still holds exactly on a loaded model.
    assert solution.velocity(1.0, 1.0) == (0.0, 0.0)
    assert solution.velocity(1.0, -1.0) == (0.0, 0.0)


def test_missing_run_directory_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="has this run finished"):
        load_run(tmp_path / "does_not_exist")


def test_mismatched_config_and_weights_raise(tiny_run, tmp_path):
    """A config that builds a different-sized model must not load silently."""
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(tiny_run, broken)
    document = yaml.safe_load((broken / "config.yaml").read_text())
    document["model"]["width"] = document["model"]["width"] * 2
    (broken / "config.yaml").write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="different models"):
        load_run(broken)


def test_vectorised_and_scalar_evaluation_agree(tiny_run):
    """Batched and single-point evaluation must agree to rounding.

    Not bitwise: XLA associates a batched matmul differently from a single
    matrix-vector product, so the two paths differ by about one ulp (measured
    1.1e-16). That is arithmetic, not a defect. A few ulp is the honest bound;
    anything larger would mean the batching is doing something else.
    """
    solution = load_run(tiny_run)
    xs = np.linspace(0.1, solution.domain_length - 0.1, 7)
    ys = np.linspace(-0.8, 0.8, 7)
    u_vec, v_vec = solution.velocity(xs, ys)
    for i, (x, y) in enumerate(zip(xs, ys)):
        u, v = solution.velocity(float(x), float(y))
        np.testing.assert_allclose(u, u_vec[i], rtol=8 * np.finfo(float).eps, atol=1e-15)
        np.testing.assert_allclose(v, v_vec[i], rtol=8 * np.finfo(float).eps, atol=1e-15)


def test_grid_shape_and_bounds(tiny_run):
    solution = load_run(tiny_run)
    x, y, u, v = solution.grid(21, 11)
    assert u.shape == (21, 11) and v.shape == (21, 11)
    assert x[0] == 0.0 and x[-1] == solution.domain_length
    assert y[0] == -1.0 and y[-1] == 1.0
    # Walls are exactly no-slip on every station.
    assert np.all(u[:, 0] == 0.0) and np.all(u[:, -1] == 0.0)


def test_every_stored_run_still_loads():
    """A finished run must never become unloadable.

    Making `physics.convective_coefficient` required locked EVERY run
    directory out of `load_run`, because their configs predate the key. That
    is a real cost: no figure could be regenerated from any completed work,
    and the failure was invisible until someone tried.

    The no-silent-defaults rule is right for a run being launched. This test
    is the other half of it: when a key becomes required, the stored runs must
    be migrated in the same change (see `src/migrate_configs.py`).
    """
    import pathlib

    from src.predict import load_run

    runs = pathlib.Path("runs")
    if not runs.is_dir():
        pytest.skip("no runs/ directory in this checkout")
    directories = [d for d in sorted(runs.iterdir())
                   if d.is_dir() and (d / "config.yaml").is_file()
                   and (d / "params.npz").is_file()]
    if not directories:
        pytest.skip("no finished runs to check")

    broken = []
    for directory in directories:
        try:
            load_run(str(directory))
        except Exception as error:  # noqa: BLE001 - any failure is the finding
            broken.append(f"{directory.name}: {type(error).__name__}: {error}")
    assert not broken, (
        "stored runs no longer load:\n  " + "\n  ".join(broken)
        + "\n\nIf a config key was just made required, migrate the stored runs "
          "in the same change: python -m src.migrate_configs --write"
    )
