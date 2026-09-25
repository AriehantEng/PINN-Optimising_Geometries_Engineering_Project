"""The standalone developing-flow PINN (developing_flow/pinn.py).

Structural checks that must hold for ANY network parameters, residuals checked
against hand-derived values (including the inertial terms, which the fully
developed case can never exercise because they vanish there), and one tiny
end-to-end run.
"""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from jax.flatten_util import ravel_pytree

from developing_flow import pinn

CONFIG_PATH = "developing_flow/re10.yaml"


def base_document():
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tiny_document():
    document = base_document()
    document["name"] = "tiny"
    document["model"] = {"width": 16, "depth": 2}
    # RAR on, so the end-to-end run exercises the adaptive path too.
    document["sampling"].update(n_interior=256, n_inlet=32, n_outlet=32, n_validation=256,
                                rar_points=32, rar_pool=512)
    document["training"] = {
        "adam": {"steps": 60, "learning_rate": 1.0e-3, "decay_rate": 0.9,
                 "decay_steps": 1000, "resample_every": 20},
        "lbfgs": {"cycles": 2, "iterations_per_cycle": 20, "gtol": 1.0e-14},
        "check_every": 10,
    }
    document["evaluation"] = {"nx": 21, "ny": 11, "profile_stations": [0.0, 1.0],
                              "n_flux_stations": 5}
    return document


@pytest.fixture(scope="module")
def config():
    return pinn.load_config(CONFIG_PATH)


def random_params(config, seed):
    return pinn.init_network(jax.random.PRNGKey(seed), config.width, config.depth)


# -- configuration ---------------------------------------------------------- #


def test_shipped_config_loads(config):
    assert config.reynolds == 10.0
    assert config.inlet_sharpness == 8


def test_missing_key_raises_rather_than_defaulting():
    document = base_document()
    del document["physics"]["reynolds"]
    with pytest.raises(pinn.ConfigError, match="physics.reynolds"):
        pinn.parse_config(document)


def test_unknown_key_is_refused():
    document = base_document()
    document["physics"]["viscosity"] = 1.0e-3
    with pytest.raises(pinn.ConfigError, match="unknown"):
        pinn.parse_config(document)


def test_sharpness_one_is_refused_because_it_is_already_developed():
    document = base_document()
    document["physics"]["inlet_sharpness"] = 1
    with pytest.raises(pinn.ConfigError, match="developed"):
        pinn.parse_config(document)


def test_a_boolean_is_not_accepted_as_an_integer():
    document = base_document()
    document["seed"] = True
    with pytest.raises(pinn.ConfigError, match="seed"):
        pinn.parse_config(document)


# -- the ansatz holds for any parameters ------------------------------------ #


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_slip_is_exactly_zero_on_both_walls(config, seed):
    params = random_params(config, seed)
    field = pinn.make_field(config.length)
    x = np.linspace(0.0, config.length, 17)
    for wall in (-1.0, 1.0):
        u, v, _ = pinn.predict(field, params, np.stack([x, np.full_like(x, wall)], axis=-1))
        assert np.all(u == 0.0) and np.all(v == 0.0)


@pytest.mark.parametrize("seed", [0, 1])
def test_pressure_gauge_is_exactly_zero_at_the_outlet(config, seed):
    params = random_params(config, seed)
    field = pinn.make_field(config.length)
    y = np.linspace(-1.0, 1.0, 17)
    _, _, p = pinn.predict(field, params, np.stack([np.full_like(y, config.length), y], axis=-1))
    assert np.all(p == 0.0)


# -- the inlet ---------------------------------------------------------------- #


@pytest.mark.parametrize("m", [2, 4, 8, 16])
def test_every_inlet_carries_the_developed_flux_and_vanishes_at_the_walls(m):
    nodes, weights = np.polynomial.legendre.leggauss(64)
    flux = float(pinn.inlet_profile(nodes, m) @ weights)
    assert abs(flux - pinn.INLET_FLUX) < 1e-13
    assert pinn.inlet_profile(np.array([-1.0, 1.0]), m).tolist() == [0.0, 0.0]


def test_the_inlet_is_not_the_developed_profile():
    y = np.linspace(-1.0, 1.0, 101)
    difference = pinn.inlet_profile(y, 8) - pinn.developed_profile(y, pinn.INLET_FLUX)
    assert np.max(np.abs(difference)) > 0.2


# -- residuals against hand-derived values ------------------------------------ #


@pytest.mark.parametrize("scaling, reynolds", [("inertial", 10.0), ("inertial", 100.0), ("viscous", 1.0)])
def test_residuals_vanish_on_the_exact_developed_solution(scaling, reynolds):
    """u = 1 - y^2, v = 0 and the matching linear pressure satisfy the full
    equations, inertia included (it vanishes identically there)."""
    _, viscous = pinn.scaling_coefficients(reynolds, scaling)

    def exact(xy):
        x, y = xy[0], xy[1]
        return jnp.stack([1.0 - y**2, 0.0 * x, -2.0 * viscous * x + 3.0])

    for point in ([0.3, -0.7], [4.0, 0.1], [9.5, 0.95]):
        d = pinn.point_derivatives(exact, jnp.array(point))
        assert abs(float(pinn.momentum_x(d, reynolds, scaling))) < 1e-13
        assert abs(float(pinn.momentum_y(d, reynolds, scaling))) < 1e-13
        assert abs(float(pinn.continuity(d))) < 1e-13


@pytest.mark.parametrize("scaling", ["inertial", "viscous"])
def test_x_inertia_matches_a_hand_derivation(scaling):
    """u = x (1 - y^2), v = 0, p = 0:
        u u_x + v u_y = x (1 - y^2)^2,  lap u = -2x,  continuity = 1 - y^2."""
    reynolds = 7.0
    inertia, viscous = pinn.scaling_coefficients(reynolds, scaling)

    def field(xy):
        x, y = xy[0], xy[1]
        return jnp.stack([x * (1.0 - y**2), 0.0 * x, 0.0 * x])

    for x, y in ((0.5, 0.2), (2.0, -0.6), (3.3, 0.9)):
        d = pinn.point_derivatives(field, jnp.array([x, y]))
        expected = inertia * x * (1.0 - y**2) ** 2 + viscous * 2.0 * x
        assert abs(float(pinn.momentum_x(d, reynolds, scaling)) - expected) < 1e-12
        assert abs(float(pinn.continuity(d)) - (1.0 - y**2)) < 1e-13


def test_y_inertia_matches_a_hand_derivation():
    """u = y, v = x y, p = 0:  u v_x + v v_y = y^2 + x^2 y,  lap v = 0."""

    def field(xy):
        x, y = xy[0], xy[1]
        return jnp.stack([y, x * y, 0.0 * x])

    for x, y in ((0.5, 0.2), (2.0, -0.6)):
        d = pinn.point_derivatives(field, jnp.array([x, y]))
        assert abs(float(pinn.momentum_y(d, 10.0, "inertial")) - (y**2 + x**2 * y)) < 1e-12


# -- the fast batched derivatives equal the single-point reference ----------- #


def test_batched_derivatives_match_the_point_reference(config):
    params = random_params(config, 3)
    field = pinn.make_field(config.length)
    f = lambda xy: field(params, xy)  # noqa: E731
    points = pinn.sample_batch(jax.random.PRNGKey(5), config, 32).interior
    batched = pinn.batched_derivatives(f, points)
    for i in range(0, 32, 7):
        reference = pinn.point_derivatives(f, points[i])
        for name in pinn.Derivatives._fields:
            a, b = float(getattr(batched, name)[i]), float(getattr(reference, name))
            assert abs(a - b) <= 1e-12 * max(1.0, abs(b)), name


# -- sampling --------------------------------------------------------------- #


def test_samples_lie_in_the_domain_and_are_a_pure_function_of_the_key(config):
    key = jax.random.PRNGKey(11)
    first = pinn.sample_batch(key, config, 1000)
    again = pinn.sample_batch(key, config, 1000)
    other = pinn.sample_batch(jax.random.PRNGKey(12), config, 1000)
    x, y = np.asarray(first.interior[:, 0]), np.asarray(first.interior[:, 1])
    assert x.min() >= 0.0 and x.max() <= config.length
    assert y.min() >= -1.0 and y.max() <= 1.0
    assert np.all(np.asarray(first.inlet[:, 0]) == 0.0)
    assert np.all(np.asarray(first.outlet[:, 0]) == config.length)
    assert np.array_equal(np.asarray(first.interior), np.asarray(again.interior))
    assert not np.array_equal(np.asarray(first.interior), np.asarray(other.interior))


def test_focus_points_crowd_the_inlet(config):
    points = np.asarray(pinn.sample_batch(jax.random.PRNGKey(0), config, 4000).interior)
    near_inlet = np.mean(points[:, 0] < 1.0)
    uniform_share = 1.0 / config.length
    assert near_inlet > 2.0 * uniform_share


def test_held_out_points_are_a_different_draw_from_the_first_training_draw(config):
    _, sample_key, heldout_key = pinn.root_keys(config.seed)
    heldout = pinn.sample_batch(heldout_key, config, 256).interior
    _, draw = jax.random.split(sample_key)
    training = pinn.sample_batch(draw, config, 256).interior
    assert not np.any(np.all(np.isclose(np.asarray(heldout)[:, None, :],
                                        np.asarray(training)[None, :, :]), axis=-1))


# -- the loss ----------------------------------------------------------------- #


def test_total_loss_is_the_weighted_sum_of_exactly_the_five_terms():
    terms = {name: jnp.asarray(float(i + 1)) for i, name in enumerate(pinn.LOSS_TERMS)}
    weights = {name: jnp.asarray(1.0) for name in pinn.LOSS_TERMS}
    assert float(pinn.total_loss(terms, weights)) == 15.0
    weights["inlet"] = jnp.asarray(0.0)
    assert float(pinn.total_loss(terms, weights)) == 11.0


# -- end to end ----------------------------------------------------------------- #


@pytest.mark.slow
def test_a_tiny_run_saves_the_parameters_training_selected(tmp_path):
    """The saved model must be the one train() SELECTED, not merely whatever
    params.npz contains. train() is deterministic, so a second training of the
    same config must reproduce the saved parameters bitwise."""
    import json
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(tiny_document()), encoding="utf-8")
    directory = pinn.run(config_path, out=tmp_path / "run", verbose=False)

    for name in ("config.yaml", "params.npz", "history.csv", "metrics.json",
                 "profiles.png", "fields.png", "diagnostics.png"):
        assert (directory / name).is_file(), name
    assert not (directory / "checkpoint.npz").exists(), "checkpoint left behind after a finished run"

    def refuse(token):
        raise AssertionError(f"metrics.json contains the non-JSON token {token}")
    metrics = json.loads((directory / "metrics.json").read_text(), parse_constant=refuse)
    for key in ("entrance_length_correlation_flat_inlet", "loss_heldout", "flux_at_outlet"):
        assert np.isfinite(metrics[key])

    config = pinn.parse_config(copy.deepcopy(tiny_document()))
    selected, _, info = pinn.train(config, verbose=False)
    saved = np.load(directory / "params.npz")["flat"]
    assert np.array_equal(saved, np.asarray(ravel_pytree(selected)[0]))
    assert metrics["loss_heldout"] == info["selected_heldout_loss"]

    model = pinn.load_run(directory)
    x, y = np.array([0.0, 0.5, 2.0]), np.array([0.0, 0.3, -0.9])
    expected = pinn.predict(pinn.make_field(config.length), selected, np.stack([x, y], axis=-1))
    for a, b in zip(model(x, y), expected):
        assert np.array_equal(a, b)
    X, Y = np.meshgrid(np.linspace(0.0, 1.0, 4), np.linspace(-0.5, 0.5, 3))
    assert all(part.shape == X.shape for part in model(X, Y))
    with pytest.raises(ValueError, match="points must lie"):
        model(np.array([config.length + 1.0]), np.array([0.0]))

    with pytest.raises(FileExistsError, match="never writes over"):
        pinn.run(config_path, out=directory, verbose=False)


def test_an_unfinished_run_does_not_load(tmp_path):
    run = tmp_path / "interrupted"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(tiny_document()), encoding="utf-8")
    (run / "checkpoint.npz").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="interrupted"):
        pinn.load_run(run)


def test_a_non_finite_loss_stops_training_instead_of_saving_the_initial_network(monkeypatch):
    config = pinn.parse_config(tiny_document())
    monkeypatch.setattr(pinn, "total_loss", lambda terms, weights: jnp.nan * terms["inlet"])
    with pytest.raises(FloatingPointError, match="non-finite"):
        pinn.train(config, verbose=False)


def test_training_never_draws_from_the_held_out_key(monkeypatch):
    """Every sample_batch call train() makes is recorded. The held-out key must
    be used exactly once, for the held-out set, and never for training."""
    config = pinn.parse_config(tiny_document())
    _, _, heldout_key = pinn.root_keys(config.seed)
    calls = []
    original = pinn.sample_batch

    def recording(key, cfg, n):
        calls.append((np.asarray(jax.random.key_data(key) if hasattr(jax.random, "key_data")
                                 else key).tobytes(), n))
        return original(key, cfg, n)

    monkeypatch.setattr(pinn, "sample_batch", recording)
    pinn.train(config, verbose=False)
    held = np.asarray(heldout_key).tobytes()
    uses = [n for key, n in calls if key == held]
    assert uses == [config.n_validation]
    expected_draws = (config.adam_steps // config.adam_resample_every) + config.lbfgs_cycles
    assert len(calls) == 1 + expected_draws


# -- the loss terms and the measurements, on fields known exactly ------------- #


def _exact_developed(config):
    _, viscous = pinn.scaling_coefficients(config.reynolds, config.pressure_scaling)
    L = config.length

    def field(params, xy):
        x, y = xy[0], xy[1]
        return jnp.stack([1.0 - y**2, 0.0 * x, 2.0 * viscous * (L - x)])
    return field


def test_loss_terms_on_the_exact_developed_field(config):
    """Interior and outlet terms vanish; the inlet term is exactly the mismatch
    between the parabola and the prescribed inlet, u first, v second."""
    field = _exact_developed(config)
    batch = pinn.sample_batch(jax.random.PRNGKey(4), config, 512)
    terms = pinn.make_loss_terms(config, field)(None, batch)
    for name in ("momentum_x", "momentum_y", "continuity", "outlet"):
        assert float(terms[name]) < 1e-26, name
    y_in = np.asarray(batch.inlet[:, 1])
    expected = np.mean((1.0 - y_in**2 - pinn.inlet_profile(y_in, config.inlet_sharpness)) ** 2)
    assert abs(float(terms["inlet"]) - expected) <= 1e-13 * expected


def test_the_inlet_term_sees_v_separately_from_u(config):
    """A field that matches the inlet in u but carries v = 1 must score exactly 1."""
    def field(params, xy):
        return jnp.stack([pinn.inlet_profile(xy[1], config.inlet_sharpness), 1.0 + 0.0 * xy[0], 0.0 * xy[0]])
    batch = pinn.sample_batch(jax.random.PRNGKey(5), config, 64)
    assert abs(float(pinn.make_loss_terms(config, field)(None, batch)["inlet"]) - 1.0) < 1e-14


def test_measurements_on_the_exact_developed_field(config):
    m, _ = pinn.measure_solution(config, _exact_developed(config), None)
    assert abs(m["flux_at_inlet_station"] - pinn.INLET_FLUX) < 1e-12
    assert m["mass_drift_from_first_station"] < 1e-12
    assert m["outlet_profile_relative_l2"] < 1e-12
    assert m["entrance_length"] == 0.0 and m["entrance_length_reliable"]
    assert abs(m["outlet_dp_dx"] - m["developed_dp_dx_expected"]) < 1e-12
    for name in ("residual_momentum_x_max", "residual_momentum_y_max", "residual_continuity_max"):
        assert m[name] < 1e-12, name


def test_a_drifting_flux_marks_the_entrance_length_unreliable(config):
    def field(params, xy):
        x, y = xy[0], xy[1]
        return jnp.stack([(1.0 + 0.01 * x / config.length) * (1.0 - y**2), 0.0 * x, 0.0 * x])
    m, _ = pinn.measure_solution(config, field, None)
    assert m["mass_drift_from_first_station"] > pinn.ENTRANCE_DRIFT_LIMIT
    assert not m["entrance_length_reliable"]


def test_predict_compiles_once_whatever_the_number_of_points(config):
    field = pinn.make_field(config.length)
    assert pinn.make_field(config.length) is field
    params = random_params(config, 0)
    pinn.predict(field, params, np.zeros((1, 2)) + [1.0, 0.1])
    # Other tests may already have compiled this field for other parameter
    # SHAPES, which legitimately needs its own compile; count only what the
    # varying point counts below add.
    before = pinn._PREDICTORS[field]._cache_size()
    for n in (3, 17, 9000):
        u, v, p = pinn.predict(field, params, np.zeros((n, 2)) + [1.0, 0.1])
        assert u.shape == (n,)
    assert pinn._PREDICTORS[field]._cache_size() == before


# -- configuration edge cases ------------------------------------------------- #


@pytest.mark.parametrize("section, key, value, message", [
    ("physics", "reynolds", float("nan"), "finite"),
    ("physics", "reynolds", float("inf"), "finite"),
    ("training.adam", "learning_rate", -1.0e-3, "learning_rate"),
    ("training.adam", "decay_rate", 1.5, "decay_rate"),
    ("training.lbfgs", "gtol", 0.0, "gtol"),
])
def test_config_refuses_values_that_cannot_describe_a_run(section, key, value, message):
    document = base_document()
    target = document
    for part in section.split("."):
        target = target[part]
    target[key] = value
    with pytest.raises(pinn.ConfigError, match=message):
        pinn.parse_config(document)


@pytest.mark.parametrize("station", [True, "a", float("nan")])
def test_profile_stations_must_be_finite_numbers(station):
    document = base_document()
    document["evaluation"]["profile_stations"] = [station, 2.0]
    with pytest.raises(pinn.ConfigError, match="profile_stations"):
        pinn.parse_config(document)


# -- residual-based adaptive refinement ------------------------------------- #


def test_rar_off_draws_exactly_the_ordinary_batch(config):
    """rar_points = 0 must reproduce the pre-RAR draws, which is what makes
    the migration's rar_points = 0 an honest description of old runs."""
    assert config.rar_points == 0
    params = random_params(config, 0)
    draw = pinn.make_draw(config, pinn.make_field(config.length))
    key = jax.random.PRNGKey(3)
    a = draw(key, params)
    b = pinn.sample_batch(key, config, config.n_interior)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_rar_adds_exactly_the_worst_residual_points_from_the_pool():
    document = tiny_document()
    config = pinn.parse_config(document)
    params = pinn.init_network(jax.random.PRNGKey(0), config.width, config.depth)
    field = pinn.make_field(config.length)
    key = jax.random.PRNGKey(9)
    batch = pinn.make_draw(config, field)(key, params)
    ordinary = pinn.sample_batch(key, config, config.n_interior)
    assert batch.interior.shape == (config.n_interior + config.rar_points, 2)
    assert np.array_equal(np.asarray(batch.interior[: config.n_interior]),
                          np.asarray(ordinary.interior))
    pool = pinn.sample_uniform(jax.random.fold_in(key, 1), config, config.rar_pool)
    scores = np.asarray(pinn.pointwise_residual(config, field, params, pool))
    added = np.asarray(pinn.pointwise_residual(config, field, params,
                                               batch.interior[config.n_interior:]))
    threshold = np.sort(scores)[-config.rar_points]
    assert np.all(added >= threshold * (1 - 1e-12))
    assert np.isclose(added.max(), scores.max())


def test_rar_pool_is_required_when_rar_is_on():
    document = tiny_document()
    del document["sampling"]["rar_pool"]
    with pytest.raises(pinn.ConfigError, match="rar_pool"):
        pinn.parse_config(document)


def test_rar_pool_is_refused_when_rar_is_off():
    document = base_document()
    document["sampling"]["rar_pool"] = 1000
    with pytest.raises(pinn.ConfigError, match="no effect"):
        pinn.parse_config(document)


def test_rar_pool_smaller_than_rar_points_is_refused():
    document = tiny_document()
    document["sampling"]["rar_pool"] = 16
    with pytest.raises(pinn.ConfigError, match="at least"):
        pinn.parse_config(document)


def test_the_rar_config_differs_from_the_baseline_only_in_rar():
    with open("developing_flow/re10_rar.yaml", encoding="utf-8") as handle:
        rar = yaml.safe_load(handle)
    base = base_document()
    assert rar.pop("name") != base.pop("name")
    rar_sampling, base_sampling = rar.pop("sampling"), base.pop("sampling")
    assert rar == base
    assert base_sampling.pop("rar_points") == 0
    assert rar_sampling.pop("rar_points") > 0 and rar_sampling.pop("rar_pool") > 0
    assert rar_sampling == base_sampling


# -- stored runs and their migration ------------------------------------------ #


def test_every_stored_developing_run_still_loads():
    """The guard CLAUDE.md asks for: a newly required key must not lock out
    finished runs. Run directories are untracked, so a fresh clone has none."""
    import pathlib
    root = pathlib.Path("developing_flow/runs")
    for directory in sorted(root.glob("*/")) if root.is_dir() else []:
        if (directory / "config.yaml").is_file() and (directory / "params.npz").is_file():
            pinn.load_run(directory)


def test_migration_records_rar_off_for_runs_that_predate_it(tmp_path):
    from developing_flow import migrate
    document = base_document()
    del document["sampling"]["rar_points"]
    run = tmp_path / "old_run"
    run.mkdir()
    (run / "config.yaml").write_text(
        "# provenance comment that must survive\n" + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8")
    with pytest.raises(pinn.ConfigError, match="rar_points"):
        pinn.load_config(run / "config.yaml")
    migrate.main(["--runs-root", str(tmp_path), "--write"])
    assert pinn.load_config(run / "config.yaml").rar_points == 0
    assert (run / "config.yaml").read_text().startswith("# provenance comment")
    assert (run / "config.migrated.json").is_file()


# -- the output-layer re-solve ---------------------------------------------- #


def test_apply_network_is_the_output_layer_over_hidden_features(config):
    params = random_params(config, 4)
    xi = jnp.array([0.3, -0.2])
    direct = pinn.apply_network(params, xi)
    layered = params["output"]["W"] @ pinn.hidden_features(params, xi) + params["output"]["b"]
    assert np.array_equal(np.asarray(direct), np.asarray(layered))


def test_refine_residual_vector_reproduces_the_training_loss_exactly():
    """refine.py assembles every derivative by hand from frozen features; its
    squared residual norm must equal the loss the trainer computes."""
    from developing_flow import refine
    config = pinn.parse_config(tiny_document())
    params = pinn.init_network(jax.random.PRNGKey(1), config.width, config.depth)
    field = pinn.make_field(config.length)
    batch = pinn.sample_batch(jax.random.PRNGKey(2), config, 128)
    weights = {k: jnp.asarray(v) for k, v in config.weights.items()}
    expected = float(pinn.total_loss(pinn.make_loss_terms(config, field)(params, batch), weights))
    interior = refine.feature_derivatives(config, params, batch.interior)
    inlet = jax.vmap(lambda xy: pinn.hidden_features(
        params, jnp.stack([2.0 * xy[0] / config.length - 1.0, xy[1]])))(batch.inlet)
    outlet = refine.feature_derivatives(config, params, batch.outlet)[:2]
    residual = refine.make_residual_vector(config, batch, interior, inlet, outlet)
    theta = jnp.concatenate([params["output"]["W"].ravel(), params["output"]["b"]])
    assert abs(float(jnp.sum(residual(theta) ** 2)) - expected) <= 1e-11 * expected


# -- seeds and the scorecard ------------------------------------------------- #


@pytest.mark.parametrize("seed", [1, 2])
def test_seed_configs_differ_from_the_baseline_only_in_name_and_seed(seed):
    with open(f"developing_flow/re10_seed{seed}.yaml", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    base = base_document()
    assert document.pop("seed") == seed and base.pop("seed") == 0
    assert document.pop("name") == f"developing_re10_seed{seed}"
    base.pop("name")
    assert document == base


def test_scorecard_judges_a_setup_only_outside_the_reference_seed_range():
    from developing_flow import scorecard
    keys = [k for k, _ in scorecard.METRICS]
    row = lambda v: {k: v for k in keys}  # noqa: E731
    groups = {"baseline": [row(1.0), row(3.0)], "a": [row(0.5)], "b": [row(2.0)], "c": [row(4.0)]}
    summary, verdicts = scorecard.summarise(groups, "baseline")
    assert summary["baseline"]["rms_x"] == {"mean": 2.0, "min": 1.0, "max": 3.0, "n": 2}
    assert verdicts["a"]["rms_x"] == "better"
    assert verdicts["b"]["rms_x"] == "within seed range"
    assert verdicts["c"]["rms_x"] == "worse"
    assert "minutes" not in verdicts["a"]


# -- Step 1 experiment configs: each changes only what it says --------------- #


def _flatten(document, prefix=""):
    flat = {}
    for key, value in document.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


STEP1_EXPERIMENTS = {
    "re10_lbfgs24.yaml": {"name", "training.lbfgs.cycles"},
    "re10_m4.yaml": {"name", "physics.inlet_sharpness"},
    "re10_m2.yaml": {"name", "physics.inlet_sharpness"},
}


@pytest.mark.parametrize("filename, allowed", sorted(STEP1_EXPERIMENTS.items()))
def test_each_step1_config_changes_only_its_own_keys(filename, allowed):
    with open(f"developing_flow/{filename}", encoding="utf-8") as handle:
        experiment = _flatten(yaml.safe_load(handle))
    base = _flatten(base_document())
    changed = {k for k in set(experiment) | set(base) if experiment.get(k) != base.get(k)}
    assert changed == allowed
    pinn.load_config(f"developing_flow/{filename}")
