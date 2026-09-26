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
    document["model"].update(width=16, depth=2)
    # RAR on, so the end-to-end run exercises the adaptive path too.
    document["sampling"].update(n_interior=256, n_inlet=32, n_outlet=32, n_validation=256,
                                rar_points=32, rar_pool=512, rar_mode="top")
    document["training"] = {
        "adam": {"steps": 60, "learning_rate": 1.0e-3, "schedule": "exponential",
                 "decay_rate": 0.9, "decay_steps": 1000, "resample_every": 20},
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
    return pinn.init_params(config, jax.random.PRNGKey(seed))


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
    field = pinn.make_field(config)
    x = np.linspace(0.0, config.length, 17)
    for wall in (-1.0, 1.0):
        u, v, _ = pinn.predict(field, params, np.stack([x, np.full_like(x, wall)], axis=-1))
        assert np.all(u == 0.0) and np.all(v == 0.0)


@pytest.mark.parametrize("seed", [0, 1])
def test_pressure_gauge_is_exactly_zero_at_the_outlet(config, seed):
    params = random_params(config, seed)
    field = pinn.make_field(config)
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


INLET_LIFT = {"physics.lift": "inlet"}
BLENDED_LIFT = {"physics.lift": "blended", "physics.lift_length": 0.5}
HARD_LINEAR = {"physics.inlet_condition": "hard", "physics.inlet_ramp": "linear", **INLET_LIFT}
HARD_EXPONENTIAL = {"physics.inlet_condition": "hard", "physics.inlet_ramp": "exponential",
                    "physics.inlet_ramp_length": 1.0, **INLET_LIFT}
STREAM = {"physics.formulation": "stream_pressure", **INLET_LIFT}

ANSATZ_VARIANTS = [
    {},
    HARD_LINEAR,
    HARD_EXPONENTIAL,
    STREAM,
    {**STREAM, **HARD_LINEAR},
    {**STREAM, **HARD_EXPONENTIAL},
    {"model.activation": "silu"},
    {"model.activation": "silu", "model.init_gain": 2.0},
    {"model.input_map": "exponential", "model.input_length": 2.0},
    {**STREAM, **HARD_EXPONENTIAL, "model.input_map": "exponential", "model.input_length": 2.0},
    {**STREAM, **BLENDED_LIFT},
    {**STREAM, **HARD_EXPONENTIAL, **BLENDED_LIFT},
    {**HARD_EXPONENTIAL, **BLENDED_LIFT},
]


def _variant_id(changes):
    return "-".join(str(v) for v in changes.values()) or "baseline"


def variant_config(changes):
    """The baseline config with `changes` applied and the loss weights made to
    match: the stream function has no continuity term, a hard inlet no inlet term."""
    document = base_document()
    for dotted, value in changes.items():
        section, key = dotted.split(".")
        document[section][key] = value
    if document["physics"]["formulation"] == "stream_pressure":
        document["weights"].pop("continuity")
    if document["physics"]["inlet_condition"] == "hard":
        document["weights"].pop("inlet")
    return pinn.parse_config(document)


@pytest.mark.parametrize("changes", ANSATZ_VARIANTS, ids=_variant_id)
def test_batched_derivatives_match_the_point_reference(changes):
    config = variant_config(changes)
    params = random_params(config, 3)
    field = pinn.make_field(config)
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
    expected = pinn.predict(pinn.make_field(config), selected, np.stack([x, y], axis=-1))
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
    field = pinn.make_field(config)
    assert pinn.make_field(config) is field
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
    draw = pinn.make_draw(config, pinn.make_field(config))
    key = jax.random.PRNGKey(3)
    a = draw(key, params)
    b = pinn.sample_batch(key, config, config.n_interior)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_rar_adds_exactly_the_worst_residual_points_from_the_pool():
    document = tiny_document()
    config = pinn.parse_config(document)
    params = pinn.init_params(config, jax.random.PRNGKey(0))
    field = pinn.make_field(config)
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
    assert rar_sampling.pop("rar_mode") == "top"
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


def test_migration_records_what_old_runs_actually_did(tmp_path):
    """A run from before every later key: after migration it must load and
    describe exactly the baseline behaviour."""
    from developing_flow import migrate
    document = base_document()
    for section, key in (("sampling", "rar_points"), ("sampling", "inlet_points"),
                         ("physics", "inlet_condition"), ("physics", "formulation"),
                         ("model", "activation"), ("model", "init_gain"),
                         ("model", "input_map"), ("training", "lm")):
        del document[section][key]
    del document["training"]["adam"]["schedule"]
    run = tmp_path / "old_run"
    run.mkdir()
    (run / "config.yaml").write_text(
        "# provenance comment that must survive\n" + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8")
    with pytest.raises(pinn.ConfigError, match="missing required key"):
        pinn.load_config(run / "config.yaml")
    (run / "params.npz").write_bytes(b"")  # a finished run; unfinished ones are skipped
    migrate.main(["--runs-root", str(tmp_path), "--write"])
    assert pinn.load_config(run / "config.yaml") == pinn.parse_config(base_document())
    assert (run / "config.yaml").read_text().startswith("# provenance comment")
    assert (run / "config.migrated.json").is_file()


def test_migration_gives_an_old_hard_inlet_run_the_linear_ramp_it_used(tmp_path):
    from developing_flow import migrate
    with open("developing_flow/re10_hardinlet.yaml", encoding="utf-8") as handle:
        text = handle.read()
    run = tmp_path / "old_hard"
    run.mkdir()
    (run / "config.yaml").write_text(text, encoding="utf-8")
    (run / "params.npz").write_bytes(b"")
    assert pinn.load_config(run / "config.yaml").inlet_ramp == "linear"
    old = "\n".join(line for line in text.splitlines() if "inlet_ramp" not in line) + "\n"
    (run / "config.yaml").write_text(old, encoding="utf-8")
    with pytest.raises(pinn.ConfigError, match="inlet_ramp"):
        pinn.load_config(run / "config.yaml")
    migrate.main(["--runs-root", str(tmp_path), "--write"])
    assert pinn.load_config(run / "config.yaml") == pinn.load_config("developing_flow/re10_hardinlet.yaml")


def test_migration_leaves_an_unfinished_run_alone(tmp_path):
    from developing_flow import migrate
    document = base_document()
    del document["model"]["init_gain"]
    run = tmp_path / "in_flight"
    run.mkdir()
    text = yaml.safe_dump(document, sort_keys=False)
    (run / "config.yaml").write_text(text, encoding="utf-8")
    migrate.main(["--runs-root", str(tmp_path), "--write"])
    assert (run / "config.yaml").read_text(encoding="utf-8") == text
    assert not (run / "config.migrated.json").exists()


# -- the output-layer re-solve ---------------------------------------------- #


def test_apply_network_is_the_output_layer_over_hidden_features(config):
    params = random_params(config, 4)
    xi = jnp.array([0.3, -0.2])
    direct = pinn.apply_network(params, xi, config.activation)
    layered = params["output"]["W"] @ pinn.hidden_features(params, xi, config.activation) + params["output"]["b"]
    assert np.array_equal(np.asarray(direct), np.asarray(layered))


def test_refine_residual_vector_reproduces_the_training_loss_exactly():
    """refine.py assembles every derivative by hand from frozen features; its
    squared residual norm must equal the loss the trainer computes."""
    from developing_flow import refine
    config = pinn.parse_config(tiny_document())
    params = pinn.init_params(config, jax.random.PRNGKey(1))
    field = pinn.make_field(config)
    batch = pinn.sample_batch(jax.random.PRNGKey(2), config, 128)
    weights = {k: jnp.asarray(v) for k, v in config.weights.items()}
    expected = float(pinn.total_loss(pinn.make_loss_terms(config, field)(params, batch), weights))
    interior = refine.feature_derivatives(config, params, batch.interior)
    inlet = jax.vmap(lambda xy: pinn.hidden_features(
        params, jnp.stack([2.0 * xy[0] / config.length - 1.0, xy[1]]), config.activation))(batch.inlet)
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


@pytest.mark.parametrize("stem", ["re10_stream", "re10_lbfgs24"])
@pytest.mark.parametrize("seed", [1, 2])
def test_step2_seed_configs_differ_from_their_seed_zero_only_in_name_and_seed(stem, seed):
    with open(f"developing_flow/{stem}_seed{seed}.yaml", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    with open(f"developing_flow/{stem}.yaml", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    assert document.pop("seed") == seed and base.pop("seed") == 0
    assert document.pop("name") == base.pop("name") + f"_seed{seed}"
    assert document == base


def test_errors_against_a_reference_are_measured_on_its_own_grid(tmp_path):
    from developing_flow import verify
    x, y = np.linspace(0.0, 10.0, 11), np.linspace(-1.0, 1.0, 5)
    X, Y = np.meshgrid(x, y, indexing="ij")
    ru, rv, rp = 1.0 - Y**2, 0.1 * np.sin(X) * (1.0 - Y**2), 0.2 * (10.0 - X)
    np.savez_compressed(tmp_path / "reference.npz", x=x, y=y, u=ru, v=rv, p=rp)
    r = verify.against_reference(tmp_path, x, y, 1.01 * ru, rv, rp + 0.02 * (10.0 - X), 3.77)
    assert abs(r["reference_u_relative_l2"] - 0.01) < 1e-14
    assert abs(r["reference_u_relative_l2_entrance"] - 0.01) < 1e-14
    assert abs(r["reference_u_relative_l2_developed"] - 0.01) < 1e-14
    assert r["reference_v_relative_l2"] == 0.0
    assert abs(r["reference_p_relative_l2"] - 0.1) < 1e-14
    assert abs(r["reference_centreline_max_abs_error"] - 0.01) < 1e-15
    # this reference IS the parabola scaled to flux 4/3, so its departure is zero
    assert r["reference_departure_from_parabola_developed"] < 1e-15
    with pytest.raises(ValueError, match="grid"):
        verify.against_reference(tmp_path, x[:-1], y, ru[:-1], rv[:-1], rp[:-1], 3.77)


def test_the_scorecard_measures_each_run_against_the_reference_for_its_own_problem(tmp_path):
    from developing_flow import scorecard
    for m in (8, 4):
        (tmp_path / f"m{m}").mkdir()
        (tmp_path / f"m{m}" / "reference.json").write_text(
            '{"reynolds": 10.0, "pressure_scaling": "inertial", "inlet_sharpness": %d, '
            '"length_sampled": 10.0}' % m)
    solutions = [tmp_path / "m8", tmp_path / "m4"]
    assert scorecard.pick_solution(pinn.load_config(CONFIG_PATH), solutions) == tmp_path / "m8"
    assert scorecard.pick_solution(pinn.load_config("developing_flow/re10_m4.yaml"),
                                   solutions) == tmp_path / "m4"
    with pytest.raises(ValueError, match="0 references match"):
        scorecard.pick_solution(pinn.load_config("developing_flow/re10_m2.yaml"), solutions)


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
    "re10_adam10k.yaml": {"name", "training.adam.steps", "training.adam.schedule",
                          "training.adam.decay_rate", "training.adam.decay_steps",
                          "training.adam.final_learning_rate"},
    "re10_inletgrid.yaml": {"name", "sampling.n_inlet", "sampling.inlet_points", "weights.inlet"},
    "re10_rad.yaml": {"name", "sampling.rar_points", "sampling.rar_pool", "sampling.rar_mode",
                      "sampling.rar_exponent", "sampling.rar_floor"},
    "re10_hardinlet.yaml": {"name", "physics.inlet_condition", "physics.inlet_ramp", "physics.lift",
                            "weights.inlet"},
    "re10_silu.yaml": {"name", "model.activation"},
    "re10_stream.yaml": {"name", "physics.formulation", "physics.lift", "weights.continuity"},
    "re10_m4.yaml": {"name", "physics.inlet_sharpness"},
    "re10_m2.yaml": {"name", "physics.inlet_sharpness"},
}


STEP2_EXPERIMENTS = {
    "re10_hardinlet_exp.yaml": {"name", "physics.inlet_condition", "physics.inlet_ramp",
                                "physics.inlet_ramp_length", "physics.lift", "weights.inlet"},
    "re10_stream_hard.yaml": {"name", "physics.formulation", "physics.inlet_condition",
                              "physics.inlet_ramp", "physics.inlet_ramp_length", "physics.lift",
                              "weights.continuity", "weights.inlet"},
    "re10_stream_hard_lift.yaml": {"name", "physics.formulation", "physics.inlet_condition",
                                   "physics.inlet_ramp", "physics.inlet_ramp_length",
                                   "physics.lift", "physics.lift_length",
                                   "weights.continuity", "weights.inlet"},
    "re10_hardinlet_lift.yaml": {"name", "physics.inlet_condition", "physics.inlet_ramp",
                                 "physics.inlet_ramp_length", "physics.lift",
                                 "physics.lift_length", "weights.inlet"},
    "re10_silu_gain2.yaml": {"name", "model.activation", "model.init_gain"},
    "re10_inputmap.yaml": {"name", "model.input_map", "model.input_length"},
    "re10_inletgrid_w1.yaml": {"name", "sampling.n_inlet", "sampling.inlet_points"},
    "re10_w40.yaml": {"name", "model.width"},
    "re10_lm.yaml": {"name", "model.width", "training.lbfgs.cycles", "training.lm.iterations",
                     "training.lm.damping", "training.lm.resample_every"},
    "re10_stream_lm.yaml": {"name", "physics.formulation", "physics.inlet_condition",
                            "physics.inlet_ramp", "physics.inlet_ramp_length", "physics.lift",
                            "physics.lift_length", "weights.continuity", "weights.inlet",
                            "model.width", "training.lbfgs.cycles", "training.lm.iterations",
                            "training.lm.damping", "training.lm.resample_every"},
}


@pytest.mark.parametrize("filename, allowed",
                         sorted({**STEP1_EXPERIMENTS, **STEP2_EXPERIMENTS}.items()))
def test_each_step1_config_changes_only_its_own_keys(filename, allowed):
    with open(f"developing_flow/{filename}", encoding="utf-8") as handle:
        experiment = _flatten(yaml.safe_load(handle))
    base = _flatten(base_document())
    changed = {k for k in set(experiment) | set(base) if experiment.get(k) != base.get(k)}
    assert changed == allowed
    pinn.load_config(f"developing_flow/{filename}")


# -- Step 1 options --------------------------------------------------------- #


@pytest.mark.parametrize("ramp", [HARD_LINEAR, HARD_EXPONENTIAL, {**HARD_EXPONENTIAL, **BLENDED_LIFT}],
                         ids=["linear", "exponential", "exponential-blended"])
@pytest.mark.parametrize("formulation", ["velocity_pressure", "stream_pressure"])
def test_a_hard_inlet_holds_exactly_for_any_parameters(formulation, ramp):
    config = variant_config({"physics.formulation": formulation, **ramp})
    y = np.linspace(-0.999, 0.999, 41)
    for seed in (0, 1):
        u, v, _ = pinn.predict(pinn.make_field(config), random_params(config, seed),
                               np.stack([np.zeros_like(y), y], axis=-1))
        assert np.max(np.abs(u - pinn.inlet_profile(y, config.inlet_sharpness))) <= 1e-15
        assert np.all(v == 0.0)


@pytest.mark.parametrize("inlet", [{}, HARD_LINEAR, HARD_EXPONENTIAL, BLENDED_LIFT,
                                   {**HARD_EXPONENTIAL, **BLENDED_LIFT}],
                         ids=["soft", "hard-linear", "hard-exponential", "soft-blended",
                              "hard-exponential-blended"])
def test_the_stream_function_conserves_mass_and_satisfies_no_slip(inlet):
    config = variant_config({**STREAM, **inlet})
    params = random_params(config, 2)
    field = pinn.make_field(config)
    nodes, weights = np.polynomial.legendre.leggauss(200)
    for x in (0.0, 1.3, 7.7, config.length):
        u, _, _ = pinn.predict(field, params, np.stack([np.full_like(nodes, x), nodes], axis=-1))
        assert abs(float(u @ weights) - pinn.INLET_FLUX) < 1e-12
    xs = np.linspace(0.0, config.length, 9)
    for wall in (-1.0, 1.0):
        u, v, _ = pinn.predict(field, params, np.stack([xs, np.full_like(xs, wall)], axis=-1))
        assert np.max(np.abs(u)) <= 1e-15 and np.max(np.abs(v)) <= 1e-15
    points = np.asarray(pinn.sample_batch(jax.random.PRNGKey(8), config, 64).interior)
    _, _, rc, *_ = pinn.residuals_at(config, field, params, points)
    assert np.max(np.abs(rc)) < 1e-12


def test_loss_terms_follow_the_formulation_and_inlet():
    stream = variant_config(STREAM)
    hard = variant_config(HARD_EXPONENTIAL)
    assert "continuity" not in stream.loss_terms and "inlet" in stream.loss_terms
    assert "inlet" not in hard.loss_terms and "continuity" in hard.loss_terms
    for config in (stream, hard):
        batch = pinn.sample_batch(jax.random.PRNGKey(1), config, 32)
        terms = pinn.make_loss_terms(config, pinn.make_field(config))(random_params(config, 0), batch)
        assert set(terms) == set(config.loss_terms) == set(config.weights)
    with pytest.raises(KeyError, match="disagree"):
        pinn.total_loss({"momentum_x": 1.0}, {"momentum_y": 1.0})


def test_init_gain_scales_every_layer_an_activation_follows_but_not_the_output():
    key = jax.random.PRNGKey(0)
    one, two = pinn.init_network(key, 8, 3, 3, 1.0), pinn.init_network(key, 8, 3, 3, 2.0)
    scaled = [(one["encoder_u"], two["encoder_u"]), (one["encoder_v"], two["encoder_v"])]
    scaled += list(zip(one["hidden"], two["hidden"]))
    for a, b in scaled:
        assert np.array_equal(2.0 * np.asarray(a["W"]), np.asarray(b["W"]))
        assert np.array_equal(np.asarray(a["b"]), np.asarray(b["b"]))
    assert np.array_equal(np.asarray(one["output"]["W"]), np.asarray(two["output"]["W"]))


def test_gain_one_is_plain_glorot():
    """Stored runs were initialised before init_gain existed; gain 1 must be that."""
    key = jax.random.PRNGKey(3)
    params = pinn.init_network(key, 8, 2, 3, 1.0)
    keys = jax.random.split(key, 2 + 3)
    expected = np.sqrt(2.0 / (2 + 8)) * np.asarray(jax.random.normal(keys[0], (8, 2), dtype=jnp.float64))
    assert np.array_equal(np.asarray(params["encoder_u"]["W"]), expected)


def test_the_exponential_ramp_vanishes_at_the_inlet_with_slope_one_over_its_length():
    for length in (0.5, 1.0, 3.0):
        assert float(pinn.inlet_ramp(0.0, 10.0, "exponential", length)) == 0.0
        slope = jax.grad(lambda x: pinn.inlet_ramp(x, 10.0, "exponential", length))(0.0)
        assert abs(float(slope) - 1.0 / length) < 1e-15
        assert abs(float(pinn.inlet_ramp(20.0 * length, 10.0, "exponential", length)) - 1.0) < 1e-8
    assert float(pinn.inlet_ramp(0.0, 10.0, "linear", None)) == 0.0
    assert float(pinn.inlet_ramp(10.0, 10.0, "linear", None)) == 1.0


def test_the_exponential_input_map_spans_the_input_range_and_stretches_the_inlet():
    L, ell = 10.0, 2.0
    x = np.linspace(0.0, L, 401)
    xi = np.array([float(pinn.input_coordinates(jnp.array([xv, 0.3]), L, "exponential", ell)[0])
                   for xv in x])
    assert abs(xi[0] + 1.0) < 1e-15 and abs(xi[-1] - 1.0) < 1e-15
    assert np.all(np.diff(xi) > 0)
    slope = jax.grad(lambda xv: pinn.input_coordinates(jnp.array([xv, 0.3]), L, "exponential", ell)[0])(0.0)
    assert abs(float(slope) - 2.0 / (ell * -np.expm1(-L / ell))) < 1e-14
    assert float(slope) > 4.9 * 2.0 / L  # at least 5x the linear map's slope at the inlet
    eta = pinn.input_coordinates(jnp.array([1.0, 0.3]), L, "exponential", ell)[1]
    assert float(eta) == 0.3


def test_silu_changes_the_network():
    tanh, silu = variant_config({}), variant_config({"model.activation": "silu"})
    params = random_params(tanh, 0)
    xi = jnp.array([0.2, -0.4])
    assert not np.allclose(np.asarray(pinn.apply_network(params, xi, "tanh")),
                           np.asarray(pinn.apply_network(params, xi, "silu")))


def test_cosine_schedule_runs_from_the_learning_rate_to_the_final_rate():
    with open("developing_flow/re10_adam10k.yaml", encoding="utf-8") as handle:
        config = pinn.parse_config(yaml.safe_load(handle))
    schedule = pinn.adam_schedule(config)
    assert float(schedule(0)) == config.adam_learning_rate
    assert abs(float(schedule(config.adam_steps)) - config.adam_final_learning_rate) < 1e-18


def test_a_fixed_inlet_grid_is_the_same_for_every_draw_and_crowds_the_corners():
    with open("developing_flow/re10_inletgrid.yaml", encoding="utf-8") as handle:
        config = pinn.parse_config(yaml.safe_load(handle))
    draw = pinn.make_draw(config, pinn.make_field(config))
    params = random_params(config, 0)
    a, b = draw(jax.random.PRNGKey(1), params), draw(jax.random.PRNGKey(2), params)
    assert np.array_equal(np.asarray(a.inlet), np.asarray(b.inlet))
    assert not np.array_equal(np.asarray(a.interior), np.asarray(b.interior))
    y = np.asarray(a.inlet[:, 1])
    assert len(y) == config.n_inlet and np.all(np.abs(y) < 1.0)
    assert np.mean(np.abs(y) > 0.9) > 0.2  # uniform would give 0.1
    heldout = pinn.sample_batch(pinn.root_keys(config.seed)[2], config, 16).inlet
    assert not np.array_equal(np.asarray(heldout), np.asarray(a.inlet[:16]))


def test_proportional_rar_picks_distinct_pool_points_weighted_toward_large_residuals():
    document = tiny_document()
    document["sampling"].update(rar_mode="proportional", rar_exponent=2.0, rar_floor=0.0)
    config = pinn.parse_config(document)
    field = pinn.make_field(config)
    params = random_params(config, 0)
    key = jax.random.PRNGKey(4)
    batch = pinn.make_draw(config, field)(key, params)
    added = np.asarray(batch.interior[config.n_interior:])
    pool = np.asarray(pinn.sample_uniform(jax.random.fold_in(key, 1), config, config.rar_pool))
    assert len(added) == config.rar_points
    assert len({tuple(p) for p in added}) == config.rar_points
    assert all(tuple(p) in {tuple(q) for q in pool} for p in added)
    scores = np.asarray(pinn.pointwise_residual(config, field, params, pool))
    chosen = np.asarray(pinn.pointwise_residual(config, field, params, added))
    assert chosen.mean() > 2.0 * scores.mean()
    top = np.sort(scores)[-config.rar_points:]
    assert chosen.mean() < top.mean(), "proportional sampling should not reduce to taking the maximum"


@pytest.mark.parametrize("changes, removed, message", [
    # decay_rate is refused or final_learning_rate is demanded; both name the schedule
    ({"training.adam.schedule": "cosine"}, (), "schedule"),
    ({"physics.formulation": "stream_pressure"}, (), "continuity"),
    ({"physics.inlet_condition": "hard"}, (), "weights.inlet"),
    ({"sampling.rar_points": 16}, (), "rar_pool"),
    ({"model.activation": "relu"}, (), "activation"),
])
def test_conditional_keys_are_required_or_refused_with_their_choice(changes, removed, message):
    document = base_document()
    for dotted, value in changes.items():
        *path, key = dotted.split(".")
        node = document
        for part in path:
            node = node[part]
        node[key] = value
    with pytest.raises(pinn.ConfigError, match=message):
        pinn.parse_config(document)


def _hard_document(**physics):
    document = base_document()
    del document["weights"]["inlet"]
    document["physics"].update(inlet_condition="hard", lift="inlet")
    document["physics"].update(physics)
    return document


@pytest.mark.parametrize("document, message", [
    (_hard_document(), "inlet_ramp"),                                   # hard needs a ramp
    (_hard_document(inlet_ramp="exponential"), "inlet_ramp_length"),    # which needs a length
    (_hard_document(inlet_ramp="linear", inlet_ramp_length=1.0), "no effect"),
    (_hard_document(inlet_ramp="cubic"), "inlet_ramp"),
    (_hard_document(inlet_ramp="exponential", inlet_ramp_length=0.0), "inlet_ramp_length"),
], ids=["no-ramp", "no-length", "linear-with-length", "unknown-ramp", "zero-length"])
def test_the_hard_inlet_ramp_is_required_and_checked(document, message):
    with pytest.raises(pinn.ConfigError, match=message):
        pinn.parse_config(document)


@pytest.mark.parametrize("physics, message", [
    ({"lift": "inlet"}, "no effect"),                       # a soft velocity inlet has no lift
    ({"formulation": "stream_pressure"}, "physics.lift"),   # the stream function needs one
    ({"formulation": "stream_pressure", "lift": "blended"}, "lift_length"),
    ({"formulation": "stream_pressure", "lift": "inlet", "lift_length": 0.5}, "no effect"),
    ({"formulation": "stream_pressure", "lift": "blended", "lift_length": -1.0}, "lift_length"),
    ({"formulation": "stream_pressure", "lift": "parabola"}, "physics.lift"),
], ids=["soft-velocity", "stream-without", "blended-without-length", "inlet-with-length",
        "negative-length", "unknown"])
def test_the_lift_is_required_exactly_where_there_is_one(physics, message):
    document = base_document()
    if physics.get("formulation") == "stream_pressure":
        del document["weights"]["continuity"]
    document["physics"].update(physics)
    with pytest.raises(pinn.ConfigError, match=message):
        pinn.parse_config(document)


def test_the_blended_lift_is_the_inlet_at_the_inlet_and_the_parabola_downstream():
    m, ell = 8, 0.5
    y = np.linspace(-1.0, 1.0, 41)
    lift = lambda x: np.array([float(pinn.lift_stream(x, yy, m, "blended", ell)) for yy in y])  # noqa: E731
    assert np.array_equal(lift(0.0), np.asarray(pinn.inlet_stream(y, m)))
    assert np.array_equal(lift(40.0 * ell), np.asarray(pinn.developed_stream(y)))
    for x in (0.0, 0.3, 1.0, 5.0):
        walls = [float(pinn.lift_stream(x, w, m, "blended", ell)) for w in (-1.0, 1.0)]
        assert abs(walls[1] - walls[0] - pinn.INLET_FLUX) < 1e-15
    slope = jax.grad(lambda x: pinn.lift_stream(x, 0.4, m, "blended", ell))
    assert float(slope(0.0)) == 0.0  # so v(0, y) = 0: the inlet stays exact
    assert float(slope(0.5)) != 0.0
    # d/dy of G_dev is the developed parabola
    u = jax.grad(lambda yy: pinn.developed_stream(yy))
    assert abs(float(u(0.3)) - float(pinn.developed_profile(0.3, pinn.INLET_FLUX))) < 1e-15


def test_the_inlet_lift_keeps_old_runs_bit_identical():
    """lift = inlet must be exactly the ansatz the stored E5 and E7 runs used."""
    for changes in (STREAM, HARD_LINEAR):
        config = variant_config(changes)
        field = pinn.make_field(config)
        params = random_params(config, 5)
        points = np.asarray(pinn.sample_batch(jax.random.PRNGKey(2), config, 16).interior)
        L, m = config.length, config.inlet_sharpness
        for point in points:
            x, y = point
            raw = pinn.apply_network(params, jnp.array([2.0 * x / L - 1.0, y]), config.activation)
            got = np.asarray(field(params, jnp.asarray(point)))
            if config.formulation == "velocity_pressure":
                ramp = x / L
                expected_u = pinn.inlet_profile(y, m) + pinn.wall_factor(y) * ramp * raw[0]
                assert got[0] == float(expected_u)
            assert got[2] == float((L - x) / L * raw[-1])


# -- Levenberg-Marquardt ------------------------------------------------------ #


def _small_config(changes=None, **training_lm):
    document = base_document()
    document["model"].update(width=8, depth=2)
    document["sampling"].update(n_interior=128, n_inlet=16, n_outlet=16, n_validation=64)
    for dotted, value in (changes or {}).items():
        section, key = dotted.split(".")
        document[section][key] = value
    if document["physics"]["formulation"] == "stream_pressure":
        document["weights"].pop("continuity")
    if document["physics"]["inlet_condition"] == "hard":
        document["weights"].pop("inlet")
    document["training"]["lm"].update(training_lm)
    return pinn.parse_config(document)


@pytest.mark.parametrize("changes", [{}, {**STREAM, **HARD_EXPONENTIAL, **BLENDED_LIFT},
                                     HARD_LINEAR], ids=["velocity-soft", "stream-hard-blended",
                                                        "velocity-hard"])
def test_the_least_squares_residuals_are_exactly_the_training_loss(changes):
    """||r||^2 must be the weighted loss and 2 J^T r its gradient, or
    Levenberg-Marquardt minimises something other than what training does."""
    config = _small_config(changes)
    field = pinn.make_field(config)
    flat, unflatten = ravel_pytree(random_params(config, 1))
    batch = pinn.sample_batch(jax.random.PRNGKey(2), config, config.n_interior)
    objective = lambda v: pinn.total_loss(  # noqa: E731
        pinn.make_loss_terms(config, field)(unflatten(v), batch), config.weights)
    normal_equations, lm_loss = pinn.make_gauss_newton(config, field, unflatten, chunk=64)
    A, g, rr = normal_equations(flat, batch)
    loss = float(objective(flat))
    assert abs(rr - loss) <= 1e-12 * loss
    assert abs(lm_loss(flat, batch) - loss) <= 1e-12 * loss
    grad = jax.grad(objective)(flat)
    assert float(jnp.linalg.norm(2.0 * g - grad)) <= 1e-12 * float(jnp.linalg.norm(grad))
    assert float(jnp.max(jnp.abs(A - A.T))) <= 1e-12 * float(jnp.max(jnp.abs(A)))


def test_levenberg_marquardt_steps_only_ever_lower_the_loss():
    config = _small_config()
    field = pinn.make_field(config)
    flat, unflatten = ravel_pytree(random_params(config, 3))
    batch = pinn.sample_batch(jax.random.PRNGKey(4), config, config.n_interior)
    normal_equations, lm_loss = pinn.make_gauss_newton(config, field, unflatten, chunk=64)
    A, g, loss = normal_equations(flat, batch)
    lam, nu, losses = 1e-3 * float(jnp.mean(jnp.diag(A))), 2.0, [loss]
    for _ in range(8):
        A, g, loss = normal_equations(flat, batch)
        delta, new, lam, nu, _ = pinn.levenberg_marquardt_step(
            A, g, loss, lambda d: lm_loss(flat + jnp.asarray(d), batch), lam, nu)
        assert delta is not None
        flat = flat + jnp.asarray(delta)
        assert abs(lm_loss(flat, batch) - new) <= 1e-12 * new
        losses.append(new)
    assert all(b < a for a, b in zip(losses, losses[1:]))
    assert losses[-1] < 0.1 * losses[0]


def test_a_rejected_step_is_retried_with_more_damping():
    A, g = np.eye(2), np.array([1.0, 0.0])
    calls = []

    def trial(delta):  # the first two trial steps fail to lower the loss
        calls.append(delta.copy())
        return 2.0 if len(calls) <= 2 else 0.5

    delta, new, lam, nu, attempts = pinn.levenberg_marquardt_step(A, g, 1.0, trial, 1.0, 2.0)
    assert attempts == 3 and new == 0.5 and nu == 2.0
    # damping went 1 -> 2 -> 8 before the accepted step, so each retry is shorter
    assert np.linalg.norm(calls[1]) < np.linalg.norm(calls[0])
    assert np.allclose(calls[2], -g / (1.0 + 8.0))


@pytest.mark.parametrize("lm, message", [
    ({"iterations": 5}, "damping"),
    ({"iterations": 5, "damping": 1e-3}, "resample_every"),
    ({"iterations": 0, "damping": 1e-3}, "no effect"),
    ({"iterations": -1}, "iterations"),
    ({"iterations": 5, "damping": 0.0, "resample_every": 2}, "damping"),
    ({"iterations": 5, "damping": 1e-3, "resample_every": 0}, "resample_every"),
])
def test_the_lm_settings_are_required_and_checked(lm, message):
    document = base_document()
    document["training"]["lm"] = lm
    with pytest.raises(pinn.ConfigError, match=message):
        pinn.parse_config(document)


@pytest.mark.slow
def test_a_tiny_run_with_levenberg_marquardt_is_deterministic():
    document = tiny_document()
    document["training"]["lm"] = {"iterations": 4, "damping": 1e-3, "resample_every": 2}
    config = pinn.parse_config(document)
    first, history, info = pinn.train(config, verbose=False)
    second, _, _ = pinn.train(config, verbose=False)
    assert np.array_equal(np.asarray(ravel_pytree(first)[0]), np.asarray(ravel_pytree(second)[0]))
    assert sum(row["phase"] == "lm" for row in history) == 4
    assert info["levenberg_marquardt"]["iterations"] == 4


def test_a_soft_inlet_refuses_a_ramp():
    document = base_document()
    document["physics"]["inlet_ramp"] = "linear"
    with pytest.raises(pinn.ConfigError, match="no effect"):
        pinn.parse_config(document)


@pytest.mark.parametrize("model, message", [
    ({"input_map": "exponential"}, "input_length"),
    ({"input_length": 2.0}, "no effect"),
    ({"input_map": "log"}, "input_map"),
    ({"init_gain": 0.0}, "init_gain"),
    ({"init_gain": -1.0}, "init_gain"),
])
def test_the_model_options_are_checked(model, message):
    document = base_document()
    document["model"].update(model)
    with pytest.raises(pinn.ConfigError, match=message):
        pinn.parse_config(document)


def test_top_rar_refuses_the_proportional_settings():
    document = tiny_document()
    document["sampling"]["rar_exponent"] = 1.0
    with pytest.raises(pinn.ConfigError, match="no effect"):
        pinn.parse_config(document)


def test_the_stream_function_is_measured_like_any_other_field():
    config = variant_config(STREAM)
    m, _ = pinn.measure_solution(config, pinn.make_field(config), random_params(config, 1))
    assert abs(m["flux_at_inlet_station"] - pinn.INLET_FLUX) < 1e-12
    assert m["residual_continuity_max"] < 1e-12
