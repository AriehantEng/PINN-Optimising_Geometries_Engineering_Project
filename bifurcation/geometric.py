"""Network 3: from the daughters' angles and a set flow share, the best widths.

Trained on the finite-element design study (bifurcation/design_study.py,
docs/bifurcation_re10-shares.json): for each pair of angles (a1, a2) and each
share q = Q1 / Q_in, the width split s = w1 / w0 with the lowest
total-pressure drop, and that drop.

What the network is asked to learn is only what physics does not already
give. Two baselines carry the rest:

    split   the friction law: plane-Poiseuille daughters cost sum Q_i^2 L / w_i^3,
            minimised under w1 + w2 = w0 by w_i ~ Q_i^(1/2):
            s_f(q) = sqrt(q) / (sqrt(q) + sqrt(1 - q)). The study's optima
            sit within 0.001-0.023 of it.
    drop    the friction law's pressure drop at that split, plus the parent's:
            1.5 nu L_d Q (sqrt(q) + sqrt(1 - q))^4 + 12 nu L_p Q / w0^3.

The network N (developing_flow's modified MLP, inputs a1, a2, q) corrects
them, built so that two rules hold for ANY parameters:

    widths    s = lo + (hi - lo) sigmoid(logit(s_f) + c): s stays inside
              split_bounds(min_width), so no width can reach zero
    mirror    swapping the daughters (a1 <-> a2, q -> 1 - q) must give the
              mirrored split 1 - s and the same drop: c is the antisymmetric
              part of N's first output, the drop correction the symmetric
              part of its second

With N = 0 the network IS the friction law; trained, it must beat it on data
it has not seen, or it adds nothing. That is what `cross_validate` (leave one
angle pair out) and `validate_against_fe` (new angle pairs, the network's
split re-solved by finite elements: the extra pressure drop paid for trusting
it, its "regret") measure.

    python -m bifurcation.geometric --config bifurcation/n3_re10_shares.yaml
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from bifurcation.geometry import split_bounds  # noqa: E402
from developing_flow import pinn  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
#: Report stems in docs/ for runs named before reports were named by run.
REPORTS = {"network3_re10_shares": "bifurcation_re10-network3"}
ROOT = HERE.parent
ConfigError = pinn.ConfigError

SCHEMA = {
    "name": str,
    "seed": int,
    "data": str,                  # the design study's JSON, relative to the repository
    "model": {"width": int, "depth": int, "activation": str, "init_gain": float},
    "fit": {"split_scale": float, "drop_scale": float, "regularisation": list,
            "restarts": int, "max_evaluations": int, "workers": int},
    "validation": {"pairs": list, "shares": list, "workers": int},
}


def parse_config(document):
    document = copy.deepcopy(document)
    pinn._check(document, SCHEMA)
    if document["model"]["activation"] not in pinn.ACTIVATIONS:
        raise ConfigError(f"model.activation must be one of {list(pinn.ACTIVATIONS)}")
    if not document["fit"]["regularisation"] or any(
            not (isinstance(v, (int, float)) and v >= 0) for v in document["fit"]["regularisation"]):
        raise ConfigError("fit.regularisation must be a list of candidates >= 0")
    for key in ("split_scale", "drop_scale"):
        if not document["fit"][key] > 0:
            raise ConfigError(f"fit.{key} must be positive")
    if document["fit"]["restarts"] < 1:
        raise ConfigError("fit.restarts must be at least 1")
    if document["fit"]["workers"] < 1:
        raise ConfigError("fit.workers must be at least 1")
    for pair in document["validation"]["pairs"]:
        if not (isinstance(pair, list) and len(pair) == 2):
            raise ConfigError(f"validation pairs must be [angle1, angle2], got {pair!r}")
    for share in document["validation"]["shares"]:
        if not 0.0 < share < 1.0:
            raise ConfigError(f"validation shares must lie in (0, 1), got {share}")
    return document


# --------------------------------------------------------------------------- #
# Data and baselines
# --------------------------------------------------------------------------- #


def load_study(path):
    """(study config, arrays a1, a2, q, best split, lowest drop): every set-share
    optimum; the drop is the fine-mesh re-solve of the optimum."""
    study = json.loads(pathlib.Path(path).read_text())
    rows = [r for r in study["rows"] if r["feasible"] and r["share"] is not None]
    data = {
        "a1": np.array([r["angles"][0] for r in rows]),
        "a2": np.array([r["angles"][1] for r in rows]),
        "q": np.array([r["share"] for r in rows]),
        "split": np.array([r["best_split"] for r in rows]),
        "drop": np.array([r["total_pressure_drop_fine"] for r in rows]),
    }
    return study["config"], data


def friction_split(q):
    """sqrt(q) / (sqrt(q) + sqrt(1 - q)), unclipped."""
    return jnp.sqrt(q) / (jnp.sqrt(q) + jnp.sqrt(1.0 - q))


def friction_drop(q, study_config):
    """The friction law's total-pressure drop at its own split, plus the parent's."""
    physics, geometry = study_config["physics"], study_config["geometry"]
    _, nu = pinn.scaling_coefficients(physics["reynolds"], physics["pressure_scaling"])
    flux = 4.0 / 3.0                                     # the study's parabolic inflow
    daughters = 1.5 * nu * geometry["daughter_length"] * flux * (
        jnp.sqrt(q) + jnp.sqrt(1.0 - q)) ** 4
    parent = 12.0 * nu * geometry["parent_length"] * flux / 2.0**3
    return daughters + parent


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #


def make_model(config, study_config):
    """predict(params, a1, a2, q) -> (split, drop), angles in degrees; vectorised."""
    lo, hi = split_bounds(study_config["geometry"]["min_width"])
    activation = config["model"]["activation"]
    edge = 1e-9                                          # keeps logit(s_f) finite at q -> 0, 1

    def raw(params, a1, a2, q):
        return pinn.apply_network(params, jnp.stack([a1 / 90.0 - 1.0, a2 / 90.0 - 1.0,
                                                     2.0 * q - 1.0]), activation)

    def one(params, a1, a2, q):
        r, mirrored = raw(params, a1, a2, q), raw(params, a2, a1, 1.0 - q)
        p = jnp.clip((friction_split(q) - lo) / (hi - lo), edge, 1.0 - edge)
        logit = jnp.log(p) - jnp.log1p(-p) + 0.5 * (r[0] - mirrored[0])
        split = lo + (hi - lo) * jax.nn.sigmoid(logit)
        drop = friction_drop(q, study_config) * jnp.exp(0.5 * (r[1] + mirrored[1]))
        return split, drop

    def predict(params, a1, a2, q):
        a1, a2, q = (jnp.asarray(v, dtype=jnp.float64) for v in (a1, a2, q))
        return jax.vmap(lambda x, y, z: one(params, x, y, z))(
            jnp.atleast_1d(a1), jnp.atleast_1d(a2), jnp.atleast_1d(q))

    return predict


def init_params(config, key):
    m = config["model"]
    return pinn.init_network(key, 3, m["width"], m["depth"], 2, m["init_gain"])


def fit(config, study_config, data, regularisation, seed):
    """Least squares on the training data, from `restarts` initialisations;
    the lowest objective wins. Returns (params, objective)."""
    predict = make_model(config, study_config)
    s_scale, d_scale = config["fit"]["split_scale"], config["fit"]["drop_scale"]
    template = init_params(config, jax.random.PRNGKey(seed))
    _, unflatten = ravel_pytree(template)
    a1, a2, q = (jnp.asarray(data[k]) for k in ("a1", "a2", "q"))
    split, log_drop = jnp.asarray(data["split"]), jnp.log(jnp.asarray(data["drop"]))
    root = math.sqrt(regularisation)

    @jax.jit
    def residuals(theta):
        s, d = predict(unflatten(theta), a1, a2, q)
        return jnp.concatenate([(s - split) / s_scale, (jnp.log(d) - log_drop) / d_scale,
                                root * theta])

    jacobian = jax.jit(jax.jacfwd(residuals))
    best = (None, math.inf)
    for k in range(config["fit"]["restarts"]):
        theta0 = ravel_pytree(init_params(config, jax.random.PRNGKey(seed + 1000 * k)))[0]
        result = least_squares(lambda t: np.asarray(residuals(jnp.asarray(t))), np.asarray(theta0),
                               jac=lambda t: np.asarray(jacobian(jnp.asarray(t))), method="lm",
                               max_nfev=config["fit"]["max_evaluations"], xtol=1e-15,
                               ftol=1e-15, gtol=1e-15)
        objective = float(np.sum(result.fun**2))
        if objective < best[1]:
            best = (unflatten(jnp.asarray(result.x)), objective)
    return best


def errors(predict, params, data):
    s, d = predict(params, data["a1"], data["a2"], data["q"])
    return {"split_abs": np.abs(np.asarray(s) - data["split"]),
            "drop_rel": np.abs(np.asarray(d) / data["drop"] - 1.0)}


def baseline_errors(study_config, data):
    lo, hi = split_bounds(study_config["geometry"]["min_width"])
    s = np.clip(np.asarray(friction_split(jnp.asarray(data["q"]))), lo, hi)
    d = np.asarray(friction_drop(jnp.asarray(data["q"]), study_config))
    return {"split_abs": np.abs(s - data["split"]), "drop_rel": np.abs(d / data["drop"] - 1.0)}


def subset(data, mask):
    return {k: v[mask] for k, v in data.items()}


def _cv_fold(task):
    """One held-out angle pair at one regularisation: fit on the rest, score it.
    A pure function of its arguments (fits are seeded from the config), so
    the result does not depend on which process runs it."""
    config, study_config, data, lam, pair = task
    held = (data["a1"] == pair[0]) & (data["a2"] == pair[1])
    params, _ = fit(config, study_config, subset(data, ~held), lam, config["seed"])
    e = errors(make_model(config, study_config), params, subset(data, held))
    return {"pair": list(pair), "split_max": float(e["split_abs"].max()),
            "split_rms": float(np.sqrt(np.mean(e["split_abs"] ** 2))),
            "drop_max": float(e["drop_rel"].max()),
            "drop_rms": float(np.sqrt(np.mean(e["drop_rel"] ** 2)))}


def cross_validate(config, study_config, data, verbose=True):
    """Leave one angle pair out: train on the others, predict the missing one.
    Run for every regularisation candidate, the folds spread over fit.workers
    processes; returns {candidate: per-pair errors}."""
    pairs = sorted({(a, b) for a, b in zip(data["a1"], data["a2"])})
    candidates = config["fit"]["regularisation"]
    tasks = [(config, study_config, data, lam, pair) for lam in candidates for pair in pairs]
    if config["fit"]["workers"] == 1:
        results = [_cv_fold(t) for t in tasks]
    else:
        with multiprocessing.get_context("spawn").Pool(config["fit"]["workers"]) as pool:
            results = pool.map(_cv_fold, tasks)
    out = {}
    for i, lam in enumerate(candidates):
        rows = results[i * len(pairs):(i + 1) * len(pairs)]
        out[lam] = rows
        if verbose:
            print(f"  regularisation {lam:g}: held-out split rms "
                  f"{np.sqrt(np.mean([r['split_rms'] ** 2 for r in rows])):.4f}, drop rms "
                  f"{np.sqrt(np.mean([r['drop_rms'] ** 2 for r in rows])):.4f}", flush=True)
    return out


def choose(cv, scale_split, scale_drop):
    """The candidate with the lowest held-out error, split and drop scaled alike."""
    def score(rows):
        return (np.mean([r["split_rms"] ** 2 for r in rows]) / scale_split**2
                + np.mean([r["drop_rms"] ** 2 for r in rows]) / scale_drop**2)
    return min(cv, key=lambda lam: score(cv[lam]))


# --------------------------------------------------------------------------- #
# Validation by finite elements on angle pairs it never saw
# --------------------------------------------------------------------------- #


def _validate_one(task):
    """FE optimum for one (angles, share), and FE drops at the network's and the
    friction law's splits. Runs in a worker process."""
    from bifurcation import design_study
    from bifurcation.geometry import GeometryError
    study_config, angles, share, proposed = task
    optimum = design_study.optimise((study_config, angles, share))
    if not optimum["feasible"]:
        return {"angles": angles, "share": share, "feasible": False}
    mesh = study_config["mesh"]
    try:
        at_proposed = design_study.solve(study_config, proposed, angles, share,
                                         mesh["max_area"], mesh["corner_levels"])
        proposed_drop = at_proposed["total_pressure_drop"]
    except GeometryError as error:
        proposed_drop, optimum["proposed_refused"] = math.inf, str(error)
    best = optimum["total_pressure_drop"]
    return {
        "angles": angles, "share": share, "feasible": True,
        "best_split": optimum["best_split"], "best_drop": best,
        "best_drop_fine": optimum["total_pressure_drop_fine"],
        "network_split": proposed, "network_drop_fe": proposed_drop,
        "network_regret": proposed_drop / best - 1.0,
        "friction_split": optimum["friction_law_split"],
        "friction_drop_fe": optimum["friction_law_drop"],
        "friction_regret": optimum["friction_law_drop"] / best - 1.0,
        "equal_widths_drop": optimum["equal_widths_drop"],
        "seconds": optimum["seconds"],
    }


def validate_against_fe(config, study_config, params, verbose=True):
    predict = make_model(config, study_config)
    tasks = []
    for pair in config["validation"]["pairs"]:
        for share in config["validation"]["shares"]:
            s, d = predict(params, pair[0], pair[1], share)
            tasks.append((study_config, [float(pair[0]), float(pair[1])], float(share),
                          float(s[0])))
    predicted = {(t[1][0], t[1][1], t[2]): None for t in tasks}
    for t in tasks:
        _, d = predict(params, t[1][0], t[1][1], t[2])
        predicted[(t[1][0], t[1][1], t[2])] = float(d[0])
    rows = []
    with multiprocessing.get_context("spawn").Pool(config["validation"]["workers"]) as pool:
        for row in pool.imap(_validate_one, tasks):
            if row["feasible"]:
                row["network_predicted_drop"] = predicted[(row["angles"][0], row["angles"][1],
                                                           row["share"])]
                row["network_drop_error"] = row["network_predicted_drop"] / row["best_drop_fine"] - 1.0
                if verbose:
                    print(f"  {row['angles'][0]:g}/{row['angles'][1]:g} share {row['share']:.2f}: "
                          f"best {row['best_split']:.4f}, network {row['network_split']:.4f} "
                          f"(regret {row['network_regret']:+.2e}), friction "
                          f"{row['friction_split']:.4f} (regret {row['friction_regret']:+.2e}); "
                          f"drop predicted {row['network_predicted_drop']:.4f} vs "
                          f"{row['best_drop_fine']:.4f}", flush=True)
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Running, saving, reporting
# --------------------------------------------------------------------------- #


def run(config_path, out=None, verbose=True):
    text = pathlib.Path(config_path).read_text()
    config = parse_config(yaml.safe_load(text))
    directory = pathlib.Path(out) if out else HERE / "runs" / config["name"]
    if (directory / "params.npz").exists():
        raise FileExistsError(f"{directory} already holds a finished run")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(text)
    study_config, data = load_study(ROOT / config["data"])
    start = time.perf_counter()
    if verbose:
        print(f"  {len(data['q'])} optima, {len({(a, b) for a, b in zip(data['a1'], data['a2'])})} "
              "angle pairs; leave-one-pair-out cross-validation:", flush=True)
    cv = cross_validate(config, study_config, data, verbose)
    lam = choose(cv, config["fit"]["split_scale"], config["fit"]["drop_scale"])
    params, objective = fit(config, study_config, data, lam, config["seed"])
    fit_seconds = time.perf_counter() - start
    predict = make_model(config, study_config)
    train = errors(predict, params, data)
    base = baseline_errors(study_config, data)
    pinn.save_params(directory / "params.npz", params)
    if verbose:
        print(f"  chosen regularisation {lam:g}; trained in {fit_seconds:.1f} s. On the training "
              f"data: split max {train['split_abs'].max():.4f} (friction law "
              f"{base['split_abs'].max():.4f}), drop max {train['drop_rel'].max():.4f} "
              f"(friction law {base['drop_rel'].max():.4f})", flush=True)
        print("  finite-element validation on angle pairs it never saw:", flush=True)
    validation = validate_against_fe(config, study_config, params, verbose)
    metrics = {
        "regularisation": lam, "objective": objective, "fit_seconds": fit_seconds,
        "cross_validation": {str(k): v for k, v in cv.items()},
        "training": {"split_max": float(train["split_abs"].max()),
                     "drop_max": float(train["drop_rel"].max())},
        "friction_law_on_training": {"split_max": float(base["split_abs"].max()),
                                     "drop_max": float(base["drop_rel"].max())},
        "validation": validation,
    }
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    report(config, study_config, data, params, metrics, ROOT / "docs" / REPORTS[config["name"]]
           if config["name"] in REPORTS else ROOT / "docs" / config["name"])
    return metrics


def load_run(directory):
    """(propose, config): propose(a1, a2, share) -> (split, w1, w2, drop)."""
    directory = pathlib.Path(directory)
    config = parse_config(yaml.safe_load((directory / "config.yaml").read_text()))
    study_config, _ = load_study(ROOT / config["data"])
    _, unflatten = ravel_pytree(init_params(config, jax.random.PRNGKey(config["seed"])))
    params = unflatten(jnp.asarray(np.load(directory / "params.npz")["flat"]))
    predict = make_model(config, study_config)

    def propose(a1, a2, share):
        s, d = predict(params, a1, a2, share)
        s = np.asarray(s)
        return s, 2.0 * s, 2.0 * (1.0 - s), np.asarray(d)

    propose.config, propose.study_config, propose.params = config, study_config, params
    return propose, config


def report(config, study_config, data, params, metrics, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    predict = make_model(config, study_config)
    cv = metrics["cross_validation"][str(metrics["regularisation"])]
    base = baseline_errors(study_config, data)
    pairs = sorted({(a, b) for a, b in zip(data["a1"], data["a2"])})
    lines = [
        "## Network 3: the best widths from the angles and a set flow share",
        "",
        f"Trained on {len(data['q'])} finite-element optima ({len(pairs)} angle pairs x "
        f"{len(set(data['q']))} shares, docs/bifurcation_re10-shares.md), Re_h = "
        f"{study_config['physics']['reynolds']:g}, daughters {study_config['geometry']['daughter_length']:g} "
        f"long. Network: modified MLP, width {config['model']['width']}, depth "
        f"{config['model']['depth']}, correcting the friction law (split) and its pressure drop. "
        f"Regularisation {metrics['regularisation']:g}, chosen by leave-one-pair-out "
        f"cross-validation; fitted in {metrics['fit_seconds']:.0f} s including that.",
        "",
        "### Leave one angle pair out (the pair's optima predicted by a network that never saw it)",
        "",
        "| held-out pair | split error, max | friction law, max | drop error, max | "
        "friction-law drop, max |",
        "|---|---|---|---|---|",
    ]
    for row in cv:
        held = (data["a1"] == row["pair"][0]) & (data["a2"] == row["pair"][1])
        lines.append(f"| {row['pair'][0]:g} / {row['pair'][1]:g} | {row['split_max']:.4f} | "
                     f"{base['split_abs'][held].max():.4f} | {100 * row['drop_max']:.2f}% | "
                     f"{100 * base['drop_rel'][held].max():.2f}% |")
    lines += ["", "### New angle pairs, checked by finite elements", "",
              "'regret' is the extra pressure drop paid for using the proposed split instead of "
              "the finite-element optimum (coarse mesh; ~1e-4 is its noise).", "",
              "| angles | share | best split | network | its regret | friction law | its regret | "
              "drop predicted | drop (FE) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in metrics["validation"]:
        if not r["feasible"]:
            lines.append(f"| {r['angles'][0]:g} / {r['angles'][1]:g} | {r['share']:.2f} | "
                         "refused | | | | | | |")
            continue
        lines.append(
            f"| {r['angles'][0]:g} / {r['angles'][1]:g} | {r['share']:.2f} | {r['best_split']:.4f} | "
            f"{r['network_split']:.4f} | {100 * r['network_regret']:+.3f}% | "
            f"{r['friction_split']:.4f} | {100 * r['friction_regret']:+.3f}% | "
            f"{r['network_predicted_drop']:.4f} | {r['best_drop_fine']:.4f} |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colours = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#6B6B6B"]
    q = np.linspace(0.03, 0.97, 200)
    for colour, pair in zip(colours, pairs):
        held = (data["a1"] == pair[0]) & (data["a2"] == pair[1])
        s, d = predict(params, np.full_like(q, pair[0]), np.full_like(q, pair[1]), q)
        axes[0].plot(q, np.asarray(s) - np.asarray(friction_split(jnp.asarray(q))), color=colour,
                     label=f"{pair[0]:g} / {pair[1]:g}")
        axes[0].plot(data["q"][held], data["split"][held]
                     - np.asarray(friction_split(jnp.asarray(data["q"][held]))), "o", color=colour)
        axes[1].plot(q, np.asarray(d), color=colour, label=f"{pair[0]:g} / {pair[1]:g}")
        axes[1].plot(data["q"][held], data["drop"][held], "o", color=colour)
    for r in metrics["validation"]:
        if r["feasible"]:
            axes[0].plot(r["share"], r["best_split"] - float(friction_split(r["share"])), "k^")
            axes[1].plot(r["share"], r["best_drop_fine"], "k^")
    axes[0].set_ylabel("best split minus the friction law")
    axes[1].set_ylabel("lowest total-pressure drop")
    for ax in axes:
        ax.set_xlabel("share of the flow through daughter 1")
        ax.grid(color="#E6E6E6")
        ax.legend(fontsize=8)
    fig.suptitle("Network 3 (lines) against the finite-element optima (dots; triangles: "
                 "angle pairs it never saw)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    run(args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
