"""A standalone physics-informed neural network for steady, DEVELOPING flow in
a 2D channel, with the inertial terms and exact no-slip walls.

What it solves
--------------
Steady, incompressible Navier-Stokes in a straight channel x* in [0, L*],
y* in [-1, 1]. Lengths are scaled on the half-height h and velocities on the
developed centreline speed U, so Re = U h / nu. In the inertial pressure
scaling (p* = p / rho U^2):

    u u_x + v u_y  +  p_x  -  (1/Re)(u_xx + u_yy)  =  0      x-momentum
    u v_x + v v_y  +  p_y  -  (1/Re)(v_xx + v_yy)  =  0      y-momentum
    u_x + v_y                                      =  0      continuity

The first pair of terms in each momentum equation is the inertia of the
fluid. It is always on here; there is no switch to remove it. (The viscous
scaling, for Re <= 1, is the same equation multiplied by Re.)

Boundary conditions
    walls   y* = +-1 : u = v = 0                  EXACT, built into the network
    inlet   x* = 0   : u = U0 (1 - y^(2m)), v = 0 soft (a loss term)
    outlet  x* = L*  : du/dx = dv/dx = 0          soft
    gauge   x* = L*  : p = 0                      EXACT, built into the network

The flow enters NOT fully developed and develops along the channel. A
perfectly flat inlet cannot be used with exact no-slip: at the two corners it
would demand an infinite network output. The inlet family above is the
regularised version: it carries the same flux (4/3) for every sharpness m,
vanishes at the walls, and approaches the flat inlet as m grows. m = 1 is the
developed parabola itself and is refused.

Why this file exists apart from src/
------------------------------------
It is self-contained (jax, optax, numpy, scipy, matplotlib, pyyaml only) so
it runs on its own, including in Colab. It differs from the main solver in two
deliberate ways, both aimed at the failure the main solver showed on developing
flow, where the held-out loss ended 238x to 72,867x above the training loss
after L-BFGS spent 6,000 iterations on ONE frozen draw of 2,048 points:

1. L-BFGS runs in CYCLES. Each cycle restarts L-BFGS on a fresh draw of
   collocation points, so no single set can be memorised, while points stay
   frozen WITHIN a cycle so L-BFGS's curvature estimate remains valid.
2. A share of the interior points is concentrated where the flow actually
   develops: near the inlet (x* sampled as focus_length * s^2) and near the
   walls (y* sampled as sin(pi t / 2)).

The parameters kept are those with the lowest loss on a separate held-out set
of points that is never trained on.

Usage
-----
    python -m developing_flow.pinn --config developing_flow/re10.yaml

In Colab, upload this file and a config, then run
    !python pinn.py --config re10.yaml

Everything a run produces goes to developing_flow/runs/<name>/ (or --out).
Using a trained model afterwards:

    from developing_flow.pinn import load_run
    model = load_run("developing_flow/runs/developing_re10")
    u, v, p = model(x, y)          # arrays of x*, y*

Entrance-length reference: Durst, Ray, Unsal & Bayoumi (2005), J. Fluids Eng.
127(6):1154-1160, plane-channel fit converted to half-height units as in the
main project (c0 = 0.3155, c1 = 0.01105, n = 1.6 per hydraulic diameter
D_h = 4h, with Re_Dh = (8/3) Re_h). It describes a FLAT inlet, so a smoothed
inlet is expected to develop slightly sooner. It is printed for comparison,
never used in the loss.
"""

from __future__ import annotations

import jax

# Double precision before any array exists. float32 caps achievable accuracy
# near 1e-4 and the symptom looks exactly like an optimiser failure.
jax.config.update("jax_enable_x64", True)

import argparse  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import functools  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import NamedTuple  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import yaml  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

#: Loss terms, in the order they are reported. Nothing else enters the loss.
LOSS_TERMS = ("momentum_x", "momentum_y", "continuity", "inlet", "outlet")

#: Flux of every inlet in the family, and of the developed parabola 1 - y^2.
INLET_FLUX = 4.0 / 3.0

#: Durst et al. (2005), per hydraulic diameter; see the module docstring.
DURST_C0, DURST_C1, DURST_N = 0.3155, 0.01105, 1.6

#: An entrance length is read where the centreline reaches 99% of its developed
#: value. If the flux itself varies along the channel by a comparable amount,
#: the centreline creeps and the crossing moves (CLAUDE.md: a 6% drift once put
#: it at 9.3 instead of 2.4). Beyond this relative drift the entrance length is
#: flagged unreliable rather than reported as a result.
ENTRANCE_DRIFT_LIMIT = 1e-3

#: Validated categorical palette (colour-vision-deficiency checked), in fixed
#: order, plus the neutral used for reference curves.
SERIES = ("#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#008300")
REFERENCE = "#333333"
MUTED = "#6B6B6B"
GRID = "#E6E6E6"


# --------------------------------------------------------------------------- #
# Configuration: every key required, unknown keys refused
# --------------------------------------------------------------------------- #


class ConfigError(ValueError):
    """A config that cannot describe a run unambiguously."""


#: The choices each named option may take.
CHOICES = {
    "physics.pressure_scaling": ("inertial", "viscous"),
    "physics.inlet_condition": ("soft", "hard"),
    "physics.formulation": ("velocity_pressure", "stream_pressure"),
    "physics.inlet_ramp": ("linear", "exponential"),
    "physics.lift": ("inlet", "blended"),
    "model.activation": ("tanh", "silu"),
    "model.input_map": ("linear", "exponential"),
    "sampling.inlet_points": ("random", "fixed"),
    "sampling.rar_mode": ("top", "proportional"),
    "training.adam.schedule": ("exponential", "cosine"),
}

#: Keys required under every choice. A leaf is the expected type; `float`
#: also accepts int.
SCHEMA = {
    "name": str,
    "seed": int,
    "physics": {"reynolds": float, "pressure_scaling": str, "inlet_sharpness": int,
                "inlet_condition": str, "formulation": str},
    "domain": {"length": float},
    "model": {"width": int, "depth": int, "activation": str, "init_gain": float,
              "input_map": str},
    "sampling": {
        "n_interior": int,
        "focus_fraction": float,
        "focus_length": float,
        "n_inlet": int,
        "n_outlet": int,
        "n_validation": int,
        "rar_points": int,
        "inlet_points": str,
    },
    "weights": {"momentum_x": float, "momentum_y": float, "outlet": float},
    "training": {
        "adam": {"steps": int, "learning_rate": float, "schedule": str, "resample_every": int},
        "lbfgs": {"cycles": int, "iterations_per_cycle": int, "gtol": float},
        "lm": {"iterations": int},
        "check_every": int,
        "init_from": str,
    },
    "evaluation": {"nx": int, "ny": int, "profile_stations": list, "n_flux_stations": int},
}

#: Keys required under some choices and REFUSED under the others, because a
#: key that has no effect is a setting nobody can see is being ignored.
CONDITIONAL = {
    ("sampling", "rar_pool"): int,          # rar_points > 0
    ("sampling", "rar_mode"): str,          # rar_points > 0
    ("sampling", "rar_exponent"): float,    # rar_mode = proportional
    ("sampling", "rar_floor"): float,       # rar_mode = proportional
    ("training.adam", "decay_rate"): float,           # schedule = exponential
    ("training.adam", "decay_steps"): int,            # schedule = exponential
    ("training.adam", "final_learning_rate"): float,  # schedule = cosine
    ("weights", "continuity"): float,       # formulation = velocity_pressure
    ("weights", "inlet"): float,            # inlet_condition = soft
    ("physics", "inlet_ramp"): str,          # inlet_condition = hard
    ("physics", "inlet_ramp_length"): float,  # inlet_ramp = exponential
    ("model", "input_length"): float,        # input_map = exponential
    ("physics", "lift"): str,                # formulation = stream_pressure or inlet = hard
    ("physics", "lift_length"): float,       # lift = blended
    ("training.lm", "damping"): float,        # lm.iterations > 0
    ("training.lm", "resample_every"): int,   # lm.iterations > 0
    ("training.lm", "average"): int,          # lm.iterations > 0
}


def _check_value(where, value, expected):
    if expected is float:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ConfigError(f"{where} must be a finite number, got {value!r}")
    elif expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer, got {value!r}")
    elif not isinstance(value, expected):
        raise ConfigError(f"{where} must be {expected.__name__}, got {value!r}")
    if where in CHOICES and value not in CHOICES[where]:
        raise ConfigError(f"{where} must be one of {list(CHOICES[where])}, got {value!r}")


def _check(document, schema, path=""):
    if not isinstance(document, dict):
        raise ConfigError(f"{path or 'config'} must be a mapping")
    for key, expected in schema.items():
        where = f"{path}.{key}" if path else key
        if key not in document:
            raise ConfigError(f"missing required key {where!r}")
        if isinstance(expected, dict):
            _check(document[key], expected, where)
        else:
            _check_value(where, document[key], expected)
    unknown = set(document) - set(schema)
    if unknown:
        raise ConfigError(f"unknown key(s) under {path or 'config'}: {sorted(unknown)}")


@dataclass(frozen=True)
class Config:
    name: str
    seed: int
    reynolds: float
    pressure_scaling: str
    inlet_sharpness: int
    inlet_condition: str
    inlet_ramp: str | None
    inlet_ramp_length: float | None
    formulation: str
    lift: str | None
    lift_length: float | None
    length: float
    width: int
    depth: int
    activation: str
    init_gain: float
    input_map: str
    input_length: float | None
    n_interior: int
    focus_fraction: float
    focus_length: float
    n_inlet: int
    n_outlet: int
    n_validation: int
    rar_points: int
    rar_pool: int | None
    rar_mode: str | None
    rar_exponent: float | None
    rar_floor: float | None
    inlet_points: str
    weights: dict
    loss_terms: tuple
    adam_steps: int
    adam_learning_rate: float
    adam_schedule: str
    adam_decay_rate: float | None
    adam_decay_steps: int | None
    adam_final_learning_rate: float | None
    adam_resample_every: int
    lbfgs_cycles: int
    lbfgs_iterations: int
    lbfgs_gtol: float
    lm_iterations: int
    lm_damping: float | None
    lm_resample_every: int | None
    lm_average: int | None
    init_from: str
    check_every: int
    nx: int
    ny: int
    profile_stations: tuple
    n_flux_stations: int

    @property
    def n_outputs(self) -> int:
        """Raw network outputs: (N_u, N_v, N_p), or (N_psi, N_p) for the stream function."""
        return 3 if self.formulation == "velocity_pressure" else 2


def _node(document, dotted):
    node = document
    for part in dotted.split("."):
        node = node[part]
    return node


def parse_config(document) -> Config:
    """Validate a config mapping. Raises ConfigError; never fills a default.

    Keys in CONDITIONAL are required exactly when the choice that uses them is
    made and refused otherwise, e.g. `sampling.rar_pool` only when
    `rar_points > 0`, `weights.inlet` only when the inlet is a soft condition.
    """
    if not isinstance(document, dict):
        raise ConfigError("config must be a mapping")
    view = copy.deepcopy(document)
    extras = {}
    for (section, key) in CONDITIONAL:
        try:
            container = _node(view, section)
        except (KeyError, TypeError):
            continue  # the missing section is reported by _check below
        if isinstance(container, dict) and key in container:
            extras[(section, key)] = container.pop(key)
    _check(view, SCHEMA)

    def take(section, key, needed, condition):
        where = f"{section}.{key}"
        present = (section, key) in extras
        if needed and not present:
            raise ConfigError(f"missing required key {where!r} (required when {condition})")
        if present and not needed:
            raise ConfigError(f"{where} has no effect unless {condition}; remove it")
        if not present:
            return None
        value = extras[(section, key)]
        _check_value(where, value, CONDITIONAL[(section, key)])
        return value

    physics, sampling = document["physics"], document["sampling"]
    adam, lbfgs = document["training"]["adam"], document["training"]["lbfgs"]
    lm = document["training"]["lm"]
    for station in document["evaluation"]["profile_stations"]:
        if (isinstance(station, bool) or not isinstance(station, (int, float))
                or not math.isfinite(station)):
            raise ConfigError(
                f"evaluation.profile_stations entries must be finite numbers, got {station!r}")
    evaluation = document["evaluation"]

    rar_on = sampling["rar_points"] > 0
    rar_pool = take("sampling", "rar_pool", rar_on, "rar_points > 0")
    rar_mode = take("sampling", "rar_mode", rar_on, "rar_points > 0")
    proportional = rar_mode == "proportional"
    rar_exponent = take("sampling", "rar_exponent", proportional, "rar_mode is 'proportional'")
    rar_floor = take("sampling", "rar_floor", proportional, "rar_mode is 'proportional'")
    exponential = adam["schedule"] == "exponential"
    decay_rate = take("training.adam", "decay_rate", exponential, "schedule is 'exponential'")
    decay_steps = take("training.adam", "decay_steps", exponential, "schedule is 'exponential'")
    final_lr = take("training.adam", "final_learning_rate", not exponential, "schedule is 'cosine'")
    with_continuity = physics["formulation"] == "velocity_pressure"
    soft_inlet = physics["inlet_condition"] == "soft"
    w_continuity = take("weights", "continuity", with_continuity,
                        "formulation is 'velocity_pressure' (the stream function "
                        "satisfies continuity identically)")
    w_inlet = take("weights", "inlet", soft_inlet,
                   "inlet_condition is 'soft' (a hard inlet holds exactly)")
    ramp = take("physics", "inlet_ramp", not soft_inlet, "inlet_condition is 'hard'")
    ramp_length = take("physics", "inlet_ramp_length", ramp == "exponential",
                       "inlet_ramp is 'exponential'")
    has_lift = physics["formulation"] == "stream_pressure" or not soft_inlet
    lift = take("physics", "lift", has_lift,
                "formulation is 'stream_pressure' or inlet_condition is 'hard' "
                "(a soft velocity inlet has no lift)")
    lift_length = take("physics", "lift_length", lift == "blended", "lift is 'blended'")
    lm_on = lm["iterations"] > 0
    lm_damping = take("training.lm", "damping", lm_on, "lm.iterations > 0")
    lm_resample = take("training.lm", "resample_every", lm_on, "lm.iterations > 0")
    lm_average = take("training.lm", "average", lm_on, "lm.iterations > 0")
    model = document["model"]
    input_length = take("model", "input_length", model["input_map"] == "exponential",
                        "input_map is 'exponential'")

    raw_weights = dict(document["weights"])
    raw_weights.pop("continuity", None)
    raw_weights.pop("inlet", None)
    if w_continuity is not None:
        raw_weights["continuity"] = w_continuity
    if w_inlet is not None:
        raw_weights["inlet"] = w_inlet
    loss_terms = tuple(term for term in LOSS_TERMS if term in raw_weights)

    config = Config(
        name=document["name"],
        seed=document["seed"],
        reynolds=float(physics["reynolds"]),
        pressure_scaling=physics["pressure_scaling"],
        inlet_sharpness=physics["inlet_sharpness"],
        inlet_condition=physics["inlet_condition"],
        inlet_ramp=ramp,
        inlet_ramp_length=None if ramp_length is None else float(ramp_length),
        formulation=physics["formulation"],
        lift=lift,
        lift_length=None if lift_length is None else float(lift_length),
        length=float(document["domain"]["length"]),
        width=model["width"],
        depth=model["depth"],
        activation=model["activation"],
        init_gain=float(model["init_gain"]),
        input_map=model["input_map"],
        input_length=None if input_length is None else float(input_length),
        n_interior=sampling["n_interior"],
        focus_fraction=float(sampling["focus_fraction"]),
        focus_length=float(sampling["focus_length"]),
        n_inlet=sampling["n_inlet"],
        n_outlet=sampling["n_outlet"],
        n_validation=sampling["n_validation"],
        rar_points=sampling["rar_points"],
        rar_pool=rar_pool,
        rar_mode=rar_mode,
        rar_exponent=None if rar_exponent is None else float(rar_exponent),
        rar_floor=None if rar_floor is None else float(rar_floor),
        inlet_points=sampling["inlet_points"],
        weights={term: float(raw_weights[term]) for term in loss_terms},
        loss_terms=loss_terms,
        adam_steps=adam["steps"],
        adam_learning_rate=float(adam["learning_rate"]),
        adam_schedule=adam["schedule"],
        adam_decay_rate=None if decay_rate is None else float(decay_rate),
        adam_decay_steps=decay_steps,
        adam_final_learning_rate=None if final_lr is None else float(final_lr),
        adam_resample_every=adam["resample_every"],
        lbfgs_cycles=lbfgs["cycles"],
        lbfgs_iterations=lbfgs["iterations_per_cycle"],
        lbfgs_gtol=float(lbfgs["gtol"]),
        lm_iterations=lm["iterations"],
        lm_damping=None if lm_damping is None else float(lm_damping),
        lm_resample_every=lm_resample,
        lm_average=lm_average,
        init_from=document["training"]["init_from"],
        check_every=document["training"]["check_every"],
        nx=evaluation["nx"],
        ny=evaluation["ny"],
        profile_stations=tuple(float(x) for x in evaluation["profile_stations"]),
        n_flux_stations=evaluation["n_flux_stations"],
    )

    if config.reynolds <= 0:
        raise ConfigError("physics.reynolds must be positive")
    if config.inlet_sharpness == 1:
        raise ConfigError(
            "physics.inlet_sharpness = 1 is the developed parabola: the flow "
            "would enter already developed. Use m >= 2."
        )
    if config.inlet_sharpness < 1:
        raise ConfigError("physics.inlet_sharpness must be an integer >= 2")
    if config.length <= 0:
        raise ConfigError("domain.length must be positive")
    if config.init_gain <= 0:
        raise ConfigError("model.init_gain must be positive")
    if config.inlet_ramp_length is not None and config.inlet_ramp_length <= 0:
        raise ConfigError("physics.inlet_ramp_length must be positive")
    if config.input_length is not None and config.input_length <= 0:
        raise ConfigError("model.input_length must be positive")
    if config.lift_length is not None and config.lift_length <= 0:
        raise ConfigError("physics.lift_length must be positive")
    if config.adam_learning_rate <= 0:
        raise ConfigError("training.adam.learning_rate must be positive")
    if exponential:
        if not 0.0 < config.adam_decay_rate <= 1.0:
            raise ConfigError("training.adam.decay_rate must lie in (0, 1]")
        if config.adam_decay_steps < 1:
            raise ConfigError("training.adam.decay_steps must be at least 1")
    elif not 0.0 < config.adam_final_learning_rate <= config.adam_learning_rate:
        raise ConfigError(
            "training.adam.final_learning_rate must lie in (0, learning_rate]")
    if config.lbfgs_gtol <= 0:
        raise ConfigError("training.lbfgs.gtol must be positive")
    for name in ("width", "depth", "n_interior", "n_inlet", "n_outlet",
                 "n_validation", "adam_resample_every",
                 "lbfgs_iterations", "check_every", "nx", "ny", "n_flux_stations"):
        if getattr(config, name) < 1:
            raise ConfigError(f"{name} must be at least 1")
    if config.rar_points < 0:
        raise ConfigError("sampling.rar_points must be >= 0 (0 turns adaptive refinement off)")
    if rar_on and rar_pool < config.rar_points:
        raise ConfigError("sampling.rar_pool must be at least sampling.rar_points")
    if proportional:
        if config.rar_exponent <= 0:
            raise ConfigError("sampling.rar_exponent must be positive")
        if config.rar_floor < 0:
            raise ConfigError("sampling.rar_floor must be >= 0")
    if config.adam_steps < 0:
        raise ConfigError("training.adam.steps must be >= 0")
    if config.lbfgs_cycles < 0:
        raise ConfigError("training.lbfgs.cycles must be >= 0")
    if not config.init_from or "/" in config.init_from or config.init_from in (".", ".."):
        raise ConfigError("training.init_from must be 'random' or the NAME of a finished run "
                          "in the same runs directory, not a path")
    if config.lm_iterations < 0:
        raise ConfigError("training.lm.iterations must be >= 0 (0 turns Levenberg-Marquardt off)")
    if lm_on and config.lm_damping <= 0:
        raise ConfigError("training.lm.damping must be positive")
    if lm_on and config.lm_resample_every < 1:
        raise ConfigError("training.lm.resample_every must be at least 1")
    if lm_on and (config.lm_average < 0 or config.lm_average == 1):
        raise ConfigError("training.lm.average must be 0 (off) or at least 2: the mean of the "
                          "last N iterates")
    if not 0.0 <= config.focus_fraction <= 1.0:
        raise ConfigError("sampling.focus_fraction must lie in [0, 1]")
    if not 0.0 < config.focus_length <= config.length:
        raise ConfigError("sampling.focus_length must lie in (0, domain.length]")
    if any(w < 0 for w in config.weights.values()):
        raise ConfigError("loss weights must be non-negative")
    stations = config.profile_stations
    if not 1 <= len(stations) <= len(SERIES):
        raise ConfigError(f"evaluation.profile_stations takes 1 to {len(SERIES)} stations")
    if any(not 0.0 <= x <= config.length for x in stations):
        raise ConfigError("every profile station must lie in [0, domain.length]")
    if config.n_flux_stations < 2:
        raise ConfigError("evaluation.n_flux_stations must be at least 2")
    return config


def load_config(path) -> Config:
    with open(path, encoding="utf-8") as handle:
        return parse_config(yaml.safe_load(handle))


# --------------------------------------------------------------------------- #
# Physics: pure functions of non-dimensional inputs
# --------------------------------------------------------------------------- #


def inlet_peak(sharpness):
    """U0 such that U0 (1 - y^(2m)) carries flux 4/3."""
    return (2.0 / 3.0) * (2 * sharpness + 1) / (2 * sharpness)


def inlet_profile(y, sharpness):
    """u*(0, y*) = U0 (1 - y^(2m)): zero at the walls, flat-topped for large m."""
    return inlet_peak(sharpness) * (1.0 - y ** (2 * sharpness))


def developed_profile(y, flux):
    """The fully developed parabola carrying `flux`: (3Q/4)(1 - y^2)."""
    return 0.75 * flux * (1.0 - y**2)


def correlation_entrance_length(reynolds_h):
    """Durst et al. (2005) entrance length, in half-heights, for a FLAT inlet."""
    re_dh = (8.0 / 3.0) * reynolds_h
    per_dh = (DURST_C0**DURST_N + (DURST_C1 * re_dh) ** DURST_N) ** (1.0 / DURST_N)
    return 4.0 * per_dh


class Derivatives(NamedTuple):
    """Values and the derivatives the residuals need. All non-dimensional."""

    u: object
    v: object
    p: object
    u_x: object
    u_y: object
    v_x: object
    v_y: object
    p_x: object
    p_y: object
    u_xx: object
    u_yy: object
    v_xx: object
    v_yy: object


def convective_x(u, v, u_x, u_y):
    """Inertia, x-component: (u . grad) u."""
    return u * u_x + v * u_y


def convective_y(u, v, v_x, v_y):
    """Inertia, y-component: (u . grad) v."""
    return u * v_x + v * v_y


def laplacian(f_xx, f_yy):
    return f_xx + f_yy


def divergence(u_x, v_y):
    return u_x + v_y


def scaling_coefficients(reynolds, pressure_scaling):
    """(inertia coefficient, viscous coefficient) for the pressure scaling."""
    if pressure_scaling == "inertial":
        return 1.0, 1.0 / reynolds
    if pressure_scaling == "viscous":
        return reynolds, 1.0
    raise ValueError(f"unknown pressure_scaling {pressure_scaling!r}")


def momentum_x(d: Derivatives, reynolds, pressure_scaling):
    """x-momentum residual, unweighted."""
    inertia, viscous = scaling_coefficients(reynolds, pressure_scaling)
    return inertia * convective_x(d.u, d.v, d.u_x, d.u_y) + d.p_x - viscous * laplacian(d.u_xx, d.u_yy)


def momentum_y(d: Derivatives, reynolds, pressure_scaling):
    """y-momentum residual, unweighted."""
    inertia, viscous = scaling_coefficients(reynolds, pressure_scaling)
    return inertia * convective_y(d.u, d.v, d.v_x, d.v_y) + d.p_y - viscous * laplacian(d.v_xx, d.v_yy)


def continuity(d: Derivatives):
    """Continuity residual, unweighted."""
    return divergence(d.u_x, d.v_y)


# --------------------------------------------------------------------------- #
# Derivatives by automatic differentiation
# --------------------------------------------------------------------------- #

_EX = jnp.array([1.0, 0.0])
_EY = jnp.array([0.0, 1.0])


def _assemble(value, d_dx, d_dy, d2_dx2, d2_dy2) -> Derivatives:
    return Derivatives(
        u=value[..., 0], v=value[..., 1], p=value[..., 2],
        u_x=d_dx[..., 0], v_x=d_dx[..., 1], p_x=d_dx[..., 2],
        u_y=d_dy[..., 0], v_y=d_dy[..., 1], p_y=d_dy[..., 2],
        u_xx=d2_dx2[..., 0], v_xx=d2_dx2[..., 1],
        u_yy=d2_dy2[..., 0], v_yy=d2_dy2[..., 1],
    )


def point_derivatives(field_fn, xy) -> Derivatives:
    """Reference implementation at ONE point, by nested forward mode."""
    jacobian = jax.jacfwd(field_fn)(xy)
    hessian = jax.jacfwd(jax.jacfwd(field_fn))(xy)
    return _assemble(field_fn(xy), jacobian[:, 0], jacobian[:, 1],
                     hessian[:, 0, 0], hessian[:, 1, 1])


def _directional(batched_fn, points, direction):
    """(value, first, second) derivative of a batched field along `direction`."""
    tangents = jnp.broadcast_to(direction, points.shape)

    def value_and_first(pts):
        return jax.jvp(batched_fn, (pts,), (tangents,))

    (value, first), (_, second) = jax.jvp(value_and_first, (points,), (tangents,))
    return value, first, second


def batched_derivatives(field_fn, points) -> Derivatives:
    """The derivative bundle for an (N, 2) array of points.

    Differentiates the already-batched field along constant directions rather
    than vmapping the single-point version: each output row depends only on its
    own input row, so one directional pass gives every point's derivative, and
    it is about 2.4x cheaper under reverse-mode training.
    """
    batched_fn = jax.vmap(field_fn)
    value, d_dx, d2_dx2 = _directional(batched_fn, points, _EX)
    _, d_dy, d2_dy2 = _directional(batched_fn, points, _EY)
    return _assemble(value, d_dx, d_dy, d2_dx2, d2_dy2)


def batched_x_derivative(field_fn, points):
    """(values, d/dx) for an (N, 2) array. Enough for the outlet condition."""
    batched_fn = jax.vmap(field_fn)
    return jax.jvp(batched_fn, (points,), (jnp.broadcast_to(_EX, points.shape),))


# --------------------------------------------------------------------------- #
# Network and the boundary-condition ansatz
# --------------------------------------------------------------------------- #


def _glorot(key, n_out, n_in, gain):
    return gain * jnp.sqrt(2.0 / (n_in + n_out)) * jax.random.normal(
        key, (n_out, n_in), dtype=jnp.float64)


def _dense(key, n_in, n_out, gain):
    return {"W": _glorot(key, n_out, n_in, gain), "b": jnp.zeros((n_out,), dtype=jnp.float64)}


def init_network(key, width, depth, n_out, gain):
    """Modified MLP (Wang, Teng & Perdikaris 2021): two input encoders blended
    into every hidden layer. Inputs (x, y); `n_out` raw outputs.

    Glorot-normal weights, scaled by `gain` on every layer an activation
    follows (encoders and hidden layers) but not on the linear output layer.
    Gain 1 is plain Glorot, right for tanh, whose slope at zero is 1. SiLU's
    slope at zero is 1/2, so under gain 1 every layer halves the signal: a
    SiLU network starts with half tanh's spatial variation and 12-32x less
    curvature. Gain 2 = 1/silu'(0) gives it the same linearisation as the
    tanh network at initialisation, so the two differ only in nonlinearity.
    """
    keys = jax.random.split(key, depth + 3)
    return {
        "encoder_u": _dense(keys[0], 2, width, gain),
        "encoder_v": _dense(keys[1], 2, width, gain),
        "hidden": [_dense(keys[2], 2, width, gain)]
        + [_dense(keys[3 + i], width, width, gain) for i in range(depth - 1)],
        "output": _dense(keys[depth + 2], width, n_out, 1.0),
    }


def init_params(config: Config, key):
    """The network a config describes, initialised from `key`."""
    return init_network(key, config.width, config.depth, config.n_outputs, config.init_gain)


#: Activation functions by config name. Applied everywhere in the modified
#: MLP, as in Wang, Teng & Perdikaris (2021), who define it for a generic one.
ACTIVATIONS = {"tanh": jnp.tanh, "silu": jax.nn.silu}


def _apply_dense(layer, x):
    return layer["W"] @ x + layer["b"]


def hidden_features(params, xi, activation):
    """The last hidden layer: the basis functions the output layer combines."""
    sigma = ACTIVATIONS[activation]
    encoded_u = sigma(_apply_dense(params["encoder_u"], xi))
    encoded_v = sigma(_apply_dense(params["encoder_v"], xi))
    h = xi
    for layer in params["hidden"]:
        z = sigma(_apply_dense(layer, h))
        h = (1.0 - z) * encoded_u + z * encoded_v
    return h


def apply_network(params, xi, activation):
    return _apply_dense(params["output"], hidden_features(params, xi, activation))


def wall_factor(y):
    """phi(y*) = 1 - y*^2: zero on both walls, positive between them."""
    return 1.0 - y**2


def inlet_stream(y, sharpness):
    """The stream function of the inlet profile: d/dy of it is inlet_profile.

    U0 (y - y^(2m+1)/(2m+1)). Its values at y = +-1 differ by exactly the flux
    4/3, and its slope vanishes at both walls, so used as the lift of a stream
    function it carries the flux and satisfies no-slip for any network output.
    """
    return inlet_peak(sharpness) * (y - y ** (2 * sharpness + 1) / (2 * sharpness + 1))


def developed_stream(y):
    """The stream function of the developed parabola 1 - y^2: y - y^3/3.

    Like inlet_stream it is +-2/3 at the walls, so the two differ by the same
    flux, 4/3, and any blend of them is a valid lift.
    """
    return y - y**3 / 3.0


def lift_blend(x, length):
    """s(x) = (1 - exp(-x/l))^2: 0 at the inlet with zero slope there, -> 1 downstream."""
    return jnp.expm1(-x / length) ** 2


def lift_stream(x, y, sharpness, lift, lift_length):
    """The stream function the network corrects: what the flow is assumed to
    be before the network says anything.

    inlet    G_in(y), the inlet profile carried unchanged down the whole
             channel. Downstream the network must then produce the steep
             near-wall difference between the parabola and the y^(2m) inlet,
             at every x, just to express developed flow.
    blended  (1 - s) G_in(y) + s G_dev(y), s = (1 - exp(-x/l))^2: the inlet at
             the inlet, the developed parabola downstream, so the network
             learns only the departure from it, which is small everywhere and
             tends to zero where the flow is developed. s'(0) = 0 keeps
             v(0, y) = 0. It is divergence-free, carries flux 4/3 at every x
             and satisfies no-slip, since both G's do.

    Because the blended lift contains the exact developed solution, accuracy
    in the developed region partly measures the lift; only the entrance region
    measures the method.
    """
    if lift == "inlet":
        return inlet_stream(y, sharpness)
    if lift == "blended":
        s = lift_blend(x, lift_length)
        return (1.0 - s) * inlet_stream(y, sharpness) + s * developed_stream(y)
    raise ValueError(f"unknown lift {lift!r}")


def make_field(config: Config):
    """field(params, xy) -> (u, v, p) at one point, for the config's ansatz.

    Cached per ansatz, so the same config always yields the same function
    object and anything compiled against it is reused.
    """
    return _make_field(config.length, config.activation, config.formulation,
                       config.inlet_condition, config.inlet_sharpness, config.inlet_ramp,
                       config.inlet_ramp_length, config.input_map, config.input_length,
                       config.lift, config.lift_length)


def inlet_ramp(x, length, ramp, ramp_length):
    """r(x): zero at the inlet, so a hard inlet holds whatever the network does.

    linear       x / L. Its slope at the inlet is 1/L = 0.1, so near the inlet
                 the network must output ~L times the correction it makes: at
                 x* = 0.1, where the flow changes fastest, r = 0.01. This was
                 E5, which failed at the inlet corners.
    exponential  1 - exp(-x / l): slope 1/l at the inlet, tending to 1 over a
                 few l. With l = 1 it is normalised to first order like a
                 distance function (Sukumar & Srivastava 2022), so the network
                 output near the inlet is the correction itself.
    """
    if ramp == "linear":
        return x / length
    if ramp == "exponential":
        return -jnp.expm1(-x / ramp_length)
    raise ValueError(f"unknown inlet_ramp {ramp!r}")


def input_coordinates(xy, length, input_map, input_length):
    """The point as the network sees it, in [-1, 1]^2.

    linear       xi = 2x/L - 1. With L = 10 the network sees x compressed 5x
                 relative to y, so the entrance region, the only place the
                 flow varies in x, occupies a narrow band of its input.
    exponential  xi = 2 (1 - exp(-x/l)) / (1 - exp(-L/l)) - 1. Deviations from
                 developed flow decay like exp(-lambda x), so this spends the
                 network's input range where the flow changes and compresses
                 the developed region, where it does not. With l = 2 the map's
                 slope at the inlet is 1, the same as y's: isotropic there.
    """
    x, y = xy[0], xy[1]
    if input_map == "linear":
        return jnp.stack([2.0 * x / length - 1.0, y])
    if input_map == "exponential":
        return jnp.stack([2.0 * jnp.expm1(-x / input_length) / jnp.expm1(-length / input_length)
                          - 1.0, y])
    raise ValueError(f"unknown input_map {input_map!r}")


@functools.lru_cache(maxsize=None)
def _make_field(length, activation, formulation, inlet_condition, sharpness,
                ramp, ramp_length, input_map, input_length, lift, lift_length):
    """The boundary-condition ansatz. In every variant:

    * no-slip holds EXACTLY on both walls, for any parameters;
    * the pressure gauge p(L, y) = 0 holds exactly: p = ((L - x)/L) N_p. It is
      written (L - x)/L, not 1 - x/L, because under jit x/L may become
      x * (1/L), and 1 - L*(1/L) is ~1e-17 rather than zero;
    * inputs are mapped to [-1, 1]^2 before entering the network
      (`input_coordinates`).

    velocity_pressure, soft inlet:  u = phi N_u,  v = phi N_v
    velocity_pressure, hard inlet:  u = u_0 + phi r(x) N_u,  v = v_0 + phi r(x) N_v
        with r the inlet ramp (`inlet_ramp`) and (u_0, v_0) the lift's
        velocity: (g(y), 0), g the inlet profile, for the 'inlet' lift, or
        the curl of the blended lift stream function. u(0, y) = g(y) and
        v(0, y) = 0 exactly either way.
    stream_pressure:  u = d(psi)/dy,  v = -d(psi)/dx,  continuity holds identically;
        soft inlet:  psi = G(x, y) + phi^2 N_psi
        hard inlet:  psi = G(x, y) + phi^2 r(x)^2 N_psi
        with G the lift (`lift_stream`). The squared ramp makes BOTH
        u(0, y) = g(y) and v(0, y) = 0 exact. psi(x, 1) - psi(x, -1) = 4/3 for
        every x, so the flux is exact by construction and cannot be tested.
    """
    L = length

    def inputs(xy):
        return input_coordinates(xy, L, input_map, input_length)

    def gauge(x):
        return (L - x) / L

    if formulation == "velocity_pressure":
        def field(params, xy):
            x, y = xy[0], xy[1]
            raw = apply_network(params, inputs(xy), activation)
            wall = wall_factor(y)
            if inlet_condition == "soft":
                u, v = wall * raw[0], wall * raw[1]
            else:
                r = inlet_ramp(x, L, ramp, ramp_length)
                if lift == "inlet":
                    u0, v0 = inlet_profile(y, sharpness), 0.0
                else:
                    slope = jax.grad(lambda point: lift_stream(
                        point[0], point[1], sharpness, lift, lift_length))(xy)
                    u0, v0 = slope[1], -slope[0]
                u = u0 + wall * r * raw[0]
                v = v0 + wall * r * raw[1]
            return jnp.stack([u, v, gauge(x) * raw[2]])
        return field

    if formulation == "stream_pressure":
        def field(params, xy):
            def psi(point):
                x, y = point[0], point[1]
                raw = apply_network(params, inputs(point), activation)
                r2 = 1.0 if inlet_condition == "soft" else inlet_ramp(x, L, ramp, ramp_length) ** 2
                return (lift_stream(x, y, sharpness, lift, lift_length)
                        + wall_factor(y) ** 2 * r2 * raw[0]), raw

            gradient, raw = jax.grad(psi, has_aux=True)(xy)
            return jnp.stack([gradient[1], -gradient[0], gauge(xy[0]) * raw[1]])
        return field

    raise ValueError(f"unknown formulation {formulation!r}")


# --------------------------------------------------------------------------- #
# Collocation points
# --------------------------------------------------------------------------- #


class Batch(NamedTuple):
    interior: jnp.ndarray
    inlet: jnp.ndarray
    outlet: jnp.ndarray


def sample_interior(key, config: Config, n: int):
    """Uniform points plus a share concentrated where the flow develops.

    Focus points: x = focus_length * s^2 crowds toward the inlet, and
    y = sin(pi t / 2) crowds toward the walls, where the entrance boundary
    layers are thin and steep.
    """
    n_focus = int(round(n * config.focus_fraction))
    n_uniform = n - n_focus
    k_ux, k_uy, k_fx, k_fy = jax.random.split(key, 4)
    uniform = jnp.stack([
        jax.random.uniform(k_ux, (n_uniform,), minval=0.0, maxval=config.length, dtype=jnp.float64),
        jax.random.uniform(k_uy, (n_uniform,), minval=-1.0, maxval=1.0, dtype=jnp.float64),
    ], axis=-1)
    s = jax.random.uniform(k_fx, (n_focus,), dtype=jnp.float64)
    t = jax.random.uniform(k_fy, (n_focus,), minval=-1.0, maxval=1.0, dtype=jnp.float64)
    focus = jnp.stack([config.focus_length * s**2, jnp.sin(0.5 * jnp.pi * t)], axis=-1)
    return jnp.concatenate([uniform, focus], axis=0)


def sample_batch(key, config: Config, n_interior: int) -> Batch:
    """One draw of training (or held-out) points. A pure function of `key`."""
    k_int, k_in, k_out = jax.random.split(key, 3)
    # Inlet y crowds toward the corners, where the inlet profile turns sharply.
    t = jax.random.uniform(k_in, (config.n_inlet,), minval=-1.0, maxval=1.0, dtype=jnp.float64)
    y_in = jnp.sin(0.5 * jnp.pi * t)
    y_out = jax.random.uniform(k_out, (config.n_outlet,), minval=-1.0, maxval=1.0, dtype=jnp.float64)
    return Batch(
        interior=sample_interior(k_int, config, n_interior),
        inlet=jnp.stack([jnp.zeros_like(y_in), y_in], axis=-1),
        outlet=jnp.stack([jnp.full_like(y_out, config.length), y_out], axis=-1),
    )


def sample_uniform(key, config: Config, n: int):
    """n points uniform over the whole channel: the RAR candidate pool."""
    k_x, k_y = jax.random.split(key)
    return jnp.stack([
        jax.random.uniform(k_x, (n,), minval=0.0, maxval=config.length, dtype=jnp.float64),
        jax.random.uniform(k_y, (n,), minval=-1.0, maxval=1.0, dtype=jnp.float64),
    ], axis=-1)


def pointwise_residual(config: Config, field, params, points):
    """R_x^2 + R_y^2 + R_c^2 at each point: where the equations are worst broken.

    Unweighted on purpose: this ranks locations for refinement, it is not a
    loss, so it does not touch the loss weights.
    """
    d = batched_derivatives(lambda xy: field(params, xy), points)
    return (momentum_x(d, config.reynolds, config.pressure_scaling) ** 2
            + momentum_y(d, config.reynolds, config.pressure_scaling) ** 2
            + continuity(d) ** 2)


def fixed_inlet_points(config: Config):
    """A static inlet grid: n_inlet points crowded toward both corners.

    y = sin(pi t / 2) on a uniform t grid, endpoints excluded (the walls, where
    u = v = 0 holds exactly anyway). Used for every training draw when
    sampling.inlet_points is 'fixed'.
    """
    t = jnp.linspace(-1.0, 1.0, config.n_inlet + 2, dtype=jnp.float64)[1:-1]
    y = jnp.sin(0.5 * jnp.pi * t)
    return jnp.stack([jnp.zeros_like(y), y], axis=-1)


def make_draw(config: Config, field):
    """draw(key, params) -> a training batch.

    The ordinary draw is `sample_batch(key, ...)`. Two optional changes, each
    only when its config asks for it, so a config using neither draws exactly
    the points it always did:

    * sampling.inlet_points = 'fixed': the inlet points are replaced by one
      static grid for every draw (`fixed_inlet_points`).
    * rar_points > 0, residual-based adaptive refinement: a fresh pool of
      `rar_pool` uniform points, never trained on, is scored by the current
      residual, and `rar_points` of them are ADDED to the interior:
        'top'           the worst ones;
        'proportional'  sampled without replacement with probability
                        proportional to score^k / mean(score^k) + c, with
                        k = rar_exponent and c = rar_floor (the RAD scheme of
                        Wu et al. 2023), so points concentrate where the
                        residual is large without all landing on the maximum.
    The HELD-OUT set never goes through this: it is always `sample_batch`.
    """
    fixed_inlet = fixed_inlet_points(config) if config.inlet_points == "fixed" else None
    score = (jax.jit(lambda p, pts: pointwise_residual(config, field, p, pts))
             if config.rar_points else None)

    def draw(key, params):
        batch = sample_batch(key, config, config.n_interior)
        if fixed_inlet is not None:
            batch = batch._replace(inlet=fixed_inlet)
        if score is None:
            return batch
        pool = sample_uniform(jax.random.fold_in(key, 1), config, config.rar_pool)
        values = score(params, pool)
        if config.rar_mode == "top":
            _, chosen = jax.lax.top_k(values, config.rar_points)
        else:
            weighted = values ** config.rar_exponent
            density = weighted / jnp.mean(weighted) + config.rar_floor
            chosen = jax.random.choice(jax.random.fold_in(key, 2), config.rar_pool,
                                       shape=(config.rar_points,), replace=False,
                                       p=density / jnp.sum(density))
        return batch._replace(interior=jnp.concatenate([batch.interior, pool[chosen]], axis=0))

    return draw


def root_keys(seed):
    """(model key, training-sample key, held-out key) from one seed."""
    return jax.random.split(jax.random.PRNGKey(seed), 3)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def make_loss_terms(config: Config, field):
    """loss_terms(params, batch) -> {term: unweighted mean-squared residual}."""
    reynolds, scaling, sharpness = config.reynolds, config.pressure_scaling, config.inlet_sharpness

    active = config.loss_terms

    def loss_terms(params, batch: Batch):
        f = lambda xy: field(params, xy)  # noqa: E731
        d = batched_derivatives(f, batch.interior)
        _, outlet_dx = batched_x_derivative(f, batch.outlet)
        terms = {
            "momentum_x": jnp.mean(momentum_x(d, reynolds, scaling) ** 2),
            "momentum_y": jnp.mean(momentum_y(d, reynolds, scaling) ** 2),
            "outlet": jnp.mean(outlet_dx[:, 0] ** 2) + jnp.mean(outlet_dx[:, 1] ** 2),
        }
        # Continuity is identically satisfied by the stream function, and a
        # hard inlet holds exactly: those terms are not in the loss at all.
        if "continuity" in active:
            terms["continuity"] = jnp.mean(continuity(d) ** 2)
        if "inlet" in active:
            inlet_values = jax.vmap(f)(batch.inlet)
            target = inlet_profile(batch.inlet[:, 1], sharpness)
            terms["inlet"] = (jnp.mean((inlet_values[:, 0] - target) ** 2)
                              + jnp.mean(inlet_values[:, 1] ** 2))
        return terms

    return loss_terms


def make_point_residuals(config: Config, field):
    """The loss's residuals one point at a time, scaled for least squares.

    Returns three functions (interior, inlet, outlet), each
    f(params, xy, n) -> the scaled residuals at one point of a group of n
    points, such that summing their squares over a batch gives exactly the
    weighted loss of `make_loss_terms` + `total_loss` (up to the ~1e-12
    difference between point and batched derivatives). Levenberg-Marquardt
    differentiates these with respect to the parameters, one point at a time.
    `inlet` is None when the inlet is not a loss term.
    """
    reynolds, scaling, sharpness = config.reynolds, config.pressure_scaling, config.inlet_sharpness
    active, w = config.loss_terms, config.weights

    def interior(params, xy, n):
        d = point_derivatives(lambda q: field(params, q), xy)
        values = [jnp.sqrt(w["momentum_x"] / n) * momentum_x(d, reynolds, scaling),
                  jnp.sqrt(w["momentum_y"] / n) * momentum_y(d, reynolds, scaling)]
        if "continuity" in active:
            values.append(jnp.sqrt(w["continuity"] / n) * continuity(d))
        return jnp.stack(values)

    def inlet(params, xy, n):
        value = field(params, xy)
        scale = jnp.sqrt(w["inlet"] / n)
        return scale * jnp.stack([value[0] - inlet_profile(xy[1], sharpness), value[1]])

    def outlet(params, xy, n):
        _, d_dx = jax.jvp(lambda q: field(params, q), (xy,), (_EX,))
        scale = jnp.sqrt(w["outlet"] / n)
        return scale * jnp.stack([d_dx[0], d_dx[1]])

    return interior, (inlet if "inlet" in active else None), outlet


#: Nielsen's damping update for Levenberg-Marquardt (Nielsen 1999; Madsen,
#: Nielsen & Tingleff 2004, eq. 2.21): after an accepted step with gain ratio
#: rho, lambda *= max(1/3, 1 - (2 rho - 1)^3); after a rejected one lambda *=
#: nu and nu doubles. Part of the algorithm's definition, not a tuning knob.
LM_MAX_ATTEMPTS = 30


def make_gauss_newton(config: Config, field, unflatten, chunk=512):
    """(normal_equations, loss) for Levenberg-Marquardt on flat parameters.

    normal_equations(theta, batch) -> (J^T J, J^T r, r.r), accumulated over
    chunks of points so the full Jacobian J is never held at once.
    loss(theta, batch) -> r.r only, for testing a trial step.
    """
    interior, inlet, outlet = make_point_residuals(config, field)

    def groups(batch):
        out = [(interior, batch.interior), (outlet, batch.outlet)]
        if inlet is not None:
            out.append((inlet, batch.inlet))
        return out

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def block(fn, theta, points, n):
        def rows(vector, xy):
            value = fn(unflatten(vector), xy, n)
            return value, value

        jac, value = jax.vmap(jax.jacrev(rows, has_aux=True), in_axes=(None, 0))(theta, points)
        jac = jac.reshape(-1, theta.shape[0])
        value = value.reshape(-1)
        return jac.T @ jac, jac.T @ value, value @ value

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def values(fn, theta, points, n):
        return jax.vmap(lambda xy: fn(unflatten(theta), xy, n))(points).reshape(-1)

    def normal_equations(theta, batch):
        size = theta.shape[0]
        A = jnp.zeros((size, size))
        g = jnp.zeros(size)
        rr = 0.0
        for fn, points in groups(batch):
            n = int(points.shape[0])
            for start in range(0, n, chunk):
                a, b, c = block(fn, theta, points[start:start + chunk], n)
                A, g, rr = A + a, g + b, rr + c
        return A, g, float(rr)

    def loss(theta, batch):
        total = 0.0
        for fn, points in groups(batch):
            n = int(points.shape[0])
            for start in range(0, n, chunk):
                r = values(fn, theta, points[start:start + chunk], n)
                total += float(r @ r)
        return total

    return normal_equations, loss


def levenberg_marquardt_step(A, g, loss_now, trial_loss, lam, nu):
    """One accepted Levenberg-Marquardt step, retrying with more damping.

    Solves (A + lam I) delta = -g by Cholesky, accepts when the actual
    reduction is positive, and updates lam by Nielsen's rule. Returns
    (delta or None, new loss, lam, nu, attempts); delta is None when no
    damping in LM_MAX_ATTEMPTS attempts reduced the loss.
    """
    import scipy.linalg

    A = np.asarray(A)
    g = np.asarray(g)
    diagonal = np.diag_indices_from(A)
    for attempt in range(1, LM_MAX_ATTEMPTS + 1):
        damped = A.copy()
        damped[diagonal] += lam
        factor = scipy.linalg.cho_factor(damped, lower=False, overwrite_a=True)
        delta = scipy.linalg.cho_solve(factor, -g)
        predicted = -(2.0 * g @ delta + delta @ (A @ delta))
        new_loss = trial_loss(delta)
        if predicted > 0 and new_loss < loss_now:
            rho = (loss_now - new_loss) / predicted
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            return delta, new_loss, lam, 2.0, attempt
        lam *= nu
        nu *= 2.0
    return None, loss_now, lam, nu, LM_MAX_ATTEMPTS


def total_loss(terms, weights):
    """The ONLY place weights meet residuals.

    Terms and weights must name the same set; a mismatch means the config's
    active terms and the loss builder have drifted apart. Summed in the fixed
    LOSS_TERMS order so the result does not depend on dict order.
    """
    if set(terms) != set(weights):
        raise KeyError(f"loss terms and weights disagree: {sorted(set(terms) ^ set(weights))}")
    return sum(weights[name] * terms[name] for name in LOSS_TERMS if name in terms)


# --------------------------------------------------------------------------- #
# Training: Adam with resampling, then L-BFGS in cycles on fresh points
# --------------------------------------------------------------------------- #


def adam_schedule(config: Config):
    """Learning rate against Adam step: exponential decay, or cosine annealing
    from learning_rate to final_learning_rate over all Adam steps."""
    if config.adam_schedule == "exponential":
        return optax.exponential_decay(
            config.adam_learning_rate, config.adam_decay_steps, config.adam_decay_rate)
    return optax.cosine_decay_schedule(
        config.adam_learning_rate, config.adam_steps,
        alpha=config.adam_final_learning_rate / config.adam_learning_rate)


#: What a warm start may NOT change: the network and the ansatz it feeds, so
#: the parent's parameters mean the same function. The Reynolds number, the
#: sampling, the weights and the training schedule may all change (a
#: curriculum in Re, Krishnapriyan et al. 2021, changes exactly the first).
WARM_START_FIXED = ("width", "depth", "activation", "input_map", "input_length", "formulation",
                    "inlet_condition", "inlet_ramp", "inlet_ramp_length", "lift", "lift_length",
                    "length", "pressure_scaling", "inlet_sharpness")


def warm_start_params(config: Config, parent_directory):
    """(params, sha256 of params.npz) of the finished run `config` starts from."""
    import hashlib

    parent_directory = pathlib.Path(parent_directory)
    if not (parent_directory / "params.npz").is_file():
        raise ConfigError(f"training.init_from = {config.init_from!r}: {parent_directory} is not a "
                          "finished run (no params.npz)")
    parent = load_config(parent_directory / "config.yaml")
    for name in WARM_START_FIXED:
        if getattr(parent, name) != getattr(config, name):
            raise ConfigError(
                f"training.init_from = {config.init_from!r}: {name} is {getattr(parent, name)!r} "
                f"there and {getattr(config, name)!r} here; a warm start keeps the network and "
                "its ansatz")
    # The seed fixes the held-out points, and a run's held-out points chose
    # its network (the minimum over hundreds of noisy iterates: G4's chosen
    # network scored 1.38e-8 on them and 2.22e-8 on fresh points). Reusing an
    # ancestor's seed would score the starting network with that bias, and
    # training would have to beat a lucky number. So the whole chain differs.
    seeds, ancestor, where = {}, parent, parent_directory
    while True:
        if ancestor.name in seeds.values():
            raise ConfigError(f"training.init_from: the chain of starting runs loops at "
                              f"{ancestor.name!r}")
        seeds.setdefault(ancestor.seed, ancestor.name)
        if ancestor.init_from == "random":
            break
        where = where.parent / ancestor.init_from
        if not (where / "config.yaml").is_file():
            raise ConfigError(f"training.init_from: {ancestor.name!r} started from "
                              f"{ancestor.init_from!r}, which is not in {where.parent}; the "
                              "seeds of the whole chain cannot be checked")
        ancestor = load_config(where / "config.yaml")
    if config.seed in seeds:
        raise ConfigError(
            f"seed {config.seed} is also the seed of {seeds[config.seed]!r}, which this run "
            "starts from: its held-out points selected that network and would score the start "
            f"with a bias. Use a seed not in {sorted(seeds)}.")
    raw = (parent_directory / "params.npz").read_bytes()
    _, unflatten = ravel_pytree(init_params(config, root_keys(config.seed)[0]))
    flat = np.load(parent_directory / "params.npz")["flat"]
    return unflatten(jnp.asarray(flat)), hashlib.sha256(raw).hexdigest()


#: Bumped whenever the training state written at checkpoints changes meaning.
TRAINING_STATE_VERSION = 1


def _config_fingerprint(config: Config) -> str:
    import dataclasses
    import hashlib

    text = json.dumps(dataclasses.asdict(config), sort_keys=True, default=repr)
    return hashlib.sha256(text.encode()).hexdigest()


def _code_fingerprint() -> str:
    """This file's bytes plus the libraries that compute the arithmetic: a
    resumed run must continue on the SAME code, or it is not one run."""
    import hashlib
    import scipy

    digest = hashlib.sha256(pathlib.Path(__file__).read_bytes())
    digest.update(f"jax {jax.__version__} optax {optax.__version__} scipy {scipy.__version__} "
                  f"numpy {np.__version__}".encode())
    return digest.hexdigest()


def save_training_state(path, arrays, meta):
    """Atomically write the state train() needs to continue from a boundary
    (the end of an Adam check, of an L-BFGS cycle, or of an LM step)."""
    temporary = pathlib.Path(path).with_suffix(".tmp.npz")
    np.savez_compressed(temporary, meta=np.array(json.dumps(meta)), **arrays)
    temporary.replace(path)


def load_training_state(path, config: Config):
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"cannot resume: {path} does not exist")
    with np.load(path) as saved:
        if "meta" not in saved:
            raise ValueError(f"cannot resume: {path} holds no training state (it was written "
                             "before runs could be resumed, and records only the best parameters)")
        meta = json.loads(str(saved["meta"]))
        arrays = {name: saved[name] for name in saved.files if name != "meta"}
    for name, expected in (("version", TRAINING_STATE_VERSION),
                           ("config", _config_fingerprint(config)),
                           ("code", _code_fingerprint())):
        if meta[name] != expected:
            raise ValueError(f"cannot resume from {path}: its {name} fingerprint differs from "
                             "this run's, so continuing would not reproduce one uninterrupted run")
    return arrays, meta


def train(config: Config, verbose: bool = True, checkpoint=None, initial=None, resume=False):
    """Returns (best params, history rows, info dict).

    `initial` is the network to start from when config.init_from names a run
    (run() loads and checks it); with init_from = 'random' it must be None.

    With a `checkpoint` path, the full training state is written there at every
    boundary: each Adam check, each completed L-BFGS cycle, each LM step. With
    `resume`, training continues from that state and reproduces the
    uninterrupted run bitwise (the tests check it); only wall-clock columns
    differ. Mid-cycle L-BFGS progress is not resumable and is redone.
    """
    if resume and checkpoint is None:
        raise ValueError("resume needs the checkpoint to resume from")
    field = make_field(config)
    model_key, sample_key, heldout_key = root_keys(config.seed)
    params = init_params(config, model_key)
    if config.init_from == "random":
        if initial is not None:
            raise ValueError("initial parameters given, but training.init_from is 'random'")
    else:
        if initial is None:
            raise ValueError(f"training.init_from = {config.init_from!r}: pass that run's "
                             "parameters as `initial` (run() does)")
        params = initial
    loss_terms = make_loss_terms(config, field)
    weights = {name: jnp.asarray(config.weights[name], dtype=jnp.float64)
               for name in config.loss_terms}

    def objective(p, batch):
        terms = loss_terms(p, batch)
        return total_loss(terms, weights), terms

    heldout_batch = sample_batch(heldout_key, config, config.n_validation)
    draw_batch = make_draw(config, field)
    heldout_loss = jax.jit(lambda p: objective(p, heldout_batch)[0])
    train_terms = jax.jit(objective)
    flat, unflatten = ravel_pytree(params)

    best = {"heldout": math.inf, "params": params, "where": "initial"}
    history = []
    cycle_log = []
    lm_log = {"iterations": 0, "rejected": 0, "stalled": 0}
    resumed_from = []
    started = time.time()
    key = sample_key
    position, saved = ("start", 0), {}
    if resume:
        saved, meta = load_training_state(checkpoint, config)
        position = (meta["phase"], meta["done"])
        best = {"heldout": meta["best_heldout"], "where": meta["best_where"],
                "params": unflatten(jnp.asarray(saved["best_flat"]))}
        history, cycle_log, lm_log = meta["history"], meta["cycle_log"], meta["lm_log"]
        resumed_from = meta["resumed_from"] + [f"{position[0]}:{position[1]}"]
        started -= meta["elapsed"]
        key = jnp.asarray(saved["key"])
        flat = jnp.asarray(saved["flat"])
        params = unflatten(flat)
        if verbose:
            print(f"  resuming after {position[0]} {position[1]} "
                  f"({meta['elapsed'] / 60:.1f} min already trained)", flush=True)

    def checkpoint_state(phase, done, current, batch=None, extra_arrays=None, **extra):
        if checkpoint is None:
            return
        arrays = {"key": np.asarray(key), "flat": np.asarray(current),
                  "best_flat": np.asarray(ravel_pytree(best["params"])[0])}
        if batch is not None:
            arrays.update({f"batch_{name}": np.asarray(value)
                           for name, value in batch._asdict().items()})
        arrays.update(extra_arrays or {})
        meta = {"version": TRAINING_STATE_VERSION, "config": _config_fingerprint(config),
                "code": _code_fingerprint(), "phase": phase, "done": done,
                "elapsed": time.time() - started, "best_heldout": best["heldout"],
                "best_where": best["where"], "history": history, "cycle_log": cycle_log,
                "lm_log": lm_log, "resumed_from": resumed_from, **extra}
        save_training_state(checkpoint, arrays, meta)

    def saved_batch():
        return Batch(**{name: jnp.asarray(saved[f"batch_{name}"]) for name in Batch._fields})

    def record(p, batch, phase, cycle, iteration, lr, average=None):
        """One history row for `p`. `average`, if given, is a second candidate
        (the mean of recent iterates): it is scored on the held-out points too
        and may be selected, labelled '<phase>_average:<iteration>'."""
        loss, terms = train_terms(p, batch)
        held = float(heldout_loss(p))
        loss = float(loss)
        held_average = math.nan if average is None else float(heldout_loss(average))
        if not (math.isfinite(loss) and math.isfinite(held)
                and (average is None or math.isfinite(held_average))):
            raise FloatingPointError(
                f"non-finite loss at {phase}:{iteration} (training {loss}, held-out {held}, "
                f"averaged {held_average}); training has diverged and nothing will be saved "
                "as a result")
        if held < best["heldout"]:
            best.update(heldout=held, params=p, where=f"{phase}:{iteration}")
        if held_average < best["heldout"]:
            best.update(heldout=held_average, params=average,
                        where=f"{phase}_average:{iteration}")
        history.append({
            "phase": phase, "cycle": cycle, "iteration": iteration,
            "train_loss": loss, "heldout_loss": held,
            "heldout_over_train": held / loss if loss > 0 else math.inf,
            "heldout_average": held_average,
            "learning_rate": lr, "seconds": time.time() - started,
            **{f"term_{k}": float(v) for k, v in terms.items()},
        })
        if verbose:
            print(f"  {phase:>6} {cycle:>5} {iteration:>9d} {loss:>12.4e} "
                  f"{held:>12.4e} {held / loss if loss > 0 else math.inf:>10.1f}"
                  + ("" if average is None else f"   averaged {held_average:.4e}"), flush=True)

    if verbose:
        print(f"  {'phase':>6} {'cycle':>5} {'iteration':>9} {'train loss':>12} "
              f"{'held-out':>12} {'ratio':>10}", flush=True)

    phases = ("start", "adam", "lbfgs", "lm")
    order = phases.index(position[0])

    if position[0] == "start" and config.init_from != "random":
        # The starting network is a result in its own right: it is the best
        # so far until training beats it on the held-out points.
        key, draw = jax.random.split(key)
        record(params, draw_batch(draw, params), "init", 0, 0, math.nan)

    # -- Adam, redrawing points every `resample_every` steps ---------------- #
    if order <= phases.index("adam") and config.adam_steps > 0:
        schedule = adam_schedule(config)
        optimiser = optax.adam(schedule)
        state = optimiser.init(params)

        @jax.jit
        def adam_step(p, s, batch):
            (loss, _), grads = jax.value_and_grad(objective, has_aux=True)(p, batch)
            updates, s = optimiser.update(grads, s, p)
            return optax.apply_updates(p, updates), s, loss

        first, batch = 0, None
        if position[0] == "adam":
            first, batch = position[1], saved_batch()
            leaves, treedef = jax.tree_util.tree_flatten(state)
            state = jax.tree_util.tree_unflatten(
                treedef, [jnp.asarray(saved[f"adam_{i}"]) for i in range(len(leaves))])
        for step in range(first, config.adam_steps):
            if step % config.adam_resample_every == 0:
                key, draw = jax.random.split(key)
                batch = draw_batch(draw, params)
            params, state, _ = adam_step(params, state, batch)
            if (step + 1) % config.check_every == 0 or step + 1 == config.adam_steps:
                record(params, batch, "adam", step // config.adam_resample_every,
                       step + 1, float(schedule(step)))
                checkpoint_state(
                    "adam", step + 1, ravel_pytree(params)[0], batch,
                    {f"adam_{i}": np.asarray(leaf)
                     for i, leaf in enumerate(jax.tree_util.tree_leaves(state))})
    if order <= phases.index("adam"):
        adam_seconds = time.time() - started
    else:
        adam_seconds = meta["adam_seconds"]

    # -- L-BFGS in cycles, each on a fresh draw ----------------------------- #
    flat = ravel_pytree(params)[0]
    value_and_grad = jax.jit(jax.value_and_grad(
        lambda vector, batch: objective(unflatten(vector), batch)[0]))
    iteration = (meta["iteration"] if resume and order >= phases.index("lbfgs")
                 else config.adam_steps)
    first_cycle = position[1] if position[0] == "lbfgs" else 0
    if order <= phases.index("lbfgs"):
        for cycle in range(first_cycle, config.lbfgs_cycles):
            key, draw = jax.random.split(key)
            batch = draw_batch(draw, unflatten(flat))

            def fun(vector):
                value, grad = value_and_grad(jnp.asarray(vector), batch)
                return float(value), np.asarray(grad, dtype=np.float64)

            counter = {"n": 0}

            def callback(vector):
                nonlocal iteration
                counter["n"] += 1
                iteration += 1
                if counter["n"] % config.check_every == 0:
                    record(unflatten(jnp.asarray(vector)), batch, "lbfgs", cycle, iteration,
                           math.nan)

            result = minimize(
                fun, np.asarray(flat, dtype=np.float64), jac=True, method="L-BFGS-B",
                callback=callback,
                options={"maxiter": config.lbfgs_iterations,
                         "maxfun": 2 * config.lbfgs_iterations,
                         "gtol": config.lbfgs_gtol, "ftol": 0.0, "maxcor": 100},
            )
            flat = jnp.asarray(result.x)
            if counter["n"] % config.check_every != 0:
                record(unflatten(flat), batch, "lbfgs", cycle, iteration, math.nan)
            cycle_log.append({"cycle": cycle, "iterations": int(result.nit),
                              "message": str(result.message)})
            checkpoint_state("lbfgs", cycle + 1, flat, iteration=iteration,
                             adam_seconds=adam_seconds)

    # -- Levenberg-Marquardt on the full network ---------------------------- #
    # Gauss-Newton with trust-region damping (Levenberg 1944; Marquardt 1963)
    # on the residuals themselves: for a least-squares loss it is the natural
    # gradient in function space, the method Jnini, Vella & Zeinhofer (2024)
    # used to solve Navier-Stokes PINNs to near single-precision accuracy.
    # J^T J is P x P, so this suits networks of a few thousand parameters.
    if config.lm_iterations > 0:
        normal_equations, lm_loss = make_gauss_newton(config, field, unflatten)
        lam, nu, first = None, 2.0, 0
        # With fresh points at every step, each step fits its own draw: the
        # iterates scatter around the solution by the sampling noise of one
        # draw. Their mean cancels much of that scatter, as averaging the
        # iterates of a stochastic method does (Polyak & Juditsky 1992).
        window = []
        if position[0] == "lm":
            lam, nu, first, batch = meta["lam"], meta["nu"], position[1], saved_batch()
            if config.lm_average:
                window = [jnp.asarray(row) for row in saved["lm_window"]]
        for step in range(first, config.lm_iterations):
            if step % config.lm_resample_every == 0:
                key, draw = jax.random.split(key)
                batch = draw_batch(draw, unflatten(flat))
            A, g, loss_now = normal_equations(flat, batch)
            if lam is None:
                lam = config.lm_damping * float(jnp.mean(jnp.diag(A)))
            delta, _, lam, nu, attempts = levenberg_marquardt_step(
                A, g, loss_now, lambda d: lm_loss(flat + jnp.asarray(d), batch), lam, nu)
            lm_log["rejected"] += attempts - 1
            if delta is None:
                lm_log["stalled"] += 1
                # No damping reduced the loss on this draw: the step is
                # skipped and the next draw is taken at once.
                key, draw = jax.random.split(key)
                batch = draw_batch(draw, unflatten(flat))
                lam, nu = config.lm_damping * float(jnp.mean(jnp.diag(A))), 2.0
            else:
                flat = flat + jnp.asarray(delta)
            iteration += 1
            lm_log["iterations"] += 1
            average = None
            if config.lm_average:
                window = (window + [flat])[-config.lm_average:]
                if len(window) == config.lm_average:
                    average = unflatten(jnp.mean(jnp.stack(window), axis=0))
            record(unflatten(flat), batch, "lm", step // config.lm_resample_every, iteration, lam,
                   average)
            checkpoint_state(
                "lm", step + 1, flat, batch,
                {"lm_window": np.stack([np.asarray(w) for w in window])} if window else None,
                iteration=iteration, adam_seconds=adam_seconds, lam=lam, nu=nu)

    if best["where"] == "initial":
        raise RuntimeError(
            "no trained parameters beat the untrained network on the held-out points; "
            "refusing to return the random initialisation as a result")
    if best["where"] == "init:0":
        raise RuntimeError(
            f"training did not improve on the starting run {config.init_from!r} on the held-out "
            "points; nothing new to save")
    info = {
        "parameter_count": int(ravel_pytree(params)[0].size),
        "adam_seconds": adam_seconds,
        "total_seconds": time.time() - started,
        "selected": best["where"],
        "selected_heldout_loss": best["heldout"],
        "lbfgs_cycles": cycle_log,
        "levenberg_marquardt": lm_log,
        "resumed_from": resumed_from,
    }
    return best["params"], history, info


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


_PREDICTORS: dict = {}
_RESIDUALS: dict = {}


def _chunked(apply, params, points, chunk):
    """Apply a compiled batch function in fixed-size chunks.

    The last chunk is padded to full size, so every call has one shape and the
    function compiles once rather than once per ragged remainder.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(points)
    padded = np.zeros((-(-n // chunk) * chunk, 2))
    padded[:n] = points
    outputs = [apply(params, jnp.asarray(padded[i:i + chunk])) for i in range(0, len(padded), chunk)]
    return outputs, n


def predict(field, params, points, chunk=8192):
    """(u, v, p) arrays at an (N, 2) array of points.

    Compiled once per field function with the parameters as an ARGUMENT, so a
    new set of parameters or a new set of points does not recompile.
    """
    apply = _PREDICTORS.get(field)
    if apply is None:
        apply = _PREDICTORS[field] = jax.jit(jax.vmap(field, in_axes=(None, 0)))
    outputs, n = _chunked(apply, params, points, chunk)
    out = np.concatenate([np.asarray(o) for o in outputs], axis=0)[:n]
    return out[:, 0], out[:, 1], out[:, 2]


def residuals_at(config: Config, field, params, points, chunk=4096):
    """(R_x, R_y, R_c, u, v, p, p_x) at an (N, 2) array of points."""
    key = (field, config.reynolds, config.pressure_scaling)
    apply = _RESIDUALS.get(key)
    if apply is None:
        def bundle(p, pts):
            d = batched_derivatives(lambda xy: field(p, xy), pts)
            return (momentum_x(d, config.reynolds, config.pressure_scaling),
                    momentum_y(d, config.reynolds, config.pressure_scaling),
                    continuity(d), d.u, d.v, d.p, d.p_x)
        apply = _RESIDUALS[key] = jax.jit(bundle)
    outputs, n = _chunked(apply, params, points, chunk)
    return tuple(np.concatenate([np.asarray(o[k]) for o in outputs])[:n] for k in range(7))


def _crossing(x, values, target):
    """First x at which `values` reaches `target`, linearly interpolated."""
    reached = values >= target
    if not np.any(reached):
        return math.nan
    i = int(np.argmax(reached))
    if i == 0:
        return float(x[0])
    return float(x[i - 1] + (target - values[i - 1]) * (x[i] - x[i - 1]) / (values[i] - values[i - 1]))


def evaluate(config: Config, params, last_batch_losses=None):
    """Everything worth knowing about a trained solution, plus plot arrays."""
    metrics, arrays = measure_solution(config, make_field(config), params)
    if last_batch_losses is not None:
        metrics.update(last_batch_losses)
    return metrics, arrays


def measure_solution(config: Config, field, params):
    """The metrics of `evaluate` for ANY field(params, xy) -> (u, v, p).

    Separate from `evaluate` so it can be checked against a field whose flux,
    centreline and residuals are known exactly.
    """
    L = config.length

    # Flux through stations along the channel, by 200-node Gauss-Legendre.
    nodes, gl_weights = np.polynomial.legendre.leggauss(200)
    stations = np.linspace(0.0, L, config.n_flux_stations)
    sx, sy = np.meshgrid(stations, nodes, indexing="ij")
    u_st, _, _ = predict(field, params, np.stack([sx.ravel(), sy.ravel()], axis=-1))
    flux = u_st.reshape(len(stations), -1) @ gl_weights
    flux_outlet = float(flux[-1])

    # Centreline development and pressure.
    xc = np.linspace(0.0, L, 1001)
    uc, _, pc = predict(field, params, np.stack([xc, np.zeros_like(xc)], axis=-1))
    developed_centreline = 0.75 * flux_outlet
    entrance = _crossing(xc, uc, 0.99 * developed_centreline)

    # Outlet profile against the developed parabola at the MEASURED flux.
    yo = np.linspace(-1.0, 1.0, 201)
    uo, vo, _ = predict(field, params, np.stack([np.full_like(yo, L), yo], axis=-1))
    parabola = developed_profile(yo, flux_outlet)

    # Residuals over a grid, and the developed pressure gradient.
    xg = np.linspace(0.0, L, config.nx)
    yg = np.linspace(-1.0, 1.0, config.ny)
    gx, gy = np.meshgrid(xg, yg, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    rx, ry, rc, ug, vg, pg, px = residuals_at(config, field, params, grid)
    inertia, viscous = scaling_coefficients(config.reynolds, config.pressure_scaling)
    developed_px = -2.0 * viscous * developed_centreline
    outlet_px = px.reshape(config.nx, config.ny)[-1, config.ny // 2]
    drift = float(np.max(np.abs(flux - flux[0])) / flux[0])

    metrics = {
        "name": config.name,
        "reynolds": config.reynolds,
        "inlet_sharpness": config.inlet_sharpness,
        "inlet_peak": inlet_peak(config.inlet_sharpness),
        "channel_length": L,
        "inlet_flux_prescribed": INLET_FLUX,
        "flux_at_inlet_station": float(flux[0]),
        "flux_at_outlet": flux_outlet,
        "flux_max_deviation_from_prescribed": float(np.max(np.abs(flux - INLET_FLUX)) / INLET_FLUX),
        "mass_drift_from_first_station": drift,
        "centreline_at_inlet": float(uc[0]),
        "centreline_developed": developed_centreline,
        "entrance_length": entrance,
        "entrance_length_reliable": bool(drift < ENTRANCE_DRIFT_LIMIT and math.isfinite(entrance)),
        "entrance_length_drift_limit": ENTRANCE_DRIFT_LIMIT,
        "entrance_length_correlation_flat_inlet": correlation_entrance_length(config.reynolds),
        "outlet_profile_relative_l2": float(np.linalg.norm(uo - parabola) / np.linalg.norm(parabola)),
        "outlet_max_abs_v": float(np.max(np.abs(vo))),
        "pressure_drop_centreline": float(pc[0] - pc[-1]),
        "outlet_dp_dx": float(outlet_px),
        "developed_dp_dx_expected": float(developed_px),
        "residual_momentum_x_max": float(np.max(np.abs(rx))),
        "residual_momentum_y_max": float(np.max(np.abs(ry))),
        "residual_continuity_max": float(np.max(np.abs(rc))),
        "residual_momentum_x_rms": float(np.sqrt(np.mean(rx**2))),
        "residual_momentum_y_rms": float(np.sqrt(np.mean(ry**2))),
        "residual_continuity_rms": float(np.sqrt(np.mean(rc**2))),
    }
    if not math.isfinite(entrance):
        metrics["entrance_length_note"] = (
            "the centreline never reached 99% of its developed value inside the channel")

    arrays = {
        "stations": stations, "flux": flux, "xc": xc, "uc": uc, "pc": pc,
        "xg": xg, "yg": yg,
        "u": ug.reshape(config.nx, config.ny), "v": vg.reshape(config.nx, config.ny),
        "p": pg.reshape(config.nx, config.ny),
        "momentum_x": rx.reshape(config.nx, config.ny),
    }
    return metrics, arrays


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def write_figures(config: Config, params, metrics, arrays, history, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    field = make_field(config)
    L = config.length

    # 1. Velocity profiles at stations, and centreline development.
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    y = np.linspace(-1.0, 1.0, 201)
    for colour, x in zip(SERIES, config.profile_stations):
        u, _, _ = predict(field, params, np.stack([np.full_like(y, x), y], axis=-1))
        left.plot(u, y, color=colour, linewidth=2, label=f"x* = {x:g}")
    left.plot(developed_profile(y, metrics["flux_at_outlet"]), y, color=REFERENCE,
              linewidth=1.5, linestyle="--", label="developed parabola")
    left.set_xlabel("u*  (streamwise velocity)")
    left.set_ylabel("y*  (across the channel)")
    left.set_title("Velocity profiles along the channel", loc="left")
    left.legend(frameon=False, fontsize=9)
    _style(left)

    right.plot(arrays["xc"], arrays["uc"], color=SERIES[0], linewidth=2, label="centreline u*")
    right.axhline(0.99 * metrics["centreline_developed"], color=MUTED, linewidth=1,
                  linestyle=":", label="99% of developed")
    if not math.isnan(metrics["entrance_length"]):
        right.axvline(metrics["entrance_length"], color=SERIES[1], linewidth=1.5,
                      label=f"entrance length {metrics['entrance_length']:.2f}")
    right.axvline(metrics["entrance_length_correlation_flat_inlet"], color=REFERENCE,
                  linewidth=1.2, linestyle="--",
                  label=f"Durst (flat inlet) {metrics['entrance_length_correlation_flat_inlet']:.2f}")
    right.set_xlabel("x*  (along the channel)")
    right.set_ylabel("u*(x*, 0)")
    right.set_title("Centreline development", loc="left")
    right.legend(frameon=False, fontsize=9, loc="lower right")
    _style(right)
    fig.suptitle(f"{config.name}: Re = {config.reynolds:g}, inlet sharpness m = "
                 f"{config.inlet_sharpness}", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(directory / "profiles.png", dpi=150)
    plt.close(fig)

    # 2. Fields.
    xg, yg = arrays["xg"], arrays["yg"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    panels = (
        ("u", "u*  streamwise velocity", "Blues", None),
        ("v", "v*  transverse velocity", "RdBu_r", "symmetric"),
        ("p", "p*  pressure", "Blues", None),
    )
    for ax, (key, title, cmap, mode) in zip(axes, panels):
        values = arrays[key].T
        kwargs = {}
        if mode == "symmetric":
            limit = float(np.max(np.abs(values))) or 1.0
            kwargs = {"vmin": -limit, "vmax": limit}
        image = ax.pcolormesh(xg, yg, values, cmap=cmap, shading="auto", **kwargs)
        fig.colorbar(image, ax=ax, pad=0.01)
        ax.set_ylabel("y*")
        ax.set_title(title, loc="left")
    axes[0].streamplot(xg, yg, arrays["u"].T, arrays["v"].T, color="white",
                       linewidth=0.6, density=0.8, arrowsize=0.6)
    axes[-1].set_xlabel("x*")
    fig.tight_layout()
    fig.savefig(directory / "fields.png", dpi=150)
    plt.close(fig)

    # 3. Training history, mass conservation, pressure.
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.2))
    it = np.array([row["iteration"] for row in history])
    a.semilogy(it, [row["train_loss"] for row in history], color=SERIES[0], linewidth=1.5, label="training points")
    a.semilogy(it, [row["heldout_loss"] for row in history], color=SERIES[1], linewidth=1.5, label="held-out points")
    averaged = np.array([row.get("heldout_average", math.nan) for row in history])
    if np.isfinite(averaged).any():
        a.semilogy(it, averaged, color=SERIES[2], linewidth=1.5,
                   label="held-out, mean of recent iterates")
    a.axvline(config.adam_steps, color=MUTED, linewidth=1, linestyle=":")
    a.set_xlabel("iteration (Adam, then L-BFGS cycles)")
    a.set_ylabel("loss")
    a.set_title("Training", loc="left")
    a.legend(frameon=False, fontsize=9)
    _style(a)

    b.plot(arrays["stations"], 100.0 * (arrays["flux"] / INLET_FLUX - 1.0), color=SERIES[2],
           linewidth=2, marker="o", markersize=3)
    b.axhline(0.0, color=REFERENCE, linewidth=1)
    b.set_xlabel("x*")
    b.set_ylabel("flux error vs inlet (%)")
    b.set_title("Mass conservation along the channel", loc="left")
    _style(b)

    c.plot(arrays["xc"], arrays["pc"], color=SERIES[3], linewidth=2)
    c.set_xlabel("x*")
    c.set_ylabel("p*(x*, 0)")
    c.set_title("Centreline pressure", loc="left")
    _style(c)
    fig.tight_layout()
    fig.savefig(directory / "diagnostics.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Saving and loading runs
# --------------------------------------------------------------------------- #


def save_params(path, params):
    flat, _ = ravel_pytree(params)
    temporary = pathlib.Path(path).with_suffix(".tmp.npz")
    np.savez_compressed(temporary, flat=np.asarray(flat, dtype=np.float64))
    temporary.replace(path)


def load_run(directory):
    """A trained model as a callable: model(x, y) -> (u, v, p) arrays."""
    directory = pathlib.Path(directory)
    if not (directory / "params.npz").is_file():
        state = ("it was interrupted: only a checkpoint exists"
                 if (directory / "checkpoint.npz").is_file() else "there are no parameters")
        raise FileNotFoundError(f"{directory} is not a finished run: {state}. "
                                "params.npz is written last, only when a run completes.")
    config = load_config(directory / "config.yaml")
    template = init_params(config, root_keys(config.seed)[0])
    flat_template, unflatten = ravel_pytree(template)
    flat = np.load(directory / "params.npz")["flat"]
    if flat.shape != flat_template.shape:
        raise ValueError("saved parameters do not match the config's network")
    params = unflatten(jnp.asarray(flat))
    field = make_field(config)

    def model(x, y):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        x, y = np.broadcast_arrays(x, y)
        if np.any((x < 0) | (x > config.length) | (y < -1) | (y > 1)):
            raise ValueError(f"points must lie in x* in [0, {config.length}], y* in [-1, 1]")
        u, v, p = predict(field, params, np.stack([x.ravel(), y.ravel()], axis=-1))
        return u.reshape(x.shape), v.reshape(x.shape), p.reshape(x.shape)

    model.config = config
    return model


def run(config_path, out=None, verbose=True, resume=False):
    """Train, evaluate and save one run. With `resume`, continue an interrupted
    run in `out` from its checkpoint: its config.yaml must be the same file,
    byte for byte, and the result equals the uninterrupted run's."""
    config = load_config(config_path)
    directory = pathlib.Path(out) if out else HERE / "runs" / config.name
    text = pathlib.Path(config_path).read_text(encoding="utf-8")
    if resume:
        if (directory / "params.npz").exists():
            raise FileExistsError(f"{directory} is a finished run; there is nothing to resume")
        stored = directory / "config.yaml"
        if not stored.is_file() or stored.read_text(encoding="utf-8") != text:
            raise ValueError(f"cannot resume {directory}: its config.yaml is missing or is not "
                             f"the same file as {config_path}")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"{directory} already holds files. A run never writes over another run: "
                "delete it or pass a different --out (or --resume an interrupted run).")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text(text, encoding="utf-8")

    if verbose:
        print(f"[{config.name}] developing flow, Re = {config.reynolds:g} "
              f"({config.pressure_scaling} pressure), inlet sharpness m = {config.inlet_sharpness}, "
              f"L* = {config.length:g}")
        print(f"[{config.name}] inertial terms ON; no-slip and pressure gauge exact; "
              f"{config.formulation}"
              + (f" ({config.lift} lift"
                 + (f", l = {config.lift_length:g})" if config.lift_length else ")")
                 if config.lift else "")
              + f", {config.inlet_condition} inlet"
              + (f" ({config.inlet_ramp} ramp"
                 + (f", l = {config.inlet_ramp_length:g})" if config.inlet_ramp_length else ")")
                 if config.inlet_ramp else "")
              + f", {config.activation} (init gain {config.init_gain:g}), "
              f"{config.input_map} input map"
              + (f" (l = {config.input_length:g})" if config.input_length else "")
              + f", Adam {config.adam_steps} steps ({config.adam_schedule}), "
              f"inlet points {config.inlet_points}")
        print(f"[{config.name}] {config.n_interior} points per draw"
              + (f" + {config.rar_points} worst of {config.rar_pool} by residual (RAR)"
                 f" ({config.rar_mode})" if config.rar_points else "")
              + f", {config.lbfgs_cycles} L-BFGS cycles of {config.lbfgs_iterations}")

    initial, warm = None, None
    if config.init_from != "random":
        initial, digest = warm_start_params(config, directory.parent / config.init_from)
        warm = {"run": config.init_from, "params_sha256": digest}
        if verbose:
            print(f"[{config.name}] starting from run {config.init_from} "
                  f"(params.npz sha256 {digest[:16]}...)")

    checkpoint = directory / "checkpoint.npz"
    params, history, info = train(config, verbose=verbose, checkpoint=checkpoint, initial=initial,
                                  resume=resume)
    if warm is not None:
        info["init_from"] = warm

    # Losses of the SELECTED parameters on a fresh draw. The held-out loss was
    # already measured, compiled, when train() selected them.
    field = make_field(config)
    _, _, heldout_key = root_keys(config.seed)
    fresh = sample_batch(jax.random.fold_in(heldout_key, 7), config, config.n_validation)
    fresh_terms = jax.jit(make_loss_terms(config, field))(params, fresh)
    weights = {k: jnp.asarray(v) for k, v in config.weights.items()}
    losses = {
        "loss_heldout": info["selected_heldout_loss"],
        "loss_fresh_points": float(total_loss(fresh_terms, weights)),
        **{f"loss_fresh_{k}": float(v) for k, v in fresh_terms.items()},
        **{f"info_{k}": v for k, v in info.items() if k != "lbfgs_cycles"},
    }
    metrics, arrays = evaluate(config, params, losses)

    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (directory / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metrics), handle, indent=2, allow_nan=False)
    with (directory / "lbfgs_cycles.json").open("w", encoding="utf-8") as handle:
        json.dump(info["lbfgs_cycles"], handle, indent=2)
    with (directory / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump({"python": sys.version, "platform": platform.platform(),
                   "jax": jax.__version__, "x64": bool(jax.config.jax_enable_x64)}, handle, indent=2)
    write_figures(config, params, metrics, arrays, history, directory)

    # LAST: params.npz exists only once everything else is written, so its
    # presence is what marks a run as finished (load_run checks for it).
    save_params(directory / "params.npz", params)
    checkpoint.unlink(missing_ok=True)

    if verbose:
        _print_summary(metrics, directory)
    return directory


def _json_safe(value):
    """Replace non-finite floats with None: NaN and Infinity are not JSON."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _print_summary(m, directory):
    print(f"\n  wrote {directory}")
    print("  --- the developing flow ---")
    print(f"  centreline speed        : {m['centreline_at_inlet']:.4f} at the inlet -> "
          f"{m['centreline_developed']:.4f} developed")
    print(f"  entrance length (99%)   : {m['entrance_length']:.3f} half-heights "
          f"(Durst, flat inlet: {m['entrance_length_correlation_flat_inlet']:.3f})")
    if not m["entrance_length_reliable"]:
        print(f"  WARNING: flux drifts by {m['mass_drift_from_first_station']:.1e} along the "
              f"channel (limit {m['entrance_length_drift_limit']:.0e}), so the centreline is "
              "still creeping and the entrance length above is NOT reliable")
    print(f"  pressure drop, centre   : {m['pressure_drop_centreline']:.5f}")
    print(f"  outlet dp/dx            : {m['outlet_dp_dx']:.5f} "
          f"(developed value {m['developed_dp_dx_expected']:.5f})")
    print("  --- how well it is solved ---")
    print(f"  outlet vs parabola (L2) : {m['outlet_profile_relative_l2']:.3e}")
    print(f"  flux drift along channel: {m['mass_drift_from_first_station']:.3e}")
    print(f"  flux vs prescribed 4/3  : {m['flux_max_deviation_from_prescribed']:.3e}")
    print(f"  loss, held-out points   : {m['loss_heldout']:.3e}")
    print(f"  loss, fresh points      : {m['loss_fresh_points']:.3e}")
    print(f"  max residual x/y/cont.  : {m['residual_momentum_x_max']:.2e} / "
          f"{m['residual_momentum_y_max']:.2e} / {m['residual_continuity_max']:.2e}")
    print(f"  wall time               : {m['info_total_seconds'] / 60:.1f} min")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Developing-flow PINN with inertia.")
    parser.add_argument("--config", required=True, help="path to a developing-flow YAML config")
    parser.add_argument("--out", default=None, help="run directory (default: developing_flow/runs/<name>)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from its checkpoint.npz")
    args = parser.parse_args(argv)
    run(args.config, args.out, verbose=not args.quiet, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
