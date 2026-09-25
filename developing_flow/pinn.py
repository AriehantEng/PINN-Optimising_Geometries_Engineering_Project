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
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
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


#: The config schema. A leaf is the expected type; `float` also accepts int.
SCHEMA = {
    "name": str,
    "seed": int,
    "physics": {"reynolds": float, "pressure_scaling": str, "inlet_sharpness": int},
    "domain": {"length": float},
    "model": {"width": int, "depth": int},
    "sampling": {
        "n_interior": int,
        "focus_fraction": float,
        "focus_length": float,
        "n_inlet": int,
        "n_outlet": int,
        "n_validation": int,
    },
    "weights": {term: float for term in LOSS_TERMS},
    "training": {
        "adam": {
            "steps": int,
            "learning_rate": float,
            "decay_rate": float,
            "decay_steps": int,
            "resample_every": int,
        },
        "lbfgs": {"cycles": int, "iterations_per_cycle": int, "gtol": float},
        "check_every": int,
    },
    "evaluation": {"nx": int, "ny": int, "profile_stations": list, "n_flux_stations": int},
}


def _check(document, schema, path=""):
    if not isinstance(document, dict):
        raise ConfigError(f"{path or 'config'} must be a mapping")
    for key, expected in schema.items():
        where = f"{path}.{key}" if path else key
        if key not in document:
            raise ConfigError(f"missing required key {where!r}")
        value = document[key]
        if isinstance(expected, dict):
            _check(value, expected, where)
        elif expected is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{where} must be a number, got {value!r}")
        elif expected is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{where} must be an integer, got {value!r}")
        elif not isinstance(value, expected):
            raise ConfigError(f"{where} must be {expected.__name__}, got {value!r}")
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
    length: float
    width: int
    depth: int
    n_interior: int
    focus_fraction: float
    focus_length: float
    n_inlet: int
    n_outlet: int
    n_validation: int
    weights: dict
    adam_steps: int
    adam_learning_rate: float
    adam_decay_rate: float
    adam_decay_steps: int
    adam_resample_every: int
    lbfgs_cycles: int
    lbfgs_iterations: int
    lbfgs_gtol: float
    check_every: int
    nx: int
    ny: int
    profile_stations: tuple
    n_flux_stations: int


def parse_config(document) -> Config:
    """Validate a config mapping. Raises ConfigError; never fills a default."""
    _check(document, SCHEMA)
    physics, sampling = document["physics"], document["sampling"]
    adam, lbfgs = document["training"]["adam"], document["training"]["lbfgs"]
    evaluation = document["evaluation"]

    config = Config(
        name=document["name"],
        seed=document["seed"],
        reynolds=float(physics["reynolds"]),
        pressure_scaling=physics["pressure_scaling"],
        inlet_sharpness=physics["inlet_sharpness"],
        length=float(document["domain"]["length"]),
        width=document["model"]["width"],
        depth=document["model"]["depth"],
        n_interior=sampling["n_interior"],
        focus_fraction=float(sampling["focus_fraction"]),
        focus_length=float(sampling["focus_length"]),
        n_inlet=sampling["n_inlet"],
        n_outlet=sampling["n_outlet"],
        n_validation=sampling["n_validation"],
        weights={term: float(document["weights"][term]) for term in LOSS_TERMS},
        adam_steps=adam["steps"],
        adam_learning_rate=float(adam["learning_rate"]),
        adam_decay_rate=float(adam["decay_rate"]),
        adam_decay_steps=adam["decay_steps"],
        adam_resample_every=adam["resample_every"],
        lbfgs_cycles=lbfgs["cycles"],
        lbfgs_iterations=lbfgs["iterations_per_cycle"],
        lbfgs_gtol=float(lbfgs["gtol"]),
        check_every=document["training"]["check_every"],
        nx=evaluation["nx"],
        ny=evaluation["ny"],
        profile_stations=tuple(float(x) for x in evaluation["profile_stations"]),
        n_flux_stations=evaluation["n_flux_stations"],
    )

    if config.reynolds <= 0:
        raise ConfigError("physics.reynolds must be positive")
    if config.pressure_scaling not in ("inertial", "viscous"):
        raise ConfigError("physics.pressure_scaling must be 'inertial' or 'viscous'")
    if config.inlet_sharpness == 1:
        raise ConfigError(
            "physics.inlet_sharpness = 1 is the developed parabola: the flow "
            "would enter already developed. Use m >= 2."
        )
    if config.inlet_sharpness < 1:
        raise ConfigError("physics.inlet_sharpness must be an integer >= 2")
    if config.length <= 0:
        raise ConfigError("domain.length must be positive")
    for name in ("width", "depth", "n_interior", "n_inlet", "n_outlet",
                 "n_validation", "adam_steps", "adam_decay_steps",
                 "adam_resample_every", "lbfgs_iterations", "check_every",
                 "nx", "ny", "n_flux_stations"):
        if getattr(config, name) < 1:
            raise ConfigError(f"{name} must be at least 1")
    if config.lbfgs_cycles < 0:
        raise ConfigError("training.lbfgs.cycles must be >= 0")
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


def _glorot(key, n_out, n_in):
    return jnp.sqrt(2.0 / (n_in + n_out)) * jax.random.normal(key, (n_out, n_in), dtype=jnp.float64)


def _dense(key, n_in, n_out):
    return {"W": _glorot(key, n_out, n_in), "b": jnp.zeros((n_out,), dtype=jnp.float64)}


def init_network(key, width, depth):
    """Modified MLP (Wang, Teng & Perdikaris 2021): two input encoders blended
    into every hidden layer. Inputs (x, y); outputs the raw (N_u, N_v, N_p)."""
    keys = jax.random.split(key, depth + 3)
    return {
        "encoder_u": _dense(keys[0], 2, width),
        "encoder_v": _dense(keys[1], 2, width),
        "hidden": [_dense(keys[2], 2, width)]
        + [_dense(keys[3 + i], width, width) for i in range(depth - 1)],
        "output": _dense(keys[depth + 2], width, 3),
    }


def _apply_dense(layer, x):
    return layer["W"] @ x + layer["b"]


def apply_network(params, xi):
    encoded_u = jnp.tanh(_apply_dense(params["encoder_u"], xi))
    encoded_v = jnp.tanh(_apply_dense(params["encoder_v"], xi))
    h = xi
    for layer in params["hidden"]:
        z = jnp.tanh(_apply_dense(layer, h))
        h = (1.0 - z) * encoded_u + z * encoded_v
    return _apply_dense(params["output"], h)


def wall_factor(y):
    """phi(y*) = 1 - y*^2: zero on both walls, positive between them."""
    return 1.0 - y**2


def make_field(length):
    """field(params, xy) -> (u, v, p) at one point.

        u = phi(y) N_u,   v = phi(y) N_v,   p = (1 - x/L) N_p

    phi vanishes on both walls, so no-slip holds exactly for ANY parameters,
    and the pressure factor vanishes at the outlet, so the gauge holds exactly.
    Inputs are mapped to [-1, 1]^2 before entering the network.

    The gauge factor is written (L - x)/L, not 1 - x/L: under jit the compiler
    may evaluate x/L as x * (1/L), and 1 - L*(1/L) is then ~1e-17 rather than
    zero. L - L is exactly zero whatever the compiler does.
    """

    def field(params, xy):
        x, y = xy[0], xy[1]
        raw = apply_network(params, jnp.stack([2.0 * x / length - 1.0, y]))
        wall = wall_factor(y)
        return jnp.stack([wall * raw[0], wall * raw[1], ((length - x) / length) * raw[2]])

    return field


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


def root_keys(seed):
    """(model key, training-sample key, held-out key) from one seed."""
    return jax.random.split(jax.random.PRNGKey(seed), 3)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def make_loss_terms(config: Config, field):
    """loss_terms(params, batch) -> {term: unweighted mean-squared residual}."""
    reynolds, scaling, sharpness = config.reynolds, config.pressure_scaling, config.inlet_sharpness

    def loss_terms(params, batch: Batch):
        f = lambda xy: field(params, xy)  # noqa: E731
        d = batched_derivatives(f, batch.interior)
        inlet_values = jax.vmap(f)(batch.inlet)
        target = inlet_profile(batch.inlet[:, 1], sharpness)
        _, outlet_dx = batched_x_derivative(f, batch.outlet)
        return {
            "momentum_x": jnp.mean(momentum_x(d, reynolds, scaling) ** 2),
            "momentum_y": jnp.mean(momentum_y(d, reynolds, scaling) ** 2),
            "continuity": jnp.mean(continuity(d) ** 2),
            "inlet": jnp.mean((inlet_values[:, 0] - target) ** 2) + jnp.mean(inlet_values[:, 1] ** 2),
            "outlet": jnp.mean(outlet_dx[:, 0] ** 2) + jnp.mean(outlet_dx[:, 1] ** 2),
        }

    return loss_terms


def total_loss(terms, weights):
    """The ONLY place weights meet residuals."""
    return sum(weights[name] * terms[name] for name in LOSS_TERMS)


# --------------------------------------------------------------------------- #
# Training: Adam with resampling, then L-BFGS in cycles on fresh points
# --------------------------------------------------------------------------- #


def train(config: Config, verbose: bool = True, checkpoint=None):
    """Returns (best params, history rows, info dict)."""
    field = make_field(config.length)
    model_key, sample_key, heldout_key = root_keys(config.seed)
    params = init_network(model_key, config.width, config.depth)
    loss_terms = make_loss_terms(config, field)
    weights = {name: jnp.asarray(config.weights[name], dtype=jnp.float64) for name in LOSS_TERMS}

    def objective(p, batch):
        terms = loss_terms(p, batch)
        return total_loss(terms, weights), terms

    heldout_batch = sample_batch(heldout_key, config, config.n_validation)
    heldout_loss = jax.jit(lambda p: objective(p, heldout_batch)[0])
    train_terms = jax.jit(objective)

    best = {"heldout": math.inf, "params": params, "where": "initial"}
    history = []
    started = time.time()

    def record(p, batch, phase, cycle, iteration, lr):
        loss, terms = train_terms(p, batch)
        held = float(heldout_loss(p))
        loss = float(loss)
        if held < best["heldout"]:
            best.update(heldout=held, params=p, where=f"{phase}:{iteration}")
            if checkpoint is not None:
                save_params(checkpoint, p)
        history.append({
            "phase": phase, "cycle": cycle, "iteration": iteration,
            "train_loss": loss, "heldout_loss": held,
            "heldout_over_train": held / loss if loss > 0 else math.inf,
            "learning_rate": lr, "seconds": time.time() - started,
            **{f"term_{k}": float(v) for k, v in terms.items()},
        })
        if verbose:
            print(f"  {phase:>6} {cycle:>5} {iteration:>9d} {loss:>12.4e} "
                  f"{held:>12.4e} {held / loss if loss > 0 else math.inf:>10.1f}", flush=True)

    if verbose:
        print(f"  {'phase':>6} {'cycle':>5} {'iteration':>9} {'train loss':>12} "
              f"{'held-out':>12} {'ratio':>10}", flush=True)

    # -- Adam, redrawing points every `resample_every` steps ---------------- #
    schedule = optax.exponential_decay(
        config.adam_learning_rate, config.adam_decay_steps, config.adam_decay_rate)
    optimiser = optax.adam(schedule)
    state = optimiser.init(params)

    @jax.jit
    def adam_step(p, s, batch):
        (loss, _), grads = jax.value_and_grad(objective, has_aux=True)(p, batch)
        updates, s = optimiser.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    key = sample_key
    batch = None
    for step in range(config.adam_steps):
        if step % config.adam_resample_every == 0:
            key, draw = jax.random.split(key)
            batch = sample_batch(draw, config, config.n_interior)
        params, state, _ = adam_step(params, state, batch)
        if (step + 1) % config.check_every == 0 or step + 1 == config.adam_steps:
            record(params, batch, "adam", step // config.adam_resample_every,
                   step + 1, float(schedule(step)))
    adam_seconds = time.time() - started

    # -- L-BFGS in cycles, each on a fresh draw ----------------------------- #
    flat, unflatten = ravel_pytree(params)
    value_and_grad = jax.jit(jax.value_and_grad(
        lambda vector, batch: objective(unflatten(vector), batch)[0]))
    iteration = config.adam_steps
    cycle_log = []
    for cycle in range(config.lbfgs_cycles):
        key, draw = jax.random.split(key)
        batch = sample_batch(draw, config, config.n_interior)

        def fun(vector):
            value, grad = value_and_grad(jnp.asarray(vector), batch)
            return float(value), np.asarray(grad, dtype=np.float64)

        counter = {"n": 0}

        def callback(vector):
            nonlocal iteration
            counter["n"] += 1
            iteration += 1
            if counter["n"] % config.check_every == 0:
                record(unflatten(jnp.asarray(vector)), batch, "lbfgs", cycle, iteration, math.nan)

        result = minimize(
            fun, np.asarray(flat, dtype=np.float64), jac=True, method="L-BFGS-B",
            callback=callback,
            options={"maxiter": config.lbfgs_iterations, "maxfun": 2 * config.lbfgs_iterations,
                     "gtol": config.lbfgs_gtol, "ftol": 0.0, "maxcor": 100},
        )
        flat = jnp.asarray(result.x)
        if counter["n"] % config.check_every != 0:
            record(unflatten(flat), batch, "lbfgs", cycle, iteration, math.nan)
        cycle_log.append({"cycle": cycle, "iterations": int(result.nit),
                          "message": str(result.message)})

    info = {
        "parameter_count": int(ravel_pytree(params)[0].size),
        "adam_seconds": adam_seconds,
        "total_seconds": time.time() - started,
        "selected": best["where"],
        "selected_heldout_loss": best["heldout"],
        "lbfgs_cycles": cycle_log,
    }
    return best["params"], history, info


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def predict(field, params, points, chunk=8192):
    """(u, v, p) arrays at an (N, 2) array of points."""
    apply = jax.jit(jax.vmap(lambda xy: field(params, xy)))
    out = [np.asarray(apply(jnp.asarray(points[i:i + chunk])))
           for i in range(0, len(points), chunk)]
    out = np.concatenate(out, axis=0)
    return out[:, 0], out[:, 1], out[:, 2]


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
    field = make_field(config.length)
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
    residual_fn = jax.jit(lambda pts: _residual_bundle(config, field, params, pts))
    parts = [residual_fn(jnp.asarray(grid[i:i + 4096])) for i in range(0, len(grid), 4096)]
    rx, ry, rc, ug, vg, pg, px = (np.concatenate([np.asarray(part[k]) for part in parts])
                                  for k in range(7))
    inertia, viscous = scaling_coefficients(config.reynolds, config.pressure_scaling)
    developed_px = -2.0 * viscous * developed_centreline
    outlet_px = px.reshape(config.nx, config.ny)[-1, config.ny // 2]

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
        "mass_drift_from_first_station": float(np.max(np.abs(flux - flux[0])) / flux[0]),
        "centreline_at_inlet": float(uc[0]),
        "centreline_developed": developed_centreline,
        "entrance_length": entrance,
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
    if last_batch_losses is not None:
        metrics.update(last_batch_losses)

    arrays = {
        "stations": stations, "flux": flux, "xc": xc, "uc": uc, "pc": pc,
        "xg": xg, "yg": yg,
        "u": ug.reshape(config.nx, config.ny), "v": vg.reshape(config.nx, config.ny),
        "p": pg.reshape(config.nx, config.ny),
        "momentum_x": rx.reshape(config.nx, config.ny),
    }
    return metrics, arrays


def _residual_bundle(config, field, params, points):
    d = batched_derivatives(lambda xy: field(params, xy), points)
    return (momentum_x(d, config.reynolds, config.pressure_scaling),
            momentum_y(d, config.reynolds, config.pressure_scaling),
            continuity(d), d.u, d.v, d.p, d.p_x)


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

    field = make_field(config.length)
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
    config = load_config(directory / "config.yaml")
    template = init_network(root_keys(config.seed)[0], config.width, config.depth)
    flat_template, unflatten = ravel_pytree(template)
    flat = np.load(directory / "params.npz")["flat"]
    if flat.shape != flat_template.shape:
        raise ValueError("saved parameters do not match the config's network")
    params = unflatten(jnp.asarray(flat))
    field = make_field(config.length)

    def model(x, y):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        x, y = np.broadcast_arrays(x, y)
        if np.any((x < 0) | (x > config.length) | (y < -1) | (y > 1)):
            raise ValueError(f"points must lie in x* in [0, {config.length}], y* in [-1, 1]")
        return predict(field, params, np.stack([x.ravel(), y.ravel()], axis=-1))

    model.config = config
    return model


def run(config_path, out=None, verbose=True):
    config = load_config(config_path)
    directory = pathlib.Path(out) if out else HERE / "runs" / config.name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(
        pathlib.Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

    if verbose:
        print(f"[{config.name}] developing flow, Re = {config.reynolds:g} "
              f"({config.pressure_scaling} pressure), inlet sharpness m = {config.inlet_sharpness}, "
              f"L* = {config.length:g}")
        print(f"[{config.name}] inertial terms ON; no-slip and pressure gauge exact; "
              f"{config.n_interior} points per draw, {config.lbfgs_cycles} L-BFGS cycles "
              f"of {config.lbfgs_iterations}")

    params, history, info = train(config, verbose=verbose, checkpoint=directory / "params.npz")
    save_params(directory / "params.npz", params)

    # Losses of the SELECTED parameters, on a fresh draw and on the held-out set.
    field = make_field(config.length)
    weights = {k: jnp.asarray(v) for k, v in config.weights.items()}
    loss_terms = make_loss_terms(config, field)
    _, _, heldout_key = root_keys(config.seed)
    fresh = sample_batch(jax.random.fold_in(heldout_key, 7), config, config.n_validation)
    heldout = sample_batch(heldout_key, config, config.n_validation)
    fresh_terms = loss_terms(params, fresh)
    losses = {
        "loss_heldout": float(total_loss(loss_terms(params, heldout), weights)),
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
        json.dump(metrics, handle, indent=2)
    with (directory / "lbfgs_cycles.json").open("w", encoding="utf-8") as handle:
        json.dump(info["lbfgs_cycles"], handle, indent=2)
    with (directory / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump({"python": sys.version, "platform": platform.platform(),
                   "jax": jax.__version__, "x64": bool(jax.config.jax_enable_x64)}, handle, indent=2)
    write_figures(config, params, metrics, arrays, history, directory)

    if verbose:
        _print_summary(metrics, directory)
    return directory


def _print_summary(m, directory):
    print(f"\n  wrote {directory}")
    print("  --- the developing flow ---")
    print(f"  centreline speed        : {m['centreline_at_inlet']:.4f} at the inlet -> "
          f"{m['centreline_developed']:.4f} developed")
    print(f"  entrance length (99%)   : {m['entrance_length']:.3f} half-heights "
          f"(Durst, flat inlet: {m['entrance_length_correlation_flat_inlet']:.3f})")
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
    args = parser.parse_args(argv)
    run(args.config, args.out, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
