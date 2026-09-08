#!/usr/bin/env python3
"""
PINN for plane Poiseuille flow — the entire project in one file.

============================================================================
HOW TO RUN
============================================================================

  1. Install the two dependencies that are not already standard:

         pip install jax optax pyyaml matplotlib

  2. Run it:

         python pinn_poiseuille_all_in_one.py            quick demo, ~3 minutes
         python pinn_poiseuille_all_in_one.py --full     full accuracy, ~30 min
         python pinn_poiseuille_all_in_one.py --check    verify only, no training

Running it creates a `src/` folder next to this file (that is the project's
code, written out so Python can import it) and a `trained_model.npz` when
training finishes.

============================================================================
WHAT YOU WILL SEE
============================================================================

During training, two numbers side by side:

    step  epoch   train loss     held-out   rel L2 vs exact      lr
     300      0   7.4260e-03   7.9012e-03       6.2084e-01  8.54e-04

  * `train loss` is the residual of the Navier-Stokes equations on the points
    being optimised. It is a DIAGNOSTIC. It can fall while the answer gets
    worse, and in this project it demonstrably does.

  * `rel L2 vs exact` is the relative L2 error against the ANALYTIC solution
    of this flow, which has been known since the 19th century. THIS is the
    accuracy number. Watch this one.

  * `held-out` is the same residual measured on points the optimiser never
    sees, which is how overfitting would show up without needing the exact
    answer (Phase 2 will not have one).

At the end it prints the accuracy against the analytic solution, then
demonstrates using the trained model to predict velocity and pressure at
arbitrary points.

============================================================================
WHAT THE NUMBERS MEAN
============================================================================

Targets from the project specification:

    relative L2 error on u    <= 1e-5   (stretch goal 1e-6)
    max |v*|                  <= 1e-5
    mass conservation drift   <= 1e-6

Best measured so far on the full schedule: 8.6e-7 on u, which beats the
stretch goal.

Everything runs in float64. That is not optional here: float32 caps the
achievable error near 1e-4, which is worse than every target above.
"""

import argparse
import json
import pathlib
import sys
import textwrap

# ===========================================================================
# THE PROJECT SOURCE, EMBEDDED VERBATIM
# ===========================================================================
#
#   *** IF YOUR EDITOR SHOWS EVERYTHING BELOW AS ONE GREY BLOCK, THAT IS  ***
#   *** SYNTAX HIGHLIGHTING, NOT A PROBLEM. The code is NOT commented out.***
#
# Each entry below is one file of the project, stored as string DATA inside
# a raw triple-quoted literal. Python reads them as data; unpack() writes
# them to disk as real .py files, which are then imported normally.
#
# Most editors colour everything between the triple quotes as a string, so
# this file LOOKS inert. It is not: it compiles, runs, trains and predicts.
# The quotes are balanced - exactly one opening and one closing delimiter
# per embedded file.
#
# To read the code as ordinary syntax-highlighted Python instead, run:
#
#       python pinn_poiseuille_all_in_one.py --extract
#
# which writes the 34 files into ./src/ and stops, without training.

FILES = {}

# ---------------------------------------------------------------------------
# src/__init__.py
# ---------------------------------------------------------------------------
FILES['src/__init__.py'] = r'''"""PINN for branching channel flow — Phase 1 (plane Poiseuille).

IMPORTANT: double precision is enabled here, at package import, *before* any
JAX array can be created by a submodule. float32 silently caps achievable
relative L2 error near 1e-4 and the symptom is indistinguishable from an
optimiser failure. Do not move, guard or make this conditional.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

import jax.numpy as _jnp  # noqa: E402  (must follow the x64 switch)

if _jnp.zeros(1).dtype != _jnp.float64:
    raise RuntimeError(
        "float64 is not active: jnp.zeros(1).dtype == "
        f"{_jnp.zeros(1).dtype}. Something imported jax and created an array "
        "before src/__init__.py ran, or JAX_ENABLE_X64 was overridden."
    )

__all__ = ["assert_float64"]


def assert_float64(*arrays, where=""):
    """Raise unless every argument is float64.

    Used at module boundaries where a float32 leak would be silent. `where` is
    a free-text label included in the error message.
    """
    for i, a in enumerate(arrays):
        dtype = _jnp.asarray(a).dtype
        if dtype != _jnp.float64:
            raise TypeError(
                f"float64 required{' in ' + where if where else ''}: "
                f"argument {i} has dtype {dtype}. See CLAUDE.md, "
                "'Non-negotiable numerical rules'."
            )
'''

# ---------------------------------------------------------------------------
# src/config.py
# ---------------------------------------------------------------------------
FILES['src/config.py'] = r'''"""Strict configuration loading.

Two rules from CLAUDE.md are enforced structurally here rather than by
convention:

* "Every run is fully determined by its config file plus its seed. No
  hyperparameters hard-coded in source, no defaults that silently apply."
* "No silent fallbacks. A missing config key raises rather than defaulting."

So every key is required and every unknown key is an error. Keys that only
apply under a particular choice (e.g. `bc.lifting_length` when the inlet is
handled by lifting) are required exactly when that choice is active and
rejected when it is not -- a config carrying a setting that does nothing is a
config that lies about what was run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .physics.conventions import EntranceLengthCorrelation, entrance_length_over_h
from .physics.scaling import PressureScaling


class ConfigError(ValueError):
    """Raised for any missing, unknown, or invalid configuration key."""


def _require(mapping: dict, key: str, path: str, kind=None):
    if not isinstance(mapping, dict):
        raise ConfigError(f"{path or '<root>'} must be a mapping, got {type(mapping).__name__}")
    if key not in mapping:
        raise ConfigError(
            f"missing required config key {_join(path, key)!r} "
            f"(present keys: {sorted(mapping)})"
        )
    value = mapping.pop(key)
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if kind is not None and not isinstance(value, kind):
        raise ConfigError(
            f"config key {_join(path, key)!r} must be {kind.__name__}, "
            f"got {type(value).__name__} ({value!r})"
        )
    return value


def _choice(mapping: dict, key: str, path: str, options):
    value = _require(mapping, key, path, str)
    if value not in options:
        raise ConfigError(
            f"config key {_join(path, key)!r} must be one of "
            f"{sorted(options)}, got {value!r}"
        )
    return value


def _reject_unused(mapping: dict, key: str, path: str, because: str):
    if key in mapping:
        raise ConfigError(
            f"config key {_join(path, key)!r} is set but has no effect {because}. "
            "Remove it: a config must describe exactly what was run."
        )


def _finish(mapping: dict, path: str):
    if mapping:
        raise ConfigError(
            f"unknown config key(s) under {path or '<root>'}: {sorted(mapping)}"
        )


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _positive(value, name: str):
    if not (value > 0):
        raise ConfigError(f"{name} must be positive, got {value!r}")
    return value


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DomainConfig:
    """Streamwise extent of the channel.

    SPEC v2 S3: `length` is DERIVED, not configured. The config supplies the
    development margin `alpha` and the outlet margin, and

        L* = alpha * (L_e / h)(Re_h) + outlet_margin

    is computed at load from the entrance-length correlation. A config may
    override `length` explicitly, but then it must still satisfy that bound and
    the loader raises if it does not -- a truncated domain produces a solution
    that looks converged and is wrong, which is exactly the failure class this
    project exists to catch, so the check fires before training rather than
    after.
    """

    length: float
    alpha: float
    outlet_margin: float
    entrance_length_over_h: float
    explicitly_overridden: bool


@dataclass(frozen=True)
class PhysicsConfig:
    reynolds: float
    pressure_scaling: PressureScaling
    formulation: str  # velocity_pressure | stream_function


@dataclass(frozen=True)
class FourierConfig:
    enabled: bool
    num_features: int | None
    bandwidth: float | None


@dataclass(frozen=True)
class RWFConfig:
    enabled: bool
    mean: float | None
    stddev: float | None


@dataclass(frozen=True)
class ModelConfig:
    arch: str  # mlp | modified_mlp
    width: int
    depth: int
    fourier: FourierConfig
    rwf: RWFConfig
    lift: str | None  # cubic | sinusoidal, psi_omega only (SPEC v2 S4)


@dataclass(frozen=True)
class BCConfig:
    inlet: str  # soft | lifting
    lifting_length: float | None
    outlet_velocity: str  # soft
    pressure_gauge: str | None  # soft | hard; None for psi_omega


@dataclass(frozen=True)
class SamplingConfig:
    n_interior: int
    n_validation: int
    n_inlet: int
    n_outlet: int
    resample_every: int
    seed: int


@dataclass(frozen=True)
class GradNormConfig:
    update_every: int
    alpha: float


@dataclass(frozen=True)
class WeightsConfig:
    scheme: str  # fixed | grad_norm
    values: Mapping[str, float]
    grad_norm: GradNormConfig | None


@dataclass(frozen=True)
class AdamConfig:
    steps: int
    lr_init: float
    decay_rate: float
    decay_steps: int


@dataclass(frozen=True)
class LBFGSConfig:
    enabled: bool
    max_iterations: int | None
    gtol: float | None


@dataclass(frozen=True)
class TrainingConfig:
    adam: AdamConfig
    lbfgs: LBFGSConfig
    select: str  # final | best_validation


@dataclass(frozen=True)
class EvaluationConfig:
    nx: int
    ny: int


@dataclass(frozen=True)
class Config:
    name: str
    seed: int
    case: str  # A | B
    inlet_uniform_value: float | None  # Case B only
    domain: DomainConfig
    correlation: EntranceLengthCorrelation
    physics: PhysicsConfig
    model: ModelConfig
    bc: BCConfig
    sampling: SamplingConfig
    weights: WeightsConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)

    # -- derived, never configurable -----------------------------------------

    @property
    def inlet_flux(self) -> float:
        """Non-dimensional flux entering at x* = 0.

        Case A: int (1 - y^2) dy = 4/3 over y in [-1, 1].
        Case B: uniform u* = U over the full height 2, so flux = 2U.
        """
        if self.case == "A":
            return 4.0 / 3.0
        return 2.0 * self.inlet_uniform_value

    @property
    def loss_terms(self) -> tuple[str, ...]:
        """Exactly the residual/BC terms this configuration produces.

        Walls never appear: no-slip is imposed by construction in the ansatz,
        so there is no wall loss term to weight (SPEC section 3).
        """
        if self.physics.formulation == "psi_omega":
            # SPEC v2 S4: two coupled second-order residuals. Continuity is
            # identical, and pressure is not in the system at all.
            terms = ["vorticity", "vorticity_transport"]
        else:
            terms = ["momentum_x", "momentum_y", "continuity"]
        if self.bc.inlet == "soft":
            terms.append("inlet")
        if self.bc.outlet_velocity == "soft":
            terms.append("outlet_velocity")
        if self.bc.pressure_gauge == "soft":
            terms.append("pressure_gauge")
        return tuple(terms)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_config(document: Mapping[str, Any]) -> Config:
    """Validate a already-loaded YAML mapping into a `Config`."""
    import copy

    raw = copy.deepcopy(dict(document))
    doc = copy.deepcopy(dict(document))

    name = _require(doc, "name", "", str)
    seed = _require(doc, "seed", "", int)
    case = _choice(doc, "case", "", {"A", "B"})

    # -- correlation (parsed before domain: S3 derives L* from it) --
    corr = _require(doc, "correlation", "", dict)
    if "citation" not in corr:
        raise ConfigError(
            "correlation.citation is required (SPEC v2 S1). The entrance-length "
            "constants differ between pipe and plane-channel geometry and must "
            "be read from the source; a wrong constant is invisible in the "
            "output, a missing citation is not."
        )
    correlation = EntranceLengthCorrelation(
        c0=_require(corr, "c0", "correlation", float),
        c1=_require(corr, "c1", "correlation", float),
        n=_require(corr, "n", "correlation", float),
        citation=_require(corr, "citation", "correlation", str),
    )
    _finish(corr, "correlation")

    # -- physics (parsed before domain: L* depends on Re_h) --
    p = _require(doc, "physics", "", dict)
    reynolds = _positive(_require(p, "reynolds", "physics", float), "physics.reynolds")
    if reynolds > 200.0:
        raise ConfigError(
            f"physics.reynolds = {reynolds} exceeds the Phase 1 ceiling of 200 "
            "(SPEC section 2: 'Do not attempt Re > 200 in Phase 1')."
        )
    pressure_scaling = _choice(p, "pressure_scaling", "physics", {"viscous", "inertial"})
    formulation = _choice(
        p, "formulation", "physics", {"velocity_pressure", "psi_omega"}
    )
    if case == "B":
        inlet_uniform_value = _positive(
            _require(p, "inlet_uniform_value", "physics", float),
            "physics.inlet_uniform_value",
        )
    else:
        _reject_unused(
            p, "inlet_uniform_value", "physics", "for case A (its inlet is parabolic)"
        )
        inlet_uniform_value = None
    physics = PhysicsConfig(reynolds, pressure_scaling, formulation)
    _finish(p, "physics")

    # -- domain (S3: L* derived from Re_h and the correlation) --
    d = _require(doc, "domain", "", dict)
    alpha = _positive(_require(d, "alpha", "domain", float), "domain.alpha")
    outlet_margin = _positive(
        _require(d, "outlet_margin", "domain", float), "domain.outlet_margin"
    )
    if outlet_margin < 5.0:
        raise ConfigError(
            f"domain.outlet_margin must be at least 5 (SPEC v2 S3), got "
            f"{outlet_margin}. The margin exists so the zero-gradient outflow "
            "condition is applied where the flow is already developed."
        )
    le_over_h = float(entrance_length_over_h(reynolds, correlation))
    required_length = alpha * le_over_h + outlet_margin
    if "length" in d:
        length = _positive(_require(d, "length", "domain", float), "domain.length")
        if length < required_length:
            raise ConfigError(
                f"domain.length = {length:g} is below the S3 bound "
                f"{required_length:g} = alpha ({alpha:g}) * L_e/h ({le_over_h:.4f}) "
                f"+ outlet_margin ({outlet_margin:g}) at Re_h = {reynolds:g}. "
                "A truncated domain produces a solution that looks converged "
                "and is wrong."
            )
        overridden = True
    else:
        length = required_length
        overridden = False
    domain = DomainConfig(
        length=length,
        alpha=alpha,
        outlet_margin=outlet_margin,
        entrance_length_over_h=le_over_h,
        explicitly_overridden=overridden,
    )
    _finish(d, "domain")

    # -- model --
    m = _require(doc, "model", "", dict)
    arch = _choice(m, "arch", "model", {"mlp", "modified_mlp"})
    width = _positive(_require(m, "width", "model", int), "model.width")
    depth = _positive(_require(m, "depth", "model", int), "model.depth")

    f = _require(m, "fourier", "model", dict)
    fourier_enabled = _require(f, "enabled", "model.fourier", bool)
    if fourier_enabled:
        fourier = FourierConfig(
            True,
            _positive(_require(f, "num_features", "model.fourier", int), "num_features"),
            _positive(_require(f, "bandwidth", "model.fourier", float), "bandwidth"),
        )
    else:
        _reject_unused(f, "num_features", "model.fourier", "when fourier.enabled is false")
        _reject_unused(f, "bandwidth", "model.fourier", "when fourier.enabled is false")
        fourier = FourierConfig(False, None, None)
    _finish(f, "model.fourier")

    r = _require(m, "rwf", "model", dict)
    rwf_enabled = _require(r, "enabled", "model.rwf", bool)
    if rwf_enabled:
        rwf = RWFConfig(
            True,
            _require(r, "mean", "model.rwf", float),
            _positive(_require(r, "stddev", "model.rwf", float), "model.rwf.stddev"),
        )
    else:
        _reject_unused(r, "mean", "model.rwf", "when rwf.enabled is false")
        _reject_unused(r, "stddev", "model.rwf", "when rwf.enabled is false")
        rwf = RWFConfig(False, None, None)
    _finish(r, "model.rwf")

    if formulation == "psi_omega":
        lift = _choice(m, "lift", "model", {"cubic", "sinusoidal"})
    else:
        _reject_unused(m, "lift", "model", "for the velocity_pressure formulation")
        lift = None
    model = ModelConfig(arch, width, depth, fourier, rwf, lift)
    _finish(m, "model")

    # -- bc --
    b = _require(doc, "bc", "", dict)
    inlet = _choice(b, "inlet", "bc", {"soft", "lifting"})
    if inlet == "lifting":
        if case == "B":
            raise ConfigError(
                "bc.inlet = 'lifting' is not available for case B: a uniform "
                "inlet u* = 1 is incompatible with the hard no-slip wall "
                "ansatz at the corners (y* = +-1, x* = 0). Use 'soft'."
            )
        lifting_length = _positive(
            _require(b, "lifting_length", "bc", float), "bc.lifting_length"
        )
    else:
        _reject_unused(b, "lifting_length", "bc", "when bc.inlet is 'soft'")
        lifting_length = None
    outlet_velocity = _choice(b, "outlet_velocity", "bc", {"soft"})
    if formulation == "psi_omega":
        # Pressure is absent from the psi-omega system, so there is no pressure
        # ansatz to choose. Recovery is post hoc from a gauge point fixed at the
        # outlet centreline -- that is the definition of the gauge, not a tunable.
        _reject_unused(
            b, "pressure_gauge", "bc",
            "for the psi_omega formulation, which has no pressure in its system "
            "(recovery is post hoc from a gauge point at the outlet centreline)",
        )
        pressure_gauge = None
    else:
        pressure_gauge = _choice(b, "pressure_gauge", "bc", {"soft", "hard"})
    bc = BCConfig(inlet, lifting_length, outlet_velocity, pressure_gauge)
    _finish(b, "bc")

    # -- sampling --
    s = _require(doc, "sampling", "", dict)
    sampling = SamplingConfig(
        n_interior=_positive(_require(s, "n_interior", "sampling", int), "n_interior"),
        # Held-out collocation points, never trained on. L-BFGS runs on a FROZEN
        # point set, so it can drive the residual down on those specific points
        # while the true error rises. Detecting that needs points the optimiser
        # has never seen -- and unlike a comparison against the analytic
        # solution, this still works in Phase 2 where no analytic solution
        # exists.
        n_validation=_positive(
            _require(s, "n_validation", "sampling", int), "n_validation"
        ),
        n_inlet=_positive(_require(s, "n_inlet", "sampling", int), "n_inlet"),
        n_outlet=_positive(_require(s, "n_outlet", "sampling", int), "n_outlet"),
        resample_every=_positive(
            _require(s, "resample_every", "sampling", int), "resample_every"
        ),
        seed=_require(s, "seed", "sampling", int),
    )
    _finish(s, "sampling")

    # -- training --
    t = _require(doc, "training", "", dict)
    a = _require(t, "adam", "training", dict)
    adam = AdamConfig(
        steps=_require(a, "steps", "training.adam", int),
        lr_init=_positive(_require(a, "lr_init", "training.adam", float), "lr_init"),
        decay_rate=_positive(
            _require(a, "decay_rate", "training.adam", float), "decay_rate"
        ),
        decay_steps=_positive(
            _require(a, "decay_steps", "training.adam", int), "decay_steps"
        ),
    )
    if adam.steps < 0:
        raise ConfigError("training.adam.steps must be >= 0")
    _finish(a, "training.adam")

    lb = _require(t, "lbfgs", "training", dict)
    lbfgs_enabled = _require(lb, "enabled", "training.lbfgs", bool)
    if lbfgs_enabled:
        lbfgs = LBFGSConfig(
            True,
            _positive(
                _require(lb, "max_iterations", "training.lbfgs", int), "max_iterations"
            ),
            _positive(_require(lb, "gtol", "training.lbfgs", float), "gtol"),
        )
    else:
        _reject_unused(lb, "max_iterations", "training.lbfgs", "when lbfgs is disabled")
        _reject_unused(lb, "gtol", "training.lbfgs", "when lbfgs is disabled")
        lbfgs = LBFGSConfig(False, None, None)
    _finish(lb, "training.lbfgs")
    select = _choice(t, "select", "training", {"final", "best_validation"})
    training = TrainingConfig(adam, lbfgs, select)
    _finish(t, "training")

    # -- evaluation --
    e = _require(doc, "evaluation", "", dict)
    evaluation = EvaluationConfig(
        nx=_positive(_require(e, "nx", "evaluation", int), "evaluation.nx"),
        ny=_positive(_require(e, "ny", "evaluation", int), "evaluation.ny"),
    )
    _finish(e, "evaluation")

    # -- weights (validated last: the active term list depends on everything
    #    above, and the config must supply a weight for exactly those terms) --
    w = _require(doc, "weights", "", dict)
    scheme = _choice(w, "scheme", "weights", {"fixed", "grad_norm"})
    values = _require(w, "values", "weights", dict)
    if scheme == "grad_norm":
        g = _require(w, "grad_norm", "weights", dict)
        grad_norm = GradNormConfig(
            update_every=_positive(
                _require(g, "update_every", "weights.grad_norm", int), "update_every"
            ),
            alpha=_require(g, "alpha", "weights.grad_norm", float),
        )
        if not 0.0 <= grad_norm.alpha <= 1.0:
            raise ConfigError("weights.grad_norm.alpha must lie in [0, 1]")
        _finish(g, "weights.grad_norm")
    else:
        _reject_unused(w, "grad_norm", "weights", "when weights.scheme is 'fixed'")
        grad_norm = None
    weights = WeightsConfig(scheme, dict(values), grad_norm)
    _finish(w, "weights")

    _finish(doc, "")

    config = Config(
        name=name,
        seed=seed,
        case=case,
        inlet_uniform_value=inlet_uniform_value,
        domain=domain,
        correlation=correlation,
        physics=physics,
        model=model,
        bc=bc,
        sampling=sampling,
        weights=weights,
        training=training,
        evaluation=evaluation,
        raw=raw,
    )

    expected_terms = set(config.loss_terms)
    given_terms = set(values)
    if given_terms != expected_terms:
        missing = sorted(expected_terms - given_terms)
        extra = sorted(given_terms - expected_terms)
        raise ConfigError(
            "weights.values must name exactly the active loss terms "
            f"{sorted(expected_terms)}."
            + (f" Missing: {missing}." if missing else "")
            + (f" Not active for this configuration: {extra}." if extra else "")
        )
    for term, value in values.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigError(f"weights.values.{term} must be a number, got {value!r}")
        if value < 0:
            raise ConfigError(f"weights.values.{term} must be >= 0, got {value!r}")

    return config


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML experiment file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ConfigError(f"{path}: top level of a config must be a mapping")
    try:
        return parse_config(document)
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from None
'''

# ---------------------------------------------------------------------------
# src/evaluation/__init__.py
# ---------------------------------------------------------------------------
FILES['src/evaluation/__init__.py'] = r'''"""Accuracy metrics, conservation checks, residual maps and plots."""
'''

# ---------------------------------------------------------------------------
# src/evaluation/accuracy.py
# ---------------------------------------------------------------------------
FILES['src/evaluation/accuracy.py'] = r'''"""Case-specific accuracy evaluation on the fixed evaluation grid.

Never on the training points (SPEC section 9). The grid is rebuilt from the
config, so the same config always evaluates on the same points regardless of
what the sampler did.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..config import Config
from ..geometry.domain import Channel
from ..geometry.sampling import evaluation_grid
from ..physics import analytic
from ..physics.conventions import entrance_length_over_h
from ..physics.derivatives import batched_field_derivatives
from ..physics import residuals, streamfunction
from ..physics.derivatives import batched_stream_derivatives
from .drift import mass_drift
from .metrics import (
    centreline_velocity,
    cross_channel_flux,
    entrance_length,
    evaluate_fields,
    flux_conservation_error,
    max_abs,
    relative_l2,
)

#: SPEC test 3 asks for ten streamwise stations.
N_FLUX_STATIONS = 10

#: Stations are trimmed 5% away from the inlet and outlet planes so that a
#: boundary-condition error is not counted as a mass-conservation error.
FLUX_STATION_MARGIN = 0.05


def _residual_norms(config, model, params, points):
    """Max and RMS of each unweighted residual on the evaluation grid.

    Reported as a DIAGNOSTIC. A small residual is not an accuracy claim: the
    failure mode this project exists to catch is exactly a low residual with a
    wrong solution.
    """
    out = {}
    if config.physics.formulation == "psi_omega":
        sd = batched_stream_derivatives(lambda xy: model.stream(params, xy), points)
        names = ("vorticity", "vorticity_transport")
        values = streamfunction.residuals(sd, config.physics.reynolds)
    else:
        d = batched_field_derivatives(lambda xy: model.field(params, xy), points)
        names = ("momentum_x", "momentum_y", "continuity")
        values = residuals.navier_stokes(
            d, config.physics.reynolds, config.physics.pressure_scaling
        )
    for name, value in zip(names, values):
        value = np.asarray(value)
        out[f"residual_{name}_max"] = float(np.max(np.abs(value)))
        out[f"residual_{name}_rms"] = float(np.sqrt(np.mean(value**2)))
    return out


def _flux_metrics(config, model, params):
    length = config.domain.length
    stations = np.linspace(
        FLUX_STATION_MARGIN * length,
        (1.0 - FLUX_STATION_MARGIN) * length,
        N_FLUX_STATIONS,
    )
    fluxes = cross_channel_flux(model.field, params, stations)
    return {
        "flux_stations": stations.tolist(),
        "flux_values": np.asarray(fluxes).tolist(),
        "flux_mean": float(np.mean(fluxes)),
        # SPEC v2 S7 defines D against the FIRST station, so a monotonic
        # downstream drift shows at full size rather than half of it. The
        # spread-about-the-mean figure is kept alongside for comparison with
        # the SPEC v1 numbers.
        "flux_conservation_relative_error": mass_drift(fluxes),
        "flux_spread_about_mean": flux_conservation_error(fluxes),
        "flux_relative_error_vs_inlet": float(
            abs(np.mean(fluxes) - config.inlet_flux) / config.inlet_flux
        ),
    }


def evaluate(config: Config, model, params) -> dict:
    """All reported metrics for a finished run."""
    channel = Channel(config.domain.length)
    points, x, y = evaluation_grid(channel, config.evaluation.nx, config.evaluation.ny)

    u, v, p = evaluate_fields(model.field, params, points)

    metrics = {
        "case": config.case,
        "reynolds": config.physics.reynolds,
        "pressure_scaling": config.physics.pressure_scaling,
        "formulation": config.physics.formulation,
        "evaluation_points": int(points.shape[0]),
        "v_max_abs": max_abs(v),
        "u_max": float(np.max(u)),
    }
    metrics["has_pressure_in_system"] = bool(model.has_pressure)
    if model.lift is not None:
        metrics["lift"] = model.lift
    metrics.update(_flux_metrics(config, model, params))
    metrics.update(_residual_norms(config, model, params, points))

    if not model.has_pressure:
        # SPEC v2 S4: pressure is absent from the psi-omega system and is
        # recovered post hoc by integrating grad p from the outlet gauge point.
        # Path independence is asserted, not assumed: grad p is only a gradient
        # if the solution actually satisfies momentum.
        metrics.update(
            _recovered_pressure_metrics(config, model, params, np.asarray(y))
        )

    grid_y = np.asarray(points[:, 1])
    grid_x = np.asarray(points[:, 0])

    if config.case == "A":
        # The exact solution is known everywhere.
        u_exact = analytic.u_nd(grid_y)
        p_exact = analytic.p_nd(
            grid_x,
            config.domain.length,
            config.physics.reynolds,
            config.physics.pressure_scaling,
        )
        metrics["u_relative_l2"] = relative_l2(u, u_exact)
        metrics["u_max_abs_error"] = max_abs(u - u_exact)
        if model.has_pressure:
            metrics["p_relative_l2"] = relative_l2(p, p_exact)
            metrics["p_max_abs_error"] = max_abs(p - p_exact)
    else:
        # No closed-form solution. Two checks apply (SPEC section 2, Case B):
        # the downstream profile, and the entrance length.
        #
        # A uniform inlet u* = 1 is DISCONTINUOUS at the corners against hard
        # no-slip walls: the ansatz u = phi(y) N_u would need N_u = 1/(1-y^2)
        # at the inlet, which is unbounded. The network therefore delivers
        # slightly less than the nominal flux, and that deficit is an inlet
        # boundary-condition artefact, not a solver error.
        #
        # So the developed profile is compared two ways, and both are reported:
        #   * against the NOMINAL flux from the config -- includes the deficit;
        #   * against the MEASURED flux -- a pure shape test, which is the
        #     question Phase 2 actually cares about ("does the flow redevelop
        #     into the right profile?"). Mass conservation is checked separately
        #     and independently by the flux metrics above.
        measured_flux = float(metrics["flux_mean"])
        nominal = analytic.developed_profile_nd(np.asarray(y), config.inlet_flux)
        shaped = analytic.developed_profile_nd(np.asarray(y), measured_flux)

        outlet_points = jnp.stack(
            [jnp.full_like(y, config.domain.length), y], axis=-1
        )
        u_outlet, _, _ = evaluate_fields(model.field, params, outlet_points)
        metrics["outlet_profile_relative_l2_vs_nominal_flux"] = relative_l2(
            u_outlet, nominal
        )
        metrics["outlet_profile_relative_l2"] = relative_l2(u_outlet, shaped)
        metrics["developed_centreline_nominal"] = float(0.75 * config.inlet_flux)
        metrics["developed_centreline_measured"] = float(0.75 * measured_flux)
        metrics["inlet_flux_deficit_relative"] = float(
            (config.inlet_flux - measured_flux) / config.inlet_flux
        )

        stations = np.linspace(0.0, config.domain.length, 2001)
        centreline = centreline_velocity(model.field, params, stations)
        # The 99% threshold is against the flux the solution actually carries.
        measured = entrance_length(stations, centreline, 0.75 * measured_flux)
        # SPEC v2 S1: the correlation consumes Re_Dh, not Re_h, and returns a
        # length in D_h. `entrance_length_over_h` converts both internally, so
        # this call site cannot get the factor of 8/3 or the factor of 4 wrong.
        correlation = float(
            entrance_length_over_h(config.physics.reynolds, config.correlation)
        )
        metrics["entrance_length_measured"] = measured
        metrics["entrance_length_correlation"] = correlation
        metrics["entrance_length_relative_error"] = (
            float(abs(measured - correlation) / correlation)
            if np.isfinite(measured)
            else float("nan")
        )
        # T-S2.2: the coupling between profile error and mass drift stays
        # visible in every run. A ratio near 1 means the profile error IS the
        # mass drift wearing an amplitude disguise. Undefined once the drift is
        # at machine zero (stream function), where it is reported as such.
        drift = metrics["flux_conservation_relative_error"]
        metrics["profile_error_over_mass_error"] = (
            float(metrics["outlet_profile_relative_l2"] / drift)
            if drift > 1e-12
            else None
        )
        metrics["centreline_stations"] = stations[::100].tolist()
        metrics["centreline_values"] = np.asarray(centreline)[::100].tolist()

    return metrics


def _recovered_pressure_metrics(config, model, params, y_grid) -> dict:
    """Post-hoc pressure for the psi-omega system, plus the path-independence check.

    Sampled on a modest set of probe points rather than the full evaluation
    grid: each probe costs a path integral whose integrand needs second
    derivatives of velocity, i.e. third derivatives of psi.
    """
    velocity_fn = lambda xy: model.velocity(params, xy)  # noqa: E731
    length = config.domain.length
    gauge = (length, 0.0)

    probes = [
        (x, yy)
        for x in np.linspace(0.05 * length, 0.95 * length, 7)
        for yy in (-0.6, 0.0, 0.6)
    ]

    x_first, y_first = [], []
    for probe in probes:
        x_first.append(
            streamfunction.recover_pressure(
                velocity_fn, probe, gauge, config.physics.reynolds,
                config.physics.pressure_scaling, path="x_first",
            )
        )
        y_first.append(
            streamfunction.recover_pressure(
                velocity_fn, probe, gauge, config.physics.reynolds,
                config.physics.pressure_scaling, path="y_first",
            )
        )
    x_first = np.asarray(x_first)
    y_first = np.asarray(y_first)

    out = {
        "pressure_path_max_disagreement": float(np.max(np.abs(x_first - y_first))),
        "pressure_recovery_probes": len(probes),
    }
    if config.case == "A":
        exact = analytic.p_nd(
            np.array([p[0] for p in probes]), length,
            config.physics.reynolds, config.physics.pressure_scaling,
        )
        out["p_relative_l2_recovered"] = relative_l2(x_first, exact)
    return out
'''

# ---------------------------------------------------------------------------
# src/evaluation/drift.py
# ---------------------------------------------------------------------------
FILES['src/evaluation/drift.py'] = r'''"""Mass-drift metric and its convergence fit (SPEC v2, S7).

With `M(x*) = integral u* dy*` on a FIXED evaluation grid at `K >= 10` stations:

    D = max_x | M(x*) - M(0) | / M(0)

Test 3 asserts `D <= 1e-6` and the tolerance stays. Test 1 measures a pointwise
field error; test 3 measures the variation of an integral, and integration
normally averages errors down, so a tighter bound on the integral is the
correct default. `D` exceeding the pointwise error means the error is
SYSTEMATIC AND ACCUMULATING DOWNSTREAM rather than cancelling -- which is the
finding, and relaxing the bound would erase it.

The discriminating measurement is the trace of `D` against L-BFGS iteration,
fitted to

    D(n) = D_inf + a n^-b

* `D_inf` significantly above 1e-6 -> formulation bias. Report it with a
  confidence interval and close the test as a characterised limitation.
* `D_inf` consistent with zero -> under-convergence, and a longer run clears it.

Either outcome is a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def mass_drift(fluxes) -> float:
    """D, referenced to the FIRST station as SPEC v2 S7 defines it.

    Note this differs from a spread-about-the-mean measure: referencing the
    inlet station makes a monotonic downstream drift show at its full size
    rather than half of it.
    """
    fluxes = np.asarray(fluxes, dtype=np.float64)
    if fluxes.size < 10:
        raise ValueError(
            f"SPEC v2 S7 requires at least 10 stations, got {fluxes.size}"
        )
    reference = float(fluxes[0])
    if reference == 0.0:
        raise ValueError("flux at the first station is zero; D is undefined")
    return float(np.max(np.abs(fluxes - reference)) / abs(reference))


@dataclass(frozen=True)
class DriftFit:
    """Result of fitting D(n) = D_inf + a n^-b."""

    d_infinity: float
    amplitude: float
    exponent: float
    d_infinity_stderr: float
    points: int
    converged: bool
    message: str
    at_machine_precision: bool = False
    tail_at_last_point: float = float("nan")
    last_iteration: float = float("nan")
    decay_range: float = float("nan")

    #: The fitted transient must decay by at least this factor across the
    #: traced range for D_inf to be a measurement rather than an extrapolation.
    #: Calibrated on synthetic traces: a decay range of 28 recovers D_inf to
    #: 4.5% and 193 to better than 0.1%, while 6.8 is 33% out and 4.2 is wrong
    #: by a factor of 25. One decade is the conservative cut between them.
    MINIMUM_DECAY_RANGE = 10.0

    @property
    def identifiable(self) -> bool:
        """Whether the trace actually pins D_inf down.

        D_inf is the asymptote of D(n) = D_inf + a n^-b. What determines
        whether it can be measured is not how big the tail is, but how much the
        tail DECAYED across the traced range: without that leverage the fit
        cannot separate the constant from the transient.

        The fitted standard error is not usable for this. On synthetic traces
        it underestimates the true error by roughly an order of magnitude
        (0.023 reported against 0.33 actual, 0.089 against 25), which is
        typical for a weakly constrained parameter in a nonlinear fit. The
        decay range separates the same cases cleanly.
        """
        if not self.converged or not np.isfinite(self.decay_range):
            return False
        return self.decay_range >= self.MINIMUM_DECAY_RANGE

    @property
    def consistent_with_zero(self) -> bool:
        """True when D_inf is within two standard errors of zero."""
        if not np.isfinite(self.d_infinity_stderr):
            return False
        return self.d_infinity <= 2.0 * self.d_infinity_stderr

    def resolves(self, tolerance=1e-6) -> bool:
        """Whether the fit can say anything AT the tolerance scale.

        Separate from `identifiable`, which asks whether the trace has enough
        decay leverage. This asks whether the resulting uncertainty is small
        enough to matter: a short trace can fit a clean power law (good decay
        range) and still leave D_inf uncertain by far more than the tolerance
        being tested, in which case "consistent with zero" is true but says
        nothing. Measured on a 400-iteration run: D_inf = 2.2e-21 +- 9.8e-3
        against a 1e-6 tolerance.
        """
        if not np.isfinite(self.d_infinity_stderr):
            return False
        return self.d_infinity_stderr <= tolerance

    def verdict(self, tolerance=1e-6) -> str:
        if self.at_machine_precision:
            return (
                "D is at machine precision throughout: this formulation "
                "conserves mass by construction, so there is no drift to "
                "characterise and no convergence claim to make. Test 3 is "
                "VACUOUS here (see docs/falsifiability-matrix.md), not passed"
            )
        if not self.converged:
            return "fit did not converge; too few points or too little variation"
        if not self.identifiable:
            return (
                f"D_inf is NOT identifiable from this trace: the fitted "
                f"transient decayed by only {self.decay_range:.1f}x across it, "
                f"against the {self.MINIMUM_DECAY_RANGE:.0f}x needed to "
                "separate the constant from the transient. No bias or "
                "under-convergence verdict can be drawn -- extend the L-BFGS "
                "budget and refit"
            )
        if not self.resolves(tolerance):
            return (
                f"the trace cannot resolve D_inf at the {tolerance:.0e} scale: "
                f"the fitted uncertainty is +-{self.d_infinity_stderr:.1e}, "
                "larger than the tolerance being tested. Any verdict here would "
                "be an over-claim -- run longer before concluding anything"
            )
        if self.consistent_with_zero:
            return (
                "D_inf is consistent with zero: the residual drift is "
                "under-convergence, and a longer run should clear it"
            )
        if self.d_infinity > tolerance:
            return (
                f"D_inf = {self.d_infinity:.3e} +- {self.d_infinity_stderr:.1e} "
                f"exceeds {tolerance:.0e}: this is formulation bias, and test 3 "
                "should be closed as a characterised limitation rather than "
                "chased with more iterations"
            )
        return (
            f"D_inf = {self.d_infinity:.3e} +- {self.d_infinity_stderr:.1e} "
            f"lies below {tolerance:.0e}: more iterations should reach the target"
        )


def fit_drift(iterations, drifts) -> DriftFit:
    """Least-squares fit of D(n) = D_inf + a n^-b."""
    from scipy.optimize import curve_fit

    iterations = np.asarray(iterations, dtype=np.float64)
    drifts = np.asarray(drifts, dtype=np.float64)

    # A formulation that conserves mass by construction traces D at the
    # rounding floor. Fitting D(n) = D_inf + a n^-b to that is fitting noise,
    # and it produced the confident nonsense "D_inf = 1.0e-10, more iterations
    # should reach the target" for a run whose D was already exactly zero.
    finite = drifts[np.isfinite(drifts)]
    if finite.size and float(np.max(finite)) < 1e-12:
        return DriftFit(
            d_infinity=float(np.max(finite)), amplitude=0.0, exponent=float("nan"),
            d_infinity_stderr=0.0, points=int(finite.size), converged=False,
            message="D is at machine precision; nothing to fit",
            at_machine_precision=True,
        )

    keep = (iterations > 0) & np.isfinite(drifts) & (drifts > 0)
    iterations, drifts = iterations[keep], drifts[keep]

    if iterations.size < 5:
        return DriftFit(np.nan, np.nan, np.nan, np.nan, int(iterations.size), False,
                        "fewer than 5 usable trace points")

    def model(n, d_inf, a, b):
        return d_inf + a * n ** (-b)

    guess = (float(drifts[-1]), float(max(drifts[0] - drifts[-1], drifts[0])), 0.5)
    try:
        popt, pcov = curve_fit(
            model, iterations, drifts, p0=guess,
            bounds=([0.0, 0.0, 1e-3], [np.inf, np.inf, 5.0]), maxfev=20000,
        )
    except Exception as error:  # noqa: BLE001 - a failed fit is a reportable state
        return DriftFit(np.nan, np.nan, np.nan, np.nan, int(iterations.size), False,
                        f"curve_fit failed: {error}")

    stderr = float(np.sqrt(np.diag(pcov))[0]) if np.all(np.isfinite(pcov)) else np.nan
    first, last = float(iterations[0]), float(iterations[-1])
    tail = float(popt[1] * last ** (-popt[2]))
    decay = float((last / first) ** popt[2]) if first > 0 else float("nan")
    return DriftFit(
        d_infinity=float(popt[0]), amplitude=float(popt[1]), exponent=float(popt[2]),
        d_infinity_stderr=stderr, points=int(iterations.size), converged=True,
        message="ok", tail_at_last_point=tail, last_iteration=last,
        decay_range=decay,
    )
'''

# ---------------------------------------------------------------------------
# src/evaluation/falsifiability.py
# ---------------------------------------------------------------------------
FILES['src/evaluation/falsifiability.py'] = r'''"""Falsifiability matrix (SPEC v2, S5).

A test `T` with assertion `A(theta)` over network parameters is VACUOUS under
ansatz `F` if `A(F(theta))` holds for all `theta`. Then `T` reports a property
of `F`, not of the trained solution.

Vacuity is decidable by sampling: if a randomly initialised, UNTRAINED network
passes a test, that test cannot distinguish a trained solution from noise.

    LIVE      the untrained model fails at every seed -- the test bites
    WEAKENED  it passes at some seeds and not others
    VACUOUS   it passes at every seed -- the test measures the ansatz

This exists because Phase 1 hit the failure mode twice and nothing reported it:
hard no-slip silently emptied the wall tests, and the stream function silently
emptied the mass-conservation test. The defect is not the disarming, it is the
absence of detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry.domain import WALL, Channel
from ..geometry.sampling import evaluation_grid
from ..models.build import build_model
from ..physics import analytic
from .metrics import cross_channel_flux, evaluate_fields, flux_conservation_error, max_abs, relative_l2

LIVE = "LIVE"
WEAKENED = "WEAKENED"
VACUOUS = "VACUOUS"

#: A check can be LIVE at random initialisation and still be nearly free,
#: because the ansatz makes the correct answer a single trivially reachable
#: point in parameter space. Sampling at the initialisation distribution cannot
#: see that. `degenerate_targets` probes for it directly by zeroing the output
#: layer, i.e. setting the network's contribution identically to zero.
DEGENERATE = "DEGENERATE-TARGET"


@dataclass(frozen=True)
class Check:
    """One assertion, evaluated on a model without training it.

    `passes(config, model, params) -> bool` must mirror the real test's
    assertion, including its tolerance.
    """

    name: str
    passes: Callable
    spec_test: str = ""


def _wall_points(length, n=64):
    y = np.concatenate([np.full(n, WALL), np.full(n, -WALL)])
    x = np.concatenate([np.linspace(0.0, length, n)] * 2)
    return jnp.asarray(np.stack([x, y], axis=-1))


def _grid(config):
    return evaluation_grid(Channel(config.domain.length), 121, 61)


def check_wall_no_slip(config, model, params):
    values = jax.vmap(lambda xy: model.field(params, xy))(
        _wall_points(config.domain.length)
    )
    return max_abs(values[:, 0]) < 1e-12 and max_abs(values[:, 1]) < 1e-12


def check_test_1_u_relative_l2(config, model, params):
    points, _, _ = _grid(config)
    u, _, _ = evaluate_fields(model.field, params, points)
    exact = analytic.u_nd(np.asarray(points[:, 1]))
    return relative_l2(u, exact) <= 1e-5


def check_test_2_v_max_abs(config, model, params):
    points, _, _ = _grid(config)
    _, v, _ = evaluate_fields(model.field, params, points)
    return max_abs(v) <= 1e-5


def check_test_3_mass_conservation(config, model, params):
    length = config.domain.length
    stations = np.linspace(0.05 * length, 0.95 * length, 10)
    fluxes = cross_channel_flux(model.field, params, stations)
    try:
        return flux_conservation_error(fluxes) <= 1e-6
    except ValueError:
        return False


def check_inlet_profile(config, model, params):
    y = jnp.linspace(-1.0, 1.0, 129)
    points = jnp.stack([jnp.zeros_like(y), y], axis=-1)
    u, v, _ = evaluate_fields(model.field, params, points)
    if config.case == "A":
        target = analytic.u_nd(np.asarray(y))
    else:
        target = np.full(len(y), config.inlet_uniform_value)
    return max_abs(u - target) < 1e-6 and max_abs(v) < 1e-6


def check_pressure_gauge(config, model, params):
    if not model.has_pressure:
        return False  # no pressure in the system: the assertion does not apply
    y = jnp.linspace(-1.0, 1.0, 65)
    points = jnp.stack([jnp.full_like(y, config.domain.length), y], axis=-1)
    _, _, p = evaluate_fields(model.field, params, points)
    return max_abs(p) == 0.0


def check_always_true(config, model, params):
    """T-S5.2: a deliberately vacuous check, to prove the detector detects."""
    return True


def check_always_false(config, model, params):
    """Control in the other direction: must always report LIVE."""
    return False


DEFAULT_CHECKS = (
    Check("wall no-slip", check_wall_no_slip, "structural"),
    Check("test 1  u relative L2", check_test_1_u_relative_l2, "1"),
    Check("test 2  max |v*|", check_test_2_v_max_abs, "2"),
    Check("test 3  mass conservation", check_test_3_mass_conservation, "3"),
    Check("inlet profile", check_inlet_profile, "structural"),
    Check("pressure gauge at outlet", check_pressure_gauge, "structural"),
    Check("CONTROL always-true", check_always_true, "T-S5.2"),
    Check("CONTROL always-false", check_always_false, "T-S5.2"),
)


def classify(config, checks=DEFAULT_CHECKS, seeds=(0, 1, 2, 3, 4)):
    """Classify every check for one configuration, at random initialisation."""
    verdicts = {}
    for check in checks:
        outcomes = []
        for seed in seeds:
            params, model = build_model(config, jax.random.PRNGKey(seed))
            outcomes.append(bool(check.passes(config, model, params)))
        if all(outcomes):
            verdicts[check.name] = VACUOUS
        elif any(outcomes):
            verdicts[check.name] = WEAKENED
        else:
            verdicts[check.name] = LIVE
    return verdicts


def build_matrix(configs, checks=DEFAULT_CHECKS, seeds=(0, 1, 2, 3, 4)):
    """`{check name: {column label: verdict}}` across several configurations."""
    matrix = {check.name: {} for check in checks}
    for label, config in configs.items():
        for name, verdict in classify(config, checks, seeds).items():
            matrix[name][label] = verdict
    return matrix


def format_matrix(matrix, columns) -> str:
    width = max(len(name) for name in matrix) + 2
    header = "| " + "check".ljust(width) + "| " + " | ".join(
        c.ljust(10) for c in columns
    ) + " |"
    rule = "|" + "-" * (width + 1) + "|" + "|".join("-" * 12 for _ in columns) + "|"
    lines = [header, rule]
    for name, row in matrix.items():
        cells = " | ".join(row.get(c, "-").ljust(10) for c in columns)
        lines.append("| " + name.ljust(width) + "| " + cells + " |")
    return "\n".join(lines)


def all_vacuous_rows(matrix):
    """Checks that are vacuous in EVERY column.

    Such a row means the property is untested by this suite and must be flagged
    in the report, never silently accepted.
    """
    return [
        name
        for name, row in matrix.items()
        if row and all(v == VACUOUS for v in row.values())
    ]


def zero_network_contribution(params):
    """Set the output layer to zero, so the network contributes nothing.

    For a psi-omega model under the cubic lift this makes psi exactly the lift,
    whose derivative IS the Case A solution -- so any check comparing against
    that solution passes perfectly. Under the sinusoidal lift it does not.
    """
    import copy

    params = copy.deepcopy(params)
    output = params["output"] if "output" in params else params["layers"][-1]
    weight_key = "W" if "W" in output else "V"
    output[weight_key] = jnp.zeros_like(output[weight_key])
    output["b"] = jnp.zeros_like(output["b"])
    return params


def degenerate_targets(config, checks=DEFAULT_CHECKS, seed=0):
    """Checks whose assertion passes when the network contributes nothing.

    A check listed here is not vacuous -- an untrained network fails it -- but
    the ansatz has placed the correct answer at a single point the optimiser
    can reach exactly, so passing it demonstrates the lift rather than the
    network's expressivity. SPEC v2 S4 flags exactly this for the cubic lift;
    this function decides it by measurement instead of by argument.
    """
    params, model = build_model(config, jax.random.PRNGKey(seed))
    zeroed = zero_network_contribution(params)
    degenerate = []
    for check in checks:
        if check.name.startswith("CONTROL"):
            continue
        try:
            if check.passes(config, model, zeroed):
                degenerate.append(check.name)
        except Exception:  # a check that cannot even run is not degenerate
            continue
    return degenerate
'''

# ---------------------------------------------------------------------------
# src/evaluation/metrics.py
# ---------------------------------------------------------------------------
FILES['src/evaluation/metrics.py'] = r'''"""Accuracy metrics.

Accuracy is ALWAYS the relative L2 error against the analytic solution,
computed on the fixed evaluation grid. The loss value is a diagnostic and is
never reported as an accuracy claim (CLAUDE.md).

The published failure mode this guards against: a plain DeepXDE setup on this
exact problem reaching a total test loss of 8.7e-5 while its relative L2 error
was 8.6% (SPEC section 5).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry.domain import WALL


def relative_l2(predicted, exact):
    """||predicted - exact||_2 / ||exact||_2.

    Raises if the reference field is identically zero: the relative L2 error of
    a zero field is not defined, and silently returning an absolute error there
    would be exactly the kind of quiet lie this project is built to avoid. Use
    `max_abs` for fields whose exact value is zero (e.g. v*).
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    denominator = np.linalg.norm(exact)
    if denominator == 0.0:
        raise ValueError(
            "relative L2 error is undefined against an identically zero "
            "reference field; use max_abs instead"
        )
    return float(np.linalg.norm(predicted - exact) / denominator)


def max_abs(values):
    return float(np.max(np.abs(np.asarray(values))))


def gauss_legendre_nodes(n: int):
    """Gauss-Legendre nodes and weights mapped onto y* in [-1, 1].

    Used for every cross-channel integral. The channel half-height is exactly
    the Legendre interval, so no rescaling is needed, and GL is exact for the
    analytic parabola to machine precision -- which matters for SPEC test 3,
    whose tolerance is 1e-6 relative.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes * WALL, weights * WALL


def evaluate_fields(field_fn, params, points, batch_size: int = 8192):
    """(u, v, p) at an (N, 2) array of points, batched to bound peak memory."""
    apply = jax.jit(jax.vmap(lambda xy: field_fn(params, xy)))
    outputs = []
    for start in range(0, points.shape[0], batch_size):
        outputs.append(np.asarray(apply(points[start : start + batch_size])))
    stacked = np.concatenate(outputs, axis=0)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


def cross_channel_flux(field_fn, params, x_stations, n_nodes: int = 200):
    """int_{-1}^{1} u*(x, y) dy at each x station, by Gauss-Legendre quadrature."""
    nodes, weights = gauss_legendre_nodes(n_nodes)
    x_stations = np.asarray(x_stations, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_stations, nodes, indexing="ij")
    points = jnp.asarray(np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1))
    u, _, _ = evaluate_fields(field_fn, params, points)
    u = u.reshape(len(x_stations), n_nodes)
    return u @ weights


def flux_conservation_error(fluxes):
    """Relative spread of the flux across stations.

    Station-to-station AGREEMENT, not agreement with the analytic flux: this is
    the mass-conservation check (SPEC test 3), and a solution that conserves
    mass at the wrong flux is a different failure from one that does not
    conserve it at all. Both are reported.
    """
    fluxes = np.asarray(fluxes, dtype=np.float64)
    mean = float(np.mean(fluxes))
    if mean == 0.0:
        raise ValueError("mean flux is zero; conservation error is undefined")
    return float(np.max(np.abs(fluxes - mean)) / abs(mean))


def centreline_velocity(field_fn, params, x_stations):
    """u*(x, 0) at each station -- the entrance-length diagnostic."""
    x_stations = np.asarray(x_stations, dtype=np.float64)
    points = jnp.asarray(np.stack([x_stations, np.zeros_like(x_stations)], axis=-1))
    u, _, _ = evaluate_fields(field_fn, params, points)
    return u


def entrance_length(x_stations, centreline, developed_value, threshold=0.99):
    """First x* at which the centreline reaches `threshold` of its developed value.

    Linear interpolation between the two bracketing stations. Returns NaN if the
    threshold is never reached inside the domain, which is a real result (the
    domain is too short) and must not be silently clipped to L*.
    """
    x_stations = np.asarray(x_stations, dtype=np.float64)
    centreline = np.asarray(centreline, dtype=np.float64)
    target = threshold * developed_value
    reached = centreline >= target
    if not np.any(reached):
        return float("nan")
    index = int(np.argmax(reached))
    if index == 0:
        return float(x_stations[0])
    x0, x1 = x_stations[index - 1], x_stations[index]
    c0, c1 = centreline[index - 1], centreline[index]
    if c1 == c0:
        return float(x1)
    return float(x0 + (target - c0) * (x1 - x0) / (c1 - c0))
'''

# ---------------------------------------------------------------------------
# src/evaluation/plots.py
# ---------------------------------------------------------------------------
FILES['src/evaluation/plots.py'] = r'''"""Figures for a finished run. Written into the run directory.

Matplotlib only, no seaborn, one figure per file, default colours. These are
diagnostic figures for the Phase 1 write-up, not publication graphics.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import Config  # noqa: E402
from ..geometry.domain import Channel  # noqa: E402
from ..geometry.sampling import evaluation_grid  # noqa: E402
from ..physics import analytic  # noqa: E402
from ..physics import residuals  # noqa: E402
from ..physics.derivatives import batched_field_derivatives  # noqa: E402
from .metrics import evaluate_fields  # noqa: E402


def _grid_fields(config: Config, model, params):
    channel = Channel(config.domain.length)
    points, x, y = evaluation_grid(channel, config.evaluation.nx, config.evaluation.ny)
    u, v, p = evaluate_fields(model.field, params, points)
    shape = (config.evaluation.nx, config.evaluation.ny)
    return np.asarray(x), np.asarray(y), u.reshape(shape), v.reshape(shape), p.reshape(shape), points


def plot_velocity_profiles(config, model, params, path: Path):
    """Predicted vs analytic profile at several stations."""
    x, y, u, _, _, _ = _grid_fields(config, model, params)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    stations = np.linspace(0, len(x) - 1, 5).astype(int)
    for index in stations:
        axes[0].plot(u[index], y, label=f"x* = {x[index]:.1f}")
    if config.case == "A":
        reference = analytic.u_nd(y)
        label = "analytic  1 - y*²"
    else:
        reference = analytic.developed_profile_nd(y, config.inlet_flux)
        label = f"developed (flux {config.inlet_flux:.3f})"
    axes[0].plot(reference, y, "k--", linewidth=2, label=label)
    axes[0].set_xlabel("u*")
    axes[0].set_ylabel("y*")
    axes[0].set_title(f"{config.name}: streamwise profiles")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    if config.case == "A":
        error = np.abs(u - analytic.u_nd(y)[None, :])
        image = axes[1].pcolormesh(x, y, error.T, shading="auto")
        fig.colorbar(image, ax=axes[1], label="|u* - u*_exact|")
        axes[1].set_title("pointwise error in u*")
    else:
        axes[1].plot(x, u[:, u.shape[1] // 2])
        axes[1].axhline(0.75 * config.inlet_flux, color="k", linestyle="--",
                        label="developed centreline")
        axes[1].set_ylabel("u*(x*, 0)")
        axes[1].set_title("centreline development")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("x*")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_residual_map(config, model, params, path: Path):
    """Where the momentum residual lives. A diagnostic, never an accuracy claim."""
    channel = Channel(config.domain.length)
    points, x, y = evaluation_grid(channel, config.evaluation.nx, config.evaluation.ny)
    shape = (config.evaluation.nx, config.evaluation.ny)
    if config.physics.formulation == "psi_omega":
        from ..physics import streamfunction
        from ..physics.derivatives import batched_stream_derivatives

        sd = batched_stream_derivatives(lambda xy: model.stream(params, xy), points)
        values_and_names = list(
            zip(
                streamfunction.residuals(sd, config.physics.reynolds),
                ("r1  omega + lap psi", "r2  vorticity transport"),
            )
        )
    else:
        d = batched_field_derivatives(lambda xy: model.field(params, xy), points)
        values_and_names = list(
            zip(
                residuals.navier_stokes(
                    d, config.physics.reynolds, config.physics.pressure_scaling
                ),
                ("momentum x", "momentum y", "continuity"),
            )
        )

    fig, axes = plt.subplots(len(values_and_names), 1, figsize=(9, 8), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (values, name) in zip(axes, values_and_names):
        magnitude = np.abs(np.asarray(values)).reshape(shape).T
        floor = max(magnitude.max(), 1e-300) * 1e-8
        image = axis.pcolormesh(
            np.asarray(x), np.asarray(y), np.log10(np.maximum(magnitude, floor)),
            shading="auto",
        )
        fig.colorbar(image, ax=axis, label="log10 |residual|")
        axis.set_ylabel("y*")
        axis.set_title(f"{name} residual (unweighted, diagnostic only)")
    axes[-1].set_xlabel("x*")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_loss_history(history_path: Path, path: Path, title: str):
    """Per-term loss history from the Adam phase."""
    if not history_path.is_file():
        return
    with history_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    steps = np.array([float(row["step"]) for row in rows])
    term_columns = sorted(c for c in rows[0] if c.startswith("term_"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(steps, [float(row["loss"]) for row in rows], "k", label="total")
    for column in term_columns:
        values = np.array([float(row[column]) for row in rows])
        axes[0].semilogy(steps, np.maximum(values, 1e-300), label=column[5:], alpha=0.8)
    axes[0].set_xlabel("Adam step")
    axes[0].set_ylabel("loss (diagnostic, not accuracy)")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    weight_columns = sorted(c for c in rows[0] if c.startswith("weight_"))
    for column in weight_columns:
        axes[1].semilogy(steps, [float(row[column]) for row in rows], label=column[7:])
    axes[1].set_xlabel("Adam step")
    axes[1].set_ylabel("loss weight")
    axes[1].set_title("loss weights")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_all(config: Config, model, params, directory: Path):
    plot_velocity_profiles(config, model, params, directory / "profiles.png")
    plot_residual_map(config, model, params, directory / "residuals.png")
    plot_loss_history(
        directory / "history.csv", directory / "loss_history.png", config.name
    )
'''

# ---------------------------------------------------------------------------
# src/geometry/__init__.py
# ---------------------------------------------------------------------------
FILES['src/geometry/__init__.py'] = r'''"""Domain definitions, wall distance functions, and collocation sampling."""
'''

# ---------------------------------------------------------------------------
# src/geometry/domain.py
# ---------------------------------------------------------------------------
FILES['src/geometry/domain.py'] = r'''"""The Phase 1 domain: a straight channel x* in [0, L*], y* in [-1, 1].

Everything is non-dimensional. The half-height is 1 by construction -- that is
what "lengths are scaled by h" means -- so it is not a free parameter and is
not configurable.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

#: Non-dimensional wall position. Walls are at y* = -WALL and y* = +WALL.
WALL = 1.0


@dataclass(frozen=True)
class Channel:
    """Straight channel of non-dimensional length `length`."""

    length: float

    def __post_init__(self):
        if not (self.length > 0):
            raise ValueError(f"channel length must be positive, got {self.length!r}")

    @property
    def bounds_lower(self):
        return jnp.array([0.0, -WALL])

    @property
    def bounds_upper(self):
        return jnp.array([self.length, WALL])


def phi(y):
    """Wall distance function: phi(y*) = 1 - y*^2.

    Vanishes on both walls and is strictly positive between them, so
    multiplying a network output by it enforces no-slip EXACTLY rather than
    through a penalty. This is the single largest accuracy lever in the project
    (SPEC section 3), which is why it is a named function with its own tests
    rather than an inline expression.
    """
    return 1.0 - y**2


def dphi_dy(y):
    """d(phi)/dy = -2y. Written out so tests can check phi against it."""
    return -2.0 * y


def normalise(xy, length):
    """Map (x*, y*) from [0, L*] x [-1, 1] onto [-1, 1]^2.

    A tanh network fed x* in [0, 20] and y* in [-1, 1] sees two inputs whose
    scales differ by a factor 20; the first hidden layer then has to undo that
    with its weights. This is a structural property of the domain, not a tuned
    hyperparameter, so it is derived from the config's domain length rather
    than being configurable.
    """
    x, y = xy[..., 0], xy[..., 1]
    return jnp.stack([2.0 * x / length - 1.0, y / WALL], axis=-1)
'''

# ---------------------------------------------------------------------------
# src/geometry/lifts.py
# ---------------------------------------------------------------------------
FILES['src/geometry/lifts.py'] = r'''"""Stream-function lifts (SPEC v2, S4).

A lift `psi_lift(y*)` must satisfy

    psi_lift(+-1) = +- Q*/2      (the wall values differ by the flux)
    psi_lift'(+-1) = 0           (no-slip: u = dpsi/dy vanishes at the wall)

so that with `psi = psi_lift + phi^2 N_psi` and `phi = 1 - y*^2` both wall
conditions hold exactly for any `N_psi`.

Two lifts are provided because ONE OF THEM DISARMS THE GATE TEST.
"""

from __future__ import annotations

import jax.numpy as jnp


def cubic(y, flux):
    """psi = (3Q/4)(y - y^3/3).

    Check: psi(1) = (3Q/4)(2/3) = Q/2, and psi' = (3Q/4)(1 - y^2) = 0 at y = +-1.

    WARNING -- this lift IS the exact Case A solution. Its derivative is
    (3Q/4)(1 - y^2), which for Q = 4/3 is exactly 1 - y^2. Under this lift the
    correct Case A answer is `N_psi = 0`, so Case A stops testing the network's
    expressivity and tests only that the lift and training machinery are wired
    up. Use it as the MACHINERY check; never quote its accuracy as a result.
    """
    return 0.75 * flux * (y - y**3 / 3.0)


def sinusoidal(y, flux):
    """psi = (Q/2) sin(pi y / 2).

    Check: psi(+-1) = +-Q/2, and psi' = (Q pi/4) cos(pi y/2) = 0 at y = +-1.

    Satisfies the same boundary conditions as `cubic` but its derivative is NOT
    the Poiseuille profile, so `N_psi` must learn a non-zero correction. This is
    the EXPRESSIVITY check and the only one whose accuracy may be quoted.
    """
    return 0.5 * flux * jnp.sin(0.5 * jnp.pi * y)


LIFTS = {"cubic": cubic, "sinusoidal": sinusoidal}

#: Lifts whose derivative is the exact fully developed profile, so that the
#: correct answer is N_psi = 0. Reported by the sentinel matrix (S5).
DEGENERATE_LIFTS = frozenset({"cubic"})


def get(name):
    if name not in LIFTS:
        raise ValueError(f"unknown lift {name!r}; expected one of {sorted(LIFTS)}")
    return LIFTS[name]
'''

# ---------------------------------------------------------------------------
# src/geometry/sampling.py
# ---------------------------------------------------------------------------
FILES['src/geometry/sampling.py'] = r'''"""Collocation sampling.

Two rules from the SPEC drive the design here:

* the sampling seed is logged and every draw is a pure function of an explicit
  PRNG key, so a run is reproducible from (config, seed) alone;
* the fixed EVALUATION grid is built by a different function from the training
  points, because reported error must never be measured on training points
  (SPEC section 9, "Resampling collocation points").
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .domain import WALL, Channel


class CollocationPoints(NamedTuple):
    """One draw of training points. All arrays are (N, 2) of (x*, y*)."""

    interior: jnp.ndarray
    inlet: jnp.ndarray
    outlet: jnp.ndarray


def sample_interior(key, channel: Channel, n: int):
    """Uniform random points strictly inside the channel.

    Wall points are excluded: no-slip is exact by construction, so a
    collocation point on the wall contributes nothing but a degenerate residual
    (phi = 0 makes u and v identically zero there for any parameters).
    """
    return jax.random.uniform(
        key,
        shape=(n, 2),
        minval=channel.bounds_lower,
        maxval=channel.bounds_upper,
        dtype=jnp.float64,
    )


def sample_inlet(key, channel: Channel, n: int):
    """Points on x* = 0, y* uniform in (-1, 1)."""
    y = jax.random.uniform(key, shape=(n,), minval=-WALL, maxval=WALL, dtype=jnp.float64)
    return jnp.stack([jnp.zeros_like(y), y], axis=-1)


def sample_outlet(key, channel: Channel, n: int):
    """Points on x* = L*, y* uniform in (-1, 1)."""
    y = jax.random.uniform(key, shape=(n,), minval=-WALL, maxval=WALL, dtype=jnp.float64)
    return jnp.stack([jnp.full_like(y, channel.length), y], axis=-1)


def sample_all(key, channel: Channel, n_interior: int, n_inlet: int, n_outlet: int):
    """One complete draw. Pure function of `key`."""
    k_int, k_in, k_out = jax.random.split(key, 3)
    return CollocationPoints(
        interior=sample_interior(k_int, channel, n_interior),
        inlet=sample_inlet(k_in, channel, n_inlet),
        outlet=sample_outlet(k_out, channel, n_outlet),
    )


def evaluation_grid(channel: Channel, nx: int, ny: int):
    """Fixed tensor-product grid for reporting error. Never used for training.

    Endpoints are included: the walls are where the hard BC is exact and the
    inlet/outlet are where the boundary conditions live, so an error metric
    that skipped them would flatter the model.

    Returns (points (nx*ny, 2), x (nx,), y (ny,)).
    """
    x = jnp.linspace(0.0, channel.length, nx, dtype=jnp.float64)
    y = jnp.linspace(-WALL, WALL, ny, dtype=jnp.float64)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    points = jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    return points, x, y


def station_grid(channel: Channel, n_stations: int, ny: int, *, margin: float = 0.0):
    """Streamwise stations for the mass-conservation check (SPEC test 3).

    `margin` trims the stations away from the inlet and outlet planes; the
    conservation check is about the interior, and a station sitting exactly on
    a boundary would fold a BC error into a conservation error.

    Returns (points (n_stations, ny, 2), x_stations (n_stations,), y (ny,)).
    """
    x = jnp.linspace(
        margin * channel.length,
        (1.0 - margin) * channel.length,
        n_stations,
        dtype=jnp.float64,
    )
    y = jnp.linspace(-WALL, WALL, ny, dtype=jnp.float64)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    return jnp.stack([xx, yy], axis=-1), x, y
'''

# ---------------------------------------------------------------------------
# src/models/__init__.py
# ---------------------------------------------------------------------------
FILES['src/models/__init__.py'] = r'''"""Network architectures and the hard boundary-condition ansatz."""
'''

# ---------------------------------------------------------------------------
# src/models/build.py
# ---------------------------------------------------------------------------
FILES['src/models/build.py'] = r'''"""Assemble a model from a validated `Config`.

Returns `(params, field_fn)` where `field_fn(params, xy) -> [u, v, p]` at a
single non-dimensional point. Everything that varies between experiments comes
from the config; nothing here has a default.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..config import Config
from .fourier import apply_fourier, init_fourier_matrix
from .hard_bc import (
    make_psi_omega_field,
    make_velocity_from_stream,
    make_velocity_pressure_field,
)
from .mlp import apply_mlp, apply_modified_mlp, init_mlp, init_modified_mlp

#: Raw network outputs per formulation.
#: velocity_pressure -> (N_u, N_v, N_p);  psi_omega -> (N_psi, N_omega).
_N_OUTPUTS = {"velocity_pressure": 3, "psi_omega": 2}


@dataclass(frozen=True)
class Model:
    """What a formulation exposes.

    `has_pressure` is False for the psi-omega system: pressure is absent from
    that system and is recovered post hoc by path integration, so any pressure
    metric taken from `field` there would be meaningless. Callers must check
    the flag rather than trusting the third slot of `field`.
    """

    formulation: str
    field: object          # (params, xy) -> [u, v, p]; p is 0 when has_pressure is False
    velocity: object       # (params, xy) -> [u, v]
    stream: object         # (params, xy) -> [psi, omega], or None
    has_pressure: bool
    lift: object = None


def build_model(config: Config, key):
    """Build parameters and the field function for `config`."""
    fourier_key, net_key = jax.random.split(key)

    n_out = _N_OUTPUTS[config.physics.formulation]

    if config.model.fourier.enabled:
        matrix = init_fourier_matrix(
            fourier_key,
            config.model.fourier.num_features,
            config.model.fourier.bandwidth,
        )
        n_in = 2 * config.model.fourier.num_features
        embed = lambda xi: apply_fourier(matrix, xi)  # noqa: E731
    else:
        n_in = 2
        embed = lambda xi: xi  # noqa: E731

    rwf = dict(
        rwf_enabled=config.model.rwf.enabled,
        rwf_mean=config.model.rwf.mean,
        rwf_stddev=config.model.rwf.stddev,
    )

    if config.model.arch == "mlp":
        params = init_mlp(net_key, n_in, n_out, config.model.width, config.model.depth, **rwf)
        core = apply_mlp
    elif config.model.arch == "modified_mlp":
        params = init_modified_mlp(
            net_key, n_in, n_out, config.model.width, config.model.depth, **rwf
        )
        core = apply_modified_mlp
    else:
        raise ValueError(f"unknown model.arch {config.model.arch!r}")

    def net_apply(p, xi):
        return core(p, embed(xi))

    if config.physics.formulation == "velocity_pressure":
        field_fn = make_velocity_pressure_field(
            net_apply,
            length=config.domain.length,
            inlet_mode=config.bc.inlet,
            lifting_length=config.bc.lifting_length,
            gauge_mode=config.bc.pressure_gauge,
        )
        model = Model(
            formulation="velocity_pressure",
            field=field_fn,
            velocity=lambda p, xy: field_fn(p, xy)[:2],
            stream=None,
            has_pressure=True,
        )
    elif config.physics.formulation == "psi_omega":
        if config.bc.inlet == "lifting":
            raise ValueError(
                "bc.inlet = 'lifting' is not implemented for the psi_omega "
                "formulation: the inlet condition there is a condition on "
                "dpsi/dy, and a lifting construction exact for both u and v at "
                "x* = 0 needs lambda'(0) = 0 as well. Use bc.inlet = 'soft'."
            )
        stream_fn = make_psi_omega_field(
            net_apply,
            length=config.domain.length,
            flux=config.inlet_flux,
            lift=config.model.lift,
        )
        velocity_fn = make_velocity_from_stream(stream_fn)
        model = Model(
            formulation="psi_omega",
            # Pressure is absent from the psi-omega system. The zero here is a
            # placeholder that `has_pressure=False` forbids anyone reporting.
            field=lambda p, xy: jnp.concatenate(
                [velocity_fn(p, xy), jnp.zeros((1,), dtype=jnp.float64)]
            ),
            velocity=velocity_fn,
            stream=stream_fn,
            has_pressure=False,
            lift=config.model.lift,
        )
    else:
        raise ValueError(f"unknown physics.formulation {config.physics.formulation!r}")

    return params, model


def count_parameters(params) -> int:
    return sum(int(leaf.size) for leaf in jax.tree.leaves(params))
'''

# ---------------------------------------------------------------------------
# src/models/dense.py
# ---------------------------------------------------------------------------
FILES['src/models/dense.py'] = r'''"""Dense layers, with optional random weight factorisation (RWF).

RWF (SPEC section 3 item 4) reparameterises each weight matrix as

    W = diag(exp(s)) V,      s ~ N(mean, stddev)

so every output neuron carries its own learnable scale. Initialisation draws W
by Glorot normal first and then factors it, so switching RWF on does not change
the initial function the network represents -- only how the optimiser moves
through parameter space.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def glorot_normal(key, n_out: int, n_in: int):
    """Glorot (Xavier) normal initialisation, float64."""
    stddev = jnp.sqrt(2.0 / (n_in + n_out))
    return stddev * jax.random.normal(key, (n_out, n_in), dtype=jnp.float64)


def init_dense(key, n_in: int, n_out: int, *, rwf_enabled: bool, rwf_mean=None, rwf_stddev=None):
    """Initialise one dense layer as a parameter pytree."""
    w_key, s_key = jax.random.split(key)
    weight = glorot_normal(w_key, n_out, n_in)
    bias = jnp.zeros((n_out,), dtype=jnp.float64)

    if not rwf_enabled:
        return {"W": weight, "b": bias}

    if rwf_mean is None or rwf_stddev is None:
        raise ValueError("RWF requires both rwf_mean and rwf_stddev")
    scale = rwf_mean + rwf_stddev * jax.random.normal(s_key, (n_out,), dtype=jnp.float64)
    # V = W / exp(s), so that diag(exp(s)) V == W exactly at initialisation.
    factor = weight / jnp.exp(scale)[:, None]
    return {"s": scale, "V": factor, "b": bias}


def apply_dense(layer, x):
    """Apply one dense layer to a single input vector."""
    if "W" in layer:
        return layer["W"] @ x + layer["b"]
    return (jnp.exp(layer["s"])[:, None] * layer["V"]) @ x + layer["b"]


def layer_weight(layer):
    """The effective weight matrix of a layer, factorised or not."""
    if "W" in layer:
        return layer["W"]
    return jnp.exp(layer["s"])[:, None] * layer["V"]
'''

# ---------------------------------------------------------------------------
# src/models/fourier.py
# ---------------------------------------------------------------------------
FILES['src/models/fourier.py'] = r'''"""Random Fourier feature embedding of the input coordinates.

    gamma(xi) = [cos(B xi), sin(B xi)],   B ~ N(0, bandwidth^2)

with `xi` the coordinates already normalised to [-1, 1]^2. Addresses the
spectral bias of a plain MLP (SPEC section 3 item 2).

B is drawn once from the run's PRNG key and then held FIXED -- it is not a
trainable parameter, so it lives outside the parameter pytree and is never
touched by the optimiser. That also keeps the L-BFGS parameter vector free of
constants.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def init_fourier_matrix(key, num_features: int, bandwidth: float, n_inputs: int = 2):
    """Draw the fixed frequency matrix B of shape (num_features, n_inputs)."""
    return bandwidth * jax.random.normal(
        key, (num_features, n_inputs), dtype=jnp.float64
    )


def apply_fourier(matrix, xi):
    """Embed a single normalised coordinate vector. Output dim = 2 * num_features."""
    projection = matrix @ xi
    return jnp.concatenate([jnp.cos(projection), jnp.sin(projection)])
'''

# ---------------------------------------------------------------------------
# src/models/hard_bc.py
# ---------------------------------------------------------------------------
FILES['src/models/hard_bc.py'] = r'''"""The boundary-condition ansatz: what the network is allowed to represent.

No-slip is NEVER a penalty term in this project (SPEC section 3). The velocity
is constructed so that it vanishes on both walls for any parameter values:

    u*(x*, y*) = phi(y*) N_u(x*, y*)
    v*(x*, y*) = phi(y*) N_v(x*, y*)      phi(y*) = 1 - y*^2

Inlet handling
--------------
Two options, both implemented, chosen by config (`bc.inlet`):

* "soft"    -- an inlet loss term. The PINN has to DISCOVER the parabolic
               profile from the PDE plus a soft nudge. This is the default for
               Case A because it is the honest test of the harness: with
               lifting, u(0, y) = phi(y) = 1 - y^2 is already the exact Case A
               inlet condition and the inlet is no longer being solved for.
* "lifting" -- u = phi(y) [1 + lambda(x) N_u], v = phi(y) lambda(x) N_v with
               lambda(x) = 1 - exp(-x / l), lambda(0) = 0. This makes the
               Case A inlet exact by construction and removes the inlet loss
               term entirely. Only available for Case A: a uniform Case B inlet
               is incompatible with hard no-slip at the corners.

Pressure gauge
--------------
Pressure is unique only up to a constant. `bc.pressure_gauge`:

* "soft" -- a loss term pinning p*(L*, y*) = 0.
* "hard" -- p* = (1 - x*/L*) N_p, which is exactly zero on the outlet plane.

Without one of these the pressure relative L2 error is meaningless (SPEC
section 9).

Stream function
---------------
u = dpsi/dy, v = -dpsi/dx satisfies continuity identically. The wall condition
needs psi_y = psi_x = 0 at y* = +-1, which phi^2 gives:

    psi = psi_base(y*) + phi(y*)^2 N_psi(x*, y*)

phi^2 alone is NOT enough, though: psi(+1) - psi(-1) equals the flux, so a psi
that is constant on both walls can only represent zero net throughflow. Hence
`psi_base`, which carries the flux and has zero slope at both walls:

    "cubic"  psi_base = (Q/4)(3y - y^3)   -> psi_base' = (3Q/4)(1 - y^2)
    "sine"   psi_base = (Q/2) sin(pi y/2) -> psi_base' = (Q pi/4) cos(pi y/2)

The cubic base IS the exact Case A solution, so with it the network only has to
learn N_psi = 0 and the Case A comparison against the velocity-pressure
formulation would be meaningless. "sine" carries the same flux with the wrong
shape, so the network still has to do work. Use "sine" for any reported
comparison; "cubic" exists to isolate optimiser behaviour.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..geometry import lifts
from ..geometry.domain import normalise, phi


def _lifting(x, lifting_length):
    """lambda(x) = 1 - exp(-x/l): zero at the inlet, ~1 within a few l."""
    return 1.0 - jnp.exp(-x / lifting_length)


def _pressure_gauge_factor(x, length, gauge_mode):
    if gauge_mode == "soft":
        return 1.0
    if gauge_mode == "hard":
        return 1.0 - x / length
    raise ValueError(f"unknown pressure gauge mode {gauge_mode!r}")


def psi_base(y, flux, base):
    """Stream function carrying `flux` with zero slope at both walls."""
    if base == "cubic":
        return 0.25 * flux * (3.0 * y - y**3)
    if base == "sine":
        return 0.5 * flux * jnp.sin(0.5 * jnp.pi * y)
    raise ValueError(f"unknown stream_base {base!r}")


def make_velocity_pressure_field(
    net_apply, *, length, inlet_mode, lifting_length, gauge_mode
):
    """Build field_fn(params, xy) -> [u, v, p] for the u-v-p formulation."""

    def field_fn(params, xy):
        x, y = xy[0], xy[1]
        raw = net_apply(params, normalise(xy, length))
        wall = phi(y)

        if inlet_mode == "soft":
            u = wall * raw[0]
            v = wall * raw[1]
        elif inlet_mode == "lifting":
            lift = _lifting(x, lifting_length)
            # u(0, y) = phi(y) = 1 - y^2 exactly; v(0, y) = 0 exactly.
            u = wall * (1.0 + lift * raw[0])
            v = wall * lift * raw[1]
        else:
            raise ValueError(f"unknown inlet mode {inlet_mode!r}")

        p = _pressure_gauge_factor(x, length, gauge_mode) * raw[2]
        return jnp.array([u, v, p])

    return field_fn


def make_stream_function_field(net_apply, *, length, flux, base, gauge_mode):
    """Build field_fn(params, xy) -> [u, v, p] for the stream-function form.

    Continuity is satisfied identically, so there is no continuity loss term.
    The cost is one extra order of differentiation in the momentum residual:
    u_xx here is a third derivative of the network.
    """

    def stream(params, xy):
        y = xy[1]
        raw = net_apply(params, normalise(xy, length))
        return psi_base(y, flux, base) + phi(y) ** 2 * raw[0]

    def field_fn(params, xy):
        gradient = jax.grad(stream, argnums=1)(params, xy)
        u = gradient[1]
        v = -gradient[0]
        raw = net_apply(params, normalise(xy, length))
        p = _pressure_gauge_factor(xy[0], length, gauge_mode) * raw[1]
        return jnp.array([u, v, p])

    return field_fn


def make_psi_omega_field(net_apply, *, length, flux, lift):
    """Build the psi-omega pair for SPEC v2 S4.

        psi   = psi_lift(y*) + phi(y*)^2 N_psi(x*, y*)
        omega = N_omega(x*, y*)                          -- NO hard constraint

    At the walls phi = 0, so psi = psi_lift there and
    d_y[phi^2 N] = 2 phi phi' N + phi^2 d_y N = 0. Both wall conditions hold
    exactly for any parameters, which is what pins each wall to its own
    streamline constant and hence fixes the flux.

    `omega` is deliberately left free. It is coupled only through
    r_1 = omega + lap psi, which the network can violate -- that residual is
    what partially restores the falsifiability the pure-psi form destroys.
    Hard-constraining omega would remove it (SPEC v2 S4, and see S5).
    """
    lift_fn = lifts.get(lift)

    def stream_fn(params, xy):
        y = xy[1]
        raw = net_apply(params, normalise(xy, length))
        psi = lift_fn(y, flux) + phi(y) ** 2 * raw[0]
        omega = raw[1]
        return jnp.array([psi, omega])

    return stream_fn


def make_velocity_from_stream(stream_fn):
    """(u, v) = (dpsi/dy, -dpsi/dx) from a psi-omega stream function."""

    def velocity_fn(params, xy):
        gradient = jax.grad(lambda p, z: stream_fn(p, z)[0], argnums=1)(params, xy)
        return jnp.array([gradient[1], -gradient[0]])

    return velocity_fn
'''

# ---------------------------------------------------------------------------
# src/models/mlp.py
# ---------------------------------------------------------------------------
FILES['src/models/mlp.py'] = r'''"""Baseline tanh MLP and the modified (two-encoder) MLP.

Both take an already-embedded input vector and return `n_out` raw outputs; the
hard boundary-condition ansatz is applied outside, in `hard_bc.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .dense import apply_dense, init_dense


def init_mlp(key, n_in: int, n_out: int, width: int, depth: int, **rwf):
    """Depth `depth` counts HIDDEN layers, so there are depth+1 dense layers."""
    sizes = [n_in] + [width] * depth + [n_out]
    keys = jax.random.split(key, len(sizes) - 1)
    return {
        "layers": [
            init_dense(k, a, b, **rwf) for k, a, b in zip(keys, sizes[:-1], sizes[1:])
        ]
    }


def apply_mlp(params, x):
    *hidden, last = params["layers"]
    for layer in hidden:
        x = jnp.tanh(apply_dense(layer, x))
    return apply_dense(last, x)


def init_modified_mlp(key, n_in: int, n_out: int, width: int, depth: int, **rwf):
    """Wang et al.'s modified MLP: two input encoders mixed multiplicatively.

        U = tanh(W_u xi + b_u),  V = tanh(W_v xi + b_v)
        H_1 = tanh(W_1 xi + b_1)
        Z_k = tanh(W_k H_k + b_k)
        H_{k+1} = (1 - Z_k) * U + Z_k * V

    The encoders let every hidden layer see the input again, which is what
    fixes the vanishing input sensitivity of a deep plain MLP.
    """
    keys = jax.random.split(key, depth + 3)
    encoder_u = init_dense(keys[0], n_in, width, **rwf)
    encoder_v = init_dense(keys[1], n_in, width, **rwf)
    hidden = [init_dense(keys[2], n_in, width, **rwf)]
    hidden += [init_dense(keys[3 + i], width, width, **rwf) for i in range(depth - 1)]
    output = init_dense(keys[depth + 2], width, n_out, **rwf)
    return {"encoder_u": encoder_u, "encoder_v": encoder_v, "hidden": hidden, "output": output}


def apply_modified_mlp(params, x):
    encoded_u = jnp.tanh(apply_dense(params["encoder_u"], x))
    encoded_v = jnp.tanh(apply_dense(params["encoder_v"], x))
    h = x
    for layer in params["hidden"]:
        z = jnp.tanh(apply_dense(layer, h))
        h = (1.0 - z) * encoded_u + z * encoded_v
    return apply_dense(params["output"], h)
'''

# ---------------------------------------------------------------------------
# src/physics/__init__.py
# ---------------------------------------------------------------------------
FILES['src/physics/__init__.py'] = r'''"""Physics layer: scaling, analytic solutions, residual operators.

This package must not import from `src.models` or `src.training` (see
CLAUDE.md, "Module layout"). Everything here is a pure function of
non-dimensional inputs, with no module-level state.
"""
'''

# ---------------------------------------------------------------------------
# src/physics/analytic.py
# ---------------------------------------------------------------------------
FILES['src/physics/analytic.py'] = r'''"""Analytic solutions for plane Poiseuille flow — the oracle for Phase 1.

Two coordinate worlds are kept strictly apart:

* `*_si` functions take SI arguments and are used only by the scaling tests.
* everything else is non-dimensional (lengths in units of the half-height h,
  velocities in units of U_max, pressure per `Scales.pressure_scale`).

Sign convention: the flow is in +x, driven by a favourable pressure gradient
G = -dp/dx > 0.
"""

from __future__ import annotations

import numpy as np

from .scaling import PressureScaling

#: Non-dimensional flux of the reference profile u* = 1 - y*^2 over y* in
#: [-1, 1]:  int_{-1}^{1} (1 - y^2) dy = 4/3.
FLUX_ND = 4.0 / 3.0

#: Non-dimensional full channel height (walls at y* = +-1).
CHANNEL_HEIGHT_ND = 2.0


# --------------------------------------------------------------------------- #
# SI forms
# --------------------------------------------------------------------------- #


def u_si(y, pressure_gradient, h, mu):
    """u(y) = (G / 2 mu) (h^2 - y^2) for G = -dp/dx > 0."""
    y = np.asarray(y, dtype=np.float64)
    return (pressure_gradient / (2.0 * mu)) * (h**2 - y**2)


def u_max_si(pressure_gradient, h, mu):
    """U_max = G h^2 / (2 mu)."""
    return pressure_gradient * h**2 / (2.0 * mu)


def flow_rate_si(pressure_gradient, h, mu):
    """Q = 2 G h^3 / (3 mu) per unit depth.

    NOTE THE EXPONENT. In 2D the flux scales as h^3, not as r^4 (Hagen-
    Poiseuille, circular pipe). Phase 3 geometry optimisation depends on this
    exponent and the two must never be mixed. See SPEC section 1.
    """
    return 2.0 * pressure_gradient * h**3 / (3.0 * mu)


def pressure_gradient_si(u_max, h, mu):
    """Inverse of `u_max_si`: G = 2 mu U_max / h^2."""
    return 2.0 * mu * u_max / h**2


# --------------------------------------------------------------------------- #
# Non-dimensional forms
# --------------------------------------------------------------------------- #


def u_nd(y):
    """u*(y*) = 1 - y*^2."""
    y = np.asarray(y, dtype=np.float64)
    return 1.0 - y**2


def v_nd(y):
    """v* = 0 everywhere. The PINN has to discover this, not be told it."""
    y = np.asarray(y, dtype=np.float64)
    return np.zeros_like(y)


def developed_profile_nd(y, flux):
    """Fully developed parabolic profile carrying a prescribed flux.

    Mass conservation fixes the amplitude:  int_{-1}^{1} u* dy* = flux, and for
    u* = A (1 - y*^2) that integral is 4A/3, so A = 3*flux/4.

    For Case A (flux = 4/3) this is exactly `u_nd`. For Case B, whose uniform
    inlet u* = 1 carries flux 2, the developed centreline velocity is 3/2 --
    NOT 1. See docs/spec-deviations.md.
    """
    y = np.asarray(y, dtype=np.float64)
    return 0.75 * flux * (1.0 - y**2)


def pressure_gradient_nd(reynolds: float, pressure_scaling: PressureScaling):
    """Constant dp*/dx* of the fully developed non-dimensional solution.

    Substituting u* = 1 - y*^2, v* = 0 into the momentum equation:

    viscous scaling   Re (u.grad)u = -grad p* + lap u*  ->  dp*/dx* = -2
    inertial scaling  (u.grad)u = -grad p* + lap(u*)/Re ->  dp*/dx* = -2/Re

    The ratio Re between the two is exactly the ratio of the pressure scales,
    which is checked in the scaling tests.
    """
    if pressure_scaling == "viscous":
        return -2.0
    if pressure_scaling == "inertial":
        return -2.0 / reynolds
    raise ValueError(f"unknown pressure_scaling {pressure_scaling!r}")


def p_nd(x, domain_length: float, reynolds: float, pressure_scaling: PressureScaling):
    """p*(x*) with the gauge p*(x* = L*) = 0.

    Pressure is unique only up to a constant. Pinning it at the outlet is the
    gauge fix used throughout Phase 1; without it the pressure L2 error is
    meaningless (SPEC section 9).
    """
    x = np.asarray(x, dtype=np.float64)
    dpdx = pressure_gradient_nd(reynolds, pressure_scaling)
    return dpdx * (x - domain_length)


def stream_function_nd(y, flux=FLUX_ND):
    """psi*(y*) with u* = dpsi/dy, for the fully developed profile.

    psi = A (y - y^3/3) with A = 3*flux/4, gauged so psi(0) = 0. The wall values
    differ by exactly `flux`, which is what makes a stream-function ansatz that
    is constant on both walls unable to represent a channel with net throughflow
    (see src/models/hard_bc.py).
    """
    y = np.asarray(y, dtype=np.float64)
    return 0.75 * flux * (y - y**3 / 3.0)


# Entrance length now lives in `conventions.py` (SPEC v2 S1).
#
# The previous `entrance_length_correlation_nd` here took a bare
# `bulk_velocity_nd` argument, and Case B called it with 1.0 (the SPEC v1 inlet
# amplitude) where the parabolic mean 2/3 was required. That inflated the
# correlation by 25% at Re_h = 10 and is exactly the unsuffixed-Reynolds bug S1
# is written to eliminate. Use `conventions.entrance_length_over_h(re_h, corr)`,
# which converts internally and cannot be called with the wrong convention.
'''

# ---------------------------------------------------------------------------
# src/physics/conventions.py
# ---------------------------------------------------------------------------
FILES['src/physics/conventions.py'] = r'''"""Reynolds-number and length-scale conventions (SPEC v2, S1).

Two Reynolds numbers live in this project and they are NOT equal.

Solver convention -- half-height `h`, centreline velocity `U_max`:

    Re_h = rho * U_max * h / mu

Correlation convention -- hydraulic diameter `D_h`, mean velocity `U_mean`:

    Re_Dh = rho * U_mean * D_h / mu

For a wide plane channel, D_h = 4A/P = 4(2hw)/(2w) = 4h, and for the parabolic
profile U_mean = (2/3) U_max, so

    Re_Dh = (8/3) Re_h

and the reference LENGTHS differ by a factor of four (D_h = 4h). Chained
through a correlation that is linear in Re, that is roughly an order of
magnitude -- which is the signature that showed up as a 289% entrance-length
miss while every other quantity from the same solution was within a few
percent.

Rule enforced by naming throughout this module: every function that consumes a
Reynolds number says WHICH ONE in its name. A parameter called plain `re` is a
bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Re_Dh / Re_h for a wide plane channel. Derivation in the module docstring.
RE_DH_OVER_RE_H = 8.0 / 3.0

#: D_h / h for a wide plane channel.
D_H_OVER_H = 4.0

#: U_mean / U_max for the parabolic profile.
U_MEAN_OVER_U_MAX = 2.0 / 3.0


class MissingCitationError(ValueError):
    """Raised when a correlation is declared without a source citation."""


@dataclass(frozen=True)
class EntranceLengthCorrelation:
    """Blended Durst-type correlation, in HYDRAULIC-DIAMETER units.

        L_e / D_h = [ c0^n + (c1 * Re_Dh)^n ]^(1/n)

    The constants differ between pipe and plane-channel geometry and must be
    read from the source, never assumed. `citation` is mandatory: a wrong
    constant here is invisible in the output, whereas a missing citation is
    not.
    """

    c0: float
    c1: float
    n: float
    citation: str

    def __post_init__(self):
        if not isinstance(self.citation, str) or not self.citation.strip():
            raise MissingCitationError(
                "an entrance-length correlation must carry a non-empty "
                "'citation'. The constants differ between pipe and plane-"
                "channel geometry and a wrong one is invisible in the output."
            )
        for name in ("c0", "c1", "n"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"correlation.{name} must be finite and positive, got {value!r}"
                )

    def length_over_dh(self, reynolds_dh):
        """L_e / D_h."""
        return (self.c0**self.n + (self.c1 * reynolds_dh) ** self.n) ** (1.0 / self.n)


# --------------------------------------------------------------------------- #
# Reynolds numbers
# --------------------------------------------------------------------------- #


def reynolds_h(rho, u_max, h, mu):
    """Solver convention: Re_h = rho U_max h / mu."""
    return rho * u_max * h / mu


def reynolds_dh(rho, u_mean, d_h, mu):
    """Correlation convention: Re_Dh = rho U_mean D_h / mu."""
    return rho * u_mean * d_h / mu


def re_h_to_re_dh(re_h):
    """Re_Dh = (8/3) Re_h. Exact for a wide plane channel."""
    return RE_DH_OVER_RE_H * re_h


def re_dh_to_re_h(re_dh):
    """Inverse of `re_h_to_re_dh`."""
    return re_dh / RE_DH_OVER_RE_H


# --------------------------------------------------------------------------- #
# Entrance length
# --------------------------------------------------------------------------- #


def entrance_length_over_h(re_h, correlation: EntranceLengthCorrelation):
    """L_e / h, converting conventions internally.

    Takes the SOLVER Reynolds number, converts to the correlation's convention,
    evaluates, and converts the length back:  L_e/h = 4 * (L_e/D_h).

    Returning in units of `h` means call sites never touch D_h and cannot get
    the factor of four wrong.
    """
    return D_H_OVER_H * correlation.length_over_dh(re_h_to_re_dh(re_h))
'''

# ---------------------------------------------------------------------------
# src/physics/derivatives.py
# ---------------------------------------------------------------------------
FILES['src/physics/derivatives.py'] = r'''"""Autodiff assembly of the derivative bundle the residual operator consumes.

Kept separate from `residuals.py` so that the residual algebra can be unit
tested against hand-derived expressions with no autodiff in the loop, and the
autodiff itself can be checked against finite differences independently
(SPEC test 5).

A "field function" here is any callable `f(xy) -> array of shape (3,)` giving
(u, v, p) at the single point `xy = [x, y]`. It may internally be a network, a
hard-BC-wrapped network, an analytic expression, or a stream-function
construction that differentiates a scalar potential -- this module does not
care, and takes second derivatives of whatever it is handed.

Two implementations of the same mathematics live here:

`field_derivatives`
    One point, nested `jacfwd`. Short, obviously correct, used as the reference.

`batched_field_derivatives`
    A whole (N, 2) array at once. Instead of vmapping the single-point version,
    it differentiates the ALREADY-BATCHED field function along a constant
    tangent direction, twice. Each output row depends only on its own input
    row, so a batch-wide directional derivative gives every point's derivative
    in that direction in one pass -- and every pass stays a full-size matrix
    product rather than a vmapped matrix-vector product.

    Measured on the baseline config (width 128, depth 4, 4096 points): the
    batched form costs 195 ms per loss-and-gradient against 460 ms for the
    vmapped one, for bitwise-comparable values. The reverse-mode pass over the
    vmapped nested `jacfwd` is the expensive part -- it stores a tangent copy
    of every activation for every point -- and the batched form cuts the number
    of those copies. `test_batched_derivatives_match_single_point` pins the two
    together so the fast path cannot drift from the reference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .residuals import Derivatives

#: Unit tangent directions in (x, y).
_EX = jnp.array([1.0, 0.0])
_EY = jnp.array([0.0, 1.0])


def field_derivatives(field_fn, xy) -> Derivatives:
    """Value, first and second derivatives of (u, v, p) at one point.

    Forward-mode throughout: the input is 2-dimensional and the output is
    3-dimensional, so forward mode is the cheaper direction for both the
    Jacobian and the Hessian.
    """

    def value_and_jacobian(point):
        value = field_fn(point)
        jacobian = jax.jacfwd(field_fn)(point)
        # `has_aux` lets the outer jacfwd differentiate the Jacobian while
        # handing back the value and Jacobian it already computed, instead of
        # recomputing them in a separate pass.
        return jacobian, (value, jacobian)

    hessian, (value, jacobian) = jax.jacfwd(value_and_jacobian, has_aux=True)(xy)

    return _assemble(value, jacobian[:, 0], jacobian[:, 1], hessian[:, 0, 0], hessian[:, 1, 1])


def _assemble(value, d_dx, d_dy, d2_dx2, d2_dy2) -> Derivatives:
    """Pack directional derivatives into the bundle. Index 0=u, 1=v, 2=p."""
    return Derivatives(
        u=value[..., 0], v=value[..., 1], p=value[..., 2],
        u_x=d_dx[..., 0], v_x=d_dx[..., 1], p_x=d_dx[..., 2],
        u_y=d_dy[..., 0], v_y=d_dy[..., 1], p_y=d_dy[..., 2],
        u_xx=d2_dx2[..., 0], v_xx=d2_dx2[..., 1],
        u_yy=d2_dy2[..., 0], v_yy=d2_dy2[..., 1],
    )


def _directional(batched_fn, points, direction):
    """(value, first, second) derivatives along a constant `direction`.

    The inner `jvp` returns the value alongside the first derivative, and the
    outer one differentiates that pair, so a single call yields all three
    without a separate primal evaluation.
    """
    tangents = jnp.broadcast_to(direction, points.shape)

    def value_and_first(pts):
        return jax.jvp(batched_fn, (pts,), (tangents,))

    (value, first), (_, second) = jax.jvp(value_and_first, (points,), (tangents,))
    return value, first, second


def batched_directional_derivatives(vector_fn, points):
    """(value, d/dx, d/dy, d2/dx2, d2/dy2) for any vector-valued `f(xy)`.

    Shared by the velocity-pressure bundle and the psi-omega bundle: both need
    exactly the value, both first derivatives and both unmixed second
    derivatives of a small vector of outputs, and neither needs the cross
    derivative.
    """
    batched_fn = jax.vmap(vector_fn)
    value, d_dx, d2_dx2 = _directional(batched_fn, points, _EX)
    _, d_dy, d2_dy2 = _directional(batched_fn, points, _EY)
    return value, d_dx, d_dy, d2_dx2, d2_dy2


def batched_field_derivatives(field_fn, points) -> Derivatives:
    """Derivative bundle for an (N, 2) array of points."""
    return _assemble(*batched_directional_derivatives(field_fn, points))


def batched_stream_derivatives(stream_fn, points):
    """Derivative bundle for the psi-omega system (SPEC v2 S4).

    `stream_fn(xy) -> [psi, omega]`. Only SECOND derivatives are taken, of both
    outputs -- the whole point of solving the coupled second-order system
    rather than substituting omega = -lap psi is to avoid fourth-order autodiff
    through tanh.
    """
    from .streamfunction import StreamDerivatives

    value, d_dx, d_dy, d2_dx2, d2_dy2 = batched_directional_derivatives(
        stream_fn, points
    )
    return StreamDerivatives(
        psi_x=d_dx[..., 0], psi_y=d_dy[..., 0],
        psi_xx=d2_dx2[..., 0], psi_yy=d2_dy2[..., 0],
        omega=value[..., 1],
        omega_x=d_dx[..., 1], omega_y=d_dy[..., 1],
        omega_xx=d2_dx2[..., 1], omega_yy=d2_dy2[..., 1],
    )


def field_and_jacobian(field_fn, xy):
    """(value, jacobian) at one point, without the Hessian.

    Boundary terms (outlet du/dx, dv/dx) need only first derivatives; computing
    the full bundle there would triple the cost for nothing.
    """
    return field_fn(xy), jax.jacfwd(field_fn)(xy)


def batched_field_and_jacobian(field_fn, points):
    """(values (N, 3), jacobians (N, 3, 2)) for an (N, 2) array of points."""
    batched_fn = jax.vmap(field_fn)
    value, d_dx = jax.jvp(batched_fn, (points,), (jnp.broadcast_to(_EX, points.shape),))
    _, d_dy = jax.jvp(batched_fn, (points,), (jnp.broadcast_to(_EY, points.shape),))
    return value, jnp.stack([d_dx, d_dy], axis=-1)
'''

# ---------------------------------------------------------------------------
# src/physics/optimum.py
# ---------------------------------------------------------------------------
FILES['src/physics/optimum.py'] = r'''"""Analytic optimum for the Phase 3 objective (SPEC v2, S8).

Objective: minimise total hydraulic resistance at fixed total channel area.
This needs no metabolic-cost coefficient, which is why it is preferable to a
literal Murray formulation here.

Plane channel, per unit depth, half-height `h`, length `L`:

    Q = 2 G h^3 / (3 mu)      =>      R = dp/Q = 3 mu L / (2 h^3)

NOTE THE EXPONENT. R ~ h^-3 and area ~ h for a 2D slot; the circular pipe has
R ~ r^-4 and volume ~ r^2. Mixing them silently is the Phase 3 hazard the SPEC
calls out, and it is what makes the answer below an exponent of 2 rather than
Murray's 3.

Two independent derivation routes are implemented and must agree; that
agreement is asserted as a test rather than assumed.
"""

from __future__ import annotations

import numpy as np

#: Exponent pairs (n, m) for `R ~ h^-n` and `area or volume ~ h^m`.
PLANE_CHANNEL = (3, 1)
CIRCULAR_PIPE = (4, 2)


def resistance_plane_channel(mu, length, half_height):
    """R = 3 mu L / (2 h^3), per unit depth."""
    return 3.0 * mu * length / (2.0 * half_height**3)


def flow_rate_plane_channel(pressure_gradient, mu, half_height):
    """Q = 2 G h^3 / (3 mu), per unit depth. Inverse of `resistance_plane_channel`."""
    return 2.0 * pressure_gradient * half_height**3 / (3.0 * mu)


def optimum_ratio_by_lagrange(mu=1.0, length_parent=1.0, length_daughter=1.0):
    """h_0/h_1 from stationarity of R_tot + lambda (A - A_0).

    Parent `(h_0, L_0)` feeding two identical daughters `(h_1, L_1)` in
    parallel:

        R_tot = (3 mu / 2) [ L_0/h_0^3 + L_1/(2 h_1^3) ]
        A     = 2 h_0 L_0 + 4 h_1 L_1

    Stationarity gives

        dL/dh_0 = -9 mu L_0 / (2 h_0^4) + 2 lambda L_0 = 0  =>  h_0^4 = 9 mu / (4 lambda)
        dL/dh_1 = -9 mu L_1 / (4 h_1^4) + 4 lambda L_1 = 0  =>  h_1^4 = 9 mu / (16 lambda)

    so h_0^4 / h_1^4 = 4 and h_0/h_1 = 4^(1/4) = sqrt(2), independent of mu,
    lambda and both lengths. The lengths are accepted as arguments precisely so
    a test can confirm that independence.
    """
    # Both stationarity conditions carry a factor L_i that cancels, so the
    # ratio is computed from the exponents alone. Written out rather than
    # returned as a literal, so the derivation is executable.
    lam = 1.0  # any positive value; it cancels
    h0_fourth = 9.0 * mu / (4.0 * lam)
    h1_fourth = 9.0 * mu / (16.0 * lam)
    return float((h0_fourth / h1_fourth) ** 0.25)


def flow_exponent(n, m):
    """Q ~ h^((n+m)/2) from minimising A Q^2 h^-n + B h^m.

    The cost is dissipation plus a maintenance term. Stationarity in `h`:

        -n A Q^2 h^(-n-1) + m B h^(m-1) = 0   =>   Q^2 ~ h^(n+m)

    Plane channel (n, m) = (3, 1) gives Q ~ h^2; circular pipe (4, 2) gives
    Q ~ r^3, which is Murray's law and the check that this route is right.
    """
    return (n + m) / 2.0


def optimum_ratio_by_flow_exponent(n=3, m=1):
    """h_0/h_1 for a symmetric bifurcation, from the flow exponent.

    With `Q ~ h^k` and the parent flux split equally between two daughters,
    `h_0^k = 2 h_1^k`, so `h_0/h_1 = 2^(1/k)`. For the plane channel k = 2 and
    the ratio is sqrt(2).
    """
    k = flow_exponent(n, m)
    return float(2.0 ** (1.0 / k))


def area_law_exponent(n=3, m=1):
    """The exponent `k` in the conserved quantity `h_0^k = h_1^k + h_2^k`.

    Plane channel: k = 2, i.e. h_0^2 = h_1^2 + h_2^2 -- the 2D analogue of
    Murray's law. The cube law is specific to circular cross-sections.
    """
    return flow_exponent(n, m)


#: The Phase 3 oracle in the fully developed limit. The DEVIATION of the PINN
#: optimum from this value is the Phase 3 result; the law itself is not.
OPTIMUM_HEIGHT_RATIO = np.sqrt(2.0)
'''

# ---------------------------------------------------------------------------
# src/physics/residuals.py
# ---------------------------------------------------------------------------
FILES['src/physics/residuals.py'] = r'''"""Unweighted residuals of the 2D steady incompressible Navier-Stokes system.

One small function per governing-equation term, so each term can be unit-tested
on its own (CLAUDE.md, "Code style"). Every function here is pure, takes
non-dimensional inputs, and returns an UNWEIGHTED residual. All loss weighting
lives in src/training/weights.py and nowhere else.

Non-dimensional forms
---------------------
inertial pressure scaling (p* = p / (rho U^2)):

    R_x = u u_x + v u_y + p_x - (1/Re) (u_xx + u_yy)
    R_y = u v_x + v v_y + p_y - (1/Re) (v_xx + v_yy)

viscous pressure scaling (p* = p h / (mu U)):

    R_x = Re (u u_x + v u_y) + p_x - (u_xx + u_yy)
    R_y = Re (u v_x + v v_y) + p_y - (v_xx + v_yy)

continuity (both):

    R_c = u_x + v_y

The two momentum forms are the same equation multiplied by Re (because
p*_viscous = Re p*_inertial), so they have identical solutions and different
conditioning. That is exactly why the SPEC makes the choice a config option.
"""

from __future__ import annotations

from typing import NamedTuple

from .scaling import PressureScaling


class Derivatives(NamedTuple):
    """Field values and the derivatives the residual operator needs.

    A NamedTuple so it is a JAX pytree and can be vmapped/jitted directly.
    All entries are non-dimensional.
    """

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


# --------------------------------------------------------------------------- #
# Individual terms
# --------------------------------------------------------------------------- #


def convective_x(u, v, u_x, u_y):
    """(u . grad) u, x-component."""
    return u * u_x + v * u_y


def convective_y(u, v, v_x, v_y):
    """(u . grad) u, y-component."""
    return u * v_x + v * v_y


def laplacian(f_xx, f_yy):
    """grad^2 f."""
    return f_xx + f_yy


def divergence(u_x, v_y):
    """div u = u_x + v_y. This is the continuity residual, unweighted."""
    return u_x + v_y


# --------------------------------------------------------------------------- #
# Assembled momentum residuals
# --------------------------------------------------------------------------- #


def _scaling_coefficients(reynolds: float, pressure_scaling: PressureScaling):
    """(convective coefficient, viscous coefficient) for the chosen scaling."""
    if pressure_scaling == "inertial":
        return 1.0, 1.0 / reynolds
    if pressure_scaling == "viscous":
        return reynolds, 1.0
    raise ValueError(
        f"unknown pressure_scaling {pressure_scaling!r}; "
        "expected 'inertial' or 'viscous'"
    )


def momentum_x(d: Derivatives, reynolds: float, pressure_scaling: PressureScaling):
    """x-momentum residual, unweighted."""
    a_conv, a_visc = _scaling_coefficients(reynolds, pressure_scaling)
    return (
        a_conv * convective_x(d.u, d.v, d.u_x, d.u_y)
        + d.p_x
        - a_visc * laplacian(d.u_xx, d.u_yy)
    )


def momentum_y(d: Derivatives, reynolds: float, pressure_scaling: PressureScaling):
    """y-momentum residual, unweighted."""
    a_conv, a_visc = _scaling_coefficients(reynolds, pressure_scaling)
    return (
        a_conv * convective_y(d.u, d.v, d.v_x, d.v_y)
        + d.p_y
        - a_visc * laplacian(d.v_xx, d.v_yy)
    )


def continuity(d: Derivatives):
    """Continuity residual, unweighted."""
    return divergence(d.u_x, d.v_y)


def navier_stokes(d: Derivatives, reynolds: float, pressure_scaling: PressureScaling):
    """All three residuals as a tuple (R_x, R_y, R_c), unweighted."""
    return (
        momentum_x(d, reynolds, pressure_scaling),
        momentum_y(d, reynolds, pressure_scaling),
        continuity(d),
    )
'''

# ---------------------------------------------------------------------------
# src/physics/scaling.py
# ---------------------------------------------------------------------------
FILES['src/physics/scaling.py'] = r'''"""Explicit, invertible SI <-> non-dimensional transform.

Every physics quantity in this project lives in one of two worlds and the
transform between them is written down here once. Nothing else in the codebase
is allowed to hold an SI-valued viscosity, density or channel height.

Reference quantities
--------------------
length    h        channel half-height
velocity  U_max    centreline velocity of the fully developed profile
pressure  depends on the chosen scaling:
            viscous   p_ref = mu * U_max / h        (use for Re <~ 1)
            inertial  p_ref = rho * U_max**2        (use for Re > 1)

Reynolds number
---------------
    Re = rho * U_max * h / mu

The two pressure scalings are related by exactly a factor Re:

    p*_viscous = p*_inertial * Re

which is a useful consistency check and is tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

PressureScaling = Literal["viscous", "inertial"]

_VALID_PRESSURE_SCALINGS = ("viscous", "inertial")

#: Relative tolerance for a "machine precision" SI -> non-dim -> SI round trip.
#: A round trip is one division followed by one multiplication, each correctly
#: rounded, so the relative error is bounded by ~2 eps. 4 eps is that bound with
#: one bit of headroom. This is a derived bound, not a fitted tolerance: it must
#: never be widened to make a test pass.
ROUNDTRIP_RTOL = 4.0 * float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class Scales:
    """Reference quantities defining the non-dimensionalisation.

    Parameters are SI. This is the only object in the codebase that is allowed
    to carry SI values; it converts them and hands non-dimensional numbers to
    everything else.

    Attributes
    ----------
    h : channel half-height [m]
    u_max : centreline velocity of the fully developed profile [m/s]
    rho : density [kg/m^3]
    mu : dynamic viscosity [Pa s]
    pressure_scaling : "viscous" or "inertial"
    """

    h: float
    u_max: float
    rho: float
    mu: float
    pressure_scaling: PressureScaling

    def __post_init__(self) -> None:
        for name in ("h", "u_max", "rho", "mu"):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"Scales.{name} must be finite, got {value!r}")
            if value <= 0.0:
                raise ValueError(f"Scales.{name} must be positive, got {value!r}")
        if self.pressure_scaling not in _VALID_PRESSURE_SCALINGS:
            raise ValueError(
                f"pressure_scaling must be one of {_VALID_PRESSURE_SCALINGS}, "
                f"got {self.pressure_scaling!r}"
            )

    # -- reference quantities -------------------------------------------------

    @property
    def reynolds(self) -> float:
        """Re = rho * U_max * h / mu."""
        return self.rho * self.u_max * self.h / self.mu

    @property
    def length_scale(self) -> float:
        return self.h

    @property
    def velocity_scale(self) -> float:
        return self.u_max

    @property
    def pressure_scale(self) -> float:
        """Pressure reference p_ref such that p* = p / p_ref."""
        if self.pressure_scaling == "viscous":
            return self.mu * self.u_max / self.h
        return self.rho * self.u_max**2

    # -- SI -> non-dimensional ------------------------------------------------

    def length_to_nondim(self, x):
        return np.asarray(x, dtype=np.float64) / self.length_scale

    def velocity_to_nondim(self, u):
        return np.asarray(u, dtype=np.float64) / self.velocity_scale

    def pressure_to_nondim(self, p):
        return np.asarray(p, dtype=np.float64) / self.pressure_scale

    def pressure_gradient_to_nondim(self, dpdx):
        """dp*/dx* = (dp/dx) * h / p_ref."""
        return (
            np.asarray(dpdx, dtype=np.float64)
            * self.length_scale
            / self.pressure_scale
        )

    # -- non-dimensional -> SI ------------------------------------------------

    def length_to_si(self, x_nd):
        return np.asarray(x_nd, dtype=np.float64) * self.length_scale

    def velocity_to_si(self, u_nd):
        return np.asarray(u_nd, dtype=np.float64) * self.velocity_scale

    def pressure_to_si(self, p_nd):
        return np.asarray(p_nd, dtype=np.float64) * self.pressure_scale

    def pressure_gradient_to_si(self, dpdx_nd):
        return (
            np.asarray(dpdx_nd, dtype=np.float64)
            * self.pressure_scale
            / self.length_scale
        )


def scales_from_reynolds(
    reynolds: float,
    pressure_scaling: PressureScaling,
    *,
    h: float = 1.0,
    u_max: float = 1.0,
    rho: float = 1.0,
) -> Scales:
    """Build `Scales` with a prescribed Reynolds number.

    mu is solved for rather than defaulted, so the returned object satisfies
    `scales.reynolds == reynolds` up to rounding. Used by configs that specify
    Re directly instead of four SI quantities.
    """
    if not np.isfinite(reynolds) or reynolds <= 0.0:
        raise ValueError(f"reynolds must be finite and positive, got {reynolds!r}")
    mu = rho * u_max * h / reynolds
    return Scales(
        h=h, u_max=u_max, rho=rho, mu=mu, pressure_scaling=pressure_scaling
    )


def recommended_pressure_scaling(reynolds: float) -> PressureScaling:
    """The SPEC's recommendation: viscous for Re <= 1, inertial above.

    Advisory only. The active scaling always comes from the config; this
    function exists so a run can *report* that it deviates from the
    recommendation, never so it can silently override the config.
    """
    return "viscous" if reynolds <= 1.0 else "inertial"
'''

# ---------------------------------------------------------------------------
# src/physics/streamfunction.py
# ---------------------------------------------------------------------------
FILES['src/physics/streamfunction.py'] = r'''"""Coupled psi-omega system and post-hoc pressure recovery (SPEC v2, S4).

    u = dpsi/dy,   v = -dpsi/dx      (continuity satisfied identically)
    omega = -lap psi                 (vorticity)

Taking the curl of the momentum equation eliminates pressure. The system is
solved as TWO COUPLED SECOND-ORDER equations in two network outputs
`(psi, omega)`:

    r_1 = omega + lap psi                                             = 0
    r_2 = psi_y omega_x - psi_x omega_y - lap(omega)/Re               = 0

rather than by substituting `omega = -lap psi` into r_2, which would give a
FOURTH-ORDER equation in psi. Fourth-order autodiff through tanh would then set
the precision floor -- and Phase 1 already measured second derivatives through
tanh costing an order of magnitude of achievable accuracy.

`omega` carries no hard constraint and is coupled only through r_1. That is
deliberate: it partially restores the falsifiability that the pure-psi form
destroys, because r_1 is a real residual the network can violate.
"""

from __future__ import annotations

from typing import NamedTuple


class StreamDerivatives(NamedTuple):
    """Derivative bundle for the psi-omega system. All non-dimensional."""

    psi_x: object
    psi_y: object
    psi_xx: object
    psi_yy: object
    omega: object
    omega_x: object
    omega_y: object
    omega_xx: object
    omega_yy: object


def laplacian(f_xx, f_yy):
    return f_xx + f_yy


def vorticity_residual(d: StreamDerivatives):
    """r_1 = omega + lap psi. Unweighted."""
    return d.omega + laplacian(d.psi_xx, d.psi_yy)


def vorticity_transport_residual(d: StreamDerivatives, reynolds: float):
    """r_2 = psi_y omega_x - psi_x omega_y - lap(omega)/Re. Unweighted.

    The convective part is (u . grad) omega with u = psi_y, v = -psi_x:

        u omega_x + v omega_y = psi_y omega_x - psi_x omega_y
    """
    return (
        d.psi_y * d.omega_x
        - d.psi_x * d.omega_y
        - laplacian(d.omega_xx, d.omega_yy) / reynolds
    )


def residuals(d: StreamDerivatives, reynolds: float):
    """(r_1, r_2), unweighted."""
    return vorticity_residual(d), vorticity_transport_residual(d, reynolds)


def velocity_from_stream(psi_x, psi_y):
    """(u, v) = (psi_y, -psi_x)."""
    return psi_y, -psi_x


# --------------------------------------------------------------------------- #
# Post-hoc pressure recovery (SPEC v2, S4)
# --------------------------------------------------------------------------- #

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from .derivatives import batched_directional_derivatives  # noqa: E402
from .residuals import _scaling_coefficients  # noqa: E402


def pressure_gradient(velocity_fn, points, reynolds, pressure_scaling):
    """(p_x, p_y) from the momentum equation, at an (N, 2) array of points.

    Rearranging the momentum residual to zero:

        p_x = a_visc (u_xx + u_yy) - a_conv (u u_x + v u_y)
        p_y = a_visc (v_xx + v_yy) - a_conv (u v_x + v v_y)

    with (a_conv, a_visc) the scaling coefficients, so this works unchanged
    under either pressure scaling. `velocity_fn(xy) -> [u, v]`.
    """
    a_conv, a_visc = _scaling_coefficients(reynolds, pressure_scaling)
    value, d_dx, d_dy, d2_dx2, d2_dy2 = batched_directional_derivatives(
        velocity_fn, points
    )
    u, v = value[..., 0], value[..., 1]
    u_x, v_x = d_dx[..., 0], d_dx[..., 1]
    u_y, v_y = d_dy[..., 0], d_dy[..., 1]
    lap_u = d2_dx2[..., 0] + d2_dy2[..., 0]
    lap_v = d2_dx2[..., 1] + d2_dy2[..., 1]

    p_x = a_visc * lap_u - a_conv * (u * u_x + v * u_y)
    p_y = a_visc * lap_v - a_conv * (u * v_x + v * v_y)
    return p_x, p_y


def _simpson(values, spacing):
    """Composite Simpson on an odd number of equally spaced samples."""
    n = len(values) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's rule needs an even number of intervals")
    weights = np.ones(n + 1)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return float(spacing / 3.0 * np.dot(weights, values))


def _segment(velocity_fn, start, end, reynolds, pressure_scaling, nodes):
    """Integral of grad p . dl along the straight segment start -> end."""
    if nodes % 2 == 0:
        nodes += 1
    t = np.linspace(0.0, 1.0, nodes)
    path = np.outer(1.0 - t, np.asarray(start)) + np.outer(t, np.asarray(end))
    p_x, p_y = pressure_gradient(
        velocity_fn, jnp.asarray(path), reynolds, pressure_scaling
    )
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    integrand = np.asarray(p_x) * delta[0] + np.asarray(p_y) * delta[1]
    return _simpson(integrand, 1.0 / (nodes - 1))


def recover_pressure(
    velocity_fn, target, gauge_point, reynolds, pressure_scaling, *, path="x_first",
    nodes=201,
):
    """p at `target` (a single (2,) point), gauged to zero at `gauge_point`.

    Integrates grad p along an L-shaped path. Two orderings are available and
    they must agree: path independence is a real consistency check on the
    recovered field, since grad p is only a gradient if the flow actually
    satisfies momentum (SPEC v2 S4, T-S4.2).

    "x_first": gauge -> (target_x, gauge_y) -> target
    "y_first": gauge -> (gauge_x, target_y) -> target
    """
    gauge = np.asarray(gauge_point, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if path == "x_first":
        corner = np.array([target[0], gauge[1]])
    elif path == "y_first":
        corner = np.array([gauge[0], target[1]])
    else:
        raise ValueError(f"unknown path {path!r}")

    total = 0.0
    for start, end in ((gauge, corner), (corner, target)):
        if not np.allclose(start, end):
            total += _segment(
                velocity_fn, start, end, reynolds, pressure_scaling, nodes
            )
    return total
'''

# ---------------------------------------------------------------------------
# src/predict.py
# ---------------------------------------------------------------------------
FILES['src/predict.py'] = r'''"""Use a trained model.

A finished run in `runs/<id>/` is a reusable solver, not just a folder of
numbers. This module loads one back and evaluates it anywhere in the channel.

    from src.predict import load_run
    solution = load_run("runs/caseA_re10")
    u, v = solution.velocity(3.0, 0.5)
    y, u = solution.profile(x=3.0)

or from a shell:

    python -m src.predict --run runs/caseA_re10 --at 3.0,0.5
    python -m src.predict --run runs/caseA_re10 --profile-at 3.0 --out profile.csv
    python -m src.predict --run runs/caseA_re10 --grid 201 101 --out field.csv

Two things this deliberately refuses to do quietly:

* It will not evaluate outside the trained domain. A PINN is fitted, not
  solved, so outside `x* in [0, L*]` and `y* in [-1, 1]` it is extrapolating
  and the residual it was trained on says nothing there. Pass
  `allow_extrapolation=True` if you mean it.
* It will not report a pressure the model does not have. The psi-omega system
  has no pressure in it, so `pressure()` recovers one post hoc by path
  integration and says so, rather than returning the placeholder zero that
  lives in the field's third slot.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .config import Config, load_config
from .geometry.domain import WALL
from .models.build import build_model
from .physics import streamfunction
from .evaluation.metrics import cross_channel_flux


@dataclass
class TrainedModel:
    """A trained solution, ready to evaluate."""

    config: Config
    params: object
    model: object
    run_directory: Path

    # -- introspection --------------------------------------------------------

    @property
    def domain_length(self) -> float:
        return self.config.domain.length

    def summary(self) -> str:
        c = self.config
        return (
            f"{c.name}: case {c.case}, Re_h = {c.physics.reynolds:g}, "
            f"{c.physics.formulation}, L* = {c.domain.length:.3f}, "
            f"pressure in system: {self.model.has_pressure}"
        )

    # -- point evaluation -----------------------------------------------------

    def _check_domain(self, x, y, allow_extrapolation):
        if allow_extrapolation:
            return
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        outside = (
            (x < 0.0) | (x > self.domain_length) | (y < -WALL) | (y > WALL)
        )
        if np.any(outside):
            first = int(np.argmax(outside))
            raise ValueError(
                f"point (x*={x[first]:g}, y*={y[first]:g}) lies outside the "
                f"trained domain x* in [0, {self.domain_length:g}], "
                f"y* in [-1, 1]. A PINN is fitted, not solved: outside the "
                "region where its residual was enforced it is extrapolating "
                "and its output is not a solution of anything. Pass "
                "allow_extrapolation=True if that is what you want."
            )

    def velocity(self, x, y, *, allow_extrapolation=False):
        """(u*, v*) at the given non-dimensional coordinates. Broadcasts."""
        self._check_domain(x, y, allow_extrapolation)
        points = self._as_points(x, y)
        out = jax.vmap(lambda p: self.model.velocity(self.params, p))(points)
        u = np.asarray(out[:, 0]).reshape(np.shape(x) if np.ndim(x) else ())
        v = np.asarray(out[:, 1]).reshape(np.shape(x) if np.ndim(x) else ())
        return (float(u), float(v)) if u.ndim == 0 else (u, v)

    def pressure(self, x, y, *, allow_extrapolation=False):
        """p* at the given coordinates, gauged to zero at the outlet centreline.

        For the velocity-pressure formulation this is a network output. For the
        psi-omega system pressure is absent from the solved system and is
        recovered by integrating grad p along a path from the gauge point,
        which is slower and carries its own consistency error -- the run's
        `pressure_path_max_disagreement` metric measures it.
        """
        self._check_domain(x, y, allow_extrapolation)
        if self.model.has_pressure:
            points = self._as_points(x, y)
            out = jax.vmap(lambda p: self.model.field(self.params, p))(points)
            p = np.asarray(out[:, 2]).reshape(np.shape(x) if np.ndim(x) else ())
            return float(p) if p.ndim == 0 else p

        velocity_fn = lambda xy: self.model.velocity(self.params, xy)  # noqa: E731
        gauge = (self.domain_length, 0.0)
        xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
        ys = np.atleast_1d(np.asarray(y, dtype=np.float64))
        xs, ys = np.broadcast_arrays(xs, ys)
        values = np.array([
            streamfunction.recover_pressure(
                velocity_fn, (float(a), float(b)), gauge,
                self.config.physics.reynolds, self.config.physics.pressure_scaling,
            )
            for a, b in zip(xs.ravel(), ys.ravel())
        ]).reshape(xs.shape)
        return float(values) if np.ndim(x) == 0 and np.ndim(y) == 0 else values

    def fields(self, x, y, *, allow_extrapolation=False) -> dict:
        u, v = self.velocity(x, y, allow_extrapolation=allow_extrapolation)
        return {
            "u": u, "v": v,
            "p": self.pressure(x, y, allow_extrapolation=allow_extrapolation),
        }

    # -- derived quantities ---------------------------------------------------

    def profile(self, x, n=201):
        """Cross-channel velocity profile at one streamwise station."""
        y = np.linspace(-WALL, WALL, n)
        u, v = self.velocity(np.full_like(y, float(x)), y)
        return y, u, v

    def flux(self, x):
        """Volumetric flux per unit depth through the station(s) x*."""
        stations = np.atleast_1d(np.asarray(x, dtype=np.float64))
        values = cross_channel_flux(self.model.field, self.params, stations)
        return float(values[0]) if np.ndim(x) == 0 else np.asarray(values)

    def grid(self, nx=201, ny=101):
        """(x, y, u, v) on a regular grid covering the whole channel."""
        x = np.linspace(0.0, self.domain_length, nx)
        y = np.linspace(-WALL, WALL, ny)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        u, v = self.velocity(xx.ravel(), yy.ravel())
        return x, y, u.reshape(nx, ny), v.reshape(nx, ny)

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _as_points(x, y):
        xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
        ys = np.atleast_1d(np.asarray(y, dtype=np.float64))
        xs, ys = np.broadcast_arrays(xs, ys)
        return jnp.asarray(np.stack([xs.ravel(), ys.ravel()], axis=-1))


def load_run(run_directory) -> TrainedModel:
    """Load a finished run back into an evaluable model.

    The parameter pytree's STRUCTURE is rebuilt from the run's own config
    rather than stored, so a run directory stays readable and version
    independent. That makes it essential that the config travels with the
    parameters -- which is why `run.py` copies it into the run directory.
    """
    directory = Path(run_directory)
    config_path = directory / "config.yaml"
    params_path = directory / "params.npz"
    for path in (config_path, params_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. A usable run directory needs both "
                "config.yaml and params.npz; has this run finished?"
            )

    config = load_config(config_path)
    template, model = build_model(config, jax.random.PRNGKey(config.seed))
    flat_template, unflatten = ravel_pytree(template)

    flat = np.load(params_path)["flat"]
    if flat.shape != flat_template.shape:
        raise ValueError(
            f"{params_path} holds {flat.shape[0]} parameters but the config in "
            f"{config_path} builds a model with {flat_template.shape[0]}. The "
            "config and the saved weights are from different models."
        )
    return TrainedModel(config, unflatten(jnp.asarray(flat)), model, directory)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained run.")
    parser.add_argument("--run", required=True, help="a finished runs/<id> directory")
    parser.add_argument("--at", help="a single point as 'x,y'")
    parser.add_argument("--profile-at", type=float, help="cross-channel profile at x*")
    parser.add_argument("--flux-at", type=float, help="flux through the station x*")
    parser.add_argument("--grid", nargs=2, type=int, metavar=("NX", "NY"),
                        help="evaluate on an NX by NY grid")
    parser.add_argument("--out", help="write CSV here instead of printing")
    parser.add_argument("--allow-extrapolation", action="store_true")
    args = parser.parse_args(argv)

    solution = load_run(args.run)
    print(solution.summary())

    if args.at:
        x, y = (float(v) for v in args.at.split(","))
        f = solution.fields(x, y, allow_extrapolation=args.allow_extrapolation)
        print(f"  at (x*={x:g}, y*={y:g}):  u* = {f['u']:+.9f}  "
              f"v* = {f['v']:+.9f}  p* = {f['p']:+.9f}")

    if args.profile_at is not None:
        y, u, v = solution.profile(args.profile_at)
        rows = list(zip(y, u, v))
        if args.out:
            _write_csv(args.out, ["y", "u", "v"], rows)
            print(f"  wrote {len(rows)} profile points to {args.out}")
        else:
            print(f"  profile at x* = {args.profile_at:g} (every 20th point):")
            for yy, uu, vv in rows[::20]:
                print(f"    y*={yy:+.4f}  u*={uu:+.9f}  v*={vv:+.9f}")

    if args.flux_at is not None:
        print(f"  flux at x* = {args.flux_at:g}: {solution.flux(args.flux_at):.12f}")

    if args.grid:
        nx, ny = args.grid
        x, y, u, v = solution.grid(nx, ny)
        rows = [
            (x[i], y[j], u[i, j], v[i, j]) for i in range(nx) for j in range(ny)
        ]
        if args.out:
            _write_csv(args.out, ["x", "y", "u", "v"], rows)
            print(f"  wrote {len(rows)} grid points to {args.out}")
        else:
            print(f"  grid {nx}x{ny}: u* in [{u.min():.6f}, {u.max():.6f}], "
                  f"max |v*| = {np.abs(v).max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

# ---------------------------------------------------------------------------
# src/report.py
# ---------------------------------------------------------------------------
FILES['src/report.py'] = r'''"""Run directory -> accuracy table.

    python -m src.report --run runs/<id>
    python -m src.report --all              # every run under runs/

Reports relative L2 error against the analytic solution as the accuracy
measure, and the loss separately and explicitly labelled as a diagnostic
(CLAUDE.md: "Loss value is a diagnostic, never an accuracy claim").

Each accuracy number is printed next to its SPEC target and a PASS/FAIL, so a
table cannot be read as a success without the reader seeing the target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS_ROOT = Path("runs")

#: SPEC section 5 targets, duplicated here for reporting only. The tests are
#: the gate; this module never decides whether something passed on its own
#: terms, it just displays the same numbers.
TARGETS = {
    "u_relative_l2": ("test 1  u relative L2", 1e-5, "<=", "A"),
    # SPEC test 2 is "Transverse velocity, Case A". Developing flow has a
    # genuine non-zero v*, so applying this target to Case B would report a
    # false failure for correct physics.
    "v_max_abs": ("test 2  max |v*|", 1e-5, "<=", "A"),
    "flux_conservation_relative_error": ("test 3  flux conservation", 1e-6, "<=", None),
    "outlet_profile_relative_l2": ("test 9  outlet profile L2", 1e-4, "<=", "B"),
    "entrance_length_relative_error": ("test 9  entrance length", 0.15, "<=", "B"),
}


def load_metrics(directory: Path) -> dict:
    path = directory / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; has this run finished?")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def format_run(metrics: dict) -> str:
    lines = []
    name = metrics.get("name", "?")
    lines.append(f"=== {name} ===")
    lines.append(
        f"  case {metrics['case']}   Re = {metrics['reynolds']:g}   "
        f"{metrics['formulation']}   {metrics['pressure_scaling']} pressure   "
        f"seed {metrics.get('seed')}"
    )

    lines.append("")
    lines.append("  ACCURACY (relative L2 against the analytic solution)")
    any_target = False
    for key, (label, target, _, only_case) in TARGETS.items():
        if key not in metrics:
            continue
        if only_case is not None and metrics["case"] != only_case:
            continue
        any_target = True
        value = metrics[key]
        verdict = "PASS" if value <= target else "FAIL"
        lines.append(f"    {label:<32} {value:10.3e}   target <= {target:<8.1e} {verdict}")
    if not any_target:
        lines.append("    (no targeted metrics present)")

    if metrics["case"] != "A":
        lines.append(
            f"    {'max |v*| (no target: case B has':<32} {metrics['v_max_abs']:10.3e}"
        )
        lines.append(f"    {'  genuine transverse velocity)':<32}")

    for key, label in (
        ("p_relative_l2", "p relative L2 (gauged at outlet)"),
        ("flux_mean", "mean flux"),
        ("flux_relative_error_vs_inlet", "flux error vs inlet flux"),
        ("outlet_profile_relative_l2_vs_nominal_flux", "outlet profile vs NOMINAL flux"),
        ("inlet_flux_deficit_relative", "inlet flux deficit (corner artefact)"),
        ("entrance_length_measured", "entrance length measured"),
        ("entrance_length_correlation", "entrance length correlation"),
    ):
        if key in metrics:
            lines.append(f"    {label:<32} {metrics[key]:10.3e}")

    lines.append("")
    lines.append("  DIAGNOSTICS -- these are NOT accuracy claims")
    for key, label in (
        ("diag_loss_after_adam", "loss after Adam"),
        ("diag_loss_after_lbfgs", "loss after L-BFGS"),
        ("diag_handoff_relative_gap", "Adam->L-BFGS loss discontinuity"),
        ("residual_momentum_x_max", "max |momentum-x residual|"),
        ("residual_momentum_y_max", "max |momentum-y residual|"),
        ("residual_continuity_max", "max |continuity residual|"),
        ("diag_parameter_count", "parameters"),
        ("diag_lbfgs_iterations", "L-BFGS iterations"),
        ("wall_seconds", "wall time (s)"),
    ):
        if key in metrics:
            value = metrics[key]
            formatted = f"{value:10.3e}" if isinstance(value, float) else f"{value:>10}"
            lines.append(f"    {label:<32} {formatted}")
    if "diag_lbfgs_message" in metrics:
        lines.append(f"    {'L-BFGS termination':<32} {metrics['diag_lbfgs_message']}")
    return "\n".join(lines)


def format_table(all_metrics: list) -> str:
    header = (
        f"{'run':<28}{'case':>5}{'Re':>7}{'form':>18}"
        f"{'u relL2':>11}{'max|v|':>11}{'flux cons':>11}{'loss':>11}"
    )
    lines = [header, "-" * len(header)]
    for metrics in all_metrics:
        loss = metrics.get("diag_loss_after_lbfgs", metrics.get("diag_loss_after_adam"))
        accuracy = metrics.get("u_relative_l2", metrics.get("outlet_profile_relative_l2"))
        lines.append(
            f"{metrics.get('name', '?'):<28}{metrics['case']:>5}"
            f"{metrics['reynolds']:>7g}{metrics['formulation']:>18}"
            f"{accuracy:>11.2e}{metrics['v_max_abs']:>11.2e}"
            f"{metrics['flux_conservation_relative_error']:>11.2e}"
            f"{(loss if loss is not None else float('nan')):>11.2e}"
        )
    lines.append("")
    lines.append(
        "u relL2 is against the analytic solution; for case B it is the outlet "
        "profile against the flux-consistent parabola."
    )
    lines.append("loss is a DIAGNOSTIC and is never an accuracy claim.")
    return "\n".join(lines)


def _write_falsifiability_matrix() -> int:
    """Regenerate the S5 matrix. Must be re-run on any ansatz change."""
    import yaml

    from .config import parse_config
    from .evaluation.falsifiability import (
        all_vacuous_rows, build_matrix, degenerate_targets, format_matrix,
    )

    columns = {
        "vp-soft": "experiments/caseA_re10_baseline.yaml",
        "vp-hard": "experiments/caseA_re10.yaml",
        "psi-cubic": "experiments/caseA_re10_psiomega_cubic.yaml",
        "psi-sin": "experiments/caseA_re10_psiomega_sinusoidal.yaml",
    }
    configs = {
        label: parse_config(yaml.safe_load(open(path, encoding="utf-8")))
        for label, path in columns.items()
    }
    matrix = build_matrix(configs)
    table = format_matrix(matrix, list(columns))

    document = [
        "# Falsifiability matrix (SPEC v2, S5)",
        "",
        "Generated by `python -m src.report --falsifiability`.",
        "**Regenerate on every ansatz change** -- a stale matrix is the silent",
        "disarming this exists to prevent, and `tests/test_sentinels.py` fails",
        "if this file does not match the code.",
        "",
        "A check is classified by running it on an UNTRAINED, randomly",
        "initialised network at five seeds:",
        "",
        "* `LIVE` - fails at every seed, so the check bites.",
        "* `WEAKENED` - passes at some seeds, so it partly measures the ansatz.",
        "* `VACUOUS` - passes at every seed, so it measures the ansatz and not",
        "  the trained solution.",
        "",
        "Columns: `vp-*` velocity-pressure with the soft/hard pressure gauge;",
        "`psi-*` the psi-omega system under the cubic/sinusoidal lift.",
        "",
        table,
        "",
    ]
    degenerate = {label: degenerate_targets(cfg) for label, cfg in configs.items()}
    document += [
        "## Degenerate targets",
        "",
        "A check can be `LIVE` -- an untrained network fails it -- and still be",
        "nearly free, because the ansatz places the correct answer at a single",
        "point the optimiser can reach exactly. Sampling at the initialisation",
        "distribution cannot see that, so this probes it directly by zeroing the",
        "output layer, i.e. removing the network's contribution entirely.",
        "",
        "Checks that PASS with no network contribution at all:",
        "",
    ]
    for label in columns:
        names = degenerate[label]
        document.append(f"* `{label}`: " + (", ".join(names) if names else "none"))
    document += [
        "",
        "Read `psi-cubic` carefully: essentially the whole Case A suite passes",
        "with the network removed, because the cubic lift's derivative IS the",
        "Case A solution. Its accuracy must never be quoted as a result.",
        "",
        "`test 2  max |v*|` is degenerate in EVERY column: the zero field passes",
        "it. It cannot distinguish a correct solution on its own and is only",
        "meaningful conjoined with test 1.",
        "",
    ]

    vacuous = all_vacuous_rows(matrix)
    if vacuous:
        document += [
            "## Untested properties",
            "",
            "These are `VACUOUS` in every column, so the suite does not test",
            "them at all:",
            "",
        ] + [f"* {name}" for name in vacuous] + [""]
    Path("docs/falsifiability-matrix.md").write_text(
        "\n".join(document), encoding="utf-8"
    )
    print("\n".join(document))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="a single run directory, e.g. runs/caseA_re10")
    group.add_argument("--all", action="store_true", help="tabulate every run")
    group.add_argument(
        "--falsifiability", action="store_true",
        help="regenerate docs/falsifiability-matrix.md (SPEC v2 S5)",
    )
    parser.add_argument("--table", action="store_true", help="one-line-per-run table")
    args = parser.parse_args(argv)

    if args.falsifiability:
        return _write_falsifiability_matrix()

    if args.run:
        metrics = load_metrics(Path(args.run))
        print(format_table([metrics]) if args.table else format_run(metrics))
        return 0

    directories = sorted(d for d in RUNS_ROOT.iterdir() if (d / "metrics.json").is_file())
    if not directories:
        print(f"no finished runs found under {RUNS_ROOT}/")
        return 1
    all_metrics = [load_metrics(d) for d in directories]
    if args.table:
        print(format_table(all_metrics))
    else:
        for metrics in all_metrics:
            print(format_run(metrics))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

# ---------------------------------------------------------------------------
# src/run.py
# ---------------------------------------------------------------------------
FILES['src/run.py'] = r'''"""Config -> experiment.

    python -m src.run --config experiments/<name>.yaml

Writes everything a run produced into runs/<name>/. That directory is written
by the code and never edited by hand (CLAUDE.md, "Module layout").
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import jax
import numpy as np
from jax.flatten_util import ravel_pytree

from .config import Config, load_config
from .evaluation.accuracy import evaluate
from .evaluation import plots
from .models.build import build_model
from .training.loop import train

RUNS_ROOT = Path("runs")


def run_directory(config: Config, override: str | None) -> Path:
    return Path(override) if override else RUNS_ROOT / config.name


def save_parameters(path: Path, params) -> None:
    """Store the flattened parameter vector.

    The pytree structure is fully determined by the config, so `load_parameters`
    rebuilds it rather than pickling it. That keeps run directories readable and
    version-independent.
    """
    flat, _ = ravel_pytree(params)
    np.savez_compressed(path, flat=np.asarray(flat, dtype=np.float64))


def load_parameters(path: Path, config: Config):
    template, model = build_model(config, jax.random.PRNGKey(config.seed))
    _, unflatten = ravel_pytree(template)
    flat = np.load(path)["flat"]
    return unflatten(flat), model


def write_history(path: Path, rows: list) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def execute(config_path: str, out: str | None = None, verbose: bool = True) -> Path:
    config = load_config(config_path)
    directory = run_directory(config, out)
    directory.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[{config.name}] case {config.case}, Re = {config.physics.reynolds}, "
              f"{config.physics.formulation}, {config.physics.pressure_scaling} pressure")
        print(f"[{config.name}] active loss terms: {', '.join(config.loss_terms)}")

    started = time.time()
    result = train(config, verbose=verbose)
    metrics = evaluate(config, result.model, result.params)
    metrics["wall_seconds"] = time.time() - started
    metrics.update({f"diag_{k}": v for k, v in result.diagnostics.items()})
    metrics["name"] = config.name
    metrics["seed"] = config.seed

    shutil.copyfile(config_path, directory / "config.yaml")
    save_parameters(directory / "params.npz", result.params)
    write_history(directory / "history.csv", result.history)
    # T-S7.2: the D(n) trace is written for every run.
    write_history(directory / "drift_trace.csv", result.drift_trace)
    with (directory / "lbfgs.json").open("w", encoding="utf-8") as handle:
        json.dump(result.lbfgs_history, handle, indent=2)
    with (directory / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    plots.write_all(config, result.model, result.params, directory)
    with (directory / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "jax": jax.__version__,
                "jax_devices": [str(d) for d in jax.devices()],
                "x64_enabled": bool(jax.config.jax_enable_x64),
            },
            handle,
            indent=2,
        )

    if verbose:
        print(f"[{config.name}] wrote {directory}")
        _print_summary(metrics)
    return directory


def _show(metrics: dict, key: str, label: str, suffix: str = "") -> None:
    """Print one metric if present. Absence is normal, not an error.

    Which metrics exist depends on the formulation and the case: the psi-omega
    system has no pressure in it, and Case A has no entrance length. Printing
    unconditionally is how this crashed every psi-omega run after its results
    were already safely written.
    """
    if key in metrics and metrics[key] is not None:
        print(f"  {label:<21}: {metrics[key]:.3e}{suffix}")


def _print_summary(metrics: dict) -> None:
    print("  --- accuracy (relative L2 against the analytic solution) ---")
    if metrics["case"] == "A":
        _show(metrics, "u_relative_l2", "u relative L2", "   (target <= 1e-5)")
        _show(metrics, "p_relative_l2", "p relative L2")
        _show(metrics, "p_relative_l2_recovered", "p relative L2 (recovered)")
    else:
        _show(metrics, "outlet_profile_relative_l2", "outlet profile L2",
              "  (target <= 1e-4)")
        if "entrance_length_measured" in metrics:
            print(f"  {'entrance length':<21}: {metrics['entrance_length_measured']:.4f} vs "
                  f"{metrics['entrance_length_correlation']:.4f} correlation "
                  f"({metrics['entrance_length_relative_error'] * 100:.1f}%, target <= 15%)")
    _show(metrics, "v_max_abs", "max |v*|",
          "   (target <= 1e-5)" if metrics["case"] == "A"
          else "   (no target: case B has genuine transverse velocity)")
    _show(metrics, "flux_conservation_relative_error", "mass drift D",
          "   (target <= 1e-6)")
    _show(metrics, "flux_relative_error_vs_inlet", "flux vs inlet flux")

    print("  --- diagnostics (NOT accuracy claims) ---")
    final = metrics.get("diag_loss_after_lbfgs", metrics.get("diag_loss_after_adam"))
    if final is not None:
        print(f"  {'final loss':<21}: {final:.3e}")
    _show(metrics, "residual_momentum_x_max", "max momentum-x resid")
    _show(metrics, "residual_vorticity_max", "max r1 resid")
    _show(metrics, "residual_vorticity_transport_max", "max r2 resid")
    _show(metrics, "pressure_path_max_disagreement", "pressure path disagree")
    # Guard at display time as well as at fit time: metrics.json is a snapshot
    # written by whatever code produced the run, so a stored verdict may predate
    # the machine-precision guard in drift.py.
    drift = metrics.get("flux_conservation_relative_error")
    if drift is not None and drift < 1e-12:
        print(f"  {'drift':<21}: D is at machine precision -- this formulation "
              "conserves mass by")
        print(f"  {'':<21}  construction, so test 3 is VACUOUS here, not passed")
    elif metrics.get("diag_drift_verdict"):
        print(f"  {'drift D_inf':<21}: {metrics['diag_drift_d_infinity']:.3e} "
              f"+- {metrics['diag_drift_d_infinity_stderr']:.1e}")
        print(f"  {'drift verdict':<21}: {metrics['diag_drift_verdict']}")
    print(f"  {'wall time':<21}: {metrics['wall_seconds']:.1f} s")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to an experiments/*.yaml")
    parser.add_argument("--out", default=None, help="override the run directory")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    execute(args.config, args.out, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

# ---------------------------------------------------------------------------
# src/training/__init__.py
# ---------------------------------------------------------------------------
FILES['src/training/__init__.py'] = r'''"""Optimisers, loss assembly, weighting, and the training loop."""
'''

# ---------------------------------------------------------------------------
# src/training/loop.py
# ---------------------------------------------------------------------------
FILES['src/training/loop.py'] = r'''"""The training loop: Adam with resampling, then L-BFGS on frozen points.

Determinism (SPEC test 8) is structural here: every random draw comes from a
PRNG key derived from `config.seed` or `config.sampling.seed`, there is no
wall-clock or hash-order dependence, and the L-BFGS phase is SciPy's
deterministic implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..config import Config
from ..geometry.domain import Channel
from ..geometry.sampling import sample_all
from ..models.build import build_model, count_parameters
from .losses import make_loss_terms
from .optimisers import adam_schedule, make_adam, run_lbfgs
from ..evaluation.drift import fit_drift, mass_drift
from ..evaluation.metrics import cross_channel_flux
from .weights import initial_weights, make_grad_norm_update, grad_norm_weights, total_loss

#: Target number of logged rows across the Adam phase. Logging cadence is
#: derived from this rather than configured, because it changes nothing about
#: the result -- only how densely the history is recorded.
TARGET_LOG_ROWS = 200


#: Points across the channel used by the live accuracy probe. Small on purpose:
#: it runs at every log interval, and the reported accuracy at the END of a run
#: is always recomputed on the full evaluation grid.
PROBE_NX, PROBE_NY = 81, 41


def _make_accuracy_probe(config, model):
    """Relative L2 against the analytic solution, as a function of parameters.

    This is the number that actually means something during training. The loss
    is a diagnostic and can fall while this rises -- that is the failure mode
    the whole project is built around -- so both are logged at every interval
    and printed side by side.

    Case A has an analytic solution everywhere. Case B does not, so the probe
    reports the outlet profile against the flux-consistent developed parabola,
    which is what SPEC test 9 measures.
    """
    from ..geometry.sampling import evaluation_grid
    from ..physics import analytic

    channel = Channel(config.domain.length)
    if config.case == "A":
        points, _, _ = evaluation_grid(channel, PROBE_NX, PROBE_NY)
        reference = analytic.u_nd(np.asarray(points[:, 1]))
    else:
        y = np.linspace(-1.0, 1.0, PROBE_NY * 4)
        points = jnp.stack(
            [jnp.full(len(y), config.domain.length), jnp.asarray(y)], axis=-1
        )
        reference = analytic.developed_profile_nd(y, config.inlet_flux)

    norm = float(np.linalg.norm(reference))

    def probe(params):
        u = np.asarray(
            jax.vmap(lambda xy: model.velocity(params, xy))(points)[:, 0]
        )
        return float(np.linalg.norm(u - reference) / norm)

    return probe


def _make_drift_probe(config, model):
    """D on the fixed evaluation stations, as a function of parameters alone."""
    length = config.domain.length
    stations = np.linspace(0.05 * length, 0.95 * length, 10)

    def probe(params):
        try:
            return mass_drift(cross_channel_flux(model.field, params, stations, 64))
        except (ValueError, FloatingPointError):
            return float("nan")

    return probe


@dataclass
class TrainingResult:
    params: Any
    model: Any
    history: list = field(default_factory=list)
    lbfgs_history: list = field(default_factory=list)
    drift_trace: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def train(config: Config, *, verbose: bool = True) -> TrainingResult:
    channel = Channel(config.domain.length)

    model_key = jax.random.PRNGKey(config.seed)
    params, model = build_model(config, model_key)

    loss_terms_fn = make_loss_terms(config, model)
    weights = initial_weights(config)

    def weighted_loss(p, points, w):
        return total_loss(loss_terms_fn(p, points), w)

    optimiser = make_adam(config)
    opt_state = optimiser.init(params)
    schedule = adam_schedule(config)

    @jax.jit
    def adam_step(p, state, points, w):
        def loss_and_terms(q):
            # `has_aux` returns the per-term losses from the SAME trace that
            # produced the gradient, instead of a second forward pass just to
            # log them.
            terms = loss_terms_fn(q, points)
            return total_loss(terms, w), terms

        (loss, terms), grads = jax.value_and_grad(loss_and_terms, has_aux=True)(p)
        updates, state = optimiser.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss, terms

    term_grad_norms = None
    if config.weights.scheme == "grad_norm":
        term_grad_norms = jax.jit(make_grad_norm_update(loss_terms_fn))

    # Held-out validation points. Drawn from a PRNG stream that the training
    # sampler never reaches, so they are disjoint from every training draw.
    validation_points = sample_all(
        jax.random.PRNGKey(config.sampling.seed + 1_000_003),
        channel,
        config.sampling.n_validation,
        config.sampling.n_inlet,
        config.sampling.n_outlet,
    )
    validation_loss = jax.jit(
        lambda p: total_loss(loss_terms_fn(p, validation_points), weights)
    )

    best = {"validation": float("inf"), "params": params, "where": "init"}

    def consider(p, where):
        """Keep the parameters with the lowest HELD-OUT loss seen so far."""
        value = float(validation_loss(p))
        if value < best["validation"]:
            best.update(validation=value, params=jax.tree.map(lambda a: a, p), where=where)
        return value

    sample_key = jax.random.PRNGKey(config.sampling.seed)
    points = sample_all(
        sample_key,
        channel,
        config.sampling.n_interior,
        config.sampling.n_inlet,
        config.sampling.n_outlet,
    )

    history = []
    log_every = max(1, config.training.adam.steps // TARGET_LOG_ROWS)
    accuracy_probe = _make_accuracy_probe(config, model)
    resample_every = config.sampling.resample_every
    started = time.time()
    if verbose:
        print(
            f"  [adam] {config.training.adam.steps} steps, resampling every "
            f"{resample_every} (one 'epoch' = one resampling interval, "
            f"{config.training.adam.steps // max(resample_every, 1)} of them)",
            flush=True,
        )
        print(f"  {'step':>8} {'epoch':>6} {'train loss':>12} {'held-out':>12} "
              f"{'rel L2 vs exact':>16} {'lr':>9}", flush=True)

    for step in range(config.training.adam.steps):
        if step > 0 and step % config.sampling.resample_every == 0:
            sample_key, draw_key = jax.random.split(sample_key)
            points = sample_all(
                draw_key,
                channel,
                config.sampling.n_interior,
                config.sampling.n_inlet,
                config.sampling.n_outlet,
            )

        if (
            term_grad_norms is not None
            and step % config.weights.grad_norm.update_every == 0
        ):
            weights = grad_norm_weights(
                term_grad_norms(params, points), weights, config.weights.grad_norm.alpha
            )

        params, opt_state, loss, terms = adam_step(params, opt_state, points, weights)

        if step % log_every == 0 or step == config.training.adam.steps - 1:
            accuracy = accuracy_probe(params)
            held_out = consider(params, f"adam:{step}")
            row = {
                "phase": "adam",
                "step": step,
                "epoch": step // max(resample_every, 1),
                "loss": float(loss),
                "validation_loss": held_out,
                "relative_l2_vs_exact": accuracy,
                "lr": float(schedule(step)),
                **{f"term_{k}": float(v) for k, v in terms.items()},
                **{f"weight_{k}": float(v) for k, v in weights.items()},
            }
            history.append(row)
            if verbose and (step % (log_every * 10) == 0):
                print(
                    f"  {step:>8d} {step // max(resample_every, 1):>6d} "
                    f"{float(loss):>12.4e} {held_out:>12.4e} {accuracy:>16.4e} "
                    f"{float(schedule(step)):>9.2e}",
                    flush=True,
                )

    adam_seconds = time.time() - started

    diagnostics = {
        "parameter_count": count_parameters(params),
        "adam_seconds": adam_seconds,
        "final_weights": {k: float(v) for k, v in weights.items()},
    }

    # -- L-BFGS on FROZEN points -------------------------------------------- #
    lbfgs_history = []
    drift_trace = []
    if config.training.lbfgs.enabled:
        frozen = points
        loss_before = float(weighted_loss(params, frozen, weights))

        def objective(p):
            return weighted_loss(p, frozen, weights)

        # SPEC v2 S7: trace the mass drift D against L-BFGS iteration. Measured
        # on the FIXED evaluation stations, never on the training collocation
        # points (T-S7.1) -- the frozen set above is the training set.
        drift_fn = _make_drift_probe(config, model)

        def trace_fn(p):
            return {
                "drift": drift_fn(p),
                "relative_l2_vs_exact": accuracy_probe(p),
                "validation_loss": consider(p, "lbfgs"),
            }

        if verbose:
            print(
                f"  [lbfgs] up to {config.training.lbfgs.max_iterations} iterations",
                flush=True,
            )
            print(f"  {'iter':>8} {'':>6} {'train loss':>12} {'held-out':>12} "
                  f"{'rel L2 vs exact':>16} {'drift D':>9}", flush=True)

        def on_trace(n, values, loss):
            if verbose:
                print(
                    f"  {n:>8d} {'':>6} {loss:>12.4e} "
                    f"{values['validation_loss']:>12.4e} "
                    f"{values['relative_l2_vs_exact']:>16.4e} {values['drift']:>9.2e}",
                    flush=True,
                )

        started = time.time()
        result = run_lbfgs(
            objective,
            params,
            config.training.lbfgs.max_iterations,
            config.training.lbfgs.gtol,
            trace_fn=trace_fn,
            on_trace=on_trace,
        )
        lbfgs_seconds = time.time() - started

        # SPEC section 9: verify the loss is continuous across the handoff. The
        # first L-BFGS evaluation is at the Adam parameters on the same frozen
        # points, so any difference is a bug in the handoff, not optimisation.
        handoff_gap = abs(result.initial_loss - loss_before)
        handoff_relative = handoff_gap / max(abs(loss_before), 1e-300)

        final_validation = consider(result.params, "lbfgs:final")
        params = result.params
        diagnostics["validation_loss_final"] = final_validation
        lbfgs_history.append(
            {
                "phase": "lbfgs",
                "loss_at_handoff_adam_side": loss_before,
                "loss_at_handoff_lbfgs_side": result.initial_loss,
                "handoff_absolute_gap": handoff_gap,
                "handoff_relative_gap": handoff_relative,
                "final_loss": result.final_loss,
                "iterations": result.iterations,
                "function_evaluations": result.function_evaluations,
                "status": result.status,
                "message": result.message,
            }
        )
        drift_trace = result.trace
        if drift_trace:
            fit = fit_drift(
                [row["iteration"] for row in drift_trace],
                [row["drift"] for row in drift_trace],
            )
            diagnostics.update(
                {
                    "drift_d_infinity": fit.d_infinity,
                    "drift_d_infinity_stderr": fit.d_infinity_stderr,
                    "drift_exponent": fit.exponent,
                    "drift_fit_points": fit.points,
                    "drift_fit_converged": fit.converged,
                    "drift_verdict": fit.verdict(),
                }
            )

        diagnostics.update(
            {
                "lbfgs_seconds": lbfgs_seconds,
                "lbfgs_iterations": result.iterations,
                "lbfgs_status": result.status,
                "lbfgs_message": result.message,
                "handoff_relative_gap": handoff_relative,
                "loss_after_adam": loss_before,
                "loss_after_lbfgs": result.final_loss,
            }
        )
        if verbose:
            print(
                f"  lbfgs {result.iterations} iters, "
                f"loss {loss_before:.6e} -> {result.final_loss:.6e}, "
                f"handoff gap {handoff_relative:.2e} ({result.message})",
                flush=True,
            )
    else:
        diagnostics["loss_after_adam"] = float(weighted_loss(params, points, weights))

    # SPEC v2 does not cover this; it is the fix for the loss/accuracy
    # divergence. Selection uses the HELD-OUT loss, never the analytic
    # solution, so the same mechanism works in Phase 2.
    diagnostics["validation_loss_best"] = best["validation"]
    diagnostics["selected_from"] = best["where"]
    diagnostics["selection_rule"] = config.training.select
    if config.training.select == "best_validation":
        final_value = diagnostics.get("validation_loss_final")
        improved = final_value is None or best["validation"] < final_value
        if verbose and best["where"] != "lbfgs:final" and improved:
            print(
                f"  selected parameters from {best['where']} "
                f"(held-out loss {best['validation']:.4e}) rather than the final "
                f"iterate ({diagnostics.get('validation_loss_final', float('nan')):.4e})",
                flush=True,
            )
        params = best["params"]

    return TrainingResult(
        drift_trace=drift_trace if config.training.lbfgs.enabled else [],
        params=params,
        model=model,
        history=history,
        lbfgs_history=lbfgs_history,
        diagnostics=diagnostics,
    )
'''

# ---------------------------------------------------------------------------
# src/training/losses.py
# ---------------------------------------------------------------------------
FILES['src/training/losses.py'] = r'''"""Per-term losses. Unweighted.

Each term is the mean squared residual of one governing equation or one
boundary condition, evaluated on the collocation points for that term. Nothing
here multiplies by a weight -- that happens in `weights.py` and nowhere else
(CLAUDE.md, "Code style").

There is deliberately no wall term. No-slip is exact by construction in the
ansatz, so a wall loss would be identically zero and would only dilute the
gradient of the terms that matter.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..config import Config
from ..physics import residuals, streamfunction
from ..physics.derivatives import (
    batched_field_and_jacobian,
    batched_field_derivatives,
    batched_stream_derivatives,
)


def inlet_profile(config: Config, y):
    """Prescribed streamwise velocity at x* = 0.

    Case A: the fully developed parabola u* = 1 - y*^2.
    Case B: the uniform profile u* = `physics.inlet_uniform_value`.
    """
    if config.case == "A":
        return 1.0 - y**2
    return jnp.full_like(y, config.inlet_uniform_value)


def make_loss_terms(config: Config, model):
    """Build `loss_terms(params, points) -> dict[str, scalar]`.

    The returned dict has exactly the keys in `config.loss_terms`, so a term
    can never be silently dropped or silently added.
    """
    reynolds = config.physics.reynolds
    pressure_scaling = config.physics.pressure_scaling
    formulation = config.physics.formulation

    def loss_terms(params, points):
        point_fn = lambda xy: model.field(params, xy)  # noqa: E731

        if formulation == "psi_omega":
            stream_point_fn = lambda xy: model.stream(params, xy)  # noqa: E731
            sd = batched_stream_derivatives(stream_point_fn, points.interior)
            r1, r2 = streamfunction.residuals(sd, reynolds)
            terms = {
                "vorticity": jnp.mean(r1**2),
                "vorticity_transport": jnp.mean(r2**2),
            }
        else:
            d = batched_field_derivatives(point_fn, points.interior)
            terms = {
                "momentum_x": jnp.mean(
                    residuals.momentum_x(d, reynolds, pressure_scaling) ** 2
                ),
                "momentum_y": jnp.mean(
                    residuals.momentum_y(d, reynolds, pressure_scaling) ** 2
                ),
                "continuity": jnp.mean(residuals.continuity(d) ** 2),
            }

        if config.bc.inlet == "soft":
            inlet_values, _ = batched_field_and_jacobian(point_fn, points.inlet)
            target = inlet_profile(config, points.inlet[:, 1])
            terms["inlet"] = jnp.mean((inlet_values[:, 0] - target) ** 2) + jnp.mean(
                inlet_values[:, 1] ** 2
            )

        needs_outlet = (
            config.bc.outlet_velocity == "soft"
            or config.bc.pressure_gauge == "soft"
        )
        if needs_outlet:
            outlet_values, outlet_jac = batched_field_and_jacobian(
                point_fn, points.outlet
            )
            if config.bc.outlet_velocity == "soft":
                # Zero streamwise gradient of velocity: du/dx = dv/dx = 0.
                terms["outlet_velocity"] = jnp.mean(
                    outlet_jac[:, 0, 0] ** 2
                ) + jnp.mean(outlet_jac[:, 1, 0] ** 2)
            if config.bc.pressure_gauge == "soft":
                # The gauge fix. Without it the pressure error is meaningless.
                terms["pressure_gauge"] = jnp.mean(outlet_values[:, 2] ** 2)

        return terms

    return loss_terms
'''

# ---------------------------------------------------------------------------
# src/training/optimisers.py
# ---------------------------------------------------------------------------
FILES['src/training/optimisers.py'] = r'''"""Optimiser schedule: Adam with exponential decay, then L-BFGS.

The handoff between the two is the delicate part. SPEC section 9 lists it as a
known trap: "Adam -> L-BFGS handoff frequently resets or mishandles state.
Verify the loss is continuous across the switch." Two things are done about it
here:

1. The collocation points are FROZEN for the whole L-BFGS phase. L-BFGS builds
   a curvature estimate from a sequence of gradients; resampling underneath it
   changes the objective between evaluations and the curvature estimate becomes
   meaningless. (This also means the loss values reported during L-BFGS are on
   a different point set from the Adam ones, which is why accuracy is always
   measured on the fixed evaluation grid instead.)

2. `run_lbfgs` returns the loss at its own first function evaluation, so the
   caller can compare it against the last Adam loss ON THE SAME POINTS and
   record the discontinuity. It should be zero to rounding.

L-BFGS itself is SciPy's, driven on a flattened float64 parameter vector.
SciPy's implementation is the well-tested Nocedal Fortran code and it is
deterministic, which SPEC test 8 depends on.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize


def adam_schedule(config):
    """Exponentially decaying learning rate, as configured."""
    return optax.exponential_decay(
        init_value=config.training.adam.lr_init,
        transition_steps=config.training.adam.decay_steps,
        decay_rate=config.training.adam.decay_rate,
        staircase=False,
    )


def make_adam(config):
    return optax.adam(learning_rate=adam_schedule(config))


class LBFGSResult(NamedTuple):
    params: object
    initial_loss: float
    final_loss: float
    iterations: int
    function_evaluations: int
    status: int
    message: str
    trace: list = []


def run_lbfgs(
    loss_fn: Callable, params, max_iterations: int, gtol: float,
    *, trace_fn=None, trace_every: int | None = None, on_trace=None,
):
    """Minimise `loss_fn(params)` with SciPy L-BFGS-B on a flat float64 vector.

    `loss_fn` must be a pure function of the parameters alone; bind the
    collocation points before calling.

    `trace_fn(params) -> float` is sampled every `trace_every` iterations and
    recorded against the iteration count. SPEC v2 S7 uses this to trace the
    mass drift D against L-BFGS iteration and fit D(n) = D_inf + a n^-b, which
    is what discriminates formulation bias from under-convergence.
    """
    flat_initial, unflatten = ravel_pytree(params)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    evaluations = {"count": 0, "first_loss": None}

    def objective(flat):
        value, grad = value_and_grad(unflatten(jnp.asarray(flat)))
        evaluations["count"] += 1
        if evaluations["first_loss"] is None:
            evaluations["first_loss"] = float(value)
        flat_grad, _ = ravel_pytree(grad)
        # SciPy needs contiguous float64 numpy arrays, not JAX arrays.
        return float(value), np.asarray(flat_grad, dtype=np.float64)

    trace = []
    if trace_fn is not None:
        if trace_every is None:
            trace_every = max(1, max_iterations // 40)
        state = {"iteration": 0}

        def callback(flat):
            state["iteration"] += 1
            n = state["iteration"]
            if n % trace_every == 0 or n == 1:
                values = trace_fn(unflatten(jnp.asarray(flat)))
                if not isinstance(values, dict):
                    values = {"value": float(values)}
                trace.append({"iteration": n, **values})
                if on_trace is not None:
                    on_trace(n, values, float(value_and_grad(unflatten(jnp.asarray(flat)))[0]))
    else:
        callback = None

    result = minimize(
        objective,
        np.asarray(flat_initial, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={
            "maxiter": max_iterations,
            "maxfun": max_iterations * 2,
            "gtol": gtol,
            # ftol = 0 so termination is governed by gtol alone; the default
            # ftol stops at ~1e-9 relative change, far short of the accuracy
            # this project targets.
            "ftol": 0.0,
            "maxcor": 100,
        },
    )

    return LBFGSResult(
        params=unflatten(jnp.asarray(result.x)),
        initial_loss=evaluations["first_loss"],
        final_loss=float(result.fun),
        iterations=int(result.nit),
        function_evaluations=int(evaluations["count"]),
        status=int(result.status),
        message=str(result.message),
        trace=trace,
    )
'''

# ---------------------------------------------------------------------------
# src/training/weights.py
# ---------------------------------------------------------------------------
FILES['src/training/weights.py'] = r'''"""The ONLY place loss weighting happens.

CLAUDE.md: "Residual functions return unweighted residuals. All loss weighting
happens in one place (src/training/weights.py)." If you find yourself wanting
to scale a residual anywhere else, the scaling belongs here instead.

Two schemes:

fixed
    The weights from the config, held constant. Start here.

grad_norm
    Learning-rate annealing of the weights (Wang, Teng & Perdikaris 2021). Every
    `update_every` steps:

        w_hat_i = (sum_j ||grad_theta L_j||) / ||grad_theta L_i||
        w_i     <- alpha * w_i + (1 - alpha) * w_hat_i

    which equalises the gradient contribution of each term. The config weights
    are the starting values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def initial_weights(config):
    """Starting weights, ordered exactly as `config.loss_terms`."""
    return {term: jnp.asarray(config.weights.values[term], dtype=jnp.float64)
            for term in config.loss_terms}


def total_loss(terms, weights):
    """Weighted sum. The single point where weights meet residuals."""
    missing = set(weights) ^ set(terms)
    if missing:
        raise KeyError(
            f"loss terms and weights disagree on {sorted(missing)}; "
            "this means the config's active-term list and the loss builder "
            "have drifted apart"
        )
    return sum(weights[name] * terms[name] for name in sorted(terms))


def grad_norm_weights(term_grad_norms, previous, alpha):
    """One grad-norm annealing update.

    `term_grad_norms` maps term name -> ||grad_theta L_term||_2 (the norm of the
    gradient of the UNWEIGHTED term). A term whose gradient norm is zero (e.g. a
    boundary condition already satisfied exactly) keeps its previous weight
    rather than producing an infinity.
    """
    total = sum(term_grad_norms.values())
    updated = {}
    for name, norm in term_grad_norms.items():
        target = jnp.where(norm > 0.0, total / jnp.where(norm > 0.0, norm, 1.0),
                           previous[name])
        updated[name] = alpha * previous[name] + (1.0 - alpha) * target
    return updated


def make_grad_norm_update(loss_terms_fn):
    """Build a jittable function computing per-term gradient norms."""

    def term_grad_norms(params, points):
        norms = {}
        for name in loss_terms_fn(params, points):
            grad = jax.grad(lambda p, n=name: loss_terms_fn(p, points)[n])(params)
            leaves = jax.tree.leaves(grad)
            norms[name] = jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))
        return norms

    return term_grad_norms
'''



# ===========================================================================
# RUNNER
# ===========================================================================


def unpack(destination="."):
    """Write the embedded source files to disk."""
    root = pathlib.Path(destination)
    for relative, text in FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return len(FILES)


def require_dependencies():
    missing = []
    for module, install in (("jax", "jax"), ("optax", "optax"),
                            ("yaml", "pyyaml"), ("numpy", "numpy"),
                            ("scipy", "scipy")):
        try:
            __import__(module)
        except ImportError:
            missing.append(install)
    if missing:
        print("Missing dependencies. Install them with:\n")
        print(f"    pip install {' '.join(sorted(set(missing)))}\n")
        sys.exit(1)


CONFIG = """
name: all_in_one
seed: 0
case: A

domain: {alpha: 1.5, outlet_margin: 5.0}

correlation:
  c0: 0.3155
  c1: 0.01105
  n: 1.6
  citation: >
    Durst, Ray, Unsal & Bayoumi (2005), J. Fluids Eng. 127:1154-1160,
    plane-channel fit, converted to hydraulic-diameter units.

physics:
  reynolds: 10.0
  pressure_scaling: inertial
  formulation: velocity_pressure

model:
  arch: modified_mlp
  width: WIDTH
  depth: DEPTH
  fourier: {enabled: false}
  rwf: {enabled: false}

bc: {inlet: soft, outlet_velocity: soft, pressure_gauge: hard}

sampling:
  n_interior: NINT
  n_validation: NINT
  n_inlet: 128
  n_outlet: 128
  resample_every: 500
  seed: 1

weights:
  scheme: fixed
  values: {momentum_x: 1.0, momentum_y: 1.0, continuity: 1.0, inlet: 1.0, outlet_velocity: 1.0}

training:
  select: best_validation
  adam:  {steps: ADAM, lr_init: 1.0e-3, decay_rate: 0.9, decay_steps: DECAY}
  lbfgs: {enabled: true, max_iterations: LBFGS, gtol: 1.0e-14}

evaluation: {nx: NX, ny: NY}
"""


def build_config(full):
    substitutions = (
        {"WIDTH": 128, "DEPTH": 4, "NINT": 2048, "ADAM": 2000,
         "DECAY": 1000, "LBFGS": 6000, "NX": 401, "NY": 201}
        if full else
        {"WIDTH": 48, "DEPTH": 3, "NINT": 768, "ADAM": 500,
         "DECAY": 250, "LBFGS": 800, "NX": 121, "NY": 61}
    )
    text = CONFIG
    for key, value in substitutions.items():
        text = text.replace(key, str(value))
    return text


def check_physics():
    """Verify the equations before trusting anything trained from them."""
    import numpy as np, jax.numpy as jnp
    from src.physics import analytic, residuals
    from src.physics.derivatives import batched_field_derivatives
    from src.physics.conventions import re_h_to_re_dh

    print("\n--- VERIFYING THE PHYSICS (no training) ---")

    Re, L = 10.0, 7.824
    field = lambda xy: jnp.array([1.0 - xy[1] ** 2, 0.0, -2.0 / Re * (xy[0] - L)])
    pts = jnp.asarray(np.random.default_rng(0).uniform([0, -1], [L, 1], size=(512, 2)))
    r = residuals.navier_stokes(batched_field_derivatives(field, pts), Re, "inertial")
    worst = float(jnp.max(jnp.abs(jnp.stack(r))))
    print(f"  analytic solution satisfies the equations : max|residual| = {worst:.2e}   (want ~0)")

    nodes, weights = np.polynomial.legendre.leggauss(200)
    flux = float(analytic.u_nd(nodes) @ weights)
    print(f"  flux of the reference profile             : {flux:.15f}")
    print(f"                                       exact: {4/3:.15f}")
    print(f"  Reynolds convention Re_Dh / Re_h          : {re_h_to_re_dh(1.0):.15f}  (want {8/3:.15f})")
    assert worst < 1e-12, "the analytic solution does not satisfy the residual operator"
    print("  OK")


def main():
    parser = argparse.ArgumentParser(description="PINN for plane Poiseuille flow")
    parser.add_argument("--full", action="store_true",
                        help="full accuracy schedule (~30 min) instead of the quick demo")
    parser.add_argument("--check", action="store_true",
                        help="verify the physics and exit, without training")
    parser.add_argument("--extract", action="store_true",
                        help="write the source files to ./src/ and exit, so they "
                             "can be read as ordinary Python")
    args = parser.parse_args()

    count = unpack()
    if args.extract:
        print(f"wrote {count} source files into ./src/\n")
        for relative in sorted(FILES):
            print(f"    {relative}")
        print("\nThese are ordinary Python files. Open them in any editor.")
        return 0

    print(f"unpacking {count} source files into ./src/")
    require_dependencies()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import yaml

    print(f"jax {jax.__version__} | devices {jax.devices()} | "
          f"float64 active: {jnp.zeros(1).dtype == jnp.float64}")

    check_physics()
    if args.check:
        return 0

    from src.config import parse_config
    from src.training.loop import train
    from src.evaluation.accuracy import evaluate
    from src.physics import analytic
    from src.predict import TrainedModel

    config = parse_config(yaml.safe_load(build_config(args.full)))
    print(f"\n--- TRAINING: {config.name} ---")
    print(f"  case {config.case}, Re_h = {config.physics.reynolds:g}, "
          f"{config.physics.formulation}")
    print(f"  entrance length L_e/h = {config.domain.entrance_length_over_h:.3f} "
          f"-> derived domain L* = {config.domain.length:.3f}")
    if not args.full:
        print("  QUICK DEMO schedule. Use --full for the accuracy targets.")

    result = train(config, verbose=True)
    metrics = evaluate(config, result.model, result.params)

    print("\n--- ACCURACY vs THE ANALYTIC SOLUTION ---")
    def row(label, key, target):
        value = metrics.get(key)
        if value is None:
            return
        verdict = "PASS" if value <= target else "FAIL"
        print(f"  {label:<34}{value:.4e}   target <= {target:.0e}   {verdict}")
    row("relative L2 error on u", "u_relative_l2", 1e-5)
    row("max |v*|", "v_max_abs", 1e-5)
    row("mass conservation drift D", "flux_conservation_relative_error", 1e-6)
    if "p_relative_l2" in metrics:
        print(f"  {'relative L2 error on p':<34}{metrics['p_relative_l2']:.4e}")

    print("\n  DIAGNOSTICS (not accuracy claims)")
    final = result.diagnostics.get("loss_after_lbfgs",
                                   result.diagnostics.get("loss_after_adam"))
    print(f"    final loss                      {final:.4e}")
    print(f"    parameters selected from        {result.diagnostics.get('selected_from')}")
    if not args.full:
        print("\n  NOTE: the quick schedule does not reach the targets. That is expected.")
        print("        Re-run with --full for the real numbers.")

    print("\n--- USING THE TRAINED MODEL ---")
    solution = TrainedModel(config, result.params, result.model, run_directory=None)
    print(f"  {solution.summary()}")
    for x, y in ((3.0, 0.5), (1.0, 0.0), (5.0, -0.75)):
        if x > config.domain.length:
            continue
        u, v = solution.velocity(x, y)
        p = solution.pressure(x, y)
        print(f"    at (x*={x:g}, y*={y:+g}):  u* = {u:+.9f}  (exact {1 - y**2:+.9f})"
              f"   v* = {v:+.9f}   p* = {p:+.9f}")
    print(f"    flux at x* = 3            :  {solution.flux(3.0):.12f}   "
          f"(exact {4/3:.12f})")

    try:
        solution.velocity(config.domain.length * 2, 0.0)
    except ValueError:
        print("    evaluating outside the trained domain is refused, correctly")

    from jax.flatten_util import ravel_pytree
    flat, _ = ravel_pytree(result.params)
    np.savez_compressed("trained_model.npz", flat=np.asarray(flat, dtype=np.float64))
    print(f"\n  saved {flat.size} trained parameters to trained_model.npz")

    print("\n--- TO USE IT YOURSELF ---")
    print(textwrap.dedent("""
        from src.predict import TrainedModel
        # (re-train, or load runs/<id> with src.predict.load_run)
        u, v = solution.velocity(3.0, 0.5)     # velocity anywhere in the channel
        y, u, v = solution.profile(x=3.0)      # a cross-channel profile
        x, y, u, v = solution.grid(201, 101)   # the whole field
    """).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
