"""Strict configuration loading.

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
