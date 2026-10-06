"""Network 2: the pressure-drop PINN on one bifurcation.

Steady incompressible Navier-Stokes on the polygon of bifurcation/geometry.py:
the problem bifurcation/reference.py solves by finite elements, with the SAME
boundary conditions, so the two can be compared point for point.

    inlet (segment D-A)   u, v = the inlet profile: the developed parabola, or
                          Network 1's flow at a station of its channel. Soft:
                          a loss term, as in Network 1.
    walls                 u = v = 0 EXACTLY: u = D(x) N_u, v = D(x) N_v.
    outlets               nu du/dn - p n = 0 (do-nothing, soft): equal outlet
                          pressures, the flow chooses its split. Or, with a
                          prescribed share, on outlet 1
                          nu du/dn - p n = -lam n, lam a trained scalar, and
                          Q1 = share * Q_in as a loss term: the PINN form of
                          the reference's flux-defective outlet.
    pressure              p = N_p, free: the outlet conditions fix its level.

Two formulations (model.formulation):
    velocity_pressure  u = D N_u, v = D N_v; continuity, and a prescribed
                       share, are loss terms. The first run lost 6.3% of the
                       mass (daughter 2 delivered 0.580 of its 0.667): soft
                       continuity drifts, as Phase 1's Case B did.
    stream_pressure    u = d(psi)/dy, v = -d(psi)/dx, psi = G + D^2 N_psi.
                       G takes a constant value on each of the three wall
                       chains (0 on daughter 2's outer wall and the parent's
                       lower wall, Q_in on daughter 1's outer wall and the
                       parent's upper wall, Q2 on the septum) and has zero
                       normal derivative there: a transfinite (Shepard)
                       interpolation G = sum_k psi_k prod_{j != k} D_j^mu /
                       sum_k prod_{j != k} D_j^mu, mu = model.blend_exponent
                       >= 2 (Sukumar & Srivastava 2022, sec. 6). Mass is then
                       conserved and Q2 is daughter 2's flux EXACTLY: set to
                       (1 - share) Q_in, or trained when the split is free.

Near a reflex corner every residual grows like 1/r (see `corner_points`);
weights.corner_radius rho > 0 multiplies the interior residuals by
min(1, r / rho), so they stay bounded and cannot dominate the fit (in the
first run, 0.5% of the points held 47% of the loss).

D is an approximate distance to the walls built with R-functions (Sukumar &
Srivastava 2022, CMAME 389, 114333): for each wall segment the trimmed
distance phi_i, which vanishes on that segment and nowhere else, then
D = (sum_i phi_i^-m)^(-1/m), the R-equivalence of order m. D is zero on every
wall, positive inside, normalised to first order (D ~ distance near a wall),
and smooth away from the corners.

The network, the equations and the Levenberg-Marquardt machinery are Network
1's (developing_flow.pinn: init_network, apply_network, point_derivatives,
batched_derivatives, momentum_x/y, continuity, gauss_newton,
levenberg_marquardt_step), so both networks rest on the same tested code.

This module is stage 3a: one fixed geometry, scored against the finite-element
solution of the same problem by relative L2 error integrated on the reference's
own mesh. Stage 3b makes the design parameters (split, angles) inputs.
"""

from __future__ import annotations

import jax

# float64 before any array exists (CLAUDE.md).
jax.config.update("jax_enable_x64", True)

import argparse  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
from typing import NamedTuple  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import yaml  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from bifurcation.geometry import Geometry, GeometryError  # noqa: E402
from developing_flow import pinn  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
NETWORK1_RUNS = HERE.parent / "developing_flow" / "runs"
ConfigError = pinn.ConfigError

#: Points per chunk when J^T J is accumulated (memory only; results agree to round-off).
LM_CHUNK = 512

#: How far inside an opening its measuring line lies (see opening_measures).
LINE_INSET = 1e-9


# --------------------------------------------------------------------------- #
# Configuration: every key required, unknown keys refused
# --------------------------------------------------------------------------- #

CHOICES = {
    "physics.pressure_scaling": ("inertial", "viscous"),
    "outlets.split": ("free", "prescribed"),
    "inlet.profile": ("parabola", "network1"),
    "model.activation": ("tanh", "silu"),
    "model.distance_order": (1, 2),
    "model.formulation": ("velocity_pressure", "stream_pressure"),
}

SCHEMA = {
    "name": str,
    "seed": int,
    "physics": {"reynolds": float, "pressure_scaling": str},
    "geometry": {"split": float, "angle1": float, "angle2": float, "parent_length": float,
                 "daughter_length": float, "min_width": float},
    "outlets": {"split": str},
    "inlet": {"profile": str},
    "model": {"width": int, "depth": int, "activation": str, "init_gain": float,
              "input_scale": float, "distance_order": int, "formulation": str,
              "wall_extension": float},
    "sampling": {"n_interior": int, "corner_fraction": float, "corner_radius": float,
                 "corner_exclusion": float, "n_inlet": int, "n_outlet": int,
                 "n_validation": int},
    "weights": {"momentum_x": float, "momentum_y": float, "inlet": float, "outlet": float,
                "corner_radius": float},
    "training": {
        "adam": {"steps": int, "learning_rate": float, "decay_rate": float, "decay_steps": int,
                 "resample_every": int},
        "lbfgs": {"cycles": int, "iterations_per_cycle": int, "gtol": float},
        "lm": {"iterations": int, "damping": float, "resample_every": int, "average": int},
        "check_every": int,
    },
    "evaluation": {"max_area": float, "corner_levels": int, "corner_radius": float,
                   "n_line": int},
}

#: Keys required under one choice and refused under the others: a key with no
#: effect is a setting nobody can see is being ignored.
CONDITIONAL = {
    # outlets.split = prescribed
    ("outlets", "outlet1_share"): float,
    # prescribed, or stream_pressure (Q_in sets the stream function's values)
    ("sampling", "n_flux"): int,
    # prescribed and velocity_pressure
    ("weights", "flux"): float,
    # velocity_pressure (continuity holds identically for a stream function)
    ("weights", "continuity"): float,
    # stream_pressure
    ("model", "blend_exponent"): float,
    # inlet.profile = network1
    ("inlet", "run"): str,
    ("inlet", "station"): float,
}


@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    seed: int
    reynolds: float
    pressure_scaling: str
    split: float
    angle1: float
    angle2: float
    parent_length: float
    daughter_length: float
    min_width: float
    outlet_split: str
    outlet1_share: float | None
    inlet_profile: str
    inlet_run: str | None
    inlet_station: float | None
    width: int
    depth: int
    activation: str
    init_gain: float
    input_scale: float
    distance_order: int
    wall_extension: float
    formulation: str
    blend_exponent: float | None
    corner_weight_radius: float
    n_interior: int
    corner_fraction: float
    corner_radius: float
    corner_exclusion: float
    n_inlet: int
    n_outlet: int
    n_validation: int
    n_flux: int | None
    weights: dict
    adam_steps: int
    adam_learning_rate: float
    adam_decay_rate: float
    adam_decay_steps: int
    adam_resample_every: int
    lbfgs_cycles: int
    lbfgs_iterations: int
    lbfgs_gtol: float
    lm_iterations: int
    lm_damping: float
    lm_resample_every: int
    lm_average: int
    check_every: int
    fe_max_area: float
    fe_corner_levels: int
    fe_corner_radius: float
    n_line: int

    @property
    def prescribed(self) -> bool:
        return self.outlet_split == "prescribed"

    @property
    def stream(self) -> bool:
        return self.formulation == "stream_pressure"

    def geometry(self) -> Geometry:
        return Geometry.from_split(self.split, self.angle1, self.angle2, self.parent_length,
                                   self.daughter_length, self.min_width)


def parse_config(document) -> Config:
    document = copy.deepcopy(document)
    # Pull the conditional keys out first, so the fixed schema sees exactly its own.
    conditional = {}
    for (section, key), expected in CONDITIONAL.items():
        node = document.get(section)
        if isinstance(node, dict) and key in node:
            conditional[(section, key)] = node.pop(key)
    pinn._check(document, SCHEMA)
    for where, allowed in CHOICES.items():
        section, key = where.split(".")
        if document[section][key] not in allowed:
            raise ConfigError(f"{where} must be one of {list(allowed)}, "
                              f"got {document[section][key]!r}")
    prescribed = document["outlets"]["split"] == "prescribed"
    network1 = document["inlet"]["profile"] == "network1"
    stream = document["model"]["formulation"] == "stream_pressure"
    needed = {("outlets", "outlet1_share"): prescribed,
              ("sampling", "n_flux"): prescribed or stream,
              ("weights", "flux"): prescribed and not stream,
              ("weights", "continuity"): not stream,
              ("model", "blend_exponent"): stream,
              ("inlet", "run"): network1, ("inlet", "station"): network1}
    for (section, key), required in needed.items():
        where = f"{section}.{key}"
        if required and (section, key) not in conditional:
            raise ConfigError(f"missing required key {where!r}")
        if not required and (section, key) in conditional:
            raise ConfigError(f"{where} has no effect under this config and is refused")
        if required:
            pinn._check_value(where, conditional[(section, key)], CONDITIONAL[(section, key)])

    share = conditional.get(("outlets", "outlet1_share"))
    if share is not None and not 0.0 < share < 1.0:
        raise ConfigError(f"outlets.outlet1_share must lie in (0, 1), got {share}")
    p, g, m, s, t = (document[k] for k in ("physics", "geometry", "model", "sampling",
                                           "training"))
    if not 0.0 <= s["corner_fraction"] < 1.0:
        raise ConfigError("sampling.corner_fraction must lie in [0, 1)")
    if not m["wall_extension"] >= 0.0:
        raise ConfigError("model.wall_extension must be >= 0 (0: walls end at the openings)")
    if not 0.0 < s["corner_exclusion"] < s["corner_radius"]:
        raise ConfigError("sampling.corner_exclusion must lie in (0, corner_radius)")
    for where, value in (("model.width", m["width"]), ("model.depth", m["depth"]),
                         ("sampling.n_interior", s["n_interior"]),
                         ("sampling.n_inlet", s["n_inlet"]), ("sampling.n_outlet", s["n_outlet"]),
                         ("sampling.n_validation", s["n_validation"]),
                         ("training.check_every", t["check_every"]),
                         ("training.adam.resample_every", t["adam"]["resample_every"]),
                         ("training.lm.resample_every", t["lm"]["resample_every"]),
                         ("evaluation.n_line", document["evaluation"]["n_line"])):
        if value < 1:
            raise ConfigError(f"{where} must be at least 1, got {value}")
    for where, value in (("model.input_scale", m["input_scale"]),
                         ("sampling.corner_radius", s["corner_radius"]),
                         ("physics.reynolds", p["reynolds"])):
        if not value > 0.0:
            raise ConfigError(f"{where} must be positive, got {value}")
    weights = dict(document["weights"])
    corner_weight_radius = float(weights.pop("corner_radius"))
    if corner_weight_radius < 0.0:
        raise ConfigError("weights.corner_radius must be >= 0 (0: no corner weighting)")
    for key in ("flux", "continuity"):
        if ("weights", key) in conditional:
            weights[key] = conditional[("weights", key)]
    blend = conditional.get(("model", "blend_exponent"))
    if blend is not None and not blend >= 2.0:
        raise ConfigError("model.blend_exponent must be >= 2: below it the blend's normal "
                          "derivative at a wall is not zero, and no-slip fails")
    config = Config(
        name=document["name"], seed=document["seed"],
        reynolds=float(p["reynolds"]), pressure_scaling=p["pressure_scaling"],
        split=float(g["split"]), angle1=float(g["angle1"]), angle2=float(g["angle2"]),
        parent_length=float(g["parent_length"]), daughter_length=float(g["daughter_length"]),
        min_width=float(g["min_width"]),
        outlet_split=document["outlets"]["split"],
        outlet1_share=None if share is None else float(share),
        inlet_profile=document["inlet"]["profile"],
        inlet_run=conditional.get(("inlet", "run")),
        inlet_station=(None if not network1 else float(conditional[("inlet", "station")])),
        width=m["width"], depth=m["depth"], activation=m["activation"],
        init_gain=float(m["init_gain"]), input_scale=float(m["input_scale"]),
        distance_order=m["distance_order"], wall_extension=float(m["wall_extension"]),
        formulation=m["formulation"],
        blend_exponent=None if blend is None else float(blend),
        corner_weight_radius=corner_weight_radius,
        n_interior=s["n_interior"], corner_fraction=float(s["corner_fraction"]),
        corner_radius=float(s["corner_radius"]),
        corner_exclusion=float(s["corner_exclusion"]), n_inlet=s["n_inlet"],
        n_outlet=s["n_outlet"],
        n_validation=s["n_validation"], n_flux=conditional.get(("sampling", "n_flux")),
        weights={k: float(v) for k, v in weights.items()},
        adam_steps=t["adam"]["steps"], adam_learning_rate=float(t["adam"]["learning_rate"]),
        adam_decay_rate=float(t["adam"]["decay_rate"]), adam_decay_steps=t["adam"]["decay_steps"],
        adam_resample_every=t["adam"]["resample_every"],
        lbfgs_cycles=t["lbfgs"]["cycles"], lbfgs_iterations=t["lbfgs"]["iterations_per_cycle"],
        lbfgs_gtol=float(t["lbfgs"]["gtol"]),
        lm_iterations=t["lm"]["iterations"], lm_damping=float(t["lm"]["damping"]),
        lm_resample_every=t["lm"]["resample_every"], lm_average=t["lm"]["average"],
        check_every=t["check_every"],
        fe_max_area=float(document["evaluation"]["max_area"]),
        fe_corner_levels=document["evaluation"]["corner_levels"],
        fe_corner_radius=float(document["evaluation"]["corner_radius"]),
        n_line=document["evaluation"]["n_line"],
    )
    config.geometry()          # refuses an impossible bifurcation now, not after training
    return config


def load_config(path) -> Config:
    with open(path) as handle:
        return parse_config(yaml.safe_load(handle))


# --------------------------------------------------------------------------- #
# The inlet profile
# --------------------------------------------------------------------------- #


def inlet_function(config: Config):
    """inlet(y) -> (u, v) numpy arrays at x* = 0, as bifurcation.reference takes it."""
    from bifurcation import reference
    if config.inlet_profile == "parabola":
        return reference.parabola_inlet
    return reference.network_inlet(NETWORK1_RUNS / config.inlet_run, config.inlet_station)


def inlet_flux(inlet, n):
    """Q_in = the integral of u over the inlet, by n-point Gauss-Legendre."""
    y, w = np.polynomial.legendre.leggauss(n)
    return float(np.dot(w, inlet(y)[0]))


# --------------------------------------------------------------------------- #
# The wall distance and the field
# --------------------------------------------------------------------------- #


def segment_distance(xy, a, b):
    """Trimmed distance to the segment a-b (Sukumar & Srivastava 2022, sec. 5.1):

        f = signed distance to the line through a and b
        t = ((L/2)^2 - |x - centre|^2) / L    (positive inside the circle on a-b)
        phi = sqrt(f^2 + ((sqrt(t^2 + f^4) - t) / 2)^2)

    phi vanishes on the segment and nowhere else: off its ends (t < 0) the
    second term stays positive. (sqrt(t^2 + f^4) - t) / 2 is evaluated as
    f^4 / (2 (sqrt(t^2 + f^4) + t)) where t > 0, which avoids cancelling.
    """
    d = b - a
    length = jnp.sqrt(d @ d)
    f = ((xy[0] - a[0]) * d[1] - (xy[1] - a[1]) * d[0]) / length
    r = xy - 0.5 * (a + b)
    t = (0.25 * length**2 - r @ r) / length
    root = jnp.sqrt(t**2 + f**4)
    positive = t > 0.0
    trim = jnp.where(positive, f**4 / (2.0 * jnp.where(positive, root + t, 1.0)),
                     0.5 * (root - t))
    return jnp.sqrt(f**2 + trim**2)


def wall_segments(geometry: Geometry, extension):
    """(S, 2, 2): the walls of the polygon as one array of segments: the three
    chains of `wall_chains`, merged and extended as it describes."""
    return np.concatenate(wall_chains(geometry, extension))


def wall_distance(xy, segments, order):
    """D = (sum_i phi_i^-m)^(-1/m): zero on every wall, positive inside."""
    phis = jnp.stack([segment_distance(xy, segments[i, 0], segments[i, 1])
                      for i in range(segments.shape[0])])
    return jnp.sum(phis ** (-order)) ** (-1.0 / order)


def wall_chains(geometry: Geometry, extension):
    """The walls as three chains, in polygon order from the inlet: daughter 2's
    outer wall with the parent's lower wall, the septum, daughter 1's outer
    wall with the parent's upper wall. Openings separate them.

    Consecutive collinear edges are merged into one segment: where a straight
    daughter continues the parent's wall, two segments meeting at 180 degrees
    give the R-function composition a Laplacian ~ 1/r at the joint, so the
    residual of u = D N grows like 1/r there (measured) although the flow is
    smooth. One segment has no joint.

    Each chain is then extended by `extension` past both of its ends, which
    are at openings (the inlet, an outlet), along its end segments, outside
    the domain. The distance functions are smooth along a segment and not at
    its end, and the stream function needs their THIRD derivatives: with no
    extension the residual near an outlet's outer corner rose to ~10 at
    r = 1e-3 for a trained network, and one point there held 86% of a
    held-out loss. Extended, no wall end lies on the domain's closure except
    at the reflex corners. 0 = no extension.
    """
    edges = geometry.edges()
    n = len(edges)
    first = next(i for i in range(n) if edges[i][2] == "inlet") + 1
    chains, current = [], None
    for k in range(n):
        a, b, kind = edges[(first + k) % n]
        if kind != "wall":
            current = None
            continue
        if current is None:
            current = []
            chains.append(current)
        if current:
            d0, d1 = current[-1][1] - current[-1][0], b - a
            cross = d0[0] * d1[1] - d0[1] * d1[0]
            if (abs(cross) <= 1e-12 * np.linalg.norm(d0) * np.linalg.norm(d1)
                    and np.dot(d0, d1) > 0.0):
                current[-1] = [current[-1][0], b.copy()]
                continue
        current.append([a.copy(), b.copy()])
    if len(chains) != 3:
        raise AssertionError(f"expected 3 wall chains, found {len(chains)}")
    chains = [np.array(chain) for chain in chains]
    if extension > 0.0:
        for chain in chains:
            for which, (start, end) in (("start", (chain[0][1], chain[0][0])),
                                        ("end", (chain[-1][0], chain[-1][1]))):
                direction = (end - start) / np.linalg.norm(end - start)
                tip = end + extension * direction
                along = end + np.linspace(0.0, 1.0, 64)[1:, None] * (tip - end)
                if geometry.contains(along).any():
                    raise GeometryError(
                        f"extending a wall by {extension:g} past an opening enters the domain: "
                        "use a shorter model.wall_extension")
                if which == "start":
                    chain[0][0] = tip
                else:
                    chain[-1][1] = tip
    return chains


def init_params(config: Config, key):
    n_out = 2 if config.stream else 3
    params = {"net": pinn.init_network(key, 2, config.width, config.depth, n_out,
                                       config.init_gain)}
    if config.stream and not config.prescribed:
        # daughter 2's share of the flow, trained: the septum's stream function
        # value is share2 * Q_in, and the flow chooses its own split
        params["share2"] = jnp.asarray(0.5, dtype=jnp.float64)
    if config.prescribed:
        # outlet 1's pressure above outlet 2's: the PINN's version of the
        # reference's Lagrange multiplier
        params["offset"] = jnp.zeros((), dtype=jnp.float64)
    return params


def make_field(config: Config, geometry: Geometry, q_in):
    """field(params, xy) -> (u, v, p) at one point. q_in, the inflow, sets the
    stream function's wall values (unused by velocity_pressure)."""
    segments = jnp.asarray(wall_segments(geometry, config.wall_extension))
    lo, hi = geometry.bounding_box()
    centre = jnp.asarray(0.5 * (lo + hi))
    scale = config.input_scale
    activation, order = config.activation, config.distance_order

    if not config.stream:
        def field(params, xy):
            raw = pinn.apply_network(params["net"], (xy - centre) / scale, activation)
            distance = wall_distance(xy, segments, order)
            return jnp.stack([distance * raw[0], distance * raw[1], raw[2]])
        return field

    chains = [jnp.asarray(chain) for chain in wall_chains(geometry, config.wall_extension)]
    mu = config.blend_exponent
    fixed_share2 = None if not config.prescribed else 1.0 - config.outlet1_share

    def psi(params, point):
        raw = pinn.apply_network(params["net"], (point - centre) / scale, activation)
        share2 = fixed_share2 if fixed_share2 is not None else params["share2"]
        values = (0.0, share2 * q_in, q_in)             # lower chain, septum, upper chain
        d = [wall_distance(point, chain, order) ** mu for chain in chains]
        # Shepard weights times prod_j D_j^mu: w_k = prod_{j != k} D_j^mu, so
        # nothing is divided by zero on a wall, and on chain k only w_k is nonzero
        w = (d[1] * d[2], d[0] * d[2], d[0] * d[1])
        blend = (values[1] * w[1] + values[2] * w[2]) / (w[0] + w[1] + w[2])
        return blend + wall_distance(point, segments, order) ** 2 * raw[0], raw

    def field(params, xy):
        gradient, raw = jax.grad(lambda q: psi(params, q), has_aux=True)(xy)
        return jnp.stack([gradient[1], -gradient[0], raw[1]])

    field.psi = psi
    return field


# --------------------------------------------------------------------------- #
# Collocation points
# --------------------------------------------------------------------------- #


class Batch(NamedTuple):
    interior: np.ndarray    # (N, 2)
    inlet: np.ndarray       # (N_in, 4): x, y, and the profile's u, v there
    outlet1: np.ndarray     # (N_out, 2)
    outlet2: np.ndarray     # (N_out, 2)


def sample_triangles(rng, tris, n):
    """n points uniform over a set of triangles (area-weighted)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    which = rng.choice(len(tris), size=n, p=area / area.sum())
    s, t = rng.random(n), rng.random(n)
    flip = s + t > 1.0
    s[flip], t[flip] = 1.0 - s[flip], 1.0 - t[flip]
    return a[which] + s[:, None] * (b - a)[which] + t[:, None] * (c - a)[which]


def outside_corners(points, corners, exclusion):
    """True for points farther than `exclusion` from every corner."""
    if len(corners) == 0:
        return np.ones(len(points), dtype=bool)
    distance = np.linalg.norm(points[:, None, :] - corners[None, :, :], axis=2)
    return np.min(distance, axis=1) > exclusion


def uniform_points(rng, geometry: Geometry, n, exclusion):
    """n points uniform over the polygon, none within `exclusion` of a singular corner."""
    corners, tris = geometry.singular_corners(), geometry.triangles()
    out, need = [], n
    while need > 0:
        trial = sample_triangles(rng, tris, need + need // 8 + 16)
        keep = trial[outside_corners(trial, corners, exclusion)]
        out.append(keep[:need])
        need -= len(out[-1])
    return np.concatenate(out)[:n]


def corner_points(rng, geometry: Geometry, n, radius, exclusion):
    """n extra points around the singular corners (Geometry.singular_corners),
    uniform in AREA over the annulus exclusion < r < radius, inside the
    polygon. Near such a corner u ~ r^0.54 while the wall distance D ~ r, so
    no network makes the residual of u = D N bounded: it grows like 1/r
    (measured). Its square is not integrable with a 1/r point density (one
    draw scored 1.4e3 against a held-out 21), so no point comes closer than
    `exclusion` and the extra points add a constant density, not a 1/r one."""
    corners = geometry.singular_corners()
    if n == 0 or len(corners) == 0:
        return np.zeros((0, 2))
    out, need = [], n
    while need > 0:
        m = 2 * need + 16
        centre = corners[rng.integers(len(corners), size=m)]
        r = np.sqrt(exclusion**2 + (radius**2 - exclusion**2) * rng.random(m))
        angle = 2.0 * np.pi * rng.random(m)
        trial = centre + r[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        keep = trial[geometry.contains(trial) & outside_corners(trial, corners, exclusion)]
        out.append(keep[:need])
        need -= len(out[-1])
    return np.concatenate(out)[:n]


def sample_batch(rng, config: Config, geometry: Geometry, inlet, n_interior) -> Batch:
    n_corner = int(round(n_interior * config.corner_fraction))
    if len(geometry.singular_corners()) == 0:
        n_corner = 0
    interior = np.concatenate([
        uniform_points(rng, geometry, n_interior - n_corner, config.corner_exclusion),
        corner_points(rng, geometry, n_corner, config.corner_radius, config.corner_exclusion)])
    # inlet y crowds toward the corners, as in Network 1
    y = np.sin(0.5 * np.pi * rng.uniform(-1.0, 1.0, config.n_inlet))
    u, v = inlet(y)
    outlet1, _ = geometry.sample_boundary(rng, "outlet1", config.n_outlet)
    outlet2, _ = geometry.sample_boundary(rng, "outlet2", config.n_outlet)
    return Batch(interior=interior,
                 inlet=np.stack([np.zeros_like(y), y, u, v], axis=1),
                 outlet1=outlet1, outlet2=outlet2)


# --------------------------------------------------------------------------- #
# Residuals: one point at a time (Levenberg-Marquardt) and batched (Adam, L-BFGS)
# --------------------------------------------------------------------------- #


def outlet_normal(geometry: Geometry, kind):
    return geometry.line_quadrature(kind, 1)[2]


def make_residuals(config: Config, geometry: Geometry, field, q_in):
    """(groups, batched_loss).

    groups(batch) -> [(fn, points)], fn(params, row, n) -> the scaled residuals
    at one point of a group of n, so that the sum of their squares over a batch
    is the weighted loss (pinn.gauss_newton's contract).
    batched_loss(params, batch) -> the same loss, with the interior derivatives
    taken by pinn.batched_derivatives (faster under reverse mode).
    """
    reynolds, scaling = config.reynolds, config.pressure_scaling
    _, nu = pinn.scaling_coefficients(reynolds, scaling)
    w = config.weights
    normal1 = jnp.asarray(outlet_normal(geometry, "outlet1"))
    normal2 = jnp.asarray(outlet_normal(geometry, "outlet2"))
    corners = jnp.asarray(geometry.singular_corners())
    rho = config.corner_weight_radius

    def corner_weight(xy):
        """min(1, r / rho), r the distance to the nearest singular corner."""
        if rho == 0.0 or corners.shape[0] == 0:
            return 1.0
        return jnp.minimum(1.0, jnp.min(jnp.linalg.norm(corners - xy, axis=1)) / rho)

    def interior_values(d, n, weight):
        out = [jnp.sqrt(w["momentum_x"] / n) * pinn.momentum_x(d, reynolds, scaling),
               jnp.sqrt(w["momentum_y"] / n) * pinn.momentum_y(d, reynolds, scaling)]
        if not config.stream:
            out.append(jnp.sqrt(w["continuity"] / n) * pinn.continuity(d))
        return [weight * r for r in out]

    def interior(params, xy, n):
        d = pinn.point_derivatives(lambda q: field(params, q), xy)
        return jnp.stack(interior_values(d, n, corner_weight(xy)))

    def inlet(params, row, n):
        value = field(params, row[:2])
        return jnp.sqrt(w["inlet"] / n) * jnp.stack([value[0] - row[2], value[1] - row[3]])

    def traction(params, xy, normal, offset):
        """nu du/dn - p n (+ lam n on a prescribed outlet 1): zero at a do-nothing outlet."""
        value, d_dn = jax.jvp(lambda q: field(params, q), (xy,), (normal,))
        return nu * d_dn[:2] - value[2] * normal + offset * normal

    def outlet1(params, xy, n):
        offset = params["offset"] if config.prescribed else 0.0
        return jnp.sqrt(w["outlet"] / n) * traction(params, xy, normal1, offset)

    def outlet2(params, xy, n):
        return jnp.sqrt(w["outlet"] / n) * traction(params, xy, normal2, 0.0)

    if config.prescribed and not config.stream:
        nodes, weights, _ = geometry.line_quadrature("outlet1", config.n_flux)
        nodes, weights = jnp.asarray(nodes), jnp.asarray(weights)
        target = config.outlet1_share * q_in

        def flux(params, row, n):
            """ONE residual for the batch: Q1 - share * Q_in, by Gauss-Legendre
            on outlet 1. `row` is a placeholder (the group has one row)."""
            values = jax.vmap(lambda q: field(params, q))(nodes)
            q1 = jnp.sum(weights * (values[:, :2] @ normal1))
            return jnp.sqrt(w["flux"]) * jnp.atleast_1d(q1 - target)

    placeholder = np.zeros((1, 1))

    def groups(batch: Batch):
        out = [(interior, batch.interior), (inlet, batch.inlet),
               (outlet1, batch.outlet1), (outlet2, batch.outlet2)]
        if config.prescribed and not config.stream:
            out.append((flux, placeholder))
        return out

    def batched_loss(params, batch: Batch):
        n = batch.interior.shape[0]
        d = pinn.batched_derivatives(lambda q: field(params, q), batch.interior)
        weight = jax.vmap(corner_weight)(jnp.asarray(batch.interior))
        total = sum(jnp.sum(r**2) for r in interior_values(d, n, weight))
        for fn, points in groups(batch)[1:]:
            m = points.shape[0]
            total = total + jnp.sum(jax.vmap(lambda row: fn(params, row, m))(points) ** 2)
        return total

    return groups, batched_loss


# --------------------------------------------------------------------------- #
# Training: Adam, L-BFGS cycles, Levenberg-Marquardt; resumable
# --------------------------------------------------------------------------- #

TRAINING_STATE_VERSION = 1


def _fingerprint(config: Config) -> dict:
    text = json.dumps(dataclasses.asdict(config), sort_keys=True, default=repr)
    code = hashlib.sha256()
    for path in (pathlib.Path(__file__), pathlib.Path(pinn.__file__), HERE / "geometry.py"):
        code.update(path.read_bytes())
    import scipy
    code.update(f"jax {jax.__version__} optax {optax.__version__} scipy {scipy.__version__} "
                f"numpy {np.__version__}".encode())
    return {"config": hashlib.sha256(text.encode()).hexdigest(), "code": code.hexdigest()}


def _streams(seed):
    """(model key, training-draw generator, held-out generator) from one seed."""
    children = np.random.SeedSequence(seed).spawn(2)
    return (jax.random.PRNGKey(seed), np.random.default_rng(children[0]),
            np.random.default_rng(children[1]))


def train(config: Config, verbose=True, checkpoint=None, resume=False):
    """Returns (best params, history, info). With `checkpoint`, the state is
    written at every boundary (Adam check, L-BFGS cycle, LM step) and `resume`
    continues from it; the draws are numpy generators whose state is saved, so
    a resumed run draws the same points as an uninterrupted one."""
    geometry = config.geometry()
    inlet = inlet_function(config)
    q_in = inlet_flux(inlet, config.n_flux) if config.n_flux is not None else None
    field = make_field(config, geometry, q_in)
    groups, batched_loss = make_residuals(config, geometry, field, q_in)
    model_key, draws, heldout_rng = _streams(config.seed)
    params = init_params(config, model_key)

    def draw():
        return sample_batch(draws, config, geometry, inlet, config.n_interior)

    heldout = sample_batch(heldout_rng, config, geometry, inlet, config.n_validation)

    def describe(p):
        """What the log shows beside the losses: the trained scalars."""
        out = {}
        if config.prescribed:
            out["offset"] = float(p["offset"])
        if config.stream and not config.prescribed:
            out["share2"] = float(p["share2"])
        return out

    best, history, info = optimise(config, params, groups, batched_loss, draws, draw, heldout,
                                   _fingerprint(config), describe, verbose, checkpoint, resume)
    info["inlet_flux"] = q_in
    return best, history, info


def optimise(config, params, groups, batched_loss, draws, draw, heldout, fingerprint, describe,
             verbose=True, checkpoint=None, resume=False):
    """The training loop Network 2 and its parametric version share: Adam with
    resampling, L-BFGS cycles on fresh draws, then Levenberg-Marquardt with
    fresh points every lm.resample_every steps and iterate averaging; the
    parameters with the lowest held-out loss are returned.

    config     the training settings (adam_*, lbfgs_*, lm_*, check_every)
    groups, batched_loss   pinn.gauss_newton's point groups and the same loss batched
    draws, draw   the numpy generator behind draw() and draw() itself: the
               generator's state goes into the checkpoint, so a resumed run
               draws the same points
    heldout    the held-out batch
    fingerprint   identifies config and code: a checkpoint from others is refused
    describe(params) -> {name: float}, logged beside the losses
    Returns (best params, history, info)."""
    if resume and checkpoint is None:
        raise ValueError("resume needs the checkpoint to resume from")
    flat, unflatten = ravel_pytree(params)
    normal_equations, lm_loss = pinn.gauss_newton(groups, unflatten, LM_CHUNK)

    def heldout_loss(vector):
        return lm_loss(vector, heldout)

    best = {"heldout": math.inf, "flat": flat, "where": "initial"}
    history = []
    lm_log = {"iterations": 0, "rejected": 0, "stalled": 0}
    started = time.time()
    position, saved, meta = ("start", 0), {}, {}
    if resume:
        with np.load(checkpoint) as stored:
            meta = json.loads(str(stored["meta"]))
            saved = {k: stored[k] for k in stored.files if k != "meta"}
        if meta["version"] != TRAINING_STATE_VERSION or meta["fingerprint"] != fingerprint:
            raise ValueError(f"cannot resume from {checkpoint}: it was written by another "
                             "config or other code")
        position = (meta["phase"], meta["done"])
        best = {"heldout": meta["best_heldout"], "where": meta["best_where"],
                "flat": jnp.asarray(saved["best_flat"])}
        history, lm_log = meta["history"], meta["lm_log"]
        started -= meta["elapsed"]
        draws.bit_generator.state = meta["draws"]
        flat = jnp.asarray(saved["flat"])
        if verbose:
            print(f"  resuming after {position[0]} {position[1]} "
                  f"({meta['elapsed'] / 60:.1f} min already trained)", flush=True)

    def save(phase, done, current, batch, extra_arrays=None, **extra):
        if checkpoint is None:
            return
        arrays = {"flat": np.asarray(current), "best_flat": np.asarray(best["flat"]),
                  **{f"batch_{k}": np.asarray(v) for k, v in batch._asdict().items()},
                  **(extra_arrays or {})}
        state = {"version": TRAINING_STATE_VERSION, "fingerprint": fingerprint, "phase": phase,
                 "done": done, "elapsed": time.time() - started, "best_heldout": best["heldout"],
                 "best_where": best["where"], "history": history, "lm_log": lm_log,
                 "draws": draws.bit_generator.state, **extra}
        temporary = pathlib.Path(checkpoint).with_suffix(".tmp.npz")
        np.savez_compressed(temporary, meta=np.array(json.dumps(state)), **arrays)
        temporary.replace(checkpoint)

    def saved_batch():
        return Batch(**{k: saved[f"batch_{k}"] for k in Batch._fields})

    def record(vector, batch, phase, iteration, average=None):
        loss = lm_loss(vector, batch)
        held = heldout_loss(vector)
        held_average = math.nan if average is None else heldout_loss(average)
        if not (math.isfinite(loss) and math.isfinite(held)
                and (average is None or math.isfinite(held_average))):
            raise FloatingPointError(f"non-finite loss at {phase}:{iteration}; training diverged")
        if held < best["heldout"]:
            best.update(heldout=held, flat=vector, where=f"{phase}:{iteration}")
        if held_average < best["heldout"]:
            best.update(heldout=held_average, flat=average, where=f"{phase}_average:{iteration}")
        row = {"phase": phase, "iteration": iteration, "train_loss": loss, "heldout_loss": held,
               "heldout_average": held_average, "seconds": time.time() - started}
        extras = describe(unflatten(vector))
        row.update(extras)
        history.append(row)
        if verbose:
            print(f"  {phase:>6} {iteration:>7d} {loss:>12.4e} {held:>12.4e}"
                  + ("" if average is None else f"   averaged {held_average:.4e}")
                  + "".join(f"   {k} {v:+.5f}" for k, v in extras.items()), flush=True)

    phases = ("start", "adam", "lbfgs", "lm")
    order = phases.index(position[0])

    # -- Adam ---------------------------------------------------------------- #
    if order <= phases.index("adam") and config.adam_steps > 0:
        schedule = optax.exponential_decay(config.adam_learning_rate, config.adam_decay_steps,
                                           config.adam_decay_rate)
        optimiser = optax.adam(schedule)
        p = unflatten(flat)
        state = optimiser.init(p)

        @jax.jit
        def adam_step(p, s, batch):
            loss, grads = jax.value_and_grad(batched_loss)(p, batch)
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
                batch = draw()
            p, state, _ = adam_step(p, state, batch)
            if (step + 1) % config.check_every == 0 or step + 1 == config.adam_steps:
                flat = ravel_pytree(p)[0]
                record(flat, batch, "adam", step + 1)
                save("adam", step + 1, flat, batch,
                     {f"adam_{i}": np.asarray(leaf)
                      for i, leaf in enumerate(jax.tree_util.tree_leaves(state))})
        flat = ravel_pytree(p)[0]

    # -- L-BFGS in cycles, each on a fresh draw ----------------------------- #
    if order <= phases.index("lbfgs"):
        value_and_grad = jax.jit(jax.value_and_grad(
            lambda vector, batch: batched_loss(unflatten(vector), batch)))
        first = position[1] if position[0] == "lbfgs" else 0
        for cycle in range(first, config.lbfgs_cycles):
            batch = draw()

            def fun(vector):
                value, grad = value_and_grad(jnp.asarray(vector), batch)
                return float(value), np.asarray(grad, dtype=np.float64)

            result = minimize(fun, np.asarray(flat, dtype=np.float64), jac=True,
                              method="L-BFGS-B",
                              options={"maxiter": config.lbfgs_iterations,
                                       "maxfun": 2 * config.lbfgs_iterations,
                                       "gtol": config.lbfgs_gtol, "ftol": 0.0, "maxcor": 100})
            flat = jnp.asarray(result.x)
            record(flat, batch, "lbfgs", cycle + 1)
            save("lbfgs", cycle + 1, flat, batch)

    # -- Levenberg-Marquardt, fresh points every resample_every steps -------- #
    if config.lm_iterations > 0:
        lam, nu_lm, first, window, batch = None, 2.0, 0, [], None
        if position[0] == "lm":
            lam, nu_lm, first, batch = meta["lam"], meta["nu"], position[1], saved_batch()
            if config.lm_average and "lm_window" in saved:
                window = [jnp.asarray(row) for row in saved["lm_window"]]
        for step in range(first, config.lm_iterations):
            if step % config.lm_resample_every == 0:
                batch = draw()
            A, g, loss_now = normal_equations(flat, batch)
            if lam is None:
                lam = config.lm_damping * float(jnp.mean(jnp.diag(A)))
            delta, _, lam, nu_lm, attempts = pinn.levenberg_marquardt_step(
                A, g, loss_now, lambda d: lm_loss(flat + jnp.asarray(d), batch), lam, nu_lm)
            lm_log["rejected"] += attempts - 1
            if delta is None:
                lm_log["stalled"] += 1
                batch = draw()
                lam, nu_lm = config.lm_damping * float(jnp.mean(jnp.diag(A))), 2.0
            else:
                flat = flat + jnp.asarray(delta)
            lm_log["iterations"] += 1
            average = None
            if config.lm_average:
                window = (window + [flat])[-config.lm_average:]
                if len(window) == config.lm_average:
                    average = jnp.mean(jnp.stack(window), axis=0)
            record(flat, batch, "lm", step + 1, average)
            save("lm", step + 1, flat, batch,
                 {"lm_window": np.stack([np.asarray(w) for w in window])} if window else None,
                 lam=lam, nu=nu_lm)

    if best["where"] == "initial":
        raise RuntimeError("no trained parameters beat the untrained network on the held-out "
                           "points; refusing to return the random initialisation")
    info = {"parameter_count": int(flat.size), "total_seconds": time.time() - started,
            "selected": best["where"], "selected_heldout_loss": best["heldout"],
            "levenberg_marquardt": lm_log}
    return unflatten(best["flat"]), history, info




# --------------------------------------------------------------------------- #
# Evaluation against the finite-element reference
# --------------------------------------------------------------------------- #


def predictor(field, params, chunk=8192):
    """model(x, y) -> (u, v, p) numpy arrays, any shape."""
    apply = jax.jit(jax.vmap(lambda xy: field(params, xy)))

    def model(x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   np.asarray(y, dtype=np.float64))
        pts = np.stack([x.ravel(), y.ravel()], axis=1)
        out = [np.asarray(apply(jnp.asarray(pts[i:i + chunk])))
               for i in range(0, len(pts), chunk)]
        out = np.concatenate(out) if out else np.zeros((0, 3))
        return tuple(out[:, k].reshape(x.shape) for k in range(3))

    return model


def opening_measures(model, geometry: Geometry, a, n_line):
    """Flux, energy flux and mean static pressure through each opening, by
    n_line-point Gauss-Legendre from point values alone, so the PINN and the
    finite-element solution are measured by ONE function (CLAUDE.md)."""
    out = {}
    for kind in ("inlet", "outlet1", "outlet2"):
        nodes, weights, normal = geometry.line_quadrature(kind, n_line)
        # 1e-9 inside: on the edge itself the reference's point search can
        # place a node outside the mesh by round-off. Both models see the same nodes.
        nodes = nodes - LINE_INSET * normal
        u, v, p = model(nodes[:, 0], nodes[:, 1])
        un = u * normal[0] + v * normal[1]
        out[kind] = {"flux": float(weights @ un),
                     "energy": float(weights @ ((p + 0.5 * a * (u**2 + v**2)) * un)),
                     "pressure": float(weights @ p) / float(weights.sum())}
    q_in, e_in = -out["inlet"]["flux"], -out["inlet"]["energy"]
    q1, q2 = out["outlet1"]["flux"], out["outlet2"]["flux"]
    return {"flux_in": q_in, "flux_out1": q1, "flux_out2": q2, "split_flux": q1 / q_in,
            "mass_imbalance": (q1 + q2 - q_in) / q_in,
            "total_pressure_drop": (e_in - out["outlet1"]["energy"]
                                    - out["outlet2"]["energy"]) / q_in,
            "static_pressure_in": out["inlet"]["pressure"],
            "static_pressure_out1": out["outlet1"]["pressure"],
            "static_pressure_out2": out["outlet2"]["pressure"]}


def relative_errors(solution, model, corners=None, exclusion=None):
    """Relative L2 errors of u, v, p against the finite-element solution,
    integrated on ITS mesh (CLAUDE.md: a grid sum over-weights the corners).
    With `corners` and `exclusion`, only over the part of the domain farther
    than `exclusion` from every corner (both norms restricted alike)."""
    import skfem as fem
    basis = solution.basis_u
    basis_p = fem.Basis(solution.mesh, solution.basis_p.elem, quadrature=basis.quadrature)
    x, y = basis.global_coordinates().value
    weights = basis.dx
    reference_u = basis.interpolate(solution.U).value
    reference_p = basis_p.interpolate(solution.P).value
    u, v, p = model(x, y)
    if corners is not None:
        keep = outside_corners(np.stack([x.ravel(), y.ravel()], axis=1), corners,
                               exclusion).reshape(x.shape)
        weights = weights * keep
    out = {}
    for name, mine, ref in (("u", u, reference_u[0]), ("v", v, reference_u[1]),
                            ("p", p, reference_p)):
        out[name] = float(np.sqrt(np.sum(weights * (mine - ref) ** 2)
                                  / np.sum(weights * ref**2)))
    return out


def evaluate(config: Config, params, verbose=True):
    """The PINN against the finite-element solution of the same problem."""
    from bifurcation import reference
    geometry = config.geometry()
    inlet = inlet_function(config)
    q_in = inlet_flux(inlet, config.n_flux) if config.n_flux is not None else None
    field = make_field(config, geometry, q_in)
    model = predictor(field, params)
    start = time.perf_counter()
    solution = reference.solve_bifurcation(
        geometry, config.reynolds, config.pressure_scaling, inlet,
        config.outlet1_share if config.prescribed else None, config.fe_max_area,
        config.fe_corner_levels, config.fe_corner_radius)
    fe_seconds = time.perf_counter() - start
    a, _ = pinn.scaling_coefficients(config.reynolds, config.pressure_scaling)
    mine = opening_measures(model, geometry, a, config.n_line)
    theirs = opening_measures(solution, geometry, a, config.n_line)
    errors = relative_errors(solution, model)
    away = relative_errors(solution, model, geometry.singular_corners(), config.corner_exclusion)
    metrics = {
        "relative_l2": errors,
        # where the PINN was trained: the corner discs hold the singularity
        "relative_l2_outside_corners": away,
        "pinn": mine, "reference": theirs,
        "reference_own": solution.measures(),
        "objective_relative_error": abs(mine["total_pressure_drop"]
                                        - theirs["total_pressure_drop"])
        / abs(theirs["total_pressure_drop"]),
        "split_error": abs(mine["split_flux"] - theirs["split_flux"]),
        "reference_seconds": fe_seconds,
    }
    if config.prescribed:
        metrics["offset"] = {"pinn": float(params["offset"]),
                             "reference": solution.info["multiplier"]}
    if verbose:
        print(f"  relative L2: u {errors['u']:.3e}  v {errors['v']:.3e}  p {errors['p']:.3e}")
        print(f"  outside the corner discs: u {away['u']:.3e}  v {away['v']:.3e}  "
              f"p {away['p']:.3e}")
        print(f"  total-pressure drop: PINN {mine['total_pressure_drop']:.6f}  "
              f"reference {theirs['total_pressure_drop']:.6f}  "
              f"(relative error {metrics['objective_relative_error']:.2e})")
        print(f"  flow split: PINN {mine['split_flux']:.6f}  reference {theirs['split_flux']:.6f}")
    return metrics, solution


# --------------------------------------------------------------------------- #
# Running and loading
# --------------------------------------------------------------------------- #


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def run(config_path, out=None, verbose=True, resume=False):
    config = load_config(config_path)
    directory = pathlib.Path(out) if out else HERE / "runs" / config.name
    if (directory / "params.npz").exists():
        raise FileExistsError(f"{directory} already holds a finished run")
    directory.mkdir(parents=True, exist_ok=True)
    stored = directory / "config.yaml"
    text = pathlib.Path(config_path).read_text()
    if resume:
        if stored.read_text() != text:
            raise ValueError("cannot resume: the stored config differs from the one given")
    else:
        if (directory / "checkpoint.npz").exists():
            raise FileExistsError(f"{directory} holds a checkpoint: resume it or remove it")
        stored.write_text(text)
    params, history, info = train(config, verbose, directory / "checkpoint.npz", resume)
    metrics, _ = evaluate(config, params, verbose)
    (directory / "history.json").write_text(json.dumps(_json_safe(history), indent=1))
    (directory / "metrics.json").write_text(json.dumps(_json_safe({**metrics, "training": info}),
                                                       indent=1))
    pinn.save_params(directory / "params.npz", params)
    return metrics


def load_run(directory):
    """(model, config, params) of a finished run; model(x, y) -> (u, v, p)."""
    directory = pathlib.Path(directory)
    config = load_config(directory / "config.yaml")
    template = init_params(config, _streams(config.seed)[0])
    flat_template, unflatten = ravel_pytree(template)
    flat = np.load(directory / "params.npz")["flat"]
    if flat.shape != flat_template.shape:
        raise ValueError("saved parameters do not match the config's network")
    params = unflatten(jnp.asarray(flat))
    q_in = (inlet_flux(inlet_function(config), config.n_flux) if config.n_flux is not None
            else None)
    model = predictor(make_field(config, config.geometry(), q_in), params)
    model.config = config
    return model, config, params


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run(args.config, args.out, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
