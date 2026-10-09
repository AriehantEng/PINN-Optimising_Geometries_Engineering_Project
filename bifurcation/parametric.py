"""Network 2, parametric: the bifurcation PINN with the width split as an input.

One network for a whole family of bifurcations: the inputs are (x, y, s), with
s = w1 / w0 the width split, over a range [split_min, split_max] at fixed
angles, fixed share of the flow through each daughter, and w1 + w2 = w0.
Trained once, it gives the flow and the total-pressure drop at any split in
the range, and so the split with the lowest pressure drop: the design loop's
question, answered by the PINN instead of a finite-element optimisation.

The construction is the fixed-geometry Network 2's (bifurcation/flow.py),
stream-function formulation, with everything that depends on the geometry
made a function of s:

    walls          the wall segments are computed from s through the
                   differentiable polygon (geometry.vertices, xp=jax.numpy),
                   merged and extended past the openings exactly as
                   flow.wall_chains does for one geometry (tested: equal to
                   it at every split)
    stream function  psi = G(x; s) + D(x; s)^2 N_psi(x, y, s), so no-slip,
                   mass conservation and the prescribed share hold exactly
                   at every s
    corners        the singular corners (B, C and the septum tip, which moves
                   with s) are recomputed for each point's s, for the corner
                   weighting
    outlet 1's extra pressure   lam(s), a Chebyshev series in s whose
                   coefficients are trained (the fixed-geometry network's
                   single scalar)

Collocation: each draw takes `designs` splits, stratified over the range, and
`points_per_design` interior points in each one's own polygon, with its inlet
and outlet points; every row carries its s.

The training loop is the fixed-geometry network's (flow.optimise): Adam, L-BFGS
cycles, Levenberg-Marquardt with iterate averaging, held-out selection,
resumable. Evaluation solves the finite-element reference at a set of splits
(fields, pressure drop, offset), and compares the PINN's optimal split with the
finite-element optimum: the argmin error, and the regret, the extra pressure
drop paid for using the PINN's split.

    python -m bifurcation.parametric --config bifurcation/p_re10_3090_equal.yaml
"""

from __future__ import annotations

import jax

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
from scipy.optimize import minimize_scalar  # noqa: E402

from bifurcation import flow  # noqa: E402
from bifurcation import geometry as geo  # noqa: E402
from bifurcation.geometry import EDGE_KINDS, Geometry, split_bounds  # noqa: E402
from developing_flow import pinn  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ConfigError = pinn.ConfigError

CHOICES = {
    "physics.pressure_scaling": ("inertial", "viscous"),
    "inlet.profile": ("parabola", "network1"),
    "model.activation": ("tanh", "silu"),
    "model.distance_order": (1, 2),
}

SCHEMA = {
    "name": str,
    "seed": int,
    "physics": {"reynolds": float, "pressure_scaling": str},
    "geometry": {"angle1": float, "angle2": float, "parent_length": float,
                 "daughter_length": float, "min_width": float},
    "design": {"split_min": float, "split_max": float, "outlet1_share": float},
    "inlet": {"profile": str},
    "model": {"width": int, "depth": int, "activation": str, "init_gain": float,
              "input_scale": float, "distance_order": int, "wall_extension": float,
              "blend_exponent": float, "offset_degree": int},
    "sampling": {"designs": int, "points_per_design": int, "corner_fraction": float,
                 "corner_radius": float, "corner_exclusion": float, "inlet_per_design": int,
                 "outlet_per_design": int, "validation_designs": int, "n_flux": int},
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
                   "n_line": int, "splits": list, "search_xatol": float, "curve_points": int},
}

#: inlet.profile = network1 only
CONDITIONAL = {("inlet", "run"): str, ("inlet", "station"): float}

#: Splits at which the range is checked: every geometry must be valid and of
#: the same shape, so one differentiable construction serves the whole range.
RANGE_CHECKS = 33


@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    seed: int
    reynolds: float
    pressure_scaling: str
    angle1: float
    angle2: float
    parent_length: float
    daughter_length: float
    min_width: float
    split_min: float
    split_max: float
    outlet1_share: float
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
    blend_exponent: float
    offset_degree: int
    designs: int
    points_per_design: int
    corner_fraction: float
    corner_radius: float
    corner_exclusion: float
    inlet_per_design: int
    outlet_per_design: int
    validation_designs: int
    n_flux: int
    weights: dict
    corner_weight_radius: float
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
    eval_splits: tuple
    search_xatol: float
    curve_points: int

    def geometry(self, split) -> Geometry:
        return Geometry.from_split(float(split), self.angle1, self.angle2, self.parent_length,
                                   self.daughter_length, self.min_width)

    @property
    def split_mid(self):
        return 0.5 * (self.split_min + self.split_max)

    @property
    def split_half(self):
        return 0.5 * (self.split_max - self.split_min)


def parse_config(document) -> Config:
    document = copy.deepcopy(document)
    conditional = {}
    for (section, key) in CONDITIONAL:
        node = document.get(section)
        if isinstance(node, dict) and key in node:
            conditional[(section, key)] = node.pop(key)
    pinn._check(document, SCHEMA)
    for where, allowed in CHOICES.items():
        section, key = where.split(".")
        if document[section][key] not in allowed:
            raise ConfigError(f"{where} must be one of {list(allowed)}, "
                              f"got {document[section][key]!r}")
    network1 = document["inlet"]["profile"] == "network1"
    for (section, key), expected in CONDITIONAL.items():
        where = f"{section}.{key}"
        if network1 and (section, key) not in conditional:
            raise ConfigError(f"missing required key {where!r}")
        if not network1 and (section, key) in conditional:
            raise ConfigError(f"{where} has no effect under this config and is refused")
        if network1:
            pinn._check_value(where, conditional[(section, key)], expected)
    g, d, m, s, t, e = (document[k] for k in ("geometry", "design", "model", "sampling",
                                                 "training", "evaluation"))
    lo, hi = split_bounds(g["min_width"])
    if not lo <= d["split_min"] < d["split_max"] <= hi:
        raise ConfigError(f"design.split_min < split_max must lie in [{lo:g}, {hi:g}]")
    if not 0.0 < d["outlet1_share"] < 1.0:
        raise ConfigError("design.outlet1_share must lie in (0, 1)")
    if m["blend_exponent"] < 2.0:
        raise ConfigError("model.blend_exponent must be >= 2 (no-slip needs it)")
    if m["wall_extension"] < 0.0 or m["offset_degree"] < 0:
        raise ConfigError("model.wall_extension and model.offset_degree must be >= 0")
    if not 0.0 <= s["corner_fraction"] < 1.0:
        raise ConfigError("sampling.corner_fraction must lie in [0, 1)")
    if not 0.0 < s["corner_exclusion"] < s["corner_radius"]:
        raise ConfigError("sampling.corner_exclusion must lie in (0, corner_radius)")
    for key in ("designs", "points_per_design", "inlet_per_design", "outlet_per_design",
                "validation_designs", "n_flux"):
        if s[key] < 1:
            raise ConfigError(f"sampling.{key} must be at least 1")
    for split in e["splits"]:
        if not d["split_min"] <= split <= d["split_max"]:
            raise ConfigError(f"evaluation split {split} lies outside the design range")
    if e["curve_points"] < 3:
        raise ConfigError("evaluation.curve_points must be at least 3")
    weights = dict(document["weights"])
    corner_weight_radius = float(weights.pop("corner_radius"))
    if corner_weight_radius < 0.0:
        raise ConfigError("weights.corner_radius must be >= 0 (0: no corner weighting)")
    config = Config(
        name=document["name"], seed=document["seed"],
        reynolds=float(document["physics"]["reynolds"]),
        pressure_scaling=document["physics"]["pressure_scaling"],
        angle1=float(g["angle1"]), angle2=float(g["angle2"]),
        parent_length=float(g["parent_length"]), daughter_length=float(g["daughter_length"]),
        min_width=float(g["min_width"]),
        split_min=float(d["split_min"]), split_max=float(d["split_max"]),
        outlet1_share=float(d["outlet1_share"]),
        inlet_profile=document["inlet"]["profile"],
        inlet_run=conditional.get(("inlet", "run")),
        inlet_station=None if not network1 else float(conditional[("inlet", "station")]),
        width=m["width"], depth=m["depth"], activation=m["activation"],
        init_gain=float(m["init_gain"]), input_scale=float(m["input_scale"]),
        distance_order=m["distance_order"], wall_extension=float(m["wall_extension"]),
        blend_exponent=float(m["blend_exponent"]), offset_degree=m["offset_degree"],
        designs=s["designs"], points_per_design=s["points_per_design"],
        corner_fraction=float(s["corner_fraction"]), corner_radius=float(s["corner_radius"]),
        corner_exclusion=float(s["corner_exclusion"]), inlet_per_design=s["inlet_per_design"],
        outlet_per_design=s["outlet_per_design"], validation_designs=s["validation_designs"],
        n_flux=s["n_flux"],
        weights={k: float(v) for k, v in weights.items()},
        corner_weight_radius=corner_weight_radius,
        adam_steps=t["adam"]["steps"], adam_learning_rate=float(t["adam"]["learning_rate"]),
        adam_decay_rate=float(t["adam"]["decay_rate"]), adam_decay_steps=t["adam"]["decay_steps"],
        adam_resample_every=t["adam"]["resample_every"],
        lbfgs_cycles=t["lbfgs"]["cycles"], lbfgs_iterations=t["lbfgs"]["iterations_per_cycle"],
        lbfgs_gtol=float(t["lbfgs"]["gtol"]),
        lm_iterations=t["lm"]["iterations"], lm_damping=float(t["lm"]["damping"]),
        lm_resample_every=t["lm"]["resample_every"], lm_average=t["lm"]["average"],
        check_every=t["check_every"],
        fe_max_area=float(e["max_area"]), fe_corner_levels=e["corner_levels"],
        fe_corner_radius=float(e["corner_radius"]), n_line=e["n_line"],
        eval_splits=tuple(float(v) for v in e["splits"]), search_xatol=float(e["search_xatol"]),
        curve_points=e["curve_points"],
    )
    structure(config)               # refuses a range one construction cannot serve
    return config


def load_config(path) -> Config:
    with open(path) as handle:
        return parse_config(yaml.safe_load(handle))


# --------------------------------------------------------------------------- #
# The geometry as a function of the split
# --------------------------------------------------------------------------- #


class Structure(NamedTuple):
    """What stays the same across the split range: which polygon vertices each
    wall segment joins, which ends are extended past an opening, which
    segments form each wall chain, and which vertices are singular corners."""
    segments: tuple      # ((start vertex, end vertex), ...)
    extend: tuple        # ((extend the start?, extend the end?), ...)
    chains: tuple        # (segment indices of the lower chain, septum, upper chain)
    singular: tuple      # vertex indices of the singular corners
    centre: tuple        # the network's input centre (bounding box at mid-range)


def _structure_at(g: Geometry):
    """The structure of one geometry, by vertex index (flow.wall_chains' rules)."""
    v = g.vertices()
    kinds = EDGE_KINDS[g.septum]
    n = len(v)
    first = kinds.index("inlet") + 1
    chains, current = [], None
    for k in range(n):
        i = (first + k) % n
        if kinds[i] != "wall":
            current = None
            continue
        a, b = i, (i + 1) % n
        if current is None:
            current = []
            chains.append(current)
        if current:
            p, q = current[-1]
            d0, d1 = v[q] - v[p], v[b] - v[a]
            cross = d0[0] * d1[1] - d0[1] * d1[0]
            if (abs(cross) <= 1e-12 * np.linalg.norm(d0) * np.linalg.norm(d1)
                    and np.dot(d0, d1) > 0.0):
                current[-1] = (p, b)
                continue
        current.append((a, b))
    segments, extend, chain_index = [], [], []
    for chain in chains:
        idx = []
        for j, seg in enumerate(chain):
            idx.append(len(segments))
            segments.append(seg)
            extend.append((j == 0, j == len(chain) - 1))
        chain_index.append(tuple(idx))
    singular = []
    for i in range(n):
        if kinds[i - 1] != "wall" or kinds[i] != "wall":
            continue
        before, after = v[i] - v[i - 1], v[(i + 1) % n] - v[i]
        turn = before[0] * after[1] - before[1] * after[0]
        if turn < -1e-12 * np.linalg.norm(before) * np.linalg.norm(after):
            singular.append(i)
    return tuple(segments), tuple(extend), tuple(chain_index), tuple(singular)


def structure(config: Config) -> Structure:
    """The structure, checked to be the same at RANGE_CHECKS splits across the
    range; refuses a range where the shape changes (say, tip to face)."""
    seen = set()
    for split in np.linspace(config.split_min, config.split_max, RANGE_CHECKS):
        try:
            g = config.geometry(split)
        except geo.GeometryError as error:
            raise ConfigError(f"split {split:.4f} in the design range: {error}") from error
        if g.septum != "tip":
            raise ConfigError("the parametric network needs the septum-tip shape throughout "
                              f"the range; split {split:.4f} gives {g.septum!r}")
        flow.wall_chains(g, config.wall_extension)      # refuses an extension entering it
        seen.add(_structure_at(g))
    if len(seen) != 1:
        raise ConfigError("the wall structure changes across the split range (a joint "
                          "becomes collinear, or a corner stops being singular); narrow it")
    segments, extend, chains, singular = seen.pop()
    lo, hi = config.geometry(config.split_mid).bounding_box()
    return Structure(segments, extend, chains, singular, tuple(0.5 * (lo + hi)))


def make_geometry(config: Config, st: Structure):
    """(vertices_of, segments_of, corners_of): differentiable functions of s."""
    a1, a2 = math.radians(config.angle1), math.radians(config.angle2)
    lp, ld, ext = config.parent_length, config.daughter_length, config.wall_extension
    width = geo.PARENT_WIDTH

    def vertices_of(s):
        return geo.vertices(width * s, width * (1.0 - s), a1, a2, lp, ld, xp=jnp)

    def segments_of(s):
        v = vertices_of(s)
        out = []
        for (i, j), (start, end) in zip(st.segments, st.extend):
            a, b = v[i], v[j]
            d = (b - a) / jnp.sqrt(jnp.sum((b - a) ** 2))
            if start:
                a = a - ext * d
            if end:
                b = b + ext * d
            out.append(jnp.stack([a, b]))
        return jnp.stack(out)

    def corners_of(s):
        return vertices_of(s)[jnp.asarray(st.singular)]

    return vertices_of, segments_of, corners_of


def outlet_normals(config: Config):
    """The outlets' outward normals: each daughter's own direction, whatever s."""
    a1, a2 = math.radians(config.angle1), math.radians(config.angle2)
    return (jnp.array([math.cos(a1), math.sin(a1)]), jnp.array([math.cos(a2), -math.sin(a2)]))


# --------------------------------------------------------------------------- #
# The network, the field and outlet 1's pressure
# --------------------------------------------------------------------------- #


def init_params(config: Config, key):
    return {"net": pinn.init_network(key, 3, config.width, config.depth, 2, config.init_gain),
            "offset": jnp.zeros(config.offset_degree + 1, dtype=jnp.float64)}


def offset(config: Config, params, s):
    """lam(s) = sum_k c_k T_k(xi), xi = (s - mid) / half: outlet 1's pressure above outlet 2's."""
    xi = (s - config.split_mid) / config.split_half
    c = params["offset"]
    total, previous, current = c[0], 1.0, xi
    for k in range(1, c.shape[0]):
        total = total + c[k] * current
        previous, current = current, 2.0 * xi * current - previous
    return total


def make_field(config: Config, st: Structure, q_in):
    """field(params, xy, s) -> (u, v, p) at one point of the design with split s."""
    _, segments_of, _ = make_geometry(config, st)
    centre = jnp.asarray(st.centre)
    scale, activation = config.input_scale, config.activation
    order, mu = config.distance_order, config.blend_exponent
    mid, half = config.split_mid, config.split_half
    values = (0.0, (1.0 - config.outlet1_share) * q_in, q_in)    # lower chain, septum, upper
    chains = [jnp.asarray(idx) for idx in st.chains]

    def psi(params, point, s):
        xi = jnp.concatenate([(point - centre) / scale, jnp.atleast_1d((s - mid) / half)])
        raw = pinn.apply_network(params["net"], xi, activation)
        segments = segments_of(s)
        d = [flow.wall_distance(point, segments[idx], order) ** mu for idx in chains]
        w = (d[1] * d[2], d[0] * d[2], d[0] * d[1])
        blend = (values[1] * w[1] + values[2] * w[2]) / (w[0] + w[1] + w[2])
        return blend + flow.wall_distance(point, segments, order) ** 2 * raw[0], raw

    def field(params, xy, s):
        gradient, raw = jax.grad(lambda q: psi(params, q, s), has_aux=True)(xy)
        return jnp.stack([gradient[1], -gradient[0], raw[1]])

    return field


# --------------------------------------------------------------------------- #
# Residuals and collocation
# --------------------------------------------------------------------------- #


def _row_derivatives(field, params, rows):
    """pinn.batched_derivatives for rows (x, y, s): derivatives along x and y only."""
    batched = jax.vmap(lambda row: field(params, row[:2], row[2]))
    value, d_dx, d2_dx2 = pinn._directional(batched, rows, jnp.array([1.0, 0.0, 0.0]))
    _, d_dy, d2_dy2 = pinn._directional(batched, rows, jnp.array([0.0, 1.0, 0.0]))
    return pinn._assemble(value, d_dx, d_dy, d2_dx2, d2_dy2)


def make_residuals(config: Config, st: Structure, field):
    """(groups, batched_loss), as flow.make_residuals, every row carrying its s."""
    reynolds, scaling = config.reynolds, config.pressure_scaling
    _, nu = pinn.scaling_coefficients(reynolds, scaling)
    w = config.weights
    normal1, normal2 = outlet_normals(config)
    _, _, corners_of = make_geometry(config, st)
    rho = config.corner_weight_radius

    def corner_weight(xy, s):
        if rho == 0.0 or not st.singular:
            return 1.0
        return jnp.minimum(1.0, jnp.min(jnp.sqrt(jnp.sum((corners_of(s) - xy) ** 2, axis=1)))
                           / rho)

    def momentum(d, n, weight):
        return [weight * jnp.sqrt(w["momentum_x"] / n) * pinn.momentum_x(d, reynolds, scaling),
                weight * jnp.sqrt(w["momentum_y"] / n) * pinn.momentum_y(d, reynolds, scaling)]

    def interior(params, row, n):
        d = pinn.point_derivatives(lambda q: field(params, q, row[2]), row[:2])
        return jnp.stack(momentum(d, n, corner_weight(row[:2], row[2])))

    def inlet(params, row, n):
        value = field(params, row[:2], row[2])
        return jnp.sqrt(w["inlet"] / n) * jnp.stack([value[0] - row[3], value[1] - row[4]])

    def traction(params, row, normal, extra):
        value, d_dn = jax.jvp(lambda q: field(params, q, row[2]), (row[:2],), (normal,))
        return nu * d_dn[:2] - value[2] * normal + extra * normal

    def outlet1(params, row, n):
        return jnp.sqrt(w["outlet"] / n) * traction(params, row, normal1,
                                                    offset(config, params, row[2]))

    def outlet2(params, row, n):
        return jnp.sqrt(w["outlet"] / n) * traction(params, row, normal2, 0.0)

    def groups(batch):
        return [(interior, batch.interior), (inlet, batch.inlet),
                (outlet1, batch.outlet1), (outlet2, batch.outlet2)]

    def batched_loss(params, batch):
        rows = jnp.asarray(batch.interior)
        n = rows.shape[0]
        d = _row_derivatives(field, params, rows)
        weight = jax.vmap(lambda row: corner_weight(row[:2], row[2]))(rows)
        total = sum(jnp.sum(r**2) for r in momentum(d, n, weight))
        for fn, points in groups(batch)[1:]:
            m = points.shape[0]
            total = total + jnp.sum(jax.vmap(lambda row: fn(params, row, m))(points) ** 2)
        return total

    return groups, batched_loss


def design_splits(rng, config: Config, n):
    """n splits stratified over the range: one uniform draw in each of n equal bins."""
    return config.split_min + (np.arange(n) + rng.random(n)) / n * (config.split_max
                                                                     - config.split_min)


def sample_batch(rng, config: Config, inlet, designs) -> flow.Batch:
    interior, inlets, outlets1, outlets2 = [], [], [], []
    for s in design_splits(rng, config, designs):
        g = config.geometry(s)
        n_corner = int(round(config.points_per_design * config.corner_fraction))
        points = np.concatenate([
            flow.uniform_points(rng, g, config.points_per_design - n_corner,
                                config.corner_exclusion),
            flow.corner_points(rng, g, n_corner, config.corner_radius, config.corner_exclusion)])
        interior.append(np.column_stack([points, np.full(len(points), s)]))
        y = np.sin(0.5 * np.pi * rng.uniform(-1.0, 1.0, config.inlet_per_design))
        u, v = inlet(y)
        inlets.append(np.column_stack([np.zeros_like(y), y, np.full_like(y, s), u, v]))
        for kind, out in (("outlet1", outlets1), ("outlet2", outlets2)):
            p, _ = g.sample_boundary(rng, kind, config.outlet_per_design)
            out.append(np.column_stack([p, np.full(len(p), s)]))
    return flow.Batch(interior=np.concatenate(interior), inlet=np.concatenate(inlets),
                      outlet1=np.concatenate(outlets1), outlet2=np.concatenate(outlets2))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _fingerprint(config: Config) -> dict:
    text = json.dumps(dataclasses.asdict(config), sort_keys=True, default=repr)
    code = hashlib.sha256()
    for path in (pathlib.Path(__file__), HERE / "flow.py", HERE / "geometry.py",
                 pathlib.Path(pinn.__file__)):
        code.update(path.read_bytes())
    import scipy
    code.update(f"jax {jax.__version__} optax {optax.__version__} scipy {scipy.__version__} "
                f"numpy {np.__version__}".encode())
    return {"config": hashlib.sha256(text.encode()).hexdigest(), "code": code.hexdigest()}


def build(config: Config):
    """(structure, inlet, q_in, field): everything training and evaluation share."""
    st = structure(config)
    inlet = flow.inlet_function(config)
    q_in = flow.inlet_flux(inlet, config.n_flux)
    return st, inlet, q_in, make_field(config, st, q_in)


def train(config: Config, verbose=True, checkpoint=None, resume=False):
    st, inlet, q_in, field = build(config)
    groups, batched_loss = make_residuals(config, st, field)
    model_key, draws, heldout_rng = flow._streams(config.seed)
    params = init_params(config, model_key)

    def draw():
        return sample_batch(draws, config, inlet, config.designs)

    heldout = sample_batch(heldout_rng, config, inlet, config.validation_designs)

    def describe(p):
        return {f"offset({s:.2f})": float(offset(config, p, s))
                for s in (config.split_min, config.split_mid, config.split_max)}

    best, history, info = flow.optimise(config, params, groups, batched_loss, draws, draw,
                                        heldout, _fingerprint(config), describe, verbose,
                                        checkpoint, resume)
    info["inlet_flux"] = q_in
    return best, history, info


# --------------------------------------------------------------------------- #
# Evaluation: fields at set splits, and the optimum against finite elements
# --------------------------------------------------------------------------- #


def predictor(field, params, chunk=8192):
    """model(x, y, s) -> (u, v, p) arrays; s a scalar or an array like x."""
    apply = jax.jit(jax.vmap(lambda row: field(params, row[:2], row[2])))

    def model(x, y, s):
        x, y, s = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                      np.asarray(y, dtype=np.float64),
                                      np.asarray(s, dtype=np.float64))
        rows = np.stack([x.ravel(), y.ravel(), s.ravel()], axis=1)
        out = np.concatenate([np.asarray(apply(jnp.asarray(rows[i:i + chunk])))
                              for i in range(0, len(rows), chunk)])
        return tuple(out[:, k].reshape(x.shape) for k in range(3))

    return model


def at_split(model, s):
    """The model at one split, as flow's measuring functions take it: (x, y) -> (u, v, p)."""
    return lambda x, y: model(x, y, np.full_like(np.asarray(x, dtype=np.float64), s))


def pinn_drop(config: Config, model, s):
    a, _ = pinn.scaling_coefficients(config.reynolds, config.pressure_scaling)
    return flow.opening_measures(at_split(model, s), config.geometry(s), a,
                                 config.n_line)["total_pressure_drop"]


def evaluate(config: Config, params, verbose=True):
    from bifurcation import reference
    st, inlet, _, field = build(config)
    model = predictor(field, params)
    a, _ = pinn.scaling_coefficients(config.reynolds, config.pressure_scaling)
    solves = {}

    def solve(s):
        key = round(float(s), 12)
        if key not in solves:
            start = time.perf_counter()
            solution = reference.solve_bifurcation(
                config.geometry(s), config.reynolds, config.pressure_scaling, inlet,
                config.outlet1_share, config.fe_max_area, config.fe_corner_levels,
                config.fe_corner_radius, continuation=())
            solves[key] = (solution, time.perf_counter() - start)
        return solves[key][0]

    rows = []
    for s in config.eval_splits:
        g = config.geometry(s)
        solution = solve(s)
        mine_model = at_split(model, s)
        errors = flow.relative_errors(solution, mine_model)
        away = flow.relative_errors(solution, mine_model, g.singular_corners(),
                                    config.corner_exclusion)
        mine = flow.opening_measures(mine_model, g, a, config.n_line)
        theirs = flow.opening_measures(solution, g, a, config.n_line)
        rows.append({
            "split": s, "relative_l2": errors, "relative_l2_outside_corners": away,
            "pinn_drop": mine["total_pressure_drop"], "reference_drop": theirs["total_pressure_drop"],
            "drop_error": mine["total_pressure_drop"] / theirs["total_pressure_drop"] - 1.0,
            "pinn_split_flux": mine["split_flux"], "pinn_mass_imbalance": mine["mass_imbalance"],
            "pinn_offset": float(offset(config, params, s)),
            "reference_offset": solution.info["multiplier"],
        })
        if verbose:
            r = rows[-1]
            print(f"  split {s:.3f}: u {errors['u']:.2e} v {errors['v']:.2e} p {errors['p']:.2e}"
                  f" | drop PINN {r['pinn_drop']:.5f} FE {r['reference_drop']:.5f} "
                  f"({100 * r['drop_error']:+.3f}%) | offset {r['pinn_offset']:+.4f} vs "
                  f"{r['reference_offset']:+.4f}", flush=True)

    # The PINN's optimum: its objective on a grid, then Brent between the
    # grid's best point and its neighbours. Its cost is line integrals only.
    curve = np.linspace(config.split_min, config.split_max, config.curve_points)
    drops = np.array([pinn_drop(config, model, s) for s in curve])

    def argmin(fun, grid, values):
        k = int(np.argmin(values))
        if k in (0, len(grid) - 1):
            return float(grid[k]), float(values[k]), True
        result = minimize_scalar(fun, bounds=(grid[k - 1], grid[k + 1]), method="bounded",
                                 options={"xatol": config.search_xatol})
        return float(result.x), float(result.fun), False

    pinn_split, pinn_min, pinn_on_bound = argmin(lambda s: pinn_drop(config, model, s),
                                                 curve, drops)
    # The reference's optimum: Brent around the best evaluation split, its
    # drop measured by the function that measures the PINN's (CLAUDE.md)
    def fe(s):
        return flow.opening_measures(solve(s), config.geometry(s), a,
                                     config.n_line)["total_pressure_drop"]

    eval_grid = np.array(sorted(config.eval_splits))
    fe_split, fe_min, fe_on_bound = argmin(fe, eval_grid, np.array([fe(s) for s in eval_grid]))
    fe_at_pinn = fe(pinn_split)
    optimum = {
        "pinn_split": pinn_split, "pinn_min_drop": pinn_min, "pinn_on_bound": pinn_on_bound,
        "reference_split": fe_split, "reference_min_drop": fe_min,
        "reference_on_bound": fe_on_bound,
        "argmin_error": pinn_split - fe_split,
        "regret": fe_at_pinn / fe_min - 1.0,
        "min_drop_error": pinn_min / fe_min - 1.0,
        "murray_split": math.sqrt(config.outlet1_share)
        / (math.sqrt(config.outlet1_share) + math.sqrt(1.0 - config.outlet1_share)),
        "curve": {"split": curve.tolist(), "pinn_drop": drops.tolist()},
        "reference_solves": len(solves),
        "reference_seconds_each": float(np.mean([t for _, t in solves.values()])),
    }
    if verbose:
        print(f"  optimum: PINN split {pinn_split:.4f} (drop {pinn_min:.5f}), reference "
              f"{fe_split:.4f} (drop {fe_min:.5f}); argmin error {optimum['argmin_error']:+.4f},"
              f" regret {100 * optimum['regret']:+.4f}%", flush=True)
    return {"splits": rows, "optimum": optimum}


def report(config: Config, metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o = metrics["optimum"]
    rows = metrics["splits"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.plot(o["curve"]["split"], o["curve"]["pinn_drop"], color="#2A78D6", label="parametric PINN")
    ax1.plot([r["split"] for r in rows], [r["reference_drop"] for r in rows], "o", color="#333333",
             label="finite elements")
    ax1.axvline(o["pinn_split"], color="#2A78D6", ls="--", lw=1)
    ax1.axvline(o["reference_split"], color="#333333", ls=":", lw=1)
    ax1.axvline(o["murray_split"], color="#EB6834", ls="-.", lw=1, label="Murray split")
    ax1.set_xlabel("width split s = w1 / w0")
    ax1.set_ylabel("total-pressure drop")
    ax1.legend(fontsize=8)
    for key, colour in (("u", "#2A78D6"), ("v", "#EB6834"), ("p", "#1BAF7A")):
        ax2.semilogy([r["split"] for r in rows], [r["relative_l2"][key] for r in rows], "o-",
                     color=colour, label=key)
    ax2.semilogy([r["split"] for r in rows], [abs(r["drop_error"]) for r in rows], "s--",
                 color="#6B6B6B", label="|drop error|")
    ax2.set_xlabel("width split s = w1 / w0")
    ax2.set_ylabel("relative error against finite elements")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(color="#E6E6E6")
    fig.suptitle(f"{config.name}: one network for every split, against finite elements",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Running and loading
# --------------------------------------------------------------------------- #


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
    pinn.save_params(directory / "params.npz", params)
    (directory / "history.json").write_text(json.dumps(flow._json_safe(history), indent=1))
    metrics = evaluate(config, params, verbose)
    (directory / "metrics.json").write_text(
        json.dumps(flow._json_safe({**metrics, "training": info}), indent=1))
    report(config, metrics, directory / "report.png")
    return metrics


def load_run(directory):
    """(model, config, params): model(x, y, s) -> (u, v, p)."""
    from jax.flatten_util import ravel_pytree
    directory = pathlib.Path(directory)
    config = load_config(directory / "config.yaml")
    template, unflatten = ravel_pytree(init_params(config, flow._streams(config.seed)[0]))
    flat = np.load(directory / "params.npz")["flat"]
    if flat.shape != template.shape:
        raise ValueError("saved parameters do not match the config's network")
    params = unflatten(jnp.asarray(flat))
    _, _, _, field = build(config)
    model = predictor(field, params)
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
