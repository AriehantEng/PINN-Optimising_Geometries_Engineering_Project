"""Flow fields for the results report: what the flow inside each bifurcation
looks like, by finite elements and by the parametric PINN.

    python -m bifurcation.fields --config bifurcation/fields_re10.yaml fe
    python -m bifurcation.fields --config bifurcation/fields_re10.yaml pinn
    python -m bifurcation.fields --config bifurcation/fields_re10.yaml figures --out docs/fields

`fe` solves every layout of the design study at its optimal width split, for
each configured share, on the study's own check mesh, and the parametric
PINN's reference at each configured split on that network's evaluation mesh.
Each field is sampled on a regular grid inside its polygon and cached under
`cache` (run outputs: never committed); a cached field is not solved again. `pinn` samples the trained parametric
network at the same points as its reference. `figures` draws them, with the
variables figure that names every symbol.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import pathlib

import numpy as np
import yaml
from matplotlib.path import Path

from bifurcation.geometry import Geometry
from developing_flow import pinn

ROOT = pathlib.Path(__file__).resolve().parent.parent
ConfigError = pinn.ConfigError

SCHEMA = {
    "name": str,
    "study": str,                        # the design study's merged JSON
    "cache": str,                        # where sampled fields are kept
    "workers": int,
    "gallery": {"shares": list, "grid": int},
    "pinn": {"config": str, "run": str, "splits": list, "grid": int},
}

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e6e3"

#: points per finite-element probe: the probe holds a points x elements array,
#: and 20,000 points on the check mesh took 11 GB, which killed pool workers
PROBE_CHUNK = 1000


def parse_config(document):
    document = copy.deepcopy(document)
    pinn._check(document, SCHEMA)
    return document


def load_config(path):
    return parse_config(yaml.safe_load(pathlib.Path(path).read_text()))


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def sample_grid(geometry: Geometry, n):
    """(X, Y, inside): a grid of spacing extent / n over the polygon's bounding
    box, and which points lie strictly inside the polygon."""
    v = geometry.vertices()
    lo, hi = v.min(axis=0), v.max(axis=0)
    h = float(max(hi - lo)) / n
    xs = np.arange(lo[0], hi[0] + h, h)
    ys = np.arange(lo[1], hi[1] + h, h)
    X, Y = np.meshgrid(xs, ys)
    # a small negative radius keeps points on the walls out of the probe
    inside = Path(v).contains_points(np.column_stack([X.ravel(), Y.ravel()]),
                                     radius=-1e-9 if _ccw(v) else 1e-9).reshape(X.shape)
    return X, Y, inside


def _ccw(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0


def sample(field, X, Y, inside):
    """(u, v, p) on the grid, NaN outside the polygon."""
    out = [np.full(X.shape, np.nan) for _ in range(3)]
    values = field(X[inside], Y[inside])
    for o, val in zip(out, values):
        o[inside] = np.asarray(val, dtype=np.float64)
    return out


def _save(path, geometry, X, Y, inside, u, v, p, meta):
    np.savez_compressed(path, X=X, Y=Y, inside=inside, u=u, v=v, p=p,
                        vertices=geometry.vertices(), meta=json.dumps(meta))


def load(path):
    d = np.load(path)
    out = {k: d[k] for k in ("X", "Y", "inside", "u", "v", "p", "vertices")}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def gallery_name(cache, angles, share):
    return pathlib.Path(cache) / f"fe_{angles[0]:g}_{angles[1]:g}_q{share:g}.npz"


def pinn_name(cache, kind, split):
    return pathlib.Path(cache) / f"{kind}_parametric_s{split:.4f}.npz"


# --------------------------------------------------------------------------- #
# Solving
# --------------------------------------------------------------------------- #


def _gallery_task(task):
    from bifurcation import reference
    study_config, angles, share, split, n, path = task
    g = study_config["geometry"]
    geometry = Geometry.from_split(split, angles[0], angles[1], g["parent_length"],
                                   g["daughter_length"], g["min_width"])
    check = study_config["check"]
    solution = reference.solve_bifurcation(
        geometry, study_config["physics"]["reynolds"], study_config["physics"]["pressure_scaling"],
        reference.parabola_inlet, share, check["max_area"], check["corner_levels"],
        study_config["mesh"]["corner_radius"])
    X, Y, inside = sample_grid(geometry, n)
    u, v, p = sample(lambda x, y: solution(x, y, chunk=PROBE_CHUNK), X, Y, inside)
    m = solution.measures()
    _save(path, geometry, X, Y, inside, u, v, p,
          {"angles": angles, "share": share, "split": split, "w1": geometry.w1,
           "w2": geometry.w2, "total_pressure_drop": m["total_pressure_drop"],
           "outlet1_pressure_offset": m["outlet1_pressure_offset"],
           "split_flux": m["split_flux"], "triangles": m["triangles"]})
    return str(path)


def _reference_task(task):
    from bifurcation import parametric, reference
    config_path, split, n, path = task
    config = parametric.load_config(config_path)
    _, inlet, _, _ = parametric.build(config)
    geometry = config.geometry(split)
    solution = reference.solve_bifurcation(
        geometry, config.reynolds, config.pressure_scaling, inlet, config.outlet1_share,
        config.fe_max_area, config.fe_corner_levels, config.fe_corner_radius)
    X, Y, inside = sample_grid(geometry, n)
    u, v, p = sample(lambda x, y: solution(x, y, chunk=PROBE_CHUNK), X, Y, inside)
    m = solution.measures()
    _save(path, geometry, X, Y, inside, u, v, p,
          {"split": split, "share": config.outlet1_share,
           "angles": [config.angle1, config.angle2], "w1": geometry.w1, "w2": geometry.w2,
           "total_pressure_drop": m["total_pressure_drop"],
           "outlet1_pressure_offset": m["outlet1_pressure_offset"],
           "triangles": m["triangles"]})
    return str(path)


def run_fe(config):
    study = json.loads((ROOT / config["study"]).read_text())
    sc = study["config"]
    cache = ROOT / config["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    tasks = []
    for share in config["gallery"]["shares"]:
        for r in study["rows"]:
            if r["share"] == share and r["feasible"] \
                    and not gallery_name(cache, r["angles"], share).exists():
                tasks.append((sc, r["angles"], share, r["best_split"], config["gallery"]["grid"],
                              gallery_name(cache, r["angles"], share)))
    refs = [(str(ROOT / config["pinn"]["config"]), s, config["pinn"]["grid"],
             pinn_name(cache, "fe", s)) for s in config["pinn"]["splits"]
            if not pinn_name(cache, "fe", s).exists()]
    with multiprocessing.get_context("spawn").Pool(config["workers"]) as pool:
        done = list(pool.imap_unordered(_gallery_task, tasks))
        done += list(pool.imap_unordered(_reference_task, refs))
    return done


def run_pinn(config):
    from bifurcation import parametric
    model, pconfig, _ = parametric.load_run(ROOT / config["pinn"]["run"])
    cache = ROOT / config["cache"]
    done = []
    for s in config["pinn"]["splits"]:
        ref = load(pinn_name(cache, "fe", s))
        u, v, p = sample(parametric.at_split(model, s), ref["X"], ref["Y"], ref["inside"])
        geometry = pconfig.geometry(s)
        a, _ = pinn.scaling_coefficients(pconfig.reynolds, pconfig.pressure_scaling)
        from bifurcation import flow
        m = flow.opening_measures(parametric.at_split(model, s), geometry, a, pconfig.n_line)
        path = pinn_name(cache, "pinn", s)
        _save(path, geometry, ref["X"], ref["Y"], ref["inside"], u, v, p,
              {"split": s, "total_pressure_drop": m["total_pressure_drop"],
               "split_flux": m["split_flux"]})
        done.append(str(path))
    return done


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def _draw_field(ax, f, scalar, norm, cmap, streamlines=True, outline=True):
    from matplotlib.patches import Polygon
    ax.pcolormesh(f["X"], f["Y"], np.ma.masked_invalid(scalar), cmap=cmap, norm=norm,
                  shading="nearest", rasterized=True)
    if streamlines:
        ax.streamplot(f["X"][0], f["Y"][:, 0], np.nan_to_num(f["u"]), np.nan_to_num(f["v"]),
                      color="white", linewidth=0.45, density=1.1, arrowsize=0.5)
    if outline:
        ax.add_patch(Polygon(f["vertices"], closed=True, fill=False, edgecolor=INK, lw=0.9))
    ax.set_aspect("equal")
    ax.axis("off")


def fig_gallery(config, share, out_path):
    """Every layout at its optimal width split for one share: speed and streamlines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    _style(plt)
    study = json.loads((ROOT / config["study"]).read_text())
    cache = ROOT / config["cache"]
    layouts = sorted({tuple(r["angles"]) for r in study["rows"]},
                     key=lambda a: (a[0] + a[1], a[1] - a[0]))
    fields = {a: load(gallery_name(cache, a, share)) for a in layouts}
    vmax = max(np.nanmax(np.hypot(f["u"], f["v"])) for f in fields.values())
    norm, cmap = Normalize(0.0, vmax), plt.get_cmap("viridis")
    rows = math.ceil(len(layouts) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(10, 3.0 * rows + 0.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, a in zip(axes.ravel(), layouts):
        f = fields[a]
        _draw_field(ax, f, np.hypot(f["u"], f["v"]), norm, cmap)
        m = f["meta"]
        ax.set_title(f"{a[0]:g}°/{a[1]:g}°, bifurcation angle {a[0] + a[1]:g}°\n"
                     f"s* = {m['split']:.4f}, w₁ = {m['w1']:.3f}, w₂ = {m['w2']:.3f}\n"
                     f"J* = {m['total_pressure_drop']:.3f}", fontsize=8, loc="left", color=INK)
    fig.suptitle(f"Each layout at its optimal width split, daughter 1's share q = {share:g}, "
                 "Re_h = 10 (finite elements; white lines: streamlines)", x=0.01, ha="left",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.985))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.25, 0.04, 0.5, 0.008])
    fig.colorbar(sm, cax=cax, orientation="horizontal").set_label(
        "speed |u| / U_c (U_c = the inlet's centreline speed)", fontsize=9)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def fig_pinn(config, quantity, out_path):
    """The parametric PINN against finite elements at the configured splits:
    one row per split, columns FE, PINN, and their difference, each row on its
    own colour scales (outlet 1's pressure offset moves the whole pressure field
    from split to split)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    _style(plt)
    cache = ROOT / config["cache"]
    splits = config["pinn"]["splits"]
    pairs = [(load(pinn_name(cache, "fe", s)), load(pinn_name(cache, "pinn", s))) for s in splits]

    def value(f):
        return np.hypot(f["u"], f["v"]) if quantity == "speed" else f["p"]

    cmap = plt.get_cmap("viridis" if quantity == "speed" else "cividis")
    label = ("speed |u| / U_c" if quantity == "speed"
             else "p / (ρ U_c²), outlet 2 at 0")
    fig = plt.figure(figsize=(10.5, 3.2 * len(splits) + 0.5))
    grid = fig.add_gridspec(len(splits), 5, width_ratios=[1, 1, 0.04, 1, 0.04], wspace=0.25,
                            left=0.02, right=0.93, top=0.91, bottom=0.02, hspace=0.32)
    for i, ((fe, net), s) in enumerate(zip(pairs, splits)):
        a, b = value(fe), value(net)
        lo = min(np.nanpercentile(a, 0.5), np.nanpercentile(b, 0.5))
        hi = max(np.nanpercentile(a, 99.5), np.nanpercentile(b, 99.5))
        norm = Normalize(lo, hi)
        d = b - a
        dmax = np.nanpercentile(np.abs(d), 99.5)
        dnorm = TwoSlopeNorm(0.0, -dmax, dmax)
        axes = [fig.add_subplot(grid[i, k]) for k in (0, 1, 3)]
        _draw_field(axes[0], fe, a, norm, cmap, streamlines=quantity == "speed")
        _draw_field(axes[1], net, b, norm, cmap, streamlines=quantity == "speed")
        _draw_field(axes[2], fe, d, dnorm, plt.get_cmap("RdBu_r"), streamlines=False)
        drop_fe, drop_net = fe["meta"]["total_pressure_drop"], net["meta"]["total_pressure_drop"]
        axes[0].set_title(f"finite elements, s = {s:.4f}\nJ = {drop_fe:.4f}", fontsize=8.5,
                          loc="left")
        axes[1].set_title(f"parametric PINN, s = {s:.4f}\nJ = {drop_net:.4f} "
                          f"({100 * (drop_net / drop_fe - 1):+.2f}%)", fontsize=8.5, loc="left")
        axes[2].set_title("PINN − finite elements", fontsize=8.5, loc="left")
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                     cax=fig.add_subplot(grid[i, 2])).set_label(label, fontsize=8)
        fig.colorbar(plt.cm.ScalarMappable(norm=dnorm, cmap=plt.get_cmap("RdBu_r")),
                     cax=fig.add_subplot(grid[i, 4])).set_label("difference", fontsize=8)
    fig.suptitle(f"Parametric PINN against finite elements: {quantity}, 30°/90°, q = 0.5, "
                 "daughters 4 long (each row on its own scale)", x=0.01, ha="left", fontsize=10)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def ideal_drops(pconfig, splits):
    """The friction law ('ideal') for the parametric network's geometry, with
    the daughters' outer-wall length (Murray, equal lengths) and with their own
    centreline lengths; the parent's fixed friction included."""
    from bifurcation import sensitivity
    study = {"geometry": {"parent_length": pconfig.parent_length,
                          "daughter_length": pconfig.daughter_length,
                          "min_width": pconfig.min_width},
             "physics": {"reynolds": pconfig.reynolds,
                         "pressure_scaling": pconfig.pressure_scaling}}
    _, nu = pinn.scaling_coefficients(pconfig.reynolds, pconfig.pressure_scaling)
    q0, w0, q = 4.0 / 3.0, 2.0, pconfig.outlet1_share
    parent = 12.0 * nu * q0 * pconfig.parent_length / w0**3
    angles = [pconfig.angle1, pconfig.angle2]
    equal = [parent + 12.0 * nu * q0 / w0**3 * pconfig.daughter_length
             * (q**2 / s**3 + (1 - q) ** 2 / (1 - s) ** 3) for s in splits]
    lengths = [parent + sensitivity.model_drop(study, angles, q, s, 0.0, 0.0) for s in splits]
    return np.array(equal), np.array(lengths)


def fig_comparison(config, out_path):
    """The parametric PINN against finite elements and the ideal friction law:
    pressure drop against split, the PINN's errors by split, and the optimum
    each route finds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar
    from bifurcation import parametric
    _style(plt)
    run = ROOT / config["pinn"]["run"]
    pconfig = parametric.load_config(run / "config.yaml")
    m = json.loads((run / "metrics.json").read_text())
    rows, o = m["splits"], m["optimum"]
    s_fe = np.array([r["split"] for r in rows])
    j_fe = np.array([r["reference_drop"] for r in rows])
    fine = np.linspace(pconfig.split_min, pconfig.split_max, 121)
    equal, lengths = ideal_drops(pconfig, fine)
    s_len = float(minimize_scalar(lambda s: ideal_drops(pconfig, [s])[1][0],
                                  bounds=(pconfig.split_min, pconfig.split_max), method="bounded",
                                  options={"xatol": 1e-6}).x)
    blue = "#2a78d6"
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.9),
                                     gridspec_kw={"width_ratios": [1.25, 1.1, 0.9]})
    # 1. pressure drop against split
    a1.plot(fine, equal, color=MUTED, ls=(0, (5, 3)), lw=1.6,
            label="ideal: Murray, equal daughter lengths")
    a1.plot(fine, lengths, color=MUTED, ls=(0, (1, 2)), lw=1.8,
            label="ideal: friction law, centreline lengths")
    a1.plot(o["curve"]["split"], o["curve"]["pinn_drop"], color=blue, lw=2.2,
            label="parametric PINN")
    a1.plot(s_fe, j_fe, "o", ms=7, mfc="none", mec=INK, mew=1.6,
            label="finite elements (reference)")
    k = int(np.argmin(np.abs(s_fe - 0.5)))
    gap = 100 * (equal[np.argmin(np.abs(fine - 0.5))] / j_fe[k] - 1)
    a1.annotate(f"ideal {gap:.1f}% below\nfinite elements", (0.5, equal.min()),
                xytext=(0.535, 3.72), fontsize=8.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    a1.set_xlabel("width split s = w₁/w₀")
    a1.set_ylabel("total-pressure drop J (ρU_c²)")
    a1.set_title("Pressure drop against width split", loc="left", fontsize=10)
    a1.legend(fontsize=8, frameon=False, loc="upper center")
    a1.set_ylim(3.5, 7.3)
    # 2. errors by split
    width = 0.012
    drop = np.array([100 * r["drop_error"] for r in rows])
    a2.axvspan(0.45 - 0.025, 0.50 + 0.025, color=GRID, lw=0, zorder=0)
    a2.text(0.43, 2.75, "the optimum lies here", fontsize=8, color=MUTED)
    a2.bar(s_fe, drop, width=width, color=blue, label="pressure-drop error (signed)", zorder=2)
    for s, d in zip(s_fe, drop):
        a2.text(s, d + (0.12 if d >= 0 else -0.3), f"{d:+.2f}%", ha="center", fontsize=7.5,
                color=INK)
    a2.plot(s_fe + 0.016, [100 * r["relative_l2"]["u"] for r in rows], "o", ms=6, color=MUTED,
            label="velocity u error (relative L2)", zorder=3)
    a2.plot(s_fe + 0.016, [100 * r["relative_l2"]["p"] for r in rows], "o", ms=6, mfc="none",
            mec=INK, mew=1.4, label="pressure p error (relative L2)", zorder=3)
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_xticks(s_fe)
    a2.set_xlabel("width split s = w₁/w₀")
    a2.set_ylabel("PINN against finite elements (%)")
    a2.set_title("The PINN's error at each split", loc="left", fontsize=10)
    a2.legend(fontsize=8, frameon=False, loc="lower left")
    a2.set_ylim(-2.6, 3.2)
    # 3. optimum by route
    routes = [("Ideal: Murray, equal lengths", o["murray_split"], 0.0, MUTED, "none"),
              ("Ideal: centreline lengths", s_len, 0.0, MUTED, "none"),
              ("Finite elements", o["reference_split"], pconfig.search_xatol, INK, INK),
              ("Parametric PINN", o["pinn_split"], pconfig.search_xatol, blue, blue)]
    for i, (name, s, tol, edge, face) in enumerate(routes):
        y = len(routes) - 1 - i
        if tol:
            a3.plot([s - tol, s + tol], [y, y], color=edge, lw=2)
        a3.plot(s, y, "o", ms=10, mfc=face, mec=edge, mew=2)
        a3.text(s, y + 0.22, f"{s:.4f}", ha="center", fontsize=8.5, color=INK)
    a3.set_yticks(range(len(routes)))
    a3.set_yticklabels([r[0] for r in routes][::-1], fontsize=8.5)
    a3.set_xlim(0.475, 0.505)
    a3.set_ylim(-0.6, len(routes) - 0.3)
    a3.set_xlabel("optimal width split s*  (bars: ±0.002 search tolerance)")
    a3.set_title("The optimum each route finds", loc="left", fontsize=10)
    for ax in (a1, a2, a3):
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"Parametric PINN against finite elements and the ideal friction law: "
                 f"{pconfig.angle1:g}°/{pconfig.angle2:g}°, q = {pconfig.outlet1_share:g}, "
                 f"daughters {pconfig.daughter_length:g} long, Re_h = {pconfig.reynolds:g}",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def fig_variables(out_path):
    """The bifurcation with every design variable drawn and named."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Polygon
    from bifurcation.geometry import corner_points
    _style(plt)
    g = Geometry.from_split(0.4, 30.0, 70.0, 3.0, 6.0, 0.2)
    v = g.vertices()
    q = corner_points(g.w1, g.w2, math.radians(g.angle1), math.radians(g.angle2),
                      g.parent_length, g.daughter_length)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.add_patch(Polygon(v, closed=True, facecolor="#dde8f4", edgecolor=INK, lw=1.2))
    accent = "#2a78d6"

    def arrow(p0, p1, **kw):
        ax.annotate("", xy=p1, xytext=p0, arrowprops={"arrowstyle": "->", "color": accent,
                                                       "lw": 1.4, **kw})

    def dim(p0, p1, text, offset, ha="center"):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        t = p1 - p0
        nrm = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        a, b = p0 + offset * nrm, p1 + offset * nrm
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "<->", "color": MUTED,
                                                     "lw": 0.8})
        mid = 0.5 * (a + b) + 0.32 * np.sign(offset) * nrm
        ax.text(mid[0], mid[1], text, ha=ha, va="center", fontsize=10, color=INK)

    # inlet profile, outlet flows
    ys = np.linspace(-0.85, 0.85, 7)
    for y in ys:
        arrow((-1.2, y), (-1.2 + 0.9 * (1 - y**2), y))
    ax.text(-1.25, 1.35, "inlet: u = 1 − y², flow Q₀", fontsize=9.5, color=INK)
    for i, (outer, inner, name) in enumerate(((q["E1o"], q["E1i"], "Q₁ = q Q₀"),
                                              (q["E2o"], q["E2i"], "Q₂ = (1 − q) Q₀"))):
        mid = 0.5 * (outer + inner)
        d = np.array([math.cos(math.radians(g.angle1)), math.sin(math.radians(g.angle1))]) if i == 0 \
            else np.array([math.cos(math.radians(g.angle2)), -math.sin(math.radians(g.angle2))])
        arrow(mid, mid + 1.1 * d)
        ax.text(*(mid + 1.35 * d), f"outlet {i + 1}: {name}", fontsize=9.5, color=INK,
                ha="left" if i == 0 else "left", va="center")
    # dimensions
    dim(v[0], v[1], "L_p", -0.45)
    dim(v[0], v[8], "w₀", -0.55)
    dim(q["C"], q["E1o"], "L_d (outer wall)", 0.45)
    dim(q["B"], q["E2o"], "L_d", -0.45)
    # widths drawn across each daughter just upstream of its outlet, clear of the outflow arrows
    for outer, inner, cornr, name in ((q["E1o"], q["E1i"], q["C"], "w₁"),
                                      (q["E2o"], q["E2i"], q["B"], "w₂")):
        back = (cornr - outer) / np.linalg.norm(cornr - outer)
        a, b = outer + 1.3 * back, inner + 1.3 * back
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "<->", "color": MUTED,
                                                     "lw": 0.8})
        m = 0.5 * (a + b) + 0.45 * back
        ax.text(m[0], m[1], name, ha="center", va="center", fontsize=10, color=INK)
    # angles
    for corner, start, extent, label in ((q["C"], 0.0, g.angle1, "α₁"),
                                         (q["B"], -g.angle2, g.angle2, "α₂")):
        ax.plot([corner[0], corner[0] + 2.0], [corner[1], corner[1]], color=MUTED, lw=0.8, ls="--")
        ax.add_patch(Arc(corner, 3.0, 3.0, theta1=start, theta2=start + extent, color=MUTED, lw=0.9))
        mid = math.radians(start + extent / 2)
        ax.text(corner[0] + 1.85 * math.cos(mid), corner[1] + 1.85 * math.sin(mid), label,
                ha="center", va="center", fontsize=11, color=INK)
    # centrelines of the daughters, for L1 and L2
    for outer, inner, cornr, name in ((q["E1o"], q["E1i"], q["C"], "L₁"),
                                      (q["E2o"], q["E2i"], q["B"], "L₂")):
        a = 0.5 * (cornr + q["T"])
        b = 0.5 * (outer + inner)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=0.9, ls=(0, (3, 2)))
        m = 0.55 * a + 0.45 * b
        ax.text(m[0] + 0.15, m[1] + 0.15, name, fontsize=10, color=MUTED)
    ax.text(q["T"][0] + 0.25, q["T"][1], "T (septum tip)", fontsize=9, color=INK, va="center")
    ax.text(*(q["E1i"] + np.array([0.1, -0.5])), "outlet 1 pressure: Δp₁ above outlet 2",
            fontsize=8.5, color=MUTED)
    ax.set_aspect("equal")
    ax.axis("off")
    lo, hi = v.min(axis=0), v.max(axis=0)
    ax.set_xlim(lo[0] - 1.5, hi[0] + 3.4)
    ax.set_ylim(lo[1] - 1.2, hi[1] + 1.3)
    ax.set_title("The design variables, drawn at α₁ = 30°, α₂ = 70°, s = w₁/w₀ = 0.4\n"
                 "Lengths in units of the parent's half-width h, so w₀ = 2",
                 fontsize=9.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def figures(config, out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig_variables(out / "variables.png")
    for share in config["gallery"]["shares"]:
        fig_gallery(config, share, out / f"gallery_q{share:g}.png")
    if all(pinn_name(ROOT / config["cache"], "pinn", s).exists() for s in config["pinn"]["splits"]):
        fig_pinn(config, "speed", out / "pinn_speed.png")
        fig_pinn(config, "pressure", out / "pinn_pressure.png")
        fig_comparison(config, out / "pinn_fe_ideal.png")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["fe", "pinn", "figures"])
    parser.add_argument("--out", help="figure directory (figures)")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "fe":
        print("\n".join(run_fe(config)))
    elif args.command == "pinn":
        print("\n".join(run_pinn(config)))
    else:
        if not args.out:
            raise SystemExit("figures needs --out")
        figures(config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
