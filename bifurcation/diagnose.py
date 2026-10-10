"""Where a parametric PINN's error lives, and what its loss says about it.

    python -m bifurcation.diagnose --config bifurcation/diagnose_re10.yaml \\
        --out docs/bifurcation_re10-diagnostic

For each run listed in the config (trained parametric Network 2 runs on the
same layout, physics and evaluation mesh) and each split, against the
finite-element solution of the same problem, every integral taken on the
reference's own mesh (CLAUDE.md):

    errors by distance to the nearest singular corner
                 for each annulus of the config's bins: its share of the area,
                 of the squared velocity error and of the squared pressure
                 error, and the relative RMS velocity error inside it. Near a
                 reflex corner the flow goes like r^z, z < 1, which
                 psi = G + D^2 N cannot represent; this shows whether the error
                 sits there or is spread through the domain.
    residual against error
                 the unweighted momentum residual at a seeded subsample of the
                 same points: its RMS in each annulus, and the rank
                 correlation of |residual| with the local velocity error. The
                 loss is a diagnostic, never an accuracy claim; this measures
                 how good a diagnostic it is.
    the drop error, by opening
                 J = (E_in - E_1 - E_2) / Q_in with E = integral of
                 (p + a |u|^2 / 2) u.n: the PINN's error in J split into each
                 opening's pressure and kinetic parts, by the line quadrature
                 flow.opening_measures uses for both models.
    training     the selected iterate's held-out loss, the held-out to
                 training ratio at the end, and the training time.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from bifurcation import flow  # noqa: E402
from bifurcation import parametric as P  # noqa: E402
from developing_flow import pinn  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ConfigError = pinn.ConfigError

SCHEMA = {
    "name": str,
    "seed": int,                  # the residual subsample
    "runs": list,                 # [{label, run}]: finished parametric runs
    "splits": list,               # evaluated at each
    "corner_bins": list,          # annulus edges in the distance to the nearest singular corner
    "residual_points": int,       # subsample size for the residual
}

#: What every run must share for one finite-element reference to judge them all.
SHARED = ("reynolds", "pressure_scaling", "angle1", "angle2", "parent_length",
          "daughter_length", "min_width", "split_min", "split_max", "outlet1_share",
          "inlet_profile", "fe_max_area", "fe_corner_levels", "fe_corner_radius", "n_line")

COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED, BAND = "#0b0b0b", "#52514e", "#e4e3df"


def parse_config(document):
    document = copy.deepcopy(document)
    pinn._check(document, SCHEMA)
    if not document["runs"]:
        raise ConfigError("runs must list at least one run")
    for k, entry in enumerate(document["runs"]):
        if not isinstance(entry, dict) or set(entry) != {"label", "run"}:
            raise ConfigError(f"runs[{k}] must be a mapping with exactly 'label' and 'run'")
    edges = [float(e) for e in document["corner_bins"]]
    if len(edges) < 2 or edges[0] != 0.0 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ConfigError("corner_bins must start at 0 and increase")
    if not document["splits"] or document["residual_points"] < 10:
        raise ConfigError("splits must be non-empty and residual_points at least 10")
    document["corner_bins"] = edges
    return document


def load_runs(document):
    runs = []
    for entry in document["runs"]:
        model, config, params = P.load_run(ROOT / entry["run"])
        runs.append({"label": entry["label"], "run": entry["run"], "model": model,
                     "config": config, "params": params})
    first = runs[0]["config"]
    for r in runs[1:]:
        for key in SHARED:
            if getattr(r["config"], key) != getattr(first, key):
                raise ConfigError(f"run {r['run']} differs from {runs[0]['run']} in {key}; "
                                  "one reference cannot judge both")
    for s in document["splits"]:
        if not first.split_min <= s <= first.split_max:
            raise ConfigError(f"split {s} lies outside the runs' design range")
    return runs


def corner_distance(points, corners):
    """Distance from each point to the nearest singular corner."""
    return np.min(np.linalg.norm(points[:, None, :] - corners[None, :, :], axis=2), axis=1)


def binned(edges, r, weights, *fields):
    """For each annulus edges[k] <= r < edges[k+1]: the integral of weights times
    each field, as arrays with one entry per annulus."""
    which = np.digitize(r, edges) - 1
    out = []
    for f in fields:
        out.append(np.array([float(np.sum((weights * f)[which == k]))
                             for k in range(len(edges) - 1)]))
    return out


def energy_parts(model, geometry, a, n_line):
    """Each opening's pressure and kinetic energy flux along its outward normal,
    and the inflow, by flow.opening_measures' quadrature."""
    out = {}
    for kind in ("inlet", "outlet1", "outlet2"):
        nodes, weights, normal = geometry.line_quadrature(kind, n_line)
        nodes = nodes - flow.LINE_INSET * normal
        u, v, p = model(nodes[:, 0], nodes[:, 1])
        un = u * normal[0] + v * normal[1]
        out[kind] = {"pressure": float(weights @ (p * un)),
                     "kinetic": float(weights @ (0.5 * a * (u**2 + v**2) * un)),
                     "flux": float(weights @ un)}
    return out


def drop_from_parts(parts):
    """J = (E_in - E_1 - E_2) / Q_in, the inlet's terms along the inward normal."""
    q_in = -parts["inlet"]["flux"]
    e = {k: parts[k]["pressure"] + parts[k]["kinetic"] for k in parts}
    return (-e["inlet"] - e["outlet1"] - e["outlet2"]) / q_in


def drop_error_parts(mine, theirs):
    """The error in J split into each opening's pressure and kinetic terms, each
    divided by the reference's inflow: they sum to J_pinn - J_fe to first order
    in the flux difference, which the stream function makes zero."""
    q = -theirs["inlet"]["flux"]
    sign = {"inlet": -1.0, "outlet1": -1.0, "outlet2": -1.0}
    return {f"{k}_{part}": sign[k] * (mine[k][part] - theirs[k][part]) / q
            for k in ("inlet", "outlet1", "outlet2") for part in ("pressure", "kinetic")}


def residual_magnitude(config, params, rows):
    """|(momentum_x, momentum_y)| unweighted, at rows (x, y, s)."""
    _, _, _, field = P.build(config)
    out = []
    for i in range(0, len(rows), 2048):
        d = P._row_derivatives(field, params, jnp.asarray(rows[i:i + 2048]))
        rx = pinn.momentum_x(d, config.reynolds, config.pressure_scaling)
        ry = pinn.momentum_y(d, config.reynolds, config.pressure_scaling)
        out.append(np.hypot(np.asarray(rx), np.asarray(ry)))
    return np.concatenate(out)


def reference_points(solution):
    """Gauss points, weights and the reference (u, v, p) there."""
    import skfem as fem
    basis = solution.basis_u
    basis_p = fem.Basis(solution.mesh, solution.basis_p.elem, quadrature=basis.quadrature)
    x, y = basis.global_coordinates().value
    ref_u = basis.interpolate(solution.U).value
    ref_p = basis_p.interpolate(solution.P).value
    return (x.ravel(), y.ravel(), np.asarray(basis.dx).ravel(), ref_u[0].ravel(),
            ref_u[1].ravel(), ref_p.ravel())


def training_summary(run):
    directory = ROOT / run
    history = json.loads((directory / "history.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    last = history[-1]
    return {"selected": metrics["training"]["selected"],
            "selected_heldout_loss": metrics["training"]["selected_heldout_loss"],
            "final_heldout_over_train": last["heldout_loss"] / last["train_loss"],
            "training_minutes": metrics["training"]["total_seconds"] / 60.0,
            "parameters": metrics["training"]["parameter_count"]}


def diagnose(document):
    from bifurcation import reference
    runs = load_runs(document)
    config = runs[0]["config"]
    _, inlet, _, _ = P.build(config)
    a, _ = pinn.scaling_coefficients(config.reynolds, config.pressure_scaling)
    edges = document["corner_bins"]
    rng = np.random.default_rng(document["seed"])
    out = {"config": document, "splits": []}
    for s in document["splits"]:
        g = config.geometry(s)
        solution = reference.solve_bifurcation(
            g, config.reynolds, config.pressure_scaling, inlet, config.outlet1_share,
            config.fe_max_area, config.fe_corner_levels, config.fe_corner_radius,
            continuation=())
        x, y, w, ru, rv, rp = reference_points(solution)
        r = corner_distance(np.stack([x, y], axis=1), g.singular_corners())
        area, speed2, pressure2 = binned(edges, r, w, np.ones_like(x), ru**2 + rv**2, rp**2)
        sample = rng.choice(len(x), size=min(document["residual_points"], len(x)),
                            replace=False)
        theirs = energy_parts(solution, g, a, config.n_line)
        entry = {"split": s, "fe_drop": drop_from_parts(theirs),
                 "area_share": (area / area.sum()).tolist(), "runs": []}
        for run in runs:
            u, v, p = run["model"](x, y, np.full_like(x, s))
            ev2, ep2 = (u - ru) ** 2 + (v - rv) ** 2, (p - rp) ** 2
            err_v, err_p = binned(edges, r, w, ev2, ep2)
            rows = np.stack([x[sample], y[sample], np.full(len(sample), s)], axis=1)
            res = residual_magnitude(run["config"], run["params"], rows)
            local = np.sqrt(ev2[sample])
            which = np.digitize(r[sample], edges) - 1
            res_rms = [float(np.sqrt(np.mean(res[which == k] ** 2))) if np.any(which == k)
                       else None for k in range(len(edges) - 1)]
            mine = energy_parts(at_split(run["model"], s), g, a, config.n_line)
            entry["runs"].append({
                "label": run["label"],
                "relative_l2": {"u": float(np.sqrt(np.sum(w * (u - ru) ** 2) / np.sum(w * ru**2))),
                                "v": float(np.sqrt(np.sum(w * (v - rv) ** 2) / np.sum(w * rv**2))),
                                "p": float(np.sqrt(np.sum(w * ep2) / np.sum(w * rp**2)))},
                "velocity_error_share": (err_v / err_v.sum()).tolist(),
                "pressure_error_share": (err_p / err_p.sum()).tolist(),
                "velocity_error_local": np.sqrt(err_v / speed2).tolist(),
                "residual_rms": res_rms,
                "residual_error_rank_correlation": float(spearmanr(res, local).statistic),
                "pinn_drop": drop_from_parts(mine),
                "drop_error_parts": drop_error_parts(mine, theirs),
            })
        out["splits"].append(entry)
    out["training"] = {run["label"]: training_summary(run["run"]) for run in runs}
    return out


def at_split(model, s):
    return lambda x, y: model(x, y, np.full_like(np.asarray(x, dtype=np.float64), s))


def bin_labels(edges):
    return [f"{a:g}–{b:g}" if b < 50 else f"> {a:g}" for a, b in zip(edges, edges[1:])]


def markdown(result):
    edges = result["config"]["corner_bins"]
    labels = bin_labels(edges)
    lines = ["## Where the parametric PINN's error lives", "",
             "Against the finite-element solution, every integral on the reference's mesh. "
             "Annuli by distance r to the nearest singular corner (B, C, the septum tip).", ""]
    lines += ["### Training", "", "| run | parameters | minutes | selected | held-out loss "
              "| final held-out / train |", "|---|---|---|---|---|---|"]
    for label, t in result["training"].items():
        lines.append(f"| {label} | {t['parameters']} | {t['training_minutes']:.0f} | "
                     f"{t['selected']} | {t['selected_heldout_loss']:.3e} | "
                     f"{t['final_heldout_over_train']:.2f} |")
    for entry in result["splits"]:
        s = entry["split"]
        lines += ["", f"### Split {s:g}", "",
                  "| run | u | v | p | drop error | rank corr. residual-error |",
                  "|---|---|---|---|---|---|"]
        for r in entry["runs"]:
            lines.append(f"| {r['label']} | {r['relative_l2']['u']:.2e} | "
                         f"{r['relative_l2']['v']:.2e} | {r['relative_l2']['p']:.2e} | "
                         f"{100 * (r['pinn_drop'] / entry['fe_drop'] - 1):+.3f}% | "
                         f"{r['residual_error_rank_correlation']:.2f} |")
        head = "| r | area | " + " | ".join(
            f"{r['label']}: vel. err. share, local rel. err., residual RMS"
            for r in entry["runs"]) + " |"
        lines += ["", head, "|---|---|" + "---|" * len(entry["runs"])]
        for k, lab in enumerate(labels):
            cells = []
            for r in entry["runs"]:
                res = r["residual_rms"][k]
                cells.append(f"{100 * r['velocity_error_share'][k]:.1f}%, "
                             f"{r['velocity_error_local'][k]:.2e}, "
                             + ("–" if res is None else f"{res:.2e}"))
            lines.append(f"| {lab} | {100 * entry['area_share'][k]:.2f}% | "
                         + " | ".join(cells) + " |")
        lines += ["", "Drop error by opening (each term divided by the inflow; they sum to "
                  "J_pinn − J_fe):", "", "| run | " + " | ".join(
                      k.replace("_", " ") for k in entry["runs"][0]["drop_error_parts"])
                  + " |", "|---|" + "---|" * len(entry["runs"][0]["drop_error_parts"])]
        for r in entry["runs"]:
            lines.append(f"| {r['label']} | " + " | ".join(
                f"{v:+.2e}" for v in r["drop_error_parts"].values()) + " |")
    return "\n".join(lines) + "\n"


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = result["config"]["corner_bins"]
    centres = [np.sqrt(max(a, 1e-3) * b) if b < 50 else 2 * a for a, b in zip(edges, edges[1:])]
    entries = result["splits"]
    fig, axes = plt.subplots(1, len(entries), figsize=(4.2 * len(entries), 3.8), sharey=True,
                             squeeze=False)
    for ax, entry in zip(axes[0], entries):
        for colour, r in zip(COLOURS, entry["runs"]):
            ax.loglog(centres, r["velocity_error_local"], color=colour, lw=2, marker="o", ms=6,
                      label=r["label"])
        ax.set_title(f"split {entry['split']:g}", fontsize=10, color=INK)
        ax.set_xlabel("distance to the nearest singular corner", fontsize=9, color=INK)
        ax.grid(color=BAND, lw=0.8, which="both")
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("relative RMS velocity error in the annulus", fontsize=9, color=INK)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(config_path, out):
    document = parse_config(yaml.safe_load(pathlib.Path(config_path).read_text()))
    result = diagnose(document)
    out = pathlib.Path(out)
    out.with_suffix(".json").write_text(json.dumps(flow._json_safe(result), indent=1))
    out.with_suffix(".md").write_text(markdown(result))
    plot(result, out.with_suffix(".png"))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run(args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
