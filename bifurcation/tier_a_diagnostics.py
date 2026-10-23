"""Why Tier A failed where it did: four checks on the stored runs, no training.

    python -m bifurcation.tier_a_diagnostics     # -> docs/re_study/tierA/diagnostics.{md,json}

1. Training histories: the held-out loss at the end of Adam, its peak during
   L-BFGS, and where Levenberg-Marquardt took it, per run.
2. Resolution of the narrow daughter: points of one draw (the trainer's own
   sampler) inside each daughter's strip at the range's narrowest split, and
   the mean spacing they give across its width.
3. The error-slope prediction (supplement S1.3): if J_PINN = J_ref (1 + e(s)),
   the optimum shifts by delta ~ -e'(s*) / (J''/J), with e' the least-squares
   slope of e through the three evaluation splits nearest the reference
   optimum and J''/J the curvature of the parabola through J_ref there,
   divided by J_ref at the optimum. Set against the measured argmin error.
4. The friction law's own optimum with each daughter's length re-measured at
   every split (centreline, and closed lengths), against s_L (lengths frozen
   at s_L) and the reference optimum.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import pathlib

import numpy as np
from matplotlib.path import Path as PolyPath
from scipy.optimize import minimize_scalar

from bifurcation import sensitivity, tier_a as T

ROOT = T.ROOT
STUDY = ROOT / "docs" / "re_study"
OUT = STUDY / "tierA"


def summaries():
    """Every exported Tier A, control and earlier 30/90 run, with where it came from."""
    out = []
    for group, pattern in (("Tier A", "tierA/pinn_*_summary.json"),
                           ("control", "tierA_controls/pinn_*_summary.json"),
                           ("amendment 3", "tierA_amend3/pinn_*_summary.json"),
                           ("earlier, q = 0.5", "pinn_30-90_Re*_seed*_summary.json")):
        for f in sorted(glob.glob(str(STUDY / pattern))):
            s = json.loads(pathlib.Path(f).read_text())
            if "failed" in s:
                continue
            s["group"], s["stem"] = group, pathlib.Path(f).name[: -len("_summary.json")]
            s["csv"] = str(pathlib.Path(f).with_name(s["stem"] + ".csv"))
            s["loss_csv"] = str(pathlib.Path(f).with_name(s["stem"] + "_loss.csv"))
            s.setdefault("share", 0.5)
            out.append(s)
    return out


def config_of(s):
    from bifurcation import parametric as P
    return P.load_config(ROOT / s["run"] / "config.yaml")


def label(s):
    c = config_of(s)
    return (f"{s['layout'][0]:g}/{s['layout'][1]:g} q {s['share']:g} Re {s['reynolds']:g} "
            f"s{s['seed']} cut {c.corner_cutoff:g}")


# --------------------------------------------------------------------------- #
# 1. Histories
# --------------------------------------------------------------------------- #


def history(s):
    rows = list(csv.DictReader(open(s["loss_csv"])))
    held = {ph: [float(r["heldout_loss"]) for r in rows if r["phase"] == ph]
            for ph in ("adam", "lbfgs", "lm")}
    return {"adam_end": held["adam"][-1], "lbfgs_peak": max(held["lbfgs"]),
            "lbfgs_end": held["lbfgs"][-1], "lm_start": held["lm"][0], "lm_min": min(held["lm"]),
            "lm_end": held["lm"][-1],
            "lbfgs_jump": max(held["lbfgs"]) / held["adam"][-1],
            "lm_recovery": held["lbfgs"][-1] / min(held["lm"]),
            "selected": s["training"]["selected_heldout_loss"]}


# --------------------------------------------------------------------------- #
# 2. Points across the narrow daughter
# --------------------------------------------------------------------------- #


def strip_points(s, split, draws=8, seed=7):
    """Mean points per draw inside each daughter's strip at `split`, and the
    mean spacing 1/sqrt(density) set against each daughter's width."""
    from bifurcation import flow
    from bifurcation.geometry import corner_points
    c = config_of(s)
    g = c.geometry(split)
    q = corner_points(g.w1, g.w2, math.radians(g.angle1), math.radians(g.angle2),
                      g.parent_length, g.daughter_length)
    tip = q["T"] if "T" in q else 0.5 * (q["start1"] + q["start2"])
    strips = {1: PolyPath(np.array([q["C"], q["E1o"], q["E1i"], tip])),
              2: PolyPath(np.array([q["B"], q["E2o"], q["E2i"], tip]))}
    rng = np.random.default_rng(seed)
    n_corner = int(round(c.points_per_design * c.corner_fraction))
    counts = {1: 0, 2: 0}
    for _ in range(draws):
        pts = np.concatenate([
            flow.uniform_points(rng, g, c.points_per_design - n_corner, c.corner_exclusion),
            flow.corner_points(rng, g, n_corner, c.corner_radius, c.corner_exclusion)])
        for k, path in strips.items():
            counts[k] += int(path.contains_points(pts).sum())
    out = {}
    for k, path in strips.items():
        v = path.vertices
        area = 0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], -1)) - np.dot(v[:, 1], np.roll(v[:, 0], -1)))
        width = g.w1 if k == 1 else g.w2
        n = counts[k] / draws
        spacing = math.sqrt(area / n) if n else math.inf
        out[k] = {"width": width, "points": n, "area": area, "spacing": spacing,
                  "across": width / spacing}
    return out


# --------------------------------------------------------------------------- #
# 3. The error slope
# --------------------------------------------------------------------------- #


def slope_prediction(s):
    rows = [r for r in csv.DictReader(open(s["csv"])) if r["J_fe"]]
    splits = np.array([float(r["s"]) for r in rows])
    j_fe = np.array([float(r["J_fe"]) for r in rows])
    e = np.array([float(r["J_pinn"]) for r in rows]) / j_fe - 1.0
    s_ref = s["s_star_reference_mesh_brent"]
    near = np.argsort(np.abs(splits - s_ref))[:3]
    near = near[np.argsort(splits[near])]
    x, je, ee = splits[near], j_fe[near], e[near]
    slope = np.polyfit(x, ee, 1)[0]
    a = np.polyfit(x, je, 2)[0]
    curvature = 2.0 * a / s["J_fe_at_reference_s_star"]
    return {"splits": x.tolist(), "e_slope": float(slope), "curvature": float(curvature),
            "predicted": float(-slope / curvature), "measured": s["argmin_error"]}


# --------------------------------------------------------------------------- #
# 4. The friction law with lengths at every split
# --------------------------------------------------------------------------- #


def friction_optimum(angles, share, lengths):
    study = {"geometry": T.geometry_config(),
             "physics": {"reynolds": 10.0, "pressure_scaling": "inertial"}}
    lo, hi = T.split_range(T.length_split(angles, share)[0])

    def drop(x):
        l1, l2 = lengths(sensitivity.make_geometry(study, list(angles), x))
        return l1 * share ** 2 / x ** 3 + l2 * (1 - share) ** 2 / (1 - x) ** 3

    return float(minimize_scalar(drop, bounds=(lo, hi), method="bounded",
                                 options={"xatol": 1e-8}).x)


def main():
    runs = summaries()
    result = {"history": [], "resolution": [], "slope": [], "friction": []}
    for s in runs:
        result["history"].append({"run": label(s), "group": s["group"], **history(s)})
        result["slope"].append({"run": label(s), "group": s["group"], **slope_prediction(s)})
    seen = set()
    for s in runs:
        key = (tuple(s["layout"]), s["share"], tuple(s.get("split_range") or (0.35, 0.65)))
        if key in seen or s["reynolds"] != 10.0:
            continue
        seen.add(key)
        lo, hi = key[2]
        narrow = lo if s["share"] <= 0.5 else hi
        for at in sorted({narrow, 0.5 * (lo + hi)}):
            r = strip_points(s, at)
            result["resolution"].append({"case": f"{key[0][0]:g}/{key[0][1]:g} q {key[1]:g}",
                                         "split": at, **{f"d{k}_{m}": v[m] for k, v in r.items()
                                                         for m in v}})
    for angles, share in sorted({(tuple(s["layout"]), s["share"]) for s in runs}):
        s_l = T.length_split(angles, share)[0]
        refs = {s["reynolds"]: s["s_star_reference_mesh_brent"] for s in runs
                if tuple(s["layout"]) == angles and s["share"] == share and s["group"] != "control"}
        result["friction"].append({
            "case": f"{angles[0]:g}/{angles[1]:g} q {share:g}", "s_M": sensitivity.murray_split(share),
            "s_L": s_l,
            "friction_centreline": friction_optimum(angles, share, sensitivity.centreline_lengths),
            "friction_closed": friction_optimum(angles, share, sensitivity.closed_lengths),
            "reference": refs})
    (OUT / "diagnostics.json").write_text(json.dumps(result, indent=1))
    (OUT / "diagnostics.md").write_text(markdown(result))
    print(markdown(result))


def markdown(r):
    L = ["## Tier A diagnostics (bifurcation/tier_a_diagnostics.py)", "",
         "### 1. Held-out loss through training", "",
         "| run | group | end of Adam | L-BFGS peak (× Adam end) | end of L-BFGS | LM minimum | "
         "L-BFGS end / LM minimum | selected |", "|---|---|---|---|---|---|---|---|"]
    for h in r["history"]:
        L.append(f"| {h['run']} | {h['group']} | {h['adam_end']:.3g} | {h['lbfgs_peak']:.3g} "
                 f"({h['lbfgs_jump']:.1f}×) | {h['lbfgs_end']:.3g} | {h['lm_min']:.3g} | "
                 f"{h['lm_recovery']:.1f} | {h['selected']:.3g} |")
    L += ["", "### 2. Collocation points in each daughter's strip (one draw of 1,024 per split, "
          "mean of 8)", "",
          "| case | split | daughter 1: width, points, spacing, across | daughter 2: width, points, "
          "spacing, across |", "|---|---|---|---|"]
    for x in r["resolution"]:
        L.append(f"| {x['case']} | {x['split']:.4f} | {x['d1_width']:.3f}, {x['d1_points']:.0f}, "
                 f"{x['d1_spacing']:.3f}, {x['d1_across']:.1f} | {x['d2_width']:.3f}, "
                 f"{x['d2_points']:.0f}, {x['d2_spacing']:.3f}, {x['d2_across']:.1f} |")
    L += ["", "'across' = width / mean spacing: the points one draw puts across the daughter.", "",
          "### 3. Error slope against the measured argmin error (S1.3)", "",
          "| run | group | e′ (per unit s) | J″/J | predicted δs | measured | measured − predicted |",
          "|---|---|---|---|---|---|---|"]
    for x in r["slope"]:
        L.append(f"| {x['run']} | {x['group']} | {100 * x['e_slope']:+.2f}% | {x['curvature']:.1f} | "
                 f"{x['predicted']:+.4f} | {x['measured']:+.4f} | "
                 f"{x['measured'] - x['predicted']:+.4f} |")
    L += ["", "### 4. The friction law with lengths re-measured at every split", "",
          "| case | s_M | s_L (frozen) | friction law, centreline lengths at every s | "
          "closed lengths at every s | reference optimum (Re_h: s*) |", "|---|---|---|---|---|---|"]
    for x in r["friction"]:
        refs = ", ".join(f"{k:g}: {v:.4f}" for k, v in sorted(x["reference"].items()))
        L.append(f"| {x['case']} | {x['s_M']:.4f} | {x['s_L']:.4f} | {x['friction_centreline']:.4f} | "
                 f"{x['friction_closed']:.4f} | {refs} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# 3b. The error slope measured locally: reference solves at s*_ref +- H
# --------------------------------------------------------------------------- #

LOCAL_STEP = 0.01


def local_slope(s, cache):
    """e(s) = J_PINN/J_ref - 1 at s*_ref - H, s*_ref, s*_ref + H, J_ref on the
    reference mesh (one solve each, cached), e' by central difference and J''/J
    by the central second difference of J_ref."""
    from bifurcation import flow, parametric as P, reference
    from developing_flow import pinn
    key = s["run"]
    if key in cache:
        return cache[key]
    model, c, _ = P.load_run(ROOT / s["run"])
    st, inlet, _, _ = P.build(c)
    a, _ = pinn.scaling_coefficients(c.reynolds, c.pressure_scaling)
    centre = s["s_star_reference_mesh_brent"]
    out = {"splits": [], "J_ref": [], "J_pinn": []}
    for x in (centre - LOCAL_STEP, centre, centre + LOCAL_STEP):
        g = c.geometry(x)
        sol = reference.solve_bifurcation(g, c.reynolds, c.pressure_scaling, inlet,
                                          c.outlet1_share, c.fe_max_area, c.fe_corner_levels,
                                          c.fe_corner_radius, continuation=c.fe_continuation)
        out["splits"].append(x)
        out["J_ref"].append(flow.opening_measures(sol, g, a, c.n_line)["total_pressure_drop"])
        out["J_pinn"].append(flow.opening_measures(P.at_split(model, x), g, a,
                                                   c.n_line)["total_pressure_drop"])
    jr, jp = np.array(out["J_ref"]), np.array(out["J_pinn"])
    e = jp / jr - 1.0
    out["e"] = e.tolist()
    out["e_slope"] = float((e[2] - e[0]) / (2 * LOCAL_STEP))
    out["curvature"] = float((jr[2] - 2 * jr[1] + jr[0]) / LOCAL_STEP ** 2 / jr[1])
    out["predicted"] = -out["e_slope"] / out["curvature"]
    out["measured"] = s["argmin_error"]
    cache[key] = out
    return out


def local_main():
    path = OUT / "diagnostics_local_slope.json"
    cache = json.loads(path.read_text()) if path.is_file() else {}
    for s in summaries():
        if s["group"] == "earlier, q = 0.5" and s["reynolds"] not in (0.1, 10.0):
            continue
        r = local_slope(s, cache)
        path.write_text(json.dumps(cache, indent=1))
        print(f"{label(s)}: e' {100 * r['e_slope']:+.2f}%/s, J''/J {r['curvature']:.1f}, "
              f"predicted {r['predicted']:+.5f}, measured {r['measured']:+.5f}", flush=True)


def local_markdown():
    """Section 3b, from the cached local solves."""
    cache = json.loads((OUT / "diagnostics_local_slope.json").read_text())
    by_run = {s["run"]: s for s in summaries()}
    L = ["", "### 3b. The error slope measured at the optimum (reference solves at s*_ref ± 0.01)",
         "", "| run | group | e′ (per unit s) | J″/J | predicted δs | measured | measured / predicted |",
         "|---|---|---|---|---|---|---|"]
    for run, r in cache.items():
        s = by_run[run]
        ratio = r["measured"] / r["predicted"] if abs(r["predicted"]) > 1e-12 else math.nan
        L.append(f"| {label(s)} | {s['group']} | {100 * r['e_slope']:+.2f}% | {r['curvature']:.1f} | "
                 f"{r['predicted']:+.5f} | {r['measured']:+.5f} | {ratio:.2f} |")
    L += ["", "The three-point estimate of section 3 uses evaluation splits 0.05 apart; this one "
          "takes e′ by central difference over ±0.01, where e is close to linear."]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# 5. What building s_L costs: J_ref(s_L)/J* - 1, reference mesh
# --------------------------------------------------------------------------- #


def s_l_cost():
    """For 30/90 at every Tier A share and both Reynolds numbers: one reference
    solve at s_L, against the reference-mesh optimum J* (Brent, from the Tier A
    run's evaluation)."""
    from bifurcation import flow, parametric as P, reference
    from developing_flow import pinn
    path = OUT / "diagnostics_s_l_cost.json"
    out = json.loads(path.read_text()) if path.is_file() else {}
    for s in summaries():
        if s["group"] != "Tier A" or tuple(s["layout"]) != (30.0, 90.0):
            continue
        key = f"q{s['share']:g}_Re{s['reynolds']:g}"
        if key in out:
            continue
        c = config_of(s)
        s_l = T.length_split((30.0, 90.0), s["share"])[0]
        g = c.geometry(s_l)
        _, inlet, _, _ = P.build(c)
        a, _ = pinn.scaling_coefficients(c.reynolds, c.pressure_scaling)
        sol = reference.solve_bifurcation(g, c.reynolds, c.pressure_scaling, inlet,
                                          c.outlet1_share, c.fe_max_area, c.fe_corner_levels,
                                          c.fe_corner_radius, continuation=c.fe_continuation)
        j_l = flow.opening_measures(sol, g, a, c.n_line)["total_pressure_drop"]
        out[key] = {"share": s["share"], "reynolds": s["reynolds"], "s_L": s_l,
                    "s_ref": s["s_star_reference_mesh_brent"], "J_s_L": j_l,
                    "J_star": s["J_fe_at_reference_s_star"],
                    "cost": j_l / s["J_fe_at_reference_s_star"] - 1.0}
        path.write_text(json.dumps(out, indent=1))
        print(key, out[key], flush=True)
