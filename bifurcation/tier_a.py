"""Tier A of the Reynolds-number study: new shares and a symmetric control.

    python -m bifurcation.tier_a screen      # reference solves, daughters 4 long
    python -m bifurcation.tier_a configs     # parametric and answer-key configs for the cases that pass

The rules are fixed in docs/re_study/tierA/preregistration.md, committed before
any of this ran: each case's trained range is s_L +- 0.15 inside [0.1, 0.9],
s_L worked out from the geometry alone; a case is trained only if the reference
solver finds an interior minimum, inside that range, with daughters 4 long.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import multiprocessing
import os
import pathlib
import tempfile

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "docs" / "re_study" / "tierA"
TEMPLATE = HERE / "p_re10_3090_corners_s22.yaml"      # the 13 earlier runs' network and schedule
ANSWER_TEMPLATE = HERE / "answer_keys_re.yaml"        # their answer keys' meshes and search

CASES = [((30.0, 90.0), 0.2), ((30.0, 90.0), 0.35), ((30.0, 90.0), 0.65), ((30.0, 90.0), 0.8),
         ((15.0, 15.0), 0.2), ((0.0, 120.0), 0.5)]
FLOWS = [{"reynolds": 0.1, "pressure_scaling": "viscous", "continuation": []},
         {"reynolds": 10.0, "pressure_scaling": "inertial", "continuation": []}]
SEED = 21
HALF_RANGE = 0.15
SPLIT_LIMITS = (0.1, 0.9)
EVALUATION_SPLITS = 7
EDGE_MARGIN = 0.05        # screening: the optimum at least this far inside the range
CUTOFF = 0.5              # the earlier runs' corner cutoff ...
CUTOFF_MARGIN = 0.2       # ... which was this short of the nearest other edge
CUTOFF_STEP = 0.05


def tag(angles, share):
    return f"{angles[0]:g}-{angles[1]:g}_q{share:g}"


def run_tag(angles, share, reynolds):
    return f"re{reynolds:g}_{angles[0]:g}{angles[1]:g}_q{share:g}_corners_s{SEED}"


def geometry_config():
    return dict(yaml.safe_load(ANSWER_TEMPLATE.read_text())["geometry"])


def length_split(angles, share):
    """s_L and the centreline lengths it is measured with, from the geometry alone."""
    from bifurcation import sensitivity
    study = {"geometry": geometry_config(),
             "physics": {"reynolds": 10.0, "pressure_scaling": "inertial"}}
    return sensitivity.length_corrected_split(study, list(angles), share, 1e-12)


def split_range(s_l):
    lo = round(max(SPLIT_LIMITS[0], s_l - HALF_RANGE), 4)
    hi = round(min(SPLIT_LIMITS[1], s_l + HALF_RANGE), 4)
    return lo, hi


def evaluation_splits(lo, hi):
    return [round(float(x), 4) for x in np.linspace(lo, hi, EVALUATION_SPLITS)]


def parametric_document(angles, share, flow, cutoff):
    """The earlier runs' config with this case's layout, share, flow and range."""
    doc = yaml.safe_load(TEMPLATE.read_text())
    s_l, _, _ = length_split(angles, share)
    lo, hi = split_range(s_l)
    doc["name"] = f"parametric_{run_tag(angles, share, flow['reynolds'])}"
    doc["seed"] = SEED
    doc["physics"] = {"reynolds": flow["reynolds"], "pressure_scaling": flow["pressure_scaling"]}
    doc["geometry"]["angle1"], doc["geometry"]["angle2"] = float(angles[0]), float(angles[1])
    doc["design"].update(split_min=lo, split_max=hi, outlet1_share=float(share))
    doc["model"]["corner_cutoff"] = float(cutoff)
    doc["evaluation"]["splits"] = evaluation_splits(lo, hi)
    doc["evaluation"]["continuation"] = list(flow["continuation"])
    return doc


def _loads(doc):
    from bifurcation import parametric as P
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(doc, f)
        path = f.name
    try:
        P.load_config(path)
        return True
    except P.ConfigError:
        return False
    finally:
        os.unlink(path)


def corner_cutoff(angles, share):
    """min(CUTOFF, R - CUTOFF_MARGIN) rounded down to CUTOFF_STEP, R the largest
    cutoff the geometry admits across the range (parametric.singular_modes)."""
    doc = parametric_document(angles, share, FLOWS[1], CUTOFF)
    lo, hi = 0.01, 3.0
    if not _loads({**doc, "model": {**doc["model"], "corner_cutoff": lo}}):
        raise RuntimeError(f"{tag(angles, share)}: no corner cutoff fits")
    while hi - lo > 1e-3:
        mid = 0.5 * (lo + hi)
        ok = _loads({**doc, "model": {**doc["model"], "corner_cutoff": mid}})
        lo, hi = (mid, hi) if ok else (lo, mid)
    chosen = min(CUTOFF, math.floor((lo - CUTOFF_MARGIN) / CUTOFF_STEP + 1e-9) * CUTOFF_STEP)
    if chosen <= 0.0:
        raise RuntimeError(f"{tag(angles, share)}: the largest cutoff, {lo:.3f}, leaves no margin")
    return round(chosen, 4), lo


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #


def screen_case(task):
    """The reference optimum over [0.1, 0.9] with daughters 4 long, and the verdict."""
    from bifurcation import answer_key, design_study
    angles, share, flow = task
    config = answer_key.parse_config(yaml.safe_load(ANSWER_TEMPLATE.read_text()))
    row = design_study.optimise((answer_key.study_config(config, flow), list(angles), share))
    s_l, l1, l2 = length_split(angles, share)
    lo, hi = split_range(s_l)
    out = {"layout": list(angles), "share": share, "reynolds": flow["reynolds"],
           "s_L": s_l, "L1": l1, "L2": l2, "range": [lo, hi], "optimum": row}
    reasons = []
    if not row["feasible"]:
        reasons.append("no feasible split")
    else:
        s = row["best_split"]
        out["s_fe"] = s
        if row["on_bound"]:
            reasons.append(f"optimum {s:.4f} on the search bound")
        limit = answer_key.feasibility_limit(row)
        if limit is not None:
            reasons.append(f"optimum {s:.4f} at the geometry's limit (split {limit:.5f} refused)")
        if not row["minimum_survives_refinement"]:
            reasons.append("the minimum does not survive the check mesh")
        if not row.get("steady_bracket", True):
            reasons.append("an unsolved split beside the optimum")
        if not lo + EDGE_MARGIN <= s <= hi - EDGE_MARGIN:
            reasons.append(f"optimum {s:.4f} not inside [{lo + EDGE_MARGIN:.4f}, "
                           f"{hi - EDGE_MARGIN:.4f}]")
    out["passes"] = not reasons
    out["reasons"] = reasons
    return out


def screen(workers=2):
    OUT.mkdir(parents=True, exist_ok=True)
    cases_dir = OUT / "screening_cases"
    cases_dir.mkdir(exist_ok=True)
    tasks = [(angles, share, flow) for angles, share in CASES for flow in FLOWS]
    todo = [t for t in tasks
            if not (cases_dir / f"{tag(t[0], t[1])}_Re{t[2]['reynolds']:g}.json").is_file()]
    with concurrent.futures.ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in concurrent.futures.as_completed([pool.submit(screen_case, t) for t in todo]):
            r = future.result()
            name = f"{tag(r['layout'], r['share'])}_Re{r['reynolds']:g}.json"
            (cases_dir / name).write_text(json.dumps(r, indent=1, default=str))
            print(f"  {name}: {'pass' if r['passes'] else 'FAIL: ' + '; '.join(r['reasons'])}",
                  flush=True)
    results = [json.loads((cases_dir / f"{tag(a, q)}_Re{f['reynolds']:g}.json").read_text())
               for (a, q, f) in tasks]
    (OUT / "screening.json").write_text(json.dumps(results, indent=1, default=str))
    (OUT / "screening.md").write_text(screening_markdown(results))
    return results


def screening_markdown(results):
    from bifurcation import sensitivity
    lines = ["## Tier A screening: the reference optimum with daughters 4 long", "",
             "Rules: docs/re_study/tierA/preregistration.md. Search [0.1, 0.9], 17-point scan, "
             "Brent to ±0.001 on the search mesh, check-mesh re-solve.", "",
             "| case | q | Re_h | s_M | s_L | trained range | s_FE | s_FE − s_L | J at s_FE | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        s = r.get("s_fe")
        lines.append(
            f"| {r['layout'][0]:g}/{r['layout'][1]:g} | {r['share']:g} | {r['reynolds']:g} | "
            f"{sensitivity.murray_split(r['share']):.4f} | {r['s_L']:.4f} | "
            f"[{r['range'][0]:.4f}, {r['range'][1]:.4f}] | "
            + ("—" if s is None else f"{s:.4f}") + " | "
            + ("—" if s is None else f"{s - r['s_L']:+.4f}") + " | "
            + ("—" if s is None else f"{r['optimum']['total_pressure_drop']:.6g}") + " | "
            + ("train" if r["passes"] else "SKIP: " + "; ".join(r["reasons"])) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #


def write_configs():
    results = json.loads((OUT / "screening.json").read_text())
    passed = {(tuple(r["layout"]), r["share"], r["reynolds"]) for r in results if r["passes"]}
    written = []
    for angles, share in CASES:
        cutoff, admitted = corner_cutoff(angles, share)
        flows = [f for f in FLOWS if (tuple(angles), share, f["reynolds"]) in passed]
        for flow in FLOWS:
            doc = parametric_document(angles, share, flow, cutoff)
            path = HERE / f"p_{run_tag(angles, share, flow['reynolds'])}.yaml"
            head = (f"# Tier A (docs/re_study/tierA/preregistration.md): the 13 earlier runs'\n"
                    f"# network and schedule; layout {angles[0]:g}/{angles[1]:g}, share {share:g},\n"
                    f"# Re_h {flow['reynolds']:g}. Range s_L +- 0.15 with s_L = "
                    f"{length_split(angles, share)[0]:.4f};\n# corner cutoff {cutoff:g} "
                    f"(the geometry admits {admitted:.3f} across the range).\n")
            path.write_text(head + yaml.safe_dump(doc, sort_keys=False))
            written.append(path)
            if flow not in flows:      # failed screening: recorded, never trained
                r = next(x for x in results if tuple(x["layout"]) == tuple(angles)
                         and x["share"] == share and x["reynolds"] == flow["reynolds"])
                run = HERE / "runs" / doc["name"]
                run.mkdir(parents=True, exist_ok=True)
                (run / "SKIPPED").write_text(
                    "failed the pre-registered screening (docs/re_study/tierA/screening.md): "
                    + "; ".join(r["reasons"]) + "\n")
                written.append(run / "SKIPPED")
        if flows:
            answer = yaml.safe_load(ANSWER_TEMPLATE.read_text())
            answer["name"] = f"answer_keys_tierA_{tag(angles, share)}"
            answer["share"] = float(share)
            answer["layouts"] = [[float(angles[0]), float(angles[1])]]
            answer["flows"] = flows
            lo, hi = split_range(length_split(angles, share)[0])
            answer["reference"]["splits"] = evaluation_splits(lo, hi)
            path = HERE / f"answer_keys_tierA_{tag(angles, share)}.yaml"
            path.write_text("# Tier A answer keys for the cases that passed screening.\n"
                            + yaml.safe_dump(answer, sort_keys=False))
            written.append(path)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["screen", "configs"])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "screen":
        screen(args.workers)
    else:
        for path in write_configs():
            print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------- #
# Controls (preregistration amendment 1): the share against the corner cutoff,
# and a second seed
# --------------------------------------------------------------------------- #

CONTROLS = [  # (angles, share, Re_h, cutoff, seed): run in this order
    ((30.0, 90.0), 0.5, 0.1, 0.15, SEED),    # q = 0.5 at a small cutoff
    ((30.0, 90.0), 0.35, 0.1, 0.40, SEED),   # off-centre at a large cutoff
    ((30.0, 90.0), 0.5, 10.0, 0.15, SEED),
    ((30.0, 90.0), 0.35, 10.0, 0.40, SEED),
    ((30.0, 90.0), 0.2, 10.0, None, 22),     # Tier A's own rule, another seed
]


def control_tag(angles, share, reynolds, cutoff, seed):
    cut = "" if cutoff is None else f"_cut{cutoff:g}"
    return f"re{reynolds:g}_{angles[0]:g}{angles[1]:g}_q{share:g}{cut}_corners_s{seed}"


def write_control_configs():
    written = []
    for angles, share, reynolds, cutoff, seed in CONTROLS:
        flow = next(f for f in FLOWS if f["reynolds"] == reynolds)
        rule, admitted = corner_cutoff(angles, share)
        used = rule if cutoff is None else cutoff
        doc = parametric_document(angles, share, flow, used)
        tag = control_tag(angles, share, reynolds, cutoff, seed)
        doc["name"], doc["seed"] = f"parametric_{tag}", seed
        if not _loads(doc):
            raise RuntimeError(f"{tag}: the geometry does not admit cutoff {used}")
        path = HERE / f"p_{tag}.yaml"
        path.write_text(
            "# Tier A control (docs/re_study/tierA/preregistration.md, amendment 1): layout "
            f"{angles[0]:g}/{angles[1]:g},\n# share {share:g}, Re_h {reynolds:g}, seed {seed}; "
            f"corner cutoff {used:g} (the rule gives {rule:g};\n# the geometry admits "
            f"{admitted:.3f} across the range).\n" + yaml.safe_dump(doc, sort_keys=False))
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# Amendment 3: the narrow daughter's sampling, and Levenberg-Marquardt given time
# --------------------------------------------------------------------------- #

SAMPLING_RUNS = [((30.0, 90.0), 0.35), ((30.0, 90.0), 0.2)]   # Re_h 10, seed 21, rule's cutoff
LM_PARENT = "parametric_re0.1_3090_q0.5_cut0.15_corners_s21"
LM_EXTRA_STEPS = 250                                         # 150 + 250 = 400 LM steps


def write_amendment3_configs():
    written = []
    flow10 = next(f for f in FLOWS if f["reynolds"] == 10.0)
    for angles, share in SAMPLING_RUNS:
        cutoff, admitted = corner_cutoff(angles, share)
        doc = parametric_document(angles, share, flow10, cutoff)
        doc["sampling"]["density"] = "width"
        tag = f"re10_{angles[0]:g}{angles[1]:g}_q{share:g}_width_corners_s{SEED}"
        doc["name"] = f"parametric_{tag}"
        if not _loads(doc):
            raise RuntimeError(f"{tag} does not load")
        path = HERE / f"p_{tag}.yaml"
        path.write_text(
            "# Tier A amendment 3: Tier A's run with interior points scaled with channel\n"
            f"# width (density ~ 1/w^2, the same total): layout {angles[0]:g}/{angles[1]:g}, "
            f"share {share:g}, Re_h 10,\n# seed {SEED}, corner cutoff {cutoff:g} by the rule.\n"
            + yaml.safe_dump(doc, sort_keys=False))
        written.append(path)
    parent = yaml.safe_load((HERE / "runs" / LM_PARENT / "config.yaml").read_text())
    doc = parent
    doc["name"] = LM_PARENT.replace("_corners_s21", "_lm400_corners_s22")
    doc["seed"] = 22               # not the parent's: its held-out points selected it
    doc["training"]["adam"]["steps"] = 0
    doc["training"]["lbfgs"]["cycles"] = 0
    doc["training"]["lm"]["iterations"] = LM_EXTRA_STEPS
    doc["training"]["init_from"] = LM_PARENT
    if not _loads(doc):
        raise RuntimeError(f"{doc['name']} does not load")
    path = HERE / f"p_{doc['name'][len('parametric_'):]}.yaml"
    path.write_text(
        f"# Tier A amendment 3: {LM_PARENT} continued for {LM_EXTRA_STEPS} more\n"
        "# Levenberg-Marquardt steps (400 in all), from its selected network; no Adam or\n"
        "# L-BFGS. Seed 22, not the parent's 21. Same cutoff, range and sampling.\n"
        + yaml.safe_dump(doc, sort_keys=False))
    written.append(path)
    return written
