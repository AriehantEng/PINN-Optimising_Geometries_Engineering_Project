"""Export one finished parametric run for the Reynolds-number study (docs/re_study).

    python -m bifurcation.re_export --run bifurcation/runs/<name> --out docs/re_study
    python -m bifurcation.re_export --index docs/re_study      # the table of every run

For a run that finished (params.npz, metrics.json, history.json), three files
named pinn_<a1>-<a2>_Re<Re>_seed<seed>:

    .csv          s, J_pinn, J_fe over the run's scan (evaluation.curve_points);
                  J_fe only where the reference was solved (evaluation.splits)
    _loss.csv     the training history: phase, iteration, step, seconds,
                  training loss, held-out loss, held-out loss of the iterate
                  average (LM only)
    _summary.json / .md
                  the PINN's s* (scan, then Brent) and its J; the loss at s*
                  (the training loss, every term, on 4 x 1,024 interior points
                  at s* from a fixed stream); relative L2 errors of u, v, p and
                  the signed drop error against the reference at the
                  evaluation splits and at s* (one more reference solve);
                  training: wall time, final training loss, selected held-out
                  loss, seed, and whether it converged (defined in the file).

A run with no params.npz is exported as failed, with the reason given.
J is in the run's pressure units: rho U_c^2 (inertial) or mu U_c / h (viscous).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

#: The held-out loss counts as still falling if its best over the last
#: STILL_FALLING_STEPS LM steps is more than STILL_FALLING_DROP below its best
#: before them.
STILL_FALLING_STEPS = 25
STILL_FALLING_DROP = 0.10
LOSS_AT_OPTIMUM_DESIGNS = 4
LOSS_AT_OPTIMUM_SEED = 2026


def stem(config):
    return (f"pinn_{config.angle1:g}-{config.angle2:g}_Re{config.reynolds:g}"
            f"_seed{config.seed}")


def batch_at(rng, config, inlet, s, designs):
    """sample_batch's draw, every design at split s."""
    from bifurcation import flow, parametric as P
    original = P.design_splits
    try:
        P.design_splits = lambda rng_, config_, n: np.full(n, float(s))
        return P.sample_batch(rng, config, inlet, designs)
    finally:
        P.design_splits = original


def history_rows(history, config):
    rows = []
    for e in history:
        phase, it = e["phase"], e["iteration"]
        step = {"adam": it, "lbfgs": config.adam_steps + it * config.lbfgs_iterations,
                "lm": config.adam_steps + config.lbfgs_cycles * config.lbfgs_iterations + it}[phase]
        rows.append({"phase": phase, "iteration": it, "step": step, "seconds": e["seconds"],
                     "train_loss": e["train_loss"], "heldout_loss": e["heldout_loss"],
                     "heldout_average": e.get("heldout_average")})
    return rows


def converged(history, metrics_training):
    lm = [e for e in history if e["phase"] == "lm"]
    finite = all(math.isfinite(e["heldout_loss"]) for e in history)
    if len(lm) <= STILL_FALLING_STEPS or not finite:
        return {"converged": False, "finite": finite,
                "why": "too few LM steps to judge" if finite else "a loss became non-finite"}
    held = [min(e["heldout_loss"], e.get("heldout_average") or math.inf) for e in lm]
    before = min(held[:-STILL_FALLING_STEPS])
    last = min(held[-STILL_FALLING_STEPS:])
    falling = last < (1.0 - STILL_FALLING_DROP) * before
    return {"converged": not falling, "finite": True, "still_falling": falling,
            "best_heldout_before_last_steps": before, "best_heldout_last_steps": last,
            "definition": f"finished, every loss finite, and the best held-out loss of the "
                          f"last {STILL_FALLING_STEPS} LM steps not more than "
                          f"{100 * STILL_FALLING_DROP:.0f}% below the best before them",
            "lm_stalled": metrics_training.get("levenberg_marquardt", {}).get("stalled")}


def export(run_directory, out_directory):
    import jax
    jax.config.update("jax_enable_x64", True)
    from bifurcation import flow, parametric as P, reference
    from developing_flow import pinn

    run_directory, out = pathlib.Path(run_directory), pathlib.Path(out_directory)
    out.mkdir(parents=True, exist_ok=True)
    config = P.load_config(run_directory / "config.yaml")
    name = stem(config)
    stopped = [run_directory / n for n in ("FAILED", "SKIPPED") if (run_directory / n).is_file()]
    if stopped or not (run_directory / "metrics.json").is_file():
        reason = (f"{stopped[0].name}: {stopped[0].read_text()}" if stopped
                  else "no metrics.json: the run did not finish")
        summary = {"run": str(run_directory), "seed": config.seed, "reynolds": config.reynolds,
                   "layout": [config.angle1, config.angle2], "failed": reason}
        (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=1))
        (out / f"{name}_summary.md").write_text(f"## {name}\n\nFAILED: {reason}\n")
        return summary

    model, config, params = P.load_run(run_directory)
    metrics = json.loads((run_directory / "metrics.json").read_text())
    history = json.loads((run_directory / "history.json").read_text())
    o = metrics["optimum"]

    # the scan, with the reference where it was solved
    fe_at = {round(r["split"], 9): r["reference_drop"] for r in metrics["splits"]}
    lines = ["s,J_pinn,J_fe"]
    for s, j in zip(o["curve"]["split"], o["curve"]["pinn_drop"]):
        fe = fe_at.get(round(s, 9))
        lines.append(f"{s:.6f},{j!r},{'' if fe is None else repr(fe)}")
    (out / f"{name}.csv").write_text("\n".join(lines) + "\n")

    rows = history_rows(history, config)
    keys = ["phase", "iteration", "step", "seconds", "train_loss", "heldout_loss",
            "heldout_average"]
    (out / f"{name}_loss.csv").write_text("\n".join(
        [",".join(keys)] + [",".join("" if r[k] is None else str(r[k]) for k in keys)
                            for r in rows]) + "\n")

    # the loss at s*, and the fields there against one more reference solve
    s_star = o["pinn_split"]
    st, inlet, _, field = P.build(config)
    _, batched_loss = P.make_residuals(config, st, field)
    batch = batch_at(np.random.default_rng(LOSS_AT_OPTIMUM_SEED), config, inlet, s_star,
                     LOSS_AT_OPTIMUM_DESIGNS)
    loss_at_optimum = float(batched_loss(params, batch))
    g = config.geometry(s_star)
    solution = reference.solve_bifurcation(
        g, config.reynolds, config.pressure_scaling, inlet, config.outlet1_share,
        config.fe_max_area, config.fe_corner_levels, config.fe_corner_radius,
        continuation=config.fe_continuation)
    a, _ = pinn.scaling_coefficients(config.reynolds, config.pressure_scaling)
    at_star = P.at_split(model, s_star)
    errors_star = flow.relative_errors(solution, at_star)
    mine = flow.opening_measures(at_star, g, a, config.n_line)["total_pressure_drop"]
    theirs = flow.opening_measures(solution, g, a, config.n_line)["total_pressure_drop"]

    fields = [{"split": r["split"], "u": r["relative_l2"]["u"], "v": r["relative_l2"]["v"],
               "p": r["relative_l2"]["p"], "drop_error": r["drop_error"]}
              for r in metrics["splits"]]
    fields.append({"split": s_star, "at": "PINN s*", "u": errors_star["u"], "v": errors_star["v"],
                   "p": errors_star["p"], "drop_error": mine / theirs - 1.0})
    t = metrics["training"]
    lm = [e for e in history if e["phase"] == "lm"]
    summary = {
        "run": str(run_directory), "layout": [config.angle1, config.angle2],
        "reynolds": config.reynolds, "pressure_scaling": config.pressure_scaling,
        "seed": config.seed,
        "s_star_pinn": s_star, "J_pinn_at_s_star": o["pinn_min_drop"],
        "s_star_pinn_on_bound": o["pinn_on_bound"],
        "loss_at_s_star": loss_at_optimum,
        "loss_at_s_star_points": f"{LOSS_AT_OPTIMUM_DESIGNS} x {config.points_per_design} "
                                 f"interior points at s*, stream {LOSS_AT_OPTIMUM_SEED}",
        "J_fe_at_s_star_pinn": theirs,
        "s_star_reference_mesh_brent": o["reference_split"],
        "J_fe_at_reference_s_star": o["reference_min_drop"],
        "argmin_error": o["argmin_error"], "regret": o["regret"],
        "fields": fields,
        "training": {"wall_seconds": t["total_seconds"],
                     "final_training_loss": lm[-1]["train_loss"] if lm else history[-1]["train_loss"],
                     "selected": t["selected"], "selected_heldout_loss": t["selected_heldout_loss"],
                     "parameters": t["parameter_count"], **converged(history, t)},
    }
    (out / f"{name}_summary.json").write_text(json.dumps(flow._json_safe(summary), indent=1))
    (out / f"{name}_summary.md").write_text(summary_markdown(name, summary))
    return summary


def summary_markdown(name, s):
    t = s["training"]
    lines = [f"## {name}", "",
             f"Layout {s['layout'][0]:g}/{s['layout'][1]:g}, Re_h = {s['reynolds']:g} "
             f"({s['pressure_scaling']} pressure), seed {s['seed']}.", "",
             f"- PINN optimum s* = {s['s_star_pinn']:.4f} (scan, then Brent), J = "
             f"{s['J_pinn_at_s_star']:.6g}; on the range's bound: {s['s_star_pinn_on_bound']}",
             f"- finite elements at the PINN's s*: J = {s['J_fe_at_s_star_pinn']:.6g} (reference mesh)",
             f"- loss at s*: {s['loss_at_s_star']:.3e} ({s['loss_at_s_star_points']})",
             f"- reference-mesh optimum (Brent from the evaluation splits): "
             f"{s['s_star_reference_mesh_brent']:.4f} (J = {s['J_fe_at_reference_s_star']:.6g}); "
             f"argmin error {s['argmin_error']:+.4f}; "
             f"regret {100 * s['regret']:+.4f}%", "",
             "| s | u | v | p | drop error |", "|---|---|---|---|---|"]
    for f in s["fields"]:
        label = f"{f['split']:.4f} (s*)" if f.get("at") else f"{f['split']:.2f}"
        lines.append(f"| {label} | {f['u']:.2e} | {f['v']:.2e} | {f['p']:.2e} | "
                     f"{100 * f['drop_error']:+.3f}% |")
    lines += ["", f"Training: {t['wall_seconds'] / 60:.0f} min, final training loss "
                  f"{t['final_training_loss']:.3e}, selected {t['selected']} with held-out loss "
                  f"{t['selected_heldout_loss']:.3e}; converged: "
                  + ("yes" if t["converged"] else "no")
                  + (": the held-out loss was still falling when the fixed schedule ended"
                     if t.get("still_falling") else "")
                  + (f" (definition: {t['definition']})" if "definition" in t
                     else f" ({t.get('why')})")]
    return "\n".join(lines) + "\n"


def index(directory):
    directory = pathlib.Path(directory)
    rows = [json.loads(p.read_text()) for p in sorted(directory.glob("pinn_*_summary.json"))]
    rows.sort(key=lambda r: (r["layout"][1], r["reynolds"], r["seed"]))
    lines = ["## PINN runs of the Reynolds-number study", "",
             "| layout | Re_h | seed | s* (PINN) | regret | drop error at 0.35 / 0.50 / 0.65 | "
             "u error at 0.50 | training (min) | converged |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        head = f"| {r['layout'][0]:g}/{r['layout'][1]:g} | {r['reynolds']:g} | {r['seed']} |"
        if "failed" in r:
            kind = "SKIPPED" if r["failed"].startswith("SKIPPED") else "FAILED"
            why = r["failed"].split(": ", 1)[-1] if kind == "SKIPPED" else r["failed"]
            lines.append(f"{head} {kind}: {why.splitlines()[0]} | | | | | |")
            continue
        by = {round(f["split"], 6): f for f in r["fields"] if not f.get("at")}
        drops = " / ".join(f"{100 * by[k]['drop_error']:+.2f}%" for k in (0.35, 0.5, 0.65) if k in by)
        u50 = by.get(0.5, {}).get("u")
        lines.append(f"{head} {r['s_star_pinn']:.4f} | {100 * r['regret']:+.4f}% | {drops} | "
                     f"{'—' if u50 is None else f'{u50:.2e}'} | "
                     f"{r['training']['wall_seconds'] / 60:.0f} | {r['training']['converged']} |")
    (directory / "pinn_runs.md").write_text("\n".join(lines) + "\n")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run")
    parser.add_argument("--out", default="docs/re_study")
    parser.add_argument("--index")
    args = parser.parse_args(argv)
    if args.run:
        export(args.run, args.out)
    if args.index:
        index(args.index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
