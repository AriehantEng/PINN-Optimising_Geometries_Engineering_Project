"""Finished parametric runs side by side, ranked against finite elements.

    python -m bifurcation.compare_runs --config bifurcation/compare_re10_options.yaml \\
        --out docs/bifurcation_re10-options

Reads each run's metrics.json and history.json (written by parametric.run on
the same evaluation splits and mesh) and tabulates the relative L2 errors, the
drop errors, the optimum, the held-out loss and the cost. The ranking column is
the error against the reference; the held-out loss is shown as the diagnostic
it is, never as the score (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from developing_flow import pinn  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ConfigError = pinn.ConfigError
SCHEMA = {"name": str, "runs": list}

#: Categorical slots 1-8 in order (validated adjacent-pairs palette).
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
           "#e34948")
INK, MUTED, BAND = "#0b0b0b", "#52514e", "#e4e3df"


def parse_config(document):
    document = copy.deepcopy(document)
    pinn._check(document, SCHEMA)
    if not 1 <= len(document["runs"]) <= len(COLOURS):
        raise ConfigError(f"runs must list 1 to {len(COLOURS)} runs")
    for k, entry in enumerate(document["runs"]):
        if not isinstance(entry, dict) or set(entry) != {"label", "run"}:
            raise ConfigError(f"runs[{k}] must be a mapping with exactly 'label' and 'run'")
    return document


def summary(label, metrics, history):
    rows = metrics["splits"]
    splits = [r["split"] for r in rows]
    t = metrics["training"]
    o = metrics["optimum"]
    return {
        "label": label, "splits": splits,
        "u": [r["relative_l2"]["u"] for r in rows], "v": [r["relative_l2"]["v"] for r in rows],
        "p": [r["relative_l2"]["p"] for r in rows],
        "drop_error": [r["drop_error"] for r in rows],
        "offset_error": [r["pinn_offset"] - r["reference_offset"] for r in rows],
        "argmin_error": o["argmin_error"], "regret": o["regret"],
        "min_drop_error": o["min_drop_error"],
        "selected": t["selected"], "heldout": t["selected_heldout_loss"],
        "start_heldout": t.get("start_heldout_loss"),
        "minutes": t["total_seconds"] / 60.0, "parameters": t["parameter_count"],
        "lm_steps": sum(1 for h in history if h["phase"] == "lm"),
    }


def gather(document):
    out = []
    for entry in document["runs"]:
        directory = ROOT / entry["run"]
        metrics = json.loads((directory / "metrics.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        out.append(summary(entry["label"], metrics, history))
    splits = out[0]["splits"]
    if any(s["splits"] != splits for s in out):
        raise ConfigError("the runs were evaluated at different splits")
    return out


def markdown(rows, name):
    lines = [f"## {name}: parametric runs against finite elements", "",
             "Relative L2 errors integrated on the reference's mesh at each evaluation split; "
             "mean and worst over the splits. Held-out loss: the diagnostic, not the score.", "",
             "| run | u mean | u worst | v mean | p mean | worst drop error | argmin error "
             "| regret | held-out loss | LM steps | minutes | parameters |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['label']} | {np.mean(r['u']):.2e} | {np.max(r['u']):.2e} | "
            f"{np.mean(r['v']):.2e} | {np.mean(r['p']):.2e} | "
            f"{100 * np.max(np.abs(r['drop_error'])):.3f}% | {r['argmin_error']:+.4f} | "
            f"{100 * r['regret']:+.4f}% | {r['heldout']:.3e} | {r['lm_steps']} | "
            f"{r['minutes']:.0f} | {r['parameters']} |")
    lines += ["", "### At each split: u error, and drop error", "",
              "| run | " + " | ".join(f"s = {s:g}" for s in rows[0]["splits"]) + " |",
              "|---|" + "---|" * len(rows[0]["splits"])]
    for r in rows:
        lines.append(f"| {r['label']} | " + " | ".join(
            f"{u:.2e} ({100 * d:+.2f}%)" for u, d in zip(r["u"], r["drop_error"])) + " |")
    return "\n".join(lines) + "\n"


def plot(rows, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for colour, r in zip(COLOURS, rows):
        left.semilogy(r["splits"], r["u"], color=colour, lw=2, marker="o", ms=6, label=r["label"])
        right.semilogy(r["splits"], np.abs(r["drop_error"]), color=colour, lw=2, marker="o",
                       ms=6, label=r["label"])
    left.set_ylabel("u, relative L2 error", fontsize=9, color=INK)
    right.set_ylabel("|total-pressure drop error|", fontsize=9, color=INK)
    for ax in (left, right):
        ax.set_xlabel("width split s = w1 / w0", fontsize=9, color=INK)
        ax.grid(color=BAND, lw=0.8, which="both")
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(rows), 4), frameon=False,
               fontsize=9)
    fig.tight_layout(rect=(0, 0.1 if len(rows) <= 4 else 0.15, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(config_path, out):
    document = parse_config(yaml.safe_load(pathlib.Path(config_path).read_text()))
    rows = gather(document)
    out = pathlib.Path(out)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    out.with_suffix(".md").write_text(markdown(rows, document["name"]))
    plot(rows, out.with_suffix(".png"))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run(args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
