"""How the junction moves the optimal split as the Reynolds number grows.

    python -m bifurcation.junction_re --config bifurcation/junction_re.yaml \\
        --out docs/bifurcation_closed-junction

Reads the closed-length design maps (bifurcation/study_closed_re*.yaml). There
both daughters have both walls over the same length, so the fully developed
friction law puts the optimum exactly at the Murray split s_M, and the departure
s* - s_M is the junction and entrance effect by construction, with no length
model in between.

For every optimum with a prescribed share: the departure, its location bound
(the mesh-induced argmin shift plus the search tolerance, from the coarse grid
and the fine-mesh bracket the study stores), whether it is resolved (beyond the
bound), which daughter it widens, and the Murray regret J(s_M)/J* - 1. Optima
whose bracket contains a split with no steady solution are listed and left out
of every statistic.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from bifurcation import sensitivity as S  # noqa: E402
from developing_flow import pinn  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ConfigError = pinn.ConfigError

SCHEMA = {
    "name": str,
    "maps": list,             # the closed-length study JSONs, one per Reynolds number
}

#: Role colours (categorical slots 1-3, validated all-pairs): the symmetric
#: layout, the intermediate one, and the most asymmetric one in each panel.
ROLE_COLOURS = {"symmetric": "#2a78d6", "intermediate": "#eb6834", "most turned": "#1baf7a"}
ROLE_MARKERS = {"symmetric": "o", "intermediate": "s", "most turned": "^"}
INK, MUTED, BAND = "#0b0b0b", "#52514e", "#e4e3df"


def parse_config(document):
    document = copy.deepcopy(document)
    pinn._check(document, SCHEMA)
    if not document["maps"]:
        raise ConfigError("maps must list at least one study JSON")
    return document


def name(angles):
    return f"{angles[0]:g}/{angles[1]:g}"


def location(row):
    """The bound on s*'s position, from the coarse grid and the fine bracket the
    study stored; None where it cannot be formed (an optimum on a width bound,
    or a fine solve missing)."""
    if row["on_bound"] or S.bracket(row) is None:
        return None
    fine = {round(f["split"], 12): f["total_pressure_drop"] for f in row["fine_bracket"]}
    (sl, _), _, (sr, _) = S.bracket(row)
    left, right = fine.get(round(sl, 12)), fine.get(round(sr, 12))
    if left is None or right is None or row["total_pressure_drop_fine"] is None:
        return None
    return S.mesh_bound(row, left, right, row["xatol"])


def records(study):
    """One record per prescribed-share optimum of one map."""
    xatol = study["config"]["search"]["xatol"]
    out = []
    for row in study["rows"]:
        if row["share"] is None:
            continue
        rec = {"angles": row["angles"], "share": row["share"], "feasible": row["feasible"],
               "unsolved": len(row.get("unsolved", []))}
        if not row["feasible"]:
            out.append({**rec, "trusted": False})
            continue
        bound = location({**row, "xatol": xatol})
        departure = row["best_split"] - row["friction_law_split"]
        a1, a2 = row["angles"]
        rec.update({
            "trusted": bool(row["steady_bracket"]) and bound is not None,
            "best_split": row["best_split"], "murray_split": row["friction_law_split"],
            "departure": departure,
            "bound": None if bound is None else bound["bound"],
            "resolved": bound is not None and abs(departure) > bound["bound"],
            # daughter 2 is the more turned one in every asymmetric layout here
            "widens_turned": None if a1 == a2 else departure < 0.0,
            "murray_regret": (None if row["friction_law_drop"] is None
                              else row["friction_law_drop"] / row["total_pressure_drop"] - 1.0),
            "drop": row["total_pressure_drop"],
        })
        out.append(rec)
    return out


def summarise(config):
    maps = []
    for path in config["maps"]:
        study = json.loads((ROOT / path).read_text())
        recs = records(study)
        maps.append({"path": path, "reynolds": study["config"]["physics"]["reynolds"],
                     "closed_length": study["config"]["geometry"].get("closed_length"),
                     "records": recs})
    if any(m["closed_length"] is None for m in maps):
        raise ConfigError("every map must use the closed-length rule (geometry.closed_length)")
    return {"config": config, "maps": sorted(maps, key=lambda m: m["reynolds"])}


def by_layout(recs):
    out = {}
    for r in recs:
        out.setdefault(name(r["angles"]), []).append(r)
    return out


def layout_order(summary):
    layouts = {tuple(r["angles"]) for m in summary["maps"] for r in m["records"]}
    return sorted(layouts, key=lambda a: (sum(a), a[1] - a[0]))


def markdown(summary):
    maps = summary["maps"]
    res = [m["reynolds"] for m in maps]
    lines = ["## The junction effect against the Reynolds number (equal closed lengths)", "",
             f"Both daughters have both walls over {maps[0]['closed_length']:g}, so the friction "
             "law's optimum is exactly the Murray split s_M and s* - s_M is the junction and "
             "entrance effect. Location bound: the mesh-induced argmin shift plus the search "
             "tolerance. Optima next to a split with no steady solution are left out (listed "
             "below).", "",
             "### Largest |s* - s_M| over the eleven shares, and the shares beyond their bound", "",
             "| layout | " + " | ".join(f"Re_h = {r:g}" for r in res) + " |",
             "|---|" + "---|" * len(res)]
    tables = [by_layout([r for r in m["records"] if r.get("trusted")]) for m in maps]
    for angles in layout_order(summary):
        n = name(angles)
        cells = []
        for t in tables:
            rs = t.get(n, [])
            if not rs:
                cells.append("-")
                continue
            worst = max(abs(r["departure"]) for r in rs)
            beyond = sum(r["resolved"] for r in rs)
            cells.append(f"{worst:.4f} ({beyond}/{len(rs)})")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "### Largest Murray regret, J(s_M)/J* - 1", "",
              "| layout | " + " | ".join(f"Re_h = {r:g}" for r in res) + " |",
              "|---|" + "---|" * len(res)]
    for angles in layout_order(summary):
        n = name(angles)
        cells = []
        for t in tables:
            rs = [r for r in t.get(n, []) if r["murray_regret"] is not None]
            cells.append(f"{100 * max(r['murray_regret'] for r in rs):.3f}%" if rs else "-")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "### Summary by Reynolds number", "",
              "| Re_h | optima trusted | beyond bound | of those in asymmetric layouts, widening "
              "the more-turned daughter | median Murray regret | max Murray regret |",
              "|---|---|---|---|---|---|"]
    for m in maps:
        trusted = [r for r in m["records"] if r.get("trusted")]
        resolved = [r for r in trusted if r["resolved"]]
        asym = [r for r in resolved if r["widens_turned"] is not None]
        regrets = [r["murray_regret"] for r in trusted if r["murray_regret"] is not None]
        lines.append(f"| {m['reynolds']:g} | {len(trusted)}/{len(m['records'])} | "
                     f"{len(resolved)} | {sum(r['widens_turned'] for r in asym)}/{len(asym)} | "
                     f"{100 * statistics.median(regrets):.4f}% | {100 * max(regrets):.3f}% |")
    left_out = [(m["reynolds"], r) for m in maps for r in m["records"] if not r.get("trusted")]
    lines += ["", "### Left out", ""]
    if not left_out:
        lines.append("None: every optimum had a steady bracket and a location bound.")
    for re, r in left_out:
        why = ("no feasible split" if not r["feasible"] else
               f"{r['unsolved']} split(s) with no steady solution" if r["unsolved"] else
               "no location bound (optimum on a width bound or a fine solve missing)")
        lines.append(f"- Re_h = {re:g}, {name(r['angles'])}, q = {r['share']:g}: {why}")
    return "\n".join(lines) + "\n"


def panels(summary):
    """{bifurcation angle: [(angles, role)]}, roles by asymmetry within the panel."""
    out = {}
    for angles in layout_order(summary):
        out.setdefault(sum(angles), []).append(angles)
    roles = {}
    for total, layouts in out.items():
        asym = sorted((a for a in layouts if a[0] != a[1]), key=lambda a: a[1] - a[0])
        roles[total] = [(a, "symmetric") for a in layouts if a[0] == a[1]]
        roles[total] += [(a, "most turned" if a == asym[-1] else "intermediate") for a in asym]
    return roles


def plot(summary, path):
    """Largest |s* - s_M| against Re_h, one panel per bifurcation angle."""
    roles = panels(summary)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True)
    res = [m["reynolds"] for m in summary["maps"]]
    tables = [by_layout([r for r in m["records"] if r.get("trusted")]) for m in summary["maps"]]
    for ax, (total, layouts) in zip(axes.ravel(), sorted(roles.items())):
        bounds = [r["bound"] for t in tables for a, _ in layouts for r in t.get(name(a), [])]
        ax.axhspan(0.0, max(bounds) if bounds else 0.001, color=BAND, lw=0, zorder=0)
        top = 0.0
        for angles, role in layouts:
            xs, ys = [], []
            for re, t in zip(res, tables):
                rs = t.get(name(angles), [])
                if rs:
                    xs.append(re)
                    ys.append(max(abs(r["departure"]) for r in rs))
            ax.plot(xs, ys, color=ROLE_COLOURS[role], lw=2, marker=ROLE_MARKERS[role], ms=6,
                    zorder=3)
            if xs:
                ax.annotate(name(angles), (xs[-1], ys[-1]), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=8, color=INK)
                top = max(top, max(ys))
        ax.set_xscale("log")
        ax.set_xticks(res, [f"{r:g}" for r in res])
        ax.minorticks_off()
        ax.set_ylim(0.0, 1.15 * max(top, 0.0015))
        ax.set_xlim(min(res) / 1.15, max(res) * 1.6)
        ax.set_title(f"{total:g}°" + (" (face septum)" if total == 180 else ""),
                     fontsize=10, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=BAND, lw=0.8)
    for ax in axes[1]:
        ax.set_xlabel("Reynolds number Re_h", fontsize=9, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("largest |s* − s_M| over the shares", fontsize=9, color=INK)
    handles = [plt.Line2D([], [], color=c, lw=2, marker=ROLE_MARKERS[k], ms=6, label=k)
               for k, c in ROLE_COLOURS.items()]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND, label="largest location bound"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Departure of the optimal split from the Murray split, daughters of equal "
                 "closed length", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(config_path, out):
    config = parse_config(yaml.safe_load(pathlib.Path(config_path).read_text()))
    summary = summarise(config)
    out = pathlib.Path(out)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    out.with_suffix(".md").write_text(markdown(summary))
    plot(summary, out.with_suffix(".png"))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run(args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
