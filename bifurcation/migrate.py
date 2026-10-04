"""Backfill config keys added after a bifurcation (Network 2 or 3) run finished.

    python -m bifurcation.migrate            # dry run, changes nothing
    python -m bifurcation.migrate --write

`parse_config` refuses a missing key rather than defaulting, so a key that
becomes required locks every stored run out of `load_run` (CLAUDE.md, "Making a
config key required"). Each stored run gets the value that describes what it
ACTUALLY did:

    model.formulation = velocity_pressure   the only formulation before the
                                            stream function existed
    weights.corner_radius = 0.0             no corner weighting existed: every
                                            interior residual had weight 1
    model.wall_extension = 0.0              walls ended at the openings
    fit.workers = 1                         (Network 3) cross-validation ran in one process

A targeted text insertion (comments survive), recorded in config.migrated.json,
as developing_flow/migrate.py does for Network 1.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re

import yaml

RUNS_ROOT = pathlib.Path(__file__).resolve().parent / "runs"

#: (dotted key, value, anchor line it follows, applies-to(document), reason).
MIGRATIONS = (
    ("model.formulation", "velocity_pressure", r"^  distance_order: .*$",
     lambda d: "formulation" not in d.get("model", {}),
     "before model.formulation existed the network always output u, v, p"),
    ("model.wall_extension", 0.0, r"^  distance_order: .*$",
     lambda d: "wall_extension" not in d.get("model", {}),
     "before model.wall_extension existed every wall ended at the openings"),
    ("weights.corner_radius", 0.0, r"^  outlet: .*$",
     lambda d: "corner_radius" not in d.get("weights", {}),
     "before weights.corner_radius existed no residual was weighted near the corners"),
)


#: The same for Network 3 (bifurcation/geometric.py) runs.
NETWORK3_MIGRATIONS = (
    ("fit.workers", 1, r"^  max_evaluations: .*$",
     lambda d: "workers" not in d.get("fit", {}),
     "before fit.workers existed cross-validation ran in one process"),
)


def plan(path: pathlib.Path):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return None
    if "outlets" in document:                         # Network 2
        table = MIGRATIONS
    elif "data" in document and "fit" in document:    # Network 3
        table = NETWORK3_MIGRATIONS
    else:
        return None
    return [(key, value, anchor, reason)
            for key, value, anchor, applies, reason in table if applies(document)]


def insert(text, additions, where):
    for key, value, anchor, _ in additions:
        match = re.search(anchor, text, re.M)
        if match is None:
            raise ValueError(f"{where}: no line matching {anchor!r} to place {key} after")
        indent = match.group(0)[: len(match.group(0)) - len(match.group(0).lstrip())]
        line = f"{indent}{key.split('.')[-1]}: {json.dumps(value)}"
        text = text[: match.end()] + "\n" + line + text[match.end():]
    return text


def apply(directory: pathlib.Path, additions) -> None:
    path = directory / "config.yaml"
    path.write_text(insert(path.read_text(encoding="utf-8"), additions, path), encoding="utf-8")
    record_path = directory / "config.migrated.json"
    history = json.loads(record_path.read_text()) if record_path.is_file() else []
    history.append({
        "migrated_by": "bifurcation/migrate.py",
        "migrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "why": "config keys became required after this run finished; values "
               "describe what the run actually computed, and no result changed",
        "added": [{"key": k, "value": v, "reason": r} for k, v, _, r in additions],
    })
    record_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="apply; without it nothing changes")
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    args = parser.parse_args(argv)
    root = pathlib.Path(args.runs_root)
    touched = 0
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        if not (directory / "config.yaml").is_file():
            continue
        if not (directory / "params.npz").is_file():
            print(f"{directory.name}: unfinished, skipped")
            continue
        additions = plan(directory / "config.yaml")
        if additions is None:
            print(f"{directory.name}: not a Network 2 or 3 run, skipped")
            continue
        if not additions:
            print(f"{directory.name}: already complete")
            continue
        touched += 1
        print(directory.name)
        for key, value, _, reason in additions:
            print(f"  + {key} = {value!r}\n      {reason}")
        if args.write:
            apply(directory, additions)
    print(f"\n{'migrated' if args.write else '(dry run) would migrate'} {touched} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
