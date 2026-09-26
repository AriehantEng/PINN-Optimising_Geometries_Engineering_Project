"""Backfill config keys added after a developing-flow run finished.

    python -m developing_flow.migrate            # dry run, changes nothing
    python -m developing_flow.migrate --write

`parse_config` refuses a missing key rather than defaulting, so a key that
becomes required locks every stored run out of `load_run` (CLAUDE.md, "Making a
config key required"). This records, for each stored run, the value that
describes what the run ACTUALLY did, from the MIGRATIONS table below:

    sampling.rar_points = 0                 RAR did not exist; 0 reproduces its draws
    sampling.inlet_points = random          inlet points were always redrawn
    sampling.rar_mode = top                 (RAR runs only) RAR always took the worst points
    physics.inlet_condition = soft          the inlet was always a loss term
    physics.formulation = velocity_pressure the network always output u, v, p
    model.activation = tanh                 the only activation there was
    training.adam.schedule = exponential    the only schedule there was
    model.init_gain = 1.0                   plain Glorot was the only initialisation
    model.input_map = linear                the network always saw xi = 2x/L - 1
    physics.inlet_ramp = linear             (hard-inlet runs only) the ramp was x/L

The edit is a targeted text insertion, not a YAML round-trip, so comments in
the stored config survive, and each migrated run gets a config.migrated.json
saying what was added and why.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re

import yaml

RUNS_ROOT = pathlib.Path(__file__).resolve().parent / "runs"

RAR_REASON = (
    "residual-based adaptive refinement did not exist when this run trained; "
    "rar_points = 0 reproduces its collocation draws exactly"
)

#: (dotted key, value, anchor line it follows, applies-to(document), reason).
#: Values describe what the stored runs ACTUALLY did: every option below was
#: added later, and its value here is the behaviour the code had before.
MIGRATIONS = (
    ("sampling.rar_points", 0, r"^  n_validation: .*$",
     lambda d: "rar_points" not in d.get("sampling", {}), RAR_REASON),
    ("physics.inlet_condition", "soft", r"^  inlet_sharpness: .*$",
     lambda d: "inlet_condition" not in d.get("physics", {}),
     "before physics.inlet_condition existed the inlet was always a soft loss term"),
    ("physics.formulation", "velocity_pressure", r"^  inlet_sharpness: .*$",
     lambda d: "formulation" not in d.get("physics", {}),
     "before physics.formulation existed the network always output u, v and p"),
    ("model.activation", "tanh", r"^  depth: .*$",
     lambda d: "activation" not in d.get("model", {}),
     "before model.activation existed the network always used tanh"),
    ("sampling.inlet_points", "random", r"^  rar_points: .*$",
     lambda d: "inlet_points" not in d.get("sampling", {}),
     "before sampling.inlet_points existed the inlet points were redrawn at random each time"),
    ("sampling.rar_mode", "top", r"^  rar_pool: .*$",
     lambda d: d.get("sampling", {}).get("rar_points", 0) > 0
     and "rar_mode" not in d.get("sampling", {}),
     "before sampling.rar_mode existed RAR always added the worst-residual points"),
    ("training.adam.schedule", "exponential", r"^    learning_rate: .*$",
     lambda d: "schedule" not in d.get("training", {}).get("adam", {}),
     "before training.adam.schedule existed the learning rate always decayed exponentially"),
    ("model.init_gain", 1.0, r"^  activation: .*$",
     lambda d: "init_gain" not in d.get("model", {}),
     "before model.init_gain existed every layer used plain Glorot initialisation (gain 1)"),
    ("model.input_map", "linear", r"^  init_gain: .*$",
     lambda d: "input_map" not in d.get("model", {}),
     "before model.input_map existed the network always saw xi = 2x/L - 1"),
    ("physics.inlet_ramp", "linear", r"^  inlet_condition: .*$",
     lambda d: d.get("physics", {}).get("inlet_condition") == "hard"
     and "inlet_ramp" not in d.get("physics", {}),
     "before physics.inlet_ramp existed a hard inlet always used the linear ramp x/L"),
)


def plan(directory: pathlib.Path):
    path = directory / "config.yaml"
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return None
    return [(key, value, anchor, reason)
            for key, value, anchor, applies, reason in MIGRATIONS if applies(document)]


def apply(directory: pathlib.Path, additions) -> None:
    """Insert each key after its anchor line, in table order.

    A targeted text edit, not a YAML round-trip, so the stored config's
    comments survive. Indentation is copied from the anchor line.
    """
    path = directory / "config.yaml"
    text = path.read_text(encoding="utf-8")
    for key, value, anchor, _ in additions:
        match = re.search(anchor, text, re.M)
        if match is None:
            raise ValueError(f"{path}: no line matching {anchor!r} to place {key} after")
        indent = match.group(0)[: len(match.group(0)) - len(match.group(0).lstrip())]
        line = f"{indent}{key.split('.')[-1]}: {json.dumps(value)}"
        text = text[: match.end()] + "\n" + line + text[match.end():]
    path.write_text(text, encoding="utf-8")
    record_path = directory / "config.migrated.json"
    history = json.loads(record_path.read_text()) if record_path.is_file() else []
    if isinstance(history, dict):
        history = [history]
    history.append({
        "migrated_by": "developing_flow/migrate.py",
        "migrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "why": "config keys became required after this run finished; values "
               "describe what the run actually computed, and no result changed",
        "added": [{"key": k, "value": v, "reason": r} for k, v, _, r in additions],
    })
    with record_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="apply; without it nothing changes")
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    args = parser.parse_args(argv)
    root = pathlib.Path(args.runs_root)
    touched = 0
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        if not directory.is_dir():
            continue
        additions = plan(directory)
        if additions is None:
            continue
        if not (directory / "params.npz").is_file():
            # params.npz is written last, so this run is unfinished and may be
            # training right now; its config is migrated once it has finished.
            print(f"{directory.name}: unfinished, skipped")
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
