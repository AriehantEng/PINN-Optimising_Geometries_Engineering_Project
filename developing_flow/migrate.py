"""Backfill config keys added after a developing-flow run finished.

    python -m developing_flow.migrate            # dry run, changes nothing
    python -m developing_flow.migrate --write

`parse_config` refuses a missing key rather than defaulting, so a key that
becomes required locks every stored run out of `load_run` (CLAUDE.md, "Making a
config key required"). This records, for each stored run, the value that
describes what the run ACTUALLY did:

    sampling.rar_points = 0
        Residual-based adaptive refinement did not exist when these runs
        trained; 0 is the setting that reproduces their draws exactly
        (`make_draw` returns `sample_batch` unchanged at 0).

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


def plan(directory: pathlib.Path):
    path = directory / "config.yaml"
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return None
    additions = []
    if "rar_points" not in document.get("sampling", {}):
        additions.append(("sampling.rar_points", 0, RAR_REASON))
    return additions


def apply(directory: pathlib.Path, additions) -> None:
    path = directory / "config.yaml"
    text = path.read_text(encoding="utf-8")
    for key, value, _ in additions:
        if key != "sampling.rar_points":  # pragma: no cover - the table above is exhaustive
            raise ValueError(f"no anchor defined for {key}")
        anchor = re.compile(r"^  n_validation: .*$", re.M)
        match = anchor.search(text)
        if match is None:
            raise ValueError(f"{path}: anchor 'n_validation' not found")
        text = text[: match.end()] + f"\n  rar_points: {json.dumps(value)}" + text[match.end():]
    path.write_text(text, encoding="utf-8")
    with (directory / "config.migrated.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "migrated_by": "developing_flow/migrate.py",
            "migrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "why": "a config key became required after this run finished; values "
                   "describe what the run actually computed, and no result changed",
            "added": [{"key": k, "value": v, "reason": r} for k, v, r in additions],
        }, handle, indent=2)


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
        if not additions:
            print(f"{directory.name}: already complete")
            continue
        touched += 1
        print(directory.name)
        for key, value, reason in additions:
            print(f"  + {key} = {value!r}\n      {reason}")
        if args.write:
            apply(directory, additions)
    print(f"\n{'migrated' if args.write else '(dry run) would migrate'} {touched} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
