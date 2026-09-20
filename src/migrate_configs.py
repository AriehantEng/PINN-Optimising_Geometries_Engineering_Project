"""Backfill config keys added after a run finished, so finished runs still load.

`load_config` refuses a missing key rather than defaulting, which is the right
rule for a RUN: a run's equations must be readable off its config. But the rule
has a side effect on runs that already happened. Adding
`physics.convective_coefficient` as a required key locked every stored run out
of `load_run`, so no figure could be regenerated from any completed work.

This backfills the keys with the values that describe what each run ACTUALLY
computed. That is the opposite of a silent default: the point is to record the
behaviour, not to guess at it.

    physics.convective_coefficient = 1.0
        Before the switch existed, `momentum_x`/`momentum_y` included
        `convective_x`/`convective_y` unconditionally. 1.0 reproduces them
        exactly; 0.0 would describe a solver that was never run.

    training.select = "final" | "best_validation"
        Taken from the run's own `diag_selection_rule` when it recorded one.
        Runs that predate held-out validation recorded nothing and used the
        final iterate, so they get "final".

    sampling.n_validation = <n_interior>
        INERT for these runs and recorded as such. Runs needing this backfill
        predate held-out validation entirely, and with `select: final` no
        validation set is consulted. The schema requires a positive integer, so
        "this run had none" cannot be expressed as 0; the value exists only to
        satisfy the schema and cannot affect any stored result.

Each migrated directory gets a `config.migrated.json` recording what was added
and why, so the provenance of the edit is itself on disk.

    python -m src.migrate_configs              # dry run, changes nothing
    python -m src.migrate_configs --write
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re

import yaml

RUNS_ROOT = pathlib.Path("runs")

#: Value, anchor line pattern, and the reason, per backfillable key.
CONVECTIVE_REASON = (
    "the convective term was included unconditionally at coefficient 1.0 "
    "before physics.convective_coefficient existed, so 1.0 reproduces what "
    "this run computed"
)
N_VALIDATION_REASON = (
    "INERT: this run predates held-out validation and selects the final "
    "iterate, so no validation set is consulted. The schema requires a "
    "positive integer, so 'none' cannot be written as 0; mirrored from "
    "n_interior purely to satisfy the schema"
)
SELECT_REASON = (
    "this run recorded no diag_selection_rule, which means it predates "
    "best-iterate selection and used the final iterate"
)
INLET_PROFILE_REASON = (
    "this run carries inlet_uniform_value, i.e. a top hat, which is what "
    "physics.inlet_profile: uniform now names explicitly. The regularised "
    "family did not exist when it ran"
)


def _insert_after(text: str, anchor: re.Pattern, line: str) -> str:
    """Insert `line` immediately after the first line matching `anchor`.

    A targeted text edit rather than a YAML round-trip, because safe_load then
    safe_dump would silently drop every comment in the file -- including the
    provenance headers explaining what each run was for.
    """
    match = anchor.search(text)
    if match is None:
        raise ValueError(f"anchor {anchor.pattern!r} not found")
    return text[: match.end()] + "\n" + line + text[match.end() :]


def plan_migration(directory: pathlib.Path):
    """Return (list of (key, value, reason), current document) for one run."""
    config_path = directory / "config.yaml"
    if not config_path.is_file():
        return None
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return None

    metrics_path = directory / "metrics.json"
    metrics = {}
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)

    additions = []
    if "convective_coefficient" not in document.get("physics", {}):
        additions.append(
            ("physics.convective_coefficient", 1.0, CONVECTIVE_REASON)
        )
    if "select" not in document.get("training", {}):
        recorded = metrics.get("diag_selection_rule")
        additions.append((
            "training.select",
            recorded or "final",
            f"taken from this run's diag_selection_rule ({recorded!r})"
            if recorded else SELECT_REASON,
        ))
    physics = document.get("physics", {})
    if document.get("case") == "B" and "inlet_profile" not in physics:
        # A stored Case B run has inlet_uniform_value and nothing else, which
        # is exactly the top hat. Naming it is a relabel, not a change.
        additions.append(
            ("physics.inlet_profile", "uniform", INLET_PROFILE_REASON)
        )
    if "n_validation" not in document.get("sampling", {}):
        additions.append((
            "sampling.n_validation",
            int(document["sampling"]["n_interior"]),
            N_VALIDATION_REASON,
        ))
    return additions, document


def apply_migration(directory: pathlib.Path, additions) -> None:
    config_path = directory / "config.yaml"
    text = config_path.read_text(encoding="utf-8")

    for key, value, _ in additions:
        rendered = f"  {key.split('.')[-1]}: {json.dumps(value)}"
        if key == "physics.convective_coefficient":
            anchor = re.compile(r"^  formulation: .*$", re.M)
        elif key == "training.select":
            anchor = re.compile(r"^training:$", re.M)
        elif key == "physics.inlet_profile":
            anchor = re.compile(r"^  inlet_uniform_value: .*$", re.M)
            # inserted BEFORE the value it describes reads better, but the
            # helper appends after the anchor; YAML is order-insensitive.
        elif key == "sampling.n_validation":
            anchor = re.compile(r"^  n_interior: .*$", re.M)
        else:  # pragma: no cover - the table above is exhaustive
            raise ValueError(f"no anchor defined for {key}")
        text = _insert_after(text, anchor, rendered)

    config_path.write_text(text, encoding="utf-8")

    record = {
        "migrated_by": "src/migrate_configs.py",
        "migrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "why": (
            "config keys became required after this run finished, which made "
            "the run unloadable. Values describe what the run actually "
            "computed; nothing about its results changed."
        ),
        "added": [
            {"key": key, "value": value, "reason": reason}
            for key, value, reason in additions
        ],
    }
    with (directory / "config.migrated.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true",
        help="apply the backfill; without it nothing is touched",
    )
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    args = parser.parse_args(argv)

    root = pathlib.Path(args.runs_root)
    touched = 0
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        if not directory.is_dir():
            continue
        planned = plan_migration(directory)
        if planned is None:
            continue
        additions, _ = planned
        if not additions:
            print(f"{directory.name}: already complete")
            continue
        touched += 1
        print(f"\n{directory.name}")
        for key, value, reason in additions:
            print(f"  + {key} = {value!r}")
            print(f"      {reason}")
        if args.write:
            apply_migration(directory, additions)

    if not args.write:
        print(f"\n(dry run: {touched} run(s) would be migrated; pass --write)")
    else:
        print(f"\nmigrated {touched} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
