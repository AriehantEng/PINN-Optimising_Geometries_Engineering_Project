"""Build docs/phase1-evidence.xlsx from the run artifacts.

Every number this project has quoted should be traceable to the file that
produced it. This reads `runs/*/` and emits one workbook whose cells carry
their own provenance, so "where did that figure come from" has an answer that
does not depend on anyone's memory.

Nothing here computes physics. It reads what the runs already wrote:

    metrics.json      final metrics, flux stations, diagnostics
    history.csv       per-term Adam loss and weight traces
    drift_trace.csv   mass drift against L-BFGS iteration
    lbfgs.json        handoff gap, termination status
    environment.json  JAX version, platform, float64 flag
    config.yaml       the full configuration the run actually used

Usage:  python -m tools.make_tables          (or python tools/make_tables.py)
"""

from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "phase1-evidence.xlsx"

sys.path.insert(0, str(ROOT))
from src.report import TARGETS, gate_results  # noqa: E402

HEADER_FILL = PatternFill("solid", fgColor="1A3B5C")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
NOTE_FONT = Font(italic=True, color="666666", size=9)
MONO = Font(name="Consolas", size=10)


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def read_json(path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_runs():
    """Every run directory that has a metrics.json, newest name order."""
    out = []
    for directory in sorted(RUNS.iterdir()) if RUNS.is_dir() else []:
        if not directory.is_dir():
            continue
        metrics = read_json(directory / "metrics.json")
        if metrics is None:
            continue
        out.append({
            "name": directory.name,
            "dir": directory,
            "metrics": metrics,
            "environment": read_json(directory / "environment.json") or {},
            # lbfgs.json holds a LIST of phase summaries (one per L-BFGS phase,
            # and empty when L-BFGS was disabled). Take the last.
            "lbfgs": (lambda v: v[-1] if isinstance(v, list) and v
                      else (v if isinstance(v, dict) else {}))(
                read_json(directory / "lbfgs.json")),
            "history": read_csv(directory / "history.csv"),
            "drift": read_csv(directory / "drift_trace.csv"),
            "config": (directory / "config.yaml").read_text(encoding="utf-8")
                      if (directory / "config.yaml").is_file() else "",
        })
    return out


def write_sheet(book, title, headers, rows, *, notes=(), widths=None, freeze="A2"):
    sheet = book.create_sheet(title)
    for note in notes:
        sheet.append([note])
        sheet.cell(sheet.max_row, 1).font = NOTE_FONT
    if notes:
        sheet.append([])
    header_row = sheet.max_row + 1
    sheet.append(list(headers))
    for column in range(1, len(headers) + 1):
        cell = sheet.cell(header_row, column)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    for row in rows:
        sheet.append(list(row))
    for index, width in enumerate(widths or [], start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    if freeze:
        sheet.freeze_panes = sheet.cell(header_row + 1, 1)
    return sheet


# --------------------------------------------------------------------------- #
# Sheets
# --------------------------------------------------------------------------- #


def sheet_runs(book, runs):
    rows = []
    for run in runs:
        m, e, l = run["metrics"], run["environment"], run["lbfgs"]
        failures, vacuous = gate_results(m)
        rows.append([
            run["name"], m.get("case"), m.get("reynolds"), m.get("formulation"),
            m.get("pressure_scaling"), m.get("seed"),
            m.get("diag_parameter_count"), m.get("diag_lbfgs_iterations"),
            round(m.get("wall_seconds", 0) / 60.0, 1),
            l.get("message", ""),
            "FAIL" if failures else ("VACUOUS" if vacuous else "PASS"),
            "; ".join(label for label, _, _ in failures) or "-",
            e.get("jax", "?"), str(e.get("x64_enabled", "?")),
            f"runs/{run['name']}/metrics.json",
        ])
    write_sheet(
        book, "runs",
        ["run", "case", "Re_h", "formulation", "p scaling", "seed", "params",
         "L-BFGS iters", "wall min", "termination", "gate", "failed targets",
         "jax", "float64", "source"],
        rows,
        notes=[
            "One row per run directory under runs/. 'gate' is recomputed here "
            "from metrics.json using the same report.gate_results() the run "
            "itself uses, so this column cannot disagree with the run's exit code.",
        ],
        widths=[32, 6, 7, 18, 11, 6, 9, 13, 10, 42, 9, 26, 9, 9, 40],
    )


def sheet_accuracy(book, runs):
    rows = []
    for run in runs:
        m = run["metrics"]
        for key, (label, target, _, only_case) in TARGETS.items():
            if key not in m:
                continue
            if only_case is not None and m.get("case") != only_case:
                continue
            value = m[key]
            vacuous = key == "flux_conservation_relative_error" and value < 1e-12
            rows.append([
                run["name"], label, value, repr(value), target,
                "VACUOUS" if vacuous else ("PASS" if value <= target else "FAIL"),
                f"metrics.json : {key}",
            ])
        for key in ("p_relative_l2", "u_max_abs_error", "p_max_abs_error",
                    "flux_spread_about_mean", "flux_relative_error_vs_inlet",
                    "entrance_length_measured", "entrance_length_correlation",
                    "entrance_length_relative_error", "inlet_flux_deficit_relative",
                    "outlet_profile_relative_l2", "diag_loss_after_adam",
                    "diag_loss_after_lbfgs", "diag_handoff_relative_gap"):
            if key in m and m[key] is not None:
                rows.append([run["name"], key, m[key], repr(m[key]), "",
                             "diagnostic", f"metrics.json : {key}"])
    write_sheet(
        book, "accuracy",
        ["run", "quantity", "value", "value (exact)", "SPEC target",
         "verdict", "source"],
        rows,
        notes=[
            "The 'value (exact)' column is the full float64 repr as TEXT. The "
            "numeric column loses the last bit (~1 ULP, 1.4e-16 relative) "
            "because XLSX stores numbers as 17-digit decimal; that is far "
            "below anything physically meaningful, but a provenance sheet "
            "should not silently drop bits.",
            "Rows with a target are the SPEC section 5 acceptance tests. Rows "
            "marked 'diagnostic' are NOT accuracy claims -- loss values in "
            "particular are never comparable across configs with different "
            "loss-term sets (see docs/phase1-findings.md).",
            "VACUOUS means the test cannot fail for that configuration and so "
            "is not evidence: psi-omega conserves mass by construction.",
        ],
        widths=[32, 38, 16, 26, 14, 11, 44],
    )


def sheet_flux(book, runs):
    rows = []
    for run in runs:
        m = run["metrics"]
        stations, values = m.get("flux_stations"), m.get("flux_values")
        if not stations or not values:
            continue
        nominal = 4.0 / 3.0
        reference = values[0]
        for x, flux in zip(stations, values):
            rows.append([
                run["name"], x, flux,
                flux / nominal - 1.0,
                (flux - reference) / reference,
                f"runs/{run['name']}/metrics.json : flux_values",
            ])
    write_sheet(
        book, "flux_stations",
        ["run", "x*", "flux", "fractional deficit vs 4/3",
         "fractional change vs station 1", "source"],
        rows,
        notes=[
            "Cross-channel flux integral of u* over y*, on 10 fixed stations "
            "spanning 0.05L* to 0.95L*, by 200-node Gauss-Legendre quadrature.",
            "The developed flux is exactly 4/3 in these units (u* is scaled by "
            "U_max, so a uniform profile of the same flux would sit at 2/3).",
            "SPEC test 3 (mass drift D) is the 'change vs station 1' column, "
            "maximised in absolute value. Deficit vs 4/3 is a DIFFERENT thing: "
            "in Case B it is dominated by the inlet flux the network fails to "
            "deliver, which is an inlet artefact, not a conservation failure.",
        ],
        widths=[32, 12, 18, 24, 28, 44],
    )


def sheet_loss_history(book, runs):
    rows, missing = [], []
    for run in runs:
        history = run["history"]
        if not history:
            continue
        columns = history[0].keys()
        if "validation_loss" not in columns:
            missing.append(run["name"])
        term_keys = sorted(c for c in columns if c.startswith("term_"))
        for row in history:
            for key in term_keys:
                rows.append([
                    run["name"], row.get("phase", ""), int(float(row["step"])),
                    key[5:], float(row[key]),
                    float(row.get(f"weight_{key[5:]}", "nan")),
                    float(row["loss"]),
                    float(row["validation_loss"]) if row.get("validation_loss") else None,
                    float(row["relative_l2_vs_exact"]) if row.get("relative_l2_vs_exact") else None,
                    f"runs/{run['name']}/history.csv",
                ])
    notes = [
        "Per-term Adam loss and its weight at each logged step (~200 rows per "
        "run). These come from the same value_and_grad trace as the gradient, "
        "not a separate forward pass.",
        "'rel L2 vs exact' is the ONLY accuracy column here. Loss and "
        "validation loss are diagnostics.",
    ]
    if missing:
        notes.append(
            "STALE DATA: these runs predate the validation_loss / "
            "relative_l2_vs_exact columns and leave them blank -- "
            + ", ".join(missing)
        )
    write_sheet(
        book, "loss_history",
        ["run", "phase", "step", "term", "term loss", "term weight",
         "total loss", "held-out loss", "rel L2 vs exact", "source"],
        rows, notes=notes,
        widths=[32, 8, 8, 18, 15, 14, 15, 15, 17, 38],
    )


def sheet_drift(book, runs):
    rows = []
    for run in runs:
        for row in run["drift"]:
            drift = row.get("drift", row.get("value"))
            rows.append([
                run["name"], int(float(row["iteration"])),
                float(drift) if drift not in (None, "") else None,
                float(row["relative_l2_vs_exact"]) if row.get("relative_l2_vs_exact") else None,
                float(row["validation_loss"]) if row.get("validation_loss") else None,
                f"runs/{run['name']}/drift_trace.csv",
            ])
    write_sheet(
        book, "drift_trace",
        ["run", "L-BFGS iteration", "mass drift D", "rel L2 vs exact",
         "held-out loss", "source"],
        rows,
        notes=[
            "D against L-BFGS iteration (~40 rows per run), measured on the "
            "fixed evaluation stations with 64-node quadrature.",
            "This trace is what settled whether the residual drift was "
            "formulation bias or under-convergence: D fell monotonically with "
            "budget from 1.622e-6 at 6000 iterations to 1.885e-7 at 30000.",
        ],
        widths=[32, 17, 17, 18, 16, 40],
    )


def sheet_conventions(book):
    rows = [
        ["Re_h", "U_max * h / nu", "Solver Reynolds number. Half-height h, MAXIMUM velocity.", "src/physics/conventions.py"],
        ["Re_Dh", "U_mean * D_h / nu", "Correlation Reynolds number. Hydraulic diameter, MEAN velocity.", "src/physics/conventions.py"],
        ["Re_Dh / Re_h", "8/3 = 2.6667", "RE_DH_OVER_RE_H. Getting this wrong invalidated a formulation comparison once.", "conventions.py : RE_DH_OVER_RE_H"],
        ["D_h / h", "4", "Wide plane channel: D_h = 2H where H = 2h is the FULL height.", "conventions.py : D_H_OVER_H"],
        ["U_mean / U_max", "2/3", "Parabolic profile.", "conventions.py : U_MEAN_OVER_U_MAX"],
        ["x*", "x / h", "Streamwise, scaled by the HALF-height.", "src/geometry/domain.py"],
        ["y*", "y / h", "Wall-normal. Walls at y* = -1 and +1, so the half-height is 1 and the full height is 2.", "domain.py : WALL = 1.0"],
        ["u*", "u / U_max", "Developed centreline is exactly 1.", "src/physics/scaling.py"],
        ["flux*", "4/3", "Integral of u* over y* in [-1,1] for the developed profile. A uniform profile of equal flux sits at u* = 2/3.", "src/physics/analytic.py"],
        ["L*", "derived", "Domain length from the entrance-length correlation, NOT set by hand.", "config.py, conventions.py"],
        ["p* (viscous)", "p / (mu U_max / h)", "Recommended for Re <= 1.", "scaling.py"],
        ["p* (inertial)", "p / (rho U_max^2)", "Recommended for Re > 1. p*_viscous = Re * p*_inertial.", "scaling.py"],
        ["dp*/dx* (Case A)", "-2/Re (inertial)", "Constant along the channel. Independent development check.", "analytic.py : pressure_gradient_nd"],
        ["ADF phi(y*)", "1 - y*^2", "Analytic. u = phi * N_u makes no-slip EXACT, not a penalty.", "domain.py : phi"],
        ["ADF (psi-omega)", "phi^2 = (1 - y*^2)^2", "Needs zero slope as well as zero value at both walls.", "src/models/hard_bc.py"],
        ["network input", "normalise(x*, y*)", "Mapped to [-1,1]^2. n_in = 2, HARD-CODED -- Phase 2 (x,y,theta) needs this changed.", "build.py:62, domain.py : normalise"],
        ["entrance length", "L_e/D_h = [c0^n + (c1 Re_Dh)^n]^(1/n)", "c0 = C0/2 = 0.3155, c1 = C1/4 = 0.01105 from the published C0=0.631, C1=0.0442 per FULL height. Two different divisors is CORRECT: one factor of 1/2 from the length scale, one from the Reynolds definition.", "conventions.py, experiments/*.yaml citation"],
    ]
    write_sheet(
        book, "conventions",
        ["symbol", "definition", "note", "defined in"], rows,
        notes=[
            "Every convention in one place. This sheet exists because an "
            "entrance-length estimate was wrong once purely from a Re "
            "convention mismatch, and that class of error is invisible until "
            "the conventions are written down.",
        ],
        widths=[20, 38, 76, 34],
    )


def sheet_hyperparameters(book, runs):
    rows = []
    for run in runs:
        for line in run["config"].splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append([run["name"], line.rstrip()])
    sheet = write_sheet(
        book, "hyperparameters",
        ["run", "config.yaml line"], rows,
        notes=[
            "The full config each run actually used, copied verbatim from its "
            "own runs/<name>/config.yaml -- not from experiments/, which may "
            "have changed since.",
            "Every run is fully determined by this file plus its seed. Note "
            "'seed' and 'sampling.seed' are SEPARATE.",
        ],
        widths=[32, 110],
    )
    for row in range(1, sheet.max_row + 1):
        sheet.cell(row, 2).font = MONO


def main() -> int:
    runs = load_runs()
    if not runs:
        print("no runs with a metrics.json found under runs/", file=sys.stderr)
        return 1

    book = Workbook()
    book.remove(book.active)

    cover = book.create_sheet("README")
    for line in [
        "Phase 1 evidence pack",
        "",
        f"Generated from runs/ by tools/make_tables.py at git {git_sha()}.",
        "Regenerate with:  python tools/make_tables.py",
        "",
        "Every sheet carries a 'source' column naming the file each number came",
        "from. Nothing here is hand-entered and nothing is recomputed from the",
        "model -- these are the artifacts the runs themselves wrote.",
        "",
        "Sheets:",
        "  runs             one row per run, with its acceptance gate re-evaluated",
        "  accuracy         every metric against its SPEC target, PASS/FAIL/VACUOUS",
        "  flux_stations    per-station flux, as fractional deficit and as drift",
        "  loss_history     per-term Adam loss and weight traces",
        "  drift_trace      mass drift against L-BFGS iteration",
        "  conventions      Re, lengths, velocities, pressure, the ADF",
        "  hyperparameters  each run's config.yaml, verbatim",
        "",
        "Reading rule: accuracy is relative L2 against the analytic solution.",
        "The loss is a diagnostic and is never an accuracy claim -- this project",
        "has a measured case where the lower-loss configuration was the less",
        "accurate one (docs/phase1-findings.md).",
    ]:
        cover.append([line])
    cover["A1"].font = Font(bold=True, size=16)
    cover.column_dimensions["A"].width = 78

    sheet_runs(book, runs)
    sheet_accuracy(book, runs)
    sheet_flux(book, runs)
    sheet_loss_history(book, runs)
    sheet_drift(book, runs)
    sheet_conventions(book)
    sheet_hyperparameters(book, runs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    book.save(OUT)
    sizes = {s.title: s.max_row for s in book.worksheets}
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")
    for title, rows_written in sizes.items():
        print(f"  {title:<18} {rows_written:>6} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
