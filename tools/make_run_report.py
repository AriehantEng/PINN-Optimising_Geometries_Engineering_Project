"""Every reading from one run, charted and explained.

Builds docs/<run>-full.xlsx: all 48 scalar metrics with a note on what each one
means and why it matters, the full Adam and L-BFGS traces, the per-station
fluxes, and four native Excel charts built on those sheets.

    python tools/make_run_report.py caseA_re10_extended

Chart colours come from the validated categorical palette in the dataviz
reference (slots assigned in fixed order, never cycled); each chart carries a
legend, and the data table sits on the same sheet, which is the relief the
palette's contrast warning requires.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference, ScatterChart, Series
from openpyxl.chart.marker import Marker
from openpyxl.drawing.line import LineProperties
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.report import TARGETS, gate_results  # noqa: E402

# Validated categorical palette, light mode, assigned in fixed order.
# node scripts/validate_palette.js "<these>" --mode light -> ALL CHECKS PASS
SERIES = ["2A78D6", "EB6834", "1BAF7A", "EDA100", "E87BA4", "008300"]

HEADER_FILL = PatternFill("solid", fgColor="1A3B5C")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
NOTE_FONT = Font(italic=True, color="666666", size=9)
MONO = Font(name="Consolas", size=10)

#: What each reading is, and why anyone should care. The second half matters
#: more than the first: a number without its significance invites the reader to
#: invent one.
SIGNIFICANCE = {
    "u_relative_l2": (
        "SPEC test 1. THE accuracy number",
        "Relative L2 error of the predicted streamwise velocity against the "
        "analytic 1 - y*^2. The network never sees that solution during "
        "training, so this is genuinely independent. Every accuracy claim in "
        "this project is this quantity or its Case B analogue -- nothing else."),
    "v_max_abs": (
        "SPEC test 2. Transverse velocity",
        "v* is identically zero in the true solution, and nothing tells the "
        "network that -- it has to discover it. CAVEAT: the zero field passes "
        "this test too, so it is degenerate in every formulation and is weak "
        "evidence on its own (docs/falsifiability-matrix.md)."),
    "flux_conservation_relative_error": (
        "SPEC test 3. Mass drift D",
        "max|M(x) - M(0)| / M(0) over 10 stations, where M is the cross-channel "
        "flux. Continuity is only a PENALTY term in this formulation, so "
        "nothing forbids mass drifting slowly downstream while the loss looks "
        "fine. This is the test that FAILED at the 6000-iteration budget "
        "(1.622e-06) and passes here."),
    "p_relative_l2": (
        "Pressure field error",
        "Diagnostic, not a gated target. Meaningful only because a pressure "
        "gauge is imposed: pressure is unique up to a constant, so without a "
        "gauge this number would be arbitrary."),
    "p_max_abs_error": (
        "Worst pointwise pressure error",
        "The L-infinity companion to p_relative_l2. A max far above the L2 "
        "would mean the error is concentrated somewhere rather than spread."),
    "u_max_abs_error": (
        "Worst pointwise velocity error",
        "Same role for u. Here it is ~2.3x the relative L2, so the error is "
        "diffuse rather than spiked at one point."),
    "u_max": (
        "Peak streamwise velocity",
        "Should be exactly 1: u* is scaled by U_max, so the developed "
        "centreline is 1 by definition of the non-dimensionalisation."),
    "flux_mean": (
        "Mean flux over the stations",
        "Exactly 4/3 for the developed profile in these units. A uniform "
        "profile carrying the same flux would sit at u* = 2/3, which is the "
        "trap that makes Case B's inlet look wrong at first glance."),
    "flux_spread_about_mean": (
        "Flux spread about the mean",
        "A DIFFERENT measure from test 3: max|M - mean|/mean rather than "
        "referenced to the first station. Recorded separately because a run "
        "predating SPEC v2 S7 once stored this under test 3's key and "
        "understated the drift by 1.8x."),
    "flux_relative_error_vs_inlet": (
        "Mean flux vs the inlet flux",
        "Whether the flow is carrying the flux it was given. Distinct from "
        "drift: a run can conserve mass perfectly while carrying the wrong "
        "amount of it, which is exactly the Case B inlet artefact."),
    "residual_momentum_x_max": (
        "Worst x-momentum residual",
        "How badly the x-momentum equation is violated at its worst point on "
        "the evaluation grid. Unweighted and diagnostic -- it is not an "
        "accuracy claim, because a low residual on a wrong solution is the "
        "expected failure mode for a PINN."),
    "residual_momentum_x_rms": (
        "RMS x-momentum residual",
        "The typical violation rather than the worst. Max/RMS ~7.7 here means "
        "the residual is fairly uniform, not concentrated at a singularity -- "
        "contrast Case B, where the ratio at the inlet corner is enormous."),
    "residual_momentum_y_max": (
        "Worst y-momentum residual",
        "As above for y-momentum. Comparable to the x value, as it should be "
        "when nothing is straining the solution in one direction."),
    "residual_momentum_y_rms": ("RMS y-momentum residual", "See above."),
    "residual_continuity_max": (
        "Worst continuity residual",
        "Pointwise div(u). Related to but not the same as mass drift: drift is "
        "the INTEGRATED consequence of this, and integration normally averages "
        "errors down, which is why test 3 has a tighter bound than test 1."),
    "residual_continuity_rms": ("RMS continuity residual", "See above."),
    "diag_loss_after_lbfgs": (
        "Final training loss",
        "A DIAGNOSTIC AND NEVER AN ACCURACY CLAIM. This project has a measured "
        "case where the configuration with the LOWER loss had the WORSE error, "
        "because removing a loss term changes what the total sums over. Rank "
        "runs by relative L2, never by this."),
    "diag_loss_after_adam": (
        "Loss when Adam handed over",
        "The starting point for L-BFGS. The five orders of magnitude between "
        "this and the final loss is what the quasi-Newton phase bought."),
    "diag_handoff_relative_gap": (
        "Adam -> L-BFGS handoff gap",
        "Relative difference between the loss Adam ended on and the loss "
        "L-BFGS started from. At 2.3e-16 the two phases agree to machine "
        "precision about the same point. A large gap would mean the handoff "
        "silently lost or altered state."),
    "diag_drift_d_infinity": (
        "Fitted asymptote of the drift",
        "D_inf from fitting D(n) = D_inf + a.n^-b to the drift trace. "
        "Indistinguishable from zero here, which is what closes Block C: the "
        "residual drift is UNDER-CONVERGENCE, not a bias baked into the "
        "formulation."),
    "diag_drift_d_infinity_stderr": (
        "Uncertainty on that asymptote",
        "The fit only says something at the 1e-6 scale if this is below 1e-6. "
        "It is (1.2e-07), so the verdict is admissible. At the 6000-iteration "
        "budget it was 8.3e-07 -- still admissible but 6.6x looser."),
    "diag_drift_exponent": (
        "Decay exponent b",
        "Algebraic convergence rate of the drift. b ~ 0.72 means more budget "
        "keeps helping, which the direct measurement confirms: D fell "
        "monotonically from 1.622e-06 to 1.885e-07 as the budget went 6000 -> "
        "30000."),
    "diag_drift_max_relative_residual": (
        "How well the drift fit tracks the data",
        "max|model - D|/D across the trace, capped at 2.0. This guard exists "
        "because curve_fit once returned the initial guess UNCHANGED while "
        "reporting success, and every downstream check passed it. A guard "
        "derived from a fitted parameter cannot detect that the parameter was "
        "never fitted."),
    "diag_drift_moved_off_guess": (
        "Did the fit actually move?",
        "False would mean curve_fit returned p0 and no fit happened. Directly "
        "catches the failure above."),
    "diag_drift_fit_points": (
        "Points in the drift fit", "Trace samples used. See lbfgs_trace."),
    "diag_drift_fit_converged": ("Did the fit converge?", "Fit status."),
    "diag_drift_verdict": (
        "Plain-language drift verdict",
        "Generated from the fit plus its guards, so it cannot claim more than "
        "the fit supports."),
    "diag_lbfgs_iterations": (
        "L-BFGS iterations used",
        "94% of this run's total iterations. The Adam phase is 2000 of 32000."),
    "diag_lbfgs_message": (
        "Why L-BFGS stopped",
        "'ITERATIONS REACHED LIMIT' means it hit the budget while STILL "
        "DESCENDING -- it did not converge to its gradient tolerance. More "
        "budget would still buy accuracy."),
    "diag_lbfgs_status": ("L-BFGS status code", "1 = iteration limit."),
    "diag_selection_rule": (
        "Which iterate was kept",
        "best_validation selects on held-out loss, so the reported model is "
        "not necessarily the final iterate. This matters because it is how "
        "overfitting would be caught in Phase 2, where no analytic answer "
        "exists."),
    "diag_selected_from": (
        "Phase the kept iterate came from", "adam or lbfgs."),
    "diag_validation_loss_best": (
        "Best held-out loss seen",
        "Equal to the final value here, so the last iterate WAS the best -- "
        "no overfitting, consistent with train and held-out tracking each "
        "other throughout."),
    "diag_validation_loss_final": ("Final held-out loss", "See above."),
    "diag_parameter_count": (
        "Network weights",
        "51,075 parameters for a problem whose exact solution is a parabola. "
        "The capacity is not the constraint here; the optimiser is."),
    "evaluation_points": (
        "Evaluation grid size",
        "401 x 201 = 80,601 points. Accuracy is measured on this grid, which "
        "is independent of the collocation points training used."),
    "wall_seconds": ("Total wall time", "Whole run, 4 CPU cores, float64."),
    "diag_adam_seconds": ("Adam wall time", "2000 steps."),
    "diag_lbfgs_seconds": (
        "L-BFGS wall time",
        "~0.30 s/iteration. The dominant cost of the run by 19x."),
    "reynolds": (
        "Reynolds number Re_h",
        "Based on the HALF-height and the MAXIMUM velocity. Correlations use "
        "Re_Dh = (8/3) Re_h = 26.7. Confusing the two is a 2.67x error, small "
        "enough to look like physics and large enough to invalidate a "
        "comparison."),
    "pressure_scaling": (
        "Pressure non-dimensionalisation",
        "'inertial' means p_ref = rho.U^2 rather than mu.U/h. It is a SCALING "
        "CHOICE ONLY and does NOT switch the inertial terms on -- that is "
        "convective_coefficient. Identical physics, different conditioning."),
    "formulation": (
        "Which system was solved",
        "velocity_pressure carries a real pressure field and a continuity "
        "penalty. psi_omega satisfies continuity by construction, which makes "
        "test 3 VACUOUS there rather than passed."),
    "has_pressure_in_system": (
        "Is pressure a solved variable?",
        "False for psi-omega, where pressure must be recovered by path "
        "integration afterwards."),
    "case": ("Case A or B", "A = fully developed. B = developing entrance flow."),
    "seed": (
        "Random seed",
        "Every run is fully determined by its config plus this. Note "
        "sampling.seed is SEPARATE. Caveat: the Adam phase is bit-reproducible "
        "across JAX versions, the L-BFGS trajectory is not."),
    "name": ("Run name", "Matches runs/<name>/."),
    "diag_final_weights": (
        "Loss weights at the end",
        "All 1.0 -- fixed weighting. Gradient-norm weighting was tried and was "
        "WORSE: it gives an already-satisfied term a vanishing gradient norm "
        "and therefore an exploding weight."),
    "diag_drift_refitted": ("Provenance note", "Recorded by src/refit_drift.py."),
    "diag_flux_metrics_recomputed": ("Provenance note", "See flux_spread_about_mean."),
}


def write_sheet(book, title, headers, rows, *, notes=(), widths=None):
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
    sheet.freeze_panes = sheet.cell(header_row + 1, 1)
    return sheet, header_row


def scatter(sheet, *, title, x_title, y_title, log_x, log_y, anchor,
            header_row, x_col, series_cols, first, last, width=24, height=12):
    """A scatter chart over columns already on `sheet`."""
    chart = ScatterChart()
    chart.title = title
    chart.x_axis.title, chart.y_axis.title = x_title, y_title
    chart.style = 2
    chart.width, chart.height = width, height
    if log_x:
        chart.x_axis.scaling.logBase = 10
    if log_y:
        chart.y_axis.scaling.logBase = 10
    # openpyxl hides axes by default when delete is unset
    chart.x_axis.delete = False
    chart.y_axis.delete = False

    xs = Reference(sheet, min_col=x_col, min_row=first, max_row=last)
    for index, column in enumerate(series_cols):
        values = Reference(sheet, min_col=column, min_row=header_row, max_row=last)
        series = Series(values, xs, title_from_data=True)
        colour = SERIES[index % len(SERIES)]
        series.marker = Marker(symbol="circle", size=5)
        series.marker.graphicalProperties.solidFill = colour
        series.marker.graphicalProperties.line.solidFill = colour
        series.graphicalProperties.line = LineProperties(solidFill=colour, w=22000)
        chart.series.append(series)
    sheet.add_chart(chart, anchor)
    return chart


def build(name: str) -> int:
    directory = ROOT / "runs" / name
    metrics = json.loads((directory / "metrics.json").read_text())
    history = list(csv.DictReader((directory / "history.csv").open())) \
        if (directory / "history.csv").is_file() else []
    trace = list(csv.DictReader((directory / "drift_trace.csv").open())) \
        if (directory / "drift_trace.csv").is_file() else []

    book = Workbook()
    book.remove(book.active)

    # ---------------------------------------------------------------- README
    cover = book.create_sheet("README")
    failures, vacuous = gate_results(metrics)
    for line in [
        f"{name}: every reading, charted and explained",
        "",
        f"Gate: {'FAIL' if failures else ('VACUOUS' if vacuous else 'PASS')}"
        f"   |   {metrics['diag_lbfgs_iterations']} L-BFGS iterations"
        f"   |   {metrics['wall_seconds'] / 60:.0f} min",
        "",
        "SHEETS",
        "  readings      all 48 scalar metrics, each with what it is and why it matters",
        "  targets       the three SPEC acceptance tests, with margin  [chart]",
        "  flux          per-station flux and its deficit               [chart]",
        "  adam_history  201 logged Adam steps, per-term losses         [chart]",
        "  lbfgs_trace   41 sampled L-BFGS iterations                   [chart]",
        "  config        the config that produced this run, verbatim",
        "",
        "HOW TO READ THIS",
        "  Accuracy means relative L2 against the analytic solution. It is the only",
        "  accuracy claim made anywhere in this project.",
        "",
        "  Loss is a DIAGNOSTIC and never an accuracy claim. There is a measured case",
        "  in this project where the lower-loss configuration was the less accurate one.",
        "",
        "  The L-BFGS phase is sampled at 41 points over 30,000 iterations and has no",
        "  per-term breakdown. It is not re-runnable at higher density: the L-BFGS",
        "  trajectory is not reproducible across environments, so a re-run would give",
        "  different numbers and invalidate what has already been reported.",
    ]:
        cover.append([line])
    cover["A1"].font = Font(bold=True, size=15)
    cover.column_dimensions["A"].width = 88

    # -------------------------------------------------------------- readings
    rows = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, list):
            continue
        what, why = SIGNIFICANCE.get(key, ("", ""))
        rows.append([
            key,
            json.dumps(value) if isinstance(value, dict) else value,
            repr(value), what, why,
        ])
    sheet, _ = write_sheet(
        book, "readings",
        ["metric", "value", "value (exact)", "what it is", "why it matters"],
        rows,
        notes=["Every scalar this run recorded. The last column is the point: a "
               "number without its significance invites the reader to invent one."],
        widths=[38, 20, 26, 34, 96],
    )
    for row in range(1, sheet.max_row + 1):
        sheet.cell(row, 5).alignment = Alignment(wrap_text=True, vertical="top")

    # --------------------------------------------------------------- targets
    target_rows = []
    for key, (label, target, _, only_case) in TARGETS.items():
        if key not in metrics or (only_case is not None and metrics["case"] != only_case):
            continue
        value = metrics[key]
        target_rows.append([label, value, target, target / value if value else None,
                            "PASS" if value <= target else "FAIL"])
    sheet, header_row = write_sheet(
        book, "targets",
        ["SPEC test", "achieved", "target", "margin (x inside target)", "verdict"],
        target_rows,
        notes=["Margin is target/achieved: how many times inside its bound each "
               "test landed. Test 3 was 0.6x (i.e. FAILING) at the 6000-iteration "
               "budget."],
        widths=[30, 18, 14, 26, 10],
    )
    # Three labelled magnitudes: a bar, not a scatter. A scatter here would
    # need an x-axis, and the only one available (the target value) carries no
    # meaning -- an invented axis is worse than no chart.
    bars = BarChart()
    bars.type, bars.style = "col", 2
    bars.title = "Margin inside the SPEC target (higher is better, log scale)"
    bars.y_axis.title = "target / achieved"
    bars.y_axis.scaling.logBase = 10
    bars.x_axis.delete = bars.y_axis.delete = False
    bars.width, bars.height = 18, 9
    bars.legend = None  # one series: the title names it
    data = Reference(sheet, min_col=4, min_row=header_row, max_row=sheet.max_row)
    labels = Reference(sheet, min_col=1, min_row=header_row + 1, max_row=sheet.max_row)
    bars.add_data(data, titles_from_data=True)
    bars.set_categories(labels)
    bars.series[0].graphicalProperties.solidFill = SERIES[0]
    sheet.add_chart(bars, "G4")

    # ------------------------------------------------------------------ flux
    stations, values = metrics["flux_stations"], metrics["flux_values"]
    reference = values[0]
    sheet, header_row = write_sheet(
        book, "flux",
        ["x*", "flux", "deficit vs 4/3", "drift vs station 1"],
        [[x, f, f / (4 / 3) - 1.0, (f - reference) / reference]
         for x, f in zip(stations, values)],
        notes=["Cross-channel flux by 200-node Gauss-Legendre quadrature. "
               "Deficit vs 4/3 and drift vs station 1 are DIFFERENT quantities: "
               "SPEC test 3 is the max absolute value of the last column. "
               "Conflating them is how an inlet artefact gets mistaken for a "
               "conservation failure."],
        widths=[12, 20, 20, 24],
    )
    scatter(sheet, title="Flux deviation along the channel",
            x_title="x*", y_title="relative deviation", log_x=False, log_y=False,
            anchor="G4", header_row=header_row, x_col=1,
            series_cols=[3, 4], first=header_row + 1, last=sheet.max_row)

    # ---------------------------------------------------------- adam_history
    if history:
        terms = sorted(c for c in history[0] if c.startswith("term_"))
        columns = ["step", "loss"] + terms
        sheet, header_row = write_sheet(
            book, "adam_history",
            ["step", "total loss"] + [t[5:] for t in terms],
            [[float(r[c]) for c in columns] for r in history],
            notes=["Per-term Adam loss at each logged step. These come from the "
                   "same value_and_grad trace as the gradient, not a second "
                   "forward pass. All are DIAGNOSTICS."],
            widths=[9] + [16] * (len(terms) + 1),
        )
        scatter(sheet, title="Adam: total and per-term loss (diagnostic, not accuracy)",
                x_title="Adam step", y_title="loss", log_x=False, log_y=True,
                anchor="K4", header_row=header_row, x_col=1,
                series_cols=list(range(2, len(columns) + 1)),
                first=header_row + 1, last=sheet.max_row)

    # ----------------------------------------------------------- lbfgs_trace
    if trace:
        has_error = "relative_l2_vs_exact" in trace[0]
        headers = ["L-BFGS iteration", "mass drift D"] + (
            ["relative L2 vs analytic"] if has_error else [])
        body = []
        for row in trace:
            entry = [float(row["iteration"]),
                     float(row.get("drift") or row.get("value"))]
            if has_error:
                entry.append(float(row["relative_l2_vs_exact"]))
            body.append(entry)
        sheet, header_row = write_sheet(
            book, "lbfgs_trace", headers, body,
            notes=["41 sampled points over the whole L-BFGS phase.",
                   "D here uses 64-node quadrature (cheap enough to run inside "
                   "the optimiser loop); the reported D in 'readings' uses the "
                   "final 200-node evaluation, so the last row differs by "
                   "~8.8e-10 relative and is NOT expected to match exactly.",
                   "This is the Block C evidence: D falls monotonically with "
                   "budget rather than plateauing, which is what makes the "
                   "residual drift under-convergence rather than formulation bias."],
            widths=[20, 20, 26],
        )
        scatter(sheet, title="Convergence against budget (log-log)",
                x_title="L-BFGS iteration", y_title="magnitude",
                log_x=True, log_y=True, anchor="F5", header_row=header_row,
                x_col=1, series_cols=[2] + ([3] if has_error else []),
                first=header_row + 1, last=sheet.max_row)

    # ---------------------------------------------------------------- config
    sheet, _ = write_sheet(
        book, "config", ["config.yaml line"],
        [[line.rstrip()] for line in (directory / "config.yaml").read_text().splitlines()],
        notes=["Verbatim. Every run is fully determined by this plus its seed."],
        widths=[112],
    )
    for row in range(1, sheet.max_row + 1):
        sheet.cell(row, 1).font = MONO

    out = ROOT / "docs" / f"{name}-full.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    book.save(out)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")
    for ws in book.worksheets:
        charts = len(getattr(ws, "_charts", []))
        print(f"  {ws.title:<15} {ws.max_row:>5} rows"
              + (f"   [{charts} chart]" if charts else ""))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", default="caseA_re10_extended")
    args = parser.parse_args(argv)
    return build(args.run)


if __name__ == "__main__":
    raise SystemExit(main())
