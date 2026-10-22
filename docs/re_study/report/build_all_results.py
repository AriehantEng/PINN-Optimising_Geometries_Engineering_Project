"""Every result of the study in one document: an overview, then the four reports.

    python docs/re_study/report/build_all_results.py
        -> docs/re_study/report/content_all_results.json
    node docs/re_study/report/build_docx.js docs/re_study/report/content_all_results.json \\
        docs/re_study/all_results.docx

Run the four builders first (build_content.py, build_paper_items.py,
build_other_layouts.py, build_tier_a.py); this reads their content files. The
overview's numbers are computed here from the stored result files.
"""
import glob
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPORT = ROOT / "docs" / "re_study" / "report"
STUDY = ROOT / "docs" / "re_study"
OUT = REPORT / "content_all_results.json"

PARTS = [("content.json", "Part A. The Reynolds-number study (30/90 and 30/120, q = 0.5)"),
         ("content_paper_items.json", "Part B. The paper's outstanding items"),
         ("content_other_layouts.json", "Part C. The other layouts (finite elements and Network 3)"),
         ("content_tier_a.json", "Part D. Tier A: new shares and a symmetric control")]


def sci(x, digits=1):
    m, e = f"{x:.{digits}e}".split("e")
    sup = str(int(e)).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m} × 10{sup}"


blocks = [{"title": "PINN optimisation of a planar bifurcation: all results"}]


def p(t): blocks.append({"p": t})


# ---- overview, from the stored files --------------------------------------- #
blocks.append({"h1": "Overview"})
p("This document gathers every result produced for the paper since the Reynolds-number study "
  "began. Part A is the Reynolds-number study of the corner-mode parametric PINN at q = 0.5; "
  "Part B answers the paper's outstanding items; Part C collects the finite-element results for "
  "layouts other than 30/90 and 30/120; Part D is Tier A, the pre-registered runs at other "
  "shares. Accuracy is always measured against the finite-element reference solution, never by "
  "loss value.")
rows = []
for re_ in ("0.1", "1", "10", "100"):
    s = [json.loads(pathlib.Path(f).read_text())
         for f in sorted(glob.glob(str(STUDY / f"pinn_30-90_Re{re_}_seed*_summary.json")))]
    e = [x["argmin_error"] for x in s]
    rows.append([re_, str(len(s)), f"{statistics.mean(e):+.4f}"
                 + (f" ± {statistics.stdev(e):.4f}" if len(e) > 1 else ""),
                 f"{100 * max(x['regret'] for x in s):.4f}%"])
outlet = json.loads((ROOT / "docs" / "developing_re10-outlet.json").read_text())
o = [r["analytical"] for r in outlet["runs"]]
tier = [json.loads(pathlib.Path(f).read_text())
        for f in sorted(glob.glob(str(STUDY / "tierA" / "pinn_*_summary.json")))]
p1 = [t for t in tier if abs(t["argmin_error"]) <= 0.002]
blocks.append({"bullets": [
    f"Part A. At q = 0.5 the parametric PINN locates 30/90's optimum to within the ±0.002 search "
    f"tolerance on average at Re_h = 10 ({rows[2][2]}), with a regret below 0.06% at every Reynolds "
    f"number; 30/120's minimum lies at the geometry's limit at Re_h ≤ 10, so it tests the pressure "
    f"drop only.",
    f"Part B. Network 1's outlet profile matches the Poiseuille profile u = 1 − y² to "
    f"{sci(statistics.mean(o))} (relative L2, mean of three seeds); Table 3's low-Re rows, the "
    f"solver (SuperLU), Network 1's L-BFGS cycles (4 × 500) and the references are settled.",
    "Part C. Thirteen other layouts have finite-element optima at Re_h = 10 and closed-length "
    "maps at six Reynolds numbers; no PINN was trained on them. On six held-out layouts the free "
    "length-corrected split s_L has a mean regret of 0.015% against Network 3's 0.016% (worst "
    "0.087% against 0.077%), and the Murray split 0.109%.",
    f"Part D. Of {len(tier)} pre-registered runs, {len(p1)} pass P1 (the PINN's optimum within "
    f"±0.002), both at q = 0.5; every off-centre share fails, the symmetric null test included, "
    f"and at Re_h = 0.1 badly. The share is confounded with a smaller corner-mode cutoff.",
]})
blocks.append({"table": {"head": ["Re_h", "Seeds", "Argmin error, mean ± sd", "Largest regret"],
                         "rows": rows, "widths": [1500, 1200, 3500, 2500]}})
blocks.append({"note": "Part A, 30/90 at q = 0.5: the PINN's optimum minus the reference "
                       "optimum, over the seeds run at each Reynolds number."})

# ---- the four parts, one heading level down -------------------------------- #
seen_images = set()
for name, heading in PARTS:
    part = json.loads((REPORT / name).read_text())
    blocks.append({"pagebreak": True})
    blocks.append({"h1": heading})
    for b in part:
        if "title" in b:
            continue
        if "h1" in b:
            blocks.append({"h2": b["h1"]})
        elif "h2" in b:
            blocks.append({"h3": b["h2"]})
        elif "img" in b:
            if b["img"] in seen_images:      # Part B repeats two of Part A's figures
                blocks.append({"note": "(Figure shown in Part A: " + b["caption"] + ")"})
                continue
            seen_images.add(b["img"])
            blocks.append(b)
        else:
            blocks.append(b)

OUT.write_text(json.dumps(blocks, indent=1, ensure_ascii=False))
print(f"{len(blocks)} blocks -> {OUT}")
