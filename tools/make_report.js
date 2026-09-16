// Builds docs/how-the-code-was-built.docx — a report on the construction of the
// Phase 1 PINN, not a results paper. Run: node tools/make_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, TableOfContents, LevelFormat, PageOrientation,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "docs", "how-the-code-was-built.docx");

// US Letter in DXA (1440 = 1 inch)
const PAGE = { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } };
const CONTENT_DXA = 12240 - 2880; // 9360

const p = (text, opts = {}) => new Paragraph({
  children: [new TextRun({ text, ...opts })],
  spacing: { after: 140, line: 276 },
  ...(opts.alignment ? { alignment: opts.alignment } : {}),
});

const h1 = (text) => new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
const h2 = (text) => new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });

const bullet = (text) => new Paragraph({
  children: [new TextRun(text)], numbering: { reference: "bullets", level: 0 },
  spacing: { after: 90, line: 276 },
});

const code = (text) => new Paragraph({
  children: [new TextRun({ text, font: "Consolas", size: 18 })],
  spacing: { after: 60, line: 240 }, indent: { left: 360 },
});

// rich paragraph: array of [text, {bold?, italics?, code?}]
const rich = (parts) => new Paragraph({
  children: parts.map(([t, o = {}]) => new TextRun({
    text: t, bold: o.bold, italics: o.italics,
    ...(o.code ? { font: "Consolas", size: 18 } : {}),
  })),
  spacing: { after: 140, line: 276 },
});

function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  if (total !== CONTENT_DXA) throw new Error(`widths sum to ${total}, need ${CONTENT_DXA}`);
  const cell = (text, { bold = false, shaded = false, w }) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: shaded ? { type: ShadingType.CLEAR, fill: "EDF1F5" } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text, bold, size: 20 })] })],
  });
  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_DXA, type: WidthType.DXA },
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((t, i) => cell(t, { bold: true, shaded: true, w: widths[i] })),
      }),
      ...rows.map((r) => new TableRow({
        children: r.map((t, i) => cell(String(t), { w: widths[i] })),
      })),
    ],
  });
}

function figure(relPath, widthPx, caption) {
  const abs = path.join(ROOT, relPath);
  if (!fs.existsSync(abs)) throw new Error(`missing figure ${relPath}`);
  // read intrinsic size from the PNG header to preserve aspect ratio
  const buf = fs.readFileSync(abs);
  const iw = buf.readUInt32BE(16), ih = buf.readUInt32BE(20);
  const heightPx = Math.round((widthPx * ih) / iw);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height: heightPx } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 18 })],
    }),
  ];
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 460, hanging: 260 } } },
      }],
    }],
  },
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal",
        run: { size: 44, bold: true }, paragraph: { spacing: { after: 120 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "1A3B5C" } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: "2E5C86" } },
    ],
  },
  sections: [{
    properties: { page: PAGE },
    children: [
      new Paragraph({ text: "Building a Physics-Informed Neural Network for Plane Poiseuille Flow", style: "Title" }),
      new Paragraph({
        spacing: { after: 320 },
        children: [new TextRun({ text: "How the Phase 1 code was designed, verified and debugged", size: 26, color: "555555" })],
      }),
      new Paragraph({
        spacing: { after: 400 },
        border: { top: { style: BorderStyle.SINGLE, size: 6, color: "AAAAAA" } },
        children: [new TextRun({ text: "" })],
      }),

      h1("1. What the code is for"),
      rich([
        ["This is the Phase 1 component of a project whose eventual goal is geometry optimisation of a fluidic junction: finding the daughter-channel heights that minimise loss when one channel branches into two. Phase 1 solves a problem that is deliberately simpler than that, and it is simpler for a specific reason: ", {}],
        ["plane Poiseuille flow has a closed-form solution", { bold: true }],
        [", so every number the solver produces can be checked against an answer that has been known since the nineteenth century.", {}],
      ]),
      p("The governing equations are the two-dimensional steady incompressible Navier–Stokes equations. In non-dimensional form, with the channel half-height as the length scale and the centreline velocity as the velocity scale, the exact solution of the fully developed problem is u* = 1 − y*², v* = 0, with a pressure gradient that is constant along the channel. The flux is 4/3."),
      rich([
        ["The purpose of Phase 1 is therefore not to discover anything about Poiseuille flow. It is to establish that the solver is correct, and to build the measurement machinery that Phase 2 will need when the analytic answer disappears. Everything in the design follows from that: the code is organised around ", {}],
        ["being checkable", { italics: true }],
        [", not around being fast or short.", {}],
      ]),

      h2("Why a neural network at all"),
      p("A finite-element solver would solve this problem faster and more accurately, and published comparisons find that PINNs do not generally beat classical methods on either speed or accuracy for problems of this kind. The justification is not accuracy on a single geometry. It is that a PINN taking (x, y, θ) as input — where θ is a vector of geometry parameters — is a differentiable surrogate over an entire design space: train once, then obtain the derivative of a loss functional with respect to the geometry directly by automatic differentiation. That is what Phase 2 needs, and it is the one thing a conventional mesh-based solver does not hand you for free."),

      h1("2. Architecture"),
      p("The code is roughly 5,000 lines of source across 34 modules, with a further 2,600 lines of tests covering 272 test cases. The module layout enforces one rule strictly:"),
      code("physics/  depends on neither models/ nor training/"),
      p("The physics functions take non-dimensional numbers and return residuals. They know nothing about neural networks, optimisers, or configuration objects. This means the governing equations can be tested against manufactured solutions in complete isolation from the machine-learning machinery — and when a residual test fails, the failure cannot be blamed on the network."),
      table(
        ["Package", "Responsibility"],
        [
          ["physics/", "Governing equations, analytic solution, non-dimensionalisation, unit conventions, derivative computation"],
          ["geometry/", "Domain definition, collocation-point sampling, stream-function lifts"],
          ["models/", "Network architectures and the hard boundary-condition wrapper"],
          ["training/", "Loss assembly, loss weighting, optimisers, the training loop"],
          ["evaluation/", "Accuracy metrics, falsifiability probes, drift analysis, figures"],
        ],
        [2200, 7160],
      ),
      p(""),

      h2("Configuration is the single source of truth"),
      rich([
        ["Every run is fully determined by its configuration file plus its seed. No hyperparameter is hard-coded in source, and — importantly — ", {}],
        ["there are no silent defaults", { bold: true }],
        [": a missing configuration key raises an error rather than quietly assuming a value. This sounds pedantic until you are comparing forty runs and need to know what each one actually solved. A run's equations must be readable off its configuration.", {}],
      ]),

      h2("Double precision, non-negotiable"),
      p("Double precision is enabled before any array is created. In single precision the achievable accuracy caps out near 1e-4, and — this is the dangerous part — the symptom looks exactly like an optimiser that has stalled. Days can be lost tuning learning rates against a problem that is actually a floating-point ceiling. The final Case A results here are around 1e-7, which single precision could not represent at all."),

      new Paragraph({ children: [new PageBreak()] }),

      h1("3. The physics layer"),
      h2("Non-dimensionalisation"),
      p("All physics is expressed in non-dimensional variables. No SI-valued viscosity, density or channel height ever enters a loss term. The reason is conditioning: in raw SI units, with micrometre lengths and Pa·s viscosities, the individual loss terms differ by ten or more orders of magnitude and no optimiser will balance them. This is the most common single cause of a PINN that will not converge."),
      p("Two pressure scalings are supported — viscous (p_ref = μU/h) and inertial (p_ref = ρU²) — differing by a factor of Re. They describe the same physical solution and differ only in conditioning; the code supports both so that the choice can be tested rather than assumed."),

      h2("Reynolds number conventions"),
      rich([
        ["The solver works in Re_h, based on the half-height and the maximum velocity. Engineering correlations are quoted in Re_Dh, based on the hydraulic diameter and the ", {}],
        ["mean", { italics: true }],
        [" velocity. For a wide plane channel D_h = 4h and U_mean = (2/3)U_max, so", {}],
      ]),
      code("Re_Dh = (8/3) Re_h"),
      p("A factor of 2.67 is small enough to look like a modelling discrepancy and large enough to invalidate a comparison. This conversion was originally wrong in this code, and the error propagated into a formulation comparison before it was caught. It is now a dedicated module with the conversion constants stated once, and every entrance-length correlation carries a mandatory citation string — a correlation without a stated source raises an exception rather than being used."),

      h2("Boundary conditions imposed exactly"),
      rich([
        ["No-slip is not a penalty term. The velocity is written as u = φ(y)·N(x, y) with φ(y) = 1 − y², so the no-slip condition holds identically for any network output whatsoever. The technique goes back to Lagaris and co-workers in 1998. It removes an entire class of failure — a network that satisfies the interior equations beautifully while quietly ignoring the walls — and it removes one term from the loss, which means one less weight to balance.", {}],
      ]),
      p("This choice has a consequence that took a while to surface. A uniform inlet profile is incompatible with exact no-slip at the inlet corners: matching a top-hat inlet requires the network output to be unbounded at those two points. That is discussed in section 5."),

      h2("Two formulations"),
      p("Both the velocity–pressure formulation and a stream function–vorticity formulation are implemented. The stream function satisfies incompressibility identically, which removes mass conservation as a failure mode — but it also removes mass conservation as an independent check, which matters more than it first appears. Having both lets each one act as a control on the other."),

      new Paragraph({ children: [new PageBreak()] }),

      h1("4. How correctness is established"),
      rich([
        ["The central discipline of this project is a single rule: ", {}],
        ["the loss value is a diagnostic and is never an accuracy claim.", { bold: true }],
        [" Accuracy is reported only as relative L2 error against the analytic solution. This is not a stylistic preference — the code contains a measured demonstration that ranking configurations by loss gives the wrong answer.", {}],
      ]),

      h2("Four layers of checking"),
      table(
        ["Layer", "What it establishes"],
        [
          ["Manufactured solutions", "The residual functions implement the equations they claim to. An analytic field is substituted in and the residual must vanish."],
          ["Finite-difference cross-check", "Automatic differentiation agrees with a fourth-order central stencil to 1e-7, on every architecture and both formulations."],
          ["Relative L2 vs the analytic solution", "The only accuracy claim made anywhere. The network never sees this solution during training."],
          ["Falsifiability probes", "Each acceptance test is checked for whether it can fail at all."],
        ],
        [2600, 6760],
      ),
      p(""),

      h2("The falsifiability matrix"),
      rich([
        ["This is the part of the design most worth explaining, because it is the part that stops a passing test suite from being self-congratulatory. Every acceptance test is classified as ", {}],
        ["LIVE", { bold: true }],
        [", ", {}],
        ["WEAKENED", { bold: true }],
        [" or ", {}],
        ["VACUOUS", { bold: true }],
        [" for each configuration, by asking a blunt question: can this test fail? The probe replaces the network's output weights with zeros — so the model predicts nothing at all — and re-runs the test. Any test the zero model passes is proving nothing.", {}],
      ]),
      p("Two findings came out of this, both uncomfortable:"),
      bullet("The transverse-velocity test (max |v*| below tolerance) is passed by the zero field in every formulation. It is degenerate everywhere, and on its own it is not evidence of anything."),
      bullet("Under the stream-function formulation, mass conservation is satisfied by construction, so the mass-conservation test is vacuous rather than passed. Reporting it as a pass would be reporting the ansatz, not the solver."),
      rich([
        ["A related trap is recorded explicitly in the specification: ", {}],
        ["never add a flux-conservation penalty to the loss.", { bold: true }],
        [" It would drive the mass-drift metric down immediately, and it would do so by training directly on the quantity the test measures. If that term is ever added, the test must be retired and replaced with something not in the loss.", {}],
      ]),
      p("There is also a degeneracy of a subtler kind. The stream-function formulation needs a base flow to carry the flux, and one natural choice — a cubic — is itself the exact solution of Case A. A configuration built on it cannot fail, no matter how poor the network is. Measured accuracy did not reveal this: the cubic lift scored slightly worse (1.10e-6) than a sinusoidal base (7.38e-7). The lesson is that a vacuous configuration is not detectable from its accuracy figure, only from its structure."),

      new Paragraph({ children: [new PageBreak()] }),

      h1("5. Defects found, and how"),
      p("What follows is the honest list. Each of these was found by a check that was built to be capable of failing, and several of them invalidated conclusions that had already been drawn."),

      h2("Loss falls while accuracy worsens"),
      p("A configuration using a lifting inlet reported a lower final loss than the soft-inlet baseline (1.63e-9 against 2.86e-9) and a worse relative L2 error (5.65e-5 against 3.48e-5). The explanation is mundane and important: lifting removes the inlet term, so the total is a sum over five terms rather than six. Part of the apparent improvement is a deleted term. Loss values are not comparable across configurations with different term sets — which is why accuracy is only ever ranked against the analytic solution."),

      h2("Gradient-norm weighting inflates a satisfied constraint"),
      p("An adaptive weighting scheme performed worse than fixed weights. It sets each weight to the total gradient norm divided by that term's gradient norm — so a term the optimiser has already satisfied has a vanishing gradient norm and therefore an exploding weight. The outlet condition, which the exact solution satisfies identically, reached a weight of 6.3e3 against roughly 10 for the momentum terms, and the optimiser then spent its budget on an already-solved constraint. Logging the weights at every step is what made this diagnosable at all."),

      h2("Mass drifts downstream when continuity is only a penalty"),
      p("The developing-flow case reproduced the entrance physics correctly and lost roughly one percent of its mass along the channel, with a loss curve that looked unremarkable throughout. The drift then corrupted an entrance-length measurement, because that length is measured where the centreline reaches 99% of its developed value and the centreline was still creeping. Since Phase 2 splits flux between two branches, this is the single most consequential Phase 1 finding."),

      h2("A curve fit that reported success without fitting anything"),
      rich([
        ["A convergence analysis reported an asymptotic drift of 1.885e-7 ± 9.1e-7, ", {}],
        ["consistent with zero", { italics: true }],
        [". Every parameter in that fit was still its initial guess. The least-squares problem spanned three and a half decades of data, so the cost at the initial guess already sat below the optimiser's default convergence tolerance, and it returned immediately reporting success. The fitted curve missed its own final data point by a factor of nine.", {}],
      ]),
      p("Both guards downstream passed it, for an instructive reason: the identifiability guard is computed from the fitted exponent, which had never been fitted. A guard derived from a fitted parameter cannot detect that the parameter was never fitted. The fix was to weight residuals relatively and to check directly how far the fitted curve sits from the data."),

      h2("A failing acceptance gate that exited zero"),
      p("The full run printed a clear FAIL against the mass-conservation target — and returned exit status zero. The gate existed only as text on a screen for a human to read; no continuous-integration job, queue script or reviewer could have caught a regression. Missing a target now returns a non-zero status, with artifacts written before the status is decided so that a failing run is still diagnosable."),

      h2("A test that measured the test, not the code"),
      p("The derivative check failed at roughly 4e-7 against a 1e-7 target, with a correct implementation. A three-point central second difference has a roundoff floor of about 4ε|f|/h², and truncation error caps the step size, so its best achievable accuracy is around 1e-7|f| — above the target for fields of the magnitude an untrained network produces. The reference method was changed to a five-point fourth-order stencil, whose measured floor is about 2e-9. The tolerance was not touched. That distinction matters: the correct response to a failing test is to establish what is actually failing, not to widen the bound until it passes."),

      new Paragraph({ children: [new PageBreak()] }),

      h1("6. Results"),
      p("Case A, the fully developed channel at Re_h = 10 (Re_Dh ≈ 27), solving the full two-dimensional steady Navier–Stokes residual with the convective terms retained. Evaluated on a grid of 80,601 points against the analytic solution:"),
      table(
        ["Quantity", "Result", "Target"],
        [
          ["Velocity profile u, relative L2", "1.837e-07", "≤ 1e-05"],
          ["max |v*|  (exactly zero in the true solution)", "3.830e-07", "≤ 1e-05"],
          ["Mass-conservation drift across 10 stations", "1.885e-07", "≤ 1e-06"],
          ["Pressure field, relative L2", "3.064e-07", "diagnostic"],
          ["Streamwise pressure drop Δp*", "1.3e-07 relative", "diagnostic"],
          ["Fitted dp*/dx* against analytic −0.2", "1.2e-09 relative", "diagnostic"],
        ],
        [4600, 2680, 2080],
      ),
      p(""),
      ...figure(
        "runs/caseA_re10_extended/profiles.png", 620,
        "Left: predicted profiles (markers) on the analytic solution (line) at five stations. Centre: the signed difference, which is the panel that actually shows the agreement — note the 1e-7 axis. Right: pointwise error over the whole domain.",
      ),
      rich([
        ["The transverse-velocity figure is the one worth dwelling on. In the true solution v* and the convective terms vanish identically — but nothing tells the network that. It has to discover it. That 3.8e-7 is a genuine check rather than a built-in identity.", {}],
      ]),

      h2("What these numbers do not claim"),
      rich([
        ["This is ", {}],
        ["verification, not validation", { bold: true }],
        [". It measures agreement with the analytic solution of an idealised problem: two-dimensional, perfectly smooth rigid walls, exactly parabolic inlet, strictly steady, Newtonian. Model error — the distance between that idealisation and a real channel — dominates by several orders of magnitude. Expressed in SI units for water in a 1 mm channel, a relative error of 1e-7 is about 10 nanometres per second, which no experiment resolves and no real channel is clean enough to exhibit.", {}],
      ]),
      p("The reason for demanding that precision is not physical realism. It is that Phase 2 ranks competing geometries by their loss, and near an optimum the objective is flat by construction, so the differences to be resolved are small. The solver's error floor has to sit far below the signal the optimisation is trying to detect. A 1e-7 floor leaves roughly four orders of headroom; a 1e-2 floor would make the comparison meaningless."),

      h2("Reproducibility, precisely stated"),
      p("A run on a different machine with a different JAX version reproduced the Adam phase bit-for-bit — every digit of the loss trace identical. The subsequent quasi-Newton phase then diverged, agreeing only to about 20%, because the line search amplifies last-bit differences in the gradient. The honest statement is therefore: seeded sampling, initialisation and the first-order phase are reproducible across environments; the second-order phase is reproducible only within one."),

      h1("7. Limits, and what Phase 2 changes"),
      bullet("Plane Poiseuille flow is close to the easiest problem in the field. Its solution is smooth and low-dimensional, and with no-slip built into the ansatz the network is fitting a small correction to something it can nearly represent exactly. Published results on non-trivial flows typically land three to four orders of magnitude worse. The 1e-7 here should not be quoted as an expectation for a branching geometry."),
      bullet("The developing-entrance case has an unresolved singularity at its inlet corners, where the top-hat inlet condition meets exact no-slip. Its residuals there are four orders of magnitude larger than in the channel core and its loss floors accordingly. The fix under consideration replaces the top hat with a family of wall-compatible profiles that approach it in a limit, so the idealisation is recovered as a limit rather than imposed."),
      bullet("Phase 2 has no analytic solution to check against. Every accuracy claim made here rests on one, so an independent reference — a finite-element solve of the same geometry — must be built and itself validated against Poiseuille while an exact answer still exists."),
      bullet("The objective functional for the geometry optimisation is not yet defined, and this is the most consequential open item. Dissipation measured over a truncated domain depends on where the inlet and outlet planes are placed, and when the branch heights change, the fully developed contribution to the pressure drop changes too. A naive comparison would confound a better junction with a wider channel."),

      new Paragraph({
        spacing: { before: 400 },
        border: { top: { style: BorderStyle.SINGLE, size: 6, color: "AAAAAA" } },
        children: [new TextRun({ text: "" })],
      }),
      rich([
        ["A closing note on method. ", {}],
        ["Most of what is described above is machinery for catching the code being wrong, not machinery for making it right.", { italics: true }],
        [" The ratio is deliberate: a physics-informed network reaching a low loss while representing the wrong solution is the expected failure mode for this class of problem, not an unusual one. Several of the defects in section 5 had already produced conclusions that were stated before they were caught. The tests that found them were the ones built to be capable of failing.", {}],
      ]),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${path.relative(ROOT, OUT)} (${(buf.length / 1024).toFixed(0)} KB)`);
});
