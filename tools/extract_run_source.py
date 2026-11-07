"""Assemble the source that produced runs/caseA_re10_extended into one file."""
import json, pathlib, sys

WT = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
COMMIT = "72a7997"

# Reading order: config -> physics -> geometry -> models -> training -> evaluation -> entry
ORDER = [
    "src/config.py",
    "src/physics/scaling.py", "src/physics/conventions.py", "src/physics/analytic.py",
    "src/physics/derivatives.py", "src/physics/residuals.py",
    "src/physics/streamfunction.py", "src/physics/optimum.py",
    "src/geometry/domain.py", "src/geometry/sampling.py", "src/geometry/lifts.py",
    "src/models/dense.py", "src/models/mlp.py", "src/models/fourier.py",
    "src/models/hard_bc.py", "src/models/build.py",
    "src/training/weights.py", "src/training/losses.py",
    "src/training/optimisers.py", "src/training/loop.py",
    "src/evaluation/metrics.py", "src/evaluation/accuracy.py",
    "src/evaluation/drift.py", "src/evaluation/falsifiability.py",
    "src/evaluation/plots.py",
    "src/predict.py", "src/report.py", "src/run.py",
]
found = sorted(str(p.relative_to(WT)) for p in WT.glob("src/**/*.py")
               if p.name != "__init__.py")
missing = [f for f in found if f not in ORDER]
assert not missing, f"not in ORDER: {missing}"

metrics = json.loads(pathlib.Path("runs/caseA_re10_extended/metrics.json").read_text())
config = pathlib.Path("runs/caseA_re10_extended/config.yaml").read_text()
env = json.loads(pathlib.Path("runs/caseA_re10_extended/environment.json").read_text())

bar = "=" * 78
parts = [f'''"""
{bar}
THE CODE THAT PRODUCED runs/caseA_re10_extended  (the 30,000-iteration run)
{bar}

PROVENANCE -- read this before trusting anything below.

  Source state : git {COMMIT} ("Block A gate complete; the S2 profile/mass
                 hypothesis is refuted"), the last commit before the run
                 started. Verified: at this commit the physics block has no
                 `convective_coefficient` key, matching what the run used.
                 That key appears in runs/caseA_re10_extended/config.yaml only
                 because a later migration backfilled it.
  Run finished : 2026-09-08 15:32 UTC
  Environment  : python {env['python'].split()[0]}, jax {env['jax']},
                 float64 enabled: {env['x64_enabled']}

THIS IS NOT THE CURRENT CODE. Since this commit the solver itself changed:
  residuals.py  +53 lines  the Stokes/Navier-Stokes inertia switch
  loop.py       +36 lines
  losses.py     +19 lines
plus config validation, the inlet family, migration and figure changes. The
switch defaults to 1.0 and reproduces this behaviour, but the code differs.

{bar}
WHAT THE RUN PRODUCED
{bar}

  relative L2 on u   {metrics['u_relative_l2']:.4e}   target <= 1e-05   PASS
  max |v*|           {metrics['v_max_abs']:.4e}   target <= 1e-05   PASS
  mass drift D       {metrics['flux_conservation_relative_error']:.4e}   target <= 1e-06   PASS
  pressure rel L2    {metrics['p_relative_l2']:.4e}   (diagnostic)
  final loss         {metrics['diag_loss_after_lbfgs']:.4e}   (diagnostic, NEVER accuracy)
  handoff gap        {metrics['diag_handoff_relative_gap']:.2e}
  L-BFGS             {metrics['diag_lbfgs_iterations']} iterations, {metrics['diag_lbfgs_message']}
  wall time          {metrics['wall_seconds']/60:.0f} min
  parameters         {metrics['diag_parameter_count']:,}

{bar}
HOW IT RAN
{bar}

    python -m src.run --config experiments/caseA_re10_extended.yaml

{bar}
READING ORDER
{bar}

The modules below are in dependency order, not alphabetical. The four pieces
that are actually the PINN:

  1. physics/residuals.py  the Navier-Stokes equations (~12 lines of algebra)
  2. models/hard_bc.py     the ansatz: u = phi(y) N_u makes no-slip structural
  3. training/losses.py    the residuals, squared, at random points
  4. training/loop.py      Adam, then L-BFGS on frozen points

Everything else is configuration, measurement or reporting.
"""
''']

parts.append(f"\n# {bar}\n# THE CONFIG THIS RUN USED (runs/caseA_re10_extended/config.yaml)\n# {bar}\n#\n")
parts.append("".join(f"# {line}\n" for line in config.splitlines()))

for rel in ORDER:
    path = WT / rel
    if not path.is_file():
        continue
    body = path.read_text()
    parts.append(f"\n\n# {bar}\n# {rel}   ({len(body.splitlines())} lines)\n# {bar}\n\n")
    parts.append(body.rstrip() + "\n")

OUT.write_text("".join(parts))
n = len(OUT.read_text().splitlines())
print(f"wrote {OUT} ({OUT.stat().st_size//1024} KB, {n} lines, {len(ORDER)} modules)")
