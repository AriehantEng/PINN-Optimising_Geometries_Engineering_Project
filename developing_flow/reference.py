"""A finite-element reference solution for the developing-flow problem.

    python -m developing_flow.reference --config developing_flow/re10.yaml \\
        --nx 100 --ny 24 --x-grading 5.0 --levels 3 \\
        --out developing_flow/runs/reference_re10_m8

Why it exists
-------------
The PINN has an exact solution in only two places: the inlet, where the
profile is prescribed, and far downstream, where the flow approaches the
Poiseuille parabola. Between them, where the flow develops and where every
option differs, nothing checked it. Worse, the parabola is approached only
exponentially, so "developed" is itself an approximation whose size nobody
had measured. This solves the same equations and boundary conditions by a
conventional method, so the PINN can be measured over the whole channel.

The problem, exactly as in developing_flow/pinn.py
--------------------------------------------------
    a (u . grad) u + grad p - nu lap u = 0,    div u = 0,    on [0, L] x [-1, 1]
    (a, nu) = (1, 1/Re) inertial or (Re, 1) viscous pressure scaling
    walls   y = +-1 : u = v = 0
    inlet   x = 0   : u = U0 (1 - y^(2m)), v = 0
    outlet  x = L   : nu du/dx - p = 0 and dv/dx = 0

The outlet line is the natural ("do-nothing") condition of this weak form.
For developed flow it reads du/dx = dv/dx = 0 with p = 0, which is what the
PINN imposes (du/dx, dv/dx softly; p = 0 exactly). The two differ only by the
outlet's departure from developed flow; `--length` solves a longer channel to
measure that.

Method
------
Taylor-Hood Q2/Q1 elements (continuous biquadratic velocity, bilinear
pressure; inf-sup stable) on a tensor mesh graded toward the inlet
(exponential in x) and the walls (Chebyshev-Lobatto in y). Stokes first, then
Newton's method to a relative update below 1e-13; each linear system is
solved directly (SuperLU). The Dirichlet data are nodal interpolants.

Verification (tests/test_reference.py)
--------------------------------------
* Poiseuille inlet: the exact solution is quadratic in y and linear in x,
  inside Q2/Q1, so the discrete solution must match it to round-off.
* Kovasznay flow: an exact Navier-Stokes solution with a nonzero convective
  term, so the Newton linearisation is checked too. Velocity converges at
  third order, pressure at second, as Q2/Q1 theory predicts.
* The reference itself is solved on `--levels` meshes, each twice as fine as
  the last. The observed order and the finest two levels' difference give the
  reference's own error, which is reported next to every PINN error measured
  against it.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import scipy.sparse
import skfem as fem
from skfem.helpers import ddot, dot, mul

from developing_flow import pinn

NEWTON_TOL = 1e-13
NEWTON_MAX = 30


# --------------------------------------------------------------------------- #
# Mesh
# --------------------------------------------------------------------------- #


def x_nodes(nx, length, grading):
    """nx + 1 nodes on [0, length], exponentially crowded toward x = 0.

    Spacing grows by the factor exp(grading / nx) from one element to the next;
    grading = 0 is uniform.
    """
    s = np.linspace(0.0, 1.0, nx + 1)
    if grading == 0.0:
        return length * s
    return length * np.expm1(grading * s) / np.expm1(grading)


def y_nodes(ny):
    """ny + 1 Chebyshev-Lobatto nodes on [-1, 1], crowded toward both walls."""
    return -np.cos(np.pi * np.arange(ny + 1) / ny)


# --------------------------------------------------------------------------- #
# Weak forms (Laplacian form, so the natural outlet condition is nu du/dn - p n = 0)
# --------------------------------------------------------------------------- #


def _div(field):
    return field.grad[0, 0] + field.grad[1, 1]


def _forms(a, nu):
    @fem.BilinearForm
    def jacobian_uu(du, w_, extra):
        u = extra["u0"]
        convection = mul(u.grad, du.value) + mul(du.grad, u.value)
        return a * dot(convection, w_.value) + nu * ddot(du.grad, w_.grad)

    @fem.BilinearForm
    def stokes_uu(du, w_, extra):
        return nu * ddot(du.grad, w_.grad)

    @fem.BilinearForm
    def jacobian_up(dp, w_, extra):  # trial pressure, test velocity
        return -dp.value * _div(w_)

    @fem.LinearForm
    def residual_u(w_, extra):
        u, p = extra["u0"], extra["p0"]
        return (a * dot(mul(u.grad, u.value), w_.value) + nu * ddot(u.grad, w_.grad)
                - p.value * _div(w_))

    @fem.LinearForm
    def residual_p(q, extra):
        return -q.value * _div(extra["u0"])

    return jacobian_uu, stokes_uu, jacobian_up, residual_u, residual_p


def taylor_hood(mesh):
    """(velocity element, pressure element): the inf-sup stable pair for the mesh's cells."""
    if isinstance(mesh, fem.MeshQuad):
        return fem.ElementVector(fem.ElementQuad2()), fem.ElementQuad1()
    if isinstance(mesh, fem.MeshTri):
        return fem.ElementVector(fem.ElementTriP2()), fem.ElementTriP1()
    raise TypeError(f"no Taylor-Hood pair defined for {type(mesh).__name__}")


def solve(mesh, a, nu, dirichlet_marker, dirichlet_value, traction=None,
          traction_facets=None, constraint=None, verbose=False, initial=None):
    """Steady Navier-Stokes by Q2/Q1 finite elements and Newton's method.

    dirichlet_marker(x) -> bool over facet midpoints selects the velocity
    Dirichlet boundary; dirichlet_value(x, y) -> (u, v) gives its values.
    Every other boundary carries the natural condition nu du/dn - p n = t,
    with t = traction(x, y, normal) on `traction_facets` (verification only)
    and t = 0 elsewhere.

    The Taylor-Hood pair follows the mesh: Q2/Q1 on quadrilaterals (the
    channel), P2/P1 on triangles (the bifurcation, bifurcation/reference.py).

    constraint(basis_u) -> (c, target), optional: one linear condition
    c . U = target on the velocity, held by a Lagrange multiplier lam that
    enters the momentum equation as lam c. When c integrates u . n over some
    outlet facets, lam c is a constant extra pressure lam on that outlet: the
    flux-defective boundary condition (Heywood, Rannacher & Turek 1996;
    Formaggia, Gerbeau, Nobile & Quarteroni 2002), which prescribes an
    outlet's flux and lets the outlet's pressure follow. lam is returned as
    info["multiplier"].

    initial, optional: Newton's starting vector (U, P and lam, if constrained),
    for continuation in the Reynolds number from a solution on the same mesh
    with the same boundary data: the U and P (and lam) of that solution,
    concatenated. Without it Newton starts from the Stokes solution.

    Returns (basis_u, basis_p, U, P, info).
    """
    eu, ep = taylor_hood(mesh)
    basis_u = fem.Basis(mesh, eu, intorder=6)
    basis_p = fem.Basis(mesh, ep, intorder=6)
    nu_dofs, np_dofs = basis_u.N, basis_p.N

    dofs = basis_u.get_dofs(dirichlet_marker)
    fixed = [dofs.all("u^1"), dofs.all("u^2")]
    U = np.zeros(nu_dofs)
    for component, index in enumerate(fixed):
        U[index] = dirichlet_value(*basis_u.doflocs[:, index])[component]
    fixed_global = np.concatenate(fixed)  # the velocity block comes first

    jacobian_uu, stokes_uu, jacobian_up, residual_u, residual_p = _forms(a, nu)
    B = fem.asm(jacobian_up, basis_p, basis_u)            # (n_u, n_p)
    load = np.zeros(nu_dofs)
    if traction is not None:
        facet_basis = fem.FacetBasis(mesh, eu, facets=traction_facets, intorder=8)

        @fem.LinearForm
        def boundary_load(w_, extra):
            t = traction(extra.x[0], extra.x[1], extra.n)
            return dot(t, w_.value)

        load = fem.asm(boundary_load, facet_basis)

    if constraint is None:
        def system(uu):
            return scipy.sparse.bmat([[uu, B], [B.T, None]], format="csr")
        extra, target = 0, None
    else:
        c, target = constraint(basis_u)
        c = scipy.sparse.csr_matrix(np.asarray(c, dtype=np.float64).reshape(-1, 1))

        def system(uu):
            return scipy.sparse.bmat([[uu, B, c], [B.T, None, None], [c.T, None, None]],
                                     format="csr")
        extra = 1

    # Stokes: a linear solve that gives Newton a start satisfying the boundary data.
    x = np.concatenate([U, np.zeros(np_dofs + extra)])
    K = system(fem.asm(stokes_uu, basis_u))
    rhs = np.concatenate([load, np.zeros(np_dofs + extra)])
    if extra:
        rhs[-1] = target
    x = fem.solve(*fem.condense(K, rhs, x=x, D=fixed_global))
    if initial is not None:
        initial = np.asarray(initial, dtype=np.float64)
        if initial.shape != x.shape:
            raise ValueError(f"initial has shape {initial.shape}, the system {x.shape}")
        if np.max(np.abs(initial[fixed_global] - x[fixed_global])) > 1e-12:
            raise ValueError("initial does not carry this problem's boundary data")
        x = initial.copy()

    history = []
    start = time.perf_counter()
    for iteration in range(1, NEWTON_MAX + 1):
        U, P = x[:nu_dofs], x[nu_dofs:nu_dofs + np_dofs]
        u0 = basis_u.interpolate(U)
        p0 = basis_p.interpolate(P)
        R = [fem.asm(residual_u, basis_u, u0=u0, p0=p0) - load,
             fem.asm(residual_p, basis_p, u0=u0)]
        if extra:
            R[0] = R[0] + x[-1] * c.toarray().ravel()
            R.append(np.ravel(c.T @ U) - target)
        R = np.concatenate(R)
        J = system(fem.asm(jacobian_uu, basis_u, u0=u0))
        delta = fem.solve(*fem.condense(J, -R, D=fixed_global))
        x = x + delta
        free = np.setdiff1d(np.arange(x.size), fixed_global)
        residual_norm = float(np.linalg.norm(R[free]))
        update = float(np.max(np.abs(delta)) / max(np.max(np.abs(x)), 1e-300))
        history.append({"iteration": iteration, "residual": residual_norm, "update": update})
        if verbose:
            print(f"    newton {iteration:2d}  |R| {residual_norm:.3e}  update {update:.3e}",
                  flush=True)
        if update < NEWTON_TOL:
            break
    else:
        raise RuntimeError(f"Newton did not converge in {NEWTON_MAX} iterations: {history[-1]}")

    U, P = x[:nu_dofs], x[nu_dofs:nu_dofs + np_dofs]
    info = {"newton": history, "seconds": time.perf_counter() - start,
            "unknowns": int(x.size), "elements": int(mesh.nelements)}
    if extra:
        info["multiplier"] = float(x[-1])
    return basis_u, basis_p, U, P, info


def _lagrange_weights(nodes, points, degree):
    """Element index and 1D Lagrange weights of `points` on a 1D mesh.

    degree 2: nodes at each element's ends and midpoint (Q2); degree 1: ends (Q1).
    Returns (first global node of the element, weights of shape (n, degree + 1)).
    """
    element = np.clip(np.searchsorted(nodes, points, side="right") - 1, 0, len(nodes) - 2)
    left, right = nodes[element], nodes[element + 1]
    t = (points - left) / (right - left)          # 0 .. 1 within the element
    if degree == 1:
        return element, np.stack([1.0 - t, t], axis=1)
    return 2 * element, np.stack([2.0 * (t - 0.5) * (t - 1.0),
                                  -4.0 * t * (t - 1.0),
                                  2.0 * t * (t - 0.5)], axis=1)


class LatticeSolution:
    """(u, v, p) of a Q2/Q1 solution at arbitrary points, from nodal values alone.

    The mesh is a tensor product of rectangles, so Q2 velocity and Q1 pressure
    are tensor products of 1D Lagrange interpolants on each element. Evaluation
    uses that directly instead of a general point search, which does not scale
    to millions of points; `tests/test_reference.py` checks it against
    scikit-fem's own `probes`. Needs no scikit-fem objects, so a solved
    reference can be stored (`save`) and evaluated anywhere later
    (`load_solution`), not only on the grid it was sampled on.
    """

    def __init__(self, x_nodes, y_nodes, lattice_u, lattice_v, lattice_p):
        self.x_nodes, self.y_nodes = np.asarray(x_nodes), np.asarray(y_nodes)
        self.lattice_u, self.lattice_v = np.asarray(lattice_u), np.asarray(lattice_v)
        self.lattice_p = np.asarray(lattice_p)
        q2 = (2 * len(self.x_nodes) - 1, 2 * len(self.y_nodes) - 1)
        if self.lattice_u.shape != q2 or self.lattice_v.shape != q2 \
                or self.lattice_p.shape != (len(self.x_nodes), len(self.y_nodes)):
            raise ValueError("lattice shapes do not match the mesh nodes")

    def save(self, path):
        np.savez_compressed(path, x_nodes=self.x_nodes, y_nodes=self.y_nodes,
                            u=self.lattice_u, v=self.lattice_v, p=self.lattice_p)

    @staticmethod
    def _to_lattice(values, doflocs, xs, ys):
        i = np.searchsorted(xs, doflocs[0])
        j = np.searchsorted(ys, doflocs[1])
        # Nodes computed two ways (mesh vertices, midpoints) may differ in the last bit.
        i = np.where((i > 0) & (np.abs(xs[np.maximum(i - 1, 0)] - doflocs[0])
                                < np.abs(xs[np.minimum(i, len(xs) - 1)] - doflocs[0])), i - 1, i)
        j = np.where((j > 0) & (np.abs(ys[np.maximum(j - 1, 0)] - doflocs[1])
                                < np.abs(ys[np.minimum(j, len(ys) - 1)] - doflocs[1])), j - 1, j)
        i, j = np.minimum(i, len(xs) - 1), np.minimum(j, len(ys) - 1)
        if (np.max(np.abs(xs[i] - doflocs[0])) > 1e-12 or np.max(np.abs(ys[j] - doflocs[1])) > 1e-12):
            raise RuntimeError("degrees of freedom do not sit on the tensor lattice")
        lattice = np.full((len(xs), len(ys)), np.nan)
        lattice[i, j] = values
        if np.isnan(lattice).any():
            raise RuntimeError("the tensor lattice is not fully covered by degrees of freedom")
        return lattice


    def __call__(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        xf, yf = x.ravel(), y.ravel()
        out = []
        for lattice, degree, xs, ys in ((self.lattice_u, 2, self.x_nodes, self.y_nodes),
                                        (self.lattice_v, 2, self.x_nodes, self.y_nodes),
                                        (self.lattice_p, 1, self.x_nodes, self.y_nodes)):
            i0, wx = _lagrange_weights(xs, xf, degree)
            j0, wy = _lagrange_weights(ys, yf, degree)
            value = np.zeros(xf.shape)
            for a in range(degree + 1):
                for b in range(degree + 1):
                    value += wx[:, a] * wy[:, b] * lattice[i0 + a, j0 + b]
            out.append(value.reshape(x.shape))
        return tuple(out)


class Solution(LatticeSolution):
    """A solved field: the scikit-fem objects, and the lattice evaluator."""

    def __init__(self, basis_u, basis_p, U, P, info, x_nodes, y_nodes):
        self.basis_u, self.basis_p, self.U, self.P, self.info = basis_u, basis_p, U, P, info
        x_nodes, y_nodes = np.asarray(x_nodes), np.asarray(y_nodes)
        (u, scalar), (v, _) = basis_u.split(U)
        # Q2 node lattice: vertices plus midpoints, (2 nx + 1) x (2 ny + 1).
        self.x2 = np.empty(2 * len(x_nodes) - 1)
        self.x2[0::2], self.x2[1::2] = x_nodes, 0.5 * (x_nodes[1:] + x_nodes[:-1])
        self.y2 = np.empty(2 * len(y_nodes) - 1)
        self.y2[0::2], self.y2[1::2] = y_nodes, 0.5 * (y_nodes[1:] + y_nodes[:-1])
        super().__init__(x_nodes, y_nodes,
                         self._to_lattice(u, scalar.doflocs, self.x2, self.y2),
                         self._to_lattice(v, scalar.doflocs, self.x2, self.y2),
                         self._to_lattice(P, basis_p.doflocs, x_nodes, y_nodes))

    def lattice(self):
        """The same solution without the scikit-fem objects: small, storable."""
        return LatticeSolution(self.x_nodes, self.y_nodes, self.lattice_u, self.lattice_v,
                               self.lattice_p)


def load_solution(directory):
    """The finest reference mesh's solution, evaluable anywhere (solution.npz)."""
    path = pathlib.Path(directory) / "solution.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: this reference was written before "
                                "solutions were stored; rerun developing_flow.reference")
    with np.load(path) as data:
        return LatticeSolution(data["x_nodes"], data["y_nodes"], data["u"], data["v"], data["p"])


#: Gauss-Legendre points per element and direction in every error norm. The
#: integrand (smooth network minus a Q2/Q1 polynomial) is integrated element by
#: element on the reference's own mesh, which crowds the inlet and the walls
#: down to elements 0.00085 x 0.00054. A uniform sampling grid cannot do this:
#: the inlet corners carry a pressure spike narrower than its spacing, and the
#: x = 0 column, weighted like any other, then supplied half of the "p error"
#: (see norm_quadrature's test). Checked converged in the tests.
NORM_GAUSS = 6


def norm_quadrature(x_nodes, y_nodes, length, n_gauss=NORM_GAUSS):
    """(X, Y, W): Gauss points in every element of a tensor mesh with x <= length,
    and their weights, as (n_x, n_y) arrays. sum(W f) integrates f."""
    xs = np.asarray(x_nodes)
    xs = xs[xs <= length * (1.0 + 1e-12)]
    if abs(xs[-1] - length) > 1e-9 * length:
        raise ValueError(f"the mesh has no node at x = {length}; cannot integrate to there")
    g, w = np.polynomial.legendre.leggauss(n_gauss)

    def composite(nodes):
        left, right = nodes[:-1, None], nodes[1:, None]
        return ((0.5 * (right - left) * g + 0.5 * (right + left)).ravel(),
                (0.5 * (right - left) * w).ravel())

    x, wx = composite(xs)
    y, wy = composite(np.asarray(y_nodes))
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y, np.outer(wx, wy)


def relative_errors(model, solution, length, developed_from, n_gauss=NORM_GAUSS):
    """Relative L2 errors of model(x, y) -> (u, v, p) against a reference
    solution, by `norm_quadrature` on the reference's mesh: whole channel for
    u, v, p, and u split at x = developed_from."""
    X, Y, W = norm_quadrature(solution.x_nodes, solution.y_nodes, length, n_gauss)
    u, v, p = model(X, Y)
    ru, rv, rp = solution(X, Y)
    entrance = X < developed_from

    def rel(a, b, mask=None):
        mask = np.ones(a.shape, bool) if mask is None else mask
        return float(np.sqrt(np.sum((W * (a - b) ** 2)[mask]) / np.sum((W * b**2)[mask])))

    return {"u": rel(u, ru), "u_entrance": rel(u, ru, entrance),
            "u_developed": rel(u, ru, ~entrance), "v": rel(v, rv), "p": rel(p, rp),
            "u_max_at_gauss_points": float(np.max(np.abs(u - ru))),
            "quadrature_points": int(X.size), "gauss_per_direction": n_gauss}


def solve_channel(reynolds, pressure_scaling, sharpness, length, nx, ny, grading,
                  verbose=False):
    """The developing-flow problem of developing_flow/pinn.py, on an nx x ny mesh."""
    a, nu = pinn.scaling_coefficients(reynolds, pressure_scaling)
    xs, ys = x_nodes(nx, length, grading), y_nodes(ny)
    mesh = fem.MeshQuad.init_tensor(xs, ys)

    def marker(x):
        return np.isclose(x[0], 0.0) | np.isclose(np.abs(x[1]), 1.0)

    def value(x, y):
        inlet = np.isclose(x, 0.0) & ~np.isclose(np.abs(y), 1.0)
        return np.where(inlet, pinn.inlet_profile(y, sharpness), 0.0), np.zeros_like(x)

    return Solution(*solve(mesh, a, nu, marker, value, verbose=verbose), xs, ys)


# --------------------------------------------------------------------------- #
# Kovasznay flow: an exact steady Navier-Stokes solution, for verification
# --------------------------------------------------------------------------- #


def kovasznay(reynolds):
    """(u, v, p, lambda) of Kovasznay (1948) for nu = 1/Re, a = 1."""
    lam = reynolds / 2.0 - math.sqrt(reynolds**2 / 4.0 + 4.0 * math.pi**2)

    def u(x, y):
        return 1.0 - np.exp(lam * x) * np.cos(2.0 * np.pi * y)

    def v(x, y):
        return lam / (2.0 * np.pi) * np.exp(lam * x) * np.sin(2.0 * np.pi * y)

    def p(x, y):
        return 0.5 * (1.0 - np.exp(2.0 * lam * x))

    return u, v, p, lam


def solve_kovasznay(reynolds, n):
    """Kovasznay flow on [-0.5, 1] x [-0.5, 1.5], n x n uniform elements.

    Velocity is prescribed on three sides; the right side carries the exact
    traction nu du/dn - p n, which also tests the natural outlet condition.
    """
    u_exact, v_exact, p_exact, lam = kovasznay(reynolds)
    nu = 1.0 / reynolds
    xs, ys = np.linspace(-0.5, 1.0, n + 1), np.linspace(-0.5, 1.5, n + 1)
    mesh = fem.MeshQuad.init_tensor(xs, ys)

    def marker(x):
        return ~np.isclose(x[0], 1.0)

    def value(x, y):
        return u_exact(x, y), v_exact(x, y)

    def traction(x, y, normal):
        # grad u of the exact solution, then nu (grad u) n - p n
        e = np.exp(lam * x)
        c, s = np.cos(2 * np.pi * y), np.sin(2 * np.pi * y)
        u_x, u_y = -lam * e * c, 2 * np.pi * e * s
        v_x, v_y = lam**2 / (2 * np.pi) * e * s, lam * e * c
        p = p_exact(x, y)
        return np.stack([nu * (u_x * normal[0] + u_y * normal[1]) - p * normal[0],
                         nu * (v_x * normal[0] + v_y * normal[1]) - p * normal[1]])

    facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], 1.0))
    solution = Solution(*solve(mesh, 1.0, nu, marker, value, traction, facets), xs, ys)
    return solution, (u_exact, v_exact, p_exact)


# --------------------------------------------------------------------------- #
# The reference: several meshes, convergence, and the fields on a fixed grid
# --------------------------------------------------------------------------- #


def energy_budget(solution, a, nu, length):
    """The mechanical-energy budget of a solution, each term to quadrature accuracy.

    Dotting momentum with u and integrating over the channel gives, for steady
    incompressible flow with no-slip walls,

        inflow - outflow = dissipation + boundary viscous work,
        inflow  = int_inlet  (p + a |u|^2 / 2) u dy,
        outflow = int_outlet (p + a |u|^2 / 2) u dy,
        dissipation = 2 nu int e:e dA,   e the strain-rate tensor.

    With these boundary conditions the boundary work vanishes: v = 0 along the
    inlet makes v_y = 0 there, hence u_x = 0, and the outlet is developed. So
    (inflow - outflow) / Q is the flux-weighted total-pressure drop, the
    objective of the bifurcation study, and it must equal dissipation / Q.

    Line integrals use Gauss-Legendre on every element of the y-mesh (exact
    for the piecewise polynomials); the dissipation is assembled on the mesh.
    """
    from skfem.helpers import transpose

    @fem.Functional
    def dissipation(w):
        e = 0.5 * (w["u"].grad + transpose(w["u"].grad))
        return 2.0 * nu * ddot(e, e)

    phi = float(dissipation.assemble(solution.basis_u,
                                     u=solution.basis_u.interpolate(solution.U)))
    nodes, weights = np.polynomial.legendre.leggauss(6)
    left, right = solution.y_nodes[:-1, None], solution.y_nodes[1:, None]
    y = (0.5 * (right - left) * nodes[None, :] + 0.5 * (right + left)).ravel()
    w = (0.5 * (right - left) * weights[None, :]).ravel()

    def through(x):
        u, v, p = solution(np.full_like(y, x), y)
        return (float(u @ w), float(((p + 0.5 * a * (u**2 + v**2)) * u) @ w),
                float((p * u) @ w))

    q_in, total_in, static_in = through(0.0)
    q_out, total_out, static_out = through(length)
    return {"flux_in": q_in, "flux_out": q_out, "energy_in": total_in, "energy_out": total_out,
            "dissipation": phi,
            "total_pressure_drop": (total_in - total_out) / q_in,
            "static_pressure_drop": (static_in - static_out) / q_in,
            "energy_consistency": abs(total_in - total_out - phi) / phi}


def grid_convergence_index(coarse, medium, fine, ratio=2.0, safety=1.25):
    """Celik et al. (2008) GCI of a scalar from three meshes refined by `ratio`.

    Returns the observed order, the Richardson-extrapolated value and the GCI
    of the finest mesh as a fraction of its value. Valid for integrated
    quantities, whose error has a smooth expansion in h (unlike values
    sampled between nodes, where Richardson extrapolation failed its
    Kovasznay check).
    """
    e32, e21 = medium - coarse, fine - medium
    if e21 == 0.0 or e32 == 0.0 or e32 / e21 <= 0.0:
        return {"order": None, "extrapolated": None, "gci_fine": None,
                "note": "oscillatory or exact convergence: no GCI"}
    order = math.log(abs(e32 / e21)) / math.log(ratio)
    extrapolated = fine + e21 / (ratio**order - 1.0)
    gci = safety * abs(e21 / fine) / (ratio**order - 1.0)
    return {"order": order, "extrapolated": extrapolated, "gci_fine": gci}


def developed_measures(solution, length):
    """Flux at 201 stations, centreline, and the entrance length of the solution."""
    nodes, weights = np.polynomial.legendre.leggauss(200)
    stations = np.linspace(0.0, length, 201)
    sx, sy = np.meshgrid(stations, nodes, indexing="ij")
    u, _, _ = solution(sx, sy)
    flux = u @ weights
    xc = np.linspace(0.0, length, pinn.CENTRELINE_POINTS)
    uc, _, pc = solution(xc, np.zeros_like(xc))
    entrance = pinn.entrance_length(solution, length)   # exactly as for a PINN
    return {"stations": stations, "flux": flux, "xc": xc, "uc": uc, "pc": pc,
            "entrance_length": entrance}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="a developing-flow PINN config")
    parser.add_argument("--nx", type=int, required=True, help="elements along x, coarsest level")
    parser.add_argument("--ny", type=int, required=True, help="elements across, coarsest level")
    parser.add_argument("--x-grading", type=float, required=True,
                        help="exponential crowding toward the inlet; 0 is uniform")
    parser.add_argument("--levels", type=int, required=True,
                        help="number of meshes, each twice as fine as the last")
    parser.add_argument("--length", type=float, default=None,
                        help="solve a LONGER channel than the config's, to measure the outlet's "
                             "influence; fields are still sampled on [0, config length]")
    parser.add_argument("--grid", type=int, nargs=2, required=True, metavar=("NX", "NY"),
                        help="uniform grid the finest fields are written on")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    config = pinn.load_config(args.config)
    length = config.length if args.length is None else args.length
    if length < config.length:
        parser.error("--length must not be shorter than the config's channel")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    levels = []
    for level in range(args.levels):
        nx, ny = args.nx * 2**level, args.ny * 2**level
        print(f"level {level}: {nx} x {ny} elements", flush=True)
        solution = solve_channel(config.reynolds, config.pressure_scaling,
                                 config.inlet_sharpness, length, nx, ny, args.x_grading,
                                 verbose=True)
        measures = developed_measures(solution, config.length)
        a, nu = pinn.scaling_coefficients(config.reynolds, config.pressure_scaling)
        # The budget closes only over the channel the PINN solves.
        energy = energy_budget(solution, a, nu, config.length) if length == config.length else None
        levels.append({"nx": nx, "ny": ny, "solution": solution.lattice(), "measures": measures,
                       "info": solution.info, "energy": energy})
        print(f"  {solution.info['unknowns']} unknowns, {solution.info['seconds']:.0f} s, "
              f"entrance length {measures['entrance_length']:.6f}", flush=True)

    # Differences between successive levels, and the observed order, as L2
    # norms integrated on the FINEST mesh (the levels nest, so every coarser
    # solution is a polynomial on each finest element and the integrals are
    # exact). The maximum is over the same Gauss points.
    names = ("u", "v", "p")
    finest_solution = levels[-1]["solution"]
    QX, QY, QW = norm_quadrature(finest_solution.x_nodes, finest_solution.y_nodes, config.length)
    fields = [L["solution"](QX, QY) for L in levels]
    convergence = []
    for k in range(1, len(levels)):
        row = {"coarse": [levels[k - 1]["nx"], levels[k - 1]["ny"]],
               "fine": [levels[k]["nx"], levels[k]["ny"]]}
        for i, name in enumerate(names):
            diff = fields[k][i] - fields[k - 1][i]
            scale = fields[k][i] if name != "v" else fields[k][0]
            row[f"{name}_l2_change"] = float(np.sqrt(np.sum(QW * diff**2) / np.sum(QW * scale**2)))
            row[f"{name}_max_change"] = float(np.max(np.abs(diff)))
        row["entrance_length_change"] = float(
            levels[k]["measures"]["entrance_length"] - levels[k - 1]["measures"]["entrance_length"])
        convergence.append(row)
    for k in range(1, len(convergence)):
        for name in names:
            before, after = convergence[k - 1][f"{name}_l2_change"], convergence[k][f"{name}_l2_change"]
            convergence[k][f"{name}_observed_order"] = float(math.log2(before / after))

    finest = levels[-1]
    solution_x = np.linspace(0.0, config.length, args.grid[0])
    solution_y = np.linspace(-1.0, 1.0, args.grid[1])
    fx, fy = np.meshgrid(solution_x, solution_y, indexing="ij")
    print(f"sampling the finest level on {args.grid[0]} x {args.grid[1]}", flush=True)
    u, v, p = solution(fx, fy)
    np.savez_compressed(out / "reference.npz", x=solution_x, y=solution_y, u=u, v=v, p=p)
    finest_solution.save(out / "solution.npz")

    m = finest["measures"]
    parabola = pinn.developed_profile(solution_y, pinn.INLET_FLUX)
    departure = np.max(np.abs(u - parabola[None, :]), axis=1)
    summary = {
        "config": str(args.config),
        "reynolds": config.reynolds,
        "pressure_scaling": config.pressure_scaling,
        "inlet_sharpness": config.inlet_sharpness,
        "length_solved": length,
        "length_sampled": config.length,
        "x_grading": args.x_grading,
        "levels": [{"nx": L["nx"], "ny": L["ny"], **L["info"],
                    "entrance_length": L["measures"]["entrance_length"],
                    "energy": L["energy"]} for L in levels],
        "energy": finest["energy"],
        "grid_convergence_index": {
            key: grid_convergence_index(*(L["energy"][key] for L in levels[-3:]))
            for key in ("dissipation", "total_pressure_drop", "static_pressure_drop")}
        if len(levels) >= 3 and finest["energy"] else {},
        "convergence": convergence,
        "convergence_norm": (f"relative L2, Gauss-Legendre {NORM_GAUSS} x {NORM_GAUSS} in every "
                             "element of the finest mesh"),
        "entrance_length": m["entrance_length"],
        "flux_max_relative_error": float(np.max(np.abs(m["flux"] / pinn.INLET_FLUX - 1.0))),
        "centreline_pressure_drop": float(m["pc"][0] - m["pc"][-1]),
        "grid": list(args.grid),
        # Largest |u - parabola| across the channel, at stations along it: how
        # far from "developed" the flow still is.
        "departure_from_parabola": {f"{x:g}": float(departure[np.argmin(np.abs(solution_x - x))])
                                    for x in (0.5, 1, 2, 3, 3.77, 4, 5, 6, 7.5, 8, 10)},
    }
    with (out / "reference.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k: summary[k] for k in ("convergence", "energy", "grid_convergence_index",
                                              "entrance_length",
                                              "flux_max_relative_error",
                                              "departure_from_parabola")}, indent=2))
    return 0


def load(directory):
    """(x, y, u, v, p) of a stored reference, on its uniform grid."""
    data = np.load(pathlib.Path(directory) / "reference.npz")
    return data["x"], data["y"], data["u"], data["v"], data["p"]


if __name__ == "__main__":
    raise SystemExit(main())
