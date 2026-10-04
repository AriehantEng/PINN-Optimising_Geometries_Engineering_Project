"""A finite-element reference for the bifurcation: the judge of Networks 2 and 3.

The same steady Navier-Stokes problem as developing_flow, on the bifurcation
polygon of bifurcation/geometry.py:

    a (u . grad) u + grad p - nu lap u = 0,   div u = 0
    inlet  x* = 0          u, v = the inlet profile (Network 1's flow at a
                            station of the parent channel, or an analytic one)
    walls                   u = v = 0
    outlets 1 and 2         nu du/dn - p n = 0 (do-nothing): equal outlet
                            pressures, so the flow chooses its own split

Taylor-Hood P2/P1 on a Triangle mesh (quality: minimum angle 30 degrees, a
maximum element area), refined locally around the three re-entrant corners
(the parent's corners B and C, and the septum tip T), where the pressure is
singular. The nonlinear solve is developing_flow.reference.solve (Stokes, then
Newton to a 1e-13 update), whose quadrilateral version is verified on
Poiseuille and Kovasznay flow; the weak forms do not depend on the cell shape.

What it measures, for a given geometry:
    flux in, flux out of each daughter, and the split Q1 / Q_in
    the flux-weighted total-pressure drop (E_in - E_out1 - E_out2) / Q_in,
        E = int (p + a |u|^2 / 2) u . n over an opening
    the dissipation Phi = 2 nu int e:e, and the resistance Phi / Q_in^2,
        which the developing-flow work found to be the robust objective
    the mean static pressure on each opening

"Lowest pressure drop" is then a statement this module can PROVE for a given
pair of angles: sweep the split at fixed angles and find the minimum of the
total-pressure drop (`sweep_split`).
"""

from __future__ import annotations

import time

import numpy as np
import skfem as fem
import triangle
from skfem.helpers import ddot, dot, sym_grad

from bifurcation.geometry import Geometry
from developing_flow import pinn
from developing_flow import reference as channel

#: The re-entrant corners, by vertex index in Geometry.vertices(): B, T, C.
CORNERS = (1, 4, 7)


def mesh(geometry: Geometry, max_area, corner_levels, corner_radius):
    """A quality triangle mesh of the polygon, refined around the corners.

    max_area       largest element area anywhere
    corner_levels  rounds of refinement near the corners; round k refines the
                   elements whose centroid lies within corner_radius / 2^k
    """
    v = geometry.vertices()
    n = len(v)
    pslg = {"vertices": v, "segments": np.array([[i, (i + 1) % n] for i in range(n)])}
    out = triangle.triangulate(pslg, f"pq30a{max_area:.12f}Q")
    m = fem.MeshTri(np.ascontiguousarray(out["vertices"].T),
                    np.ascontiguousarray(out["triangles"].T))
    corners = v[list(CORNERS)]
    for k in range(corner_levels):
        centroid = m.p[:, m.t].mean(axis=1).T
        distance = np.min(np.linalg.norm(centroid[:, None, :] - corners[None], axis=2), axis=1)
        marked = np.nonzero(distance < corner_radius / 2.0**k)[0]
        if len(marked):
            m = m.refined(marked)
    return m


def _distance_to_segment(points, a, b):
    t = np.clip(((points - a) @ (b - a)) / np.dot(b - a, b - a), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * (b - a)), axis=1)


def boundary_facets(m, geometry: Geometry, tol=1e-9):
    """{kind: boundary facet indices}, each facet assigned to the edge it lies on."""
    facets = m.boundary_facets()
    mid = m.p[:, m.facets[:, facets]].mean(axis=1).T
    edges = geometry.edges()
    distance = np.stack([_distance_to_segment(mid, a, b) for a, b, _ in edges], axis=1)
    nearest = np.argmin(distance, axis=1)
    if np.max(distance[np.arange(len(facets)), nearest]) > tol * max(1.0, geometry.daughter_length):
        raise RuntimeError("a boundary facet lies on no edge of the polygon")
    kinds = {}
    for (_, _, kind), i in zip(edges, range(len(edges))):
        kinds.setdefault(kind, []).append(facets[nearest == i])
    return {kind: np.concatenate(parts) for kind, parts in kinds.items()}


def solve_bifurcation(geometry: Geometry, reynolds, pressure_scaling, inlet, max_area,
                      corner_levels, corner_radius, verbose=False):
    """Solve one geometry. `inlet(y) -> (u, v)` arrays: the inflow at x* = 0."""
    a, nu = pinn.scaling_coefficients(reynolds, pressure_scaling)
    m = mesh(geometry, max_area, corner_levels, corner_radius)
    facets = boundary_facets(m, geometry)
    dirichlet = np.concatenate([facets["inlet"], facets["wall"]])
    on_inlet_line = geometry.edges()[-1]
    a_in, b_in = on_inlet_line[0], on_inlet_line[1]

    def marker(x):
        # skfem calls this with facet midpoints; mark the inlet and every wall
        pts = np.asarray(x).T
        d_in = _distance_to_segment(pts, a_in, b_in)
        d_wall = np.min(np.stack([_distance_to_segment(pts, p0, p1)
                                  for p0, p1, k in geometry.edges() if k == "wall"]), axis=0)
        return (d_in < 1e-9) | (d_wall < 1e-9)

    def value(x, y):
        x, y = np.asarray(x), np.asarray(y)
        u, v = np.zeros_like(x), np.zeros_like(x)
        at_inlet = np.abs(x) < 1e-12
        if at_inlet.any():
            ui, vi = inlet(y[at_inlet])
            u[at_inlet], v[at_inlet] = ui, vi
        return u, v

    start = time.perf_counter()
    basis_u, basis_p, U, P, info = channel.solve(m, a, nu, marker, value, verbose=verbose)
    info = {**info, "total_seconds": time.perf_counter() - start, "triangles": int(m.nelements),
            "dirichlet_facets": int(len(dirichlet))}
    return BifurcationSolution(geometry, m, facets, basis_u, basis_p, U, P, info, a, nu)


class BifurcationSolution:
    def __init__(self, geometry, m, facets, basis_u, basis_p, U, P, info, a, nu):
        self.geometry, self.mesh, self.facets = geometry, m, facets
        self.basis_u, self.basis_p, self.U, self.P, self.info = basis_u, basis_p, U, P, info
        self.a, self.nu = a, nu

    def _opening(self, kind):
        """(flux out, energy out, mean static pressure) through one opening, by
        facet quadrature; 'out' along the outward normal (negative at the inlet)."""
        fb_u = fem.FacetBasis(self.mesh, self.basis_u.elem, facets=self.facets[kind], intorder=8)
        fb_p = fem.FacetBasis(self.mesh, self.basis_p.elem, facets=self.facets[kind], intorder=8)
        u = fb_u.interpolate(self.U)
        p = fb_p.interpolate(self.P)
        a = self.a
        flux = fem.Functional(lambda w: dot(w["u"].value, w.n)).assemble(fb_u, u=u)
        energy = fem.Functional(
            lambda w: (w["p"].value + 0.5 * a * dot(w["u"].value, w["u"].value))
            * dot(w["u"].value, w.n)).assemble(fb_u, u=u, p=p)
        length = fem.Functional(lambda w: 1.0 + 0.0 * w.x[0]).assemble(fb_p)
        pressure = fem.Functional(lambda w: w["p"].value).assemble(fb_p, p=p) / length
        return float(flux), float(energy), float(pressure)

    def measures(self):
        q_in, e_in, p_in = self._opening("inlet")
        q1, e1, p1 = self._opening("outlet1")
        q2, e2, p2 = self._opening("outlet2")
        q_in, e_in = -q_in, -e_in                      # inflow, along the inward normal
        nu = self.nu
        dissipation = float(fem.Functional(
            lambda w: 2.0 * nu * ddot(sym_grad(w["u"]), sym_grad(w["u"]))
        ).assemble(self.basis_u, u=self.basis_u.interpolate(self.U)))
        return {
            "flux_in": q_in, "flux_out1": q1, "flux_out2": q2,
            "split_flux": q1 / q_in,
            "mass_imbalance": (q1 + q2 - q_in) / q_in,
            "total_pressure_drop": (e_in - e1 - e2) / q_in,
            "dissipation": dissipation,
            "resistance": dissipation / q_in**2,
            "energy_consistency": abs(e_in - e1 - e2 - dissipation) / dissipation,
            "static_pressure_in": p_in, "static_pressure_out1": p1, "static_pressure_out2": p2,
            "seconds": self.info["total_seconds"], "triangles": self.info["triangles"],
            "unknowns": self.info["unknowns"],
        }

    def __call__(self, x, y, chunk=20000):
        """(u, v, p) at arbitrary points inside the polygon."""
        pts = np.stack([np.ravel(x), np.ravel(y)])
        (u_dofs, scalar), (v_dofs, _) = self.basis_u.split(self.U)
        out = [[], [], []]
        for i in range(0, pts.shape[1], chunk):
            block = pts[:, i:i + chunk]
            probe = scalar.probes(block)
            out[0].append(probe @ u_dofs)
            out[1].append(probe @ v_dofs)
            out[2].append(self.basis_p.probes(block) @ self.P)
        shape = np.shape(x)
        return tuple(np.concatenate(o).reshape(shape) for o in out)


def parabola_inlet(y):
    """The developed inflow 1 - y^2, v = 0: an inlet with a known answer upstream."""
    y = np.asarray(y, dtype=np.float64)
    return 1.0 - y**2, np.zeros_like(y)


def network_inlet(run_directory, station):
    """Network 1's flow at x* = `station` of its channel, as an inlet(y)."""
    model = pinn.load_run(run_directory)
    if not 0.0 <= station <= model.config.length:
        raise ValueError(f"station {station} is outside Network 1's channel "
                         f"[0, {model.config.length}]")

    def inlet(y):
        y = np.asarray(y, dtype=np.float64)
        u, v, _ = model(np.full_like(y, station), y)
        return u, v

    inlet.config = model.config
    return inlet


def sweep_split(splits, angle1, angle2, parent_length, daughter_length, min_width, reynolds,
                pressure_scaling, inlet, max_area, corner_levels, corner_radius, verbose=False):
    """The objective at each split for fixed angles: the brute-force proof of a minimum."""
    rows = []
    for s in splits:
        g = Geometry.from_split(float(s), angle1, angle2, parent_length, daughter_length,
                                min_width)
        sol = solve_bifurcation(g, reynolds, pressure_scaling, inlet, max_area, corner_levels,
                                corner_radius)
        rows.append({"split": float(s), **sol.measures()})
        if verbose:
            r = rows[-1]
            print(f"  split {s:.4f}: total-pressure drop {r['total_pressure_drop']:.8f}, "
                  f"flow split {r['split_flux']:.5f}, {r['seconds']:.1f} s", flush=True)
    return rows
