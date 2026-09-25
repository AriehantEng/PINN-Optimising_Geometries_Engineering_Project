"""Re-solve the output layer of a trained developing-flow PINN on frozen features.

    python -m developing_flow.refine developing_flow/runs/developing_re10 \\
        --out developing_flow/runs/developing_re10_refined \\
        --points 16384 --boundary-points 1024 --iterations 50 --seed 2024

The hidden layers are frozen, so the network becomes a fixed set of basis
functions h_k(x, y), k = 1..width, and the only unknowns are the output-layer
coefficients: 3 x width weights plus 3 biases. With the ansatz

    u = phi(y) (a_u . h + c_u),  v = phi(y) (a_v . h + c_v),
    p = ((L - x)/L)(a_p . h + c_p),

every quantity the loss needs is LINEAR in those coefficients except the
inertial terms u u_x + v u_y, which are quadratic. So this is NOT the single
linear solve that works for a linear PDE: it is a small nonlinear least-squares
problem, solved here by Levenberg-Marquardt on the exact Jacobian.

The residual vector is scaled so that its squared norm is exactly the
training loss of `pinn.total_loss`; that identity is checked before solving.
The re-solved coefficients are written as a new run directory, which
`compare` and `verify` accept like any other.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from developing_flow import pinn
from developing_flow.compare import load


def feature_derivatives(config, params, points):
    """(H, H_x, H_y, H_xx, H_yy): the frozen basis and its derivatives."""
    L = config.length

    def features(xy):
        return pinn.hidden_features(params, jnp.stack([2.0 * xy[0] / L - 1.0, xy[1]]),
                                    config.activation)

    batched = jax.vmap(features)
    value, d_dx, d2_dx2 = pinn._directional(batched, points, pinn._EX)
    _, d_dy, d2_dy2 = pinn._directional(batched, points, pinn._EY)
    return value, d_dx, d_dy, d2_dx2, d2_dy2


def make_residual_vector(config, batch, interior, inlet, outlet):
    """r(theta) with ||r||^2 equal to the weighted training loss."""
    width = interior[0].shape[1]
    L, w = config.length, config.weights
    inertia, viscous = pinn.scaling_coefficients(config.reynolds, config.pressure_scaling)
    H, Hx, Hy, Hxx, Hyy = interior
    x, y = batch.interior[:, 0], batch.interior[:, 1]
    phi, dphi, d2phi = 1.0 - y**2, -2.0 * y, -2.0
    g, dg = (L - x) / L, -1.0 / L
    H_in = inlet
    phi_in = 1.0 - batch.inlet[:, 1] ** 2
    target = pinn.inlet_profile(batch.inlet[:, 1], config.inlet_sharpness)
    H_out, Hx_out = outlet
    phi_out = 1.0 - batch.outlet[:, 1] ** 2
    n, n_in, n_out = len(x), len(target), len(phi_out)

    def split(theta):
        W = theta[: 3 * width].reshape(3, width)
        b = theta[3 * width:]
        return W, b

    def residual(theta):
        W, b = split(theta)
        a_u, a_v, a_p = W
        c_u, c_v, c_p = b
        Nu, Nv, Np = H @ a_u + c_u, H @ a_v + c_v, H @ a_p + c_p
        u = phi * Nu
        u_x = phi * (Hx @ a_u)
        u_y = dphi * Nu + phi * (Hy @ a_u)
        u_xx = phi * (Hxx @ a_u)
        u_yy = d2phi * Nu + 2.0 * dphi * (Hy @ a_u) + phi * (Hyy @ a_u)
        v = phi * Nv
        v_x = phi * (Hx @ a_v)
        v_y = dphi * Nv + phi * (Hy @ a_v)
        v_xx = phi * (Hxx @ a_v)
        v_yy = d2phi * Nv + 2.0 * dphi * (Hy @ a_v) + phi * (Hyy @ a_v)
        p_x = dg * Np + g * (Hx @ a_p)
        p_y = g * (Hy @ a_p)
        r_x = inertia * (u * u_x + v * u_y) + p_x - viscous * (u_xx + u_yy)
        r_y = inertia * (u * v_x + v * v_y) + p_y - viscous * (v_xx + v_yy)
        r_c = u_x + v_y
        u_in = phi_in * (H_in @ a_u + c_u)
        v_in = phi_in * (H_in @ a_v + c_v)
        ux_out = phi_out * (Hx_out @ a_u)
        vx_out = phi_out * (Hx_out @ a_v)
        return jnp.concatenate([
            math.sqrt(w["momentum_x"] / n) * r_x,
            math.sqrt(w["momentum_y"] / n) * r_y,
            math.sqrt(w["continuity"] / n) * r_c,
            math.sqrt(w["inlet"] / n_in) * (u_in - target),
            math.sqrt(w["inlet"] / n_in) * v_in,
            math.sqrt(w["outlet"] / n_out) * ux_out,
            math.sqrt(w["outlet"] / n_out) * vx_out,
        ])

    return residual


def refine(run, out, n_points, n_boundary, iterations, seed, verbose=True):
    run, out = pathlib.Path(run), pathlib.Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already holds files; a run never writes over another run")
    config, params = load(run)
    if config.formulation != "velocity_pressure" or config.inlet_condition != "soft":
        raise ValueError(
            "the output-layer re-solve assembles the velocity-pressure, soft-inlet ansatz "
            f"by hand; this run uses {config.formulation} with a {config.inlet_condition} inlet")
    field = pinn.make_field(config)
    loss_terms = pinn.make_loss_terms(config, field)
    weights = {k: jnp.asarray(v) for k, v in config.weights.items()}
    loss = jax.jit(lambda p, batch: pinn.total_loss(loss_terms(p, batch), weights))

    # Fitting points: a fresh draw from the training sampler, never the held-out key.
    from dataclasses import replace
    fit_config = replace(config, n_inlet=n_boundary, n_outlet=n_boundary)
    batch = pinn.sample_batch(jax.random.PRNGKey(seed), fit_config, n_points)
    _, _, heldout_key = pinn.root_keys(config.seed)
    heldout = pinn.sample_batch(heldout_key, config, config.n_validation)

    interior = feature_derivatives(config, params, batch.interior)
    inlet = jax.vmap(lambda xy: pinn.hidden_features(
        params, jnp.stack([2.0 * xy[0] / config.length - 1.0, xy[1]]), config.activation))(batch.inlet)
    out_value, out_dx = feature_derivatives(config, params, batch.outlet)[:2]
    residual = jax.jit(make_residual_vector(config, batch, interior, inlet, (out_value, out_dx)))
    jacobian = jax.jit(jax.jacfwd(residual))

    theta = jnp.concatenate([params["output"]["W"].ravel(), params["output"]["b"]])
    before_fit = float(loss(params, batch))
    identity = float(jnp.sum(residual(theta) ** 2))
    if abs(identity - before_fit) > 1e-9 * before_fit:
        raise RuntimeError(
            f"residual vector does not reproduce the training loss: {identity:.6e} "
            f"vs {before_fit:.6e}; the hand-assembled derivatives are wrong")

    def with_output(t):
        W = t[: 3 * config.width].reshape(3, config.width)
        return {**params, "output": {"W": W, "b": t[3 * config.width:]}}

    before_heldout = float(loss(params, heldout))
    cost = identity
    damping = 1e-3
    log = []
    for it in range(iterations):
        r = np.asarray(residual(theta))
        J = np.asarray(jacobian(theta))
        normal = J.T @ J
        gradient = J.T @ r
        diagonal = np.diag(np.diag(normal))
        while True:
            step = np.linalg.solve(normal + damping * diagonal, -gradient)
            candidate = theta + jnp.asarray(step)
            new_cost = float(jnp.sum(residual(candidate) ** 2))
            if new_cost < cost:
                theta, damping = candidate, max(damping / 3.0, 1e-12)
                break
            damping *= 3.0
            if damping > 1e12:
                break
        improvement = (cost - new_cost) / cost if new_cost < cost else 0.0
        cost = min(cost, new_cost)
        held = float(loss(with_output(theta), heldout))
        log.append({"iteration": it + 1, "fit_loss": cost, "heldout_loss": held, "damping": damping})
        if verbose:
            print(f"  LM {it + 1:>3}  fit loss {cost:.4e}  held-out {held:.4e}  damping {damping:.1e}",
                  flush=True)
        if damping > 1e12 or improvement < 1e-10:
            break

    refined = with_output(theta)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run / "config.yaml", out / "config.yaml")
    summary = {
        "source_run": str(run),
        "method": "Levenberg-Marquardt on the output layer, hidden layers frozen",
        "unknowns": int(theta.size),
        "fit_points": {"interior": n_points, "inlet": n_boundary, "outlet": n_boundary, "seed": seed},
        "fit_loss_before": before_fit,
        "fit_loss_after": cost,
        "heldout_loss_before": before_heldout,
        "heldout_loss_after": float(loss(refined, heldout)),
        "iterations": log,
    }
    with (out / "refine.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    # Last, so that a params.npz means the refined run is complete.
    pinn.save_params(out / "params.npz", refined)
    if verbose:
        print(f"\n  held-out loss {before_heldout:.4e} -> {summary['heldout_loss_after']:.4e} "
              f"({before_heldout / summary['heldout_loss_after']:.2f}x)")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--points", type=int, required=True)
    parser.add_argument("--boundary-points", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    refine(args.run, args.out, args.points, args.boundary_points, args.iterations, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
