"""Global / stochastic search for the stage-one example ``desc/examples/precise_QA.py``.

Same initial surface, objective (QuasisymmetryTwoTerm on rho = 0.6, 0.8, 1 +
AspectRatio(6) + RotationalTransform(0.42)), constraints and optimizer
(``proximal-lsq-exact``) as the DESC example. The resolution is reduced (default
M = N = 6 instead of 8, boundary modes up to k_max = 3) so that many local solves
fit in a CPU budget.

Strategies
----------
baseline
    The example: continuation over boundary mode number k = 1..k_max.
multistart
    Random smooth perturbations of the *initial* boundary (modes of order <= k_max,
    major radius untouched), each followed by the full continuation.
basin
    Basin hopping on the final problem (k = k_max): perturb the incumbent boundary,
    re-optimize, Metropolis accept/reject on log(cost), adaptive step size.

Boundary perturbations are applied with ``Equilibrium.perturb`` (first order), and
``ProximalProjection`` re-solves force balance before each local optimization.

Usage::

    python stage1_precise_QA.py --method baseline
    python stage1_precise_QA.py --method multistart --budget 3000
    python stage1_precise_QA.py --method basin --budget 3000
"""

import argparse
import json
import os
import time
import warnings

import jax

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/jax-cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import numpy as np  # noqa: E402

from desc.continuation import solve_continuation_automatic  # noqa: E402
from desc.equilibrium import Equilibrium  # noqa: E402
from desc.geometry import FourierRZToroidalSurface  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives import (  # noqa: E402
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
    RotationalTransform,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results", "stage1")


def initial_eq(M):
    """Initial equilibrium of the precise QA example at resolution M = N."""
    fname = os.path.join(OUTDIR, f"initial_eq_M{M}.h5")
    if os.path.exists(fname):
        return load(fname)
    surf = FourierRZToroidalSurface(
        R_lmn=[1, 0.166, 0.1],
        Z_lmn=[-0.166, -0.1],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
        NFP=2,
    )
    eq = Equilibrium(M=M, N=M, Psi=0.087, surface=surf)
    eq = solve_continuation_automatic(eq, objective="force", bdry_step=0.5, verbose=0)[
        -1
    ]
    eq.save(fname)
    return eq


def make_objective(eq):
    grid = LinearGrid(
        M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True
    )
    return ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(eq=eq, helicity=(1, 0), grid=grid, normalize=False),
            AspectRatio(eq=eq, target=6, weight=1e1, normalize=False),
            RotationalTransform(eq=eq, target=0.42, weight=10, normalize=False),
        )
    )


def make_constraints(eq, k):
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :],
        )
    )
    Z_modes = eq.surface.Z_basis.modes[np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :]
    return (
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )


def evaluate(eq):
    """Objective cost 0.5*||f||^2 plus its components."""
    obj = make_objective(eq)
    obj.build(verbose=0)
    params = obj.unpack_state(obj.x(eq), per_objective=True)
    out = {}
    for name, o, p in zip(["qs", "aspect", "iota"], obj.objectives, params):
        f = np.asarray(o.compute_scaled_error(*p))
        out[name] = float(0.5 * np.sum(f**2))
    out["cost"] = float(sum(out.values()))
    d = eq.compute(["R0/a", "iota"], grid=LinearGrid(rho=np.array([1.0]), M=4, N=4, NFP=eq.NFP))
    out["aspect_ratio"] = float(d["R0/a"])
    out["iota_edge"] = float(np.mean(d["iota"]))
    return out


def local_step(eq, k, maxiter):
    """One step of the example: optimize boundary modes of order <= k."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq_new, _ = eq.optimize(
            objective=make_objective(eq),
            constraints=make_constraints(eq, k),
            optimizer="proximal-lsq-exact",
            maxiter=maxiter,
            verbose=0,
            copy=True,
            options={"initial_trust_radius": 0.5},
        )
    return eq_new


def continuation(eq, k_max, maxiter, log=None):
    for k in range(1, k_max + 1):
        eq = local_step(eq, k, maxiter)
        if log is not None:
            log(f"   k={k}: cost {evaluate(eq)['cost']:.4e}")
    return eq


def boundary_scales(eq, length_scale, alpha, max_order):
    """Smooth perturbation scales for Rb_lmn, Zb_lmn (R00 excluded)."""
    out = {}
    for name, basis in (("Rb_lmn", eq.surface.R_basis), ("Zb_lmn", eq.surface.Z_basis)):
        order = np.max(np.abs(basis.modes), axis=1).astype(float)
        s = length_scale * np.exp(-alpha * order)
        s[order > max_order] = 0
        s[order == 0] = 0  # keep the major radius fixed
        out[name] = s
    return out


def perturbed(eq, scales, sigma, rng):
    deltas = {k: sigma * s * rng.standard_normal(s.size) for k, s in scales.items()}
    eq = eq.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq.perturb(deltas=deltas, order=1, verbose=0, copy=False)
    return eq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="baseline", choices=["baseline", "multistart", "basin"])
    p.add_argument("--M", type=int, default=6)
    p.add_argument("--kmax", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=50)
    p.add_argument("--budget", type=float, default=3000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--length-scale", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument(
        "--init",
        default=None,
        help="basin hopping only: start from this saved (already optimized) eq",
    )
    p.add_argument("--tag", default="")
    args = p.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    eq0 = initial_eq(args.M)
    log(f"initial eq ready, cost {evaluate(eq0)['cost']:.4e}")
    scales = boundary_scales(eq0, args.length_scale, 0.5, args.kmax)
    trace, minima = [], []
    best_eq, best = None, np.inf

    def record(eq, label):
        nonlocal best_eq, best
        c = evaluate(eq)["cost"]
        minima.append(c)
        if c < best:
            best_eq, best = eq, c
        trace.append(dict(time=time.time() - t0, best_cost=best, cost=c, label=label))
        log(f"{label}: cost {c:.4e}  best {best:.4e}")
        return c

    if args.method == "baseline":
        eq = continuation(eq0, args.kmax, args.maxiter, log)
        record(eq, "baseline")
    elif args.method == "multistart":
        i = 0
        while time.time() - t0 < args.budget:
            eqs = eq0 if i == 0 else perturbed(eq0, scales, args.sigma, rng)
            try:
                eq = continuation(eqs, args.kmax, args.maxiter)
                record(eq, f"start {i}")
            except Exception as e:
                log(f"start {i} failed: {e!r}")
            i += 1
    elif args.method == "basin":
        sigma = args.sigma
        if args.init is not None:
            eq_cur = load(args.init)
        else:
            eq_cur = continuation(eq0, args.kmax, args.maxiter)
        c_cur = record(eq_cur, "hop 0")
        k, n_acc = 0, 0
        while time.time() - t0 < args.budget:
            k += 1
            try:
                eq = local_step(perturbed(eq_cur, scales, sigma, rng), args.kmax, args.maxiter)
                c = record(eq, f"hop {k} (sigma {sigma:.2f})")
            except Exception as e:
                log(f"hop {k} failed: {e!r}")
                continue
            acc = c < c_cur or rng.random() < np.exp(
                -(np.log(c) - np.log(c_cur)) / args.temperature
            )
            if acc:
                eq_cur, c_cur = eq, c
                n_acc += 1
            if k % 4 == 0:  # adapt step size towards ~40% acceptance
                sigma = sigma / 0.75 if n_acc / 4 > 0.4 else sigma * 0.75
                sigma = float(np.clip(sigma, 0.05, 5))
                n_acc = 0
            if minima and c_cur > best and k % 6 == 0:  # restart walker from best
                eq_cur, c_cur = best_eq, best

    final = evaluate(best_eq)
    summary = dict(
        method=args.method,
        seed=args.seed,
        M=args.M,
        kmax=args.kmax,
        maxiter=args.maxiter,
        sigma=args.sigma,
        length_scale=args.length_scale,
        budget=args.budget,
        wall_time=time.time() - t0,
        best=final,
        local_minima=minima,
        trace=trace,
    )
    tag = f"{args.method}{args.tag}_M{args.M}_k{args.kmax}_seed{args.seed}"
    with open(os.path.join(OUTDIR, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    best_eq.save(os.path.join(OUTDIR, f"{tag}_eq.h5"))
    log(json.dumps(final))


if __name__ == "__main__":
    main()
