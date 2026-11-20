"""Global / stochastic search for the stage-two coil problem of the DESC tutorial.

The problem is exactly the first one in DESC's
``docs/notebooks/tutorials/coil_stage_two_optimization.ipynb``: 3 unique FourierXYZ
modular coils (N=10) for the vacuum precise QA equilibrium, started from circular
coils, with quadratic flux, coil-coil and plasma-coil distance, curvature and length
terms, and one coil current fixed.

Usage::

    python stage2_precise_QA.py --method baseline
    python stage2_precise_QA.py --method multistart --budget 1200
    python stage2_precise_QA.py --method basin --budget 1200
    python stage2_precise_QA.py --method cma --budget 1200
    python stage2_precise_QA.py --method noisy --budget 1200

Results (JSON summary + best coilset .h5) are written to ``results/stage2/``.
"""

import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/jax-cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import numpy as np  # noqa: E402

import desc.examples  # noqa: E402
from desc.coils import initialize_modular_coils  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.objectives import (  # noqa: E402
    CoilCurvature,
    CoilLength,
    CoilSetMinDistance,
    FixCoilCurrent,
    ObjectiveFunction,
    PlasmaCoilSetMinDistance,
    QuadraticFlux,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_global import (  # noqa: E402
    ProjectedProblem,
    annealed_noisy_gradient,
    basin_hopping,
    cma_es,
    multistart,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results", "stage2")

WEIGHTS = {
    "quadratic flux": 200,
    "coil-coil min dist": 100,
    "plasma-coil min dist": 10,
    "coil curvature": 185,
    "coil length": 20,
}
TERM_NAMES = ["quadratic flux", "coil-coil dist", "plasma-coil dist", "curvature", "length"]


def setup(num_coils=3, r_over_a=3.0, N=10):
    """Return (eq, coilset, objective, constraints) as in the DESC tutorial."""
    eq = desc.examples.get("precise_QA")
    coilset = initialize_modular_coils(eq, num_coils=num_coils, r_over_a=r_over_a)
    coilset = coilset.to_FourierXYZ(N=N)
    coil_grid = LinearGrid(N=50)
    plasma_grid = LinearGrid(M=25, N=25, NFP=eq.NFP, sym=eq.sym)
    w = WEIGHTS
    obj = ObjectiveFunction(
        (
            QuadraticFlux(
                eq,
                field=coilset,
                eval_grid=plasma_grid,
                field_grid=coil_grid,
                vacuum=True,
                weight=w["quadratic flux"],
            ),
            CoilSetMinDistance(
                coilset,
                bounds=(0.1, np.inf),
                normalize_target=False,
                grid=coil_grid,
                weight=w["coil-coil min dist"],
            ),
            PlasmaCoilSetMinDistance(
                eq,
                coilset,
                bounds=(0.25, np.inf),
                normalize_target=False,
                plasma_grid=plasma_grid,
                coil_grid=coil_grid,
                eq_fixed=True,
                weight=w["plasma-coil min dist"],
            ),
            CoilCurvature(
                coilset,
                bounds=(-1, 2),
                normalize_target=False,
                grid=coil_grid,
                weight=w["coil curvature"],
            ),
            CoilLength(
                coilset,
                bounds=(0, 2 * np.pi * (coilset[0].compute("length")["length"])),
                normalize_target=True,
                grid=coil_grid,
                weight=w["coil length"],
            ),
        )
    )
    fix = [False] * len(coilset)
    fix[0] = True
    constraints = (FixCoilCurrent(coilset, indices=fix),)
    return eq, coilset, obj, constraints


def average_normalized_field(field, eq):
    """<|B.n|>/<|B|> on the LCFS (the metric used in the DESC tutorial)."""
    grid = LinearGrid(M=80, N=80, NFP=eq.NFP)
    Bn, coords = field.compute_Bnormal(eq.surface, eval_grid=grid)
    B = field.compute_magnetic_field(coords)
    return float(np.mean(np.abs(Bn)) / np.mean(np.linalg.norm(B, axis=1)))


def metrics(problem, eq, coilset, x):
    """Physics metrics and per-term cost breakdown for full vector x."""
    problem.set_x_full(x)
    obj = problem.objective
    params = obj.unpack_state(jax.numpy.asarray(x), per_objective=True)
    terms = {}
    for name, o, p in zip(TERM_NAMES, obj.objectives, params):
        f = np.asarray(o.compute_scaled_error(*p))
        terms[name] = float(0.5 * np.sum(f**2))
    raw = [np.asarray(o.compute_unscaled(*p)) for o, p in zip(obj.objectives, params)]
    curv_norm = obj.objectives[3].normalization
    return dict(
        cost=float(problem.cost(x)),
        terms=terms,
        Bn_over_B=average_normalized_field(coilset, eq),
        min_coil_coil_dist=float(np.min(raw[1])),
        min_plasma_coil_dist=float(np.min(raw[2])),
        max_curvature_normalized=float(np.max(raw[3]) / curv_norm),
        min_curvature_normalized=float(np.min(raw[3]) / curv_norm),
        coil_lengths=[float(v) for v in np.ravel(raw[4])],
        currents=[float(c.current) for c in coilset],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        default="baseline",
        choices=["baseline", "converged", "multistart", "basin", "cma", "noisy"],
    )
    p.add_argument("--budget", type=float, default=1200.0, help="seconds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--length-scale", type=float, default=0.05, help="meters")
    p.add_argument(
        "--max-order",
        type=int,
        default=None,
        help="cma only: search only coil Fourier modes |n| <= max-order",
    )
    p.add_argument("--tag", default="")
    p.add_argument("--smoke", action="store_true", help="tiny budgets for testing")
    args = p.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    eq, coilset, obj, cons = setup()
    problem = ProjectedProblem(
        coilset,
        obj,
        cons,
        local_kwargs=dict(maxiter=100, ftol=1e-4, xtol=1e-6),
        length_scale=args.length_scale,
        alpha=0.5,
        current_rel=0.1,
    )
    x0 = problem.x_full()
    print(f"initial cost {problem.cost(x0):.4e}")
    smoke = args.smoke
    m = args.method
    if m == "baseline":  # the tutorial: single lsq-exact run, ftol=1e-4, maxiter 100
        res = problem.local_solve(x0)
        result = dict(x=res["x"], cost=res["cost"], trace=[], history=[], method=m)
        result["trace"] = [dict(time=problem.elapsed(), nlocal=1, best_cost=res["cost"])]
    elif m == "converged":  # same local start, run to much tighter tolerances
        res = problem.local_solve(x0, maxiter=1000, ftol=1e-10, xtol=1e-12, gtol=1e-12)
        result = dict(x=res["x"], cost=res["cost"], trace=[], history=[], method=m)
        result["trace"] = [dict(time=problem.elapsed(), nlocal=1, best_cost=res["cost"])]
    elif m == "multistart":
        result = multistart(
            problem,
            n_starts=2 if smoke else 1000,
            sigma=args.sigma,
            x0=x0,
            seed=args.seed,
            time_budget=args.budget,
        )
    elif m == "basin":
        result = basin_hopping(
            problem,
            n_hops=2 if smoke else 1000,
            sigma=args.sigma,
            x0=x0,
            seed=args.seed,
            temperature=0.5,
            restart_after=6,
            time_budget=args.budget,
        )
    elif m == "cma":
        # CMA-ES is used as a population-based explorer around the first local
        # minimum (from circular coils it spends most evaluations on the large
        # initial descent that the local solver does in ~40 s)
        res0 = problem.local_solve(x0)
        print(f"first local minimum: {res0['cost']:.4e}")
        result = cma_es(
            problem,
            sigma0=args.sigma,
            x0=res0["x"],
            seed=args.seed,
            max_order=args.max_order,
            max_evals=200 if smoke else 10**6,
            n_polish=1 if smoke else 3,
            time_budget=30 if smoke else max(args.budget - 300, 60),
        )
        if res0["cost"] < result["cost"]:
            result["x"], result["cost"] = res0["x"], res0["cost"]
    elif m == "noisy":
        # like CMA-ES, start the Langevin excursions from the first local minimum
        res0 = problem.local_solve(x0)
        print(f"first local minimum: {res0['cost']:.4e}")
        result = annealed_noisy_gradient(
            problem,
            n_cycles=1 if smoke else 1000,
            steps_per_cycle=20 if smoke else 200,
            temperature0=0.3 * args.sigma**2,
            x0=res0["x"],
            seed=args.seed,
            time_budget=30 if smoke else args.budget - res0["time"],
        )
        result["history"].insert(0, dict(cost=res0["cost"], x=res0["x"]))
    met = metrics(problem, eq, coilset, result["x"])
    tag = f"{m}{args.tag}_seed{args.seed}" + ("_smoke" if smoke else "")
    summary = dict(
        method=m,
        tag=args.tag,
        seed=args.seed,
        sigma=args.sigma,
        length_scale=args.length_scale,
        budget=args.budget,
        wall_time=time.time() - t0,
        n_local_solves=problem.nlocal,
        nfev=problem.nfev,
        ngev=problem.ngev,
        best_cost=float(result["cost"]),
        metrics=met,
        trace=result["trace"],
        local_minima=[
            float(h["cost"]) for h in result["history"] if "x" in h
        ],
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "trace"}, indent=1))
    with open(os.path.join(OUTDIR, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    problem.set_x_full(result["x"])
    coilset.save(os.path.join(OUTDIR, f"{tag}_coilset.h5"))


if __name__ == "__main__":
    main()
