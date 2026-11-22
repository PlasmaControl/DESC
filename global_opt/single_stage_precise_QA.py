"""Single-stage optimization of vacuum precise QA and its coils: pool search vs. one solve.

Problem
-------
The plasma is DESC's precise QA example (vacuum, NFP = 2), run at
L = M = N = 6 by default (8 is the example's own resolution, but each solver
iteration then takes ~50 s on 3 CPU cores instead of ~10 s) and the coils are the stage-two tutorial's three unique modular
``FourierXYZCoil`` s (N = 10), started from the tutorial's stage-two result
(``results/stage2_fixed/baseline_seed0_coilset.h5``, cost 31.0 there). Both are
optimized together. The objective joins the two stages:

* stage one, as in ``desc/examples/precise_QA.py``: ``QuasisymmetryTwoTerm`` on
  rho = 0.6, 0.8, 1; aspect ratio 6; rotational transform 0.42;
* the coupling term ``VacuumBoundaryError(eq, coils, field_fixed=False)``: the
  coils' normal field on the plasma boundary and the jump in |B|^2 across it;
* the tutorial's coil terms and bounds: coil-coil distance, plasma-coil distance,
  curvature and length.

Constraints: ``ForceBalance`` (so every local solve uses a ``ProximalProjection``
that re-solves the equilibrium, as ``proximal-lsq-exact`` does), fixed pressure,
current and Psi, the major radius R00 and every boundary mode of order > ``kmax``,
and one coil current. The term weights are chosen so that the stage-one terms and
the coupling term both matter at the start (see ``WEIGHTS``).

Methods
-------
baseline
    One ``proximal-lsq-exact`` solve from the start (up to 100 iterations, ftol
    1e-4): the usual single-stage procedure.
converged
    The same solve to tight tolerances (1e-8, up to 500 iterations), to check that
    the baseline stops at a genuine local minimum.
pool
    Deflated pool search (:func:`desc_global.pool_search`) for ``--budget`` seconds.

Usage::

    python single_stage_precise_QA.py --method probe        # term values, timing
    python single_stage_precise_QA.py --method baseline
    python single_stage_precise_QA.py --method pool --budget 7200 --seed 0

Results go to ``results/single_stage/<method>_seed<seed>.json`` with the final
equilibrium and coils (``_eq.h5``, ``_coilset.h5``).
"""

import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import desc.examples  # noqa: E402
from desc.coils import initialize_modular_coils  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.objectives import (  # noqa: E402
    AspectRatio,
    CoilCurvature,
    CoilLength,
    CoilSetMinDistance,
    FixBoundaryR,
    FixBoundaryZ,
    FixCoilCurrent,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    PlasmaCoilSetMinDistance,
    QuasisymmetryTwoTerm,
    RotationalTransform,
    VacuumBoundaryError,
)

from desc_global import ProjectedProblem, pool_search  # noqa: E402

OUTDIR = os.path.join(HERE, "results", "single_stage")
START_COILS = os.path.join(HERE, "results", "stage2_fixed", "baseline_seed0_coilset.h5")
# Stage one: the weights of desc/examples/precise_QA.py (QS 1, aspect ratio 10,
# iota 10), all multiplied by STAGE1_SCALE. Coil terms: the stage-two tutorial's.
# With STAGE1_SCALE = 1000 the QS term (8.9 at the start) and the boundary error
# (10.7 at the start, weight 1) are of similar size, so the solve has to trade
# plasma quality against how well the coils fit it.
STAGE1_SCALE = 1000.0
WEIGHTS = {
    "qs": 1.0 * STAGE1_SCALE,
    "aspect ratio": 10.0 * STAGE1_SCALE,
    "iota": 10.0 * STAGE1_SCALE,
    "boundary error": 1.0,
    "coil-coil min dist": 100.0,
    "plasma-coil min dist": 10.0,
    "coil curvature": 185.0,
    "coil length": 20.0,
}
TERMS = ["qs", "aspect ratio", "iota", "boundary error", "coil-coil dist",
         "plasma-coil dist", "curvature", "length"]


def setup(M=8, kmax=4, weights=None):
    """Return (eq, coilset, objective, constraints) of the single-stage problem."""
    w = dict(WEIGHTS, **(weights or {}))
    eq = desc.examples.get("precise_QA")
    if M != eq.M:
        eq.change_resolution(L=M, M=M, N=M, L_grid=2 * M, M_grid=2 * M, N_grid=2 * M)
    coilset = initialize_modular_coils(eq, num_coils=3, r_over_a=3.0).to_FourierXYZ(N=10)
    len0 = float(coilset[0].compute("length")["length"])  # circular coil, tutorial bound
    coil_grid = LinearGrid(N=50)
    plasma_grid = LinearGrid(M=25, N=25, NFP=eq.NFP, sym=eq.sym)
    qs_grid = LinearGrid(M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True)
    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(eq, helicity=(1, 0), grid=qs_grid, normalize=False,
                                 weight=w["qs"]),
            AspectRatio(eq, target=6, weight=w["aspect ratio"], normalize=False),
            RotationalTransform(eq, target=0.42, weight=w["iota"], normalize=False),
            VacuumBoundaryError(eq, coilset, field_fixed=False, field_grid=coil_grid,
                                weight=w["boundary error"]),
            CoilSetMinDistance(coilset, bounds=(0.1, np.inf), grid=coil_grid,
                               weight=w["coil-coil min dist"]),
            PlasmaCoilSetMinDistance(eq, coilset, bounds=(0.25, np.inf),
                                     normalize_target=False, plasma_grid=plasma_grid,
                                     coil_grid=coil_grid, eq_fixed=False,
                                     weight=w["plasma-coil min dist"]),
            CoilCurvature(coilset, bounds=(-1, 2), normalize_target=False,
                          grid=coil_grid, weight=w["coil curvature"]),
            CoilLength(coilset, bounds=(0, 2 * np.pi * len0), normalize_target=True,
                       grid=coil_grid, weight=w["coil length"]),
        )
    )
    order = lambda modes: np.max(np.abs(modes), axis=1)  # noqa: E731
    R_modes = eq.surface.R_basis.modes
    Z_modes = eq.surface.Z_basis.modes
    constraints = (
        ForceBalance(eq),
        FixBoundaryR(eq, modes=np.vstack(([0, 0, 0], R_modes[order(R_modes) > kmax]))),
        FixBoundaryZ(eq, modes=Z_modes[order(Z_modes) > kmax]),
        FixPressure(eq),
        FixCurrent(eq),
        FixPsi(eq),
        FixCoilCurrent(coilset, indices=[True, False, False]),
    )
    return eq, coilset, objective, constraints


def load_start_coils(coilset):
    """Set the coils to the stage-two tutorial's result.

    Call this after the objective is built: the coil terms set their normalization
    from the coils at build time, and the tutorial builds them with circular coils.
    """
    for c, d in zip(coilset, load(START_COILS).params_dict):
        c.params_dict = d


def term_costs(objective):
    """Cost 0.5 * ||f_i||^2 of every term at the things' current state."""
    x = objective.x(*objective.things)
    params = objective.unpack_state(x, per_objective=True)
    out = {}
    for name, o, p in zip(TERMS, objective.objectives, params):
        f = np.asarray(o.compute_scaled_error(*p, constants=o.constants))
        out[name] = float(0.5 * np.sum(f**2))
    return out


def metrics(problem, eq, coilset, x):
    """Per-term costs and physics metrics of full state x (an equilibrium)."""
    problem.set_x_full(x)
    out = dict(cost=problem.cost(x), terms=term_costs(problem.objective))
    g = LinearGrid(rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    d = eq.compute(["R0/a", "iota"], grid=g)
    out["aspect_ratio"] = float(d["R0/a"])
    out["iota_edge"] = float(np.mean(d["iota"]))
    fb = ObjectiveFunction(ForceBalance(eq))
    fb.build(verbose=0)
    f = np.asarray(fb.compute_scaled_error(fb.x(eq), fb.constants))
    out["force_residual_rms"] = float(np.sqrt(np.mean(f**2)))
    grid = LinearGrid(M=40, N=40, NFP=eq.NFP)
    Bn, coords = coilset.compute_Bnormal(eq.surface, eval_grid=grid)
    B = coilset.compute_magnetic_field(coords)
    out["Bn_over_B"] = float(np.mean(np.abs(Bn)) / np.mean(np.linalg.norm(B, axis=1)))
    cg = LinearGrid(N=50)
    out["coil_lengths"] = [float(np.asarray(c.compute("length", grid=cg)["length"]))
                           for c in coilset]
    return out


def main():
    """Run one method and write its summary, equilibrium and coils."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["probe", "baseline", "converged", "pool"])
    ap.add_argument("--budget", type=float, default=7200.0, help="pool: seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=int, default=6, help="L = M = N resolution")
    ap.add_argument("--kmax", type=int, default=4, help="boundary modes optimized")
    ap.add_argument("--maxiter", type=int, default=100, help="iterations per solve")
    ap.add_argument("--length-scale", type=float, default=0.05,
                    help="kick scale of coil coefficients [m]")
    ap.add_argument("--boundary-scale", type=float, default=0.02,
                    help="kick scale of plasma boundary coefficients [m]")
    ap.add_argument("--w-qs", type=float, default=None)
    ap.add_argument("--w-be", type=float, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    w = {}
    if args.w_qs is not None:
        w["qs"] = args.w_qs
    if args.w_be is not None:
        w["boundary error"] = args.w_be
    eq, coilset, objective, constraints = setup(args.M, args.kmax, w)
    problem = ProjectedProblem(
        [eq, coilset], objective, constraints,
        local_kwargs=dict(maxiter=args.maxiter, ftol=1e-4, xtol=1e-6),
        length_scale=args.length_scale,
        # the boundary gets its own scale (stage one used 0.02 m), still decaying
        # as exp(-0.5 k) with the mode order k
        overrides={
            name: (lambda e, b=basis: args.boundary_scale
                   * np.exp(-0.5 * np.max(np.abs(getattr(e.surface, b).modes), axis=1)))
            for name, basis in (("Rb_lmn", "R_basis"), ("Zb_lmn", "Z_basis"))
        },
        deflation_slots=8 if args.method == "pool" else 0,
    )
    print(f"built in {time.time() - t0:.0f}s", flush=True)
    load_start_coils(coilset)
    x0 = problem.x_full()
    if args.method == "probe":
        print("term costs at the start (weights", dict(WEIGHTS, **w), "):")
        for k, v in term_costs(objective).items():
            print(f"  {k:18s} {v:.4e}")
        print(f"start cost {problem.cost(x0):.4e}", flush=True)
        for it in (1, 3):
            r = problem.local_solve(x0, maxiter=it)
            print(f"local solve, {it} iterations: {r['time']:.0f}s, cost {r['cost']:.4e}",
                  flush=True)
        return
    if args.method in ("baseline", "converged"):
        tight = dict(maxiter=500, ftol=1e-8, xtol=1e-10, gtol=1e-10)
        res = problem.local_solve(x0, **(tight if args.method == "converged" else {}))
        result = dict(x=res["x"], cost=res["cost"], history=[dict(cost=res["cost"], x=res["x"])],
                      trace=[dict(time=problem.elapsed(), nlocal=1, best_cost=res["cost"])])
        extra = dict(nit=res["nit"])
    else:
        result = pool_search(problem, x0=x0, seed=args.seed, time_budget=args.budget)
        extra = dict(pool_stats=result["stats"], pool_costs=result["pool_costs"])
    met = metrics(problem, eq, coilset, result["x"])
    tag = f"{args.method}{args.tag}_seed{args.seed}"
    summary = dict(
        method=args.method, tag=args.tag, seed=args.seed, M=args.M, kmax=args.kmax,
        maxiter=args.maxiter, length_scale=args.length_scale,
        boundary_scale=args.boundary_scale, budget=args.budget,
        weights=dict(WEIGHTS, **w), wall_time=time.time() - t0,
        n_local_solves=problem.nlocal, best_cost=float(result["cost"]), metrics=met,
        trace=result["trace"],
        local_minima=[float(h["cost"]) for h in result["history"] if "x" in h],
        hops=[{k: (float(v) if isinstance(v, (float, np.floating)) else v)
               for k, v in h.items() if k != "x"} for h in result["history"]],
        **extra,
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("trace", "hops")},
                     indent=1))
    with open(os.path.join(OUTDIR, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    problem.set_x_full(result["x"])
    eq.save(os.path.join(OUTDIR, f"{tag}_eq.h5"))
    coilset.save(os.path.join(OUTDIR, f"{tag}_coilset.h5"))


if __name__ == "__main__":
    main()
