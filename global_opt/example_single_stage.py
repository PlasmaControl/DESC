"""Global search on a small single-stage problem: equilibrium and coils both vary.

This is a usage example and a smoke test, not a benchmark. The equilibrium is the
precise QA example at low resolution (L = M = N = 4) and the coils are the
stage-two tutorial's three coils, started from the tutorial's result
(``results/stage2_fixed/baseline_seed0_coilset.h5``) when it exists, otherwise from
circular coils. Objective:

* aspect ratio of the plasma (target 6),
* ``VacuumBoundaryError(eq, coils, field_fixed=False)``: the single-stage coupling
  term, zero when the coils' field is tangent to the plasma boundary and matches
  the field inside,
* coil length <= 8 m and plasma-coil distance >= 0.25 m.

Constraints: ``ForceBalance`` (nonlinear, so :class:`desc_global.ProjectedProblem`
wraps the objective in a ``ProximalProjection``), fixed pressure, current and Psi,
and one coil current. Each local solve therefore re-solves the equilibrium at every
step, as ``proximal-lsq-exact`` does.

Usage::

    python example_single_stage.py              # a few hops of each method, ~3 min
    python example_single_stage.py --budget 1800 --maxiter 50

On 4 CPU cores the build takes ~30 s and a 6-iteration local solve ~30 s.
"""

import argparse
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
    CoilLength,
    FixCoilCurrent,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    PlasmaCoilSetMinDistance,
    VacuumBoundaryError,
)

from desc_global import ProjectedProblem, basin_hopping, pool_search  # noqa: E402


def setup(resolution=4):
    """Low-resolution precise QA equilibrium, three coils, objective and constraints."""
    eq = desc.examples.get("precise_QA")
    r = resolution
    eq.change_resolution(L=r, M=r, N=r, L_grid=2 * r, M_grid=2 * r, N_grid=2 * r)
    coilset = initialize_modular_coils(eq, num_coils=3, r_over_a=3.0).to_FourierXYZ(N=10)
    start = os.path.join(HERE, "results", "stage2_fixed", "baseline_seed0_coilset.h5")
    if os.path.exists(start):
        for c, d in zip(coilset, load(start).params_dict):
            c.params_dict = d
    objective = ObjectiveFunction(
        (
            AspectRatio(eq, target=6.0, weight=10),
            VacuumBoundaryError(
                eq, coilset, field_fixed=False,
                eval_grid=LinearGrid(M=2 * r, N=2 * r, NFP=eq.NFP, sym=False),
            ),
            CoilLength(coilset, bounds=(0, 8.0), normalize_target=False),
            PlasmaCoilSetMinDistance(
                eq, coilset, bounds=(0.25, np.inf), normalize_target=False
            ),
        )
    )
    constraints = (
        ForceBalance(eq),  # nonlinear: handled with a ProximalProjection
        FixPressure(eq),
        FixCurrent(eq),
        FixPsi(eq),
        FixCoilCurrent(coilset, indices=[True, False, False]),
    )
    return eq, coilset, objective, constraints


def main():
    """Build the problem, then run basin hopping and deflated pool search."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None, help="seconds per method")
    ap.add_argument("--maxiter", type=int, default=6, help="iterations per local solve")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    t0 = time.time()
    eq, coilset, objective, constraints = setup()
    problem = ProjectedProblem(
        [eq, coilset], objective, constraints,
        local_kwargs=dict(maxiter=args.maxiter, ftol=1e-4),
        # one length scale for boundary and coil coefficients; pass
        # overrides={"Rb_lmn": ..., "Zb_lmn": ...} to perturb the boundary less
        length_scale=0.02,
        deflation_slots=8,
    )
    print(f"built in {time.time() - t0:.0f}s, proximal mode: {problem.proximal}")
    res = problem.local_solve(problem.x_full())
    print(f"single local solve: cost {res['cost']:.4e}")
    hops = dict(n_hops=2) if args.budget is None else dict(time_budget=args.budget)
    r_bh = basin_hopping(problem, x0=res["x"], sigma=1.0, seed=args.seed, **hops)
    hops = dict(n_hops=3) if args.budget is None else dict(time_budget=args.budget)
    r_pool = pool_search(problem, x0=res["x"], seed=args.seed, **hops)
    print(f"basin hopping best {r_bh.cost:.4e}; deflated pool search best "
          f"{r_pool.cost:.4e} {r_pool['stats']}; total {time.time() - t0:.0f}s")
    # the things now hold pool search's best result: a force-balanced equilibrium
    # and its coils
    eq.save(os.path.join(HERE, "results", "single_stage_eq.h5"))
    coilset.save(os.path.join(HERE, "results", "single_stage_coilset.h5"))


if __name__ == "__main__":
    main()
