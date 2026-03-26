#!/usr/bin/env python
"""Full-resolution coil opt with blocked mode + per-objective jac_chunk_size.

Usage: CUDA_VISIBLE_DEVICES=X python profiling/fullres_blocked_chunked.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from instrument import setup_gpu
setup_gpu()

import numpy as np

sys.path.insert(0, os.path.expanduser("~/finding_better_basins"))
from configs import build_config, load_data, setup_jax

config = build_config(platform="perlmutter", lightweight=False, seed=0)
setup_jax(config)
data = load_data(config)

# Build objectives manually so we can set per-objective jac_chunk_size
from desc.grid import LinearGrid
from desc.objectives import (
    CoilLength, CoilSetMinDistance, FixCoilCurrent, ObjectiveFunction,
    PlasmaCoilSetDistanceBound, QuadraticFlux,
)
from desc.optimize import Optimizer

eq = data["eq"]
shaping = data["shaping"]
encircling = data["encircling"]
tol = config["tolerances"]
w = config["weights"]
g = config["grid"]
sm = config["softmin"]

JAC_CHUNK = int(os.environ.get("JAC_CHUNK", "32"))

objectives = [
    QuadraticFlux(
        eq, field=[shaping, encircling],
        eval_grid=LinearGrid(M=g["eval_grid_M"], N=g["eval_grid_N"], NFP=eq.NFP),
        vacuum=True, weight=w["quadratic_flux"],
        jac_chunk_size=JAC_CHUNK,
    ),
    CoilSetMinDistance(
        shaping, bounds=(tol["coil_coil_dist_min"], np.inf),
        use_softmin=sm["use_softmin"], softmin_alpha=sm["softmin_alpha"],
        grid=LinearGrid(N=g["coil_grid_N"]), weight=w["coil_coil_distance"],
        jac_chunk_size=JAC_CHUNK,
    ),
    FixCoilCurrent(
        shaping, bounds=(-tol["max_coil_current"], tol["max_coil_current"]),
        weight=w["fix_coil_current"],
    ),
    PlasmaCoilSetDistanceBound(
        eq, shaping, bounds=tol["plasma_coil_dist"], eq_fixed=True,
        use_softmin=sm["use_softmin"], softmin_alpha=sm["softmin_alpha"],
        plasma_grid=LinearGrid(M=g["plasma_grid_M"], N=g["plasma_grid_N"], NFP=eq.NFP),
        coil_grid=LinearGrid(N=g["coil_grid_N"]), weight=w["plasma_coil_distance"],
        jac_chunk_size=JAC_CHUNK,
    ),
    CoilLength(
        shaping, weight=w["coil_length"],
        jac_chunk_size=JAC_CHUNK,
    ),
]

shaping_fix = [{"r_n": True} for _ in range(len(shaping))]
from desc.objectives import FixParameters
constraints = (
    FixParameters(shaping, shaping_fix),
    FixParameters(encircling),
)

things = [shaping, encircling]

OPTIMIZER = os.environ.get("OPT", "lsq-exact")
MAXITER = int(os.environ.get("MAXITER", "3"))

for mode in ["batched", "blocked"]:
    print(f"\n{'='*60}")
    print(f"deriv_mode={mode}, jac_chunk={JAC_CHUNK}, optimizer={OPTIMIZER}")
    print(f"{'='*60}")

    if mode == "batched":
        # Batched mode: chunk at ObjectiveFunction level, not per-objective
        # Rebuild objectives without per-objective chunk sizes
        objs_no_chunk = [
            QuadraticFlux(
                eq, field=[shaping, encircling],
                eval_grid=LinearGrid(M=g["eval_grid_M"], N=g["eval_grid_N"], NFP=eq.NFP),
                vacuum=True, weight=w["quadratic_flux"],
            ),
            CoilSetMinDistance(
                shaping, bounds=(tol["coil_coil_dist_min"], np.inf),
                use_softmin=sm["use_softmin"], softmin_alpha=sm["softmin_alpha"],
                grid=LinearGrid(N=g["coil_grid_N"]), weight=w["coil_coil_distance"],
            ),
            FixCoilCurrent(
                shaping, bounds=(-tol["max_coil_current"], tol["max_coil_current"]),
                weight=w["fix_coil_current"],
            ),
            PlasmaCoilSetDistanceBound(
                eq, shaping, bounds=tol["plasma_coil_dist"], eq_fixed=True,
                use_softmin=sm["use_softmin"], softmin_alpha=sm["softmin_alpha"],
                plasma_grid=LinearGrid(M=g["plasma_grid_M"], N=g["plasma_grid_N"], NFP=eq.NFP),
                coil_grid=LinearGrid(N=g["coil_grid_N"]), weight=w["plasma_coil_distance"],
            ),
            CoilLength(shaping, weight=w["coil_length"]),
        ]
        obj_fn = ObjectiveFunction(
            tuple(objs_no_chunk), deriv_mode="batched", jac_chunk_size=JAC_CHUNK,
        )
    else:
        # Blocked mode: chunk per-objective
        obj_fn = ObjectiveFunction(
            tuple(objectives), deriv_mode="blocked",
        )
    obj_fn.build(verbose=0)
    x = obj_fn.x(*things)
    print(f"DOFs: {len(x)}, dim_f: {obj_fn.dim_f}")

    opt = Optimizer(OPTIMIZER)
    t0 = time.perf_counter()
    try:
        _things_out, result = opt.optimize(
            things=things, objective=obj_fn, constraints=constraints,
            maxiter=MAXITER, ftol=0, xtol=0, gtol=0, verbose=2, copy=True,
        )
        wall = time.perf_counter() - t0
        print(f"\n{mode}: {wall:.1f}s, njev={result.njev}")
    except Exception as e:
        wall = time.perf_counter() - t0
        print(f"\n{mode}: FAILED after {wall:.1f}s — {e}")
