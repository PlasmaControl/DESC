#!/usr/bin/env python
"""Test dist_chunk_size + blocked mode at full resolution.

Hypothesis: reducing dist_chunk_size on CoilSetMinDistance reduces forward-pass
memory enough to allow blocked mode with jac_chunk_size > 1.

Usage: CUDA_VISIBLE_DEVICES=X python profiling/fullres_dist_chunk_blocked.py
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

from desc.grid import LinearGrid
from desc.objectives import (
    CoilLength, CoilSetMinDistance, FixCoilCurrent, FixParameters,
    ObjectiveFunction, PlasmaCoilSetDistanceBound, QuadraticFlux,
)
from desc.optimize import Optimizer

eq = data["eq"]
shaping = data["shaping"]
encircling = data["encircling"]
tol = config["tolerances"]
w = config["weights"]
g = config["grid"]
sm = config["softmin"]

things = [shaping, encircling]
constraints = (FixParameters(shaping, [{"r_n": True} for _ in range(len(shaping))]),
               FixParameters(encircling))

print(f"Full resolution: {len(shaping)} coils, {shaping.num_coils} effective")

# Test combinations of dist_chunk_size and jac_chunk_size in blocked mode
configs = [
    # (dist_chunk, jac_chunk, label)
    (None, 1, "blocked/dist=None/jac=1"),
    (10, 1, "blocked/dist=10/jac=1"),
    (10, 2, "blocked/dist=10/jac=2"),
    (10, 4, "blocked/dist=10/jac=4"),
    (5, 4, "blocked/dist=5/jac=4"),
    (5, 8, "blocked/dist=5/jac=8"),
]

for dist_chunk, jac_chunk, label in configs:
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    objectives = [
        QuadraticFlux(eq, field=[shaping, encircling],
            eval_grid=LinearGrid(M=g["eval_grid_M"], N=g["eval_grid_N"], NFP=eq.NFP),
            vacuum=True, weight=w["quadratic_flux"],
            jac_chunk_size=jac_chunk),
        CoilSetMinDistance(shaping, bounds=(tol["coil_coil_dist_min"], np.inf),
            use_softmin=sm["use_softmin"], softmin_alpha=sm["softmin_alpha"],
            grid=LinearGrid(N=g["coil_grid_N"]), weight=w["coil_coil_distance"],
            jac_chunk_size=jac_chunk, dist_chunk_size=dist_chunk),
        FixCoilCurrent(shaping, bounds=(-tol["max_coil_current"], tol["max_coil_current"]),
            weight=w["fix_coil_current"]),
        PlasmaCoilSetDistanceBound(eq, shaping, bounds=tol["plasma_coil_dist"], eq_fixed=True,
            use_softmin=sm["use_softmin"], softmin_alpha=sm["softmin_alpha"],
            plasma_grid=LinearGrid(M=g["plasma_grid_M"], N=g["plasma_grid_N"], NFP=eq.NFP),
            coil_grid=LinearGrid(N=g["coil_grid_N"]), weight=w["plasma_coil_distance"],
            jac_chunk_size=jac_chunk),
        CoilLength(shaping, weight=w["coil_length"],
            jac_chunk_size=jac_chunk),
    ]

    try:
        obj_fn = ObjectiveFunction(tuple(objectives), deriv_mode="blocked")
        obj_fn.build(verbose=0)
        x = obj_fn.x(*things)
        print(f"DOFs: {len(x)}, dim_f: {obj_fn.dim_f}")

        # Just time one Jacobian eval (warmup + 1 timed)
        import jax
        J = obj_fn.jac_scaled(x)
        J.block_until_ready()
        print("Warmup OK")

        t0 = time.perf_counter()
        J = obj_fn.jac_scaled(x)
        J.block_until_ready()
        jac_time = time.perf_counter() - t0
        print(f"Jacobian: {jac_time:.2f}s, shape={J.shape}")

        # Run 2 optimizer iterations
        opt = Optimizer("lsq-exact")
        t0 = time.perf_counter()
        _things_out, result = opt.optimize(
            things=things, objective=obj_fn, constraints=constraints,
            maxiter=2, ftol=0, xtol=0, gtol=0, verbose=2, copy=True,
        )
        wall = time.perf_counter() - t0
        print(f"Optimizer: {wall:.1f}s, njev={result.njev}")

    except Exception as e:
        err = str(e)[:80]
        print(f"FAILED: {err}")
