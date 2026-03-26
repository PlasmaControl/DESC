#!/usr/bin/env python
"""Sweep jac_chunk_size 1-8 at full resolution, batched mode.

Usage: CUDA_VISIBLE_DEVICES=X python profiling/fullres_chunk_sweep.py [chunks...]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from instrument import setup_gpu
setup_gpu()

import numpy as np
import json

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

CHUNKS = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 3, 4, 6, 8]

things = [shaping, encircling]
constraints = (FixParameters(shaping, [{"r_n": True} for _ in range(len(shaping))]),
               FixParameters(encircling))

print(f"Full resolution: {len(shaping)} coils")

results = []
for chunk in CHUNKS:
    print(f"\n{'='*60}")
    print(f"jac_chunk_size = {chunk}")
    print(f"{'='*60}")

    objectives = [
        QuadraticFlux(eq, field=[shaping, encircling],
            eval_grid=LinearGrid(M=g["eval_grid_M"], N=g["eval_grid_N"], NFP=eq.NFP),
            vacuum=True, weight=w["quadratic_flux"]),
        CoilSetMinDistance(shaping, bounds=(tol["coil_coil_dist_min"], np.inf),
            use_softmin=sm["use_softmin"], softmin_alpha=sm["softmin_alpha"],
            grid=LinearGrid(N=g["coil_grid_N"]), weight=w["coil_coil_distance"]),
        FixCoilCurrent(shaping, bounds=(-tol["max_coil_current"], tol["max_coil_current"]),
            weight=w["fix_coil_current"]),
        PlasmaCoilSetDistanceBound(eq, shaping, bounds=tol["plasma_coil_dist"], eq_fixed=True,
            use_softmin=sm["use_softmin"], softmin_alpha=sm["softmin_alpha"],
            plasma_grid=LinearGrid(M=g["plasma_grid_M"], N=g["plasma_grid_N"], NFP=eq.NFP),
            coil_grid=LinearGrid(N=g["coil_grid_N"]), weight=w["plasma_coil_distance"]),
        CoilLength(shaping, weight=w["coil_length"]),
    ]

    try:
        obj_fn = ObjectiveFunction(tuple(objectives), deriv_mode="batched", jac_chunk_size=chunk)
        obj_fn.build(verbose=0)
        x = obj_fn.x(*things)
        print(f"DOFs: {len(x)}, dim_f: {obj_fn.dim_f}")

        opt = Optimizer("lsq-exact")
        t0 = time.perf_counter()
        _things_out, result = opt.optimize(
            things=things, objective=obj_fn, constraints=constraints,
            maxiter=3, ftol=0, xtol=0, gtol=0, verbose=2, copy=True,
        )
        wall = time.perf_counter() - t0
        print(f"chunk={chunk}: {wall:.1f}s, njev={result.njev}")
        results.append({"chunk": chunk, "wall_s": wall, "njev": result.njev, "status": "ok"})
    except Exception as e:
        print(f"chunk={chunk}: FAILED — {e}")
        results.append({"chunk": chunk, "status": "oom"})

print(f"\n{'='*60}")
print(f"SUMMARY (full-res batched, lsq-exact, 3 iter)")
print(f"{'='*60}")
print(f"{'chunk':>6s} {'wall_s':>8s} {'njev':>5s}")
baseline = next((r["wall_s"] for r in results if r["status"] == "ok"), 1)
for r in results:
    if r["status"] == "ok":
        speedup = baseline / r["wall_s"]
        print(f"{r['chunk']:>6d} {r['wall_s']:>7.1f}s {r['njev']:>5d}  {speedup:.2f}x")
    else:
        print(f"{r['chunk']:>6d}     OOM")
