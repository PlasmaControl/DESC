#!/usr/bin/env python
"""Sweep bs_chunk_size on coil optimization (Case 1) at fixed jac_chunk_size=128."""
import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from instrument import setup_gpu
setup_gpu()

import numpy as np

sys.path.insert(0, os.path.expanduser("~/finding_better_basins"))
from configs import build_config, build_objectives, load_data, setup_jax

config = build_config(platform="perlmutter", lightweight=True, seed=0)
setup_jax(config)

print("Loading data...")
data = load_data(config)

# We need to rebuild objectives with different bs_chunk_size each time
from desc.grid import LinearGrid
from desc.objectives import (
    CoilSetMinDistance, FixCoilCurrent, ObjectiveFunction,
    PlasmaCoilSetDistanceBound, QuadraticFlux,
)

eq = data["eq"]
shaping = data["shaping"]
encircling = data["encircling"]
tol = config["tolerances"]
w = config["weights"]
g = config["grid"]
sm = config["softmin"]

JAC_CHUNK = 128  # Fixed at good value from prior sweep

BS_CHUNKS = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [None, 1, 4, 16, 64, 256]

results = []
for bs_chunk in BS_CHUNKS:
    print(f"\n{'='*60}")
    print(f"bs_chunk_size = {bs_chunk}")
    print(f"{'='*60}")

    objectives = [
        QuadraticFlux(
            eq, field=[shaping, encircling],
            eval_grid=LinearGrid(M=g["eval_grid_M"], N=g["eval_grid_N"], NFP=eq.NFP),
            vacuum=True, weight=w["quadratic_flux"],
            bs_chunk_size=bs_chunk,
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
    ]

    obj_fn = ObjectiveFunction(tuple(objectives), deriv_mode="batched", jac_chunk_size=JAC_CHUNK)
    obj_fn.build(verbose=0)
    x0 = obj_fn.x(*[shaping, encircling])
    print(f"DOFs: {len(x0)}, dim_f: {obj_fn.dim_f}, jac_chunk: {JAC_CHUNK}, bs_chunk: {bs_chunk}")

    t0 = time.perf_counter()
    J = obj_fn.jac_scaled(x0); J.block_until_ready()
    warmup = time.perf_counter() - t0
    print(f"Warmup (JIT): {warmup:.1f}s")

    times = []
    for i in range(3):
        t0 = time.perf_counter()
        J = obj_fn.jac_scaled(x0); J.block_until_ready()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  trial {i+1}: {elapsed:.2f}s")

    med = float(np.median(times))
    print(f"  median: {med:.2f}s")
    results.append({
        "bs_chunk_size": bs_chunk, "jac_chunk_size": JAC_CHUNK,
        "warmup_s": warmup, "median_s": med, "times_s": times,
        "dofs": len(x0), "dim_f": obj_fn.dim_f,
    })

print(f"\n{'='*60}")
print(f"SUMMARY (bs_chunk_size sweep, jac_chunk={JAC_CHUNK})")
print(f"{'='*60}")
print(f"{'bs_chunk':>10s} {'median_s':>10s} {'warmup_s':>10s} {'speedup':>8s}")
baseline = results[0]["median_s"]
for r in results:
    speedup = baseline / r["median_s"] if r["median_s"] > 0 else 0
    bs = str(r["bs_chunk_size"])
    print(f"{bs:>10s} {r['median_s']:>9.2f}s {r['warmup_s']:>9.1f}s {speedup:>7.1f}x")

with open("profiling/results_bs_chunk_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: profiling/results_bs_chunk_sweep.json")
