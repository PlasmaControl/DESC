#!/usr/bin/env python
"""Sweep jac_chunk_size on coil optimization Jacobian.

Usage: CUDA_VISIBLE_DEVICES=X python profiling/sweep_jac_chunk.py [chunk_sizes...]
       CUDA_VISIBLE_DEVICES=0 python profiling/sweep_jac_chunk.py 1 8 32 128
"""
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
problem = build_objectives(data, config)

from desc.objectives import ObjectiveFunction

# Parse chunk sizes from argv
if len(sys.argv) > 1:
    CHUNKS = [int(x) for x in sys.argv[1:]]
else:
    CHUNKS = [1, 8, 32, 128, 512, 2048, 3561, 10000]

results = []
for chunk in CHUNKS:
    print(f"\n{'='*60}")
    print(f"jac_chunk_size = {chunk}")
    print(f"{'='*60}")

    obj_fn = ObjectiveFunction(
        tuple(problem["objective_list"]),
        deriv_mode="batched",
        jac_chunk_size=chunk,
    )
    obj_fn.build(verbose=0)
    x0 = obj_fn.x(*problem["things"])
    n_passes = int(np.ceil(len(x0) / chunk))
    print(f"DOFs: {len(x0)}, dim_f: {obj_fn.dim_f}, "
          f"chunk: {chunk}, forward passes: {n_passes}")

    # Warmup
    t0 = time.perf_counter()
    J = obj_fn.jac_scaled(x0)
    J.block_until_ready()
    warmup = time.perf_counter() - t0
    print(f"Warmup (JIT): {warmup:.1f}s")

    # Timed: 3 evals
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        J = obj_fn.jac_scaled(x0)
        J.block_until_ready()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  trial {i+1}: {elapsed:.2f}s")

    med = float(np.median(times))
    print(f"  median: {med:.2f}s")

    results.append({
        "jac_chunk_size": chunk,
        "n_passes": n_passes,
        "warmup_s": warmup,
        "median_s": med,
        "times_s": times,
        "dofs": len(x0),
        "dim_f": obj_fn.dim_f,
    })

# Summary table
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"{'chunk':>8s} {'passes':>8s} {'median_s':>10s} {'warmup_s':>10s} {'speedup':>8s}")
baseline = results[0]["median_s"] if results else 1
for r in results:
    speedup = baseline / r["median_s"] if r["median_s"] > 0 else 0
    print(f"{r['jac_chunk_size']:>8d} {r['n_passes']:>8d} "
          f"{r['median_s']:>9.2f}s {r['warmup_s']:>9.1f}s {speedup:>7.1f}x")

# Save
out_path = "profiling/results_jac_chunk_sweep.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
