#!/usr/bin/env python
"""Sweep jac_chunk_size on free boundary equilibrium (Case 3)."""
import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from instrument import setup_gpu
setup_gpu()

import numpy as np
from desc.io import load
from desc.equilibrium import EquilibriaFamily
from desc.grid import LinearGrid
from desc.objectives import (
    BoundaryError, FixAtomicNumber, FixCurrent, FixElectronDensity,
    FixElectronTemperature, FixIonTemperature, FixPsi, ForceBalance,
    ObjectiveFunction,
)

HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "dynamic-accessibility/half_beta_half_slew_eq/second_free_boundary_proper-tf")

print("Loading data...")
encircling = load(os.path.join(DATA_DIR, "data/midbeta/encircling_midbeta.h5"))
shaping = load(os.path.join(DATA_DIR, "data/midbeta/shaping_midbeta.h5"))
eq0 = load(os.path.join(DATA_DIR, "data/midbeta/equil_G1635_DESC_fixed.h5"))
if isinstance(eq0, EquilibriaFamily):
    eq0 = eq0[-1]
eq = eq0.copy()
eq.Psi = 0.925 * eq.Psi
print(f"Equilibrium: L={eq.L}, M={eq.M}, N={eq.N}, NFP={eq.NFP}")

grid_enc = LinearGrid(N=max(encircling[0].N, 32))
grid_shp = LinearGrid(N=max(shaping[0].N, 16))
grid_lcfs = LinearGrid(rho=np.array([1.0]), M=8, N=8, NFP=eq.NFP, sym=False)

CHUNKS = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 8, 32, 128, 256, 384]

results = []
for chunk in CHUNKS:
    print(f"\n{'='*60}")
    print(f"jac_chunk_size = {chunk}")
    print(f"{'='*60}")

    objective = ObjectiveFunction(
        BoundaryError(
            eq=eq, field=[encircling, shaping], target=0,
            source_grid=grid_lcfs, eval_grid=grid_lcfs,
            field_grid=[grid_enc, grid_shp], field_fixed=True,
        ),
        deriv_mode="batched",
        jac_chunk_size=chunk,
    )
    objective.build(verbose=0)
    x0 = objective.x(eq)
    n_passes = int(np.ceil(len(x0) / chunk))
    print(f"DOFs: {len(x0)}, dim_f: {objective.dim_f}, chunk: {chunk}, passes: {n_passes}")

    t0 = time.perf_counter()
    J = objective.jac_scaled(x0); J.block_until_ready()
    warmup = time.perf_counter() - t0
    print(f"Warmup (JIT): {warmup:.1f}s")

    times = []
    for i in range(3):
        t0 = time.perf_counter()
        J = objective.jac_scaled(x0); J.block_until_ready()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  trial {i+1}: {elapsed:.2f}s")

    med = float(np.median(times))
    print(f"  median: {med:.2f}s")
    results.append({
        "jac_chunk_size": chunk, "n_passes": n_passes,
        "warmup_s": warmup, "median_s": med, "times_s": times,
        "dofs": len(x0), "dim_f": objective.dim_f,
    })

print(f"\n{'='*60}")
print(f"SUMMARY (Case 3: Free Boundary)")
print(f"{'='*60}")
print(f"{'chunk':>8s} {'passes':>8s} {'median_s':>10s} {'warmup_s':>10s} {'speedup':>8s}")
baseline = results[0]["median_s"]
for r in results:
    speedup = baseline / r["median_s"] if r["median_s"] > 0 else 0
    print(f"{r['jac_chunk_size']:>8d} {r['n_passes']:>8d} "
          f"{r['median_s']:>9.2f}s {r['warmup_s']:>9.1f}s {speedup:>7.1f}x")

with open("profiling/results_jac_chunk_sweep_case3.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: profiling/results_jac_chunk_sweep_case3.json")
