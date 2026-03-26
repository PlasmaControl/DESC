#!/usr/bin/env python
"""Sweep jac_chunk_size on single-stage (Case 4)."""
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
from desc.coils import CoilSet, MixedCoilSet
from desc.grid import LinearGrid, ConcentricGrid
from desc.objectives import (
    BoundaryError, BootstrapRedlConsistency, FixAtomicNumber,
    FixBoundaryR, FixBoundaryZ, FixCoilCurrent, FixCurrent,
    FixElectronDensity, FixElectronTemperature, FixIonTemperature,
    FixParameters, FixPsi, ForceBalance, ObjectiveFunction,
    QuasisymmetryTwoTerm,
)
from desc.objectives._coils import FieldNormalError
from desc.optimize import Optimizer

# Read case 4 script to reuse its setup
# We import the already-written case_4 and just rebuild with different chunk sizes
# Actually, easier to just load data and build objectives inline

HOME = os.path.expanduser("~")

# Use the case_4 script's CONFIG - read it to get paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Minimal reproduction: load eq + coils, build same objectives as case_4
# but only vary jac_chunk_size

# Load case_4 results to get the problem dimensions
print("Loading Case 4 setup...")

# Import case_4's setup by exec'ing up to the objective build
# This is hacky but avoids duplicating 100 lines of setup
import importlib.util
spec = importlib.util.spec_from_file_location("case4", "profiling/case_4_single_stage.py")
# Actually, let's just run case_4 and extract what we need

# Simpler approach: read the case_4 JSON results for dimensions,
# then build a minimal version here
print("Building minimal single-stage setup...")

DATA_BASE = os.path.join(HOME, "dynamic-accessibility/half_beta_half_slew_eq/slew_from_halfslew_eq-QS-divLeg")
eq0 = load(os.path.join(
    HOME, "dynamic-accessibility/half_beta_half_slew_eq/"
    "first_slew_from_halfslew_eq-QS/output/single_stage_20260313_171631/"
    "eq_single_stage_QS.h5"
))
if isinstance(eq0, EquilibriaFamily):
    eq0 = eq0[-1]
eq = eq0.copy()
eq.Psi = 0.925 * eq.Psi

encircling = load(os.path.join(DATA_BASE, "data/midbeta/encircling_midbeta.h5"))
shaping_raw = load(os.path.join(DATA_BASE, "data/midbeta/shaping_midbeta.h5"))

# LIGHTWEIGHT subsample (2%)
n_keep = max(2, int(len(shaping_raw) * 0.02))
indices = np.linspace(0, len(shaping_raw) - 1, n_keep, dtype=int)
from desc.grid import LinearGrid as LG
shaping_coils = [shaping_raw[int(i)] for i in indices]
shaping = CoilSet(*shaping_coils)

coils = MixedCoilSet(encircling, shaping)
print(f"Eq: L={eq.L}, M={eq.M}, N={eq.N}, Coils: enc={len(encircling)}, shp={len(shaping)}")

# Build lightweight objectives (subset - skip divertor legs for sweep simplicity)
grid_lcfs = LinearGrid(rho=np.array([1.0]), M=6, N=6, NFP=eq.NFP, sym=False)

objectives = [
    BoundaryError(eq=eq, field=coils, field_fixed=True,
                  eval_grid=grid_lcfs, source_grid=grid_lcfs),
    QuasisymmetryTwoTerm(eq=eq, helicity=(1, 0), grid=ConcentricGrid(L=2, M=2, N=2, NFP=eq.NFP)),
]

constraints = (
    FixCurrent(eq=eq), FixPsi(eq=eq),
    FixElectronDensity(eq=eq), FixElectronTemperature(eq=eq),
    FixIonTemperature(eq=eq), FixAtomicNumber(eq=eq),
    ForceBalance(eq=eq),
)

CHUNKS = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 8, 32, 128, 256, 384]

results = []
for chunk in CHUNKS:
    print(f"\n{'='*60}")
    print(f"jac_chunk_size = {chunk}")
    print(f"{'='*60}")

    obj_fn = ObjectiveFunction(tuple(objectives), deriv_mode="batched", jac_chunk_size=chunk)
    obj_fn.build(verbose=0)
    x0 = obj_fn.x(eq)
    n_passes = int(np.ceil(len(x0) / chunk))
    print(f"DOFs: {len(x0)}, dim_f: {obj_fn.dim_f}, chunk: {chunk}, passes: {n_passes}")

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
        "jac_chunk_size": chunk, "n_passes": n_passes,
        "warmup_s": warmup, "median_s": med, "times_s": times,
        "dofs": len(x0), "dim_f": obj_fn.dim_f,
    })

print(f"\n{'='*60}")
print(f"SUMMARY (Case 4: Single-Stage, simplified)")
print(f"{'='*60}")
print(f"{'chunk':>8s} {'passes':>8s} {'median_s':>10s} {'warmup_s':>10s} {'speedup':>8s}")
baseline = results[0]["median_s"]
for r in results:
    speedup = baseline / r["median_s"] if r["median_s"] > 0 else 0
    print(f"{r['jac_chunk_size']:>8d} {r['n_passes']:>8d} "
          f"{r['median_s']:>9.2f}s {r['warmup_s']:>9.1f}s {speedup:>7.1f}x")

with open("profiling/results_jac_chunk_sweep_case4.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: profiling/results_jac_chunk_sweep_case4.json")
