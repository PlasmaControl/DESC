#!/usr/bin/env python
"""Test and benchmark the n_neighbors pruning in DESC's CoilSetMinDistance.

1. Correctness: compare n_neighbors=20 vs full at lightweight resolution
2. Benchmark: full resolution Jacobian timing
3. Memory: test larger chunk sizes enabled by pruning
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
from desc import set_device
set_device("gpu")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.expanduser("~/finding_better_basins"))
from configs import build_config, load_data, setup_jax

from desc.grid import LinearGrid
from desc.objectives import (
    CoilLength, CoilSetMinDistance, FixCoilCurrent, FixParameters,
    ObjectiveFunction, PlasmaCoilSetDistanceBound, QuadraticFlux,
)
from desc.optimize import Optimizer

# ============================================================
# Part 1: Correctness at lightweight
# ============================================================
print("=== Part 1: Correctness (lightweight) ===")
config = build_config(platform="perlmutter", lightweight=True, seed=0)
setup_jax(config)
data = load_data(config)

eq = data["eq"]
shaping = data["shaping"]
encircling = data["encircling"]
tol = config["tolerances"]
w = config["weights"]
g = config["grid"]
sm = config["softmin"]

# Build with n_neighbors=None (full)
obj_full = CoilSetMinDistance(
    shaping, bounds=(tol["coil_coil_dist_min"], np.inf),
    use_softmin=sm["use_softmin"], softmin_alpha=sm["softmin_alpha"],
    grid=LinearGrid(N=g["coil_grid_N"]), weight=w["coil_coil_distance"],
)
obj_full.build(verbose=0)
f_full = obj_full.compute(shaping.params_dict)
f_full.block_until_ready()

for K in [5, 10, 20, 50]:
    obj_pruned = CoilSetMinDistance(
        shaping, bounds=(tol["coil_coil_dist_min"], np.inf),
        use_softmin=sm["use_softmin"], softmin_alpha=sm["softmin_alpha"],
        grid=LinearGrid(N=g["coil_grid_N"]), weight=w["coil_coil_distance"],
        n_neighbors=K,
    )
    obj_pruned.build(verbose=0)
    f_pruned = obj_pruned.compute(shaping.params_dict)
    f_pruned.block_until_ready()
    diff = jnp.abs(f_pruned - f_full)
    print(f"  K={K:>3d}: max_diff={float(jnp.max(diff)):.2e}, "
          f"n_wrong={int(jnp.sum(diff > 1e-6))}/{len(f_full)}")

# Jacobian correctness
print("\nJacobian correctness (lightweight):")
obj_fn_full = ObjectiveFunction(
    (CoilSetMinDistance(
        shaping, bounds=(tol["coil_coil_dist_min"], np.inf),
        use_softmin=sm["use_softmin"], softmin_alpha=sm["softmin_alpha"],
        grid=LinearGrid(N=g["coil_grid_N"]),
    ),),
    deriv_mode="batched", jac_chunk_size=1,
)
obj_fn_full.build(verbose=0)
x = obj_fn_full.x(shaping)
J_full = obj_fn_full.jac_scaled(x)
J_full.block_until_ready()

for K in [10, 20]:
    obj_fn_pruned = ObjectiveFunction(
        (CoilSetMinDistance(
            shaping, bounds=(tol["coil_coil_dist_min"], np.inf),
            use_softmin=sm["use_softmin"], softmin_alpha=sm["softmin_alpha"],
            grid=LinearGrid(N=g["coil_grid_N"]),
            n_neighbors=K,
        ),),
        deriv_mode="batched", jac_chunk_size=1,
    )
    obj_fn_pruned.build(verbose=0)
    J_pruned = obj_fn_pruned.jac_scaled(x)
    J_pruned.block_until_ready()
    diff = jnp.abs(J_pruned - J_full)
    print(f"  K={K:>3d}: max_jac_diff={float(jnp.max(diff)):.2e}")

# ============================================================
# Part 2: Full resolution benchmark
# ============================================================
print(f"\n=== Part 2: Full Resolution Benchmark ===")
config_fr = build_config(platform="perlmutter", lightweight=False, seed=0)
data_fr = load_data(config_fr)
eq_fr = data_fr["eq"]
shaping_fr = data_fr["shaping"]
encircling_fr = data_fr["encircling"]
tol_fr = config_fr["tolerances"]
w_fr = config_fr["weights"]
g_fr = config_fr["grid"]
sm_fr = config_fr["softmin"]

things = [shaping_fr, encircling_fr]
constraints = (
    FixParameters(shaping_fr, [{"r_n": True} for _ in range(len(shaping_fr))]),
    FixParameters(encircling_fr),
)

print(f"Coils: {len(shaping_fr)} unique, {shaping_fr.num_coils} effective")

# Benchmark: full multi-objective with pruning vs without
configs_to_test = [
    ("baseline (chunk=1)", 1, None),
    ("baseline (chunk=4)", 4, None),
    ("pruned K=20 (chunk=4)", 4, 20),
    ("pruned K=20 (chunk=8)", 8, 20),
    ("pruned K=20 (chunk=16)", 16, 20),
    ("pruned K=20 (chunk=32)", 32, 20),
]

results = []
for label, chunk, K in configs_to_test:
    print(f"\n--- {label} ---")
    try:
        objectives = [
            QuadraticFlux(eq_fr, field=[shaping_fr, encircling_fr],
                eval_grid=LinearGrid(M=g_fr["eval_grid_M"], N=g_fr["eval_grid_N"], NFP=eq_fr.NFP),
                vacuum=True, weight=w_fr["quadratic_flux"]),
            CoilSetMinDistance(shaping_fr, bounds=(tol_fr["coil_coil_dist_min"], np.inf),
                use_softmin=sm_fr["use_softmin"], softmin_alpha=sm_fr["softmin_alpha"],
                grid=LinearGrid(N=g_fr["coil_grid_N"]), weight=w_fr["coil_coil_distance"],
                n_neighbors=K),
            FixCoilCurrent(shaping_fr,
                bounds=(-tol_fr["max_coil_current"], tol_fr["max_coil_current"]),
                weight=w_fr["fix_coil_current"]),
            PlasmaCoilSetDistanceBound(eq_fr, shaping_fr,
                bounds=tol_fr["plasma_coil_dist"], eq_fixed=True,
                use_softmin=sm_fr["use_softmin"], softmin_alpha=sm_fr["softmin_alpha"],
                plasma_grid=LinearGrid(M=g_fr["plasma_grid_M"], N=g_fr["plasma_grid_N"], NFP=eq_fr.NFP),
                coil_grid=LinearGrid(N=g_fr["coil_grid_N"]),
                weight=w_fr["plasma_coil_distance"]),
            CoilLength(shaping_fr, weight=w_fr["coil_length"]),
        ]
        obj_fn = ObjectiveFunction(tuple(objectives), deriv_mode="batched", jac_chunk_size=chunk)
        obj_fn.build(verbose=0)
        x = obj_fn.x(*things)
        print(f"DOFs: {len(x)}, dim_f: {obj_fn.dim_f}")

        # Run lsq-exact for 3 iterations
        opt = Optimizer("lsq-exact")
        t0 = time.perf_counter()
        _things_out, result = opt.optimize(
            things=things, objective=obj_fn, constraints=constraints,
            maxiter=3, ftol=0, xtol=0, gtol=0, verbose=2, copy=True,
        )
        wall = time.perf_counter() - t0
        print(f"{label}: {wall:.1f}s, njev={result.njev}")
        results.append({"label": label, "wall_s": wall, "njev": result.njev})
    except Exception as e:
        err = str(e)[:80]
        print(f"{label}: FAILED — {err}")
        results.append({"label": label, "wall_s": None, "status": "oom"})

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY (full-res, lsq-exact, 3 iter)")
print(f"{'='*60}")
baseline = next((r["wall_s"] for r in results if r.get("wall_s") and "baseline" in r["label"]), None)
for r in results:
    if r.get("wall_s"):
        speedup = f"{baseline/r['wall_s']:.2f}x" if baseline else ""
        print(f"  {r['label']:<35s} {r['wall_s']:>7.1f}s  njev={r['njev']}  {speedup}")
    else:
        print(f"  {r['label']:<35s}     OOM")
