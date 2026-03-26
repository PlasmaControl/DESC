#!/usr/bin/env python
"""Profile auglag at full resolution: where does wall time go?

Instruments lsqtr fun/jac calls to get per-iteration breakdown.
Also instruments LCP build separately.
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
from desc.optimize import least_squares as ls_mod
from desc.optimize._constraint_wrappers import LinearConstraintProjection

eq = data["eq"]
shaping = data["shaping"]
encircling = data["encircling"]
tol = config["tolerances"]
w = config["weights"]
g = config["grid"]
sm = config["softmin"]

JAC_CHUNK = int(os.environ.get("JAC_CHUNK", "1"))

objectives = [
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
constraints = (FixParameters(shaping, [{"r_n": True} for _ in range(len(shaping))]),
               FixParameters(encircling))
things = [shaping, encircling]

# Instrument lsqtr
call_log = []
orig_lsqtr = ls_mod.lsqtr

def instrumented_lsqtr(fun, x0, jac, **kwargs):
    call_log.clear()
    def timed_fun(x, *args):
        t0 = time.perf_counter()
        result = fun(x, *args)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        call_log.append(("fun", time.perf_counter() - t0))
        return result
    def timed_jac(x, *args):
        t0 = time.perf_counter()
        result = jac(x, *args)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        call_log.append(("jac", time.perf_counter() - t0))
        return result
    return orig_lsqtr(timed_fun, x0, timed_jac, **kwargs)

ls_mod.lsqtr = instrumented_lsqtr

# Also instrument the auglag wrapper to capture its outer loop
from desc.optimize import _desc_wrappers as dw
orig_aug = dw._optimize_desc_aug_lagrangian
aug_outer_log = []

def instrumented_aug(*args, **kwargs):
    # Wrap the lsqtr calls to track per-outer-iteration
    aug_outer_log.append(("aug_start", time.perf_counter()))
    result = orig_aug(*args, **kwargs)
    aug_outer_log.append(("aug_end", time.perf_counter()))
    return result

dw._optimize_desc_aug_lagrangian = instrumented_aug

print(f"Full resolution: {len(shaping)} coils, jac_chunk={JAC_CHUNK}")

# Build and time LCP
obj_fn = ObjectiveFunction(
    tuple(objectives), deriv_mode="batched", jac_chunk_size=JAC_CHUNK,
)
t_build_0 = time.perf_counter()
obj_fn.build(verbose=0)
t_build = time.perf_counter() - t_build_0
print(f"ObjectiveFunction build: {t_build:.1f}s")

x = obj_fn.x(*things)
print(f"DOFs: {len(x)}, dim_f: {obj_fn.dim_f}")

# Run auglag
opt = Optimizer("lsq-auglag")
t0 = time.perf_counter()
_things_out, result = opt.optimize(
    things=things, objective=obj_fn, constraints=constraints,
    maxiter=3, ftol=0, xtol=0, gtol=0, verbose=2, copy=True,
)
total_wall = time.perf_counter() - t0

# Restore
ls_mod.lsqtr = orig_lsqtr

# Report
fun_times = [t for label, t in call_log if label == "fun"]
jac_times = [t for label, t in call_log if label == "jac"]
total_fun = sum(fun_times)
total_jac = sum(jac_times)
other = total_wall - total_fun - total_jac

print(f"\n{'='*60}")
print(f"FULL-RES AUGLAG PROFILE (jac_chunk={JAC_CHUNK})")
print(f"{'='*60}")
print(f"Total wall time:     {total_wall:.1f}s")
print(f"ObjFn build:         {t_build:.1f}s")
print(f"Fun evals:           {len(fun_times):>4d} calls, {total_fun:.1f}s "
      f"({100*total_fun/total_wall:.0f}%)")
print(f"Jac evals:           {len(jac_times):>4d} calls, {total_jac:.1f}s "
      f"({100*total_jac/total_wall:.0f}%)")
print(f"Other (LCP+TR+aug):  {other:.1f}s ({100*other/total_wall:.0f}%)")

if fun_times:
    print(f"\nFun per-call: mean={np.mean(fun_times):.2f}s, "
          f"min={np.min(fun_times):.2f}s, max={np.max(fun_times):.2f}s")
if jac_times:
    print(f"Jac per-call: mean={np.mean(jac_times):.2f}s, "
          f"min={np.min(jac_times):.2f}s, max={np.max(jac_times):.2f}s")

print(f"\nTimeline:")
for i, (label, t) in enumerate(call_log):
    print(f"  {i:>3d}. {label:4s} {t:.2f}s")
