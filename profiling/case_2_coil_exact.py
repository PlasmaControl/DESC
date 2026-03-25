#!/usr/bin/env python
"""Case 2: Coil optimization with lsq-exact (lightweight).

Same problem setup as Case 1, different optimizer.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from instrument import (
    setup_gpu,
    monkey_patch_lsqtr,
    time_per_objective_jac,
    print_lsqtr_breakdown,
    print_objective_breakdown,
    save_results,
)

setup_gpu()

import numpy as np

sys.path.insert(0, os.path.expanduser("~/finding_better_basins"))
from configs import build_config, build_objectives, load_data, setup_jax

config = build_config(platform="perlmutter", lightweight=True, seed=0)
setup_jax(config)

call_log, restore_lsqtr = monkey_patch_lsqtr()

print("Loading data...")
data = load_data(config)

print("Building objectives...")
problem = build_objectives(data, config)

from desc.objectives import ObjectiveFunction
from desc.optimize import Optimizer

obj_fn = ObjectiveFunction(
    tuple(problem["objective_list"]),
    deriv_mode="batched",
    jac_chunk_size=config["probe"]["jac_chunk_size"],
)
obj_fn.build(verbose=0)

x0 = np.asarray(obj_fn.x(*problem["things"])).copy()
print(f"DOFs: {len(x0)}, dim_f: {obj_fn.dim_f}")

# Per-objective Jacobian breakdown
print("\nPer-objective Jacobian profiling...")
constants = obj_fn.constants
obj_results = time_per_objective_jac(obj_fn, x0, constants)
print_objective_breakdown(obj_results, label="(Case 2: coil exact)")

# Run optimizer
print(f"\nRunning lsq-exact (5 iterations)...")
opt = Optimizer("lsq-exact")
t0 = time.perf_counter()
_things_out, result = opt.optimize(
    things=problem["things"],
    objective=obj_fn,
    constraints=problem["constraints"],
    maxiter=5,
    ftol=0, xtol=0, gtol=0,
    verbose=2,
    copy=True,
)
wall = time.perf_counter() - t0

print_lsqtr_breakdown(call_log, label="(Case 2: coil exact)")
print(f"\nTotal wall time: {wall:.1f}s")

save_results({
    "case": "case_2_coil_exact",
    "wall_time_s": wall,
    "dofs": len(x0),
    "dim_f": obj_fn.dim_f,
    "call_log": call_log,
    "objective_breakdown": obj_results,
}, "profiling/results_case_2.json")

restore_lsqtr()
