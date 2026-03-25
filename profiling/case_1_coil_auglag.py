"""Profile Case 1: lightweight coil optimization using lsq-auglag."""

import os
import sys
import time

# Must be before any JAX/DESC imports
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

# Load configs from finding_better_basins
sys.path.insert(0, os.path.expanduser("~/finding_better_basins"))
from configs import build_config, build_objectives, load_data, setup_jax

# Build lightweight config
config = build_config(platform="perlmutter", lightweight=True, seed=0)

# Setup JAX
setup_jax(config)

# Monkey-patch lsqtr before optimization
call_log, restore_lsqtr = monkey_patch_lsqtr()

# Load data and build objectives
data = load_data(config)
problem = build_objectives(data, config)

# Build ObjectiveFunction
from desc.objectives import ObjectiveFunction

obj_fn = ObjectiveFunction(
    tuple(problem["objective_list"]),
    deriv_mode="batched",
    jac_chunk_size=config["probe"]["jac_chunk_size"],
)
obj_fn.build(verbose=0)

# Get initial point
x0 = np.asarray(obj_fn.x(*problem["things"])).copy()

print(f"DOFs: {len(x0)}")
print(f"dim_f: {obj_fn.dim_f}")

# Per-objective Jacobian profiling
constants = obj_fn.constants  # property, not method call
obj_results = time_per_objective_jac(obj_fn, x0, constants)
print_objective_breakdown(obj_results, label="(Case 1: coil auglag)")

# Run optimizer
from desc.optimize import Optimizer

opt = Optimizer("lsq-auglag")

t_start = time.perf_counter()
_things_out, result = opt.optimize(
    things=problem["things"],
    objective=obj_fn,
    constraints=problem["constraints"],
    maxiter=5,
    ftol=0,
    xtol=0,
    gtol=0,
    verbose=2,
    copy=True,
)
wall = time.perf_counter() - t_start

# Print breakdown and timing
print_lsqtr_breakdown(call_log)
print(f"Total wall time: {wall:.2f}s")

# Save results
save_results(
    {
        "case": "case_1_coil_auglag",
        "wall_time_s": wall,
        "dofs": len(x0),
        "dim_f": obj_fn.dim_f,
        "call_log": call_log,
        "objective_breakdown": obj_results,
    },
    "profiling/results_case_1.json",
)

# Restore lsqtr
restore_lsqtr()
