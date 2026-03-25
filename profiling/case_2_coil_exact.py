"""Case 2: coil optimization with lsq-exact optimizer.

Identical to case_1 except uses Optimizer("lsq-exact").
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from instrument import (
    monkey_patch_lsqtr,
    print_lsqtr_breakdown,
    print_objective_breakdown,
    save_results,
    setup_gpu,
    time_per_objective_jac,
)

# Must be called before any JAX/DESC imports
setup_gpu()

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/finding_better_basins"))
from configs import build_config, build_objectives, load_data, setup_jax  # noqa: E402

from desc.objectives import ObjectiveFunction  # noqa: E402
from desc.optimize import Optimizer  # noqa: E402

config = build_config(platform="perlmutter", lightweight=True, seed=0)
setup_jax(config)

call_log, restore_lsqtr = monkey_patch_lsqtr()

data = load_data(config)
problem = build_objectives(data, config)

obj_fn = ObjectiveFunction(
    problem["objectives"],
    deriv_mode="batched",
    jac_chunk_size=config["probe"]["jac_chunk_size"],
)
obj_fn.build()

x0 = np.asarray(obj_fn.x(*problem["things"])).copy()

# Per-objective Jacobian profiling
constants = obj_fn.constants
obj_jac_results = time_per_objective_jac(obj_fn, x0, constants)

optimizer = Optimizer("lsq-exact")
result = optimizer.optimize(
    obj_fn,
    constraints=problem.get("constraints"),
    x0=x0,
    maxiter=5,
    ftol=0,
    xtol=0,
    gtol=0,
    copy=False,
    verbose=3,
)

restore_lsqtr()

print_lsqtr_breakdown(call_log, label="case_2_coil_exact")
print_objective_breakdown(obj_jac_results, label="case_2_coil_exact")

fun_times = [t for name, t in call_log if name == "fun"]
jac_times = [t for name, t in call_log if name == "jac"]

results_dict = {
    "case": "case_2_coil_exact",
    "optimizer": "lsq-exact",
    "maxiter": 5,
    "n_fun_calls": len(fun_times),
    "n_jac_calls": len(jac_times),
    "total_fun_time_s": float(sum(fun_times)),
    "total_jac_time_s": float(sum(jac_times)),
    "mean_fun_time_s": float(np.mean(fun_times)) if fun_times else None,
    "mean_jac_time_s": float(np.mean(jac_times)) if jac_times else None,
    "call_log": call_log,
    "per_objective_jac": obj_jac_results,
}

save_results(
    results_dict,
    os.path.join(os.path.dirname(__file__), "results_case_2.json"),
)
