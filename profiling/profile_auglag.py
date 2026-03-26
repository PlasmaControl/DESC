#!/usr/bin/env python
"""Profile auglag internals: where does time go beyond the Jacobian?

Instruments the auglag optimizer to time each phase:
- LCP build
- Objective/constraint evaluation
- Jacobian evaluation
- Trust-region subproblem solve
- Penalty/multiplier updates
"""
import os
import sys
import time
from functools import wraps

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from instrument import setup_gpu
setup_gpu()

import numpy as np

sys.path.insert(0, os.path.expanduser("~/finding_better_basins"))
from configs import build_config, build_objectives, load_data, setup_jax

config = build_config(platform="perlmutter", lightweight=True, seed=0)
setup_jax(config)
data = load_data(config)
problem = build_objectives(data, config)

from desc.objectives import ObjectiveFunction
from desc.optimize import Optimizer
from desc.optimize import least_squares as ls_mod
from desc.optimize import _constraint_wrappers as cw_mod

MODE = os.environ.get("MODE", "blocked")

# ============================================================
# Instrument key functions
# ============================================================
timings = {}

def instrument(module, func_name, label):
    """Wrap a function to record cumulative time."""
    orig = getattr(module, func_name)
    timings[label] = {"count": 0, "total": 0.0, "times": []}

    @wraps(orig)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig(*args, **kwargs)
        # block_until_ready if JAX array
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        elapsed = time.perf_counter() - t0
        timings[label]["count"] += 1
        timings[label]["total"] += elapsed
        timings[label]["times"].append(elapsed)
        return result

    setattr(module, func_name, wrapper)
    return orig  # return original for restoration

# Instrument lsqtr (trust-region solver)
orig_lsqtr = instrument(ls_mod, "lsqtr", "lsqtr")

# Instrument LCP build and jac
orig_lcp_jac = None
orig_lcp_build = None

# We need to instrument after the LCP is created, so wrap Optimizer.optimize
from desc.optimize.optimizer import Optimizer as OptClass

_orig_optimize = OptClass.optimize

@wraps(_orig_optimize)
def instrumented_optimize(self, *args, **kwargs):
    # Instrument LCP._jac after it's created inside optimize
    # We do this by wrapping the constraint wrapper's _jac method
    result = _orig_optimize(self, *args, **kwargs)
    return result

# Instead, let's instrument at the module level for key functions
# that get called during optimization

# Instrument the auglag wrapper's key operations
from desc.optimize import _desc_wrappers as dw_mod

# The auglag function is _optimize_desc_aug_lagrangian or similar
# Let's find it
import desc.optimize._desc_wrappers as dw

print(f"Available auglag functions: {[x for x in dir(dw) if 'aug' in x.lower() or 'lagrang' in x.lower()]}")

# ============================================================
# Run optimization with timing
# ============================================================
obj_fn = ObjectiveFunction(
    tuple(problem["objective_list"]),
    deriv_mode=MODE,
    jac_chunk_size=1 if MODE == "batched" else None,
)
obj_fn.build(verbose=0)
x0 = obj_fn.x(*problem["things"])
print(f"DOFs: {len(x0)}, dim_f: {obj_fn.dim_f}, deriv_mode={MODE}")

# Time the full optimization
opt = Optimizer("lsq-auglag")
t_total_0 = time.perf_counter()
_things_out, result = opt.optimize(
    things=problem["things"],
    objective=obj_fn,
    constraints=problem["constraints"],
    maxiter=5, ftol=0, xtol=0, gtol=0, verbose=2, copy=True,
)
t_total = time.perf_counter() - t_total_0

# Report
print(f"\n{'='*60}")
print(f"AUGLAG PROFILING ({MODE})")
print(f"{'='*60}")
print(f"Total wall time: {t_total:.1f}s")
print(f"\nInstrumented functions:")
for label, data in sorted(timings.items(), key=lambda x: -x[1]["total"]):
    if data["count"] > 0:
        print(f"  {label:<30s} {data['count']:>5d} calls  {data['total']:>8.2f}s total  "
              f"{data['total']/data['count']:>8.3f}s avg")
        if len(data["times"]) <= 20:
            print(f"    per-call: {[f'{t:.2f}' for t in data['times']]}")

# Also time individual components manually
print(f"\n{'='*60}")
print(f"COMPONENT TIMING")
print(f"{'='*60}")

# Rebuild for clean timing
obj_fn2 = ObjectiveFunction(
    tuple(problem["objective_list"]),
    deriv_mode=MODE,
    jac_chunk_size=1 if MODE == "batched" else None,
)
obj_fn2.build(verbose=0)
x = obj_fn2.x(*problem["things"])

# Time compute_scaled
from desc.backend import jnp
f = obj_fn2.compute_scaled(x)
f.block_until_ready()
times_f = []
for _ in range(5):
    t0 = time.perf_counter()
    f = obj_fn2.compute_scaled(x)
    f.block_until_ready()
    times_f.append(time.perf_counter() - t0)
print(f"compute_scaled:       {np.median(times_f)*1000:.1f}ms (median)")

# Time jac_scaled
J = obj_fn2.jac_scaled(x)
J.block_until_ready()
times_j = []
for _ in range(3):
    t0 = time.perf_counter()
    J = obj_fn2.jac_scaled(x)
    J.block_until_ready()
    times_j.append(time.perf_counter() - t0)
print(f"jac_scaled:           {np.median(times_j)*1000:.1f}ms (median)")

# Time LCP build
from desc.optimize._constraint_wrappers import LinearConstraintProjection
t0 = time.perf_counter()
lcp = LinearConstraintProjection(
    obj_fn2,
    ObjectiveFunction(problem["constraints"]),
)
lcp.build(verbose=0)
t_build = time.perf_counter() - t0
print(f"LCP build:            {t_build*1000:.0f}ms")

# Time LCP._jac
x_r = lcp.recover(lcp.project(x))[:lcp.dim_x]
# Actually get x_reduced properly
import jax.numpy as jnp2
x_reduced = lcp.project(x)
J_lcp = lcp.jac_scaled(x_reduced)
J_lcp.block_until_ready()
times_lcp_j = []
for _ in range(3):
    t0 = time.perf_counter()
    J_lcp = lcp.jac_scaled(x_reduced)
    J_lcp.block_until_ready()
    times_lcp_j.append(time.perf_counter() - t0)
print(f"LCP jac_scaled:       {np.median(times_lcp_j)*1000:.1f}ms (median)")

# Time just the trust-region linear algebra (solve the least-squares subproblem)
# Simulate: given J and f, solve the TR subproblem
f_val = obj_fn2.compute_scaled(x)
f_val.block_until_ready()
J_val = obj_fn2.jac_scaled(x)
J_val.block_until_ready()
from desc.optimize.least_squares import trust_region_step_exact_cho
from jax.numpy.linalg import cho_factor
g = J_val.T @ f_val
B = J_val.T @ J_val
times_tr = []
for _ in range(5):
    t0 = time.perf_counter()
    try:
        step = trust_region_step_exact_cho(g, B, 1.0)
        if hasattr(step, 'block_until_ready'):
            step.block_until_ready()
    except Exception:
        pass
    times_tr.append(time.perf_counter() - t0)
print(f"TR subproblem solve:  {np.median(times_tr)*1000:.1f}ms (median)")
