# DESC GPU Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Profile 4 DESC optimization cases with Nsight Systems + JAX profiler to identify wall-time bottlenecks and per-objective Jacobian cost.

**Architecture:** Each case gets a self-contained Python script that (1) sets up a lightweight optimization, (2) runs it under JAX profiler annotations, and (3) measures per-objective Jacobian cost. A separate shell script wraps each with `nsys profile`. Two analysis scripts parse the outputs into a summary report.

**Tech Stack:** Python, JAX, NVIDIA Nsight Systems (`nsys`), SQLite3, DESC

---

### Task 1: Create shared profiling instrumentation module

**Files:**
- Create: `profiling/instrument.py`

This module provides monkey-patching utilities and timing helpers used by all 4 case scripts.

- [ ] **Step 1: Create `profiling/instrument.py`**

```python
"""Shared instrumentation for DESC GPU profiling.

Provides:
- monkey_patch_lsqtr(): wraps lsqtr fun/jac with timing + block_until_ready
- time_per_objective_jac(): measures each sub-objective's JVP cost
- setup_gpu(): standard GPU/JAX setup for Perlmutter
"""
import os
import time
import json
import numpy as np


def setup_gpu():
    """Configure GPU and JAX for profiling on Perlmutter."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

    from desc import set_device
    set_device("gpu")

    import jax
    cache_dir = os.path.join(os.environ.get("PSCRATCH", "/tmp"), "jax_cache")
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def monkey_patch_lsqtr():
    """Patch lsqtr to time fun/jac calls with block_until_ready.

    Returns (call_log, restore_fn).
    call_log: list of (label, seconds) tuples, populated during optimize().
    restore_fn: call to undo the monkey-patch.
    """
    from desc.optimize import least_squares as ls_mod

    call_log = []
    orig_lsqtr = ls_mod.lsqtr

    def patched_lsqtr(fun, x0, jac, **kwargs):
        call_log.clear()

        def timed_fun(x, *args):
            t0 = time.perf_counter()
            result = fun(x, *args)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            call_log.append(("fun", time.perf_counter() - t0))
            return result

        def timed_jac(x, *args):
            t0 = time.perf_counter()
            result = jac(x, *args)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            call_log.append(("jac", time.perf_counter() - t0))
            return result

        return orig_lsqtr(timed_fun, x0, timed_jac, **kwargs)

    ls_mod.lsqtr = patched_lsqtr

    def restore():
        ls_mod.lsqtr = orig_lsqtr

    return call_log, restore


def time_per_objective_jac(obj_fn, x, constants, n_warmup=1, n_trials=3):
    """Time each sub-objective's JVP contribution to the Jacobian.

    Mirrors the splitting logic in ObjectiveFunction._jvp_blocked:
    x is split by thing.dim_x, then per-objective slices are selected
    via _things_per_objective_idx.

    Parameters
    ----------
    obj_fn : ObjectiveFunction (already built)
    x : array, current parameter vector
    constants : list of dicts, from obj_fn.constants()
    n_warmup : int, warmup calls (JIT compilation)
    n_trials : int, timed calls (take median)

    Returns
    -------
    results : list of dicts with keys: name, dim_f, dim_x, median_ms, times_ms
    """
    import jax.numpy as jnp

    # Split x into per-thing arrays (same as _jvp_blocked)
    xs_splits = np.cumsum([t.dim_x for t in obj_fn.things])
    xs = jnp.split(jnp.asarray(x), xs_splits)

    results = []

    for k, obj in enumerate(obj_fn.objectives):
        # Which things does this objective use?
        thing_idx = obj_fn._things_per_objective_idx[k]
        xi = [xs[i] for i in thing_idx]

        # Build identity tangent vectors per thing (for JVP -> full Jacobian)
        vi = [jnp.eye(xii.shape[0]) for xii in xi]

        const_k = constants[k] if constants is not None else None

        times = []
        for trial in range(n_warmup + n_trials):
            t0 = time.perf_counter()
            J = obj.jvp_scaled(vi, xi, constants=const_k)
            if hasattr(J, "block_until_ready"):
                J.block_until_ready()
            elapsed = time.perf_counter() - t0
            if trial >= n_warmup:
                times.append(elapsed * 1000)  # ms

        total_dim_x = sum(xii.shape[0] for xii in xi)
        results.append({
            "name": getattr(obj, "name", obj.__class__.__name__),
            "dim_f": obj.dim_f,
            "dim_x": total_dim_x,
            "median_ms": float(np.median(times)),
            "times_ms": times,
        })

    return results


def print_lsqtr_breakdown(call_log, label=""):
    """Print a formatted breakdown of lsqtr timing."""
    fun_times = [t for tag, t in call_log if tag == "fun"]
    jac_times = [t for tag, t in call_log if tag == "jac"]
    total_fun = sum(fun_times)
    total_jac = sum(jac_times)
    total = total_fun + total_jac
    overhead = 0  # We don't know total wall from call_log alone

    print(f"\n{'=' * 60}")
    print(f"lsqtr breakdown {label}")
    print(f"{'=' * 60}")
    print(f"  fun calls: {len(fun_times)}, total: {total_fun:.2f}s")
    if fun_times:
        print(f"    mean: {np.mean(fun_times):.3f}s, "
              f"min: {np.min(fun_times):.3f}s, max: {np.max(fun_times):.3f}s")
    print(f"  jac calls: {len(jac_times)}, total: {total_jac:.2f}s")
    if jac_times:
        print(f"    mean: {np.mean(jac_times):.3f}s, "
              f"min: {np.min(jac_times):.3f}s, max: {np.max(jac_times):.3f}s")
    if total > 0:
        print(f"  fun%: {100*total_fun/total:.0f}%, jac%: {100*total_jac/total:.0f}%")


def print_objective_breakdown(results, label=""):
    """Print per-objective Jacobian cost table."""
    total = sum(r["median_ms"] for r in results)
    print(f"\n{'=' * 60}")
    print(f"Per-objective Jacobian cost {label}")
    print(f"{'=' * 60}")
    print(f"  {'Objective':<30s} {'dim_f':>6s} {'dim_x':>6s} "
          f"{'ms':>8s} {'%':>6s}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for r in sorted(results, key=lambda x: -x["median_ms"]):
        pct = 100 * r["median_ms"] / total if total > 0 else 0
        print(f"  {r['name']:<30s} {r['dim_f']:>6d} {r['dim_x']:>6d} "
              f"{r['median_ms']:>7.1f}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<30s} {'':>6s} {'':>6s} {total:>7.1f}  100.0%")


def save_results(results_dict, path):
    """Save profiling results to JSON."""

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2, default=convert)
    print(f"Saved: {path}")
```

- [ ] **Step 2: Verify file was created**

Run: `python -c "import sys; sys.path.insert(0, 'profiling'); import instrument; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add profiling/instrument.py
git commit -m "profiling: add shared instrumentation module"
```

---

### Task 2: Create Case 1 -- Coil optimization with lsq-auglag

**Files:**
- Create: `profiling/case_1_coil_auglag.py`

Adapts `finding_better_basins/run_baseline.py` config/setup at lightweight resolution with profiling instrumentation.

- [ ] **Step 1: Create `profiling/case_1_coil_auglag.py`**

```python
#!/usr/bin/env python
"""Case 1: Coil optimization with lsq-auglag (lightweight).

Profiles:
- lsqtr fun/jac timing via monkey-patch
- Per-objective Jacobian cost breakdown
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

# GPU setup must happen before any JAX/DESC imports
setup_gpu()

import numpy as np
# Import the finding_better_basins config machinery
sys.path.insert(0, os.path.expanduser("~/finding_better_basins"))
from configs import build_config, build_objectives, load_data, setup_jax

config = build_config(platform="perlmutter", lightweight=True, seed=0)
setup_jax(config)

# Monkey-patch lsqtr for timing
call_log, restore_lsqtr = monkey_patch_lsqtr()

# Load data and build problem
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

# --- Layer 2+3: Per-objective Jacobian breakdown ---
print("\nPer-objective Jacobian profiling...")
constants = obj_fn.constants
obj_results = time_per_objective_jac(obj_fn, x0, constants)
print_objective_breakdown(obj_results, label="(Case 1: coil auglag)")

# --- Layer 2: Run optimizer with instrumented lsqtr ---
print(f"\nRunning lsq-auglag (5 iterations)...")
opt = Optimizer("lsq-auglag")
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

print_lsqtr_breakdown(call_log, label="(Case 1: coil auglag)")
print(f"\nTotal wall time: {wall:.1f}s")

# Save results
save_results({
    "case": "case_1_coil_auglag",
    "wall_time_s": wall,
    "dofs": len(x0),
    "dim_f": obj_fn.dim_f,
    "call_log": call_log,
    "objective_breakdown": obj_results,
}, "profiling/results_case_1.json")

restore_lsqtr()
```

- [ ] **Step 2: Smoke test (import only, no GPU needed)**

Run: `python -c "import ast; ast.parse(open('profiling/case_1_coil_auglag.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add profiling/case_1_coil_auglag.py
git commit -m "profiling: add case 1 coil opt lsq-auglag"
```

---

### Task 3: Create Case 2 -- Coil optimization with lsq-exact

**Files:**
- Create: `profiling/case_2_coil_exact.py`

Same setup as Case 1, but uses `lsq-exact` optimizer instead.

- [ ] **Step 1: Create `profiling/case_2_coil_exact.py`**

```python
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
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('profiling/case_2_coil_exact.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add profiling/case_2_coil_exact.py
git commit -m "profiling: add case 2 coil opt lsq-exact"
```

---

### Task 4: Create Case 3 -- Free boundary equilibrium with proximal-lsq-exact

**Files:**
- Create: `profiling/case_3_free_boundary.py`

Adapts `dynamic-accessibility/.../free_boundary_half-slew_lower-res.py` at lightweight resolution. This case uses equilibrium DOFs (not coil DOFs) and `proximal-lsq-exact`.

- [ ] **Step 1: Create `profiling/case_3_free_boundary.py`**

```python
#!/usr/bin/env python
"""Case 3: Free boundary equilibrium with proximal-lsq-exact (lightweight).

Adapts the half-slew free-boundary script at reduced resolution.
Only runs the boundary optimization step (not bootstrap).
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

from desc.io import load
from desc.equilibrium import EquilibriaFamily
from desc.grid import LinearGrid
from desc.magnetic_fields import FourierCurrentPotentialField
from desc.objectives import (
    BoundaryError, FixAtomicNumber, FixCurrent, FixElectronDensity,
    FixElectronTemperature, FixIonTemperature, FixPsi, ForceBalance,
    ObjectiveFunction,
)
from desc.optimize import Optimizer

# --- Data paths (Perlmutter) ---
HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(
    HOME, "dynamic-accessibility/half_beta_half_slew_eq/"
    "second_free_boundary_proper-tf"
)

COIL_ENC = os.path.join(DATA_DIR, "data/midbeta/encircling_midbeta.h5")
COIL_SHP = os.path.join(DATA_DIR, "data/midbeta/shaping_midbeta.h5")
EQ_PATH = os.path.join(DATA_DIR, "data/midbeta/equil_G1635_DESC_fixed.h5")

# Load coils and equilibrium
print("Loading coils...")
encircling = load(COIL_ENC)
shaping = load(COIL_SHP)
print(f"Encircling: {len(encircling)} coils, Shaping: {len(shaping)} coils")

print("Loading equilibrium...")
eq0 = load(EQ_PATH)
if isinstance(eq0, EquilibriaFamily):
    eq0 = eq0[-1]
eq = eq0.copy()
eq.Psi = 0.925 * eq.Psi

# Convert surface for free-boundary
if not isinstance(eq.surface, FourierCurrentPotentialField):
    eq.surface = FourierCurrentPotentialField.from_surface(
        eq.surface, M_Phi=eq.M, N_Phi=eq.N
    )

print(f"Equilibrium: L={eq.L}, M={eq.M}, N={eq.N}, NFP={eq.NFP}")

# Monkey-patch lsqtr
call_log, restore_lsqtr = monkey_patch_lsqtr()

# Build lightweight objective (small grids)
grid_enc = LinearGrid(N=max(encircling[0].N, 32))
grid_shp = LinearGrid(N=max(shaping[0].N, 16))
grid_lcfs = LinearGrid(
    rho=np.array([1.0]), M=8, N=8, NFP=eq.NFP, sym=False
)

print("Building objectives...")
objective = ObjectiveFunction(
    BoundaryError(
        eq=eq, field=[encircling, shaping], target=0,
        source_grid=grid_lcfs, eval_grid=grid_lcfs,
        field_grid=[grid_enc, grid_shp], field_fixed=True,
    ),
    deriv_mode="batched",
    jac_chunk_size=8,
)

constraints = (
    FixAtomicNumber(eq=eq), FixCurrent(eq=eq), FixElectronDensity(eq=eq),
    FixElectronTemperature(eq=eq), FixIonTemperature(eq=eq),
    FixPsi(eq=eq), ForceBalance(eq=eq),
)

objective.build(verbose=0)

x0 = np.asarray(objective.x(eq)).copy()
print(f"DOFs: {len(x0)}, dim_f: {objective.dim_f}")

# Per-objective Jacobian breakdown
print("\nPer-objective Jacobian profiling...")
constants = objective.constants
obj_results = time_per_objective_jac(objective, x0, constants)
print_objective_breakdown(obj_results, label="(Case 3: free boundary)")

# Run optimizer
print(f"\nRunning proximal-lsq-exact (3 outer iterations)...")
opt = Optimizer("proximal-lsq-exact")
t0 = time.perf_counter()
[eq_out], result = opt.optimize(
    things=eq, objective=objective, constraints=constraints,
    x_scale="ess", maxiter=3, ftol=1e-4, gtol=1e-16,
    options={
        "solve_options": {
            "ftol": 1e-4, "xtol": 1e-6, "gtol": 1e-6, "maxiter": 5,
        },
    },
    verbose=2, copy=True,
)
wall = time.perf_counter() - t0

print_lsqtr_breakdown(call_log, label="(Case 3: free boundary)")
print(f"\nTotal wall time: {wall:.1f}s")

save_results({
    "case": "case_3_free_boundary",
    "wall_time_s": wall,
    "dofs": len(x0),
    "dim_f": objective.dim_f,
    "call_log": call_log,
    "objective_breakdown": obj_results,
}, "profiling/results_case_3.json")

restore_lsqtr()
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('profiling/case_3_free_boundary.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add profiling/case_3_free_boundary.py
git commit -m "profiling: add case 3 free boundary proximal-lsq-exact"
```

---

### Task 5: Create Case 4 -- Single-stage with proximal-lsq-exact

**Files:**
- Create: `profiling/case_4_single_stage.py`

Adapts the single-stage optimization script at `LIGHTWEIGHT=True`. This is the most complex case -- simultaneous equilibrium + coil current optimization with many objectives.

- [ ] **Step 1: Read the full single-stage script to understand objective setup**

Read: `/global/u2/t/telder/dynamic-accessibility/half_beta_half_slew_eq/slew_from_halfslew_eq-QS-divLeg/free_boundary_half-slew_eq-QS-divLeg-increasing-bdy-weight-and-slew-lower-res.py` (lines 100-250 for objective construction)

- [ ] **Step 2: Create `profiling/case_4_single_stage.py`**

This script must replicate the single-stage objective setup at lightweight resolution. Since the script is complex and has many data dependencies (divertor curves, equilibrium, coils), the implementation should:
1. Import directly from the source script's directory where possible
2. Use `LIGHTWEIGHT=True` settings from the script's CONFIG
3. Set `maxiter=3`, `maxiter_inner=3` for profiling only
4. Skip all plotting and diagnostics

The exact implementation depends on reading the full source script in Step 1 -- the objective list, constraints, and `things` must match the source but at reduced resolution. The engineer should adapt the objective construction section (approximately lines 100-250 of the source) into this profiling script, using the same pattern as cases 1-3.

Template structure:
```python
#!/usr/bin/env python
"""Case 4: Single-stage optimization with proximal-lsq-exact (lightweight).

Adapts the half-slew QS+divLeg single-stage script at LIGHTWEIGHT=True.
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
# ... (load eq, coils, build objectives matching source script's LIGHTWEIGHT config)
# ... (same instrumentation pattern as cases 1-3)
```

- [ ] **Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('profiling/case_4_single_stage.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add profiling/case_4_single_stage.py
git commit -m "profiling: add case 4 single-stage proximal-lsq-exact"
```

---

### Task 6: Create Nsight Systems analysis script

**Files:**
- Create: `profiling/analyze_nsight.py`

Parses `nsys export --type=sqlite` output to extract GPU utilization metrics.

- [ ] **Step 1: Create `profiling/analyze_nsight.py`**

```python
#!/usr/bin/env python
"""Parse Nsight Systems SQLite exports and print GPU utilization summary.

Usage:
    nsys export --type=sqlite profiling/case_N.nsys-rep
    python profiling/analyze_nsight.py profiling/case_N.sqlite

Or analyze all:
    python profiling/analyze_nsight.py profiling/case_*.sqlite
"""
import sqlite3
import sys
import os


def analyze_nsight_db(db_path):
    """Extract GPU profiling metrics from nsys SQLite export."""
    conn = sqlite3.connect(db_path)

    results = {"db_path": db_path}

    # Total trace duration (nanoseconds)
    try:
        row = conn.execute(
            "SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
        if row and row[0] is not None:
            kernel_start_ns, kernel_end_ns = row
            results["kernel_span_ns"] = kernel_end_ns - kernel_start_ns
        else:
            results["kernel_span_ns"] = 0
    except sqlite3.OperationalError:
        results["kernel_span_ns"] = 0

    # GPU kernel stats
    try:
        row = conn.execute(
            "SELECT COUNT(*), SUM(end - start), AVG(end - start), "
            "MIN(end - start), MAX(end - start) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
        results["kernel_count"] = row[0] or 0
        results["kernel_total_ns"] = row[1] or 0
        results["kernel_avg_ns"] = row[2] or 0
        results["kernel_min_ns"] = row[3] or 0
        results["kernel_max_ns"] = row[4] or 0
    except sqlite3.OperationalError:
        results["kernel_count"] = 0
        results["kernel_total_ns"] = 0

    # Top 10 kernels by cumulative time
    try:
        rows = conn.execute(
            "SELECT demangledName, COUNT(*) as cnt, "
            "SUM(end - start) as total_ns, AVG(end - start) as avg_ns "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "GROUP BY demangledName ORDER BY total_ns DESC LIMIT 10"
        ).fetchall()
        results["top_kernels"] = [
            {"name": r[0][:80], "count": r[1],
             "total_ms": r[2] / 1e6, "avg_ms": r[3] / 1e6}
            for r in rows
        ]
    except sqlite3.OperationalError:
        results["top_kernels"] = []

    # Memory copy stats
    try:
        for direction, table_suffix in [
            ("HtoD", "MEMCPY"), ("DtoH", "MEMCPY")
        ]:
            row = conn.execute(
                f"SELECT COUNT(*), SUM(end - start), SUM(bytes) "
                f"FROM CUPTI_ACTIVITY_KIND_{table_suffix} "
                f"WHERE copyKind LIKE '%{direction}%'"
            ).fetchone()
            results[f"memcpy_{direction}_count"] = row[0] or 0
            results[f"memcpy_{direction}_ns"] = row[1] or 0
            results[f"memcpy_{direction}_bytes"] = row[2] or 0
    except sqlite3.OperationalError:
        pass

    conn.close()
    return results


def print_nsight_summary(results):
    """Print formatted summary of Nsight results."""
    name = os.path.basename(results["db_path"]).replace(".sqlite", "")
    span_ms = results.get("kernel_span_ns", 0) / 1e6
    kernel_ms = results.get("kernel_total_ns", 0) / 1e6
    util = (kernel_ms / span_ms * 100) if span_ms > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"Nsight Systems: {name}")
    print(f"{'=' * 60}")
    print(f"  GPU kernel time:  {kernel_ms:>10.1f} ms")
    print(f"  Kernel span:      {span_ms:>10.1f} ms")
    print(f"  GPU utilization:  {util:>9.1f}%")
    print(f"  Kernel launches:  {results['kernel_count']:>10d}")
    if results["kernel_count"] > 0:
        print(f"  Avg kernel:       {results['kernel_avg_ns']/1e6:>10.3f} ms")
        print(f"  Min kernel:       {results['kernel_min_ns']/1e6:>10.3f} ms")
        print(f"  Max kernel:       {results['kernel_max_ns']/1e6:>10.3f} ms")

    for d in ["HtoD", "DtoH"]:
        cnt = results.get(f"memcpy_{d}_count", 0)
        ns = results.get(f"memcpy_{d}_ns", 0)
        mb = results.get(f"memcpy_{d}_bytes", 0) / 1e6
        if cnt > 0:
            print(f"  Memcpy {d}:     {cnt:>6d} calls, "
                  f"{ns/1e6:.1f} ms, {mb:.1f} MB")

    if results.get("top_kernels"):
        print(f"\n  Top kernels by cumulative time:")
        print(f"  {'Kernel':<60s} {'#':>5s} {'ms':>8s} {'avg_ms':>8s}")
        for k in results["top_kernels"]:
            print(f"  {k['name']:<60s} {k['count']:>5d} "
                  f"{k['total_ms']:>7.1f}  {k['avg_ms']:>7.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <nsys_export.sqlite> ...")
        sys.exit(1)

    for path in sys.argv[1:]:
        results = analyze_nsight_db(path)
        print_nsight_summary(results)
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('profiling/analyze_nsight.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add profiling/analyze_nsight.py
git commit -m "profiling: add Nsight Systems SQLite analysis script"
```

---

### Task 7: Create combined report script

**Files:**
- Create: `profiling/report.py`

Reads all `results_case_*.json` files and prints the combined profiling report.

- [ ] **Step 1: Create `profiling/report.py`**

```python
#!/usr/bin/env python
"""Print combined DESC GPU profiling report from JSON results.

Usage: python profiling/report.py
"""
import json
import glob
import os


def load_results(pattern="profiling/results_case_*.json"):
    """Load all result JSON files."""
    results = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            results.append(json.load(f))
    return results


def print_report(results):
    """Print the combined profiling report."""
    print("=" * 70)
    print("DESC GPU Profiling Report")
    print("=" * 70)

    for r in results:
        case = r["case"]
        wall = r["wall_time_s"]
        dofs = r["dofs"]
        dim_f = r["dim_f"]

        # Step breakdown from call_log
        call_log = r.get("call_log", [])
        fun_total = sum(t for tag, t in call_log if tag == "fun")
        jac_total = sum(t for tag, t in call_log if tag == "jac")
        other = wall - fun_total - jac_total

        print(f"\n{'─' * 70}")
        print(f"  {case}")
        print(f"  DOFs: {dofs}, Residuals: {dim_f}, Wall: {wall:.1f}s")
        if wall > 0:
            print(f"  Step breakdown: "
                  f"Jacobian {100*jac_total/wall:.0f}% | "
                  f"Obj eval {100*fun_total/wall:.0f}% | "
                  f"Other {100*other/wall:.0f}%")

        # Per-objective breakdown
        obj_results = r.get("objective_breakdown", [])
        if obj_results:
            total_ms = sum(o["median_ms"] for o in obj_results)
            print(f"  Jacobian by objective:")
            for o in sorted(obj_results, key=lambda x: -x["median_ms"]):
                pct = 100 * o["median_ms"] / total_ms if total_ms > 0 else 0
                print(f"    {o['name']:<30s} {o['median_ms']:>7.1f}ms "
                      f"({pct:>5.1f}%)  "
                      f"[{o['dim_f']}x{o['dim_x']}]")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    results = load_results()
    if not results:
        print("No results found. Run the case scripts first.")
    else:
        print_report(results)
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('profiling/report.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add profiling/report.py
git commit -m "profiling: add combined report script"
```

---

### Task 8: Create run_all.sh orchestration script

**Files:**
- Create: `profiling/run_all.sh`

Orchestrates running all 4 cases with both nsys and JAX profiling, then prints the report.

- [ ] **Step 1: Create `profiling/run_all.sh`**

```bash
#!/usr/bin/env bash
# Run all DESC GPU profiling cases.
#
# Usage (on Perlmutter GPU node):
#   bash profiling/run_all.sh
#
# Or run individual cases:
#   bash profiling/run_all.sh 1    # just case 1
#   bash profiling/run_all.sh 1 2  # cases 1 and 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
NSYS="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Systems/bin/nsys"

# Which cases to run
if [ $# -gt 0 ]; then
    CASES=("$@")
else
    CASES=(1 2 3 4)
fi

CASE_SCRIPTS=(
    [1]="profiling/case_1_coil_auglag.py"
    [2]="profiling/case_2_coil_exact.py"
    [3]="profiling/case_3_free_boundary.py"
    [4]="profiling/case_4_single_stage.py"
)

for c in "${CASES[@]}"; do
    script="${CASE_SCRIPTS[$c]}"
    echo ""
    echo "============================================================"
    echo "Running Case $c: $script"
    echo "============================================================"

    # Layer 1: Nsight Systems
    echo "--- nsys profile ---"
    "$NSYS" profile \
        --output="profiling/nsys_case_${c}" \
        --force-overwrite=true \
        --trace=cuda,nvtx \
        python "$script" 2>&1 | tee "profiling/case_${c}_stdout.log"

    # Export to SQLite
    echo "--- nsys export ---"
    "$NSYS" export \
        --type=sqlite \
        --output="profiling/nsys_case_${c}.sqlite" \
        "profiling/nsys_case_${c}.nsys-rep" 2>&1 || true

    echo "Case $c complete."
done

# Print reports
echo ""
echo "============================================================"
echo "Nsight Analysis"
echo "============================================================"
python profiling/analyze_nsight.py profiling/nsys_case_*.sqlite 2>/dev/null || \
    echo "(No nsys SQLite files found)"

echo ""
echo "============================================================"
echo "Combined Report"
echo "============================================================"
python profiling/report.py
```

- [ ] **Step 2: Make executable**

Run: `chmod +x profiling/run_all.sh`

- [ ] **Step 3: Commit**

```bash
git add profiling/run_all.sh
git commit -m "profiling: add run_all.sh orchestration script"
```

---

### Task 9: Run profiling on Perlmutter GPU node

**Files:**
- No new files

This task runs the actual profiling. Must be executed on a Perlmutter GPU node (not login node).

- [ ] **Step 1: Get a GPU node**

Either via interactive allocation or existing job:
```bash
salloc -N 1 -t 60 -C gpu -A <account> -q interactive
```

- [ ] **Step 2: Run all cases**

```bash
bash profiling/run_all.sh
```

Expected: Each case prints its timing breakdown and saves a `results_case_N.json`. The nsys exports produce `nsys_case_N.sqlite` files.

If a case fails (e.g., missing data file), note the error and run the remaining cases:
```bash
bash profiling/run_all.sh 1 2  # just the cases that work
```

- [ ] **Step 3: Review the combined report**

```bash
python profiling/report.py
```

Verify the output matches the expected format from the design spec. Check that the per-objective breakdowns are plausible (e.g., QuadraticFlux should be significant for coil cases).

- [ ] **Step 4: Review Nsight analysis**

```bash
python profiling/analyze_nsight.py profiling/nsys_case_*.sqlite
```

Check GPU utilization numbers. If utilization is very low (<30%), that itself is a major finding.

- [ ] **Step 5: Save stdout logs and commit results**

```bash
git add profiling/results_case_*.json profiling/case_*_stdout.log
git commit -m "profiling: add profiling results from GPU run"
```

Note: Do NOT commit `.nsys-rep` or `.sqlite` files (they are large binary files).

---

### Task 10: Interpret results and write recommendations

**Files:**
- No new files (output to stdout)

- [ ] **Step 1: Analyze the report output**

Based on the combined report, answer:

1. **Which case is Jacobian-dominated vs balanced?** If Jacobian is >80% of wall time, Jacobian speedups are high-impact. If it's <50%, look at the "Other" category (Python overhead, trust-region solve).

2. **Which objectives dominate Jacobian cost?** If QuadraticFlux/FieldNormalError are >70% of Jacobian time across cases, block-sparse structure only helps the remaining 30%. If per-coil objectives (CoilLength, CoilSetMinDistance) dominate, block-sparse is a big win.

3. **Is the GPU earning its keep?** If GPU utilization is <30%, kernel launch overhead is killing performance and CPU might be competitive. Recommend a CPU vs GPU comparison.

4. **Do proximal and non-proximal cases differ?** proximal-lsq-exact has additional overhead from the proximal sub-problem and constraint Jacobian. Compare the "Other" % between cases 1/2 (non-proximal) and 3/4 (proximal).

- [ ] **Step 2: Print recommendations to the user**

Summarize findings in plain text, referencing the specific numbers from the report.
