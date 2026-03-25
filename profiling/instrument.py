"""Shared profiling instrumentation for DESC optimization benchmarking.

Provides GPU setup, lsqtr monkey-patching, per-objective JVP timing,
and result reporting/saving utilities.
"""

import json
import os
import time

import numpy as np


def setup_gpu():
    """Configure GPU/JAX for profiling on Perlmutter."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    from desc import set_device

    set_device("gpu")
    import jax

    cache_dir = os.path.join(os.environ.get("PSCRATCH", "/tmp"), "jax_cache")
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def monkey_patch_lsqtr():
    """Wrap DESC's lsqtr fun/jac args with timing and block_until_ready.

    Returns
    -------
    call_log : list of (str, float)
        List of ("fun"|"jac", seconds) tuples appended during optimization.
    restore_fn : callable
        Call to restore the original lsqtr function.
    """
    import jax
    import desc.optimize.least_squares as _ls_mod

    _original_lsqtr = _ls_mod.lsqtr
    call_log = []

    def _timed(label, fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            # Ensure GPU computation is complete before stopping timer
            jax.block_until_ready(result)
            elapsed = time.perf_counter() - t0
            call_log.append((label, elapsed))
            return result

        return wrapper

    def _patched_lsqtr(fun, x0, jac, *args, **kwargs):
        timed_fun = _timed("fun", fun)
        timed_jac = _timed("jac", jac)
        return _original_lsqtr(timed_fun, x0, timed_jac, *args, **kwargs)

    _ls_mod.lsqtr = _patched_lsqtr

    def restore_fn():
        _ls_mod.lsqtr = _original_lsqtr

    return call_log, restore_fn


def time_per_objective_jac(obj_fn, x, constants, n_warmup=1, n_trials=3):
    """Measure each sub-objective's JVP cost.

    Parameters
    ----------
    obj_fn : desc ObjectiveFunction
        The composite objective whose sub-objectives will be timed.
    x : array-like
        The parameter vector at which to evaluate JVPs.
    constants : list of dict
        obj_fn.constants — one dict per sub-objective.
    n_warmup : int
        Number of warmup calls before timing (triggers JIT compilation).
    n_trials : int
        Number of timed trials; median is reported.

    Returns
    -------
    results : list of dict
        One dict per objective with keys:
        - "name": str
        - "median_s": float (median wall time in seconds)
        - "times_s": list of float (all trial times)
    """
    import jax
    import jax.numpy as jnp

    xs_splits = np.cumsum([t.dim_x for t in obj_fn.things])
    xs = jnp.split(jnp.asarray(x), xs_splits)

    results = []
    for k, obj in enumerate(obj_fn.objectives):
        thing_idx = obj_fn._things_per_objective_idx[k]
        xi = [xs[i] for i in thing_idx]
        vi = [jnp.eye(xii.shape[0]) for xii in xi]
        const_k = constants[k]

        # Warmup to trigger JIT compilation
        for _ in range(n_warmup):
            out = obj.jvp_scaled(vi, xi, constants=const_k)
            jax.block_until_ready(out)

        # Timed trials
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            out = obj.jvp_scaled(vi, xi, constants=const_k)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)

        results.append(
            {
                "name": type(obj).__name__,
                "median_s": float(np.median(times)),
                "times_s": [float(t) for t in times],
            }
        )

    return results


def print_lsqtr_breakdown(call_log, label=""):
    """Print a formatted breakdown of lsqtr fun/jac timing.

    Parameters
    ----------
    call_log : list of ("fun"|"jac", float)
        Timing log from monkey_patch_lsqtr.
    label : str
        Optional label printed as a header.
    """
    if label:
        print(f"\n=== lsqtr timing breakdown: {label} ===")
    else:
        print("\n=== lsqtr timing breakdown ===")

    fun_times = [t for name, t in call_log if name == "fun"]
    jac_times = [t for name, t in call_log if name == "jac"]

    total_fun = sum(fun_times)
    total_jac = sum(jac_times)
    total = total_fun + total_jac

    n_fun = len(fun_times)
    n_jac = len(jac_times)

    print(f"  fun calls : {n_fun:4d}   total={total_fun:8.3f}s", end="")
    if n_fun > 0:
        print(f"   mean={total_fun/n_fun:.4f}s   median={np.median(fun_times):.4f}s")
    else:
        print()

    print(f"  jac calls : {n_jac:4d}   total={total_jac:8.3f}s", end="")
    if n_jac > 0:
        print(f"   mean={total_jac/n_jac:.4f}s   median={np.median(jac_times):.4f}s")
    else:
        print()

    print(f"  total     :        {total:8.3f}s")
    if total > 0:
        print(
            f"  jac share : {100*total_jac/total:.1f}%   fun share: {100*total_fun/total:.1f}%"
        )
    print()


def print_objective_breakdown(results, label=""):
    """Print a per-objective Jacobian cost table.

    Parameters
    ----------
    results : list of dict
        Output from time_per_objective_jac.
    label : str
        Optional label printed as a header.
    """
    if label:
        print(f"\n=== Per-objective JVP cost: {label} ===")
    else:
        print("\n=== Per-objective JVP cost ===")

    if not results:
        print("  (no results)")
        return

    total_median = sum(r["median_s"] for r in results)
    name_width = max(len(r["name"]) for r in results)
    name_width = max(name_width, 20)

    header = f"  {'Objective':<{name_width}}  {'median (s)':>10}  {'share':>7}"
    print(header)
    print("  " + "-" * (name_width + 22))

    for r in results:
        share = 100.0 * r["median_s"] / total_median if total_median > 0 else 0.0
        print(f"  {r['name']:<{name_width}}  {r['median_s']:>10.4f}  {share:>6.1f}%")

    print("  " + "-" * (name_width + 22))
    print(f"  {'TOTAL':<{name_width}}  {total_median:>10.4f}  {'100.0%':>7}")
    print()


def save_results(results_dict, path):
    """Save profiling results to a JSON file.

    Parameters
    ----------
    results_dict : dict
        Arbitrary dict of profiling results (must be JSON-serialisable).
    path : str
        File path to write; parent directories are created if needed.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results_dict, fh, indent=2)
    print(f"Results saved to {path}")
