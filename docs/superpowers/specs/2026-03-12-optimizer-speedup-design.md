# Optimizer Jacobian Speedup: Reverse-Mode AD + Broyden Updates

**Date:** 2026-03-12
**Author:** Todd Elder + Claude
**Branch:** `tme/descspedup`
**Status:** Design approved, pending implementation

## Problem

Coil optimization with `lsq-exact` on Perlmutter A100 GPUs is bottlenecked by
Jacobian computation. Current per-iteration profiling (trefoil workload, full
coil count):

| Phase | Time | Share |
|---|---|---|
| Objective eval | 0.160s | 9.5% |
| **Jacobian** | **0.978s** | **57.8%** |
| TR solve (QR) | 0.553s | 32.7% |
| **Total/iter** | **1.691s** | |

The Jacobian is a (191, 972) matrix — 191 residual outputs, 972 reduced
parameters — but is computed using **forward-mode AD** (972 JVP passes).
Reverse-mode would need only 191 backward passes.

## Root Cause

`LinearConstraintProjection._jac` in `desc/optimize/_constraint_wrappers.py`
(line 386-390) unconditionally uses forward-mode JVP:

```python
def _jac(self, x_reduced, constants=None, op="scaled"):
    x = self.recover(x_reduced)
    v = self._feasible_tangents          # shape (4521, 972)
    df = getattr(self._objective, "jvp_" + op)(v.T, x, constants)  # 972 fwd passes
    return df.T
```

This is the single chokepoint for **all** least-squares optimizations that use
linear constraints (which is every `lsq-exact` call). The individual objective
auto-mode heuristic (which would correctly pick reverse-mode for this shape) is
bypassed because LCP calls `jvp_*` directly.

## Changes

### Change A: Reverse-Mode Jacobian in LinearConstraintProjection

**File:** `desc/optimize/_constraint_wrappers.py`

Replace the JVP-based `_jac` with reverse-mode AD when `dim_f < dim_x_reduced`:

```python
def _jac(self, x_reduced, constants=None, op="scaled"):
    x = self.recover(x_reduced)
    if self._ad_mode == "rev":
        fun = lambda x: getattr(self._objective, "compute_" + op)(x, constants)
        J_full = jacrev_chunked(fun, chunk_size=None)(x)  # None = no chunking
        return J_full @ self._feasible_tangents
    else:
        v = self._feasible_tangents
        df = getattr(self._objective, "jvp_" + op)(v.T, x, constants)
        return df.T
```

**Why `chunk_size=None`:** `LinearConstraintProjection` does not own a `_jac_chunk_size`
attribute — it would fall through to the inner objective's value via `__getattr__`,
but that chunk size is tuned for forward-mode column counts, not reverse-mode row
counts. With `dim_f=191` the full reverse pass is small enough to run unchunked.
If future workloads have very large `dim_f`, a dedicated `_rev_chunk_size` can be
added to LCP.

**AD mode selection** (at the end of `build()`, after line 159 where
`self._feasible_tangents` is assigned, before `self._built = True` at line 161):

```python
if self._objective.dim_f < self._dim_x_reduced:
    self._ad_mode = "rev"   # fewer outputs than inputs → reverse is faster
else:
    self._ad_mode = "fwd"   # more outputs than inputs → forward is faster
```

**Math:** By the chain rule, `df/dx_reduced = df/dx_full @ dx_full/dx_reduced`.
`_feasible_tangents` is `dx_full/dx_reduced` (shape `(dim_x_full, dim_x_reduced)`).
We compute `df/dx_full` via `jacrev_chunked` (shape `(dim_f, dim_x_full)`), then
multiply. The intermediate matrix is (191, 4521) = ~700 KB — negligible on GPU.

**Import needed:** `from desc.batching import jacrev_chunked`

**Expected impact:** Jacobian 2-5x faster (972 fwd passes → 191 rev passes).
Reverse passes are typically 1-3x more expensive per pass than forward, so
net speedup is ~2-5x depending on problem structure.

### Change B: Broyden Rank-1 Jacobian Updates in lsqtr

**File:** `desc/optimize/least_squares.py`

Instead of recomputing the full Jacobian every accepted step (line 391), use
Broyden rank-1 updates between periodic full recomputes:

```
J_{k+1} = J_k + (delta_f - J_k * delta_x) * delta_x^T / (delta_x^T * delta_x)
```

where `delta_f = f_new - f_prev` and `delta_x = x_new - x_prev`.

**Integration in the iteration loop** (line 385-409):

```python
if actual_reduction > 0:
    x_prev, f_prev = x, f
    x = x_new
    f = f_new
    cost = cost_new

    steps_since_full_jac += 1
    force_full_jac = (
        steps_since_full_jac >= jac_recompute_every
        or reduction_ratio < broyden_quality_threshold
        or broyden_force_recompute  # safety trigger from inner-loop retries
    )

    if force_full_jac:
        J = jac(x, *args).block_until_ready()
        njev += 1
        steps_since_full_jac = 0
        broyden_force_recompute = False
    else:
        delta_f = f - f_prev
        delta_x = x - x_prev
        J = J_unscaled + jnp.outer(       # J_unscaled kept from last full/Broyden
            delta_f - J_unscaled @ delta_x, delta_x
        ) / jnp.dot(delta_x, delta_x)

    # After computing J (full or Broyden), keep an unscaled copy BEFORE scaling:
    J_unscaled = J.copy()   # ~1.5 MB for (191, 972), negligible
    # Then apply scaling as before:
    J *= d
    J_h = J
    # NOTE: The existing `del J` (source line 409) must be REMOVED or moved
    # after `J_unscaled = J.copy()`. The `del J` was an optimization to free
    # memory early, but with Broyden we need the unscaled copy to survive.
    # Since `J_h = J` is an alias (not a copy), `del J` just drops the name
    # binding — `J_h` still holds the data. So removing `del J` is safe.
```

**Unscaled Jacobian bookkeeping:** The existing code scales `J` in place
(`J *= d`, then `J_h = J`), and `d` is recomputed each iteration. Attempting to
recover the unscaled Jacobian via `J_h * (1/d)` would use the **new** `d` to undo
scaling applied with the **old** `d`. Instead, we keep `J_unscaled` as a separate
copy before scaling is applied. Cost: one extra (191, 972) array = ~1.5 MB.

**Important:** The existing `del J` at source line 409 must be removed (or moved
after the `J_unscaled` copy). This is safe because `J_h = J` creates an alias, not
a copy — `del J` only drops the name, and `J_h` retains the array.

**New options exposed via `options` dict:**

| Option | Type | Default | Description |
|---|---|---|---|
| `jac_recompute_every` | int | 6 | Full Jacobian every N accepted steps. 1 = always (disables Broyden). |
| `broyden_quality_threshold` | float | 0.1 | Force full recompute if reduction_ratio drops below this. |
| `max_inner_retries` | int | 3 | Force full recompute if inner TR loop takes more than this many shrink steps. |

**New options must be popped before the unknown-options error check** (line 241).
Add `jac_recompute_every`, `broyden_quality_threshold`, and `max_inner_retries`
to the `options.pop()` block (around lines 188-239) so they don't trigger
`errorif(len(options) > 0, ...)`.

**Safety mechanisms:**
- If a Broyden-updated Jacobian leads to a rejected step (actual_reduction <= 0)
  and the inner TR loop exhausts `max_nfev`, set `broyden_force_recompute = True`
  to force a full recompute on the next accepted step.
- Additionally, if the inner TR loop takes more than `max_inner_retries` (default 3)
  shrink steps before finding a valid step, set `broyden_force_recompute = True`.
  This catches cases where the Broyden approximation is degrading but hasn't fully
  failed yet.
- `jac_recompute_every=1` recovers exact current behavior (no Broyden).
- First iteration always uses full Jacobian.
- `broyden_force_recompute` is initialized to `False` and reset after each full
  recompute.
- `x_prev` and `f_prev` must be initialized before the main loop to the initial
  `x` and `f` values (the values after the pre-loop Jacobian computation at
  line 183). This ensures the first Broyden update has valid reference values.

**Expected impact:** With K=6, only 1 in 6 iterations pays the full Jacobian
cost. Amortized Jacobian time drops by ~5-6x.

## Files Modified

| File | Change |
|---|---|
| `desc/optimize/_constraint_wrappers.py` | Reverse-mode `_jac`, auto AD mode selection |
| `desc/optimize/least_squares.py` | Broyden updates, new options, safety logic |

No changes to objectives, coils, magnetic fields, or any other code.

## Expected Performance

On the trefoil workload (Perlmutter A100, full coil count):

| Scenario | Jacobian/iter | Total/iter | vs Current |
|---|---|---|---|
| Current (fwd-mode, every iter) | 0.978s | 1.691s | baseline |
| After A (rev-mode, every iter) | ~0.2-0.5s | ~0.9-1.2s | 1.4-1.9x |
| After A+B (rev-mode, Broyden K=6) | ~0.05-0.1s amortized | ~0.5-0.8s | 2.1-3.4x |

## Testing and Validation

### Change A

1. **Numerical correctness:** Compare `_jac` output (fwd vs rev) on the trefoil
   workload. Must match to ~1e-10 relative error.
2. **Existing tests:** Run all optimizer tests (`tests/test_optimizer.py` and any
   tests that exercise `LinearConstraintProjection`).
3. **Profile on PM:** Measure actual Jacobian speedup with profiler Phase 4.

### Change B

1. **Convergence:** Run 20-iteration trefoil optimization. Compare final cost
   and iteration count between `jac_recompute_every=1` (baseline) and
   `jac_recompute_every=6`.
2. **Edge cases:** Verify rejected-step recompute triggers correctly. Verify
   `jac_recompute_every=1` produces identical results to current code.
3. **Wall-clock time:** Total optimization time is the real metric.

### Integration

1. Run `profile_desc_opt.py` on PM with both changes active.
2. Run `debug_coilset.py` to verify coil BS evaluation is unaffected.
3. Run full `pytest tests/test_coils.py` suite.

## Risks

| Risk | Mitigation |
|---|---|
| Reverse-mode more memory per pass | Intermediate (191, 4521) is ~700 KB — negligible |
| Broyden updates degrade convergence | Quality threshold + periodic full recompute + `jac_recompute_every=1` escape hatch |
| Reverse-mode slower for wide Jacobians | Auto mode selection: only uses rev when `dim_f < dim_x_reduced` |
| Broyden update numerics (division by small step) | Step is bounded by trust region; degenerate case triggers full recompute via quality threshold |
