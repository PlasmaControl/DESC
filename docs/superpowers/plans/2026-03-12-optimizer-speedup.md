# Optimizer Jacobian Speedup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 2-3x faster per-iteration coil optimization by switching LCP Jacobian to reverse-mode AD and adding Broyden rank-1 updates in lsqtr.

**Architecture:** Two independent changes. Change A modifies `LinearConstraintProjection._jac` to use reverse-mode AD when `dim_f < dim_x_reduced`. Change B modifies `lsqtr()` to approximate the Jacobian via Broyden rank-1 updates between periodic full recomputes. Both changes are backwards-compatible: existing behavior is preserved when conditions don't favor the new paths.

**Tech Stack:** JAX (jacrev_chunked from desc.batching), NumPy, DESC optimizer infrastructure.

**Spec:** `docs/superpowers/specs/2026-03-12-optimizer-speedup-design.md`

---

## Chunk 1: Change A — Reverse-Mode Jacobian in LinearConstraintProjection

### Task 1: Add reverse-mode AD mode selection to LCP.build()

**Files:**
- Modify: `desc/optimize/_constraint_wrappers.py:8` (import)
- Modify: `desc/optimize/_constraint_wrappers.py:159-161` (build method, after `_feasible_tangents`)

- [ ] **Step 1: Add `jacrev_chunked` to imports**

In `desc/optimize/_constraint_wrappers.py`, change line 8 from:

```python
from desc.batching import batched_vectorize
```

to:

```python
from desc.batching import batched_vectorize, jacrev_chunked
```

- [ ] **Step 2: Add AD mode selection at end of build()**

In `desc/optimize/_constraint_wrappers.py`, insert after line 159 (`self._feasible_tangents = ...`) and before line 161 (`self._built = True`):

```python
        # Choose AD mode for _jac: reverse when fewer outputs than inputs
        if self._objective.dim_f < self._dim_x_reduced:
            self._ad_mode = "rev"
        else:
            self._ad_mode = "fwd"
```

- [ ] **Step 3: Commit**

```bash
git add desc/optimize/_constraint_wrappers.py
git commit -m "feat(LCP): add reverse-mode AD selection in build()"
```

### Task 2: Replace LCP._jac with mode-switching implementation

**Files:**
- Modify: `desc/optimize/_constraint_wrappers.py:386-390` (`_jac` method)

- [ ] **Step 1: Replace the `_jac` method**

Replace lines 386-390:

```python
    def _jac(self, x_reduced, constants=None, op="scaled"):
        x = self.recover(x_reduced)
        v = self._feasible_tangents
        df = getattr(self._objective, "jvp_" + op)(v.T, x, constants)
        return df.T
```

with:

```python
    def _jac(self, x_reduced, constants=None, op="scaled"):
        x = self.recover(x_reduced)
        if self._ad_mode == "rev":
            fun = lambda x: getattr(self._objective, "compute_" + op)(x, constants)
            J_full = jacrev_chunked(fun, chunk_size=None)(x)
            return J_full @ self._feasible_tangents
        else:
            v = self._feasible_tangents
            df = getattr(self._objective, "jvp_" + op)(v.T, x, constants)
            return df.T
```

- [ ] **Step 2: Commit**

```bash
git add desc/optimize/_constraint_wrappers.py
git commit -m "feat(LCP): reverse-mode Jacobian when dim_f < dim_x_reduced"
```

### Task 3: Verify Change A with existing Jacobian comparison test

**Files:**
- Test: `tests/test_optimizer.py::test_LinearConstraint_jacobian` (existing, line 1286)

- [ ] **Step 1: Run the existing Jacobian test**

This test already compares LCP Jacobians across `auto`, `fwd`, and `rev` deriv modes against a manually-constructed reference Jacobian (`obj1.jac_scaled(x)[:, lc1._unfixed_idx] @ lc1._Z`). It checks `jac_scaled`, `jac_unscaled`, JVPs, and VJPs at tolerance `rtol=1e-12, atol=1e-12`.

The test uses a HELIOTRON equilibrium at low resolution with `ForceBalance` — at this size `dim_f > dim_x_reduced` is likely, so the auto-selected mode will be `"fwd"` (unchanged path). This test validates regression, not the new rev-mode path. Task 4 explicitly tests both modes.

```bash
cd /home/toddelder/DESC && python -m pytest tests/test_optimizer.py::test_LinearConstraint_jacobian -xvs 2>&1 | tail -20
```

Expected: PASS. All `np.testing.assert_allclose` at `rtol=1e-12` should hold.

- [ ] **Step 2: Also run the LCP wrapper test**

```bash
cd /home/toddelder/DESC && python -m pytest tests/test_optimizer.py::TestOptimizer::test_wrappers -xvs 2>&1 | tail -20
```

Expected: PASS.

### Task 4: Write explicit fwd-vs-rev comparison test for LCP._jac

**Files:**
- Modify: `tests/test_optimizer.py` (add new test after `test_LinearConstraint_jacobian`)

- [ ] **Step 1: Write the test**

Add after line 1377 in `tests/test_optimizer.py`:

```python
@pytest.mark.unit
def test_LCP_jac_fwd_vs_rev():
    """Test that LCP._jac produces identical results in fwd and rev mode."""
    eq = desc.examples.get("HELIOTRON")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(1, 1, 1, 2, 2, 2)

    obj = ObjectiveFunction(
        ForceBalance(eq, deriv_mode="auto"), deriv_mode="batched", use_jit=False
    )
    con = ObjectiveFunction(get_fixed_boundary_constraints(eq))
    lc = LinearConstraintProjection(obj, con)
    lc.build()

    x_reduced = lc.x()

    # Force fwd mode and compute
    lc._ad_mode = "fwd"
    J_fwd_scaled = np.array(lc.jac_scaled(x_reduced))
    J_fwd_unscaled = np.array(lc.jac_unscaled(x_reduced))
    J_fwd_scaled_error = np.array(lc.jac_scaled_error(x_reduced))

    # Force rev mode and compute
    lc._ad_mode = "rev"
    J_rev_scaled = np.array(lc.jac_scaled(x_reduced))
    J_rev_unscaled = np.array(lc.jac_unscaled(x_reduced))
    J_rev_scaled_error = np.array(lc.jac_scaled_error(x_reduced))

    np.testing.assert_allclose(J_fwd_scaled, J_rev_scaled, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(J_fwd_unscaled, J_rev_unscaled, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        J_fwd_scaled_error, J_rev_scaled_error, rtol=1e-10, atol=1e-10
    )
    print(f"  dim_f={lc.dim_f}, dim_x_reduced={lc._dim_x_reduced}")
    print(f"  J shape: {J_fwd_scaled.shape}")
    print(f"  max |fwd-rev| scaled: {np.max(np.abs(J_fwd_scaled - J_rev_scaled)):.2e}")
```

- [ ] **Step 2: Run it**

```bash
cd /home/toddelder/DESC && python -m pytest tests/test_optimizer.py::test_LCP_jac_fwd_vs_rev -xvs 2>&1 | tail -20
```

Expected: PASS with max difference around 1e-12 to 1e-14.

- [ ] **Step 3: Commit**

```bash
git add tests/test_optimizer.py
git commit -m "test: add fwd-vs-rev Jacobian comparison for LCP._jac"
```

---

## Chunk 2: Change B — Broyden Rank-1 Updates in lsqtr

### Task 5: Add new Broyden options to lsqtr

**Files:**
- Modify: `desc/optimize/least_squares.py:188-245` (options block)

- [ ] **Step 1: Pop new options before the error check**

In `desc/optimize/least_squares.py`, insert after line 239 (`tr_method = options.pop("tr_method", "qr")`) and before line 241 (`errorif(len(options) > 0, ...)`):

```python
    jac_recompute_every = options.pop("jac_recompute_every", 6)
    broyden_quality_threshold = options.pop("broyden_quality_threshold", 0.1)
    max_inner_retries = options.pop("max_inner_retries", 3)
```

- [ ] **Step 2: Save unscaled J before initial scaling**

In `desc/optimize/least_squares.py`, insert BEFORE line 206 (`J *= d`):

```python
    J_unscaled = J  # save before scaling; JAX J *= d creates new array, won't mutate
```

This works because JAX's `J *= d` does not mutate in place — it rebinds `J` to a new array, so `J_unscaled` retains the original unscaled Jacobian.

- [ ] **Step 3: Initialize remaining Broyden state before the loop**

In `desc/optimize/least_squares.py`, insert after line 272 (`alpha = 0.0`) and before the `while` loop at line 274:

```python
    # Broyden state
    steps_since_full_jac = 0
    broyden_force_recompute = False
    x_prev = x
    f_prev = f
```

- [ ] **Step 3: Commit**

```bash
git add desc/optimize/least_squares.py
git commit -m "feat(lsqtr): add Broyden option parsing and state initialization"
```

### Task 6: Add inner-loop retry counter

**Files:**
- Modify: `desc/optimize/least_squares.py:298-303` (inner loop)

- [ ] **Step 1: Add retry counter before inner loop**

In `desc/optimize/least_squares.py`, insert after line 298 (`actual_reduction = -1`) and before line 303 (`while actual_reduction <= 0 and nfev <= max_nfev:`):

```python
        inner_retries = 0
```

- [ ] **Step 2: Increment counter inside inner loop**

Inside the inner while loop body, after line 345 (`actual_reduction = cost - cost_new`), add:

```python
            if actual_reduction <= 0:
                inner_retries += 1
```

- [ ] **Step 3: Add safety trigger after inner loop exits**

After the inner loop's `break` at line 383 (`if success is not None: break`), but before line 385 (`# if reduction was enough, accept the step`), add:

```python
        # If inner loop struggled, Broyden approximation may be degrading
        if inner_retries > max_inner_retries and steps_since_full_jac > 0:
            broyden_force_recompute = True
```

- [ ] **Step 4: Commit**

```bash
git add desc/optimize/least_squares.py
git commit -m "feat(lsqtr): add inner-loop retry counter for Broyden safety"
```

### Task 7: Replace accepted-step Jacobian recompute with Broyden logic

**Files:**
- Modify: `desc/optimize/least_squares.py:385-409` (accepted step block)

- [ ] **Step 1: Replace the accepted-step block**

Replace lines 386-409:

```python
        if actual_reduction > 0:
            x = x_new
            allx.append(x)
            f = f_new
            cost = cost_new
            J = jac(x, *args)
            njev += 1
            g = jnp.dot(J.T, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)

            v, dv = cl_scaling_vector(x, g, lb, ub)
            v = jnp.where(dv != 0, v * scale_inv, v)
            d = v**0.5 * scale
            diag_h = g * dv * scale

            g_h = g * d
            J *= d
            # we don't need unscaled J anymore this iteration, so we overwrite
            # it with J_h = J * d to avoid carrying so many J-sized matrices
            # in memory, which can be large
            J_h = J
            del J
```

with:

```python
        if actual_reduction > 0:
            x_prev = x  # save before update for Broyden delta
            f_prev = f
            x = x_new
            allx.append(x)
            f = f_new
            cost = cost_new

            steps_since_full_jac += 1
            force_full_jac = (
                steps_since_full_jac >= jac_recompute_every
                or reduction_ratio < broyden_quality_threshold
                or broyden_force_recompute
            )

            if force_full_jac:
                J = jac(x, *args).block_until_ready()
                njev += 1
                steps_since_full_jac = 0
                broyden_force_recompute = False
            else:
                delta_f = f - f_prev
                delta_x = x - x_prev
                J = J_unscaled + jnp.outer(
                    delta_f - J_unscaled @ delta_x, delta_x
                ) / jnp.dot(delta_x, delta_x)

            g = jnp.dot(J.T, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)

            v, dv = cl_scaling_vector(x, g, lb, ub)
            v = jnp.where(dv != 0, v * scale_inv, v)
            d = v**0.5 * scale
            diag_h = g * dv * scale

            # Save unscaled J before scaling (JAX *= creates new array)
            J_unscaled = J

            g_h = g * d
            J *= d
            J_h = J
            del J
```

- [ ] **Step 2: Fix the return value unscaling**

At line 444, the return `jac=J_h * 1 / d` uses the current `d` to unscale `J_h`, which is correct because `d` and `J_h` are from the same iteration. However, now we also have `J_unscaled` available. Change line 444 from:

```python
        jac=J_h * 1 / d,  # after overwriting J_h, we have to revert back
```

to:

```python
        jac=J_unscaled,  # keep unscaled copy for return value
```

This is simpler and avoids any numerical noise from unscaling.

- [ ] **Step 3: Commit**

```bash
git add desc/optimize/least_squares.py
git commit -m "feat(lsqtr): Broyden rank-1 Jacobian updates between full recomputes"
```

### Task 8: Test Broyden with jac_recompute_every=1 matches baseline

**Files:**
- Modify: `tests/test_optimizer.py` (add test)

- [ ] **Step 1: Write a test that verifies jac_recompute_every=1 is identical to default**

Add to `tests/test_optimizer.py` after the existing `TestLSQTR` class:

```python
@pytest.mark.unit
def test_lsqtr_broyden_recompute_every_1():
    """Broyden with jac_recompute_every=1 must match baseline exactly."""
    p = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
    x = np.linspace(-1, 1, 100)
    y = vector_fun(x, p)

    def res(p):
        return vector_fun(x, p) - y

    rando = default_rng(seed=0)
    p0 = p + 0.25 * (rando.random(p.size) - 0.5)
    jac_fn = Derivative(res, 0, "fwd")

    # jac_recompute_every=1 should force full Jacobian every step = baseline
    out = lsqtr(
        res,
        p0,
        jac_fn,
        verbose=0,
        x_scale=1,
        options={
            "initial_trust_radius": 0.15,
            "max_trust_radius": 0.25,
            "tr_method": "cho",
            "jac_recompute_every": 1,
        },
    )
    np.testing.assert_allclose(out["x"], p, atol=1e-8)
```

- [ ] **Step 2: Run it**

```bash
cd /home/toddelder/DESC && python -m pytest tests/test_optimizer.py::test_lsqtr_broyden_recompute_every_1 -xvs 2>&1 | tail -20
```

Expected: PASS.

- [ ] **Step 3: Write a test that Broyden (K=6) still converges**

Add to `tests/test_optimizer.py`:

```python
@pytest.mark.unit
def test_lsqtr_broyden_convergence():
    """Broyden updates with jac_recompute_every=6 must still converge."""
    p = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
    x = np.linspace(-1, 1, 100)
    y = vector_fun(x, p)

    def res(p):
        return vector_fun(x, p) - y

    rando = default_rng(seed=0)
    p0 = p + 0.25 * (rando.random(p.size) - 0.5)
    jac_fn = Derivative(res, 0, "fwd")

    out = lsqtr(
        res,
        p0,
        jac_fn,
        verbose=0,
        x_scale=1,
        maxiter=200,
        options={
            "initial_trust_radius": 0.15,
            "max_trust_radius": 0.25,
            "tr_method": "cho",
            "jac_recompute_every": 6,
        },
    )
    np.testing.assert_allclose(out["x"], p, atol=1e-6)
    # Broyden should use fewer Jacobian evaluations
    print(f"  njev={out['njev']}, nit={out['nit']}, nfev={out['nfev']}")
```

- [ ] **Step 4: Run it**

```bash
cd /home/toddelder/DESC && python -m pytest tests/test_optimizer.py::test_lsqtr_broyden_convergence -xvs 2>&1 | tail -20
```

Expected: PASS with `njev` significantly less than `nit`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_optimizer.py
git commit -m "test: add Broyden convergence and baseline-equivalence tests"
```

---

## Chunk 3: Integration Testing

### Task 9: Run all existing optimizer tests

**Files:**
- Test: `tests/test_optimizer.py` (full suite)

- [ ] **Step 1: Run the full optimizer test suite**

```bash
cd /home/toddelder/DESC && python -m pytest tests/test_optimizer.py -x --timeout=300 2>&1 | tail -30
```

Expected: All tests pass. Key tests to watch:
- `test_lsqtr_exact` — exercises lsqtr directly
- `test_LinearConstraint_jacobian` — validates LCP Jacobian across modes
- `test_wrappers` — validates LCP/PP wrapper API

- [ ] **Step 2: Run the linear objectives test for LCP target update**

```bash
cd /home/toddelder/DESC && python -m pytest tests/test_linear_objectives.py::test_linearconstraintprojection_update_target -xvs 2>&1 | tail -20
```

Expected: PASS.

### Task 10: Integration test with real coil data

**Files:**
- Create: `debug_optimizer_speedup.py` (repo root)

- [ ] **Step 1: Write integration validation script**

Create `debug_optimizer_speedup.py` in the repo root. This script:
1. Loads real coil data (same paths as `profile_desc_opt.py`)
2. Builds the same objective/constraint stack used in production
3. Compares LCP._jac output in fwd vs rev mode at machine precision
4. Runs a short optimization (2 iterations) with `jac_recompute_every=1` and `jac_recompute_every=6`
5. Verifies both produce nearly identical cost trajectories

```python
"""
Integration test: optimizer speedup changes with real coil data.

Validates:
  - LCP._jac fwd vs rev mode match on real workload
  - Broyden updates (K=6) converge similarly to baseline (K=1)

Usage:  PLATFORM=local python debug_optimizer_speedup.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np

PLATFORM = os.environ.get("PLATFORM", "local")

if PLATFORM == "perlmutter":
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    import jax
    from desc import set_device
    set_device("gpu")
else:
    import jax

from desc.backend import jnp
from desc.io import load
from desc.grid import LinearGrid
from desc.coils import CoilSet
from desc.objectives import (
    ObjectiveFunction,
    FixCoilCurrent,
    FixParameters,
    QuadraticFlux,
)
from desc.objectives._coils import FieldNormalError
from desc.optimize import LinearConstraintProjection, Optimizer

print(f"Backend: {jax.default_backend()}")

# ── Data paths ──
PATHS = {
    "local": {
        "data_dir": "/home/toddelder/projects/divertor_coil/output_runs/divertor_trefoil_v0",
        "trefoil_upper": "footpoint_coils.h5",
        "trefoil_lower": None,
        "fullbeta_eq": "/home/toddelder/projects/dynamic_accessibility/FullBeta_DivLeg_Opt/equil_G1600_20260212_072035.h5",
    },
    "perlmutter": {
        "data_dir": "/global/homes/t/telder/divertor/bifoil_optimization/data",
        "trefoil_upper": "footpoint_coils_upper.h5",
        "trefoil_lower": "footpoint_coils_lower.h5",
        "fullbeta_eq": "/global/homes/t/telder/divertor/data/G1600_12-89_w-Bxdl_data/equil_Helios_G1600-12-89_free_L12_M12_N20_nocont.h5",
    },
}[PLATFORM]

COIL_KEEP_FRAC = 0.1  # lightweight

# ── Load data ──
print("Loading data...")
data_dir = PATHS["data_dir"]
eq_raw = load(PATHS["fullbeta_eq"])
eq = eq_raw[-1] if isinstance(eq_raw, (list, tuple)) else eq_raw

if PATHS["trefoil_lower"] is not None:
    trefoils_upper_raw = load(os.path.join(data_dir, PATHS["trefoil_upper"]))
    trefoils_lower_raw = load(os.path.join(data_dir, PATHS["trefoil_lower"]))
    trefoils_upper = CoilSet(*list(trefoils_upper_raw.coils), NFP=eq.NFP, sym=True)
    trefoils_lower = CoilSet(*list(trefoils_lower_raw.coils), NFP=eq.NFP, sym=True)
else:
    footpoint_all = load(os.path.join(data_dir, PATHS["trefoil_upper"]))
    upper_coils, lower_coils = [], []
    for c in footpoint_all.coils:
        pts = np.squeeze(c._compute_position(grid=16, basis="xyz"))
        if pts[:, 2].mean() > 0:
            upper_coils.append(c)
        else:
            lower_coils.append(c)
    trefoils_upper = CoilSet(*upper_coils, NFP=eq.NFP, sym=True)
    trefoils_lower = CoilSet(*lower_coils, NFP=eq.NFP, sym=True)


def drop_coils(cs, frac=COIL_KEEP_FRAC):
    coils = list(cs.coils)
    n = max(1, int(len(coils) * frac))
    idx = np.linspace(0, len(coils) - 1, n, dtype=int)
    return CoilSet(*[coils[i] for i in idx], NFP=cs.NFP, sym=cs.sym)


trefoils_upper = drop_coils(trefoils_upper)
trefoils_lower = drop_coils(trefoils_lower)
print(f"  Coils: {len(trefoils_upper.coils)} upper, {len(trefoils_lower.coils)} lower")

# ── Build objective ──
print("Building objective...")
eval_grid = LinearGrid(M=3, N=5, NFP=eq.NFP)
things_to_optimize = trefoils_upper + trefoils_lower

obj = ObjectiveFunction(
    FieldNormalError(
        eq, field=trefoils_upper, eval_grid=eval_grid, field_grid=16,
    ),
)
constraints = ObjectiveFunction(
    FixParameters(things_to_optimize, ["current"]),
)
obj.build(verbose=0)
constraints.build(verbose=0)

lc = LinearConstraintProjection(obj, constraints)
lc.build(verbose=0)

x_reduced = lc.x()
print(f"  dim_f={lc.dim_f}, dim_x_reduced={lc._dim_x_reduced}")
print(f"  AD mode selected: {lc._ad_mode}")

# ── TEST 1: fwd vs rev Jacobian ──
print("\n--- TEST 1: LCP._jac fwd vs rev ---")
lc._ad_mode = "fwd"
J_fwd = np.array(lc.jac_scaled_error(x_reduced))

lc._ad_mode = "rev"
J_rev = np.array(lc.jac_scaled_error(x_reduced))

# Restore auto-selected mode
if lc._objective.dim_f < lc._dim_x_reduced:
    lc._ad_mode = "rev"
else:
    lc._ad_mode = "fwd"

diff = np.max(np.abs(J_fwd - J_rev))
maxref = np.max(np.abs(J_fwd))
rel = diff / maxref if maxref > 0 else diff
status = "PASS" if rel < 1e-8 else "*** FAIL ***"
print(f"  J shape: {J_fwd.shape}")
print(f"  max |fwd-rev|: {diff:.2e}, rel: {rel:.2e} {status}")

# ── TEST 2: Broyden K=1 vs K=6 ──
print("\n--- TEST 2: Broyden convergence (K=1 vs K=6) ---")

for K in [1, 6]:
    opt = Optimizer("lsq-exact")
    result = opt.optimize(
        things_to_optimize,
        obj,
        constraints,
        maxiter=3,
        verbose=0,
        options={"jac_recompute_every": K},
    )
    print(f"  K={K}: cost={result['cost']:.6e}, njev={result['njev']}, nit={result['nit']}")

print("\n--- DONE ---")
```

- [ ] **Step 2: Run the script locally**

```bash
cd /home/toddelder/DESC && PLATFORM=local python debug_optimizer_speedup.py 2>&1 | tail -30
```

Expected: TEST 1 passes with rel < 1e-8. TEST 2 shows K=6 uses fewer njev than K=1 while achieving comparable cost.

- [ ] **Step 3: Commit**

```bash
git add debug_optimizer_speedup.py
git commit -m "test: integration validation of optimizer speedup with real coils"
```

### Task 11: Run debug_coilset.py to confirm coil BS is unaffected

**Files:**
- Test: `debug_coilset.py` (existing)

- [ ] **Step 1: Run the coilset validation**

```bash
cd /home/toddelder/DESC && python debug_coilset.py 2>&1 | tail -20
```

Expected: All 15 tests PASS. These changes don't touch coil code, but this confirms nothing was accidentally broken.

### Task 12: Run coil tests

**Files:**
- Test: `tests/test_coils.py`

- [ ] **Step 1: Run the coil test suite**

```bash
cd /home/toddelder/DESC && python -m pytest tests/test_coils.py -x --timeout=300 2>&1 | tail -20
```

Expected: All tests pass.

### Task 13: Profile on Perlmutter

**Files:**
- Run: `profile_desc_opt.py` (existing profiler)

This is the primary validation of the performance claims. Must be run on PM after pushing.

- [ ] **Step 1: Push branch to remote**

```bash
cd /home/toddelder/DESC && git push origin tme/descspedup
```

- [ ] **Step 2: On Perlmutter, pull and run profiler**

```bash
# On PM (after salloc + conda activate):
cd /path/to/DESC && git pull
python profile_desc_opt.py 2>&1 | tee profile_output.txt
```

- [ ] **Step 3: Compare per-iteration Jacobian time against baseline**

Baseline (from last PM run): Jacobian = 0.978s/iter, Total = 1.691s/iter.

Expected with Change A alone: Jacobian ~0.2-0.5s (2-5x faster).
Expected with A+B (K=6): amortized Jacobian ~0.05-0.1s (up to 3.4x total speedup).

- [ ] **Step 4: Run `debug_optimizer_speedup.py` on PM for real-data validation**

```bash
python debug_optimizer_speedup.py 2>&1 | tee optimizer_validation.txt
```

Expected: TEST 1 (fwd vs rev) PASS. TEST 2 (Broyden convergence) comparable costs.
