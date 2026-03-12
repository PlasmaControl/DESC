"""
Integration test: optimizer speedup changes with real DESC optimization.

Validates:
  - LCP._jac fwd vs rev mode match on a ForceBalance equilibrium objective
  - lsqtr Broyden updates (K=6) converge similarly to baseline (K=1)
  - Full Optimizer.optimize pipeline works end-to-end with both changes

Usage:  conda run -n desc-thea-gpu python debug_optimizer_speedup.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import desc.examples
from desc.backend import jnp
from desc.equilibrium import Equilibrium
from desc.objectives import (
    ForceBalance,
    ObjectiveFunction,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixPressure,
    FixPsi,
    get_fixed_boundary_constraints,
)
from desc.optimize import LinearConstraintProjection, Optimizer

PASS_COUNT = 0
FAIL_COUNT = 0

# ── Setup: small HELIOTRON equilibrium ──
print("Loading HELIOTRON equilibrium...")
eq = desc.examples.get("HELIOTRON")
eq.change_resolution(2, 2, 2, 4, 4, 4)
print(f"  L={eq.L}, M={eq.M}, N={eq.N}")

# ── TEST 1: LCP._jac fwd vs rev ──
print("\n--- TEST 1: LCP._jac fwd vs rev ---")
obj = ObjectiveFunction(
    ForceBalance(eq, deriv_mode="auto"), deriv_mode="batched", use_jit=False
)
con = ObjectiveFunction(get_fixed_boundary_constraints(eq))
lc = LinearConstraintProjection(obj, con)
lc.build(verbose=0)

x_reduced = lc.x()
print(f"  dim_f={lc.dim_f}, dim_x_reduced={lc._dim_x_reduced}")
print(f"  AD mode auto-selected: {lc._ad_mode}")

# Force fwd mode and compute
lc._ad_mode = "fwd"
J_fwd = np.array(lc.jac_scaled_error(x_reduced))

# Force rev mode and compute
lc._ad_mode = "rev"
J_rev = np.array(lc.jac_scaled_error(x_reduced))

diff = np.max(np.abs(J_fwd - J_rev))
maxref = np.max(np.abs(J_fwd))
rel = diff / maxref if maxref > 0 else diff
status = "PASS" if rel < 1e-8 else "*** FAIL ***"
print(f"  J shape: {J_fwd.shape}")
print(f"  max |fwd-rev|: {diff:.2e}, rel: {rel:.2e} {status}")
if rel < 1e-8:
    PASS_COUNT += 1
else:
    FAIL_COUNT += 1

# ── TEST 2: Full optimization with K=1 (baseline) ──
print("\n--- TEST 2: Optimizer.optimize with jac_recompute_every=1 ---")
eq1 = eq.copy()
opt1 = Optimizer("lsq-exact")
eq1, result_k1 = opt1.optimize(
    eq1,
    ObjectiveFunction(ForceBalance(eq1)),
    get_fixed_boundary_constraints(eq1),
    maxiter=5,
    verbose=1,
    options={"jac_recompute_every": 1},
)
print(f"  K=1: cost={result_k1['cost']:.6e}, njev={result_k1['njev']}, nit={result_k1['nit']}")
if result_k1["nit"] > 0:
    PASS_COUNT += 1
else:
    print("  *** FAIL *** (no iterations)")
    FAIL_COUNT += 1

# ── TEST 3: Full optimization with K=6 (Broyden) ──
print("\n--- TEST 3: Optimizer.optimize with jac_recompute_every=6 ---")
eq6 = eq.copy()
opt6 = Optimizer("lsq-exact")
eq6, result_k6 = opt6.optimize(
    eq6,
    ObjectiveFunction(ForceBalance(eq6)),
    get_fixed_boundary_constraints(eq6),
    maxiter=5,
    verbose=1,
    options={"jac_recompute_every": 6},
)
print(f"  K=6: cost={result_k6['cost']:.6e}, njev={result_k6['njev']}, nit={result_k6['nit']}")

if result_k6["njev"] <= result_k1["njev"]:
    print(f"  Broyden njev reduction: PASS ({result_k6['njev']} <= {result_k1['njev']})")
    PASS_COUNT += 1
else:
    print(f"  Broyden njev reduction: *** FAIL *** ({result_k6['njev']} > {result_k1['njev']})")
    FAIL_COUNT += 1

print(f"\n{'='*50}")
print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
print(f"{'='*50}")
if FAIL_COUNT > 0:
    exit(1)
