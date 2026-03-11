"""
DESC Coil Optimization Profiler
================================
Profiles the hot path of a finite-beta coil optimization using the
trefoil workload. Measures:

  1. Data loading & setup
  2. Objective build
  3. Full optimizer pipeline (2 iters)
  4. Compiled function timings (objective eval + Jacobian)
  5. Biot-Savart isolation (per-coilset breakdown)
  6. Old (scan/fori_loop) vs New (vmap) CoilSet evaluation
  7. Trust-region linear algebra
  8. Jacobian: scan vs vmap under AD

Usage
-----
  Local (CPU):       PLATFORM=local python profile_desc_opt.py
  Perlmutter (GPU):  python profile_desc_opt.py   (after salloc + activating DESC env)

Set PLATFORM env var or toggle the default below.
"""

import os
import sys
import time
import warnings
import numpy as np

# =============================================================================
# PLATFORM
# =============================================================================
PLATFORM = os.environ.get("PLATFORM", "perlmutter")  # "local" | "perlmutter"

# GPU setup
if PLATFORM == "perlmutter":
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    _jax_cache = os.path.join(os.environ.get("PSCRATCH", "/tmp"), "jax_cache")
    import jax
    jax.config.update("jax_compilation_cache_dir", _jax_cache)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    from desc import set_device
    set_device("gpu")
else:
    import jax

from desc.backend import jnp
from desc.io import load
from desc.grid import LinearGrid, QuadratureGrid
from desc.magnetic_fields import SumMagneticField
from desc.coils import (
    FourierXYZCoil, FourierPlanarCoil, CoilSet, MixedCoilSet,
    biot_savart_general, biot_savart_general_vector_potential,
)
from desc.objectives import (
    ObjectiveFunction,
    FixCoilCurrent,
    FixParameters,
    QuadraticFlux,
)
from desc.objectives._coils import FieldNormalError
from desc.optimize import Optimizer
from desc.backend import fori_loop, tree_stack
from desc.utils import reflection_matrix
from desc.utils import rpz2xyz, xyz2rpz, rpz2xyz_vec, xyz2rpz_vec
from desc.compute import get_params
from jax.lax import scan as jax_scan

# =============================================================================
# DATA PATHS
# =============================================================================
PATHS = {
    "local": {
        "data_dir":        "/home/toddelder/projects/divertor_coil/output_runs/divertor_trefoil_v0",
        "trefoil_upper":   "footpoint_coils.h5",
        "trefoil_lower":   None,
        "shaping_file":    "shaping_filtered.h5",
        "encircling_file": "/home/toddelder/projects/dynamic_accessibility/G1600_12-89_data/G1600-12-89_large.h5",
        "fullbeta_eq":     "/home/toddelder/projects/dynamic_accessibility/FullBeta_DivLeg_Opt/equil_G1600_20260212_072035.h5",
        "xpt_pts":         "/home/toddelder/projects/self_intersecting_surface/data/G1600_xpt_coil_location_20cm.txt",
        "inner_fps":       "/home/toddelder/projects/self_intersecting_surface/data/strikeline_xyz_inner.npy",
        "outer_fps":       "/home/toddelder/projects/self_intersecting_surface/data/strikeline_xyz_outer.npy",
    },
    "perlmutter": {
        "data_dir":        "/global/homes/t/telder/divertor/bifoil_optimization/data",
        "trefoil_upper":   "footpoint_coils_upper.h5",
        "trefoil_lower":   "footpoint_coils_lower.h5",
        "shaping_file":    "shaping_filtered.h5",
        "encircling_file": "/global/homes/t/telder/divertor/data/G1600_12-89_w-Bxdl_data/G1600-12-89_large.h5",
        "fullbeta_eq":     "/global/homes/t/telder/divertor/data/G1600_12-89_w-Bxdl_data/equil_Helios_G1600-12-89_free_L12_M12_N20_nocont.h5",
        "xpt_pts":         "/global/homes/t/telder/divertor/data/G1600_12-89_w-Bxdl_data/G1600_xpt_coil_location_20cm.txt",
        "inner_fps":       "/global/homes/t/telder/divertor/data/G1600_12-89_w-Bxdl_data/strikeline_xyz_inner.npy",
        "outer_fps":       "/global/homes/t/telder/divertor/data/G1600_12-89_w-Bxdl_data/strikeline_xyz_outer.npy",
    },
}[PLATFORM]

# =============================================================================
# CONFIG — tune problem size here
# =============================================================================
JAC_CHUNK_SIZE = 200
EQ_GRID_L, EQ_GRID_M, EQ_GRID_N = 5, 10, 5
PLOT_GRID_M, PLOT_GRID_N = 3, 5
N_SURF = 2
NQUAD = 16
NMODES = 12
EXTENT, MIN_EXTENT = 0.50, -1.0
COIL_KEEP_FRAC = 0.1  # 1.0 = all coils, 0.1 = ~10%
N_WARM = 5


# ── Timer helper ──────────────────────────────────────────────────────
class Stopwatch:
    def __init__(self):
        self.timings = {}

    def time(self, label):
        return _StopwatchCtx(self, label)

    def report(self):
        print("\n" + "=" * 70)
        print("ALL TIMINGS")
        print("=" * 70)
        ml = max(len(k) for k in self.timings) if self.timings else 20
        for label, t in self.timings.items():
            print(f"  {label:<{ml+2}} {t:>10.3f} s")
        print("=" * 70)


class _StopwatchCtx:
    def __init__(self, sw, label):
        self.sw, self.label = sw, label
    def __enter__(self):
        jax.effects_barrier()
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        jax.effects_barrier()
        elapsed = time.perf_counter() - self.t0
        self.sw.timings[self.label] = elapsed
        print(f"  [{self.label}] {elapsed:.3f} s")


def bench(fn, n=N_WARM, label=""):
    """Run fn n times, return list of wall-clock times."""
    times = []
    for _ in range(n):
        jax.effects_barrier()
        t0 = time.perf_counter()
        result = fn()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        jax.effects_barrier()
        times.append(time.perf_counter() - t0)
    avg = np.mean(times)
    if label:
        print(f"    {label}: {[f'{t:.3f}' for t in times]} avg={avg:.3f} s")
    return times, avg


sw = Stopwatch()

print("DESC Coil Optimization Profiler")
print(f"  Platform: {PLATFORM} | Backend: {jax.default_backend()} | Devices: {jax.devices()}")
print()


def drop_coils(cs, frac=COIL_KEEP_FRAC):
    """Subsample coils from a CoilSet, keeping at least 1."""
    coils = list(cs.coils)
    n = max(1, int(len(coils) * frac))
    idx = np.linspace(0, len(coils) - 1, n, dtype=int)
    return CoilSet(*[coils[i] for i in idx], NFP=cs.NFP, sym=cs.sym)


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING
# ══════════════════════════════════════════════════════════════════════
print("PHASE 1: Data Loading")
print("-" * 40)

data_dir = PATHS["data_dir"]

with sw.time("Load equilibrium"):
    eq_raw = load(PATHS["fullbeta_eq"])
    eq = eq_raw[-1] if isinstance(eq_raw, (list, tuple)) else eq_raw
    print(f"    eq: NFP={eq.NFP}, L={eq.L}, M={eq.M}, N={eq.N}")

with sw.time("Load coils"):
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
        n_half_u = max(1, len(upper_coils) // 2)
        n_half_l = max(1, len(lower_coils) // 2)
        trefoils_upper = CoilSet(*upper_coils[:n_half_u], NFP=eq.NFP, sym=True)
        trefoils_lower = CoilSet(*lower_coils[:n_half_l], NFP=eq.NFP, sym=True)

    shaping_raw = load(os.path.join(data_dir, PATHS["shaping_file"]))
    encircling = load(PATHS["encircling_file"])

with sw.time("Convert shaping -> FourierPlanar"):
    raw_coils = list(shaping_raw.coils)
    n_keep = max(1, int(len(raw_coils) * COIL_KEEP_FRAC))
    keep_idx = np.linspace(0, len(raw_coils) - 1, n_keep, dtype=int)
    raw_coils = [raw_coils[i] for i in keep_idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shaping_fp = [
            coil.to_FourierPlanar(N=2, grid=LinearGrid(N=32), basis="xyz", name=f"sh_{i}")
            for i, coil in enumerate(raw_coils)
        ]
    shaping = CoilSet(*shaping_fp, NFP=eq.NFP, sym=True)

trefoils_upper = drop_coils(trefoils_upper, COIL_KEEP_FRAC)
trefoils_lower = drop_coils(trefoils_lower, COIL_KEEP_FRAC)

print(f"    trefoils_upper: {len(trefoils_upper)} fund, {trefoils_upper.num_coils} total")
print(f"    trefoils_lower: {len(trefoils_lower)} fund, {trefoils_lower.num_coils} total")
print(f"    shaping: {len(shaping)} fund, {shaping.num_coils} total")
print(f"    encircling: {len(encircling)} coils")

with sw.time("Build divertor geometry"):
    xpt_pts = np.loadtxt(PATHS["xpt_pts"])
    inner_fps = np.load(PATHS["inner_fps"])
    outer_fps = np.load(PATHS["outer_fps"])
    xpt_curve = FourierXYZCoil.from_values(current=0, N=NMODES, coords=xpt_pts, basis="xyz")
    fpi_curve = FourierXYZCoil.from_values(current=0, N=NMODES, coords=inner_fps, basis="xyz")
    fpo_curve = FourierXYZCoil.from_values(current=0, N=NMODES, coords=outer_fps, basis="xyz")

    def get_coords(curve, nq=NQUAD):
        return np.squeeze(curve._compute_position(grid=nq, basis="xyz"))
    def make_surface(c_start, c_end, n_res, ext, min_ext):
        v = np.linspace(min_ext, ext, n_res).reshape(1, -1, 1)
        r1, r2 = c_start[:, None, :], c_end[:, None, :]
        R_vec = r2 - r1
        pts = r1 + v * R_vec
        Tu = np.gradient(pts, axis=0)
        Tv = np.tile(R_vec, (1, n_res, 1))
        normals = np.cross(Tu, Tv)
        mag = np.linalg.norm(normals, axis=2, keepdims=True)
        mag[mag < 1e-9] = 1.0
        return pts.reshape(-1, 3), (normals / mag).reshape(-1, 3)

    xpt_c, fpi_c, fpo_c = get_coords(xpt_curve), get_coords(fpi_curve), get_coords(fpo_curve)
    pts_in, norms_in = make_surface(xpt_c, fpi_c, N_SURF, EXTENT, MIN_EXTENT)
    pts_out, norms_out = make_surface(xpt_c, fpo_c, N_SURF, EXTENT, MIN_EXTENT)
    print(f"    Inner leg: {len(pts_in)} pts, Outer leg: {len(pts_out)} pts")


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: BUILD OBJECTIVES
# ══════════════════════════════════════════════════════════════════════
print("\nPHASE 2: Build Objectives")
print("-" * 40)

eq_source_grid = QuadratureGrid(L=EQ_GRID_L, M=EQ_GRID_M, N=EQ_GRID_N, NFP=eq.NFP)
for c in shaping.coils:
    c.current = 0

with sw.time("Create QuadraticFlux"):
    qf = QuadraticFlux(
        eq,
        field=[trefoils_upper, trefoils_lower, shaping, encircling],
        eval_grid=LinearGrid(M=PLOT_GRID_M, N=max(2, PLOT_GRID_N // 4), NFP=eq.NFP),
        vacuum=False, weight=1, name="QuadFlux",
    )

with sw.time("Create FieldNormalError (inner)"):
    fne_inner = FieldNormalError(
        points=pts_in, normal_vectors=norms_in,
        eq=eq, basis="xyz",
        field=[trefoils_upper, trefoils_lower, shaping, encircling],
        target=0, weight=100, field_fixed=False, eq_grid=eq_source_grid,
        name="Inner B*n",
    )

with sw.time("Create FieldNormalError (outer)"):
    fne_outer = FieldNormalError(
        points=pts_out, normal_vectors=norms_out,
        eq=eq, basis="xyz",
        field=[trefoils_upper, trefoils_lower, shaping, encircling],
        target=0, weight=100, field_fixed=False, eq_grid=eq_source_grid,
        name="Outer B*n",
    )

cur_bound_u = FixCoilCurrent(trefoils_upper, bounds=(-4.5e6, 4.5e6), weight=1.0)
cur_bound_l = FixCoilCurrent(trefoils_lower, bounds=(-4.5e6, 4.5e6), weight=1.0)

obj_fun = ObjectiveFunction(
    (qf, cur_bound_u, cur_bound_l, fne_inner, fne_outer),
    deriv_mode="batched", jac_chunk_size=JAC_CHUNK_SIZE,
)

constraints = (
    FixParameters(trefoils_upper, [{"center": True, "r_n": True, "rotmat": True, "shift": True}] * len(trefoils_upper)),
    FixParameters(trefoils_lower, [{"center": True, "r_n": True, "rotmat": True, "shift": True}] * len(trefoils_lower)),
    FixParameters(shaping, [{"center": True, "normal": True, "r_n": True, "current": True}] * len(shaping)),
    FixParameters(encircling),
)

things = [trefoils_upper, trefoils_lower, shaping, encircling]


# ══════════════════════════════════════════════════════════════════════
# PHASE 3: OPTIMIZER (2 iterations — build + compile + iterate)
# ══════════════════════════════════════════════════════════════════════
print("\nPHASE 3: Optimizer (2 iterations)")
print("-" * 40)

optimizer = Optimizer("lsq-exact")

with sw.time("Optimizer.optimize (2 iters)"):
    _, result = optimizer.optimize(
        things=things, objective=obj_fun, constraints=constraints,
        maxiter=2, options={"max_nfev": 10}, verbose=2, copy=True,
    )

print(f"    Result: {result.message}")
print(f"    nfev={result.nfev}, njev={result.njev}")


# ══════════════════════════════════════════════════════════════════════
# PHASE 4: COMPILED FUNCTION TIMINGS
# ══════════════════════════════════════════════════════════════════════
print("\nPHASE 4: Compiled Function Timings")
print("-" * 40)

from desc.objectives.utils import combine_args
from desc.optimize._constraint_wrappers import LinearConstraintProjection

obj_fun2 = ObjectiveFunction(
    (qf, cur_bound_u, cur_bound_l, fne_inner, fne_outer),
    deriv_mode="batched", jac_chunk_size=JAC_CHUNK_SIZE,
)
constraint_objs = (
    FixParameters(trefoils_upper, [{"center": True, "r_n": True, "rotmat": True, "shift": True}] * len(trefoils_upper)),
    FixParameters(trefoils_lower, [{"center": True, "r_n": True, "rotmat": True, "shift": True}] * len(trefoils_lower)),
    FixParameters(shaping, [{"center": True, "normal": True, "r_n": True, "current": True}] * len(shaping)),
    FixParameters(encircling),
)
linear_constraint = ObjectiveFunction(constraint_objs)

obj_fun2.build(verbose=0)
linear_constraint.build(verbose=0)
obj_fun2, linear_constraint = combine_args(obj_fun2, linear_constraint)

lcp = LinearConstraintProjection(obj_fun2, linear_constraint)
lcp.build(verbose=0)

x0 = lcp.x(*[things[things.index(t)] for t in lcp.things])

print(f"    dim_x={lcp.dim_x}, dim_f={lcp.dim_f}, projected dim_x={x0.shape[0]}")

# JIT compilation
with sw.time("JIT: compute_scaled_error"):
    f = lcp.compute_scaled_error(x0, lcp.constants)
    f.block_until_ready()

with sw.time("JIT: jac_scaled_error"):
    J = lcp.jac_scaled_error(x0, lcp.constants)
    J.block_until_ready()

print(f"    f shape={f.shape}, J shape={J.shape}")

# Warm calls
times_f, avg_f = bench(
    lambda: lcp.compute_scaled_error(x0, lcp.constants),
    label="compute_scaled_error",
)
sw.timings["Warm: compute_scaled_error"] = avg_f

times_j, avg_j = bench(
    lambda: lcp.jac_scaled_error(x0, lcp.constants),
    label="jac_scaled_error",
)
sw.timings["Warm: jac_scaled_error"] = avg_j


# ══════════════════════════════════════════════════════════════════════
# PHASE 5: BIOT-SAVART ISOLATION
# ══════════════════════════════════════════════════════════════════════
print("\nPHASE 5: Biot-Savart Isolation (new vectorized code)")
print("-" * 40)

all_eval_pts = jnp.concatenate([jnp.asarray(pts_in), jnp.asarray(pts_out)])
n_eval = all_eval_pts.shape[0]
print(f"    {n_eval} evaluation points")

# Per-coilset breakdown
for name, cs in [("trefoils_upper", trefoils_upper), ("trefoils_lower", trefoils_lower),
                  ("shaping", shaping), ("encircling", encircling)]:
    # warmup
    _ = cs.compute_magnetic_field(all_eval_pts, basis="xyz")
    _, avg = bench(
        lambda cs=cs: cs.compute_magnetic_field(all_eval_pts, basis="xyz"),
        label=f"{name} ({len(cs)}f/{cs.num_coils}t)",
    )
    sw.timings[f"BS new: {name}"] = avg

# Total
total_field = SumMagneticField([trefoils_upper, trefoils_lower, shaping, encircling])
_ = total_field.compute_magnetic_field(all_eval_pts, basis="xyz")
_, avg_bs = bench(
    lambda: total_field.compute_magnetic_field(all_eval_pts, basis="xyz"),
    label="TOTAL (all coilsets)",
)
sw.timings["BS new: total"] = avg_bs


# ══════════════════════════════════════════════════════════════════════
# PHASE 6: OLD vs NEW CoilSet Evaluation
# ══════════════════════════════════════════════════════════════════════
print("\nPHASE 6: Old (scan/fori_loop) vs New (vmap) CoilSet Evaluation")
print("-" * 40)


def old_coilset_compute_B(cs, coords, basis="xyz"):
    """Reimplementation of the OLD CoilSet._compute_A_or_B using scan/fori_loop.

    This is what the code did before the vectorized rewrite:
    - Convert to RPZ
    - fori_loop over NFP periods (shifting phi)
    - scan over coils within each period
    - Each coil does its own rpz<->xyz internally
    """
    coords = jnp.atleast_2d(jnp.asarray(coords))
    if basis == "rpz":
        coords_xyz = rpz2xyz(coords)
    else:
        coords_xyz = coords

    params = [get_params(["x_s", "x", "s", "ds"], coil, basis="rpz") for coil in cs]
    for par, coil in zip(params, cs):
        par["current"] = coil.current

    if cs.sym:
        normal = jnp.array(
            [-jnp.sin(jnp.pi / cs.NFP), jnp.cos(jnp.pi / cs.NFP), 0]
        )
        coords_sym = (
            coords_xyz
            @ reflection_matrix(normal).T
            @ reflection_matrix([0, 0, 1]).T
        )
        coords_xyz_full = jnp.vstack((coords_xyz, coords_sym))
    else:
        coords_xyz_full = coords_xyz

    coords_rpz = xyz2rpz(coords_xyz_full)
    op = cs[0].compute_magnetic_field

    def nfp_loop(k, AB):
        coords_nfp = coords_rpz + jnp.array([0, 2 * jnp.pi * k / cs.NFP, 0])

        def body(AB, x):
            AB += op(coords_nfp, params=x, basis="rpz", source_grid=None, chunk_size=None)
            return AB, None

        AB += jax_scan(body, jnp.zeros(coords_nfp.shape), tree_stack(params))[0]
        return AB

    AB = fori_loop(0, cs.NFP, nfp_loop, jnp.zeros_like(coords_rpz))

    if cs.sym:
        n_orig = coords.shape[0]
        AB = AB[:n_orig] + AB[n_orig:] * jnp.array([-1, 1, 1])

    if basis == "xyz":
        AB = rpz2xyz_vec(AB, x=coords[:, 0], y=coords[:, 1])
    return AB


# Test each coilset that uses the CoilSet._compute_A_or_B path
for name, cs in [("trefoils_upper", trefoils_upper), ("trefoils_lower", trefoils_lower),
                  ("shaping", shaping)]:
    print(f"  {name}: {len(cs)} fund, {cs.num_coils} total, NFP={cs.NFP}, sym={cs.sym}")

    # Warmup both
    B_new = cs.compute_magnetic_field(all_eval_pts, basis="xyz")
    B_old = old_coilset_compute_B(cs, all_eval_pts, basis="xyz")

    # Verify agreement
    diff = jnp.max(jnp.abs(B_new - B_old))
    rel_diff = diff / jnp.max(jnp.abs(B_old))
    print(f"    Max |new - old| = {diff:.2e}, relative = {rel_diff:.2e}")

    # Bench new (vectorized vmap)
    _, avg_new = bench(
        lambda cs=cs: cs.compute_magnetic_field(all_eval_pts, basis="xyz"),
        label="NEW (vmap)",
    )

    # Bench old (scan/fori_loop)
    _, avg_old = bench(
        lambda cs=cs: old_coilset_compute_B(cs, all_eval_pts, basis="xyz"),
        label="OLD (scan/fori_loop)",
    )

    speedup = avg_old / max(avg_new, 1e-12)
    print(f"    >>> Speedup: {speedup:.2f}x")
    sw.timings[f"Old vs New {name}: OLD"] = avg_old
    sw.timings[f"Old vs New {name}: NEW"] = avg_new


# ══════════════════════════════════════════════════════════════════════
# PHASE 7: TRUST-REGION LINEAR ALGEBRA
# ══════════════════════════════════════════════════════════════════════
print("\nPHASE 7: Trust-Region Linear Algebra")
print("-" * 40)

J_jax = jnp.asarray(J)
f_jax = jnp.asarray(f)
print(f"    J shape: {J_jax.shape} ({J_jax.nbytes / 1e6:.1f} MB)")

with sw.time("QR factorization"):
    Q_j, R_j = jnp.linalg.qr(J_jax, mode="reduced")
    Q_j.block_until_ready()

with sw.time("QR solve"):
    p_j, _, _, _ = jnp.linalg.lstsq(R_j, -Q_j.T @ f_jax, rcond=None)
    p_j.block_until_ready()

with sw.time("J^T @ J"):
    JtJ_j = J_jax.T @ J_jax
    JtJ_j.block_until_ready()

with sw.time("Cholesky factor"):
    try:
        L_cho = jnp.linalg.cholesky(JtJ_j + 1e-6 * jnp.eye(JtJ_j.shape[0]))
        L_cho.block_until_ready()
    except Exception as e:
        print(f"    Cholesky failed: {e}")
        sw.timings["Cholesky factor"] = float("nan")


# ══════════════════════════════════════════════════════════════════════
# PHASE 8: JACOBIAN — scan vs vmap under AD
# ══════════════════════════════════════════════════════════════════════
print("\nPHASE 8: Jacobian of Biot-Savart (scan vs vmap under AD)")
print("-" * 40)

# Pick the largest CoilSet for isolated Jacobian test
test_cs = max([trefoils_upper, trefoils_lower, shaping], key=lambda cs: cs.num_coils)
test_name = {id(trefoils_upper): "trefoils_upper", id(trefoils_lower): "trefoils_lower",
             id(shaping): "shaping"}[id(test_cs)]
print(f"    Testing with: {test_name} ({len(test_cs)} fund, {test_cs.num_coils} total)")

# Pre-compute stacked coil data
all_x, all_t, all_I = [], [], []
for coil in test_cs:
    sg = LinearGrid(N=2 * coil.N * getattr(coil, "NFP", 1) + 5)
    data = coil.compute(["x", "x_s", "ds"], grid=sg, basis="xyz")
    all_x.append(data["x"])
    all_t.append(data["x_s"] * data["ds"][:, None])
    all_I.append(coil.current)
coil_x, coil_t, coil_I = jnp.stack(all_x), jnp.stack(all_t), jnp.array(all_I)
print(f"    Coil data: x={coil_x.shape}, t={coil_t.shape}, I={coil_I.shape}")

flat_params = jnp.concatenate([coil_x.ravel(), coil_t.ravel(), coil_I])
n_x, n_t = coil_x.size, coil_t.size
n_params = flat_params.shape[0]
n_out = all_eval_pts.shape[0] * 3
print(f"    n_params={n_params}, n_output={n_out}")


def bs_vmap_flat(params):
    cx = params[:n_x].reshape(coil_x.shape)
    ct = params[n_x:n_x + n_t].reshape(coil_t.shape)
    ci = params[n_x + n_t:]
    B_all = jax.vmap(
        lambda x, t, i: biot_savart_general(all_eval_pts, x, i * t),
        in_axes=(0, 0, 0),
    )(cx, ct, ci)
    return B_all.sum(axis=0).ravel()


def bs_scan_flat(params):
    cx = params[:n_x].reshape(coil_x.shape)
    ct = params[n_x:n_x + n_t].reshape(coil_t.shape)
    ci = params[n_x + n_t:]
    def body(B_acc, args):
        x, t, i = args
        B_acc += biot_savart_general(all_eval_pts, x, i * t)
        return B_acc, None
    B, _ = jax_scan(body, jnp.zeros((all_eval_pts.shape[0], 3)), (cx, ct, ci))
    return B.ravel()


jac_vmap_fn = jax.jit(jax.jacfwd(bs_vmap_flat))
jac_scan_fn = jax.jit(jax.jacfwd(bs_scan_flat))

# JIT compile
with sw.time("JIT: jacfwd(vmap BS)"):
    J_vmap = jac_vmap_fn(flat_params)
    J_vmap.block_until_ready()

with sw.time("JIT: jacfwd(scan BS)"):
    J_scan = jac_scan_fn(flat_params)
    J_scan.block_until_ready()

# Warm calls
_, avg_jv = bench(lambda: jac_vmap_fn(flat_params), label="jacfwd(vmap BS)")
_, avg_js = bench(lambda: jac_scan_fn(flat_params), label="jacfwd(scan BS)")
sw.timings["Warm: jacfwd(vmap BS)"] = avg_jv
sw.timings["Warm: jacfwd(scan BS)"] = avg_js
print(f"    Speedup (scan/vmap): {avg_js / max(avg_jv, 1e-12):.2f}x")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n")
print("=" * 60)
print("SUMMARY")
print("=" * 60)

t_fun = avg_f
t_jac = avg_j
t_qr = sw.timings.get("QR factorization", 0) + sw.timings.get("QR solve", 0)
t_iter = t_fun + t_jac + t_qr

print(f"\nPer-iteration cost breakdown:")
print(f"  Objective eval:  {t_fun:>8.3f} s  ({100*t_fun/t_iter:>5.1f}%)")
print(f"  Jacobian:        {t_jac:>8.3f} s  ({100*t_jac/t_iter:>5.1f}%)")
print(f"  TR solve (QR):   {t_qr:>8.3f} s  ({100*t_qr/t_iter:>5.1f}%)")
print(f"  ───────────────────────────────────────")
print(f"  Total/iteration: {t_iter:>8.3f} s")
print(f"  Jac/fun ratio:   {t_jac/max(t_fun,1e-9):.1f}x")
jit_overhead = sw.timings.get("JIT: compute_scaled_error", 0) + sw.timings.get("JIT: jac_scaled_error", 0)
print(f"  JIT overhead:    {jit_overhead:>8.1f} s (one-time)")

print(f"\nOld vs New CoilSet evaluation:")
for name in ["trefoils_upper", "trefoils_lower", "shaping"]:
    old_key = f"Old vs New {name}: OLD"
    new_key = f"Old vs New {name}: NEW"
    if old_key in sw.timings and new_key in sw.timings:
        old_t = sw.timings[old_key]
        new_t = sw.timings[new_key]
        print(f"  {name}: OLD={old_t:.3f}s, NEW={new_t:.3f}s, speedup={old_t/max(new_t,1e-12):.2f}x")

print(f"\nIsolated BS Jacobian ({test_name}, fund domain only):")
print(f"  jacfwd(vmap): {avg_jv:.4f} s")
print(f"  jacfwd(scan): {avg_js:.4f} s")
print(f"  speedup:      {avg_js/max(avg_jv,1e-12):.2f}x")

sw.report()
