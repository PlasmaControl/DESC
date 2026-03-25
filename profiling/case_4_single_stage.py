#!/usr/bin/env python
"""Case 4: Single-stage optimization with proximal-lsq-exact (lightweight).

Replicates the objective/constraint setup from:
  dynamic-accessibility/half_beta_half_slew_eq/slew_from_halfslew_eq-QS-divLeg/
  free_boundary_half-slew_eq-QS-divLeg-increasing-bdy-weight-and-slew-lower-res.py

using LIGHTWEIGHT=True config values.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from instrument import (
    setup_gpu, monkey_patch_lsqtr, time_per_objective_jac,
    print_lsqtr_breakdown, print_objective_breakdown, save_results,
)

setup_gpu()

import numpy as np

from desc.coils import CoilSet, FourierXYZCoil, MixedCoilSet
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid
from desc.io import load
from desc.magnetic_fields import FourierCurrentPotentialField
from desc.objectives import (
    BootstrapRedlConsistency, BoundaryError, FieldNormalError,
    FixAtomicNumber, FixCoilCurrent, FixCurrent, FixElectronDensity,
    FixElectronTemperature, FixIonTemperature, FixParameters, FixPsi,
    ForceBalance, ObjectiveFunction, QuasisymmetryTwoTerm,
)
from desc.optimize import Optimizer

# ==========================================
# CONFIGURATION  (LIGHTWEIGHT = True)
# ==========================================
LIGHTWEIGHT = True

HOME = os.path.expanduser("~")
SOURCE_DIR = os.path.join(
    HOME,
    "dynamic-accessibility/half_beta_half_slew_eq/"
    "slew_from_halfslew_eq-QS-divLeg",
)

CONFIG = {
    # Coil paths (half-slew)
    "coil_encircling": os.path.join(SOURCE_DIR, "data/midbeta/encircling_midbeta.h5"),
    "coil_shaping":    os.path.join(SOURCE_DIR, "data/midbeta/shaping_midbeta.h5"),
    # Initial equilibrium (from 1st free-boundary QS solve)
    "eq_path": os.path.join(
        HOME,
        "dynamic-accessibility/half_beta_half_slew_eq/"
        "first_slew_from_halfslew_eq-QS/output/single_stage_20260313_171631/"
        "eq_single_stage_QS.h5",
    ),
    # Divertor leg curves
    "xpt_pts":   "/global/homes/t/telder/divertor/data/G1600_12-89_w-Bxdl_data/G1600_xpt_coil_location_20cm.txt",
    "inner_fps": "/global/homes/t/telder/divertor/data/G1600_12-89_w-Bxdl_data/strikeline_xyz_inner.npy",
    "outer_fps": "/global/homes/t/telder/divertor/data/G1600_12-89_w-Bxdl_data/strikeline_xyz_outer.npy",
    # Physics
    "helicity": (1, 0),   # QA
    "psi_scale": 0.925,
    "slew_limit": 500e3,  # +/-500 kA
    # Optimization (LIGHTWEIGHT)
    "jac_chunk_size": 1,
    "maxiter": 3,
    "maxiter_inner": 3,
    "ftol": 1e-4,
    "xtol": 1e-6,
    "gtol": 1e-6,
    # Objective weights
    "weight_boundary":     1e0,
    "weight_qs":           0.5,
    "weight_bootstrap":    0.5,
    "weight_coil_current": 2.0,
    "weight_divertor":     0.5,
    # Divertor ruled surface (LIGHTWEIGHT)
    "div_nmodes":   4,
    "div_nquad":    16,
    "div_n_radial": 3,
    "div_extent":   0.5,
    "div_min_extent": 0.0,
    # Subsample shaping coils (LIGHTWEIGHT: keep 2%)
    "shaping_keep_fraction": 0.02,
}

# ==========================================
# LOAD EQUILIBRIUM
# ==========================================
print("Loading equilibrium...")
eq = load(CONFIG["eq_path"])
print(f"Loaded eq: L={eq.L}, M={eq.M}, N={eq.N}, NFP={eq.NFP}")

# Scale Psi
eq.Psi = CONFIG["psi_scale"] * eq.Psi
print(f"Scaled Psi: {eq.Psi:.4f} Wb")

# Ensure surface is FourierCurrentPotentialField for free-boundary
if not isinstance(eq.surface, FourierCurrentPotentialField):
    eq.surface = FourierCurrentPotentialField.from_surface(
        eq.surface, M_Phi=eq.M, N_Phi=eq.N
    )

# Reduce resolution for lightweight testing
eq.change_resolution(M=2, N=2, M_grid=6, N_grid=6, L=2, L_grid=6)
print(f"Reduced eq: L={eq.L}, M={eq.M}, N={eq.N}, M_grid={eq.M_grid}, N_grid={eq.N_grid}")

# Re-solve force balance with scaled Psi
print("\nRe-solving force balance after Psi scaling...")
eq.solve(
    objective="force",
    optimizer="lsq-exact",
    ftol=CONFIG["ftol"],
    xtol=CONFIG["xtol"],
    gtol=CONFIG["gtol"],
    maxiter=CONFIG["maxiter_inner"],
    verbose=2,
    copy=False,
)

# ==========================================
# LOAD COILS
# ==========================================
print("\nLoading coils...")
encircling = load(CONFIG["coil_encircling"])
shaping_full = load(CONFIG["coil_shaping"])
print(f"Encircling coils: {len(encircling)}")
print(f"Shaping coils (full): {len(shaping_full)}")

# Subsample shaping coils
frac = CONFIG["shaping_keep_fraction"]
n_full = len(shaping_full)
n_keep = max(2, int(n_full * frac))
keep_idx = np.linspace(0, n_full - 1, n_keep, dtype=int)
shaping = CoilSet(
    *[shaping_full[i] for i in keep_idx],
    NFP=getattr(shaping_full, "NFP", 1),
    sym=getattr(shaping_full, "sym", False),
)
print(f"Shaping coils (subsampled {frac:.0%}): {len(shaping)}")

coils = MixedCoilSet((encircling, shaping))

# Record half-slew currents for bounds
n_shp = len(shaping)
I_shp_halfslew = np.array([shaping[i].current for i in range(n_shp)])
slew = CONFIG["slew_limit"]
I_lower = I_shp_halfslew - slew
I_upper = I_shp_halfslew + slew

# ==========================================
# LOAD DIVERTOR LEG CURVES
# ==========================================
print("\nLoading divertor leg curves...")
xpt_pts   = np.loadtxt(CONFIG["xpt_pts"])
inner_fps = np.load(CONFIG["inner_fps"])
outer_fps = np.load(CONFIG["outer_fps"])

nmodes = CONFIG["div_nmodes"]
xpt_curve = FourierXYZCoil.from_values(
    current=0, N=nmodes, coords=xpt_pts, basis="xyz", name="x-point"
)
inner_fp_curve = FourierXYZCoil.from_values(
    current=0, N=nmodes, coords=inner_fps, basis="xyz", name="inner-fps"
)
outer_fp_curve = FourierXYZCoil.from_values(
    current=0, N=nmodes, coords=outer_fps, basis="xyz", name="outer-fps"
)
print(f"  X-point:  {len(xpt_pts)} pts -> {nmodes} Fourier modes")
print(f"  Inner FP: {len(inner_fps)} pts -> {nmodes} Fourier modes")
print(f"  Outer FP: {len(outer_fps)} pts -> {nmodes} Fourier modes")

# QuadratureGrid for plasma field evaluation in FieldNormalError (eq_fixed=False)
eq_grid_div = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)

# ==========================================
# BUILD OBJECTIVES
# ==========================================
print("\nBuilding objectives and constraints...")

grid_lcfs = LinearGrid(
    rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False
)
grid_encircling = LinearGrid(N=max(encircling[0].N, 64))
grid_shaping    = LinearGrid(N=max(shaping[0].N, 32))
grid_boot = LinearGrid(
    M=eq.M_grid, N=eq.M_grid, NFP=eq.NFP, sym=eq.sym,
    rho=np.linspace(1 / eq.L_grid, 1, eq.L_grid) - 1 / (2 * eq.L_grid),
)
grid_qs = ConcentricGrid(
    L=eq.L_grid, M=eq.M_grid, N=eq.N_grid,
    NFP=eq.NFP, sym=eq.sym, axis=False,
)

objective = ObjectiveFunction(
    (
        BoundaryError(
            eq=eq, field=coils, target=0,
            source_grid=grid_lcfs, eval_grid=grid_lcfs,
            field_grid=[grid_encircling, grid_shaping],
            field_fixed=False,
            weight=CONFIG["weight_boundary"],
        ),
        QuasisymmetryTwoTerm(
            eq=eq,
            helicity=CONFIG["helicity"],
            grid=grid_qs,
            weight=CONFIG["weight_qs"],
        ),
        BootstrapRedlConsistency(
            eq=eq, grid=grid_boot, helicity=CONFIG["helicity"],
            weight=CONFIG["weight_bootstrap"],
        ),
        FixCoilCurrent(
            coil=coils, indices=[False, True],
            bounds=(I_lower, I_upper),
            weight=CONFIG["weight_coil_current"],
        ),
        FieldNormalError(
            field=coils,
            curve_start=xpt_curve,
            curve_end=inner_fp_curve,
            nquad=CONFIG["div_nquad"],
            n_radial=CONFIG["div_n_radial"],
            extent=CONFIG["div_extent"],
            min_extent=CONFIG["div_min_extent"],
            eq=eq,
            eq_fixed=False,
            field_fixed=False,
            eq_grid=eq_grid_div,
            basis="xyz",
            target=0,
            weight=CONFIG["weight_divertor"],
            name="Inner Leg B*n",
        ),
        FieldNormalError(
            field=coils,
            curve_start=xpt_curve,
            curve_end=outer_fp_curve,
            nquad=CONFIG["div_nquad"],
            n_radial=CONFIG["div_n_radial"],
            extent=CONFIG["div_extent"],
            min_extent=CONFIG["div_min_extent"],
            eq=eq,
            eq_fixed=False,
            field_fixed=False,
            eq_grid=eq_grid_div,
            basis="xyz",
            target=0,
            weight=CONFIG["weight_divertor"],
            name="Outer Leg B*n",
        ),
    ),
    deriv_mode="batched",
    jac_chunk_size=CONFIG["jac_chunk_size"],
)

constraints = (
    # Coils: encircling fully fixed, shaping geometry fixed + current free
    FixParameters(
        coils,
        [
            {"X": True, "Y": True, "Z": True, "current": True},
            {"X": True, "Y": True, "Z": True, "current": False},
        ],
    ),
    # Kinetic profiles fixed
    FixAtomicNumber(eq=eq),
    FixElectronDensity(eq=eq),
    FixElectronTemperature(eq=eq),
    FixIonTemperature(eq=eq),
    # First two current coefficients fixed, rest free for bootstrap
    FixCurrent(eq=eq, indices=np.array([0, 1])),
    # Scaled Psi fixed
    FixPsi(eq=eq),
    # MHD force balance
    ForceBalance(eq=eq),
)

# Monkey-patch lsqtr BEFORE building
call_log, restore_lsqtr = monkey_patch_lsqtr()

objective.build(verbose=2)

x0 = np.asarray(objective.x(eq, coils)).copy()
print(f"DOFs: {len(x0)}, dim_f: {objective.dim_f}")

# ==========================================
# PER-OBJECTIVE JACOBIAN PROFILING
# ==========================================
print("\nPer-objective Jacobian profiling...")
constants = objective.constants  # property!
obj_results = time_per_objective_jac(objective, x0, constants)
print_objective_breakdown(obj_results, label="(Case 4: single-stage)")

# ==========================================
# RUN OPTIMIZER
# ==========================================
print(f"\nRunning proximal-lsq-exact (maxiter={CONFIG['maxiter']}, "
      f"maxiter_inner={CONFIG['maxiter_inner']})...")

opt = Optimizer("proximal-lsq-exact")
t0 = time.perf_counter()
[eq_out, coils_out], result = opt.optimize(
    things=[eq, coils],
    objective=objective,
    constraints=constraints,
    maxiter=CONFIG["maxiter"],
    ftol=CONFIG["ftol"],
    gtol=1e-16,
    options={
        "solve_options": {
            "ftol": CONFIG["ftol"],
            "xtol": CONFIG["xtol"],
            "gtol": CONFIG["gtol"],
            "maxiter": CONFIG["maxiter_inner"],
        },
    },
    verbose=3,
    copy=True,
)
wall = time.perf_counter() - t0

print_lsqtr_breakdown(call_log, label="(Case 4: single-stage)")
print(f"\nTotal wall time: {wall:.1f}s")

save_results({
    "case": "case_4_single_stage",
    "wall_time_s": wall,
    "dofs": len(x0),
    "dim_f": objective.dim_f,
    "call_log": call_log,
    "objective_breakdown": obj_results,
}, "profiling/results_case_4.json")

restore_lsqtr()
