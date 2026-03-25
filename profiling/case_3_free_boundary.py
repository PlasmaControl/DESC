#!/usr/bin/env python
"""Case 3: Free boundary equilibrium with proximal-lsq-exact (lightweight)."""
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

# Data paths on Perlmutter
HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(
    HOME, "dynamic-accessibility/half_beta_half_slew_eq/"
    "second_free_boundary_proper-tf"
)
COIL_ENC = os.path.join(DATA_DIR, "data/midbeta/encircling_midbeta.h5")
COIL_SHP = os.path.join(DATA_DIR, "data/midbeta/shaping_midbeta.h5")
EQ_PATH = os.path.join(DATA_DIR, "data/midbeta/equil_G1635_DESC_fixed.h5")

# Load
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

if not isinstance(eq.surface, FourierCurrentPotentialField):
    eq.surface = FourierCurrentPotentialField.from_surface(
        eq.surface, M_Phi=eq.M, N_Phi=eq.N
    )

print(f"Equilibrium: L={eq.L}, M={eq.M}, N={eq.N}, NFP={eq.NFP}")

call_log, restore_lsqtr = monkey_patch_lsqtr()

# Build lightweight objective (small grids)
grid_enc = LinearGrid(N=max(encircling[0].N, 32))
grid_shp = LinearGrid(N=max(shaping[0].N, 16))
grid_lcfs = LinearGrid(rho=np.array([1.0]), M=8, N=8, NFP=eq.NFP, sym=False)

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

# Per-objective Jacobian profiling
print("\nPer-objective Jacobian profiling...")
constants = objective.constants  # property!
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
