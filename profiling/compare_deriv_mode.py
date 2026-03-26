#!/usr/bin/env python
"""Compare batched vs blocked deriv_mode on a full optimizer run.

Usage: CUDA_VISIBLE_DEVICES=X CASE=N python profiling/compare_deriv_mode.py

CASE=1: Coil opt (lsq-auglag)
CASE=3: Free boundary (proximal-lsq-exact)
CASE=4: Single-stage (proximal-lsq-exact)
"""
import os
import sys
import time

CASE = int(os.environ.get("CASE", "1"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from instrument import setup_gpu
setup_gpu()

import numpy as np

# ============================================================
# Case-specific setup
# ============================================================

if CASE in (1, 2):
    sys.path.insert(0, os.path.expanduser("~/finding_better_basins"))
    from configs import build_config, build_objectives, load_data, setup_jax
    config = build_config(platform="perlmutter", lightweight=True, seed=0)
    setup_jax(config)
    data = load_data(config)
    problem = build_objectives(data, config)

    from desc.objectives import ObjectiveFunction
    from desc.optimize import Optimizer

    def run_case(deriv_mode):
        obj_fn = ObjectiveFunction(
            tuple(problem["objective_list"]),
            deriv_mode=deriv_mode,
            jac_chunk_size=1 if deriv_mode == "batched" else None,
        )
        obj_fn.build(verbose=0)
        opt = Optimizer("lsq-auglag")
        t0 = time.perf_counter()
        _things_out, result = opt.optimize(
            things=problem["things"],
            objective=obj_fn,
            constraints=problem["constraints"],
            maxiter=5, ftol=0, xtol=0, gtol=0, verbose=2, copy=True,
        )
        wall = time.perf_counter() - t0
        return wall, result

elif CASE == 3:
    from desc.io import load
    from desc.equilibrium import EquilibriaFamily
    from desc.grid import LinearGrid
    from desc.objectives import (
        BoundaryError, FixAtomicNumber, FixCurrent, FixElectronDensity,
        FixElectronTemperature, FixIonTemperature, FixPsi, ForceBalance,
        ObjectiveFunction,
    )
    from desc.optimize import Optimizer

    HOME = os.path.expanduser("~")
    DATA_DIR = os.path.join(HOME, "dynamic-accessibility/half_beta_half_slew_eq/second_free_boundary_proper-tf")
    encircling = load(os.path.join(DATA_DIR, "data/midbeta/encircling_midbeta.h5"))
    shaping = load(os.path.join(DATA_DIR, "data/midbeta/shaping_midbeta.h5"))
    eq0 = load(os.path.join(DATA_DIR, "data/midbeta/equil_G1635_DESC_fixed.h5"))
    if isinstance(eq0, EquilibriaFamily):
        eq0 = eq0[-1]

    grid_lcfs = LinearGrid(rho=np.array([1.0]), M=8, N=8, NFP=eq0.NFP, sym=False)
    grid_enc = LinearGrid(N=max(encircling[0].N, 32))
    grid_shp = LinearGrid(N=max(shaping[0].N, 16))

    def run_case(deriv_mode):
        eq = eq0.copy()
        eq.Psi = 0.925 * eq.Psi
        objective = ObjectiveFunction(
            BoundaryError(
                eq=eq, field=[encircling, shaping], target=0,
                source_grid=grid_lcfs, eval_grid=grid_lcfs,
                field_grid=[grid_enc, grid_shp], field_fixed=True,
            ),
            deriv_mode=deriv_mode,
            jac_chunk_size=8 if deriv_mode == "batched" else None,
        )
        constraints = (
            FixAtomicNumber(eq=eq), FixCurrent(eq=eq), FixElectronDensity(eq=eq),
            FixElectronTemperature(eq=eq), FixIonTemperature(eq=eq),
            FixPsi(eq=eq), ForceBalance(eq=eq),
        )
        objective.build(verbose=0)
        opt = Optimizer("proximal-lsq-exact")
        t0 = time.perf_counter()
        [eq_out], result = opt.optimize(
            things=eq, objective=objective, constraints=constraints,
            x_scale="ess", maxiter=3, ftol=1e-4, gtol=1e-16,
            options={"solve_options": {"ftol": 1e-4, "xtol": 1e-6, "gtol": 1e-6, "maxiter": 5}},
            verbose=2, copy=True,
        )
        wall = time.perf_counter() - t0
        return wall, result

elif CASE == 4:
    # Simplified single-stage (BoundaryError + QS only)
    from desc.io import load
    from desc.equilibrium import EquilibriaFamily
    from desc.coils import CoilSet, MixedCoilSet
    from desc.grid import LinearGrid, ConcentricGrid
    from desc.objectives import (
        BoundaryError, FixAtomicNumber, FixCurrent, FixElectronDensity,
        FixElectronTemperature, FixIonTemperature, FixPsi, ForceBalance,
        ObjectiveFunction, QuasisymmetryTwoTerm,
    )
    from desc.optimize import Optimizer

    HOME = os.path.expanduser("~")
    DATA_BASE = os.path.join(HOME, "dynamic-accessibility/half_beta_half_slew_eq/slew_from_halfslew_eq-QS-divLeg")
    eq0 = load(os.path.join(
        HOME, "dynamic-accessibility/half_beta_half_slew_eq/"
        "first_slew_from_halfslew_eq-QS/output/single_stage_20260313_171631/"
        "eq_single_stage_QS.h5"
    ))
    if isinstance(eq0, EquilibriaFamily):
        eq0 = eq0[-1]
    encircling = load(os.path.join(DATA_BASE, "data/midbeta/encircling_midbeta.h5"))
    shaping_raw = load(os.path.join(DATA_BASE, "data/midbeta/shaping_midbeta.h5"))
    n_keep = max(2, int(len(shaping_raw) * 0.02))
    indices = np.linspace(0, len(shaping_raw) - 1, n_keep, dtype=int)
    shaping = CoilSet(*[shaping_raw[int(i)] for i in indices])
    coils = MixedCoilSet(encircling, shaping)

    grid_lcfs = LinearGrid(rho=np.array([1.0]), M=6, N=6, NFP=eq0.NFP, sym=False)

    def run_case(deriv_mode):
        eq = eq0.copy()
        eq.Psi = 0.925 * eq.Psi
        objectives = [
            BoundaryError(eq=eq, field=coils, field_fixed=True,
                          eval_grid=grid_lcfs, source_grid=grid_lcfs),
            QuasisymmetryTwoTerm(eq=eq, helicity=(1, 0),
                                 grid=ConcentricGrid(L=2, M=2, N=2, NFP=eq.NFP)),
        ]
        constraints = (
            FixCurrent(eq=eq), FixPsi(eq=eq),
            FixElectronDensity(eq=eq), FixElectronTemperature(eq=eq),
            FixIonTemperature(eq=eq), FixAtomicNumber(eq=eq),
            ForceBalance(eq=eq),
        )
        objective = ObjectiveFunction(tuple(objectives), deriv_mode=deriv_mode,
                                       jac_chunk_size=1 if deriv_mode == "batched" else None)
        objective.build(verbose=0)
        opt = Optimizer("proximal-lsq-exact")
        t0 = time.perf_counter()
        [eq_out], result = opt.optimize(
            things=eq, objective=objective, constraints=constraints,
            x_scale="ess", maxiter=3, ftol=1e-4, gtol=1e-16,
            options={"solve_options": {"ftol": 1e-4, "xtol": 1e-6, "gtol": 1e-6, "maxiter": 3}},
            verbose=2, copy=True,
        )
        wall = time.perf_counter() - t0
        return wall, result

# ============================================================
# Run both modes
# ============================================================
print(f"\n{'='*60}")
print(f"Case {CASE}: deriv_mode=batched")
print(f"{'='*60}")
wall_batched, res_batched = run_case("batched")
print(f"\nWall time (batched): {wall_batched:.1f}s")

print(f"\n{'='*60}")
print(f"Case {CASE}: deriv_mode=blocked")
print(f"{'='*60}")
wall_blocked, res_blocked = run_case("blocked")
print(f"\nWall time (blocked): {wall_blocked:.1f}s")

print(f"\n{'='*60}")
print(f"SUMMARY Case {CASE}")
print(f"{'='*60}")
print(f"  batched: {wall_batched:.1f}s")
print(f"  blocked: {wall_blocked:.1f}s")
print(f"  speedup: {wall_batched/wall_blocked:.1f}x")
