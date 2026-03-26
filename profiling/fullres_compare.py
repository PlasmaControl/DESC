#!/usr/bin/env python
"""Full-resolution batched vs blocked comparison.

Usage:
  CUDA_VISIBLE_DEVICES=X CASE=1 MODE=batched python profiling/fullres_compare.py
  CUDA_VISIBLE_DEVICES=X CASE=3 MODE=blocked python profiling/fullres_compare.py

CASE=1: Coil opt (lsq-exact), full resolution
CASE=3: Free boundary (proximal-lsq-exact), M=12, N=18
CASE=4: Single-stage (proximal-lsq-exact), M=12, N=18
"""
import os
import sys
import time

CASE = int(os.environ.get("CASE", "1"))
MODE = os.environ.get("MODE", "blocked")

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from instrument import setup_gpu
setup_gpu()

import numpy as np

print(f"Case {CASE}, deriv_mode={MODE}, full resolution")

if CASE == 1:
    sys.path.insert(0, os.path.expanduser("~/finding_better_basins"))
    from configs import build_config, build_objectives, load_data, setup_jax
    config = build_config(platform="perlmutter", lightweight=False, seed=0)
    setup_jax(config)
    data = load_data(config)
    problem = build_objectives(data, config)

    from desc.objectives import ObjectiveFunction
    from desc.optimize import Optimizer

    obj_fn = ObjectiveFunction(
        tuple(problem["objective_list"]),
        deriv_mode=MODE,
        jac_chunk_size=1 if MODE == "batched" else None,
    )
    obj_fn.build(verbose=0)
    x0 = obj_fn.x(*problem["things"])
    print(f"DOFs: {len(x0)}, dim_f: {obj_fn.dim_f}")

    opt = Optimizer("lsq-exact")
    t0 = time.perf_counter()
    _things_out, result = opt.optimize(
        things=problem["things"],
        objective=obj_fn,
        constraints=problem["constraints"],
        maxiter=3, ftol=0, xtol=0, gtol=0, verbose=2, copy=True,
    )
    wall = time.perf_counter() - t0
    print(f"\nCase {CASE} ({MODE}): {wall:.1f}s, njev={result.njev}")

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
    DATA_DIR = os.path.join(
        HOME, "dynamic-accessibility/half_beta_half_slew_eq/"
        "second_free_boundary_proper-tf"
    )
    print("Loading coils...")
    encircling = load(os.path.join(DATA_DIR, "data/midbeta/encircling_midbeta.h5"))
    shaping = load(os.path.join(DATA_DIR, "data/midbeta/shaping_midbeta.h5"))
    print("Loading equilibrium...")
    eq0 = load(os.path.join(DATA_DIR, "data/midbeta/equil_G1635_DESC_fixed.h5"))
    if isinstance(eq0, EquilibriaFamily):
        eq0 = eq0[-1]
    eq = eq0.copy()
    eq.change_resolution(M=12, N=18)
    eq.Psi = 0.925 * eq.Psi
    print(f"Equilibrium: L={eq.L}, M={eq.M}, N={eq.N}, NFP={eq.NFP}")

    grid_enc = LinearGrid(N=max(encircling[0].N, 32))
    grid_shp = LinearGrid(N=max(shaping[0].N, 16))
    grid_lcfs = LinearGrid(
        rho=np.array([1.0]), M=12, N=18, NFP=eq.NFP, sym=False
    )

    objective = ObjectiveFunction(
        BoundaryError(
            eq=eq, field=[encircling, shaping], target=0,
            source_grid=grid_lcfs, eval_grid=grid_lcfs,
            field_grid=[grid_enc, grid_shp], field_fixed=True,
        ),
        deriv_mode=MODE,
        jac_chunk_size=8 if MODE == "batched" else None,
    )
    constraints = (
        FixAtomicNumber(eq=eq), FixCurrent(eq=eq), FixElectronDensity(eq=eq),
        FixElectronTemperature(eq=eq), FixIonTemperature(eq=eq),
        FixPsi(eq=eq), ForceBalance(eq=eq),
    )
    objective.build(verbose=0)
    x0 = objective.x(eq)
    print(f"DOFs: {len(x0)}, dim_f: {objective.dim_f}")

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
    print(f"\nCase {CASE} ({MODE}): {wall:.1f}s")

elif CASE == 4:
    from desc.io import load
    from desc.equilibrium import EquilibriaFamily
    from desc.coils import CoilSet, MixedCoilSet
    from desc.grid import LinearGrid, ConcentricGrid
    from desc.objectives import (
        BoundaryError, BootstrapRedlConsistency, FixAtomicNumber,
        FixCoilCurrent, FixCurrent, FixElectronDensity,
        FixElectronTemperature, FixIonTemperature, FixParameters,
        FixPsi, ForceBalance, ObjectiveFunction, QuasisymmetryTwoTerm,
    )
    from desc.optimize import Optimizer

    HOME = os.path.expanduser("~")
    DATA_BASE = os.path.join(
        HOME, "dynamic-accessibility/half_beta_half_slew_eq/"
        "slew_from_halfslew_eq-QS-divLeg"
    )
    print("Loading equilibrium...")
    eq0 = load(os.path.join(
        HOME, "dynamic-accessibility/half_beta_half_slew_eq/"
        "first_slew_from_halfslew_eq-QS/output/single_stage_20260313_171631/"
        "eq_single_stage_QS.h5"
    ))
    if isinstance(eq0, EquilibriaFamily):
        eq0 = eq0[-1]
    eq = eq0.copy()
    eq.change_resolution(M=12, N=18)
    eq.Psi = 0.925 * eq.Psi

    print("Loading coils...")
    encircling = load(os.path.join(DATA_BASE, "data/midbeta/encircling_midbeta.h5"))
    shaping = load(os.path.join(DATA_BASE, "data/midbeta/shaping_midbeta.h5"))
    coils = MixedCoilSet(encircling, shaping)
    print(f"Eq: L={eq.L}, M={eq.M}, N={eq.N}, NFP={eq.NFP}")
    print(f"Coils: enc={len(encircling)}, shp={len(shaping)}")

    grid_lcfs = LinearGrid(
        rho=np.array([1.0]), M=12, N=18, NFP=eq.NFP, sym=False
    )

    objectives = [
        BoundaryError(
            eq=eq, field=coils, field_fixed=True,
            eval_grid=grid_lcfs, source_grid=grid_lcfs,
        ),
        QuasisymmetryTwoTerm(
            eq=eq, helicity=(1, 0),
            grid=ConcentricGrid(L=6, M=6, N=6, NFP=eq.NFP),
        ),
    ]

    constraints = (
        FixCurrent(eq=eq), FixPsi(eq=eq),
        FixElectronDensity(eq=eq), FixElectronTemperature(eq=eq),
        FixIonTemperature(eq=eq), FixAtomicNumber(eq=eq),
        ForceBalance(eq=eq),
    )

    objective = ObjectiveFunction(
        tuple(objectives),
        deriv_mode=MODE,
        jac_chunk_size=1 if MODE == "batched" else None,
    )
    objective.build(verbose=0)
    x0 = objective.x(eq)
    print(f"DOFs: {len(x0)}, dim_f: {objective.dim_f}")

    opt = Optimizer("proximal-lsq-exact")
    t0 = time.perf_counter()
    [eq_out], result = opt.optimize(
        things=eq, objective=objective, constraints=constraints,
        x_scale="ess", maxiter=3, ftol=1e-4, gtol=1e-16,
        options={
            "solve_options": {
                "ftol": 1e-4, "xtol": 1e-6, "gtol": 1e-6, "maxiter": 3,
            },
        },
        verbose=2, copy=True,
    )
    wall = time.perf_counter() - t0
    print(f"\nCase {CASE} ({MODE}): {wall:.1f}s")
