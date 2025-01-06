"""Example script for recreating the precise QA configuration of Landreman and Paul."""

from desc import set_device

set_device("gpu")

import pickle

import jax
import numpy as np

import desc
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import ConcentricGrid, LinearGrid
from desc.objectives import (
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    MagneticWell,
    MeanCurvature,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
    RotationalTransform,
    Volume,
)
from desc.optimize import Optimizer

eq = desc.io.load("precise_QA_AL.h5")[-2]
eqfam = EquilibriaFamily(eq)

grid = LinearGrid(
    M=eq.M_grid,
    N=eq.N_grid,
    NFP=eq.NFP,
    rho=np.linspace(0.1, 1, 10),
    sym=True,
)

# optimize in steps
for n in range(1, eq.M + 1, 2):
    jax.clear_caches()
    print("\n==================================")
    print("Optimizing boundary modes M,N <= {}".format(n))
    print("====================================")

    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(
                eq=eqfam[-1], helicity=(1, 0), grid=grid, normalize=True
            ),
        ),
        verbose=0,
    )
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > n, :
            ],
        )
    )
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > n, :
    ]
    constraints = (
        ForceBalance(eq=eqfam[-1], weight=100),  # J x B - grad(p) = 0
        FixBoundaryR(eq=eqfam[-1], modes=R_modes),
        FixBoundaryZ(eq=eqfam[-1], modes=Z_modes),
        FixPressure(eq=eqfam[-1]),
        FixCurrent(eq=eqfam[-1]),
        FixPsi(eq=eqfam[-1]),
        AspectRatio(eq=eqfam[-1], target=6),
        RotationalTransform(eq=eqfam[-1], bounds=(0.43, 0.5)),
        Volume(eq=eqfam[-1], target=eqfam[0].compute("V")["V"]),
        MagneticWell(eq=eqfam[-1], bounds=(lambda x: x * 1e-4, np.inf)),
        MeanCurvature(eq=eqfam[-1], bounds=(-np.inf, 0.5), normalize=False),
    )
    optimizer = Optimizer("lsq-auglag")
    eq_new, out = eqfam[-1].optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        maxiter=1000,
        ftol=1e-6,
        xtol=1e-6,
        ctol=1e-6,
        verbose=3,
        copy=True,
        options={
            "initial_multipliers": (0 if n == 1 else out["y"]),
            "initial_penalty_parameter": (10 if n == 1 else out["penalty_param"]),
            "alpha_eta": 1,
            "alpha_omega": 0.2,
            "beta_eta": 0.2,
            "beta_omega": 0.2,
            "eta": 1e-2,
            "omega": 100,
            "initial_trust_radius": "scipy",
        },
    )
    eqfam.append(eq_new)
    eqfam[-1].solve(copy=False, verbose=3)
    eqfam.save("precise_QA_AL_well2.h5")

with open("precise_QA_AL_well2.pkl", "wb+") as f:
    pickle.dump(out, f)

eq_new = eqfam[-1].copy()
eq_new.change_resolution(12, 12, 12, 24, 24, 24)
eq_new = solve_continuation_automatic(eq_new)[-1]
eqfam.append(eq_new)
eqfam.save("precise_QA_AL_well2.h5")
