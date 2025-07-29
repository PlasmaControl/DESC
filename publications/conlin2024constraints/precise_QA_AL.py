"""Example script for recreating the precise QA configuration of Landreman and Paul."""

from desc import set_device

set_device("gpu")

import pickle

import jax
import numpy as np

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
    MeanCurvature,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
    RotationalTransform,
    Volume,
)
from desc.optimize import Optimizer

surf = FourierRZToroidalSurface(
    R_lmn=[1, 0.166, 0.0],
    Z_lmn=[-0.166, -0.0],
    modes_R=[[0, 0], [1, 0], [0, 1]],
    modes_Z=[[-1, 0], [0, -1]],
    NFP=2,
)
eq = Equilibrium(M=10, N=10, Psi=0.087, surface=surf)
eq = solve_continuation_automatic(eq, objective="force", bdry_step=0.5, verbose=3)[-1]
eqfam = EquilibriaFamily(eq)

grid = LinearGrid(
    M=eq.M_grid,
    N=eq.N_grid,
    NFP=eq.NFP,
    rho=np.linspace(0.1, 1, 10),
    sym=True,
)

# optimize in steps
for k in range(1, eq.M + 1):
    jax.clear_caches()

    print("\n==================================")
    print("Optimizing boundary modes M,N <= {}".format(k))
    print("====================================")

    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(
                eq=eqfam[-1], helicity=(1, 0), grid=grid, normalize=False
            ),
        ),
        verbose=0,
    )
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :
            ],
        )
    )
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :
    ]
    constraints = (
        ForceBalance(eq=eqfam[-1]),  # J x B - grad(p) = 0
        FixBoundaryR(eq=eqfam[-1], modes=R_modes),
        FixBoundaryZ(eq=eqfam[-1], modes=Z_modes),
        FixPressure(eq=eqfam[-1]),
        FixCurrent(eq=eqfam[-1]),
        FixPsi(eq=eqfam[-1]),
        AspectRatio(eq=eqfam[-1], target=6),
        Volume(eq=eqfam[-1], target=eqfam[0].compute("V")["V"]),
        RotationalTransform(eq=eqfam[-1], bounds=(0.43, 0.5)),
        MeanCurvature(eq=eqfam[-1], bounds=(-np.inf, 0)),
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
            "initial_multipliers": (0 if k == 1 else out["y"]),
            "initial_penalty_parameter": (10 if k == 1 else out["penalty_param"]),
            "alpha_eta": 1,
            "alpha_omega": 0.2,
            "beta_eta": 0.2,
            "beta_omega": 0.2,
            "eta": 1e-2,
            "omega": 100,
        },
    )
    eqfam.append(eq_new)
    eqfam[-1].solve(copy=False, verbose=3)
    eqfam.save("precise_QA_AL.h5")

with open("precise_QA_AL.pkl", "wb+") as f:
    pickle.dump(out, f)

eq_new = eqfam[-1].copy()
eq_new.change_resolution(12, 12, 12, 24, 24, 24)
eq_new = solve_continuation_automatic(eq_new)[-1]
eqfam.append(eq_new)
eqfam.save("precise_QA_AL.h5")
