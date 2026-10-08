"""Example script for recreating the "precise QA" configuration of Landreman and Paul.

Note that this resembles their optimization process in SIMSOPT, but the final optimized
equilibrium is slightly different from their VMEC solution.
"""

from desc import set_device

# need to do this before importing other DESC stuff so JAX initializes properly
set_device("gpu")

import numpy as np

from desc.continuation import solve_continuation_automatic
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.objectives import (
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
    RotationalTransform,
)
from desc.optimize import Optimizer

# create initial surface. Aspect ratio ~ 6, circular cross section with slight
# axis torsion to make it nonplanar
surf = FourierRZToroidalSurface(
    R_lmn=[1, 0.166, 0.1],
    Z_lmn=[-0.166, -0.1],
    modes_R=[[0, 0], [1, 0], [0, 1]],
    modes_Z=[[-1, 0], [0, -1]],
    NFP=2,
)
# create initial equilibrium. Psi chosen to give B ~ 1 T. Could also give profiles here,
# default is zero pressure and zero current
eq = Equilibrium(M=8, N=8, Psi=0.087, surface=surf)
# this is usually all you need to solve a fixed boundary equilibrium
eq = solve_continuation_automatic(eq, objective="force", bdry_step=0.5, verbose=3)[-1]
# it will be helpful to store intermediate results
eqfam = EquilibriaFamily(eq)

# create grid where we want to minimize QS error. Here we do it on 3 surfaces
grid = LinearGrid(M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True)

# optimize in steps
for k in range(1, eq.M + 1):
    print("\n==================================")
    print("Optimizing boundary modes M,N <= {}".format(k))
    print("====================================")
    objective = ObjectiveFunction(
        (
            # pass in the grid we defined, and don't forget the target helicity!
            QuasisymmetryTwoTerm(
                eq=eqfam[-1], helicity=(1, 0), grid=grid, normalize=False
            ),
            AspectRatio(eq=eqfam[-1], target=6, weight=1e1, normalize=False),
            # this targets a profile pointwise, which is ok because we expect it to be
            # fairly flat
            RotationalTransform(eq=eqfam[-1], target=0.42, weight=10, normalize=False),
            # we could optionally set normalize=True which would compute things in
            # normalized/dimensionless units, effectively changing the weights
        ),
    )
    # as opposed to SIMSOPT and STELLOPT where variables are assumed fixed, in DESC
    # we assume variables are free. Here we decide which ones to fix, starting with
    # the major radius (R mode = [0,0,0]) and all modes with m,n > k
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
    # next we create the constraints, using the mode number arrays just created
    # if we didn't pass those in, it would fix all the modes (like for the profiles)
    constraints = (
        ForceBalance(eq=eqfam[-1]),  # J x B - grad(p) = 0
        FixBoundaryR(eq=eqfam[-1], modes=R_modes),
        FixBoundaryZ(eq=eqfam[-1], modes=Z_modes),
        FixPressure(eq=eqfam[-1]),
        FixCurrent(eq=eqfam[-1]),
        FixPsi(eq=eqfam[-1]),
    )
    # this is the default optimizer, which re-solves the equilibrium at each step
    optimizer = Optimizer("proximal-lsq-exact")
    # we get a new equilibrium by optimizing the old one and passing copy=True.
    # otherwise, we could modify the original equilibrium in place
    eq_new, out = eqfam[-1].optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        maxiter=50,
        verbose=3,
        copy=True,
        options={
            # sometimes the default initial trust radius is too big, allowing the
            # optimizer to take too large a step in a bad direction. If this happens,
            # we can manually specify a smaller starting radius.
            "initial_trust_radius": 0.5,
        },
    )
    # add our new equilibrium to the family
    eqfam.append(eq_new)

# save all the steps of the optimization for later analysis
eqfam.save("precise_QA_output.h5")
