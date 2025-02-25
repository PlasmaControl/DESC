import os
import sys

# Add the path to the parent directory to augment search for module
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

# These will be used for diving the single CPU into multiple virtual CPUs
# such that JAX and XLA thinks there are multiple devices
from desc import _set_cpu_count, set_device

num_device = 3
_set_cpu_count(num_device)
set_device("cpu", num_device=num_device)

import numpy as np
from mpi4py import MPI

from desc.backend import jax, jnp
from desc.examples import get
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
)
from desc.optimize import Optimizer

if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()

    eq = get("precise_QA")
    eq.change_resolution(3, 3, 3, 6, 6, 6)

    # create two grids with different rho values, this will effectively separate
    # the quasisymmetry objective into two parts
    grid1 = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=jnp.linspace(0.2, 0.5, 4), sym=True
    )
    grid2 = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=jnp.linspace(0.6, 1.0, 6), sym=True
    )

    # when using parallel objectives, the user needs to supply the device_id
    obj1 = QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid1, device_id=0)
    obj2 = QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid2, device_id=1)
    obj3 = AspectRatio(eq=eq, target=8, weight=100, device_id=2)

    objs = [obj1, obj2, obj3]
    # this part will probably be automatized in the future
    for obji in objs:
        obji.build(verbose=3)
        obji = jax.device_put(obji, obji._device)
        obji.things[0] = eq

    # Parallel objective function needs the MPI communicator
    objective = ObjectiveFunction(objs, mpi=MPI)
    objective.build(verbose=3)

    # we will fix some modes as usual
    k = 1
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
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixPsi(eq=eq),
        FixCurrent(eq=eq),
    )
    optimizer = Optimizer("proximal-lsq-exact")

    # Until this line, the code is performed on all ranks, so it might print some
    # information multiple times. The following part will only be performed on the
    # master rank

    # this context manager will put the workers in a loop to listen to the master
    # to compute the objective function and its derivatives
    with objective as objective:
        # apart from cost evaluation and derivatives, everything else will be only
        # performed on the master rank
        if rank == 0:
            eq.optimize(
                objective=objective,
                constraints=constraints,
                optimizer=optimizer,
                maxiter=1,
                verbose=3,
                options={
                    "initial_trust_ratio": 1.0,
                },
            )

    # if you put a code here, it will be performed on all ranks
