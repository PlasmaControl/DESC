import os
import sys

# Add the path to the parent directory to augment search for module
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../../"))

from mpi4py import MPI

from desc import _set_cpu_count, set_device

kind = "cpu"  # or "gpu"
num_device = 2
# ====== Using CPUs ======
# These will be used for diving the single CPU into multiple virtual CPUs
# such that JAX and XLA thinks there are multiple devices
if kind == "cpu":
    # !!! If you have multiple CPUs, you shouldn't call `_set_cpu_count` !!!
    _set_cpu_count(num_device)
    set_device("cpu", num_device=num_device, mpi=MPI)

# ====== Using GPUs ======
# When we have multiple processes using the same devices (for example, 3 processes
# using 3 GPUs), each process will try to pre-allocate 75% of the GPU memory which will
# cause the memory allocation to fail. To avoid this, we can set the allocator to `platform`
# such that there is no pre-allocation. This is a bit conservative (and probably there is room
# for improvement), but if a process needs more memory, it can use more memory on the fly.
elif kind == "gpu":
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    set_device("gpu", num_device=num_device)


import numpy as np

from desc import config as desc_config
from desc.backend import jax, jnp, print_backend_info
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
    size = MPI.COMM_WORLD.Get_size()
    if rank == 0:
        print(f"====== TOTAL OF {size} RANKS ======")

    # see which rank is running on which device
    # Note: JAX has 2 functions for this: `jax.devices()` and `jax.local_devices()`
    # `jax.devices()` will return all devices available to JAX, while `jax.local_devices()`
    # will return only the devices that are available to the current process. This is
    # useful when you have multiple processes running on multiple nodes and you want
    # to see which devices are available to each process.
    if desc_config["kind"] == "gpu":
        print(
            f"Rank {rank} is running on {jax.local_devices(backend='gpu')} "
            f"and {jax.local_devices(backend='cpu')}\n"
        )
    else:
        print(f"Rank {rank} is running on {jax.local_devices(backend='cpu')}\n")

    if rank == 0:
        print("====== BACKEND INFO ======")
        print_backend_info()
        print("\n")

    eq = get("precise_QA")
    if desc_config["kind"] == "cpu":
        eq.change_resolution(M=3, N=2, M_grid=6, N_grid=4)

    # create two grids with different rho values, this will effectively separate
    # the quasisymmetry objective into two parts
    grid1 = LinearGrid(
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=eq.NFP,
        rho=jnp.linspace(0.2, 0.5, 4),
        sym=True,
    )
    grid2 = LinearGrid(
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=eq.NFP,
        rho=jnp.linspace(0.6, 1.0, 6),
        sym=True,
    )

    # when using parallel objectives, the user needs to supply the device_id
    obj1 = QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid1, device_id=0)
    obj2 = QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid2, device_id=1)
    obj3 = AspectRatio(eq=eq, target=8, weight=100, device_id=0)
    objs = [obj1, obj2, obj3]

    # Parallel objective function needs the MPI communicator
    # If you don't specify `deriv_mode=blocked`, you will get a warning and DESC will
    # automatically switch to `blocked`.
    objective = ObjectiveFunction(
        objs, deriv_mode="blocked", mpi=MPI, rank_per_objective=np.array([0, 1, 0])
    )
    if rank == 0:
        objective.build(verbose=3)
    else:
        objective.build(verbose=0)

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
    with objective:
        # apart from cost evaluation and derivatives, everything else will be only
        # performed on the master rank
        if rank == 0:
            eq.optimize(
                objective=objective,
                constraints=constraints,
                optimizer=optimizer,
                maxiter=3,
                verbose=3,
                options={
                    "initial_trust_ratio": 1.0,
                },
            )

    # if you put a code here, it will be performed on all ranks
