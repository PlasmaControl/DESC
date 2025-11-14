import os
import sys

# Add the path to the parent directory to augment search for module
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../../"))

import numpy as np
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

from desc import config as desc_config
from desc.backend import jax, print_backend_info
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import ForceBalance, ObjectiveFunction
from desc.objectives.getters import get_fixed_boundary_constraints

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
            f"Rank {rank} can see {jax.local_devices(backend='gpu')} "
            f"and {jax.local_devices(backend='cpu')}\n"
        )
    else:
        print(f"Rank {rank} can see {jax.local_devices(backend='cpu')}\n")

    if rank == 0:
        print("====== BACKEND INFO ======")
        print_backend_info()
        print("\n")

    eq = get("HELIOTRON")
    if desc_config["kind"] == "cpu":
        # for local testing use lower resolution
        eq.change_resolution(M=3, N=2, M_grid=6, N_grid=4)

    # setup 2 grids for 2 objectives covering different flux surfaces
    rhos = np.linspace(0.1, 1.0, eq.L_grid)
    grid1 = LinearGrid(
        rho=rhos[: rhos.size // 2],
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=eq.NFP,
    )
    grid2 = LinearGrid(
        rho=rhos[rhos.size // 2 :],
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=eq.NFP,
    )
    obj = ObjectiveFunction(
        [
            ForceBalance(eq, grid=grid1, device_id=0),
            ForceBalance(eq, grid=grid2, device_id=1),
        ],
        mpi=MPI,
        deriv_mode="blocked",
    )
    obj.build()
    cons = get_fixed_boundary_constraints(eq)

    # Until this line, the code is performed on all ranks, so it might print some
    # information multiple times. The following part will only be performed on the
    # master rank

    # this context manager will put the workers in a loop to listen to the master
    # to compute the objective function and its derivatives
    with obj:
        # apart from cost evaluation and derivatives, everything else will be only
        # performed on the master rank
        if rank == 0:
            eq.solve(
                objective=obj,
                constraints=cons,
                maxiter=10,
                ftol=0,
                gtol=0,
                xtol=0,
                verbose=3,
            )

    # if you put a code here, it will be performed on all ranks
