import os
import sys

# Add the path to the parent directory to augment search for module
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

from desc import _set_cpu_count, set_device

# ====== Using CPUs ======
num_device = 4
# These will be used for diving the single CPU into multiple virtual CPUs
# such that JAX and XLA thinks there are multiple devices
# If you have multiple CPUs, you don't need to call `_set_cpu_count`
_set_cpu_count(num_device)
set_device("cpu", num_device=num_device)

# ====== Using GPUs ======
# When we have multiple processing using the same devices (for example, 3 processes
# using 3 GPUs), each process will try to pre-allocate 75% of the GPU memory which will
# cause the memory allocation to fail. To avoid this, we can set the memory fraction
# to 1/(num_device + 2) which will allow each process to allocate 1/(num_device + 2) of
# the GPU memory. This is a bit conservative, but if a process needs more memory, it can
# allocate more memory on the fly.
#
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(1 / (num_device + 2))
# set_device("gpu", num_device=num_device)

from mpi4py import MPI

from desc import config as desc_config
from desc.backend import jax, print_backend_info
from desc.examples import get
from desc.objectives.getters import (
    get_fixed_boundary_constraints,
    get_parallel_forcebalance,
)

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
            f"Rank {rank} is running on {jax.local_devices(backend="gpu")} "
            f"and {jax.local_devices(backend="cpu")}"
        )
    else:
        print(f"Rank {rank} is running on {jax.local_devices(backend='cpu')}")
    print_backend_info()

    eq = get("HELIOTRON")
    eq.change_resolution(6, 6, 6, 12, 12, 12)

    # this will create a parallel objective function
    # user can create their own parallel objective function as well which will be
    # shown in the next example
    obj = get_parallel_forcebalance(eq, num_device=num_device, mpi=MPI, verbose=1)
    cons = get_fixed_boundary_constraints(eq)

    # Until this line, the code is performed on all ranks, so it might print some
    # information multiple times. The following part will only be performed on the
    # master rank

    # this context manager will put the workers in a loop to listen to the master
    # to compute the objective function and its derivatives
    with obj as obj:
        # apart from cost evaluation and derivatives, everything else will be only
        # performed on the master rank
        if rank == 0:
            eq.solve(
                objective=obj,
                constraints=cons,
                maxiter=3,
                ftol=0,
                gtol=0,
                xtol=0,
                verbose=3,
            )

    # if you put a code here, it will be performed on all ranks
