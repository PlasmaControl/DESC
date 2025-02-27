import os
import sys

# Add the path to the parent directory to augment search for module
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

# These will be used for diving the single CPU into multiple virtual CPUs
# such that JAX and XLA thinks there are multiple devices
from desc import _set_cpu_count, set_device

num_device = 4
_set_cpu_count(num_device)
set_device("cpu", num_device=num_device)

from mpi4py import MPI

from desc.examples import get
from desc.objectives.getters import (
    get_fixed_boundary_constraints,
    get_parallel_forcebalance,
)

if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
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
