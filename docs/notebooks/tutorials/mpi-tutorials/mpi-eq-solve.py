import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

from desc import _set_cpu_count, set_device

num_device = 4
_set_cpu_count(num_device)
set_device("cpu", num_device=num_device)

import numpy as np
from mpi4py import MPI

from desc.backend import jax
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import ForceBalance, ObjectiveFunction
from desc.objectives.getters import (
    get_fixed_boundary_constraints,
    get_parallel_forcebalance,
)

if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    eq = get("HELIOTRON")
    eq.change_resolution(6, 6, 6, 12, 12, 12)

    obj = get_parallel_forcebalance(eq, num_device=num_device, mpi=MPI, verbose=1)
    cons = get_fixed_boundary_constraints(eq)
    with obj as obj:
        if rank == 0:
            eq.solve(
                objective=obj,
                constraints=cons,
                maxiter=1,
                ftol=0,
                gtol=0,
                xtol=0,
                verbose=3,
            )
