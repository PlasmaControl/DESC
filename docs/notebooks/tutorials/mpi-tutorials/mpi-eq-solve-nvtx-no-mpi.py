import os
import sys

# Add the path to the parent directory to augment search for module
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../../"))

import nvtx

from desc import set_device

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
set_device("gpu")
import desc
from desc import config as desc_config
from desc.backend import jax, jnp, print_backend_info
from desc.examples import get
from desc.objectives import ForceBalance, ObjectiveFunction
from desc.objectives.getters import get_fixed_boundary_constraints

if __name__ == "__main__":
    print("====== BACKEND INFO ======")
    print_backend_info()
    print("\n")

    with nvtx.annotate("Setup"):
        eq = get("HELIOTRON")
        rhos = jnp.linspace(0.01, 1.0, eq.L_grid)
        grid1 = desc.grid.LinearGrid(
            rho=rhos[0 : eq.L_grid // 2],
            # kind of experimental way of set giving
            # less grid points to inner part, but seems
            # to make transforms way slower
            # M=int(eq.M_grid * i / num_device), # noqa: E800
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
        )
        grid2 = desc.grid.LinearGrid(
            rho=rhos[eq.L_grid // 2 :],
            # kind of experimental way of set giving
            # less grid points to inner part, but seems
            # to make transforms way slower
            # M=int(eq.M_grid * i / num_device), # noqa: E800
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
        )
        obj = ObjectiveFunction(
            [
                ForceBalance(eq, grid=grid1),
                ForceBalance(eq, grid=grid2),
            ],
            deriv_mode="blocked",
        )
        obj.build()
        cons = get_fixed_boundary_constraints(eq)

    with nvtx.annotate("Solve"):
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
