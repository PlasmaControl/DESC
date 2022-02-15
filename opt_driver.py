import numpy as np

from desc.utils import Timer
from desc.equilibrium import EquilibriaFamily
from desc.objectives import (
    ObjectiveFunction,
    FixedBoundaryR,
    FixedBoundaryZ,
    FixedPressure,
    FixedIota,
    FixedPsi,
    LCFSBoundary,
    QuasisymmetryBoozer,
    QuasisymmetryFluxFunction,
    QuasisymmetryTripleProduct,
)


qs = "C"
order = 1
name = "f{}_or{}".format(qs, order)

path = "/projects/EKOLEMEN/QS/STELLOPT_QS/"
input_path = path + "RBC-5E-02_ZBS-4E-02/output.h5"
output_path = path + "data/eq_" + name + ".h5"

timer = Timer()
timer.start("Total")

# equilibrium
eqf = EquilibriaFamily.load(input_path)
eq = eqf[-1]

# XXX: remove
eq.surface.R_basis._create_idx()
eq.surface.Z_basis._create_idx()

# objective
constraints = (
    FixedBoundaryR(),
    FixedBoundaryZ(),
    FixedPressure(),
    FixedIota(),
    FixedPsi(),
    LCFSBoundary(),
)
if qs == "B":
    objective = ObjectiveFunction(
        QuasisymmetryBoozer(helicity=(1, eq.NFP)), constraints
    )
elif qs == "C":
    objective = ObjectiveFunction(
        QuasisymmetryFluxFunction(helicity=(1, eq.NFP)), constraints
    )
elif qs == "T":
    objective = ObjectiveFunction(QuasisymmetryTripleProduct(), constraints)


opt_subspace = np.zeros((2, eq.Rb_lmn.size + eq.Zb_lmn.size))
opt_subspace[0, eq.surface.R_basis.get_idx(M=1, N=2)] = 1
opt_subspace[0, eq.surface.R_basis.get_idx(M=-1, N=-2)] = -1


perturb_options = {
    "dRb": True,
    "dZb": True,
    "opt_subspace": opt_subspace,
    "tr_ratio": [0.1, 0.25],
    "verbose": 2,
}
solve_options = {"verbose": 3, "maxiter": 50}

eq.optimize(
    objective,
    ftol=1e-2,
    xtol=1e-6,
    copy=False,
    verbose=2,
    perturb_options=perturb_options,
    solve_options=solve_options,
)
eq.save(output_path)

timer.stop("Total")
timer.disp("Total")
