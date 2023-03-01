"""Driver script for optimization examples in dudt2022."""

import numpy as np
from scipy.linalg import block_diag, null_space

from desc.equilibrium import Equilibrium
from desc.objectives import (
    FixedBoundaryR,
    FixedBoundaryZ,
    FixedIota,
    FixedPressure,
    FixedPsi,
    LCFSBoundary,
    ObjectiveFunction,
    QuasisymmetryBoozer,
    QuasisymmetryFluxFunction,
    QuasisymmetryTripleProduct,
)
from desc.utils import Timer


def getOptSubspace(eq):
    """Get opt_subspace matrix."""
    idxRcc = eq.surface.R_basis.get_idx(M=1, N=2)
    idxRss = eq.surface.R_basis.get_idx(M=-1, N=-2)
    idxZsc = eq.surface.Z_basis.get_idx(M=-1, N=2)
    idxZcs = eq.surface.Z_basis.get_idx(M=1, N=-2)

    Rb_subspace = np.eye(eq.Rb_lmn.size)
    Rb_subspace = np.delete(Rb_subspace, (idxRcc, idxRss), 0)
    Rb_constraint = np.zeros((eq.Rb_lmn.size,))
    Rb_constraint[idxRcc] = 1
    Rb_constraint[idxRss] = -1
    Rb_subspace = np.vstack((Rb_subspace, Rb_constraint))

    Zb_subspace = np.eye(eq.Zb_lmn.size)
    Zb_subspace = np.delete(Zb_subspace, (idxZsc, idxZcs), 0)
    Zb_constraint = np.zeros((eq.Zb_lmn.size,))
    Zb_constraint[idxZsc] = 1
    Zb_constraint[idxZcs] = 1
    Zb_subspace = np.vstack((Zb_subspace, Zb_constraint))

    vmec_constraints = block_diag(Rb_subspace, Zb_subspace)
    opt_subspace = null_space(vmec_constraints)
    return opt_subspace


qs = "B"
order = 1
name = "f{}_or{}".format(qs, order)

input_path = "data/initial.h5"
output_path = "data/eq_" + name + ".h5"

timer = Timer()
timer.start("Total")

# equilibrium
eq = Equilibrium.load(input_path)
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
        QuasisymmetryBoozer(norm=True, helicity=(1, eq.NFP), M_booz=24, N_booz=12),
        constraints,
    )
elif qs == "C":
    objective = ObjectiveFunction(
        QuasisymmetryFluxFunction(norm=True, helicity=(1, eq.NFP)), constraints
    )
elif qs == "T":
    objective = ObjectiveFunction(QuasisymmetryTripleProduct(norm=True), constraints)

perturb_options = {
    "dRb": True,
    "dZb": True,
    "opt_subspace": getOptSubspace(eq),
    "order": order,
    "verbose": 2,
}
solve_options = {"verbose": 3}

eq = eq.optimize(
    objective,
    verbose=2,
    perturb_options=perturb_options,
    solve_options=solve_options,
)
eq.save(output_path)

timer.stop("Total")
timer.disp("Total")
