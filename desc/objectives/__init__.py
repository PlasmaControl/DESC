"""Classes defining objectives for equilibrium and optimization."""

from ._equilibrium import (
    CurrentDensity,
    Energy,
    ForceBalance,
    HelicalForceBalance,
    RadialForceBalance,
)
from ._generic import GenericObjective, ToroidalCurrent
from ._geometry import AspectRatio, Volume
from ._qs import QuasisymmetryBoozer, QuasisymmetryTripleProduct, QuasisymmetryTwoTerm
from ._stability import MagneticWell, MercierStability
from ._wrappers import WrappedEquilibriumObjective
from .linear_objectives import (
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixLambdaGauge,
    FixPressure,
    FixPsi,
)
from .objective_funs import ObjectiveFunction


def get_fixed_boundary_constraints(profiles=True, iota=True):
    """Get the constraints necessary for a typical fixed-boundary equilibrium problem.

    Parameters
    ----------
    profiles : bool
        Whether to also return constraints to fix input profiles.
    iota : bool
        Whether to add FixIota or FixCurrent as a constraint.

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in fixed-boundary problems.

    """
    constraints = (
        FixBoundaryR(fixed_boundary=True),
        FixBoundaryZ(fixed_boundary=True),
        FixLambdaGauge(),
        FixPsi(),
    )
    if profiles:
        constraints += (FixPressure(),)

        if iota:
            constraints += (FixIota(),)
        else:
            constraints += (FixCurrent(),)
    return constraints


def get_equilibrium_objective(mode="force"):
    """Get the objective function for a typical force balance equilibrium problem.

    Parameters
    ----------
    mode : {"force", "forces", "energy", "vacuum"}
        which objective to return. "force" computes force residuals on unified grid.
        "forces" uses two different grids for radial and helical forces. "energy" is
        for minimizing MHD energy. "vacuum" directly minimizes current density.

    Returns
    -------
    objective, ObjectiveFunction
        An objective function with default force balance objectives.

    """
    if mode == "energy":
        objectives = Energy()
    elif mode == "force":
        objectives = ForceBalance()
    elif mode == "forces":
        objectives = (RadialForceBalance(), HelicalForceBalance())
    elif mode == "vacuum":
        objectives = CurrentDensity()
    else:
        raise ValueError("got an unknown equilibrium objective type '{}'".format(mode))
    return ObjectiveFunction(objectives)
