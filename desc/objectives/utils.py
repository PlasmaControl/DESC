from .objective_funs import ObjectiveFunction
from .linear_objectives import (
    LCFSBoundaryR,
    LCFSBoundaryZ,
    LambdaGauge,
    PoincareLambda,
    FixedPressure,
    FixedIota,
    FixedPsi,
    PoincareBoundaryR,
    PoincareBoundaryZ,
)
from .nonlinear_objectives import RadialForceBalance, HelicalForceBalance, Energy


def get_fixed_boundary_constraints():
    """Get the constraints necessary for a typical fixed-boundary equilibrium problem.

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in fixed-boundary problems.

    """
    constraints = (
        LCFSBoundaryR(),
        LCFSBoundaryZ(),
        LambdaGauge(),
        FixedPressure(),
        FixedIota(),
        FixedPsi(),
    )
    return constraints


def get_poincare_boundary_constraints():
    """Get the constraints necessary for a Poincare-boundary equilibrium problem.

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in Poincare-boundary problems.

    """
    constraints = (
        PoincareBoundaryR(),
        PoincareBoundaryZ(),
        LambdaGauge(),
        FixedPressure(),
        FixedIota(),
        FixedPsi(),
    )
    return constraints


def get_force_balance_objective():
    """Get the objective function for a typical force balance equilibrium problem.

    Returns
    -------
    objective, ObjectiveFunction
        An objective function with default force balance objectives.

    """
    objectives = (RadialForceBalance(), HelicalForceBalance())
    constraints = get_fixed_boundary_constraints()
    return ObjectiveFunction(objectives, constraints)


def get_force_balance_poincare_objective():
    """Get the objective function for a poincare force balance equilibrium problem.

    Returns
    -------
    objective, ObjectiveFunction
        An objective function with default force balance objectives.

    """
    objectives = (RadialForceBalance(), HelicalForceBalance())
    constraints = get_poincare_boundary_constraints()
    return ObjectiveFunction(objectives, constraints)


def get_energy_objective():
    """Get the objective function for a typical energy equilibrium problem.

    Returns
    -------
    objective, ObjectiveFunction
        An objective function with default energy objectives.

    """
    objectives = Energy()
    constraints = get_fixed_boundary_constraints()
    return ObjectiveFunction(objectives, constraints)
