"""Utilities for getting standard groups of objectives and constraints."""

from ._equilibrium import (
    CurrentDensity,
    Energy,
    ForceBalance,
    HelicalForceBalance,
    RadialForceBalance,
)
from .linear_objectives import (
    AxisRSelfConsistency,
    AxisZSelfConsistency,
    BoundaryRSelfConsistency,
    BoundaryZSelfConsistency,
    FixAtomicNumber,
    FixAxisR,
    FixAxisZ,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixIota,
    FixLambdaGauge,
    FixPressure,
    FixPsi,
)
from .nae_utils import make_RZ_cons_1st_order
from .objective_funs import ObjectiveFunction


def get_equilibrium_objective(eq=None, mode="force", normalize=True):
    """Get the objective function for a typical force balance equilibrium problem.

    Parameters
    ----------
    mode : one of {"force", "forces", "energy", "vacuum"}
        which objective to return. "force" computes force residuals on unified grid.
        "forces" uses two different grids for radial and helical forces. "energy" is
        for minimizing MHD energy. "vacuum" directly minimizes current density.
    normalize : bool
        Whether to normalize units of objective.

    Returns
    -------
    objective, ObjectiveFunction
        An objective function with default force balance objectives.
    """
    if mode == "energy":
        objectives = Energy(eq=eq, normalize=normalize, normalize_target=normalize)
    elif mode == "force":
        objectives = ForceBalance(
            eq=eq, normalize=normalize, normalize_target=normalize
        )
    elif mode == "forces":
        objectives = (
            RadialForceBalance(eq=eq, normalize=normalize, normalize_target=normalize),
            HelicalForceBalance(eq=eq, normalize=normalize, normalize_target=normalize),
        )
    elif mode == "vacuum":
        objectives = CurrentDensity(
            eq=eq, normalize=normalize, normalize_target=normalize
        )
    else:
        raise ValueError("got an unknown equilibrium objective type '{}'".format(mode))
    return ObjectiveFunction(objectives)


def get_fixed_axis_constraints(
    eq=None, profiles=True, iota=True, kinetic=False, normalize=True
):
    """Get the constraints necessary for a fixed-axis equilibrium problem.

    Parameters
    ----------
    profiles : bool
        Whether to also return constraints to fix input profiles.
    iota : bool
        Whether to add FixIota or FixCurrent as a constraint.
    kinetic : bool
        Whether to add constraints to fix kinetic profiles or pressure
    normalize : bool
        Whether to apply constraints in normalized units.

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in fixed-axis problems.

    """
    constraints = (
        FixAxisR(eq=eq, normalize=normalize, normalize_target=normalize),
        FixAxisZ(eq=eq, normalize=normalize, normalize_target=normalize),
        FixPsi(eq=eq, normalize=normalize, normalize_target=normalize),
    )
    if profiles:
        if kinetic:
            constraints += (
                FixElectronDensity(
                    eq=eq, normalize=normalize, normalize_target=normalize
                ),
                FixElectronTemperature(
                    eq=eq, normalize=normalize, normalize_target=normalize
                ),
                FixIonTemperature(
                    eq=eq, normalize=normalize, normalize_target=normalize
                ),
                FixAtomicNumber(eq=eq, normalize=normalize, normalize_target=normalize),
            )
        else:
            constraints += (
                FixPressure(eq=eq, normalize=normalize, normalize_target=normalize),
            )

        if iota:
            constraints += (
                FixIota(eq=eq, normalize=normalize, normalize_target=normalize),
            )
        else:
            constraints += (
                FixCurrent(eq=eq, normalize=normalize, normalize_target=normalize),
            )
    return constraints


def get_fixed_boundary_constraints(
    eq=None, profiles=True, iota=True, kinetic=False, normalize=True
):
    """Get the constraints necessary for a typical fixed-boundary equilibrium problem.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to constraint.
    profiles : bool
        Whether to also return constraints to fix input profiles.
    iota : bool
        Whether to add FixIota or FixCurrent as a constraint.
    kinetic : bool
        Whether to also fix kinetic profiles.
    normalize : bool
        Whether to apply constraints in normalized units.

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in fixed-boundary problems.

    """
    constraints = (
        FixBoundaryR(eq=eq, normalize=normalize, normalize_target=normalize),
        FixBoundaryZ(eq=eq, normalize=normalize, normalize_target=normalize),
        FixPsi(eq=eq, normalize=normalize, normalize_target=normalize),
    )
    if profiles:
        if kinetic:
            constraints += (
                FixElectronDensity(
                    eq=eq, normalize=normalize, normalize_target=normalize
                ),
                FixElectronTemperature(
                    eq=eq, normalize=normalize, normalize_target=normalize
                ),
                FixIonTemperature(
                    eq=eq, normalize=normalize, normalize_target=normalize
                ),
                FixAtomicNumber(eq=eq, normalize=normalize, normalize_target=normalize),
            )
        else:
            constraints += (
                FixPressure(eq=eq, normalize=normalize, normalize_target=normalize),
            )

        if iota:
            constraints += (
                FixIota(eq=eq, normalize=normalize, normalize_target=normalize),
            )
        else:
            constraints += (
                FixCurrent(eq=eq, normalize=normalize, normalize_target=normalize),
            )
    return constraints


def get_NAE_constraints(
    desc_eq,
    qsc_eq,
    order=1,
    profiles=True,
    iota=False,
    kinetic=False,
    normalize=True,
    N=None,
):
    """Get the constraints necessary for fixing NAE behavior in an equilibrium problem.

    Parameters
    ----------
    desc_eq : Equilibrium
        Equilibrium to constrain behavior of
        (assumed to be a fit from the NAE equil using .from_near_axis()).
    qsc_eq : Qsc
        Qsc object defining the near-axis equilibrium to constrain behavior to.
    order : int
        order (in rho) of near-axis behavior to constrain
    profiles : bool
        Whether to also return constraints to fix input profiles.
    iota : bool
        Whether to add FixIota or FixCurrent as a constraint.
    kinetic : bool
        Whether to also fix kinetic profiles.
    normalize : bool
        Whether to apply constraints in normalized units.
    N : int,
        max toroidal resolution to constrain.
        If None, defaults to equilibrium's toroidal resolution

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in fixed-axis problems.
    """
    constraints = (
        FixAxisR(eq=desc_eq, normalize=normalize, normalize_target=normalize),
        FixAxisZ(eq=desc_eq, normalize=normalize, normalize_target=normalize),
        FixPsi(eq=desc_eq, normalize=normalize, normalize_target=normalize),
    )
    if profiles:
        if kinetic:
            constraints += (
                FixElectronDensity(
                    eq=desc_eq, normalize=normalize, normalize_target=normalize
                ),
                FixElectronTemperature(
                    eq=desc_eq, normalize=normalize, normalize_target=normalize
                ),
                FixIonTemperature(
                    eq=desc_eq, normalize=normalize, normalize_target=normalize
                ),
                FixAtomicNumber(
                    eq=desc_eq, normalize=normalize, normalize_target=normalize
                ),
            )
        else:
            constraints += (
                FixPressure(
                    eq=desc_eq, normalize=normalize, normalize_target=normalize
                ),
            )

        if iota:
            constraints += (
                FixIota(eq=desc_eq, normalize=normalize, normalize_target=normalize),
            )
        else:
            constraints += (
                FixCurrent(eq=desc_eq, normalize=normalize, normalize_target=normalize),
            )
    if order >= 1:  # first order constraints
        constraints += make_RZ_cons_1st_order(qsc=qsc_eq, desc_eq=desc_eq, N=N)
    if order >= 2:  # 2nd order constraints
        raise NotImplementedError("NAE constraints only implemented up to O(rho) ")

    return constraints


def maybe_add_self_consistency(eq, constraints):
    """Add self consistency constraints if needed."""

    def _is_any_instance(things, cls):
        return any([isinstance(t, cls) for t in things])

    if not _is_any_instance(constraints, BoundaryRSelfConsistency):
        constraints += (BoundaryRSelfConsistency(eq=eq),)
    if not _is_any_instance(constraints, BoundaryZSelfConsistency):
        constraints += (BoundaryZSelfConsistency(eq=eq),)
    if not _is_any_instance(constraints, FixLambdaGauge):
        constraints += (FixLambdaGauge(eq=eq),)
    if not _is_any_instance(constraints, AxisRSelfConsistency):
        constraints += (AxisRSelfConsistency(eq=eq),)
    if not _is_any_instance(constraints, AxisZSelfConsistency):
        constraints += (AxisZSelfConsistency(eq=eq),)
    return constraints
