"""Utilities for getting standard groups of objectives and constraints."""

import numpy as np

from desc.utils import is_any_instance

from ._equilibrium import Energy, ForceBalance, HelicalForceBalance, RadialForceBalance
from .linear_objectives import (
    AxisRSelfConsistency,
    AxisZSelfConsistency,
    BoundaryRSelfConsistency,
    BoundaryZSelfConsistency,
    FixAnisotropy,
    FixAtomicNumber,
    FixAxisR,
    FixAxisZ,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixCurveRotation,
    FixCurveShift,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixIota,
    FixLambdaGauge,
    FixParameter,
    FixPressure,
    FixPsi,
    FixSectionLambda,
    FixSectionR,
    FixSectionZ,
    SectionLambdaSelfConsistency,
    SectionRSelfConsistency,
    SectionZSelfConsistency,
)
from .nae_utils import calc_zeroth_order_lambda, make_RZ_cons_1st_order
from .objective_funs import ObjectiveFunction

_PROFILE_CONSTRAINTS = {
    "pressure": FixPressure,
    "iota": FixIota,
    "current": FixCurrent,
    "electron_density": FixElectronDensity,
    "electron_temperature": FixElectronTemperature,
    "ion_temperature": FixIonTemperature,
    "atomic_number": FixAtomicNumber,
    "anisotropy": FixAnisotropy,
}


def get_equilibrium_objective(eq, mode="force", normalize=True):
    """Get the objective function for a typical force balance equilibrium problem.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    mode : one of {"force", "forces", "energy"}
        which objective to return. "force" computes force residuals on unified grid.
        "forces" uses two different grids for radial and helical forces. "energy" is
        for minimizing MHD energy.
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
    else:
        raise ValueError("got an unknown equilibrium objective type '{}'".format(mode))
    return ObjectiveFunction(objectives)


def get_fixed_axis_constraints(eq, profiles=True, normalize=True):
    """Get the constraints necessary for a fixed-axis equilibrium problem.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to constrain.
    profiles : bool
        If True, also include constraints to fix all profiles assigned to equilibrium.
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
        for name, con in _PROFILE_CONSTRAINTS.items():
            if getattr(eq, name) is not None:
                constraints += (
                    con(eq=eq, normalize=normalize, normalize_target=normalize),
                )
    for param in ["I", "G", "Phi_mn"]:
        if np.array(getattr(eq, param, [])).size:
            constraints += (FixParameter(eq, param),)

    return constraints


def get_fixed_boundary_constraints(
    eq,
    profiles=True,
    normalize=True,
):
    """Get the constraints necessary for a typical fixed-boundary equilibrium problem.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to constrain.
    profiles : bool
        If True, also include constraints to fix all profiles assigned to equilibrium.
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
        for name, con in _PROFILE_CONSTRAINTS.items():
            if getattr(eq, name) is not None:
                constraints += (
                    con(eq=eq, normalize=normalize, normalize_target=normalize),
                )
    for param in ["I", "G", "Phi_mn"]:
        if np.array(getattr(eq, param, [])).size:
            constraints += (FixParameter(eq, param),)

    return constraints


def get_fixed_xsection_constraints(
    eq,
    profiles=True,
    normalize=True,
):
    """Get the constraints necessary for a fixed cross-section equilibrium problem.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to constrain.
    profiles : bool
        If True, also include constraints to fix all profiles assigned to equilibrium.
    normalize : bool
        Whether to apply constraints in normalized units.

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in fixed-boundary problems.

    """
    constraints = (
        FixSectionR(eq=eq, normalize=normalize, normalize_target=normalize),
        FixSectionZ(eq=eq, normalize=normalize, normalize_target=normalize),
        FixSectionLambda(eq=eq, normalize=normalize, normalize_target=normalize),
        FixPsi(eq=eq, normalize=normalize, normalize_target=normalize),
    )
    if profiles:
        for name, con in _PROFILE_CONSTRAINTS.items():
            if getattr(eq, name) is not None:
                constraints += (
                    con(eq=eq, normalize=normalize, normalize_target=normalize),
                )

    return constraints


def get_NAE_constraints(
    desc_eq,
    qsc_eq,
    order=1,
    profiles=True,
    normalize=True,
    N=None,
    fix_lambda=False,
):
    """Get the constraints necessary for fixing NAE behavior in an equilibrium problem.

    Parameters
    ----------
    desc_eq : Equilibrium
        Equilibrium to constrain behavior of
        (assumed to be a fit from the NAE equil using `.from_near_axis()`).
    qsc_eq : Qsc
        Qsc object defining the near-axis equilibrium to constrain behavior to.
    order : int
        order (in rho) of near-axis behavior to constrain
    profiles : bool
        If True, also include constraints to fix all profiles assigned to equilibrium.
    normalize : bool
        Whether to apply constraints in normalized units.
    N : int
        max toroidal resolution to constrain.
        If `None`, defaults to equilibrium's toroidal resolution
    fix_lambda : bool or int
        Whether to constrain lambda to match that of the NAE near-axis
        if an `int`, fixes lambda up to that order in rho {0,1}
        if `True`, fixes lambda up to the specified order given by `order`

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in fixed-axis problems.
    """
    if not isinstance(fix_lambda, bool):
        fix_lambda = int(fix_lambda)
    constraints = (
        FixAxisR(eq=desc_eq, normalize=normalize, normalize_target=normalize),
        FixAxisZ(eq=desc_eq, normalize=normalize, normalize_target=normalize),
        FixPsi(eq=desc_eq, normalize=normalize, normalize_target=normalize),
    )

    if profiles:
        for name, con in _PROFILE_CONSTRAINTS.items():
            if getattr(desc_eq, name) is not None:
                constraints += (
                    con(eq=desc_eq, normalize=normalize, normalize_target=normalize),
                )
    for param in ["I", "G", "Phi_mn"]:
        if np.array(getattr(desc_eq, param, [])).size:
            constraints += (FixParameter(desc_eq, param),)

    if fix_lambda or (fix_lambda >= 0 and type(fix_lambda) is int):
        L_axis_constraints, _, _ = calc_zeroth_order_lambda(
            qsc=qsc_eq, desc_eq=desc_eq, N=N
        )
        constraints += L_axis_constraints
    if order >= 1:  # first order constraints
        constraints += make_RZ_cons_1st_order(
            qsc=qsc_eq, desc_eq=desc_eq, N=N, fix_lambda=fix_lambda and fix_lambda > 0
        )
    if order >= 2:  # 2nd order constraints
        raise NotImplementedError("NAE constraints only implemented up to O(rho) ")

    return constraints


def maybe_add_self_consistency(thing, constraints):
    """Add self consistency constraints if needed."""
    # Equilibrium
    if (
        hasattr(thing, "Ra_n")
        and hasattr(thing, "Za_n")
        and hasattr(thing, "Rb_lmn")
        and hasattr(thing, "Zb_lmn")
        and hasattr(thing, "Rp_lmn")
        and hasattr(thing, "Zp_lmn")
        and hasattr(thing, "Lp_lmn")
        and hasattr(thing, "L_lmn")
    ):
        # Section self-consistency constraints
        if not is_any_instance(constraints, SectionRSelfConsistency):
            constraints += (SectionRSelfConsistency(eq=thing),)
        if not is_any_instance(constraints, SectionZSelfConsistency):
            constraints += (SectionZSelfConsistency(eq=thing),)
        if not is_any_instance(constraints, SectionLambdaSelfConsistency):
            constraints += (SectionLambdaSelfConsistency(eq=thing),)

        # Boundary self-consistency constraints
        if not is_any_instance(constraints, BoundaryRSelfConsistency):
            constraints += (BoundaryRSelfConsistency(eq=thing),)
        if not is_any_instance(constraints, BoundaryZSelfConsistency):
            constraints += (BoundaryZSelfConsistency(eq=thing),)
        if not is_any_instance(constraints, FixLambdaGauge):
            constraints += (FixLambdaGauge(eq=thing),)
        if not is_any_instance(constraints, AxisRSelfConsistency):
            constraints += (AxisRSelfConsistency(eq=thing),)
        if not is_any_instance(constraints, AxisZSelfConsistency):
            constraints += (AxisZSelfConsistency(eq=thing),)

        # Lambda gauge constraint for theta=0
        if not is_any_instance(constraints, FixLambdaGauge):
            constraints += (FixLambdaGauge(eq=thing),)

    # Curve
    elif hasattr(thing, "shift") and hasattr(thing, "rotmat"):
        if not is_any_instance(constraints, FixCurveShift):
            constraints += (FixCurveShift(curve=thing),)
        if not is_any_instance(constraints, FixCurveRotation):
            constraints += (FixCurveRotation(curve=thing),)

    return constraints
