"""Utilities for getting standard groups of objectives and constraints."""

from desc.utils import flatten_list, get_all_instances, isposint, unique_list

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
    FixNearAxisLambda,
    FixNearAxisR,
    FixNearAxisZ,
    FixPressure,
    FixPsi,
    FixSheetCurrent,
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


def get_equilibrium_objective(eq, mode="force", normalize=True, jac_chunk_size="auto"):
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
    jac_chunk_size : int or ``auto``, optional
        If `"batched"` deriv_mode is used, will calculate the Jacobian
        ``jac_chunk_size`` columns at a time, instead of all at once.
        The memory usage of the Jacobian calculation is roughly
        ``memory usage = m0 + m1*jac_chunk_size``: the smaller the chunk size,
        the less memory the Jacobian calculation will require (with some baseline
        memory usage). The time it takes to compute the Jacobian is roughly
        ``t = t0 + t1/jac_chunk_size`` so the larger the ``jac_chunk_size``, the faster
        the calculation takes, at the cost of requiring more memory.
        If None, it will use the largest size i.e ``obj.dim_x``.
        Defaults to ``chunk_size="auto"`` which will use a conservative
        chunk size based off of a heuristic estimate of the memory usage.


    Returns
    -------
    objective, ObjectiveFunction
        An objective function with default force balance objectives.

    """
    kwargs = {"eq": eq, "normalize": normalize, "normalize_target": normalize}
    if mode == "energy":
        objectives = Energy(**kwargs)
    elif mode == "force":
        objectives = ForceBalance(**kwargs)
    elif mode == "forces":
        objectives = (RadialForceBalance(**kwargs), HelicalForceBalance(**kwargs))
    else:
        raise ValueError("got an unknown equilibrium objective type '{}'".format(mode))
    deriv_mode = "batched" if isposint(jac_chunk_size) else "auto"
    return ObjectiveFunction(
        objectives, jac_chunk_size=jac_chunk_size, deriv_mode=deriv_mode
    )


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
    kwargs = {"eq": eq, "normalize": normalize, "normalize_target": normalize}
    constraints = (FixAxisR(**kwargs), FixAxisZ(**kwargs), FixPsi(**kwargs))
    if profiles:
        for name, con in _PROFILE_CONSTRAINTS.items():
            if getattr(eq, name) is not None:
                constraints += (con(**kwargs),)
    constraints += (FixSheetCurrent(**kwargs),)

    return constraints


def get_fixed_boundary_constraints(eq, profiles=True, normalize=True):
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
    kwargs = {"eq": eq, "normalize": normalize, "normalize_target": normalize}
    constraints = (FixBoundaryR(**kwargs), FixBoundaryZ(**kwargs), FixPsi(**kwargs))
    if profiles:
        for name, con in _PROFILE_CONSTRAINTS.items():
            if getattr(eq, name) is not None:
                constraints += (con(**kwargs),)
    constraints += (FixSheetCurrent(**kwargs),)

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
    qsc_eq : Qsc, optional
        Qsc object defining the near-axis equilibrium to constrain behavior to.
        if None, will instead fix the current near-axis behavior of the ``desc_eq``
    order : {0,1,2}
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
    kwargs = {"eq": desc_eq, "normalize": normalize, "normalize_target": normalize}
    if not isinstance(fix_lambda, bool):
        fix_lambda = int(fix_lambda)
    constraints = (FixPsi(**kwargs),)

    if profiles:
        for name, con in _PROFILE_CONSTRAINTS.items():
            if getattr(desc_eq, name) is not None:
                constraints += (con(**kwargs),)
    constraints += (FixSheetCurrent(**kwargs),)

    constraints += (
        FixNearAxisR(
            eq=desc_eq,
            target=qsc_eq,
            N=N,
            order=order,
            normalize=normalize,
        ),
        FixNearAxisZ(
            eq=desc_eq,
            target=qsc_eq,
            N=N,
            order=order,
            normalize=normalize,
        ),
    )
    if fix_lambda or (fix_lambda >= 0 and type(fix_lambda) is int):
        constraints += (
            FixNearAxisLambda(
                eq=desc_eq,
                target=qsc_eq,
                N=N,
                order=int(fix_lambda),
                normalize=normalize,
            ),
        )

    return constraints


def _get_NAE_constraints(
    desc_eq,
    qsc_eq,
    order=1,
    N=None,
    fix_lambda=False,
    normalize=True,
):
    """Get the constraints necessary for fixing NAE behavior in an equilibrium problem.

    NOTE: This will return tuples of FixSumModes__, this is not intended to be directly
    used by the user. Instead, call the ``get_NAE_constraints`` function, or use the
    FixNearAxis{R,Z,Lambda} objectives along with the FixAxis{R,Z} objectives. This
    instead is a helper function to get the needed constraints for the
    FixNearAxis{R,Z,Lambda} objectives, which use those constraints to form
    the full constraint matrices to properly constrain the behavior of that
    part of the equilibrium.

    Parameters
    ----------
    desc_eq : Equilibrium
        Equilibrium to constrain behavior of
        (assumed to be a fit from the NAE equil using `.from_near_axis()`).
    qsc_eq : Qsc
        Qsc object defining the near-axis equilibrium to constrain behavior to.
        order : {0,1,2}
        order (in rho) of near-axis behavior to constrain
    normalize : bool
        Whether to apply constraints in normalized units.
    N : int
        max toroidal resolution to constrain.
        If `None`, defaults to equilibrium's toroidal resolution
    fix_lambda : bool or int
        Whether to constrain lambda to match that of the NAE near-axis
        if an `int`, fixes lambda up to that order in rho {0,1}
        if `True`, fixes lambda up to the specified order given by `order`
    normalize : bool
        Whether to apply constraints in normalized units.

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in fixed-near-axis behavior problems.
    """
    if not isinstance(fix_lambda, bool):
        fix_lambda = int(fix_lambda)
    constraints = ()

    kwargs = {"eq": desc_eq, "normalize": normalize, "normalize_target": normalize}
    if not isinstance(fix_lambda, bool):
        fix_lambda = int(fix_lambda)

    if fix_lambda or (fix_lambda >= 0 and type(fix_lambda) is int):
        L_axis_constraints, _, _ = calc_zeroth_order_lambda(
            qsc=qsc_eq, desc_eq=desc_eq, N=N
        )
        constraints += L_axis_constraints

    # Axis constraints
    constraints += (
        FixAxisR(**kwargs),
        FixAxisZ(**kwargs),
        AxisRSelfConsistency(desc_eq),
        AxisZSelfConsistency(desc_eq),
    )

    if order >= 1:  # first order constraints
        constraints += make_RZ_cons_1st_order(
            qsc=qsc_eq, desc_eq=desc_eq, N=N, fix_lambda=fix_lambda and fix_lambda > 0
        )
    if order >= 2:  # 2nd order constraints
        raise NotImplementedError("NAE constraints only implemented up to O(rho) ")

    return constraints


def maybe_add_self_consistency(thing, constraints):
    """Add self consistency constraints if needed."""

    def add_if_multiple(constraints, cls):
        cons = get_all_instances(constraints, cls)
        if cons is not None:
            cons_on_this_thing = [con for con in cons if con.things[0] == thing]
            if not len(cons_on_this_thing):
                constraints += (cls(thing),)
        else:
            constraints += (cls(thing),)
        return constraints

    params = set(unique_list(flatten_list(thing.optimizable_params))[0])

    # Equilibrium
    if {"R_lmn", "Rb_lmn"} <= params:
        constraints = add_if_multiple(constraints, BoundaryRSelfConsistency)

    if {"Z_lmn", "Zb_lmn"} <= params:
        constraints = add_if_multiple(constraints, BoundaryZSelfConsistency)

    if {"L_lmn"} <= params:
        constraints = add_if_multiple(constraints, FixLambdaGauge)

    if {"R_lmn", "Ra_n"} <= params:
        constraints = add_if_multiple(constraints, AxisRSelfConsistency)

    if {"Z_lmn", "Za_n"} <= params:
        constraints = add_if_multiple(constraints, AxisZSelfConsistency)

    # Curve
    if {"shift"} <= params:
        constraints = add_if_multiple(constraints, FixCurveShift)
    if {"rotmat"} <= params:
        constraints = add_if_multiple(constraints, FixCurveRotation)

    return constraints


def get_parallel_forcebalance(eq, num_device, mpi, grid=None, use_jit=True, verbose=1):
    """Get an ObjectiveFunction for parallel computing ForceBalance.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to constrain.
    num_device : int
        Number of devices to use for parallel computing.

    Returns
    -------
    obj : ObjectiveFunction
        A built objective function with force balance objectives. Each objective is
        computed on a separate device.
    """
    from desc.backend import desc_config, jax, jnp
    from desc.grid import LinearGrid

    if desc_config["num_device"] < num_device:
        raise ValueError(
            f"Number of devices in desc_config ({desc_config['num_device']}) "
            f"is less than the number of devices in input ({num_device})."
        )
    if grid is not None:
        if len(grid) != num_device:
            raise ValueError(
                f"Number of grids and num_device must be the same! Got "
                f"{len(grid)=} and {num_device=}."
            )
    if eq.L_grid % num_device == 0:
        k = eq.L_grid // num_device
        L = eq.L_grid
    else:
        k = eq.L_grid // num_device + 1
        L = k * num_device

    rhos = jnp.linspace(0.01, 1.0, L)
    objs = ()
    for i in range(num_device):
        if grid is None:
            gridi = LinearGrid(
                rho=rhos[i * k : (i + 1) * k],
                # kind of experimental way of set giving
                # less grid points to inner part, but seems
                # to make transforms way slower
                # M=int(eq.M_grid * i / num_device), # noqa: E800
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
            )
        else:
            gridi = grid[i]
        obj = ForceBalance(eq, grid=gridi, device_id=i)
        obj.build(use_jit=use_jit, verbose=verbose)
        obj = jax.device_put(obj, obj._device)
        # if the eq is also distrubuted across GPUs, then some internal logic
        # that checks if the things are different will fail, so we need to
        # set the eq to be the same manually
        obj._things[0] = eq
        objs += (obj,)
    objective = ObjectiveFunction(objs, mpi=mpi, deriv_mode="blocked")
    objective.build(use_jit=use_jit, verbose=verbose)
    return objective
