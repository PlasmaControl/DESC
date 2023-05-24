"""Functions for getting common objectives and constraints."""

import numpy as np

from desc.backend import jnp, put
from desc.compute import arg_order
from desc.utils import Index, flatten_list, svd_inv_null

from ._equilibrium import (
    CurrentDensity,
    Energy,
    ForceBalance,
    HelicalForceBalance,
    RadialForceBalance,
)
from .linear_objectives import (
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


def get_fixed_boundary_constraints(
    profiles=True, iota=True, kinetic=False, normalize=True
):
    """Get the constraints necessary for a typical fixed-boundary equilibrium problem.

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
        A list of the linear constraints used in fixed-boundary problems.

    """
    constraints = (
        FixBoundaryR(normalize=normalize, normalize_target=normalize),
        FixBoundaryZ(normalize=normalize, normalize_target=normalize),
        FixPsi(normalize=normalize, normalize_target=normalize),
    )
    if profiles:
        if kinetic:
            constraints += (
                FixElectronDensity(normalize=normalize, normalize_target=normalize),
                FixElectronTemperature(normalize=normalize, normalize_target=normalize),
                FixIonTemperature(normalize=normalize, normalize_target=normalize),
                FixAtomicNumber(normalize=normalize, normalize_target=normalize),
            )
        else:
            constraints += (
                FixPressure(normalize=normalize, normalize_target=normalize),
            )

        if iota:
            constraints += (FixIota(normalize=normalize, normalize_target=normalize),)
        else:
            constraints += (
                FixCurrent(normalize=normalize, normalize_target=normalize),
            )
    return constraints


def maybe_add_self_consistency(constraints):
    """Add self consistency constraints if needed."""

    def _is_any_instance(things, cls):
        return any([isinstance(t, cls) for t in things])

    if not _is_any_instance(constraints, BoundaryRSelfConsistency):
        constraints += (BoundaryRSelfConsistency(),)
    if not _is_any_instance(constraints, BoundaryZSelfConsistency):
        constraints += (BoundaryZSelfConsistency(),)
    if not _is_any_instance(constraints, FixLambdaGauge):
        constraints += (FixLambdaGauge(),)
    return constraints


def get_fixed_axis_constraints(profiles=True, iota=True):
    """Get the constraints necessary for a fixed-axis equilibrium problem.

    Parameters
    ----------
    profiles : bool
        Whether to also return constraints to fix input profiles.
    iota : bool
        Whether to add FixIota or FixCurrent as a constraint.

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in fixed-axis problems.

    """
    constraints = (
        FixAxisR(),
        FixAxisZ(),
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


def get_NAE_constraints(desc_eq, qsc_eq, profiles=True, iota=False, order=1):
    """Get the constraints necessary for fixing NAE behavior in an equilibrium problem. # noqa D205

    Parameters
    ----------
    desc_eq : Equilibrium
        Equilibrium to constrain behavior of
        (assumed to be a fit from the NAE equil using .from_near_axis()).
    qsc_eq : Qsc
        Qsc object defining the near-axis equilibrium to constrain behavior to.
    profiles : bool
        Whether to also return constraints to fix input profiles.
    iota : bool
        Whether to add FixIota or FixCurrent as a constraint.
    order : int
        order (in rho) of near-axis behavior to constrain

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in fixed-axis problems.
    """

    constraints = (
        FixAxisR(),
        FixAxisZ(),
        FixPsi(),
    )
    if profiles:
        constraints += (FixPressure(),)

        if iota:
            constraints += (FixIota(),)
        else:
            constraints += (FixCurrent(),)
    if order >= 1:  # first order constraints
        constraints += make_RZ_cons_1st_order(qsc=qsc_eq, desc_eq=desc_eq)
    if order >= 2:  # 2nd order constraints
        raise NotImplementedError("NAE constraints only implemented up to O(rho) ")

    return constraints


def get_equilibrium_objective(mode="force", normalize=True):
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
        objectives = Energy(normalize=normalize, normalize_target=normalize)
    elif mode == "force":
        objectives = ForceBalance(normalize=normalize, normalize_target=normalize)
    elif mode == "forces":
        objectives = (
            RadialForceBalance(normalize=normalize, normalize_target=normalize),
            HelicalForceBalance(normalize=normalize, normalize_target=normalize),
        )
    elif mode == "vacuum":
        objectives = CurrentDensity(normalize=normalize, normalize_target=normalize)
    else:
        raise ValueError("got an unknown equilibrium objective type '{}'".format(mode))
    return ObjectiveFunction(objectives)


def factorize_linear_constraints(constraints, objective_args):  # noqa: C901
    """Compute and factorize A to get pseudoinverse and nullspace.

    Given constraints of the form Ax=b, factorize A to find a particular solution xp
    and the null space Z st. Axp=b and AZ=0, so that the full space of solutions to
    Ax=b can be written as x=xp + Zy where y is now unconstrained.


    Parameters
    ----------
    constraints : tuple of Objectives
        linear objectives/constraints to factorize for projection method.
    objective_args : list of str
        names of all arguments used by the desired objective.

    Returns
    -------
    xp : ndarray
        particular solution to Ax=b
    A : ndarray ndarray
        Combined constraint matrix, such that A @ x[unfixed_idx] == b
    b : list of ndarray
        Combined rhs vector
    Z : ndarray
        Null space operator for full combined A such that A @ Z == 0
    unfixed_idx : ndarray
        indices of x that correspond to non-fixed values
    project, recover : function
        functions to project full vector x into reduced vector y,
        and recovering x from y.

    """
    # set state vector
    args = np.concatenate([obj.args for obj in constraints])
    args = np.concatenate((args, objective_args))
    # this is all args used by both constraints and objective
    args = [arg for arg in arg_order if arg in args]
    dimensions = constraints[0].dimensions
    dim_x = 0
    x_idx = {}
    for arg in args:
        x_idx[arg] = np.arange(dim_x, dim_x + dimensions[arg])
        dim_x += dimensions[arg]

    A = []
    b = []
    xp = jnp.zeros(dim_x)  # particular solution to Ax=b

    # linear constraint matrices for each objective
    for obj_ind, obj in enumerate(constraints):
        if obj.bounds is not None:
            raise ValueError("Linear constraints must use target instead of bounds.")
        A_ = {
            arg: obj.derivatives["jac_scaled"][arg](
                *[jnp.zeros(obj.dimensions[arg]) for arg in obj.args]
            )
            for arg in args
        }
        # using obj.compute instead of obj.target to allow for correct scale/weight
        b_ = -obj.compute_scaled_error(
            *[jnp.zeros(obj.dimensions[arg]) for arg in obj.args]
        )
        A.append(A_)
        b.append(b_)

    A_full = jnp.vstack([jnp.hstack([Ai[arg] for arg in args]) for Ai in A])
    b_full = jnp.concatenate(b)
    # fixed just means there is a single element in A, so A_ij*x_j = b_i
    fixed_rows = np.where(np.count_nonzero(A_full, axis=1) == 1)[0]
    # indices of x that are fixed = cols of A where rows have 1 nonzero val.
    _, fixed_idx = np.where(A_full[fixed_rows])
    unfixed_rows = np.setdiff1d(np.arange(A_full.shape[0]), fixed_rows)
    unfixed_idx = np.setdiff1d(np.arange(xp.size), fixed_idx)
    if len(fixed_rows):
        # something like 0.5 x1 = 2 is the same as x1 = 4
        b_full = put(
            b_full, fixed_rows, b_full[fixed_rows] / np.sum(A_full[fixed_rows], axis=1)
        )
        A_full = put(
            A_full,
            Index[fixed_rows, :],
            A_full[fixed_rows] / np.sum(A_full[fixed_rows], axis=1)[:, None],
        )
        xp = put(xp, fixed_idx, b_full[fixed_rows])
        # some values might be fixed, but they still show up in other constraints
        # this is where the fixed cols have >1 nonzero val
        # for fixed variables, we delete that row and col of A, but that means
        # we need to subtract the fixed value from b so that the equation is balanced.
        # eg 2 x1 + 3 x2 + 1 x3= 4 ;    4 x1 = 2
        # combining gives 3 x2 + 1 x3 = 3, with x1 now removed
        b_full = put(
            b_full,
            unfixed_rows,
            b_full[unfixed_rows]
            - A_full[unfixed_rows][:, fixed_idx] @ b_full[fixed_rows],
        )
    A_full = A_full[unfixed_rows][:, unfixed_idx]
    b_full = b_full[unfixed_rows]
    if A_full.size:
        Ainv_full, Z = svd_inv_null(A_full)
    else:
        Ainv_full = A_full.T
        Z = np.eye(A_full.shape[1])
    xp = put(xp, unfixed_idx, Ainv_full @ b_full)

    def project(x):
        """Project a full state vector into the reduced optimization vector."""
        x_reduced = jnp.dot(Z.T, (x - xp)[unfixed_idx])
        return jnp.atleast_1d(jnp.squeeze(x_reduced))

    def recover(x_reduced):
        """Recover the full state vector from the reduced optimization vector."""
        dx = put(jnp.zeros(dim_x), unfixed_idx, Z @ x_reduced)
        return jnp.atleast_1d(jnp.squeeze(xp + dx))

    # check that all constraints are actually satisfiable
    xp_dict = {arg: xp[x_idx[arg]] for arg in x_idx.keys()}
    for con in constraints:
        res = con.compute_scaled_error(**xp_dict)
        x = np.concatenate([xp_dict[arg] for arg in con.args])
        # stuff like density is O(1e19) so need some adjustable tolerance here.
        atol = max(1e-8, np.finfo(x.dtype).eps * np.linalg.norm(x) / x.size)
        np.testing.assert_allclose(
            res,
            0,
            atol=atol,
            err_msg="Incompatible constraints detected, cannot satisfy "
            + f"constraint {con}",
        )

    return xp, A_full, b_full, Z, unfixed_idx, project, recover


def align_jacobian(Fx, objective_f, objective_g):
    """Pad Jacobian with zeros in the right places so that the arguments line up.

    Parameters
    ----------
    Fx : ndarray
        Jacobian wrt args the objective_f takes
    objective_f : ObjectiveFunction
        Objective corresponding to Fx
    objective_g : ObjectiveFunction
        Other objective we want to align Jacobian against

    Returns
    -------
    A : ndarray
        Jacobian matrix, reordered and padded so that it broadcasts
        correctly against the other Jacobian
    """
    x_idx = objective_f.x_idx
    args = objective_f.args

    dim_f = Fx.shape[:1]
    A = {arg: Fx.T[x_idx[arg]] for arg in args}
    allargs = np.concatenate([objective_f.args, objective_g.args])
    allargs = [arg for arg in arg_order if arg in allargs]
    for arg in allargs:
        if arg not in A.keys():
            A[arg] = jnp.zeros((objective_f.dimensions[arg],) + dim_f)
    A = jnp.concatenate([A[arg] for arg in allargs])
    return A.T


def combine_args(*objectives):
    """Given ObjectiveFunctions, modify all to take the same state vector.

    The new state vector will be a combination of all arguments taken by any objective.

    Parameters
    ----------
    objectives : ObjectiveFunction
        ObjectiveFunctions to modify.

    Returns
    -------
    objectives : ObjectiveFunction
        Original ObjectiveFunctions modified to take the same state vector.
    """
    args = flatten_list([obj.args for obj in objectives])
    args = [arg for arg in arg_order if arg in args]

    for obj in objectives:
        obj.set_args(*args)

    return objectives
