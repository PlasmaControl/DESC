"""Functions for getting common objectives and constraints."""

import numpy as np

from desc.backend import block_diag, jnp, put
from desc.compute import arg_order
from desc.utils import svd_inv_null

from ._equilibrium import (
    CurrentDensity,
    Energy,
    ForceBalance,
    HelicalForceBalance,
    RadialForceBalance,
)
from .linear_objectives import (
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
        FixBoundaryR(
            fixed_boundary=True, normalize=normalize, normalize_target=normalize
        ),
        FixBoundaryZ(
            fixed_boundary=True, normalize=normalize, normalize_target=normalize
        ),
        FixLambdaGauge(normalize=normalize, normalize_target=normalize),
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
        FixAxisR(fixed_boundary=True),
        FixAxisZ(fixed_boundary=True),
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
        FixLambdaGauge(),
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
    A : dict of ndarray
        Individual constraint matrices, keyed by argument
    Ainv : dict of ndarray
        Individual pseudoinverses of constraint matrices
    b : dict of ndarray
        Individual rhs vectors
    Z : ndarray
        Null space operator for full combined A
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
    for arg in objective_args:
        x_idx[arg] = np.arange(dim_x, dim_x + dimensions[arg])
        dim_x += dimensions[arg]

    A = {}
    _A_unweighted = {}  # keep unweighted A here and use this to find Ainv
    b = {}
    Ainv = {}
    arg_obj_lists = {}
    xp = jnp.zeros(dim_x)  # particular solution to Ax=b
    constraint_args = []  # all args used in constraints
    unfixed_args = []  # subset of constraint args for unfixed objectives

    # linear constraint matrices for each objective
    for obj_ind, obj in enumerate(constraints):
        if len(obj.args) > 1:
            raise ValueError("Linear constraints must have only 1 argument.")
        if obj.bounds is not None:
            raise ValueError("Linear constraints must use target instead of bounds.")
        arg = obj.args[0]
        if arg not in objective_args:
            continue
        if arg not in constraint_args:  # hope this does not mess
            constraint_args.append(arg)
        if obj.fixed and obj.dim_f == dimensions[obj.target_arg]:
            # if all coefficients are fixed the constraint matrices are not needed
            xp = put(xp, x_idx[obj.target_arg], obj.target)

        else:
            A_ = obj.derivatives["jac"][arg](jnp.zeros(obj.dimensions[arg]))
            # using obj.compute instead of obj.target to allow for correct scale/weight
            b_ = -obj.compute_scaled(jnp.zeros(obj.dimensions[arg]))
            if arg not in A.keys():
                A[arg] = A_

                _A_unweighted[arg] = (
                    obj.normalization * (A_.T / obj.weight).T
                )  # undo normalization here for each indiv A
                b[arg] = b_
                unfixed_args.append(arg)
                arg_obj_lists[arg] = [obj]
            elif b_.size > 0:  # we want to stack these vertically if the same arg
                # but only if the constraint actually constrains anything
                # if not, i.e. if attempted to fix a mode that is not in the basis,
                # b_ is an empty array and does not need to be added
                rk_before = jnp.linalg.matrix_rank(A[arg])
                A[arg] = jnp.vstack((A[arg], A_))
                rk_after = jnp.linalg.matrix_rank(A[arg])
                if (
                    rk_after < rk_before + A_.shape[0]
                ):  # if less, there are some lin dependent rows of A_ and A[arg]
                    # for example, could be trying to constrain same mode with 2
                    # different values
                    raise RuntimeError(
                        f"Incompatible constraints given for {arg}!"
                        + f" constraint {obj} is incompatible with"
                        + f" {*arg_obj_lists[arg],}"
                    )
                _A_unweighted[arg] = jnp.vstack(
                    (A[arg], A_ / obj.weight * obj.normalization)
                )
                arg_obj_lists[arg] += [obj]

                b[arg] = jnp.hstack((b[arg], b_))
    # find inverse of the now-combined constraint matrices for each arg
    for key in list(A.keys()):
        if A[key].shape[0]:
            Ainv[key], Z_ = svd_inv_null(_A_unweighted[key])
        else:
            Ainv[key] = _A_unweighted[
                key
            ].T  # make inverse matrices using already unweighted A

    # catch any arguments that are not constrained
    for arg in x_idx.keys():
        if arg not in constraint_args:
            unfixed_args.append(arg)
            A[arg] = jnp.zeros((1, dimensions[arg]))
            b[arg] = jnp.zeros((1,))

    # full A matrix for all unfixed constraints
    if len(A):
        unfixed_idx = jnp.concatenate(
            [x_idx[arg] for arg in arg_order if arg in A.keys()]
        )
        A_full = block_diag(*[A[arg] for arg in arg_order if arg in A.keys()])
        b_full = jnp.concatenate([b[arg] for arg in arg_order if arg in b.keys()])
        Ainv_full, Z = svd_inv_null(A_full)
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
        arg = con.args[0]
        if arg not in objective_args:
            continue
        res = con.compute_scaled(**xp_dict)
        x = xp_dict[arg]
        # stuff like density is O(1e19) so need some adjustable tolerance here.
        atol = max(1e-8, np.finfo(x).eps * np.linalg.norm(x) / x.size)
        if not np.allclose(res, 0, atol=atol):
            raise ValueError(
                f"Incompatible constraints detected, cannot satisfy constraint {con}"
            )

    return xp, A, Ainv, b, Z, unfixed_idx, project, recover


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
