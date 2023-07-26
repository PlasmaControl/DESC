"""Misc utilities needed by objectives.

Functions in this module should not depend on any other submodules in desc.objectives.
"""

import warnings

import numpy as np
from jax import lax
from jax.scipy.special import logsumexp

from desc.backend import jnp, put
from desc.utils import Index, flatten_list, sort_args, svd_inv_null


def factorize_linear_constraints(constraints, objective):  # noqa: C901
    """Compute and factorize A to get pseudoinverse and nullspace.

    Given constraints of the form Ax=b, factorize A to find a particular solution xp
    and the null space Z st. Axp=b and AZ=0, so that the full space of solutions to
    Ax=b can be written as x=xp + Zy where y is now unconstrained.


    Parameters
    ----------
    constraints : tuple of Objectives
        linear objectives/constraints to factorize for projection method.
    objective : ObjectiveFunction
        Objective being optimized.

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
    for con in constraints:
        for thing in con.things:
            if thing not in objective.things:
                warnings.warn(
                    f"Optimizeable object {thing} is constrained"
                    + " but not included in Objective"
                )
    # set state vector
    xz = objective.unpack_state(np.zeros(objective.dim_x))
    xp = jnp.zeros(objective.dim_x)  # particular solution to Ax=b
    A = []
    b = []

    from desc.optimize import ProximalProjection

    prox_flag = isinstance(objective, ProximalProjection)

    # linear constraint matrices for each objective
    for con in constraints:
        if con.bounds is not None:
            raise ValueError(
                f"Linear constraint {con} must use target instead of bounds."
            )
        A_per_thing = []
        # computing A matrix for each constraint for each thing in the optimization
        for thing in objective.things:
            if thing in con.things:
                # for now we implicitly assume that each linear constraint is bound to
                # only 1  thing, to generalize we need to make jac_scaled work for all
                # positional args not just the first one.
                A_ = con.jac_scaled(map_params(xz, con, objective.things)[0])
            else:
                A_ = {
                    arg: jnp.zeros((con.dim_f, dimx))
                    for arg, dimx in thing.dimensions.items()
                }
            args = objective._args if prox_flag else thing.optimizeable_params
            A_per_thing.append(jnp.hstack([A_[arg] for arg in args]))
        # using obj.compute instead of obj.target to allow for correct scale/weight
        b_ = -con.compute_scaled_error(map_params(xz, con, objective.things)[0])
        A.append(A_per_thing)
        b.append(b_)

    A_full = jnp.vstack([jnp.hstack(Ai) for Ai in A])
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
        x_reduced = Z.T @ ((x - xp)[unfixed_idx])
        return jnp.atleast_1d(jnp.squeeze(x_reduced))

    def recover(x_reduced):
        """Recover the full state vector from the reduced optimization vector."""
        dx = put(jnp.zeros(objective.dim_x), unfixed_idx, Z @ x_reduced)
        return jnp.atleast_1d(jnp.squeeze(xp + dx))

    # check that all constraints are actually satisfiable
    xp_ = objective.unpack_state(xp)
    for con in constraints:
        y1 = con.compute_unscaled(*map_params(xp_, con, objective.things))
        y2 = con.target
        y1, y2 = np.broadcast_arrays(y1, y2)
        np.testing.assert_allclose(
            y1,
            y2,
            atol=1e-14,
            rtol=1e-14,
            err_msg="Incompatible constraints detected, cannot satisfy "
            + f"constraint {con}",
        )

    return xp, A_full, b_full, Z, unfixed_idx, project, recover


def jax_softmax(arr, alpha):
    """JAX softmax implementation.

    Inspired by https://www.johndcook.com/blog/2010/01/13/soft-maximum/
    and https://www.johndcook.com/blog/2010/01/20/how-to-compute-the-soft-maximum/

    Will automatically multiply array values by 2 / min_val if the min_val of
    the array is <1. This is to avoid inaccuracies that arise when values <1
    are present in the softmax, which can cause inaccurate maxes or even incorrect
    signs of the softmax versus the actual max.

    Parameters
    ----------
    arr: ndarray, the array which we would like to apply the softmax function to.
    alpha: float, the parameter smoothly transitioning the function to a hardmax.
        as alpha increases, the value returned will come closer and closer to
        max(arr).

    Returns
    -------
    softmax: float, the soft-maximum of the array.
    """
    arr_times_alpha = alpha * arr
    min_val = jnp.min(jnp.abs(arr_times_alpha)) + 1e-4  # buffer value in case min is 0
    return lax.cond(
        jnp.any(min_val < 1),
        lambda arr_times_alpha: logsumexp(
            arr_times_alpha / min_val * 2
        )  # adjust to make vals>1
        / alpha
        * min_val
        / 2,
        lambda arr_times_alpha: logsumexp(arr_times_alpha) / alpha,
        arr_times_alpha,
    )


def jax_softmin(arr, alpha):
    """JAX softmin implementation, by taking negative of softmax(-arr).

    Parameters
    ----------
    arr: ndarray, the array which we would like to apply the softmin function to.
    alpha: float, the parameter smoothly transitioning the function to a hardmin.
        as alpha increases, the value returned will come closer and closer to
        min(arr).

    Returns
    -------
    softmin: float, the soft-minimum of the array.
    """
    return -jax_softmax(-arr, alpha)


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
    args = sort_args(args)

    for obj in objectives:
        obj.set_args(*args)

    return objectives


def map_params(params, objective, things):
    """Return a list of parameters for the things objective is tied to."""
    return [p for p, t in zip(params, things) if t in objective.things]
