"""Misc utilities needed by objectives.

Functions in this module should not depend on any other submodules in desc.objectives.
"""

import numpy as np

from desc.backend import cond, jit, jnp, logsumexp, put
from desc.utils import Index, errorif, flatten_list, svd_inv_null, unique_list, warnif


def factorize_linear_constraints(objective, constraint):  # noqa: C901
    """Compute and factorize A to get pseudoinverse and nullspace.

    Given constraints of the form Ax=b, factorize A to find a particular solution xp
    and the null space Z st. Axp=b and AZ=0, so that the full space of solutions to
    Ax=b can be written as x=xp + Zy where y is now unconstrained.

    Parameters
    ----------
    objective : ObjectiveFunction
        Objective function to optimize.
    constraint : ObjectiveFunction
        Objective function of linear constraints to enforce.

    Returns
    -------
    xp : ndarray
        Particular solution to Ax=b.
    A : ndarray ndarray
        Combined constraint matrix, such that A @ x[unfixed_idx] == b.
    b : list of ndarray
        Combined RHS vector.
    Z : ndarray
        Null space operator for full combined A such that A @ Z == 0.
    unfixed_idx : ndarray
        Indices of x that correspond to non-fixed values.
    project, recover : function
        Functions to project full vector x into reduced vector y,
        and to recover x from y.

    """
    for con in constraint.objectives:
        errorif(
            not con.linear,
            ValueError,
            "Cannot handle nonlinear constraint {con}.",
        )
        errorif(
            con.bounds is not None,
            ValueError,
            f"Linear constraint {con} must use target instead of bounds.",
        )
        for thing in con.things:
            warnif(
                thing not in objective.things,
                UserWarning,
                f"Optimizable object {thing} is constrained by {con}"
                + " but not included in objective.",
            )

    from desc.optimize import ProximalProjection

    # particular solution to Ax=b
    xp = jnp.zeros(objective.dim_x)

    # linear constraints Ax=b
    x0 = jnp.zeros(constraint.dim_x)
    A = constraint.jac_scaled(x0)
    b = -constraint.compute_scaled_error(x0)

    if isinstance(objective, ProximalProjection):
        # remove cols of A corresponding to ["R_lmn", "Z_lmn", "L_lmn", "Ra_n", "Za_n"]
        c = 0
        cols = np.array([], dtype=int)
        for t in objective.things:
            if t is objective._eq:
                for arg, dim in objective._eq.dimensions.items():
                    if arg in objective._args:  # these Equilibrium args are kept
                        cols = np.append(cols, np.arange(c, c + dim))
                    c += dim  # other Equilibrium args are removed
            else:  # non-Equilibrium args are always included
                cols = np.append(cols, np.arange(c, c + t.dim_x))
                c += t.dim_x
        A = A[:, cols]
    assert A.shape[1] == xp.size

    # will store the global index of the unfixed rows, idx
    indices_row = np.arange(A.shape[0])
    indices_idx = np.arange(A.shape[1])

    # while loop has problems updating JAX arrays, convert them to numpy arrays
    A = np.array(A)
    b = np.array(b)
    while len(np.where(np.count_nonzero(A, axis=1) == 1)[0]):
        # fixed just means there is a single element in A, so A_ij*x_j = b_i
        fixed_rows = np.where(np.count_nonzero(A, axis=1) == 1)[0]
        # indices of x that are fixed = cols of A where rows have 1 nonzero val.
        _, fixed_idx = np.where(A[fixed_rows])
        unfixed_rows = np.setdiff1d(np.arange(A.shape[0]), fixed_rows)
        unfixed_idx = np.setdiff1d(np.arange(A.shape[1]), fixed_idx)

        # find the global index of the fixed variables of this iteration
        global_fixed_idx = indices_idx[fixed_idx]
        # find the global index of the unfixed variables by removing the fixed variables
        # from the indices arrays.
        indices_idx = np.delete(indices_idx, fixed_idx)  # fixed indices are removed
        indices_row = np.delete(indices_row, fixed_rows)  # fixed rows are removed

        if len(fixed_rows):
            # something like 0.5 x1 = 2 is the same as x1 = 4
            b = put(b, fixed_rows, b[fixed_rows] / np.sum(A[fixed_rows], axis=1))
            A = put(
                A,
                Index[fixed_rows, :],
                A[fixed_rows] / np.sum(A[fixed_rows], axis=1)[:, None],
            )
            xp = put(xp, global_fixed_idx, b[fixed_rows])
            # Some values might be fixed, but they still show up in other constraints
            # this is where the fixed cols have >1 nonzero val.
            # For fixed variables, we delete that row and col of A, but that means
            # we need to subtract the fixed value from b so that the equation is
            # balanced.
            # e.g., 2 x1 + 3 x2 + 1 x3 = 4 ; 4 x1 = 2
            # combining gives 3 x2 + 1 x3 = 3, with x1 now removed
            b = put(
                b,
                unfixed_rows,
                b[unfixed_rows] - A[unfixed_rows][:, fixed_idx] @ b[fixed_rows],
            )
        A = A[unfixed_rows][:, unfixed_idx]
        b = b[unfixed_rows]
    unfixed_idx = indices_idx
    if A.size:
        Ainv_full, Z = svd_inv_null(A)
    else:
        Ainv_full = A.T
        Z = np.eye(A.shape[1])
    Ainv_full = jnp.asarray(Ainv_full)
    Z = jnp.asarray(Z)
    b = jnp.asarray(b)
    xp = put(xp, unfixed_idx, Ainv_full @ b)
    xp = jnp.asarray(xp)

    @jit
    def project(x):
        """Project a full state vector into the reduced optimization vector."""
        x_reduced = Z.T @ ((x - xp)[unfixed_idx])
        return jnp.atleast_1d(jnp.squeeze(x_reduced))

    @jit
    def recover(x_reduced):
        """Recover the full state vector from the reduced optimization vector."""
        dx = put(jnp.zeros(objective.dim_x), unfixed_idx, Z @ x_reduced)
        return jnp.atleast_1d(jnp.squeeze(xp + dx))

    # check that all constraints are actually satisfiable
    params = objective.unpack_state(xp, False)
    for con in constraint.objectives:
        xpi = [params[i] for i, t in enumerate(objective.things) if t in con.things]
        y1 = con.compute_unscaled(*xpi)
        y2 = con.target
        y1, y2 = np.broadcast_arrays(y1, y2)

        # If the error is very large, likely want to error out as
        # it probably is due to a real mistake instead of just numerical
        # roundoff errors.
        np.testing.assert_allclose(
            y1,
            y2,
            atol=1e-6,
            rtol=1e-1,
            err_msg="Incompatible constraints detected, cannot satisfy constraint "
            + f"{con}.",
        )

        # else check with tighter tols and throw an error, these tolerances
        # could be tripped due to just numerical roundoff or poor scaling between
        # constraints, so don't want to error out but we do want to warn the user.
        atol = 3e-14
        rtol = 3e-14

        try:
            np.testing.assert_allclose(
                y1,
                y2,
                atol=atol,
                rtol=rtol,
                err_msg="Incompatible constraints detected, cannot satisfy constraint "
                + f"{con}.",
            )
        except AssertionError as e:
            warnif(
                True,
                UserWarning,
                str(e) + "\n This may indicate incompatible constraints, "
                "or be due to floating point error.",
            )

    return xp, A, b, Z, unfixed_idx, project, recover


def softmax(arr, alpha):
    """JAX softmax implementation.

    Inspired by https://www.johndcook.com/blog/2010/01/13/soft-maximum/
    and https://www.johndcook.com/blog/2010/01/20/how-to-compute-the-soft-maximum/

    Will automatically multiply array values by 2 / min_val if the min_val of
    the array is <1. This is to avoid inaccuracies that arise when values <1
    are present in the softmax, which can cause inaccurate maxes or even incorrect
    signs of the softmax versus the actual max.

    Parameters
    ----------
    arr : ndarray
        The array which we would like to apply the softmax function to.
    alpha : float
        The parameter smoothly transitioning the function to a hardmax.
        as alpha increases, the value returned will come closer and closer to
        max(arr).

    Returns
    -------
    softmax : float
        The soft-maximum of the array.

    """
    arr_times_alpha = alpha * arr
    min_val = jnp.min(jnp.abs(arr_times_alpha)) + 1e-4  # buffer value in case min is 0
    return cond(
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


def softmin(arr, alpha):
    """JAX softmin implementation, by taking negative of softmax(-arr).

    Parameters
    ----------
    arr : ndarray
        The array which we would like to apply the softmin function to.
    alpha: float
        The parameter smoothly transitioning the function to a hardmin.
        as alpha increases, the value returned will come closer and closer to
        min(arr).

    Returns
    -------
    softmin: float
        The soft-minimum of the array.

    """
    return -softmax(-arr, alpha)


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
    # unique list of things from all objectives
    things = unique_list(flatten_list([obj.things for obj in objectives]))[0]
    for obj in objectives:
        obj._set_things(things)  # obj.things will have same order for all objectives
    return objectives


def _parse_callable_target_bounds(target, bounds, x):
    if x.ndim > 1:
        x = x[:, 0]
    if callable(target):
        target = target(x)
    if bounds is not None and callable(bounds[0]):
        bounds = (bounds[0](x), bounds[1])
    if bounds is not None and callable(bounds[1]):
        bounds = (bounds[0], bounds[1](x))
    return target, bounds
