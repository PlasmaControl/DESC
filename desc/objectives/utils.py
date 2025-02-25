"""Misc utilities needed by objectives.

Functions in this module should not depend on any other submodules in desc.objectives.
"""

import numpy as np

from desc.backend import jit, jnp, put, softargmax
from desc.io import IOAble
from desc.utils import Index, errorif, flatten_list, svd_inv_null, unique_list, warnif


def factorize_linear_constraints(objective, constraint, x_scale="auto"):  # noqa: C901
    """Compute and factorize A to get particular solution and nullspace.

    Given constraints of the form Ax=b, factorize A to find a particular solution xp
    and the null space Z st. Axp=b and AZ=0, so that the full space of solutions to
    Ax=b can be written as x=xp + Zy where y is now unconstrained.

    Parameters
    ----------
    objective : ObjectiveFunction
        Objective function to optimize.
    constraint : ObjectiveFunction
        Objective function of linear constraints to enforce.
    x_scale : array_like or ``'auto'``, optional
        Characteristic scale of each variable. Setting ``x_scale`` is equivalent
        to reformulating the problem in scaled variables ``xs = x / x_scale``.
        If set to ``'auto'``, the scale is determined from the initial state vector.

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
    D : ndarray
        Scale of the full state vector x, as set by the parameter ``x_scale``.
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

    # check for degenerate rows and delete if necessary
    # augment A with b so that it only deletes actual degenerate constraints
    # which are duplicate rows of A that also have duplicate entries of b,
    # if the entries of b aren't the same then the constraints are actually
    # incompatible and so we will leave those to be caught later.
    A_augmented = np.hstack([A, np.reshape(b, (A.shape[0], 1))])

    # Find unique rows of A_augmented
    _, unique_indices = np.unique(A_augmented, axis=0, return_index=True)

    # Sort the indices to preserve the order of appearance
    unique_indices = np.sort(unique_indices)
    # Find the indices of the degenerate rows
    degenerate_idx = np.setdiff1d(np.arange(A_augmented.shape[0]), unique_indices)

    # Extract the unique rows
    A_augmented = A_augmented[unique_indices]
    A = A_augmented[:, :-1]
    b = np.atleast_1d(A_augmented[:, -1].squeeze())

    A_nondegenerate = A.copy()

    # remove fixed parameters from A and b
    A, b, xp, unfixed_idx, fixed_idx = remove_fixed_parameters(A, b, xp)

    # compute x_scale if not provided
    # Note: this x_scale is not the same as the x_scale as in solve_options["x_scale"]
    if x_scale == "auto":
        x_scale = objective.x(*objective.things)
    errorif(
        x_scale.shape != xp.shape,
        ValueError,
        "x_scale must be the same size as the full state vector. "
        + f"Got size {x_scale.size} for state vector of size {xp.size}.",
    )
    D = np.where(np.abs(x_scale) < 1e2, 1, np.abs(x_scale))

    # null space & particular solution
    A = A * D[None, unfixed_idx]
    if A.size:
        A_inv, Z = svd_inv_null(A)
    else:
        A_inv = A.T
        Z = np.eye(A.shape[1])
    xp = put(xp, unfixed_idx, A_inv @ b)
    xp = put(xp, fixed_idx, ((1 / D) * xp)[fixed_idx])
    # cast to jnp arrays
    # TODO: might consider sharding these too
    xp = jnp.asarray(xp)
    A = jnp.asarray(A)
    b = jnp.asarray(b)
    Z = jnp.asarray(Z)
    D = jnp.asarray(D)

    project = _Project(Z, D, xp, unfixed_idx)
    recover = _Recover(Z, D, xp, unfixed_idx, objective.dim_x)

    # check that all constraints are actually satisfiable
    params = objective.unpack_state(D * xp, False)
    for con in constraint.objectives:
        xpi = [params[i] for i, t in enumerate(objective.things) if t in con.things]
        y1 = con.compute_unscaled(*xpi)
        y2 = con.target
        y1, y2 = np.broadcast_arrays(y1, y2)

        # If the error is very large, likely want to error out as
        # it probably is due to a real mistake instead of just numerical
        # round-off errors.
        np.testing.assert_allclose(
            y1,
            y2,
            atol=1e-6,
            rtol=1e-1,
            err_msg="Incompatible constraints detected, cannot satisfy constraint "
            + f"{con}.",
        )

        # else check with tighter tols and throw an error, these tolerances
        # could be tripped due to just numerical round-off or poor scaling between
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

    return (
        xp,
        A,
        b,
        Z,
        D,
        unfixed_idx,
        project,
        recover,
        A_inv,
        A_nondegenerate,
        degenerate_idx,
    )


class _Project(IOAble):
    _io_attrs_ = ["Z", "D", "xp", "unfixed_idx"]

    def __init__(self, Z, D, xp, unfixed_idx):
        self.Z = Z
        self.D = D
        self.xp = xp
        self.unfixed_idx = unfixed_idx

    @jit
    def __call__(self, x_full):
        """Project a full state vector into the reduced optimization vector."""
        x_reduced = self.Z.T @ ((1 / self.D) * x_full - self.xp)[self.unfixed_idx]
        return jnp.atleast_1d(jnp.squeeze(x_reduced))


class _Recover(IOAble):
    _io_attrs_ = ["Z", "D", "xp", "unfixed_idx", "dim_x"]
    _static_attrs = ["dim_x"]

    def __init__(self, Z, D, xp, unfixed_idx, dim_x):
        self.Z = Z
        self.D = D
        self.xp = xp
        self.unfixed_idx = unfixed_idx
        self.dim_x = dim_x

    @jit
    def __call__(self, x_reduced):
        """Recover the full state vector from the reduced optimization vector."""
        dx = put(jnp.zeros(self.dim_x), self.unfixed_idx, self.Z @ x_reduced)
        x_full = self.D * (self.xp + dx)
        return jnp.atleast_1d(jnp.squeeze(x_full))


def softmax(arr, alpha):
    """JAX softmax implementation.

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
    arr = arr.flatten()
    arr_times_alpha = alpha * arr
    return softargmax(arr_times_alpha).dot(arr)


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


def check_if_points_are_inside_perimeter(R, Z, Rcheck, Zcheck):
    """Function to check if the given points is inside the given polyognal perimeter.

    Rcheck, Zcheck are the points to check, and R, Z define the perimeter
    in which to check. This function assumes that all points are in the same
    plane. Function will return an array of signs (+/- 1), with positive sign meaning
    the point is inside of the given perimeter, and a negative sign meaning the point
    is outside of the given perimeter.

    NOTE: it does not matter if the input coordinates are cylindrical (R,Z) or
    cartesian (X,Y), these are equivalent as long as they are in the same phi plane.
    This function will work even if points are not in the same phi plane, but the
    input coordinates must then be the equivalent of cartesian coordinates for whatever
    plane the points lie in.

    Algorithm based off of "An Incremental Angle Point in Polygon Test",
    K. Weiler, https://doi.org/10.1016/B978-0-12-336156-1.50012-4

    Parameters
    ----------
    R,Z : ndarray
        1-D arrays of coordinates of the points defining the polygonal
        perimeter. The function will determine if the check point is inside
        or outside of this perimeter. These should form a closed curve.
    Rcheck, Zcheck : ndarray
        coordinates of the points being checked if they are inside or outside of the
        given perimeter.

    Returns
    -------
    pt_sign : ndarray of {-1,1}
        Integers corresponding to if the given point is inside or outside of the given
        perimeter, with pt_sign[i]>0 meaning the point given by Rcheck[i], Zcheck[i] is
        inside of the given perimeter, and a negative sign meaning the point is outside
        of the given perimeter.

    """
    # R Z are the perimeter points
    # Rcheck Zcheck are the points being checked for whether
    # or not they are inside the check

    Rbool = R[:, None] > Rcheck
    Zbool = Z[:, None] > Zcheck
    # these are now size (Ncheck, Nperimeter)
    quadrants = jnp.zeros_like(Rbool)
    quadrants = jnp.where(jnp.logical_and(jnp.logical_not(Rbool), Zbool), 1, quadrants)
    quadrants = jnp.where(
        jnp.logical_and(jnp.logical_not(Rbool), jnp.logical_not(Zbool)),
        2,
        quadrants,
    )
    quadrants = jnp.where(jnp.logical_and(Rbool, jnp.logical_not(Zbool)), 3, quadrants)
    deltas = quadrants[1:, :] - quadrants[0:-1, :]
    deltas = jnp.where(deltas == 3, -1, deltas)
    deltas = jnp.where(deltas == -3, 1, deltas)
    # then flip sign if the R intercept is > Rcheck and the
    # quadrant flipped over a diagonal
    b = (Z[1:] / R[1:] - Z[0:-1] / R[0:-1]) / (Z[1:] - Z[0:-1])
    Rint = Rcheck[:, None] - b * (R[1:] - R[0:-1]) / (Z[1:] - Z[0:-1])
    deltas = jnp.where(
        jnp.logical_and(jnp.abs(deltas) == 2, Rint.T > Rcheck),
        -deltas,
        deltas,
    )
    pt_sign = jnp.sum(deltas, axis=0)
    # positive distance if the check pt is inside the perimeter, else
    # negative distance is assigned
    # pt_sign = 0 : Means point is OUTSIDE of the perimeter,
    #               assign positive distance
    # pt_sign = +/-4: Means point is INSIDE perimeter, so
    #                 assign negative distance
    pt_sign = jnp.where(jnp.isclose(pt_sign, 0), 1, -1)
    return pt_sign


def remove_fixed_parameters(A, b, xp):
    """Remove fixed parameters from the linear constraint matrix and RHS vector.

    Given a linear constraint matrix A and RHS vector b, remove fixed parameters from A
    and b. Fixed parameters are those that have only a single nonzero value in A, so
    that the equation is already balanced. This function will remove the fixed
    parameters from A and b, will also update the correcponding sections of the
    particular solution xp.

    Parameters
    ----------
    A : ndarray
        Constraint matrix.
    b : ndarray
        RHS vector.
    xp : ndarray
        Particular solution vector for the constraint Ax=b.

    Returns
    -------
    A : ndarray
        Constraint matrix with fixed parameters removed.
    b : ndarray
        RHS vector with fixed parameters removed.
    xp : ndarray
        Particular solution with fixed parameters updated.
    unfixed_idx : ndarray
        Indices of the unfixed parameters.
    fixed_idx : ndarray
        Indices of the fixed parameters
    """
    # will store the global index of the unfixed rows, idx
    indices_row = np.arange(A.shape[0])
    indices_idx = np.arange(A.shape[1])

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
    fixed_idx = np.delete(np.arange(xp.size), unfixed_idx)

    return A, b, xp, unfixed_idx, fixed_idx
