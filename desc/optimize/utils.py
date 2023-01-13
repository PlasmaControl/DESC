"""Utility functions used in optimization problems."""

import numpy as np

from desc.backend import cond, fori_loop, jit, jnp, put


@jit
def _cholmod(A):
    """Modified Cholesky factorization of indefinite matrix.

    If matrix is positive definite, returns the regular Cholesky factorization,
    otherwise performs a modified factorization to ensure the factors are positive
    definite.

    Algorithm adapted from https://www.cs.umd.edu/users/oleary/tr/tr4807.pdf

    Note this is much slower than the built in cholesky factorization, so should
    only be used when the matrix is known to be indefinite.

    Parameters
    ----------
    A : ndarray
        Matrix to factorize. Should be hermitian. No checking is done.

    Returns
    -------
    L : ndarray
        Lower triangular cholesky factor, such that L@L.T ~ A

    """
    A = jnp.asarray(A)
    n = A.shape[0]

    delta = jnp.finfo(A.dtype).eps * jnp.linalg.norm(A, "fro")
    xi = jnp.max(jnp.abs(A - jnp.diag(jnp.diag(A))))
    eta = jnp.max(jnp.abs(jnp.diag(A)))
    beta = jnp.sqrt(jnp.asarray([eta, xi / n, jnp.finfo(A.dtype).eps]).max())

    L = jnp.zeros_like(A)
    D = jnp.zeros(n)
    c = jnp.zeros_like(A)

    def sum_slice(D, L, i, j, kmax):
        s = 0
        bodyfun = lambda k, s: s + D[k] * L[i, k] * L[j, k]
        return fori_loop(0, kmax, bodyfun, s)

    def inner_loop(j, cLDi):
        c, L, D, i = cLDi
        s = sum_slice(D, L, i, j, j)
        c = put(c, (i, j), A[i, j] - s)

        def truefun(args):
            i, c, D = args
            theta = jnp.max(jnp.abs(c[i, :]))
            D = put(D, i, jnp.asarray([abs(c[i, i]), (theta / beta) ** 2, delta]).max())
            return i, c, D

        def falsefun(args):
            return args

        i, c, D = cond(i == j, truefun, falsefun, (i, c, D))
        L = put(L, (i, j), c[i, j] / D[j])

        return c, L, D, i

    def outer_loop(i, cLD):
        c, L, D = cLD
        c, L, D, i = fori_loop(0, i + 1, inner_loop, (c, L, D, i))
        return c, L, D

    c, L, D = fori_loop(0, n, outer_loop, (c, L, D))
    return L * jnp.sqrt(D)


@jit
def chol(A):
    """Cholesky factorization of possibly indefinite matrix.

    If matrix is positive definite, returns the regular Cholesky factorization,
    otherwise performs a modified factorization to ensure the factors are positive
    definite.

    Parameters
    ----------
    A : ndarray
        Matrix to factorize. Should be hermitian. No checking is done.

    Returns
    -------
    L : ndarray
        Lower triangular cholesky factor, such that L@L.T = A

    """
    L = jnp.linalg.cholesky(A)
    L = cond(jnp.any(jnp.isnan(L)), lambda A: _cholmod(A), lambda A: L, A)
    return L


def evaluate_quadratic_form_hess(x, f, g, H, scale=None):
    """Compute values of a quadratic function arising in trust region subproblem.

    The function is 0.5 * x.T * H * x + g.T * x + f.

    Parameters
    ----------
    x : ndarray, shape(n,)
        position where to evaluate quadratic form
    f : float
        constant term
    g : ndarray, shape(n,)
        Gradient, defines the linear term.
    H : ndarray
        Hessian matrix
    scale : ndarray, shape(n,)
        scaling to apply. Scales hess -> scale*hess*scale, g-> scale*g

    Returns
    -------
    values : float
        Value of the function.
    """
    scale = scale if scale is not None else 1
    q = (x * scale) @ H @ (x * scale)
    l = jnp.dot(scale * g, x)

    return f + l + 1 / 2 * q


def evaluate_quadratic_form_jac(J, g, s, diag=None):
    """Compute values of a quadratic function arising in least squares.

    The function is 0.5 * s.T * (J.T * J + diag) * s + g.T * s.

    Parameters
    ----------
    J : ndarray, sparse matrix or LinearOperator, shape (m, n)
        Jacobian matrix, affects the quadratic term.
    g : ndarray, shape (n,)
        Gradient, defines the linear term.
    s : ndarray, shape (k, n) or (n,)
        Array containing steps as rows.
    diag : ndarray, shape (n,), optional
        Addition diagonal part, affects the quadratic term.
        If None, assumed to be 0.

    Returns
    -------
    values : ndarray with shape (k,) or float
        Values of the function. If `s` was 2-D, then ndarray is
        returned, otherwise, float is returned.
    """
    if s.ndim == 1:
        Js = J.dot(s)
        q = jnp.dot(Js, Js)
        if diag is not None:
            q += jnp.dot(s * diag, s)
    else:
        Js = J.dot(s.T)
        q = jnp.sum(Js**2, axis=0)
        if diag is not None:
            q += jnp.sum(diag * s**2, axis=1)

    l = jnp.dot(s, g)

    return 0.5 * q + l


def print_header_nonlinear():
    """Print a pretty header."""
    print(
        "{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}".format(
            "Iteration",
            "Total nfev",
            "Cost",
            "Cost reduction",
            "Step norm",
            "Optimality",
        )
    )


def print_iteration_nonlinear(
    iteration, nfev, cost, cost_reduction, step_norm, optimality
):
    """Print a line of optimizer output."""
    if iteration is None or abs(iteration) == np.inf:
        iteration = " " * 15
    else:
        iteration = "{:^15}".format(iteration)

    if nfev is None or abs(nfev) == np.inf:
        nfev = " " * 15
    else:
        nfev = "{:^15}".format(nfev)

    if cost is None or abs(cost) == np.inf:
        cost = " " * 15
    else:
        cost = "{:^15.4e}".format(cost)

    if cost_reduction is None or abs(cost_reduction) == np.inf:
        cost_reduction = " " * 15
    else:
        cost_reduction = "{:^15.2e}".format(cost_reduction)

    if step_norm is None or abs(step_norm) == np.inf:
        step_norm = " " * 15
    else:
        step_norm = "{:^15.2e}".format(step_norm)

    if optimality is None or abs(optimality) == np.inf:
        optimality = " " * 15
    else:
        optimality = "{:^15.2e}".format(optimality)

    print(
        "{}{}{}{}{}{}".format(
            iteration, nfev, cost, cost_reduction, step_norm, optimality
        )
    )


STATUS_MESSAGES = {
    "success": "Optimization terminated successfully.",
    "xtol": "`xtol` condition satisfied.",
    "ftol": "`ftol` condition satisfied.",
    "gtol": "`gtol` condition satisfied.",
    "max_nfev": "Maximum number of function evaluations has been exceeded.",
    "max_ngev": "Maximum number of gradient evaluations has been exceeded.",
    "max_nhev": "Maximum number of Jacobian/Hessian evaluations has been exceeded.",
    "maxiter": "Maximum number of iterations has been exceeded.",
    "pr_loss": "Desired error not necessarily achieved due to precision loss.",
    "nan": "NaN result encountered.",
    "out_of_bounds": "The result is outside of the provided bounds.",
    "err": "A linalg error occurred, such as a non-psd Hessian.",
    "approx": "A bad approximation caused failure to predict improvement.",
    "callback": "User supplied callback triggered termination",
    None: None,
}


def check_termination(
    dF,
    F,
    dx_norm,
    x_norm,
    g_norm,
    reduction_ratio,
    ftol,
    xtol,
    gtol,
    iteration,
    maxiter,
    nfev,
    max_nfev,
    ngev,
    max_ngev,
    nhev,
    max_nhev,
    **kwargs,
):
    """Check termination condition and get message."""
    ftol_satisfied = dF < abs(ftol * F) and reduction_ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (xtol + x_norm) and reduction_ratio > 0.25
    gtol_satisfied = g_norm < gtol

    if any([ftol_satisfied, xtol_satisfied, gtol_satisfied]):
        message = STATUS_MESSAGES["success"]
        success = True
        if ftol_satisfied:
            message += "\n" + STATUS_MESSAGES["ftol"]
        if xtol_satisfied:
            message += "\n" + STATUS_MESSAGES["xtol"]
        if gtol_satisfied:
            message += "\n" + STATUS_MESSAGES["gtol"]
    elif iteration >= maxiter:
        success = False
        message = STATUS_MESSAGES["maxiter"]
    elif nfev >= max_nfev:
        success = False
        message = STATUS_MESSAGES["max_nfev"]
    elif ngev >= max_ngev:
        success = False
        message = STATUS_MESSAGES["max_ngev"]
    elif nhev >= max_nhev:
        success = False
        message = STATUS_MESSAGES["max_nhev"]
    elif dx_norm < kwargs.get("min_trust_radius", np.finfo(x_norm.dtype).eps):
        success = False
        message = STATUS_MESSAGES["approx"]
    elif kwargs.get("dx_total", 0) > kwargs.get("max_dx", np.inf):
        success = False
        message = STATUS_MESSAGES["out_of_bounds"]
    else:
        success = None
        message = None

    return success, message


def compute_jac_scale(A, prev_scale_inv=None):
    """Compute scaling factor based on column norm of Jacobian matrix."""
    scale_inv = jnp.sum(A**2, axis=0) ** 0.5
    scale_inv = jnp.where(scale_inv == 0, 1, scale_inv)

    if prev_scale_inv is not None:
        scale_inv = jnp.maximum(scale_inv, prev_scale_inv)
    return 1 / scale_inv, scale_inv


def compute_hess_scale(H, prev_scale_inv=None):
    """Compute scaling factors based on diagonal of Hessian matrix."""
    scale_inv = jnp.abs(jnp.diag(H))
    scale_inv = jnp.where(scale_inv == 0, 1, scale_inv)

    if prev_scale_inv is not None:
        scale_inv = jnp.maximum(scale_inv, prev_scale_inv)
    return 1 / scale_inv, scale_inv


def f_where_x(x, xs, fs):
    """Return fs where x==xs.

    Parameters
    ----------
    x : ndarray, shape(k,)
        array to find
    xs : list of ndarray of shape(k,)
        list to compare x against
    fs : list of float, ndarray
        list of values to return value from

    Returns
    -------
    f : float or ndarray
        value of fs[i] where x==xs[i]
    """
    x, xs, fs = map(np.asarray, (x, xs, fs))
    assert len(xs) == len(fs)
    assert len(xs) == 0 or x.shape == xs[0].shape
    eps = np.finfo(x.dtype).eps
    i = np.where(np.all(np.isclose(x, xs, rtol=eps, atol=eps), axis=1))[0]
    # sometimes two things are within eps of x, we want the most recent one
    if len(i) > 1:
        i = i[-1]
    return fs[i].squeeze()
