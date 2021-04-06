import numpy as np
from desc.backend import jnp
import scipy.sparse
from scipy.optimize import OptimizeResult
import numba


def min_eig_est(A, tol=1e-2):
    """Estimate the minimum eigenvalue of a matrix

    Uses Lanzcos method through scipy.sparse.linalg

    Parameters
    ----------
    A : ndarray
        matrix, should be square
    tol : float
        precision for estimate of eigenvalue

    Returns
    -------
    e1 : float
        estimate for the minimum eigenvalue. Should be
        accurate to with +/- tol
    v1 : ndarray
        approximate eigenvector corresponding to e1
    """

    return scipy.sparse.linalg.eigsh(A, k=1, which="SA", tol=tol)


def make_spd(A, delta=1e-2, tol=1e-2):
    """Modify a matrix to make it positive definite

    Shifts the spectrum to make all eigenvalues > delta.
    Uses iterative Lanzcos method to approximate the smallest
    eigenvalue.

    Parameters
    -----------
    A : ndarray
        matrix, should be square and symmetric
    delta : float
        minimum allowed eigenvalue
    tol : float
        precision for estimate of minimum eigenvalue

    Returns
    -------
    A : ndarray
        A, but shifted by tau*I where tau is an approximation to
        the minimum eigenvalue
    """

    A = np.asarray(A)
    n = A.shape[0]
    eig_1, eigvec_1 = min_eig_est(A, tol)
    tau = max(0, (1 + tol) * (delta - eig_1))
    A = A + tau * np.eye(n)
    return 0.5 * (A + A.T)


@numba.njit()
def chol_U_update(U, x, alpha):
    """Rank 1 update to a cholesky decomposition

    Given cholesky decomposition A = U.T * U
    compute cholesky decomposition to A + alpha*x.T*x where
    x is a vector and alpha is a scalar.

    Parameters
    ----------
    U : ndarray
        upper triangular cholesky factor
    x : ndarray
        rank 1 update vector
    alpha : float
        scalar coefficient

    Returns
    -------
    U : ndarray
        updated cholesky factor
    """
    U = U.copy()
    sign = np.sign(alpha)
    a = np.sqrt(np.abs(alpha))
    x = a * x
    for k in range(x.size):
        r = np.sqrt(U[k, k] ** 2 + sign * x[k] ** 2)
        c = r / U[k, k]
        s = x[k] / U[k, k]
        U[k, k] = r
        U[k, k + 1 :] = (U[k, k + 1 :] + sign * s * x[k + 1 :]) / c
        x[k + 1 :] = c * x[k + 1 :] - s * U[k, k + 1 :]
    return U


def evaluate_quadratic_form(x, f, g, HorJ, scale=None):
    """Compute values of a quadratic function arising in least squares.
    The function is 0.5 * x.T * H * x + g.T * x + f.

    Parameters
    ----------
    x : ndarray, shape(n,)
        position where to evaluate quadratic form
    f : float
        constant term
    g : ndarray, shape(n,)
        Gradient, defines the linear term.
    HorJ : ndarray, LinearOperator or OptimizerDerivative
        Hessian/Jacobian matrix or operator
    scale : ndarray, shape(n,)
        scaling to apply. Scales hess -> scale*hess*scale, g-> scale*g

    Returns
    -------
    values : float
        Value of the function.
    """
    scale = scale if scale is not None else 1
    q = HorJ.quadratic(x * scale, x * scale)
    l = jnp.dot(scale * g, x)

    return f + l + 1 / 2 * q


def print_header_nonlinear():
    print(
        "{0:^15}{1:^15}{2:^15}{3:^15}{4:^15}{5:^15}".format(
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
        cost = "{0:^15.4e}".format(cost)

    if cost_reduction is None or abs(cost_reduction) == np.inf:
        cost_reduction = " " * 15
    else:
        cost_reduction = "{0:^15.2e}".format(cost_reduction)

    if step_norm is None or abs(step_norm) == np.inf:
        step_norm = " " * 15
    else:
        step_norm = "{0:^15.2e}".format(step_norm)

    if optimality is None or abs(optimality) == np.inf:
        optimality = " " * 15
    else:
        optimality = "{0:^15.2e}".format(optimality)

    print(
        "{0}{1}{2}{3}{4}{5}".format(
            iteration, nfev, cost, cost_reduction, step_norm, optimality
        )
    )


status_messages = {
    "success": "Optimization terminated successfully.",
    "xtol": "`xtol` condition satisfied.",
    "ftol": "`ftol` condition satisfied.",
    "gtol": "`agol` condition satisfied.",
    "max_nfev": "Maximum number of function evaluations has " "been exceeded.",
    "max_ngev": "Maximum number of gradient evaluations has " "been exceeded.",
    "max_nhev": "Maximum number of jacobian/hessian evaluations has " "been exceeded.",
    "maxiter": "Maximum number of iterations has been " "exceeded.",
    "pr_loss": "Desired error not necessarily achieved due " "to precision loss.",
    "nan": "NaN result encountered.",
    "out_of_bounds": "The result is outside of the provided " "bounds.",
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
    ratio,
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
):
    """Check termination condition and get message."""
    ftol_satisfied = dF < abs(ftol * F) and ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (xtol + x_norm)
    gtol_satisfied = g_norm < gtol

    if any([ftol_satisfied, xtol_satisfied, gtol_satisfied]):
        message = status_messages["success"]
        success = True
        if ftol_satisfied:
            message += "\n" + status_messages["ftol"]
        if xtol_satisfied:
            message += "\n" + status_messages["xtol"]
        if gtol_satisfied:
            message += "\n" + status_messages["gtol"]
    elif iteration >= maxiter:
        success = False
        message = status_messages["maxiter"]
    elif nfev >= max_nfev:
        success = False
        message = status_messages["max_nfev"]
    elif ngev >= max_ngev:
        success = False
        message = status_messages["max_ngev"]
    elif nhev >= max_nhev:
        success = False
        message = status_messages["max_nhev"]
    else:
        success = None
        message = None

    return success, message


def compute_jac_scale(A, prev_scale_inv=None):
    scale_inv = jnp.sum(A ** 2, axis=0) ** 0.5
    scale_inv = jnp.where(scale_inv == 0, 1, scale_inv)

    if prev_scale_inv is not None:
        scale_inv = jnp.maximum(scale_inv, prev_scale_inv)
    return 1 / scale_inv, scale_inv


def evaluate_quadratic(J, g, s, diag=None):
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
        q = jnp.sum(Js ** 2, axis=0)
        if diag is not None:
            q += jnp.sum(diag * s ** 2, axis=1)

    l = jnp.dot(s, g)

    return 0.5 * q + l
