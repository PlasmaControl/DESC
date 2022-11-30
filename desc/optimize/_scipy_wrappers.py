"""Functions for wrapping scipy.optimize methods and handle custom stopping criteria."""

import numpy as np
import scipy.optimize
from scipy.optimize import OptimizeResult

from desc.backend import jnp

from .utils import (
    check_termination,
    evaluate_quadratic_form_jac,
    f_where_x,
    print_header_nonlinear,
    print_iteration_nonlinear,
)


def _optimize_scipy_least_squares(
    fun, jac, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for scipy.optimize.least_squares.

    Parameters
    ----------
    fun : callable
        Function to minimize.
    jac : callable
        Jacobian of fun.
    x0 : ndarray
        Starting point.
    method : str
        Name of the method to use. Any of the options in scipy.optimize.least_squares.
    x_scale : array_like or ‘jac’, optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to ‘jac’, the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1-2 : display a termination report.
        * 3 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol",
        "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See the code for more details.

    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    options = {} if options is None else options
    # need to use some "global" variables here
    fun_allx = []
    fun_allf = []
    jac_allx = []
    jac_allf = []
    message = [""]
    success = [None]

    def fun_wrapped(x):
        # record all the xs and fs we see
        fun_allx.append(x)
        f = fun(x)
        fun_allf.append(f)
        return f

    def jac_wrapped(x):
        # record all the xs and jacs we see
        jac_allx.append(x)
        J = jac(x)
        jac_allf.append(J)
        callback(x)
        return J

    def callback(x1):
        f1 = f_where_x(x1, fun_allx, fun_allf)
        c1 = 1 / 2 * np.dot(f1, f1)
        J1 = f_where_x(x1, jac_allx, jac_allf)
        g1 = J1.T @ f1
        g_norm = np.linalg.norm(g1, ord=np.inf)
        x_norm = np.linalg.norm(x1)

        if len(jac_allx) < 2:
            df = np.inf
            dx_norm = np.inf
            reduction_ratio = 0
        else:
            x0 = jac_allx[-2]
            f0 = f_where_x(x0, fun_allx, fun_allf)
            c0 = 1 / 2 * np.dot(f0, f0)
            df = c0 - c1
            dx = x1 - x0
            dx_norm = jnp.linalg.norm(dx)

            predicted_reduction = -evaluate_quadratic_form_jac(J1, g1, dx)
            if predicted_reduction > 0:
                reduction_ratio = df / predicted_reduction
            elif predicted_reduction == df == 0:
                reduction_ratio = 1
            else:
                reduction_ratio = 0

        if verbose > 2:
            print_iteration_nonlinear(
                len(jac_allx), len(fun_allx), c1, df, dx_norm, g_norm
            )
        success[0], message[0] = check_termination(
            df,
            c1,
            dx_norm,
            x_norm,
            g_norm,
            reduction_ratio,
            stoptol["ftol"],
            stoptol["xtol"],
            stoptol["gtol"],
            len(fun_allf),  # iteration,
            stoptol["maxiter"],
            len(fun_allf),
            stoptol["max_nfev"],
            len(jac_allf),  # ngev
            stoptol["max_njev"],  # max_ngev,
            len(jac_allf),  # nhev,
            stoptol["max_njev"],  # max_nhev,
        )

        if success[0] is not None:
            raise StopIteration

    EPS = 2 * np.finfo(x0.dtype).eps
    print_header_nonlinear()
    try:
        _ = scipy.optimize.least_squares(
            fun_wrapped,
            x0=x0,
            args=(),
            jac=jac_wrapped,
            method=method,
            x_scale=x_scale,
            ftol=EPS,
            xtol=EPS,
            gtol=EPS,
            max_nfev=np.inf,
            verbose=0,
            **options,
        )

    except StopIteration:
        x = jac_allx[-1]
        f = f_where_x(x, fun_allx, fun_allf)
        c = 1 / 2 * np.dot(f, f)
        J = f_where_x(x, jac_allx, jac_allf)
        g = J.T @ f
        result = OptimizeResult(
            x=x,
            success=success[0],
            cost=c,
            fun=f,
            grad=g,
            jac=J,
            optimality=np.linalg.norm(g, ord=np.inf),
            nfev=len(fun_allx),
            njev=len(jac_allx),
            nit=len(jac_allx),
            message=message[0],
            allx=jac_allx,
        )

    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print("         Current function value: {:.3e}".format(result["cost"]))
        print("         Iterations: {:d}".format(result["nit"]))
        print("         Function evaluations: {:d}".format(result["nfev"]))
        print("         Jacobian evaluations: {:d}".format(result["njev"]))

    return result
