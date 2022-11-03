"""Function for minimizing a scalar function of multiple variables."""

import numpy as np
from scipy.optimize import OptimizeResult

from .utils import (
    STATUS_MESSAGES,
    check_termination,
    print_header_nonlinear,
    print_iteration_nonlinear,
)


def sgd(
    fun,
    x0,
    grad,
    args=(),
    method="sgd",
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    verbose=1,
    maxiter=None,
    callback=None,
    options={},
):
    """Minimize a scalar function using stochastic gradient descent with momentum.

    Update rule is x_{k+1} = x_{k} - alpha*v_{k}
                   v_{k}   = beta*v_{k-1} + (1-beta)*grad(x)

    Where alpha is the step size and beta is the momentum parameter.

    Parameters
    ----------
    fun : callable
        objective to be minimized. Should have a signature like fun(x,*args)-> float
    x0 : array-like
        initial guess
    grad : callable
        function to compute gradient, df/dx. Should take the same arguments as fun
    args : tuple
        additional arguments passed to fun, grad, and hess
    method : str
        Step size update rule. Currently only the default "sgd" is available. Future
        updates may include RMSProp, Adam, etc.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function. Default
        is 1e-8. The optimization process is stopped when ``dF < ftol * F``,
        and there was an adequate agreement between a local quadratic model and
        the true model in the last step. If None, the termination by this
        condition is disabled.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        Default is 1e-8. Optimization is stopped when
        ``norm(dx) < xtol * (xtol + norm(x))``. If None, the termination by
        this condition is disabled.
    gtol : float or None, optional
        Absolute tolerance for termination by the norm of the gradient. Default is 1e-6.
        Optimizer teriminates when ``norm(g) < gtol``, where
        If None, the termination by this condition is disabled.
    verbose : {0, 1, 2}, optional
        * 0 (default) : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    maxiter : int, optional
        maximum number of iterations. Defaults to size(x)*100
    callback : callable, optional
        Called after each iteration. Should be a callable with
        the signature:

            ``callback(xk, OptimizeResult state) -> bool``

        where ``xk`` is the current parameter vector. and ``state``
        is an ``OptimizeResult`` object, with the same fields
        as the ones from the return. If callback returns True
        the algorithm execution is terminated.
    options : dict, optional
        dictionary of optional keyword arguments to override default solver settings.
        See the code for more details.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully.

    """
    nfev = 0
    ngev = 0
    iteration = 0

    N = x0.size
    x = x0.copy()
    v = np.zeros_like(x)
    f = fun(x, *args)
    nfev += 1
    g = grad(x, *args)
    ngev += 1

    if maxiter is None:
        maxiter = N * 1000
    gnorm_ord = options.pop("gnorm_ord", np.inf)
    xnorm_ord = options.pop("xnorm_ord", 2)
    g_norm = np.linalg.norm(g, ord=gnorm_ord)
    x_norm = np.linalg.norm(x, ord=xnorm_ord)
    return_all = options.pop("return_all", True)
    alpha = options.pop("alpha", 1e-2 * x_norm / g_norm)
    beta = options.pop("beta", 0.9)
    assert len(options) == 0, "sgd got an unexpected option {}".format(options.keys())

    success = None
    message = None
    step_norm = np.inf
    df_norm = np.inf

    if verbose > 1:
        print_header_nonlinear()

    if return_all:
        allx = [x]

    while True:
        success, message = check_termination(
            df_norm,
            f,
            step_norm,
            x_norm,
            g_norm,
            1,
            ftol,
            xtol,
            gtol,
            iteration,
            maxiter,
            nfev,
            np.inf,
            0,
            np.inf,
            0,
            np.inf,
        )
        if success is not None:
            break

        v = beta * v + (1 - beta) * g
        x = x - alpha * v
        g = grad(x, *args)
        ngev += 1
        step_norm = np.linalg.norm(alpha * v, ord=xnorm_ord)
        g_norm = np.linalg.norm(g, ord=gnorm_ord)
        fnew = fun(x, *args)
        nfev += 1
        df = f - fnew
        f = fnew

        if return_all:
            allx.append(x)
        if verbose > 1:
            print_iteration_nonlinear(iteration, nfev, f, df, step_norm, g_norm)

        if callback is not None:
            stop = callback(np.copy(x), *args)
            if stop:
                success = False
                message = STATUS_MESSAGES["callback"]

        iteration += 1

    result = OptimizeResult(
        x=x,
        success=success,
        fun=f,
        grad=g,
        optimality=g_norm,
        nfev=nfev,
        ngev=ngev,
        nit=iteration,
        message=message,
    )
    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print("         Current function value: {:.3e}".format(result["fun"]))
        print("         Iterations: {:d}".format(result["nit"]))
        print("         Function evaluations: {:d}".format(result["nfev"]))
        print("         Gradient evaluations: {:d}".format(result["ngev"]))
    if return_all:
        result["allx"] = allx
    return result
