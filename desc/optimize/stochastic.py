"""Function for minimizing a scalar function of multiple variables."""

from scipy.optimize import OptimizeResult

from desc.backend import jnp
from desc.utils import errorif, setdefault

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
    options=None,
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
        additional arguments passed to fun and grad
    method : str
        Step size update rule. Currently only the default "sgd" is available. Future
        updates may include RMSProp, Adam, etc.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function.
        The optimization process is stopped when ``dF < ftol * F``.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        Optimization is stopped when ``norm(dx) < xtol * (xtol + norm(x))``.
        If None, the termination by this condition is disabled.
    gtol : float or None, optional
        Absolute tolerance for termination by the norm of the gradient.
        Optimizer terminates when ``max(abs(g)) < gtol``.
        If None, the termination by this condition is disabled.
    verbose : {0, 1, 2}, optional
        * 0 : work silently.
        * 1 (default) : display a termination report.
        * 2 : display progress during iterations
    maxiter : int, optional
        maximum number of iterations. Defaults to size(x)*100
    callback : callable, optional
        Called after each iteration. Should be a callable with
        the signature:

            ``callback(xk, *args) -> bool``

        where ``xk`` is the current parameter vector. and ``args``
        are the same arguments passed to fun and grad. If callback returns True
        the algorithm execution is terminated.
    options : dict, optional
        dictionary of optional keyword arguments to override default solver settings.

        - ``"alpha"`` : (float > 0) Step size parameter. Default
          ``1e-2 * norm(x)/norm(grad(x))``
        - ``"beta"`` : (float > 0) Momentum parameter. Default 0.9

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully.

    """
    options = {} if options is None else options
    nfev = 0
    ngev = 0
    iteration = 0

    N = x0.size
    x = x0.copy()
    v = jnp.zeros_like(x)
    f = fun(x, *args)
    nfev += 1
    g = grad(x, *args)
    ngev += 1

    maxiter = setdefault(maxiter, N * 100)
    g_norm = jnp.linalg.norm(g, ord=2)
    x_norm = jnp.linalg.norm(x, ord=2)
    alpha = options.pop("alpha", 1e-2 * x_norm / g_norm)
    beta = options.pop("beta", 0.9)

    errorif(
        len(options) > 0,
        ValueError,
        "Unknown options: {}".format([key for key in options]),
    )

    callback = setdefault(callback, lambda *args: False)

    success = None
    message = None
    step_norm = jnp.inf
    df_norm = jnp.inf

    if verbose > 2:
        print("Solver options:")
        print("-" * 40)
        print(f"{'Alpha':<15}: {alpha:.3e}")
        print(f"{'Beta':<15}: {beta:.3e}")
        print("-" * 40, "\n")

    if verbose > 1:
        print_header_nonlinear()
        print_iteration_nonlinear(iteration, nfev, f, None, step_norm, g_norm)

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
            jnp.inf,
        )
        if success is not None:
            break

        v = beta * v + (1 - beta) * g
        x = x - alpha * v
        g = grad(x, *args)
        ngev += 1
        step_norm = jnp.linalg.norm(alpha * v, ord=2)
        g_norm = jnp.linalg.norm(g, ord=jnp.inf)
        fnew = fun(x, *args)
        nfev += 1
        df = f - fnew
        f = fnew

        allx.append(x)
        iteration += 1
        if verbose > 1:
            print_iteration_nonlinear(iteration, nfev, f, df, step_norm, g_norm)

        if callback(jnp.copy(x), *args):
            success, message = False, STATUS_MESSAGES["callback"]

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
        allx=allx,
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

    return result


def adam(
    fun,
    x0,
    grad,
    args=(),
    x_scale="auto",
    method="adam",
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    verbose=1,
    maxiter=None,
    callback=None,
    options=None,
):
    """Minimize a scalar function using the ADAM optimizer.

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        x_t = x_{t-1} - alpha * m_hat / (sqrt(v_hat) + epsilon)

    Parameters
    ----------
    fun : callable
        objective to be minimized. Should have a signature like fun(x,*args)-> float
    x0 : array-like
        initial guess
    grad : callable
        function to compute gradient, df/dx. Should take the same arguments as fun
    args : tuple
        additional arguments passed to fun and grad
    x_scale : 'auto' or array-like
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. Improved
        convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function.
    method : str
        Step size update rule.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
    gtol : float or None, optional
        Absolute tolerance for termination by the norm of the gradient.
    verbose : {0, 1, 2}, optional
        * 0 : work silently.
        * 1 (default) : display a termination report.
        * 2 : display progress during iterations
    maxiter : int, optional
        maximum number of iterations. Defaults to size(x)*100
    callback : callable, optional
        Called after each iteration. Should be a callable with
        the signature: ``callback(xk, *args) -> bool``
    options : dict, optional
        dictionary of optional keyword arguments to override default solver settings.

        - ``"alpha"`` : (float > 0) Learning rate. Default 1e-3.
        - ``"beta1"`` : (float > 0) Exponential decay rate for the first moment
          estimates. Default 0.9.
        - ``"beta2"`` : (float > 0) Exponential decay rate for the second moment
          estimates. Default 0.999.
        - ``"epsilon"``: (float > 0) Small constant for numerical stability.
          Default 1e-8.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
    """
    errorif(
        isinstance(x_scale, str) and x_scale not in ["auto"],
        ValueError,
        "x_scale should be one of 'auto' or array-like, got {}".format(x_scale),
    )
    if isinstance(x_scale, str):
        x_scale = 1.0
    options = {} if options is None else options
    nfev = 0
    ngev = 0
    iteration = 0

    N = x0.size
    x = x0.copy()
    x_scale = jnp.broadcast_to(x_scale, x.shape)

    # Adam state initialization
    m = jnp.zeros_like(x)  # First moment estimate
    v = jnp.zeros_like(x)  # Second moment estimate

    f = fun(x, *args)
    nfev += 1
    g = grad(x, *args) / x_scale
    ngev += 1
    # scaled and unscaled norms
    g_norm = jnp.linalg.norm(g * x_scale, ord=2)
    gs_norm = jnp.linalg.norm(g, ord=2)
    x_norm = jnp.linalg.norm(x, ord=2)
    xs_norm = jnp.linalg.norm(x / x_scale, ord=2)
    maxiter = setdefault(maxiter, N * 100)

    # Adam Hyperparameters
    alpha = options.pop("alpha", 1e-1 * xs_norm / gs_norm)
    beta1 = options.pop("beta1", 0.9)
    beta2 = options.pop("beta2", 0.999)
    epsilon = options.pop("epsilon", 1e-8)

    errorif(
        len(options) > 0,
        ValueError,
        "Unknown options: {}".format([key for key in options]),
    )

    callback = setdefault(callback, lambda *args: False)

    success = None
    message = None
    step_norm = jnp.inf
    df_norm = jnp.inf

    if verbose > 2:
        print("Solver options:")
        print("-" * 40)
        print(f"{'Alpha':<15}: {alpha:.3e}")
        print(f"{'Beta1':<15}: {beta1:.3e}")
        print(f"{'Beta2':<15}: {beta2:.3e}")
        print(f"{'Epsilon':<15}: {epsilon:.3e}")
        print("-" * 40, "\n")

    if verbose > 1:
        print_header_nonlinear()
        print_iteration_nonlinear(iteration, nfev, f, None, step_norm, g_norm)

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
            jnp.inf,
        )
        if success is not None:
            break

        t = iteration + 1

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        dx = alpha * m_hat / (jnp.sqrt(v_hat) + epsilon)
        x = x - dx * x_scale
        g = grad(x, *args) / x_scale
        ngev += 1

        # unscaled norms
        step_norm = jnp.linalg.norm(dx * x_scale, ord=2)
        g_norm = jnp.linalg.norm(g * x_scale, ord=jnp.inf)

        fnew = fun(x, *args)
        nfev += 1
        df = f - fnew
        f = fnew

        allx.append(x)
        iteration += 1

        if verbose > 1:
            print_iteration_nonlinear(iteration, nfev, f, df, step_norm, g_norm)

        if callback(jnp.copy(x), *args):
            success, message = False, STATUS_MESSAGES["callback"]

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
        allx=allx,
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

    return result
