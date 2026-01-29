"""Function for minimizing a scalar function of multiple variables."""

import warnings

import optax
from scipy.optimize import OptimizeResult

from desc.backend import jnp
from desc.utils import errorif, setdefault, warnif

from .utils import (
    STATUS_MESSAGES,
    check_termination,
    print_header_nonlinear,
    print_iteration_nonlinear,
)


def sgd(  # noqa: C901
    fun,
    x0,
    grad,
    args=(),
    method="sgd",
    x_scale="auto",
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    verbose=1,
    maxiter=None,
    callback=None,
    options=None,
):
    r"""Minimize a scalar function using one of stochastic gradient descent methods.

    This is the generic function. The update method is chosen based on the `method`
    argument.

    Update rule for ``'sgd'``,

    v_k     =  β * v_{k-1} + (1-β) * ∇f(x_k)

    x_{k+1} =  x_k - α * v_k

    where α is the step size and β is the momentum parameter.

    Additionally, optax optimizers can be used by specifying the method as
    ``'optax-<optimizer_name>'``, where ``<optimizer_name>`` is any valid optax
    optimizer. Hyperparameters for the optax optimizer must be passed via the
    `optax-options` key of `options` dictionary.

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
        Name of the method to use. Available options are `'sgd'`.
        Additionally, ``optax`` optimizers can be used by specifying the method as
        ``'optax-<optimizer_name>'``, where ``<optimizer_name>`` is any valid ``optax``
        optimizer. Hyperparameters for the ``optax`` optimizer must be passed via the
        ``'optax-options'`` key of ``options`` dictionary. A custom ``optax``
        optimizer can be used by specifying the method as ``'optax-custom'`` and
        passing the ``optax`` optimizer via the ``'update-rule'`` key of
        ``'optax-options'`` in the ``options`` dictionary.
    x_scale : array_like or 'auto', optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. Improved
        convergence may be achieved by setting x_scale such that a step of a given
        size along any of the scaled variables has a similar effect on the cost
        function. Defaults to 'auto', meaning no scaling.
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
        dictionary of optional keyword arguments to override default solver settings
        for the update rule chosen.

        - ``"alpha"`` : (float > 0) Learning rate. Defaults to
          1e-2 * ||x_scaled|| / ||g_scaled||.
        - ``"beta"`` : (float > 0) Exponential decay rate for the first moment
          estimates. Default 0.9.

        For optax optimizers, hyperparameters specific to the chosen optimizer
        must be passed via the `optax-options` key of `options` dictionary.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully.

    Examples
    --------
    One can use custom ``optax`` optimizers as follows:

    .. code-block:: python

        import optax
        from desc.optimize import Optimizer
        from desc.examples import get

        eq = get("DSHAPE")

        # Optimizer
        opt = optax.chain(
            optax.sgd(learning_rate=1.0),
            optax.scale_by_zoom_linesearch(max_linesearch_steps=15),
        )
        optimizer = Optimizer("optax-custom")
        eq.solve(optimizer=optimizer, options={"optax-options": {"update_rule": opt}})

    """
    errorif(
        method not in ["sgd"] and "optax-" not in method,
        "Available options for method are 'sgd' or "
        f"any optax optimizer wrapper by 'optax-', but {method} "
        "is given.",
    )
    errorif(
        isinstance(x_scale, str) and x_scale not in ["auto"],
        ValueError,
        "x_scale should be one of 'auto' or array-like, got {}".format(x_scale),
    )
    deprecated_sgd = False
    if method == "sgd":
        # warn the user but do not fail the pytest
        with warnings.catch_warnings():
            warnings.filterwarnings("default", "'sgd' method is deprecated")
            warnif(
                True,
                DeprecationWarning,
                "'sgd' method is deprecated and will be removed in a future "
                "release in favor of 'optax-sgd'",
            )
        method = "optax-sgd"
        deprecated_sgd = True
    if isinstance(x_scale, str):
        x_scale = 1.0

    options = {} if options is None else options
    nfev = 0
    ngev = 0
    iteration = 0

    N = x0.size
    x = x0.copy()
    f = fun(x, *args)
    nfev += 1
    # Scaled state xs = x / x_scale
    # Scaled gradient df/dxs = df/dx * dx/dxs = df/dx * x_scale
    gs = grad(x, *args) * x_scale
    ngev += 1
    # scaled and unscaled norms
    g_norm = jnp.linalg.norm(gs / x_scale, ord=2)
    gs_norm = jnp.linalg.norm(gs, ord=2)
    x_norm = jnp.linalg.norm(x, ord=2)
    xs_norm = jnp.linalg.norm(x / x_scale, ord=2)
    maxiter = setdefault(maxiter, N * 100)

    alpha = options.pop("alpha", 1e-2 * xs_norm / gs_norm)
    # check for zero or nan step size
    if alpha == 0 or jnp.isnan(alpha):
        alpha = 1e-3  # default small step size
    if "optax-" in method and method != "optax-custom":
        method_options = options.pop("optax-options", {})
        if method not in ["optax-lbfgs", "optax-polyak_sgd"]:
            # L-BFGS uses its own line search, so don't set learning rate if not given
            method_options["learning_rate"] = method_options.get("learning_rate", alpha)
        if method == "optax-noisy_sgd":
            # noisy_sgd requires a key for random number generation
            method_options["key"] = method_options.get("key", 0)
        if deprecated_sgd and "momentum" not in method_options:
            method_options["momentum"] = options.pop("beta", 0.9)
            method_options["nesterov"] = True

    elif method == "optax-custom":
        method_options = options.pop("optax-options", {})
        errorif(
            "update_rule" not in method_options,
            ValueError,
            "For 'optax-custom' method, 'update_rule' must be provided in "
            "'optax-options' of options dictionary.",
        )
        optax_method = method_options.pop("update_rule")
        errorif(
            not (hasattr(optax_method, "init") and hasattr(optax_method, "update")),
            ValueError,
            "'update_rule' must be a valid optax optimizer with 'init' and "
            "'update' methods.",
        )
        errorif(
            len(method_options) > 0,
            ValueError,
            "Cannot pass hyperparameters for custom optax optimizer.",
        )

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
        for key, val in method_options.items():
            print(f"{key:<15}: {val}")
        print("-" * 40, "\n")

    if verbose > 1:
        print_header_nonlinear()
        print_iteration_nonlinear(iteration, nfev, f, None, step_norm, g_norm)

    allx = [x]
    if method != "optax-custom":
        optax_method = getattr(optax, method.replace("optax-", ""))(**method_options)
    opt_state = optax_method.init(x)
    # necessary for linesearch
    optax_fun = lambda xs, *args: fun(xs * x_scale, *args)

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

        dxs, opt_state = optax_method.update(
            gs, opt_state, x / x_scale, value=f, grad=gs, value_fn=optax_fun
        )
        dx = dxs * x_scale
        x = x + dx
        gs = grad(x, *args) * x_scale
        g_norm = jnp.linalg.norm(gs / x_scale, ord=2)
        step_norm = jnp.linalg.norm(dx, ord=2)
        fnew = fun(x, *args)
        df = f - fnew
        df_norm = jnp.abs(df)
        x_norm = jnp.linalg.norm(x, ord=2)
        f = fnew

        ngev += 1
        nfev += 1

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
        grad=gs / x_scale,
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
