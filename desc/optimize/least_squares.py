import numpy as np
from termcolor import colored
from desc.backend import jnp
from .utils import (
    check_termination,
    OptimizeResult,
    print_header_nonlinear,
    print_iteration_nonlinear,
    status_messages,
    compute_jac_scale,
    evaluate_quadratic,
)
from .tr_subproblems import (
    trust_region_step_exact_svd,
    trust_region_step_exact_cho,
    update_tr_radius,
)


def lsqtr(
    fun,
    x0,
    jac,
    args=(),
    x_scale=1,
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    verbose=1,
    maxiter=None,
    tr_method="svd",
    callback=None,
    options={},
):
    """Solve a least squares problem using a (quasi)-Newton trust region method

    Parameters
    ----------
    fun : callable
        objective to be minimized. Should have a signature like fun(x,*args)-> 1d array
    x0 : array-like
        initial guess
    jac : callable:
        function to compute jacobian matrix of fun
    args : tuple
        additional arguments passed to fun, grad, and jac
    x_scale : array_like or ``'jac'``, optional
        Characteristic scale of each variable. Setting ``x_scale`` is equivalent
        to reformulating the problem in scaled variables ``xs = x / x_scale``.
        An alternative view is that the size of a trust region along jth
        dimension is proportional to ``x_scale[j]``. Improved convergence may
        be achieved by setting ``x_scale`` such that a step of a given size
        along any of the scaled variables has a similar effect on the cost
        function. If set to ``'jac'``, the scale is iteratively updated using the
        inverse norms of the columns of the jacobian matrix.
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
        Absolute tolerance for termination by the norm of the gradient. Default is 1e-8.
        Optimizer teriminates when ``norm(g) < gtol``, where
        If None, the termination by this condition is disabled.
    verbose : {0, 1, 2}, optional
        * 0 (default) : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    maxiter : int, optional
        maximum number of iterations. Defaults to size(x)*100
    tr_method : {'cho', 'svd'}
        method to use for solving the trust region subproblem. 'cho' uses a sequence of
        cholesky factorizations (generally 2-3), while 'svd' uses one singular value
        decomposition. 'cho' is generally faster for large systems, especially on GPU,
        but may be less accurate in some cases.
    callback : callable, optional
        Called after each iteration. Should be a callable with
        the signature:

            ``callback(xk, *args) -> bool``

        where ``xk`` is the current parameter vector. and ``args``
        are the same arguments passed to fun and jac. If callback returns True
        the algorithm execution is terminated.
    options : dict, optional
        dictionary of optional keyword arguments to override default solver settings.
        See the code for more details.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        ``OptimizeResult`` for a description of other attributes.

    """
    if tr_method not in ["cho", "svd"]:
        raise ValueError(
            "tr_method should be one of 'cho', 'svd', got {}".format(tr_method)
        )
    if isinstance(x_scale, str) and x_scale not in ["jac", "auto"]:
        raise ValueError(
            "x_scale should be one of 'jac', 'auto' or array-like, got {}".format(
                x_scale
            )
        )

    nfev = 0
    njev = 0
    iteration = 0

    n = x0.size
    x = x0.copy()
    f = fun(x, *args)
    m = f.size
    nfev += 1
    cost = 0.5 * jnp.dot(f, f)
    J = jac(x, *args)
    njev += 1
    g = jnp.dot(J.T, f)

    if maxiter is None:
        maxiter = n * 100
    max_nfev = options.pop("max_nfev", maxiter)
    max_njev = options.pop("max_njev", max_nfev)
    gnorm_ord = options.pop("gnorm_ord", np.inf)
    xnorm_ord = options.pop("xnorm_ord", 2)

    ga_fd_step = options.pop("ga_fd_step", 1e-3)
    ga_tr_ratio = options.pop("ga_tr_ratio", 0)
    return_all = options.pop("return_all", True)
    return_tr = options.pop("return_tr", True)

    jac_scale = isinstance(x_scale, str) and x_scale in ["jac", "auto"]
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        x_scale = np.broadcast_to(x_scale, x.shape)
        scale, scale_inv = x_scale, 1 / x_scale

    # initial trust region radius
    trust_radius = options.pop("initial_trust_radius", np.linalg.norm(x * scale_inv))
    max_trust_radius = options.pop("max_trust_radius", trust_radius * 1000.0)
    min_trust_radius = options.pop("min_trust_radius", 0)
    tr_increase_threshold = options.pop("tr_increase_threshold", 0.75)
    tr_decrease_threshold = options.pop("tr_decrease_threshold", 0.25)
    tr_increase_ratio = options.pop("tr_increase_ratio", 2)
    tr_decrease_ratio = options.pop("tr_decrease_ratio", 0.25)

    if trust_radius == 0:
        trust_radius = 1.0
    if len(options) > 0:
        raise ValueError(
            colored("Unknown options: {}".format([key for key in options]), "red")
        )

    x_norm = np.linalg.norm(x, ord=xnorm_ord)
    success = None
    step_norm = np.inf
    actual_reduction = np.inf
    ratio = 0  # ratio between actual reduction and predicted reduction

    if verbose > 1:
        print_header_nonlinear()

    if return_all:
        allx = [x]
    if return_tr:
        alltr = [trust_radius]

    alpha = 0  # "Levenberg-Marquardt" parameter

    while True:

        g_norm = np.linalg.norm(g, ord=gnorm_ord)
        if g_norm < gtol:
            success = True
        if verbose > 1:
            print_iteration_nonlinear(
                iteration, nfev, cost, actual_reduction, step_norm, g_norm
            )
        if success is not None:
            break

        g_h = g * scale
        J_h = J * scale
        if tr_method == "svd":
            U, s, Vt = jnp.linalg.svd(J_h, full_matrices=False)
        elif tr_method == "cho":
            B_h = jnp.dot(J_h.T, J_h)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            # Solve the sub-problem.
            # This gives us the proposed step relative to the current position
            # and it tells us whether the proposed step
            # has reached the trust region boundary or not.
            if tr_method == "svd":
                step_h, hits_boundary, alpha = trust_region_step_exact_svd(
                    f, U, s, Vt.T, trust_radius, alpha
                )
            elif tr_method == "cho":
                step_h, hits_boundary, alpha = trust_region_step_exact_cho(
                    g_h, B_h, trust_radius, alpha
                )

            step_h_norm = np.linalg.norm(step_h, ord=xnorm_ord)

            # geodesic acceleration
            if ga_tr_ratio > 0:
                f0 = f
                f1 = fun(x + ga_fd_step * step_h * scale, *args)
                nfev += 1
                df = (f1 - f0) / ga_fd_step
                RHS = 2 / ga_fd_step * (df - J.dot(step_h * scale))
                if tr_method == "svd":
                    ga_step_h, _, _ = trust_region_step_exact_svd(
                        RHS, U, s, Vt.T, ga_tr_ratio * step_h_norm, alpha
                    )
                elif tr_method == "cho":
                    RHS = jnp.dot(J_h.T, RHS)
                    ga_step_h, _, _ = trust_region_step_exact_cho(
                        RHS, B_h, ga_tr_ratio * step_h_norm, alpha
                    )
                step_h += ga_step_h

            # calculate the predicted value at the proposed point
            predicted_reduction = -evaluate_quadratic(J_h, g_h, step_h)

            # calculate actual reduction and step norm
            step = scale * step_h
            step_norm = np.linalg.norm(step, ord=xnorm_ord)

            x_new = x + step
            f_new = fun(x_new, *args)
            nfev += 1

            cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new

            # update the trust radius according to the actual/predicted ratio
            tr_old = trust_radius
            trust_radius, ratio = update_tr_radius(
                trust_radius,
                actual_reduction,
                predicted_reduction,
                step_h_norm,
                hits_boundary,
                max_trust_radius,
                min_trust_radius,
                tr_increase_threshold,
                tr_increase_ratio,
                tr_decrease_threshold,
                tr_decrease_ratio,
            )
            if return_tr:
                alltr.append(trust_radius)
            alpha *= tr_old / trust_radius

            success, message = check_termination(
                actual_reduction,
                cost,
                step_norm,
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
                0,
                np.inf,
                njev,
                max_njev,
            )
            if success is not None:
                break

        # if reduction was enough, accept the step
        if actual_reduction > 0:
            x_old = x
            x = x_new
            if return_all:
                allx.append(x)
            f_old = f
            f = f_new
            cost = cost_new
            J = jac(x, *args)
            njev += 1
            g = jnp.dot(J.T, f)
            x_norm = np.linalg.norm(x, ord=xnorm_ord)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)

            if callback is not None:
                stop = callback(np.copy(x), *args)
                if stop:
                    success = False
                    message = status_messages["callback"]
                    break

        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

    result = OptimizeResult(
        x=x,
        success=success,
        cost=cost,
        fun=f,
        grad=g,
        jac=J,
        optimality=g_norm,
        nfev=nfev,
        njev=njev,
        nit=iteration,
        message=message,
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

    if return_all:
        result["allx"] = allx
    if return_tr:
        result["alltr"] = alltr
    return result
