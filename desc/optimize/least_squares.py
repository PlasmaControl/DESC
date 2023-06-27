"""Function for solving nonlinear least squares problems."""

from scipy.optimize import OptimizeResult
from termcolor import colored

from desc.backend import jnp

from .bound_utils import (
    cl_scaling_vector,
    find_active_constraints,
    in_bounds,
    make_strictly_feasible,
    select_step,
)
from .tr_subproblems import (
    trust_region_step_exact_cho,
    trust_region_step_exact_svd,
    update_tr_radius,
)
from .utils import (
    STATUS_MESSAGES,
    check_termination,
    compute_jac_scale,
    print_header_nonlinear,
    print_iteration_nonlinear,
)


def lsqtr(  # noqa: C901 - FIXME: simplify this
    fun,
    x0,
    jac,
    bounds=(-jnp.inf, jnp.inf),
    args=(),
    x_scale="jac",
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    verbose=1,
    maxiter=None,
    tr_method="svd",
    callback=None,
    options=None,
):
    """Solve a least squares problem using a (quasi)-Newton trust region method.

    Parameters
    ----------
    fun : callable
        objective to be minimized. Should have a signature like fun(x,*args)-> 1d array
    x0 : array-like
        initial guess
    jac : callable:
        function to compute Jacobian matrix of fun
    bounds : tuple of array-like
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each array must match the size of x0 or be a scalar, in the latter case a
        bound will be the same for all variables. Use np.inf with an appropriate sign
        to disable bounds on all or some variables.
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
        inverse norms of the columns of the Jacobian matrix.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function.
        The optimization process is stopped when ``dF < ftol * F``,
        and there was an adequate agreement between a local quadratic model and
        the true model in the last step. If None, the termination by this
        condition is disabled.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        Optimization is stopped when ``norm(dx) < xtol * (xtol + norm(x))``.
        If None, the termination by this condition is disabled.
    gtol : float or None, optional
        Absolute tolerance for termination by the norm of the gradient.
        Optimizer teriminates when ``max(abs(g)) < gtol``.
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
    options = {} if options is None else options
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
    lb, ub = jnp.broadcast_to(bounds[0], x.size), jnp.broadcast_to(bounds[1], x.size)
    bounded = jnp.any(lb != -jnp.inf) | jnp.any(ub != jnp.inf)
    assert in_bounds(x, lb, ub), "x0 is infeasible"
    x = make_strictly_feasible(x, lb, ub)

    f = fun(x, *args)
    nfev += 1
    cost = 0.5 * jnp.dot(f, f)
    J = jac(x, *args)
    njev += 1
    g = jnp.dot(J.T, f)

    if maxiter is None:
        maxiter = n * 100
    max_nfev = options.pop("max_nfev", 5 * maxiter + 1)
    max_njev = options.pop("max_njev", maxiter + 1)
    gnorm_ord = options.pop("gnorm_ord", jnp.inf)
    xnorm_ord = options.pop("xnorm_ord", 2)
    max_dx = options.pop("max_dx", jnp.inf)

    return_all = options.pop("return_all", True)
    return_tr = options.pop("return_tr", True)

    jac_scale = isinstance(x_scale, str) and x_scale in ["jac", "auto"]
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        x_scale = jnp.broadcast_to(x_scale, x.shape)
        scale, scale_inv = x_scale, 1 / x_scale

    v, dv = cl_scaling_vector(x, g, lb, ub)
    v = jnp.where(dv != 0, v * scale_inv, v)
    d = v**0.5 * scale
    diag_h = g * dv * scale

    g_h = g * d
    J_h = J * d
    g_norm = jnp.linalg.norm(g * v, ord=gnorm_ord)

    # initial trust region radius is based on the geometric mean of 2 possible rules:
    # first is the norm of the cauchy point, as recommended in ch17 of Conn & Gould
    # second is the norm of the scaled x, as used in scipy
    # in practice for our problems the C&G one is too small, while scipy is too big,
    # but the geometric mean seems to work well
    init_tr = {
        "scipy": jnp.linalg.norm(x * scale_inv / v**0.5),
        "conngould": jnp.sum(g_h**2) / jnp.sum((J_h @ g_h) ** 2),
        "mix": jnp.sqrt(
            jnp.sum(g_h**2)
            / jnp.sum((J_h @ g_h) ** 2)
            * jnp.linalg.norm(x * scale_inv / v**0.5)
        ),
    }
    trust_radius = options.pop("initial_trust_radius", "scipy")
    tr_ratio = options.pop("initial_trust_ratio", 1.0)
    trust_radius = init_tr.get(trust_radius, trust_radius)
    trust_radius *= tr_ratio

    max_trust_radius = options.pop("max_trust_radius", trust_radius * 1000.0)
    min_trust_radius = options.pop("min_trust_radius", jnp.finfo(x0.dtype).eps)
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

    x_norm = jnp.linalg.norm(x, ord=xnorm_ord)
    success = None
    message = None
    step_norm = jnp.inf
    actual_reduction = jnp.inf
    reduction_ratio = 0

    if verbose > 1:
        print_header_nonlinear()
        print_iteration_nonlinear(
            iteration, nfev, cost, actual_reduction, step_norm, g_norm
        )

    if return_all:
        allx = [x]
    if return_tr:
        alltr = [trust_radius]
    if g_norm < gtol:
        success = True
        message = STATUS_MESSAGES["gtol"]

    alpha = 0  # "Levenberg-Marquardt" parameter

    while iteration < maxiter and success is None:

        # we don't want to factorize the extra stuff if we don't need to
        if bounded:
            J_a = jnp.vstack([J_h, jnp.diag(diag_h**0.5)])
            f_a = jnp.concatenate([f, jnp.zeros(diag_h.size)])
        else:
            J_a = J_h
            f_a = f

        if tr_method == "svd":
            U, s, Vt = jnp.linalg.svd(J_a, full_matrices=False)
        elif tr_method == "cho":
            B_h = jnp.dot(J_a.T, J_a)

        actual_reduction = -1

        # theta controls step back step ratio from the bounds.
        theta = max(0.995, 1 - g_norm)

        while actual_reduction <= 0 and nfev <= max_nfev:
            # Solve the sub-problem.
            # This gives us the proposed step relative to the current position
            # and it tells us whether the proposed step
            # has reached the trust region boundary or not.
            if tr_method == "svd":
                step_h, hits_boundary, alpha = trust_region_step_exact_svd(
                    f_a, U, s, Vt.T, trust_radius, alpha
                )
            elif tr_method == "cho":
                step_h, hits_boundary, alpha = trust_region_step_exact_cho(
                    g_h, B_h, trust_radius, alpha
                )
            step = d * step_h  # Trust-region solution in the original space.

            step, step_h, predicted_reduction = select_step(
                x,
                J_h,
                diag_h,
                g_h,
                step,
                step_h,
                d,
                trust_radius,
                lb,
                ub,
                theta,
                mode="jac",
            )

            step_h_norm = jnp.linalg.norm(step_h, ord=xnorm_ord)
            step_norm = jnp.linalg.norm(step, ord=xnorm_ord)

            x_new = make_strictly_feasible(x + step, lb, ub, rstep=0)
            f_new = fun(x_new, *args)
            nfev += 1

            cost_new = 0.5 * jnp.dot(f_new, f_new)
            actual_reduction = cost - cost_new

            # update the trust radius according to the actual/predicted ratio
            tr_old = trust_radius
            trust_radius, reduction_ratio = update_tr_radius(
                trust_radius,
                actual_reduction,
                predicted_reduction,
                step_h_norm,
                hits_boundary,
                max_trust_radius,
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
                reduction_ratio,
                ftol,
                xtol,
                gtol,
                iteration,
                maxiter,
                nfev,
                max_nfev,
                0,
                jnp.inf,
                njev,
                max_njev,
                min_trust_radius=min_trust_radius,
                dx_total=jnp.linalg.norm(x - x0),
                max_dx=max_dx,
            )
            if success is not None:
                break

        # if reduction was enough, accept the step
        if actual_reduction > 0:
            x = x_new
            if return_all:
                allx.append(x)
            f = f_new
            cost = cost_new
            J = jac(x, *args)
            njev += 1
            g = jnp.dot(J.T, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)

            v, dv = cl_scaling_vector(x, g, lb, ub)
            v = jnp.where(dv != 0, v * scale_inv, v)
            d = v**0.5 * scale
            diag_h = g * dv * scale

            g_h = g * d
            J_h = J * d
            x_norm = jnp.linalg.norm(x, ord=xnorm_ord)
            g_norm = jnp.linalg.norm(g * v, ord=gnorm_ord)
            if g_norm < gtol:
                success = True
                message = STATUS_MESSAGES["gtol"]

            if callback is not None:
                stop = callback(jnp.copy(x), *args)
                if stop:
                    success = False
                    message = STATUS_MESSAGES["callback"]

        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1
        if verbose > 1:
            print_iteration_nonlinear(
                iteration, nfev, cost, actual_reduction, step_norm, g_norm
            )

    if g_norm < gtol:
        success = True
        message = STATUS_MESSAGES["gtol"]
    if (iteration == maxiter) and success is None:
        success = False
        message = STATUS_MESSAGES["maxiter"]
    active_mask = find_active_constraints(x, lb, ub, rtol=xtol)
    result = OptimizeResult(
        x=x,
        success=success,
        cost=cost,
        fun=f,
        grad=g,
        v=v,
        jac=J,
        optimality=g_norm,
        nfev=nfev,
        njev=njev,
        nit=iteration,
        message=message,
        active_mask=active_mask,
    )
    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print("         Current function value: {:.3e}".format(result["cost"]))
        print(
            "         Total delta_x: {:.3e}".format(jnp.linalg.norm(x0 - result["x"]))
        )
        print("         Iterations: {:d}".format(result["nit"]))
        print("         Function evaluations: {:d}".format(result["nfev"]))
        print("         Jacobian evaluations: {:d}".format(result["njev"]))

    if return_all:
        result["allx"] = allx
    if return_tr:
        result["alltr"] = alltr
    return result
