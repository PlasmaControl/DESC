"""Function for solving nonlinear least squares problems."""

from scipy.optimize import OptimizeResult

from desc.backend import jnp, qr
from desc.utils import errorif, safediv, setdefault

from .bound_utils import (
    cl_scaling_vector,
    find_active_constraints,
    in_bounds,
    make_strictly_feasible,
    select_step,
)
from .tr_subproblems import (
    trust_region_step_exact_cho,
    trust_region_step_exact_qr,
    trust_region_step_exact_svd,
    update_tr_radius,
)
from .utils import (
    STATUS_MESSAGES,
    check_termination,
    compute_jac_scale,
    print_header_nonlinear,
    print_iteration_nonlinear,
    solve_triangular_regularized,
)


def lsqtr(  # noqa: C901
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

        where ``xk`` is the current parameter vector, and ``args``
        are the same arguments passed to fun and jac. If callback returns True
        the algorithm execution is terminated.
    options : dict, optional
        dictionary of optional keyword arguments to override default solver settings.

        - ``"max_nfev"`` : (int > 0) Maximum number of function evaluations (each
          iteration may take more than one function evaluation). Default is
          ``5*maxiter+1``
        - ``"max_dx"`` : (float > 0) Maximum allowed change in the norm of x from its
          starting point. Default np.inf.
        - ``"initial_trust_radius"`` : (``"scipy"``, ``"conngould"``, ``"mix"`` or
          float > 0) Initial trust region radius. ``"scipy"`` uses the scaled norm of
          x0, which is the default behavior in ``scipy.optimize.least_squares``.
          ``"conngould"`` uses the norm of the Cauchy point, as recommended in ch17
          of [1]_. ``"mix"`` uses the geometric mean of the previous two options. A
          float can also be passed to specify the trust radius directly.
          Default is ``"scipy"``.
        - ``"initial_trust_ratio"`` : (float > 0) A extra scaling factor that is
          applied after one of the previous heuristics to determine the initial trust
          radius. Default 1.
        - ``"max_trust_radius"`` : (float > 0) Maximum allowable trust region radius.
          Default ``np.inf``.
        - ``"min_trust_radius"`` : (float >= 0) Minimum allowable trust region radius.
          Optimization is terminated if the trust region falls below this value.
          Default ``np.finfo(x0.dtype).eps``.
        - ``"tr_increase_threshold"`` : (0 < float < 1) Increase the trust region
          radius when the ratio of actual to predicted reduction exceeds this threshold.
          Default 0.75.
        - ``"tr_decrease_threshold"`` : (0 < float < 1) Decrease the trust region
          radius when the ratio of actual to predicted reduction is less than this
          threshold. Default 0.25.
        - ``"tr_increase_ratio"`` : (float > 1) Factor to increase the trust region
          radius by when  the ratio of actual to predicted reduction exceeds threshold.
          Default 2.
        - ``"tr_decrease_ratio"`` : (0 < float < 1) Factor to decrease the trust region
          radius by when  the ratio of actual to predicted reduction falls below
          threshold. Default 0.25.
        - ``"tr_method"`` : (``"qr"``, ``"svd"``, ``"cho"``) Method to use for solving
          the trust region subproblem. ``"qr"`` and ``"cho"`` uses a sequence of QR or
          Cholesky factorizations (generally 2-3), while ``"svd"`` uses one singular
          value decomposition. ``"cho"`` is generally the fastest for large systems,
          especially on GPU, but may be less accurate for badly scaled systems.
          ``"svd"`` is the most accurate but significantly slower. Default ``"qr"``.
        - ``"scaled_termination"`` : Whether to evaluate termination criteria for
          ``xtol`` and ``gtol`` in scaled / normalized units (default) or base units.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        ``OptimizeResult`` for a description of other attributes.

    References
    ----------
    .. [1] Conn, Andrew, and Gould, Nicholas, and Toint, Philippe. "Trust-region
           methods" (2000).

    """
    options = {} if options is None else options
    errorif(
        isinstance(x_scale, str) and x_scale not in ["jac", "auto"],
        ValueError,
        "x_scale should be one of 'jac', 'auto' or array-like, got {}".format(x_scale),
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
    J = jac(x, *args).block_until_ready()  # FIXME: block is needed for jaxify util
    njev += 1
    g = jnp.dot(J.T, f)

    maxiter = setdefault(maxiter, n * 100)
    max_nfev = options.pop("max_nfev", 5 * maxiter + 1)
    max_dx = options.pop("max_dx", jnp.inf)
    scaled_termination = options.pop("scaled_termination", True)

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
    g_norm = jnp.linalg.norm(
        (g * v * scale if scaled_termination else g * v), ord=jnp.inf
    )

    # conngould : norm of the cauchy point, as recommended in ch17 of Conn & Gould
    # scipy : norm of the scaled x, as used in scipy
    # mix : geometric mean of conngould and scipy
    tr_scipy = jnp.linalg.norm(x * scale_inv / v**0.5)
    conngould = safediv(jnp.sum(g_h**2), jnp.sum((J_h @ g_h) ** 2))
    init_tr = {
        "scipy": tr_scipy,
        "conngould": conngould,
        "mix": jnp.sqrt(conngould * tr_scipy),
    }
    trust_radius = options.pop("initial_trust_radius", "scipy")
    tr_ratio = options.pop("initial_trust_ratio", 1.0)
    trust_radius = init_tr.get(trust_radius, trust_radius)
    trust_radius *= tr_ratio
    trust_radius = trust_radius if (trust_radius > 0) else 1.0

    max_trust_radius = options.pop("max_trust_radius", jnp.inf)
    min_trust_radius = options.pop("min_trust_radius", jnp.finfo(x0.dtype).eps)
    tr_increase_threshold = options.pop("tr_increase_threshold", 0.75)
    tr_decrease_threshold = options.pop("tr_decrease_threshold", 0.25)
    tr_increase_ratio = options.pop("tr_increase_ratio", 2)
    tr_decrease_ratio = options.pop("tr_decrease_ratio", 0.25)
    tr_method = options.pop("tr_method", "qr")

    errorif(
        len(options) > 0,
        ValueError,
        "Unknown options: {}".format([key for key in options]),
    )
    errorif(
        tr_method not in ["cho", "svd", "qr"],
        ValueError,
        "tr_method should be one of 'cho', 'svd', 'qr', got {}".format(tr_method),
    )

    callback = setdefault(callback, lambda *args: False)

    x_norm = jnp.linalg.norm(((x * scale_inv) if scaled_termination else x), ord=2)
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

    allx = [x]
    alltr = [trust_radius]
    if g_norm < gtol:
        success, message = True, STATUS_MESSAGES["gtol"]

    alpha = None  # "Levenberg-Marquardt" parameter

    while iteration < maxiter and success is None:

        # we don't want to factorize the extra stuff if we don't need to
        J_a = jnp.vstack([J_h, jnp.diag(diag_h**0.5)]) if bounded else J_h
        f_a = jnp.concatenate([f, jnp.zeros(diag_h.size)]) if bounded else f

        if tr_method == "svd":
            U, s, Vt = jnp.linalg.svd(J_a, full_matrices=False)
        elif tr_method == "cho":
            B_h = jnp.dot(J_a.T, J_a)
        elif tr_method == "qr":
            # try full newton step
            tall = J_a.shape[0] >= J_a.shape[1]
            if tall:
                Q, R = qr(J_a, mode="economic")
                p_newton = solve_triangular_regularized(R, -Q.T @ f_a)
            else:
                Q, R = qr(J_a.T, mode="economic")
                p_newton = Q @ solve_triangular_regularized(R.T, -f_a, lower=True)

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
            elif tr_method == "qr":
                step_h, hits_boundary, alpha = trust_region_step_exact_qr(
                    p_newton, f_a, J_a, trust_radius, alpha
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

            step_h_norm = jnp.linalg.norm(step_h, ord=2)
            step_norm = jnp.linalg.norm(step, ord=2)

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
            alltr.append(trust_radius)
            alpha *= tr_old / trust_radius

            success, message = check_termination(
                actual_reduction,
                cost,
                (step_h_norm if scaled_termination else step_norm),
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
                min_trust_radius=min_trust_radius,
                dx_total=jnp.linalg.norm(x - x0),
                max_dx=max_dx,
            )
            if success is not None:
                break

        # if reduction was enough, accept the step
        if actual_reduction > 0:
            x = x_new
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
            x_norm = jnp.linalg.norm(
                ((x * scale_inv) if scaled_termination else x), ord=2
            )
            g_norm = jnp.linalg.norm(
                (g * v * scale if scaled_termination else g * v), ord=jnp.inf
            )

            if g_norm < gtol:
                success, message = True, STATUS_MESSAGES["gtol"] + f" ({gtol=:.2e})"

            if callback(jnp.copy(x), *args):
                success, message = False, STATUS_MESSAGES["callback"]

        else:
            step_norm = step_h_norm = actual_reduction = 0

        iteration += 1
        if verbose > 1:
            print_iteration_nonlinear(
                iteration, nfev, cost, actual_reduction, step_norm, g_norm
            )

    if g_norm < gtol:
        success, message = True, STATUS_MESSAGES["gtol"] + f" ({gtol=:.2e})"
    if (iteration == maxiter) and success is None:
        success, message = False, STATUS_MESSAGES["maxiter"]
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
        allx=allx,
        alltr=alltr,
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

    return result
