"""Function for minimizing a scalar function of multiple variables."""

from scipy.optimize import BFGS, OptimizeResult
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
    solve_trust_region_2d_subspace,
    solve_trust_region_dogleg,
    trust_region_step_exact_cho,
    update_tr_radius,
)
from .utils import (
    STATUS_MESSAGES,
    check_termination,
    compute_hess_scale,
    print_header_nonlinear,
    print_iteration_nonlinear,
)


def fmintr(  # noqa: C901 - FIXME: simplify this
    fun,
    x0,
    grad,
    hess="bfgs",
    bounds=(-jnp.inf, jnp.inf),
    args=(),
    method="exact",
    x_scale="hess",
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    verbose=1,
    maxiter=None,
    callback=None,
    options=None,
):
    """Minimize a scalar function using a (quasi)-Newton trust region method.

    Parameters
    ----------
    fun : callable
        objective to be minimized. Should have a signature like fun(x,*args)-> float
    x0 : array-like
        initial guess
    grad : callable
        function to compute gradient, df/dx. Should take the same arguments as fun
    hess : callable or ``'bfgs'``, optional:
        function to compute Hessian matrix of fun, or ``'bfgs'`` in which case the BFGS
        method will be used to approximate the Hessian.
    bounds : tuple of array-like
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each array must match the size of x0 or be a scalar, in the latter case a
        bound will be the same for all variables. Use np.inf with an appropriate sign
        to disable bounds on all or some variables.
    args : tuple
        additional arguments passed to fun, grad, and hess
    method : ``'exact'``, ``'dogleg'`` or ``'subspace'``
        method to use for trust region subproblem. 'exact' uses a series of cholesky
        factorizations (usually 2-3) to find the optimal step. `dogleg` approximates the
        optimal step using Powell's dogleg method. 'subspace' solves a reduced
        subproblem over the space spanned by the gradient and newton direction.
    x_scale : array_like or ``'hess'``, optional
        Characteristic scale of each variable. Setting ``x_scale`` is equivalent
        to reformulating the problem in scaled variables ``xs = x / x_scale``.
        An alternative view is that the size of a trust region along jth
        dimension is proportional to ``x_scale[j]``. Improved convergence may
        be achieved by setting ``x_scale`` such that a step of a given size
        along any of the scaled variables has a similar effect on the cost
        function. If set to ``'hess'``, the scale is iteratively updated using the
        inverse norms of the columns of the Hessian matrix.
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
    options = {} if options is None else options
    nfev = 0
    ngev = 0
    nhev = 0
    iteration = 0

    N = x0.size
    x = x0.copy()
    lb, ub = jnp.broadcast_to(bounds[0], x.size), jnp.broadcast_to(bounds[1], x.size)
    bounded = jnp.any(lb != -jnp.inf) | jnp.any(ub != jnp.inf)
    assert in_bounds(x, lb, ub), "x0 is infeasible"
    x = make_strictly_feasible(x, lb, ub)

    f = fun(x, *args)
    nfev += 1
    g = grad(x, *args)
    ngev += 1

    if isinstance(hess, str) and hess.lower() == "bfgs":
        hess_init_scale = options.pop("hessian_init_scale", "auto")
        hess_exception_strategy = options.pop(
            "hessian_exception_strategy", "damp_update"
        )
        hess_min_curvature = options.pop("hessian_minimum_curvature", None)
        hess = BFGS(hess_exception_strategy, hess_min_curvature, hess_init_scale)
    if callable(hess):
        H = hess(x, *args)
        nhev += 1
        bfgs = False
    elif isinstance(hess, BFGS):
        if hasattr(hess, "n"):  # assume its already been initialized
            assert hess.approx_type == "hess"
            assert hess.n == N
        else:
            hess.initialize(N, "hess")
        bfgs = True
        H = hess.get_matrix()
    else:
        raise ValueError(colored("hess should either be a callable or 'bfgs'", "red"))

    if method == "dogleg":
        subproblem = solve_trust_region_dogleg
    elif method == "subspace":
        subproblem = solve_trust_region_2d_subspace
    elif method == "exact":
        subproblem = trust_region_step_exact_cho
    else:
        raise ValueError(
            colored("method should be one of 'exact', 'dogleg' or 'subspace'", "red")
        )

    if maxiter is None:
        maxiter = N * 100
    max_nfev = options.pop("max_nfev", 5 * maxiter + 1)
    max_ngev = options.pop("max_ngev", maxiter + 1)
    max_nhev = options.pop("max_nhev", maxiter + 1)
    gnorm_ord = options.pop("gnorm_ord", jnp.inf)
    xnorm_ord = options.pop("xnorm_ord", 2)
    return_all = options.pop("return_all", True)
    return_tr = options.pop("return_tr", True)
    max_dx = options.pop("max_dx", jnp.inf)

    hess_scale = isinstance(x_scale, str) and x_scale in ["hess", "auto"]
    if hess_scale:
        scale, scale_inv = compute_hess_scale(H)
    else:
        x_scale = jnp.broadcast_to(x_scale, x.shape)
        scale, scale_inv = x_scale, 1 / x_scale

    v, dv = cl_scaling_vector(x, g, lb, ub)
    v = jnp.where(dv != 0, v * scale_inv, v)
    d = v**0.5 * scale
    diag_h = g * dv * scale

    g_h = g * d
    H_h = d * H * d[:, None]
    g_norm = jnp.linalg.norm(g * v, ord=gnorm_ord)

    # initial trust region radius is based on the geometric mean of 2 possible rules:
    # first is the norm of the cauchy point, as recommended in ch17 of Conn & Gould
    # second is the norm of the scaled x, as used in scipy
    # in practice for our problems the C&G one is too small, while scipy is too big,
    # but the geometric mean seems to work well
    init_tr = {
        "scipy": jnp.linalg.norm(x * scale_inv / v**0.5),
        "conngould": (g_h @ g_h) / abs(g_h @ H_h @ g_h),
        "mix": jnp.sqrt(
            (g_h @ g_h)
            / abs(g_h @ H_h @ g_h)
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
    reduction_ratio = 1

    if verbose > 1:
        print_header_nonlinear()
        print_iteration_nonlinear(
            iteration, nfev, f, actual_reduction, step_norm, g_norm
        )

    if return_all:
        allx = [x]
    if return_tr:
        alltr = [trust_radius]

    alpha = 0  # "Levenberg-Marquardt" parameter

    if g_norm < gtol:
        success = True
        message = STATUS_MESSAGES["gtol"]

    while iteration < maxiter and success is None:

        if bounded:
            H_a = H_h + jnp.diag(diag_h)
        else:
            H_a = H_h

        actual_reduction = -1

        # theta controls step back step ratio from the bounds.
        theta = max(0.995, 1 - g_norm)

        while actual_reduction <= 0 and nfev <= max_nfev:
            # Solve the sub-problem.
            # This gives us the proposed step relative to the current position
            # and it tells us whether the proposed step
            # has reached the trust region boundary or not.
            step_h, hits_boundary, alpha = subproblem(g_h, H_a, trust_radius, alpha)

            step = d * step_h  # Trust-region solution in the original space.

            step, step_h, predicted_reduction = select_step(
                x,
                H_h,
                diag_h,
                g_h,
                step,
                step_h,
                d,
                trust_radius,
                lb,
                ub,
                theta,
                mode="hess",
            )

            # calculate actual reduction and step norm
            step_h_norm = jnp.linalg.norm(step_h, ord=xnorm_ord)
            step_norm = jnp.linalg.norm(step, ord=xnorm_ord)

            x_new = make_strictly_feasible(x + step, lb, ub, rstep=0)
            f_new = fun(x_new, *args)
            nfev += 1
            actual_reduction = f - f_new

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
                f,
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
                ngev,
                max_ngev,
                nhev,
                max_nhev,
                min_trust_radius=min_trust_radius,
                dx_total=jnp.linalg.norm(x - x0),
                max_dx=max_dx,
            )
            if success is not None:
                break

        # if reduction was enough, accept the step
        if actual_reduction > 0:
            x_old = x
            x = x_new
            f = f_new
            g_old = g
            g = grad(x, *args)
            ngev += 1
            if bfgs:
                hess.update(x - x_old, g - g_old)
                H = hess.get_matrix()
            else:
                H = hess(x, *args)
                nhev += 1

            if hess_scale:
                scale, scale_inv = compute_hess_scale(H)

            v, dv = cl_scaling_vector(x, g, lb, ub)
            v = jnp.where(dv != 0, v * scale_inv, v)
            d = v**0.5 * scale
            diag_h = g * dv * scale

            g_h = g * d
            H_h = d * H * d[:, None]

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

            if return_all:
                allx.append(x)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1
        if verbose > 1:
            print_iteration_nonlinear(
                iteration, nfev, f, actual_reduction, step_norm, g_norm
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
        fun=f,
        grad=g,
        v=v,
        hess=H,
        optimality=g_norm,
        nfev=nfev,
        ngev=ngev,
        nhev=nhev,
        nit=iteration,
        message=message,
        active_mask=active_mask,
    )
    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print("         Current function value: {:.3e}".format(result["fun"]))
        print(
            "         Total delta_x: {:.3e}".format(jnp.linalg.norm(x0 - result["x"]))
        )
        print("         Iterations: {:d}".format(result["nit"]))
        print("         Function evaluations: {:d}".format(result["nfev"]))
        print("         Gradient evaluations: {:d}".format(result["ngev"]))
        print("         Hessian evaluations: {:d}".format(result["nhev"]))
    if return_all:
        result["allx"] = allx
    if return_tr:
        result["alltr"] = alltr
    return result
