"""Function for minimizing a scalar function of multiple variables."""

from scipy.optimize import BFGS, OptimizeResult

from desc.backend import jnp
from desc.utils import errorif, safediv, setdefault

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


def fmintr(  # noqa: C901
    fun,
    x0,
    grad,
    hess="bfgs",
    bounds=(-jnp.inf, jnp.inf),
    args=(),
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
        are the same arguments passed to fun and grad. If callback returns True
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
        - ``"tr_method"`` : (``"exact"``, ``"dogleg"``, ``"subspace"``) Method to use
          for trust region subproblem. ``"exact"`` uses a series of cholesky
          factorizations (usually 2-3) to find the optimal step. ``"dogleg"``
          approximates the optimal step using Powell's dogleg method. ``"subspace"``
          solves a reduced subproblem over the space spanned by the gradient and Newton
          direction. Default ``"exact"``
        - ``"hessian_exception_strategy"`` : (``"skip_update"``, ``"damp_update"``)
          If BFGS is used, defines how to proceed when the curvature condition is
          violated. Set it to 'skip_update' to just skip the update. Or, alternatively,
          set it to 'damp_update' to interpolate between the actual BFGS
          result and the unmodified matrix. Both exceptions strategies
          are explained  in [2]_, p.536-537. Default is ``"damp_update"``.
        - ``"hessian_min_curvature"`` : (float) If BFGS is used, this number, scaled by
          a normalization factor, defines the minimum curvature
          ``dot(delta_grad, delta_x)`` allowed to go unaffected by the exception
          strategy. By default is equal to 1e-8 when ``exception_strategy =
          "skip_update"`` and equal to 0.2 when ``exception_strategy = "damp_update"``.
        - ``"hessian_init_scale"`` : (float, ``"auto"``) If BFGS is used, the matrix
          scale at first iteration. At the first iteration the Hessian matrix or its
          inverse will be initialized with ``init_scale*np.eye(n)``, where ``n`` is the
          problem dimension. Set it to ``"auto"`` in order to use an automatic heuristic
          for choosing the initial scale. The heuristic is described in [2]_, p.143.
          By default uses ``"auto"``.
        - ``"scaled_termination"`` : Whether to evaluate termination criteria for
          ``xtol`` and ``gtol`` in scaled / normalized units (default) or base units.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully.

    References
    ----------
    .. [1] Conn, Andrew, and Gould, Nicholas, and Toint, Philippe. "Trust-region
           methods" (2000).
    .. [2] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).

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
    errorif(
        not (isinstance(hess, BFGS) or callable(hess)),
        ValueError,
        "hess should either be a callable or 'bfgs'",
    )

    if maxiter is None:
        maxiter = N * 100
    max_nfev = options.pop("max_nfev", 5 * maxiter + 1)
    max_dx = options.pop("max_dx", jnp.inf)
    scaled_termination = options.pop("scaled_termination", True)

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
    # we don't need unscaled H anymore this iteration, so we overwrite
    # it with H_h = d * H * d[:, None] to avoid carrying so many H-sized matrices
    # in memory, which can be large
    # TODO: place this function under JIT (#1669)
    # doing operation H = d * H * d[:, None]
    H *= d[:, None]
    H *= d
    H_h = H
    del H

    g_norm = jnp.linalg.norm(
        (g * v * scale if scaled_termination else g * v), ord=jnp.inf
    )

    # conngould : norm of the cauchy point, as recommended in ch17 of Conn & Gould
    # scipy : norm of the scaled x, as used in scipy
    # mix : geometric mean of conngould and scipy
    tr_scipy = jnp.linalg.norm(x * scale_inv / v**0.5)
    conngould = safediv(g_h @ g_h, abs(g_h @ H_h @ g_h))
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
    tr_method = options.pop("tr_method", "exact")

    errorif(
        len(options) > 0,
        ValueError,
        "Unknown options: {}".format([key for key in options]),
    )

    callback = setdefault(callback, lambda *args: False)

    methods = {
        "dogleg": solve_trust_region_dogleg,
        "subspace": solve_trust_region_2d_subspace,
        "exact": trust_region_step_exact_cho,
    }
    errorif(
        tr_method not in methods,
        ValueError,
        f"tr_method should be one of {methods.keys()}, got {tr_method}",
    )
    subproblem = methods[tr_method]

    x_norm = jnp.linalg.norm(((x * scale_inv) if scaled_termination else x), ord=2)
    success = None
    message = None
    step_norm = jnp.inf
    actual_reduction = jnp.inf
    reduction_ratio = 1

    if verbose > 2:
        print("Solver options:")
        print("-" * 60)
        print(f"{'Maximum Function Evaluations':<35}: {max_nfev}")
        print(f"{'Maximum Allowed Total Δx Norm':<35}: {max_dx:.3e}")
        print(f"{'Scaled Termination':<35}: {scaled_termination}")
        print(f"{'Trust Region Method':<35}: {tr_method}")
        print(f"{'Initial Trust Radius':<35}: {trust_radius:.3e}")
        print(f"{'Maximum Trust Radius':<35}: {max_trust_radius:.3e}")
        print(f"{'Minimum Trust Radius':<35}: {min_trust_radius:.3e}")
        print(f"{'Trust Radius Increase Ratio':<35}: {tr_increase_ratio:.3e}")
        print(f"{'Trust Radius Decrease Ratio':<35}: {tr_decrease_ratio:.3e}")
        print(f"{'Trust Radius Increase Threshold':<35}: {tr_increase_threshold:.3e}")
        print(f"{'Trust Radius Decrease Threshold':<35}: {tr_decrease_threshold:.3e}")
        print("-" * 60, "\n")

    if verbose > 1:
        print_header_nonlinear()
        print_iteration_nonlinear(
            iteration, nfev, f, actual_reduction, step_norm, g_norm
        )

    allx = [x]
    alltr = [trust_radius]

    alpha = 0  # "Levenberg-Marquardt" parameter

    if g_norm < gtol:
        success, message = True, STATUS_MESSAGES["gtol"]

    while iteration < maxiter and success is None:

        H_a = H_h + jnp.diag(diag_h) if bounded else H_h

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
            step_h_norm = jnp.linalg.norm(step_h, ord=2)
            step_norm = jnp.linalg.norm(step, ord=2)

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
            alltr.append(trust_radius)
            alpha *= tr_old / trust_radius

            success, message = check_termination(
                actual_reduction,
                f,
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

            # we don't need unscaled H anymore this iteration, so we overwrite
            # it with H_h = d * H * d[:, None] to avoid carrying so many H-sized
            # matrices in memory, which can be large
            # doing operation H = d * H * d[:, None]
            H *= d[:, None]
            H *= d
            H_h = H
            del H

            x_norm = jnp.linalg.norm(
                ((x * scale_inv) if scaled_termination else x), ord=2
            )
            g_norm = jnp.linalg.norm(
                (g * v * scale if scaled_termination else g * v), ord=jnp.inf
            )

            if g_norm < gtol:
                success, message = True, STATUS_MESSAGES["gtol"]

            if callback(jnp.copy(x), *args):
                success, message = False, STATUS_MESSAGES["callback"]

            allx.append(x)
        else:
            step_norm = step_h_norm = actual_reduction = 0

        iteration += 1
        if verbose > 1:
            print_iteration_nonlinear(
                iteration, nfev, f, actual_reduction, step_norm, g_norm
            )

    if g_norm < gtol:
        success, message = True, STATUS_MESSAGES["gtol"]
    if (iteration == maxiter) and success is None:
        success, message = False, STATUS_MESSAGES["maxiter"]
    active_mask = find_active_constraints(x, lb, ub, rtol=xtol)
    result = OptimizeResult(
        x=x,
        success=success,
        fun=f,
        grad=g,
        v=v,
        hess=H_h / d[:, None] / d,  # unscale the hessian
        optimality=g_norm,
        nfev=nfev,
        ngev=ngev,
        nhev=nhev,
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
        print("         Current function value: {:.3e}".format(result["fun"]))
        print(
            "         Total delta_x: {:.3e}".format(jnp.linalg.norm(x0 - result["x"]))
        )
        print("         Iterations: {:d}".format(result["nit"]))
        print("         Function evaluations: {:d}".format(result["nfev"]))
        print("         Gradient evaluations: {:d}".format(result["ngev"]))
        print("         Hessian evaluations: {:d}".format(result["nhev"]))

    return result
