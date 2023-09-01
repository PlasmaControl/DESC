"""Augmented Lagrangian for scalar valued objectives."""

from scipy.optimize import BFGS, NonlinearConstraint, OptimizeResult

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
    inequality_to_bounds,
    print_header_nonlinear,
    print_iteration_nonlinear,
)


def fmin_auglag(  # noqa: C901 - FIXME: simplify this
    fun,
    x0,
    grad,
    hess="bfgs",
    bounds=(-jnp.inf, jnp.inf),
    constraint=None,
    args=(),
    x_scale=1,
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    ctol=1e-6,
    verbose=1,
    maxiter=None,
    options={},
):
    """Minimize a function with constraints using an augmented Lagrangian method.

    Parameters
    ----------
    fun : callable
        objective to be minimized. Should have a signature like fun(x,*args)-> float
    x0 : array-like
        initial guess
    grad : callable
        function to compute gradient, df/dx. Should take the same arguments as fun
    hess : callable
        function to compute Hessian matrix of fun
    bounds : tuple of array-like
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each array must match the size of x0 or be a scalar, in the latter case a
        bound will be the same for all variables. Use np.inf with an appropriate sign
        to disable bounds on all or some variables.
    constraint : scipy.optimize.NonlinearConstraint
        constraint to be satisfied
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
    ctol : float, optional
        Tolerance for stopping based on infinity norm of the constraint violation.
        Optimizer terminates when ``max(abs(constr_violation)) < ctol`` AND one or more
        of the other tolerances are met (``ftol``, ``xtol``, ``gtol``)
    verbose : {0, 1, 2}, optional
        * 0 (default) : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    maxiter : int, optional
        maximum number of iterations. Defaults to size(x)*100
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
    if constraint is None:
        # create a dummy constraint
        constraint = NonlinearConstraint(
            fun=lambda x, *args: jnp.array([0.0]),
            lb=0.0,
            ub=0.0,
            jac=lambda x, *args: jnp.zeros((1, x.size)),
        )
    (
        z0,
        fun_wrapped,
        grad_wrapped,
        hess_wrapped,
        constraint_wrapped,
        zbounds,
        z2xs,
    ) = inequality_to_bounds(
        x0,
        fun,
        grad,
        hess,
        constraint,
        bounds,
    )

    def lagfun(f, c, y, mu, *args):
        return f - jnp.dot(y, c) + mu / 2 * jnp.dot(c, c)

    def laggrad(z, y, mu, *args):
        c = constraint_wrapped.fun(z, *args)
        yJ = constraint_wrapped.vjp(y - mu * c, z, *args)
        return grad_wrapped(z, *args) - yJ

    if isinstance(hess_wrapped, str) and hess_wrapped.lower() == "bfgs":
        bfgs = True
        hess_init_scale = options.pop("hessian_init_scale", "auto")
        hess_exception_strategy = options.pop(
            "hessian_exception_strategy", "damp_update"
        )
        hess_min_curvature = options.pop("hessian_minimum_curvature", None)
        hess_wrapped = BFGS(
            hess_exception_strategy, hess_min_curvature, hess_init_scale
        )
    if isinstance(hess_wrapped, BFGS):
        bfgs = True
        if hasattr(hess_wrapped, "n"):  # assume its already been initialized
            assert hess_wrapped.approx_type == "hess"
            assert hess.n == z0.size
        else:
            hess_wrapped.initialize(z0.size, "hess")
        laghess = hess_wrapped

    elif callable(constraint_wrapped.hess) and callable(hess_wrapped):
        bfgs = False

        def laghess(z, y, mu, *args):
            c = constraint_wrapped.fun(z, *args)
            Hf = hess_wrapped(z, *args)
            Jc = constraint_wrapped.jac(z, *args)
            Hc1 = constraint_wrapped.hess(z, y)
            Hc2 = constraint_wrapped.hess(z, c)
            return Hf - Hc1 + mu * (Hc2 + jnp.dot(Jc.T, Jc))

    elif callable(hess_wrapped):
        bfgs = False

        def laghess(z, y, mu, *args):
            H = hess_wrapped(z, *args)
            J = constraint_wrapped.jac(z, *args)
            # ignoring higher order derivatives of constraints for now
            return H + mu * jnp.dot(J.T, J)

    else:
        raise ValueError("hess should either be a callable or 'bfgs'", "red")

    nfev = 0
    ngev = 0
    nhev = 0
    iteration = 0

    z = z0.copy()
    f = fun_wrapped(z, *args)
    c = constraint_wrapped.fun(z)
    constr_violation = jnp.linalg.norm(c, ord=jnp.inf)
    nfev += 1

    lb, ub = zbounds
    bounded = jnp.any(lb != -jnp.inf) | jnp.any(ub != jnp.inf)
    assert in_bounds(z, lb, ub), "x0 is infeasible"
    z = make_strictly_feasible(z, lb, ub)

    mu = options.pop("initial_penalty_parameter", 10)
    y = options.pop("initial_multipliers", jnp.zeros_like(c))
    if y == "least_squares":  # use least squares multiplier estimates
        _J = constraint_wrapped.jac(z, *args)
        _g = grad_wrapped(z, *args)
        y = jnp.linalg.lstsq(_J.T, _g)[0]
        y = jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # notation following Conn & Gould, algorithm 14.4.2, but with our mu = their mu^-1
    omega = options.pop("omega", 1.0)
    eta = options.pop("eta", 1.0)
    alpha_omega = options.pop("alpha_omega", 1.0)
    beta_omega = options.pop("beta_omega", 1.0)
    alpha_eta = options.pop("alpha_eta", 0.1)
    beta_eta = options.pop("beta_eta", 0.9)
    tau = options.pop("tau", 10)

    gtolk = omega / mu**alpha_omega
    ctolk = eta / mu**alpha_eta

    L = lagfun(f, c, y, mu)
    g = laggrad(z, y, mu, *args)
    ngev += 1
    if bfgs:
        H = laghess.get_matrix()
    else:
        H = laghess(z, y, mu, *args)
        nhev += 1

    allx = []
    return_all = options.pop("return_all", True)
    return_tr = options.pop("return_tr", True)

    if maxiter is None:
        maxiter = z.size * 100
    max_nfev = options.pop("max_nfev", 5 * maxiter + 1)
    max_ngev = options.pop("max_ngev", maxiter + 1)
    max_nhev = options.pop("max_nhev", maxiter + 1)
    max_dx = options.pop("max_dx", jnp.inf)

    hess_scale = isinstance(x_scale, str) and x_scale in ["hess", "auto"]
    if hess_scale:
        scale, scale_inv = compute_hess_scale(H)
    else:
        x_scale = jnp.broadcast_to(x_scale, z.shape)
        scale, scale_inv = x_scale, 1 / x_scale

    v, dv = cl_scaling_vector(z, g, lb, ub)
    v = jnp.where(dv != 0, v * scale_inv, v)
    d = v**0.5 * scale
    diag_h = g * dv * scale

    g_h = g * d
    H_h = d * H * d[:, None]
    g_norm = jnp.linalg.norm(g * v, ord=jnp.inf)

    # initial trust region radius is based on the geometric mean of 2 possible rules:
    # first is the norm of the cauchy point, as recommended in ch17 of Conn & Gould
    # second is the norm of the scaled x, as used in scipy
    # in practice for our problems the C&G one is too small, while scipy is too big,
    # but the geometric mean seems to work well
    init_tr = {
        "scipy": jnp.linalg.norm(z * scale_inv / v**0.5),
        "conngould": (g_h @ g_h) / abs(g_h @ H_h @ g_h),
        "mix": jnp.sqrt(
            (g_h @ g_h)
            / abs(g_h @ H_h @ g_h)
            * jnp.linalg.norm(z * scale_inv / v**0.5)
        ),
    }
    trust_radius = options.pop("initial_trust_radius", "scipy")
    tr_ratio = options.pop("initial_trust_ratio", 1.0)
    trust_radius = init_tr.get(trust_radius, trust_radius)
    trust_radius *= tr_ratio

    max_trust_radius = options.pop("max_trust_radius", trust_radius * 1000.0)
    min_trust_radius = options.pop("min_trust_radius", jnp.finfo(z.dtype).eps)
    tr_increase_threshold = options.pop("tr_increase_threshold", 0.75)
    tr_decrease_threshold = options.pop("tr_decrease_threshold", 0.25)
    tr_increase_ratio = options.pop("tr_increase_ratio", 2)
    tr_decrease_ratio = options.pop("tr_decrease_ratio", 0.25)
    tr_method = options.pop("tr_method", "exact")
    methods = {
        "dogleg": solve_trust_region_dogleg,
        "subspace": solve_trust_region_2d_subspace,
        "exact": trust_region_step_exact_cho,
    }
    subproblem = methods[tr_method]
    if trust_radius == 0:
        trust_radius = 1.0
    if len(options) > 0:
        raise ValueError("Unknown options: {}".format([key for key in options]), "red")

    z_norm = jnp.linalg.norm(z, ord=2)
    success = None
    message = None
    step_norm = jnp.inf
    actual_reduction = jnp.inf
    Lactual_reduction = jnp.inf
    alpha = 0  # "Levenberg-Marquardt" parameter

    if return_all:
        allx = [z]
    if return_tr:
        alltr = [trust_radius]
    if g_norm < gtol and constr_violation < ctol:
        success = True
        message = STATUS_MESSAGES["gtol"]

    if verbose > 1:
        print_header_nonlinear(True, "Penalty param", "max(|mltplr|)")
        print_iteration_nonlinear(
            iteration,
            nfev,
            f,
            actual_reduction,
            step_norm,
            g_norm,
            constr_violation,
            mu,
            jnp.max(jnp.abs(y)),
        )

    while iteration < maxiter and success is None:

        if bounded:
            H_a = H_h + jnp.diag(diag_h)
        else:
            H_a = H_h

        actual_reduction = -1
        Lactual_reduction = -1

        # theta controls step back step ratio from the bounds.
        theta = max(0.995, 1 - g_norm)

        while Lactual_reduction <= 0 and nfev <= max_nfev:
            # Solve the sub-problem.
            # This gives us the proposed step relative to the current position
            # and it tells us whether the proposed step
            # has reached the trust region boundary or not.
            step_h, hits_boundary, alpha = subproblem(g_h, H_a, trust_radius, alpha)

            step = d * step_h  # Trust-region solution in the original space.

            step, step_h, Lpredicted_reduction = select_step(
                z,
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

            z_new = make_strictly_feasible(z + step, lb, ub, rstep=0)
            f_new = fun_wrapped(z_new, *args)
            c_new = constraint_wrapped.fun(z_new)
            L_new = lagfun(f_new, c_new, y, mu)
            nfev += 1

            actual_reduction = f - f_new
            Lactual_reduction = L - L_new

            # update the trust radius according to the actual/predicted ratio
            tr_old = trust_radius
            trust_radius, Lreduction_ratio = update_tr_radius(
                trust_radius,
                Lactual_reduction,
                Lpredicted_reduction,
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
                z_norm,
                g_norm,
                Lreduction_ratio,
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
                dx_total=jnp.linalg.norm(z - z0),
                max_dx=max_dx,
                constr_violation=constr_violation,
                ctol=ctol,
            )
            if success is not None:
                break

        # if reduction was enough, accept the step
        if Lactual_reduction > 0:
            z_old = z
            z = z_new
            if return_all:
                allx.append(z)
            f = f_new
            c = c_new
            constr_violation = jnp.linalg.norm(c, ord=jnp.inf)
            L = L_new
            g_old = g
            g = laggrad(z, y, mu, *args)
            ngev += 1
            if bfgs:
                laghess.update(z - z_old, g - g_old)
                H = laghess.get_matrix()
            else:
                H = laghess(z, y, mu, *args)
                nhev += 1

            if hess_scale:
                scale, scale_inv = compute_hess_scale(H)
            v, dv = cl_scaling_vector(z, g, lb, ub)
            v = jnp.where(dv != 0, v * scale_inv, v)
            g_norm = jnp.linalg.norm(g * v, ord=jnp.inf)

            # updating augmented lagrangian params
            if g_norm < gtolk:  # TODO: maybe also add ftolk, xtolk?
                if constr_violation < ctolk:
                    y = y - mu * c
                    ctolk = ctolk / (mu**beta_eta)
                    gtolk = gtolk / (mu**beta_omega)
                else:
                    mu = tau * mu
                    ctolk = eta / (mu**alpha_eta)
                    gtolk = omega / (mu**alpha_omega)
                # if we update lagrangian params, need to recompute L and J
                L = lagfun(f, c, y, mu)
                g = laggrad(z, y, mu, *args)
                ngev += 1

                if bfgs:
                    laghess.update(z - z_old, g - g_old)
                    H = laghess.get_matrix()
                else:
                    H = laghess(z, y, mu, *args)
                    nhev += 1

                if hess_scale:
                    scale, scale_inv = compute_hess_scale(H, scale_inv)

                v, dv = cl_scaling_vector(z, g, lb, ub)
                v = jnp.where(dv != 0, v * scale_inv, v)
                g_norm = jnp.linalg.norm(g * v, ord=jnp.inf)

            z_norm = jnp.linalg.norm(z, ord=2)
            d = v**0.5 * scale
            diag_h = g * dv * scale
            g_h = g * d
            H_h = d * H * d[:, None]

            if g_norm < gtol and constr_violation < ctol:
                success = True
                message = STATUS_MESSAGES["gtol"]

        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1
        if verbose > 1:
            print_iteration_nonlinear(
                iteration,
                nfev,
                f,
                actual_reduction,
                step_norm,
                g_norm,
                constr_violation,
                mu,
                jnp.max(jnp.abs(y)),
            )

    if g_norm < gtol and constr_violation < ctol:
        success = True
        message = STATUS_MESSAGES["gtol"]
    if (iteration == maxiter) and success is None:
        success = False
        message = STATUS_MESSAGES["maxiter"]
    x, s = z2xs(z)
    active_mask = find_active_constraints(z, zbounds[0], zbounds[1], rtol=xtol)
    result = OptimizeResult(
        x=x,
        s=s,
        y=y,
        success=success,
        fun=f,
        grad=g,
        v=v,
        optimality=g_norm,
        nfev=nfev,
        ngev=ngev,
        nhev=nhev,
        nit=iteration,
        message=message,
        active_mask=active_mask,
        constr_violation=constr_violation,
    )
    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print(f"""         Current function value: {result["fun"]:.3e}""")
        print(f"""         Constraint violation: {result['constr_violation']:.3e}""")
        print(f"""         Total delta_x: {jnp.linalg.norm(x0 - result["x"]):.3e}""")
        print(f"""         Iterations: {result["nit"]:d}""")
        print(f"""         Function evaluations: {result["nfev"]:d}""")
        print(f"""         Gradient evaluations: {result["ngev"]:d}""")
        print(f"""         Hessian evaluations: {result["nhev"]:d}""")

    result["allx"] = [z2xs(x)[0] for x in allx]

    return result
