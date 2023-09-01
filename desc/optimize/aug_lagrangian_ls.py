"""Augmented Lagrangian for vector valued objectives."""

from scipy.optimize import NonlinearConstraint, OptimizeResult

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
    inequality_to_bounds,
    print_header_nonlinear,
    print_iteration_nonlinear,
)


def lsq_auglag(  # noqa: C901 - FIXME: simplify this
    fun,
    x0,
    jac,
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

    The objective function is assumed to be vector valued, and is minimized in the least
    squares sense.

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
        Optimizer terminates when ``norm(g) < gtol``, where
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
        jac_wrapped,
        _,
        constraint_wrapped,
        zbounds,
        z2xs,
    ) = inequality_to_bounds(
        x0,
        fun,
        jac,
        None,
        constraint,
        bounds,
    )

    # L(x,y,mu) = 1/2 f(x)^2 - y*c(x) + mu/2 c(x)^2 + y^2/(2*mu)
    # = 1/2 f(x)^2 + 1/2 [-y/sqrt(mu) + sqrt(mu) c(x)]^2

    def lagfun(f, c, y, mu):
        c = -y / jnp.sqrt(mu) + jnp.sqrt(mu) * c
        return jnp.concatenate((f, c))

    def lagjac(z, y, mu, *args):
        Jf = jac_wrapped(z, *args)
        Jc = constraint_wrapped.jac(z, *args)
        Jc = jnp.sqrt(mu) * Jc
        return jnp.vstack((Jf, Jc))

    nfev = 0
    njev = 0
    iteration = 0

    z = z0.copy()
    f = fun_wrapped(z, *args)
    cost = 1 / 2 * jnp.dot(f, f)
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
        _g = f @ jac_wrapped(z, *args)
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
    J = lagjac(z, y, mu, *args)
    Lcost = 1 / 2 * jnp.dot(L, L)
    g = L @ J

    allx = []
    return_all = options.pop("return_all", True)
    return_tr = options.pop("return_tr", True)

    if maxiter is None:
        maxiter = z.size * 100
    max_nfev = options.pop("max_nfev", 5 * maxiter + 1)
    max_njev = options.pop("max_njev", maxiter + 1)
    max_dx = options.pop("max_dx", jnp.inf)

    jac_scale = isinstance(x_scale, str) and x_scale in ["jac", "auto"]
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        x_scale = jnp.broadcast_to(x_scale, z.shape)
        scale, scale_inv = x_scale, 1 / x_scale

    v, dv = cl_scaling_vector(z, g, lb, ub)
    v = jnp.where(dv != 0, v * scale_inv, v)
    d = v**0.5 * scale
    diag_h = g * dv * scale

    g_h = g * d
    J_h = J * d
    g_norm = jnp.linalg.norm(g * v, ord=jnp.inf)

    init_tr = {
        "scipy": jnp.linalg.norm(z * scale_inv / v**0.5),
        "conngould": jnp.sum(g_h**2) / jnp.sum((J_h @ g_h) ** 2),
        "mix": jnp.sqrt(
            jnp.sum(g_h**2)
            / jnp.sum((J_h @ g_h) ** 2)
            * jnp.linalg.norm(z * scale_inv / v**0.5)
        ),
    }
    trust_radius = options.pop("initial_trust_radius", "scipy")
    tr_ratio = options.pop("initial_trust_ratio", 1.0)
    trust_radius = init_tr.get(trust_radius, trust_radius)
    trust_radius *= tr_ratio
    if trust_radius == 0:
        trust_radius = 1.0

    max_trust_radius = options.pop("max_trust_radius", jnp.inf)
    min_trust_radius = options.pop("min_trust_radius", jnp.finfo(z.dtype).eps)
    tr_increase_threshold = options.pop("tr_increase_threshold", 0.75)
    tr_decrease_threshold = options.pop("tr_decrease_threshold", 0.25)
    tr_increase_ratio = options.pop("tr_increase_ratio", 2)
    tr_decrease_ratio = options.pop("tr_decrease_ratio", 0.25)
    tr_method = options.pop("tr_method", "svd")

    if len(options) > 0:
        raise ValueError("Unknown options: {}".format([key for key in options]))

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
            cost,
            actual_reduction,
            step_norm,
            g_norm,
            constr_violation,
            mu,
            jnp.max(jnp.abs(y)),
        )

    while iteration < maxiter and success is None:

        # we don't want to factorize the extra stuff if we don't need to
        if bounded:
            J_a = jnp.vstack([J_h, jnp.diag(diag_h**0.5)])
            L_a = jnp.concatenate([L, jnp.zeros(diag_h.size)])
        else:
            J_a = J_h
            L_a = L

        if tr_method == "svd":
            U, s, Vt = jnp.linalg.svd(J_a, full_matrices=False)
        elif tr_method == "cho":
            B_h = jnp.dot(J_a.T, J_a)

        actual_reduction = -1
        Lactual_reduction = -1

        # theta controls step back step ratio from the bounds.
        theta = max(0.995, 1 - g_norm)

        while Lactual_reduction <= 0 and nfev <= max_nfev:
            # Solve the sub-problem.
            # This gives us the proposed step relative to the current position
            # and it tells us whether the proposed step
            # has reached the trust region boundary or not.
            if tr_method == "svd":
                step_h, hits_boundary, alpha = trust_region_step_exact_svd(
                    L_a, U, s, Vt.T, trust_radius, alpha
                )
            elif tr_method == "cho":
                step_h, hits_boundary, alpha = trust_region_step_exact_cho(
                    g_h, B_h, trust_radius, alpha
                )
            step = d * step_h  # Trust-region solution in the original space.

            step, step_h, Lpredicted_reduction = select_step(
                z,
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

            z_new = make_strictly_feasible(z + step, lb, ub, rstep=0)
            f_new = fun_wrapped(z_new, *args)
            cost_new = 0.5 * jnp.dot(f_new, f_new)
            c_new = constraint_wrapped.fun(z_new)
            L_new = lagfun(f_new, c_new, y, mu)
            nfev += 1

            Lcost_new = 0.5 * jnp.dot(L_new, L_new)
            actual_reduction = cost - cost_new
            Lactual_reduction = Lcost - Lcost_new

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
                cost,
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
                0,
                jnp.inf,
                njev,
                max_njev,
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
            z = z_new
            if return_all:
                allx.append(z)
            f = f_new
            c = c_new
            constr_violation = jnp.linalg.norm(c, ord=jnp.inf)
            L = L_new
            cost = cost_new
            Lcost = Lcost_new
            J = lagjac(z, y, mu, *args)
            njev += 1
            g = jnp.dot(J.T, L)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
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
                Lcost = 0.5 * jnp.dot(L, L)
                J = lagjac(z, y, mu, *args)
                njev += 1
                g = jnp.dot(J.T, L)

                if jac_scale:
                    scale, scale_inv = compute_jac_scale(J, scale_inv)

                v, dv = cl_scaling_vector(z, g, lb, ub)
                v = jnp.where(dv != 0, v * scale_inv, v)
                g_norm = jnp.linalg.norm(g * v, ord=jnp.inf)

            z_norm = jnp.linalg.norm(z, ord=2)
            d = v**0.5 * scale
            diag_h = g * dv * scale
            g_h = g * d
            J_h = J * d

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
                cost,
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
        constr_violation=constr_violation,
    )
    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print(f"""         Current function value: {result["cost"]:.3e}""")
        print(f"""         Constraint violation: {result['constr_violation']:.3e}""")
        print(f"""         Total delta_x: {jnp.linalg.norm(x0 - result["x"]):.3e}""")
        print(f"""         Iterations: {result["nit"]:d}""")
        print(f"""         Function evaluations: {result["nfev"]:d}""")
        print(f"""         Jacobian evaluations: {result["njev"]:d}""")

    result["allx"] = [z2xs(x)[0] for x in allx]

    return result
