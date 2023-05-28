"""Augmented Langrangian for vector valued objectives."""

import numpy as np
from scipy.optimize import NonlinearConstraint, OptimizeResult

from desc.backend import jnp
from desc.optimize.least_squares import lsqtr

from .bound_utils import find_active_constraints
from .utils import (
    check_termination,
    inequality_to_bounds,
    print_header_nonlinear,
    print_iteration_nonlinear,
)


def fmin_lag_ls_stel(  # noqa: C901 - FIXME: simplify this
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
    """Minimize a function with constraints using an augmented Langrangian method.

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

    def lagfun(z, lmbda, mu, *args):
        f = fun_wrapped(z, *args)
        c = constraint_wrapped.fun(z, *args)
        c = 1 / (np.sqrt(2 * mu)) * (-lmbda + mu * c)
        return jnp.concatenate((f, c))

    def lagjac(z, lmbda, mu, *args):
        Jf = jac_wrapped(z, *args)
        Jc = constraint_wrapped.jac(z, *args)
        Jc = 1 / (np.sqrt(2 * mu)) * (mu * Jc)
        return jnp.vstack((Jf, Jc))

    nfev = 0
    njev = 0
    iteration = 0

    z = z0.copy()
    f = fun_wrapped(z, *args)
    J = jac_wrapped(z, *args)
    cost = 1 / 2 * jnp.dot(f, f)
    g = jnp.dot(f, J)
    c = constraint_wrapped.fun(z)
    nfev += 1
    njev += 1

    if maxiter is None:
        maxiter = z.size
    mu = options.pop("initial_penalty_parameter", 10)
    lmbda = options.pop("initial_multipliers", None)
    if lmbda is None:  # use least squares multiplier estimates
        _J = constraint_wrapped.jac(z, *args)
        _g = f @ jac_wrapped(z, *args)
        lmbda = jnp.linalg.lstsq(_J.T, _g)[0]

    maxiter_inner = options.pop("maxiter_inner", 20)
    max_nfev = options.pop("max_nfev", 5 * maxiter * maxiter_inner + 1)
    max_njev = options.pop("max_njev", maxiter * maxiter_inner + 1)

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
    zold = z
    cost_old = cost
    allx = []

    success = None
    message = None
    step_norm = jnp.inf
    actual_reduction = jnp.inf
    g_norm = np.linalg.norm(g, ord=np.inf)
    constr_violation = np.linalg.norm(c, ord=np.inf)

    options.setdefault("initial_trust_radius", "scipy")
    options["return_tr"] = True

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
            jnp.max(jnp.abs(lmbda)),
        )

    while iteration < maxiter:
        result = lsqtr(
            lagfun,
            z,
            lagjac,
            bounds=zbounds,
            args=(lmbda, mu) + args,
            x_scale=x_scale,
            ftol=0,
            xtol=0,
            gtol=gtolk,
            maxiter=maxiter_inner,
            verbose=0,
            options=options.copy(),
        )
        # update outer counters
        allx += result["allx"]
        nfev += result["nfev"]
        njev += result["njev"]
        z = result["x"]
        f = fun_wrapped(z, *args)
        cost = 1 / 2 * jnp.dot(f, f)
        c = constraint_wrapped.fun(z)
        nfev += 1
        constr_violation = np.linalg.norm(c, ord=np.inf)
        step_norm = np.linalg.norm(zold - z)
        z_norm = np.linalg.norm(z)
        g_norm = result["optimality"]
        actual_reduction = cost_old - cost
        # don't stop if we increased cost
        reduction_ratio = jnp.sign(actual_reduction)
        # reuse the previous trust radius on the next pass
        options["initial_trust_radius"] = float(result["alltr"][-1])
        iteration = iteration + 1

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
                jnp.max(jnp.abs(lmbda)),
            )

        # check if we can stop the outer loop
        success, message = check_termination(
            actual_reduction,
            cost,
            step_norm,
            z_norm,
            g_norm,
            reduction_ratio,
            ftol,
            xtol,
            gtol,
            iteration,
            maxiter,
            nfev,
            max_nfev,
            njev,
            max_njev,
            0,
            np.inf,
            constr_violation=constr_violation,
            ctol=ctol,
        )
        if success is not None:
            break

        if not result["success"]:  # did the subproblem actually finish, or maxiter?
            continue
        elif constr_violation < ctolk:
            lmbda = lmbda - mu * c
            ctolk = ctolk / (mu**beta_eta)
            gtolk = gtolk / (mu**beta_omega)
        else:
            mu = tau * mu
            ctolk = eta / (mu**alpha_eta)
            gtolk = omega / (mu**alpha_omega)

        zold = z
        cost_old = cost

    x, s = z2xs(z)
    active_mask = find_active_constraints(z, zbounds[0], zbounds[1], rtol=xtol)
    result = OptimizeResult(
        x=x,
        y=lmbda,
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
