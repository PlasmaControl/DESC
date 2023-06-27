"""Augmented Langrangian for scalar valued objectives."""

from scipy.optimize import BFGS, NonlinearConstraint, OptimizeResult
from termcolor import colored

from desc.backend import jnp
from desc.optimize.fmin_scalar import fmintr

from .bound_utils import find_active_constraints
from .utils import (
    check_termination,
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
    """Minimize a function with constraints using an augmented Langrangian method.

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
        Optimizer teriminates when ``max(abs(g)) < gtol``.
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

    def lagfun(z, y, mu, *args):
        c = constraint_wrapped.fun(z, *args)
        return fun_wrapped(z, *args) - jnp.dot(y, c) + mu / 2 * jnp.dot(c, c)

    def laggrad(z, y, mu, *args):
        c = constraint_wrapped.fun(z, *args)
        yJ = constraint_wrapped.vjp(y - mu * c, z, *args)
        return grad_wrapped(z, *args) - yJ

    if isinstance(hess_wrapped, str) and hess_wrapped.lower() == "bfgs":
        hess_init_scale = options.pop("hessian_init_scale", "auto")
        hess_exception_strategy = options.pop(
            "hessian_exception_strategy", "damp_update"
        )
        hess_min_curvature = options.pop("hessian_minimum_curvature", None)
        hess_wrapped = BFGS(
            hess_exception_strategy, hess_min_curvature, hess_init_scale
        )
    if isinstance(hess_wrapped, BFGS):
        if hasattr(hess_wrapped, "n"):  # assume its already been initialized
            assert hess_wrapped.approx_type == "hess"
            assert hess.n == z0.size
        else:
            hess_wrapped.initialize(z0.size, "hess")
        laghess = hess_wrapped

    elif callable(constraint_wrapped.hess) and callable(hess_wrapped):

        def laghess(z, y, mu, *args):
            c = constraint_wrapped.fun(z, *args)
            Hf = hess_wrapped(z, *args)
            Jc = constraint_wrapped.jac(z, *args)
            Hc1 = constraint_wrapped.hess(z, y)
            Hc2 = constraint_wrapped.hess(z, c)
            return Hf - Hc1 + mu * (Hc2 + jnp.dot(Jc.T, Jc))

    elif callable(hess_wrapped):

        def laghess(z, y, mu, *args):
            H = hess_wrapped(z, *args)
            J = constraint_wrapped.jac(z, *args)
            # ignoring higher order derivatives of constraints for now
            return H + mu * jnp.dot(J.T, J)

    else:
        raise ValueError(colored("hess should either be a callable or 'bfgs'", "red"))

    nfev = 0
    ngev = 0
    nhev = 0
    iteration = 0

    z = z0.copy()
    f = fun_wrapped(z, *args)
    c = constraint_wrapped.fun(z)
    nfev += 1

    mu = options.pop("initial_penalty_parameter", 10)
    y = options.pop("initial_multipliers", jnp.zeros_like(c))
    if y == "least_squares":  # use least squares multiplier estimates
        _J = constraint_wrapped.jac(z, *args)
        _g = grad_wrapped(z, *args)
        y = jnp.linalg.lstsq(_J.T, _g)[0]
        y = jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if maxiter is None:
        maxiter = z.size * 100
    maxiter_inner = options.pop("maxiter_inner", 20)
    max_nfev = options.pop("max_nfev", 5 * maxiter + 1)
    max_ngev = options.pop("max_ngev", maxiter + 1)
    max_nhev = options.pop("max_nhev", maxiter + 1)

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
    fold = f
    allx = []

    success = None
    message = None
    step_norm = jnp.inf
    actual_reduction = jnp.inf
    g_norm = jnp.linalg.norm(laggrad(z, y, mu, *args), ord=jnp.inf)
    constr_violation = jnp.linalg.norm(c, ord=jnp.inf)

    options.setdefault("initial_trust_radius", "scipy")
    options["return_tr"] = True

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

    while iteration < maxiter:
        result = fmintr(
            lagfun,
            z,
            grad=laggrad,
            hess=laghess,
            bounds=zbounds,
            args=(y, mu) + args,
            x_scale=x_scale,
            ftol=0,
            xtol=0,
            gtol=gtolk,
            maxiter=maxiter_inner,
            verbose=0,
            options=options.copy(),
        )
        allx += result["allx"]
        nfev += result["nfev"]
        ngev += result["ngev"]
        nhev += result["nhev"]
        iteration += result["nit"]
        zold = z
        fold = f
        z = result["x"]
        f = fun_wrapped(z, *args)
        c = constraint_wrapped.fun(z, *args)
        nfev += 1
        constr_violation = jnp.linalg.norm(c, ord=jnp.inf)
        step_norm = jnp.linalg.norm(zold - z)
        z_norm = jnp.linalg.norm(z)
        g_norm = result["optimality"]
        actual_reduction = fold - f
        # don't stop if we increased cost
        reduction_ratio = jnp.sign(actual_reduction)
        # reuse the previous trust radius on the next pass
        options["initial_trust_radius"] = float(result["alltr"][-1])

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

        success, message = check_termination(
            actual_reduction,
            f,
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
            ngev,
            max_ngev,
            nhev,
            max_nhev,
            constr_violation=constr_violation,
            ctol=ctol,
        )
        if success is not None:
            break

        if not result["success"]:  # did the subproblem actually finish, or maxiter?
            continue
        elif constr_violation < ctolk:
            y = y - mu * c
            ctolk = ctolk / (mu**beta_eta)
            gtolk = gtolk / (mu**beta_omega)
        else:
            mu = tau * mu
            ctolk = eta / (mu**alpha_eta)
            gtolk = omega / (mu**alpha_omega)

    x, s = z2xs(z)
    active_mask = find_active_constraints(z, zbounds[0], zbounds[1], rtol=xtol)
    result = OptimizeResult(
        x=x,
        s=s,
        y=y,
        success=success,
        fun=f,
        grad=result["grad"],
        v=result["v"],
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
