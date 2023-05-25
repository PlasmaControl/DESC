"""Augmented Langrangian for scalar valued objectives."""

import numpy as np
from scipy.optimize import OptimizeResult

from desc.backend import jnp
from desc.optimize.fmin_scalar import fmintr

from .bound_utils import find_active_constraints
from .utils import (
    check_termination,
    inequality_to_bounds,
    print_header_nonlinear,
    print_iteration_nonlinear,
)


def fmin_lag_stel(  # noqa: C901 - FIXME: simplify this
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
    hess : callable or ``'bfgs'``, optional:
        function to compute Hessian matrix of fun, or ``'bfgs'`` in which case the BFGS
        method will be used to approximate the Hessian.
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
        # just do regular unconstrained stuff
        return fmintr(
            fun,
            x0,
            grad,
            hess,
            bounds,
            args,
            x_scale=x_scale,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            verbose=verbose,
            maxiter=maxiter,
            options=options,
        )
    else:
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

    def lagfun(z, lmbda, mu, *args):
        c = constraint_wrapped.fun(z, *args)
        return fun_wrapped(z, *args) - jnp.dot(lmbda, c) + mu / 2 * jnp.dot(c, c)

    def laggrad(x, lmbda, mu, *args):
        c = constraint_wrapped.fun(x, *args)
        J = constraint_wrapped.jac(x, *args)
        return grad_wrapped(x, *args) - jnp.dot(lmbda, J) + mu * jnp.dot(c, J)

    if callable(hess_wrapped):
        if callable(constraint_wrapped.hess):

            def laghess(x, lmbda, mu, *args):
                c = constraint_wrapped.fun(x, *args)
                Hf = hess_wrapped(x, *args)
                Jc = constraint_wrapped.jac(x, *args)
                Hc1 = constraint_wrapped.hess(x, lmbda)
                Hc2 = constraint_wrapped.hess(x, c)
                # ignoring higher order derivatives of constraints for now
                return Hf - Hc1 + mu * (Hc2 + jnp.dot(Jc.T, Jc))

        else:

            def laghess(x, lmbda, mu, *args):
                H = hess_wrapped(x, *args)
                J = constraint_wrapped.jac(x, *args)
                # ignoring higher order derivatives of constraints for now
                return H + mu * jnp.dot(J.T, J)

    else:
        laghess = "bfgs"

    nfev = 0
    ngev = 0
    nhev = 0
    iteration = 0

    z = z0.copy()
    f = fun_wrapped(z, *args)
    g = grad_wrapped(z, *args)
    c = constraint_wrapped.fun(z)
    nfev += 1
    ngev += 1

    if maxiter is None:
        maxiter = z.size
    mu = options.pop("initial_penalty_parameter", 10)
    lmbda = options.pop("initial_multipliers", jnp.zeros(len(c)))
    maxiter_inner = options.pop("maxiter_inner", 20)
    max_nfev = options.pop("max_nfev", 5 * maxiter * maxiter_inner + 1)
    max_ngev = options.pop("max_ngev", maxiter * maxiter_inner + 1)
    max_nhev = options.pop("max_nhev", maxiter * maxiter_inner + 1)

    gtolk = 1 / (10 * np.linalg.norm(mu))
    ctolk = 1 / (np.linalg.norm(mu) ** (0.1))
    zold = z
    fold = f
    allx = []

    success = None
    message = None
    step_norm = jnp.inf
    actual_reduction = jnp.inf
    g_norm = np.linalg.norm(g, ord=np.inf)
    constr_violation = np.linalg.norm(c, ord=np.inf)

    if verbose > 1:
        print_header_nonlinear(constrained=True)
        print_iteration_nonlinear(
            iteration, nfev, f, actual_reduction, step_norm, g_norm, constr_violation
        )

    while iteration < maxiter:
        result = fmintr(
            lagfun,
            z,
            grad=laggrad,
            hess=laghess,
            bounds=zbounds,
            args=(lmbda, mu) + args,
            x_scale=x_scale,
            ftol=ftol,
            xtol=xtol,
            gtol=gtolk,
            maxiter=maxiter_inner,
            verbose=0,
        )
        allx += result["allx"]
        nfev += result["nfev"]
        ngev += result["ngev"]
        nhev += result["nhev"]
        z = result["x"]
        f = fun_wrapped(z, *args)
        c = constraint_wrapped.fun(z, *args)
        nfev += 1
        constr_violation = np.linalg.norm(c, ord=np.inf)
        step_norm = np.linalg.norm(zold - z)
        z_norm = np.linalg.norm(z)
        g_norm = result["optimality"]
        actual_reduction = fold - f
        iteration = iteration + 1

        if verbose > 1:
            print_iteration_nonlinear(
                iteration,
                nfev,
                f,
                actual_reduction,
                step_norm,
                g_norm,
                constr_violation,
            )

        success, message = check_termination(
            actual_reduction,
            f,
            step_norm,
            z_norm,
            g_norm,
            1,
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

        if constr_violation < ctolk:
            lmbda = lmbda - mu * c
            ctolk = ctolk / (mu ** (0.9))
            gtolk = gtolk / (mu)
        else:
            mu = 2 * mu
            ctolk = constr_violation / (mu ** (0.1))
            gtolk = gtolk / mu

        zold = z
        fold = f

    x, s = z2xs(z)
    active_mask = find_active_constraints(z, zbounds[0], zbounds[1], rtol=xtol)
    result = OptimizeResult(
        x=x,
        y=lmbda,
        success=success,
        fun=f,
        grad=g,
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
