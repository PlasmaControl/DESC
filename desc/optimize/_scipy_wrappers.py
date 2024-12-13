"""Functions for wrapping scipy.optimize methods and handle custom stopping criteria."""

import numpy as np
import scipy.optimize
from scipy.optimize import NonlinearConstraint, OptimizeResult

from desc.backend import jnp

from .optimizer import register_optimizer
from .utils import (
    check_termination,
    compute_hess_scale,
    compute_jac_scale,
    evaluate_quadratic_form_hess,
    evaluate_quadratic_form_jac,
    f_where_x,
    print_header_nonlinear,
    print_iteration_nonlinear,
)


@register_optimizer(
    name=[
        "scipy-bfgs",
        "scipy-CG",
        "scipy-Newton-CG",
        "scipy-dogleg",
        "scipy-trust-exact",
        "scipy-trust-ncg",
        "scipy-trust-krylov",
    ],
    description=[
        "BFGS quasi-newton method with line search. "
        + "See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html",
        "Nonlinear conjugate gradient method. "
        + "See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html",
        "Newton conjugate gradient method. "
        + "See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-newtoncg.html",  # noqa: E501
        "Trust region method with dogleg step. Hessian must be positive definite. "
        + "See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-dogleg.html",  # noqa: E501
        "Trust region method using 'exact' method to solve subproblem. "
        + "See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustexact.html",  # noqa: E501
        "Trust region method using conjugate gradient to solve subproblem. "
        + "See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustncg.html",  # noqa: E501
        "Trust region method using Krylov iterations to solve subproblem. "
        + "See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustkrylov.html",  # noqa: E501
    ],
    scalar=True,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=False,
    hessian=[False, False, True, True, True, True, True],
    GPU=False,
)
def _optimize_scipy_minimize(  # noqa: C901
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for scipy.optimize.minimize.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy - not supported by this method
    x0 : ndarray
        Starting point.
    method : str
        Name of the method to use. Any of the options in scipy.optimize.minimize.
    x_scale : array_like
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``scipy.optimize.minimize`` for details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    assert constraint is None, f"method {method} doesn't support constraints"
    options = {} if options is None else options
    options.setdefault("maxiter", stoptol["maxiter"])
    options.setdefault("disp", False)
    fun, grad, hess = objective.compute_scalar, objective.grad, objective.hess
    if isinstance(x_scale, str) and x_scale == "auto":
        H = hess(x0)
        scale, _ = compute_hess_scale(H)
    else:
        scale = x_scale
    if method in ["scipy-trust-exact", "scipy-trust-ncg"]:
        options.setdefault("initial_trust_radius", 1e-2 * np.linalg.norm(x0 / scale))
        options.setdefault("max_trust_radius", np.inf)
    # need to use some "global" variables here
    allx = [x0]
    func_allx = []
    func_allf = []
    grad_allx = []
    grad_allf = []
    hess_allx = []
    hess_allf = []
    message = [""]
    success = [None]

    def fun_wrapped(xs):
        # record all the x and fs we see
        x = xs * scale
        if len(func_allx):
            f = f_where_x(x, func_allx, func_allf, dim=0)
        else:
            f = np.array([])
        if not f.size:
            func_allx.append(x)
            f = fun(x, objective.constants)
            func_allf.append(f)
        return f

    def grad_wrapped(xs):
        # record all the x and grad we see
        x = xs * scale
        if len(grad_allx):
            g = f_where_x(x, grad_allx, grad_allf, dim=1)
        else:
            g = np.array([])
        if not g.size:
            grad_allx.append(x)
            g = grad(x, objective.constants)
            grad_allf.append(g)
        return g * scale

    def hess_wrapped(xs):
        # record all the x and hess we see
        x = xs * scale
        if len(hess_allx):
            H = f_where_x(x, hess_allx, hess_allf, dim=2)
        else:
            H = np.array([[]])
        if not H.size:
            hess_allx.append(x)
            H = hess(x, objective.constants)
            hess_allf.append(H)
        return H * (np.atleast_2d(scale).T * np.atleast_2d(scale))

    hess_wrapped = None if method in ["scipy-bfgs", "scipy-CG"] else hess_wrapped

    def callback(xs):
        x1 = xs * scale
        # if x not updated, means last step wasn't accepted, keep going
        eps = np.finfo(x1.dtype).eps
        if np.all(np.isclose(x1, allx[-1], rtol=eps, atol=eps)):
            return
        f1 = f_where_x(x1, func_allx, func_allf, dim=0)
        if not f1.size:
            f1 = fun_wrapped(xs)
        g1 = f_where_x(x1, grad_allx, grad_allf, dim=1)
        if not g1.size:
            g1 = grad_wrapped(xs) / scale
        allx.append(x1)
        g_norm = np.linalg.norm(g1, ord=np.inf)
        x_norm = np.linalg.norm(x1)

        if len(allx) < 2:
            df = np.inf
            dx_norm = np.inf
            reduction_ratio = 0
        else:
            x2 = allx[-2]
            f2 = f_where_x(x2, func_allx, func_allf, dim=0)
            df = f2 - f1
            dx = x1 - x2
            dx_norm = jnp.linalg.norm(dx)
            if len(hess_allx):
                H1 = f_where_x(x1, hess_allx, hess_allf, dim=2)
            else:
                H1 = np.eye(x1.size) / dx_norm
            if not H1.size:
                H1 = np.eye(x1.size) / dx_norm
            predicted_reduction = -evaluate_quadratic_form_hess(H1, g1, dx)
            if predicted_reduction > 0:
                reduction_ratio = df / predicted_reduction
            elif predicted_reduction == df == 0:
                reduction_ratio = 1
            else:
                reduction_ratio = 0

        if verbose > 1:
            print_iteration_nonlinear(
                len(allx) - 1, len(func_allx), f1, df, dx_norm, g_norm
            )

        success[0], message[0] = check_termination(
            df,
            f1,
            dx_norm,
            x_norm,
            g_norm,
            reduction_ratio,
            stoptol["ftol"],
            stoptol["xtol"],
            stoptol["gtol"],
            len(allx) - 1,
            stoptol["maxiter"],
            len(func_allx),
            stoptol["max_nfev"],
            dx_total=np.linalg.norm(x1 - x0),
            max_dx=options.get("max_dx", np.inf),
        )

        if success[0] is not None:
            raise StopIteration

    if verbose > 1:
        print_header_nonlinear()
        f1 = fun_wrapped(x0 / scale)
        g1 = grad_wrapped(x0 / scale) / scale
        g_norm = np.linalg.norm(g1, ord=np.inf)
        print_iteration_nonlinear(
            len(allx) - 1, len(func_allx), f1, np.inf, np.inf, g_norm
        )

    try:
        result = scipy.optimize.minimize(
            fun_wrapped,
            x0=x0 / scale,
            args=(),
            method=method.replace("scipy-", ""),
            jac=grad_wrapped,
            hess=hess_wrapped,
            tol=stoptol["gtol"],
            options=options,
            callback=callback,
        )
        result["allx"] = allx
        result["nfev"] = len(func_allx)
        result["ngev"] = len(grad_allx)
        result["nhev"] = len(hess_allx)
        result["nit"] = len(allx) - 1
    except StopIteration:
        x = allx[-1]
        f = f_where_x(x, func_allx, func_allf, dim=0)
        g = f_where_x(x, grad_allx, grad_allf, dim=1)
        if len(hess_allx):
            H = f_where_x(x, hess_allx, hess_allf, dim=2)
        else:
            H = None
        result = OptimizeResult(
            x=x,
            success=success[0],
            fun=f,
            grad=g,
            hess=H,
            optimality=np.linalg.norm(g, ord=np.inf),
            nfev=len(func_allx),
            ngev=len(grad_allx),
            nhev=len(hess_allx),
            nit=len(allx) - 1,
            message=message[0],
            allx=allx,
        )
    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print("         Current function value: {:.3e}".format(result["fun"]))
        print("         Total delta_x: {:.3e}".format(np.linalg.norm(x0 - result["x"])))
        print("         Iterations: {:d}".format(result["nit"]))
        print("         Function evaluations: {:d}".format(result["nfev"]))
        print("         Gradient evaluations: {:d}".format(result["ngev"]))
        print("         Hessian evaluations: {:d}".format(result["nhev"]))

    return result


@register_optimizer(
    name=["scipy-trf", "scipy-lm", "scipy-dogbox"],
    description=[
        "Trust region least squares method. "
        + "See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html",  # noqa: E501
        "Levenberg-Marquardt implicit trust region method. "
        + "See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html",  # noqa: E501
        "Dogleg method with box shaped trust region. "
        + "See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html",  # noqa: E501
    ],
    scalar=False,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=False,
    hessian=False,
    GPU=False,
)
def _optimize_scipy_least_squares(  # noqa: C901
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for scipy.optimize.least_squares.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy - not supported by this method
    x0 : ndarray
        Starting point.
    method : str
        Name of the method to use. Any of the options in scipy.optimize.least_squares.
    x_scale : array_like or ‘jac’, optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to ‘jac’, the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``scipy.optimize.least_squares`` for details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    assert constraint is None, f"method {method} doesn't support constraints"
    options = {} if options is None else options
    x_scale = "jac" if x_scale == "auto" else x_scale
    fun, jac = objective.compute_scaled_error, objective.jac_scaled_error
    # need to use some "global" variables here
    fun_allx = []
    fun_allf = []
    jac_allx = []
    jac_allf = []
    message = [""]
    success = [None]

    def fun_wrapped(x):
        # record all the xs and fs we see
        fun_allx.append(x)
        f = jnp.atleast_1d(fun(x, objective.constants))
        fun_allf.append(f)
        return f

    def jac_wrapped(x):
        # record all the xs and jacobians we see
        jac_allx.append(x)
        J = jac(x, objective.constants)
        jac_allf.append(J)
        callback(x)
        return J

    def callback(x1):
        f1 = f_where_x(x1, fun_allx, fun_allf, dim=1)
        c1 = 1 / 2 * np.dot(f1, f1)
        J1 = f_where_x(x1, jac_allx, jac_allf, dim=2)
        g1 = J1.T @ f1

        g_norm = np.linalg.norm(g1, ord=np.inf)
        x_norm = np.linalg.norm(x1)

        if len(jac_allx) < 2:
            df = np.inf
            dx_norm = np.inf
            reduction_ratio = 0
        else:
            x2 = jac_allx[-2]
            f2 = f_where_x(x2, fun_allx, fun_allf, dim=1)
            c2 = 1 / 2 * np.dot(f2, f2)
            df = c2 - c1
            dx = np.atleast_1d(x1 - x2)
            dx_norm = np.linalg.norm(dx)

            predicted_reduction = -evaluate_quadratic_form_jac(J1, g1, dx)
            if predicted_reduction > 0:
                reduction_ratio = df / predicted_reduction
            elif predicted_reduction == df == 0:
                reduction_ratio = 1
            else:
                reduction_ratio = 0

        if verbose > 1:
            print_iteration_nonlinear(
                len(jac_allx), len(fun_allx), c1, df, dx_norm, g_norm
            )
        success[0], message[0] = check_termination(
            df,
            c1,
            dx_norm,
            x_norm,
            g_norm,
            reduction_ratio,
            stoptol["ftol"],
            stoptol["xtol"],
            stoptol["gtol"],
            len(fun_allf),  # iteration,
            stoptol["maxiter"],
            len(fun_allf),
            stoptol["max_nfev"],
            dx_total=np.linalg.norm(x1 - x0),
            max_dx=options.get("max_dx", np.inf),
        )

        if success[0] is not None:
            raise StopIteration

    EPS = 2 * np.finfo(x0.dtype).eps
    if verbose > 1:
        print_header_nonlinear()
    try:
        result = scipy.optimize.least_squares(
            fun_wrapped,
            x0=x0,
            args=(),
            jac=jac_wrapped,
            method=method.replace("scipy-", ""),
            x_scale=x_scale,
            ftol=EPS,
            xtol=EPS,
            gtol=EPS,
            max_nfev=stoptol["max_nfev"],
            verbose=0,
            **options,
        )
        result["allx"] = jac_allx
        result["nfev"] = len(fun_allx)
        result["njev"] = len(jac_allx)
        result["nit"] = len(jac_allx)
    except StopIteration:
        x = jac_allx[-1]
        f = f_where_x(x, fun_allx, fun_allf, dim=1)
        c = 1 / 2 * np.dot(f, f)
        J = f_where_x(x, jac_allx, jac_allf, dim=2)
        g = np.atleast_1d(J.T @ f)
        result = OptimizeResult(
            x=x,
            success=success[0],
            cost=c,
            fun=f,
            grad=g,
            jac=J,
            optimality=np.linalg.norm(g, ord=np.inf),
            nfev=len(fun_allx),
            njev=len(jac_allx),
            nit=len(jac_allx),
            message=message[0],
            allx=jac_allx,
        )

    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print("         Current function value: {:.3e}".format(result["cost"]))
        print("         Total delta_x: {:.3e}".format(np.linalg.norm(x0 - result["x"])))
        print("         Iterations: {:d}".format(result["nit"]))
        print("         Function evaluations: {:d}".format(result["nfev"]))
        print("         Jacobian evaluations: {:d}".format(result["njev"]))

    return result


@register_optimizer(
    name=[
        "scipy-trust-constr",
        "scipy-SLSQP",
    ],
    description=[
        "Trust region interior point method. "
        + "See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html",  # noqa: E501
        "Sequential least squares programming method. "
        + "See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html",  # noqa: E501
    ],
    scalar=True,
    equality_constraints=True,
    inequality_constraints=True,
    stochastic=False,
    hessian=[True, False],
    GPU=False,
)
def _optimize_scipy_constrained(  # noqa: C901
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for scipy.optimize.minimize.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy
    x0 : ndarray
        Starting point.
    method : str
        Name of the method to use. Any of the options in scipy.optimize.minimize.
    x_scale : array_like
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``scipy.optimize.minimize`` for details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    options = {} if options is None else options
    options.setdefault("maxiter", stoptol["max_nfev"])
    options.setdefault("disp", False)
    fun, grad, hess = objective.compute_scalar, objective.grad, objective.hess

    if isinstance(x_scale, str) and x_scale == "auto":
        H = hess(x0)
        scale, _ = compute_hess_scale(H)
    else:
        scale = x_scale

    if constraint is not None:
        num_equality = np.count_nonzero(
            constraint.bounds_scaled[0] == constraint.bounds_scaled[1]
        )
        if num_equality > len(x0):
            raise ValueError(
                "scipy constrained optimizers cannot handle systems with more "
                + "equality constraints than free variables. Suggest reducing the grid "
                + "resolution of constraints"
            )
        if isinstance(x_scale, str) and x_scale == "auto":
            J = constraint.jac_scaled(x0)
            Jscale, _ = compute_jac_scale(J)
            scale = jnp.sqrt(scale * Jscale)

        def cfun_wrapped(xs):
            # record all the x and fs we see
            x = xs * scale
            if len(cfun_allx):
                f = f_where_x(x, cfun_allx, cfun_allf, dim=1)
            else:
                f = np.array([])
            if not f.size:
                cfun_allx.append(x)
                f = constraint.compute_scaled(x, constraint.constants)
                cfun_allf.append(f)
            return f

        def cjac_wrapped(xs):
            x = xs * scale
            if len(cjac_allx):
                J = f_where_x(x, cjac_allx, cjac_allf, dim=2)
            else:
                J = np.array([[]])
            if not J.size:
                cjac_allx.append(x)
                J = constraint.jac_scaled(x, constraint.constants)
                cjac_allf.append(J)
            return J * scale

        lb, ub = constraint.bounds_scaled
    else:

        def cfun_wrapped(xs):
            return 0.0

        def cjac_wrapped(xs):
            return jnp.zeros_like(xs)

        lb, ub = 0, 0

    def constraint_violation(xs):
        if constraint is not None:
            f = constraint.compute_scaled_error(xs * scale, constraint.constants)
        else:
            f = 0.0
        return jnp.max(jnp.abs(f))

    constraint_wrapped = NonlinearConstraint(
        cfun_wrapped,
        lb,
        ub,
        cjac_wrapped,
    )

    # need to use some "global" variables here
    allx = [x0]
    func_allx = []
    func_allf = []
    grad_allx = []
    grad_allf = []
    hess_allx = []
    hess_allf = []
    cfun_allx = []
    cfun_allf = []
    cjac_allx = []
    cjac_allf = []
    message = [""]
    success = [None]

    def fun_wrapped(xs):
        # record all the x and fs we see
        x = xs * scale
        if len(func_allx):
            f = f_where_x(x, func_allx, func_allf, dim=0)
        else:
            f = np.array([])
        if not f.size:
            func_allx.append(x)
            f = fun(x, objective.constants)
            func_allf.append(f)
        return f

    def grad_wrapped(xs):
        # record all the x and grad we see
        x = xs * scale
        if len(grad_allx):
            g = f_where_x(x, grad_allx, grad_allf, dim=1)
        else:
            g = np.array([])
        if not g.size:
            grad_allx.append(x)
            g = grad(x, objective.constants)
            grad_allf.append(g)
        return g * scale

    def hess_wrapped(xs):
        # record all the x and hess we see
        x = xs * scale
        if len(hess_allx):
            H = f_where_x(x, hess_allx, hess_allf, dim=2)
        else:
            H = np.array([[]])
        if not H.size:
            hess_allx.append(x)
            H = hess(x, objective.constants)
            hess_allf.append(H)
        return H * (np.atleast_2d(scale).T * np.atleast_2d(scale))

    hess_wrapped = None if method in ["scipy-SLSQP"] else hess_wrapped

    def callback(xs, *args):  # need to take args bc trust-constr passes extra stuff
        x1 = xs * scale
        # if x not updated, means last step wasn't accepted, keep going
        eps = np.finfo(x1.dtype).eps
        if np.all(np.isclose(x1, allx[-1], rtol=eps, atol=eps)):
            return
        f1 = f_where_x(x1, func_allx, func_allf, dim=0)
        if not f1.size:
            f1 = fun_wrapped(xs)
        g1 = f_where_x(x1, grad_allx, grad_allf, dim=1)
        if not g1.size:
            g1 = grad_wrapped(xs) / scale
        allx.append(x1)
        g_norm = np.linalg.norm(g1, ord=np.inf)
        x_norm = np.linalg.norm(x1)

        x2 = allx[-2]
        f2 = f_where_x(x2, func_allx, func_allf)
        if not f2.size:
            f2 = fun_wrapped(x2 / scale)
        df = f2 - f1
        dx = x1 - x2
        dx_norm = jnp.linalg.norm(dx)
        if len(hess_allx):
            H1 = f_where_x(x1, hess_allx, hess_allf)
        else:
            H1 = np.eye(x1.size) / dx_norm
        if not H1.size:
            H1 = np.eye(x1.size) / dx_norm
        predicted_reduction = -evaluate_quadratic_form_hess(H1, g1, dx)
        if predicted_reduction > 0:
            reduction_ratio = df / predicted_reduction
        elif predicted_reduction == df == 0:
            reduction_ratio = 1
        else:
            reduction_ratio = 0

        constr_violation = constraint_violation(x1 / scale)
        if verbose > 1:
            print_iteration_nonlinear(
                len(allx) - 1,
                len(func_allx),
                f1,
                df,
                dx_norm,
                g_norm,
                constr_violation,
            )
        success[0], message[0] = check_termination(
            df,
            f1,
            dx_norm,
            x_norm,
            g_norm,
            reduction_ratio,
            stoptol["ftol"],
            stoptol["xtol"],
            stoptol["gtol"],
            len(allx) - 1,
            stoptol["maxiter"],
            len(func_allx),
            stoptol["max_nfev"],
            dx_total=np.linalg.norm(x1 - x0),
            max_dx=options.get("max_dx", np.inf),
            ctol=stoptol["ctol"],
            constr_violation=constr_violation,
        )

        if success[0] is not None:
            raise StopIteration

    if verbose > 1:
        print_header_nonlinear(constrained=True)
        f1 = fun_wrapped(x0 / scale)
        g1 = grad_wrapped(x0 / scale) / scale
        g_norm = np.linalg.norm(g1, ord=np.inf)
        constr_violation = constraint_violation(x0 / scale)
        print_iteration_nonlinear(
            len(allx) - 1, len(func_allx), f1, np.inf, np.inf, g_norm, constr_violation
        )

    try:
        result = scipy.optimize.minimize(
            fun_wrapped,
            x0=x0 / scale,
            args=(),
            method=method.replace("scipy-", ""),
            jac=grad_wrapped,
            hess=hess_wrapped,
            constraints=(constraint_wrapped if constraint else None),
            tol=stoptol["gtol"],
            callback=callback,
            options=options,
        )
    except StopIteration:
        x = allx[-1]
        f = f_where_x(x, func_allx, func_allf)
        g = f_where_x(x, grad_allx, grad_allf)
        if len(hess_allx):
            H = f_where_x(x, hess_allx, hess_allf)
        else:
            H = None
        result = OptimizeResult(
            x=x,
            success=success[0],
            fun=f,
            grad=g,
            hess=H,
            optimality=np.linalg.norm(g, ord=np.inf),
            message=message[0],
        )

    result["allx"] = allx
    result["nfev"] = len(func_allx)
    result["ngev"] = len(grad_allx)
    result["nhev"] = len(hess_allx)
    result["ncfev"] = len(cfun_allx)
    result["ncjev"] = len(cjac_allx)
    result["nit"] = len(allx) - 1
    result["constr_violation"] = constraint_violation(result["x"] / scale)

    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print("         Current function value: {:.3e}".format(result["fun"]))
        print(
            "         Max constraint violation: {:.3e}".format(
                result["constr_violation"]
            )
        )
        print("         Total delta_x: {:.3e}".format(np.linalg.norm(x0 - result["x"])))
        print("         Iterations: {:d}".format(result["nit"]))
        print("         Function   evaluations: {:d}".format(result["nfev"]))
        print("         Gradient   evaluations: {:d}".format(result["ngev"]))
        print("         Hessian    evaluations: {:d}".format(result["nhev"]))
        print("         Constraint evaluations: {:d}".format(result["ncfev"]))
        print("         Jacobian   evaluations: {:d}".format(result["ncjev"]))

    return result
