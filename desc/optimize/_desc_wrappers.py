from scipy.optimize import NonlinearConstraint

from desc.backend import jnp

import numpy as np

from .aug_lagrangian import fmin_auglag
from .aug_lagrangian_ls import lsq_auglag
from .fmin_scalar import fmintr
from .least_squares import lsqtr
from .optimizer import register_optimizer
from .stochastic import sgd


@register_optimizer(
    name=["fmin-auglag", "fmin-auglag-bfgs"],
    description=[
        "Augmented Lagrangian method with trust region subproblem.",
        "Augmented Lagrangian method with trust region subproblem. Uses BFGS to"
        + " approximate hessian",
    ],
    scalar=True,
    equality_constraints=True,
    inequality_constraints=True,
    stochastic=False,
    hessian=[True, False],
    GPU=True,
)
def _optimize_desc_aug_lagrangian(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.fmin_lag_ls_stel.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy
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
        "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See the code for more details.

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
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_ratio", 1e-3)
        options.setdefault("max_trust_radius", 1.0)
    options["max_nfev"] = stoptol["max_nfev"]
    options["max_ngev"] = stoptol["max_ngev"]
    options["max_nhev"] = stoptol["max_nhev"]
    hess = objective.hess if "bfgs" not in method else "bfgs"

    if constraint is not None:
        lb, ub = constraint.bounds_scaled
        constraint_wrapped = NonlinearConstraint(
            constraint.compute_scaled,
            lb,
            ub,
            constraint.jac_scaled,
        )
        constraint_wrapped.vjp = constraint.vjp_scaled
    else:
        constraint_wrapped = None

    result = fmin_auglag(
        objective.compute_scalar,
        x0=x0,
        grad=objective.grad,
        hess=hess,
        bounds=(-jnp.inf, jnp.inf),
        constraint=constraint_wrapped,
        args=(),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        ctol=stoptol["ctol"],
        verbose=verbose,
        maxiter=stoptol["maxiter"],
        options=options,
    )
    return result


@register_optimizer(
    name="lsq-auglag",
    description="Least Squares Augmented Lagrangian approach "
    + "to constrained optimization",
    scalar=False,
    equality_constraints=True,
    inequality_constraints=True,
    stochastic=False,
    hessian=False,
    GPU=True,
)
def _optimize_desc_aug_lagrangian_least_squares(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.fmin_lag_ls_stel.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy
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
        "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See the code for more details.

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
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_radius", 1e-3)
        options.setdefault("max_trust_radius", 1.0)
    options["max_nfev"] = stoptol["max_nfev"]
    options["max_njev"] = stoptol["max_njev"]

    if constraint is not None:
        lb, ub = constraint.bounds_scaled
        constraint_wrapped = NonlinearConstraint(
            constraint.compute_scaled,
            lb,
            ub,
            constraint.jac_scaled,
        )
    else:
        constraint_wrapped = None

    result = lsq_auglag(
        objective.compute_scaled_error,
        x0=x0,
        jac=objective.jac_scaled,
        bounds=(-jnp.inf, jnp.inf),
        constraint=constraint_wrapped,
        args=(),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        ctol=stoptol["ctol"],
        verbose=verbose,
        maxiter=stoptol["maxiter"],
        options=options,
    )
    return result


@register_optimizer(
    name="lsq-exact",
    description="Trust region least squares method, "
    + "similar to the `trf` method in scipy",
    scalar=False,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=False,
    hessian=False,
    GPU=True,
)
def _optimize_desc_least_squares(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.lsqtr.

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
        "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See the code for more details.

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
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_radius", 1e-3)
        options.setdefault("max_trust_radius", 1.0)
    elif options.get("initial_trust_radius", "scipy") == "scipy":
        options.setdefault("initial_trust_ratio", 0.1)
    options["max_nfev"] = stoptol["max_nfev"]
    options["max_njev"] = stoptol["max_njev"]

    result = lsqtr(
        objective.compute_scaled_error,
        x0=x0,
        jac=objective.jac_scaled,
        args=(objective.constants,),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=None,
        options=options,
    )
    return result


@register_optimizer(
    name=[
        "fmin-exact",
        "fmin-dogleg",
        "fmin-subspace",
        "fmin-exact-bfgs",
        "fmin-dogleg-bfgs",
        "fmin-subspace-bfgs",
    ],
    description=[
        "Trust region method using iterative cholesky method to exactly solve the "
        + "trust region subproblem.",
        "Trust region method using Powell's dogleg method to approximately solve the "
        + "trust region subproblem.",
        "Trust region method solving the subproblem over the 2d subspace spanned by "
        + "the gradient and newton direction.",
        "Trust region method using iterative cholesky method to exactly solve the "
        + "trust region subproblem. Uses BFGS to approximate hessian",
        "Trust region method using Powell's dogleg method to approximately solve the "
        + "trust region subproblem. Uses BFGS to approximate hessian",
        "Trust region method solving the subproblem over the 2d subspace spanned by "
        + "the gradient and newton direction. Uses BFGS to approximate hessian",
    ],
    scalar=True,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=False,
    hessian=[True, True, True, False, False, False],
    GPU=True,
)
def _optimize_desc_fmin_scalar(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.fmintr.

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
        "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See the code for more details.

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
    hess = objective.hess if "bfgs" not in method else "bfgs"
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_ratio", 1e-3)
        options.setdefault("max_trust_radius", 1.0)
    elif options.get("initial_trust_radius", "scipy") == "scipy":
        options.setdefault("initial_trust_ratio", 0.1)
    options["max_nfev"] = stoptol["max_nfev"]
    options["max_ngev"] = stoptol["max_ngev"]
    options["max_nhev"] = stoptol["max_nhev"]

    result = fmintr(
        objective.compute_scalar,
        x0=x0,
        grad=objective.grad,
        hess=hess,
        args=(),
        method=method.replace("-bfgs", "").replace("fmin-", ""),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=None,
        options=options,
    )
    return result


@register_optimizer(
    name="sgd",
    description="Stochastic gradient descent with Nesterov momentum",
    scalar=True,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=True,
    hessian=False,
    GPU=True,
)
def _optimize_desc_stochastic(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.sgd.

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
        "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See the code for more details.

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

    def grad_fd(x_reduced,x0,f):
        x = x_reduced
        fx = f
        print("FX IS " + str(fx))
        dx = 1.0e-1*np.abs(x)               
        x = jnp.asarray(x).at[x == 0].set(0.001)

        tang = np.eye(len(dx))
        jac = np.zeros((len(fx),len(tang)))
        for i in range(len(tang)):
            tang[i][i] = dx[i]
        for i in range(len(tang)):
            df = (objective.compute(x_reduced+tang[:,i].T) - objective.compute(x_reduced-tang[:,i].T))/(np.linalg.norm(objective.recover(tang[:,i].T)))
            jac[:,i] = df
        return fx.T @ jac

    def grad_spsa(x_reduced,*args):
        fx, fd_step, num_grad = args

        x = x_reduced
        h = fd_step*jnp.abs(x)
        h = jnp.asarray(h).at[h == 0].set(min(h[jnp.nonzero(h)]))

        jac = jnp.zeros((len(fx),len(h)))
        
        for j in range(num_grad):
            djac = np.zeros((len(fx),len(h)))
            dx = (np.random.binomial(1,0.5,x.shape)*2-1)*h
            print("dx is " + str(dx))
            
            print("x + dx is " + str(x + dx))
            ob = objective.compute_scaled_error(x + dx)
            print("ob is " + str(ob))
            obm = objective.compute_scaled_error(x - dx)
            df = ob - obm
            for i in range(len(x)):
                zx = np.zeros(len(x))
                zx[i] = dx[i]
                djac[:,i] = jnp.abs(dx[i])/dx[i]*df/(2*jnp.linalg.norm(objective.recover(zx)))
            jac = jac + djac
        jac = jac/num_grad

        return fx.T @ jac

    result = sgd(
        objective,
        x0=x0,
        grad=grad_spsa,
        args=(objective.constants,),
        method=method,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=None,
        options=options,
    )
    return result
