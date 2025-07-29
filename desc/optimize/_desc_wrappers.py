from scipy.optimize import NonlinearConstraint

from desc.backend import jnp

from .aug_lagrangian import fmin_auglag
from .aug_lagrangian_ls import lsq_auglag
from .fmin_scalar import fmintr
from .least_squares import lsqtr
from .optimizer import register_optimizer
from .stochastic import sgd


@register_optimizer(
    name=["fmin-auglag", "fmin-auglag-bfgs"],
    description=[
        "Augmented Lagrangian trust region method for minimizing scalar valued "
        + "multivariate function. "
        + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.fmin_auglag.html",  # noqa: E501
        "Augmented Lagrangian trust region method for minimizing scalar valued "
        + "multivariate function. Uses BFGS to approximate Hessian. "
        + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.fmin_auglag.html",  # noqa: E501
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
    """Wrapper for desc.optimize.fmin_auglag.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy
    x0 : ndarray
        Starting point.
    method : {"fmin-auglag", "fmin-auglag-bfgs"}
        Name of the method to use.
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
        settings. See ``desc.optimize.fmin_auglag`` for details.

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
    # local lambdas to handle constants from both objective and constraint
    hess = (lambda x, *c: objective.hess(x, c[0])) if "bfgs" not in method else "bfgs"

    if constraint is not None:
        lb, ub = constraint.bounds_scaled
        constraint_wrapped = NonlinearConstraint(
            lambda x, *c: constraint.compute_scaled(x, c[1]),
            lb,
            ub,
            lambda x, *c: constraint.jac_scaled(x, c[1]),
        )
        # TODO (#1394): can't pass constants dict into vjp for now
        constraint_wrapped.vjp = lambda v, x, *args: constraint.vjp_scaled(v, x)
    else:
        constraint_wrapped = None

    result = fmin_auglag(
        lambda x, *c: objective.compute_scalar(x, c[0]),
        x0=x0,
        grad=lambda x, *c: objective.grad(x, c[0]),
        hess=hess,
        bounds=(-jnp.inf, jnp.inf),
        constraint=constraint_wrapped,
        args=(objective.constants, constraint.constants if constraint else None),
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
    description="Least squares augmented Lagrangian for constrained optimization"
    + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.lsq_auglag.html",  # noqa: E501
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
    """Wrapper for desc.optimize.lsq_auglag.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy
    x0 : ndarray
        Starting point.
    method : {"lsq-auglag"}
        Name of the method to use.
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
        settings. See ``desc.optimize.lsq_auglag`` for details.

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

    if constraint is not None:
        lb, ub = constraint.bounds_scaled
        constraint_wrapped = NonlinearConstraint(
            lambda x, *c: constraint.compute_scaled(x, c[1]),
            lb,
            ub,
            lambda x, *c: constraint.jac_scaled(x, c[1]),
        )
    else:
        constraint_wrapped = None

    result = lsq_auglag(
        lambda x, *c: objective.compute_scaled_error(x, c[0]),
        x0=x0,
        jac=lambda x, *c: objective.jac_scaled_error(x, c[0]),
        bounds=(-jnp.inf, jnp.inf),
        constraint=constraint_wrapped,
        args=(objective.constants, constraint.constants if constraint else None),
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
    description="Trust region least squares, similar to the `trf` method in scipy"
    + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.lsqtr.html",  # noqa: E501
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
    method : {"lsq-exact"}
        Name of the method to use.
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
        settings. See ``desc.optimize.lsqtr`` for details.

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

    result = lsqtr(
        objective.compute_scaled_error,
        x0=x0,
        jac=objective.jac_scaled_error,
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
        "fmintr",
        "fmintr-bfgs",
    ],
    description=[
        "Trust region method for minimizing scalar valued multivariate function. See "
        + "https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.fmintr.html",  # noqa: E501
        "Trust region method for minimizing scalar valued multivariate function. Uses "
        + "BFGS to approximate the Hessian. See "
        + "https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.fmintr.html",  # noqa: E501
    ],
    scalar=True,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=False,
    hessian=[True, False],
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
        Name of the method to use.
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

    result = fmintr(
        objective.compute_scalar,
        x0=x0,
        grad=objective.grad,
        hess=hess,
        args=(),
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
    description="Stochastic gradient descent with Nesterov momentum"
    + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.sgd.html",  # noqa: E501
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
        Name of the method to use.
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
        settings. See ``desc.optimize.sgd`` for details.

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
    result = sgd(
        objective.compute_scalar,
        x0=x0,
        grad=objective.grad,
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
