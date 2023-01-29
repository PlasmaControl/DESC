"""Class for wrapping a number of common optimization methods."""

import warnings

import numpy as np
from termcolor import colored

from desc.backend import jnp
from desc.io import IOAble
from desc.objectives import FixCurrent, FixIota, ObjectiveFunction
from desc.utils import Timer

from ._constraint_wrappers import LinearConstraintProjection, ProximalProjection
from ._scipy_wrappers import _optimize_scipy_least_squares, _optimize_scipy_minimize
from .fmin_scalar import fmintr
from .least_squares import lsqtr
from .stochastic import sgd


class Optimizer(IOAble):
    """A helper class to wrap several optimization routines.

    Offers all the ``scipy.optimize.least_squares`` routines  and several of the most
    useful ``scipy.optimize.minimize`` routines.
    Also offers several custom routines specifically designed for DESC, both scalar and
    least squares routines with and without Jacobian/Hessian information.

    Parameters
    ----------
    method : str
        name of the optimizer to use. Options are:

        * scipy scalar routines: ``'scipy-bfgs'``, ``'scipy-trust-exact'``,
          ``'scipy-trust-ncg'``, ``'scipy-trust-krylov'``
        * scipy least squares routines: ``'scipy-trf'``, ``'scipy-lm'``,
          ``'scipy-dogbox'``
        * desc scalar routines: ``'dogleg'``, ``'subspace'``, ``'dogleg-bfgs'``,
          ``'subspace-bfgs'``
        * desc least squares routines: ``'lsq-exact'``

    objective : ObjectiveFunction
        objective to be optimized

    """

    _io_attrs_ = ["_method"]

    # TODO: better way to organize these:
    _wrappers = [None, "prox", "proximal"]
    _scipy_least_squares_methods = ["scipy-trf", "scipy-lm", "scipy-dogbox"]
    _scipy_scalar_methods = [
        "scipy-bfgs",
        "scipy-trust-exact",
        "scipy-trust-ncg",
        "scipy-trust-krylov",
    ]
    _desc_scalar_methods = ["dogleg", "subspace", "dogleg-bfgs", "subspace-bfgs"]
    _desc_stochastic_methods = ["sgd"]
    _desc_least_squares_methods = ["lsq-exact"]
    _hessian_free_methods = ["scipy-bfgs", "dogleg-bfgs", "subspace-bfgs"]
    _scipy_constrained_scalar_methods = ["scipy-trust-constr"]
    _scipy_constrained_least_squares_methods = []
    _desc_constrained_scalar_methods = []
    _desc_constrained_least_squares_methods = []
    _scalar_methods = (
        _desc_scalar_methods
        + _scipy_scalar_methods
        + _scipy_constrained_scalar_methods
        + _desc_constrained_scalar_methods
        + _desc_stochastic_methods
    )
    _least_squares_methods = (
        _scipy_least_squares_methods
        + _desc_least_squares_methods
        + _scipy_constrained_least_squares_methods
        + _desc_constrained_least_squares_methods
    )
    _scipy_methods = (
        _scipy_least_squares_methods
        + _scipy_scalar_methods
        + _scipy_constrained_scalar_methods
        + _scipy_constrained_least_squares_methods
    )
    _desc_methods = (
        _desc_least_squares_methods
        + _desc_scalar_methods
        + _desc_constrained_scalar_methods
        + _desc_constrained_least_squares_methods
        + _desc_stochastic_methods
    )
    _constrained_methods = (
        _desc_constrained_scalar_methods
        + _desc_constrained_least_squares_methods
        + _scipy_constrained_scalar_methods
        + _scipy_constrained_least_squares_methods
    )
    _all_methods = (
        _scipy_least_squares_methods
        + _scipy_scalar_methods
        + _desc_scalar_methods
        + _desc_least_squares_methods
        + _desc_stochastic_methods
    )

    def __init__(self, method):

        self.method = method

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (method={})".format(self.method)
        )

    @property
    def method(self):
        """str: Name of the optimization method."""
        return self._method

    @method.setter
    def method(self, method):
        wrapper, submethod = _parse_method(method)
        if wrapper not in self._wrappers:
            raise ValueError("got unknown wrapper {}".format(wrapper))
        if submethod not in Optimizer._all_methods:
            raise NotImplementedError(
                colored(
                    "method must be one of {}".format(
                        ".".join([Optimizer._all_methods])
                    ),
                    "red",
                )
            )
        self._method = method

    # TODO: add copy argument and return the equilibrium?
    def optimize(  # noqa: C901 - FIXME: simplify this
        self,
        eq,
        objective,
        constraints=(),
        ftol=None,
        xtol=None,
        gtol=None,
        x_scale="auto",
        verbose=1,
        maxiter=None,
        options={},
    ):
        """Optimize an objective function.

        Parameters
        ----------
        eq : Equilibrium
            Initial equilibrium.
        objective : ObjectiveFunction
            Objective function to optimize.
        constraints : tuple of Objective, optional
            List of objectives to be used as constraints during optimization.
        ftol : float or None, optional
            Tolerance for termination by the change of the cost function.
            The optimization process is stopped when ``dF < ftol * F``,
            and there was an adequate agreement between a local quadratic model and the
            true model in the last step.
            If None, defaults to 1e-2 (or 1e-6 for stochastic).
        xtol : float or None, optional
            Tolerance for termination by the change of the independent variables.
            Optimization is stopped when ``norm(dx) < xtol * (xtol + norm(x))``.
            If None, defaults to 1e-6.
        gtol : float or None, optional
            Absolute tolerance for termination by the norm of the gradient.
            Optimizer terminates when ``norm(g) < gtol``, where
            If None, defaults to 1e-8.
        x_scale : array_like or ``'auto'``, optional
            Characteristic scale of each variable. Setting ``x_scale`` is equivalent
            to reformulating the problem in scaled variables ``xs = x / x_scale``.
            An alternative view is that the size of a trust region along jth
            dimension is proportional to ``x_scale[j]``. Improved convergence may
            be achieved by setting ``x_scale`` such that a step of a given size
            along any of the scaled variables has a similar effect on the cost
            function. If set to ``'auto'``, the scale is iteratively updated using the
            inverse norms of the columns of the Jacobian or Hessian matrix.
        verbose : integer, optional
            * 0  : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations
        maxiter : int, optional
            Maximum number of iterations. Defaults to 100.
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
        # TODO: document options
        timer = Timer()
        wrapper, method = _parse_method(self.method)

        linear_constraints, nonlinear_constraint = _parse_constraints(constraints)
        objective = _maybe_wrap_nonlinear_constraints(
            objective, nonlinear_constraint, self.method, options
        )
        if len(linear_constraints):
            objective = LinearConstraintProjection(objective, linear_constraints)
        if not objective.built:
            objective.build(eq, verbose=verbose)
        if not objective.compiled:
            mode = "scalar" if method in Optimizer._scalar_methods else "lsq"
            objective.compile(mode, verbose)

        if objective.scalar and (method in Optimizer._least_squares_methods):
            warnings.warn(
                colored(
                    "method {} is not intended for scalar objective function".format(
                        ".".join([method])
                    ),
                    "yellow",
                )
            )

        x0 = objective.x(eq)

        stoptol = _get_default_tols(
            method,
            ftol,
            xtol,
            gtol,
            maxiter,
            options,
        )

        if verbose > 0:
            print("Number of parameters: {}".format(x0.size))
            print("Number of objectives: {}".format(objective.dim_f))

        if verbose > 0:
            print("Starting optimization")
        timer.start("Solution time")

        if method in Optimizer._scipy_scalar_methods:

            x_scale = 1 if x_scale == "auto" else x_scale
            if isinstance(x_scale, str):
                raise ValueError(
                    f"Method {method} does not support x_scale type {x_scale}"
                )
            result = _optimize_scipy_minimize(
                objective.compute_scalar,
                objective.grad,
                objective.hess,
                x0,
                method.replace("scipy-", ""),
                x_scale,
                verbose,
                stoptol,
                options,
            )

        elif method in Optimizer._scipy_least_squares_methods:

            x_scale = "jac" if x_scale == "auto" else x_scale

            result = _optimize_scipy_least_squares(
                objective.compute,
                objective.jac,
                x0,
                method.replace("scipy-", ""),
                x_scale,
                verbose,
                stoptol,
                options,
            )

        elif method in Optimizer._desc_scalar_methods:

            hess = objective.hess if "bfgs" in method else "bfgs"
            if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
                options.setdefault("initial_trust_ratio", 1e-3)
                options.setdefault("max_trust_radius", 1.0)

            result = fmintr(
                objective.compute_scalar,
                x0=x0,
                grad=objective.grad,
                hess=hess,
                args=(),
                method=method.replace("-bfgs", ""),
                x_scale=x_scale,
                ftol=stoptol["ftol"],
                xtol=stoptol["xtol"],
                gtol=stoptol["gtol"],
                maxiter=stoptol["maxiter"],
                verbose=verbose,
                callback=None,
                options=options,
            )

        elif method in Optimizer._desc_stochastic_methods:

            result = sgd(
                objective.compute_scalar,
                x0=x0,
                grad=objective.grad,
                args=(),
                method=method,
                ftol=stoptol["ftol"],
                xtol=stoptol["xtol"],
                gtol=stoptol["gtol"],
                maxiter=stoptol["maxiter"],
                verbose=verbose,
                callback=None,
                options=options,
            )

        elif method in Optimizer._desc_least_squares_methods:

            if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
                options.setdefault("initial_trust_radius", 1e-3)
                options.setdefault("max_trust_radius", 1.0)

            result = lsqtr(
                objective.compute,
                x0=x0,
                jac=objective.jac,
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

        if isinstance(objective, LinearConstraintProjection):
            # remove wrapper to get at underlying objective
            result["allx"] = [objective.recover(x) for x in result["allx"]]
            objective = objective._objective

        if isinstance(objective, ProximalProjection):
            result["history"] = objective.history
        else:
            result["history"] = {}
            for arg in objective.args:
                result["history"][arg] = []
            for x in result["allx"]:
                kwargs = objective.unpack_state(x)
                for arg in kwargs:
                    result["history"][arg].append(kwargs[arg])

        timer.stop("Solution time")

        if verbose > 1:
            timer.disp("Solution time")
            timer.pretty_print(
                "Avg time per step",
                timer["Solution time"] / (result.get("nit", result.get("nfev")) + 1),
            )
        for key in ["hess", "hess_inv", "jac", "grad", "active_mask"]:
            _ = result.pop(key, None)

        return result


def _parse_method(method):
    """Split string into wrapper and method parts."""
    wrapper = None
    submethod = method
    for key in Optimizer._wrappers[1:]:
        if method.lower().startswith(key):
            wrapper = key
            submethod = method[len(key) + 1 :]
    return wrapper, submethod


def _parse_constraints(constraints):
    """Break constraints into linear and nonlinear, and combine nonlinear constraints.

    Parameters
    ----------
    constraints : tuple of Objective
        constraints to parse

    Returns
    -------
    linear_constraints : tuple of Objective
        Individual linear constraints
    nonlinear_constraints : ObjectiveFunction or None
        if any nonlinear constraints are present, they are combined into a single
        ObjectiveFunction, otherwise returns None
    """
    if not isinstance(constraints, (tuple, list)):
        constraints = (constraints,)
    linear_constraints = tuple(
        constraint for constraint in constraints if constraint.linear
    )
    nonlinear_constraints = tuple(
        constraint for constraint in constraints if not constraint.linear
    )
    # check for incompatible constraints
    if any(isinstance(lc, FixCurrent) for lc in linear_constraints) and any(
        isinstance(lc, FixIota) for lc in linear_constraints
    ):
        raise ValueError(
            "Toroidal current and rotational transform cannot be "
            + "constrained simultaneously."
        )
    # make sure any nonlinear constraints are combined into a single ObjectiveFunction
    if len(nonlinear_constraints) == 1 and isinstance(
        nonlinear_constraints[0], ObjectiveFunction
    ):
        nonlinear_constraints = nonlinear_constraints[0]
    elif len(nonlinear_constraints):
        nonlinear_constraints = ObjectiveFunction(nonlinear_constraints)
    else:
        nonlinear_constraints = None
    return linear_constraints, nonlinear_constraints


def _maybe_wrap_nonlinear_constraints(objective, nonlinear_constraint, method, options):
    """Use ProximalProjection to handle nonlinear constraints."""
    wrapper, method = _parse_method(method)
    if nonlinear_constraint is None:
        if wrapper is not None:
            warnings.warn(
                f"No nonlinear constraints detected, ignoring wrapper method {wrapper}"
            )
        return objective
    if wrapper is None:
        warnings.warn(
            FutureWarning(
                f"""
                Nonlinear constraints detected but method {method} does not support
                nonlinear constraints. Defaulting to method "proximal-{method}"
                In the future this will raise an error. To ignore this warnging, specify
                a wrapper "proximal-" to convert the nonlinearly constrained problem
                into an unconstrained one,
                """
            )
        )
        wrapper = "proximal"
    if wrapper.lower() in ["prox", "proximal"]:
        perturb_options = options.pop("perturb_options", {})
        solve_options = options.pop("solve_options", {})
        objective = ProximalProjection(
            objective,
            constraint=nonlinear_constraint,
            perturb_options=perturb_options,
            solve_options=solve_options,
        )
    return objective


def _get_default_tols(
    method,
    ftol=None,
    xtol=None,
    gtol=None,
    maxiter=None,
    options=None,
):
    """Parse and set defaults for stopping tolerances."""
    if options is None:
        options = {}
    stoptol = {}
    if xtol is not None:
        stoptol["xtol"] = xtol
    if ftol is not None:
        stoptol["ftol"] = ftol
    if gtol is not None:
        stoptol["gtol"] = gtol
    if maxiter is not None:
        stoptol["maxiter"] = maxiter
    stoptol.setdefault(
        "xtol",
        options.pop("xtol", 1e-6),
    )
    stoptol.setdefault(
        "ftol",
        options.pop(
            "ftol", 1e-6 if method in Optimizer._desc_stochastic_methods else 1e-2
        ),
    )
    stoptol.setdefault("gtol", options.pop("gtol", 1e-8))
    stoptol.setdefault("maxiter", options.pop("maxiter", 100))

    stoptol["max_nfev"] = options.pop("max_nfev", np.inf)
    stoptol["max_ngev"] = options.pop("max_ngev", np.inf)
    stoptol["max_njev"] = options.pop("max_njev", np.inf)
    stoptol["max_nhev"] = options.pop("max_nhev", np.inf)

    return stoptol
