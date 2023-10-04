"""Class for wrapping a number of common optimization methods."""

import warnings

import numpy as np
from termcolor import colored

from desc.io import IOAble
from desc.objectives import (
    FixCurrent,
    FixIota,
    ObjectiveFunction,
    maybe_add_self_consistency,
)
from desc.objectives.utils import combine_args
from desc.utils import Timer

from ._constraint_wrappers import LinearConstraintProjection, ProximalProjection


class Optimizer(IOAble):
    """A helper class to wrap several optimization routines.

    Offers all the ``scipy.optimize.least_squares`` routines  and several of the most
    useful ``scipy.optimize.minimize`` routines.
    Also offers several custom routines specifically designed for DESC, both scalar and
    least squares routines with and without Jacobian/Hessian information.

    Parameters
    ----------
    method : str
        name of the optimizer to use. Options can be found as desc.optimize.optimizers

    objective : ObjectiveFunction
        objective to be optimized

    """

    _io_attrs_ = ["_method"]
    _wrappers = [None, "prox", "proximal"]

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
        if submethod not in optimizers:
            raise NotImplementedError(
                colored(
                    "method must be one of {}".format(".".join([*optimizers.keys()])),
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
        ctol=None,
        x_scale="auto",
        verbose=1,
        maxiter=None,
        options=None,
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
        ctol : float or None, optional
            Stopping tolerance on infinity norm of the constraint violation.
            Optimization will stop when ctol and one of the other tolerances
            are satisfied. If None, defaults to 1e-4.
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
        options = {} if options is None else options
        # TODO: document options
        timer = Timer()
        options = {} if options is None else options
        wrapper, method = _parse_method(self.method)

        linear_constraints, nonlinear_constraint = _parse_constraints(constraints)
        objective, nonlinear_constraint = _maybe_wrap_nonlinear_constraints(
            eq, objective, nonlinear_constraint, self.method, options
        )

        if not isinstance(objective, ProximalProjection):
            # need to include self consistency constraints
            linear_constraints = maybe_add_self_consistency(eq, linear_constraints)
        if len(linear_constraints):
            objective = LinearConstraintProjection(objective, linear_constraints, eq=eq)
            if nonlinear_constraint is not None:
                nonlinear_constraint = LinearConstraintProjection(
                    nonlinear_constraint, linear_constraints, eq=eq
                )
        if not objective.built:
            objective.build(eq, verbose=verbose)
        if nonlinear_constraint is not None and not nonlinear_constraint.built:
            nonlinear_constraint.build(eq, verbose=verbose)
        if nonlinear_constraint is not None:
            objective, nonlinear_constraint = combine_args(
                objective, nonlinear_constraint
            )
        if len(linear_constraints) and not isinstance(x_scale, str):
            # need to project x_scale down to correct size
            Z = objective._Z
            x_scale = np.broadcast_to(x_scale, objective._objective.dim_x)
            x_scale = np.abs(
                np.diag(Z.T @ np.diag(x_scale[objective._unfixed_idx]) @ Z)
            )
            x_scale = np.where(x_scale < np.finfo(x_scale.dtype).eps, 1, x_scale)

        if not objective.compiled:
            if optimizers[method]["scalar"] and optimizers[method]["hessian"]:
                mode = "scalar"
            elif optimizers[method]["scalar"]:
                mode = "bfgs"
            else:
                mode = "lsq"
            try:
                objective.compile(mode, verbose)
            except ValueError:
                objective.build(eq, verbose=verbose)
                objective.compile(mode, verbose=verbose)
        if nonlinear_constraint is not None and not nonlinear_constraint.compiled:
            try:
                nonlinear_constraint.compile("lsq", verbose)
            except ValueError:
                nonlinear_constraint.build(eq, verbose=verbose)
                nonlinear_constraint.compile("lsq", verbose)

        if objective.scalar and (not optimizers[method]["scalar"]):
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
            ctol,
            maxiter,
            options,
        )

        if verbose > 0:
            print("Number of parameters: {}".format(x0.size))
            print("Number of objectives: {}".format(objective.dim_f))
            if nonlinear_constraint is not None:
                num_equality = np.count_nonzero(
                    nonlinear_constraint.bounds_scaled[0]
                    == nonlinear_constraint.bounds_scaled[1]
                )
                print("Number of equality constraints: {}".format(num_equality))
                print(
                    "Number of inequality constraints: {}".format(
                        nonlinear_constraint.dim_f - num_equality
                    )
                )

        if verbose > 0:
            print("Starting optimization")
            print("Using method: " + str(self.method))

        timer.start("Solution time")

        result = optimizers[method]["fun"](
            objective,
            nonlinear_constraint,
            x0,
            method,
            x_scale,
            verbose,
            stoptol,
            options,
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
    if len(nonlinear_constraints):
        nonlinear_constraints = ObjectiveFunction(nonlinear_constraints)
    else:
        nonlinear_constraints = None
    return linear_constraints, nonlinear_constraints


def _maybe_wrap_nonlinear_constraints(
    eq, objective, nonlinear_constraint, method, options
):
    """Use ProximalProjection to handle nonlinear constraints."""
    wrapper, method = _parse_method(method)
    if nonlinear_constraint is None:
        if wrapper is not None:
            warnings.warn(
                f"No nonlinear constraints detected, ignoring wrapper method {wrapper}"
            )
        return objective, nonlinear_constraint
    if wrapper is None and not optimizers[method]["equality_constraints"]:
        warnings.warn(
            FutureWarning(
                f"""
                Nonlinear constraints detected but method {method} does not support
                nonlinear constraints. Defaulting to method "proximal-{method}"
                In the future this will raise an error. To ignore this warning, specify
                a wrapper "proximal-" to convert the nonlinearly constrained problem
                into an unconstrained one.
                """
            )
        )
        wrapper = "proximal"
    if wrapper is not None and wrapper.lower() in ["prox", "proximal"]:
        perturb_options = options.pop("perturb_options", {})
        solve_options = options.pop("solve_options", {})
        objective = ProximalProjection(
            objective,
            constraint=nonlinear_constraint,
            perturb_options=perturb_options,
            solve_options=solve_options,
            eq=eq,
        )
        nonlinear_constraint = None
    return objective, nonlinear_constraint


def _get_default_tols(
    method,
    ftol=None,
    xtol=None,
    gtol=None,
    ctol=None,
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
    if ctol is not None:
        stoptol["ctol"] = ctol
    if maxiter is not None:
        stoptol["maxiter"] = maxiter
    stoptol.setdefault(
        "xtol",
        options.pop("xtol", 1e-6),
    )
    stoptol.setdefault(
        "ftol",
        options.pop(
            "ftol",
            1e-6 if optimizers[method]["stochastic"] or "auglag" in method else 1e-2,
        ),
    )
    stoptol.setdefault("gtol", options.pop("gtol", 1e-8))
    stoptol.setdefault("ctol", options.pop("ctol", 1e-4))
    stoptol.setdefault(
        "maxiter", options.pop("maxiter", 500 if "auglag" in method else 100)
    )

    # if we define an "iteration" as a successful step, it can take a few function
    # evaluations per iteration
    stoptol["max_nfev"] = options.pop("max_nfev", 5 * stoptol["maxiter"] + 1)

    return stoptol


optimizers = {}


def register_optimizer(
    name,
    description,
    scalar,
    equality_constraints,
    inequality_constraints,
    stochastic,
    hessian,
    GPU=False,
    **kwargs,
):
    """Decorator to wrap a function for optimization.

    Function being wrapped should have a signature of the form
    fun(objective, constraint, x0, method, x_scale, verbose, stoptol, options=None)
    and should return a scipy.optimize.OptimizeResult object

    Function should take the following arguments:

    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy
    x0 : ndarray
        Starting point.
    method : str
        Name of the method to use.
    x_scale : array_like or ‘jac’, optional
        Characteristic scale of each variable.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings.

    Parameters
    ----------
    name : str or array-like of str
        Name of the optimizer method. If one function supports multiple methods,
        provide a list of names.
    description : str or array-like of str
        Short description of the optimizer method, with references if possible.
    scalar : bool or array-like of bool
        Whether the method assumes a scalar residual, or a vector of residuals for
        least squares.
    equality_constraints : bool or array-like of bool
        Whether the method handles equality constraints.
    inequality_constraints : bool or array-like of bool
        Whether the method handles inequality constraints.
    stochastic : bool or array-like of bool
        Whether the method can handle noisy objectives.
    hessian : bool or array-like of bool
        Whether the method requires calculation of the full hessian matrix.
    GPU : bool or array-like of bool
        Whether the method supports running on GPU
    """
    (
        name,
        description,
        scalar,
        equality_constraints,
        inequality_constraints,
        stochastic,
        hessian,
        GPU,
    ) = map(
        np.atleast_1d,
        (
            name,
            description,
            scalar,
            equality_constraints,
            inequality_constraints,
            stochastic,
            hessian,
            GPU,
        ),
    )
    (
        name,
        description,
        scalar,
        equality_constraints,
        inequality_constraints,
        stochastic,
        hessian,
        GPU,
    ) = np.broadcast_arrays(
        name,
        description,
        scalar,
        equality_constraints,
        inequality_constraints,
        stochastic,
        hessian,
        GPU,
    )

    def _decorator(func):

        for i, nm in enumerate(name):
            d = {
                "description": description[i % len(name)],
                "scalar": scalar[i % len(name)],
                "equality_constraints": equality_constraints[i % len(name)],
                "inequality_constraints": inequality_constraints[i % len(name)],
                "stochastic": stochastic[i % len(name)],
                "hessian": hessian[i % len(name)],
                "GPU": GPU[i % len(name)],
                "fun": func,
            }
            optimizers[nm] = d
        return func

    return _decorator
