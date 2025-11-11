"""Class for wrapping a number of common optimization methods."""

import contextlib
import io
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
from desc.utils import (
    PRINT_WIDTH,
    Timer,
    errorif,
    flatten_list,
    get_instance,
    is_any_instance,
    unique_list,
    warnif,
)

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
        _, submethod = _parse_method(method)
        if submethod not in optimizers:
            raise NotImplementedError(
                colored(
                    "method must be one of {}".format(".".join([*optimizers.keys()])),
                    "red",
                )
            )
        self._method = method

    def optimize(  # noqa: C901
        self,
        things,
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
        copy=False,
    ):
        """Optimize an objective function.

        Parameters
        ----------
        things : Optimizable or tuple/list of Optimizable
            Things to optimize, eg Equilibrium.
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
            settings. Options that are universal to all optimizers are:

            - ``"linear_constraint_options"`` : Dictionary of keyword arguments to pass
              to ``LinearConstraintProjection``.
            - ``"perturb_options"`` : Dictionary of keyword arguments to pass to
              ``ProximalProjection`` as its ``perturb_options``.
            - ``"solve_options"`` : Dictionary of keyword arguments to pass to
              ``ProximalProjection`` as its ``solve_options``.

            See the documentation page [Optimizers
            Supported](https://desc-docs.readthedocs.io/en/stable/optimizers.html)
            for more details on the options available to each specific optimizer.

        copy : bool
            Whether to return the current things or a copy (leaving the original
            unchanged).

        Returns
        -------
        things : list,
            list of optimized things
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.
            Additionally, stores the before and after values of the objectives
            and constraints in the ``Objective values`` key.

        """
        is_linear_proj = isinstance(objective, LinearConstraintProjection)
        if not isinstance(constraints, (tuple, list)) and not is_linear_proj:
            constraints = (constraints,)
        elif is_linear_proj:
            constraints = objective._constraint.objectives
        errorif(
            not isinstance(objective, ObjectiveFunction),
            TypeError,
            "objective should be of type ObjectiveFunction.",
        )

        # get unique things
        things, indices = unique_list(flatten_list(things, flatten_tuple=True))
        counts = np.unique(indices, return_counts=True)[1]
        duplicate_idx = np.where(counts > 1)[0]
        warnif(
            len(duplicate_idx),
            UserWarning,
            f"{[things[idx] for idx in duplicate_idx]} is duplicated in things.",
        )
        things0 = [t.copy() for t in things]

        # need local import to avoid circular dependencies
        from desc.equilibrium import Equilibrium
        from desc.objectives import QuadraticFlux

        # eq may be None
        eq = get_instance(things, Equilibrium)
        if eq is not None:
            if not is_linear_proj:
                # check if stage 2 objectives are here:
                all_objs = list(constraints) + list(objective.objectives)
                errorif(
                    is_any_instance(all_objs, QuadraticFlux),
                    ValueError,
                    "QuadraticFlux objective assumes Equilibrium is fixed but "
                    + "Equilibrium is in things to optimize.",
                )
            # save these for later
            eq_params_init = eq.params_dict.copy()

        options = {} if options is None else options
        timer = Timer()
        options = {} if options is None else options
        _, method = _parse_method(self.method)

        timer.start("Initializing the optimization")
        if not is_linear_proj:
            objective, nonlinear_constraint, x_scale = (
                get_combined_constraint_objectives(
                    eq,
                    constraints,
                    objective,
                    things,
                    x_scale,
                    self.method,
                    method,
                    verbose,
                    options,
                )
            )
        else:
            nonlinear_constraint = None
            if not objective.built:
                objective.build(verbose=verbose)
        # we have to use this cumbersome indexing in this method when passing things
        # to objective to guard against the passed-in things having an ordering
        # different from objective.things, to ensure the correct order is passed
        # to the objective
        x0 = objective.x(*[things[things.index(t)] for t in objective.things])

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
        timer.stop("Initializing the optimization")
        if verbose > 1:
            timer.disp("Initializing the optimization")

        if verbose > 0:
            print("\nStarting optimization")
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
            # reset eq params to initial
            if eq is not None:
                eq.params_dict = eq_params_init
            result["history"] = objective.history
            objective = objective._objective
        else:
            result["history"] = [
                objective.unpack_state(xi, False) for xi in result["allx"]
            ]

        timer.stop("Solution time")

        if verbose > 1:
            timer.disp("Solution time")
            timer.pretty_print(
                "Avg time per step",
                timer["Solution time"] / (result.get("nit", result.get("nfev")) + 1),
            )
        for key in ["hess", "hess_inv", "jac", "grad", "active_mask"]:
            _ = result.pop(key, None)

        # temporarily assign new stuff for printing, might get replaced later
        for thing, params in zip(objective.things, result["history"][-1]):
            # more indexing here to ensure the correct params are assigned to the
            # correct thing, as the order of things and objective.things might differ
            ind = things.index(thing)
            things[ind].params_dict = params

        # we always want to populate result dict with before/after values, but may
        # not always want to print, so we first capture the output and then decide
        # what to do about it.
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            result = _print_output(things, things0, objective, constraints, result)
            text_out = buf.getvalue()
        if verbose > 0:
            print(text_out)

        if copy:
            # need to swap things and things0, since things should be unchanged
            for t, t0 in zip(things, things0):
                init_params = t0.params_dict.copy()
                final_params = t.params_dict.copy()
                t.params_dict = init_params
                t0.params_dict = final_params
            return things0, result

        return things, result


def _print_output(things, things0, objective, constraints, result):
    """Print summary stats after optimization."""
    state_0 = [things0[things.index(t)] for t in objective.things]
    state = [things[things.index(t)] for t in objective.things]

    # put a divider
    w_divider = 50
    print("{:=<{}}".format("", PRINT_WIDTH + w_divider))

    print(f"{'Start  -->   End':>{PRINT_WIDTH+21}}")
    values = objective.print_value(objective.x(*state), objective.x(*state_0))
    for con in constraints:
        arg_inds_for_this_con = [things.index(t) for t in things if t in con.things]
        args_for_this_con = [things[ind] for ind in arg_inds_for_this_con]
        args0_for_this_con = [things0[ind] for ind in arg_inds_for_this_con]
        _val = con.print_value(con.xs(*args_for_this_con), con.xs(*args0_for_this_con))
        if con._print_value_fmt in values:
            values[con._print_value_fmt].append(_val)
        else:
            values[con._print_value_fmt] = [_val]
    print("{:=<{}}".format("", PRINT_WIDTH + w_divider))
    result["Objective values"] = values
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


def _combine_constraints(constraints):
    """Combine constraints into a single ObjectiveFunction.

    Parameters
    ----------
    constraints : tuple of Objective
        Constraints to combine.

    Returns
    -------
    objective : ObjectiveFunction or None
        If constraints are present, they are combined into a single ObjectiveFunction.
        Otherwise returns None.

    """
    if len(constraints):
        objective = ObjectiveFunction(constraints)
    else:
        objective = None
    return objective


def _parse_constraints(constraints):
    """Break constraints into linear and nonlinear.

    Parameters
    ----------
    constraints : tuple of Objective
        Constraints to parse.

    Returns
    -------
    linear_constraints : tuple of Objective
        Individual linear constraints.
    nonlinear_constraints : tuple of Objective
        Individual nonlinear constraints.

    """
    if not isinstance(constraints, (tuple, list)):
        constraints = (constraints,)
    # we treat linear bound constraints as nonlinear since they can't be easily
    # factorized like linear equality constraints
    linear_constraints = tuple(
        constraint
        for constraint in constraints
        if (constraint.linear and (constraint.bounds is None))
    )
    nonlinear_constraints = tuple(
        constraint for constraint in constraints if constraint not in linear_constraints
    )
    # check for incompatible constraints
    if any(isinstance(lc, FixCurrent) for lc in linear_constraints) and any(
        isinstance(lc, FixIota) for lc in linear_constraints
    ):
        raise ValueError(
            "Toroidal current and rotational transform cannot be "
            + "constrained simultaneously."
        )
    return linear_constraints, nonlinear_constraints


def _maybe_wrap_nonlinear_constraints(
    eq, objective, nonlinear_constraints, method, options
):
    """Use ProximalProjection to handle nonlinear constraints."""
    if eq is None:  # not deal with an equilibrium problem -> no ProximalProjection
        return objective, nonlinear_constraints
    wrapper, method = _parse_method(method)
    if not len(nonlinear_constraints):
        if wrapper is not None:
            warnings.warn(
                f"No nonlinear constraints detected, ignoring wrapper method {wrapper}."
            )
        return objective, nonlinear_constraints
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
            constraint=_combine_constraints(nonlinear_constraints),
            perturb_options=perturb_options,
            solve_options=solve_options,
            eq=eq,
        )
        nonlinear_constraints = ()
    return objective, nonlinear_constraints


def get_combined_constraint_objectives(
    eq,
    constraints,
    objective,
    things,
    x_scale,
    opt_method,
    method,
    verbose,
    options,
):
    """Combine constraints and objective.

    Checks, combines, wraps and builds constraints and objective functions into a single
    objective and nonlinear constraints that can be passed to the optimizer.
    """
    from desc.equilibrium import Equilibrium

    # parse and combine constraints into linear & nonlinear objective functions
    linear_constraints, nonlinear_constraints = _parse_constraints(constraints)
    objective, nonlinear_constraints = _maybe_wrap_nonlinear_constraints(
        eq, objective, nonlinear_constraints, opt_method, options
    )
    is_prox = isinstance(objective, ProximalProjection)
    for t in things:
        if isinstance(t, Equilibrium) and is_prox:
            # don't add Equilibrium self-consistency if proximal is used
            # see ProximalProjection._set_eq_state_vector
            continue
        linear_constraints = maybe_add_self_consistency(t, linear_constraints)
    linear_constraint = _combine_constraints(linear_constraints)
    nonlinear_constraint = _combine_constraints(nonlinear_constraints)

    # make sure everything is built
    if objective is not None and not objective.built:
        objective.build(verbose=verbose)
    if linear_constraint is not None and not linear_constraint.built:
        linear_constraint.build(verbose=verbose)
    if nonlinear_constraint is not None and not nonlinear_constraint.built:
        nonlinear_constraint.build(verbose=verbose)

    # combine arguments from all three objective functions
    inconsistent_things_error_msg = (
        "Detected an inconsistency in the "
        "things passed to the optimizer and/or the things targeted by the objective "
        "and constraints.\n"
        "\nOne reason this can happen is if you forgot to pass in every thing \n"
        "expected by the objectives and constraints (e.g. did not pass in coils for \n"
        "an optimization where the coils are allowed to change), or if you passed in "
        "things that were not"
        " targeted by the objectives or constraints.\n"
        "\nAnother reason could be that some of the things passed to the optimizer "
        "are different objects than what the objective was created with. \n"
        "This may happen if, for example, you are trying to re-use an objective but "
        "not with the same exact thing it was created with. \nE.g. if you make an"
        " objective for "
        "an Equilibrium object, then try to optimize a different Equilibrium object\n"
        "with that same objective, or you try to optimize a copy of that Equilibrium\n"
        "object (which has a different Python object ID and therefore is a different"
        " object), this error will occur.\n"
        "Check the below inconsistency for things like two of the same object type "
        "where\nyou only expected one, or for an object that you were not intending "
        "to allow to be optimizable, etc.\n\n"
    )
    if linear_constraint is not None and nonlinear_constraint is not None:
        objective, linear_constraint, nonlinear_constraint = combine_args(
            objective, linear_constraint, nonlinear_constraint
        )
        assert set(objective.things) == set(linear_constraint.things), (
            inconsistent_things_error_msg
            + f"objective+constraints expected things {set(objective.things)}, \n"
            f"linear constraints expected things {set(linear_constraint.things)} "
        )
        assert set(objective.things) == set(nonlinear_constraint.things), (
            inconsistent_things_error_msg
            + f"objective+constraints expected things {set(objective.things)}, \n"
            f"nonlinear constraints expected things {set(nonlinear_constraint.things)}"
        )
    elif linear_constraint is not None:
        objective, linear_constraint = combine_args(objective, linear_constraint)
        assert set(objective.things) == set(linear_constraint.things), (
            inconsistent_things_error_msg
            + f"objective+constraints expected things {set(objective.things)}, \n"
            f"linear constraints expected things {set(linear_constraint.things)}"
        )
    elif nonlinear_constraint is not None:
        objective, nonlinear_constraint = combine_args(objective, nonlinear_constraint)
        assert set(objective.things) == set(nonlinear_constraint.things), (
            inconsistent_things_error_msg
            + f"objective+constraints expected things {set(objective.things)}, \n"
            f"nonlinear constraints expected things {set(nonlinear_constraint.things)}"
        )
    assert set(objective.things) == set(things), (
        inconsistent_things_error_msg
        + f"objective+constraints expected things {set(objective.things)}, \n"
        f"optimizer got things {set(things)}"
    )

    # wrap to handle linear constraints
    if linear_constraint is not None:
        linear_constraint_options = options.pop("linear_constraint_options", {})
        objective = LinearConstraintProjection(
            objective, linear_constraint, **linear_constraint_options
        )
        objective.build(verbose=verbose)
        if nonlinear_constraint is not None:
            nonlinear_constraint = LinearConstraintProjection(
                nonlinear_constraint, linear_constraint
            )
            nonlinear_constraint.build(verbose=verbose)

    if linear_constraint is not None and not isinstance(x_scale, str):
        # need to project x_scale down to correct size
        Z = objective._Z
        x_scale = np.broadcast_to(x_scale, objective._objective.dim_x)
        x_scale = np.abs(np.diag(Z.T @ np.diag(x_scale[objective._unfixed_idx]) @ Z))
        x_scale = np.where(x_scale < np.finfo(x_scale.dtype).eps, 1, x_scale)

    if objective.scalar and (not optimizers[method]["scalar"]):
        warnings.warn(
            colored(
                "method {} is not intended for scalar objective".format(
                    ".".join([method]) + " function"
                ),
                "yellow",
            )
        )

    return objective, nonlinear_constraint, x_scale


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
