"""Class for wrapping a number of common optimization methods."""

import warnings

import numpy as np
from termcolor import colored

from desc.backend import jnp
from desc.io import IOAble
from desc.objectives import (
    CurrentDensity,
    FixCurrent,
    FixIota,
    ForceBalance,
    HelicalForceBalance,
    ObjectiveFunction,
    RadialForceBalance,
    WrappedEquilibriumObjective,
)
from desc.objectives.utils import factorize_linear_constraints
from desc.utils import Timer

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
        if method not in Optimizer._all_methods:
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
            If None, the termination by this condition is disabled.
        xtol : float or None, optional
            Tolerance for termination by the change of the independent variables.
            Optimization is stopped when ``norm(dx) < xtol * (xtol + norm(x))``.
            If None, the termination by this condition is disabled.
        gtol : float or None, optional
            Absolute tolerance for termination by the norm of the gradient.
            Optimizer terminates when ``norm(g) < gtol``, where
            If None, the termination by this condition is disabled.
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
            * 1-2 : display a termination report.
            * 3 : display progress during iterations
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
        # scipy optimizers expect disp={0,1,2} while we use verbose={0,1,2,3}
        disp = verbose - 1 if verbose > 1 else verbose

        if (
            self.method in Optimizer._desc_methods
            and self.method not in Optimizer._desc_stochastic_methods
        ):
            if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
                options.setdefault("initial_trust_radius", 0.5)
                options.setdefault("max_trust_radius", 1.0)

        linear_constraints, nonlinear_constraints = _parse_constraints(constraints)
        # wrap nonlinear constraints if necessary
        wrapped = False
        if len(nonlinear_constraints) > 0 and (
            self.method not in Optimizer._constrained_methods
        ):
            wrapped = True
            objective = _wrap_nonlinear_constraints(
                objective, nonlinear_constraints, self.method, options
            )

        if not objective.built:
            objective.build(eq, verbose=verbose)
        if not objective.compiled:
            mode = "scalar" if self.method in Optimizer._scalar_methods else "lsq"
            objective.compile(mode, verbose)
        for constraint in linear_constraints:
            if not constraint.built:
                constraint.build(eq, verbose=verbose)

        if objective.scalar and (self.method in Optimizer._least_squares_methods):
            warnings.warn(
                colored(
                    "method {} is not intended for scalar objective function".format(
                        ".".join([self.method])
                    ),
                    "yellow",
                )
            )

        if verbose > 0:
            print("Factorizing linear constraints")
        timer.start("linear constraint factorize")
        (
            compute_wrapped,
            compute_scalar_wrapped,
            grad_wrapped,
            hess_wrapped,
            jac_wrapped,
            project,
            recover,
        ) = _wrap_objective_with_constraints(objective, linear_constraints, self.method)
        timer.stop("linear constraint factorize")
        if verbose > 1:
            timer.disp("linear constraint factorize")

        x0_reduced = project(objective.x(eq))

        stoptol = _get_default_tols(
            self.method,
            ftol,
            xtol,
            gtol,
            maxiter,
            options,
        )

        if verbose > 0:
            print("Number of parameters: {}".format(x0_reduced.size))
            print("Number of objectives: {}".format(objective.dim_f))

        if verbose > 0:
            print("Starting optimization")
        timer.start("Solution time")

        if self.method in Optimizer._scipy_scalar_methods:

            method = self.method[len("scipy-") :]
            x_scale = 1 if x_scale == "auto" else x_scale
            if isinstance(x_scale, str):
                raise ValueError(
                    f"Method {self.method} does not support x_scale type {x_scale}"
                )
            result = _optimize_scipy_minimize(
                compute_scalar_wrapped,
                grad_wrapped,
                hess_wrapped,
                x0_reduced,
                method,
                x_scale,
                verbose,
                stoptol,
                options,
            )

        elif self.method in Optimizer._scipy_least_squares_methods:

            x_scale = "jac" if x_scale == "auto" else x_scale
            method = self.method[len("scipy-") :]

            result = _optimize_scipy_least_squares(
                compute_wrapped,
                jac_wrapped,
                x0_reduced,
                method,
                x_scale,
                verbose,
                stoptol,
                options,
            )

        elif self.method in Optimizer._desc_scalar_methods:

            hess = hess_wrapped if "bfgs" not in self.method else "bfgs"
            method = (
                self.method if "bfgs" not in self.method else self.method.split("-")[0]
            )
            if isinstance(x_scale, str):
                if x_scale == "auto":
                    if "bfgs" not in self.method:
                        x_scale = "hess"
                    else:
                        x_scale = 1
            result = fmintr(
                compute_scalar_wrapped,
                x0=x0_reduced,
                grad=grad_wrapped,
                hess=hess,
                args=(),
                method=method,
                x_scale=x_scale,
                ftol=stoptol["ftol"],
                xtol=stoptol["xtol"],
                gtol=stoptol["gtol"],
                maxiter=stoptol["maxiter"],
                verbose=disp,
                callback=None,
                options=options,
            )

        elif self.method in Optimizer._desc_stochastic_methods:

            result = sgd(
                compute_scalar_wrapped,
                x0=x0_reduced,
                grad=grad_wrapped,
                args=(),
                method=self.method,
                ftol=stoptol["ftol"],
                xtol=stoptol["xtol"],
                gtol=stoptol["gtol"],
                maxiter=stoptol["maxiter"],
                verbose=disp,
                callback=None,
                options=options,
            )

        elif self.method in Optimizer._desc_least_squares_methods:

            result = lsqtr(
                compute_wrapped,
                x0=x0_reduced,
                jac=jac_wrapped,
                args=(),
                x_scale=x_scale,
                ftol=stoptol["ftol"],
                xtol=stoptol["xtol"],
                gtol=stoptol["gtol"],
                maxiter=stoptol["maxiter"],
                verbose=disp,
                callback=None,
                options=options,
            )

        if wrapped:
            result["history"] = objective.history
        else:
            result["history"] = {}
            for arg in objective.args:
                result["history"][arg] = []
            for x_reduced in result["allx"]:
                x = recover(x_reduced)
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


def _parse_constraints(constraints):
    if not isinstance(constraints, tuple):
        constraints = (constraints,)
    linear_constraints = tuple(
        constraint for constraint in constraints if constraint.linear
    )
    nonlinear_constraints = tuple(
        constraint for constraint in constraints if not constraint.linear
    )
    if any(isinstance(lc, FixCurrent) for lc in linear_constraints) and any(
        isinstance(lc, FixIota) for lc in linear_constraints
    ):
        raise ValueError(
            "Toroidal current and rotational transform cannot be "
            + "constrained simultaneously."
        )
    return linear_constraints, nonlinear_constraints


def _wrap_objective_with_constraints(objective, linear_constraints, method):
    """Factorize constraints and make new functions that project/recover + evaluate."""
    _, _, _, _, Z, unfixed_idx, project, recover = factorize_linear_constraints(
        linear_constraints, objective.args
    )

    def compute_wrapped(x_reduced):
        x = recover(x_reduced)
        f = objective.compute(x)
        if method in Optimizer._scalar_methods:
            return f.squeeze()
        else:
            return jnp.atleast_1d(f)

    def compute_scalar_wrapped(x_reduced):
        x = recover(x_reduced)
        return objective.compute_scalar(x)

    def grad_wrapped(x_reduced):
        x = recover(x_reduced)
        df = objective.grad(x)
        return df[unfixed_idx] @ Z

    def hess_wrapped(x_reduced):
        x = recover(x_reduced)
        df = objective.hess(x)
        return Z.T @ df[unfixed_idx, :][:, unfixed_idx] @ Z

    def jac_wrapped(x_reduced):
        x = recover(x_reduced)
        df = objective.jac(x)
        return df[:, unfixed_idx] @ Z

    return (
        compute_wrapped,
        compute_scalar_wrapped,
        grad_wrapped,
        hess_wrapped,
        jac_wrapped,
        project,
        recover,
    )


def _wrap_nonlinear_constraints(objective, nonlinear_constraints, method, options):
    """Use WrappedEquilibriumObjective to hanle nonlinear equilibrium constraints."""
    for constraint in nonlinear_constraints:
        if not isinstance(
            constraint,
            (
                ForceBalance,
                RadialForceBalance,
                HelicalForceBalance,
                CurrentDensity,
            ),
        ):
            raise ValueError(
                "optimizer method {} ".format(method)
                + "cannot handle general nonlinear constraint {}.".format(constraint)
            )
    perturb_options = options.pop("perturb_options", {})
    perturb_options.setdefault("verbose", 0)
    perturb_options.setdefault("include_f", False)
    solve_options = options.pop("solve_options", {})
    solve_options.setdefault("verbose", 0)
    objective = WrappedEquilibriumObjective(
        objective,
        eq_objective=ObjectiveFunction(nonlinear_constraints),
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
        options.pop(
            "xtol", 1e-6 if method in Optimizer._desc_stochastic_methods else 1e-4
        ),
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
