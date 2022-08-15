import scipy.optimize
import warnings
from termcolor import colored

from desc.backend import jnp
from desc.utils import Timer
from desc.io import IOAble
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    RadialForceBalance,
    HelicalForceBalance,
    CurrentDensity,
    WrappedEquilibriumObjective,
)
from desc.objectives.utils import factorize_linear_constraints
from desc.optimize import fmintr, lsqtr
from .utils import check_termination, print_header_nonlinear, print_iteration_nonlinear


class Optimizer(IOAble):
    """A helper class to wrap several different optimization routines

    Offers all of the ``scipy.optimize.least_squares`` routines  and several of the most
    useful ``scipy.optimize.minimize`` routines.
    Also offers several custom routines specifically designed for DESC, both scalar and
    least squares routines with and without jacobian/hessian information.

    Parameters
    ----------
    method : str
        name of the optimizer to use. Options are:

        * scipy scalar routines: ``'scipy-bfgs'``, ``'scipy-trust-exact'``,
          ``'scipy-trust-ncg'``, ``'scipy-trust-krylov'``
        * scipy least squares routines: ``'scipy-trf'``, ``'scipy-lm'``, ``'scipy-dogbox'``
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
    )

    def __init__(self, method):

        self.method = method

    def __repr__(self):
        """string form of the object"""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (method={})".format(self.method)
        )

    @property
    def method(self):
        """str : name of the optimization method"""
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
    def optimize(
        self,
        eq,
        objective,
        constraints=(),
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
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
            Default is 1e-8. The optimization process is stopped when ``dF < ftol * F``,
            and there was an adequate agreement between a local quadratic model and the
            true model in the last step.
            If None, the termination by this condition is disabled.
        xtol : float or None, optional
            Tolerance for termination by the change of the independent variables.
            Default is 1e-8.
            Optimization is stopped when ``norm(dx) < xtol * (xtol + norm(x))``.
            If None, the termination by this condition is disabled.
        gtol : float or None, optional
            Absolute tolerance for termination by the norm of the gradient.
            Default is 1e-8. Optimizer teriminates when ``norm(g) < gtol``, where
            # FIXME: missing documentation!
            If None, the termination by this condition is disabled.
        x_scale : array_like or ``'auto'``, optional
            Characteristic scale of each variable. Setting ``x_scale`` is equivalent
            to reformulating the problem in scaled variables ``xs = x / x_scale``.
            An alternative view is that the size of a trust region along jth
            dimension is proportional to ``x_scale[j]``. Improved convergence may
            be achieved by setting ``x_scale`` such that a step of a given size
            along any of the scaled variables has a similar effect on the cost
            function. If set to ``'auto'``, the scale is iteratively updated using the
            inverse norms of the columns of the jacobian or hessian matrix.
        verbose : integer, optional
            * 0  : work silently.
            * 1-2 : display a termination report.
            * 3 : display progress during iterations
        maxiter : int, optional
            Maximum number of iterations. Defaults to size(x)*100.
        options : dict, optional
            Dictionary of optional keyword arguments to override default solver settings.
            See the code for more details.

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
        # scipy optimizers expect disp={0,1,2} while we use verbose={0,1,2,3}
        disp = verbose - 1 if verbose > 1 else verbose

        timer = Timer()

        if self.method in Optimizer._desc_methods:
            if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
                options.setdefault("initial_trust_radius", 0.5)
                options.setdefault("max_trust_radius", 1.0)

        if not isinstance(constraints, tuple):
            constraints = (constraints,)
        linear_constraints = tuple(
            constraint for constraint in constraints if constraint.linear
        )
        nonlinear_constraints = tuple(
            constraint for constraint in constraints if not constraint.linear
        )

        # wrap nonlinear constraints if necessary
        wrapped = False
        if len(nonlinear_constraints) > 0 and (
            self.method not in Optimizer._constrained_methods
        ):
            wrapped = True
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
                        "optimizer method {} ".format(self.method)
                        + "cannot handle general nonlinear constraint {}.".format(
                            constraint
                        )
                    )
            perturb_options = options.pop("perturb_options", {})
            perturb_options.setdefault("verbose", 0)
            solve_options = options.pop("solve_options", {})
            solve_options.setdefault("verbose", 0)
            objective = WrappedEquilibriumObjective(
                objective,
                eq_objective=ObjectiveFunction(nonlinear_constraints),
                perturb_options=perturb_options,
                solve_options=solve_options,
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
        _, _, _, _, Z, unfixed_idx, project, recover = factorize_linear_constraints(
            linear_constraints, objective.args
        )
        timer.stop("linear constraint factorize")
        if verbose > 1:
            timer.disp("linear constraint factorize")

        x0_reduced = project(objective.x(eq))

        if verbose > 0:
            print("Number of parameters: {}".format(x0_reduced.size))
            print("Number of objectives: {}".format(objective.dim_f))

        if verbose > 0:
            print("Starting optimization")
        timer.start("Solution time")

        def compute_wrapped(x_reduced):
            x = recover(x_reduced)
            f = objective.compute(x)
            if self.method in Optimizer._scalar_methods:
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

        if self.method in Optimizer._scipy_scalar_methods:

            allx = []
            allf = []
            msg = [""]

            def callback(x_reduced):
                x = recover(x_reduced)
                if len(allx) > 0:
                    dx = allx[-1] - x_reduced
                    dx_norm = jnp.linalg.norm(dx)
                    if dx_norm > 0:
                        fx = objective.compute_scalar(x)
                        df = allf[-1] - fx
                        allx.append(x_reduced)
                        allf.append(fx)
                        x_norm = jnp.linalg.norm(x_reduced)
                        if verbose > 2:
                            print_iteration_nonlinear(
                                len(allx), None, fx, df, dx_norm, None
                            )
                        success, message = check_termination(
                            df,
                            fx,
                            dx_norm,
                            x_norm,
                            jnp.inf,
                            1,
                            ftol,
                            xtol,
                            0,
                            len(allx),
                            maxiter,
                            0,
                            jnp.inf,
                            0,
                            jnp.inf,
                            0,
                            jnp.inf,
                        )
                        if success:
                            msg[0] = message
                            raise StopIteration
                else:
                    dx = None
                    df = None
                    fx = objective.compute_scalar(x)
                    allx.append(x_reduced)
                    allf.append(fx)
                    dx_norm = None
                    x_norm = jnp.linalg.norm(x_reduced)
                    if verbose > 2:
                        print_iteration_nonlinear(
                            len(allx), None, fx, df, dx_norm, None
                        )

            print_header_nonlinear()
            try:
                result = scipy.optimize.minimize(
                    compute_scalar_wrapped,
                    x0=x0_reduced,
                    args=(),
                    method=self.method[len("scipy-") :],
                    jac=grad_wrapped,
                    hess=hess_wrapped,
                    tol=gtol,
                    callback=callback,
                    options={"maxiter": maxiter, "disp": disp, **options},
                )
                result["allx"] = allx
            except StopIteration:
                result = {
                    "x": allx[-1],
                    "allx": allx,
                    "fun": allf[-1],
                    "message": msg[0],
                    "nit": len(allx),
                    "success": True,
                }
                if verbose > 1:
                    print(msg[0])
                    print(
                        "         Current function value: {:.3e}".format(result["fun"])
                    )
                    print("         Iterations: {:d}".format(result["nit"]))

        elif self.method in Optimizer._scipy_least_squares_methods:

            allx = []
            x_scale = "jac" if x_scale == "auto" else x_scale

            def jac(x_reduced):
                allx.append(x_reduced)
                return jac_wrapped(x_reduced)

            result = scipy.optimize.least_squares(
                compute_wrapped,
                x0=x0_reduced,
                args=(),
                jac=jac,
                method=self.method[len("scipy-") :],
                x_scale=x_scale,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                max_nfev=maxiter,
                verbose=disp,
            )
            result["allx"] = allx

        elif self.method in Optimizer._desc_scalar_methods:

            hess = hess_wrapped if "bfgs" not in self.method else "bfgs"
            method = (
                self.method if "bfgs" not in self.method else self.method.split("-")[0]
            )
            x_scale = "hess" if x_scale == "auto" else x_scale

            result = fmintr(
                compute_scalar_wrapped,
                x0=x0_reduced,
                grad=grad_wrapped,
                hess=hess,
                args=(),
                method=method,
                x_scale=x_scale,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                verbose=disp,
                maxiter=maxiter,
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
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                verbose=disp,
                maxiter=maxiter,
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
                timer["Solution time"] / result.get("nit", result.get("nfev")),
            )
        for key in ["hess", "hess_inv", "jac", "grad", "active_mask"]:
            _ = result.pop(key, None)

        return result
