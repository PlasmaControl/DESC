import numpy as np
import scipy.optimize
from termcolor import colored
from desc.utils import Timer
from desc.optimize import fmintr, lsqtr
from desc.io import IOAble
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
    _scalar_methods = _desc_scalar_methods + _scipy_scalar_methods
    _least_squares_methods = _scipy_least_squares_methods + _desc_least_squares_methods
    _scipy_methods = _scipy_least_squares_methods + _scipy_scalar_methods
    _desc_methods = _desc_least_squares_methods + _desc_scalar_methods
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

    def optimize(
        self,
        objective,
        x_init,
        args=(),
        x_scale="auto",
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=1,
        maxiter=None,
        options={},
    ):
        """Optimize the objective function

        Parameters
        ----------
        objective : ObjectiveFunction
            objective function to optimize
        x_init : ndarray
            initial guess. Should satisfy any constraints on x
        args : tuple, optional
            additional arguments passed to objective fun and derivatives
        x_scale : array_like or ``'auto'``, optional
            Characteristic scale of each variable. Setting ``x_scale`` is equivalent
            to reformulating the problem in scaled variables ``xs = x / x_scale``.
            An alternative view is that the size of a trust region along jth
            dimension is proportional to ``x_scale[j]``. Improved convergence may
            be achieved by setting ``x_scale`` such that a step of a given size
            along any of the scaled variables has a similar effect on the cost
            function. If set to ``'auto'``, the scale is iteratively updated using the
            inverse norms of the columns of the jacobian or hessian matrix.
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
        verbose : integer, optional
            * 0  : work silently.
            * 1-2 : display a termination report.
            * 3 : display progress during iterations
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
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.

        """
        # TODO: document options
        timer = Timer()
        if objective.scalar and (self.method in Optimizer._least_squares_methods):
            raise ValueError(
                colored(
                    "method {} is incompatible with scalar objective function".format(
                        ".".join([self.method])
                    ),
                    "red",
                )
            )

        if not objective.compiled:
            mode = "scalar" if self.method in Optimizer._scalar_methods else "lsq"
            objective.compile(x_init, args, verbose, mode=mode)

        # need some weird logic because scipy optimizers expect disp={0,1,2}
        # while we use verbose={0,1,2,3}
        disp = verbose - 1 if verbose > 1 else verbose

        if self.method in Optimizer._desc_methods:
            if not isinstance(x_scale, str) and np.allclose(x_scale, 1):
                options.setdefault("initial_trust_radius", 0.5)
                options.setdefault("max_trust_radius", 1.0)

        if verbose > 0:
            print("Starting optimization")
        timer.start("Solution time")

        if self.method in Optimizer._scipy_scalar_methods:

            allx = []
            allf = []
            msg = [""]

            def callback(x):
                if len(allx) > 0:
                    dx = allx[-1] - x
                    dx_norm = np.linalg.norm(dx)
                    if dx_norm > 0:
                        fx = objective.compute_scalar(x, *args)
                        df = allf[-1] - fx
                        allx.append(x)
                        allf.append(fx)
                        x_norm = np.linalg.norm(x)
                        if verbose > 2:
                            print_iteration_nonlinear(
                                len(allx), None, fx, df, dx_norm, None
                            )
                        success, message = check_termination(
                            df,
                            fx,
                            dx_norm,
                            x_norm,
                            np.inf,
                            1,
                            ftol,
                            xtol,
                            0,
                            len(allx),
                            maxiter,
                            0,
                            np.inf,
                            0,
                            np.inf,
                            0,
                            np.inf,
                        )
                        if success:
                            msg[0] = message
                            raise StopIteration
                else:
                    dx = None
                    df = None
                    fx = objective.compute_scalar(x, *args)
                    allx.append(x)
                    allf.append(fx)
                    dx_norm = None
                    x_norm = np.linalg.norm(x)
                    if verbose > 2:
                        print_iteration_nonlinear(
                            len(allx), None, fx, df, dx_norm, None
                        )

            print_header_nonlinear()
            # print_iteration_nonlinear(1, None, allf[0], None, None, None)
            try:
                result = scipy.optimize.minimize(
                    objective.compute_scalar,
                    x0=x_init,
                    args=args,
                    method=self.method[len("scipy-") :],
                    jac=objective.grad_x,
                    hess=objective.hess_x,
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

            x_scale = "jac" if x_scale == "auto" else x_scale
            allx = []

            def jac_wrapped(x, *args):
                allx.append(x)
                return objective.jac_x(x, *args)

            result = scipy.optimize.least_squares(
                objective.compute,
                x0=x_init,
                args=args,
                jac=jac_wrapped,
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

            x_scale = "hess" if x_scale == "auto" else x_scale
            method = (
                self.method if "bfgs" not in self.method else self.method.split("-")[0]
            )
            hess = objective.hess_x if "bfgs" not in self.method else "bfgs"
            result = fmintr(
                objective.compute_scalar,
                x0=x_init,
                grad=objective.grad_x,
                hess=hess,
                args=args,
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
                objective.compute,
                x0=x_init,
                jac=objective.jac_x,
                args=args,
                x_scale=x_scale,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                verbose=disp,
                maxiter=maxiter,
                callback=None,
                options=options,
            )

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
