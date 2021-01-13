import numpy as np
from scipy.optimize import least_squares, minimize
from termcolor import colored
from desc.utils import Timer
from desc.backend import jit
from desc.optimize import fmin_scalar


class Optimizer:

    _scipy_least_squares_methods = ["scipy-trf", "scipy-lm", "scipy-dogbox"]
    _scipy_minimize_methods = [
        "scipy-bfgs",
        "scipy-dogleg",
        "scipy-trust-exact",
        "scipy-trust-ncg",
        "scipy-trust-krylov",
    ]
    _desc_scalar_methods = ["dogleg", "subspace"]
    _hessian_free_methods = ["scipy-bfgs"]
    _scalar_methods = _desc_scalar_methods + _scipy_minimize_methods
    _least_squares_methods = _scipy_least_squares_methods
    _all_methods = (
        _scipy_least_squares_methods + _scipy_minimize_methods + _desc_scalar_methods
    )

    def __init__(self, method, objective, use_jit=True, device=None):

        self._check_method_objective(method, objective)
        self._method = method
        self._objective = objective
        self.use_jit = use_jit
        self._device = device
        self.timer = Timer()
        self._set_compute_funs()
        self.compiled = False

    def _check_method_objective(self, method, objective):
        if method not in Optimizer._all_methods:
            raise NotImplementedError(
                colored(
                    "method must be one of {}".format(
                        ".".join([Optimizer._all_methods])
                    ),
                    "red",
                )
            )
        if objective.scalar and (method in Optimizer._least_squares_methods):
            raise ValueError(
                colored(
                    "method {} is incompatible with scalar objective function".format(
                        ".".join([method])
                    ),
                    "red",
                )
            )

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        self._check_method_objective(method, self.objective)
        if (
            method in Optimizer._scalar_methods
            and self.method in Optimizer._least_squares_methods
        ) or (
            method in Optimizer._least_squares_methods
            and self.method in Optimizer._scalar_methods
        ):
            self.compiled = False

        self._method = method
        self._set_compute_funs()

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, objective):
        self._check_method_objective(self.method, objective)
        self._objective = objective
        self.method = self.method
        self._set_compute_funs()
        self.compiled = False

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device
        self._set_compute_funs()
        self.compiled = False

    def _set_compute_funs(self):

        if self.use_jit:
            if self.method in Optimizer._scalar_methods:
                self._fun = jit(self.objective.compute_scalar, device=self.device)
                self._grad = jit(self.objective.grad_x, device=self.device)
                if self.method in Optimizer._hessian_free_methods:
                    self._hess = None
                else:
                    self._hess = jit(self.objective.hess_x, device=self.device)
            else:
                self._fun = jit(self.objective.compute, device=self.device)
                self._jac = jit(self.objective.jac_x, device=self.device)
        else:
            if self.method in Optimizer._scalar_methods:
                self._fun = self.objective.compute_scalar
                self._grad = self.objective.grad_x
                if self.method in Optimizer._hessian_free_methods:
                    self._hess = None
                else:
                    self._hess = self.objective.hess_x
            else:
                self._fun = self.objective.compute
                self._jac = self.objective.jac_x

    def compile(self, x, args, verbose=1):

        if not self.use_jit:
            return
        if verbose > 0:
            print("Compiling objective function")
        self.timer.start("Compilation time")

        if self.method in Optimizer._scalar_methods:
            f0 = self._fun(x, *args)
            g0 = self._grad(x, *args)
            if self.method not in Optimizer._hessian_free_methods:
                H0 = self._hess(x, *args)
        else:
            f0 = self._fun(x, *args)
            J0 = self._jac(x, *args)
        self.timer.stop("Compilation time")
        if verbose > 1:
            self.timer.disp("Compilation time")
        self.compiled = True

    def optimize(
        self,
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
        if self.use_jit and not self.compiled:
            self.compile(x_init, args, verbose)

        # need some weird logic because scipy optimizers expect disp={0,1,2}
        # while we use verbose={0,1,2,3}
        disp = verbose - 1 if verbose > 1 else verbose

        if verbose > 0:
            print("Starting optimization")

        if self.method in Optimizer._scipy_minimize_methods:

            out = minimize(
                self._fun,
                x0=x_init,
                args=args,
                method=self.method[len("scipy-") :],
                jac=self._grad,
                hess=self._hess,
                tol=gtol,
                options={"maxiter": maxiter, "disp": disp, **options},
            )

        elif self.method in Optimizer._scipy_least_squares_methods:

            x_scale = "jac" if x_scale == "auto" else x_scale

            out = least_squares(
                self._fun,
                x0=x_init,
                args=args,
                jac=self._jac,
                method=self.method[len("scipy-") :],
                x_scale=x_scale,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                max_nfev=maxiter,
                verbose=disp,
            )

        elif self.method in Optimizer._desc_scalar_methods:

            x_scale = "hess" if x_scale == "auto" else x_scale

            out = fmin_scalar(
                self._fun,
                x0=x_init,
                grad=self._grad,
                hess=self._hess,
                init_hess=None,
                args=args,
                method=self.method,
                x_scale=x_scale,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                verbose=disp,
                maxiter=maxiter,
                callback=None,
                options=options,
            )

        return out
