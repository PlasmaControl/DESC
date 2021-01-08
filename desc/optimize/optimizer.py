import numpy as np
from scipy.optimize import least_squares, minimize
from termcolor import colored

from desc.backend import jit
from desc.optimize import fmin_scalar


class Optimizer():

    def __init__(self, method):

        self.method = method
        self._scipy_least_squares_methods = [
            'scipy-trf', 'scipy-lm', 'scipy-dogbox']
        self._scipy_minimize_methods = ['scipy-bfgs', 'scipy-dogleg', 'scipy-trust-exact',
                                        'scipy-trust-ncg', 'scipy-trust-krylov']
        self._desc_scalar_methods = ['dogleg', 'subspace']
        self._all_methods = self._scipy_least_squares_methods + \
            self._scipy_minimize_methods + self._desc_scalar_methods

        if self.method not in self._all_methods:
            raise NotImplementedError(colored(
                "method must be one of {}".format('.'.join([self._all_methods])), 'red'))

    def optimize(self, objective,
                 x_init,
                 # TODO: get rid of init hess, just call the function once
                 init_hess=None,
                 args=(),
                 x_scale='auto',
                 ftol=1e-8,
                 xtol=1e-8,
                 agtol=1e-8,
                 rgtol=1e-8,
                 verbose=1,
                 maxiter=None,
                 callback=None,  # TODO: what is callback doing?
                 options={}):

        if self.method in self._scipy_minimize_methods:
            obj_jit = jit(objective.compute_scalar)
            grad_jit = jit(objective.grad)
            hess_jit = jit(objective.hess)

            f0 = obj_jit(x_init, *args)
            g0 = grad_jit(x_init, *args)
            if 'bfgs' not in self.method:
                h0 = hess_jit(x_init, *args)

            out = minimize(obj_jit,
                           x0=x_init,
                           args=args,
                           method=self.method[len('scipy-'):],
                           jac=grad_jit,
                           hess=hess_jit,
                           tol=agtol,
                           options={'maxiter': maxiter,
                                    'disp': verbose,
                                    **options})

        elif self.method in self._scipy_least_squares_methods:
            obj_jit = jit(objective.compute)
            jac_jit = jit(objective.jac)

            f0 = obj_jit(x_init, *args)
            j0 = jac_jit(x_init, *args)

            x_scale = 'jac' if x_scale == 'auto' else x_scale

            out = least_squares(obj_jit,
                                x0=x_init,
                                args=args,
                                jac=jac_jit,
                                method=self.method,
                                x_scale=x_scale,
                                ftol=ftol,
                                xtol=xtol,
                                gtol=agtol,
                                max_nfev=maxiter,
                                verbose=verbose)

        elif self.method in self._desc_scalar_methods:
            obj_jit = jit(objective.compute_scalar)
            grad_jit = jit(objective.grad)
            hess_jit = jit(objective.hess)

            f0 = obj_jit(x_init, *args)
            g0 = grad_jit(x_init, *args)
            if 'bfgs' not in self.method:
                h0 = hess_jit(x_init, *args)

            x_scale = 'hess' if x_scale == 'auto' else x_scale
            out = fmin_scalar(obj_jit,
                              x0=x_init,
                              grad=grad_jit,
                              hess=hess_jit,
                              init_hess=None,
                              args=args,
                              method=self.method,
                              x_scale=x_scale,
                              ftol=ftol,
                              xtol=xtol,
                              agtol=agtol,
                              rgtol=rgtol,
                              verbose=verbose,
                              maxiter=maxiter,
                              callback=None,
                              options=options)

        return out
