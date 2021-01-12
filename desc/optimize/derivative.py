import numpy as np
import functools
from abc import ABC, abstractmethod
from termcolor import colored

from desc.backend import jnp, cho_factor, cho_solve, jit, use_jax
from desc.optimize.utils import make_spd, chol_U_update


class OptimizerDerivative(ABC):
    """Abstract base class for hessians and jacobians used in the optimizer"""

    shape = None
    min_eig = None
    is_pos_def = None
    negative_curvature_direction = None

    @abstractmethod
    def update(self, x_new, x_old, grad_new, grad_old, iteration):
        """Update the internal matrix A"""
        pass

    @abstractmethod
    def recompute(self, x):
        """Recompute the full correct internal matrix at the point x"""
        pass

    @abstractmethod
    def get_matrix(self):
        """Return the internal matrix A"""
        pass

    @abstractmethod
    def get_inverse(self):
        """Return the inverse of the internal matrix A^-1"""
        pass

    @abstractmethod
    def dot(self, x):
        """Compute dot(A,x)"""
        pass

    @abstractmethod
    def solve(self, b):
        """Solve A*x = b for x"""
        pass

    @abstractmethod
    def get_scale(self, prev_scale=None):
        """Compute scaling vector"""
        pass


class CholeskyHessian(OptimizerDerivative):
    def __init__(
        self,
        n,
        init_hess="auto",
        hessfun=None,
        hessfun_args=(),
        exception_strategy="damp_update",
        min_curvature=None,
        damp_ratio=0.2,
    ):
        self._n = n
        self.shape = (n, n)
        self.is_pos_def = True
        self.min_eig = None
        self.negative_curvature_direction = None
        self._damp_ratio = damp_ratio

        if hessfun is not None:
            if callable(hessfun):
                self._hessfun = hessfun
                self._hessfun_args = hessfun_args
            else:
                raise ValueError(colored("hessfun should be callable or None", "red"))
        else:
            self._hessfun = None
            self._hessfun_args = ()

        if exception_strategy == "skip_update":
            if min_curvature is not None:
                self.min_curvature = min_curvature
            else:
                self.min_curvature = 1e-8
        elif exception_strategy == "damp_update":
            if min_curvature is not None:
                self.min_curvature = min_curvature
            else:
                self.min_curvature = 0.2
        else:
            raise ValueError(
                colored(
                    "'exception_strategy' must be 'skip_update' " "or 'damp_update'",
                    "red",
                )
            )
        self.exception_strategy = exception_strategy

        if init_hess is None and hessfun is None:
            self._U = np.eye(n)
            self._initialized = True
            self._initialization = "eye"
        elif init_hess is None and hessfun is not None:
            self._U = np.eye(n)
            self._initialized = False
            self._initialization = "hessfun"
        elif isinstance(init_hess, str) and init_hess == "auto":
            self._U = np.eye(n)
            self._initialized = False
            self._initialization = "auto"
        elif isinstance(init_hess, str):
            raise ValueError(colored("unknown hessian initialization", "red"))
        else:
            init_hess = make_spd(init_hess, delta=self.min_curvature, tol=0.1)
            self._U = self._cholesky(init_hess).T
            self._initialized = True
            self._initialization = "init_hess"

    def _auto_scale_init(self, delta_x, delta_grad):
        # Heuristic to scale matrix at first iteration.
        # Described in Nocedal and Wright "Numerical Optimization"
        # p.143 formula (6.20).
        s_norm2 = np.dot(delta_x, delta_x)
        y_norm2 = np.dot(delta_grad, delta_grad)
        ys = np.abs(np.dot(delta_grad, delta_x))
        if ys == 0.0 or y_norm2 == 0 or s_norm2 == 0:
            scale = 1
        else:
            scale = y_norm2 / ys
        self._U = self._U * np.sqrt(scale)
        self._initialized = True

    def recompute(self, x):
        H = self._hessfun(x, *self._hessfun_args)
        H = make_spd(H, delta=self.min_curvature, tol=0.1)
        U = self._cholesky(H)
        self._U = U  # self._damp_ratio*self._U + (1-self._damp_ratio)*U

    def update(self, x_new, x_old, grad_new, grad_old):
        """Update internal matrix.
        Update Hessian matrix or its inverse (depending on how 'approx_type'
        is defined) using information about the last evaluated points.
        Parameters
        ----------
        delta_x : ndarray
            The difference between two points the gradient
            function have been evaluated at: ``delta_x = x2 - x1``.
        delta_grad : ndarray
            The difference between the gradients:
            ``delta_grad = grad(x2) - grad(x1)``.
        """
        x_new = np.asarray(x_new)
        x_old = np.asarray(x_old)
        grad_new = np.asarray(grad_new)
        grad_old = np.asarray(grad_old)

        delta_x = x_new - x_old
        delta_grad = grad_new - grad_old

        if np.all(delta_x == 0.0):
            return
        if np.all(delta_grad == 0.0):
            return
        if not self._initialized:
            if self._initialization == "auto":
                self._auto_scale_init(delta_x, delta_grad)
            elif self._initialization == "hessfun":
                self.recompute(x_new)

        self._bfgs_update(delta_x, delta_grad)

    def _bfgs_update(self, delta_x, delta_grad):

        if np.all(delta_x == 0.0):
            return
        if np.all(delta_grad == 0.0):
            return

        s = delta_x
        y = delta_grad

        # Do some common operations
        sy = np.dot(s, y)
        Bs = self.dot(s)
        sBs = Bs.dot(s)

        # Check if curvature condition is violated
        if sy <= self.min_curvature * sBs:

            if self.exception_strategy == "skip_update":
                return
            # interpolate between the actual BFGS
            # result and the unmodified matrix.
            elif self.exception_strategy == "damp_update":
                update_factor = (1 - self.min_curvature) / (1 - sy / sBs)
                y = update_factor * y + (1 - update_factor) * Bs
                sy = np.dot(s, y)

        u = np.asarray(y)
        v = np.asarray(Bs)
        alpha = np.asarray(1 / sy)
        beta = np.asarray(-1 / sBs)

        self._U = chol_U_update(np.asarray(self._U), u, alpha)
        self._U = chol_U_update(np.asarray(self._U), v, beta)

    def get_matrix(self):
        return jnp.dot(self._U.T, self._U)

    def get_inverse(self):
        return cho_solve((self._U, False), jnp.eye(self._n))

    def dot(self, x):
        return self._dot(self._U.T, self._dot(self._U, x))

    def solve(self, x):
        return self._cho_solve(self._U, x)

    @functools.partial(jit, static_argnums=(0,))
    def _dot(self, A, b):
        return jnp.dot(A, b)

    @functools.partial(jit, static_argnums=(0,))
    def _cho_solve(self, U, x):
        return cho_solve((U, False), x)

    @functools.partial(jit, static_argnums=(0,))
    def _cholesky(self, A):
        return jnp.linalg.cholesky(A).T

    def get_scale(self, prev_scale=None):

        return compute_jac_scale(self._U, prev_scale)


def compute_jac_scale(A, prev_scale_inv=None):
    scale_inv = jnp.sum(A ** 2, axis=0) ** 0.5
    scale_inv = jnp.where(scale_inv == 0, 1, scale_inv)

    if prev_scale_inv is not None:
        scale_inv = jnp.maximum(scale_inv, prev_scale_inv)
    return 1 / scale_inv, scale_inv
