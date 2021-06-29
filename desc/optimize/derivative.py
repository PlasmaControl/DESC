import numpy as np
import functools
from abc import ABC, abstractmethod
from termcolor import colored

from desc.backend import jnp, cho_factor, cho_solve, solve_triangular, qr, jit, use_jax
from desc.optimize.utils import make_spd, chol_U_update, compute_jac_scale
import scipy.linalg


class OptimizerDerivative(ABC):
    """Abstract base class for hessians and jacobians used in the optimizer"""

    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    @abstractmethod
    def min_eig(self):
        pass

    @property
    @abstractmethod
    def is_pos_def(self):
        pass

    @property
    @abstractmethod
    def negative_curvature_direction(self):
        pass

    @abstractmethod
    def update(self, x_new, x_old, grad_new, grad_old):
        """Update the internal matrix A"""

    @abstractmethod
    def recompute(self, x):
        """Recompute the full correct internal matrix at the point x"""

    @abstractmethod
    def get_matrix(self):
        """Return the internal matrix A"""

    @abstractmethod
    def get_inverse(self):
        """Return the inverse of the internal matrix A^-1"""

    @abstractmethod
    def dot(self, x):
        """Compute dot(A,x)"""

    @abstractmethod
    def solve(self, b):
        """Solve A*x = b for x"""

    @abstractmethod
    def get_scale(self, prev_scale=None):
        """Compute scaling vector"""

    @abstractmethod
    def quadratic(self, u, v):
        """Evaluate quadratic form u.T * H * v"""


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
        self._shape = (n, n)
        self._is_pos_def = True
        self._min_eig = None
        self._negative_curvature_direction = None
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
        elif init_hess in [None, "auto"] and hessfun is not None:
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
            self._U = jnp.linalg.cholesky(init_hess).T
            self._initialized = True
            self._initialization = "init_hess"

    @property
    def shape(self):
        return self._shape

    @property
    def min_eig(self):
        """an estimate for the minimum eigenvalue of the matrix"""
        return self._min_eig

    @property
    def is_pos_def(self):
        """whether the matrix is positive definite"""
        return self._is_pos_def

    @property
    def negative_curvature_direction(self):
        """a direction corresponding to a negative eigenvalue"""
        return self._negative_curvature_direction

    def _auto_scale_init(self, delta_x, delta_grad):
        """Heuristic to scale matrix at first iteration"""
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
        """recompute the full matrix at the current point"""
        H = self._hessfun(x, *self._hessfun_args)
        H = make_spd(H, delta=self.min_curvature, tol=0.1)
        self._U = jnp.linalg.cholesky(H).T

    def update(self, x_new, x_old, grad_new, grad_old):
        """Update internal matrix"""
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
                return
        self._bfgs_update(delta_x, delta_grad)

    def _bfgs_update(self, delta_x, delta_grad):
        """rank 2 update using BFGS rule"""
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
        """get the current internal matrix"""
        return jnp.dot(self._U.T, self._U)

    def get_inverse(self):
        """get the inverse of the internal matrix"""
        return cho_solve((self._U, False), jnp.eye(self._n))

    def dot(self, x):
        """compute H@x"""
        return jnp.dot(self._U.T, jnp.dot(self._U, x))

    def solve(self, b):
        """solve Hx=b for x"""
        return cho_solve((self._U, False), b)

    def get_scale(self, prev_scale=None):
        """get diagonal scaling vector"""
        return compute_jac_scale(self._U, prev_scale)

    def quadratic(self, u, v):
        """evaluate quadratic form"""
        uu = jnp.dot(self._U, u)
        vv = jnp.dot(self._U, v)
        return jnp.dot(uu.T, vv)
