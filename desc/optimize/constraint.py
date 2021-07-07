import numpy as np
from desc.backend import jnp
import scipy.linalg
from termcolor import colored
from desc.io import IOAble


class LinearEqualityConstraint(IOAble):
    """Linear constraint for optimization

    solution vector x must satisfy Ax = b


    The method is to find a particular solution x0 that satisfies
    A x0 = b, and then to define x = x0 + dx where dx = Zy where Z
    is a representation for the nullspace of A. The optimization is
    then done over y instead of x.

    Parameters
    ----------
    A : ndarray
        Constraint matrix, shape(m,n) where m is the number of constraints
        and n is the dimension of x
    b : ndarray
        Constraint vector, shape(m,1)
    build : bool
        whether to compute null space and pseudoinverse now or wait until needed.

    """

    _io_attrs_ = ["_A", "_b", "_dimx"]

    def __init__(self, A, b, build=True):

        self._A = np.atleast_2d(A)
        self._b = np.atleast_1d(b)
        self._built = False
        self._Z = None
        self._Ainv = None
        self._dimx = self.A.shape[1]
        self._dimy = None
        if build:
            self.build()

    def build(self):
        """Builds linear constraint by factorizing A to get pseudoinverse and nullspace"""

        if self.built:
            return

        A = self.A
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        M, N = u.shape[0], vh.shape[1]
        K = min(M, N)
        rcond = np.finfo(A.dtype).eps * max(M, N)

        tol = np.amax(s) * rcond
        large = s > tol
        num = np.sum(large, dtype=int)
        Z = vh[num:, :].T.conj()

        uk = u[:, :K]
        vhk = vh[:K, :]
        s = np.divide(1, s, where=large, out=s)
        s[(~large,)] = 0
        Ainv = np.matmul(
            np.transpose(vhk), np.multiply(s[..., np.newaxis], np.transpose(uk))
        )

        self._Z = Z
        self._Ainv = Ainv
        self._dimy = Z.shape[1]
        self._x0 = Ainv.dot(self.b)

        self._built = True

    @property
    def built(self):
        return self.__dict__.setdefault("_built", False)

    @property
    def dimy(self):
        if not self.built:
            self._dimy = self.A.shape[1] - np.linalg.matrix_rank(self.A)
        return self._dimy

    @property
    def dimx(self):
        return self._dimx

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        self._b = b
        if not self.is_feasible(self.x0):
            self.x0 = self.Ainv.dot(self.b)

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, A):
        self._A = A
        self._built = False
        self.build()

    @property
    def Ainv(self):
        if not self.built:
            self.build()
        return self._Ainv

    @property
    def Z(self):
        if not self.built:
            self.build()
        return self._Z

    def compute_residual(self, x):
        """computes the residual of Ax=b"""
        res = self.A.dot(x) - self.b
        return res

    def is_feasible(self, x, tol=1e-8):
        """Checks that a given solution x is feasible, to with tolerance

        eg, norm(Ax-b) < tol
        """
        res = self.compute_residual(x)
        return np.linalg.norm(res) < tol

    def make_feasible(self, x):
        """make a vector x feasible by projecting onto the nullspace of A"""
        dx = x - self.x0
        y = self.Z.T.dot(dx)
        return self.x0 + self.Z.dot(y)

    @property
    def x0(self):
        """particular feasible solution"""
        if not self.built:
            self.build()
        return self._x0

    @x0.setter
    def x0(self, x0):
        if not self.is_feasible(x0):
            raise ValueError(colored("x0 is not feasible", "red"))
        self._x0 = x0

    def recover(self, y):
        """Recover the full solution x from the optimization variable y"""
        if not self.built:
            self.build()
        x = self.x0 + jnp.dot(self.Z, y)
        return jnp.squeeze(x)

    def project(self, x):
        """Project a full solution x to the optimization variable y"""
        if not self.built:
            self.build()
        dx = x - self.x0
        y = jnp.dot(self.Z.T, dx)
        return jnp.squeeze(y)
