import numpy as np
from desc.backend import jnp
import scipy.linalg
from termcolor import colored


class LinearEqualityConstraint:
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
    x0 : ndarray, optional
        initial feasible solution. If not provided, one will be generated
        from the least norm solution of Ax=b
    build : bool
        whether to compute null space and pseudoinverse now or wait until needed.

    """

    def __init__(self, A, b, x0=None, build=True):

        self._A = np.atleast_2d(A)
        self._b = np.atleast_1d(b)
        if x0 is not None and not self.is_feasible(x0):
            raise ValueError(colored("x0 is not feasible", "red"))
        self._x0 = x0
        self._built = False
        self._Z = None
        self._Ainv = None
        self._dimx = self._A.shape[1]
        self._dimy = None
        if build:
            self.build()

    def __add__(self, other):
        if not isinstance(other, LinearEqualityConstraint):
            raise ValueError(
                colored(
                    "cannot combine LinearConstraint with object of type {}".format(
                        type(other)
                    ),
                    "red",
                )
            )

        newA = np.vstack([self._A, other._A])
        newb = np.concatenate([self._b, other._b])
        return LinearEqualityConstraint(newA, newb)

    def build(self):
        """Builds linear constraint by factorizing A to get pseudoinverse and nullspace"""

        if self._built:
            return

        A = self._A
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
        s[~large] = 0
        Ainv = np.matmul(
            np.transpose(vhk), np.multiply(s[..., np.newaxis], np.transpose(uk))
        )

        self._Z = Z
        self._Ainv = Ainv
        self._dimy = self._Z.shape[1]
        if self._x0 is None:
            self._x0 = self._Ainv.dot(self._b)

        self._built = True

    def remove_duplicates(self):
        """Delete duplicate constraints (ie duplicate rows in A,b)"""
        temp = np.hstack([self._A, self._b.reshape((-1, 1))])
        temp = np.unique(temp, axis=0)
        self._A = np.atleast_2d(temp[:, :-1])
        self._b = temp[:, -1].flatten()

    @property
    def built(self):
        return self._built

    @property
    def dimy(self):
        if self._dimy is None:
            self._dimy = self._A.shape[1] - np.linalg.matrix_rank(self._A)
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
        if not self.is_feasible(self._x0):
            self._x0 = self._Ainv.dot(self._b)

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
            raise ValueError(
                "constraint must be build with constraint.build() to create Ainv"
            )
        return self._Ainv

    @property
    def Z(self):
        if not self.built:
            raise ValueError(
                "constraint must be build with constraint.build() to create Z"
            )
        return self._Z

    def compute_residual(self, x):
        """computes the residual of Ax=b"""
        res = self._A.dot(x) - self._b
        return res

    def is_feasible(self, x, tol=1e-8):
        """Checks that a given solution x is feasible, to with tolerance

        eg, norm(Ax-b) < tol
        """
        res = self.compute_residual(x)
        return np.linalg.norm(res) < tol

    def make_feasible(self, x):
        """make a vector x feasible by projecting onto the nullspace of A"""
        if not self.built:
            raise ValueError(
                "constraint must be build with constraint.build() before using"
            )
        dx = x - self._x0
        y = self._Z.T.dot(dx)
        return self._x0 + self._Z.dot(y)

    @property
    def x0(self):
        """particular feasible solution"""
        return self._x0

    @x0.setter
    def x0(self, x0):
        if not self.is_feasible(x0):
            raise ValueError(colored("x0 is not feasible", "red"))
        self._x0 = x0

    def recover(self, y):
        """Recover the full solution x from the optimization variable y"""
        if not self.built:
            raise ValueError(
                "constraint must be build with constraint.build() before using"
            )
        x = self._x0 + jnp.dot(self._Z, y)
        return jnp.squeeze(x)

    def project(self, x):
        """Project a full solution x to the optimization variable y"""
        if not self.built:
            raise ValueError(
                "constraint must be build with constraint.build() before using"
            )
        dx = x - self._x0
        y = jnp.dot(self._Z.T, dx)
        return jnp.squeeze(y)
