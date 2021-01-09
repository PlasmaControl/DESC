import numpy as np
from desc.backend import jnp
import scipy.linalg
from termcolor import colored


class LinearEqualityConstraint():
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

    """

    def __init__(self, A, b, x0=None):

        self._A = np.atleast_2d(A)
        self._b = np.atleast_1d(b)
        self.remove_duplicates()

        self._Z = scipy.linalg.null_space(self._A)
        self._Ainv = np.linalg.pinv(self._A)
        self.m = self._A.shape[0]
        self.n = self._A.shape[1]
        self.k = self._Z.shape[1]

        if x0 is None:
            self._x0 = self._Ainv.dot(self._b)
        elif not self.is_feasible(x0):
            raise ValueError(colored("x0 is not feasible", 'red'))
        else:
            self._x0 = x0

    def __add__(self, other):
        if not isinstance(other, LinearEqualityConstraint):
            raise ValueError(colored(
                "cannot combine LinearConstraint with object of type {}".format(type(other)), 'red'))

        newA = np.vstack([self._A, other._A])
        newb = np.concatenate([self._b, other._b])
        return LinearEqualityConstraint(newA, newb)

    def remove_duplicates(self):
        """Delete duplicate constraints (ie duplicate rows in A,b)"""
        temp = np.hstack([self._A, self._b.reshape((-1, 1))])
        temp = np.unique(temp, axis=0)
        self._A = np.atleast_2d(temp[:, :-1])
        self._b = temp[:, -1].flatten()

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
        self._Ainv = np.linalg.pinv(A)
        self._Z = scipy.linalg.null_space(A)

    @property
    def Ainv(self):
        return self._Ainv

    @property
    def Z(self):
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
            raise ValueError(colored("x0 is not feasible", 'red'))
        self._x0 = x0

    def recover(self, y):
        """Recover the full solution x from the optimization variable y"""
        x = self._x0 + jnp.dot(self._Z, y)
        return jnp.squeeze(x)

    def project(self, x):
        """Project a full solution x to the optimization variable y"""
        dx = x - self._x0
        y = jnp.dot(self._Z.T, dx)
        return jnp.squeeze(y)
