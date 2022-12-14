"""Wrapper classes for JAX automatic differentiation and finite differences."""

from abc import ABC, abstractmethod

import numpy as np
from termcolor import colored

from desc.backend import jnp, put, use_jax

if use_jax:
    import jax


class _Derivative(ABC):
    """_Derivative is an abstract base class for derivative matrix calculations.

    Parameters
    ----------
    fun : callable
        Function to be differentiated.
    argnums : int, optional
        Specifies which positional argument to differentiate with respect to

    """

    @abstractmethod
    def __init__(self, fun, argnum=0, mode=None, **kwargs):
        pass

    @abstractmethod
    def compute(self, *args):
        """Compute the derivative matrix.

        Parameters
        ----------
        *args : list
            Arguments of the objective function where the derivative is to be
            evaluated at.

        Returns
        -------
        D : ndarray of float
            derivative of f evaluated at x, where f is the output of the function
            fun and x is the input argument at position argnum. Exact shape and meaning
            will depend on "mode"

        """

    @property
    def fun(self):
        """Callable : function being differentiated."""
        return self._fun

    @fun.setter
    def fun(self, fun):
        self._fun = fun

    @property
    def argnum(self):
        """Integer : argument being differentiated with respect to."""
        return self._argnum

    @argnum.setter
    def argnum(self, argnum):
        self._argnum = argnum

    @property
    def mode(self):
        """String : the kind of derivative being computed (eg ``'grad'``)."""
        return self._mode

    def __call__(self, *args):
        """Compute the derivative matrix.

        Parameters
        ----------
        *args : list
            Arguments of the objective function where the derivative is to be
            evaluated at.

        Returns
        -------
        D : ndarray of float
            derivative of f evaluated at x, where f is the output of the function
            fun and x is the input argument at position argnum. Exact shape and meaning
            will depend on "mode"

        """
        return self.compute(*args)

    def __repr__(self):
        """String form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (fun={}, argnum={}, mode={})".format(
                repr(self.fun), self.argnum, self.mode
            )
        )


class AutoDiffDerivative(_Derivative):
    """Computes derivatives using automatic differentiation with JAX.

    Parameters
    ----------
    fun : callable
        Function to be differentiated.
    argnum : int, optional
        Specifies which positional argument to differentiate with respect to
    mode : str, optional
        Automatic differentiation mode.
        One of ``'fwd'`` (forward mode jacobian), ``'rev'`` (reverse mode jacobian),
        ``'grad'`` (gradient of a scalar function),
        ``'hess'`` (hessian of a scalar function),
        or ``'jvp'`` (jacobian vector product)
        Default = ``'fwd'``

    Raises
    ------
    ValueError, if mode is not supported

    """

    def __init__(self, fun, argnum=0, mode="fwd", **kwargs):

        self._fun = fun
        self._argnum = argnum

        self._set_mode(mode)

    def compute(self, *args):
        """Compute the derivative matrix.

        Parameters
        ----------
        *args : list
            Arguments of the objective function where the derivative is to be
            evaluated at.

        Returns
        -------
        D : ndarray of float
            derivative of f evaluated at x, where f is the output of the function
            fun and x is the input argument at position argnum. Exact shape and meaning
            will depend on "mode"

        """
        return self._compute(*args)

    @classmethod
    def compute_jvp(cls, fun, argnum, v, *args):
        """Compute df/dx*v.

        Parameters
        ----------
        fun : callable
            function to differentiate
        argnum : int or tuple
            arguments to differentiate with respect to
        v : array-like or tuple of array-like
            tangent vectors. Should be one for each argnum
        args : tuple
            arguments passed to f

        Returns
        -------
        jvp : array-like
            jacobian times vectors v, summed over different argnums

        """
        tangents = list(nested_zeros_like(args))
        if jnp.isscalar(argnum):
            argnum = (argnum,)
        else:
            argnum = tuple(argnum)
        v = (v,) if not isinstance(v, tuple) else v

        for i, vi in enumerate(v):
            tangents[argnum[i]] = vi
        y, u = jax.jvp(fun, args, tuple(tangents))
        return u

    @classmethod
    def compute_jvp2(cls, fun, argnum1, argnum2, v1, v2, *args):
        """Compute d^2f/dx^2*v1*v2.

        Parameters
        ----------
        fun : callable
            function to differentiate
        argnum1, argnum2 : int or tuple of int
            arguments to differentiate with respect to. First entry corresponds to v1,
            second to v2
        v1,v2 : array-like or tuple of array-like
            tangent vectors. Should be one for each argnum
        args : tuple
            arguments passed to f

        Returns
        -------
        jvp2 : array-like
            second derivative times vectors v1, v2, summed over different argnums

        """
        if np.isscalar(argnum1):
            v1 = (v1,) if not isinstance(v1, tuple) else v1
            argnum1 = (argnum1,)
        else:
            v1 = tuple(v1)

        if np.isscalar(argnum2):
            argnum2 = (argnum2 + 1,)
            v2 = (v2,) if not isinstance(v2, tuple) else v2
        else:
            argnum2 = tuple([i + 1 for i in argnum2])
            v2 = tuple(v2)

        dfdx = lambda dx1, *args: cls.compute_jvp(fun, argnum1, dx1, *args)
        d2fdx2 = lambda dx1, dx2: cls.compute_jvp(dfdx, argnum2, dx2, dx1, *args)
        return d2fdx2(v1, v2)

    @classmethod
    def compute_jvp3(cls, fun, argnum1, argnum2, argnum3, v1, v2, v3, *args):
        """Compute d^3f/dx^3*v1*v2*v3.

        Parameters
        ----------
        fun : callable
            function to differentiate
        argnum1, argnum2, argnum3 : int or tuple of int
            arguments to differentiate with respect to. First entry corresponds to v1,
            second to v2 etc
        v1,v2,v3 : array-like or tuple of array-like
            tangent vectors. Should be one for each argnum
        args : tuple
            arguments passed to f

        Returns
        -------
        jvp3 : array-like
            third derivative times vectors v2, v3, v3, summed over different argnums

        """
        if np.isscalar(argnum1):
            v1 = (v1,) if not isinstance(v1, tuple) else v1
            argnum1 = (argnum1,)
        else:
            v1 = tuple(v1)

        if np.isscalar(argnum2):
            argnum2 = (argnum2 + 1,)
            v2 = (v2,) if not isinstance(v2, tuple) else v2
        else:
            argnum2 = tuple([i + 1 for i in argnum2])
            v2 = tuple(v2)

        if np.isscalar(argnum3):
            argnum3 = (argnum3 + 2,)
            v3 = (v3,) if not isinstance(v3, tuple) else v3
        else:
            argnum3 = tuple([i + 2 for i in argnum3])
            v3 = tuple(v3)

        dfdx = lambda dx1, *args: cls.compute_jvp(fun, argnum1, dx1, *args)
        d2fdx2 = lambda dx1, dx2, *args: cls.compute_jvp(dfdx, argnum2, dx2, dx1, *args)
        d3fdx3 = lambda dx1, dx2, dx3: cls.compute_jvp(
            d2fdx2, argnum3, dx3, dx2, dx1, *args
        )
        return d3fdx3(v1, v2, v3)

    def _compute_jvp(self, v, *args):
        return self.compute_jvp(self._fun, self.argnum, v, *args)

    def _set_mode(self, mode) -> None:
        if mode not in ["fwd", "rev", "grad", "hess", "jvp"]:
            raise ValueError(
                colored("invalid mode option for automatic differentiation", "red")
            )

        self._mode = mode
        if self._mode == "fwd":
            self._compute = jax.jacfwd(self._fun, self._argnum)
        elif self._mode == "rev":
            self._compute = jax.jacrev(self._fun, self._argnum)
        elif self._mode == "grad":
            self._compute = jax.grad(self._fun, self._argnum)
        elif self._mode == "hess":
            self._compute = jax.hessian(self._fun, self._argnum)
        elif self._mode == "jvp":
            self._compute = self._compute_jvp


class FiniteDiffDerivative(_Derivative):
    """Computes derivatives using 2nd order centered finite differences.

    Parameters
    ----------
    fun : callable
        Function to be differentiated.
    argnum : int, optional
        Specifies which positional argument to differentiate with respect to
    mode : str, optional
        Automatic differentiation mode.
        One of ``'fwd'`` (forward mode jacobian), ``'rev'`` (reverse mode jacobian),
        ``'grad'`` (gradient of a scalar function),
        ``'hess'`` (hessian of a scalar function),
        or ``'jvp'`` (jacobian vector product)
        Default = ``'fwd'``
    rel_step : float, optional
        Relative step size: dx = max(1, abs(x))*rel_step
        Default = 1e-3

    """

    def __init__(self, fun, argnum=0, mode="fwd", rel_step=1e-3, **kwargs):

        self._fun = fun
        self._argnum = argnum
        self.rel_step = rel_step
        self._set_mode(mode)

    def _compute_hessian(self, *args):
        """Compute the hessian matrix using 2nd order centered finite differences.

        Parameters
        ----------
        *args : list
            Arguments of the objective function where the derivative is to be
            evaluated at.

        Returns
        -------
        H : ndarray of float, shape(len(x),len(x))
            d^2f/dx^2, where f is the output of the function fun and x is the input
            argument at position argnum.

        """

        def f(x):
            tempargs = args[0 : self._argnum] + (x,) + args[self._argnum + 1 :]
            return self._fun(*tempargs)

        x = np.atleast_1d(args[self._argnum])
        n = len(x)
        fx = f(x)
        h = np.maximum(1.0, np.abs(x)) * self.rel_step
        ee = np.diag(h)
        hess = np.outer(h, h)

        for i in range(n):
            eei = ee[i, :]
            hess[i, i] = (f(x + 2 * eei) - 2 * fx + f(x - 2 * eei)) / (4.0 * hess[i, i])
            for j in range(i + 1, n):
                eej = ee[j, :]
                hess[i, j] = (
                    f(x + eei + eej)
                    - f(x + eei - eej)
                    - f(x - eei + eej)
                    + f(x - eei - eej)
                ) / (4.0 * hess[j, i])
                hess[j, i] = hess[i, j]

        return hess

    def _compute_grad_or_jac(self, *args):
        """Compute the gradient or jacobian matrix (ie, first derivative).

        Parameters
        ----------
        *args : list
            Arguments of the objective function where the derivative is to be
            evaluated at.

        Returns
        -------
        J : ndarray of float, shape(len(f),len(x))
            df/dx, where f is the output of the function fun and x is the input
            argument at position argnum.

        """

        def f(x):
            tempargs = args[0 : self._argnum] + (x,) + args[self._argnum + 1 :]
            return self._fun(*tempargs)

        x0 = np.atleast_1d(args[self._argnum])
        f0 = f(x0)
        m = f0.size
        n = x0.size
        J = np.zeros((m, n))
        h = np.maximum(1.0, np.abs(x0)) * self.rel_step
        h_vecs = np.diag(np.atleast_1d(h))
        for i in range(n):
            x1 = x0 - h_vecs[i]
            x2 = x0 + h_vecs[i]
            dx = x2[i] - x1[i]
            f1 = f(x1)
            f2 = f(x2)
            df = f2 - f1
            dfdx = df / dx
            J = put(J.T, i, dfdx.flatten()).T
        if m == 1:
            J = np.ravel(J)
        return J

    @classmethod
    def compute_jvp(cls, fun, argnum, v, *args, **kwargs):
        """Compute df/dx*v.

        Parameters
        ----------
        fun : callable
            function to differentiate
        argnum : int or tuple
            arguments to differentiate with respect to
        v : array-like or tuple of array-like
            tangent vectors. Should be one for each argnum
        args : tuple
            arguments passed to f

        Returns
        -------
        jvp : array-like
            jacobian times vectors v, summed over different argnums

        """
        rel_step = kwargs.get("rel_step", 1e-3)

        if np.isscalar(argnum):
            nargs = 1
            argnum = (argnum,)
        else:
            nargs = len(argnum)
        v = (v,) if not isinstance(v, tuple) else v

        f = np.array(
            [
                cls._compute_jvp_1arg(fun, argnum[i], v[i], *args, rel_step=rel_step)
                for i in range(nargs)
            ]
        )
        return np.sum(f, axis=0)

    @classmethod
    def compute_jvp2(cls, fun, argnum1, argnum2, v1, v2, *args):
        """Compute d^2f/dx^2*v1*v2.

        Parameters
        ----------
        fun : callable
            function to differentiate
        argnum1, argnum2 : int or tuple of int
            arguments to differentiate with respect to. First entry corresponds to v1,
            second to v2
        v1,v2 : array-like or tuple of array-like
            tangent vectors. Should be one for each argnum
        args : tuple
            arguments passed to f

        Returns
        -------
        jvp2 : array-like
            second derivative times vectors v1, v2, summed over different argnums

        """
        if np.isscalar(argnum1):
            v1 = (v1,) if not isinstance(v1, tuple) else v1
            argnum1 = (argnum1,)
        else:
            v1 = tuple(v1)

        if np.isscalar(argnum2):
            argnum2 = (argnum2 + 1,)
            v2 = (v2,) if not isinstance(v2, tuple) else v2
        else:
            argnum2 = tuple([i + 1 for i in argnum2])
            v2 = tuple(v2)

        dfdx = lambda dx1, *args: cls.compute_jvp(fun, argnum1, dx1, *args)
        d2fdx2 = lambda dx1, dx2: cls.compute_jvp(dfdx, argnum2, dx2, dx1, *args)
        return d2fdx2(v1, v2)

    @classmethod
    def compute_jvp3(cls, fun, argnum1, argnum2, argnum3, v1, v2, v3, *args):
        """Compute d^3f/dx^3*v1*v2*v3.

        Parameters
        ----------
        fun : callable
            function to differentiate
        argnum1, argnum2, argnum3 : int or tuple of int
            arguments to differentiate with respect to. First entry corresponds to v1,
            second to v2 etc
        v1,v2,v3 : array-like or tuple of array-like
            tangent vectors. Should be one for each argnum
        args : tuple
            arguments passed to f

        Returns
        -------
        jvp3 : array-like
            third derivative times vectors v2, v3, v3, summed over different argnums

        """
        if np.isscalar(argnum1):
            v1 = (v1,) if not isinstance(v1, tuple) else v1
            argnum1 = (argnum1,)
        else:
            v1 = tuple(v1)

        if np.isscalar(argnum2):
            argnum2 = (argnum2 + 1,)
            v2 = (v2,) if not isinstance(v2, tuple) else v2
        else:
            argnum2 = tuple([i + 1 for i in argnum2])
            v2 = tuple(v2)

        if np.isscalar(argnum3):
            argnum3 = (argnum3 + 2,)
            v3 = (v3,) if not isinstance(v3, tuple) else v3
        else:
            argnum3 = tuple([i + 2 for i in argnum3])
            v3 = tuple(v3)

        dfdx = lambda dx1, *args: cls.compute_jvp(fun, argnum1, dx1, *args)
        d2fdx2 = lambda dx1, dx2, *args: cls.compute_jvp(dfdx, argnum2, dx2, dx1, *args)
        d3fdx3 = lambda dx1, dx2, dx3: cls.compute_jvp(
            d2fdx2, argnum3, dx3, dx2, dx1, *args
        )
        return d3fdx3(v1, v2, v3)

    def _compute_jvp(self, v, *args):
        return self.compute_jvp(
            self._fun, self._argnum, v, *args, rel_step=self.rel_step
        )

    @classmethod
    def _compute_jvp_1arg(cls, fun, argnum, v, *args, **kwargs):
        """Compute a jvp wrt a single argument."""
        rel_step = kwargs.get("rel_step", 1e-3)
        normv = np.linalg.norm(v)
        if normv != 0:
            vh = v / normv
        else:
            vh = v
        x = args[argnum]

        def f(x):
            tempargs = args[0:argnum] + (x,) + args[argnum + 1 :]
            return fun(*tempargs)

        h = rel_step
        df = (f(x + h * vh) - f(x - h * vh)) / (2 * h)
        return df * normv

    def _set_mode(self, mode):
        if mode not in ["fwd", "rev", "grad", "hess", "jvp"]:
            raise ValueError(
                colored(
                    "invalid mode option for finite difference differentiation", "red"
                )
            )

        self._mode = mode
        if self._mode == "fwd":
            self._compute = self._compute_grad_or_jac
        elif self._mode == "rev":
            self._compute = self._compute_grad_or_jac
        elif self._mode == "grad":
            self._compute = self._compute_grad_or_jac
        elif self._mode == "hess":
            self._compute = self._compute_hessian
        elif self._mode == "jvp":
            self._compute = self._compute_jvp

    def compute(self, *args):
        """Compute the derivative matrix.

        Parameters
        ----------
        *args : list
            Arguments of the objective function where the derivative is to be
            evaluated at.

        Returns
        -------
        D : ndarray of float
            derivative of f evaluated at x, where f is the output of the function
            fun and x is the input argument at position argnum. Exact shape and meaning
            will depend on "mode"

        """
        return self._compute(*args)


def nested_zeros_like(x):
    """Get a nested pytree of zeros like a given pytree."""
    if x is None:
        return None
    if jnp.isscalar(x):
        return 0.0
    if isinstance(x, tuple):
        return tuple([nested_zeros_like(a) for a in x])
    if isinstance(x, list):
        return list([nested_zeros_like(a) for a in x])
    return jnp.zeros_like(x)


if use_jax:
    Derivative = AutoDiffDerivative
else:
    Derivative = FiniteDiffDerivative
