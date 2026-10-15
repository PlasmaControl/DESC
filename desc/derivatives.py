"""Wrapper classes for JAX automatic differentiation and finite differences."""

from abc import ABC, abstractmethod

import numpy as np

from desc.backend import jnp, use_jax
from desc.utils import ensure_tuple

if use_jax:
    import jax

    from desc.batching import jacfwd_chunked, jacrev_chunked


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
    def compute(self, *args, **kwargs):
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

    def __call__(self, *args, **kwargs):
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
        return self.compute(*args, **kwargs)

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
        One of ``'fwd'`` (forward mode Jacobian), ``'rev'`` (reverse mode Jacobian),
        ``'grad'`` (gradient of a scalar function),
        ``'hess'`` (Hessian of a scalar function),
        or ``'jvp'`` (Jacobian vector product)
        Default = ``'fwd'``

    Raises
    ------
    ValueError, if mode is not supported

    """

    def __init__(self, fun, argnum=0, mode="fwd", chunk_size=None, **kwargs):

        self._fun = fun
        self._argnum = argnum
        self._chunk_size = chunk_size
        self._set_mode(mode)

    def compute(self, *args, **kwargs):
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
        return self._compute(*args, **kwargs)

    @classmethod
    def compute_vjp(cls, fun, argnum, v, *args, **kwargs):
        """Compute v.T * df/dx.

        Parameters
        ----------
        fun : callable
            function to differentiate
        argnum : int or tuple
            arguments to differentiate with respect to
        v : array-like or tuple of array-like
            tangent vectors. Should be one for each output of fun.
        args : tuple
            arguments passed to fun
        kwargs : dict
            keyword arguments passed to fun

        Returns
        -------
        vjp : array-like
            Vector v times Jacobian, summed over different argnums

        """
        assert jnp.isscalar(argnum), "vjp for multiple args not currently supported"
        _ = kwargs.pop("rel_step", None)  # unused by autodiff

        def _fun(*args):
            return v.T @ fun(*args, **kwargs)

        return jax.grad(_fun, argnum)(*args)

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
            arguments passed to fun
        kwargs : dict
            keyword arguments passed to fun

        Returns
        -------
        jvp : array-like
            Jacobian times vectors v, summed over different argnums

        """
        _ = kwargs.pop("rel_step", None)  # unused by autodiff
        argnum = (argnum,) if jnp.isscalar(argnum) else tuple(argnum)
        v = ensure_tuple(v)

        def _fun(*x):
            _args = list(args)
            for i, xi in zip(argnum, x):
                _args[i] = xi
            return fun(*_args, **kwargs)

        y, u = jax.jvp(_fun, tuple(args[i] for i in argnum), v)
        return u

    @classmethod
    def compute_jvp2(cls, fun, argnum1, argnum2, v1, v2, *args, **kwargs):
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
            arguments passed to fun
        kwargs : dict
            keyword arguments passed to fun

        Returns
        -------
        jvp2 : array-like
            second derivative times vectors v1, v2, summed over different argnums

        """
        if np.isscalar(argnum1):
            v1 = ensure_tuple(v1)
            argnum1 = (argnum1,)
        else:
            v1 = tuple(v1)

        if np.isscalar(argnum2):
            argnum2 = (argnum2 + 1,)
            v2 = ensure_tuple(v2)
        else:
            argnum2 = tuple([i + 1 for i in argnum2])
            v2 = tuple(v2)

        dfdx = lambda dx1, *args: cls.compute_jvp(fun, argnum1, dx1, *args, **kwargs)
        d2fdx2 = lambda dx1, dx2: cls.compute_jvp(
            dfdx, argnum2, dx2, dx1, *args, **kwargs
        )
        return d2fdx2(v1, v2)

    @classmethod
    def compute_jvp3(cls, fun, argnum1, argnum2, argnum3, v1, v2, v3, *args, **kwargs):
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
            arguments passed to fun
        kwargs : dict
            keyword arguments passed to fun

        Returns
        -------
        jvp3 : array-like
            third derivative times vectors v2, v3, v3, summed over different argnums

        """
        if np.isscalar(argnum1):
            v1 = ensure_tuple(v1)
            argnum1 = (argnum1,)
        else:
            v1 = tuple(v1)

        if np.isscalar(argnum2):
            argnum2 = (argnum2 + 1,)
            v2 = ensure_tuple(v2)
        else:
            argnum2 = tuple([i + 1 for i in argnum2])
            v2 = tuple(v2)

        if np.isscalar(argnum3):
            argnum3 = (argnum3 + 2,)
            v3 = ensure_tuple(v3)
        else:
            argnum3 = tuple([i + 2 for i in argnum3])
            v3 = tuple(v3)

        dfdx = lambda dx1, *args: cls.compute_jvp(fun, argnum1, dx1, *args, **kwargs)
        d2fdx2 = lambda dx1, dx2, *args: cls.compute_jvp(
            dfdx, argnum2, dx2, dx1, *args, **kwargs
        )
        d3fdx3 = lambda dx1, dx2, dx3: cls.compute_jvp(
            d2fdx2, argnum3, dx3, dx2, dx1, *args, **kwargs
        )
        return d3fdx3(v1, v2, v3)

    def _compute_jvp(self, v, *args, **kwargs):
        return self.compute_jvp(self._fun, self.argnum, v, *args, **kwargs)

    def _set_mode(self, mode) -> None:
        if mode not in ["fwd", "rev", "grad", "hess", "jvp"]:
            raise ValueError("invalid mode option for automatic differentiation")

        self._mode = mode
        if self._mode == "fwd":
            self._compute = jacfwd_chunked(
                self._fun, self._argnum, chunk_size=self._chunk_size
            )
        elif self._mode == "rev":
            self._compute = jacrev_chunked(
                self._fun, self._argnum, chunk_size=self._chunk_size
            )
        elif self._mode == "grad":
            self._compute = jax.grad(self._fun, self._argnum)
        elif self._mode == "hess":
            self._compute = jacfwd_chunked(
                jacrev_chunked(self._fun, self._argnum, chunk_size=self._chunk_size),
                self._argnum,
                chunk_size=self._chunk_size,
            )
        elif self._mode == "jvp":
            self._compute = self._compute_jvp


Derivative = AutoDiffDerivative
