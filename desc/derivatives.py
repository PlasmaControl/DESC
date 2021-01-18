import numpy as np
from abc import ABC, abstractmethod
from termcolor import colored

from desc.backend import use_jax, put, jnp

if use_jax:
    import jax


class _Derivative(ABC):
    """_Derivative is an abstract base class for derivative matrix calculations"""

    @abstractmethod
    def __init__(self, fun: callable, argnum: int = 0, **kwargs) -> None:
        """Initializes a Derivative object

        Parameters
        ----------
        fun : callable
            Function to be differentiated.
        argnums : int, optional
            Specifies which positional argument to differentiate with respect to

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def compute(self, *args):
        """Computes the derivative matrix

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
        pass

    @property
    def fun(self) -> callable:
        return self._fun

    @fun.setter
    def fun(self, fun: callable) -> None:
        self._fun = fun

    @property
    def argnum(self) -> int:
        return self._argnum

    @argnum.setter
    def argnum(self, argnum: int) -> None:
        self._argnum = argnum

    def __call__(self, *args):
        return self.compute(*args)


class AutoDiffDerivative(_Derivative):
    """Computes derivatives using automatic differentiation with JAX"""

    def __init__(
        self, fun: callable, argnum: int = 0, mode: str = "fwd", **kwargs
    ) -> None:
        """Initializes an AutoDiffDerivative

        Parameters
        ----------
        fun : callable
            Function to be differentiated.
        argnum : int, optional
            Specifies which positional argument to differentiate with respect to
        mode : str, optional
            Automatic differentiation mode.
            One of 'fwd' (forward mode jacobian), 'rev' (reverse mode jacobian),
            'grad' (gradient of a scalar function), 'hess' (hessian of a scalar function),
            or 'jvp' (jacobian vector product)
            Default = 'fwd'

        Raises
        ------
        ValueError, if mode is not supported

        Returns
        -------
        None

        """
        self._fun = fun
        self._argnum = argnum

        if ("block_size" in kwargs or "num_blocks" in kwargs) and mode in [
            "fwd",
            "rev",
            "hess",
        ]:
            self._init_blocks(mode, kwargs)
        else:
            self.mode = mode

    def _init_blocks(self, mode, kwargs):

        if mode in ["fwd", "rev"]:
            self._block_fun = self._fun
            self._mode = "blocked-rev"
        elif mode in ["hess"]:
            self._block_fun = jax.grad(self._fun, self._argnum)
            self._mode = "blocked-hess"

        try:
            self.shape = kwargs["shape"]
        except KeyError as e:
            raise ValueError(
                "Block derivative requires the shape of the derivative matrix to be specified with the 'shape' keyword argument"
            ) from e

        N, M = self.shape
        block_size = kwargs.get("block_size", None)
        num_blocks = kwargs.get("num_blocks", None)
        # TODO: some sort of "auto" sizing option by checking available memory
        if block_size is not None and num_blocks is not None:
            raise ValueError(
                colored("can specify either block_size or num_blocks, not both", "red")
            )

        elif block_size is None and num_blocks is None:
            self._block_size = N
            self._num_blocks = 1
        elif block_size is not None:
            self._block_size = block_size
            self._num_blocks = np.ceil(N / block_size).astype(int)
        else:
            self._num_blocks = num_blocks
            self._block_size = np.ceil(N / num_blocks).astype(int)

        devices = kwargs.get("devices", None)
        if type(devices) in [list, tuple]:
            self._devices = devices
        else:
            self._devices = [devices]

        self._use_jit = kwargs.get("use_jit", True)
        self._f_blocks = []
        self._jac_blocks = []

        for i in range(self._num_blocks):
            # need the i=i in the lambda signature, otherwise i is scoped to
            # the loop and get overwritten, making each function compute the same subset
            self._f_blocks.append(
                lambda *args, i=i: self._block_fun(*args)[
                    i * self._block_size : (i + 1) * self._block_size
                ]
            )
            # need to use jacrev here to actually get memory savings
            # (plus, these blocks should be wide and short)
            if self._use_jit:
                self._jac_blocks.append(
                    jax.jit(
                        jax.jacrev(self._f_blocks[i], self._argnum),
                        device=self._devices[i % len(self._devices)],
                    )
                )
            else:
                self._jac_blocks.append(jax.jacrev(self._f_blocks[i], self._argnum))

        self._compute = self._compute_blocks

    def _compute_blocks(self, *args):
        return jnp.vstack([jac(*args) for jac in self._jac_blocks])

    def compute(self, *args):
        """Computes the derivative matrix

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
        return self._compute(*args)

    def _compute_jvp(self, v, *args):
        tangents = list(args)
        tangents[self.argnum] = v
        y = jax.jvp(self._fun, args, tangents)
        return y

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
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
    """Computes derivatives using 2nd order centered finite differences"""

    def __init__(
        self, fun: callable, argnum: int = 0, mode: str = "fwd", rel_step: float = 1e-3
    ) -> None:
        """Initializes a FiniteDiffDerivative

        Parameters
        ----------
        fun : callable
            Function to be differentiated.
        argnum : int, optional
            Specifies which positional argument to differentiate with respect to
        mode : str, optional
            Automatic differentiation mode.
            One of 'fwd' (forward mode jacobian), 'rev' (reverse mode jacobian),
            'grad' (gradient of a scalar function), 'hess' (hessian of a scalar function),
            or 'jvp' (jacobian vector product)
            Default = 'fwd'
        rel_step : float, optional
            Relative step size: dx = max(1, abs(x))*rel_step
            Default = 1e-3

        Returns
        -------
        None

        """
        self._fun = fun
        self._argnum = argnum
        self.rel_step = rel_step
        self.mode = mode

    def _compute_hessian(self, *args):
        """Computes the hessian matrix using 2nd order centered finite differences.

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
        dtype = fx.dtype
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
        """Computes the gradient or jacobian matrix (ie, first derivative)

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

    def _compute_jvp(self, v, *args):

        normv = np.linalg.norm(v)
        vh = v / normv
        x = args[self.argnum]

        def f(x):
            tempargs = args[0 : self._argnum] + (x,) + args[self._argnum + 1 :]
            return self._fun(*tempargs)

        h = self.rel_step
        df = (f(x + h * vh) - f(x - h * vh)) / (2 * h)
        return df * normv

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
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
        """Computes the derivative matrix

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
        return self._compute(*args)


if use_jax:
    Derivative = AutoDiffDerivative
else:
    Derivative = FiniteDiffDerivative


# TODO: make this thing an option for autodiff, probably just by adding a dict of kwargs?


class BlockJacobian:
    """Computes a jacobian matrix in smaller blocks.
    Takes a large jacobian and splits it into smaller blocks
    (row-wise) for easier computation, possibly allowing each
    block to be computed independently on different devices in
    parallel. Also helps to reduce memory load, allowing
    computation of larger jacobians on limited memory GPUs
    Parameters
    ----------
    fun : callable
        function to take jacobian of
    N : int
        dimension of fun(x)
    M : int
        dimension of x
    block_size : int
        size (number of rows) of each block.
        the last block may be smaller depending on N and block_size
    num_blocks : int
        number of blocks (only used if block size
        is not given).
    devices : list or tuple of jax.device
        list of jax devices to use.
        Blocks will be split evenly across them.
    usejit : bool
        whether to apply JIT compilation. Generally
        only worth it if jacobian will be called many times
    Returns
    -------
    jac_fun : callable
        object that computes the jacobian of fun.
    """

    def __init__(
        self, fun, N, M, block_size=None, num_blocks=None, devices=None, usejit=False
    ):

        self.fun = fun
        self.N = N
        self.M = M

        # could probably add some fancier logic here to look at M as well when deciding how
        # to split blocks? Though we can't really split the jacobian columnwise without a lot
        # of surgery on the objective function
        if block_size is not None and num_blocks is not None:
            raise ValueError(
                colored("can specify either block_size or num_blocks, not both", "red")
            )

        elif block_size is None and num_blocks is None:
            self.block_size = N
            self.num_blocks = 1
        elif block_size is not None:
            self.block_size = block_size
            self.num_blocks = np.ceil(N / block_size).astype(int)
        else:
            self.num_blocks = num_blocks
            self.block_size = np.ceil(N / num_blocks).astype(int)

        if type(devices) in [list, tuple]:
            self.devices = devices
        else:
            self.devices = [devices]

        self.usejit = usejit
        self.f_blocks = []
        self.jac_blocks = []

        for i in range(self.num_blocks):
            # need the i=i in the lambda signature, otherwise i is scoped to
            # the loop and get overwritten, making each function compute the same subset
            self.f_blocks.append(
                lambda x, *args, i=i: self.fun(x, *args)[
                    i * self.block_size : (i + 1) * self.block_size
                ]
            )
            # need to use jacrev here to actually get memory savings
            # (plus, these blocks should be wide and short)
            if self.usejit:
                self.jac_blocks.append(
                    jit(
                        jacrev(self.f_blocks[i]),
                        device=self.devices[i % len(self.devices)],
                    )
                )
            else:
                self.jac_blocks.append(jacrev(self.f_blocks[i]))

    def compute(self, x, *args):

        return np.vstack([jac(x, *args) for jac in self.jac_blocks])
