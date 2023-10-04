"""Backend functions for DESC, with options for JAX or regular numpy."""

import os
import warnings

import numpy as np
from termcolor import colored

import desc
from desc import config as desc_config
from desc import set_device

if os.environ.get("DESC_BACKEND") == "numpy":
    jnp = np
    use_jax = False
    set_device(kind="cpu")
    print(
        "DESC version {}, using numpy backend, version={}, dtype={}".format(
            desc.__version__, np.__version__, np.linspace(0, 1).dtype
        )
    )
else:
    if desc_config.get("device") is None:
        set_device("cpu")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jax
            import jax.numpy as jnp
            import jaxlib
            from jax.config import config as jax_config

            jax_config.update("jax_enable_x64", True)
            if desc_config.get("kind") == "gpu" and len(jax.devices("gpu")) == 0:
                warnings.warn(
                    "JAX failed to detect GPU, are you sure you "
                    + "installed JAX with GPU support?"
                )
                set_device("cpu")
            x = jnp.linspace(0, 5)
            y = jnp.exp(x)
        use_jax = True
        print(
            f"DESC version {desc.__version__},"
            + f"using JAX backend, jax version={jax.__version__}, "
            + f"jaxlib version={jaxlib.__version__}, dtype={y.dtype}"
        )
        del x, y
    except ModuleNotFoundError:
        jnp = np
        x = jnp.linspace(0, 5)
        y = jnp.exp(x)
        use_jax = False
        set_device(kind="cpu")
        warnings.warn(colored("Failed to load JAX", "red"))
        print(
            "DESC version {}, using NumPy backend, version={}, dtype={}".format(
                desc.__version__, np.__version__, y.dtype
            )
        )
print(
    "Using device: {}, with {:.2f} GB available memory".format(
        desc_config.get("device"), desc_config.get("avail_mem")
    )
)

if use_jax:  # noqa: C901 - FIXME: simplify this, define globally and then assign?
    jit = jax.jit
    fori_loop = jax.lax.fori_loop
    cond = jax.lax.cond
    switch = jax.lax.switch
    while_loop = jax.lax.while_loop
    vmap = jax.vmap
    bincount = jnp.bincount
    from jax import custom_jvp
    from jax.experimental.ode import odeint
    from jax.scipy.linalg import block_diag, cho_factor, cho_solve, qr, solve_triangular
    from jax.scipy.special import gammaln, logsumexp
    from jax.tree_util import register_pytree_node

    def put(arr, inds, vals):
        """Functional interface for array "fancy indexing".

        Provides a way to do arr[inds] = vals in a way that works with JAX.

        Parameters
        ----------
        arr : array-like
            Array to populate
        inds : array-like of int
            Indices to populate
        vals : array-like
            Values to insert

        Returns
        -------
        arr : array-like
            Input array with vals inserted at inds.

        """
        return jnp.asarray(arr).at[inds].set(vals)

    def sign(x):
        """Sign function, but returns 1 for x==0.

        Parameters
        ----------
        x : array-like
            array of input values

        Returns
        -------
        y : array-like
            1 where x>=0, -1 where x<0

        """
        x = jnp.atleast_1d(x)
        y = jnp.where(x == 0, 1, jnp.sign(x))
        return y

else:
    jit = lambda func, *args, **kwargs: func
    from scipy.integrate import odeint  # noqa: F401
    from scipy.linalg import (  # noqa: F401
        block_diag,
        cho_factor,
        cho_solve,
        qr,
        solve_triangular,
    )
    from scipy.special import gammaln, logsumexp  # noqa: F401

    def register_pytree_node(foo, *args):
        """Dummy decorator for non-jax pytrees."""
        return foo

    def put(arr, inds, vals):
        """Functional interface for array "fancy indexing".

        Provides a way to do arr[inds] = vals in a way that works with JAX.

        Parameters
        ----------
        arr : array-like
            Array to populate
        inds : array-like of int
            Indices to populate
        vals : array-like
            Values to insert

        Returns
        -------
        arr : array-like
            Input array with vals inserted at inds.

        """
        arr[inds] = vals
        return arr

    def sign(x):
        """Sign function, but returns 1 for x==0.

        Parameters
        ----------
        x : array-like
            array of input values

        Returns
        -------
        y : array-like
            1 where x>=0, -1 where x<0

        """
        x = np.atleast_1d(x)
        y = np.where(x == 0, 1, np.sign(x))
        return y

    def fori_loop(lower, upper, body_fun, init_val):
        """Loop from lower to upper, applying body_fun to init_val.

        This version is for the numpy backend, for jax backend see jax.lax.fori_loop

        Parameters
        ----------
        lower : int
            an integer representing the loop index lower bound (inclusive)
        upper : int
            an integer representing the loop index upper bound (exclusive)
        body_fun : callable
            function of type ``(int, a) -> a``.
        init_val : array-like or container
            initial loop carry value of type ``a``

        Returns
        -------
        final_val: array-like or container
            Loop value from the final iteration, of type ``a``.

        """
        val = init_val
        for i in np.arange(lower, upper):
            val = body_fun(i, val)
        return val

    def cond(pred, true_fun, false_fun, *operand):
        """Conditionally apply true_fun or false_fun.

        This version is for the numpy backend, for jax backend see jax.lax.cond

        Parameters
        ----------
        pred: bool
            which branch function to apply.
        true_fun: callable
            Function (A -> B), to be applied if pred is True.
        false_fun: callable
            Function (A -> B), to be applied if pred is False.
        operand: any
            input to either branch depending on pred. The type can be a scalar, array,
            or any pytree (nested Python tuple/list/dict) thereof.

        Returns
        -------
        value: any
            value of either true_fun(operand) or false_fun(operand), depending on the
            value of pred. The type can be a scalar, array, or any pytree (nested
            Python tuple/list/dict) thereof.

        """
        if pred:
            return true_fun(*operand)
        else:
            return false_fun(*operand)

    def switch(index, branches, operand):
        """Apply exactly one of branches given by index.

        If index is out of bounds, it is clamped to within bounds.

        Parameters
        ----------
        index: int
            which branch function to apply.
        branches: Sequence[Callable]
            sequence of functions (A -> B) to be applied based on index.
        operand: any
            input to whichever branch is applied.

        Returns
        -------
        value: any
            output of branches[index](operand)

        """
        index = np.clip(index, 0, len(branches) - 1)
        return branches[index](operand)

    def while_loop(cond_fun, body_fun, init_val):
        """Call body_fun repeatedly in a loop while cond_fun is True.

        Parameters
        ----------
        cond_fun: callable
            function of type a -> bool.
        body_fun: callable
            function of type a -> a.
        init_val: any
            value of type a, a type that can be a scalar, array, or any pytree (nested
            Python tuple/list/dict) thereof, representing the initial loop carry value.

        Returns
        -------
        value: any
            The output from the final iteration of body_fun, of type a.

        """
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val

    def vmap(fun, out_axes=0):
        """A numpy implementation of jax.lax.map whose API is a subset of jax.vmap.

        Like Python's builtin map,
        except inputs and outputs are in the form of stacked arrays,
        and the returned object is a vectorized version of the input function.

        Parameters
        ----------
        fun: callable
            Function (A -> B)
        out_axes: int
            An integer indicating where the mapped axis should appear in the output.

        Returns
        -------
        fun_vmap: callable
            Vectorized version of fun.

        """

        def fun_vmap(fun_inputs):
            return np.stack([fun(fun_input) for fun_input in fun_inputs], axis=out_axes)

        return fun_vmap

    def bincount(x, weights=None, minlength=None, length=None):
        """Same as np.bincount but with a dummy parameter to match jnp.bincount API."""
        return np.bincount(x, weights, minlength)

    def custom_jvp(fun, *args, **kwargs):
        """Dummy function for custom_jvp without JAX."""
        fun.defjvp = lambda *args, **kwargs: None
        fun.defjvps = lambda *args, **kwargs: None
        return fun
