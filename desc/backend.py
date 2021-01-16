import numpy as np
import warnings
import desc
import os
from termcolor import colored

# we divide by zero in a few places but then overwrite with the
# correct values, so lets suppress annoying warnings about that
np.seterr(divide="ignore", invalid="ignore")


os.environ["JAX_PLATFORM_NAME"] = "cpu"

if os.environ.get("DESC_USE_NUMPY"):
    jnp = np
    use_jax = False
    print(
        "DESC version {}, using numpy backend, version={}, dtype={}".format(
            desc.__version__, np.__version__, np.linspace(0, 1).dtype
        )
    )
else:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jax
            import jaxlib
            import jax.numpy as jnp
            from jax.config import config

            config.update("jax_enable_x64", True)
            x = jnp.linspace(0, 5)
            y = jnp.exp(x)
        use_jax = True
        print(
            "DESC version {}, using JAX backend, jax version={}, jaxlib version={}, dtype={}".format(
                desc.__version__, jax.__version__, jaxlib.__version__, y.dtype
            )
        )
    except:
        jnp = np
        x = jnp.linspace(0, 5)
        y = jnp.exp(x)
        use_jax = False
        warnings.warn(colored("Failed to load JAX", "red"))
        print(
            "DESC version {}, using NumPy backend, version={}, dtype={}".format(
                desc.__version__, np.__version__, y.dtype
            )
        )


if use_jax:
    jit = jax.jit
    fori_loop = jax.lax.fori_loop
    from jax.scipy.linalg import cho_factor, cho_solve, qr

    def put(arr, inds, vals):
        """Functional interface for array "fancy indexing"

        basically a way to do arr[inds] = vals in a way that plays nice with jit/autodiff.

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

        return jax.ops.index_update(arr, inds, vals)

    @jit
    def factorial(n):
        """Factorial function for jax backend

        Parameters
        ----------
        n : array-like of int
            input values. if n<0, returns 0

        Returns
        -------
        n! : array-like of float
            factorial of n

        """
        x = jnp.asarray(n + 1)
        y = jnp.exp(jax.scipy.special.gammaln(x))
        y = jnp.where(x < 1, 0, y)
        return y


else:
    jit = lambda func, *args, **kwargs: func
    from scipy.special import factorial
    from scipy.linalg import cho_factor, cho_solve, qr

    def put(arr, inds, vals):
        """Functional interface for array "fancy indexing"

        basically a way to do arr[inds] = vals in a way that plays nice with jit/autodiff.

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

    def fori_loop(lower, upper, body_fun, init_val):
        """Loop from lower to upper, applying body_fun to init_val

        This version is for the numpy backend, for jax backend see jax.lax.fori_loop
        The semantics of ``fori_loop`` are given by this Python implementation::

            def fori_loop(lower, upper, body_fun, init_val):
                val = init_val
                for i in range(lower, upper):
                    val = body_fun(i, val)
                return val

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
