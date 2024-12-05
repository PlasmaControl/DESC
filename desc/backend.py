"""Backend functions for DESC, with options for JAX or regular numpy."""

import functools
import os
import warnings

import numpy as np
from termcolor import colored

import desc
from desc import config as desc_config
from desc import set_device

if os.environ.get("DESC_BACKEND") == "numpy":  # pragma: no cover
    use_jax = False
    warnings.warn(
        "DESC_BACKEND=numpy is deprecated. Please use DESC_BACKEND=jax instead.",
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
            from jax import config as jax_config

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
    except ModuleNotFoundError:  # pragma: no cover
        use_jax = False
        warnings.warn(
            colored(
                "Failed to load JAX. Numpy will not be supported anymore. "
                + "Please check your JAX installation",
                "red",
            )
        )
print(
    "Using device: {}, with {:.2f} GB available memory".format(
        desc_config.get("device"), desc_config.get("avail_mem")
    )
)

if use_jax:  # noqa: C901
    from jax import custom_jvp, jit, vmap

    imap = jax.lax.map
    from jax.experimental.ode import odeint
    from jax.lax import cond, fori_loop, scan, switch, while_loop
    from jax.nn import softmax as softargmax
    from jax.numpy import bincount, flatnonzero, repeat, take
    from jax.numpy.fft import irfft, rfft, rfft2
    from jax.scipy.fft import dct, idct
    from jax.scipy.linalg import block_diag, cho_factor, cho_solve, qr, solve_triangular
    from jax.scipy.special import gammaln
    from jax.tree_util import (
        register_pytree_node,
        tree_flatten,
        tree_leaves,
        tree_map,
        tree_structure,
        tree_unflatten,
        treedef_is_leaf,
    )

    if hasattr(jnp, "trapezoid"):
        trapezoid = jnp.trapezoid  # for JAX 0.4.26 and later
    elif hasattr(jax.scipy, "integrate"):
        trapezoid = jax.scipy.integrate.trapezoid
    else:
        trapezoid = jnp.trapz  # for older versions of JAX, deprecated by jax 0.4.16

    def execute_on_cpu(func):
        """Decorator to set default device to CPU for a function.

        Parameters
        ----------
        func : callable
            Function to decorate

        Returns
        -------
        wrapper : callable
            Decorated function that will always run on CPU even if
            there are available GPUs.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with jax.default_device(jax.devices("cpu")[0]):
                return func(*args, **kwargs)

        return wrapper

    # JAX implementation is not differentiable on gpu.
    eigh_tridiagonal = execute_on_cpu(jax.scipy.linalg.eigh_tridiagonal)

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
            Copy of input array with vals inserted at inds.
            In some cases JAX may decide a copy is not necessary.

        """
        if isinstance(arr, np.ndarray):
            arr = arr.copy()
            arr[inds] = vals
            return arr
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
        x = jnp.asarray(x)
        y = jnp.where(x == 0, 1, jnp.sign(x))
        return y

    @jit
    def tree_stack(trees):
        """Takes a list of trees and stacks every corresponding leaf.

        For example, given two trees ((a, b), c) and ((a', b'), c'), returns
        ((stack(a, a'), stack(b, b')), stack(c, c')).
        Useful for turning a list of objects into something you can feed to a
        vmapped function.
        """
        # from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
        import jax.tree_util as jtu

        return jtu.tree_map(lambda *v: jnp.stack(v), *trees)

    @jit
    def tree_unstack(tree):
        """Takes a tree and turns it into a list of trees. Inverse of tree_stack.

        For example, given a tree ((a, b), c), where a, b, and c all have first
        dimension k, will make k trees
        [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
        Useful for turning the output of a vmapped function into normal objects.
        """
        # from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
        import jax.tree_util as jtu

        leaves, treedef = jtu.tree_flatten(tree)
        return [treedef.unflatten(leaf) for leaf in zip(*leaves)]

    def root_scalar(
        fun,
        x0,
        jac=None,
        args=(),
        tol=1e-6,
        maxiter=20,
        maxiter_ls=5,
        alpha=0.1,
        fixup=None,
        full_output=False,
    ):
        """Find x where fun(x, *args) == 0.

        Parameters
        ----------
        fun : callable
            Function to find the root of. Should have a signature of the form
            fun(x, *args)- > float.
        x0 : float
            Initial guess for the root.
        jac : callable
            Jacobian of fun, should have a signature of the form jac(x, *args) -> float.
            Defaults to using jax.jacfwd
        args : tuple, optional
            Additional arguments to pass to fun and jac.
        tol : float, optional
            Stopping tolerance. Stops when norm(fun(x)) < tol.
        maxiter : int > 0, optional
            Maximum number of iterations.
        maxiter_ls : int >=0, optional
            Maximum number of sub-iterations for the backtracking line search.
        alpha : 0 < float < 1, optional
            Backtracking line search decrease factor. Line search first tries full
            Newton step, then alpha*Newton step, then alpha**2*Newton step etc.
        fixup : callable, optional
            Function to modify x after each update, ie to enforce periodicity. Should
            have a signature of the form fixup(x, *args) -> x'.
        full_output : bool, optional
            If True, also return a tuple where the first element is the residual from
            the root finding and the second is the number of iterations.

        Returns
        -------
        xk : float
            Root, or best approximation
        info : tuple of (float, int)
            Residual of fun at xk and number of iterations of outer loop

        """
        if fixup is None:
            fixup = lambda x, *args: x
        if jac is None:
            jac = jax.jacfwd(fun)
        jac2 = lambda x: jac(x, *args)
        res = lambda x: fun(x, *args)

        def solve(resfun, guess):
            def condfun_ls(state_ls):
                xk2, xk1, fk2, fk1, d, alphak2, k2 = state_ls
                return (k2 <= maxiter_ls) & (jnp.dot(fk2, fk2) >= jnp.dot(fk1, fk1))

            def bodyfun_ls(state_ls):
                xk2, xk1, fk2, fk1, d, alphak2, k2 = state_ls
                xk2 = fixup(xk1 - alphak2 * d, *args)
                fk2 = resfun(xk2)
                return xk2, xk1, fk2, fk1, d, alpha * alphak2, k2 + 1

            def backtrack(xk1, fk1, d):
                state_ls = (xk1, xk1, fk1, fk1, d, 1.0, 0)
                state_ls = jax.lax.while_loop(condfun_ls, bodyfun_ls, state_ls)
                xk2, xk1, fk2, fk1, d, alphak2, k2 = state_ls
                return xk2, fk2

            def condfun(state):
                xk1, fk1, k1 = state
                return (k1 < maxiter) & (jnp.dot(fk1, fk1) > tol**2)

            def bodyfun(state):
                xk1, fk1, k1 = state
                J = jac2(xk1)
                d = fk1 / J
                xk1, fk1 = backtrack(xk1, fk1, d)
                return xk1, fk1, k1 + 1

            state = guess, res(guess), 0.0
            state = jax.lax.while_loop(condfun, bodyfun, state)
            if full_output:
                return state[0], state[1:]
            else:
                return state[0]

        def tangent_solve(g, y):
            A = jax.jacfwd(g)(y)
            return y / A

        if full_output:
            x, (res, niter) = jax.lax.custom_root(
                res, x0, solve, tangent_solve, has_aux=True
            )
            return x, (abs(res), niter)
        else:
            x = jax.lax.custom_root(res, x0, solve, tangent_solve, has_aux=False)
            return x

    def root(
        fun,
        x0,
        jac=None,
        args=(),
        tol=1e-6,
        maxiter=20,
        maxiter_ls=0,
        alpha=0.1,
        fixup=None,
        full_output=False,
    ):
        """Find x where fun(x, *args) == 0.

        Parameters
        ----------
        fun : callable
            Function to find the root of. Should have a signature of the form
            fun(x, *args)- > 1d array.
        x0 : ndarray
            Initial guess for the root.
        jac : callable
            Jacobian of fun, should have a signature of the form
            jac(x, *args) -> 2d array. Defaults to using jax.jacfwd
        args : tuple, optional
            Additional arguments to pass to fun and jac.
        tol : float, optional
            Stopping tolerance. Stops when norm(fun(x)-f) < tol.
        maxiter : int > 0, optional
            Maximum number of iterations.
        maxiter_ls : int >=0, optional
            Maximum number of sub-iterations for the backtracking line search.
        alpha : 0 < float < 1, optional
            Backtracking line search decrease factor. Line search first tries full
            Newton step, then alpha*Newton step, then alpha**2*Newton step etc.
        fixup : callable, optional
            Function to modify x after each update, ie to enforce periodicity. Should
            have a signature of the form fixup(x, *args) -> 1d array.
        full_output : bool, optional
            If True, also return a tuple where the first element is the residual from
            the root finding and the second is the number of iterations.

        Returns
        -------
        xk : ndarray
            Root, or best approximation
        info : tuple of (ndarray, int)
            Residual of fun at xk and number of iterations of outer loop

        Notes
        -----
        This routine may be used on over or under-determined systems, in which case it
        will solve it in a least squares / least norm sense.
        """
        from desc.utils import safenorm

        if fixup is None:
            fixup = lambda x, *args: x
        if jac is None:
            jac2 = lambda x: jnp.atleast_2d(jax.jacfwd(fun)(x, *args))
        else:
            jac2 = lambda x: jnp.atleast_2d(jac(x, *args))

        res = lambda x: jnp.atleast_1d(fun(x, *args)).flatten()

        # want to use least squares for rank-defficient systems, but
        # jnp.linalg.lstsq doesn't have JVP defined and is slower than needed
        # so we use the normal equations with regularized cholesky
        def _lstsq(a, b):
            a = jnp.atleast_2d(a)
            b = jnp.atleast_1d(b)
            tall = a.shape[-2] >= a.shape[-1]
            A = a.T @ a if tall else a @ a.T
            B = a.T @ b if tall else b @ a
            A += jnp.sqrt(jnp.finfo(A.dtype).eps) * jnp.eye(A.shape[0])
            return cho_solve(cho_factor(A), B)

        def solve(resfun, guess):
            def condfun_ls(state_ls):
                xk2, xk1, fk2, fk1, d, alphak2, k2 = state_ls
                return (k2 <= maxiter_ls) & (jnp.dot(fk2, fk2) >= jnp.dot(fk1, fk1))

            def bodyfun_ls(state_ls):
                xk2, xk1, fk2, fk1, d, alphak2, k2 = state_ls
                xk2 = fixup(xk1 - alphak2 * d, *args)
                fk2 = resfun(xk2)
                return xk2, xk1, fk2, fk1, d, alpha * alphak2, k2 + 1

            def backtrack(xk1, fk1, d):
                state_ls = (xk1, xk1, fk1, fk1, d, 1.0, 0)
                state_ls = jax.lax.while_loop(condfun_ls, bodyfun_ls, state_ls)
                xk2, xk1, fk2, fk1, d, alphak2, k2 = state_ls
                return xk2, fk2

            def condfun(state):
                xk1, fk1, k1 = state
                return (k1 < maxiter) & (jnp.dot(fk1, fk1) > tol**2)

            def bodyfun(state):
                xk1, fk1, k1 = state
                J = jac2(xk1)
                d = _lstsq(J, fk1)
                xk1, fk1 = backtrack(xk1, fk1, d)
                return xk1, fk1, k1 + 1

            state = (
                jnp.atleast_1d(jnp.asarray(guess)),
                jnp.atleast_1d(resfun(guess)),
                0.0,
            )
            state = jax.lax.while_loop(condfun, bodyfun, state)
            if full_output:
                return state[0], state[1:]
            else:
                return state[0]

        def tangent_solve(g, y):
            A = jnp.atleast_2d(jax.jacfwd(g)(y))
            return _lstsq(A, jnp.atleast_1d(y))

        if full_output:
            x, (res, niter) = jax.lax.custom_root(
                res, x0, solve, tangent_solve, has_aux=True
            )
            return x, (safenorm(res), niter)
        else:
            x = jax.lax.custom_root(res, x0, solve, tangent_solve, has_aux=False)
            return x

else:  # pragma: no cover
    warnings.warn(
        colored(
            "Failed to load JAX. Numpy will not be supported anymore. "
            + "Please check your JAX installation",
            "red",
        )
    )
