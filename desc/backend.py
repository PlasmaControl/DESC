"""Backend functions for DESC, with options for JAX or regular numpy."""

import functools
import os
import warnings

import numpy as np
from packaging.version import Version
from termcolor import colored

import desc
from desc import config as desc_config
from desc import set_device

if os.environ.get("DESC_BACKEND") == "numpy":
    jnp = np
    use_jax = False
    set_device(kind="cpu")
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
    except ModuleNotFoundError:
        jnp = np
        x = jnp.linspace(0, 5)
        y = jnp.exp(x)
        use_jax = False
        set_device(kind="cpu")
        warnings.warn(colored("Failed to load JAX", "red"))


def print_backend_info():
    """Prints DESC version, backend type & version, device type & memory."""
    print(f"DESC version={desc.__version__}.")
    if use_jax:
        print(
            f"Using JAX backend: jax version={jax.__version__}, "
            + f"jaxlib version={jaxlib.__version__}, dtype={y.dtype}."
        )
    else:
        print(f"Using NumPy backend: version={np.__version__}, dtype={y.dtype}.")
    print(
        "Using device: {}, with {:.2f} GB available memory.".format(
            desc_config.get("device"), desc_config.get("avail_mem")
        )
    )


def _diag_to_full(d, e):
    j = np.arange(d.shape[-1])
    return (
        jnp.zeros(d.shape + (d.shape[-1],))
        .at[..., j, j]
        .set(d, indices_are_sorted=True, unique_indices=True)
        .at[..., j[:-1], j[1:]]
        .set(e, indices_are_sorted=True, unique_indices=True)
        .at[..., j[1:], j[:-1]]
        .set(e, indices_are_sorted=True, unique_indices=True)
    )


if use_jax:  # noqa: C901
    from jax import custom_jvp, jit, vmap
    from jax.experimental.ode import odeint
    from jax.lax import cond, fori_loop, scan, switch, while_loop
    from jax.nn import softmax as softargmax
    from jax.numpy import bincount, flatnonzero, repeat, take
    from jax.numpy.fft import ifft, irfft, irfft2, rfft, rfft2
    from jax.scipy.fft import dct, idct
    from jax.scipy.linalg import block_diag, cho_factor, cho_solve, qr, solve_triangular
    from jax.scipy.special import gammaln
    from jax.tree_util import (
        register_pytree_node,
        tree_flatten,
        tree_leaves,
        tree_map,
        tree_map_with_path,
        tree_structure,
        tree_unflatten,
        treedef_is_leaf,
    )

    # TODO: update this when JAX min version >= 0.4.26
    if hasattr(jnp, "trapezoid"):
        trapezoid = jnp.trapezoid  # for JAX 0.4.26 and later
    elif hasattr(jax.scipy, "integrate"):
        trapezoid = jax.scipy.integrate.trapezoid
    else:
        trapezoid = jnp.trapz  # for older versions of JAX, deprecated by jax 0.4.16

    # TODO: update this when JAX min version >= 0.4.35
    if Version(jax.__version__) >= Version("0.4.35"):

        def pure_callback(func, result_shape_dtype, *args, vectorized=False, **kwargs):
            """Wrapper for jax.pure_callback for versions >=0.4.35."""
            return jax.pure_callback(
                func,
                result_shape_dtype,
                *args,
                vmap_method="expand_dims" if vectorized else "sequential",
                **kwargs,
            )

    else:

        def pure_callback(func, result_shape_dtype, *args, vectorized=False, **kwargs):
            """Wrapper for jax.pure_callback for versions <0.4.35."""
            return jax.pure_callback(
                func, result_shape_dtype, *args, vectorized=vectorized, **kwargs
            )

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

    _eigh_tridiagonal = jnp.vectorize(
        jax.scipy.linalg.eigh_tridiagonal,
        signature="(m),(n)->(m)",
        excluded={"eigvals_only", "select", "select_range", "tol"},
    )
    if desc_config["kind"] == "gpu":
        # JAX eigh_tridiagonal is not differentiable on gpu.
        # https://github.com/jax-ml/jax/issues/23650
        # # TODO (#1750): Eventually use this once it supports kwargs.
        # https://docs.jax.dev/en/latest/_autosummary/jax.lax.platform_dependent.html

        def eigh_tridiagonal(
            d,
            e,
            *,
            eigvals_only=False,
            select="a",
            select_range=None,
            tol=None,
        ):
            """Wrapper for eigh_tridiagonal which is partially implemented in JAX.

            Calls linalg.eigh when on GPU or when eigenvectors are requested.
            """
            return jax.scipy.linalg.eigh(
                _diag_to_full(d, e), eigvals_only=eigvals_only, eigvals=select_range
            )

    else:

        def eigh_tridiagonal(
            d,
            e,
            *,
            eigvals_only=False,
            select="a",
            select_range=None,
            tol=None,
        ):
            """Wrapper for eigh_tridiagonal which is partially implemented in JAX.

            Calls linalg.eigh when on GPU or when eigenvectors are requested.
            """
            # Reverse mode also not differentiable on CPU.
            # TODO (#1750): Update logic when resolving the linked issue?
            if True or not eigvals_only:
                # https://github.com/jax-ml/jax/issues/14019
                return jax.scipy.linalg.eigh(
                    _diag_to_full(d, e), eigvals_only=eigvals_only, eigvals=select_range
                )
            return _eigh_tridiagonal(
                d,
                e,
                eigvals_only=eigvals_only,
                select=select,
                select_range=select_range,
                tol=tol,
            )

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
            return y / g(1.0)

        if full_output:
            x, (res, niter) = jax.lax.custom_root(
                res, x0, solve, tangent_solve, has_aux=True
            )
            return x, (abs(res), niter)
        else:
            x = jax.lax.custom_root(res, x0, solve, tangent_solve, has_aux=False)
            return x

    def _lstsq(A, y):
        """Cholesky factorized least-squares.

        jnp.linalg.lstsq doesn't have JVP defined and is slower than needed,
        so we use regularized cholesky.

        For square systems, solves Ax=y directly.
        """
        A = jnp.atleast_2d(A)
        y = jnp.atleast_1d(y)
        eps = jnp.sqrt(jnp.finfo(A.dtype).eps)
        if A.shape[-2] == A.shape[-1]:
            return jnp.linalg.solve(A, y) if y.size > 1 else jnp.squeeze(y / A)
        elif A.shape[-2] > A.shape[-1]:
            P = A.T @ A + eps * jnp.eye(A.shape[-1])
            return cho_solve(cho_factor(P), A.T @ y)
        else:
            P = A @ A.T + eps * jnp.eye(A.shape[-2])
            return A.T @ cho_solve(cho_factor(P), y)

    def _tangent_solve(g, y):
        return _lstsq(jax.jacfwd(g)(y), y)

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

        if full_output:
            x, (res, niter) = jax.lax.custom_root(
                res, x0, solve, _tangent_solve, has_aux=True
            )
            return x, (safenorm(res), niter)
        else:
            x = jax.lax.custom_root(res, x0, solve, _tangent_solve, has_aux=False)
            return x


# we can't really test the numpy backend stuff in automated testing, so we ignore it
# for coverage purposes
else:  # pragma: no cover
    jit = lambda func, *args, **kwargs: func
    execute_on_cpu = lambda func: func
    import scipy.optimize
    from numpy.fft import ifft, irfft, irfft2, rfft, rfft2  # noqa: F401
    from scipy.fft import dct, idct  # noqa: F401
    from scipy.integrate import odeint  # noqa: F401
    from scipy.linalg import (  # noqa: F401
        block_diag,
        cho_factor,
        cho_solve,
        qr,
        solve_triangular,
    )
    from scipy.special import gammaln  # noqa: F401
    from scipy.special import softmax as softargmax  # noqa: F401

    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    eigh_tridiagonal = np.vectorize(
        scipy.linalg.eigh_tridiagonal,
        signature="(m),(n)->(m)",
        excluded={"eigvals_only", "select", "select_range", "tol"},
    )

    def _map(f, xs, *, batch_size=None, in_axes=0, out_axes=0):
        """Generalizes jax.lax.map; uses numpy."""
        if not isinstance(xs, np.ndarray):
            raise NotImplementedError(
                "Require numpy array input, or install jax to support pytrees."
            )
        xs = np.moveaxis(xs, source=in_axes, destination=0)
        return np.stack([f(x) for x in xs], axis=out_axes)

    def vmap(fun, in_axes=0, out_axes=0):
        """A numpy implementation of jax.lax.map whose API is a subset of jax.vmap.

        Like Python's builtin map,
        except inputs and outputs are in the form of stacked arrays,
        and the returned object is a vectorized version of the input function.

        Parameters
        ----------
        fun: callable
            Function (A -> B)
        in_axes: int
            Axis to map over.
        out_axes: int
            An integer indicating where the mapped axis should appear in the output.

        Returns
        -------
        fun_vmap: callable
            Vectorized version of fun.

        """
        return lambda xs: _map(fun, xs, in_axes=in_axes, out_axes=out_axes)

    def pure_callback(*args, **kwargs):
        """IO callback for numpy backend."""
        raise NotImplementedError

    def tree_stack(*args, **kwargs):
        """Stack pytree for numpy backend."""
        raise NotImplementedError

    def tree_unstack(*args, **kwargs):
        """Unstack pytree for numpy backend."""
        raise NotImplementedError

    def tree_flatten(*args, **kwargs):
        """Flatten pytree for numpy backend."""
        raise NotImplementedError

    def tree_unflatten(*args, **kwargs):
        """Unflatten pytree for numpy backend."""
        raise NotImplementedError

    def tree_map(*args, **kwargs):
        """Map pytree for numpy backend."""
        raise NotImplementedError

    def tree_map_with_path(*args, **kwargs):
        """Map pytree with path for numpy backend."""
        raise NotImplementedError

    def tree_structure(*args, **kwargs):
        """Get structure of pytree for numpy backend."""
        raise NotImplementedError

    def tree_leaves(*args, **kwargs):
        """Get leaves of pytree for numpy backend."""
        raise NotImplementedError

    def treedef_is_leaf(*args, **kwargs):
        """Check is leaf of pytree for numpy backend."""
        raise NotImplementedError

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
            Copy of input array with vals inserted at inds.

        """
        arr = arr.copy()
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

    def cond(pred, true_fun, false_fun, *operands):
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
        operands: any
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
            return true_fun(*operands)
        else:
            return false_fun(*operands)

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

    def scan(f, init, xs, length=None, reverse=False, unroll=1):
        """Scan a function over leading array axes while carrying along state.

        Parameters
        ----------
        f : callable
            Python function to be scanned of type c -> a -> (c, b), meaning that f
            accepts two arguments where the first is a value of the loop carry and the
            second is a slice of xs along its leading axis, and that f returns a pair
            where the first element represents a new value for the loop carry and the
            second represents a slice of the output.
        init : ndarray
            an initial loop carry value of type c.
        xs : ndarray
            the value of type [a] over which to scan along the leading axis.
        length : int, optional
            optional integer specifying the number of loop iterations, which must agree
            with the sizes of leading axes of the arrays in xs (but can be used to
            perform scans where no input xs are needed).
        reverse : bool
            optional boolean specifying whether to run the scan iteration forward
            (the default) or in reverse, equivalent to reversing the leading axes of
            the arrays in both xs and in ys.
        unroll : int, optional
            optional positive int specifying, in the underlying operation of the scan
            primitive, how many scan iterations to unroll within a single iteration
            of a loop.
        """
        if xs is None:
            xs = [None] * length
        carry = init
        ys = []
        if reverse:
            xs = xs[::-1]
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, np.stack(ys)

    def bincount(x, weights=None, minlength=0, length=None):
        """A numpy implementation of jnp.bincount."""
        x = np.clip(x, 0, None)
        if length is None:
            length = max(minlength, x.max() + 1)
        else:
            minlength = max(minlength, length)
        return np.bincount(x, weights, minlength)[:length]

    def repeat(a, repeats, axis=None, total_repeat_length=None):
        """A numpy implementation of jnp.repeat."""
        out = np.repeat(a, repeats, axis)
        if total_repeat_length is not None:
            out = out[:total_repeat_length]
        return out

    def custom_jvp(fun, *args, **kwargs):
        """Dummy function for custom_jvp without JAX."""
        fun.defjvp = lambda *args, **kwargs: None
        fun.defjvps = lambda *args, **kwargs: None
        return fun

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
            have a signature of the form fixup(x) -> x'.
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
        out = scipy.optimize.root_scalar(
            fun, args, x0=x0, fprime=jac, xtol=tol, rtol=tol
        )
        if full_output:
            return out.root, out
        else:
            return out.root

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
        will solve it in a least squares sense.
        """
        out = scipy.optimize.root(fun, x0, args, jac=jac, tol=tol)
        if full_output:
            return out.x, out
        else:
            return out.x

    def flatnonzero(a, size=None, fill_value=0):
        """A numpy implementation of jnp.flatnonzero."""
        nz = np.flatnonzero(a)
        if size is not None:
            nz = np.pad(nz, (0, max(size - nz.size, 0)), constant_values=fill_value)
        return nz

    def take(
        a,
        indices,
        axis=None,
        out=None,
        mode="fill",
        unique_indices=False,
        indices_are_sorted=False,
        fill_value=None,
    ):
        """A numpy implementation of jnp.take."""
        if mode == "fill":
            if fill_value is None:
                # copy jax logic
                # https://jax.readthedocs.io/en/latest/_modules/jax/_src/lax/slicing.html#gather
                if np.issubdtype(a.dtype, np.inexact):
                    fill_value = np.nan
                elif np.issubdtype(a.dtype, np.signedinteger):
                    fill_value = np.iinfo(a.dtype).min
                elif np.issubdtype(a.dtype, np.unsignedinteger):
                    fill_value = np.iinfo(a.dtype).max
                elif a.dtype == np.bool_:
                    fill_value = True
                else:
                    raise ValueError(f"Unsupported dtype {a.dtype}.")
            out = np.where(
                (-a.size <= indices) & (indices < a.size),
                np.take(a, indices, axis, out, mode="wrap"),
                fill_value,
            )
        else:
            out = np.take(a, indices, axis, out, mode)
        return out
