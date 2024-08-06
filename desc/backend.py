"""Backend functions for DESC, with options for JAX or regular numpy."""

import functools
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
    from functools import partial

    import jax
    import jax.numpy as jnp
    import jaxlib
    from jax import config as jax_config

    repeat = jnp.repeat
    take = jnp.take
    scan = jax.lax.scan
    from jax import custom_jvp
    from jax.experimental.ode import odeint
    from jax.scipy.linalg import block_diag, cho_factor, cho_solve, qr, solve_triangular
    from jax.scipy.special import gammaln, logsumexp
    from jax.tree_util import (
        register_pytree_node,
        tree_flatten,
        tree_leaves,
        tree_map,
        tree_structure,
        tree_unflatten,
        treedef_is_leaf,
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
            Input array with vals inserted at inds.

        """
        if isinstance(arr, np.ndarray):
            arr[inds] = vals
            return arr
        return jnp.asarray(arr).at[inds].set(vals)

    def execute_on_cpu(func):
        """Decorator to set default device to CPU for a function.

        Parameters
        ----------
        func : callable
            Function to decorate

        Returns
        -------
        wrapper : callable
            Decorated function that will run always on CPU even if
            there are available GPUs.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with jax.default_device(jax.devices("cpu")[0]):
                return func(*args, **kwargs)

        return wrapper

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

    _eigvals_cpu = jax.jit(jnp.linalg.eigvals, device=jax.devices("cpu")[0])

    @jax.custom_jvp
    def eigvals(A):
        """
        Eigenvalue solver.

        Returns the eigenvalues of the square matrix A.
        eigvals only run on CPUs
        """
        u = jax.pure_callback(
            _eigvals_cpu, jnp.zeros_like(A[..., -1]) + 1j, A, vectorized=True
        )
        return u

    @eigvals.defjvp
    def _eigvals_jvp(primals, tangents):

        u = eigvals(primals[0])

        @partial(jnp.vectorize, signature="(n,n),(n,n)->(n)")
        def jvpfun(primals, tangents):
            u, du = jax.jvp(_eigvals_cpu, (primals,), (tangents,))
            return du.squeeze()

        du = jax.pure_callback(jvpfun, u, *primals, *tangents, vectorized=True)
        return u, du

    _gen_eigval_cpu = jax.jit(jax.scipy.linalg.eigh, device=jax.devices("cpu")[0])

    @jax.custom_jvp
    def gen_eigval(A):
        """
        Generalize eigenvalue solver.

        Returns the top n eigenvalues of the square matrix A. Calculation is
        being performed on a CPU. If the CPU version can provide the top eigenvalue,
        the calculation should be faster on a CPU.
        Currently doesn't work because of the limitations of the jax functionality.
        """
        neigs, N, _ = jnp.shape(A)
        u = jnp.zeros((N,))
        i = jnp.arange(N)

        u = u.at[i].set(
            jax.pure_callback(
                _gen_eigval_cpu,
                jnp.zeros_like(A[i, :, :]),
                A[i, :, :],
                k=1,
                sigma=0.42,
                vectorized=True,
            )
        )
        return u

    @gen_eigval.defjvp
    def _gen_eigval_jvp(primals, tangents):

        u = gen_eigval(primals[0])

        @partial(jnp.vectorize, signature="(n,n),(n,n)->(n)")
        def jvpfun(primals, tangents):
            u, du = jax.jvp(_gen_eigval_cpu, (primals,), (tangents,))
            return du.squeeze()

        du = jax.pure_callback(jvpfun, u, *primals, *tangents, vectorized=True)
        return u, du

    @jit
    def simspson_integrator(y, dx):
        """Simpsons integrations scheme for high-order accurate integrals."""
        if len(y[..., :]) % 2 == 1:
            raise ValueError("n must be even")

        S = (
            dx
            / 3
            * (
                y[..., 0]
                + y[..., -1]
                + 4 * jnp.sum(y[..., 1:-1:2], axis=-1)
                + 2 * jnp.sum(y[..., 2:-2:2], axis=-1)
            )
        )

        return S

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

            state = guess, res(guess), 0
            state = jax.lax.while_loop(condfun, bodyfun, state)
            return state[0], state[1:]

        def tangent_solve(g, y):
            A = jax.jacfwd(g)(y)
            return y / A

        x, (res, niter) = jax.lax.custom_root(
            res, x0, solve, tangent_solve, has_aux=True
        )
        return x, (abs(res), niter)

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
                0,
            )
            state = jax.lax.while_loop(condfun, bodyfun, state)
            return state[0], state[1:]

        def tangent_solve(g, y):
            A = jnp.atleast_2d(jax.jacfwd(g)(y))
            return _lstsq(A, jnp.atleast_1d(y))

        x, (res, niter) = jax.lax.custom_root(
            res, x0, solve, tangent_solve, has_aux=True
        )
        return x, (jnp.linalg.norm(res), niter)


# we can't really test the numpy backend stuff in automated testing, so we ignore it
# for coverage purposes
else:  # pragma: no cover
    jit = lambda func, *args, **kwargs: func
    import numpy as np

    execute_on_cpu = lambda func: func
    import scipy.optimize
    from scipy.integrate import odeint  # noqa: F401
    from scipy.linalg import (  # noqa: F401
        block_diag,
        cho_factor,
        cho_solve,
        qr,
        solve_triangular,
    )
    from scipy.special import gammaln, logsumexp  # noqa: F401

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

    def bincount(x, weights=None, minlength=None, length=None):
        """Same as np.bincount but with a dummy parameter to match jnp.bincount API."""
        return np.bincount(x, weights, minlength)

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

    def eigvals(A):
        """
        Eigenvalue solver.

        Returns the eigenvalues of the square matrix A.
        """
        u = np.linalg.eigvals(A)
        return u

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
        return out.root, out

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
        return out.x, out

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
