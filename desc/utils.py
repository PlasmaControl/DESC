"""Utility functions, independent of the rest of DESC."""

import functools
import operator
import warnings
from itertools import combinations_with_replacement, permutations

import numpy as np
from scipy.special import factorial
from termcolor import colored

from desc.backend import flatnonzero, fori_loop, jax, jit, jnp, pure_callback, take

PRINT_WIDTH = 60  # current longest name is BootstrapRedlConsistency with pre-text


class Timer:
    """Simple object for organizing timing info.

    Create a Timer object, which can then keep track of
    multiple concurrent performance timers, each associated with
    a given name.

    Individual timers can be started and stopped with
    ``timer.start(name)`` and ``timer.stop(name)``

    The elapsed time can be printed with ``timer.disp(name)``

    Raw values of elapsed time (in seconds) can be retrieved
    with ``timer[name]``

    Parameters
    ----------
    ns : bool, optional
       use nanosecond timing if available

    """

    def __init__(self, ns=True):
        import time

        self._times = {}
        self._timers = {}
        self._ns = ns
        if self._ns:
            try:
                self.op = time.perf_counter_ns
            except AttributeError:
                self.op = time.perf_counter
                self._ns = False
                warnings.warn(
                    colored(
                        "nanosecond timing not available on this system,"
                        + " reverting to microsecond timing",
                        "yellow",
                    )
                )

        else:
            self.op = time.perf_counter

    def start(self, name):
        """Start a timer.

        Parameters
        ----------
        name : str
            name to associate with timer

        """
        self._timers[name] = [self.op()]

    def stop(self, name):
        """Stop a running timer.

        Parameters
        ----------
        name : str
            name of timer to stop

        Raises
        ------
        ValueError
            if timer ``'name'`` has not been started

        """
        try:
            self._timers[name].append(self.op())
        except KeyError:
            raise ValueError(
                colored("timer '{}' has not been started".format(name), "red")
            ) from None
        self._times[name] = np.diff(self._timers[name])[0]
        if self._ns:
            self._times[name] = self._times[name] / 1e9
        del self._timers[name]

    @staticmethod
    def pretty_print(name, time):
        """Pretty print time interval.

        Does not modify or use any internal timer data,
        this is just a helper for pretty printing arbitrary time data

        Parameters
        ----------
        name : str
            text to print before time
        time : float
            time (in seconds) to print

        """
        us = time * 1e6
        ms = us / 1000
        sec = ms / 1000
        mins = sec / 60
        hrs = mins / 60

        if us < 100:
            out = "{:.3f}".format(us)[:4] + " us"
        elif us < 1000:
            out = "{:.3f}".format(us)[:3] + " us"
        elif ms < 100:
            out = "{:.3f}".format(ms)[:4] + " ms"
        elif ms < 1000:
            out = "{:.3f}".format(ms)[:3] + " ms"
        elif sec < 60:
            out = "{:.3f}".format(sec)[:4] + " sec"
        elif mins < 60:
            out = "{:.3f}".format(mins)[:4] + " min"
        else:
            out = "{:.3f}".format(hrs)[:4] + " hrs"

        print(colored("Timer: {} = {}".format(name, out), "green"))

    def disp(self, name):
        """Pretty print elapsed time.

        If the timer has been stopped, it reports the time delta between
        start and stop. If it has not been stopped, it reports the current
        elapsed time and keeps the timing running.

        Parameters
        ----------
        name : str
            name of the timer to display

        Raises
        ------
        ValueError
            if timer ``'name'`` has not been started

        """
        try:  # has the timer been stopped?
            time = self._times[name]
        except KeyError:  # might still be running, let's check
            try:
                start = self._timers[name][0]
                now = self.op()  # don't stop it, just report current elapsed time
                time = float(now - start) / 1e9 if self._ns else (now - start)
            except KeyError:
                raise ValueError(
                    colored("timer '{}' has not been started".format(name), "red")
                ) from None

        self.pretty_print(name, time)

    def __getitem__(self, key):
        return self._times[key]

    def __setitem__(self, key, val):
        self._times[key] = val


class _Indexable:
    """Helper object for building indexes for indexed update functions.

    This is a singleton object that overrides the ``__getitem__`` method
    to return the index it is passed.
    >>> Index[1:2, 3, None, ..., ::2]
    (slice(1, 2, None), 3, None, Ellipsis, slice(None, None, 2))
    copied from jax.ops.index to work with either backend
    """

    __slots__ = ()

    def __getitem__(self, index):
        return index

    @staticmethod
    def get(stuff, axis, ndim):
        slices = [slice(None)] * ndim
        slices[axis] = stuff
        slices = tuple(slices)
        return slices


"""
Helper object for building indexes for indexed update functions.
This is a singleton object that overrides the ``__getitem__`` method
to return the index it is passed.
>>> Index[1:2, 3, None, ..., ::2]
(slice(1, 2, None), 3, None, Ellipsis, slice(None, None, 2))
copied from jax.ops.index to work with either backend
"""
Index = _Indexable()


def equals(a, b):
    """Compare (possibly nested) objects, such as dicts and lists.

    Parameters
    ----------
    a :
        reference object
    b :
        comparison object

    Returns
    -------
    bool
        a == b

    """
    if hasattr(a, "shape") and hasattr(b, "shape"):
        return a.shape == b.shape and np.allclose(a, b)
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(equals(a[key], b[key]) for key in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all([equals(a[i], b[i]) for i in range(len(a))])
    if hasattr(a, "equiv"):
        return a.equiv(b)
    return a == b


def flatten_list(x, flatten_tuple=False):
    """Flatten a nested list.

    Parameters
    ----------
    x : list
        nested list of lists to flatten
    flatten_tuple : bool
        Whether to also flatten nested tuples.

    Returns
    -------
    x : list
        flattened input

    """
    types = (list,)
    if flatten_tuple:
        types += (tuple,)
    if isinstance(x, types):
        return [a for i in x for a in flatten_list(i, flatten_tuple)]
    else:
        return [x]


def issorted(x, axis=None, tol=1e-12):
    """Check if an array is sorted, within a given tolerance.

    Checks whether x[i+1] - x[i] > tol

    Parameters
    ----------
    x : array-like
        input values
    axis : int
        axis along which to check if the array is sorted.
        If None, the flattened array is used
    tol : float
        tolerance for determining order. Array is still considered sorted
        if the difference between adjacent values is greater than -tol

    Returns
    -------
    is_sorted : bool
        whether the array is sorted along specified axis

    """
    x = np.asarray(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    return np.all(np.diff(x, axis=axis) >= -tol)


def isalmostequal(x, axis=-1, rtol=1e-6, atol=1e-12):
    """Check if all values of an array are equal, to within a given tolerance.

    Parameters
    ----------
    x : array-like
        input values
    axis : int
        axis along which to make comparison. If None, the flattened array is used
    rtol : float
        relative tolerance for comparison.
    atol : float
        absolute tolerance for comparison.
        If the following equation is element-wise True, then returns True.
            absolute(a - b) <= (atol + rtol * absolute(b))
        where a= x[0] and b is every other element of x, if flattened array,
        or if axis is not None, a = x[:,0,:] and b = x[:,i,:] for all i, and
        the 0,i placement is in the dimension indicated by axis

    Returns
    -------
    is_almost_equal : bool
        whether the array is equal along specified axis

    """
    x = np.asarray(x)
    if x.ndim == 0 or x.size == 0:
        return True
    if axis is None or x.ndim == 1:
        x = x.flatten()
        return np.allclose(x[0], x, atol=atol, rtol=rtol)

    # some fancy indexing, basically this is to be able to use np.allclose
    # and broadcast the desired array we want matching along the specified axis,
    inds = [0] * x.ndim
    # want slice for all except axis
    for i, dim in enumerate(x.shape):
        inds[i] = slice(0, dim)
    inds[axis] = 0
    inds = tuple(inds)
    # array we want to be the same along the specified axis
    arr_match = x[inds]

    # this just puts a np.newaxis where the specified axis is
    # so that we can tell np.allclose we want this array
    # broadcast to match the size of our original array
    inds_broadcast = list(inds)
    inds_broadcast[axis] = np.newaxis
    inds_broadcast = tuple(inds_broadcast)

    return np.allclose(x, arr_match[inds_broadcast], atol=atol, rtol=rtol)


def islinspaced(x, axis=-1, rtol=1e-6, atol=1e-12):
    """Check if all values of an array are linearly spaced, to within a given tolerance.

    Parameters
    ----------
    x : array-like
        input values
    axis : int
        axis along which to make comparison. If None, the flattened array is used
    rtol : float
        relative tolerance for comparison.
    atol : float
        absolute tolerance for comparison.

    Returns
    -------
    is_linspaced : bool
        whether the array is linearly spaced along specified axis

    """
    x = np.asarray(x)
    if x.ndim == 0 or x.size == 0:
        return True
    if axis is None or x.ndim == 1:
        x = x.flatten()
        xdiff = np.diff(x)
        return np.allclose(xdiff[0], xdiff, atol=atol, rtol=rtol)

    return isalmostequal(np.diff(x, axis=axis), rtol=rtol, atol=atol, axis=axis)


@jit
def copy_coeffs(c_old, modes_old, modes_new, c_new=None):
    """Copy coefficients from one resolution to another."""
    modes_old, modes_new = jnp.atleast_1d(jnp.asarray(modes_old)), jnp.atleast_1d(
        jnp.asarray(modes_new)
    )

    if modes_old.ndim == 1:
        modes_old = modes_old.reshape((-1, 1))
    if modes_new.ndim == 1:
        modes_new = modes_new.reshape((-1, 1))

    if c_new is None:
        c_new = jnp.zeros((modes_new.shape[0],))
    c_old, c_new = jnp.asarray(c_old), jnp.asarray(c_new)

    def body(i, c_new):
        mask = (modes_old[i, :] == modes_new).all(axis=1)
        c_new = jnp.where(mask, c_old[i], c_new)
        return c_new

    if c_old.size:
        c_new = fori_loop(0, modes_old.shape[0], body, c_new)
    return c_new


def svd_inv_null(A):
    """Compute pseudo-inverse and null space of a matrix using an SVD.

    Parameters
    ----------
    A : ndarray
        Matrix to invert and find null space of.

    Returns
    -------
    Ainv : ndarray
        Pseudo-inverse of A.
    Z : ndarray
        Null space of A.

    """
    u, s, vh = jnp.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    K = min(M, N)
    rcond = np.finfo(A.dtype).eps * max(M, N)
    tol = jnp.amax(s) * rcond
    large = s > tol
    num = jnp.sum(large, dtype=int)
    uk = u[:, :K]
    vhk = vh[:K, :]
    s = jnp.where(large, 1 / s, 0)
    Ainv = vhk.T @ jnp.diag(s) @ uk.T
    Z = vh[num:, :].T.conj()
    return Ainv, Z


def combination_permutation(m, n, equals=True):
    """Compute all m-tuples of non-negative ints that sum to less than or equal to n.

    Parameters
    ----------
    m : int
        Size of tuples. IE, number of items being combined.
    n : int
        Maximum sum
    equals : bool
        If True, return only where sum == n, else return where sum <= n

    Returns
    -------
    out : ndarray
        m tuples that sum to n, or less than n if equals=False
    """
    out = []
    combos = combinations_with_replacement(range(n + 1), m)
    for combo in list(combos):
        perms = set(permutations(combo))
        for perm in list(perms):
            out += [perm]
    out = np.array(out)
    if equals:
        out = out[out.sum(axis=-1) == n]
    else:
        out = out[out.sum(axis=-1) <= n]
    return out


def multinomial_coefficients(m, n):
    """Number of ways to place n objects into m bins."""
    k = combination_permutation(m, n)
    num = factorial(n, exact=True)
    den = factorial(k, exact=True).prod(axis=-1)
    return num / den


def is_broadcastable(shp1, shp2):
    """Determine if 2 shapes will broadcast without error.

    Parameters
    ----------
    shp1, shp2 : tuple of int
        Shapes of the arrays to check.

    Returns
    -------
    is_broadcastable : bool
        Whether the arrays can be broadcast.
    """
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def get_instance(things, cls):
    """Get first thing from an iterable of things that is instance of cls."""
    foo = [t for t in things if isinstance(t, cls)]
    return foo[0] if len(foo) else None


def get_all_instances(things, cls):
    """Get every thing from an iterable of things that is instance of cls."""
    foo = [t for t in things if isinstance(t, cls)]
    return foo if len(foo) else None


def parse_argname_change(arg, kwargs, oldname, newname):
    """Warn and parse arguments whose names have changed."""
    if oldname in kwargs:
        warnings.warn(
            FutureWarning(
                f"argument {oldname} has been renamed to {newname}, "
                + f"{oldname} will be removed in a future release"
            )
        )
        arg = kwargs.pop(oldname)
    return arg


def setdefault(val, default, cond=None):
    """Return val if condition is met, otherwise default.

    If cond is None, then it checks if val is not None, returning val
    or default accordingly.
    """
    return val if cond or (cond is None and val is not None) else default


def isnonnegint(x):
    """Determine if x is a non-negative integer."""
    try:
        _ = operator.index(x)
    except TypeError:
        return False
    return x >= 0


def isposint(x):
    """Determine if x is a strictly positive integer."""
    return isnonnegint(x) and (x > 0)


def errorif(cond, err=ValueError, msg=""):
    """Raise an error if condition is met.

    Similar to assert but allows wider range of Error types, rather than
    just AssertionError.
    """
    if cond:
        raise err(colored(msg, "red"))


class ResolutionWarning(UserWarning):
    """Warning for insufficient resolution."""

    pass


def warnif(cond, err=UserWarning, msg=""):
    """Throw a warning if condition is met."""
    if cond:
        warnings.warn(colored(msg, "yellow"), err)


def check_nonnegint(x, name="", allow_none=True):
    """Throw an error if x is not a non-negative integer."""
    if allow_none:
        errorif(
            not ((x is None) or isnonnegint(x)),
            ValueError,
            f"{name} should be a non-negative integer or None, got {x}",
        )
    else:
        errorif(
            not isnonnegint(x),
            ValueError,
            f"{name} should be a non-negative integer, got {x}",
        )
    return x


def check_posint(x, name="", allow_none=True):
    """Throw an error if x is not a positive integer."""
    if allow_none:
        errorif(
            not ((x is None) or isposint(x)),
            ValueError,
            f"{name} should be a positive integer or None, got {x}",
        )
    else:
        errorif(
            not isposint(x), ValueError, f"{name} should be a positive integer, got {x}"
        )
    return x


def only1(*args):
    """Return True if 1 and only 1 of args evaluates to True."""
    # copied from https://stackoverflow.com/questions/16801322/
    i = iter(args)
    return any(i) and not any(i)


def unique_list(thelist):
    """Get the unique elements from a list, and indices to recreate it.

    Parameters
    ----------
    thelist : list
        List to get unique elements from.

    Returns
    -------
    unique : list
        Unique elements from the input.
    inds : list of int
        Indices of unique elements in original list, such that
        unique[inds[i]] == thelist[i]
    """
    inds = []
    unique = []
    for i, x in enumerate(thelist):
        if x not in unique:
            unique.append(x)
        inds.append(unique.index(x))
    return unique, inds


def is_any_instance(things, cls):
    """Check if any of things is an instance of cls."""
    return any([isinstance(t, cls) for t in things])


def broadcast_tree(tree_in, tree_out, dtype=int):
    """Broadcast tree_in to the same pytree structure as tree_out.

    Both trees must be nested lists of dicts with string keys and array values.
    Or the values can be bools, where False broadcasts to an empty array and True
    broadcasts to the corresponding array from tree_out.

    Parameters
    ----------
    tree_in : pytree
        Tree to broadcast.
    tree_out : pytree
        Tree with structure to broadcast to.
    dtype : optional
        Data type of array values. Default = int.

    Returns
    -------
    tree : pytree
        Tree with the leaves of tree_in broadcast to the structure of tree_out.

    """
    # both trees at leaf layer
    if isinstance(tree_in, dict) and isinstance(tree_out, dict):
        tree_new = {}
        for key, value in tree_in.items():
            errorif(
                key not in tree_out.keys(),
                ValueError,
                f"dict key '{key}' of tree_in must be a subset of those in tree_out: "
                + f"{list(tree_out.keys())}",
            )
            if isinstance(value, bool):
                if value:
                    tree_new[key] = np.atleast_1d(tree_out[key]).astype(dtype=dtype)
                else:
                    tree_new[key] = np.array([], dtype=dtype)
            else:
                tree_new[key] = np.atleast_1d(value).astype(dtype=dtype)
        for key, value in tree_out.items():
            if key not in tree_new.keys():
                tree_new[key] = np.array([], dtype=dtype)
            errorif(
                not np.all(np.isin(tree_new[key], value)),
                ValueError,
                f"dict value {tree_new[key]} of tree_in must be a subset "
                + f"of those in tree_out: {value}",
            )
        return tree_new
    # tree_out is deeper than tree_in
    elif isinstance(tree_in, dict) and isinstance(tree_out, list):
        return [broadcast_tree(tree_in.copy(), branch) for branch in tree_out]
    # both trees at branch layer
    elif isinstance(tree_in, list) and isinstance(tree_out, list):
        errorif(
            len(tree_in) != len(tree_out),
            ValueError,
            "tree_in must have the same number of branches as tree_out",
        )
        return [broadcast_tree(tree_in[k], tree_out[k]) for k in range(len(tree_out))]
    # tree_in is deeper than tree_out
    elif isinstance(tree_in, list) and isinstance(tree_out, dict):
        raise ValueError("tree_in cannot have a deeper structure than tree_out")
    # invalid tree structure
    else:
        raise ValueError("trees must be nested lists of dicts")


@functools.partial(
    jnp.vectorize, signature="(m),(m)->(n)", excluded={"size", "fill_value"}
)
def take_mask(a, mask, /, *, size=None, fill_value=None):
    """JIT compilable method to return ``a[mask][:size]`` padded by ``fill_value``.

    Parameters
    ----------
    a : jnp.ndarray
        The source array.
    mask : jnp.ndarray
        Boolean mask to index into ``a``. Should have same shape as ``a``.
    size : int
        Elements of ``a`` at the first size True indices of ``mask`` will be returned.
        If there are fewer elements than size indicates, the returned array will be
        padded with ``fill_value``. The size default is ``mask.size``.
    fill_value : Any
        When there are fewer than the indicated number of elements, the remaining
        elements will be filled with ``fill_value``. Defaults to NaN for inexact types,
        the largest negative value for signed types, the largest positive value for
        unsigned types, and True for booleans.

    Returns
    -------
    result : jnp.ndarray
        Shape (size, ).

    """
    assert a.shape == mask.shape
    idx = flatnonzero(mask, size=setdefault(size, mask.size), fill_value=mask.size)
    return take(
        a,
        idx,
        mode="fill",
        fill_value=fill_value,
        unique_indices=True,
        indices_are_sorted=True,
    )


def flatten_matrix(y):
    """Flatten matrix to vector."""
    return y.reshape(*y.shape[:-2], -1)


# TODO: Eventually remove and use numpy's stuff.
# https://github.com/numpy/numpy/issues/25805
def atleast_nd(ndmin, ary):
    """Adds dimensions to front if necessary."""
    return jnp.array(ary, ndmin=ndmin) if jnp.ndim(ary) < ndmin else ary


def jaxify(func, abstract_eval, vectorized=False, abs_step=1e-4, rel_step=0):
    """Make an external (python) function work with JAX.

    Positional arguments to func can be differentiated,
    use keyword args for static values and non-differentiable stuff.

    Note: Only forward mode differentiation is supported currently.

    Parameters
    ----------
    func : callable
        Function to wrap. Should be a "pure" function, in that it has no side
        effects and doesn't maintain state. Does not need to be JAX transformable.
    abstract_eval : callable
        Auxiliary function that computes the output shape and dtype of func.
        **Must be JAX transformable**. Should be of the form

            abstract_eval(*args, **kwargs) -> Pytree with same shape and dtype as
            func(*args, **kwargs)

        For example, if func always returns a scalar:

            abstract_eval = lambda *args, **kwargs: jnp.array(1.)

        Or if func takes an array of shape(n) and returns a dict of arrays of
        shape(n-2):

            abstract_eval = lambda arr, **kwargs:
            {"out1": jnp.empty(arr.size-2), "out2": jnp.empty(arr.size-2)}
    vectorized : bool, optional
        Whether or not the wrapped function is vectorized. Default = False.
    abs_step : float, optional
        Absolute finite difference step size. Default = 1e-4.
        Total step size is ``abs_step + rel_step * mean(abs(x))``.
    rel_step : float, optional
        Relative finite difference step size. Default = 0.
        Total step size is ``abs_step + rel_step * mean(abs(x))``.

    Returns
    -------
    func : callable
        New function that behaves as func but works with jit/vmap/jacfwd etc.

    """

    def wrap_pure_callback(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result_shape_dtype = abstract_eval(*args, **kwargs)
            return pure_callback(
                func, result_shape_dtype, *args, vectorized=vectorized, **kwargs
            )

        return wrapper

    def define_fd_jvp(func):
        func = jax.custom_jvp(func)

        @func.defjvp
        def func_jvp(primals, tangents):
            primal_out = func(*primals)

            # flatten everything into 1D vectors for easier finite differences
            y, unflaty = jax.flatten_util.ravel_pytree(primal_out)
            x, unflatx = jax.flatten_util.ravel_pytree(primals)
            v, _______ = jax.flatten_util.ravel_pytree(tangents)

            # finite difference step size
            fd_step = abs_step + rel_step * jnp.mean(jnp.abs(x))

            # scale tangents to unit norm if nonzero
            normv = jnp.linalg.norm(v)
            vh = jnp.where(normv == 0, v, v / normv)

            def f(x):
                return jax.flatten_util.ravel_pytree(func(*unflatx(x)))[0]

            tangent_out = (f(x + fd_step * vh) - y) / fd_step * normv
            tangent_out = unflaty(tangent_out)

            return primal_out, tangent_out

        return func

    return define_fd_jvp(wrap_pure_callback(func))


def atleast_3d_mid(ary):
    """Like np.atleast_3d but if adds dim at axis 1 for 2d arrays."""
    ary = jnp.atleast_2d(ary)
    return ary[:, jnp.newaxis] if ary.ndim == 2 else ary


def atleast_2d_end(ary):
    """Like np.atleast_2d but if adds dim at axis 1 for 1d arrays."""
    ary = jnp.atleast_1d(ary)
    return ary[:, jnp.newaxis] if ary.ndim == 1 else ary


def dot(a, b, axis=-1):
    """Batched vector dot product.

    Parameters
    ----------
    a : array-like
        First array of vectors.
    b : array-like
        Second array of vectors.
    axis : int
        Axis along which vectors are stored.

    Returns
    -------
    y : array-like
        y = sum(a*b, axis=axis)

    """
    return jnp.sum(a * b, axis=axis, keepdims=False)


def cross(a, b, axis=-1):
    """Batched vector cross product.

    Parameters
    ----------
    a : array-like
        First array of vectors.
    b : array-like
        Second array of vectors.
    axis : int
        Axis along which vectors are stored.

    Returns
    -------
    y : array-like
        y = a x b

    """
    return jnp.cross(a, b, axis=axis)


def safenorm(x, ord=None, axis=None, fill=0, threshold=0):
    """Like jnp.linalg.norm, but without nan gradient at x=0.

    Parameters
    ----------
    x : ndarray
        Vector or array to norm.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of norm.
    axis : {None, int, 2-tuple of ints}, optional
        Axis to take norm along.
    fill : float, ndarray, optional
        Value to return where x is zero.
    threshold : float >= 0
        How small is x allowed to be.

    """
    is_zero = (jnp.abs(x) <= threshold).all(axis=axis, keepdims=True)
    y = jnp.where(is_zero, jnp.ones_like(x), x)  # replace x with ones if is_zero
    n = jnp.linalg.norm(y, ord=ord, axis=axis)
    n = jnp.where(is_zero.squeeze(), fill, n)  # replace norm with zero if is_zero
    return n


def safenormalize(x, ord=None, axis=None, fill=0, threshold=0):
    """Normalize a vector to unit length, but without nan gradient at x=0.

    Parameters
    ----------
    x : ndarray
        Vector or array to norm.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of norm.
    axis : {None, int, 2-tuple of ints}, optional
        Axis to take norm along.
    fill : float, ndarray, optional
        Value to return where x is zero.
    threshold : float >= 0
        How small is x allowed to be.

    """
    is_zero = (jnp.abs(x) <= threshold).all(axis=axis, keepdims=True)
    y = jnp.where(is_zero, jnp.ones_like(x), x)  # replace x with ones if is_zero
    n = safenorm(x, ord, axis, fill, threshold) * jnp.ones_like(x)
    # return unit vector with equal components if norm <= threshold
    return jnp.where(n <= threshold, jnp.ones_like(y) / jnp.sqrt(y.size), y / n)


def safediv(a, b, fill=0, threshold=0):
    """Divide a/b with guards for division by zero.

    Parameters
    ----------
    a, b : ndarray
        Numerator and denominator.
    fill : float, ndarray, optional
        Value to return where b is zero.
    threshold : float >= 0
        How small is b allowed to be.
    """
    mask = jnp.abs(b) <= threshold
    num = jnp.where(mask, fill, a)
    den = jnp.where(mask, 1, b)
    return num / den


def ensure_tuple(x):
    """Returns x as a tuple of arrays."""
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)
