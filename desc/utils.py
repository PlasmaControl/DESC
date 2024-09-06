"""Utility functions, independent of the rest of DESC."""

import operator
import warnings
from functools import partial
from itertools import combinations_with_replacement, permutations
from typing import Callable, Optional

import numpy as np
from scipy.special import factorial
from termcolor import colored

from desc.backend import flatnonzero, fori_loop, functools, jax, jit, jnp, take

if jax.__version_info__ >= (0, 4, 16):
    from jax.extend import linear_util as lu
else:
    from jax import linear_util as lu

from jax._src.numpy.vectorize import (
    _apply_excluded,
    _check_output_dims,
    _parse_gufunc_signature,
    _parse_input_dimensions,
)


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
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    K = min(M, N)
    rcond = np.finfo(A.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    large = s > tol
    num = np.sum(large, dtype=int)
    uk = u[:, :K]
    vhk = vh[:K, :]
    s = np.divide(1, s, where=large, out=s)
    s[(~large,)] = 0
    Ainv = np.matmul(vhk.T, np.multiply(s[..., np.newaxis], uk.T))
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


# The following section of this code is derived from the NetKet project
# https://github.com/netket/netket/blob/9881c9fb217a2ac4dc9274a054bf6e6a2993c519/
# netket/jax/_chunk_utils.py
#
# The original copyright notice is as follows
# Copyright 2021 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");


def _treeify(f):
    def _f(x, *args, **kwargs):
        return jax.tree_util.tree_map(lambda y: f(y, *args, **kwargs), x)

    return _f


@_treeify
def _unchunk(x):
    return x.reshape((-1,) + x.shape[2:])


@_treeify
def _chunk(x, chunk_size=None):
    # chunk_size=None -> add just a dummy chunk dimension, same as np.expand_dims(x, 0)
    if x.ndim == 0:
        raise ValueError("x cannot be chunked as it has 0 dimensions.")
    n = x.shape[0]
    if chunk_size is None:
        chunk_size = n

    n_chunks, residual = divmod(n, chunk_size)
    if residual != 0:
        raise ValueError(
            "The first dimension of x must be divisible by chunk_size."
            + f"\n            Got x.shape={x.shape} but chunk_size={chunk_size}."
        )
    return x.reshape((n_chunks, chunk_size) + x.shape[1:])


def _chunk_size(x):
    b = set(map(lambda x: x.shape[:2], jax.tree_util.tree_leaves(x)))
    if len(b) != 1:
        raise ValueError(
            "The arrays in x have inconsistent chunk_size or number of chunks"
        )
    return b.pop()[1]


def unchunk(x_chunked):
    """Merge the first two axes of an array (or a pytree of arrays).

    Parameters
    ----------
    x_chunked: an array (or pytree of arrays) of at least 2 dimensions

    Returns
    -------
    (x, chunk_fn) : tuple
        where x is x_chunked reshaped to (-1,)+x.shape[2:]
        and chunk_fn is a function which restores x given x_chunked

    """
    return _unchunk(x_chunked), functools.partial(
        _chunk, chunk_size=_chunk_size(x_chunked)
    )


def chunk(x, chunk_size=None):
    """Split an array (or a pytree of arrays) into chunks along the first axis.

    Parameters
    ----------
        x: an array (or pytree of arrays)
        chunk_size: an integer or None (default)
            The first axis in x must be a multiple of chunk_size

    Returns
    -------
    (x_chunked, unchunk_fn): tuple
        - x_chunked is x reshaped to (-1, chunk_size)+x.shape[1:]
          if chunk_size is None then it defaults to x.shape[0], i.e. just one chunk
        - unchunk_fn is a function which restores x given x_chunked
    """
    return _chunk(x, chunk_size), _unchunk


####

# The following section of this code is derived from the NetKet project
# https://github.com/netket/netket/blob/9881c9fb217a2ac4dc9274a054bf6e6a2993c519/
# netket/jax/_scanmap.py
#
# The original copyright notice is as follows
# Copyright 2021 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

_tree_add = functools.partial(jax.tree_util.tree_map, jax.lax.add)
_tree_zeros_like = functools.partial(
    jax.tree_util.tree_map, lambda x: jnp.zeros(x.shape, dtype=x.dtype)
)


# TODO put it somewhere
def _multimap(f, *args):
    try:
        return tuple(map(lambda a: f(*a), zip(*args)))
    except TypeError:
        return f(*args)


def scan_append_reduce(f, x, append_cond, op=_tree_add, zero_fun=_tree_zeros_like):
    """Evaluate f element by element in x while appending and/or reducing the results.

    Parameters
    ----------
        f: a function that takes elements of the leading dimension of x
        x: a pytree where each leaf array has the same leading dimension
        append_cond: a bool (if f returns just one result) or a tuple of
                     bools (if f returns multiple values)
                     which indicates whether the individual result should
                     be appended or reduced
        op: a function to (pairwise) reduce the specified results. Defaults to a sum.
        zero_fun: a function which prepares the zero element of op for a given input
                  shape/dtype tree. Defaults to zeros.

    Returns
    -------
        The (tuple of) results corresponding to the output of f
        where each result is given by:
        if append_cond is True:
            a (pytree of) array(s) with leading dimension same as x,
            containing the evaluation of f at each element in x
        else (append_cond is False):
            a (pytree of) array(s) with the same shape as the corresponding
            output of f, containing the reduction over op of f evaluated at each x


    Example:

        import jax.numpy as jnp
        from netket.jax import scan_append_reduce

        def f(x):
             y = jnp.sin(x)
             return y, y, y**2

        N = 100
        x = jnp.linspace(0.,jnp.pi,N)

        y, s, s2 = scan_append_reduce(f, x, (True, False, False))
        mean = s/N
        var = s2/N - mean**2
    """
    # TODO: different op for each result

    x0 = jax.tree_util.tree_map(lambda x: x[0], x)

    # special code path if there is only one element
    # to avoid having to rely on xla/llvm to optimize the overhead away
    if jax.tree_util.tree_leaves(x)[0].shape[0] == 1:
        return _multimap(
            lambda c, x: jnp.expand_dims(x, 0) if c else x, append_cond, f(x0)
        )

    # the original idea was to use pytrees,
    # however for now just operate on the return value tuple
    _get_append_part = functools.partial(
        _multimap, lambda c, x: x if c else None, append_cond
    )
    _get_op_part = functools.partial(
        _multimap, lambda c, x: x if not c else None, append_cond
    )
    _tree_select = functools.partial(
        _multimap, lambda c, t1, t2: t1 if c else t2, append_cond
    )

    carry_init = True, _get_op_part(zero_fun(jax.eval_shape(f, x0)))

    def f_(carry, x):
        is_first, y_carry = carry
        y = f(x)
        y_op = _get_op_part(y)
        y_append = _get_append_part(y)
        y_reduce = op(y_carry, y_op)
        return (False, y_reduce), y_append

    (_, res_op), res_append = jax.lax.scan(f_, carry_init, x, unroll=1)
    # reconstruct the result from the reduced and appended parts in the two trees
    return _tree_select(res_append, res_op)


scan_append = functools.partial(scan_append_reduce, append_cond=True)
scan_reduce = functools.partial(scan_append_reduce, append_cond=False)


# TODO in_axes a la vmap?
def _scanmap(fun, scan_fun, argnums=0):
    """A helper function to wrap f with a scan_fun.

    Example
    -------
        import jax.numpy as jnp
        from functools import partial

        from desc.utils import _scanmap, scan_append_reduce

        scan_fun = partial(scan_append_reduce, append_cond=(True, False, False))

        @partial(_scanmap, scan_fun=scan_fun, argnums=1)
        def f(c, x):
             y = jnp.sin(x) + c
             return y, y, y**2

        N = 100
        x = jnp.linspace(0.,jnp.pi,N)
        c = 1.


        y, s, s2 = f(c, x)
        mean = s/N
        var = s2/N - mean**2
    """

    def f_(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = jax.api_util.argnums_partial(
            f, argnums, args, require_static_args_hashable=False
        )
        return scan_fun(lambda x: f_partial.call_wrapped(*x), dyn_args)

    return f_


# The following section of this code is derived from the NetKet project
# https://github.com/netket/netket/blob/9881c9fb217a2ac4dc9274a054bf6e6a2993c519/
# netket/jax/_vmap_chunked.py
#
# The original copyright notice is as follows
# Copyright 2021 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");


def _eval_fun_in_chunks(vmapped_fun, chunk_size, argnums, *args, **kwargs):
    n_elements = jax.tree_util.tree_leaves(args[argnums[0]])[0].shape[0]
    n_chunks, n_rest = divmod(n_elements, chunk_size)

    if n_chunks == 0 or chunk_size >= n_elements:
        y = vmapped_fun(*args, **kwargs)
    else:
        # split inputs
        def _get_chunks(x):
            x_chunks = jax.tree_util.tree_map(
                lambda x_: x_[: n_elements - n_rest, ...], x
            )
            x_chunks = _chunk(x_chunks, chunk_size)
            return x_chunks

        def _get_rest(x):
            x_rest = jax.tree_util.tree_map(
                lambda x_: x_[n_elements - n_rest :, ...], x
            )
            return x_rest

        args_chunks = [
            _get_chunks(a) if i in argnums else a for i, a in enumerate(args)
        ]
        args_rest = [_get_rest(a) if i in argnums else a for i, a in enumerate(args)]

        y_chunks = _unchunk(
            _scanmap(vmapped_fun, scan_append, argnums)(*args_chunks, **kwargs)
        )

        if n_rest == 0:
            y = y_chunks
        else:
            y_rest = vmapped_fun(*args_rest, **kwargs)
            y = jax.tree_util.tree_map(
                lambda y1, y2: jnp.concatenate((y1, y2)), y_chunks, y_rest
            )
    return y


def _chunk_vmapped_function(
    vmapped_fun: Callable,
    chunk_size: Optional[int],
    argnums=0,
) -> Callable:
    """Takes a vmapped function and computes it in chunks."""
    if chunk_size is None:
        return vmapped_fun

    if isinstance(argnums, int):
        argnums = (argnums,)
    return functools.partial(_eval_fun_in_chunks, vmapped_fun, chunk_size, argnums)


def _parse_in_axes(in_axes):
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    if not set(in_axes).issubset((0, None)):
        raise NotImplementedError("Only in_axes 0/None are currently supported")

    argnums = tuple(
        map(lambda ix: ix[0], filter(lambda ix: ix[1] is not None, enumerate(in_axes)))
    )
    return in_axes, argnums


def apply_chunked(
    f: Callable,
    in_axes=0,
    *,
    chunk_size: Optional[int],
) -> Callable:
    """Compute f in smaller chunks over axis 0.

    Takes an implicitly vmapped function over the axis 0 and uses scan to
    do the computations in smaller chunks over the 0-th axis of all input arguments.

    For this to work, the function `f` should be `vectorized` along the `in_axes`
    of the arguments. This means that the function `f` should respect the following
    condition:

    .. code-block:: python

        assert f(x) == jnp.concatenate([f(x_i) for x_i in x], axis=0)

    which is automatically satisfied if `f` is obtained by vmapping a function,
    such as:

    .. code-block:: python

        f = jax.vmap(f_orig)


    Parameters
    ----------
        f: A function that satisfies the condition above
        in_axes: The axes that should be scanned along. Only supports `0` or `None`
        chunk_size: The maximum size of the chunks to be used. If it is `None`,
           chunking is disabled

    """
    _, argnums = _parse_in_axes(in_axes)
    return _chunk_vmapped_function(
        f,
        chunk_size,
        argnums,
    )


def vmap_chunked(
    f: Callable,
    in_axes=0,
    *,
    chunk_size: Optional[int],
) -> Callable:
    """Behaves like jax.vmap but uses scan to chunk the computations in smaller chunks.

    This function is essentially equivalent to:

    .. code-block:: python

        nk.jax.apply_chunked(jax.vmap(f, in_axes), in_axes, chunk_size)

    Some limitations to `in_axes` apply.

    Parameters
    ----------
        f: The function to be vectorised.
        in_axes: The axes that should be scanned along. Only supports `0` or `None`
        chunk_size: The maximum size of the chunks to be used. If it is `None`,
            chunking is disabled


    Returns
    -------
        f: A vectorised and chunked function
    """
    in_axes, argnums = _parse_in_axes(in_axes)
    vmapped_fun = jax.vmap(f, in_axes=in_axes)
    return _chunk_vmapped_function(vmapped_fun, chunk_size, argnums)


def batched_vectorize(pyfunc, *, excluded=frozenset(), signature=None, chunk_size=None):
    """Define a vectorized function with broadcasting and batching.

    below is taken from JAX
    FIXME: change restof docstring
    :func:`vectorize` is a convenience wrapper for defining vectorized
    functions with broadcasting, in the style of NumPy's
    `generalized universal functions
    <https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html>`_.
    It allows for defining functions that are automatically repeated across
    any leading dimensions, without the implementation of the function needing to
    be concerned about how to handle higher dimensional inputs.

    :func:`jax.numpy.vectorize` has the same interface as
    :class:`numpy.vectorize`, but it is syntactic sugar for an auto-batching
    transformation (:func:`vmap`) rather than a Python loop. This should be
    considerably more efficient, but the implementation must be written in terms
    of functions that act on JAX arrays.

    Parameters
    ----------
        pyfunc: function to vectorize.
        excluded: optional set of integers representing positional arguments for
        which the function will not be vectorized. These will be passed directly
        to ``pyfunc`` unmodified.
        signature: optional generalized universal function signature, e.g.,
        ``(m,n),(n)->(m)`` for vectorized matrix-vector multiplication. If
        provided, ``pyfunc`` will be called with (and expected to return) arrays
        with shapes given by the size of corresponding core dimensions. By
        default, pyfunc is assumed to take scalars arrays as input and output.
        chunk_size: the size of the batches to pass to vmap. if 1, will only

    Returns
    -------
        Vectorized version of the given function.

    """
    if any(not isinstance(exclude, (str, int)) for exclude in excluded):
        raise TypeError(
            "jax.numpy.vectorize can only exclude integer or string arguments, "
            "but excluded={!r}".format(excluded)
        )
    if any(isinstance(e, int) and e < 0 for e in excluded):
        raise ValueError(f"excluded={excluded!r} contains negative numbers")

    @functools.wraps(pyfunc)
    def wrapped(*args, **kwargs):
        error_context = (
            "on vectorized function with excluded={!r} and "
            "signature={!r}".format(excluded, signature)
        )
        excluded_func, args, kwargs = _apply_excluded(pyfunc, excluded, args, kwargs)

        if signature is not None:
            input_core_dims, output_core_dims = _parse_gufunc_signature(signature)
        else:
            input_core_dims = [()] * len(args)
            output_core_dims = None

        none_args = {i for i, arg in enumerate(args) if arg is None}
        if any(none_args):
            if any(input_core_dims[i] != () for i in none_args):
                raise ValueError(
                    f"Cannot pass None at locations {none_args} with {signature=}"
                )
            excluded_func, args, _ = _apply_excluded(excluded_func, none_args, args, {})
            input_core_dims = [
                dim for i, dim in enumerate(input_core_dims) if i not in none_args
            ]

        args = tuple(map(jnp.asarray, args))

        broadcast_shape, dim_sizes = _parse_input_dimensions(
            args, input_core_dims, error_context
        )

        checked_func = _check_output_dims(
            excluded_func, dim_sizes, output_core_dims, error_context
        )

        # Rather than broadcasting all arguments to full broadcast shapes, prefer
        # expanding dimensions using vmap. By pushing broadcasting
        # into vmap, we can make use of more efficient batching rules for
        # primitives where only some arguments are batched (e.g., for
        # lax_linalg.triangular_solve), and avoid instantiating large broadcasted
        # arrays.

        squeezed_args = []
        rev_filled_shapes = []

        for arg, core_dims in zip(args, input_core_dims):
            noncore_shape = arg.shape[: arg.ndim - len(core_dims)]

            pad_ndim = len(broadcast_shape) - len(noncore_shape)
            filled_shape = pad_ndim * (1,) + noncore_shape
            rev_filled_shapes.append(filled_shape[::-1])

            squeeze_indices = tuple(
                i for i, size in enumerate(noncore_shape) if size == 1
            )
            squeezed_arg = jnp.squeeze(arg, axis=squeeze_indices)
            squeezed_args.append(squeezed_arg)

        vectorized_func = checked_func
        dims_to_expand = []
        for negdim, axis_sizes in enumerate(zip(*rev_filled_shapes)):
            in_axes = tuple(None if size == 1 else 0 for size in axis_sizes)
            if all(axis is None for axis in in_axes):
                dims_to_expand.append(len(broadcast_shape) - 1 - negdim)
            else:
                # change the vmap here to chunked_vmap
                vectorized_func = vmap_chunked(
                    vectorized_func, in_axes, chunk_size=chunk_size
                )
        result = vectorized_func(*squeezed_args)

        if not dims_to_expand:
            return result
        elif isinstance(result, tuple):
            return tuple(jnp.expand_dims(r, axis=dims_to_expand) for r in result)
        else:
            return jnp.expand_dims(result, axis=dims_to_expand)

    return wrapped


@partial(jnp.vectorize, signature="(m),(m)->(n)", excluded={"size", "fill_value"})
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


PRINT_WIDTH = 60  # current longest name is BootstrapRedlConsistency with pre-text


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


def cumtrapz(y, x=None, dx=1.0, axis=-1, initial=None):
    """Cumulatively integrate y(x) using the composite trapezoidal rule.

    Taken from SciPy, but changed NumPy references to JAX.NumPy:
        https://github.com/scipy/scipy/blob/v1.10.1/scipy/integrate/_quadrature.py

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        The coordinate to integrate along. If None (default), use spacing `dx`
        between consecutive elements in `y`.
    dx : float, optional
        Spacing between elements of `y`. Only used if `x` is None.
    axis : int, optional
        Specifies the axis to cumulate. Default is -1 (last axis).
    initial : scalar, optional
        If given, insert this value at the beginning of the returned result.
        Typically, this value should be 0. Default is None, which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    """
    y = jnp.asarray(y)
    if x is None:
        d = dx
    else:
        x = jnp.asarray(x)
        if x.ndim == 1:
            d = jnp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the " "same as y.")
        else:
            d = jnp.diff(x, axis=axis)

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError(
                "If given, length of x along axis must be the " "same as y."
            )

    def tupleset(t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)

    nd = len(y.shape)
    slice1 = tupleset((slice(None),) * nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),) * nd, axis, slice(None, -1))
    res = jnp.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)

    if initial is not None:
        if not jnp.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = jnp.concatenate(
            [jnp.full(shape, initial, dtype=res.dtype), res], axis=axis
        )

    return res
