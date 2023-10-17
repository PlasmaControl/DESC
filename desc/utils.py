"""Utility functions, independent of the rest of DESC."""

import numbers
import warnings
from itertools import combinations_with_replacement, permutations

import numpy as np
from scipy.special import factorial
from termcolor import colored

from desc.backend import fori_loop, jit, jnp


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
    if hasattr(a, "eq"):
        return a.eq(b)
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
    modes_old, modes_new = jnp.atleast_1d(modes_old), jnp.atleast_1d(modes_new)

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
    num = factorial(n)
    den = factorial(k).prod(axis=-1)
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
    """Get thing from a collection of things that is the instance of a given class."""
    return [t for t in things if isinstance(t, cls)][0]


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
    return isinstance(x, numbers.Real) and (x == int(x)) and (x >= 0)


def isposint(x):
    """Determine if x is a strictly positive integer."""
    return isinstance(x, numbers.Real) and (x == int(x)) and (x > 0)


def errorif(cond, err=ValueError, msg=""):
    """Raise an error if condition is met.

    Similar to assert but allows wider range of Error types, rather than
    just AssertionError.
    """
    if cond:
        raise err(msg)


def warnif(cond, err=UserWarning, msg=""):
    """Throw a warning if condition is met."""
    if cond:
        warnings.warn(msg, err)


def only1(*args):
    """Return True if 1 and only 1 of args evaluates to True."""
    # copied from https://stackoverflow.com/questions/16801322/
    i = iter(args)
    return any(i) and not any(i)
