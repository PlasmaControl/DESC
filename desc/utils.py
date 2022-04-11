import numpy as np
import warnings
from termcolor import colored
from shapely.geometry import Polygon


# Helper Classes -----------------------------------------------------------------------


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


# Helper Functions -----------------------------------------------------------


def unpack_state(x, nR, nZ):
    """Unpack the state vector x into R_lmn, Z_lmn, L_lmn components.
    Parameters
    ----------
    x : ndarray
        vector to unpack: x = [cR, cZ, cL]
    nR : int
        number of R_lmn coefficients
    nZ : int
        number of Z_lmn coefficients
    Returns
    -------
    R_lmn : ndarray
        spectral coefficients of R
    Z_lmn : ndarray
        spectral coefficients of Z
    L_lmn : ndarray
        spectral coefficients of lambda
    """
    R_lmn = x[:nR]
    Z_lmn = x[nR : nR + nZ]
    L_lmn = x[nR + nZ :]
    return R_lmn, Z_lmn, L_lmn


def area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2):
    """Compute area difference between coordinate curves

    Parameters
    ----------
    args : ndarray
        R and Z coordinates of constant rho (r) or vartheta (v) contours.
        Arrays should be indexed as [rho,theta,zeta]

    Returns
    -------
    area_rho : ndarray, shape(Nr, Nz)
        normalized area difference of rho contours, computed as the symmetric
        difference divided by the intersection
    area_theta : ndarray, shape(Nt, Nz)
        normalized area difference between vartheta contours, computed as the area
        of the polygon created by closing the two vartheta contours divided by the
        perimeter squared
    """
    assert Rr1.shape == Rr2.shape == Zr1.shape == Zr2.shape
    assert Rv1.shape == Rv2.shape == Zv1.shape == Zv2.shape

    poly_r1 = np.array(
        [
            [Polygon(np.array([R, Z]).T) for R, Z in zip(Rr1[:, :, i], Zr1[:, :, i])]
            for i in range(Rr1.shape[2])
        ]
    )
    poly_r2 = np.array(
        [
            [Polygon(np.array([R, Z]).T) for R, Z in zip(Rr2[:, :, i], Zr2[:, :, i])]
            for i in range(Rr2.shape[2])
        ]
    )
    poly_v = np.array(
        [
            [
                Polygon(np.array([R, Z]).T)
                for R, Z in zip(
                    np.hstack([Rv1[:, :, i].T, Rv2[::-1, :, i].T]),
                    np.hstack([Zv1[:, :, i].T, Zv2[::-1, :, i].T]),
                )
            ]
            for i in range(Rv1.shape[2])
        ]
    )

    diff_rho = np.array(
        [
            poly1.symmetric_difference(poly2).area
            for poly1, poly2 in zip(poly_r1.flat, poly_r2.flat)
        ]
    ).reshape((Rr1.shape[2], Rr1.shape[0]))
    intersect_rho = np.array(
        [
            poly1.intersection(poly2).area
            for poly1, poly2 in zip(poly_r1.flat, poly_r2.flat)
        ]
    ).reshape((Rr1.shape[2], Rr1.shape[0]))
    area_rho = np.where(diff_rho > 0, diff_rho / intersect_rho, 0)
    area_theta = np.array(
        [poly.area / (poly.length) ** 2 for poly in poly_v.flat]
    ).reshape((Rv1.shape[1], Rv1.shape[2]))
    return area_rho, area_theta


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


def flatten_list(x):
    """Flatten a nested list.

    Parameters
    ----------
    x : list
        nested list of lists to flatten

    Returns
    -------
    x : list
        flattened input

    """
    if isinstance(x, list):
        return [a for i in x for a in flatten_list(i)]
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
    issorted : bool
        whether the array is sorted along specified axis

    """
    x = np.asarray(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    return np.all(np.diff(x, axis=axis) >= -tol)


def isalmostequal(x, axis=-1, tol=1e-12):
    """Check if all values of an array are equal, to within a given tolerance.

    Parameters
    ----------
    x : array-like
        input values
    axis : int
        axis along which to make comparison. If None, the flattened array is used
    tol : float
        tolerance for comparison.
        Array is considered equal if std(x)*len(x)< tol along axis

    Returns
    -------
    isalmostequal : bool
        whether the array is equal along specified axis

    """
    x = np.asarray(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    return np.all(x.std(axis=axis) * x.shape[axis] < tol)


def islinspaced(x, axis=-1, tol=1e-12):
    """Check if all values of an array are linearly spaced, to within a given tolerance.

    Parameters
    ----------
    x : array-like
        input values
    axis : int
        axis along which to make comparison. If None, the flattened array is used
    tol : float
        tolerance for comparison.
        Array is considered linearly spaced if std(diff(x)) < tol along axis

    Returns
    -------
    islinspaced : bool
        whether the array is linearly spaced along specified axis

    """
    x = np.asarray(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    return np.all(np.diff(x, axis=axis).std() < tol)


def copy_coeffs(c_old, modes_old, modes_new, c_new=None):
    """Copy coefficients from one resolution to another."""
    modes_old, modes_new = np.atleast_1d(modes_old), np.atleast_1d(modes_new)
    if modes_old.ndim == 1:
        modes_old = modes_old.reshape((-1, 1))
    if modes_new.ndim == 1:
        modes_new = modes_new.reshape((-1, 1))

    num_modes = modes_new.shape[0]
    if c_new is None:
        c_new = np.zeros((num_modes,))

    for i in range(num_modes):
        idx = np.where((modes_old == modes_new[i, :]).all(axis=1))[0]
        if len(idx):
            c_new[i] = c_old[idx]
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
