import numpy as np
import functools
import warnings
from termcolor import colored
from desc.backend import jnp


# Helper Classes -------------------------------------------------------------

class Timer():
    """Simple object for organizing timing info

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

    Returns
    -------

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
                warnings.warn(colored(
                              'nanosecond timing not available on this system,'
                              + ' reverting to microsecond timing', 'yellow'))

        else:
            self.op = time.perf_counter

    def start(self, name):
        """Starts a timer

        Parameters
        ----------
        name : str
            name to associate with timer

        Returns
        -------

        """

        self._timers[name] = [self.op()]

    def stop(self, name):
        """Stops a running timer:

        Parameters
        ----------
        name : str
            name of timer to stop

        Returns
        -------

        Raises
        ------
        ValueError
            if timer 'name' has not been started

        """

        try:
            self._timers[name].append(self.op())
        except KeyError:
            raise ValueError(
                colored("timer '{}' has not been started".format(name), 'red')) from None
        self._times[name] = np.diff(self._timers[name])[0]
        if self._ns:
            self._times[name] = self._times[name]/1e9
        del self._timers[name]

    @staticmethod
    def pretty_print(name, time):
        """Pretty prints time interval

        Does not modify or use any internal timer data,
        this is just a helper for pretty printing arbitrary time data

        Parameters
        ----------
        name : str
            text to print before time
        time : float
            time (in seconds) to print

        Returns
        -------

        """
        us = time*1e6
        ms = us / 1000
        sec = ms / 1000
        mins = sec / 60
        hrs = mins / 60

        if us < 100:
            out = '{:.3f}'.format(us)[:4] + ' us'
        elif us < 1000:
            out = '{:.3f}'.format(us)[:3] + ' us'
        elif ms < 100:
            out = '{:.3f}'.format(ms)[:4] + ' ms'
        elif ms < 1000:
            out = '{:.3f}'.format(ms)[:3] + ' ms'
        elif sec < 60:
            out = '{:.3f}'.format(sec)[:4] + ' sec'
        elif mins < 60:
            out = '{:.3f}'.format(mins)[:4] + ' min'
        else:
            out = '{:.3f}'.format(hrs)[:4] + ' hrs'

        print(colored('Timer: {} = {}'.format(name, out), 'green'))

    def disp(self, name):
        """Pretty prints elapsed time

        If the timer has been stopped, it reports the time delta between
        start and stop. If it has not been stopped, it reports the current
        elapsed time and keeps the timing running.

        Parameters
        ----------
        name : str
            name of the timer to display

        Returns
        -------

        Raises
        ------
        ValueError
            if timer 'name' has not been started

        """

        try:     # has the timer been stopped?
            time = self._times[name]
        except KeyError:  # might still be running, let's check
            try:
                start = self._timers[name][0]
                now = self.op()   # don't stop it, just report current elapsed time
                time = float(now-start)/1e9 if self._ns else (now-start)
            except KeyError:
                raise ValueError(
                    colored("timer '{}' has not been started".format(name), 'red')) from None

        self.pretty_print(name, time)

    def __getitem__(self, key):
        return self._times[key]

    def __setitem__(self, key, val):
        self._times[key] = val


"""
Tristate is used with Basis to determine type of stellarator symmetry:
    True for cos(m*t-n*z) symmetry
    False for sin(m*t-n*z) symmetry
    None for no symmetry (Default)
"""


class Tristate(object):
    """Tristate is used to represent logical values with 3 possible states:
       True, False, or None

    """

    def __init__(self, value=None):
        if any(value is v for v in (True, False, None)):
            self.value = value
        else:
            raise ValueError("Tristate value must be True, False, or None")

    def __eq__(self, other):
        return (self.value is other.value if isinstance(other, Tristate)
                else self.value is other)

    def __ne__(self, other):
        return not self == other

    def __bool__(self):
        raise TypeError("Tristate object may not be used as a Boolean")

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return "Tristate(%s)" % self.value


class _Indexable():
    """Helper object for building indexes for indexed update functions.
    This is a singleton object that overrides the ``__getitem__`` method
    to return the index it is passed.
    >>> opsindex[1:2, 3, None, ..., ::2]
    (slice(1, 2, None), 3, None, Ellipsis, slice(None, None, 2))
    copied from jax.ops.index to work with either backend
    Parameters
    ----------
    Returns
    -------
    """
    __slots__ = ()

    def __getitem__(self, index):
        return index


"""
Helper object for building indexes for indexed update functions.
This is a singleton object that overrides the ``__getitem__`` method
to return the index it is passed.
>>> opsindex[1:2, 3, None, ..., ::2]
(slice(1, 2, None), 3, None, Ellipsis, slice(None, None, 2))
copied from jax.ops.index to work with either backend
"""
opsindex = _Indexable()


# Helper Functions -----------------------------------------------------------

def conditional_decorator(dec, condition, *args, **kwargs):
    """Apply arbitrary decorator to a function if condition is met

    Parameters
    ----------
    dec : decorator
        Decorator to apply
    condition : bool
        condition that must be met for decorator to be applied
    args : tuple, optional
        Arguments to pass to decorator
    kwargs : dict, optional
        Keyword arguments to pass to decorator


    Returns
    -------
    cond_dec : decorator
        Decorator that acts like ``dec`` if ``condition``,

    """
    @functools.wraps(dec)
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func, *args, **kwargs)
    return decorator


def unpack_state(x, nR0, nZ0, nr, nl):
    """Unpacks the state vector x into R0_n, Z0_n, r_lmn, l_lmn components

    Parameters
    ----------
    x : ndarray
        vector to unpack: x = [cR, cZ, cL]
    nR0 : int
        number of R0_n coefficients
    nZ0 : int
        number of Z0_n coefficients
    nr : int
        number of r_lmn coefficients
    nl : int
        number of l_lmn coefficients

    Returns
    -------
    R0_n : ndarray
        spectral coefficients of R0
    Z0_n : ndarray
        spectral coefficients of Z0
    r_lmn : ndarray
        spectral coefficients of r
    l_lmn : ndarray
        spectral coefficients of lambda

    """

    R0_n = x[:nR0]
    Z0_n = x[nR0:nR0+nZ0]
    r_lmn = x[nR0+nZ0:nR0+nZ0+nr]
    l_lmn = x[nR0+nZ0+nr:nR0+nZ0+nr+nl]
    return R0_n, Z0_n, r_lmn, l_lmn


def equals(a, b) -> bool:
    """Compares dictionaries that have numpy array values

    Parameters
    ----------
    a : dict
        reference dictionary
    b : dict
        comparison dictionary

    Returns
    -------
    bool
        a == b

    """
    if a.keys() != b.keys():
        return False
    return all(equals(a[key], b[key]) if isinstance(a[key], dict)
               else jnp.allclose(a[key], b[key]) if isinstance(a[key], jnp.ndarray)
               else (a[key] == b[key])
               for key in a)


def flatten_list(x):
    """Flattens a nested list

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
    """Checks if an array is sorted, within a given tolerance

    Checks whether x[i+1] - x[i] > tol

    Parameters
    ----------
    x : array-like
        input values
    axis : int
        axis along which to check if the array is sorted.
        If None, the flattened array is used. (Default value = None)
    tol : float
        tolerance for determining order. Array is still considered sorted
        if the difference between adjacent values is greater than -tol (Default value = 1e-12)

    Returns
    -------
    issorted : bool
        whether the array is sorted along specified axis

    """
    if axis is None:
        x = x.flatten()
        axis = 0
    return np.all(np.diff(x, axis=axis) >= -tol)


def isalmostequal(x, axis=-1, tol=1e-12):
    """Checks if all values of an array are equal, to within a given tolerance

    Parameters
    ----------
    x : array-like
        input values
    axis : int
        axis along which to make comparison. If None, the flattened array is used (Default value = -1)
    tol : float
        tolerance for comparison. Array is considered equal if std(x)*len(x)< tol along axis (Default value = 1e-12)

    Returns
    -------
    isalmostequal : bool
        whether the array is equal along specified axis

    """
    if axis is None:
        x = x.flatten()
        axis = 0
    return np.all(x.std(axis=axis)*x.shape[axis] < tol)


def sign(x):
    """Sign function, but returns 1 for x==0

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


def dot(a, b, axis):
    """Batched vector dot product

    Parameters
    ----------
    a : array-like
        first array of vectors
    b : array-like
        second array of vectors
    axis : int
        axis along which vectors are stored

    Returns
    -------
    y : array-like
        y = sum(a*b, axis=axis)

    """
    return jnp.sum(a*b, axis=axis, keepdims=False)


def cross(a, b, axis):
    """Batched vector cross product

    Parameters
    ----------
    a : array-like
        first array of vectors
    b : array-like
        second array of vectors
    axis : int
        axis along which vectors are stored

    Returns
    -------
    y : array-like
        y = a x b

    """
    return jnp.cross(a, b, axis=axis)


def rms(x):
    """Compute rms value of an array

    Parameters
    ----------
    x : array-like
        input array

    Returns
    -------
    y : float
        rms value of x, eg sqrt(sum(x**2))

    """
    return jnp.sqrt(jnp.mean(x**2))


def softmax(x, a=1):
    """Softmax function (or softmin for a<0)

    Smooth approximation to max/min of array

    Parameters
    ----------
    x : array-like
        input array
    a : float
        strength of approximation.
        Softmax -> max as a-> infty
        Softmin -> min as a -> -infty

    Returns
    -------
    y : float
        soft max/min of x
    """
    num = jnp.sum(x*jnp.exp(a*x))
    den = jnp.sum(jnp.exp(a*x))
    return num/den
