import numpy as np
import functools
import warnings
import desc
import os
os.environ["JAX_PLATFORM_NAME"] = 'cpu'


class TextColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    TIMER = '\033[32m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


if os.environ.get('DESC_USE_NUMPY'):
    jnp = np
    use_jax = False
    print('DESC version {}, using numpy backend, version={}, dtype={}'.format(desc.__version__,
                                                                              np.__version__, np.linspace(0, 1).dtype))
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
        print('DESC version {}, using JAX backend, jax version={}, jaxlib version={}, dtype={}'.format(
            desc.__version__, jax.__version__, jaxlib.__version__, x.dtype))
    except:
        jnp = np
        use_jax = False
        warnings.warn(TextColors.WARNING +
                      'Failed to load JAX' + TextColors.ENDC)
        print('DESC version {}, using numpy backend, version={}, dtype={}'.format(desc.__version__,
                                                                                  np.__version__, np.linspace(0, 1).dtype))

if use_jax:
    jit = jax.jit
    fori_loop = jax.lax.fori_loop

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
        x = jnp.asarray(n+1)
        y = jnp.exp(jax.scipy.special.gammaln(x))
        y = jnp.where(x < 1, 0, y)
        return y

else:
    jit = lambda func, *args, **kwargs: func
    from scipy.special import factorial

    # we divide by zero in a few places but then overwrite with the
    # correct asmptotic values, so lets suppress annoying warnings about that
    np.seterr(divide='ignore', invalid='ignore')

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
                warnings.warn(TextColors.WARNING +
                              'nanosecond timing not available on this system, reverting to microsecond timing' + TextColors.ENDC)
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
                TextColors.FAIL + "timer '{}' has not been started".format(name) + TextColors.ENDC) from None
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

        print(TextColors.TIMER + 'Timer: {} = {}'.format(name, out) + TextColors.ENDC)

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
                    TextColors.FAIL + "timer '{}' has not been started".format(name) + TextColors.ENDC) from None

        self.pretty_print(name, time)

    def __getitem__(self, key):
        return self._times[key]

    def __setitem__(self, key, val):
        self._times[key] = val


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


class Tristate(object):
    """ Tristate to determine type of symmetry for R,Z, and L.
    
        Possible values are:
            True for cos(m*t-n*z) symmetry
            False for sin(m*t-n*z) symmetry
            None for no symmetry (Default)

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
