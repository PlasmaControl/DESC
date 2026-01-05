"""Functions related to most/least rational numbers, etc."""

import numpy as np
from scipy import optimize

from desc.backend import jnp


def cf_to_dec(cf):
    """Compute decimal form of a continued fraction.

    Parameters
    ----------
    cf : array-like
        coefficients of continued fraction.

    Returns
    -------
    x : float
        floating point representation of cf

    """
    if len(cf) == 1:
        return cf[0]
    else:
        return cf[0] + 1 / cf_to_dec(cf[1:])


def dec_to_cf(x, dmax=20, itol=1e-14):
    """Compute continued fraction form of a number.

    Parameters
    ----------
    x : float
        floating point form of number
    dmax : int
        maximum iterations (ie, number of coefficients of continued fraction).
        (Default value = 20)
    itol : float, optional
        tolerance for rounding float to nearest int

    Returns
    -------
    cf : ndarray of int
        coefficients of continued fraction form of x.

    """
    x = float(_round(x, itol))
    cf = []
    q = np.floor(x).astype(int)
    cf.append(q)
    x = _round(x - q, itol)
    i = 0
    while x != 0 and i < dmax:
        q = np.floor(1 / x).astype(int)
        cf.append(q)
        x = _round(1 / x - q, itol)
        i = i + 1
    return np.array(cf).astype(int)


def find_most_distant(pts, n, a=None, b=None, atol=1e-14, **kwargs):
    """Find n points in interval that are maximally distant from pts and each other.

    Parameters
    ----------
    pts : ndarray
        Points to avoid
    n : int
        Number of points to find.
    a, b : float, optional
        Start and end points for interval. Default is min/max of pts
    atol : float, optional
        Stopping tolerance for minimization
    """

    def foo(x, xs):
        xs = np.atleast_1d(xs)
        d = x - xs[:, None]
        return -np.prod(np.abs(d), axis=0).squeeze()

    if a is None:
        a = np.min(pts)
    if b is None:
        b = np.max(pts)

    pts = list(pts)
    nsamples = kwargs.get("nsamples", 1000)
    x = np.linspace(a, b, nsamples)
    out = []
    for i in range(n):
        y = foo(x, pts)
        x0 = x[np.argmin(y)]
        bracket = (max(a, x0 - 5 / nsamples), min(x0 + 5 / nsamples, b))
        c = optimize.minimize_scalar(
            foo, bracket=bracket, bounds=(a, b), args=(pts,), options={"xatol": atol}
        ).x
        pts.append(c)
        out.append(c)
    return np.array(out)


def find_most_rational_surfaces(iota, n, atol=1e-14, itol=1e-14, eps=1e-12, **kwargs):
    """Find "most rational" surfaces for a give iota profile.

    By most rational, we generally mean lowest order ie smallest continued fraction.

    Note: May not work as expected for non-monotonic profiles with duplicate rational
    surfaces. (generally only 1 of each rational is found)

    Parameters
    ----------
    iota : Profile
        iota profile to search
    n : integer
        number of rational surfaces to find
    atol : float, optional
        stopping tolerance for root finding
    itol : float, optional
        tolerance for rounding float to nearest int
    eps : float, optional
        amount to displace points to avoid duplicates

    Returns
    -------
    rho : ndarray
        sorted radial locations of rational surfaces
    rationals: ndarray
        values of iota at rational surfaces
    """
    # find approx min/max
    r = np.linspace(0, 1, kwargs.get("nsamples", 1000))
    io = iota(r)
    iomin, iomax = np.min(io), np.max(io)
    # find rational values of iota and corresponding rho
    io_rational = n_most_rational(iomin, iomax, n, itol=itol, eps=eps)
    rho = _find_rho(iota, io_rational, tol=atol)
    idx = np.argsort(rho)
    return rho[idx], io_rational[idx]


def find_least_rational_surfaces(
    iota, n, nrational=100, atol=1e-14, itol=1e-14, eps=1e-12, **kwargs
):
    """Find "least rational" surfaces for given iota profile.

    By least rational we mean points farthest in iota from the nrational lowest
    order rational surfaces and each other.

    Note: May not work as expected for non-monotonic profiles with duplicate rational
    surfaces. (generally only 1 of each rational is found)

    Parameters
    ----------
    iota : Profile
        iota profile to search
    n : integer
        number of approximately irrational surfaces to find
    nrational : int, optional
        number of lowest order rational surfaces to avoid.
    atol : float, optional
        Stopping tolerance for minimization
    itol : float, optional
        tolerance for rounding float to nearest int
    eps : float, optional
        amount to displace points to avoid duplicates

    Returns
    -------
    rho : ndarray
        locations of least rational surfaces
    io : ndarray
        values of iota at least rational surfaces
    rho_rat : ndarray
        rho values of lowest order rational surfaces
    io_rat : ndarray
        iota values at lowest order rational surfaces
    """
    rho_rat, io_rat = find_most_rational_surfaces(
        iota, nrational, atol, itol, eps, **kwargs
    )
    a, b = iota([0.0, 1.0])
    io = find_most_distant(io_rat, n, a, b, tol=atol, **kwargs)
    rho = _find_rho(iota, io, tol=atol)
    return rho, io


def most_rational(a, b, itol=1e-14):
    """Compute the most rational number in the range [a,b].

    Parameters
    ----------
    a,b : float
        lower and upper bounds
    itol : float, optional
        tolerance for rounding float to nearest int

    Returns
    -------
    x : float
        most rational number between [a,b]

    """
    a = float(_round(a, itol))
    b = float(_round(b, itol))

    # Handle empty range
    if a == b:
        return a

    # Return 0 if in range
    if np.sign(a * b) <= 0:
        return 0

    # Handle negative ranges
    if np.sign(a) < 0:
        s = -1
        a *= -1
        b *= -1
    else:
        s = 1

    # Ensure a < b
    if a > b:
        a, b = b, a

    a_cf = dec_to_cf(a)
    b_cf = dec_to_cf(b)
    idx = 0  # first index of dissimilar digits
    for i in range(min(a_cf.size, b_cf.size)):
        if a_cf[i] != b_cf[i]:
            idx = i
            break
    f = 1
    while True:
        dec = cf_to_dec(np.append(a_cf[0:idx], f))
        if a <= dec <= b:
            return dec * s
        f += 1


def n_most_rational(a, b, n, eps=1e-12, itol=1e-14):
    """Find the n most rational numbers in a given interval.

    Parameters
    ----------
    a, b : float
        start and end points of the interval
    n : integer
        number of rationals to find
    eps : float, optional
        amount to displace points to avoid duplicates
    itol : float, optional
        tolerance for rounding float to nearest int

    Returns
    -------
    c : ndarray
        most rational points in (a,b), in approximate
        order of "rationality"
    """
    assert eps > itol
    a = float(_round(a, itol))
    b = float(_round(b, itol))
    # start with the full interval, find first most rational
    # then subdivide at that point and look for next most
    # rational in the largest sub-interval
    out = []
    intervals = np.array(sorted([a, b]))
    for i in range(n):
        i = np.argmax(np.diff(intervals))
        ai, bi = intervals[i : i + 2]
        if ai in out:
            ai += eps
        if bi in out:
            bi -= eps
        c = most_rational(ai, bi)
        out.append(c)
        j = np.searchsorted(intervals, c)
        intervals = np.insert(intervals, j, c)
    return np.array(out)


def periodic_spacing(x, period=2 * jnp.pi, sort=False, jnp=jnp):
    """Compute dx between points in x assuming periodicity.

    Parameters
    ----------
    x : Array
        Points, assumed sorted in the cyclic domain [0, period], unless
        specified otherwise.
    period : float
        Number such that f(x + period) = f(x) for any function f on this domain.
    sort : bool
        Set to true if x is not sorted in the cyclic domain [0, period].

    Returns
    -------
    x, dx : Array
        Points in [0, period] and assigned spacing.

    """
    x = jnp.atleast_1d(x)
    x = jnp.where(x == period, x, x % period)
    if sort:
        x = jnp.sort(x, axis=0)
    # choose dx to be half the distance between its neighbors
    if x.size > 1:
        if np.isfinite(period):
            dx_0 = x[1] + (period - x[-1]) % period
            dx_1 = x[0] + (period - x[-2]) % period
        else:
            # just set to 0 to stop nan gradient, even though above gives expected value
            dx_0 = 0
            dx_1 = 0
        if x.size == 2:
            # then dx[0] == period and dx[-1] == 0, so fix this
            dx_1 = dx_0
        dx = jnp.hstack([dx_0, x[2:] - x[:-2], dx_1]) / 2
    else:
        dx = jnp.array([period])
    return x, dx


def midpoint_spacing(x, jnp=jnp):
    """Compute dx between points in x in [0, 1].

    Parameters
    ----------
    x : Array
        Points in [0, 1], assumed sorted.

    Returns
    -------
    dx : Array
        Spacing assigned to points in x.

    """
    x = jnp.atleast_1d(x)
    if x.size > 1:
        # choose dx such that cumulative sums of dx[] are node midpoints
        # and the total sum is 1
        dx_0 = (x[0] + x[1]) / 2
        dx_1 = 1 - (x[-2] + x[-1]) / 2
        dx = jnp.hstack([dx_0, (x[2:] - x[:-2]) / 2, dx_1])
    else:
        dx = jnp.array([1.0])
    return dx


def _find_rho(iota, iota_vals, tol=1e-14):
    """Find rho values for iota_vals in iota profile."""
    r = np.linspace(0, 1, 1000)
    io = iota(r)
    rho = []
    for ior in iota_vals:
        f = lambda r: iota(np.atleast_1d(r))[0] - ior
        df = lambda r: iota(np.atleast_1d(r), dr=1)[0]
        # nearest neighbor search for initial guess
        x0 = r[np.argmin(np.abs(io - ior))]
        rho_i = optimize.root_scalar(f, x0=x0, fprime=df, xtol=tol).root
        rho.append(rho_i)
    return np.array(rho)


def _round(x, tol):
    # we do this to avoid some floating point issues with things
    # that are basically low order rationals to near machine precision
    if abs(x % 1) < tol or abs((x % 1) - 1) < tol:
        return round(x)
    return x
