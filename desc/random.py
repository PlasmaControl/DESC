"""Utilities for creating random surfaces and profiles."""

import numpy as np
import scipy.optimize
from numpy.random import default_rng

from desc.backend import jnp
from desc.basis import DoubleFourierSeries
from desc.derivatives import Derivative
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile


def random_surface(
    M=8,
    N=8,
    R0=(1, 10),
    R_scale=(0.5, 2),
    Z_scale=(0.5, 2),
    NFP=(1, 10),
    sym=None,
    alpha=(1, 4),
    rng=None,
):
    """Create a "random" toroidal surface.

    Uses a double Fourier series representation with random coefficients.
    The coefficients are given by

    X_mn =  X_scale * U(-1,1) exp(-alpha*(|m| + |n|))

    Where U(-1,1) is a uniform random variable on [-1,1].

    Parameters
    ----------
    M, N : int
        Poloidal and toroidal resolution of the double Fourier series.
    R0 : float or tuple
        Major radius. If a tuple, treats as min/max for random value.
    R_scale, Z_scale : float or tuple
        Scale factors for R and Z coordinates. If a tuple, treats as min/max for random
        values. The aspect ratio of the surface will be approximately
        R0/sqrt(R_scale*Z_scale)
    NFP : int or tuple
        Number of field periods. If a tuple, treats as min/max for random int
    sym : bool or None
        Whether the surface should be stellarator symmetric. If None, selects randomly.
    alpha : int or tuple
        Spectral decay factor. Larger values of alpha will tend to create simpler
        surfaces. If a tuple, treats as min/max for random int.
    rng : numpy.random.Generator
        Random number generator. If None, uses numpys default_rng

    Returns
    -------
    surf : FourierRZToroidalSurface
        Random toroidal surface.
    """
    if rng is None:
        rng = default_rng()
    if sym is None:
        sym = rng.choice([True, False])
    if isinstance(alpha, tuple):
        alpha = rng.integers(alpha[0], alpha[1] + 1)
    if isinstance(NFP, tuple):
        NFP = rng.integers(NFP[0], NFP[1] + 1)
    if isinstance(R_scale, tuple):
        R_scale = (R_scale[1] - R_scale[0]) * rng.random() + R_scale[0]
    if isinstance(Z_scale, tuple):
        Z_scale = (Z_scale[1] - Z_scale[0]) * rng.random() + Z_scale[0]
    if isinstance(R0, tuple):
        R0 = (R0[1] - R0[0]) * rng.random() + R0[0]

    R_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym="cos" if sym else False)
    Z_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym="sin" if sym else False)
    R_norm = np.exp(-alpha * np.sum(abs(R_basis.modes), axis=-1))
    Z_norm = np.exp(-alpha * np.sum(abs(Z_basis.modes), axis=-1))
    R_mn = (1 - 2 * rng.random(R_basis.num_modes)) * R_norm * R_scale
    Z_mn = (1 - 2 * rng.random(Z_basis.num_modes)) * Z_norm * Z_scale
    R_mn[R_basis.get_idx(0, 0, 0)] = R0
    surf = FourierRZToroidalSurface(
        R_mn,
        Z_mn,
        R_basis.modes[:, 1:],
        Z_basis.modes[:, 1:],
        NFP=NFP,
        sym=sym,
        check_orientation=False,
    )
    if surf._compute_orientation() == -1:
        surf._flip_orientation()
        assert surf._compute_orientation() == 1
    return surf


def random_pressure(n=(8, 16), p0=(1e3, 1e4), rng=None):
    """Create a random monotonic pressure profile.

    Profile will be a PowerSeriesProfile with even symmetry,
    enforced to be monotonically decreasing from p0 at r=0 to 0 at r=1

    Could also be used for other monotonically decreasing profiles
    such as temperature or density.

    Parameters
    ----------
    n : int or tuple
        Order of polynomial. If a tuple, treats as min/max for random int.
    p0 : float or tuple
        Pressure on axis. If a tuple, treats as min/max for random value.
    rng : numpy.random.Generator
        Random number generator. If None, uses numpys default_rng

    Returns
    -------
    pressure : PowerSeriesProfile
        Random pressure profile.
    """
    if rng is None:
        rng = default_rng()
    if isinstance(n, tuple):
        n = rng.integers(n[0] // 2, (n[1] + 1) // 2) * 2  # ensure its even
    if isinstance(p0, tuple):
        p0 = rng.uniform(p0[0], p0[1])

    # first create random even coeffs
    p = 1 - 2 * np.random.random(n // 2 + 1)
    # make it sum to 0 -> p=0 at r=1
    p[0] -= p.sum()
    # make p(0) = 1
    p = p / p[0]
    # this inserts zeros for all the odd modes
    p1 = jnp.vstack([p, jnp.zeros_like(p)]).flatten(order="F")[::-1]
    r = jnp.linspace(0, 1, 40)
    y = jnp.polyval(p1, r)

    def fun(x):
        x = jnp.vstack([x, jnp.zeros_like(x)]).flatten(order="F")[::-1]
        y_ = jnp.polyval(x, r)
        return jnp.sum((y - y_) ** 2)

    # constrain it so that it is monotonically decreasing, goes through (0,1) and (1,0)
    def con(x):
        x = jnp.vstack([x, jnp.zeros_like(x)]).flatten(order="F")[::-1]
        dx = jnp.polyder(x, 1)
        dy = jnp.polyval(dx, r)
        return jnp.concatenate([dy, jnp.atleast_1d(jnp.sum(x)), jnp.atleast_1d(x[-1])])

    hess = Derivative(fun, mode="hess")
    grad = Derivative(fun, mode="grad")
    A = Derivative(con, mode="fwd")(0 * p)
    l = np.concatenate([-np.inf * np.ones_like(r), jnp.array([0, 1])])
    u = np.concatenate([np.zeros_like(r), jnp.array([0, 1])])

    out = scipy.optimize.minimize(
        fun,
        p,
        jac=grad,
        hess=hess,
        constraints=scipy.optimize.LinearConstraint(A, l, u),
        method="trust-constr",
    )

    p = np.vstack([out.x, np.zeros_like(out.x)]).flatten(order="F")[::-1]
    return PowerSeriesProfile(p[::-1] * p0)
