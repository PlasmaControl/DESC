import numpy as np
from termcolor import colored
from abc import ABC, abstractmethod
import warnings
import scipy.optimize

from desc.backend import jnp, put
from desc.io import IOAble
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.interpolate import interp1d
from desc.transform import Transform
from desc.basis import PowerSeries
from desc.utils import copy_coeffs


class Profile(IOAble, ABC):
    """Abstract base class for profiles.

    All profile classes inherit from this, and must implement
    the transform() and compute() methods.

    The transform method should take an array of parameters and return the value
    of the profile or its derivatives on the default grid that is assigned to Profile.grid.
    This allows the profile to be used in solving and optimizing an equilibrium.

    The compute method should take an array of nodes and an optional array of parameters
    and compute the value or derivative of the profile at the specified nodes. If the
    parameters are not given, the ones assigned to the profile should be used.

    Subclasses must also implement getter and setting methods for name, grid, and params

    """

    _io_attrs_ = ["_name", "_grid", "_params"]

    @property
    def name(self):
        """Name of the profile"""
        return self._name

    @name.setter
    def name(self, new):
        self._name = new

    @property
    @abstractmethod
    def grid(self):
        """Default grid for computation"""

    @grid.setter
    @abstractmethod
    def grid(self, new):
        """Set default grid for computation"""

    @property
    @abstractmethod
    def params(self):
        """Default parameters for computation"""

    @params.setter
    @abstractmethod
    def params(self, new):
        """Set default params for computation"""

    @abstractmethod
    def compute(params=None, grid=None, dr=0, dt=0, dz=0):
        """compute values on specified nodes, default to using self.params"""

    def __call__(self, grid=None, params=None, dr=0, dt=0, dz=0):
        return self.compute(params, grid, dr, dt, dz)

    def __repr__(self):
        """string form of the object"""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, grid={})".format(self.name, self.grid)
        )


class PowerSeriesProfile(Profile):
    """Profile represented by a monic power series

    f(x) = a[0] + a[1]*x + a[2]*x**2 + ...

    Parameters
    ----------
    params: array-like
        coefficients of the series. If modes is not supplied, assumed to be in ascending order
        with no missing values. If modes is given, coefficients can be in any order or indexing.
    modes : array-like
        mode numbers for the associated coefficients. eg a[modes[i]] = params[i]
    grid : Grid
        default grid to use for computing values using transform method
    name : str
        name of the profile

    """

    _io_attrs_ = Profile._io_attrs_ + ["_basis", "_transform"]

    def __init__(self, params, modes=None, grid=None, name=None):

        self._name = name
        params = np.atleast_1d(params)
        if modes is None:
            modes = np.arange(params.size)
        self._basis = PowerSeries(L=int(np.max(abs(modes))))
        self._params = np.zeros(self.basis.num_modes, dtype=float)
        for m, c in zip(modes, params):
            idx = np.where(self.basis.modes[:, 0] == int(m))[0]
            self._params[idx] = c
        if grid is None:
            grid = Grid(np.empty((0, 3)))
        self._grid = grid
        self._transform = self._get_transform(grid)

    def _get_transform(self, grid):
        if grid is None:
            return self._transform
        if not isinstance(grid, Grid):
            if np.isscalar(grid):
                grid = np.linspace(0, 1, grid)
            grid = np.atleast_1d(grid)
            if grid.ndim == 1:
                grid = np.pad(grid[:, np.newaxis], ((0, 0), (0, 2)))
            grid = Grid(grid, sort=False)
        transform = Transform(
            grid,
            self.basis,
            derivs=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
        )
        return transform

    def __repr__(self):
        s = super().__repr__
        s = s[:-1]
        s += ", basis={})".format(self.basis)
        return s

    @property
    def basis(self):
        """Spectral basis for power series"""
        return self._basis

    @property
    def grid(self):
        """Default grid for computation"""
        return self._grid

    @grid.setter
    def grid(self, new):
        if isinstance(new, Grid):
            self._grid = new
        elif isinstance(new, (np.ndarray, jnp.ndarray)):
            self._grid = Grid(new, sort=False)
        else:
            raise TypeError(
                f"grid should be a Grid or subclass, or ndarray, got {type(new)}"
            )
        self._transform.grid = self.grid

    @property
    def params(self):
        """Parameter values"""
        return self._params

    @params.setter
    def params(self, new):
        if len(new) == self._basis.num_modes:
            self._params = jnp.asarray(new)
        else:
            raise ValueError(
                f"params should have the same size as the basis, got {len(new)} for basis with {self._basis.num_modes} modes"
            )

    def get_params(self, l):
        """Get power series coefficients for given mode number(s)"""
        l = np.atleast_1d(l).astype(int)
        a = np.zeros_like(l).astype(float)

        idx = np.where(l[:, np.newaxis] == self.basis.modes[:, 0])

        a[idx[0]] = self.params[idx[1]]
        return a

    def set_params(self, l, a=None):
        """set specific power series coefficients"""
        l, a = np.atleast_1d(l), np.atleast_1d(a)
        a = np.broadcast_to(a, l.shape)
        for ll, aa in zip(l, a):
            idx = self.basis.get_idx(ll, 0, 0)
            if aa is not None:
                self.params[idx] = aa

    def get_idx(self, l):
        """get index into params array for given mode number(s)"""
        return self.basis.get_idx(L=l)

    def change_resolution(self, L):
        """set a new maximum mode number"""
        modes_old = self.basis.modes
        self.basis.change_resolution(L)
        self._transform = self._get_transform(self.grid)
        self.params = copy_coeffs(self.params, modes_old, self.basis.modes)

    def compute(self, params=None, grid=None, dr=0, dt=0, dz=0):
        """Compute values of profile at specified nodes

        Parameters
        ----------
        params : array-like
            polynomial coefficients to use, in ascending order. If not given, uses the
            values given by the params attribute
        grid : Grid or array-like
            locations to compute values at. Defaults to self.grid
        dr, dt, dz : int
            derivative order in rho, theta, zeta

        Returns
        -------
        values : ndarray
            values of the profile or its derivative at the points specified

        """
        if params is None:
            params = self.params
        transform = self._get_transform(grid)
        return transform.transform(params, dr=dr, dt=dt, dz=dz)

    @classmethod
    def from_values(cls, x, y, order=6, rcond=None, w=None, grid=None, name=None):
        """Fit a PowerSeriesProfile from point data

        Parameters
        ----------
        x : array-like, shape(M,)
            coordinate locations
        y : array-like, shape(M,)
            function values
        order : int
            order of the polynomial to fit
        rcond : float
            Relative condition number of the fit. Singular values smaller than this
            relative to the largest singular value will be ignored. The default value
            is len(x)*eps, where eps is the relative precision of the float type, about
            2e-16 in most cases.
        w : array-like, shape(M,)
            Weights to apply to the y-coordinates of the sample points. For gaussian
            uncertainties, use 1/sigma (not 1/sigma**2).
        grid : Grid
            default grid to use for computing values using transform method
        name : str
            name of the profile

        Returns
        -------
        profile : PowerSeriesProfile
            profile in power series basis fit to given data.

        """
        params = np.polyfit(x, y, order, rcond=rcond, w=w, full=False)[::-1]
        return cls(params, grid=grid, name=name)

    def to_powerseries(self, order=6, xs=100, rcond=None, w=None):
        """Convert this profile to a PowerSeriesProfile

        Parameters
        ----------
        order : int
            polynomial order
        xs : int or ndarray
            x locations to use for fit. If an integer, uses that many points linearly
            spaced between 0,1
        rcond : float
            Relative condition number of the fit. Singular values smaller than this
            relative to the largest singular value will be ignored. The default value
            is len(x)*eps, where eps is the relative precision of the float type, about
            2e-16 in most cases.
        w : array-like, shape(M,)
            Weights to apply to the y-coordinates of the sample points. For gaussian
            uncertainties, use 1/sigma (not 1/sigma**2).

        Returns
        -------
        profile : PowerSeriesProfile
            profile in power series form.

        """
        if len(self.params) == order + 1:
            params = self.params
        elif len(self.params) > order + 1:
            params = self.params[: order + 1]
        elif len(self.params) < order + 1:
            params = np.pad(self.params, (0, order + 1 - len(self.params)))
        modes = np.arange(order + 1)

        return PowerSeriesProfile(params, modes, self.grid, self.name)

    def to_spline(self, knots=20, method="cubic2"):
        """Convert this profile to a SplineProfile

        Parameters
        ----------
        knots : int or ndarray
            x locations to use for spline. If an integer, uses that many points linearly
            spaced between 0,1
        method : str
            method of interpolation
            - `'nearest'`: nearest neighbor interpolation
            - `'linear'`: linear interpolation
            - `'cubic'`: C1 cubic splines (aka local splines)
            - `'cubic2'`: C2 cubic splines (aka natural splines)
            - `'catmull-rom'`: C1 cubic centripedal "tension" splines

        Returns
        -------
        profile : SplineProfile
            profile in spline form.

        """
        if np.isscalar(knots):
            knots = np.linspace(0, 1, knots)
        values = self.compute(grid=knots)
        return SplineProfile(values, knots, self.grid, method, self.name)

    def to_mtanh(
        self, order=4, xs=100, w=None, p0=None, pmax=None, pmin=None, **kwargs
    ):
        """Convert this profile to modified hyperbolic tangent + poly form.

        Parameters
        ----------
        order : int
            order of the core polynomial to fit
        xs : int or array-like, shape(M,)
            coordinate locations to evaluate for fitting. If an integer, assumes
            that many linearly spaced ints in (0,1)
        w : array-like, shape(M,)
            Weights to apply to the y-coordinates of the sample points. For gaussian
            uncertainties, use 1/sigma (not 1/sigma**2).
        p0 : array-like, shape(5+order,)
            initial guess for parameter values
        pmin : float or array-like, shape(5+order,)
            lower bounds for parameter values
        pmax : float or array-like, shape(5+order,)
            upper bounds for parameter values

        Returns
        -------
        profile : MTanhProfile
            profile in mtanh + polynomial form.

        """
        if np.isscalar(xs):
            xs = np.linspace(0, 1, xs)
        ys = self.compute(grid=xs)
        return MTanhProfile.from_values(
            xs,
            ys,
            order=order,
            w=w,
            p0=p0,
            pmax=pmax,
            pmin=pmin,
            grid=self.grid,
            name=self.name,
            **kwargs,
        )


class SplineProfile(Profile):
    """Profile represented by a piecewise cubic spline


    Parameters
    ----------
    params: array-like
        Values of the function at knot locations.
    knots : int or ndarray
        x locations to use for spline. If an integer, uses that many points linearly
        spaced between 0,1
    method : str
        method of interpolation
        - `'nearest'`: nearest neighbor interpolation
        - `'linear'`: linear interpolation
        - `'cubic'`: C1 cubic splines (aka local splines)
        - `'cubic2'`: C2 cubic splines (aka natural splines)
        - `'catmull-rom'`: C1 cubic centripedal "tension" splines
    grid : Grid
        default grid to use for computing values using transform method
    name : str
        name of the profile

    """

    _io_attrs_ = Profile._io_attrs_ + ["_knots", "_method"]

    def __init__(self, values, knots=None, grid=None, method="cubic2", name=None):

        if knots is None:
            knots = np.linspace(0, 1, values.size)
        self._name = name
        self._knots = knots
        self._params = values
        self._method = method

        if grid is None:
            grid = Grid(np.empty((0, 3)))
        self.grid = grid

    def __repr__(self):
        s = super().__repr__
        s = s[:-1]
        s += ", method={}, num_knots={})".format(self._method, len(self._knots))
        return s

    @property
    def grid(self):
        """Default grid for computation"""
        return self._grid

    @grid.setter
    def grid(self, new):
        if isinstance(new, Grid):
            self._grid = new
        elif isinstance(new, (np.ndarray, jnp.ndarray)):
            self._grid = Grid(new, sort=False)
        else:
            raise TypeError(
                f"grid should be a Grid or subclass, or ndarray, got {type(new)}"
            )

    @property
    def params(self):
        """Alias for values"""
        return self._params

    @params.setter
    def params(self, new):
        if len(new) == len(self._knots):
            self._params = jnp.asarray(new)
        else:
            raise ValueError(
                f"params should have the same size as the knots, got {len(new)} values for {len(self._knots)} knots"
            )

    @property
    def values(self):
        """Value of the function at knots"""
        return self._params

    @values.setter
    def values(self, new):
        if len(new) == len(self._knots):
            self._params = jnp.asarray(new)
        else:
            raise ValueError(
                f"params should have the same size as the knots, got {len(new)} values for {len(self._knots)} knots"
            )

    def _get_xq(self, grid):
        if grid is None:
            return self.grid.nodes[:, 0]
        if isinstance(grid, Grid):
            return grid.nodes[:, 0]
        if np.isscalar(grid):
            return np.linspace(0, 1, grid)
        grid = np.atleast_1d(grid)
        if grid.ndim == 1:
            return grid
        return grid[:, 0]

    def compute(self, params=None, grid=None, dr=0, dt=0, dz=0):
        """Compute values of profile at specified nodes

        Parameters
        ----------
        nodes : ndarray, shape(k,) or (k,3)
            locations to compute values at
        params : array-like
            spline values to use. If not given, uses the
            values given by the params attribute
        dr, dt, dz : int
            derivative order in rho, theta, zeta

        Returns
        -------
        values : ndarray
            values of the profile or its derivative at the points specified

        """
        if params is None:
            params = self.params
        xq = self._get_xq(grid)
        if dt != 0 or dz != 0:
            return jnp.zeros_like(xq)
        x = self._knots
        f = params
        fq = interp1d(xq, x, f, method=self._method, derivative=dr, extrap=True)
        return fq

    def to_powerseries(self, order=6, xs=100, rcond=None, w=None):
        """Convert this profile to a PowerSeriesProfile

        Parameters
        ----------
        order : int
            polynomial order
        xs : int or ndarray
            x locations to use for fit. If an integer, uses that many points linearly
            spaced between 0,1
        rcond : float
            Relative condition number of the fit. Singular values smaller than this
            relative to the largest singular value will be ignored. The default value
            is len(x)*eps, where eps is the relative precision of the float type, about
            2e-16 in most cases.
        w : array-like, shape(M,)
            Weights to apply to the y-coordinates of the sample points. For gaussian
            uncertainties, use 1/sigma (not 1/sigma**2).

        Returns
        -------
        profile : PowerSeriesProfile
            profile in power series form.

        """
        if np.isscalar(xs):
            xs = np.linspace(0, 1, xs)
        fs = self.compute(grid=xs)
        p = PowerSeriesProfile.from_values(xs, fs, order, rcond=rcond, w=w)
        p.grid = self.grid
        p.name = self.name
        return p

    def to_spline(self, knots=20, method="cubic2"):
        """Convert this profile to a SplineProfile

        Parameters
        ----------
        knots : int or ndarray
            x locations to use for spline. If an integer, uses that many points linearly
            spaced between 0,1
        method : str
            method of interpolation
            - `'nearest'`: nearest neighbor interpolation
            - `'linear'`: linear interpolation
            - `'cubic'`: C1 cubic splines (aka local splines)
            - `'cubic2'`: C2 cubic splines (aka natural splines)
            - `'catmull-rom'`: C1 cubic centripedal "tension" splines

        Returns
        -------
        profile : SplineProfile
            profile in spline form.

        """
        if np.isscalar(knots):
            knots = np.linspace(0, 1, knots)
        values = self.compute(grid=knots)
        return SplineProfile(values, knots, self.grid, method, self.name)

    def to_mtanh(
        self, order=4, xs=100, w=None, p0=None, pmax=None, pmin=None, **kwargs
    ):
        """Convert this profile to modified hyperbolic tangent + poly form.

        Parameters
        ----------
        order : int
            order of the core polynomial to fit
        xs : int or array-like, shape(M,)
            coordinate locations to evaluate for fitting. If an integer, assumes
            that many linearly spaced ints in (0,1)
        w : array-like, shape(M,)
            Weights to apply to the y-coordinates of the sample points. For gaussian
            uncertainties, use 1/sigma (not 1/sigma**2).
        p0 : array-like, shape(5+order,)
            initial guess for parameter values
        pmin : float or array-like, shape(5+order,)
            lower bounds for parameter values
        pmax : float or array-like, shape(5+order,)
            upper bounds for parameter values

        Returns
        -------
        profile : MTanhProfile
            profile in mtanh + polynomial form.

        """
        if np.isscalar(xs):
            xs = np.linspace(0, 1, xs)
        ys = self.compute(grid=xs)
        return MTanhProfile.from_values(
            xs,
            ys,
            order=order,
            w=w,
            p0=p0,
            pmax=pmax,
            pmin=pmin,
            grid=self.grid,
            name=self.name,
            **kwargs,
        )


class MTanhProfile(Profile):
    """Profile represented by a modified hyperbolic tangent + polynomial

    Profile is parameterized by pedestal height (ped, :math:`p`), SOL height (offset, :math:`o`),
    pedestal symmetry point (sym, :math:`s`), pedestal width (width, :math:`w`), and a polynomial:

    .. math::

        y = o + \\frac{1}{2} \\left(o - p\\right) \\left(\\tanh{\\left(z \\right)} - 1\\right) + \\frac{\\left(o - p\\right) f{\\left(\\frac{z}{e^{2 z} + 1} \\right)}}{2}

    Where :math:`z=(x-s)/w` and :math:`f` is a polynomial (with no constant term)

    Parameters
    ----------
    params: array-like
        parameters for mtanh + poly. ``params = [ped, offset, sym, width, *core_poly]`` where
        core poly are the polynomial coefficients in ascending order, without a constant term
    grid : Grid
        default grid to use for computing values using transform method
    name : str
        name of the profile

    """

    def __init__(self, params, grid=None, name=None):

        self._name = name
        self._params = params

        if grid is None:
            grid = Grid(np.empty((0, 3)))
        self.grid = grid

    def __repr__(self):
        s = super().__repr__
        s = s[:-1]
        s += ", num_params={})".format(len(self._params))
        return s

    @property
    def grid(self):
        """Default grid for computation"""
        return self._grid

    @grid.setter
    def grid(self, new):
        if isinstance(new, Grid):
            self._grid = new
        elif isinstance(new, (np.ndarray, jnp.ndarray)):
            self._grid = Grid(new, sort=False)
        else:
            raise TypeError(
                f"grid should be a Grid or subclass, or ndarray, got {type(new)}"
            )

    @property
    def params(self):
        """Parameter values"""
        return self._params

    @params.setter
    def params(self, new):
        if len(new) >= 5:
            self._params = jnp.asarray(new)
        else:
            raise ValueError(
                f"params should have at least 5 elements [ped, offset, sym, width, *core_poly]  got only {len(new)} values"
            )

    @staticmethod
    def _mtanh(x, ped, offset, sym, width, core_poly, dx=0):
        """modified tanh + polynomial profile

        Parameters
        ----------
        x : ndarray
            evaluation locations
        ped : float
            height of pedestal
        offset : float
            height of SOL
        sym : float
            symmetry point
        width : float
            width of pedestal
        core_poly : ndarray
            polynomial coefficients in ascending order [x^1,...x^n]
        dx : int
           radial derivative order

        Returns
        -------
        y : ndarray
            profile evaluated at x
        """
        core_poly = jnp.pad(jnp.asarray(core_poly), ((1, 0)))
        z = (x - sym) / width

        if dx == 0:
            y = 1 / 2 * (ped - offset) * (1 - jnp.tanh(z)) + offset
        elif dx == 1:
            y = -1 / (2 * width) * (1 - jnp.tanh(z) ** 2) * (ped - offset)
        elif dx == 2:
            y = (ped - offset) * (jnp.tanh(-z) ** 2 - 1) * jnp.tanh(-z) / width ** 2

        e2z = jnp.exp(2 * z)
        zz = z / (1 + e2z)
        if dx == 0:
            f = jnp.polyval(core_poly[::-1], zz)
        elif dx == 1:
            dz = ((1 + e2z) - 2 * z * e2z) / (width * (1 + e2z) ** 2)
            f = jnp.polyval(jnp.polyder(core_poly[::-1], 1), zz) * dz
        elif dx == 2:
            dz = ((1 + e2z) - 2 * z * e2z) / (width * (1 + e2z) ** 2)
            ddz = (
                4
                * (-width * (1 + e2z) + (1 - e2z) * (sym - x))
                * e2z
                / (width ** 3 * (e2z + 1) ** 3)
            )
            f = (
                jnp.polyval(jnp.polyder(core_poly[::-1], 2)) * dz ** 2
                + jnp.polyval(jnp.polyder(core_poly[::-1], 1), zz) * ddz
            )

        y = y + f * (offset - ped) / 2
        return y

    def _get_xq(self, grid):
        if grid is None:
            return self.grid.nodes[:, 0]
        if isinstance(grid, Grid):
            return grid.nodes[:, 0]
        if np.isscalar(grid):
            return np.linspace(0, 1, grid)
        grid = np.atleast_1d(grid)
        if grid.ndim == 1:
            return grid
        return grid[:, 0]

    def compute(self, params=None, grid=None, dr=0, dt=0, dz=0):
        """Compute values of profile at specified nodes

        Parameters
        ----------
        nodes : ndarray, shape(k,) or (k,3)
            locations to compute values at
        params : array-like
            coefficients to use, in order. [ped, offset, sym, width, core_poly]
            If not given, uses the values given by the params attribute
        dr, dt, dz : int
            derivative order in rho, theta, zeta

        Returns
        -------
        values : ndarray
            values of the profile or its derivative at the points specified

        """
        if params is None:
            params = self.params

        xq = self._get_xq(grid)
        ped = params[0]
        offset = params[1]
        sym = params[2]
        width = params[3]
        core_poly = params[4:]

        if dt != 0 or dz != 0:
            return jnp.zeros_like(xq)

        y = MTanhProfile._mtanh(xq, ped, offset, sym, width, core_poly, dx=dr)
        return y

    @classmethod
    def from_values(
        cls,
        x,
        y,
        order=4,
        w=None,
        p0=None,
        pmax=None,
        pmin=None,
        grid=None,
        name=None,
        **kwargs,
    ):
        """Fit a MTanhProfile from point data

        Parameters
        ----------
        x : array-like, shape(M,)
            coordinate locations
        y : array-like, shape(M,)
            function values
        order : int
            order of the core polynomial to fit
        w : array-like, shape(M,)
            Weights to apply to the y-coordinates of the sample points. For gaussian
            uncertainties, use 1/sigma (not 1/sigma**2).
        p0 : array-like, shape(4+order,)
            initial guess for parameter values [ped, offset, sym, width, core_poly].
            Use a value of "None" to use the default initial guess for that parameter
        pmin : float or array-like, shape(4+order,)
            lower bounds for parameter values
            Use a value of "None" to use the default bound for that parameter
        pmax : float or array-like, shape(4+order,)
            upper bounds for parameter values
            Use a value of "None" to use the default bound for that parameter
        grid : Grid
            default grid to use for computing values using transform method
        name : str
            name of the profile
        kwargs :
            additional keyword arguments passed to scipy.optimize.least_squares

        Returns
        -------
        profile : MTanhProfile
            profile in mtanh + polynomial form.

        """
        if w is None:
            w = np.ones_like(x)
        fun = (
            lambda args: (
                cls._mtanh(x, args[0], args[1], args[2], args[3], args[4:]) - y
            )
            / w
        )

        ped0 = np.clip(interp1d([0.93], x, y, "cubic2", extrap=True), 0, np.inf)[0]
        off0 = np.clip(interp1d([0.98], x, y, "cubic2", extrap=True), 0, np.inf)[0]
        default_pmax = np.array([np.inf, np.inf, 1.02, 0.2, np.inf])
        default_pmin = np.array([0, 0, 0.9, 0.0, -np.inf])
        default_p0 = np.array([ped0, off0, 0.95, 0.1, 0])

        p0_ = np.atleast_1d(p0)
        pmin_ = np.atleast_1d(pmax)
        pmax_ = np.atleast_1d(pmin)
        p0 = np.zeros(order + 4)
        pmax = np.zeros(order + 4)
        pmin = np.zeros(order + 4)
        for i in range(order + 4):
            if i < len(p0_) and p0_[i] is not None:
                p0[i] = p0_[i]
            else:
                p0[i] = default_p0[np.clip(i, 0, len(default_p0) - 1)]
            if i < len(pmax_) and pmax_[i] is not None:
                pmax[i] = pmax_[i]
            else:
                pmax[i] = default_pmax[np.clip(i, 0, len(default_pmax) - 1)]
            if i < len(pmin_) and pmin_[i] is not None:
                pmin[i] = pmin_[i]
            else:
                pmin[i] = default_pmin[np.clip(i, 0, len(default_pmin) - 1)]

        out = scipy.optimize.least_squares(
            fun, x0=p0, method="trf", bounds=(pmin, pmax), **kwargs
        )
        if not out.success:
            warnings.warn("Fitting did not converge, parameters may not be correct")
        params = out.x
        return MTanhProfile(params, grid, name)

    def to_powerseries(self, order=6, xs=100, rcond=None, w=None):
        """Convert this profile to a PowerSeriesProfile

        Parameters
        ----------
        order : int
            polynomial order
        xs : int or ndarray
            x locations to use for fit. If an integer, uses that many points linearly
            spaced between 0,1
        rcond : float
            Relative condition number of the fit. Singular values smaller than this
            relative to the largest singular value will be ignored. The default value
            is len(x)*eps, where eps is the relative precision of the float type, about
            2e-16 in most cases.
        w : array-like, shape(M,)
            Weights to apply to the y-coordinates of the sample points. For gaussian
            uncertainties, use 1/sigma (not 1/sigma**2).

        Returns
        -------
        profile : PowerSeriesProfile
            profile in power series form.

        """
        if np.isscalar(xs):
            xs = np.linspace(0, 1, xs)
        fs = self.compute(grid=xs)
        p = PowerSeriesProfile.from_values(xs, fs, order, rcond=rcond, w=w)
        p.grid = self.grid
        p.name = self.name
        return p

    def to_spline(self, knots=20, method="cubic2"):
        """Convert this profile to a SplineProfile

        Parameters
        ----------
        knots : int or ndarray
            x locations to use for spline. If an integer, uses that many points linearly
            spaced between 0,1
        method : str
            method of interpolation
            - `'nearest'`: nearest neighbor interpolation
            - `'linear'`: linear interpolation
            - `'cubic'`: C1 cubic splines (aka local splines)
            - `'cubic2'`: C2 cubic splines (aka natural splines)
            - `'catmull-rom'`: C1 cubic centripedal "tension" splines

        Returns
        -------
        profile : SplineProfile
            profile in spline form.

        """
        if np.isscalar(knots):
            knots = np.linspace(0, 1, knots)
        values = self.compute(grid=knots)
        return SplineProfile(values, knots, self.grid, method, self.name)
