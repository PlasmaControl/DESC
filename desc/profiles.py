"""Profile objects for representing pressure, rotational transform, etc."""

import warnings
from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize

from desc.backend import jit, jnp, put, sign
from desc.basis import FourierZernikeBasis, PowerSeries
from desc.derivatives import Derivative
from desc.grid import Grid, LinearGrid
from desc.interpolate import _approx_df, interp1d
from desc.io import IOAble
from desc.transform import Transform
from desc.utils import copy_coeffs


class Profile(IOAble, ABC):
    """Abstract base class for profiles.

    All profile classes inherit from this, and must implement
    the compute() methods.

    The compute method should take an array of nodes and an optional array of parameters
    and compute the value or derivative of the profile at the specified nodes.
    If the parameters are not given, the ones assigned to the profile should be used.

    Subclasses must also implement getter and setter methods for params
    """

    _io_attrs_ = ["_name", "_grid", "_params"]

    def __init__(self, grid=None, name=""):

        self.name = name
        self.grid = grid if grid is not None else LinearGrid(L=20)

    @property
    def name(self):
        """str: Name of the profile."""
        return self._name

    @name.setter
    def name(self, new):
        self._name = new

    @property
    def grid(self):
        """Grid: Nodes for computation."""
        return self._grid

    @grid.setter
    def grid(self, grid):
        if isinstance(grid, Grid):
            self._grid = grid
            return
        if np.isscalar(grid):
            grid = np.linspace(0, 1, grid)
        if isinstance(grid, (np.ndarray, jnp.ndarray)):
            if grid.ndim == 1:
                grid = np.pad(grid[:, np.newaxis], ((0, 0), (0, 2)))
            self._grid = Grid(grid, sort=False)
        else:
            raise TypeError(
                f"grid should be a Grid or subclass, or ndarray, got {type(grid)}"
            )

    @property
    @abstractmethod
    def params(self):
        """ndarray: Parameters for computation."""

    @params.setter
    @abstractmethod
    def params(self, new):
        """Set default params for computation."""

    @abstractmethod
    def compute(self, params=None, grid=None, dr=0, dt=0, dz=0):
        """Compute values on specified nodes, default to using self.params."""

    def to_powerseries(self, order=6, xs=100, sym="auto", rcond=None, w=None):
        """Convert this profile to a PowerSeriesProfile.

        Parameters
        ----------
        order : int
            polynomial order
        xs : int or ndarray
            x locations to use for fit. If an integer, uses that many points linearly
            spaced between 0,1
        sym : bool or "auto"
            Whether to enforce explicit even parity
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
        p = PowerSeriesProfile.from_values(xs, fs, order, rcond=rcond, w=w, sym=sym)
        p.grid = self.grid
        p.name = self.name
        return p

    def to_fourierzernike(self, L=6, M=0, N=0, NFP=1, xs=100, w=None):
        """Convert this profile to a FourierZernikeProfile.

        Parameters
        ----------
        L, M, N : int
           maximum mode numbers
        NFP : int
            number of field periods
        xs : int or ndarray
            x locations to use for fit. If an integer, uses that many points linearly
            spaced between 0,1
        w : array-like, shape(M,)
            Weights to apply to the y-coordinates of the sample points. For gaussian
            uncertainties, use 1/sigma (not 1/sigma**2).

        Returns
        -------
        profile : FourierZernikeProfile
            profile in power series form.

        """
        if np.isscalar(xs):
            xs = np.linspace(0, 1, xs)
        f = self.compute(grid=xs)
        r = xs
        t = np.zeros_like(xs)
        z = np.zeros_like(xs)
        p = FourierZernikeProfile.from_values(
            r, t, z, f, L, M, N, NFP, w, self.grid, self.name
        )
        return p

    def to_spline(self, knots=20, method="cubic2"):
        """Convert this profile to a SplineProfile.

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

    def __call__(self, grid=None, params=None, dr=0, dt=0, dz=0):
        """Evaluate the profile at a given set of points."""
        return self.compute(params, grid, dr, dt, dz)

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, grid={})".format(self.name, self.grid)
        )

    def __mul__(self, x):
        """Multiply this profile by another or a constant."""
        if np.isscalar(x):
            return ScaledProfile(x, self)
        elif isinstance(x, Profile):
            return ProductProfile(self, x)
        else:
            raise NotImplementedError()

    def __rmul__(self, x):
        """Multiply this profile by another or a constant."""
        return self.__mul__(x)

    def __add__(self, x):
        """Add this profile with another."""
        if isinstance(x, Profile):
            return SumProfile(self, x)
        else:
            raise NotImplementedError()

    def __neg__(self):
        """Invert the sign of this profile."""
        return ScaledProfile(-1, self)

    def __sub__(self, x):
        """Subtract another profile from this one."""
        return self.__add__(-x)


class ScaledProfile(Profile):
    """Profile times a constant value.

    f_1(x) = a*f(x)

    Parameters
    ----------
    profile : Profile
        Base profile to scale.
    scale : float
        Scale factor.

    """

    _io_attrs_ = Profile._io_attrs_ + ["_profile", "_scale"]

    def __init__(self, scale, profile, **kwargs):
        assert isinstance(
            profile, Profile
        ), "profile in a ScaledProfile must be a Profile or subclass, got {}.".format(
            str(profile)
        )
        assert np.isscalar(scale), "scale must be a scalar."

        self._profile = profile.copy()
        self._scale = scale

        kwargs.setdefault("name", profile.name)
        kwargs.setdefault("grid", profile.grid)
        super().__init__(**kwargs)

    @property
    def grid(self):
        """Grid: Nodes for computation."""
        return self._grid

    @grid.setter
    def grid(self, new):
        Profile.grid.fset(self, new)
        self._profile.grid = new

    @property
    def params(self):
        """ndarray: Parameters for computation."""
        return (self._scale, self._profile.params)

    @params.setter
    def params(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 2:
            params = x[1]
            scale = x[0]
        elif np.isscalar(x):
            scale = x
            params = self._profile.params
        else:
            scale = self._scale
            params = x
        self._scale = scale
        self._profile.params = params

    def compute(self, params=None, grid=None, dr=0, dt=0, dz=0):
        """Compute values of profile at specified nodes.

        Parameters
        ----------
        params : array-like
            Parameters to use. If not given, uses the
            values given by the self.params attribute.
        grid : Grid or array-like
            locations to compute values at. Defaults to self.grid.
        dr, dt, dz : int
            derivative order in rho, theta, zeta.

        Returns
        -------
        values : ndarray
            values of the profile or its derivative at the points specified.

        """
        f = self._profile.compute(params, grid, dr, dt, dz)
        return self._scale * f

    def __repr__(self):
        """Get the string form of the object."""
        s = super().__repr__()
        s = s[:-1]
        s += ", scale={})".format(self._scale)
        return s


class SumProfile(Profile):
    """Sum of two or more Profiles.

    f(x) = f1(x) + f2(x) + f3(x) ...

    Parameters
    ----------
    profiles : Profile
        Profiles to sum.

    """

    _io_attrs_ = Profile._io_attrs_ + ["_profiles"]

    def __init__(self, *profiles, **kwargs):
        self._profiles = []
        for profile in profiles:
            assert isinstance(profile, Profile), (
                "Each profile in a SumProfile must be a Profile or "
                + "subclass, got {}.".format(str(profile))
            )
            if isinstance(profile, SumProfile):
                self._profiles += [pro.copy() for pro in profile._profiles]
            else:
                self._profiles.append(profile.copy())
        super().__init__(**kwargs)

    @property
    def grid(self):
        """Grid: Nodes for computation."""
        return self._grid

    @grid.setter
    def grid(self, new):
        Profile.grid.fset(self, new)
        for profile in self._profiles:
            profile.grid = new

    @property
    def params(self):
        """ndarray: Parameters for computation."""
        return tuple(profile.params for profile in self._profiles)

    @params.setter
    def params(self, x):
        if isinstance(x, (list, tuple)) and len(x) == len(self._profiles):
            for i, profile in enumerate(self._profiles):
                profile.params = x[i] if x[i] is not None else profile.params
        else:
            for profile in self._profiles:
                profile.params = x

    def compute(self, params=None, grid=None, dr=0, dt=0, dz=0):
        """Compute values of profile at specified nodes.

        Parameters
        ----------
        params : array-like
            Parameters to use. If not given, uses the
            values given by the self.params attribute.
        grid : Grid or array-like
            locations to compute values at. Defaults to self.grid.
        dr, dt, dz : int
            derivative order in rho, theta, zeta.

        Returns
        -------
        values : ndarray
            values of the profile or its derivative at the points specified.

        """
        if params is None:
            params = [None] * len(self._profiles)
        if isinstance(params, dict):
            params = [params]
        f = 0
        for i, profile in enumerate(self._profiles):
            f += profile.compute(params[i % len(params)], grid, dr, dt, dz)
        return f

    def __repr__(self):
        """Get the string form of the object."""
        s = super().__repr__()
        s = s[:-1]
        s += ", with {} profiles)".format(len(self._profiles))
        return s


class ProductProfile(Profile):
    """Product of two or more Profiles.

    f(x) = f1(x) * f2(x) * f3(x) ...

    Parameters
    ----------
    profiles : Profile
        Profiles to multiply.

    """

    _io_attrs_ = Profile._io_attrs_ + ["_profiles"]

    def __init__(self, *profiles, **kwargs):

        self._profiles = []
        for profile in profiles:
            assert isinstance(profile, Profile), (
                "Each profile in a ProductProfile must be a Profile or "
                + "subclass, got {}.".format(str(profile))
            )
            if isinstance(profile, ProductProfile):
                self._profiles += [pro.copy() for pro in profile._profiles]
            else:
                self._profiles.append(profile.copy())
        super().__init__(**kwargs)

    @property
    def grid(self):
        """Grid: Nodes for computation."""
        return self._grid

    @grid.setter
    def grid(self, new):
        Profile.grid.fset(self, new)
        for profile in self._profiles:
            profile.grid = new

    @property
    def params(self):
        """ndarray: Parameters for computation."""
        return tuple(profile.params for profile in self._profiles)

    @params.setter
    def params(self, x):
        if isinstance(x, (list, tuple)) and len(x) == len(self._profiles):
            for i, profile in enumerate(self._profiles):
                profile.params = x[i] if x[i] is not None else profile.params
        else:
            for profile in self._profiles:
                profile.params = x

    def compute(self, params=None, grid=None, dr=0, dt=0, dz=0):
        """Compute values of profile at specified nodes.

        Parameters
        ----------
        params : array-like
            Parameters to use. If not given, uses the
            values given by the self.params attribute.
        grid : Grid or array-like
            locations to compute values at. Defaults to self.grid.
        dr, dt, dz : int
            derivative order in rho, theta, zeta.

        Returns
        -------
        values : ndarray
            values of the profile or its derivative at the points specified.

        """
        if params is None:
            params = [None] * len(self._profiles)
        if isinstance(params, dict):
            params = [params]
        f = 1
        for i, profile in enumerate(self._profiles):
            f *= profile.compute(params[i % len(params)], grid, dr, dt, dz)
        return f

    def __repr__(self):
        """Get the string form of the object."""
        s = super().__repr__()
        s = s[:-1]
        s += ", with {} profiles)".format(len(self._profiles))
        return s


class PowerSeriesProfile(Profile):
    """Profile represented by a monic power series.

    f(x) = a[0] + a[1]*x + a[2]*x**2 + ...

    Parameters
    ----------
    params: array-like
        Coefficients of the series. If modes is not supplied, assumed to be in ascending
        order with no missing values. If modes is given, coefficients can be in any
        order or indexing.
    modes : array-like
        Mode numbers for the associated coefficients. eg a[modes[i]] = params[i]
    grid : Grid
        Default grid to use for computing values using transform method.
    sym : bool
        Whether the basis should only contain even powers (True) or all powers (False).
    name : str
        Name of the profile.
    """

    _io_attrs_ = Profile._io_attrs_ + ["_basis", "_transform"]

    def __init__(self, params, modes=None, grid=None, sym="auto", name=""):
        super().__init__(grid, name)

        params = np.atleast_1d(params)

        if sym == "auto":  # sym = "even" if all odd modes are zero, else sym = False
            if modes is None:
                modes = np.arange(params.size)
            else:
                modes = np.atleast_1d(modes)
            sym = np.all(params[modes % 2 != 0] == 0)
        sym = "even" if sym else False
        if modes is None:
            if sym:
                modes = np.arange(2 * params.size, step=2)
            else:
                modes = np.arange(params.size)
        else:
            modes = np.atleast_1d(modes)
        self._basis = PowerSeries(L=int(np.max(abs(modes))), sym=sym)
        self._params = np.zeros(self.basis.num_modes, dtype=float)
        for m, c in zip(modes, params):
            idx = np.where(self.basis.modes[:, 0] == int(m))[0]
            self._params[idx] = c
        self._transform = self._get_transform(self.grid)

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
            build=True,
        )
        return transform

    def __repr__(self):
        """Get the string form of the object."""
        s = super().__repr__()
        s = s[:-1]
        s += ", basis={})".format(self.basis)
        return s

    @property
    def sym(self):
        """str: Symmetry type of the power series."""
        return self.basis.sym

    @property
    def basis(self):
        """PowerSeriesBasis: Spectral basis for power series."""
        return self._basis

    @property
    def grid(self):
        """Grid: Nodes for computation."""
        return self._grid

    @grid.setter
    def grid(self, new):
        Profile.grid.fset(self, new)
        if hasattr(self, "_transform"):
            self._transform.grid = self.grid
            self._transform.build()

    @property
    def params(self):
        """ndarray: Parameter values."""
        return self._params

    @params.setter
    def params(self, new):
        new = jnp.atleast_1d(new)
        if new.size == self._basis.num_modes:
            self._params = jnp.asarray(new)
        else:
            raise ValueError(
                "params should have the same size as the basis, "
                + f"got {len(new)} for basis with {self._basis.num_modes} modes"
            )

    def get_params(self, l):
        """Get power series coefficients for given mode number(s)."""
        l = np.atleast_1d(l).astype(int)
        a = np.zeros_like(l).astype(float)

        idx = np.where(l[:, np.newaxis] == self.basis.modes[:, 0])

        a[idx[0]] = self.params[idx[1]]
        return a

    def set_params(self, l, a=None):
        """Set specific power series coefficients."""
        l, a = np.atleast_1d(l), np.atleast_1d(a)
        a = np.broadcast_to(a, l.shape)
        for ll, aa in zip(l, a):
            idx = self.basis.get_idx(ll, 0, 0)
            if aa is not None:
                self.params = put(self.params, idx, aa)

    def change_resolution(self, L, M=None, N=None):
        """Set a new maximum mode number."""
        modes_old = self.basis.modes
        self.basis.change_resolution(L)
        self._transform = self._get_transform(self.grid)
        self.params = copy_coeffs(self.params, modes_old, self.basis.modes)

    def compute(self, params=None, grid=None, dr=0, dt=0, dz=0):
        """Compute values of profile at specified nodes.

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
    def from_values(
        cls, x, y, order=6, rcond=None, w=None, grid=None, sym="auto", name=""
    ):
        """Fit a PowerSeriesProfile from point data.

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
        sym : bool
            Whether the basis should only contain even powers (T) or all powers (F).
        name : str
            name of the profile

        Returns
        -------
        profile : PowerSeriesProfile
            profile in power series basis fit to given data.

        """
        params = np.polyfit(x, y, order, rcond=rcond, w=w, full=False)[::-1]
        return cls(params, grid=grid, sym=sym, name=name)


class SplineProfile(Profile):
    """Profile represented by a piecewise cubic spline.

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
        - `'catmull-rom'`: C1 cubic centripetal "tension" splines
    grid : Grid
        default grid to use for computing values using transform method
    name : str
        name of the profile

    """

    _io_attrs_ = Profile._io_attrs_ + ["_knots", "_method", "_Dx"]

    def __init__(self, values, knots=None, grid=None, method="cubic2", name=""):

        super().__init__(grid, name)

        values = np.atleast_1d(values)
        if knots is None:
            knots = np.linspace(0, 1, values.size)
        else:
            knots = np.atleast_1d(knots)
        self._knots = knots
        self._params = values
        self._method = method
        self._Dx = _approx_df(
            self._knots, np.eye(self._knots.size), self._method, axis=0
        )

    def __repr__(self):
        """Get the string form of the object."""
        s = super().__repr__()
        s = s[:-1]
        s += ", method={}, num_knots={})".format(self._method, len(self._knots))
        return s

    @property
    def grid(self):
        """Grid: Nodes for computation."""
        return self._grid

    @grid.setter
    def grid(self, new):
        Profile.grid.fset(self, new)

    @property
    def params(self):
        """ndarray: Parameters for computation."""
        return self._params

    @params.setter
    def params(self, new):
        if len(new) == len(self._knots):
            self._params = jnp.asarray(new)
        else:
            raise ValueError(
                "params should have the same size as the knots, "
                + f"got {len(new)} values for {len(self._knots)} knots"
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
        """Compute values of profile at specified nodes.

        Parameters
        ----------
        params : array-like
            spline values to use. If not given, uses the
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
        xq = self._get_xq(grid)
        if dt != 0 or dz != 0:
            return jnp.zeros_like(xq)
        x = self._knots
        f = params
        df = self._Dx @ f
        fq = interp1d(xq, x, f, method=self._method, derivative=dr, extrap=True, df=df)
        return fq


class MTanhProfile(Profile):
    r"""Profile represented by a modified hyperbolic tangent + polynomial.

    Profile is parameterized by pedestal height (ped, :math:`p`), SOL height
    (offset, :math:`o`), pedestal symmetry point (sym, :math:`s`), pedestal width
    (width, :math:`w`), and a polynomial:

    .. math::

        f = o + 1/2 (o - p) (\tanh(z) - 1) + 1/2 (o - p) g(y)

    Where :math:`z=(x-s)/w`, :math:`y=e^z/(e^{2z}+1)`, and :math:`g` is a polynomial
    with no constant term

    Parameters
    ----------
    params: array-like
        parameters for mtanh + poly. ``params = [ped, offset, sym, width, *core_poly]``
        where core poly are the polynomial coefficients in ascending order, without
        a constant term
    grid : Grid
        default grid to use for computing values using transform method
    name : str
        name of the profile

    """

    def __init__(self, params, grid=None, name=""):

        super().__init__(grid, name)

        self._params = params

    def __repr__(self):
        """Get the string form of the object."""
        s = super().__repr__()
        s = s[:-1]
        s += ", num_params={})".format(len(self._params))
        return s

    @property
    def params(self):
        """ndarray: Parameter values."""
        return self._params

    @params.setter
    def params(self, new):
        new = jnp.atleast_1d(new)
        if new.size >= 5:
            self._params = jnp.asarray(new)
        else:
            raise ValueError(
                "params should have at least 5 elements [ped, offset, sym, width,"
                + f"*core_poly]  got only {new.size} values"
            )

    @staticmethod
    def _mtanh(x, ped, offset, sym, width, core_poly, dx=0):
        """Compute modified tanh + polynomial profile.

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
            y = (ped - offset) * (jnp.tanh(-z) ** 2 - 1) * jnp.tanh(-z) / width**2

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
                / (width**3 * (e2z + 1) ** 3)
            )
            f = (
                jnp.polyval(jnp.polyder(core_poly[::-1], 2), zz) * dz**2
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
        """Compute values of profile at specified nodes.

        Parameters
        ----------
        params : array-like
            coefficients to use, in order. [ped, offset, sym, width, core_poly]
            If not given, uses the values given by the params attribute
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
        name="",
        **kwargs,
    ):
        """Fit a MTanhProfile from point data.

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
        jac = jit(Derivative(fun, 0, "fwd").compute)
        fun = jit(fun)
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
            fun, jac=jac, x0=p0, method="trf", bounds=(pmin, pmax), **kwargs
        )
        if not out.success:
            warnings.warn("Fitting did not converge, parameters may not be correct")
        params = out.x
        return MTanhProfile(params, grid, name)


class FourierZernikeProfile(Profile):
    """Possibly anisotropic profile represented by Fourier-Zernike basis.

    Parameters
    ----------
    params: array-like, shape(k,)
        coefficients of the series. If modes is not supplied, assumed to be only radial
        modes in ascending order with no missing values. If modes is given, coefficients
        can be in any order or indexing.
    modes : array-like, shape(k,3)
        mode numbers for the associated coefficients. eg a[modes[i]] = params[i].
        If None, assumes params are only the m=0 n=0 modes
    grid : Grid
        default grid to use for computing values using transform method
    sym : {"auto", "sin", "cos", False}
        Whether the basis should be stellarator symmetric.
    name : str
        name of the profile.

    """

    _io_attrs_ = Profile._io_attrs_ + ["_basis", "_transform"]

    def __init__(self, params, modes=None, grid=None, sym="auto", NFP=1, name=""):
        super().__init__(grid, name)

        params = np.atleast_1d(params)

        if modes is None:
            modes = np.hstack(
                [
                    np.atleast_2d(np.arange(0, 2 * len(params), 2)).T,
                    np.zeros((len(params), 2)),
                ]
            )
        modes = np.asarray(modes)
        assert np.all(modes.astype(int) == modes), "mode numbers should be integers"
        modes = modes.astype(int)

        L = np.max(abs(modes[:, 0]))
        M = np.max(abs(modes[:, 1]))
        N = np.max(abs(modes[:, 2]))
        if sym == "auto":
            if np.all(params[np.where(sign(modes[:, 1]) != sign(modes[:, 2]))] == 0):
                sym = "cos"
            elif np.all(params[np.where(sign(modes[:, 1]) == sign(modes[:, 2]))] == 0):
                sym = "sin"
            else:
                sym = False

        self._basis = FourierZernikeBasis(L=L, M=M, N=N, NFP=NFP, sym=sym)

        self._params = copy_coeffs(params, modes, self.basis.modes)

        self._transform = self._get_transform(self.grid)

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
            build=True,
        )
        return transform

    def __repr__(self):
        """Get the string form of the object."""
        s = super().__repr__()
        s = s[:-1]
        s += ", basis={})".format(self.basis)
        return s

    @property
    def basis(self):
        """FourierZernikeBasis: Spectral basis for Fourier-Zernike series."""
        return self._basis

    @property
    def grid(self):
        """Grid: Nodes for computation."""
        return self._grid

    @grid.setter
    def grid(self, new):
        Profile.grid.fset(self, new)
        if hasattr(self, "_transform"):
            self._transform.grid = self.grid
            self._transform.build()

    @property
    def params(self):
        """ndarray: Parameter values."""
        return self._params

    @params.setter
    def params(self, new):
        new = jnp.atleast_1d(new)
        if new.size == self._basis.num_modes:
            self._params = jnp.asarray(new)
        else:
            raise ValueError(
                f"params should have the same size as the basis, got {new.size} "
                + f"for basis with {self._basis.num_modes} modes"
            )

    def get_params(self, l, m, n):
        """Get Fourier-Zernike coefficients for given mode number(s)."""
        l = np.atleast_1d(l).astype(int)
        m = np.atleast_1d(m).astype(int)
        n = np.atleast_1d(n).astype(int)
        a = np.zeros_like(l).astype(float)

        for i, (ll, mm, nn) in enumerate(zip(l, m, n)):
            idx = self.basis.get_idx(ll, mm, nn)
            a[i] = self.params[idx]
        return a

    def set_params(self, l, m, n, a=None):
        """Set specific Fourier-Zernike coefficients."""
        l, m, n, a = map(np.atleast_1d, (l, m, n, a))
        a = np.broadcast_to(a, l.shape)
        for ll, mm, nn, aa in zip(l, m, n, a):
            idx = self.basis.get_idx(ll, mm, nn)
            if aa is not None:
                self.params = put(self.params, idx, aa)

    def change_resolution(self, L=None, M=None, N=None):
        """Set a new maximum mode number."""
        modes_old = self.basis.modes
        L = L if L is not None else self.basis.L
        M = M if M is not None else self.basis.M
        N = N if N is not None else self.basis.N
        self.basis.change_resolution(L, M, N)
        self._transform = self._get_transform(self.grid)
        self.params = copy_coeffs(self.params, modes_old, self.basis.modes)

    def compute(self, params=None, grid=None, dr=0, dt=0, dz=0):
        """Compute values of profile at specified nodes.

        Parameters
        ----------
        params : array-like
            Fourier-Zernike coefficients to use, in ascending order. If not given,
            uses the values given by the params attribute
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
    def from_values(cls, r, t, z, f, L=6, M=0, N=0, NFP=1, w=None, grid=None, name=""):
        """Fit a FourierZernikeProfile from point data.

        Parameters
        ----------
        r, t, z : array-like, shape(k,)
            coordinate locations in rho, theta, zeta
        f : array-like, shape(k,)
            function values
        L, M, N : int
            maximum mode numbers to fit
        NFP : int
            number of field periods
        w : array-like, shape(k,)
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
        nodes = np.vstack([r, t, z]).T
        basis = FourierZernikeBasis(L, M, N, NFP)
        A = basis.evaluate(nodes)
        if w is not None:
            A *= w[:, np.newaxis]
            f *= w
        scale = np.sqrt((A * A).sum(axis=0))
        scale = np.where(scale == 0, 1, scale)
        A /= scale
        c, resids, rank, s = np.linalg.lstsq(A, f, rcond=None)
        c = (c.T / scale).T  # broadcast scale coefficients
        return cls(c, modes=basis.modes, NFP=NFP, grid=grid, name=name)
