import numpy as np
from termcolor import colored
from abc import ABC, abstractmethod
import copy
import scipy.optimize

from desc.backend import jnp, put
from desc.io import IOAble
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.interpolate import interp1d
from desc.transform import Transform
from desc.basis import PowerSeries


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

    _object_lib_ = {
        "Grid": Grid,
        "LinearGrid": LinearGrid,
        "ConcentricGrid": ConcentricGrid,
        "QuadratureGrid": QuadratureGrid,
    }

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
    def transform(self, params, dr=0, dt=0, dz=0):
        """compute profile values on default grid using specified coefficients"""

    @abstractmethod
    def compute(nodes, params=None, dr=0, dt=0, dz=0):
        """compute values on specified nodes, default to using self.params"""

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this profile."""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        return new

    def __call__(self, nodes, params=None, dr=0, dt=0, dz=0):
        return self.compute(nodes, params, dr, dt, dz)


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
    _object_lib_ = Profile._object_lib_
    _object_lib_.update({"PowerSeries": PowerSeries, "Transform": Transform})

    def __init__(self, params, modes=None, grid=None, name=None):

        self._name = name
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
        self._transform = Transform(
            self.grid, self.basis, derivs=np.array([[0, 0, 0], [1, 0, 0]])
        )

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

    # TODO: methods for setting individual components, getting indices of modes etc
    @params.setter
    def params(self, new):
        if len(new) == self._basis.num_modes:
            self._params = jnp.asarray(new)
        else:
            raise ValueError(
                f"params should have the same size as the basis, got {len(new)} for basis with {self._basis.num_modes} modes"
            )

    def transform(self, params, dr=0, dt=0, dz=0):
        """Compute values using specified coefficients

        Parameters
        ----------
        params: array-like
            polynomial coefficients to use
        dr, dt, dz : int
            derivative order in rho, theta, zeta

        Returns
        -------
        values : ndarray
            values of the profile or its derivative at the points specified by the
            grid attribute

        """
        return self._transform.transform(params, dr=dr, dt=dt, dz=dz)

    def compute(self, nodes, params=None, dr=0, dt=0, dz=0):
        """Compute values of profile at specified nodes

        Parameters
        ----------
        nodes : ndarray, shape(k,) or (k,3)
            locations to compute values at
        params : array-like
            polynomial coefficients to use, in ascending order. If not given, uses the
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
        if nodes.ndim == 1:
            nodes = np.pad(nodes[:, np.newaxis], ((0, 0), (0, 2)))
        A = self.basis.evaluate(nodes, derivatives=[dr, dt, dz])
        return jnp.dot(A, params)

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
        values = self.compute(knots)
        return SplineProfile(values, knots, self.grid, method, self.name)

    def to_mtanh(self, order=4, xs=100, w=None, p0=None, pmax=np.inf, pmin=-np.inf):
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
        ys = self.compute(xs)
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

    def transform(self, params, dr=0, dt=0, dz=0):
        """Compute values using specified coefficients

        Parameters
        ----------
        params: array-like
            polynomial coefficients to use
        dr, dt, dz : int
            derivative order in rho, theta, zeta

        Returns
        -------
        values : ndarray
            values of the profile or its derivative at the points specified by the
            grid attribute

        """
        xq = self.grid.nodes[:, 0]
        if dt != 0 or dz != 0:
            return jnp.zeros_like(xq)
        x = self._knots
        f = params
        fq = interp1d(xq, x, f, method=self._method, derivative=dr, extrap=True)
        return fq

    def compute(self, nodes, params=None, dr=0, dt=0, dz=0):
        """Compute values of profile at specified nodes

        Parameters
        ----------
        nodes : ndarray, shape(k,) or (k,3)
            locations to compute values at
        params : array-like
            polynomial coefficients to use, in ascending order. If not given, uses the
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

        xq = nodes[:, 0] if nodes.ndim > 1 else nodes
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
        fs = self.compute(xs)
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
        values = self.compute(knots)
        return SplineProfile(values, knots, self.grid, method, self.name)

    def to_mtanh(self, order=4, xs=100, w=None, p0=None, pmax=np.inf, pmin=-np.inf):
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
        ys = self.compute(xs)
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
        )


class MTanhProfile(Profile):
    """Profile represented by a modified hyperbolic tangent + polynomial


    Parameters
    ----------
    params: array-like
        parameters for mtanh + poly. p = [height, offset, sym, width, *core_poly] where
        core poly are the polynomial coefficients in ascending order
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
                f"params should have at least 5 elements [height, offset, sym, width, core_poly[0]]  got only {len(new)} values"
            )

    # TODO: check these parameter definitions and formulas
    @staticmethod
    def _mtanh(x, height, offset, sym, width, core_poly, dx=0):
        """modified tanh + polynomial profile

        Parameters
        ----------
        x : ndarray
            evaluation locations
        height : float
            height of pedestal
        offset : float
            height of SOL
        sym : float
            symmetry point
        width : float
            width of pedestal
        core_poly : ndarray
            polynomial coefficients in descending order [x^n, x^n-1,...x^0]
        dx : int
           radial derivative order

        Returns
        -------
        y : ndarray
            profile evaluated at x
        """
        # Z shifted function
        z = (sym - x) / width
        # core polynomial
        p0 = jnp.polyval(core_poly, z)
        # mtanh part
        m0 = MTanhProfile._m(z)
        # offsets + scale
        A = (height + offset) / 2
        B = (height - offset) / 2
        if dx == 0:
            y = A * p0 * m0 + B
        elif dx == 1:
            dzdx = -1 / width
            p1 = jnp.polyval(jnp.polyder(core_poly, 1), z)
            m1 = MTanhProfile._m(z, dz=1)
            y = A * dzdx * (p1 * m0 + p0 * m1)
        elif dx == 2:
            dzdx = -1 / width
            p1 = jnp.polyval(jnp.polyder(core_poly, 1), z)
            m1 = MTanhProfile._m(z, dz=1)
            p2 = jnp.polyval(jnp.polyder(core_poly, 2), z)
            m2 = MTanhProfile._m(z, dz=2)
            y = A * dzdx ** 2 * (p2 * m0 + 2 * m1 * p1 + m2 * p0)

        return y

    @staticmethod
    def _m(z, dz=0):
        if dz == 0:
            m = 1 / (1 + jnp.exp(-2 * z))
        elif dz == 1:
            m0 = MTanhProfile._m(z, dz=0)
            m = 2 * jnp.exp(-2 * z) * m0 ** 2
        elif dz == 2:
            m0 = MTanhProfile._m(z, dz=0)
            m1 = MTanhProfile._m(z, dz=1)
            m = 4 * jnp.exp(-2 * z) * m1 * m0 - 2 * m1
        return m

    def transform(self, params, dr=0, dt=0, dz=0):
        """Compute values using specified coefficients

        Parameters
        ----------
        params: array-like
            polynomial coefficients to use
        dr, dt, dz : int
            derivative order in rho, theta, zeta

        Returns
        -------
        values : ndarray
            values of the profile or its derivative at the points specified by the
            grid attribute

        """
        nodes = self.grid.nodes
        y = self.compute(nodes, params, dr, dt, dz)
        return y

    def compute(self, nodes, params=None, dr=0, dt=0, dz=0):
        """Compute values of profile at specified nodes

        Parameters
        ----------
        nodes : ndarray, shape(k,) or (k,3)
            locations to compute values at
        params : array-like
            polynomial coefficients to use, in ascending order. If not given, uses the
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

        xq = nodes[:, 0] if nodes.ndim > 1 else nodes
        height = params[0]
        offset = params[1]
        sym = params[2]
        width = params[3]
        core_poly = params[4:]

        if dt != 0 or dz != 0:
            return jnp.zeros_like(xq)

        y = MTanhProfile._mtanh(xq, height, offset, sym, width, core_poly, dx=dr)
        return y

    @classmethod
    def from_values(
        cls,
        x,
        y,
        order=4,
        w=None,
        p0=None,
        pmax=np.inf,
        pmin=-np.inf,
        grid=None,
        name=None,
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
        p0 : array-like, shape(5+order,)
            initial guess for parameter values
        pmin : float or array-like, shape(5+order,)
            lower bounds for parameter values
        pmax : float or array-like, shape(5+order,)
            upper bounds for parameter values
        grid : Grid
            default grid to use for computing values using transform method
        name : str
            name of the profile

        Returns
        -------
        profile : MTanhProfile
            profile in mtanh + polynomial form.

        """
        fun = lambda x, *args: cls._mtanh(
            x, args[0], args[1], args[2], args[3], args[4:]
        )
        if p0 is None:
            p0 = np.zeros(order + 5)
            p0[0] = 1.0
            p0[1] = 1.0
            p0[3] = 1.0
            p0[4] = 1.0
        params, unc = scipy.optimize.curve_fit(
            fun, x, y, p0, w, method="trf", bounds=(pmin, pmax)
        )
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
        fs = self.compute(xs)
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
        values = self.compute(knots)
        return SplineProfile(values, knots, self.grid, method, self.name)
