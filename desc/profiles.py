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
    _io_attrs_ = ["_name", "_grid", "_coeffs"]

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
    def coeffs(self):
        """Default coefficients for computation"""

    @coeffs.setter
    @abstractmethod
    def coeffs(self, new):
        """Set default coeffs for computation"""

    @abstractmethod
    def transform(self, coeffs, dr=0, dt=0, dz=0):
        """compute profile values on default grid using specified coefficients"""

    @abstractmethod
    def compute(nodes, coeffs=None, dr=0, dt=0, dz=0):
        """compute values on specified nodes, default to using self.coeffs"""

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this profile."""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        return new

    def __call__(self, nodes, coeffs=None, dr=0, dt=0, dz=0):
        return self.compute(nodes, coeffs, dr, dt, dz)


class PowerSeriesProfile(Profile):
    _io_attrs_ = Profile._io_attrs_ + ["_basis", "_transform"]
    _object_lib_ = Profile._object_lib_
    _object_lib_.update({"PowerSeries": PowerSeries, "Transform": Transform})

    def __init__(self, coeffs, modes=None, grid=None, name=None):

        self._name = name
        if modes is None:
            modes = np.arange(coeffs.size)
        self._basis = PowerSeries(L=int(np.max(abs(modes))))
        self._coeffs = np.zeros(self.basis.num_modes, dtype=float)
        for m, c in zip(modes, coeffs):
            idx = np.where(self.basis.modes[:, 0] == int(m))[0]
            self._coeffs[idx] = c
        if grid is None:
            grid = Grid(np.empty((0, 3)))
        self._grid = grid
        self._transform = Transform(
            self.grid, self.basis, derivs=np.array([[0, 0, 0], [1, 0, 0]])
        )

    @property
    def basis(self):
        return self._basis

    @property
    def grid(self):
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
    def coeffs(self):
        return self._coeffs

    # TODO: methods for setting individual components, getting indices of modes etc
    @coeffs.setter
    def coeffs(self, new):
        if len(new) == self._basis.num_modes:
            self._coeffs = jnp.asarray(new)
        else:
            raise ValueError(
                f"coeffs should have the same size as the basis, got {len(new)} for basis with {self._basis.num_modes} modes"
            )

    def transform(self, coeffs, dr=0, dt=0, dz=0):
        return self._transform.transform(coeffs, dr=dr, dt=dt, dz=dz)

    def compute(self, nodes, coeffs=None, dr=0, dt=0, dz=0):
        if coeffs is None:
            coeffs = self.coeffs
        if nodes.ndim == 1:
            nodes = np.pad(nodes[:, np.newaxis], ((0, 0), (0, 2)))
        A = self.basis.evaluate(nodes, derivatives=[dr, dt, dz])
        return jnp.dot(A, coeffs)

    @classmethod
    def from_values(cls, x, y, order=6, rcond=None, w=None):
        coeffs = np.polyfit(x, y, order, rcond=rcond, w=w, full=False)[::-1]
        return cls(coeffs)

    def to_powerseries(self, order=6, xs=100, rcond=None, w=None):
        if len(self.coeffs) == order + 1:
            coeffs = self.coeffs
        elif len(self.coeffs) > order + 1:
            coeffs = self.coeffs[: order + 1]
        elif len(self.coeffs) < order + 1:
            coeffs = np.pad(self.coeffs, (0, order + 1 - len(self.coeffs)))
        modes = np.arange(order + 1)

        return PowerSeriesProfile(coeffs, modes, self.grid, self.name)

    def to_spline(self, knots=20, method="cubic2"):
        if np.isscalar(knots):
            knots = np.linspace(0, 1, knots)
        values = self.compute(knots)
        return SplineProfile(values, knots, self.grid, method, self.name)

    def to_mtanh(self, order=4, xs=100, w=None, p0=None, pmax=np.inf, pmin=-np.inf):

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
    _io_attrs_ = Profile._io_attrs_ + ["_knots", "_method"]

    def __init__(self, values, knots=None, grid=None, method="cubic2", name=None):

        if knots is None:
            knots = np.linspace(0, 1, values.size)
        self._name = name
        self._knots = knots
        self._coeffs = values
        self._method = method

        if grid is None:
            grid = Grid(np.empty((0, 3)))
        self.grid = grid

    @property
    def grid(self):
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
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, new):
        if len(new) == len(self._knots):
            self._coeffs = jnp.asarray(new)
        else:
            raise ValueError(
                f"coeffs should have the same size as the knots, got {len(new)} values for {len(self._knots)} knots"
            )

    def transform(self, coeffs, dr=0, dt=0, dz=0):

        xq = self.grid.nodes[:, 0]
        if dt != 0 or dz != 0:
            return jnp.zeros_like(xq)
        x = self._knots
        f = coeffs
        fq = interp1d(xq, x, f, method=self._method, derivative=dr, extrap=True)
        return fq

    def compute(self, nodes, coeffs=None, dr=0, dt=0, dz=0):
        if coeffs is None:
            coeffs = self.coeffs

        xq = nodes[:, 0] if nodes.ndim > 1 else nodes
        if dt != 0 or dz != 0:
            return jnp.zeros_like(xq)
        x = self._knots
        f = coeffs
        fq = interp1d(xq, x, f, method=self._method, derivative=dr, extrap=True)
        return fq

    def to_powerseries(self, order=6, xs=100, rcond=None, w=None):
        if np.isscalar(xs):
            xs = np.linspace(0, 1, xs)
        fs = self.compute(xs)
        p = PowerSeriesProfile.from_values(xs, fs, order, rcond=rcond, w=w)
        p.grid = self.grid
        p.name = self.name
        return p

    def to_mtanh(self, order=4, xs=100, w=None, p0=None, pmax=np.inf, pmin=-np.inf):

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
    def __init__(self, coeffs, grid=None, name=None):

        self._name = name
        self._coeffs = coeffs

        if grid is None:
            grid = Grid(np.empty((0, 3)))
        self.grid = grid

    @property
    def grid(self):
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
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, new):
        if len(new) >= 5:
            self._coeffs = jnp.asarray(new)
        else:
            raise ValueError(
                f"coeffs should have at least 5 elements [height, offset, sym, width, core_poly[0]]  got only {len(new)} values"
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

    def transform(self, coeffs, dr=0, dt=0, dz=0):

        nodes = self.grid.nodes
        y = self.compute(nodes, coeffs, dr, dt, dz)
        return y

    def compute(self, nodes, coeffs=None, dr=0, dt=0, dz=0):

        if coeffs is None:
            coeffs = self.coeffs

        xq = nodes[:, 0] if nodes.ndim > 1 else nodes
        height = coeffs[0]
        offset = coeffs[1]
        sym = coeffs[2]
        width = coeffs[3]
        core_poly = coeffs[4:]

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

        fun = lambda x, *args: cls._mtanh(
            x, args[0], args[1], args[2], args[3], args[4:]
        )
        if p0 is None:
            p0 = np.zeros(order + 5)
            p0[0] = 1.0
            p0[1] = 1.0
            p0[3] = 1.0
            p0[4] = 1.0
        coeffs, unc = scipy.optimize.curve_fit(
            fun, x, y, p0, w, method="trf", bounds=(pmin, pmax)
        )
        return MTanhProfile(coeffs, grid, name)

    def to_powerseries(self, order=6, xs=100, rcond=None, w=None):
        if np.isscalar(xs):
            xs = np.linspace(0, 1, xs)
        fs = self.compute(xs)
        p = PowerSeriesProfile.from_values(xs, fs, order, rcond=rcond, w=w)
        p.grid = self.grid
        p.name = self.name
        return p

    def to_spline(self, knots=20, method="cubic2"):
        if np.isscalar(knots):
            knots = np.linspace(0, 1, knots)
        values = self.compute(knots)
        return SplineProfile(values, knots, self.grid, method, self.name)
