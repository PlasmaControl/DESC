"""Base classes for curves and surfaces."""

import numbers
from abc import ABC, abstractmethod

import numpy as np

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import data_index
from desc.compute.utils import (
    _parse_parameterization,
    get_data_deps,
    get_params,
    get_transforms,
)
from desc.grid import LinearGrid, QuadratureGrid, _Grid
from desc.io import IOAble
from desc.optimizable import Optimizable, optimizable_parameter
from desc.utils import errorif, reflection_matrix, rotation_matrix


class Curve(IOAble, Optimizable, ABC):
    """Abstract base class for 1D curves in 3D space."""

    _io_attrs_ = ["_name", "_shift", "_rotmat"]
    _static_attrs = Optimizable._static_attrs + ["_name"]

    def __init__(self, name=""):
        self._shift = jnp.array([0, 0, 0], dtype=float)
        self._rotmat = jnp.eye(3, dtype=float).flatten()
        self._name = name

    def _set_up(self):
        """Set things after loading."""
        if hasattr(self, "_NFP"):
            self._NFP = int(self._NFP)
        if not hasattr(self, "_shift"):
            self.shift
        if not hasattr(self, "_rotmat"):
            self.rotmat

    @optimizable_parameter
    @property
    def shift(self):
        """Displacement of curve in X, Y, Z."""
        return self.__dict__.setdefault("_shift", jnp.array([0, 0, 0], dtype=float))

    @shift.setter
    def shift(self, new):
        if len(new) == 3:
            self._shift = jnp.asarray(new)
        else:
            raise ValueError("shift should be a 3 element vector, got {}".format(new))

    @optimizable_parameter
    @property
    def rotmat(self):
        """Rotation matrix of curve in X, Y, Z."""
        return self.__dict__.setdefault("_rotmat", jnp.eye(3, dtype=float).flatten())

    @rotmat.setter
    def rotmat(self, new):
        if len(new) == 9:
            self._rotmat = jnp.asarray(new)
        else:
            self._rotmat = jnp.asarray(new.flatten())

    @property
    def name(self):
        """Name of the curve."""
        return self.__dict__.setdefault("_name", "")

    @name.setter
    def name(self, new):
        self._name = str(new)

    def compute(
        self,
        names,
        grid=None,
        params=None,
        transforms=None,
        data=None,
        override_grid=True,
        **kwargs,
    ):
        """Compute the quantity given by name on grid.

        Parameters
        ----------
        names : str or array-like of str
            Name(s) of the quantity(s) to compute.
        grid : Grid or int, optional
            Grid of coordinates to evaluate at. Defaults to a Linear grid.
            If an integer, uses that many equally spaced points.
        params : dict of ndarray
            Parameters from the equilibrium. Defaults to attributes of self.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from grid
        data : dict of ndarray
            Data computed so far, generally output from other compute functions.
            Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
            v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
            of the cylindrical coordinates R, ϕ, Z.
        override_grid : bool
            If True, override the user supplied grid if necessary and use a full
            resolution grid to compute quantities and then downsample to user requested
            grid. If False, uses only the user specified grid, which may lead to
            inaccurate values for surface or volume averages.

        Returns
        -------
        data : dict of ndarray
            Computed quantity and intermediate variables.

        """
        if isinstance(names, str):
            names = [names]
        if grid is None:
            grid = LinearGrid(N=2 * self.N * getattr(self, "NFP", 1) + 5)
        elif isinstance(grid, numbers.Integral):
            grid = LinearGrid(N=grid)
        errorif(
            not isinstance(grid, _Grid),
            TypeError,
            f"grid argument must be a Grid object or an integer, got type {type(grid)}",
        )

        if params is None:
            params = get_params(names, obj=self, basis=kwargs.get("basis", "rpz"))
        if transforms is None:
            transforms = get_transforms(
                names,
                obj=self,
                grid=grid,
                jitable=True,
            )
        if data is None:
            data = {}
        profiles = {}

        p = _parse_parameterization(self)
        deps = list(set(get_data_deps(names, obj=p) + names))
        dep0d = [
            dep
            for dep in deps
            if (data_index[p][dep]["coordinates"] == "") and (dep not in data)
        ]
        calc0d = bool(len(dep0d))
        # see if the grid we're already using will work for desired qtys
        if calc0d and (grid.N >= 2 * self.N + 5) and isinstance(grid, LinearGrid):
            calc0d = False

        if calc0d and override_grid:
            grid0d = LinearGrid(N=2 * self.N * getattr(self, "NFP", 1) + 5)
            data0d = compute_fun(
                self,
                dep0d,
                params=params,
                transforms=get_transforms(
                    dep0d,
                    obj=self,
                    grid=grid0d,
                    jitable=True,
                    **kwargs,
                ),
                profiles={},
                data=None,
                **kwargs,
            )
            # these should all be 0d quantities so don't need to compress/expand
            data0d = {key: val for key, val in data0d.items() if key in dep0d}
            data.update(data0d)

        data = compute_fun(
            self,
            names,
            params=params,
            transforms=transforms,
            profiles=profiles,
            data=data,
            **kwargs,
        )
        return data

    def translate(self, displacement=[0, 0, 0]):
        """Translate the curve by a rigid displacement in X,Y,Z coordinates."""
        self.shift = self.shift + jnp.asarray(displacement)

    def rotate(self, axis=[0, 0, 1], angle=0):
        """Rotate the curve by a fixed angle about axis in X,Y,Z coordinates."""
        R = rotation_matrix(axis=axis, angle=angle)
        self.rotmat = (R @ self.rotmat.reshape(3, 3)).flatten()
        self.shift = self.shift @ R.T

    def flip(self, normal=[0, 0, 1]):
        """Flip the curve about the plane with specified normal in X,Y,Z coordinates."""
        F = reflection_matrix(normal)
        self.rotmat = (F @ self.rotmat.reshape(3, 3)).flatten()
        self.shift = self.shift @ F.T

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={})".format(self.name)
        )

    def to_FourierXYZ(self, N=10, grid=None, s=None, name=""):
        """Convert Curve to FourierXYZCurve representation.

        Parameters
        ----------
        N : int
            Fourier resolution of the new X,Y,Z representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierXYZCurve.
            If an integer, uses that many equally spaced points.
        s : ndarray or "arclength"
            arbitrary curve parameter to use for the fitting.
            Should be monotonic, 1D array of same length as
            coords. if None, defaults linearly spaced in [0,2pi)
            Alternative, can pass "arclength" to use normalized distance between points.
        name : str
            name for this curve

        Returns
        -------
        curve : FourierXYZCurve
            New representation of the curve parameterized by Fourier series for X,Y,Z.

        """
        from .curve import FourierXYZCurve

        if (grid is None) and (s is not None) and (not isinstance(s, str)):
            grid = LinearGrid(zeta=s)
        if grid is None:
            grid = LinearGrid(N=2 * N + 1)
        coords = self.compute("x", grid=grid, basis="xyz")["x"]
        return FourierXYZCurve.from_values(coords, N=N, s=s, basis="xyz", name=name)

    def to_SplineXYZ(self, knots=None, grid=None, method="cubic", name=""):
        """Convert Curve to SplineXYZCurve.

        Parameters
        ----------
        knots : ndarray or "arclength"
            arbitrary curve parameter values to use for spline knots,
            should be an 1D ndarray of same length as the input.
            (input length in this case is determined by grid argument, since
            the input coordinates come from Curve.compute("x",grid=grid))
            If None, defaults to using an linearly spaced points in [0, 2pi) as the
            knots. If supplied, should lie in [0,2pi].
            Alternatively, the string "arclength" can be supplied to use the normalized
            distance between points.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with SplineXYZCurve.
            If an integer, uses that many equally spaced points.
        method : str
            method of interpolation
            - `'nearest'`: nearest neighbor interpolation
            - `'linear'`: linear interpolation
            - `'cubic'`: C1 cubic splines (aka local splines)
            - `'cubic2'`: C2 cubic splines (aka natural splines)
            - `'catmull-rom'`: C1 cubic centripetal "tension" splines
        name : str
            name for this curve

        Returns
        -------
        SplineXYZCurve: SplineXYZCurve
            New representation of the curve parameterized by a spline for X,Y,Z.

        """
        from .curve import SplineXYZCurve

        if (grid is None) and (knots is not None) and (not isinstance(knots, str)):
            grid = LinearGrid(zeta=knots)
        coords = self.compute("x", grid=grid, basis="xyz")["x"]
        return SplineXYZCurve.from_values(
            coords, knots=knots, method=method, name=name, basis="xyz"
        )

    def to_FourierRZ(self, N=10, grid=None, NFP=None, sym=False, name=""):
        """Convert Curve to FourierRZCurve representation.

        Note that some types of curves may not be representable in this basis.

        Parameters
        ----------
        N : int
            Fourier resolution of the new R,Z representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierRZCurve.
            If an integer, uses that many equally spaced points.
        NFP : int
            Number of field periods, the curve will have a discrete toroidal symmetry
            according to NFP.
        sym : bool, optional
            Whether the curve is stellarator-symmetric or not. Default is False.
        name : str
            name for this curve

        Returns
        -------
        curve : FourierRZCurve
            New representation of the curve parameterized by Fourier series for R,Z.

        """
        from .curve import FourierRZCurve

        NFP = 1 or NFP
        if grid is None:
            grid = LinearGrid(N=2 * N + 1)
        coords = self.compute("x", grid=grid, basis="xyz")["x"]
        return FourierRZCurve.from_values(
            coords, N=N, NFP=NFP, basis="xyz", name=name, sym=sym
        )

    def to_FourierPlanar(self, N=10, grid=None, basis="xyz", name=""):
        """Convert Curve to FourierPlanarCurve representation.

        Note that some types of curves may not be representable in this basis.
        In this case, a least-squares fit will be done to find the
        planar curve that best represents the curve.

        Parameters
        ----------
        N : int
            Fourier resolution of the new FourierPlanarCurve representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierPlanarCurve.
            If an integer, uses that many equally spaced points.
        basis : {'xyz', 'rpz'}
            Coordinate system for center and normal vectors. Default = 'xyz'.
        name : str
            name for this curve

        Returns
        -------
        curve : FourierPlanarCurve
            New representation of the curve parameterized by Fourier series for
            minor radius r in a plane specified by a center position and normal
            vector.

        """
        from .curve import FourierPlanarCurve

        if grid is None:
            grid = LinearGrid(N=2 * N + 1)
        coords = self.compute("x", grid=grid, basis=basis)["x"]
        return FourierPlanarCurve.from_values(coords, N=N, basis=basis, name=name)

    def to_FourierXY(self, N=10, grid=None, s=None, basis="xyz", name=""):
        """Convert Curve to FourierXYCurve representation.

        Note that some types of curves may not be representable in this basis.
        In this case, a least-squares fit will be done to find the
        planar curve that best represents the curve.

        Parameters
        ----------
        N : int
            Fourier resolution of the new FourierXYCurve representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierXYCurve.
            If an integer, uses that many equally spaced points.
        s : ndarray or "arclength"
            Arbitrary curve parameter to use for the fitting.
            Should be monotonic, 1D array of same length as
            coords. if None, defaults linearly spaced in [0,2pi)
            Alternative, can pass "arclength" to use normalized distance between points.
        basis : {'xyz', 'rpz'}
            Coordinate system for center and normal vectors. Default = 'xyz'.
        name : str
            name for this curve

        Returns
        -------
        curve : FourierXYCurve
            New representation of the curve parameterized by Fourier series for the
            X and Y coordinates in a plane specified by a center position and normal
            vector.

        """
        from .curve import FourierXYCurve

        if (grid is None) and (s is not None) and (not isinstance(s, str)):
            grid = LinearGrid(zeta=s)
        if grid is None:
            grid = LinearGrid(N=2 * N + 1)
        coords = self.compute("x", grid=grid, basis=basis)["x"]
        return FourierXYCurve.from_values(coords, N=N, s=s, basis=basis, name=name)


class Surface(IOAble, Optimizable, ABC):
    """Abstract base class for 2d surfaces in 3d space."""

    _io_attrs_ = ["_name", "_sym", "_L", "_M", "_N"]
    _static_attrs = Optimizable._static_attrs + ["_name", "_sym", "_L", "_M", "_N"]

    def _set_up(self):
        """Set things after loading."""
        if hasattr(self, "_NFP"):
            self._NFP = int(self._NFP)
        self._L = int(self._L)
        self._M = int(self._M)
        self._N = int(self._N)

    @property
    def name(self):
        """str: Name of the surface."""
        return self.__dict__.setdefault("_name", "")

    @name.setter
    def name(self, new):
        self._name = str(new)

    @property
    def L(self):
        """int: Maximum radial mode number."""
        return self._L

    @property
    def M(self):
        """int: Maximum poloidal mode number."""
        return self._M

    @property
    def N(self):
        """int: Maximum toroidal mode number."""
        return self._N

    @property
    def sym(self):
        """bool: Whether or not the surface is stellarator symmetric."""
        return self._sym

    def _compute_orientation(self):
        """Handedness of coordinate system.

        Returns
        -------
        orientation : float
            +1 for right handed coordinate system (theta increasing CW),
            -1 for left handed coordinates (theta increasing CCW),
            or 0 for a singular coordinate system (no volume)
        """
        R0 = self.R_lmn[self.R_basis.get_idx(0, 0, 0, False)]
        R0 = R0 if R0.size > 0 else 0
        Rsin = self.R_lmn[self.R_basis.get_idx(0, -1, 0, False)]
        Rsin = Rsin if Rsin.size > 0 else 0
        Rcos = self.R_lmn[self.R_basis.get_idx(0, 1, 0, False)]
        Rcos = Rcos if Rcos.size > 0 else 0
        Zsin = self.Z_lmn[self.Z_basis.get_idx(0, -1, 0, False)]
        Zsin = Zsin if Zsin.size > 0 else 0
        Zcos = self.Z_lmn[self.Z_basis.get_idx(0, 1, 0, False)]
        Zcos = Zcos if Zcos.size > 0 else 0
        out = np.sign((R0 + Rcos) * (Rsin * Zcos - Rcos * Zsin))
        assert (out == -1) or (out == 0) or (out == 1)
        return out

    def _flip_orientation(self):
        """Flip the orientation of theta."""
        one = np.ones_like(self.R_lmn)
        one[self.R_basis.modes[:, 1] < 0] *= -1
        self.R_lmn *= one
        one = np.ones_like(self.Z_lmn)
        one[self.Z_basis.modes[:, 1] < 0] *= -1
        self.Z_lmn *= one

    @abstractmethod
    def change_resolution(self, *args, **kwargs):
        """Change the maximum resolution."""

    def compute(
        self,
        names,
        grid=None,
        params=None,
        transforms=None,
        data=None,
        override_grid=True,
        **kwargs,
    ):
        """Compute the quantity given by name on grid.

        Parameters
        ----------
        names : str or array-like of str
            Name(s) of the quantity(s) to compute.
        grid : Grid, optional
            Grid of coordinates to evaluate at. Defaults to a Linear grid for constant
            rho surfaces or a Quadrature grid for constant zeta surfaces.
        params : dict of ndarray
            Parameters from the equilibrium. Defaults to attributes of self.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from grid
        data : dict of ndarray
            Data computed so far, generally output from other compute functions
        override_grid : bool
            If True, override the user supplied grid if necessary and use a full
            resolution grid to compute quantities and then downsample to user requested
            grid. If False, uses only the user specified grid, which may lead to
            inaccurate values for surface or volume averages.

        Returns
        -------
        data : dict of ndarray
            Computed quantity and intermediate variables.

        """
        if isinstance(names, str):
            names = [names]
        if grid is None:
            if hasattr(self, "rho"):  # constant rho surface
                grid = LinearGrid(
                    rho=np.array(self.rho),
                    M=2 * self.M + 5,
                    N=2 * self.N + 5,
                    NFP=self.NFP,
                )
            elif hasattr(self, "zeta"):  # constant zeta surface
                grid = QuadratureGrid(L=2 * self.L + 5, M=2 * self.M + 5, N=0, NFP=1)
                grid._nodes[:, 2] = self.zeta
        elif not isinstance(grid, _Grid):
            raise TypeError(
                "must pass in a Grid object or an integer for argument grid!"
                f" instead got type {type(grid)}"
            )

        if params is None:
            params = get_params(names, obj=self, basis=kwargs.get("basis", "rpz"))
        if transforms is None:
            transforms = get_transforms(
                names,
                obj=self,
                grid=grid,
                jitable=kwargs.pop("jitable", False),
                **kwargs,
            )
        if data is None:
            data = {}
        profiles = {}

        p = _parse_parameterization(self)
        deps = list(set(get_data_deps(names, obj=p) + names))
        dep0d = [
            dep
            for dep in deps
            if (data_index[p][dep]["coordinates"] == "") and (dep not in data)
        ]
        calc0d = bool(len(dep0d))
        # see if the grid we're already using will work for desired qtys
        if calc0d and override_grid and hasattr(self, "rho"):  # constant rho surface
            if (
                (grid.N >= 2 * self.N + 5)
                and (grid.M > 2 * self.M + 5)
                and isinstance(grid, LinearGrid)
            ):
                calc0d = False
            else:
                grid0d = LinearGrid(
                    rho=np.array(self.rho),
                    M=2 * self.M + 5,
                    N=2 * self.N + 5,
                    NFP=self.NFP,
                )
        elif (
            calc0d and override_grid and hasattr(self, "zeta")
        ):  # constant zeta surface
            if (
                (grid.L >= self.L + 1)
                and (grid.M > 2 * self.M + 5)
                and isinstance(grid, QuadratureGrid)
            ):
                calc0d = False
            else:
                grid0d = QuadratureGrid(L=2 * self.L + 5, M=2 * self.M + 5, N=0, NFP=1)
                grid0d._nodes[:, 2] = self.zeta

        if calc0d and override_grid:
            data0d = compute_fun(
                self,
                dep0d,
                params=params,
                transforms=get_transforms(
                    dep0d,
                    obj=self,
                    grid=grid0d,
                    jitable=kwargs.pop("jitable", False),
                    **kwargs,
                ),
                profiles={},
                data=None,
                **kwargs,
            )
            # these should all be 0d quantities so don't need to compress/expand
            data0d = {key: val for key, val in data0d.items() if key in dep0d}
            data.update(data0d)

        data = compute_fun(
            self,
            names,
            params=params,
            transforms=transforms,
            profiles=profiles,
            data=data,
            **kwargs,
        )
        return data

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={})".format(self.name)
        )
