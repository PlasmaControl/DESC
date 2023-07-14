"""Base classes for curves and surfaces."""

from abc import ABC, abstractmethod

import numpy as np

from desc.backend import jnp
from desc.compute.utils import compute as compute_fun
from desc.compute.utils import get_params, get_transforms
from desc.grid import LinearGrid
from desc.io import IOAble

from .utils import reflection_matrix, rotation_matrix


class Curve(IOAble, ABC):
    """Abstract base class for 1D curves in 3D space."""

    _io_attrs_ = ["_name", "shift", "rotmat"]

    def __init__(self, name=""):
        self.shift = jnp.array([0, 0, 0])
        self.rotmat = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.name = name

    @property
    def name(self):
        """Name of the curve."""
        return self._name

    @name.setter
    def name(self, new):
        self._name = new

    @property
    @abstractmethod
    def grid(self):
        """Grid: Nodes for computation."""

    def compute(
        self,
        names,
        grid=None,
        params=None,
        transforms=None,
        data=None,
        **kwargs,
    ):
        """Compute the quantity given by name on grid.

        Parameters
        ----------
        names : str or array-like of str
            Name(s) of the quantity(s) to compute.
        grid : Grid, optional
            Grid of coordinates to evaluate at. Defaults to the Linear grid.
        params : dict of ndarray
            Parameters from the equilibrium. Defaults to attributes of self.
        transforms : dict of Transform
            Transforms for R, Z, lambda, etc. Default is to build from grid
        data : dict of ndarray
            Data computed so far, generally output from other compute functions

        Returns
        -------
        data : dict of ndarray
            Computed quantity and intermediate variables.

        """
        if isinstance(names, str):
            names = [names]
        if grid is None:
            NFP = self.NFP if hasattr(self, "NFP") else 1
            grid = LinearGrid(N=2 * self.N + 5, NFP=NFP, endpoint=True)

        if params is None:
            params = get_params(names, obj=self)
        if transforms is None:
            transforms = get_transforms(names, obj=self, grid=grid, **kwargs)
        if data is None:
            data = {}
        profiles = {}

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
        """Translate the curve by a rigid displacement in x, y, z."""
        self.shift += jnp.asarray(displacement)

    def rotate(self, axis=[0, 0, 1], angle=0):
        """Rotate the curve by a fixed angle about axis in xyz coordinates."""
        R = rotation_matrix(axis, angle)
        self.rotmat = R @ self.rotmat
        self.shift = self.shift @ R.T

    def flip(self, normal):
        """Flip the curve about the plane with specified normal."""
        F = reflection_matrix(normal)
        self.rotmat = F @ self.rotmat
        self.shift = self.shift @ F.T

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={})".format(self.name)
        )


class Surface(IOAble, ABC):
    """Abstract base class for 2d surfaces in 3d space."""

    _io_attrs_ = ["_name", "_sym", "_L", "_M", "_N"]

    @property
    def name(self):
        """str: Name of the surface."""
        return self._name

    @name.setter
    def name(self, new):
        self._name = new

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

    @property
    @abstractmethod
    def grid(self):
        """Grid: Nodes for computation."""

    @abstractmethod
    def change_resolution(self, *args, **kwargs):
        """Change the maximum resolution."""

    @abstractmethod
    def compute_coordinates(self, params=None, grid=None, dt=0, dz=0):
        """Compute coordinate values at specified nodes."""

    @abstractmethod
    def compute_normal(self, params=None, grid=None):
        """Compute normal vectors to the surface on predefined grid."""

    @abstractmethod
    def compute_surface_area(self, params=None, grids=None):
        """Compute surface area via quadrature."""

    def compute_curvature(self, R_lmn=None, Z_lmn=None, grid=None):
        """Compute gaussian and mean curvature.

        Parameters
        ----------
        R_lmn, Z_lmn: array-like
            fourier coefficients for R, Z. Defaults to self.R_lmn, self.Z_lmn
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)

        Returns
        -------
        K, H, k1, k2 : ndarray, shape(k,)
            Gaussian, mean and 2 principle curvatures at points specified in grid.

        """
        # following notation from
        # https://en.wikipedia.org/wiki/Parametric_surface#Curvature
        E, F, G = self._compute_first_fundamental_form(R_lmn, Z_lmn, grid)
        L, M, N = self._compute_second_fundamental_form(R_lmn, Z_lmn, grid)
        # coeffs of quadratic eqn for determinant
        a = E * G - F**2
        b = F * M - L * G - E * N
        c = L * N - M**2
        r1 = (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
        r2 = (-b - jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)
        k1 = jnp.maximum(r1, r2)
        k2 = jnp.minimum(r1, r2)
        K = k1 * k2
        H = (k1 + k2) / 2
        return K, H, k1, k2

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={})".format(self.name)
        )
