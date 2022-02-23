import numpy as np
from collections.abc import MutableSequence
from desc.backend import jnp
from desc.geometry.core import Curve, xyz2rpz, xyz2rpz_vec, rpz2xyz, rpz2xyz_vec
from desc.magnetic_fields import MagneticField, biot_savart
from desc.grid import Grid


class Coil(MagneticField, Curve):
    """Class representing a magnetic field coil, as a combination of a curve and current

    Parameters
    ----------
    curve : Curve
        underlying geometric curve definining path of coil
    current : float
        current passing through the coil, in Amperes
    """

    _io_attrs_ = MagneticField._io_attrs_ + ["_curve", "_current"]

    def __init__(self, curve, current, name=""):
        super(Coil, self).__init__(name)
        assert isinstance(curve, Curve)
        self._curve = curve
        self._current = current

    @property
    def curve(self):
        return self._curve

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._current = new

    @property
    def grid(self):
        """Default grid for computation."""
        return self.curve.grid

    @grid.setter
    def grid(self, new):
        if isinstance(new, Grid):
            self.curve.grid = new
        elif isinstance(new, (np.ndarray, jnp.ndarray)):
            self.curve.grid = Grid(new, sort=False)
        else:
            raise TypeError(
                f"grid should be a Grid or subclass, or ndarray, got {type(new)}"
            )

    def compute_coordinates(self, *args, **kwargs):
        """Compute real space coordinates using underlying curve method."""
        return self.curve.compute_coordinates(*args, **kwargs)

    def compute_frenet_frame(self, *args, **kwargs):
        """Compute Frenet frame using underlying curve method."""
        return self.curve.compute_frenet_frame(*args, **kwargs)

    def compute_curvature(self, *args, **kwargs):
        """Compute curvature using underlying curve method."""
        return self.curve.compute_curvature(*args, **kwargs)

    def compute_torsion(self, *args, **kwargs):
        """Compute torsion using underlying curve method."""
        return self.curve.compute_torsion(*args, **kwargs)

    def compute_length(self, *args, **kwargs):
        """Compute the length of the curve using underlying curve method."""
        return self.curve.compute_length(*args, **kwargs)

    def translate(self, *args, **kwargs):
        """translate the coil along an axis"""
        self.curve.translate(*args, **kwargs)

    def rotate(self, *args, **kwargs):
        """rotate the coil about an axis"""
        self.curve.rotate(*args, **kwargs)

    def flip(self, *args, **kwargs):
        """flip the coil across a plane"""
        self.curve.flip(*args, **kwargs)

    def compute_magnetic_field(self, coords, params={}, basis="rpz"):
        """Compute magnetic field at a set of points

        Parameters
        ----------
        coords : array-like shape(n,3) or Grid
            coordinates to evaluate field at [R,phi,Z] or [x,y,z]
        params : dict, optional
            parameters to pass to curve
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field

        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates
        """
        assert basis.lower() in ["rpz", "xyz"]
        if isinstance(coords, Grid):
            coords = coords.nodes
        coords = jnp.atleast_2d(coords)
        if basis == "rpz":
            coords = rpz2xyz(coords)
        current = params.pop("current", self.current)
        coil_coords = self.curve.compute_coordinates(**params, basis="xyz")
        B = biot_savart(coords, coil_coords, current)
        if basis == "rpz":
            B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        return B
