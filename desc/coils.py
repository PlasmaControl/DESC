from desc.backend import jnp
from desc.geometry.core import Curve, xyz2rpz, xyz2rpz_vec, rpz2xyz, rpz2xyz_vec
from desc.magnetic_fields import MagneticField, biot_savart
from desc.grid import Grid


class Coil(Curve, MagneticField):
    """Class representing a magnetic field coil, as a combination of a curve and current

    Parameters
    ----------
    curve : Curve
        underlying geometric curve definining path of coil
    current : float
        current passing through the coil, in Amperes
    """

    def __init__(self, curve, current):
        assert isinstance(curve, Curve)
        self.curve = curve
        self.current = current

    def compute_coordinates(self, *args, **kwargs):
        """Compute real space coordinates on predefined grid."""
        return self.curve.compute_coordinates(*args, **kwargs)

    def compute_frenet_frame(self, *args, **kwargs):
        """Compute Frenet frame on predefined grid."""
        return self.curve.compute_frenet_frame(*args, **kwargs)

    def compute_curvature(self, *args, **kwargs):
        """Compute curvature on predefined grid."""
        return self.curve.compute_curvature(*args, **kwargs)

    def compute_torsion(self, *args, **kwargs):
        """Compute torsion on predefined grid."""
        return self.curve.compute_torsion(*args, **kwargs)

    def compute_length(self, *args, **kwargs):
        """Compute the length of the curve using specified nodes for quadrature."""
        return self.curve.compute_length(*args, **kwargs)

    def compute_magnetic_field(
        self, coords, curve_grid=None, dR=0, dp=0, dZ=0, params={}
    ):
        """Compute magnetic field at a set of points

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical coordinates to evaluate field at [R,phi,Z]
        curve_grid : Grid, ndarray or int
            grid to use for evaluating curve geometry. If an integer, assumes that
            many equally spaced points in (0,2pi)
        dR, dp, dZ : int, optional
            order of derivative to take in R,phi,Z directions
        params : dict, optional
            parameters to pass to curve

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points, in cylindrical form [BR, Bphi,BZ]

        """
        if any([(dR != 0), (dp != 0), (dZ != 0)]):
            raise NotImplementedError(
                "Derivatives of coil fields have not been implemented"
            )
        if isinstance(coords, Grid):
            coords = coords.nodes
        coords = jnp.atleast_2d(coords)
        coordsxyz = rpz2xyz(coords)
        current = params.pop("current", self.current)
        coil_coords = self.compute_coordinates(grid=curve_grid, **params, basis="xyz")
        B = biot_savart(coordsxyz, coil_coords, current)
        B = xyz2rpz_vec(B, phi=coords[..., 1])
        return B
