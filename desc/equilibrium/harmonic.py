"""Classes for parameterized 3D space curves."""

import numpy as np

from desc.backend import jnp, put
from desc.basis import FourierSeries
from desc.compute import rpz2xyz, xyz2rpz
from desc.grid import LinearGrid
from desc.io import InputReader
from desc.optimizable import optimizable_parameter
from desc.transform import Transform
from desc.utils import check_nonnegint, check_posint, copy_coeffs, errorif

from .core import Curve

__all__ = ["FourierRZCurve", "FourierXYZCurve", "FourierPlanarCurve", "SplineXYZCurve"]

class HarmonicSuperPos(Misc, Optimizable):
    """Curve parameterized by Fourier series for R,Z in terms of toroidal angle phi.

    Parameters
    ----------
    a,b: array-like
        Scalars coefficients for harmonic fields
    NFP : int
        Number of field periods.
    sym : bool
        Whether to enforce stellarator symmetry.
    name : str
        Name for this curve.

    """

    _io_attrs_ = Curve._io_attrs_ + [
        "_a,
        "_b",
        "_sym",
        "_NFP",
    ]

    def __init__(
        self,
        eq,
        field,
        a = 1,
        b = 1,
        modes_R=None,
        modes_Z=None,
        plasma_grid=None,
        NFP=1,
        sym="auto",
        name="Superposition of harmonic fields of a surface",
    ):

        if target is None and bounds is None:
            bounds = (1, np.inf)
        self._eq = eq
        self._plasma_grid = plasma_grid
        
        super().__init__(name)
        a, b = np.atleast_1d(a), np.atleast_1d(b)

        if a.size == 0:
            raise ValueError("At least 1 coefficient must be supplied")
            
        if b.size == 0:
            b = np.array([0.0])

        if sym == "auto":
            if a == 0 and b == 0:
                sym = True
            else:
                sym = False
        
        self._sym = sym
        self._NFP = check_posint(NFP, "NFP", False)
        
        self._field = field
        self._a = a
        self._b = b

    @property
    def sym(self):
        """Whether this curve has stellarator symmetry."""
        return self._sym

    @optimizable_parameter
    @property
    def a(self):
        """Spectral coefficients for R."""
        return self._a

    @optimizable_parameter
    @property
    def b(self):
        """Spectral coefficients for Z."""
        return self._a

    
    @classmethod
    def from_values(cls, coords, N=10, NFP=1, basis="rpz", name="", sym=False):
        """Fit coordinates to FourierRZCurve representation.

        Parameters
        ----------
        coords: ndarray, shape (num_coords,3)
            coordinates to fit a FourierRZCurve object with each column
            corresponding to xyz or rpz depending on the basis argument.
        N : int
            Fourier resolution of the new R,Z representation.
        NFP : int
            Number of field periods, the curve will have a discrete toroidal symmetry
            according to NFP.
        basis : {"rpz", "xyz"}
            basis for input coordinates. Defaults to "rpz"

        Returns
        -------
        curve : FourierRZCurve
            New representation of the curve parameterized by Fourier series for R,Z.

        """
        if basis == "rpz":
            coords_rpz = coords
            coords_xyz = rpz2xyz(coords)
        else:
            coords_rpz = xyz2rpz(coords)
            coords_xyz = coords
        R = coords_rpz[:, 0]
        phi = coords_rpz[:, 1]

        X = coords_xyz[:, 0]
        Y = coords_xyz[:, 1]
        Z = coords_rpz[:, 2]

        # easiest to check closure in XYZ coordinates
        X, Y, Z, _, _, _, input_curve_was_closed = _unclose_curve(X, Y, Z)
        if input_curve_was_closed:
            R = R[0:-1]
            phi = phi[0:-1]
            Z = coords_rpz[0:-1, 2]
        # check if any phi are negative, and make them positive instead
        # so we can more easily check if it is monotonic
        inds_negative = np.where(phi < 0)
        phi = phi.at[inds_negative].set(phi[inds_negative] + 2 * np.pi)

        # assert the curve is not doubling back on itself in phi,
        # which can't be represented with a curve parameterized by phi
        errorif(
            not np.all(np.diff(phi) > 0), ValueError, "Supplied phi must be monotonic"
        )

        grid = LinearGrid(zeta=phi, NFP=1, sym=sym)
        basis = FourierSeries(N=N, NFP=NFP, sym=sym)
        transform = Transform(grid, basis, build_pinv=True)
        R_n = transform.fit(R)
        Z_n = transform.fit(Z)
        
        return FourierRZCurve(R_n=R_n, Z_n=Z_n, NFP=NFP, name=name, sym=sym)