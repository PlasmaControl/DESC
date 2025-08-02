"""Classes for parameterized 3D space curves."""

import os
import warnings

import numpy as np

from desc.backend import jnp, put
from desc.basis import FourierSeries
from desc.grid import LinearGrid
from desc.io import InputReader
from desc.optimizable import optimizable_parameter
from desc.transform import Transform
from desc.utils import (
    check_nonnegint,
    check_posint,
    copy_coeffs,
    errorif,
    rotation_matrix,
    rpz2xyz,
    rpz2xyz_vec,
    warnif,
    xyz2rpz,
    xyz2rpz_vec,
)

from .core import Curve

__all__ = [
    "FourierPlanarCurve",
    "FourierRZCurve",
    "FourierXYCurve",
    "FourierXYZCurve",
    "SplineXYZCurve",
]


class FourierRZCurve(Curve):
    """Curve parameterized by Fourier series for R,Z in terms of toroidal angle phi.

    Parameters
    ----------
    R_n, Z_n: array-like
        Fourier coefficients for R, Z.
    modes_R : array-like, optional
        Mode numbers associated with R_n. If not given defaults to [-n:n].
    modes_Z : array-like, optional
        Mode numbers associated with Z_n, If not given defaults to [-n:n]].
    NFP : int
        Number of field periods.
    sym : bool
        Whether to enforce stellarator symmetry.
    name : str
        Name for this curve.

    """

    _io_attrs_ = Curve._io_attrs_ + [
        "_R_n",
        "_Z_n",
        "_R_basis",
        "_Z_basis",
        "_sym",
        "_NFP",
    ]

    _static_attrs = Curve._static_attrs + ["_sym", "_NFP", "_R_basis", "_Z_basis"]

    def __init__(
        self,
        R_n=10,
        Z_n=0,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        name="",
    ):
        super().__init__(name)
        R_n, Z_n = np.atleast_1d(R_n), np.atleast_1d(Z_n)
        if modes_R is None:
            modes_R = np.arange(-(R_n.size // 2), R_n.size // 2 + 1)
        if modes_Z is None:
            modes_Z = np.arange(-(Z_n.size // 2), Z_n.size // 2 + 1)

        if R_n.size == 0:
            raise ValueError("At least 1 coefficient for R must be supplied")
        if Z_n.size == 0:
            Z_n = np.array([0.0])
            modes_Z = np.array([0])

        modes_R, modes_Z = np.asarray(modes_R), np.asarray(modes_Z)

        assert R_n.size == modes_R.size, "R_n size and modes_R must be the same size"
        assert Z_n.size == modes_Z.size, "Z_n size and modes_Z must be the same size"

        assert issubclass(modes_R.dtype.type, np.integer)
        assert issubclass(modes_Z.dtype.type, np.integer)

        if sym == "auto":
            if np.all(R_n[modes_R < 0] == 0) and np.all(Z_n[modes_Z >= 0] == 0):
                sym = True
            else:
                sym = False
        self._sym = sym
        NR = np.max(abs(modes_R))
        NZ = np.max(abs(modes_Z))
        N = max(NR, NZ)
        self._NFP = check_posint(NFP, "NFP", False)
        self._R_basis = FourierSeries(N, int(NFP), sym="cos" if sym else False)
        self._Z_basis = FourierSeries(N, int(NFP), sym="sin" if sym else False)

        self._R_n = copy_coeffs(R_n, modes_R, self.R_basis.modes[:, 2])
        self._Z_n = copy_coeffs(Z_n, modes_Z, self.Z_basis.modes[:, 2])

    @property
    def sym(self):
        """bool: Whether or not the curve is stellarator symmetric."""
        return self._sym

    @property
    def R_basis(self):
        """Spectral basis for R_Fourier series."""
        return self._R_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z_Fourier series."""
        return self._Z_basis

    @property
    def NFP(self):
        """Number of field periods."""
        return self._NFP

    @property
    def N(self):
        """Maximum mode number."""
        return max(self.R_basis.N, self.Z_basis.N)

    def change_resolution(self, N=None, NFP=None, sym=None):
        """Change the maximum toroidal resolution."""
        N = check_nonnegint(N, "N")
        NFP = check_posint(NFP, "NFP")
        if (
            ((N is not None) and (N != self.N))
            or ((NFP is not None) and (NFP != self.NFP))
            or ((sym is not None) and (sym != self.sym))
        ):
            self._NFP = int(NFP if NFP is not None else self.NFP)
            self._sym = bool(sym) if sym is not None else self.sym
            N = int(N if N is not None else self.N)
            R_modes_old = self.R_basis.modes
            Z_modes_old = self.Z_basis.modes
            self.R_basis.change_resolution(
                N=N, NFP=self.NFP, sym="cos" if self.sym else self.sym
            )
            self.Z_basis.change_resolution(
                N=N, NFP=self.NFP, sym="sin" if self.sym else self.sym
            )
            self.R_n = copy_coeffs(self.R_n, R_modes_old, self.R_basis.modes)
            self.Z_n = copy_coeffs(self.Z_n, Z_modes_old, self.Z_basis.modes)

    def get_coeffs(self, n):
        """Get Fourier coefficients for given mode number(s)."""
        n = np.atleast_1d(n).astype(int)
        R = np.zeros_like(n).astype(float)
        Z = np.zeros_like(n).astype(float)

        idxR = np.where(n[:, np.newaxis] == self.R_basis.modes[:, 2])
        idxZ = np.where(n[:, np.newaxis] == self.Z_basis.modes[:, 2])

        R[idxR[0]] = self.R_n[idxR[1]]
        Z[idxZ[0]] = self.Z_n[idxZ[1]]
        return R, Z

    def set_coeffs(self, n, R=None, Z=None):
        """Set specific Fourier coefficients."""
        n, R, Z = np.atleast_1d(n), np.atleast_1d(R), np.atleast_1d(Z)
        R = np.broadcast_to(R, n.shape)
        Z = np.broadcast_to(Z, n.shape)
        for nn, RR, ZZ in zip(n, R, Z):
            if RR is not None:
                idxR = self.R_basis.get_idx(0, 0, nn)
                self.R_n = put(self.R_n, idxR, RR)
            if ZZ is not None:
                idxZ = self.Z_basis.get_idx(0, 0, nn)
                self.Z_n = put(self.Z_n, idxZ, ZZ)

    @optimizable_parameter
    @property
    def R_n(self):
        """Spectral coefficients for R."""
        return self._R_n

    @R_n.setter
    def R_n(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_n should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
            )

    @optimizable_parameter
    @property
    def Z_n(self):
        """Spectral coefficients for Z."""
        return self._Z_n

    @Z_n.setter
    def Z_n(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_n should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.Z_basis.num_modes} modes"
            )

    @classmethod
    def from_input_file(cls, path, **kwargs):
        """Create a axis curve from Fourier coefficients in a DESC or VMEC input file.

        Parameters
        ----------
        path : Path-like or str
            Path to DESC or VMEC input file.
        **kwargs : dict, optional
            keyword arguments to pass to the constructor of the
            FourierRZCurve being created.

        Returns
        -------
        curve : FourierRZToroidalCurve
            Axis with given Fourier coefficients.

        """
        path = os.path.expanduser(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = InputReader().parse_inputs(path)[-1]
        curve = FourierRZCurve(
            inputs["axis"][:, 1],
            inputs["axis"][:, 2],
            inputs["axis"][:, 0].astype(int),
            inputs["axis"][:, 0].astype(int),
            inputs["NFP"],
            inputs["sym"],
            **kwargs,
        )
        return curve

    @classmethod
    def from_values(cls, coords, N=10, NFP=1, sym=False, basis="rpz", name=""):
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
        sym : bool
            Whether to enforce stellarator symmetry.
        basis : {"rpz", "xyz"}
            basis for input coordinates. Defaults to "rpz"
        name : str
            Name for this curve.

        Returns
        -------
        curve : FourierRZCurve
            New representation of the curve parameterized by Fourier series for R,Z.

        """
        coords = jnp.atleast_2d(coords)
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
        R_basis = FourierSeries(N=N, NFP=NFP, sym="cos" if sym else False)
        Z_basis = FourierSeries(N=N, NFP=NFP, sym="sin" if sym else False)
        with warnings.catch_warnings():
            # grid and basis have uneven NFP because we want to allow the user to either
            # pass in an entire curve (in which case phi : 0-> 2pi) which has some
            # field-periodicity, or only a portion of the curve
            # (in which case phi: 0->2pi/NFP) and still have this fit work.
            warnings.filterwarnings("ignore", message="Unequal number of field periods")
            R_transform = Transform(grid, R_basis, build_pinv=True)
            Z_transform = Transform(grid, Z_basis, build_pinv=True)
        R_n = R_transform.fit(R)
        Z_n = Z_transform.fit(Z)
        return FourierRZCurve(
            R_n=R_n,
            Z_n=Z_n,
            modes_R=R_basis.modes[:, 2],
            modes_Z=Z_basis.modes[:, 2],
            NFP=NFP,
            sym=sym,
            name=name,
        )


def _unclose_curve(X, Y, Z):
    if (
        np.allclose([X[0], Y[0], Z[0]], [X[-1], Y[-1], Z[-1]], atol=1e-14)
        and X.size != 1
    ):
        closedX, closedY, closedZ = X.copy(), Y.copy(), Z.copy()
        X, Y, Z = X[:-1], Y[:-1], Z[:-1]
        flag = True
    else:
        closedX, closedY, closedZ = (
            np.append(X, X[0]),
            np.append(Y, Y[0]),
            np.append(Z, Z[0]),
        )
        flag = False
    return X, Y, Z, closedX, closedY, closedZ, flag


class FourierXYZCurve(Curve):
    """Curve parameterized by Fourier series for X,Y,Z in terms of an arbitrary angle s.

    Parameters
    ----------
    X_n, Y_n, Z_n: array-like
        Fourier coefficients for X, Y, Z
    modes : array-like
        mode numbers associated with X_n etc.
    name : str
        name for this curve

    """

    _io_attrs_ = Curve._io_attrs_ + [
        "_X_n",
        "_Y_n",
        "_Z_n",
        "_X_basis",
        "_Y_basis",
        "_Z_basis",
    ]

    _static_attrs = Curve._static_attrs + ["_X_basis", "_Y_basis", "_Z_basis"]

    def __init__(
        self,
        X_n=[0, 10, 2],
        Y_n=[0, 0, 0],
        Z_n=[-2, 0, 0],
        modes=None,
        name="",
    ):
        super().__init__(name)
        X_n, Y_n, Z_n = np.atleast_1d(X_n), np.atleast_1d(Y_n), np.atleast_1d(Z_n)
        if modes is None:
            modes = np.arange(-(X_n.size // 2), X_n.size // 2 + 1)
        else:
            modes = np.asarray(modes)

        assert issubclass(modes.dtype.type, np.integer)

        assert X_n.size == modes.size, "X_n and modes must be the same size"
        assert Y_n.size == modes.size, "Y_n and modes must be the same size"
        assert Z_n.size == modes.size, "Z_n and modes must be the same size"

        N = np.max(abs(modes))
        self._X_basis = FourierSeries(N, NFP=1, sym=False)
        self._Y_basis = FourierSeries(N, NFP=1, sym=False)
        self._Z_basis = FourierSeries(N, NFP=1, sym=False)
        self._X_n = copy_coeffs(X_n, modes, self.X_basis.modes[:, 2])
        self._Y_n = copy_coeffs(Y_n, modes, self.Y_basis.modes[:, 2])
        self._Z_n = copy_coeffs(Z_n, modes, self.Z_basis.modes[:, 2])

    @property
    def X_basis(self):
        """Spectral basis for X Fourier series."""
        return self._X_basis

    @property
    def Y_basis(self):
        """Spectral basis for Y Fourier series."""
        return self._Y_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z Fourier series."""
        return self._Z_basis

    @property
    def N(self):
        """Maximum mode number."""
        return max(self.X_basis.N, self.Y_basis.N, self.Z_basis.N)

    def change_resolution(self, N=None):
        """Change the maximum angular resolution."""
        N = check_nonnegint(N, "N")
        if (N is not None) and (N != self.N):
            N = int(N)
            Xmodes_old = self.X_basis.modes
            Ymodes_old = self.Y_basis.modes
            Zmodes_old = self.Z_basis.modes
            self.X_basis.change_resolution(N=N)
            self.Y_basis.change_resolution(N=N)
            self.Z_basis.change_resolution(N=N)
            self.X_n = copy_coeffs(self.X_n, Xmodes_old, self.X_basis.modes)
            self.Y_n = copy_coeffs(self.Y_n, Ymodes_old, self.Y_basis.modes)
            self.Z_n = copy_coeffs(self.Z_n, Zmodes_old, self.Z_basis.modes)

    def get_coeffs(self, n):
        """Get Fourier coefficients for given mode number(s)."""
        n = np.atleast_1d(n).astype(int)
        X = np.zeros_like(n).astype(float)
        Y = np.zeros_like(n).astype(float)
        Z = np.zeros_like(n).astype(float)

        Xidx = np.where(n[:, np.newaxis] == self.X_basis.modes[:, 2])
        Yidx = np.where(n[:, np.newaxis] == self.Y_basis.modes[:, 2])
        Zidx = np.where(n[:, np.newaxis] == self.Z_basis.modes[:, 2])

        X[Xidx[0]] = self.X_n[Xidx[1]]
        Y[Yidx[0]] = self.Y_n[Yidx[1]]
        Z[Zidx[0]] = self.Z_n[Zidx[1]]
        return X, Y, Z

    def set_coeffs(self, n, X=None, Y=None, Z=None):
        """Set specific Fourier coefficients."""
        n, X, Y, Z = (
            np.atleast_1d(n),
            np.atleast_1d(X),
            np.atleast_1d(Y),
            np.atleast_1d(Z),
        )
        X = np.broadcast_to(X, n.shape)
        Y = np.broadcast_to(Y, n.shape)
        Z = np.broadcast_to(Z, n.shape)
        for nn, XX in zip(n, X):
            idx = self.X_basis.get_idx(0, 0, nn)
            if XX is not None:
                self.X_n = put(self.X_n, idx, XX)

        for nn, YY in zip(n, Y):
            idx = self.Y_basis.get_idx(0, 0, nn)
            if YY is not None:
                self.Y_n = put(self.Y_n, idx, YY)

        for nn, ZZ in zip(n, Z):
            idx = self.Z_basis.get_idx(0, 0, nn)
            if ZZ is not None:
                self.Z_n = put(self.Z_n, idx, ZZ)

    @optimizable_parameter
    @property
    def X_n(self):
        """Spectral coefficients for X."""
        return self._X_n

    @X_n.setter
    def X_n(self, new):
        if len(new) == self.X_basis.num_modes:
            self._X_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"X_n should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.X_basis.num_modes} modes."
            )

    @optimizable_parameter
    @property
    def Y_n(self):
        """Spectral coefficients for Y."""
        return self._Y_n

    @Y_n.setter
    def Y_n(self, new):
        if len(new) == self.Y_basis.num_modes:
            self._Y_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"Y_n should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.Y_basis.num_modes} modes."
            )

    @optimizable_parameter
    @property
    def Z_n(self):
        """Spectral coefficients for Z."""
        return self._Z_n

    @Z_n.setter
    def Z_n(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_n should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.Z_basis.num_modes} modes."
            )

    @classmethod
    def from_values(cls, coords, N=10, s=None, basis="xyz", name=""):
        """Fit coordinates to FourierXYZCurve representation.

        Parameters
        ----------
        coords: ndarray, shape (num_coords,3)
            Coordinates to fit a FourierXYZCurve object, with each column
            corresponding to xyz or rpz depending on the basis argument.
        N : int
            Fourier resolution of the new X,Y,Z representation.
        s : ndarray or "arclength"
            Arbitrary curve parameter to use for the fitting.
            Should be monotonic, 1D array of same length as
            coords. if None, defaults linearly spaced in [0,2pi)
            Alternative, can pass "arclength" to use normalized distance between points.
        basis : {"rpz", "xyz"}
            Basis for input coordinates. Defaults to "xyz".

        Returns
        -------
        curve : FourierXYZCurve
            New representation of the curve parameterized by Fourier series for X,Y,Z.

        """
        if basis == "xyz":
            coords_xyz = coords
        else:
            coords_xyz = rpz2xyz(coords)
        X = coords_xyz[:, 0]
        Y = coords_xyz[:, 1]
        Z = coords_xyz[:, 2]

        X, Y, Z, closedX, closedY, closedZ, input_curve_was_closed = _unclose_curve(
            X, Y, Z
        )

        if isinstance(s, str):
            assert s == "arclength", f"got unknown specification for s {s}"
            # find equal arclength angle-like variable, and use that as theta
            # L_along_curve / L = theta / 2pi
            lengths = np.sqrt(
                np.diff(closedX) ** 2 + np.diff(closedY) ** 2 + np.diff(closedZ) ** 2
            )
            thetas = 2 * np.pi * np.cumsum(lengths) / np.sum(lengths)
            thetas = np.insert(thetas, 0, 0)
            s = thetas[:-1]
        elif s is None:
            s = np.linspace(0, 2 * np.pi, X.size, endpoint=False)
        else:
            s = np.atleast_1d(s)
            s = s[:-1] if input_curve_was_closed else s
            errorif(
                not np.all(np.diff(s) > 0),
                ValueError,
                "supplied s must be monotonically increasing",
            )
            errorif(s[0] < 0, ValueError, "s must lie in [0, 2pi]")
            errorif(s[-1] > 2 * np.pi, ValueError, "s must lie in [0, 2pi]")

        grid = LinearGrid(zeta=s, NFP=1, sym=False)
        basis = FourierSeries(N=N, NFP=1, sym=False)
        transform = Transform(grid, basis, build_pinv=True)
        X_n = transform.fit(X)
        Y_n = transform.fit(Y)
        Z_n = transform.fit(Z)
        return FourierXYZCurve(
            X_n=X_n, Y_n=Y_n, Z_n=Z_n, modes=basis.modes[:, 2], name=name
        )


class FourierPlanarCurve(Curve):
    """Curve that lies in a plane.

    Parameterized by a point (the center of the curve), a vector (normal to the plane),
    and a Fourier series defining the radius from the center as a function of
    a polar angle theta.

    Parameters
    ----------
    center : array-like, shape(3,)
        Coordinates of center of curve, in system determined by basis.
    normal : array-like, shape(3,)
        Components of normal vector to planar surface, in system determined by basis.
    r_n : array-like
        Fourier coefficients for radius from center as function of polar angle
    modes : array-like
        mode numbers associated with r_n
    basis : {'xyz', 'rpz'}
        Coordinate system for center and normal vectors. Default = 'xyz'.
    name : str
        Name for this curve.

    """

    _io_attrs_ = Curve._io_attrs_ + ["_r_n", "_center", "_normal", "_r_basis", "_basis"]

    _static_attrs = Curve._static_attrs + ["_basis", "_r_basis"]

    # Reference frame is centered at the origin with normal in the +Z direction.
    # Curve is computed in reference frame, then displaced/rotated to the desired frame.
    def __init__(
        self,
        center=[10, 0, 0],
        normal=[0, 1, 0],
        r_n=2,
        modes=None,
        basis="xyz",
        name="",
    ):
        super().__init__(name)
        r_n = np.atleast_1d(r_n)
        if modes is None:
            modes = np.arange(-(r_n.size // 2), r_n.size // 2 + 1)
        else:
            modes = np.asarray(modes)
        assert issubclass(modes.dtype.type, np.integer)
        assert r_n.size == modes.size, "r_n size and modes must be the same size"
        assert basis.lower() in ["xyz", "rpz"]

        N = np.max(abs(modes))
        self._r_basis = FourierSeries(N, NFP=1, sym=False)
        self._r_n = copy_coeffs(r_n, modes, self.r_basis.modes[:, 2])

        self._basis = basis
        self.normal = normal
        self.center = center

    @property
    def r_basis(self):
        """Spectral basis for Fourier series."""
        return self._r_basis

    @property
    def N(self):
        """Maximum mode number."""
        return self.r_basis.N

    def change_resolution(self, N=None):
        """Change the maximum angular resolution."""
        N = check_nonnegint(N, "N")
        if (N is not None) and (N != self.N):
            N = int(N)
            modes_old = self.r_basis.modes
            self.r_basis.change_resolution(N=N)
            self.r_n = copy_coeffs(self.r_n, modes_old, self.r_basis.modes)

    @optimizable_parameter
    @property
    def center(self):
        """Center of planar curve polar coordinates."""
        return self._center

    @center.setter
    def center(self, new):
        if len(new) == 3:
            self._center = np.asarray(new)
        else:
            raise ValueError(
                "center should be a 3 element vector in "
                + self.basis
                + " coordinates, got {}".format(new)
            )

    @optimizable_parameter
    @property
    def normal(self):
        """Normal vector to plane."""
        return self._normal

    @normal.setter
    def normal(self, new):
        if len(np.asarray(new)) == 3:
            self._normal = np.asarray(new) / np.linalg.norm(new)
        else:
            raise ValueError(
                "normal should be a 3 element vector in "
                + self.basis
                + " coordinates, got {}".format(new)
            )

    @optimizable_parameter
    @property
    def r_n(self):
        """Spectral coefficients for r."""
        return self._r_n

    @r_n.setter
    def r_n(self, new):
        if len(np.asarray(new)) == self.r_basis.num_modes:
            self._r_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"r_n should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.r_basis.num_modes} modes."
            )

    @property
    def basis(self):
        """Coordinate system for center and normal vectors."""
        return self._basis

    @basis.setter
    def basis(self, new):
        assert new.lower() in ["xyz", "rpz"]
        if new != self.basis:
            if new == "xyz":
                self.normal = rpz2xyz_vec(self.normal, phi=self.center[1])
                self.center = rpz2xyz(self.center)
            else:
                self.center = xyz2rpz(self.center)
                self.normal = xyz2rpz_vec(self.normal, phi=self.center[1])
            self._basis = new

    def get_coeffs(self, n):
        """Get Fourier coefficients for given mode number(s)."""
        n = np.atleast_1d(n).astype(int)
        r = np.zeros_like(n).astype(float)

        idx = np.where(n[:, np.newaxis] == self.r_basis.modes[:, 2])

        r[idx[0]] = self.r_n[idx[1]]
        return r

    def set_coeffs(self, n, r=None):
        """Set specific Fourier coefficients."""
        n, r = np.atleast_1d(n), np.atleast_1d(r)
        r = np.broadcast_to(r, n.shape)
        for nn, rr in zip(n, r):
            idx = self.r_basis.get_idx(0, 0, nn)
            if rr is not None:
                self.r_n = put(self.r_n, idx, rr)

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
        return super().compute(
            names=names,
            grid=grid,
            params=params,
            transforms=transforms,
            data=data,
            override_grid=override_grid,
            basis_in=self.basis,
            **kwargs,
        )

    @classmethod
    def from_values(cls, coords, N=10, basis="xyz", name=""):
        """Fit coordinates to FourierPlanarCurve representation.

        Parameters
        ----------
        coords: ndarray, shape (num_coords,3)
            Coordinates to fit a FourierPlanarCurve object with each column
            corresponding to xyz or rpz depending on the basis argument.
        N : int
            Fourier resolution of the new r representation.
        basis : {"rpz", "xyz"}
            Basis for input coordinates. Defaults to "xyz".
        name : str
            Name for this curve.

        Returns
        -------
        curve : FourierPlanarCurve
            New representation of the curve parameterized by a Fourier series for r.

        """
        # convert to xyz basis
        if basis == "rpz":
            coords = rpz2xyz(coords)
        coords = np.atleast_2d(coords)

        # center
        center = np.mean(coords, axis=0)
        coords_centered = coords - center  # shift to origin

        # normal
        U, _, _ = np.linalg.svd(coords_centered.T)
        normal = U[:, -1].T  # left singular vector of the least singular value

        # axis and angle of rotation
        Z_axis = np.array([0, 0, 1])
        axis = np.cross(Z_axis, normal)
        dotprod = np.dot(Z_axis, normal)
        angle = np.arccos(dotprod)
        rotmat = np.where(  # handle the case where normal is aligned with the -Z axis
            np.allclose(dotprod, -1.0),
            np.diag(np.array([1.0, -1.0, -1.0])),
            rotation_matrix(axis, angle),
        )
        coords_rotated = coords_centered @ rotmat  # rotate to X-Y plane

        warnif(
            np.max(np.abs(coords_rotated[:, 2])) > 1e-14,  # check Z=0 for all points
            UserWarning,
            "Curve values are not planar! Using the projection onto a plane.",
        )

        # polar angle
        s = np.arctan2(coords_rotated[:, 1], coords_rotated[:, 0])
        unwrapped_s = np.unwrap(s)
        # Determine if the sequence is monotonically increasing or decreasing
        if np.all(np.diff(unwrapped_s) > 0):
            curve_sign = 1
        elif np.all(np.diff(unwrapped_s) < 0):
            curve_sign = -1
        else:
            warnings.warn(
                "The curve parameter s is not strictly increasing or decreasing. "
                "Assuming default direction.",
                UserWarning,
            )
            curve_sign = 1

        if curve_sign == -1:
            # original curve was parameterized "backwards" (clockwise),
            # compared to FourierPlanarCurve assumption
            normal = -normal  # flip normal vector direction
            rotmat = rotation_matrix(axis, np.pi)
            coords_rotated = (
                coords_rotated @ rotmat
            )  # flip on X-Y plane to match normal vector

        # polar radius and angle
        r = np.sqrt(coords_rotated[:, 0] ** 2 + coords_rotated[:, 1] ** 2)
        s = np.arctan2(coords_rotated[:, 1], coords_rotated[:, 0])
        s = np.mod(s + 2 * np.pi, 2 * np.pi)  # mod angle to range [0, 2*pi)
        idx = np.argsort(s)  # sort angle to be monotonically increasing
        r = r[idx]
        s = s[idx]

        # Fourier transform
        basis = FourierSeries(N, NFP=1, sym=False)
        grid_fit = LinearGrid(zeta=s, NFP=1)
        transform_fit = Transform(grid_fit, basis, build_pinv=True)
        r_n = transform_fit.fit(r)

        return FourierPlanarCurve(
            center=center,
            normal=normal,
            r_n=r_n,
            modes=basis.modes[:, 2],
            basis="xyz",
            name=name,
        )


class FourierXYCurve(Curve):
    """Curve that lies in a plane.

    Parameterized by a point (the center of the curve), a vector (normal to the plane),
    and Fourier series defining the X and Y coordinates in the plane as a function of
    an arbitrary angle s.

    Parameters
    ----------
    center : array-like, shape(3,)
        Coordinates of center of curve, in system determined by basis.
    normal : array-like, shape(3,)
        Components of normal vector to planar surface, in system determined by basis.
    X_n : array-like
        Fourier coefficients of the X coordinate in the plane.
    Y_n : array-like
        Fourier coefficients of the Y coordinate in the plane.
    modes : array-like
        Mode numbers associated with X_n and Y_n. The n=0 mode will be ignored.
    basis : {'xyz', 'rpz'}
        Coordinate system for center and normal vectors. Default = 'xyz'.
    name : str
        Name for this curve.

    """

    _io_attrs_ = Curve._io_attrs_ + [
        "_center",
        "_normal",
        "_X_n",
        "_Y_n",
        "_X_basis",
        "_Y_basis",
        "_basis",
    ]
    _static_attrs = Curve._static_attrs + ["_basis", "_X_basis", "_Y_basis"]

    # Reference frame is centered at the origin with normal in the +Z direction.
    # Curve is computed in reference frame, then displaced/rotated to the desired frame.
    def __init__(
        self,
        center=[10, 0, 0],
        normal=[0, 1, 0],
        X_n=[0, 2],
        Y_n=[2, 0],
        modes=None,
        basis="xyz",
        name="",
    ):
        super().__init__(name)
        X_n, Y_n = np.atleast_1d(X_n), np.atleast_1d(Y_n)
        if modes is None:
            modes = np.arange(-(X_n.size // 2), X_n.size // 2 + 1)
            if len(X_n) % 2 == 0:
                modes = modes[modes != 0]
        else:
            modes = np.asarray(modes)

        assert issubclass(modes.dtype.type, np.integer)
        assert X_n.size == modes.size, "X_n and modes must be the same size"
        assert Y_n.size == modes.size, "Y_n and modes must be the same size"
        assert basis.lower() in ["xyz", "rpz"]

        idx0 = np.where(modes == 0)[0]
        if len(idx0):
            warnif(
                True,
                UserWarning,
                "Ignoring n=0 mode because it is redundant with center.",
            )
            modes = np.delete(modes, idx0)
            X_n = np.delete(X_n, idx0)
            Y_n = np.delete(Y_n, idx0)

        N = np.max(abs(modes))
        self._X_basis = FourierSeries(N, NFP=1, sym="no n=0")
        self._Y_basis = FourierSeries(N, NFP=1, sym="no n=0")
        self._X_n = copy_coeffs(X_n, modes, self.X_basis.modes[:, 2])
        self._Y_n = copy_coeffs(Y_n, modes, self.Y_basis.modes[:, 2])

        self._basis = basis
        self.normal = normal
        self.center = center

    @optimizable_parameter
    @property
    def center(self):
        """Center of planar curve."""
        return self._center

    @center.setter
    def center(self, new):
        if len(new) == 3:
            self._center = np.asarray(new)
        else:
            raise ValueError(
                "center should be a 3 element vector in "
                + self.basis
                + " coordinates, got {}".format(new)
            )

    @optimizable_parameter
    @property
    def normal(self):
        """Normal vector to plane."""
        return self._normal

    @normal.setter
    def normal(self, new):
        if len(np.asarray(new)) == 3:
            self._normal = np.asarray(new) / np.linalg.norm(new)
        else:
            raise ValueError(
                "normal should be a 3 element vector in "
                + self.basis
                + " coordinates, got {}".format(new)
            )

    @optimizable_parameter
    @property
    def X_n(self):
        """Spectral coefficients for X."""
        return self._X_n

    @X_n.setter
    def X_n(self, new):
        if len(new) == self.X_basis.num_modes:
            self._X_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"X_n should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.X_basis.num_modes} modes."
            )

    @optimizable_parameter
    @property
    def Y_n(self):
        """Spectral coefficients for Y."""
        return self._Y_n

    @Y_n.setter
    def Y_n(self, new):
        if len(new) == self.Y_basis.num_modes:
            self._Y_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"Y_n should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.Y_basis.num_modes} modes."
            )

    @property
    def X_basis(self):
        """Spectral basis for X Fourier series."""
        return self._X_basis

    @property
    def Y_basis(self):
        """Spectral basis for Y Fourier series."""
        return self._Y_basis

    @property
    def basis(self):
        """Coordinate system for center and normal vectors."""
        return self._basis

    @basis.setter
    def basis(self, new):
        assert new.lower() in ["xyz", "rpz"]
        if new != self.basis:
            if new == "xyz":
                self.normal = rpz2xyz_vec(self.normal, phi=self.center[1])
                self.center = rpz2xyz(self.center)
            else:
                self.center = xyz2rpz(self.center)
                self.normal = xyz2rpz_vec(self.normal, phi=self.center[1])
            self._basis = new

    @property
    def N(self):
        """Maximum mode number."""
        return max(self.X_basis.N, self.Y_basis.N)

    def change_resolution(self, N=None):
        """Change the maximum angular resolution."""
        N = check_posint(N, "N")
        if (N is not None) and (N != self.N):
            N = int(N)
            Xmodes_old = self.X_basis.modes
            Ymodes_old = self.Y_basis.modes
            self.X_basis.change_resolution(N=N)
            self.Y_basis.change_resolution(N=N)
            self.X_n = copy_coeffs(self.X_n, Xmodes_old, self.X_basis.modes)
            self.Y_n = copy_coeffs(self.Y_n, Ymodes_old, self.Y_basis.modes)

    def get_coeffs(self, n):
        """Get Fourier coefficients for given mode number(s)."""
        n = np.atleast_1d(n).astype(int)
        X = np.zeros_like(n).astype(float)
        Y = np.zeros_like(n).astype(float)

        Xidx = np.where(n[:, np.newaxis] == self.X_basis.modes[:, 2])
        Yidx = np.where(n[:, np.newaxis] == self.Y_basis.modes[:, 2])

        X[Xidx[0]] = self.X_n[Xidx[1]]
        Y[Yidx[0]] = self.Y_n[Yidx[1]]
        return X, Y

    def set_coeffs(self, n, X=None, Y=None):
        """Set specific Fourier coefficients."""
        n, X, Y = np.atleast_1d(n), np.atleast_1d(X), np.atleast_1d(Y)
        X = np.broadcast_to(X, n.shape)
        Y = np.broadcast_to(Y, n.shape)
        for nn, XX in zip(n, X):
            idx = self.X_basis.get_idx(0, 0, nn)
            if XX is not None:
                self.X_n = put(self.X_n, idx, XX)

        for nn, YY in zip(n, Y):
            idx = self.Y_basis.get_idx(0, 0, nn)
            if YY is not None:
                self.Y_n = put(self.Y_n, idx, YY)

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
        return super().compute(
            names=names,
            grid=grid,
            params=params,
            transforms=transforms,
            data=data,
            override_grid=override_grid,
            basis_in=self.basis,
            **kwargs,
        )

    @classmethod
    def from_values(cls, coords, N=10, s=None, basis="xyz", name=""):
        """Fit coordinates to FourierXYCurve representation.

        Parameters
        ----------
        coords: ndarray, shape (num_coords,3)
            Coordinates to fit a FourierXYCurve object with each column
            corresponding to xyz or rpz depending on the basis argument.
        N : int
            Fourier resolution of the new X & Y representation.
        s : ndarray or "arclength"
            Arbitrary curve parameter to use for the fitting.
            Should be monotonic, 1D array of same length as
            coords. if None, defaults linearly spaced in [0,2pi)
            Alternative, can pass "arclength" to use normalized distance between points.
        basis : {"rpz", "xyz"}
            Basis for input coordinates. Defaults to "xyz".
        name : str
            Name for this curve.

        Returns
        -------
        curve : FourierXYCurve
            New representation of the curve parameterized by Fourier series for X and Y.

        """
        # convert to xyz basis
        if basis == "rpz":
            coords = rpz2xyz(coords)
        coords = np.atleast_2d(coords)

        # center
        center = np.mean(coords, axis=0)
        coords = coords - center  # shift to origin

        # normal
        U, _, _ = np.linalg.svd(coords.T)
        normal = U[:, -1].T  # left singular vector of the least singular value

        # axis and angle of rotation
        Z_axis = np.array([0, 0, 1])
        axis = np.cross(Z_axis, normal)
        dotprod = np.dot(Z_axis, normal)
        angle = np.arccos(dotprod)
        rotmat = np.where(  # handle the case where normal is aligned with the -Z axis
            np.allclose(dotprod, -1.0),
            np.diag(np.array([1.0, -1.0, -1.0])),
            rotation_matrix(axis, angle),
        )
        coords = coords @ rotmat  # rotate to X-Y plane

        X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]
        X, Y, Z, closedX, closedY, closedZ, input_curve_was_closed = _unclose_curve(
            X, Y, Z
        )
        warnif(
            np.max(np.abs(Z)) > 1e-14,  # check that Z=0 for all points
            UserWarning,
            "Curve values are not planar! Using the projection onto a plane.",
        )

        if isinstance(s, str):
            assert s == "arclength", f"got unknown specification for s {s}"
            # find equal arclength angle-like variable, and use that as theta
            # L_along_curve / L = theta / 2pi
            lengths = np.sqrt(np.diff(closedX) ** 2 + np.diff(closedY) ** 2)
            thetas = 2 * np.pi * np.cumsum(lengths) / np.sum(lengths)
            thetas = np.insert(thetas, 0, 0)
            s = thetas[:-1]
        elif s is None:
            s = np.linspace(0, 2 * np.pi, X.size, endpoint=False)
        else:
            s = np.atleast_1d(s)
            s = s[:-1] if input_curve_was_closed else s
            errorif(
                not np.all(np.diff(s) > 0),
                ValueError,
                "supplied s must be monotonically increasing",
            )
            errorif(s[0] < 0, ValueError, "s must lie in [0, 2pi]")
            errorif(s[-1] > 2 * np.pi, ValueError, "s must lie in [0, 2pi]")

        # Fourier transform
        grid = LinearGrid(zeta=s, NFP=1)
        basis = FourierSeries(N, NFP=1, sym=False)
        transform = Transform(grid, basis, build_pinv=True)
        idx = np.where(basis.modes[:, 2] != 0)  # exclude n=0 mode
        X_n = transform.fit(X)
        Y_n = transform.fit(Y)
        center_from_n0 = np.array([X_n[N], Y_n[N], 0]) @ np.linalg.inv(rotmat)

        return FourierXYCurve(
            center=center + center_from_n0,  # add n=0 mode to center
            normal=normal,
            X_n=X_n[idx],
            Y_n=Y_n[idx],
            modes=basis.modes[:, 2][idx],
            basis="xyz",
            name=name,
        )


class SplineXYZCurve(Curve):
    """Curve parameterized by spline knots in X,Y,Z.

    Parameters
    ----------
    X, Y, Z: array-like
        Points for X, Y, Z describing the curve. If the endpoint is included
        (ie, X[0] == X[-1]), then the final point will be dropped.
    knots : ndarray or "arclength"
        arbitrary curve parameter values to use for spline knots,
        should be a monotonic, 1D ndarray of same length as the input X,Y,Z.
        If None, defaults to using an linearly spaced points in [0, 2pi) as the knots.
        If supplied, should lie in [0,2pi].
        Alternatively, the string "arclength" can be supplied to use the normalized
        distance between points.
    method : str
        method of interpolation

        - ``'nearest'``: nearest neighbor interpolation
        - ``'linear'``: linear interpolation
        - ``'cubic'``: C1 cubic splines (aka local splines)
        - ``'cubic2'``: C2 cubic splines (aka natural splines)
        - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
        - ``'cardinal'``: C1 cubic general tension splines. If used, default tension of
          c = 0 will be used
        - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in the
          data, and will not introduce new extrema in the interpolated points
        - ``'monotonic-0'``: same as `'monotonic'` but with 0 first derivatives at both
          endpoints

    name : str
        name for this curve

    """

    _io_attrs_ = Curve._io_attrs_ + ["_X", "_Y", "_Z", "_knots", "_method"]

    _static_attrs = Curve._static_attrs + ["_method"]

    def __init__(
        self,
        X,
        Y,
        Z,
        knots=None,
        method="cubic",
        name="",
    ):
        super().__init__(name)
        X, Y, Z = np.atleast_1d(X), np.atleast_1d(Y), np.atleast_1d(Z)
        X, Y, Z = np.broadcast_arrays(X, Y, Z)

        X, Y, Z, closedX, closedY, closedZ, closed_flag = _unclose_curve(X, Y, Z)

        self._X = X
        self._Y = Y
        self._Z = Z

        if isinstance(knots, str):
            assert knots == "arclength", f"got unknown arclength specification {knots}"
            # find equal arclength angle-like variable, and use that as theta
            # L_along_curve / L = theta / 2pi
            lengths = np.sqrt(
                np.diff(closedX) ** 2 + np.diff(closedY) ** 2 + np.diff(closedZ) ** 2
            )
            thetas = 2 * np.pi * np.cumsum(lengths) / np.sum(lengths)
            thetas = np.insert(thetas, 0, 0)
            knots = thetas[:-1]
        elif knots is None:
            knots = np.linspace(0, 2 * np.pi, len(self._X), endpoint=False)
        else:
            knots = np.atleast_1d(knots)
            errorif(
                not np.all(np.diff(knots) > 0),
                ValueError,
                "supplied knots must be monotonically increasing",
            )
            errorif(knots[0] < 0, ValueError, "knots must lie in [0, 2pi]")
            errorif(knots[-1] > 2 * np.pi, ValueError, "knots must lie in [0, 2pi]")
            knots = knots[:-1] if closed_flag else knots

        self._knots = knots
        self._method = method

    @optimizable_parameter
    @property
    def X(self):
        """Coordinates for X."""
        return self._X

    @X.setter
    def X(self, new):
        if len(new) == len(self.knots):
            self._X = jnp.asarray(new)
        else:
            raise ValueError(
                "X should have the same size as the knots, "
                + f"got {len(new)} X values for {len(self.knots)} knots"
            )

    @optimizable_parameter
    @property
    def Y(self):
        """Coordinates for Y."""
        return self._Y

    @Y.setter
    def Y(self, new):
        if len(new) == len(self.knots):
            self._Y = jnp.asarray(new)
        else:
            raise ValueError(
                "Y should have the same size as the knots, "
                + f"got {len(new)} Y values for {len(self.knots)} knots"
            )

    @optimizable_parameter
    @property
    def Z(self):
        """Coordinates for Z."""
        return self._Z

    @Z.setter
    def Z(self, new):
        if len(new) == len(self.knots):
            self._Z = jnp.asarray(new)
        else:
            raise ValueError(
                "Z should have the same size as the knots, "
                + f"got {len(new)} Z values for {len(self.knots)} knots"
            )

    @property
    def knots(self):
        """Knots for spline."""
        return self._knots

    @knots.setter
    def knots(self, new):
        if len(new) == len(self.knots):
            knots = jnp.atleast_1d(jnp.asarray(new))
            errorif(
                not np.all(np.diff(knots) > 0),
                ValueError,
                "supplied knots must be monotonically increasing",
            )
            errorif(knots[0] < 0, ValueError, "knots must lie in [0, 2pi]")
            errorif(knots[-1] > 2 * np.pi, ValueError, "knots must lie in [0, 2pi]")
            self._knots = jnp.asarray(knots)
        else:
            raise ValueError(
                "new knots should have the same size as the current knots, "
                + f"got {len(new)} new knots, but expected {len(self.knots)} knots"
            )

    @property
    def N(self):
        """Number of knots in the spline."""
        return self.knots.size

    @property
    def method(self):
        """Method of interpolation to use."""
        return self._method

    @method.setter
    def method(self, new):
        possible_methods = [
            "nearest",
            "linear",
            "cubic",
            "cubic2",
            "catmull-rom",
            "monotonic",
            "monotonic-0",
            "cardinal",
        ]
        if new in possible_methods:
            self._method = new
        else:
            raise ValueError(
                "Method must be one of {possible_methods}, "
                + f"instead got unknown method {new} "
            )

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
        grid : Grid or int, optional
            Grid of coordinates to evaluate at. Defaults to a Linear grid.
            If an integer, uses that many equally spaced points.
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
        return super().compute(
            names=names,
            grid=grid,
            params=params,
            transforms=transforms,
            data=data,
            method=self._method,
            **kwargs,
        )

    @classmethod
    def from_values(cls, coords, knots=None, method="cubic", basis="xyz", name=""):
        """Create SplineXYZCurve from coordinate values.

        Parameters
        ----------
        coords: ndarray, shape (num_coords,3)
            Points for X, Y, Z (or R, phi, Z) describing the curve with each column
            corresponding to xyz or rpz depending on the basis argument. If the
            endpoint is included (ie, X[0] == X[-1]), then the final point will be
            dropped.
        knots : ndarray
            Arbitrary curve parameter values to use for spline knots,
            should be an 1D ndarray of same length as the input.
            (input length in this case is determined by grid argument, since
            the input coordinates come from
            Curve.compute("x",grid=grid))
            If None, defaults to using an equal-arclength angle as the knots
            If supplied, will be rescaled to lie in [0,2pi]
        method : str
            method of interpolation

            - `'nearest'`: nearest neighbor interpolation
            - `'linear'`: linear interpolation
            - `'cubic'`: C1 cubic splines (aka local splines)
            - `'cubic2'`: C2 cubic splines (aka natural splines)
            - `'catmull-rom'`: C1 cubic centripetal "tension" splines

        basis : {"rpz", "xyz"}
            Basis for input coordinates. Defaults to "xyz".
        name : str
            Name for this curve.

        Returns
        -------
        curve: SplineXYZCurve
            New representation of the curve parameterized by splines in X,Y,Z.

        """
        if basis == "rpz":
            coords = rpz2xyz(coords)
        return SplineXYZCurve(
            X=coords[:, 0],
            Y=coords[:, 1],
            Z=coords[:, 2],
            knots=knots,
            method=method,
            name=name,
        )
