import numpy as np

from desc.backend import jnp
from desc.basis import FourierSeries
from .core import Curve, cart2polvec, pol2cartvec
from desc.transform import Transform
from desc.grid import Grid, LinearGrid
from desc.utils import copy_coeffs

__all__ = [
    "FourierRZCurve",
    "FourierXYZCurve",
    "FourierPlanarCurve",
]


class FourierRZCurve(Curve):
    """Curve parameterized by fourier series for R,Z in terms of toroidal angle phi.

    Parameters
    ----------
    R_n, Z_n: array-like
        fourier coefficients for R, Z
    modes_R : array-like
        mode numbers associated with R_n. If not given defaults to [-n:n]
    modes_Z : array-like
        mode numbers associated with Z_n, defaults to modes_R
    NFP : int
        number of field periods
    sym : bool
        whether to enforce stellarator symmetry
    grid : Grid
        default grid or computation
    name : str
        name for this curve
    """

    _io_attrs_ = Curve._io_attrs_ + [
        "_R_n",
        "_Z_n",
        "_R_basis",
        "_Z_basis",
        "_R_transform",
        "_Z_transform",
        "_sym",
    ]

    def __init__(
        self,
        R_n=10,
        Z_n=0,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        grid=None,
        name="",
    ):

        R_n, Z_n = np.atleast_1d(R_n), np.atleast_1d(Z_n)
        if modes_R is None:
            modes_R = np.arange(-(R_n.size // 2), R_n.size // 2 + 1)
        if modes_Z is None:
            modes_Z = modes_R

        modes_R, modes_Z = np.asarray(modes_R), np.asarray(modes_Z)

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
        self._R_basis = FourierSeries(NR, NFP, sym="cos" if sym else False)
        self._Z_basis = FourierSeries(NZ, NFP, sym="sin" if sym else False)

        self._R_n = copy_coeffs(R_n, modes_R, self.R_basis.modes[:, 2])
        self._Z_n = copy_coeffs(Z_n, modes_Z, self.Z_basis.modes[:, 2])

        if grid is None:
            grid = LinearGrid(N=2 * max(NR, NZ) + 1, endpoint=True)
        self._grid = grid
        self._R_transform, self._Z_transform = self._get_transforms(grid)
        self.name = name

    @property
    def sym(self):
        """Whether this curve has stellarator symmetry."""
        return self._sym

    @property
    def R_basis(self):
        """Spectral basis for R_fourier series."""
        return self._R_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z_fourier series."""
        return self._Z_basis

    @property
    def grid(self):
        """Default grid for computation."""
        return self._grid

    def change_resolution(self, N):
        """Change the maximum toroidal resolution."""
        R_modes_old = self.R_basis.modes
        Z_modes_old = self.Z_basis.modes
        self.R_basis.change_resolution(N=N)
        self.Z_basis.change_resolution(N=N)
        self._R_transform, self._Z_transform = self._get_transforms(self.grid)
        self.R_n = copy_coeffs(self.R_n, R_modes_old, self.R_basis.modes)
        self.Z_n = copy_coeffs(self.Z_n, Z_modes_old, self.Z_basis.modes)

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
        self._R_transform.grid = self.grid
        self._Z_transform.grid = self.grid

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
            idxR = self.R_basis.get_idx(0, 0, nn)
            idxZ = self.Z_basis.get_idx(0, 0, nn)
            if RR is not None:
                self.R_n[idxR] = RR
            if ZZ is not None:
                self.Z_n[idxZ] = ZZ

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
                f"R_n should have the same size as the basis, got {len(new)} for basis with {self.R_basis.num_modes} modes"
            )

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
                f"Z_n should have the same size as the basis, got {len(new)} for basis with {self.Z_basis.num_modes} modes"
            )

    def _get_transforms(self, grid=None):
        if grid is None:
            return self._R_transform, self._Z_transform
        if not isinstance(grid, Grid):
            if np.isscalar(grid):
                grid = np.linspace(0, 2 * np.pi, grid)
            grid = np.atleast_1d(grid)
            if grid.ndim == 1:
                grid = np.pad(grid[:, np.newaxis], ((0, 0), (2, 0)))
            grid = Grid(grid, sort=False)
        R_transform = Transform(
            grid,
            self.R_basis,
            derivs=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        )
        Z_transform = Transform(
            grid,
            self.Z_basis,
            derivs=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        )
        return R_transform, Z_transform

    def compute_coordinates(self, R_n=None, Z_n=None, grid=None, dt=0):
        """Compute values using specified coefficients.

        Parameters
        ----------
        R_n, Z_n: array-like
            fourier coefficients for R, Z. Defaults to self.R_n, self.Z_n
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)
        dt: int
            derivative order to compute

        Returns
        -------
        values : ndarray, shape(k,3)
            R, phi, Z coordinates of the curve at specified grid locations in phi
        """
        if R_n is None:
            R_n = self.R_n
        if Z_n is None:
            Z_n = self.Z_n
        R_transform, Z_transform = self._get_transforms(grid)
        if dt == 0:
            R = R_transform.transform(R_n, dz=0)
            Z = Z_transform.transform(Z_n, dz=0)
            phi = R_transform.grid.nodes[:, 2]
            return jnp.stack([R, phi, Z], axis=1)
        if dt == 1:
            R0 = R_transform.transform(R_n, dz=0)
            dR = R_transform.transform(R_n, dz=dt)
            dZ = Z_transform.transform(Z_n, dz=dt)
            dphi = R0
            return jnp.stack([dR, dphi, dZ], axis=1)
        if dt == 2:
            R0 = R_transform.transform(R_n, dz=0)
            dR = R_transform.transform(R_n, dz=1)
            d2R = R_transform.transform(R_n, dz=2)
            d2Z = Z_transform.transform(Z_n, dz=2)
            R = d2R - R0
            Z = d2Z
            # 2nd derivative wrt to phi = 0
            phi = 2 * dR
            return jnp.stack([R, phi, Z], axis=1)
        if dt == 3:
            R0 = R_transform.transform(R_n, dz=0)
            dR = R_transform.transform(R_n, dz=1)
            d2R = R_transform.transform(R_n, dz=2)
            d3R = R_transform.transform(R_n, dz=3)
            d3Z = Z_transform.transform(Z_n, dz=3)
            R = d3R - 3 * dR
            Z = d3Z
            phi = 3 * d2R - R0
            return jnp.stack([R, phi, Z], axis=1)
        raise NotImplementedError(
            "Derivatives higher than 3 have not been implemented in cylindrical coordinates"
        )

    def compute_frenet_frame(self, R_n=None, Z_n=None, grid=None, coords="rpz"):
        """Compute Frenet frame vectors using specified coefficients.

        Parameters
        ----------
        R_n, Z_n: array-like
            fourier coefficients for R, Z. Defaults to self.R_n, self.Z_n
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
        coords : {"rpz", "xyz"}
            basis vectors to use for Frenet vector representation

        Returns
        -------
        T, N, B : ndarrays, shape(k,3)
            tangent, normal, and binormal vectors of the curve at specified grid locations in phi
        """
        assert coords.lower() in ["rpz", "xyz"]
        if R_n is None:
            R_n = self.R_n
        if Z_n is None:
            Z_n = self.Z_n
        R_transform, Z_transform = self._get_transforms(grid)
        T = self.compute_coordinates(R_n, Z_n, grid, dt=1)
        N = self.compute_coordinates(R_n, Z_n, grid, dt=2)

        T = T / jnp.linalg.norm(T, axis=1)[:, jnp.newaxis]
        N = N / jnp.linalg.norm(N, axis=1)[:, jnp.newaxis]
        B = jnp.cross(T, N, axis=1)

        if coords.lower() == "xyz":
            phi = R_transform.grid.nodes[:, 2]
            T = pol2cartvec(T, phi=phi)
            N = pol2cartvec(N, phi=phi)
            B = pol2cartvec(B, phi=phi)

        return T, N, B

    def compute_curvature(self, R_n=None, Z_n=None, grid=None):
        """Compute curvature using specified coefficients.

        Parameters
        ----------
        R_n, Z_n: array-like
            fourier coefficients for R, Z. Defaults to self.R_n, self.Z_n
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)

        Returns
        -------
        kappa : ndarray, shape(k,)
            curvature of the curve at specified grid locations in phi
        """
        dx = self.compute_coordinates(R_n, Z_n, grid, dt=1)
        d2x = self.compute_coordinates(R_n, Z_n, grid, dt=2)
        dxn = jnp.linalg.norm(dx, axis=1)[:, jnp.newaxis]
        kappa = jnp.linalg.norm(jnp.cross(dx, d2x, axis=1) / dxn ** 3, axis=1)
        return kappa

    def compute_torsion(self, R_n=None, Z_n=None, grid=None):
        """Compute torsion using specified coefficients.

        Parameters
        ----------
        R_n, Z_n: array-like
            fourier coefficients for R, Z. Defaults to self.R_n, self.Z_n
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)

        Returns
        -------
        tau : ndarray, shape(k,)
            torsion of the curve at specified grid locations in phi
        """
        dx = self.compute_coordinates(R_n, Z_n, grid, dt=1)
        d2x = self.compute_coordinates(R_n, Z_n, grid, dt=2)
        d3x = self.compute_coordinates(R_n, Z_n, grid, dt=3)
        dxd2x = jnp.cross(dx, d2x, axis=1)
        tau = (
            jnp.sum(dxd2x * d3x, axis=1)
            / jnp.linalg.norm(dxd2x, axis=1)[:, jnp.newaxis] ** 2
        )
        return tau

    def compute_length(self, R_n=None, Z_n=None, grid=None):
        """Compute the length of the curve using specified nodes for quadrature.

        Parameters
        ----------
        R_n, Z_n: array-like
            fourier coefficients for R, Z. If not given, defaults to values given
            by R_n, Z_n attributes
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)

        Returns
        -------
        length : float
            length of the curve approximated by quadrature
        """
        R_transform, Z_transform = self._get_transforms(grid)
        T = self.compute_coordinates(R_n, Z_n, grid, dt=1)
        T = jnp.linalg.norm(T, axis=1)
        phi = R_transform.grid.nodes[:, 2]
        return jnp.trapz(T, phi)


class FourierXYZCurve(Curve):
    """Curve parameterized by fourier series for X,Y,Z in terms of arbitrary angle phi.

    Parameters
    ----------
    X_n, Y_n, Z_n: array-like
        fourier coefficients for X, Y, Z
    modes : array-like
        mode numbers associated with X_n etc.
    grid : Grid
        default grid or computation
    name : str
        name for this curve

    """

    _io_attrs_ = Curve._io_attrs_ + ["_X_n", "_Y_n", "_Z_n", "_basis", "_transform"]

    def __init__(
        self,
        X_n=[0, 10, 2],
        Y_n=[0, 0, 0],
        Z_n=[2, 0, 0],
        modes=None,
        grid=None,
        name="",
    ):

        X_n, Y_n, Z_n = np.atleast_1d(X_n), np.atleast_1d(Y_n), np.atleast_1d(Z_n)
        if modes is None:
            modes = np.arange(-(X_n.size // 2), X_n.size // 2 + 1)
        else:
            modes = np.asarray(modes)

        assert issubclass(modes.dtype.type, np.integer)

        N = np.max(abs(modes))
        self._basis = FourierSeries(N, NFP=1, sym=False)
        self._X_n = copy_coeffs(X_n, modes, self.basis.modes[:, 2])
        self._Y_n = copy_coeffs(Y_n, modes, self.basis.modes[:, 2])
        self._Z_n = copy_coeffs(Z_n, modes, self.basis.modes[:, 2])

        if grid is None:
            grid = LinearGrid(N=2 * N + 1, endpoint=True)
        self._grid = grid
        self._transform = self._get_transforms(grid)
        self.name = name

    @property
    def basis(self):
        """Spectral basis for Fourier series."""
        return self._basis

    @property
    def grid(self):
        """Default grid for computation."""
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

    def change_resolution(self, N):
        """Change the maximum angular resolution."""
        modes_old = self.basis.modes
        self.basis.change_resolution(N=N)
        self._transform = self._get_transforms(self.grid)
        self.X_n = copy_coeffs(self.X_n, modes_old, self.basis.modes)
        self.Y_n = copy_coeffs(self.Y_n, modes_old, self.basis.modes)
        self.Z_n = copy_coeffs(self.Z_n, modes_old, self.basis.modes)

    def get_coeffs(self, n):
        """Get Fourier coefficients for given mode number(s)."""
        n = np.atleast_1d(n).astype(int)
        X = np.zeros_like(n).astype(float)
        Y = np.zeros_like(n).astype(float)
        Z = np.zeros_like(n).astype(float)

        idx = np.where(n[:, np.newaxis] == self.basis.modes[:, 2])

        X[idx[0]] = self.X_n[idx[1]]
        Y[idx[0]] = self.Y_n[idx[1]]
        Z[idx[0]] = self.Z_n[idx[1]]
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
        for nn, XX, YY, ZZ in zip(n, X, Y, Z):
            idx = self.basis.get_idx(0, 0, nn)
            if XX is not None:
                self.X_n[idx] = XX
            if YY is not None:
                self.Y_n[idx] = YY
            if ZZ is not None:
                self.Z_n[idx] = ZZ

    @property
    def X_n(self):
        """Spectral coefficients for X."""
        return self._X_n

    @X_n.setter
    def X_n(self, new):
        if len(new) == self._basis.num_modes:
            self._X_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"X_n should have the same size as the basis, got {len(new)} for basis with {self._basis.num_modes} modes"
            )

    @property
    def Y_n(self):
        """Spectral coefficients for Y."""
        return self._Y_n

    @Y_n.setter
    def Y_n(self, new):
        if len(new) == self._basis.num_modes:
            self._Y_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"Y_n should have the same size as the basis, got {len(new)} for basis with {self._basis.num_modes} modes"
            )

    @property
    def Z_n(self):
        """Spectral coefficients for Z."""
        return self._Z_n

    @Z_n.setter
    def Z_n(self, new):
        if len(new) == self._basis.num_modes:
            self._Z_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_n should have the same size as the basis, got {len(new)} for basis with {self._basis.num_modes} modes"
            )

    def _get_transforms(self, grid=None):
        if grid is None:
            return self._transform
        if not isinstance(grid, Grid):
            if np.isscalar(grid):
                grid = np.linspace(0, 2 * np.pi, grid)
            grid = np.atleast_1d(grid)
            if grid.ndim == 1:
                grid = np.pad(grid[:, np.newaxis], ((0, 0), (2, 0)))
            grid = Grid(grid, sort=False)
        transform = Transform(
            grid,
            self.basis,
            derivs=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        )
        return transform

    def compute_coordinates(self, X_n=None, Y_n=None, Z_n=None, grid=None, dt=0):
        """Compute values using specified coefficients.

        Parameters
        ----------
        X_n, Y_n, Z_n: array-like
            fourier coefficients for X, Y, Z. If not given, defaults to values given
            by X_n, Y_n, Z_n attributes
        grid : Grid or array-like
            dependent coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)
        dt: int
            derivative order to compute

        Returns
        -------
        values : ndarray, shape(k,3)
            X, Y, Z coordinates of the curve at specified grid locations in phi
        """
        if X_n is None:
            X_n = self.X_n
        if Y_n is None:
            Y_n = self.Y_n
        if Z_n is None:
            Z_n = self.Z_n

        transform = self._get_transforms(grid)
        X = transform.transform(X_n, dz=dt)
        Y = transform.transform(Y_n, dz=dt)
        Z = transform.transform(Z_n, dz=dt)

        return jnp.stack([X, Y, Z], axis=1)

    def compute_frenet_frame(
        self, X_n=None, Y_n=None, Z_n=None, grid=None, coords="rpz"
    ):
        """Compute Frenet frame vectors using specified coefficients.

        Parameters
        ----------
        X_n, Y_n, Z_n: array-like
            fourier coefficients for X, Y, Z. If not given, defaults to values given
            by X_n, Y_n, Z_n attributes
        grid : Grid or array-like
            dependent coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)
        coords : {"rpz", "xyz"}
            basis vectors to use for Frenet vector representation

        Returns
        -------
        T, N, B : ndarrays, shape(k,3)
            tangent, normal, and binormal vectors of the curve at specified grid locations
        """
        assert coords.lower() in ["rpz", "xyz"]
        if X_n is None:
            X_n = self.X_n
        if Y_n is None:
            Y_n = self.Y_n
        if Z_n is None:
            Z_n = self.Z_n

        transform = self._get_transforms(grid)
        X = transform.transform(X_n, dz=0)
        Y = transform.transform(Y_n, dz=0)

        dX = transform.transform(X_n, dz=1)
        dY = transform.transform(Y_n, dz=1)
        dZ = transform.transform(Z_n, dz=1)

        d2X = transform.transform(X_n, dz=2)
        d2Y = transform.transform(Y_n, dz=2)
        d2Z = transform.transform(Z_n, dz=2)

        T = jnp.stack([dX, dY, dZ], axis=1)
        N = jnp.stack([d2X, d2Y, d2Z], axis=1)

        T = T / jnp.linalg.norm(T, axis=1)[:, jnp.newaxis]
        N = N / jnp.linalg.norm(N, axis=1)[:, jnp.newaxis]
        B = jnp.cross(T, N, axis=1)

        if coords.lower() == "rpz":
            T = cart2polvec(T, x=X, y=Y)
            N = cart2polvec(N, x=X, y=Y)
            B = cart2polvec(B, x=X, y=Y)

        return T, N, B

    def compute_curvature(self, X_n=None, Y_n=None, Z_n=None, grid=None):
        """Compute curvature using specified coefficients.

        Parameters
        ----------
        X_n, Y_n, Z_n: array-like
            fourier coefficients for X, Y, Z. If not given, defaults to values given
            by X_n, Y_n, Z_n attributes
        grid : Grid or array-like
            dependent coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)

        Returns
        -------
        kappa : ndarray, shape(k,)
            curvature of the curve at specified grid locations in phi
        """
        dx = self.compute_coordinates(X_n, Y_n, Z_n, grid, dt=1)
        d2x = self.compute_coordinates(X_n, Y_n, Z_n, grid, dt=2)
        dxn = jnp.linalg.norm(dx, axis=1)[:, jnp.newaxis]
        kappa = jnp.linalg.norm(jnp.cross(dx, d2x, axis=1) / dxn ** 3, axis=1)
        return kappa

    def compute_torsion(self, X_n=None, Y_n=None, Z_n=None, grid=None):
        """Compute torsion using specified coefficientsnp.empty((0, 3)).

        Parameters
        ----------
        X_n, Y_n, Z_n: array-like
            fourier coefficients for X, Y, Z. If not given, defaults to values given
            by X_n, Y_n, Z_n attributes
        grid : Grid or array-like
            dependent coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)

        Returns
        -------
        tau : ndarray, shape(k,)
            torsion of the curve at specified grid locations in phi
        """
        dx = self.compute_coordinates(X_n, Y_n, Z_n, grid, dt=1)
        d2x = self.compute_coordinates(X_n, Y_n, Z_n, grid, dt=2)
        d3x = self.compute_coordinates(X_n, Y_n, Z_n, grid, dt=3)
        dxd2x = jnp.cross(dx, d2x, axis=1)
        tau = (
            jnp.sum(dxd2x * d3x, axis=1)
            / jnp.linalg.norm(dxd2x, axis=1)[:, jnp.newaxis] ** 2
        )
        return tau

    def compute_length(self, X_n=None, Y_n=None, Z_n=None, grid=None):
        """Compute the length of the curve using specified nodes for quadrature.

        Parameters
        ----------
        X_n, Y_n, Z_n: array-like
            fourier coefficients for X, Y, Z. If not given, defaults to values given
            by X_n, Y_n, Z_n attributes
        grid : Grid or array-like
            dependent coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)

        Returns
        -------
        length : float
            length of the curve approximated by quadrature
        """
        transform = self._get_transforms(grid)
        T = self.compute_coordinates(X_n, Y_n, Z_n, grid, dt=1)
        T = jnp.linalg.norm(T, axis=1)
        theta = transform.grid.nodes[:, 2]
        return jnp.trapz(T, theta)

    # TODO: methods for converting between representations


class FourierPlanarCurve(Curve):
    """Curve that lines in a plane, parameterized by a point (the center of the curve),
    a vector (normal to the plane), and a fourier series defining the radius from the
    center as a function of a polar angle theta.

    Parameters
    ----------
    center : array-like, shape(3,)
        x,y,z coordinates of center of curve
    normal : array-like, shape(3,)
        x,y,z components of normal vector to planar surface
    r_n : array-like
        fourier coefficients for radius from center as function of polar angle
    modes : array-like
        mode numbers associated with r_n
    grid : Grid
        default grid for computation
    name : str
        name for this curve

    """

    _io_attrs_ = Curve._io_attrs_ + [
        "_r_n",
        "_center",
        "_normal",
        "_basis",
        "_transform",
    ]

    # We define a reference frame with center at the origin and normal in the +Z direction.
    # The curve is computed in this frame and then shifted/rotated to the correct frame
    def __init__(
        self,
        center=[10, 0, 0],
        normal=[0, 1, 0],
        r_n=2,
        modes=None,
        grid=None,
        name="",
    ):
        r_n = np.atleast_1d(r_n)
        if modes is None:
            modes = np.arange(-(r_n.size // 2), r_n.size // 2 + 1)
        else:
            modes = np.asarray(modes)
        assert issubclass(modes.dtype.type, np.integer)

        N = np.max(abs(modes))
        self._basis = FourierSeries(N, NFP=1, sym=False)
        self._r_n = copy_coeffs(r_n, modes, self.basis.modes[:, 2])

        self.normal = normal
        self.center = center
        if grid is None:
            grid = LinearGrid(N=2 * N + 1, endpoint=True)
        self._grid = grid
        self._transform = self._get_transforms(grid)
        self.name = name

    @property
    def basis(self):
        """Spectral basis for Fourier series."""
        return self._basis

    @property
    def grid(self):
        """Default grid for computation."""
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

    def change_resolution(self, N):
        """Change the maximum angular resolution."""
        modes_old = self.basis.modes
        self.basis.change_resolution(N=N)
        self._transform = self._get_transforms(self.grid)
        self.r_n = copy_coeffs(self.r_n, modes_old, self.basis.modes)

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
                "center should be a 3 element vector [cx, cy, cz], got {}".format(new)
            )

    @property
    def normal(self):
        """Normal vector to plane."""
        return self._normal

    @normal.setter
    def normal(self, new):
        if len(new) == 3:
            self._normal = np.asarray(new) / np.linalg.norm(new)
        else:
            raise ValueError(
                "normal should be a 3 element vector [nx, ny, nz], got {}".format(new)
            )

    @property
    def r_n(self):
        """Spectral coefficients for r."""
        return self._r_n

    @r_n.setter
    def r_n(self, new):
        if len(new) == self._basis.num_modes:
            self._r_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"r_n should have the same size as the basis, got {len(new)} for basis with {self._basis.num_modes} modes"
            )

    def get_coeffs(self, n):
        """Get Fourier coefficients for given mode number(s)."""
        n = np.atleast_1d(n).astype(int)
        r = np.zeros_like(n).astype(float)

        idx = np.where(n[:, np.newaxis] == self.basis.modes[:, 2])

        r[idx[0]] = self.r_n[idx[1]]
        return r

    def set_coeffs(self, n, r=None):
        """Set specific fourier coefficients."""
        n, r = np.atleast_1d(n), np.atleast_1d(r)
        r = np.broadcast_to(r, n.shape)
        for nn, rr in zip(n, r):
            idx = self.basis.get_idx(0, 0, nn)
            if rr is not None:
                self.r_n[idx] = rr

    def _rotmat(self, normal=None):
        """Rotation matrix to rotate z axis into plane normal."""
        if normal is None:
            normal = self.normal

        nx, ny, nz = normal
        nxny = jnp.sqrt(nx ** 2 + ny ** 2)

        R = jnp.array(
            [
                [ny / nxny, -nx / nxny, 0],
                [nx * nx / nxny, ny * nz / nxny, -nxny],
                [nx, ny, nz],
            ]
        ).T
        return R

    def _get_transforms(self, grid=None):
        if grid is None:
            return self._transform
        if not isinstance(grid, Grid):
            if np.isscalar(grid):
                grid = np.linspace(0, 2 * np.pi, grid)
            grid = np.atleast_1d(grid)
            if grid.ndim == 1:
                grid = np.pad(grid[:, np.newaxis], ((0, 0), (2, 0)))
            grid = Grid(grid, sort=False)
        transform = Transform(
            grid,
            self.basis,
            derivs=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        )
        return transform

    def compute_coordinates(self, center=None, normal=None, r_n=None, grid=None, dt=0):
        """Compute values using specified coefficients.

        Parameters
        ----------
        center : array-like, shape(3,)
            x,y,z coordinates of center of curve. If not given, defaults to self.center
        normal : array-like, shape(3,)
            x,y,z components of normal vector to planar surface. If not given, defaults
            to self.normal
        r_n : array-like
            fourier coefficients for radius from center as function of polar angle.
            If not given defaults to self.r_n
        grid : Grid or array-like
            dependent coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)
        dt: int
            derivative order to compute

        Returns
        -------
        values : ndarray, shape(k,3)
            X, Y, Z coordinates of the curve at specified grid locations in theta
        """
        if center is None:
            center = self.center
        if normal is None:
            normal = self.normal
        if r_n is None:
            r_n = self.r_n
        transform = self._get_transforms(grid)
        if dt == 0:
            r = transform.transform(r_n, dz=dt)
            t = transform.grid.nodes[:, -1]
            X = r * jnp.cos(t)
            Y = r * jnp.sin(t)
            Z = np.zeros_like(r)
            coords = jnp.array([X, Y, Z])
        elif dt == 1:
            t = transform.grid.nodes[:, -1]
            r = transform.transform(r_n, dz=0)
            dr = transform.transform(r_n, dz=1)

            dX = dr * jnp.cos(t) - r * jnp.sin(t)
            dY = dr * jnp.sin(t) + r * jnp.cos(t)
            Z = np.zeros_like(r)
            coords = jnp.array([dX, dY, Z])
        elif dt == 2:
            t = transform.grid.nodes[:, -1]
            r = transform.transform(r_n, dz=0)
            dr = transform.transform(r_n, dz=1)
            d2r = transform.transform(r_n, dz=2)
            d2X = d2r * jnp.cos(t) - 2 * dr * jnp.sin(t) - r * jnp.cos(t)
            d2Y = d2r * jnp.sin(t) + 2 * dr * jnp.cos(t) - r * jnp.sin(t)
            Z = np.zeros_like(r)
            coords = jnp.array([d2X, d2Y, Z])
        elif dt == 3:
            t = transform.grid.nodes[:, -1]
            r = transform.transform(r_n, dz=0)
            dr = transform.transform(r_n, dz=1)
            d2r = transform.transform(r_n, dz=2)
            d3r = transform.transform(r_n, dz=3)
            d3X = (
                d3r * jnp.cos(t)
                - 3 * d2r * jnp.sin(t)
                - 3 * dr * jnp.cos(t)
                + r * jnp.sin(t)
            )
            d3Y = (
                d3r * jnp.sin(t)
                + 3 * d2r * jnp.cos(t)
                - 3 * dr * jnp.sin(t)
                - r * jnp.cos(t)
            )
            coords = jnp.array([d3X, d3Y, Z])
        else:
            raise NotImplementedError(
                "derivatives higher than 3 have not been implemented for planar curves"
            )
        R = self._rotmat(normal)
        coords = jnp.matmul(R, coords) + (center[:, jnp.newaxis] * (dt == 0))

        return coords.T

    def compute_frenet_frame(
        self, center=None, normal=None, r_n=None, grid=None, coords="rpz"
    ):
        """Compute Frenet frame vectors using specified coefficients.

        Parameters
        ----------
        center : array-like, shape(3,)
            x,y,z coordinates of center of curve. If not given, defaults to self.center
        normal : array-like, shape(3,)
            x,y,z components of normal vector to planar surface. If not given, defaults
            to self.normal
        r_n : array-like
            fourier coefficients for radius from center as function of polar angle.
            If not given defaults to self.r_n
        grid : Grid or array-like
            dependent coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)
        coords : {"rpz", "xyz"}
            basis vectors to use for Frenet vector representation

        Returns
        -------
        T, N, B : ndarrays, shape(k,3)
            tangent, normal, and binormal vectors of the curve at specified grid locations in theta
        """
        assert coords.lower() in ["rpz", "xyz"]
        if center is None:
            center = self.center
        if normal is None:
            normal = self.normal
        if r_n is None:
            r_n = self.r_n
        transform = self._get_transforms(grid)

        T = self.compute_coordinates(center, normal, r_n, grid, dt=1)
        N = self.compute_coordinates(center, normal, r_n, grid, dt=2)

        T = T / jnp.linalg.norm(T, axis=1)[:, jnp.newaxis]
        N = N / jnp.linalg.norm(N, axis=1)[:, jnp.newaxis]
        B = jnp.cross(T, N, axis=1)

        if coords.lower() == "rpz":
            r = transform.transform(r_n)
            t = transform.grid.nodes[:, 2]
            R = self._rotmat(normal)
            xyz = jnp.array([r * jnp.cos(t), r * jnp.sin(t), jnp.zeros_like(r)])
            x, y, z = jnp.matmul(R, xyz) + center[:, jnp.newaxis]
            T = cart2polvec(T, x=x, y=y)
            N = cart2polvec(N, x=x, y=y)
            B = cart2polvec(B, x=x, y=y)

        return T, N, B

    def compute_curvature(self, center=None, normal=None, r_n=None, grid=None):
        """Compute curvature using specified coefficients.

        Parameters
        ----------
        center : array-like, shape(3,)
            x,y,z coordinates of center of curve. If not given, defaults to self.center
        normal : array-like, shape(3,)
            x,y,z components of normal vector to planar surface. If not given, defaults
            to self.normal
        r_n : array-like
            fourier coefficients for radius from center as function of polar angle.
            If not given defaults to self.r_n
        grid : Grid or array-like
            dependent coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)

        Returns
        -------
        kappa : ndarray, shape(k,)
            curvature of the curve at specified grid locations in theta
        """
        dx = self.compute_coordinates(center, normal, r_n, grid, dt=1)
        d2x = self.compute_coordinates(center, normal, r_n, grid, dt=2)
        dxn = jnp.linalg.norm(dx, axis=1)[:, jnp.newaxis]
        kappa = jnp.linalg.norm(jnp.cross(dx, d2x, axis=1) / dxn ** 3, axis=1)
        return kappa

    def compute_torsion(self, center=None, normal=None, r_n=None, grid=None):
        """Compute torsion using specified coefficients.

        Parameters
        ----------
        center : array-like, shape(3,)
            x,y,z coordinates of center of curve. If not given, defaults to self.center
        normal : array-like, shape(3,)
            x,y,z components of normal vector to planar surface. If not given, defaults
            to self.normal
        r_n : array-like
            fourier coefficients for radius from center as function of polar angle.
            If not given defaults to self.r_n
        grid : Grid or array-like
            dependent coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)

        Returns
        -------
        tau : ndarray, shape(k,)
            torsion of the curve at specified grid locations in phi
        """
        # torsion is zero for planar curves
        transform = self._get_transforms(grid)
        torsion = jnp.zeros_like(transform.grid.nodes[:, -1])
        return torsion

    def compute_length(self, center=None, normal=None, r_n=None, grid=None):
        """Compute the length of the curve using specified nodes for quadrature.

        Parameters
        ----------
        center : array-like, shape(3,)
            x,y,z coordinates of center of curve. If not given, defaults to self.center
        normal : array-like, shape(3,)
            x,y,z components of normal vector to planar surface. If not given, defaults
            to self.normal
        r_n : array-like
            fourier coefficients for radius from center as function of polar angle.
            If not given defaults to self.r_n
        grid : Grid or array-like
            dependent coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)

        Returns
        -------
        length : float
            length of the curve approximated by quadrature
        """
        transform = self._get_transforms(grid)
        T = self.compute_coordinates(center, normal, r_n, grid, dt=1)
        T = jnp.linalg.norm(T, axis=1)
        theta = transform.grid.nodes[:, 2]
        return jnp.trapz(T, theta)
