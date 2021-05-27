import numpy as np
from abc import ABC, abstractmethod
import copy

from desc.backend import jnp
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.io import IOAble
from desc.basis import FourierSeries


class Curve(ABC, IOAble):
    """Abstract base class for 1D curves in 3D space"""

    _io_attrs_ = ["_name", "_grid"]
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

    @abstractmethod
    def transform(self, params, dt=0):
        """Compute real space coordinates on predefined grid"""

    @abstractmethod
    def compute_frenet_frame(self, params):
        """Compute frenet frame on predefined grid"""

    @abstractmethod
    def compute_curvature(self, params):
        """Compute curvature on predefined grid"""

    @abstractmethod
    def compute_torsion(self, params):
        """Compute torsion on predefined grid"""

    @abstractmethod
    def compute_length(self, nodes=None):
        """Compute the length of the curve using specified nodes for quadrature"""

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this curve."""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        return new


class FourierRZCurve(Curve):
    """Curve parameterized by fourier series for R,Z in terms of
    toroidal angle phi"""

    _io_attrs_ = Curve._io_attrs_ + ["_R_n", "_Z_n", "_basis", "_transform"]
    _object_lib_ = Curve._object_lib_
    _object_lib_.update({"FourierSeries": FourierSeries, "Transform": Transform})

    def __init__(self, R_n, Z_n, modes=None, NFP=1, grid=None, name=None):

        R_n, Z_n = np.atleast_1d(R_n), np.atleast_1d(Z_n)
        if modes is None:
            modes = np.arange(-R_n.size // 2, R_n.size // 2 + 1)
        N = np.max(abs(modes))
        self._basis = FourierSeries(N, NFP, sym=False)
        self._R_n = np.zeros(len(modes))
        self._Z_n = np.zeros(len(modes))
        for n, cR, cZ in zip(modes, R_n, Z_n):
            idx = np.where(self.basis.modes[:, 2] == int(n))[0]
            self._R_n[idx] = cR
            self._Z_n[idx] = cZ

        if grid is None:
            grid = Grid(np.empty((0, 3)))
        self._grid = grid
        self._transform = Transform(
            self.grid,
            self.basis,
            derivs=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        )

    @property
    def basis(self):
        """Spectral basis for fourier series"""
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
    def R_n(self):
        """Spectral coefficients for R"""
        return self._R_n

    # TODO: methods for setting individual components, getting indices of modes etc
    @R_n.setter
    def R_n(self, new):
        if len(new) == self._basis.num_modes:
            self._R_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_n should have the same size as the basis, got {len(new)} for basis with {self._basis.num_modes} modes"
            )

    @property
    def Z_n(self):
        """Spectral coefficients for Z"""
        return self._Z_n

    # TODO: methods for setting individual components, getting indices of modes etc
    @Z_n.setter
    def Z_n(self, new):
        if len(new) == self._basis.num_modes:
            self._Z_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_n should have the same size as the basis, got {len(new)} for basis with {self._basis.num_modes} modes"
            )

    def transform(self, R_n, Z_n, dt=0):
        """Compute values using specified coefficients

        Parameters
        ----------
        R_n, Z_n: array-like
            fourier coefficients for R, Z
        dt: int
            derivative order to compute

        Returns
        -------
        values : ndarray, shape(k,3)
            R, phi, Z coordinates of the curve at specified grid locations in phi
        """
        R = self._transform.transform(R_n, dz=dt)
        Z = self._transform.transform(Z_n, dz=dt)
        phi = self.grid.nodes[:, 2] ** (dt == 1) * (dt > 1)

        return jnp.stack([R, phi, Z], axis=1)

    def compute_coordinates(self, nodes, R_n=None, Z_n=None, dt=0):
        """Compute coordinate values at specified nodes

        Parameters
        ----------
        nodes : array-like, shape(k,)
            toroidal angles to compute coordinates at
        R_n, Z_n: array-like
            fourier coefficients for R, Z. If not given, defaults to values given
            by R_n, Z_n attributes
        dt: int
            derivative order to compute

        Returns
        -------
        values : ndarray, shape(k,3)
            R, phi, Z coordinates of the curve at specified nodes in phi
        """
        if R_n is None:
            R_n = self.R_n
        if Z_n is None:
            Z_n = self.Z_n
        if nodes.ndim == 1:
            nodes = np.pad(nodes[:, np.newaxis], ((0, 0), (2, 0)))
        A = self.basis.evaluate(nodes, derivatives=[0, 0, dt])

        R = jnp.dot(A, R_n)
        Z = jnp.dot(A, Z_n)
        phi = nodes ** (dt == 1) * (dt > 1)
        return jnp.stack([R, phi, Z], axis=1)

    def compute_frenet_frame(self, R_n, Z_n):
        """Compute frenet frame vectors using specified coefficients

        Parameters
        ----------
        R_n, Z_n: array-like
            fourier coefficients for R, Z

        Returns
        -------
        T, N, B : ndarrays, shape(k,3)
            tangent, normal, and binormal vectors of the curve at specified grid locations in phi
        """

        dR = self._transform.transform(R_n, dz=1)
        dZ = self._transform.transform(Z_n, dz=1)
        dphi = jnp.ones_like(self.grid.nodes[:, 2])

        d2R = self._transform.transform(R_n, dz=2)
        d2Z = self._transform.transform(Z_n, dz=2)
        d2phi = jnp.zeros_like(self.grid.nodes[:, 2])

        T = jnp.stack([dR, dphi, dZ], axis=1)
        N = jnp.stack([d2R, d2phi, d2Z], axis=1)

        T = T / jnp.linalg.norm(T, axis=1)
        N = N / jnp.linalg.norm(T, axis=1)
        B = jnp.cross(T, N, axis=1)

        return T, N, B

    def compute_curvature(self, R_n, Z_n):
        """Compute curvature using specified coefficients

        Parameters
        ----------
        R_n, Z_n: array-like
            fourier coefficients for R, Z

        Returns
        -------
        kappa : ndarray, shape(k,)
            curvature of the curve at specified grid locations in phi
        """
        d2R = self._transform.transform(R_n, dz=2)
        d2Z = self._transform.transform(Z_n, dz=2)
        d2phi = jnp.zeros_like(self.grid.nodes[:, 2])

        kappa = jnp.sqrt(d2R ** 2 + d2phi ** 2 + d2Z ** 2)
        return kappa

    def compute_torsion(self, R_n, Z_n):
        """Compute torsion using specified coefficients

        Parameters
        ----------
        R_n, Z_n: array-like
            fourier coefficients for R, Z

        Returns
        -------
        tau : ndarray, shape(k,)
            torsion of the curve at specified grid locations in phi
        """
        # tau = -N * B'
        # B = TxN
        # B' = T'xN + TxN'
        # tau = -N*(T'xN) - N*(TxN')
        #         ^ this is zero
        dR = self._transform.transform(R_n, dz=1)
        dZ = self._transform.transform(Z_n, dz=1)
        dphi = jnp.ones_like(self.grid.nodes[:, 2])

        d2R = self._transform.transform(R_n, dz=2)
        d2Z = self._transform.transform(Z_n, dz=2)
        d2phi = jnp.zeros_like(self.grid.nodes[:, 2])

        d3R = self._transform.transform(R_n, dz=3)
        d3Z = self._transform.transform(Z_n, dz=3)
        d3phi = jnp.zeros_like(self.grid.nodes[:, 2])

        T = jnp.stack([dR, dphi, dZ], axis=1)
        T = T / jnp.linalg.norm(T, axis=1)

        N = jnp.stack([d2R, d2phi, d2Z], axis=1)
        kappa = jnp.sqrt(d2R ** 2 + d2phi ** 2 + d2Z ** 2)
        N = N / kappa
        dN = jnp.stack([d3R, d3phi, d3Z], axis=1) / kappa

        tau = jnp.cross(T, dN, axis=1)
        tau = jnp.sum(-N * tau, axis=1)
        return tau

    def compute_length(self, nodes=None, R_n=None, Z_n=None):
        """Compute the length of the curve using specified nodes for quadrature

        Parameters
        ----------
        nodes : int or array-like
            points to use for quadrature. If not supplied, uses the default grid.
            If an integer, assumes that many linearly spaced points in (0,2pi)
        R_n, Z_n: array-like
            fourier coefficients for R, Z. If not given, defaults to values given
            by R_n, Z_n attributes

        Returns
        -------
        length : float
            length of the curve approximated by quadrature
        """
        if R_n is None:
            R_n = self.R_n
        if Z_n is None:
            Z_n = self.Z_n
        if nodes is not None and nodes.ndim == 1:
            nodes = np.pad(nodes[:, np.newaxis], ((0, 0), (2, 0)))

        if nodes is None:
            R = self._transform.transform(R_n, dz=0)
            Z = self._transform.transform(Z_n, dz=0)
            phi = self.grid.nodes[:, 2]
            coords = jnp.stack([R, phi, Z], axis=1)
        else:
            coords = self.compute_coordinates(nodes, R_n, Z_n)
        dl = jnp.linalg.norm(jnp.diff(coords, axis=0), axis=1)
        return jnp.trapz(dl)


class FourierXYZCurve(Curve):
    """Curve parameterized by fourier series for X,Y,Z in terms of
    arbitrary angle phi"""

    _io_attrs_ = Curve._io_attrs_ + ["_X_n", "_Y_n", "_Z_n", "_basis", "_transform"]
    _object_lib_ = Curve._object_lib_
    _object_lib_.update({"FourierSeries": FourierSeries, "Transform": Transform})

    def __init__(self, X_n, Y_n, Z_n, modes=None, NFP=1, grid=None, name=None):

        X_n, Y_n, Z_n = np.atleast_1d(X_n), np.atleast_1d(Y_n), np.atleast_1d(Z_n)
        if modes is None:
            modes = np.arange(-X_n.size // 2, X_n.size // 2 + 1)
        N = np.max(abs(modes))
        self._basis = FourierSeries(N, NFP, sym=False)
        self._X_n = np.zeros(len(modes))
        self._Y_n = np.zeros(len(modes))
        self._Z_n = np.zeros(len(modes))
        for n, cX, cY, cZ in zip(modes, X_n, Y_n, Z_n):
            idx = np.where(self.basis.modes[:, 2] == int(n))[0]
            self._X_n[idx] = cX
            self._Y_n[idx] = cY
            self._Z_n[idx] = cZ

        if grid is None:
            grid = Grid(np.empty((0, 3)))
        self._grid = grid
        self._transform = Transform(
            self.grid,
            self.basis,
            derivs=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        )

    @property
    def basis(self):
        """Spectral basis for fourier series"""
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
    def X_n(self):
        """Spectral coefficients for X"""
        return self._X_n

    # TODO: methods for setting individual components, getting indices of modes etc
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
        """Spectral coefficients for Y"""
        return self._Y_n

    # TODO: methods for setting individual components, getting indices of modes etc
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
        """Spectral coefficients for Z"""
        return self._Z_n

    # TODO: methods for setting individual components, getting indices of modes etc
    @Z_n.setter
    def Z_n(self, new):
        if len(new) == self._basis.num_modes:
            self._Z_n = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_n should have the same size as the basis, got {len(new)} for basis with {self._basis.num_modes} modes"
            )

    def transform(self, X_n, Y_n, Z_n, dt=0):
        """Compute values using specified coefficients

        Parameters
        ----------
        X_n, Y_n, Z_n: array-like
            fourier coefficients for X, Y, Z
        dt: int
            derivative order to compute

        Returns
        -------
        values : ndarray, shape(k,3)
            X, Y, Z coordinates of the curve at specified grid locations in phi
        """
        X = self._transform.transform(X_n, dz=dt)
        Y = self._transform.transform(Y_n, dz=dt)
        Z = self._transform.transform(Z_n, dz=dt)

        return jnp.stack([X, Y, Z], axis=1)

    def compute_coordinates(self, nodes, X_n=None, Y_n=None, Z_n=None, dt=0):
        """Compute coordinate values at specified nodes

        Parameters
        ----------
        nodes : array-like, shape(k,)
            toroidal angles to compute coordinates at
        X_n, Y_n, Z_n: array-like
            fourier coefficients for X, Y, Z. If not given, defaults to values given
            by X_n, Y_n, Z_n attributes
        dt: int
            derivative order to compute

        Returns
        -------
        values : ndarray, shape(k,3)
            X, Y, Z coordinates of the curve at specified nodes in phi
        """
        if X_n is None:
            X_n = self.X_n
        if Y_n is None:
            Y_n = self.Y_n
        if Z_n is None:
            Z_n = self.Z_n
        if nodes.ndim == 1:
            nodes = np.pad(nodes[:, np.newaxis], ((0, 0), (2, 0)))
        A = self.basis.evaluate(nodes, derivatives=[0, 0, dt])

        X = jnp.dot(A, X_n)
        Y = jnp.dot(A, Y_n)
        Z = jnp.dot(A, Z_n)

        return jnp.stack([X, Y, Z], axis=1)

    def compute_frenet_frame(self, X_n, Y_n, Z_n):
        """Compute frenet frame vectors using specified coefficients

        Parameters
        ----------
        X_n, Y_n, Z_n: array-like
            fourier coefficients for X, Y, Z

        Returns
        -------
        T, N, B : ndarrays, shape(k,3)
            tangent, normal, and binormal vectors of the curve at specified grid locations in phi
        """

        dX = self._transform.transform(X_n, dz=1)
        dY = self._transform.transform(Y_n, dz=1)
        dZ = self._transform.transform(Z_n, dz=1)

        d2X = self._transform.transform(X_n, dz=2)
        d2Y = self._transform.transform(Y_n, dz=2)
        d2Z = self._transform.transform(Z_n, dz=2)

        T = jnp.stack([dX, dY, dZ], axis=1)
        N = jnp.stack([d2X, d2Y, d2Z], axis=1)

        T = T / jnp.linalg.norm(T, axis=1)
        N = N / jnp.linalg.norm(T, axis=1)
        B = jnp.cross(T, N, axis=1)

        return T, N, B

    def compute_curvature(self, X_n, Y_n, Z_n):
        """Compute curvature using specified coefficients

        Parameters
        ----------
        X_n, Y_n, Z_n: array-like
            fourier coefficients for X, Y, Z

        Returns
        -------
        kappa : ndarray, shape(k,)
            curvature of the curve at specified grid locations in phi
        """
        d2X = self._transform.transform(X_n, dz=2)
        d2Y = self._transform.transform(Y_n, dz=2)
        d2Z = self._transform.transform(Z_n, dz=2)

        kappa = jnp.sqrt(d2X ** 2 + d2Y ** 2 + d2Z ** 2)
        return kappa

    def compute_torsion(self, X_n, Y_n, Z_n):
        """Compute torsion using specified coefficients

        Parameters
        ----------
        X_n, Y_n, Z_n: array-like
            fourier coefficients for X, Y, Z

        Returns
        -------
        tau : ndarray, shape(k,)
            torsion of the curve at specified grid locations in phi
        """
        # tau = -N * B'
        # B = TxN
        # B' = T'xN + TxN'
        # tau = -N*(T'xN) - N*(TxN')
        #         ^ this is zero
        dX = self._transform.transform(X_n, dz=1)
        dY = self._transform.transform(Y_n, dz=1)
        dZ = self._transform.transform(Z_n, dz=1)

        d2X = self._transform.transform(X_n, dz=2)
        d2Y = self._transform.transform(Y_n, dz=2)
        d2Z = self._transform.transform(Z_n, dz=2)

        d3X = self._transform.transform(X_n, dz=3)
        d3Y = self._transform.transform(Y_n, dz=3)
        d3Z = self._transform.transform(Z_n, dz=3)

        T = jnp.stack([dX, dY, dZ], axis=1)
        T = T / jnp.linalg.norm(T, axis=1)

        N = jnp.stack([d2X, d2Y, d2Z], axis=1)
        kappa = jnp.sqrt(d2X ** 2 + d2Y ** 2 + d2Z ** 2)
        N = N / kappa
        dN = jnp.stack([d3X, d3Y, d3Z], axis=1) / kappa

        tau = jnp.cross(T, dN, axis=1)
        tau = jnp.sum(-N * tau, axis=1)
        return tau

    def compute_length(self, nodes=None, X_n=None, Y_n=None, Z_n=None):
        """Compute the length of the curve using specified nodes for quadrature

        Parameters
        ----------
        nodes : int or array-like
            points to use for quadrature. If not supplied, uses the default grid.
            If an integer, assumes that many linearly spaced points in (0,2pi)
        X_n, Y_n, Z_n: array-like
            fourier coefficients for X, Y, Z. If not given, defaults to values given
            by X_n, Y_n, Z_n attributes

        Returns
        -------
        length : float
            length of the curve approximated by quadrature
        """
        if X_n is None:
            X_n = self.X_n
        if Y_n is None:
            Y_n = self.Y_n
        if Z_n is None:
            Z_n = self.Z_n
        if nodes is not None and nodes.ndim == 1:
            nodes = np.pad(nodes[:, np.newaxis], ((0, 0), (2, 0)))

        if nodes is None:
            X = self._transform.transform(X_n, dz=0)
            X = self._transform.transform(Y_n, dz=0)
            Z = self._transform.transform(Z_n, dz=0)
            coords = jnp.stack([X, Y, Z], axis=1)
        else:
            coords = self.compute_coordinates(nodes, X_n, Y_n, Z_n)
        dl = jnp.linalg.norm(jnp.diff(coords, axis=0), axis=1)
        return jnp.trapz(dl)

    # TODO: methods for converting between representations
