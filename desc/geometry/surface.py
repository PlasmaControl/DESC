import numpy as np

from desc.backend import jnp
from desc.utils import sign
from desc.grid import Grid, LinearGrid
from desc.basis import DoubleFourierSeries, ZernikePolynomial
from desc.transform import Transform
from .core import Surface


class FourierRZToroidalSurface(Surface):
    """Toroidal surface represented by a double fourier series in poloidal angle
    theta and toroidal angle phi/zeta

    Parameters
    ----------
    R_mn, Z_mn : array-like, shape(k,)
        Fourier coefficients for R and Z in cylindrical coordinates
    modes : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_mn and Z_mn
    NFP : int
        number of field periods
    sym : bool
        whether to enforce stellarator symmetry. Default is "auto" which enforces if
        modes are symmetric. If True, non-symmetric modes will be truncated.
    grid : Grid
        default grid for computation
    name : str
        name for this surface

    """

    _io_attrs_ = Surface._io_attrs_ + [
        "_R_mn",
        "_Z_mn",
        "_R_basis",
        "_Z_basis",
        "_R_transform",
        "_Z_transform",
        "_NFP",
    ]

    def __init__(self, R_mn, Z_mn, modes, NFP=1, sym="auto", grid=None, name=None):

        M = np.max(abs(modes[:, 0]))
        N = np.max(abs(modes[:, 1]))
        if sym == "auto":
            if np.all(
                R_mn[np.where(sign(modes[:, 0]) != sign(modes[:, 1]))] == 0
            ) and np.all(Z_mn[np.where(sign(modes[:, 0]) == sign(modes[:, 1]))] == 0):
                sym = True
            else:
                sym = False

        self._R_basis = DoubleFourierSeries(
            M=M, N=N, NFP=NFP, sym="cos" if sym else False
        )
        self._Z_basis = DoubleFourierSeries(
            M=M, N=N, NFP=NFP, sym="sin" if sym else False
        )
        self._R_mn = np.zeros(len(modes))
        self._Z_mn = np.zeros(len(modes))
        for m, n, cR, cZ in zip(modes, R_mn, Z_mn):
            idxR = np.where(self.R_basis.modes[:, -2:] == [int(m), int(n)])[0]
            idxZ = np.where(self.Z_basis.modes[:, -2:] == [int(m), int(n)])[0]
            self._R_mn[idxR] = cR
            self._Z_mn[idxZ] = cZ
        if grid is None:
            grid = Grid(np.empty((0, 3)))
        self._grid = grid
        self._R_transform = Transform(
            self.grid,
            self.R_basis,
            derivs=np.array(
                [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 0, 2], [0, 1, 1]]
            ),
        )
        self._Z_transform = Transform(
            self.grid,
            self.Z_basis,
            derivs=np.array(
                [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 0, 2], [0, 1, 1]]
            ),
        )
        self.name = name
        self._NFP = NFP
        self._sym = sym

    @property
    def NFP(self):
        """number of toroidal field periods"""
        return self._NFP

    @property
    def R_basis(self):
        """Spectral basis for R double fourier series"""
        return self._R_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z double fourier series"""
        return self._Z_basis

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
    def R_mn(self):
        """Spectral coefficients for R"""
        return self._R_mn

    @R_mn.setter
    def R_mn(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_mn = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_mn should have the same size as the basis, got {len(new)} for basis with {self.R_basis.num_modes} modes"
            )

    @property
    def Z_mn(self):
        """Spectral coefficients for Z"""
        return self._Z_mn

    @Z_mn.setter
    def Z_mn(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_mn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_mn should have the same size as the basis, got {len(new)} for basis with {self.Z_basis.num_modes} modes"
            )

    def transform(self, R_mn, Z_mn, dt=0, dz=0):
        """Compute values using specified coefficients

        Parameters
        ----------
        R_mn, Z_mn: array-like
            fourier coefficients for R, Z
        dt, dz: int
            derivative order to compute in theta, zeta

        Returns
        -------
        values : ndarray, shape(k,3)
            R, phi, Z coordinates of the surface at points specified in grid
        """
        R = self._R_transform.transform(R_mn, dt=dt, dz=dz)
        Z = self._Z_transform.transform(Z_mn, dt=dt, dz=dz)
        phi = self.grid.nodes[:, 2] ** (dz == 1) * (dz > 1)

        return jnp.stack([R, phi, Z], axis=1)

    def compute_coordinates(self, nodes, R_mn=None, Z_mn=None, dt=0, dz=0):
        """Compute coordinate values at specified nodes

        Parameters
        ----------
        nodes : array-like, shape(k,2)
            poloidal and toroidal angles to compute coordinates at
        R_mn, Z_mn: array-like
            fourier coefficients for R, Z. If not given, defaults to values given
            by R_mn, Z_mn attributes
        dt, dz: int
            derivative order to compute in theta, zeta

        Returns
        -------
        values : ndarray, shape(k,3)
            R, phi, Z coordinates of the surface at specified nodes
        """
        if R_mn is None:
            R_mn = self.R_mn
        if Z_mn is None:
            Z_mn = self.Z_mn
        nodes = np.atleast_2d(nodes)
        if nodes.shape[-1] == 2:
            nodes = np.pad(nodes, ((0, 0), (1, 0)))
        AR = self.R_basis.evaluate(nodes, derivatives=[0, dt, dz])
        AZ = self.Z_basis.evaluate(nodes, derivatives=[0, dt, dz])

        R = jnp.dot(AR, R_mn)
        Z = jnp.dot(AZ, Z_mn)
        phi = nodes[:, -1] ** (dz == 1) * (dz > 1)
        return jnp.stack([R, phi, Z], axis=1)

    def compute_normal(self, R_mn, Z_mn):
        """Compute normal vector to surface on default grid

        Parameters
        ----------
        R_mn, Z_mn: array-like
            fourier coefficients for R, Z

        Returns
        -------
        N : ndarray, shape(k,3)
            normal vector to surface in X,Y,Z coordinates
        """
        R = self._R_transform.transform(R_mn)
        R_t = self._R_transform.transform(R_mn, dt=1)
        R_z = self._R_transform.transform(R_mn, dz=1)
        Z = self._Z_transform.transform(Z_mn)
        Z_t = self._Z_transform.transform(Z_mn, dt=1)
        Z_z = self._Z_transform.transform(Z_mn, dz=1)
        phi = self.grid.nodes[:, -1]
        # X = R*cos(phi)
        X_t = R_t * jnp.cos(phi)
        X_z = R_z * jnp.cos(phi) - R * jnp.sin(phi)

        # Y = R*sin(phi)
        Y_t = R_t * jnp.sin(phi)
        Y_z = R_z * jnp.sin(phi) + R * jnp.cos(phi)

        r_t = jnp.array([X_t, Y_t, Z_t]).T
        r_z = jnp.array([X_z, Y_z, Z_z]).T

        N = jnp.cross(r_t, r_z, axis=1)
        N = N / jnp.linalg.norm(N, axis=1)
        return N

    # TODO: compute_normal_XYZ, compute_normal_RpZ

    def compute_surface_area(self, nodes=None, R_mn=None, Z_mn=None):
        """Compute surface area via quadrature

        Parameters
        ----------
        nodes : tuple of 2 int or array-like, shape(k,2)
            quadrature nodes in theta, zeta to use. If integers, assumes that many
            equally spaced points in theta, zeta. If None, uses default grid
        R_mn, Z_mn : array-like
            fourier coefficients for R, Z

        Returns
        -------
        area : float
            surface area
        """
        if nodes is not None and np.isscalar(nodes):
            nodes = (nodes, nodes)
        if nodes is not None:
            if len(nodes) == 2:
                grid = LinearGrid(rho=1, M=nodes[0], N=nodes[1], NFP=self.NFP)
            else:
                if nodes.shape[1] == 2:
                    nodes = np.pad(nodes, ((0, 0), (1, 0)))
                grid = Grid(nodes)
            R_transform = Transform(
                grid,
                self.R_basis,
                derivs=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
            )
            Z_transform = Transform(
                grid,
                self.Z_basis,
                derivs=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
            )
        else:
            R_transform = self._R_transform
            Z_transform = self._Z_transform
            grid = self.grid

        R = R_transform.transform(R_mn)
        R_t = R_transform.transform(R_mn, dt=1)
        R_z = R_transform.transform(R_mn, dz=1)
        Z = Z_transform.transform(Z_mn)
        Z_t = Z_transform.transform(Z_mn, dt=1)
        Z_z = Z_transform.transform(Z_mn, dz=1)
        phi = grid.nodes[:, -1]
        # X = R*cos(phi)
        X_t = R_t * jnp.cos(phi)
        X_z = R_z * jnp.cos(phi) - R * jnp.sin(phi)

        # Y = R*sin(phi)
        Y_t = R_t * jnp.sin(phi)
        Y_z = R_z * jnp.sin(phi) + R * jnp.cos(phi)

        r_t = jnp.array([X_t, Y_t, Z_t]).T
        r_z = jnp.array([X_z, Y_z, Z_z]).T

        N = jnp.cross(r_t, r_z, axis=1)
        return jnp.sum(grid.weights * jnp.linalg.norm(N, axis=1))


class ZernikeToroidalSection(Surface):
    """A toroidal cross section represented by a zernike polynomial in R,Z

    Parameters
    ----------
    R_lm, Z_lm : array-like, shape(k,)
        zernike coefficients
    modes : array-like, shape(k,2)
        radial and poloidal mode numbers [l,m] for R_lm and Z_lm
    sym : bool
        whether to enforce stellarator symmetry. Default is "auto" which enforces if
        modes are symmetric. If True, non-symmetric modes will be truncated.
    spectral_indexing : {``'ansi'``, ``'fringe'``}
        Indexing method, default value = ``'fringe'``

        For L=0, all methods are equivalent and give a "chevron" shaped
        basis (only the outer edge of the zernike pyramid of width M).
        For L>0, the indexing scheme defines order of the basis functions:

        ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
        decreasing size, ending in a triagle shape. For L == M,
        the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
        to the bottom of the pyramid, increasing L while keeping M constant,
        giving a "house" shape

        ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
        decreasing size, ending in a diamond shape for L=2*M where
        the traditional fringe/U of Arizona indexing is recovered.
        For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond
    grid : Grid
        default grid for computation
    name : str
        name for this surface

    """

    _io_attrs_ = Surface._io_attrs_ + [
        "_R_lm",
        "_Z_lm",
        "_R_basis",
        "_Z_basis",
        "_R_transform",
        "_Z_transform",
        "_spectral_indexing",
    ]

    def __init__(
        self,
        R_lm,
        Z_lm,
        modes,
        spectral_indexing="fringe",
        sym="auto",
        grid=None,
        name=None,
    ):

        L = np.max(abs(modes[:, 0]))
        M = np.max(abs(modes[:, 1]))

        if sym == "auto":
            if np.all(
                R_lm[np.where(sign(modes[:, 0]) != sign(modes[:, 1]))] == 0
            ) and np.all(Z_lm[np.where(sign(modes[:, 0]) == sign(modes[:, 1]))] == 0):
                sym = True
            else:
                sym = False

        self._R_basis = ZernikePolynomial(
            L=max(L, M),
            M=max(L, M),
            spectral_indexing=spectral_indexing,
            sym="cos" if sym else False,
        )
        self._Z_basis = ZernikePolynomial(
            L=max(L, M),
            M=max(L, M),
            spectral_indexing=spectral_indexing,
            sym="sin" if sym else False,
        )

        self._R_lm = np.zeros(self.R_basis.num_modes)
        self._Z_lm = np.zeros(self.Z_basis.num_modes)
        for l, m, cR, cZ in zip(modes, R_lm, Z_lm):
            idxR = np.where(self.R_basis.modes[:, :2] == [int(l), int(m)])[0]
            idxZ = np.where(self.Z_basis.modes[:, :2] == [int(l), int(m)])[0]
            self._R_lm[idxR] = cR
            self._Z_lm[idxZ] = cZ
        if grid is None:
            grid = Grid(np.empty((0, 3)))
        self._grid = grid
        self._R_transform = Transform(
            self.grid,
            self.R_basis,
            derivs=np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [1, 1, 0]]
            ),
        )
        self._Z_transform = Transform(
            self.grid,
            self.Z_basis,
            derivs=np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [1, 1, 0]]
            ),
        )
        self._sym = sym
        self._spectral_indexing = spectral_indexing
        self.name = name

    @property
    def spectral_indexing(self):
        return self._spectral_indexing

    @property
    def R_basis(self):
        """Spectral basis for R zernike polynomial"""
        return self._R_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z zernike polynomial"""
        return self._Z_basis

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
    def R_lm(self):
        """Spectral coefficients for R"""
        return self._R_lm

    @R_lm.setter
    def R_lm(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_lm = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_lm should have the same size as the basis, got {len(new)} for basis with {self.R_basis.num_modes} modes"
            )

    @property
    def Z_lm(self):
        """Spectral coefficients for Z"""
        return self._Z_lm

    @Z_lm.setter
    def Z_lm(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_lm = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_lm should have the same size as the basis, got {len(new)} for basis with {self.Z_basis.num_modes} modes"
            )

    def transform(self, R_lm, Z_lm, dr=0, dt=0):
        """Compute values using specified coefficients

        Parameters
        ----------
        R_lm, Z_lm: array-like
            zernike coefficients for R, Z
        dr, dt: int
            derivative order to compute in rho, theta

        Returns
        -------
        values : ndarray, shape(k,3)
            R, phi, Z coordinates of the surface at points specified in grid
        """
        R = self._R_transform.transform(R_lm, dr=dr, dt=dt)
        Z = self._Z_transform.transform(Z_lm, dr=dr, dt=dt)
        phi = self.grid.nodes[:, 2]

        return jnp.stack([R, phi, Z], axis=1)

    def compute_coordinates(self, nodes, R_lm=None, Z_lm=None, dr=0, dt=0):
        """Compute coordinate values at specified nodes

        Parameters
        ----------
        nodes : array-like, shape(k,2)
            radial and poloidal values to compute coordinates at
        R_lm, Z_lm: array-like
            zernike coefficients for R, Z. If not given, defaults to values given
            by R_lm, Z_lm attributes
        dr, dt: int
            derivative order to compute in rho, theta

        Returns
        -------
        values : ndarray, shape(k,3)
            R, phi, Z coordinates of the surface at specified nodes
        """
        if R_lm is None:
            R_lm = self.R_lm
        if Z_lm is None:
            Z_lm = self.Z_lm
        nodes = np.atleast_2d(nodes)
        if nodes.shape[-1] == 2:
            nodes = np.pad(nodes, ((0, 0), (0, 1)))
        AR = self.R_basis.evaluate(nodes, derivatives=[dr, dt, 0])
        AZ = self.Z_basis.evaluate(nodes, derivatives=[dr, dt, 0])

        R = jnp.dot(AR, R_lm)
        Z = jnp.dot(AZ, Z_lm)
        phi = nodes[:, -1]
        return jnp.stack([R, phi, Z], axis=1)

    def compute_normal(self, R_lm, Z_lm):
        """Compute normal vector to surface on default grid

        Parameters
        ----------
        R_lm, Z_lm: array-like
            zernike coefficients for R, Z

        Returns
        -------
        N : ndarray, shape(k,3)
            normal vector to surface in X,Y,Z coordinates
        """
        R = self._R_transform.transform(R_lm)
        R_t = self._R_transform.transform(R_lm, dt=1)
        R_r = self._R_transform.transform(R_lm, dr=1)
        Z = self._Z_transform.transform(Z_lm)
        Z_t = self._Z_transform.transform(Z_lm, dt=1)
        Z_r = self._Z_transform.transform(Z_lm, dr=1)
        phi = self.grid.nodes[:, -1]
        # X = R*cos(phi)
        X_t = R_t * jnp.cos(phi)
        X_r = R_r * jnp.cos(phi)

        # Y = R*sin(phi)
        Y_t = R_t * jnp.sin(phi)
        Y_r = R_r * jnp.sin(phi)

        r_t = jnp.array([X_t, Y_t, Z_t]).T
        r_r = jnp.array([X_r, Y_r, Z_r]).T

        N = jnp.cross(r_r, r_t, axis=1)
        N = N / jnp.linalg.norm(N, axis=1)
        return N

    # TODO: compute_normal_XYZ, compute_normal_RpZ

    def compute_surface_area(self, nodes=None, R_lm=None, Z_lm=None):
        """Compute surface area via quadrature

        Parameters
        ----------
        nodes : tuple of 2 int or array-like, shape(k,2)
            quadrature nodes in rho, theta to use. If integers, assumes that many
            equally spaced points in rho, theta. If None, uses default grid
        R_lm, Z_lm : array-like
            zernike coefficients for R, Z

        Returns
        -------
        area : float
            surface area
        """
        if nodes is not None and np.isscalar(nodes):
            nodes = (nodes, nodes)
        if nodes is not None:
            if len(nodes) == 2:
                grid = LinearGrid(L=nodes[0], M=nodes[1], zeta=0)
            else:
                if nodes.shape[1] == 2:
                    nodes = np.pad(nodes, ((0, 0), (0, 1)))
                grid = Grid(nodes)
            R_transform = Transform(
                nodes,
                self.R_basis,
                derivs=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            )
            Z_transform = Transform(
                nodes,
                self.Z_basis,
                derivs=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            )
        else:
            R_transform = self._R_transform
            Z_transform = self._Z_transform
            grid = self.grid

        R = R_transform.transform(R_lm)
        R_t = R_transform.transform(R_lm, dt=1)
        R_r = R_transform.transform(R_lm, dr=1)
        Z = Z_transform.transform(Z_lm)
        Z_t = Z_transform.transform(Z_lm, dt=1)
        Z_r = Z_transform.transform(Z_lm, dr=1)
        phi = grid.nodes[:, -1]
        # X = R*cos(phi)
        X_t = R_t * jnp.cos(phi)
        X_r = R_r * jnp.cos(phi)

        # Y = R*sin(phi)
        Y_t = R_t * jnp.sin(phi)
        Y_r = R_r * jnp.sin(phi)

        r_t = jnp.array([X_t, Y_t, Z_t]).T
        r_r = jnp.array([X_r, Y_r, Z_r]).T

        N = jnp.cross(r_r, r_t, axis=1)
        return jnp.sum(grid.weights * jnp.linalg.norm(N, axis=1))
