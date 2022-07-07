import tempfile
import numpy as np
import warnings
import numbers
from desc.backend import jnp, sign, put
from desc.utils import copy_coeffs
from desc.grid import Grid, LinearGrid
from desc.basis import DoubleFourierSeries, ZernikePolynomial
from desc.transform import Transform
from desc.io import InputReader
from .core import Surface
from .utils import xyz2rpz_vec, rpz2xyz_vec, xyz2rpz, rpz2xyz

__all__ = ["FourierRZToroidalSurface", "ZernikeRZToroidalSection"]


class FourierRZToroidalSurface(Surface):
    """Toroidal surface represented by Fourier series in poloidal and toroidal angles.

    Parameters
    ----------
    R_lmn, Z_lmn : array-like, shape(k,)
        Fourier coefficients for R and Z in cylindrical coordinates
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn.
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
    NFP : int
        number of field periods
    sym : bool
        whether to enforce stellarator symmetry. Default is "auto" which enforces if
        modes are symmetric. If True, non-symmetric modes will be truncated.
    rho : float (0,1)
        flux surface label for the toroidal surface
    grid : Grid
        default grid for computation
    name : str
        name for this surface

    """

    _io_attrs_ = Surface._io_attrs_ + [
        "_R_lmn",
        "_Z_lmn",
        "_R_basis",
        "_Z_basis",
        "_R_transform",
        "_Z_transform",
        "rho",
        "_NFP",
    ]

    def __init__(
        self,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        rho=1,
        grid=None,
        name="",
    ):

        if R_lmn is None:
            R_lmn = np.array([10, 1])
            modes_R = np.array([[0, 0], [1, 0]])
        if Z_lmn is None:
            Z_lmn = np.array([0, 1])
            modes_Z = np.array([[0, 0], [-1, 0]])
        if modes_Z is None:
            modes_Z = modes_R
        R_lmn, Z_lmn, modes_R, modes_Z = map(
            np.asarray, (R_lmn, Z_lmn, modes_R, modes_Z)
        )

        assert issubclass(modes_R.dtype.type, np.integer)
        assert issubclass(modes_Z.dtype.type, np.integer)

        MR = np.max(abs(modes_R[:, 0]))
        NR = np.max(abs(modes_R[:, 1]))
        MZ = np.max(abs(modes_Z[:, 0]))
        NZ = np.max(abs(modes_Z[:, 1]))
        self._L = 0
        self._M = max(MR, MZ)
        self._N = max(NR, NZ)
        if sym == "auto":
            if np.all(
                R_lmn[np.where(sign(modes_R[:, 0]) != sign(modes_R[:, 1]))] == 0
            ) and np.all(
                Z_lmn[np.where(sign(modes_Z[:, 0]) == sign(modes_Z[:, 1]))] == 0
            ):
                sym = True
            else:
                sym = False

        self._R_basis = DoubleFourierSeries(
            M=MR, N=NR, NFP=NFP, sym="cos" if sym else False
        )
        self._Z_basis = DoubleFourierSeries(
            M=MZ, N=NZ, NFP=NFP, sym="sin" if sym else False
        )

        self._R_lmn = copy_coeffs(R_lmn, modes_R, self.R_basis.modes[:, 1:])
        self._Z_lmn = copy_coeffs(Z_lmn, modes_Z, self.Z_basis.modes[:, 1:])
        self._NFP = NFP
        self._sym = sym
        self.rho = rho
        if grid is None:
            grid = LinearGrid(
                rho=self.rho,
                M=4 * self.M + 1,
                N=4 * self.N + 1,
                endpoint=True,
            )
        self._grid = grid
        self._R_transform, self._Z_transform = self._get_transforms(grid)
        self.name = name

    @property
    def NFP(self):
        """Number of (toroidal) field periods (int)."""
        return self._NFP

    @NFP.setter
    def NFP(self, new):
        assert (
            isinstance(new, numbers.Real) and int(new) == new and new > 0
        ), f"NFP should be a positive integer, got {type(new)}"
        self.change_resolution(NFP=new)

    @property
    def R_basis(self):
        """Spectral basis for R double Fourier series."""
        return self._R_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z double Fourier series."""
        return self._Z_basis

    @property
    def grid(self):
        """Grid for computation."""
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
        self._R_transform.grid = self.grid
        self._Z_transform.grid = self.grid

    def change_resolution(self, *args, **kwargs):
        """Change the maximum poloidal and toroidal resolution"""
        assert ((len(args) in [2, 3]) and (len(kwargs) == 0 or "NFP" in kwargs)) or (
            len(args) == 0
        ), "change_resolution should be called with 2 or 3 positional arguments or only keyword arguments"
        L = kwargs.get("L", None)
        M = kwargs.get("M", None)
        N = kwargs.get("N", None)
        NFP = kwargs.get("NFP", None)
        self._NFP = NFP if NFP is not None else self.NFP
        if L is not None:
            warnings.warn(
                "FourierRZToroidalSurface does not have a radial resolution, ignoring L"
            )
        if len(args) == 2:
            M, N = args
        elif len(args) == 3:
            L, M, N = args

        if (
            ((N is not None) and (N != self.N))
            or ((M is not None) and (M != self.M))
            or (NFP is not None)
        ):
            M = M if M is not None else self.M
            N = N if N is not None else self.N
            R_modes_old = self.R_basis.modes
            Z_modes_old = self.Z_basis.modes
            self.R_basis.change_resolution(M=M, N=N, NFP=self.NFP)
            self.Z_basis.change_resolution(M=M, N=N, NFP=self.NFP)
            self._R_transform, self._Z_transform = self._get_transforms(self.grid)
            self.R_lmn = copy_coeffs(self.R_lmn, R_modes_old, self.R_basis.modes)
            self.Z_lmn = copy_coeffs(self.Z_lmn, Z_modes_old, self.Z_basis.modes)
            self._M = M
            self._N = N

    @property
    def R_lmn(self):
        """Spectral coefficients for R."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_lmn should have the same size as the basis, got {len(new)} for basis with {self.R_basis.num_modes} modes"
            )

    @property
    def Z_lmn(self):
        """Spectral coefficients for Z."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_lmn should have the same size as the basis, got {len(new)} for basis with {self.Z_basis.num_modes} modes"
            )

    def get_coeffs(self, m, n=0):
        """Get Fourier coefficients for given mode number(s)."""
        n = np.atleast_1d(n).astype(int)
        m = np.atleast_1d(m).astype(int)

        m, n = np.broadcast_arrays(m, n)
        R = np.zeros_like(m).astype(float)
        Z = np.zeros_like(m).astype(float)

        mn = np.array([m, n]).T
        idxR = np.where(
            (mn[:, np.newaxis, :] == self.R_basis.modes[np.newaxis, :, 1:]).all(axis=-1)
        )
        idxZ = np.where(
            (mn[:, np.newaxis, :] == self.Z_basis.modes[np.newaxis, :, 1:]).all(axis=-1)
        )

        R[idxR[0]] = self.R_lmn[idxR[1]]
        Z[idxZ[0]] = self.Z_lmn[idxZ[1]]
        return R, Z

    def set_coeffs(self, m, n=0, R=None, Z=None):
        """Set specific Fourier coefficients."""
        m, n, R, Z = (
            np.atleast_1d(m),
            np.atleast_1d(n),
            np.atleast_1d(R),
            np.atleast_1d(Z),
        )
        m, n, R, Z = np.broadcast_arrays(m, n, R, Z)
        for mm, nn, RR, ZZ in zip(m, n, R, Z):
            if RR is not None:
                idxR = self.R_basis.get_idx(0, mm, nn)
                self.R_lmn = put(self.R_lmn, idxR, RR)
            if ZZ is not None:
                idxZ = self.Z_basis.get_idx(0, mm, nn)
                self.Z_lmn = put(self.Z_lmn, idxZ, ZZ)

    def _get_transforms(self, grid=None):
        if grid is None:
            return self._R_transform, self._Z_transform
        if not isinstance(grid, Grid):
            if np.isscalar(grid):
                grid = LinearGrid(rho=1, M=grid, N=grid, NFP=self.NFP)
            elif len(grid) == 2:
                grid = LinearGrid(rho=1, M=grid[0], N=grid[1], NFP=self.NFP)
            elif grid.shape[1] == 2:
                grid = np.pad(grid, ((0, 0), (1, 0)), constant_values=self.rho)
                grid = Grid(grid, sort=False)
            else:
                grid = Grid(grid, sort=False)
        R_transform = Transform(
            grid,
            self.R_basis,
            derivs=np.array(
                [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]]
            ),
        )
        Z_transform = Transform(
            grid,
            self.Z_basis,
            derivs=np.array(
                [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]]
            ),
        )
        return R_transform, Z_transform

    def compute_curvature(self, params=None, grid=None):
        """Compute gaussian and mean curvature."""
        raise NotImplementedError()

    def compute_coordinates(
        self, R_lmn=None, Z_lmn=None, grid=None, dt=0, dz=0, basis="rpz"
    ):
        """Compute values using specified coefficients.

        Parameters
        ----------
        R_lmn, Z_lmn: array-like
            fourier coefficients for R, Z. Defaults to self.R_lmn, self.Z_lmn
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)
        dt, dz: int
            derivative order to compute in theta, zeta
        basis : {"rpz", "xyz"}
            coordinate system for returned points

        Returns
        -------
        values : ndarray, shape(k,3)
            R, phi, Z or x, y, z coordinates of the surface at points specified in grid
        """
        assert basis.lower() in ["rpz", "xyz"]
        if R_lmn is None:
            R_lmn = self.R_lmn
        if Z_lmn is None:
            Z_lmn = self.Z_lmn
        R_transform, Z_transform = self._get_transforms(grid)

        if dz == 0:
            R = R_transform.transform(R_lmn, dt=dt, dz=0)
            Z = Z_transform.transform(Z_lmn, dt=dt, dz=0)
            phi = R_transform.grid.nodes[:, 2] * (dt == 0)
            coords = jnp.stack([R, phi, Z], axis=1)
        elif dz == 1:
            R0 = R_transform.transform(R_lmn, dt=dt, dz=0)
            dR = R_transform.transform(R_lmn, dt=dt, dz=1)
            dZ = Z_transform.transform(Z_lmn, dt=dt, dz=1)
            dphi = R0 * (dt == 0)
            coords = jnp.stack([dR, dphi, dZ], axis=1)
        elif dz == 2:
            R0 = R_transform.transform(R_lmn, dt=dt, dz=0)
            dR = R_transform.transform(R_lmn, dt=dt, dz=1)
            d2R = R_transform.transform(R_lmn, dt=dt, dz=2)
            d2Z = Z_transform.transform(Z_lmn, dt=dt, dz=2)
            R = d2R - R0
            Z = d2Z
            # 2nd derivative wrt to phi = 0
            phi = 2 * dR * (dt == 0)
            coords = jnp.stack([R, phi, Z], axis=1)
        elif dz == 3:
            R0 = R_transform.transform(R_lmn, dt=dt, dz=0)
            dR = R_transform.transform(R_lmn, dt=dt, dz=1)
            d2R = R_transform.transform(R_lmn, dt=dt, dz=2)
            d3R = R_transform.transform(R_lmn, dt=dt, dz=3)
            d3Z = Z_transform.transform(Z_lmn, dt=dt, dz=3)
            R = d3R - 3 * dR
            Z = d3Z
            phi = (3 * d2R - R0) * (dt == 0)
            coords = jnp.stack([R, phi, Z], axis=1)
        else:
            raise NotImplementedError(
                "Derivatives higher than 3 have not been implemented in cylindrical coordinates"
            )
        if basis.lower() == "xyz":
            if (dt > 0) or (dz > 0):
                coords = rpz2xyz_vec(coords, phi=R_transform.grid.nodes[:, 2])
            else:
                coords = rpz2xyz(coords)
        return coords

    def compute_normal(self, R_lmn=None, Z_lmn=None, grid=None, basis="rpz"):
        """Compute normal vector to surface on default grid.

        Parameters
        ----------
        R_lmn, Z_lmn: array-like
            fourier coefficients for R, Z. Defaults to self.R_lmn, self.Z_lmn
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)
        basis : {"rpz", "xyz"}
            basis vectors to use for normal vector representation

        Returns
        -------
        N : ndarray, shape(k,3)
            normal vector to surface in specified coordinates
        """
        assert basis.lower() in ["rpz", "xyz"]

        if R_lmn is None:
            R_lmn = self.R_lmn
        if Z_lmn is None:
            Z_lmn = self.Z_lmn
        R_transform, Z_transform = self._get_transforms(grid)

        r_t = self.compute_coordinates(R_lmn, Z_lmn, grid, dt=1)
        r_z = self.compute_coordinates(R_lmn, Z_lmn, grid, dz=1)

        N = jnp.cross(r_t, r_z, axis=1)
        N = N / jnp.linalg.norm(N, axis=1)[:, jnp.newaxis]
        if basis.lower() == "xyz":
            phi = R_transform.grid.nodes[:, 2]
            N = rpz2xyz_vec(N, phi=phi)
        return N

    def compute_surface_area(self, R_lmn=None, Z_lmn=None, grid=None):
        """Compute surface area via quadrature.

        Parameters
        ----------
        R_lmn, Z_lmn: array-like
            fourier coefficients for R, Z. Defaults to self.R_lmn, self.Z_lmn
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)

        Returns
        -------
        area : float
            surface area
        """
        if R_lmn is None:
            R_lmn = self.R_lmn
        if Z_lmn is None:
            Z_lmn = self.Z_lmn
        R_transform, Z_transform = self._get_transforms(grid)

        r_t = self.compute_coordinates(R_lmn, Z_lmn, grid, dt=1)
        r_z = self.compute_coordinates(R_lmn, Z_lmn, grid, dz=1)

        N = jnp.cross(r_t, r_z, axis=1)
        return jnp.sum(R_transform.grid.weights * jnp.linalg.norm(N, axis=1))

    @classmethod
    def from_input_file(cls, path):
        """Create a surface objective from Fourier coefficients in a DESC or VMEC input file

        Parameters
        ----------
        path : Path-like or str
            path to DESC or VMEC input file

        Returns
        -------
        surface : FourierRZToroidalSurface
            surface with given Fourier coefficients

        """
        f = open(path, "r")
        if "&INDATA" in f.readlines()[0]:  # vmec input, convert to desc
            inputs = InputReader.parse_vmec_inputs(f)
        else:
            inputs = InputReader().parse_inputs(f)[-1]
        if inputs["bdry_ratio"] != 1:
            warnings.warn(
                "boundary_ratio = {} != 1, surface may not be as expected".format(
                    inputs["bdry_ratio"]
                )
            )
        surf = cls(
            inputs["surface"][:, 3],
            inputs["surface"][:, 4],
            inputs["surface"][:, 1:3].astype(int),
            inputs["surface"][:, 1:3].astype(int),
            inputs["NFP"],
            inputs["sym"],
        )
        return surf


class ZernikeRZToroidalSection(Surface):
    """A toroidal cross section represented by a Zernike polynomial in R,Z.

    Parameters
    ----------
    R_lmn, Z_lmn : array-like, shape(k,)
        zernike coefficients
    modes_R : array-like, shape(k,2)
        radial and poloidal mode numbers [l,m] for R_lmn
    modes_Z : array-like, shape(k,2)
        radial and poloidal mode numbers [l,m] for Z_lmn. If None defaults to modes_R.
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
    zeta : float (0,2pi)
        toroidal angle for the section.
    grid : Grid
        default grid for computation
    name : str
        name for this surface

    """

    _io_attrs_ = Surface._io_attrs_ + [
        "_R_lmn",
        "_Z_lmn",
        "_R_basis",
        "_Z_basis",
        "_R_transform",
        "_Z_transform",
        "zeta",
        "_spectral_indexing",
    ]

    def __init__(
        self,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        spectral_indexing="fringe",
        sym="auto",
        zeta=0,
        grid=None,
        name="",
    ):
        if R_lmn is None:
            R_lmn = np.array([10, 1])
            modes_R = np.array([[0, 0], [1, 1]])
        if Z_lmn is None:
            Z_lmn = np.array([0, 1])
            modes_Z = np.array([[0, 0], [1, -1]])
        if modes_Z is None:
            modes_Z = modes_R
        R_lmn, Z_lmn, modes_R, modes_Z = map(
            np.asarray, (R_lmn, Z_lmn, modes_R, modes_Z)
        )

        assert issubclass(modes_R.dtype.type, np.integer)
        assert issubclass(modes_Z.dtype.type, np.integer)

        LR = np.max(abs(modes_R[:, 0]))
        MR = np.max(abs(modes_R[:, 1]))
        LZ = np.max(abs(modes_Z[:, 0]))
        MZ = np.max(abs(modes_Z[:, 1]))
        self._L = max(LR, LZ)
        self._M = max(MR, MZ)
        self._N = 0

        if sym == "auto":
            if np.all(
                R_lmn[np.where(sign(modes_R[:, 0]) != sign(modes_R[:, 1]))] == 0
            ) and np.all(
                Z_lmn[np.where(sign(modes_Z[:, 0]) == sign(modes_Z[:, 1]))] == 0
            ):
                sym = True
            else:
                sym = False

        self._R_basis = ZernikePolynomial(
            L=max(LR, MR),
            M=max(LR, MR),
            spectral_indexing=spectral_indexing,
            sym="cos" if sym else False,
        )
        self._Z_basis = ZernikePolynomial(
            L=max(LZ, MZ),
            M=max(LZ, MZ),
            spectral_indexing=spectral_indexing,
            sym="sin" if sym else False,
        )

        self._R_lmn = copy_coeffs(R_lmn, modes_R, self.R_basis.modes[:, :2])
        self._Z_lmn = copy_coeffs(Z_lmn, modes_Z, self.Z_basis.modes[:, :2])
        self._sym = sym
        self._spectral_indexing = spectral_indexing

        self.zeta = zeta
        if grid is None:
            grid = LinearGrid(
                L=2 * self.L, M=4 * self.M + 1, zeta=self.zeta, endpoint=True
            )
        self._grid = grid
        self._R_transform, self._Z_transform = self._get_transforms(grid)
        self.name = name

    @property
    def spectral_indexing(self):
        return self._spectral_indexing

    @property
    def R_basis(self):
        """Spectral basis for R Zernike polynomial."""
        return self._R_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z Zernike polynomial."""
        return self._Z_basis

    @property
    def grid(self):
        """Grid for computation."""
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
        self._R_transform.grid = self.grid
        self._Z_transform.grid = self.grid

    def change_resolution(self, *args, **kwargs):
        """Change the maximum radial and poloidal resolution"""
        assert ((len(args) in [2, 3]) and len(kwargs) == 0) or (
            len(args) == 0
        ), "change_resolution should be called with 2 or 3 positional arguments or only keyword arguments"
        L = kwargs.get("L", None)
        M = kwargs.get("M", None)
        N = kwargs.get("N", None)
        if N is not None:
            warnings.warn(
                "ZernikeRZToroidalSection does not have a toroidal resolution, ignoring N"
            )
        if len(args) == 2:
            L, M = args
        elif len(args) == 3:
            L, M, N = args

        if ((L is not None) and (L != self.L)) or ((M is not None) and (M != self.M)):
            R_modes_old = self.R_basis.modes
            Z_modes_old = self.Z_basis.modes
            self.R_basis.change_resolution(L=L, M=M)
            self.Z_basis.change_resolution(L=L, M=M)
            self._R_transform, self._Z_transform = self._get_transforms(self.grid)
            self.R_lmn = copy_coeffs(self.R_lmn, R_modes_old, self.R_basis.modes)
            self.Z_lmn = copy_coeffs(self.Z_lmn, Z_modes_old, self.Z_basis.modes)
            self._L = L
            self._M = M

    @property
    def R_lmn(self):
        """Spectral coefficients for R."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_lmn should have the same size as the basis, got {len(new)} for basis with {self.R_basis.num_modes} modes"
            )

    @property
    def Z_lmn(self):
        """Spectral coefficients for Z."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_lmn should have the same size as the basis, got {len(new)} for basis with {self.Z_basis.num_modes} modes"
            )

    def get_coeffs(self, l, m=0):
        """Get Zernike coefficients for given mode number(s)."""

        l = np.atleast_1d(l).astype(int)
        m = np.atleast_1d(m).astype(int)

        l, m = np.broadcast_arrays(l, m)
        R = np.zeros_like(m).astype(float)
        Z = np.zeros_like(m).astype(float)

        lm = np.array([l, m]).T
        idxR = np.where(
            (lm[:, np.newaxis, :] == self.R_basis.modes[np.newaxis, :, :2]).all(axis=-1)
        )
        idxZ = np.where(
            (lm[:, np.newaxis, :] == self.Z_basis.modes[np.newaxis, :, :2]).all(axis=-1)
        )

        R[idxR[0]] = self.R_lmn[idxR[1]]
        Z[idxZ[0]] = self.Z_lmn[idxZ[1]]
        return R, Z

    def set_coeffs(self, l, m=0, R=None, Z=None):
        """Set specific Zernike coefficients."""

        l, m, R, Z = (
            np.atleast_1d(l),
            np.atleast_1d(m),
            np.atleast_1d(R),
            np.atleast_1d(Z),
        )
        l, m, R, Z = np.broadcast_arrays(l, m, R, Z)
        for ll, mm, RR, ZZ in zip(l, m, R, Z):
            if RR is not None:
                idxR = self.R_basis.get_idx(ll, mm, 0)
                self.R_lmn = put(self.R_lmn, idxR, RR)
            if ZZ is not None:
                idxZ = self.Z_basis.get_idx(ll, mm, 0)
                self.Z_lmn = put(self.Z_lmn, idxZ, ZZ)

    def _get_transforms(self, grid=None):
        if grid is None:
            return self._R_transform, self._Z_transform
        if not isinstance(grid, Grid):
            if np.isscalar(grid):
                grid = LinearGrid(L=grid, M=grid, zeta=0, NFP=1)
            elif len(grid) == 2:
                grid = LinearGrid(L=grid[0], M=grid[1], zeta=0, NFP=1)
            elif grid.shape[1] == 2:
                grid = np.pad(grid, ((0, 0), (0, 1)), constant_values=self.zeta)
                grid = Grid(grid, sort=False)
            else:
                grid = Grid(grid, sort=False)
        R_transform = Transform(
            grid,
            self.R_basis,
            derivs=np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [1, 1, 0]]
            ),
        )
        Z_transform = Transform(
            grid,
            self.Z_basis,
            derivs=np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [1, 1, 0]]
            ),
        )
        return R_transform, Z_transform

    def compute_curvature(self, params=None, grid=None):
        """Compute gaussian and mean curvature."""
        raise NotImplementedError()

    def compute_coordinates(
        self, R_lmn=None, Z_lmn=None, grid=None, dr=0, dt=0, basis="rpz"
    ):
        """Compute values using specified coefficients.

        Parameters
        ----------
        R_lmn, Z_lmn: array-like
            zernike coefficients for R, Z. Defaults to self.R_lmn, self.Z_lmn
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,1)x(0,2pi)
        dr, dt: int
            derivative order to compute in rho, theta
        basis : {"rpz", "xyz"}
            coordinate system for returned points

        Returns
        -------
        values : ndarray, shape(k,3)
            R, phi, Z or x, y, z coordinates of the surface at points specified in grid
        """
        assert basis.lower() in ["rpz", "xyz"]
        if R_lmn is None:
            R_lmn = self.R_lmn
        if Z_lmn is None:
            Z_lmn = self.Z_lmn
        R_transform, Z_transform = self._get_transforms(grid)

        R = R_transform.transform(R_lmn, dr=dr, dt=dt)
        Z = Z_transform.transform(Z_lmn, dr=dr, dt=dt)
        phi = R_transform.grid.nodes[:, 2] * (dr == 0) * (dt == 0)
        coords = jnp.stack([R, phi, Z], axis=1)
        if basis.lower() == "xyz":
            if (dt > 0) or (dr > 0):
                coords = rpz2xyz_vec(coords, phi=R_transform.grid.nodes[:, 2])
            else:
                coords = rpz2xyz(coords)
        return coords

    def compute_normal(self, R_lmn=None, Z_lmn=None, grid=None, basis="rpz"):
        """Compute normal vector to surface on default grid.

        Parameters
        ----------
        R_lmn, Z_lmn: array-like
            zernike coefficients for R, Z. Defaults to self.R_lmn, self.Z_lmn
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,1)x(0,2pi)
        basis : {"rpz", "xyz"}
            basis vectors to use for normal vector representation

        Returns
        -------
        N : ndarray, shape(k,3)
            normal vector to surface in specified coordinates
        """
        assert basis.lower() in ["rpz", "xyz"]

        R_transform, Z_transform = self._get_transforms(grid)

        phi = R_transform.grid.nodes[:, -1]

        # normal vector is a constant 1*phihat
        N = jnp.array([jnp.zeros_like(phi), jnp.ones_like(phi), jnp.zeros_like(phi)]).T

        if basis.lower() == "xyz":
            N = rpz2xyz_vec(N, phi=phi)

        return N

    def compute_surface_area(self, R_lmn=None, Z_lmn=None, grid=None):
        """Compute surface area via quadrature.

        Parameters
        ----------
        R_lmn, Z_lmn: array-like
            zernike coefficients for R, Z. Defaults to self.R_lmn, self.Z_lmn
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,1)x(0,2pi)

        Returns
        -------
        area : float
            surface area
        """
        if R_lmn is None:
            R_lmn = self.R_lmn
        if Z_lmn is None:
            Z_lmn = self.Z_lmn
        R_transform, Z_transform = self._get_transforms(grid)

        r_r = self.compute_coordinates(R_lmn, Z_lmn, grid, dr=1)
        r_t = self.compute_coordinates(R_lmn, Z_lmn, grid, dt=1)

        N = jnp.cross(r_r, r_t, axis=1)
        return jnp.sum(R_transform.grid.weights * jnp.linalg.norm(N, axis=1)) / (
            2 * np.pi
        )
