"""Classes for 2D surfaces embedded in 3D space."""

import numbers
import warnings

import numpy as np

from desc.backend import jnp, put, sign
from desc.basis import DoubleFourierSeries, ZernikePolynomial
from desc.grid import Grid, LinearGrid
from desc.io import InputReader
from desc.transform import Transform
from desc.utils import copy_coeffs

from .core import Surface
from .utils import rpz2xyz, rpz2xyz_vec

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
    check_orientation : bool
        ensure that this surface has a right handed orientation. Do not set to False
        unless you are sure the parameterization you have given is right handed
        (ie, e_theta x e_zeta points outward from the surface).

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
        check_orientation=True,
    ):
        if R_lmn is None:
            R_lmn = np.array([10, 1])
            modes_R = np.array([[0, 0], [1, 0]])
        if Z_lmn is None:
            Z_lmn = np.array([0, -1])
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

        if check_orientation and self._compute_orientation() == -1:
            warnings.warn(
                "Left handed coordinates detected, switching sign of theta."
                + " To avoid this warning in the future, switch the sign of all"
                + " modes with m<0"
            )
            self._flip_orientation()
            assert self._compute_orientation() == 1

        if grid is None:
            grid = LinearGrid(
                M=2 * self.M,
                N=2 * self.N,
                NFP=self.NFP,
                rho=np.asarray(self.rho),
                endpoint=True,
            )
        self._grid = grid
        self._R_transform, self._Z_transform = self._get_transforms(grid)
        self.name = name

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self._NFP

    @NFP.setter
    def NFP(self, new):
        assert (
            isinstance(new, numbers.Real) and int(new) == new and new > 0
        ), f"NFP should be a positive integer, got {type(new)}"
        self.change_resolution(NFP=new)

    @property
    def R_basis(self):
        """DoubleFourierSeries: Spectral basis for R."""
        return self._R_basis

    @property
    def Z_basis(self):
        """DoubleFourierSeries: Spectral basis for Z."""
        return self._Z_basis

    @property
    def grid(self):
        """Grid: Nodes for computation."""
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
        """Change the maximum poloidal and toroidal resolution."""
        assert (
            ((len(args) in [2, 3]) and len(kwargs) == 0)
            or ((len(args) in [2, 3]) and len(kwargs) in [1, 2])
            or (len(args) == 0)
        ), (
            "change_resolution should be called with 2 (M,N) or 3 (L,M,N) "
            + "positional arguments or only keyword arguments."
        )
        L = kwargs.pop("L", None)
        M = kwargs.pop("M", None)
        N = kwargs.pop("N", None)
        NFP = kwargs.pop("NFP", None)
        sym = kwargs.pop("sym", None)
        assert len(kwargs) == 0, "change_resolution got unexpected kwarg: {kwargs}"
        self._NFP = NFP if NFP is not None else self.NFP
        self._sym = sym if sym is not None else self.sym
        if L is not None:
            warnings.warn(
                "FourierRZToroidalSurface does not have radial resolution, ignoring L"
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
            self.R_basis.change_resolution(
                M=M, N=N, NFP=self.NFP, sym="cos" if self.sym else self.sym
            )
            self.Z_basis.change_resolution(
                M=M, N=N, NFP=self.NFP, sym="sin" if self.sym else self.sym
            )
            if hasattr(self.grid, "change_resolution"):
                self.grid.change_resolution(
                    self.grid.L, self.grid.M, self.grid.N, self.NFP
                )
            self._R_transform, self._Z_transform = self._get_transforms(self.grid)
            self.R_lmn = copy_coeffs(self.R_lmn, R_modes_old, self.R_basis.modes)
            self.Z_lmn = copy_coeffs(self.Z_lmn, Z_modes_old, self.Z_basis.modes)
            self._M = M
            self._N = N

    @property
    def R_lmn(self):
        """ndarray: Spectral coefficients for R."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
            )

    @property
    def Z_lmn(self):
        """ndarray: Spectral coefficients for Z."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
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
                grid = LinearGrid(
                    M=grid, N=grid, rho=np.asarray(self.rho), NFP=self.NFP
                )
            elif len(grid) == 2:
                grid = LinearGrid(
                    M=grid[0], N=grid[1], rho=np.asarray(self.rho), NFP=self.NFP
                )
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

    def _compute_first_fundamental_form(self, R_lmn=None, Z_lmn=None, grid=None):
        """Compute coefficients for the first fundamental form."""
        rt = self.compute_coordinates(R_lmn, Z_lmn, grid, dt=1)
        rz = self.compute_coordinates(R_lmn, Z_lmn, grid, dz=1)
        E = jnp.sum(rt * rt, axis=-1)
        F = jnp.sum(rt * rz, axis=-1)
        G = jnp.sum(rz * rz, axis=-1)
        return E, F, G

    def _compute_second_fundamental_form(self, R_lmn=None, Z_lmn=None, grid=None):
        """Compute coefficients for the second fundamental form."""
        rtt = self.compute_coordinates(R_lmn, Z_lmn, grid, dt=2)
        rtz = self.compute_coordinates(R_lmn, Z_lmn, grid, dt=1, dz=1)
        rzz = self.compute_coordinates(R_lmn, Z_lmn, grid, dz=2)
        n = self.compute_normal(R_lmn, Z_lmn, grid)
        L = jnp.sum(rtt * n, axis=-1)
        M = jnp.sum(rtz * n, axis=-1)
        N = jnp.sum(rzz * n, axis=-1)
        return L, M, N

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
            # 2nd derivative wrt phi = 0
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
                "Derivatives higher than 3 have not been implemented in "
                + "cylindrical coordinates."
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
        """Create a surface from Fourier coefficients in a DESC or VMEC input file.

        Parameters
        ----------
        path : Path-like or str
            Path to DESC or VMEC input file.

        Returns
        -------
        surface : FourierRZToroidalSurface
            Surface with given Fourier coefficients.

        """
        f = open(path)
        if "&INDATA" in f.readlines()[0]:  # vmec input, convert to desc
            inputs = InputReader.parse_vmec_inputs(f)[-1]
        else:
            inputs = InputReader().parse_inputs(f)[-1]
        if (inputs["bdry_ratio"] is not None) and (inputs["bdry_ratio"] != 1):
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

    @classmethod
    def from_near_axis(cls, aspect_ratio, elongation, mirror_ratio, axis_Z, NFP=1):
        """Create a surface from a near-axis model for quasi-poloidal/quasi-isodynamic.

        Parameters
        ----------
        aspect_ratio : float
            Aspect ratio of the geometry = major radius / average cross-sectional area.
        elongation : float
            Elongation of the elliptical surface = major axis / minor axis.
        mirror_ratio : float
            Mirror ratio generated by toroidal variation of the cross-sectional area.
            Must be < 2.
        axis_Z : float
            Vertical extent of the magnetic axis Z coordinate.
            Coefficient of sin(2*phi).
        NFP : int
            Number of field periods.

        Returns
        -------
        surface : FourierRZToroidalSurface
            Surface with given geometric properties.

        """
        assert mirror_ratio <= 2
        a = np.sqrt(elongation) / aspect_ratio  # major axis
        b = 1 / (aspect_ratio * np.sqrt(elongation))  # minor axis
        epsilon = (2 - np.sqrt(4 - mirror_ratio**2)) / mirror_ratio

        R_lmn = np.array(
            [
                1,
                (elongation + 1) * b / 2,
                -1 / 5,
                a * epsilon,
                (elongation - 1) * b / 2,
                (elongation - 1) * b / 2,
            ]
        )
        Z_lmn = np.array(
            [
                -(elongation + 1) * b / 2,
                axis_Z,
                -b * epsilon,
                -(elongation - 1) * b / 2,
                (elongation - 1) * b / 2,
            ]
        )
        modes_R = np.array([[0, 0], [1, 0], [0, 2], [1, 1], [1, 2], [-1, -2]])
        modes_Z = np.array([[-1, 0], [0, -2], [-1, 1], [1, -2], [-1, 2]])

        surf = cls(R_lmn=R_lmn, Z_lmn=Z_lmn, modes_R=modes_R, modes_Z=modes_Z, NFP=NFP)
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
    check_orientation : bool
        ensure that this surface has a right handed orientation. Do not set to False
        unless you are sure the parameterization you have given is right handed
        (ie, e_theta x e_zeta points outward from the surface).

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
        zeta=0.0,
        grid=None,
        name="",
        check_orientation=True,
    ):
        if R_lmn is None:
            R_lmn = np.array([10, 1])
            modes_R = np.array([[0, 0], [1, 1]])
        if Z_lmn is None:
            Z_lmn = np.array([0, -1])
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

        if check_orientation and self._compute_orientation() == -1:
            warnings.warn(
                "Left handed coordinates detected, switching sign of theta."
                + " To avoid this warning in the future, switch the sign of all"
                + " modes with m<0"
            )
            self._flip_orientation()
            assert self._compute_orientation() == 1

        if grid is None:
            grid = LinearGrid(
                L=self.L, M=2 * self.M, zeta=np.asarray(self.zeta), endpoint=True
            )
        self._grid = grid
        self._R_transform, self._Z_transform = self._get_transforms(grid)
        self.name = name

    @property
    def spectral_indexing(self):
        """str: Type of spectral indexing for Zernike basis."""
        return self._spectral_indexing

    @property
    def R_basis(self):
        """ZernikePolynomial: Spectral basis for R."""
        return self._R_basis

    @property
    def Z_basis(self):
        """ZernikePolynomial: Spectral basis for Z."""
        return self._Z_basis

    @property
    def grid(self):
        """Grid: Nodes for computation."""
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
        """Change the maximum radial and poloidal resolution."""
        assert (
            ((len(args) in [2, 3]) and len(kwargs) == 0)
            or ((len(args) in [2, 3]) and len(kwargs) in [1, 2])
            or (len(args) == 0)
        ), (
            "change_resolution should be called with 2 (M,N) or 3 (L,M,N) "
            + "positional arguments or only keyword arguments."
        )
        L = kwargs.pop("L", None)
        M = kwargs.pop("M", None)
        N = kwargs.pop("N", None)
        sym = kwargs.pop("sym", None)
        assert len(kwargs) == 0, "change_resolution got unexpected kwarg: {kwargs}"
        self._sym = sym if sym is not None else self.sym
        if N is not None:
            warnings.warn(
                "ZernikeRZToroidalSection does not have toroidal resolution, ignoring N"
            )
        if len(args) == 2:
            L, M = args
        elif len(args) == 3:
            L, M, N = args

        if ((L is not None) and (L != self.L)) or ((M is not None) and (M != self.M)):
            L = L if L is not None else self.L
            M = M if M is not None else self.M
            R_modes_old = self.R_basis.modes
            Z_modes_old = self.Z_basis.modes
            self.R_basis.change_resolution(
                L=L, M=M, sym="cos" if self.sym else self.sym
            )
            self.Z_basis.change_resolution(
                L=L, M=M, sym="sin" if self.sym else self.sym
            )
            if hasattr(self.grid, "change_resolution"):
                self.grid.change_resolution(self.grid.L, self.grid.M, self.grid.N)
            self._R_transform, self._Z_transform = self._get_transforms(self.grid)
            self.R_lmn = copy_coeffs(self.R_lmn, R_modes_old, self.R_basis.modes)
            self.Z_lmn = copy_coeffs(self.Z_lmn, Z_modes_old, self.Z_basis.modes)
            self._L = L
            self._M = M

    @property
    def R_lmn(self):
        """ndarray: Spectral coefficients for R."""
        return self._R_lmn

    @R_lmn.setter
    def R_lmn(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
            )

    @property
    def Z_lmn(self):
        """ndarray: Spectral coefficients for Z."""
        return self._Z_lmn

    @Z_lmn.setter
    def Z_lmn(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_lmn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_lmn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.R_basis.num_modes} modes."
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
                grid = LinearGrid(L=grid, M=grid, zeta=np.asarray(self.zeta))
            elif len(grid) == 2:
                grid = LinearGrid(L=grid[0], M=grid[1], zeta=np.asarray(self.zeta))
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

    def _compute_first_fundamental_form(self, R_lmn=None, Z_lmn=None, grid=None):
        """Compute coefficients for the first fundamental form."""
        rr = self.compute_coordinates(R_lmn, Z_lmn, grid, dr=1)
        rt = self.compute_coordinates(R_lmn, Z_lmn, grid, dt=1)
        E = jnp.sum(rr * rr, axis=-1)
        F = jnp.sum(rr * rt, axis=-1)
        G = jnp.sum(rt * rt, axis=-1)
        return E, F, G

    def _compute_second_fundamental_form(self, R_lmn=None, Z_lmn=None, grid=None):
        """Compute coefficients for the second fundamental form."""
        rrr = self.compute_coordinates(R_lmn, Z_lmn, grid, dr=2)
        rrt = self.compute_coordinates(R_lmn, Z_lmn, grid, dr=1, dt=1)
        rtt = self.compute_coordinates(R_lmn, Z_lmn, grid, dt=2)
        n = self.compute_normal(R_lmn, Z_lmn, grid)
        L = jnp.sum(rrr * n, axis=-1)
        M = jnp.sum(rrt * n, axis=-1)
        N = jnp.sum(rtt * n, axis=-1)
        return L, M, N

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
