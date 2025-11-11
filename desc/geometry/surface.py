"""Classes for 2D surfaces embedded in 3D space."""

import os
import warnings

import numpy as np

from desc.backend import (
    block_diag,
    execute_on_cpu,
    jit,
    jnp,
    put,
    root_scalar,
    sign,
    vmap,
)
from desc.basis import DoubleFourierSeries, ZernikePolynomial
from desc.grid import Grid, LinearGrid
from desc.io import InputReader
from desc.optimizable import optimizable_parameter
from desc.transform import Transform
from desc.utils import (
    check_nonnegint,
    check_posint,
    copy_coeffs,
    errorif,
    rpz2xyz_vec,
    setdefault,
    xyz2rpz,
    xyz2rpz_vec,
)

from .core import Surface

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
    M, N: int or None
        Maximum poloidal and toroidal mode numbers. Defaults to maximum from modes_R
        and modes_Z.
    rho : float [0,1]
        flux surface label for the toroidal surface
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
        "_NFP",
        "_rho",
    ]
    _static_attrs = Surface._static_attrs + ["_NFP", "_R_basis", "_Z_basis"]

    @execute_on_cpu
    def __init__(
        self,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        M=None,
        N=None,
        rho=1,
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

        assert (
            R_lmn.size == modes_R.shape[0]
        ), "R_lmn size and modes_R.shape[0] must be the same size!"
        assert (
            Z_lmn.size == modes_Z.shape[0]
        ), "Z_lmn size and modes_Z.shape[0] must be the same size!"

        assert issubclass(modes_R.dtype.type, np.integer)
        assert issubclass(modes_Z.dtype.type, np.integer)

        MR = np.max(abs(modes_R[:, 0]))
        NR = np.max(abs(modes_R[:, 1]))
        MZ = np.max(abs(modes_Z[:, 0]))
        NZ = np.max(abs(modes_Z[:, 1]))
        self._L = 0
        M = check_nonnegint(M, "M")
        N = check_nonnegint(N, "N")
        NFP = check_posint(NFP, "NFP", False)
        self._M = setdefault(M, max(MR, MZ))
        self._N = setdefault(N, max(NR, NZ))
        self._NFP = NFP

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
            M=self._M, N=self._N, NFP=NFP, sym="cos" if sym else False
        )
        self._Z_basis = DoubleFourierSeries(
            M=self._M, N=self._N, NFP=NFP, sym="sin" if sym else False
        )

        self._R_lmn = copy_coeffs(R_lmn, modes_R, self.R_basis.modes[:, 1:])
        self._Z_lmn = copy_coeffs(Z_lmn, modes_Z, self.Z_basis.modes[:, 1:])
        self._sym = bool(sym)
        self._rho = rho

        if check_orientation and self._compute_orientation() == -1:
            warnings.warn(
                "Left handed coordinates detected, switching sign of theta."
                + " To avoid this warning in the future, switch the sign of all"
                + " modes with m<0. You may also need to switch the sign of iota or"
                + " current profiles."
            )
            self._flip_orientation()
            assert self._compute_orientation() == 1

        self.name = name

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self._NFP

    @property
    def R_basis(self):
        """DoubleFourierSeries: Spectral basis for R."""
        return self._R_basis

    @property
    def Z_basis(self):
        """DoubleFourierSeries: Spectral basis for Z."""
        return self._Z_basis

    @property
    def rho(self):
        """float: Flux surface label."""
        if not (hasattr(self, "_rho")) or self._rho is None:
            self._rho = 1.0
        return self._rho

    @rho.setter
    def rho(self, rho):
        self._rho = rho

    @execute_on_cpu
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
        if L is not None:
            warnings.warn(
                "FourierRZToroidalSurface does not have radial resolution, ignoring L"
            )
        if len(args) == 2:
            M, N = args
        elif len(args) == 3:
            _, M, N = args

        M = check_nonnegint(M, "M")
        N = check_nonnegint(N, "N")
        NFP = check_posint(NFP, "NFP")
        self._NFP = int(NFP if NFP is not None else self.NFP)

        if (
            ((N is not None) and (N != self.N))
            or ((M is not None) and (M != self.M))
            or (NFP is not None)
            or ((sym is not None) and (sym != self.sym))
        ):
            self._sym = sym if sym is not None else self.sym
            M = int(M if M is not None else self.M)
            N = int(N if N is not None else self.N)
            R_modes_old = self.R_basis.modes
            Z_modes_old = self.Z_basis.modes
            self.R_basis.change_resolution(
                M=M, N=N, NFP=self.NFP, sym="cos" if self.sym else self.sym
            )
            self.Z_basis.change_resolution(
                M=M, N=N, NFP=self.NFP, sym="sin" if self.sym else self.sym
            )
            self.R_lmn = copy_coeffs(self.R_lmn, R_modes_old, self.R_basis.modes)
            self.Z_lmn = copy_coeffs(self.Z_lmn, Z_modes_old, self.Z_basis.modes)
            self._M = M
            self._N = N

    @optimizable_parameter
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

    @optimizable_parameter
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
                + f"basis with {self.Z_basis.num_modes} modes."
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

    @classmethod
    def from_input_file(cls, path, **kwargs):
        """Create a surface from Fourier coefficients in a DESC or VMEC input file.

        Parameters
        ----------
        path : Path-like or str
            Path to DESC or VMEC input file.
        **kwargs : dict, optional
            keyword arguments to pass to the constructor of the
            FourierRZToroidalSurface being created.

        Returns
        -------
        surface : FourierRZToroidalSurface
            Surface with given Fourier coefficients.

        """
        path = os.path.expanduser(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = InputReader().parse_inputs(path)[-1]
        if (inputs["bdry_ratio"] is not None) and (inputs["bdry_ratio"] != 1):
            warnings.warn(
                "`bdry_ratio` is intended as an input for the continuation method."
                "`bdry_ratio`=1 uses the given surface modes as is, any other  "
                "scalar value will scale the non-axisymmetric  modes by that "
                "value. The final value of `bdry_ratio` in the input file is "
                f"{inputs['bdry_ratio']}, this means the created "
                "FourierRZToroidalSurface will be a scaled version of the "
                "input file boundary."
            )
        surf = cls(
            inputs["surface"][:, 3],
            inputs["surface"][:, 4],
            inputs["surface"][:, 1:3].astype(int),
            inputs["surface"][:, 1:3].astype(int),
            inputs["NFP"],
            inputs["sym"],
            **kwargs,
        )
        return surf

    # TODO (#1385): add k value for number of rotations per field period
    @classmethod
    def from_qp_model(
        cls,
        major_radius=1,
        aspect_ratio=10,
        elongation=2,
        mirror_ratio=0.1,
        torsion=0,
        NFP=1,
        sym=True,
        positive_iota=True,
    ):
        """Create a surface from a near-axis model for quasi-poloidal symmetry.

        Model is based off of section III of Goodman et. al. [1]_

        Parameters
        ----------
        major_radius : float
            Average major radius. Constant term in the R coordinate.
        aspect_ratio : float
            Aspect ratio of the geometry = major radius / average cross-sectional area.
        elongation : float
            Elongation of the elliptical surface = major axis / minor axis.
        mirror_ratio : float
            Mirror ratio generated by toroidal variation of the cross-sectional area.
            Must be <= 1.
        torsion : float
            Vertical extent of the magnetic axis Z coordinate.
            Coefficient of sin(2*phi).
        NFP : int
            Number of field periods.
        sym : bool (optional)
            Whether to enforce stellarator symmetry.
        positive_iota : bool (optional)
            Whether the rotational transform should be positive or negative.

        Returns
        -------
        surface : FourierRZToroidalSurface
            Surface with given geometric properties.

        References
        ----------
        .. [1] Goodman, Alan, et al. "Constructing Precisely Quasi-Isodynamic
          Magnetic Fields." Journal of Plasma Physics 89.5 (2023): 905890504.

        """
        assert mirror_ratio <= 1
        a = major_radius * np.sqrt(elongation) / aspect_ratio  # major axis
        b = major_radius / (aspect_ratio * np.sqrt(elongation))  # minor axis
        epsilon = (1 - np.sqrt(1 - mirror_ratio**2)) / mirror_ratio
        iota_sign = 2 * positive_iota - 1

        R_lmn = np.array(
            [
                major_radius,  # m=0, n=0
                -(elongation + 1) * b / 2 * iota_sign,  # m=1, n=0
                -major_radius / (1 + 4 * NFP**2),  # m=0, n=2
                a * epsilon * iota_sign,  # m=1, n=1
                -(elongation - 1) * b / 2 * iota_sign,  # m=1, n=2
                -(elongation - 1) * b / 2,  # m=-1, n=-2
            ]
        )
        Z_lmn = np.array(
            [
                (elongation + 1) * b / 2 * iota_sign,  # m=-1, n=0
                torsion,  # m=0, n=-2
                -b * epsilon * iota_sign,  # m=-1, n=1
                (elongation - 1) * b / 2,  # m=1, n=-2
                -(elongation - 1) * b / 2 * iota_sign,  # m=-1, n=2
            ]
        )
        modes_R = np.array([[0, 0], [1, 0], [0, 2], [1, 1], [1, 2], [-1, -2]])
        modes_Z = np.array([[-1, 0], [0, -2], [-1, 1], [1, -2], [-1, 2]])

        surf = cls(
            R_lmn=R_lmn, Z_lmn=Z_lmn, modes_R=modes_R, modes_Z=modes_Z, NFP=NFP, sym=sym
        )
        return surf

    @classmethod
    def from_values(
        cls,
        coords,
        theta,
        zeta=None,
        M=6,
        N=6,
        NFP=1,
        sym=True,
        check_orientation=True,
        rcond=None,
        w=None,
    ):
        """Create a surface from given R,Z coordinates in real space.

        Parameters
        ----------
        coords : array-like shape(num_points,3) or Grid
            cylindrical coordinates (R,phi,Z) to fit as a FourierRZToroidalSurface
        theta : ndarray, shape(num_points,)
            Locations in poloidal angle theta where real space coordinates are given.
            Expects same number of angles as coords (num_points),
            This determines the poloidal angle for the resulting surface.
        zeta : ndarray, shape(num_points,)
            Locations in toroidal angle zeta where real space coordinates are given.
            Expects same number of angles as coords (num_points),
            This determines the toroidal angle for the resulting surface.
            if None, defaults to assuming the toroidal angle is the cylindrical phi
            and so sets zeta = phi = coords[:,1]
        M : int
            poloidal resolution of basis used to fit surface with.
            It is recommended to fit with M < num_theta points per toroidal plane,
            i.e. if num_points = num_theta*num_zeta , then want to ensure M < num_theta
        N : int
            toroidal resolution of basis used to fit surface with
            It is recommended to fit with N < num_zeta points per poloidal plane.
            i.e. if num_points = num_theta*num_zeta , then want to ensure N < num_zeta
        NFP : int
            number of toroidal field periods for surface
        sym : bool
            True if surface is stellarator-symmetric
        check_orientation : bool
            whether to check left-handedness of coordinates and flip if necessary.
        rcond : float
            Relative condition number of the fit. Singular values smaller than this
            relative to the largest singular value will be ignored. The default value
            is len(x)*eps, where eps is the relative precision of the float type, about
            2e-16 in most cases.
        w : array-like, shape(num_points,)
            Weights to apply to the sample coordinates. For gaussian
            uncertainties, use 1/sigma (not 1/sigma**2).

        Returns
        -------
        surface : FourierRZToroidalSurface
            Surface with Fourier coefficients fitted from input coords.

        """
        M = check_nonnegint(M, "M", False)
        N = check_nonnegint(N, "N", False)
        NFP = check_posint(NFP, "NFP", False)
        theta = np.asarray(theta)
        assert (
            coords.shape[0] == theta.size
        ), "coords first dimension and theta must have same size"
        if zeta is None:
            zeta = coords[:, 1]
        else:
            raise NotImplementedError("zeta != phi not yet implemented")
        nodes = Grid(
            np.vstack([np.ones_like(theta), theta, coords[:, 1]]).T,
            sort=False,
            jitable=True,
        )

        R = coords[:, 0]
        Z = coords[:, 2]
        R_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym="cos" if sym else False)
        Z_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym="sin" if sym else False)
        if w is None:  # unweighted fit
            transform = Transform(
                nodes, R_basis, build=False, build_pinv=True, rcond=rcond
            )
            Rb_lmn = transform.fit(R)

            transform = Transform(
                nodes, Z_basis, build=False, build_pinv=True, rcond=rcond
            )
            Zb_lmn = transform.fit(Z)
        else:  # perform weighted fit
            # solves system W A x = W b
            # where A is the transform matrix, W is the diagonal weight matrix
            # of weights w, and b is the vector of data points
            w = np.asarray(w)
            W = np.diag(w)
            assert w.size == R.size, "w must same length as number of points being fit"

            transform = Transform(
                nodes, R_basis, build=True, build_pinv=False, method="direct1"
            )
            AR = transform.matrices[transform.method][0][0][0]

            transform = Transform(
                nodes, Z_basis, build=True, build_pinv=False, method="direct1"
            )
            AZ = transform.matrices[transform.method][0][0][0]

            A = block_diag(W @ AR, W @ AZ)
            b = np.concatenate([w * R, w * Z])
            x_lmn = np.linalg.lstsq(A, b, rcond=rcond)[0]

            Rb_lmn = x_lmn[0 : R_basis.num_modes]
            Zb_lmn = x_lmn[R_basis.num_modes :]

        surf = cls(
            Rb_lmn,
            Zb_lmn,
            R_basis.modes[:, 1:],
            Z_basis.modes[:, 1:],
            NFP,
            sym,
            check_orientation=check_orientation,
        )
        return surf

    @classmethod
    def from_shape_parameters(
        cls,
        major_radius=10.0,
        aspect_ratio=10.0,
        elongation=1.0,
        triangularity=0.0,
        squareness=0.0,
        eccentricity=0.0,
        torsion=0.0,
        twist=0.0,
        NFP=1,
        sym=True,
    ):
        """Create a surface using a generalized Miller parameterization.

        Parameters
        ----------
        major_radius : float > 0
            Average major radius.
        aspect_ratio : float > 0
            Ratio of major radius / minor radius.
        elongation : float > 0
            Elongation of the cross section = major axis / minor axis. Value of 1 gives
            a circular cross section. Value > 1 gives vertical elongated cross section,
            value < 1 gives horizontally elongated section.
        triangularity : float
            Positive triangularity makes a "▷" like cross section, negative
            triangularity makes a "◁" like cross section. Surface may self-intersect
            if abs(triangularity) > 1
        squareness : float
            Positive squareness makes a "□" type cross section. Negative squareness
            makes a "+" like cross section. Surface may self-intersect if
            abs(squareness) > 0.5
        eccentricity : float in [0, 1)
            Eccentricity of the magnetic axis. Value of 0 gives circular axis, value of
            1 gives infinitely elongated axis.
        torsion : float
            How non-planar the magnetic axis is.
        twist : float
            How many times the cross section rotates per field period. For integer
            values it is a rigid rotation in phi, for non-integer values it also deforms
            as it goes around toroidally. Values > 0 give a CCW rotation,
            values < 0 give CW rotation. Cross section has zero volume if
            abs(twist)%1 == 1/2.
        NFP : int
            Number of field periods.
        sym : bool (optional)
            Whether to enforce stellarator symmetry.

        Returns
        -------
        surface : FourierRZToroidalSurface
            Surface with given geometric properties.

        """
        errorif(major_radius <= 0, ValueError, "major_radius must be positive")
        errorif(aspect_ratio <= 0, ValueError, "aspect_ratio must be positive")
        errorif(elongation <= 0, ValueError, "elongation should be positive")
        errorif(eccentricity < 0, ValueError, "eccentricity must be in [0,1)")
        errorif(eccentricity >= 1, ValueError, "eccentricity must be in [0,1)")
        errorif(
            abs(twist) % 1 == 0.5,
            ValueError,
            "surface has no volume for abs(twist)%1 == 0.5",
        )
        grid = LinearGrid(L=0, M=30, N=30, NFP=NFP, endpoint=True)
        theta = grid.nodes[:, 1]
        zeta = grid.nodes[:, 2]

        # area = pi*a*b = pi*r^2
        # r ~ sqrt(a*b) -> a = r^2/b
        # elongation = a/b = r^2/b^2 -> b = r/sqrt(elongation)
        r = major_radius / aspect_ratio
        b = r / np.sqrt(elongation)
        a = r * np.sqrt(elongation)

        # create cross section shape using miller parameterization
        Rp = b * np.cos(
            theta + triangularity * np.sin(theta) - squareness * np.sin(2 * theta)
        )
        Zp = -a * np.sin(theta + squareness * np.sin(2 * theta))

        # create axis shape using ellipse + torsion
        a = 2 * major_radius / (1 + np.sqrt(1 - eccentricity**2))
        b = a * np.sqrt(1 - eccentricity**2)
        Ra = (a * b) / np.sqrt(
            b**2 * np.cos(NFP * zeta) ** 2 + a**2 * np.sin(NFP * zeta) ** 2
        )
        # max(2, NFP) ensures that nonzero torsion gives nonplanar axis even for NFP=1
        # otherwise it just tilts the whole axis.
        Za = torsion * np.sin(max(2, NFP) * zeta)
        # combine axis + cross section with twist
        R = Ra + Rp * np.cos(twist * NFP * zeta) - Zp * np.sin(twist * NFP * zeta)
        Z = Za + Zp * np.cos(twist * NFP * zeta) + Rp * np.sin(twist * NFP * zeta)

        # self._compute_orientation falsely returns -1 when twist = 1 since there is
        # no m=1, n=0 mode, but by construction this should be right handed
        # so we can skip that check.
        return cls.from_values(
            np.array([R, zeta, Z]).T, theta, NFP=NFP, sym=sym, check_orientation=False
        )

    def constant_offset_surface(
        self, offset, grid=None, M=None, N=None, full_output=False
    ):
        """Create a FourierRZSurface with constant offset from the base surface (self).

        Implementation of algorithm described in Appendix B of
        "An improved current potential method for fast computation of
        stellarator coil shapes", Landreman (2017)
        https://iopscience.iop.org/article/10.1088/1741-4326/aa57d4

        NOTE: Must have the toroidal angle as the cylindrical toroidal angle
        in order for this algorithm to work properly

        Parameters
        ----------
        base_surface : FourierRZToroidalSurface
            Surface from which the constant offset surface will be found.
        offset : float
            constant offset (in m) of the desired surface from the input surface
            offset will be in the normal direction to the surface.
        grid : Grid, optional
            Grid object of the points on the given surface to evaluate the
            offset points at, from which the offset surface will be created by fitting
            offset points with the basis defined by the given M and N.
            If None, defaults to a LinearGrid with M and N and NFP equal to the
            base_surface.M and base_surface.N and base_surface.NFP
        M : int, optional
            Poloidal resolution of the basis used to fit the offset points
            to create the resulting constant offset surface, by default equal
            to base_surface.M
        N : int, optional
            Toroidal resolution of the basis used to fit the offset points
            to create the resulting constant offset surface, by default equal
            to base_surface.N
        full_output : bool, optional
            If True, also return a dict of useful data about the surfaces and a
            tuple where the first element is the residual from
            the root finding and the second is the number of iterations.

        Returns
        -------
        offset_surface : FourierRZToroidalSurface
            FourierRZToroidalSurface, created from fitting points offset from the input
            surface by the given constant offset.
        data : dict
            dictionary containing  the following data, in the cylindrical basis:
                ``n`` : (``grid.num_nodes`` x 3) array of the unit surface normal on
                    the base_surface evaluated at the input ``grid``
                ``x`` : (``grid.num_nodes`` x 3) array of coordinates on
                    the base_surface evaluated at the input ``grid``
                ``x_offset_surface`` : (``grid.num_nodes`` x 3) array of the
                    coordinates on the offset surface, corresponding to the
                    ``x`` points on the base_surface (i.e. the points to which the
                    offset surface was fit)
        info : tuple
            2 element tuple containing residuals and number of iterations
            for each point. Only returned if ``full_output`` is True

        """
        M = check_nonnegint(M, "M")
        N = check_nonnegint(N, "N")

        base_surface = self
        if grid is None:
            grid = LinearGrid(
                M=base_surface.M * 2,
                N=base_surface.N * 2,
                NFP=base_surface.NFP,
                sym=base_surface.sym,
            )
        assert isinstance(
            base_surface, FourierRZToroidalSurface
        ), "base_surface must be a FourierRZToroidalSurface!"
        M = base_surface.M if M is None else int(M)
        N = base_surface.N if N is None else int(N)

        def n_and_r_jax(nodes):
            data = base_surface.compute(
                ["X", "Y", "Z", "n_rho"],
                grid=Grid(nodes, jitable=True, sort=False),
                method="jitable",
            )

            phi = nodes[:, 2]
            re = jnp.vstack([data["X"], data["Y"], data["Z"]]).T
            n = data["n_rho"]
            n = rpz2xyz_vec(n, phi=phi)
            r_offset = re + offset * n
            return n, re, r_offset

        def fun_jax(zeta_hat, theta, zeta):
            nodes = jnp.vstack((jnp.ones_like(theta), theta, zeta_hat)).T
            n, r, r_offset = n_and_r_jax(nodes)
            return jnp.arctan(r_offset[0, 1] / r_offset[0, 0]) - zeta

        vecroot = jit(
            vmap(
                lambda x0, *p: root_scalar(
                    fun_jax, x0, jac=None, args=p, full_output=full_output
                )
            )
        )
        if full_output:
            zetas, (res, niter) = vecroot(
                grid.nodes[:, 2], grid.nodes[:, 1], grid.nodes[:, 2]
            )
        else:
            zetas = vecroot(grid.nodes[:, 2], grid.nodes[:, 1], grid.nodes[:, 2])

        zetas = np.asarray(zetas)
        nodes = np.vstack((np.ones_like(grid.nodes[:, 1]), grid.nodes[:, 1], zetas)).T
        n, x, x_offsets = n_and_r_jax(nodes)

        data = {}
        data["n"] = xyz2rpz_vec(n, phi=nodes[:, 1])
        data["x"] = xyz2rpz(x)
        data["x_offset_surface"] = xyz2rpz(x_offsets)

        offset_surface = FourierRZToroidalSurface.from_values(
            data["x_offset_surface"],
            theta=nodes[:, 1],
            M=M,
            N=N,
            NFP=base_surface.NFP,
            sym=base_surface.sym,
        )
        if full_output:
            return offset_surface, data, (res, niter)
        else:
            return offset_surface

    def get_axis(self):
        """Get the axis of the surface.

        This method calculates the axis of the surface by finding the mid point of the
        R and Z values at theta=0 and pi degrees for a bunch of toroidal angles and
        fitting a Fourier curve to it.
        For general non-convex surfaces, the geometric center (aka centroid) might not
        be inside the given surface. Since we use the axis for the initial guess of the
        magnetic axis, it is important to have the axis inside the surface, and form
        good nested flux surfaces. This method is a simple way to get a good initial
        guess for the axis.

        Returns
        -------
        axis : FourierRZCurve
            Axis of the surface.

        """
        from desc.geometry import FourierRZCurve

        # over-sample to get a good axis fit
        grid = LinearGrid(rho=1, theta=2, zeta=self.N * 4, NFP=self.NFP)
        data = self.compute(["R", "Z"], grid=grid)
        R = data["R"]
        Z = data["Z"]
        # Calculate the R and Z values at theta=0 and pi degrees for bunch of toroidal
        # angles, find the mid point of them and fit a Fourier curve to it.
        Rout = R[::2]
        Rin = R[1::2]
        Zout = Z[::2]
        Zin = Z[1::2]
        Rmid = (Rout + Rin) / 2
        Zmid = (Zout + Zin) / 2
        phis = (
            jnp.linspace(0, 2 * np.pi / self.NFP, len(Rmid), endpoint=False)
            if self.N > 0
            else jnp.zeros_like(Rmid)
        )
        axis = FourierRZCurve.from_values(
            jnp.vstack([Rmid, phis, Zmid]).T, N=self.N, NFP=self.NFP, sym=self.sym
        )
        return axis


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
        Indexing method, default value = ``'ansi'``

        For L=0, all methods are equivalent and give a "chevron" shaped
        basis (only the outer edge of the zernike pyramid of width M).
        For L>0, the indexing scheme defines order of the basis functions:

        ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
        decreasing size, ending in a triangle shape. For L == M,
        the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
        to the bottom of the pyramid, increasing L while keeping M constant,
        giving a "house" shape

        ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
        decreasing size, ending in a diamond shape for L=2*M where
        the traditional fringe/U of Arizona indexing is recovered.
        For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond
    L, M : int or None
        Maximum radial and poloidal mode numbers. Defaults to max from modes_R and
        modes_Z.
    zeta : float [0,2pi)
        toroidal angle for the section.
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
        "_spectral_indexing",
        "_zeta",
    ]

    _static_attrs = Surface._static_attrs + [
        "_spectral_indexing",
        "_R_basis",
        "_Z_basis",
    ]

    @execute_on_cpu
    def __init__(
        self,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        spectral_indexing="ansi",
        sym="auto",
        L=None,
        M=None,
        zeta=0.0,
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

        assert (
            R_lmn.size == modes_R.shape[0]
        ), "R_lmn size and modes_R.shape[0] must be the same size!"
        assert (
            Z_lmn.size == modes_Z.shape[0]
        ), "Z_lmn size and modes_Z.shape[0] must be the same size!"

        assert issubclass(modes_R.dtype.type, np.integer)
        assert issubclass(modes_Z.dtype.type, np.integer)

        LR = np.max(abs(modes_R[:, 0]))
        MR = np.max(abs(modes_R[:, 1]))
        LZ = np.max(abs(modes_Z[:, 0]))
        MZ = np.max(abs(modes_Z[:, 1]))
        L = check_nonnegint(L, "L")
        M = check_nonnegint(M, "M")
        self._L = setdefault(L, max(LR, LZ))
        self._M = setdefault(M, max(MR, MZ))
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
            L=self._L,
            M=self._M,
            spectral_indexing=spectral_indexing,
            sym="cos" if sym else False,
        )
        self._Z_basis = ZernikePolynomial(
            L=self._L,
            M=self._M,
            spectral_indexing=spectral_indexing,
            sym="sin" if sym else False,
        )

        self._R_lmn = copy_coeffs(R_lmn, modes_R, self.R_basis.modes[:, :2])
        self._Z_lmn = copy_coeffs(Z_lmn, modes_Z, self.Z_basis.modes[:, :2])
        self._sym = bool(sym)
        self._spectral_indexing = str(spectral_indexing)

        self._zeta = zeta

        if check_orientation and self._compute_orientation() == -1:
            warnings.warn(
                "Left handed coordinates detected, switching sign of theta."
                + " To avoid this warning in the future, switch the sign of all"
                + " modes with m<0"
            )
            self._flip_orientation()
            assert self._compute_orientation() == 1

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
    def zeta(self):
        """float: Toroidal angle."""
        return self._zeta

    @zeta.setter
    def zeta(self, zeta):
        self._zeta = zeta

    @execute_on_cpu
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
        if N is not None:
            warnings.warn(
                "ZernikeRZToroidalSection does not have toroidal resolution, ignoring N"
            )
        if len(args) == 2:
            L, M = args
        elif len(args) == 3:
            L, M, _ = args

        L = check_nonnegint(L, "L")
        M = check_nonnegint(M, "M")

        if (
            ((L is not None) and (L != self.L))
            or ((M is not None) and (M != self.M))
            or ((sym is not None) and (sym != self.sym))
        ):
            self._sym = sym if sym is not None else self.sym
            L = int(L if L is not None else self.L)
            M = int(M if M is not None else self.M)
            R_modes_old = self.R_basis.modes
            Z_modes_old = self.Z_basis.modes
            self.R_basis.change_resolution(
                L=L, M=M, sym="cos" if self.sym else self.sym
            )
            self.Z_basis.change_resolution(
                L=L, M=M, sym="sin" if self.sym else self.sym
            )
            self.R_lmn = copy_coeffs(self.R_lmn, R_modes_old, self.R_basis.modes)
            self.Z_lmn = copy_coeffs(self.Z_lmn, Z_modes_old, self.Z_basis.modes)
            self._L = L
            self._M = M

    @optimizable_parameter
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

    @optimizable_parameter
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

    def get_axis(self):
        """Get the axis of the surface.

        Computes the R and Z value at rho=0 and creates N=0 FourierRZCurve
        to represent the axis of the cross section.

        Returns
        -------
        axis : FourierRZCurve
            Circular axis of the surface.

        """
        from desc.geometry import FourierRZCurve

        grid = LinearGrid(rho=0)
        data = self.compute(["R", "Z"], grid=grid)
        axis = FourierRZCurve(R_n=data["R"][0], Z_n=data["Z"][0], sym=self.sym)
        return axis
