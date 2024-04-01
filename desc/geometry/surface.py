"""Classes for 2D surfaces embedded in 3D space."""

import numbers
import time
import warnings

import numpy as np
from scipy.sparse.linalg import splu
from scipy.special import roots_sh_jacobi

from desc.backend import jnp, put, sign
from desc.basis import DoubleFiniteElementBasis, DoubleFourierSeries, ZernikePolynomial
from desc.io import InputReader
from desc.optimizable import optimizable_parameter
from desc.utils import copy_coeffs, isposint

from .core import Surface

__all__ = [
    "FourierRZToroidalSurface",
    "ZernikeRZToroidalSection",
    "convert_spectral_to_FE",
    "FiniteElementRZToroidalSurface",
]


def convert_spectral_to_FE(
    R_lmn,
    Z_lmn,
    L_lmn,
    R_basis,
    Z_basis,
    L_basis,
    Rprime_basis,
    Zprime_basis,
    Lprime_basis,
):
    """Converts between double Fourier and double FE representation.

    Parameters
    ----------
    R_lmn : ndarray, shape(k, 3)
        Fourier coefficients of R(rho, theta, zeta)
    Z_lmn : ndarray, shape(k, 3)
        Fourier coefficients of Z(rho, theta, zeta)
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn.
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
    L_lmn : ndarray, shape(k, 3)
        Fourier coefficients of Lambda(rho, theta, zeta), Default = None
    R_basis : DoubleFourier or FourierZernike, shape (L, 2M + 1, 2N + 1)
        Basis elements representing the dependence in (rho, theta, phi).
    Z_basis : DoubleFourier or FourierZernike, shape (L, 2M + 1, 2N + 1)
        Basis elements representing the dependence in (rho, theta, phi).
    L_basis : DoubleFourier or FourierZernike, shape (L, 2M + 1, 2N + 1)
        Basis elements representing the dependence in (rho, theta, phi).
    Rprime_basis : DoubleFE, shape (I, Q) or shape (L, I, Q)
        Basis elements representing the dependence in (rho, theta, phi).
    Zprime_basis : DoubleFE, shape (I, Q) or shape (L, I, Q)
        Basis elements representing the dependence in (rho, theta, phi).
    Lprime_basis : DoubleFE, shape (I, Q) or shape (L, I, Q)
        Basis elements representing the dependence in (rho, theta, phi).

    Returns
    -------
    tildeR_lmn : ndarray, shape (kk, 3)
        Finite element coefficients of R(rho, theta, zeta)
    tildeZ_lmn : ndarray, shape (kk, 3)
        Finite element coefficients of Z(rho, theta, zeta)
    tildeL_lmn : ndarray, shape (kk, 3)
        Finite element coefficients of L(rho, theta, zeta)

    """
    # Assume uniform grid
    N = R_basis.N
    I = Rprime_basis.I_2MN
    Q = Rprime_basis.Q
    L = Rprime_basis.L
    lmodes = (L + 3) // 2
    nmodes = lmodes * I * Q
    mesh = Rprime_basis.mesh
    nquad = mesh.nquad

    # Get the angular quadrature points
    quadpoints_mesh = np.array(mesh.return_quadrature_points())
    if N == 0:
        quadpoints = (
            np.array(
                np.meshgrid(np.ones(1), quadpoints_mesh, np.zeros(1), indexing="ij")
            ).reshape(3, I * nquad)
        ).T
    else:
        quadpoints = (
            np.array(np.meshgrid(np.ones(1), quadpoints_mesh, indexing="ij")).reshape(
                3, I * nquad
            )
        ).T

    # Compute the A matrix in Ax = b, should be exactly the same
    # as in the 2D case.
    t1 = time.time()
    Rprime_pre_evaluated = Rprime_basis.evaluate(nodes=quadpoints)
    Zprime_pre_evaluated = Zprime_basis.evaluate(nodes=quadpoints)
    Lprime_pre_evaluated = Lprime_basis.evaluate(nodes=quadpoints)
    IQ = I * Q
    Bjb_Z = mesh.integrate(
        (
            Zprime_pre_evaluated[:, np.newaxis, :IQ]
            * Zprime_pre_evaluated[:, :IQ, np.newaxis]
        ).reshape(I * nquad, -1)
    ).reshape(I, Q, I, Q)
    Bjb_R = mesh.integrate(
        (
            Rprime_pre_evaluated[:, np.newaxis, :IQ]
            * Rprime_pre_evaluated[:, :IQ, np.newaxis]
        ).reshape(I * nquad, -1)
    ).reshape(I, Q, I, Q)
    Bjb_L = mesh.integrate(
        (
            Lprime_pre_evaluated[:, np.newaxis, :IQ]
            * Lprime_pre_evaluated[:, :IQ, np.newaxis]
        ).reshape(I * nquad, -1)
    ).reshape(I, Q, I, Q)
    # Commenting out k_plus1 = np.arange(0, L + 1, 2) + 1

    t2 = time.time()
    print("Time to construct A matrix = ", t2 - t1)

    t1 = time.time()

    # quadrature using the roots of the shifted jacobi polynomials
    # Integrate 2*Nrho - 1 order polynomial exactly
    Nrho = 2 * (L + 3)

    # weights for integral rho * R_l * R_k
    rho, rho_weights = roots_sh_jacobi(Nrho, 2, 2)

    # Get the angular quadrature points
    quadpoints_mesh = np.array(mesh.return_quadrature_points())
    if N == 0:
        quadpoints = (
            np.array(
                np.meshgrid(rho, quadpoints_mesh, np.zeros(1), indexing="ij")
            ).reshape(3, I * nquad * Nrho)
        ).T
    else:
        quadpoints = (
            np.array(np.meshgrid(rho, quadpoints_mesh, indexing="ij")).reshape(
                3, I * nquad * Nrho
            )
        ).T
    R_pre_evaluated = R_basis.evaluate(nodes=quadpoints)
    Z_pre_evaluated = Z_basis.evaluate(nodes=quadpoints)
    L_pre_evaluated = L_basis.evaluate(nodes=quadpoints)

    # Evaluated FE basis functions but using the usual R_k(rho) basis
    # functions without any rho factors.
    Rprime_pre_evaluated = Rprime_basis.evaluate(nodes=quadpoints)
    Zprime_pre_evaluated = Zprime_basis.evaluate(nodes=quadpoints)
    Lprime_pre_evaluated = Lprime_basis.evaluate(nodes=quadpoints)
    R_sum_pre_evaluated = R_pre_evaluated @ R_lmn
    Z_sum_pre_evaluated = Z_pre_evaluated @ Z_lmn
    L_sum_pre_evaluated = L_pre_evaluated @ L_lmn
    ZZ = Z_sum_pre_evaluated[:, np.newaxis] * Zprime_pre_evaluated
    RR = R_sum_pre_evaluated[:, np.newaxis] * Rprime_pre_evaluated
    LL = L_sum_pre_evaluated[:, np.newaxis] * Lprime_pre_evaluated

    # Note integral rho R_k(rho) = 0 unless k = 0 by orthogonality
    Aj_Z = mesh.integrate(
        # Integrate rho * basis functions over the rho direction
        np.sum(
            rho_weights[:, np.newaxis, np.newaxis] * (ZZ).reshape(Nrho, I * nquad, -1),
            axis=0,
        )
    )
    Aj_R = mesh.integrate(
        np.sum(
            rho_weights[:, np.newaxis, np.newaxis] * (RR).reshape(Nrho, I * nquad, -1),
            axis=0,
        )
    )
    Aj_L = mesh.integrate(
        np.sum(
            rho_weights[:, np.newaxis, np.newaxis] * (LL).reshape(Nrho, I * nquad, -1),
            axis=0,
        )
    )
    t2 = time.time()
    print("Time to construct vector b = ", t2 - t1)

    # Constructed the matrices such that Bjb * Rprime = Aj and now need to solve
    # this linear system of equations. Use an LU
    lu = splu(Bjb_R)  # [:nmodes - (L + 3) // 2, :nmodes - (L + 3) // 2])
    Rprime = lu.solve(Aj_R)  # [:nmodes - (L + 3) // 2])
    Rprime_lmn = Rprime.reshape(nmodes)  # - (L + 3) // 2)
    lu = splu(Bjb_Z)  # [:nmodes - (L + 3) // 2, :nmodes - (L + 3) // 2])
    Zprime = lu.solve(Aj_Z)  # [:nmodes - (L + 3) // 2])
    Zprime_lmn = Zprime.reshape(nmodes)  # - (L + 3) // 2)
    lu = splu(Bjb_L)  # [:nmodes - (L + 3) // 2, :nmodes - (L + 3) // 2])
    Lprime = lu.solve(Aj_L)  # [:nmodes - (L + 3) // 2])
    Lprime_lmn = Lprime.reshape(nmodes)  # - (L + 3) // 2)
    t2 = time.time()
    print("Time to solve Ax = b, ", t2 - t1)
    return Rprime_lmn, Zprime_lmn, Lprime_lmn


class FiniteElementRZToroidalSurface(Surface):
    """Toroidal surface represented by finite elements in poloidal and toroidal angles.

    Parameters
    ----------
    R_lmn, Z_lmn : array-like, shape(k,)
        Finite Element coefficients for R and Z in cylindrical coordinates
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
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
        "rho",
        "_NFP",
    ]

    def __init__(
        self,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        K=1,
        NFP=1,
        sym="auto",
        rho=1,
        name="",
        check_orientation=True,
    ):
        # Initialize a FourierRZToroidalSurface
        self.fs = FourierRZToroidalSurface(
            R_lmn=R_lmn,
            Z_lmn=Z_lmn,
            modes_R=modes_R,
            modes_Z=modes_Z,
            NFP=NFP,
            sym=sym,
            rho=rho,
            name=name,
            check_orientation=check_orientation,
        )
        # Convert coefficients to a FE representation
        self._R_basis = DoubleFiniteElementBasis(M=self.fs._M, N=self.fs._N)
        self._Z_basis = DoubleFiniteElementBasis(M=self.fs._M, N=self.fs._N)
        R_lmn, Z_lmn = convert_spectral_to_FE(
            self.fs.R_lmn,
            self.fs.Z_lmn,
            self.fs.R_basis,
            self.fs.Z_basis,
            self.fs.L_basis,
            self._R_basis,
            self._Z_basis,
            self._L_basis,
        )
        I = 2 * self.fs._M * self.fs._N
        self.K = K
        Q = int((K + 1) * (K + 2) / 2.0)
        modes_R, modes_Z = np.meshgrid(I, Q, indexing="ij")
        modes_R = modes_R.reshape(-1, 2)
        modes_Z = modes_Z.reshape(-1, 2)
        self.R_lmn = R_lmn
        self.Z_lmn = Z_lmn
        self.rho = rho
        self._R_lmn = copy_coeffs(R_lmn, modes_R, self.R_basis.modes[:, 1:])
        self._Z_lmn = copy_coeffs(Z_lmn, modes_Z, self.Z_basis.modes[:, 1:])

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self.fs._NFP

    @NFP.setter
    def NFP(self, new):
        assert (
            isinstance(new, numbers.Real) and int(new) == new and new > 0
        ), f"NFP should be a positive integer, got {type(new)}"
        self.fs.change_resolution(NFP=new)

    @property
    def R_basis(self):
        """Double finite element basis for R."""
        return self._R_basis

    @property
    def Z_basis(self):
        """Double finite element basis for Z."""
        return self._Z_basis

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
            self.R_basis.change_resolution(I=M, J=N)
            self.Z_basis.change_resolution(I=M, J=N)
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
        """Get finite element coefficients for given mode number(s)."""
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
        """Set specific finite element coefficients."""
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
    def from_input_file(cls, path):
        """Create a finite element surface from Fourier coefficients in a file.

        Parameters
        ----------
        path : Path-like or str
            Path to DESC or VMEC input file.

        Returns
        -------
        surface : FiniteElementRZToroidalSurface
            Surface with given finite element coefficients.

        """
        surf = cls()
        fs = surf.fs.from_input_file(path=path)
        R_lmn, Z_lmn = convert_spectral_to_FE(
            fs.R_lmn,
            fs.Z_lmn,
            fs.L_lmn,
            fs.R_basis,
            fs.Z_basis,
            fs.L_basis,
            surf._R_basis,
            surf._Z_basis,
            surf._L_basis,
        )
        I = fs._M * 2
        J = fs._N * 2
        modes_R, modes_Z = np.meshgrid(I, J, indexing="ij")
        modes_R = modes_R.reshape(-1, 2)
        modes_Z = modes_Z.reshape(-1, 2)
        surf = cls(R_lmn=R_lmn, Z_lmn=Z_lmn, modes_R=modes_R, modes_Z=modes_Z)
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
        surf = cls()
        fs = surf.fs.from_near_axis(aspect_ratio, elongation, mirror_ratio, axis_Z, NFP)
        R_lmn, Z_lmn = convert_spectral_to_FE(
            fs.R_lmn,
            fs.Z_lmn,
            fs.L_lmn,
            fs.R_basis,
            fs.Z_basis,
            fs.L_basis,
            surf._R_basis,
            surf._Z_basis,
            surf._L_basis,
        )
        I = fs._M * 2
        J = fs._N * 2
        modes_R, modes_Z = np.meshgrid(I, J, indexing="ij")
        modes_R = modes_R.reshape(-1, 2)
        modes_Z = modes_Z.reshape(-1, 2)
        surf = cls(R_lmn=R_lmn, Z_lmn=Z_lmn, modes_R=modes_R, modes_Z=modes_Z)
        return surf


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
        assert isposint(NFP)
        NFP = int(NFP)
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
        self._NFP = int(NFP if NFP is not None else self.NFP)
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
        isVMEC = False
        for line in f.readlines():
            if "&INDATA" in line.upper():
                isVMEC = True
                break
        if isVMEC:  # vmec input, convert to desc
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
        "zeta",
        "_spectral_indexing",
    ]

    def __init__(
        self,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        spectral_indexing="ansi",
        sym="auto",
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
