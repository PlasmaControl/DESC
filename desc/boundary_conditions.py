import numpy as np
from abc import ABC, abstractmethod
from termcolor import colored

from desc.backend import jnp, put
from desc.basis import zernike_radial_coeffs
from desc.optimize.constraint import LinearEqualityConstraint
from desc.grid import LinearGrid
from desc.transform import Transform
from desc.utils import unpack_state, copy_coeffs


__all__ = [
    "LCFSConstraint",
    "PoincareConstraint",
    "UmbilicConstraint",
    "get_boundary_condition",
]


class BoundaryCondition(LinearEqualityConstraint, ABC):
    """Defines the template for different types of boundary conditions.

    Parameters
    ----------
    R_basis : Basis
        Fourier-Zernike basis for R
    Z_basis : Basis
        Fourier-Zernike basis for Z
    L_basis : Basis
        Fourier-Zernike basis for lambda
    Rb_basis : Basis
        Double Fourier basis for boundary R
    Zb_basis : Basis
        Double Fourier basis for boundary Z
    Rb_lmn : ndarray
        Array of spectral coefficients for boundary R
    Zb_lmn : ndarray
        Array of spectral coefficients for boundary Z
    build : bool
        whether to compute null space and pseudoinverse now or wait until needed.
    """

    @abstractmethod
    def __init__(
        self, R_basis, Z_basis, L_basis, Rb_basis, Zb_basis, Rb_lmn, Zb_lmn, build=True
    ):
        pass

    @abstractmethod
    def recover_from_constraints(self, y, Rb_lmn=None, Zb_lmn=None):
        """Recover full state vector that satifies linear constraints."""

    @property
    @abstractmethod
    def name(self):
        """Name of objective function (str)."""


class LCFSConstraint(BoundaryCondition):
    """Fixed-boundary constraint where the rho=1 surface is given.

    enforces:

    * R(1,theta,zeta) = Rb(theta,zeta)
    * Z(1,theta,zeta) = Zb(theta,zeta)
    * lambda(0,theta,zeta) == 0
    * lambda(rho,0,0) == 0

    Parameters
    ----------
    R_basis : Basis
        Fourier-Zernike basis for R
    Z_basis : Basis
        Fourier-Zernike basis for Z
    L_basis : Basis
        Fourier-Zernike basis for lambda
    Rb_basis : Basis
        Double Fourier basis for boundary R
    Zb_basis : Basis
        Double Fourier basis for boundary Z
    Rb_lmn : ndarray
        Array of spectral coefficients for boundary R
    Zb_lmn : ndarray
        Array of spectral coefficients for boundary Z
    build : bool
        whether to compute null space and pseudoinverse now or wait until needed.
    """

    _io_attrs_ = BoundaryCondition._io_attrs_ + [
        "_brmask",
        "_bzmask",
        "_rmask",
        "_zmask",
    ]

    def __init__(
        self, R_basis, Z_basis, L_basis, Rb_basis, Zb_basis, Rb_lmn, Zb_lmn, build=True
    ):

        Rb_basis_full = Rb_basis.copy()
        Zb_basis_full = Zb_basis.copy()
        # make sure the boundary bases have same resolution as flux surface bases so that
        # all modes are properly constrainted
        Rb_basis_full.change_resolution(M=R_basis.M, N=R_basis.N)
        Zb_basis_full.change_resolution(M=Z_basis.M, N=Z_basis.N)
        Rb_lmn_full = copy_coeffs(Rb_lmn, Rb_basis.modes, Rb_basis_full.modes)
        Zb_lmn_full = copy_coeffs(Zb_lmn, Zb_basis.modes, Zb_basis_full.modes)

        # find indices where to insert boundary coeffs into constraint array
        self._brmask = np.where(
            (Rb_basis_full.modes[:, np.newaxis] == Rb_basis.modes[np.newaxis, :, :])
            .all(axis=-1)
            .any(-1)
        )[0]
        self._bzmask = (
            np.where(
                (Zb_basis_full.modes[:, np.newaxis] == Zb_basis.modes[np.newaxis, :, :])
                .all(axis=-1)
                .any(-1)
            )[0]
            + Rb_basis_full.num_modes
        )
        # find which coeffs can acually be constrainted
        self._rmask = np.where(
            (Rb_basis.modes[:, np.newaxis] == Rb_basis_full.modes[np.newaxis, :, :])
            .all(axis=-1)
            .any(-1)
        )[0]
        self._zmask = np.where(
            (Zb_basis.modes[:, np.newaxis] == Zb_basis_full.modes[np.newaxis, :, :])
            .all(axis=-1)
            .any(-1)
        )[0]

        A_lcfs, b_lcfs = _get_lcfs_bc(
            R_basis,
            Z_basis,
            L_basis,
            Rb_basis_full,
            Zb_basis_full,
            Rb_lmn_full,
            Zb_lmn_full,
        )
        A_axis, b_axis = _get_axis_bc(R_basis, Z_basis, L_basis)
        A_gauge, b_gauge = _get_gauge_bc(R_basis, Z_basis, L_basis)

        A = np.vstack([A_lcfs, A_axis, A_gauge])
        b = np.concatenate([b_lcfs, b_axis, b_gauge])

        super(BoundaryCondition, self).__init__(A, b, build)

    def recover_from_constraints(self, y, Rb_lmn=None, Zb_lmn=None):
        """Recover full state vector that satifies linear constraints."""
        if Rb_lmn is not None and Zb_lmn is not None:
            b = jnp.zeros(self.b.shape)
            b = put(b, self._brmask, Rb_lmn[self._rmask])
            b = put(b, self._bzmask, Zb_lmn[self._zmask])
        else:
            b = self.b

        x0 = jnp.dot(self.Ainv, b)
        x = x0 + jnp.dot(self.Z, y)
        return x

    @property
    def name(self):
        """Name of objective function (str)."""
        return "lcfs"


class PoincareConstraint(BoundaryCondition):
    """Boundary condition where the Poincare section at zeta=0 is given.

    enforces:

    * R(rho,theta,0) = Rb(rho,theta)
    * Z(rho,theta,0) = Zb(rho,theta)
    * lambda(rho,theta,zeta) = 0

    Parameters
    ----------
    R_basis : Basis
        Fourier-Zernike basis for R
    Z_basis : Basis
        Fourier-Zernike basis for Z
    L_basis : Basis
        Fourier-Zernike basis for lambda
    Rb_basis : Basis
        Zernike polynomial basis for boundary R
    Zb_basis : Basis
        Zernike polynomial basis for boundary Z
    Rb_lmn : ndarray
        Array of spectral coefficients for boundary R
    Zb_lmn : ndarray
        Array of spectral coefficients for boundary Z
    build : bool
        whether to compute null space and pseudoinverse now or wait until needed.
    """

    _io_attrs_ = BoundaryCondition._io_attrs_ + [
        "_brmask",
        "_bzmask",
        "_rmask",
        "_zmask",
    ]

    def __init__(
        self, R_basis, Z_basis, L_basis, Rb_basis, Zb_basis, Rb_lmn, Zb_lmn, build=True
    ):
        Rb_basis_full = Rb_basis.copy()
        Zb_basis_full = Zb_basis.copy()
        Rb_basis_full.change_resolution(L=R_basis.L, M=R_basis.M)
        Zb_basis_full.change_resolution(L=Z_basis.L, M=Z_basis.M)
        Rb_lmn_full = copy_coeffs(Rb_lmn, Rb_basis.modes, Rb_basis_full.modes)
        Zb_lmn_full = copy_coeffs(Zb_lmn, Zb_basis.modes, Zb_basis_full.modes)

        # find indices where to insert boundary coeffs into constraint array
        self._brmask = np.where(
            (Rb_basis_full.modes[:, np.newaxis] == Rb_basis.modes[np.newaxis, :, :])
            .all(axis=-1)
            .any(-1)
        )[0]
        self._bzmask = (
            np.where(
                (Zb_basis_full.modes[:, np.newaxis] == Zb_basis.modes[np.newaxis, :, :])
                .all(axis=-1)
                .any(-1)
            )[0]
            + Rb_basis_full.num_modes
        )
        # find which coeffs can acually be constrainted
        self._rmask = np.where(
            (Rb_basis.modes[:, np.newaxis] == Rb_basis_full.modes[np.newaxis, :, :])
            .all(axis=-1)
            .any(-1)
        )[0]
        self._zmask = np.where(
            (Zb_basis.modes[:, np.newaxis] == Zb_basis_full.modes[np.newaxis, :, :])
            .all(axis=-1)
            .any(-1)
        )[0]

        A_poincare, b_poincare = _get_poincare_bc(
            R_basis,
            Z_basis,
            L_basis,
            Rb_basis_full,
            Zb_basis_full,
            Rb_lmn_full,
            Zb_lmn_full,
        )
        A_sfl, b_sfl = _get_sfl_bc(R_basis, Z_basis, L_basis)

        A = np.vstack([A_poincare, A_sfl])
        b = np.concatenate([b_poincare, b_sfl])

        super(BoundaryCondition, self).__init__(A, b, build)

    def recover_from_constraints(self, y, Rb_lmn=None, Zb_lmn=None):
        """Recover full state vector that satifies linear constraints."""
        if Rb_lmn is not None and Zb_lmn is not None:
            b = jnp.zeros(self.b.shape)
            b = put(b, self._brmask, Rb_lmn[self._rmask])
            b = put(b, self._bzmask, Zb_lmn[self._zmask])
        else:
            b = self.b

        x0 = jnp.dot(self.Ainv, b)
        x = x0 + jnp.dot(self.Z, y)
        return x

    @property
    def name(self):
        """Name of objective function (str)."""
        return "poincare"


class UmbilicConstraint(BoundaryCondition):
    """Boundary condition for umbilic-type stellarators.

    enforces:

    * R(rho,theta,zeta) = a(rho)*cos(m*theta)*cos(zeta) + b(rho)*cos(2*m*theta)*cos(zeta)
      - a(rho)*sin(m*theta)*sin(zeta) + b(rho)*sin(2*m*theta)*sin(zeta)
    * Z(rho,theta,zeta) = a(rho)*cos(m*theta)*sin(zeta) + b(rho)*cos(2*m*theta)*sin(zeta)
      + a(rho)*sin(m*theta)*cos(zeta) - b(rho)*sin(2*m*theta)*cos(zeta)
    * lambda(rho,theta,zeta) = 0

    Parameters
    ----------
    R_basis : Basis
        Fourier-Zernike basis for R
    Z_basis : Basis
        Fourier-Zernike basis for Z
    L_basis : Basis
        Fourier-Zernike basis for lambda
    Rb_basis : Basis
        Zernike polynomial basis for boundary R
    Zb_basis : Basis
        Zernike polynomial basis for boundary Z
    Rb_lmn : ndarray
        Array of spectral coefficients for boundary R
    Zb_lmn : ndarray
        Array of spectral coefficients for boundary Z
    build : bool
        whether to compute null space and pseudoinverse now or wait until needed.
    """

    _io_attrs_ = BoundaryCondition._io_attrs_ + ["_dim_umbilic", "_dim_sfl"]

    def __init__(
        self, R_basis, Z_basis, L_basis, Rb_basis, Zb_basis, Rb_lmn, Zb_lmn, build=True
    ):

        A_umbilic, b_umbilic = _get_umbilic_bc(R_basis, Z_basis)
        A_sfl, b_sfl = _get_sfl_bc(R_basis, Z_basis, L_basis)

        self._dim_umbilic = b_umbilic.size
        self._dim_sfl = b_sfl.size

        A = np.vstack([A_umbilic, A_sfl])
        b = np.concatenate([b_umbilic, b_sfl])

        super(BoundaryCondition, self).__init__(A, b, build)

    def recover_from_constraints(self, y, Rb_lmn=None, Zb_lmn=None):
        """Recover full state vector that satifies linear constraints."""
        if Rb_lmn is not None and Zb_lmn is not None:
            z1 = jnp.zeros(self._dim_umbilic)
            z2 = jnp.zeros(self._dim_sfl)
            b = jnp.concatenate([z1, z2])
        else:
            b = self.b

        x0 = jnp.dot(self.Ainv, b)
        x = x0 + jnp.dot(self.Z, y)
        return x

    @property
    def name(self):
        """Name of objective function (str)."""
        return "umbilic"


def get_boundary_condition(
    condition, R_basis, Z_basis, L_basis, Rb_basis, Zb_basis, Rb_lmn, Zb_lmn, build=True
):
    """Get a boundary condition by name.

    Parameters
    ----------
    condition : str
        name of the desired boundary condition, eg ``'lcfs'`` or ``'poincare'``
    R_basis : Basis
        Fourier-Zernike basis for R
    Z_basis : Basis
        Fourier-Zernike basis for Z
    L_basis : Basis
        Fourier-Zernike basis for lambda
    Rb_basis : Basis
        Zernike polynomial basis for boundary R
    Zb_basis : Basis
        Zernike polynomial basis for boundary Z
    Rb_lmn : ndarray
        Array of spectral coefficients for boundary R
    Zb_lmn : ndarray
        Array of spectral coefficients for boundary Z
    build : bool
        whether to compute null space and pseudoinverse now or wait until needed.

    Returns
    -------
    bdry_con : BoundaryCondition
        boundary condition initialized with the given bases and constraints

    """
    if condition == "lcfs":
        bdry_con = LCFSConstraint(
            R_basis=R_basis,
            Z_basis=Z_basis,
            L_basis=L_basis,
            Rb_basis=Rb_basis,
            Zb_basis=Zb_basis,
            Rb_lmn=Rb_lmn,
            Zb_lmn=Zb_lmn,
            build=build,
        )
    elif condition == "poincare":
        bdry_con = PoincareConstraint(
            R_basis=R_basis,
            Z_basis=Z_basis,
            L_basis=L_basis,
            Rb_basis=Rb_basis,
            Zb_basis=Zb_basis,
            Rb_lmn=Rb_lmn,
            Zb_lmn=Zb_lmn,
            build=build,
        )
    elif condition == "umbilic":
        bdry_con = UmbilicConstraint(
            R_basis=R_basis,
            Z_basis=Z_basis,
            L_basis=L_basis,
            Rb_basis=Rb_basis,
            Zb_basis=Zb_basis,
            Rb_lmn=Rb_lmn,
            Zb_lmn=Zb_lmn,
            build=build,
        )
    else:
        raise ValueError(
            colored(
                "Requested Boundary Condition is not implemented. "
                + "Available boundary conditions are: "
                + "'lcfs', 'poincare', 'umbilic'",
                "red",
            )
        )

    return bdry_con


def _get_lcfs_bc(R_basis, Z_basis, L_basis, Rb_basis, Zb_basis, Rb_lmn, Zb_lmn):
    """Compute constraint matrices for the shape of the last closed flux surface.

    enforces:
    R(1,theta,zeta) = Rb(theta,zeta)
    Z(1,theta,zeta) = Zb(theta,zeta)

    Parameters
    ----------
    R_basis : Basis
        Fourier-Zernike basis for R
    Z_basis : Basis
        Fourier-Zernike basis for Z
    L_basis : Basis
        Fourier-Zernike basis for lambda
    Rb_basis : Basis
        Double Fourier basis for boundary R
    Zb_basis : Basis
        Double Fourier basis for boundary Z
    Rb_lmn : ndarray
        Array of spectral coefficients for boundary R
    Zb_lmn : ndarray
        Array of spectral coefficients for boundary Z

    Returns
    -------
    A, b : ndarray
        Constraint matrix and vector such that the constraint is satisfied
        if and only if Ax=b

    """
    assert (
        len(Rb_lmn) == Rb_basis.num_modes
    ), f"Mismatch between number of boundary modes {Rb_basis.num_modes} and number of boundary coefficients {len(Rb_lmn)}"
    assert (
        len(Zb_lmn) == Zb_basis.num_modes
    ), f"Mismatch between number of boundary modes {Zb_basis.num_modes} and number of boundary coefficients {len(Zb_lmn)}"
    assert R_basis.M <= Rb_basis.M, "Boundary does not fully constraint R_basis"
    assert Z_basis.N <= Zb_basis.N, "Boundary does not fully constraint Z_basis"

    R_modes = R_basis.modes
    Z_modes = Z_basis.modes
    Rb_modes = Rb_basis.modes
    Zb_modes = Zb_basis.modes

    dim_R = R_basis.num_modes
    dim_Z = Z_basis.num_modes
    dim_L = L_basis.num_modes
    dim_Rb = Rb_basis.num_modes
    dim_Zb = Zb_basis.num_modes

    dimx = dim_R + dim_Z + dim_L

    AR = np.zeros((dim_Rb, dimx))
    AZ = np.zeros((dim_Zb, dimx))
    bR = Rb_lmn
    bZ = Zb_lmn

    for i, (l, m, n) in enumerate(R_modes):
        j = np.argwhere(np.logical_and(Rb_modes[:, 1] == m, Rb_modes[:, 2] == n))
        AR[j, i] = 1

    for i, (l, m, n) in enumerate(Z_modes):
        j = np.argwhere(np.logical_and(Zb_modes[:, 1] == m, Zb_modes[:, 2] == n))
        AZ[j, dim_R + i] = 1

    A = np.vstack([AR, AZ])
    b = np.concatenate([bR, bZ])

    return A, b


def _get_poincare_bc(R_basis, Z_basis, L_basis, Rb_basis, Zb_basis, Rb_lmn, Zb_lmn):
    """Compute constraint matrices for the shape of a Poincare surface.

    enforces:
    R(rho,theta,0) = Rb(rho,theta)
    Z(rho,theta,0) = Zb(rho,theta)

    Parameters
    ----------
    R_basis : Basis
        Fourier-Zernike basis for R
    Z_basis : Basis
        Fourier-Zernike basis for Z
    L_basis : Basis
        Fourier-Zernike basis for lambda
    Rb_basis : Basis
        Double Fourier basis for boundary R
    Zb_basis : Basis
        Double Fourier basis for boundary Z
    Rb_lmn : ndarray
        Array of spectral coefficients for boundary R
    Zb_lmn : ndarray
        Array of spectral coefficients for boundary Z

    Returns
    -------
    A, b : ndarray
        Constraint matrix and vector such that the constraint is satisfied
        if and only if Ax=b

    """
    assert (
        len(Rb_lmn) == Rb_basis.num_modes
    ), f"Mismatch between number of boundary modes {Rb_basis.num_modes} and number of boundary coefficients {len(Rb_lmn)}"
    assert (
        len(Zb_lmn) == Zb_basis.num_modes
    ), f"Mismatch between number of boundary modes {Zb_basis.num_modes} and number of boundary coefficients {len(Zb_lmn)}"
    assert R_basis.L <= Rb_basis.L, "Boundary does not fully constraint R_basis"
    assert Z_basis.M <= Zb_basis.M, "Boundary does not fully constraint Z_basis"

    dim_R = R_basis.num_modes
    dim_Z = Z_basis.num_modes
    dim_Rb = Rb_basis.modes.shape[0]
    dim_Zb = Zb_basis.modes.shape[0]

    AR = np.zeros((dim_Rb, dim_R))
    AZ = np.zeros((dim_Zb, dim_Z))

    for i, (l, m, n) in enumerate(R_basis.modes):
        j = np.where(
            np.logical_and(
                (Rb_basis.modes[:, :2] == [l, m]).all(axis=1),
                Rb_basis.modes[:, -1] >= 0,
            )
        )[0]
        AR[j, i] = 1

    for i, (l, m, n) in enumerate(Z_basis.modes):
        j = np.where(
            np.logical_and(
                (Zb_basis.modes[:, :2] == [l, m]).all(axis=1),
                Zb_basis.modes[:, -1] >= 0,
            )
        )[0]
        AZ[j, i] = 1

    A = np.block([[AR, np.zeros((dim_Rb, dim_Z))], [np.zeros((dim_Zb, dim_R)), AZ]])
    b = np.concatenate([Rb_lmn, Zb_lmn])

    return A, b


def _get_axis_bc(R_basis, Z_basis, L_basis):
    """Compute constraint matrices for the magnetic axis.

    lambda(0,theta,zeta) == 0

    Parameters
    ----------
    R_basis : Basis
        Fourier-Zernike basis for R
    Z_basis : Basis
        Fourier-Zernike basis for Z
    L_basis : Basis
        Fourier-Zernike basis for lambda

    Returns
    -------
    A, b : ndarray
        Constraint matrix and vector such that the constraint is satisfied
        if and only if Ax=b

    """
    dim_R = R_basis.num_modes
    dim_Z = Z_basis.num_modes
    dim_L = L_basis.num_modes

    dimx = dim_R + dim_Z + dim_L

    N = L_basis.N
    ns = np.arange(-N, N + 1)
    A = np.zeros((len(ns), dimx))
    b = np.zeros((len(ns)))

    # l(0,t,z) = 0
    lmn = L_basis.modes
    for i, (l, m, n) in enumerate(lmn):
        if m != 0:
            continue
        if (l // 2) % 2 == 0:
            j = np.argwhere(n == ns)
            A[j, i + dim_R + dim_Z] = 1
        else:
            j = np.argwhere(n == ns)
            A[j, i + dim_R + dim_Z] = -1

    return A, b


def _get_sfl_bc(R_basis, Z_basis, L_basis):
    """Compute constraint matrices for straigh field-line coordinates.

    enforces lambda(rho,theta,zeta) = 0

    Parameters
    ----------
    R_basis : Basis
        Fourier-Zernike basis for R
    Z_basis : Basis
        Fourier-Zernike basis for Z
    L_basis : Basis
        Fourier-Zernike basis for lambda

    Returns
    -------
    A, b : ndarray
        Constraint matrix and vector such that the constraint is satisfied
        if and only if Ax=b

    """
    dim_R = R_basis.num_modes
    dim_Z = Z_basis.num_modes
    dim_L = L_basis.num_modes

    A = np.hstack(
        (np.zeros((dim_L, dim_R)), np.zeros((dim_L, dim_Z)), np.identity(dim_L))
    )
    b = np.zeros((dim_L,))

    return A, b


def _get_gauge_bc(R_basis, Z_basis, L_basis):
    """Compute constraint matrices for gauge freedom of lambda.

    enforces lambda(rho,0,0) == 0

    Parameters
    ----------
    R_basis : Basis
        Fourier-Zernike basis for R
    Z_basis : Basis
        Fourier-Zernike basis for Z
    L_basis : Basis
        Fourier-Zernike basis for lambda

    Returns
    -------
    A, b : ndarray
        Constraint matrix and vector such that the constraint is satisfied
        if and only if Ax=b

    """
    dim_R = R_basis.num_modes
    dim_Z = Z_basis.num_modes
    dim_L = L_basis.num_modes

    dimx = dim_R + dim_Z + dim_L

    L_modes = L_basis.modes
    mnpos = np.where((L_modes[:, 1:] >= [0, 0]).all(axis=1))[0]
    l_lmn = L_modes[mnpos, :]
    if len(l_lmn) > 0:
        c = zernike_radial_coeffs(l_lmn[:, 0], l_lmn[:, 1])
    else:
        c = np.zeros((0, 0))

    A = np.zeros((c.shape[1], dimx))
    A[:, mnpos + dim_R + dim_Z] = c.T
    b = np.zeros((c.shape[1]))

    return A, b


def _get_umbilic_bc(R_basis, Z_basis):
    """Compute constraint matrices for umbilic rotation.

    enforces:
    R(rho,theta,zeta) = a(rho)*cos(m*theta)*cos(zeta) + b(rho)*cos(2*m*theta)*cos(zeta)
                      - a(rho)*sin(m*theta)*sin(zeta) + b(rho)*sin(2*m*theta)*sin(zeta)
    Z(rho,theta,zeta) = a(rho)*cos(m*theta)*sin(zeta) + b(rho)*cos(2*m*theta)*sin(zeta)
                      + a(rho)*sin(m*theta)*cos(zeta) - b(rho)*sin(2*m*theta)*cos(zeta)

    Parameters
    ----------
    R_basis : Basis
        Fourier-Zernike basis for R
    Z_basis : Basis
        Fourier-Zernike basis for Z

    Returns
    -------
    A, b : ndarray
        Constraint matrix and vector such that the constraint is satisfied
        if and only if Ax=b

    """
    dim_R = R_basis.num_modes
    dim_Z = Z_basis.num_modes
    dim_x = dim_R + dim_Z
    dim_y = np.sum(np.nonzero(R_basis.modes[:, 2] == 0))

    R_modes = R_basis.modes
    Z_modes = Z_basis.modes

    AR = np.zeros((dim_R - dim_y, dim_x))
    AZ = np.zeros((dim_Z, dim_x))

    i = 0
    for l, m, n in enumerate(R_modes):
        j = np.where((R_modes == [int(l), int(m), int(n)]).all(axis=1))[0]
        if m != 0 and n == 0:  # no n=0 modes besides m=0
            AR[i, j] = 1
            i += 1
        elif n > 0:  # symmetry between (m,n) and (-m,-n)
            k = np.where((R_modes == [int(l), -int(m), -int(n)]).all(axis=1))[0]
            AR[i, j] = 1
            AR[i, k] = 2 * (int(m) % 2) + 1  # even m = -1, odd m = +1
            i += 1

    for i, (l, m, n) in enumerate(Z_modes):
        j = np.where((Z_modes == [int(l), int(m), int(n)]).all(axis=1))[0]
        AZ[i, dim_R + j] = 1  # no n=0 modes
        if n != 0:  # symmetry between R and Z coefficients
            k = np.where((R_modes == [int(l), int(m), -int(n)]).all(axis=1))[0]
            if m > 0:
                AZ[i, k] = -1
            elif m < 0:
                AZ[i, k] = -2 * (int(m) % 2) + 1  # even m = +1, odd m = -1

    A = np.vstack((AR, AZ))
    b = np.zeros((dim_x - dim_y,))

    return A, b


class RadialConstraint:
    """Penalty term to enforce nested flux surfaces.

    Penalizes the spacing between flux surfaces to ensure that r is a positive monotonic
    function of rho.

    Parameters
    ----------
    r_basis : Basis
        spectral basis for r
    L : integer, optional
        how many flux surfaces to test (default 10)
    M : integer, optional
        how many theta angles to test (default 2*basis.M + 1)
    N : integer, optional
        how many poincare sections to test (default 2*basis.N + 1)
    a : float, optional
        strength of softmin function. (default 10*L)
    scalar : bool, optional
        whether to compute a scalar or vector loss
    """

    def __init__(self, R_basis, Z_basis, L=10, M=None, N=None, a=None, scalar=False):

        self._R_basis = R_basis
        self._Z_basis = Z_basis
        self._L = L
        self._M = M if M is not None else 2 * R_basis.M + 1
        self._N = M if M is not None else 2 * R_basis.N + 1
        self._a = a or self._L * 10
        self._scalar = scalar
        self._grid = LinearGrid(L=self._L, M=self._M, N=self._N, axis=True)
        self._R_transform = Transform(self._grid, self._R_basis)
        self._Z_transform = Transform(self._grid, self._Z_basis)

    def compute(self, x, *args):
        R_lmn, Z_lmn, L_lmn = unpack_state(
            x, self._R_transform.basis.num_modes, self._Z_transform.basis.num_modes
        )
        R = self._R_transform.transform(R_lmn)
        Z = self._R_transform.transform(Z_lmn)
        R0 = jnp.mean(R[self._grid.axis])
        Z0 = jnp.mean(Z[self._grid.axis])
        r2 = (R - R0) ** 2 + (Z - Z0) ** 2
        dr = jnp.diff(
            r2.reshape((self._L, self._M, self._N), order="F"), axis=0
        ).flatten()

        if self._scalar:
            # apply softmin
            num = jnp.sum(dr * jnp.exp(-self._a * dr))
            den = jnp.sum(jnp.exp(-self._a * dr))
            return num / den
        return -jnp.log(dr) / self._a
