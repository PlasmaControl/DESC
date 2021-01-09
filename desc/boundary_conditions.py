import numpy as np

from desc.backend import jnp
from desc.basis import jacobi_coeffs
from desc.optimize.constraint import LinearEqualityConstraint


class BoundaryConstraint(LinearEqualityConstraint):
    """Linear equality constraint for boundary conditions and gauge freedom

    enforces:
    r(0,theta,zeta) == 0
    lambda(0,theta,zeta) == 0
    R0 + r(1,theta,zeta)*cos(theta) == R1
    Z0 + r(1,theta,zeta)*sin(theta) == Z1
    lambda(rho,0,0) == 0

    Parameters
    ----------
    R0_basis : Basis
        Fourier basis for R0
    Z0_basis : Basis
        Fourier basis for Z0
    r_basis : Basis
        Fourier-Zernike basis for r
    l_basis : Basis
        Fourier-Zernike basis for lambda
    R1_basis : Basis
        Double Fourier basis for boundary R
    Z1_basis : Basis
        Double Fourier basis for boundary Z
    R1_mn : ndarray
        Array of spectral coefficients for boundary R
    Z1_mn : ndarray
        Array of spectral coefficients for boundary Z
    x0 : ndarray, optional
        particular solution for Ax=b. If not supplied it will
        be calculated using the least norm solution of the constraint.

    """

    def __init__(self, R0_basis, Z0_basis, r_basis, l_basis, x0=None):

        Aaxis, baxis = get_axis_bc_matrices(
            R0_basis, Z0_basis, r_basis, l_basis)
        Alcfs, blcfs = get_lcfs_bc_matrices(
            R0_basis, Z0_basis, r_basis, l_basis)
        Agauge, bgauge = get_gauge_bc_matrices(
            R0_basis, Z0_basis, r_basis, l_basis)

        A = np.vstack([Aaxis, Alcfs, Agauge])
        b = np.concatenate([baxis, blcfs, bgauge])

        self._Aaxis = Aaxis
        self._Alcfs = Alcfs
        self._Agauge = Agauge

        self._baxis = baxis
        self._blcfs = blcfs
        self._bgauge = bgauge

        super().__init__(A, b)


def get_gauge_bc_matrices(R0_basis, Z0_basis, r_basis, l_basis):
    """Compute constraint matrices for gauge freedom of lambda

    enforces lambda(rho,0,0) == 0

    Parameters
    ----------
    R0_basis : Basis
        Fourier basis for R0
    Z0_basis : Basis
        Fourier basis for Z0
    r_basis : Basis
        Fourier-Zernike basis for r
    l_basis : Basis
        Fourier-Zernike basis for lambda

    Returns
    -------
    A, b : ndarray
        Constraint matrix and vector such that the constraint is satisfied
        if and only if Ax=b
    """
    R0_modes = R0_basis.modes
    Z0_modes = Z0_basis.modes
    r_modes = r_basis.modes
    l_modes = l_basis.modes
    dim_R0 = len(R0_modes)
    dim_Z0 = len(Z0_modes)
    dim_r = len(r_modes)
    dim_l = len(l_modes)
    dimx = dim_R0 + dim_Z0 + dim_r + dim_l

    mnpos = np.where(np.logical_and(l_modes[:, 1] >= 0, l_modes[:, 2] >= 0))[0]
    l_lmn = l_modes[mnpos, :]
    if len(l_lmn) > 0:
        c = jacobi_coeffs(l_lmn[:, 0], l_lmn[:, 1])
    else:
        c = np.zeros((0, 0))

    A = np.zeros((c.shape[1], dimx))
    A[:, mnpos+dim_R0 + dim_Z0 + dim_r] = c.T
    b = np.zeros((c.shape[1]))

    return A, b


def get_lcfs_bc_matrices(R0_basis, Z0_basis, r_basis, l_basis):
    """Compute constraint matrices for the shape of the last closed flux surface.

    enforces r(1,theta,zeta) == 1

    Parameters
    ----------
    R0_basis : Basis
        Fourier basis for R0
    Z0_basis : Basis
        Fourier basis for Z0
    r_basis : Basis
        Fourier-Zernike basis for r
    l_basis : Basis
        Fourier-Zernike basis for lambda

    Returns
    -------
    A, b : ndarray
        Constraint matrix and vector such that the constraint is satisfied
        if and only if Ax=b
    """

    R0_modes = R0_basis.modes
    Z0_modes = Z0_basis.modes
    r_modes = r_basis.modes
    l_modes = l_basis.modes

    dim_R0 = len(R0_modes)
    dim_Z0 = len(Z0_modes)
    dim_r = len(r_modes)
    dim_l = len(l_modes)
    dimx = dim_R0 + dim_Z0 + dim_r + dim_l

    MN = np.unique(r_modes[:, 1:], axis=0)
    numMN = len(MN)

    A = np.zeros((numMN, dimx))
    b = np.zeros((numMN))

    for i, (m, n) in enumerate(MN):
        j = np.argwhere(np.logical_and(r_modes[:, 1] == m, r_modes[:, 2] == n))
        A[i, dim_R0 + dim_Z0 + j] = 1
        if m == 0 and n == 0:
            b[i] = 1
        else:
            b[i] = 0

    return A, b


def get_axis_bc_matrices(R0_basis, Z0_basis, r_basis, l_basis):
    """Compute constraint matrices for the magnetic axis

    enforces r(0,theta,zeta) == 0 and lambda(0,theta,zeta) == 0

    Parameters
    ----------
    R0_basis : Basis
        Fourier basis for R0
    Z0_basis : Basis
        Fourier basis for Z0
    r_basis : Basis
        Fourier-Zernike basis for r
    l_basis : Basis
        Fourier-Zernike basis for lambda

    Returns
    -------
    A, b : ndarray
        Constraint matrix and vector such that the constraint is satisfied
        if and only if Ax=b
    """

    dim_R0 = len(R0_basis.modes)
    dim_Z0 = len(Z0_basis.modes)
    dim_r = len(r_basis.modes)
    dim_l = len(l_basis.modes)
    dimx = dim_R0 + dim_Z0 + dim_r + dim_l

    N = max(r_basis.N, l_basis.N)
    ns = np.arange(-N, N+1)
    A = np.zeros((len(ns)*2, dimx))
    b = np.zeros((len(ns)*2))

    # r(0,t,z) = 0
    lmn = r_basis.modes
    for i, (l, m, n) in enumerate(lmn):
        if m != 0:
            continue
        if (l//2) % 2 == 0:
            j = np.argwhere(n == ns)
            A[j, i + dim_R0 + dim_Z0] = 1
        else:
            j = np.argwhere(n == ns)
            A[j, i + dim_R0 + dim_Z0] = -1

    # l(0,t,z) = 0
    lmn = l_basis.modes
    for i, (l, m, n) in enumerate(lmn):
        if m != 0:
            continue
        if (l//2) % 2 == 0:
            j = np.argwhere(n == ns) + len(ns)
            A[j, i + dim_R0 + dim_Z0 + dim_r] = 1
        else:
            j = np.argwhere(n == ns) + len(ns)
            A[j, i + dim_R0 + dim_Z0 + dim_r] = -1

    return A, b
