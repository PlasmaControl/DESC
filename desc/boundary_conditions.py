import numpy as np

from desc.backend import jnp
from desc.basis import jacobi_coeffs
from desc.optimize.constraint import LinearEqualityConstraint


class BoundaryConstraint(LinearEqualityConstraint):
    """Linear equality constraint for boundary conditions and gauge freedom

    enforces:

    R(1,theta,zeta) = Rb(theta,zeta)

    Z(1,theta,zeta) = Zb(theta,zeta)

    lambda(0,theta,zeta) == 0

    lambda(rho,0,0) == 0


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
    Rb_mn : ndarray
        Array of spectral coefficients for boundary R
    Zb_mn : ndarray
        Array of spectral coefficients for boundary Z
    x0 : ndarray, optional
        particular solution for Ax=b. If not supplied it will
        be calculated using the least norm solution of the constraint.
    build : bool
        whether to compute null space and pseudoinverse now or wait until needed.
    """

    def __init__(
        self,
        R_basis,
        Z_basis,
        L_basis,
        Rb_basis,
        Zb_basis,
        Rb_mn,
        Zb_mn,
        x0=None,
        build=True,
    ):

        Alcfs, blcfs = _get_lcfs_bc_matrices(
            R_basis, Z_basis, L_basis, Rb_basis, Zb_basis, Rb_mn, Zb_mn
        )
        Aaxis, baxis = _get_axis_bc_matrices(R_basis, Z_basis, L_basis)
        Agauge, bgauge = _get_gauge_bc_matrices(R_basis, Z_basis, L_basis)

        A = np.vstack([Alcfs, Aaxis, Agauge])
        b = np.concatenate([blcfs, baxis, bgauge])

        self._Aaxis = Aaxis
        self._Alcfs = Alcfs
        self._Agauge = Agauge

        self._baxis = baxis
        self._blcfs = blcfs
        self._bgauge = bgauge

        super().__init__(A, b, x0, build)

    def recover_from_bdry(self, y, Rb_mn=None, Zb_mn=None):

        if Rb_mn is not None and Zb_mn is not None:
            b = self._get_b(Rb_mn, Zb_mn)
        else:
            b = self.b

        x0 = jnp.dot(self.Ainv, b)
        x = x0 + jnp.dot(self.Z, y)
        return x

    def _get_b(self, Rb_mn, Zb_mn):

        z1 = jnp.zeros(self._baxis.size)
        z2 = jnp.zeros(self._bgauge.size)

        b = jnp.concatenate([Rb_mn.flatten(), Zb_mn.flatten(), z1, z2])
        return b


def _get_gauge_bc_matrices(R_basis, Z_basis, L_basis):
    """Compute constraint matrices for gauge freedom of lambda

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
        c = jacobi_coeffs(l_lmn[:, 0], l_lmn[:, 1])
    else:
        c = np.zeros((0, 0))

    A = np.zeros((c.shape[1], dimx))
    A[:, mnpos + dim_R + dim_Z] = c.T
    b = np.zeros((c.shape[1]))

    return A, b


def _get_lcfs_bc_matrices(R_basis, Z_basis, L_basis, Rb_basis, Zb_basis, Rb_mn, Zb_mn):
    """Compute constraint matrices for the shape of the last closed flux surface.

    enforces r(1,theta,zeta) == 1

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
    Rb_mn : ndarray
        Array of spectral coefficients for boundary R
    Zb_mn : ndarray
        Array of spectral coefficients for boundary Z

    Returns
    -------
    A, b : ndarray
        Constraint matrix and vector such that the constraint is satisfied
        if and only if Ax=b
    """

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
    bR = Rb_mn
    bZ = Zb_mn

    for i, (l, m, n) in enumerate(Rb_modes):
        j = np.argwhere(np.logical_and(R_modes[:, 1] == m, R_modes[:, 2] == n))
        AR[i, j] = 1

    for i, (l, m, n) in enumerate(Zb_modes):
        j = np.argwhere(np.logical_and(Z_modes[:, 1] == m, Z_modes[:, 2] == n))
        AZ[i, dim_R + j] = 1

    A = np.vstack([AR, AZ])
    b = np.concatenate([bR, bZ])

    return A, b


def _get_axis_bc_matrices(R_basis, Z_basis, L_basis):
    """Compute constraint matrices for the magnetic axis

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
