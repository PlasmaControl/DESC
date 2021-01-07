import numpy as np

from desc.backend import jnp
from desc.utils import Tristate
from desc.grid import Grid
from desc.basis import DoubleFourierSeries, jacobi_coeffs
from desc.transform import Transform
from desc.optimize.constraint import LinearEqualityConstraint


class BoundaryConstraint(LinearEqualityConstraint):
    """Linear equality constraint for boundary conditions and gauge freedom

    enforces:
    r(0,theta,zeta) == 0
    lambda(0,theta,zeta) == 0
    R0 + r(1,theta,zeta)*cos(theta) == Rb
    Z0 + r(1,theta,zeta)*sin(theta) == Zb
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
    Rb_basis : Basis
        Double Fourier basis for boundary R
    Zb_basis : Basis
        Double Fourier basis for boundary Z
    R1_mn : ndarray
        Array of spectral coefficients for boundary R
    Z1_mn : ndarray
        Array of spectral coefficients for boundary Z
    x0 : ndarray, optional
        particular solution for Ax=b. If not supplied it will
        be calculated using the least norm solution of the constraint.

    """

    def __init__(self, R0_basis, Z0_basis, r_basis, l_basis,
                 Rb_basis, Zb_basis, R1_mn, Z1_mn, x0=None):

        Aaxis, baxis = get_axis_bc_matrices(
            R0_basis, Z0_basis, r_basis, l_basis)
        Alcfs, blcfs = get_lcfs_bc_matrices(
            R0_basis, Z0_basis, r_basis, l_basis, Rb_basis, Zb_basis, R1_mn, Z1_mn)
        Agauge, bgauge = get_gauge_bc_matrices(
            R0_basis, Z0_basis, r_basis, l_basis)

        A = np.vstack([Aaxis, Alcfs, Agauge])
        b = np.vstack([baxis, blcfs, bgauge])

        self._Aaxis = Aaxis
        self._Alcfs = Alcfs
        self._Agauge = Agauge

        self._baxis = baxis
        self._blcfs = blcfs
        self._bgauge = bgauge

        super().__init__(A, b)

    def get_b(self, R1_mn, Z1_mn):

        z1 = jnp.zeros(self._baxis.size)
        z2 = jnp.zeros(self._bgauge.size)

        b = jnp.concatenate([z1, R1_mn.flatten(), Z1_mn.flatten(), z2])
        return b


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
    b = np.zeros((c.shape[1], 1))

    return A, b


def get_lcfs_bc_matrices(R0_basis, Z0_basis, r_basis, l_basis,
                         Rb_basis, Zb_basis, R1_mn, Z1_mn):
    """Compute constraint matrices for the shape of the last closed flux surface.

    enforces R0 + r(1,theta,zeta)*cos(theta) == Rb and
    Z0 + r(1,theta,zeta)*sin(theta) == Zb

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
    Rb_basis : Basis
        Double Fourier basis for boundary R
    Zb_basis : Basis
        Double Fourier basis for boundary Z
    R1_mn : ndarray
        Array of spectral coefficients for boundary R
    Z1_mn : ndarray
        Array of spectral coefficients for boundary Z

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
    Rb_modes = Rb_basis.modes
    Zb_modes = Zb_basis.modes

    dim_R0 = len(R0_modes)
    dim_Z0 = len(Z0_modes)
    dim_r = len(r_modes)
    dim_l = len(l_modes)
    dimx = dim_R0 + dim_Z0 + dim_r + dim_l
    dim_Rb = len(Rb_modes)
    dim_Zb = len(Zb_modes)

    AR = np.zeros((dim_Rb, dimx))
    AZ = np.zeros((dim_Zb, dimx))
    bR = R1_mn.reshape((-1, 1))
    bZ = Z1_mn.reshape((-1, 1))

    for i, (l, m, n) in enumerate(Rb_modes):
        if m == 0:
            j = np.argwhere(R0_modes == n)
            AR[i, j] = 1
            j = np.argwhere(np.logical_and(
                r_modes[:, 1] == 1, r_modes[:, 2] == n)) + dim_R0 + dim_Z0
            AR[i, j] = 1/2
        elif abs(m) == 1:
            j = np.argwhere(np.logical_and(
                r_modes[:, 1] == 0, r_modes[:, 2] == n)) + dim_R0 + dim_Z0
            AR[i, j] = 1
            j = np.argwhere(np.logical_and(
                r_modes[:, 1] == 1, r_modes[:, 2] == n)) + dim_R0 + dim_Z0
            AR[i, j] = 1/2
        else:
            j = np.argwhere(np.logical_and(
                r_modes[:, 1] == m+1, r_modes[:, 2] == n)) + dim_R0 + dim_Z0
            AR[i, j] = 1/2
            j = np.argwhere(np.logical_and(
                r_modes[:, 1] == m-1, r_modes[:, 2] == n)) + dim_R0 + dim_Z0
            AR[i, j] = -1/2

    for i, (l, m, n) in enumerate(Zb_modes):
        if m == 0:
            j = np.argwhere(Z0_modes == n) + dim_R0
            AZ[i, j] = 1
            j = np.argwhere(np.logical_and(
                r_modes[:, 1] == -1, r_modes[:, 2] == n)) + dim_R0 + dim_Z0
            AZ[i, j] = 1/2
        elif abs(m) == 1:
            j = np.argwhere(np.logical_and(
                r_modes[:, 1] == 0, r_modes[:, 2] == n)) + dim_R0 + dim_Z0
            AZ[i, j] = 1
            j = np.argwhere(np.logical_and(
                r_modes[:, 1] == 1, r_modes[:, 2] == n)) + dim_R0 + dim_Z0
            AZ[i, j] = -1/2
        else:
            j = np.argwhere(np.logical_and(
                r_modes[:, 1] == -m-1, r_modes[:, 2] == n)) + dim_R0 + dim_Z0
            AZ[i, j] = 1/2
            j = np.argwhere(np.logical_and(
                r_modes[:, 1] == -m+1, r_modes[:, 2] == n)) + dim_R0 + dim_Z0
            AZ[i, j] = -1/2

    A = np.vstack([AR, AZ])
    b = np.vstack([bR, bZ])
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
    b = np.zeros((len(ns)*2, 1))

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


# XXX: Note that this method cannot be improved with FFT due to non-uniform grid
def compute_bdry_err(cR, cZ, cL, cRb, cZb, R1_transform, Z1_transform, L_transform, bdry_ratio):
    """Compute boundary error in (theta,phi) Fourier coefficients from non-uniform interpolation grid

    Parameters
    ----------
    cR : ndarray, shape(RZ_transform.num_modes,)
        spectral coefficients of R
    cZ : ndarray, shape(RZ_transform.num_modes,)
        spectral coefficients of Z
    cL : ndarray, shape(L_transform.num_modes,)
        spectral coefficients of lambda
    cRb : ndarray, shape(bdry_basis.num_modes,)
        spectral coefficients of R boundary
    cZb : ndarray, shape(bdry_basis.num_modes,)
        spectral coefficients of Z boundary
    bdry_ratio : float
        fraction in range [0,1] of the full non-axisymmetric boundary to use
    R1_transform : Transform
        transforms cR to physical space at the boundary
    Z1_transform : Transform
        transforms cZ to physical space at the boundary
    L_transform : Transform
        transforms cL to physical space

    Returns
    -------
    errR : ndarray, shape(N_bdry_pts,)
        vector of R errors in boundary spectral coeffs
    errZ : ndarray, shape(N_bdry_pts,)
        vector of Z errors in boundary spectral coeffs

    """
    # coordinates
    rho = L_transform.grid.nodes[:, 0]
    vartheta = L_transform.grid.nodes[:, 1]
    zeta = L_transform.grid.nodes[:, 2]
    lamda = L_transform.transform(cL)
    theta = vartheta - lamda
    phi = zeta

    # cannot use Transform object with JAX
    nodes = jnp.array([rho, theta, phi]).T
    if L_transform.basis.sym == None:
        A = L_transform.basis.evaluate(nodes)
        pinv_R = jnp.linalg.pinv(A, rcond=1e-6)
        pinv_Z = pinv_R
        ratio_Rb = jnp.where(L_transform.basis.modes[:, 2] != 0, bdry_ratio, 1)
        ratio_Zb = ratio_Rb
    else:
        Rb_basis = DoubleFourierSeries(
            M=L_transform.basis.M, N=L_transform.basis.N,
            NFP=L_transform.basis.NFP, sym=Tristate(True))
        Zb_basis = DoubleFourierSeries(
            M=L_transform.basis.M, N=L_transform.basis.N,
            NFP=L_transform.basis.NFP, sym=Tristate(False))
        AR = Rb_basis.evaluate(nodes)
        AZ = Zb_basis.evaluate(nodes)
        pinv_R = jnp.linalg.pinv(AR, rcond=1e-6)
        pinv_Z = jnp.linalg.pinv(AZ, rcond=1e-6)
        ratio_Rb = jnp.where(Rb_basis.modes[:, 2] != 0, bdry_ratio, 1)
        ratio_Zb = jnp.where(Zb_basis.modes[:, 2] != 0, bdry_ratio, 1)

    # LCFS transform and fit
    R = R1_transform.transform(cR)
    Z = Z1_transform.transform(cZ)
    cR_lcfs = jnp.matmul(pinv_R, R)
    cZ_lcfs = jnp.matmul(pinv_Z, Z)

    # compute errors
    errR = cR_lcfs - cRb*ratio_Rb
    errZ = cZ_lcfs - cZb*ratio_Zb
    return errR, errZ


# FIXME: this method might not be stable, but could yield speed improvements
def compute_bdry_err_sfl(cR, cZ, cL, cRb, cZb, RZ_transform, L_transform, bdry_transform, bdry_ratio):
    """Compute boundary error in (theta,phi) Fourier coefficients from non-uniform interpolation grid

    Parameters
    ----------
    cR : ndarray, shape(RZ_transform.num_modes,)
        spectral coefficients of R
    cZ : ndarray, shape(RZ_transform.num_modes,)
        spectral coefficients of Z
    cL : ndarray, shape(L_transform.num_modes,)
        spectral coefficients of lambda
    cRb : ndarray, shape(bdry_basis.num_modes,)
        spectral coefficients of R boundary
    cZb : ndarray, shape(bdry_basis.num_modes,)
        spectral coefficients of Z boundary
    bdry_ratio : float
        fraction in range [0,1] of the full non-axisymmetric boundary to use
    RZ_transform : Transform
        transforms cR and cZ to physical space
    L_transform : Transform
        transforms cL to physical space
    bdry_transform : Transform
        transforms cRb and cZb to physical space

    Returns
    -------
    errR : ndarray, shape(N_bdry_pts,)
        vector of R errors in boundary spectral coeffs
    errZ : ndarray, shape(N_bdry_pts,)
        vector of Z errors in boundary spectral coeffs

    """
    # coordinates
    rho = L_transform.grid.nodes[:, 0]
    vartheta = L_transform.grid.nodes[:, 1]
    zeta = L_transform.grid.nodes[:, 2]
    lamda = L_transform.transform(cL)
    theta = vartheta - lamda
    phi = zeta

    # boundary transform
    nodes = np.array([rho, theta, phi]).T
    grid = Grid(nodes)
    transf = Transform(grid, bdry_transform.basis)

    # transform to real space and fit back to sfl spectral basis
    R = transf.transform(cRb)
    Z = transf.transform(cZb)
    cRb_sfl = bdry_transform.fit(R)
    cZb_sfl = bdry_transform.fit(Z)

    # compute errors
    errR = np.zeros_like(cRb_sfl)
    errZ = np.zeros_like(cZb_sfl)
    i = 0
    for l, m, n in bdry_transform.modes:
        idx = np.where(np.logical_and(
            RZ_transform.basis.modes[:, 1] == m,
            RZ_transform.basis.modes[:, 2] == n))[0]
        errR[i] = np.sum(cR[idx]) - cRb_sfl[i]
        errZ[i] = np.sum(cZ[idx]) - cZb_sfl[i]
        i += 1

    return errR, errZ


# XXX: this function is used in callback()
def compute_lambda_err(cL, L_basis: DoubleFourierSeries):
    """Computes the error in the constraint lambda(t=0, p=0) = 0

    Parameters
    ----------
    cL : ndarray, shape(L_basis.num_modes)
        lambda spectral coefficients
    L_basis : DoubleFourierSeries
        indices for lambda spectral basis, ie an array of [m,n] for each spectral coefficient

    Returns
    -------
    errL : float
        sum of cL_mn where m, n > 0

    """

    errL = jnp.sum(jnp.where(jnp.logical_and(L_basis.modes[:, 1] >= 0,
                                             L_basis.modes[:, 2] >= 0), cL, 0))
    return errL


# XXX: Where is this function used?
def get_lambda_constraint_matrix(zern_idx, lambda_idx):
    """Computes a linear constraint matrix to enforce vartheta = 0 at theta=0.
    We require sum(lambda_mn) = 0, expressed in matrix for as Cx = 0

    Parameters
    ----------
    zern_idx : ndarray, shape(N_coeffs,3)
        indices for R,Z spectral basis,
        ie an array of [l,m,n] for each spectral coefficient
    lambda_idx : ndarray, shape(Nlambda,2)
        indices for lambda spectral basis,
        ie an array of [m,n] for each spectral coefficient

    Returns
    -------
    C : ndarray, shape(2*N_coeffs + Nlambda,)
        linear constraint matrix,
        so ``np.matmul(C,x)`` is the error in the lambda constraint

    """

    # assumes x = [cR, cZ, cL]
    offset = 2*len(zern_idx)
    mn_pos = np.where(np.logical_and(
        lambda_idx[:, 0] >= 0, lambda_idx[:, 1] >= 0))[0]
    C = np.zeros(offset + len(lambda_idx))
    C[offset+mn_pos] = 1

    return C
