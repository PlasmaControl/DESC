import numpy as np

from desc.backend import jnp, put, Tristate
from desc.grid import Grid
from desc.basis import DoubleFourierSeries
from desc.transform import Transform


def format_bdry(bdry, Rb_basis:DoubleFourierSeries,
                Zb_basis:DoubleFourierSeries, mode:str='spectral'):
    """Formats arrays for boundary conditions and converts between
    real space and fourier representations

    Parameters
    ----------
    bdry : ndarray, shape(Nbdry,4)
        array of fourier coeffs [m,n,Rcoeff, Zcoeff]
        or array of real space coordinates, [theta,phi,R,Z]
    Rb_basis : DoubleFourierSeries
        spectral basis for R boundary coefficients
    Zb_basis : DoubleFourierSeries
        spectral basis for Z boundary coefficients
    mode : str
        one of 'real', 'spectral'. Whether bdry is specified in real or spectral space.

    Returns
    -------
    cRb : ndarray
        spectral coefficients for R boundary
    cZb : ndarray
        spectral coefficients for Z boundary

    """
    if mode == 'real':
        theta = bdry[:, 0]
        phi = bdry[:, 1]
        rho = np.ones_like(theta)

        nodes = np.array([rho, theta, phi]).T
        grid = Grid(nodes)
        Rb_transf = Transform(grid, Rb_basis)
        Zb_transf = Transform(grid, Zb_basis)

        # fit real data to spectral coefficients
        cRb = Rb_transf.fit(bdry[:, 2])
        cZb = Zb_transf.fit(bdry[:, 3])

    else:
        cRb = np.zeros((Rb_basis.num_modes,))
        cZb = np.zeros((Zb_basis.num_modes,))

        for m, n, bR, bZ in bdry:
            idx_R = np.where(np.logical_and(Rb_basis.modes[:, 1] == int(m),
                                            Rb_basis.modes[:, 2] == int(n)))[0]
            idx_Z = np.where(np.logical_and(Zb_basis.modes[:, 1] == int(m),
                                            Zb_basis.modes[:, 2] == int(n)))[0]
            cRb = put(cRb, idx_R, bR)
            cZb = put(cZb, idx_Z, bZ)

    return cRb, cZb


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
def compute_lambda_err(cL, L_basis:DoubleFourierSeries):
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
