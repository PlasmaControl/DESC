import numpy as np

from desc.backend import jnp, put
from desc.grid import Grid
from desc.basis import DoubleFourierSeries
from desc.transform import Transform


def format_bdry(bdry, basis:DoubleFourierSeries, mode:str='spectral'):
    """Formats arrays for boundary conditions and converts between
    real space and fourier representations

    Parameters
    ----------
    bdry : ndarray, shape(Nbdry,4)
        array of fourier coeffs [m,n,Rcoeff, Zcoeff]
        or array of real space coordinates, [theta,phi,R,Z]
    basis : DoubleFourierSeries
        spectral basis for boundary coefficients
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
        transf = Transform(grid, basis)

        # fit real data to spectral coefficients
        cRb = transf.fit(bdry[:, 2])
        cZb = transf.fit(bdry[:, 3])

    else:
        bdryR = np.zeros((basis.num_modes,))
        bdryZ = np.zeros((basis.num_modes,))

        for m, n, bR, bZ in bdry:
            idx = np.where(np.logical_and(basis.modes[:, 1] == int(m),
                                          basis.modes[:, 2] == int(n)))[0]
            cRb = put(bdryR, idx, bR)
            cZb = put(bdryZ, idx, bZ)

    return cRb, cZb


# XXX: Note that this method cannot be improved with FFT due to non-uniform grid
def compute_bdry_err(cR, cZ, cL, cRb, cZb, RZb_transform, L_transform, bdry_ratio):
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
    RZb_transform : Transform
        transforms cR and cZ to physical space at the boundary
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

    # build fitting matrix
    nodes = jnp.array([rho, theta, phi]).T
    A = L_transform.basis.evaluate(nodes)
    pinv = jnp.linalg.pinv(A, rcond=1e-6)

    # LCFS transform and fit
    R = RZb_transform.transform(cR)
    Z = RZb_transform.transform(cZ)
    cR_lcfs = jnp.matmul(pinv, R)
    cZ_lcfs = jnp.matmul(pinv, Z)

    # ratio of non-axisymmetric boundary modes to use
    ratio = jnp.where(L_transform.basis.modes[:, 2] != 0, bdry_ratio, 1)

    # compute errors
    errR = cR_lcfs - cRb*ratio
    errZ = cZ_lcfs - cZb*ratio
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
