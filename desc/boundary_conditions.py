import numpy as np
from desc.backend import jnp, put, sign
from desc.zernike import fourzern, double_fourier_basis, eval_double_fourier
from desc.nodes import get_nodes_surf


def format_bdry(M, N, NFP, bdry, in_mode, out_mode, ntheta=None, nphi=None):
    """Formats arrays for boundary conditions and converts between
    real space and fourier representations

    Parameters
    ----------
    M : int
        maximum poloidal resolution
    N : int
        maximum toroidal resolution
    NFP : int
        number of field periods
    bdry : ndarray, shape(Nbdry,4)
        array of fourier coeffs [m,n,Rcoeff, Zcoeff]
        or array of real space coordinates, [theta,phi,R,Z]
    in_mode : str
        one of 'real', 'spectral'. Whether bdry is specified in real space or fourier.
    out_mode : str
        one of 'real', 'spectral'. Whether output should be specified in real space or fourier.
    ntheta,nphi : int
        number of grid points to use in poloidal and toroidal directions.
        only used if in_mode = 'spectral' and out_mode = 'real'. Defaults to 4*M and 4*N respectively

    Returns
    -------
    bdry_poloidal : ndarray
        poloidal mode numbers OR poloidal angle variables.
    bdry_toroidal : ndarray
        toroidal mode numbers OR toroidal angle variables.
    bdryR : ndarray
        R coeffs, where bdryR[i] has m=bdry_poloidal[i], n=bdry_toroidal[i].
        or R values at bdry, where bdryR[i] is at theta = bdry_poloidal[i],
        phi = bdry_toroidal[i].
    bdryZ : ndarray
        Z coeffs, where bdryZ[i] has m=bdry_poloidal[i], n=bdry_toroidal[i].
        or R values at bdry, where bdryR[i] is at theta = bdry_poloidal[i],
        phi = bdry_toroidal[i].

    """

    if in_mode == 'real' and out_mode == 'real':
        # just need to unpack the array
        bdry_theta = bdry[:, 0]
        bdry_phi = bdry[:, 1]
        bdryR = bdry[:, 2]
        bdryZ = bdry[:, 3]
        return bdry_theta, bdry_phi, bdryR, bdryZ

    if in_mode == 'spectral' and out_mode == 'spectral':
        # basically going from a sparse matrix representation to dense
        bdryM = np.arange(-M, M+1)
        bdryN = np.arange(-N, N+1)
        bdryM, bdryN = np.meshgrid(bdryM, bdryN, indexing='ij')
        bdryM = bdryM.flatten()
        bdryN = bdryN.flatten()
        bdryR = np.zeros(len(bdryM), dtype=np.float64)
        bdryZ = np.zeros(len(bdryM), dtype=np.float64)

        for m, n, bR, bZ in bdry:
            bdryR = put(bdryR, np.where(np.logical_and(
                bdryM == int(m), bdryN == int(n)))[0], bR)
            bdryZ = put(bdryZ, np.where(np.logical_and(
                bdryM == int(m), bdryN == int(n)))[0], bZ)
        return bdryM, bdryN, bdryR, bdryZ

    if in_mode == 'spectral' and out_mode == 'real':
        # just evaulate fourier series at nodes
        ntheta = ntheta if ntheta else 4*M
        nphi = nphi if nphi else 4*N
        bdry_theta = np.linspace(0, 2*np.pi, ntheta)
        bdry_phi = np.linspace(0, 2*np.pi/NFP, nphi)
        bdryR = eval_double_fourier(
            bdry[:, 2], bdry[:, :2], NFP, bdry_theta, bdry_phi)
        bdryZ = eval_double_fourier(
            bdry[:, 3], bdry[:, :2], NFP, bdry_theta, bdry_phi)
        return bdry_theta, bdry_phi, bdryR, bdryZ

    if in_mode == 'real' and out_mode == 'spectral':
        # fit to fourier series
        bdry_theta = bdry[:, 0]
        bdry_phi = bdry[:, 1]
        bdryR = bdry[:, 2]
        bdryZ = bdry[:, 3]
        bdryM = np.arange(-M, M+1)
        bdryN = np.arange(-N, N+1)
        bdryM, bdryN = np.meshgrid(bdryM, bdryN, indexing='ij')
        bdryM = bdryM.flatten()
        bdryN = bdryN.flatten()
        interp = double_fourier_basis(bdry_theta, bdry_phi, bdryM, bdryN, NFP)
        bR, bZ = np.linalg.lstsq(interp, np.array([bdryR, bdryZ]).T)[0].T
        return bdryM, bdryN, bR, bZ


# TODO: this method is not stable, but could yield speed improvements
def compute_bdry_err_four_sfl(cR, cZ, cL, bdry_ratio, zern_idx, lambda_idx, bdryR, bdryZ, bdryM, bdryN, NFP, sample=1.5):
    """Compute boundary error in (vartheta,zeta) Fourier coefficients

    Parameters
    ----------
    cR : ndarray, shape(N_coeffs,)
        Fourier-Zernike coefficients of R
    cZ : ndarray, shape(N_coeffs,)
        Fourier-Zernike coefficients of Z
    cL : ndarray, shape(2M+1)*(2N+1)
        double Fourier coefficients of lambda
    bdry_ratio : float
        fraction in range [0,1] of the full non-axisymmetric boundary to use
    zern_idx : ndarray, shape(N_coeffs,3)
        indices for R,Z spectral basis, ie an
        array of [l,m,n] for each spectral coefficient
    lambda_idx : ndarray, shape(2M+1)*(2N+1)
        indices for lambda spectral basis,
        ie an array of [m,n] for each spectral coefficient
    bdryR : ndarray, shape(N_bdry_modes,)
        R coefficients of boundary shape
    bdryZ : ndarray, shape(N_bdry_modes,)
        Z coefficients of boundary shape
    bdryM : ndarray, shape(N_bdry_modes,)
        poloidal mode numbers
    bdryN : ndarray, shape(N_bdry_modes,)
        toroidal mode numbers
    NFP : int
        number of field periods
    sample : float
        sampling factor (eg, 1.0 would be no oversampling) (Default value = 1.5)

    Returns
    -------
    errR : ndarray, shape(N_bdry_pts,)
        vector of R errors in boundary spectral coeffs
    errZ : ndarray, shape(N_bdry_pts,)
        vector of Z errors in boundary spectral coeffs

    """

    # sfl grid
    M = np.ceil(sample*max(bdryM))
    N = np.ceil(sample*max(bdryN))
    nodes, vols = get_nodes_surf(M, N, NFP, surf=1.0)
    vartheta = nodes[1, :]
    zeta = nodes[2, :]

    # grid in boundary coordinates
    lamda = eval_double_fourier(cL, lambda_idx, NFP, vartheta, zeta)
    theta = vartheta - lamda
    phi = zeta

    # ratio of non-axisymmetric boundary modes to use
    ratio = jnp.where(bdryN != 0, bdry_ratio, 1)

    # interpolate boundary
    bdry_idx = np.stack([bdryM, bdryN], axis=1)
    R = eval_double_fourier(bdryR*ratio, bdry_idx, NFP, theta, phi)
    Z = eval_double_fourier(bdryZ*ratio, bdry_idx, NFP, theta, phi)

    # transform BC to sfl Fourier coefficients
    four_bdry_interp = double_fourier_basis(vartheta, zeta, bdryM, bdryN, NFP)
    four_bdry_interp_pinv = jnp.linalg.pinv(four_bdry_interp, rcond=1e-6)
    bR, bZ = jnp.matmul(four_bdry_interp_pinv, jnp.array([R, Z]).T).T

    # compute errors
    errR = np.zeros_like(bdryM)
    errZ = np.zeros_like(bdryN)
    for i in range(len(bdryM)):
        idx = np.where(np.logical_and(
            zern_idx[:, 1] == bdryM[i], zern_idx[:, 2] == bdryN[i]))[0]
        errR = jax.ops.index_update(
            errR, jax.ops.index[i], bR[i] - np.sum(cR[idx]))
        errZ = jax.ops.index_update(
            errZ, jax.ops.index[i], bZ[i] - np.sum(cR[idx]))

    return errR, errZ


# TODO: Note that this method cannot be improved with FFT due to non-uniform grid
# TODO: The SVD Fourier transform could be unstable if lambda has a large amplitude
def compute_bdry_err_four(cR, cZ, cL, bdry_ratio, zernike_transform, lambda_idx, bdryR, bdryZ, bdryM, bdryN, NFP, sym=False):
    """Compute boundary error in (theta,phi) Fourier coefficients from non-uniform interpolation grid

    Parameters
    ----------
    cR : ndarray, shape(N_coeffs,)
        Fourier-Zernike coefficients of R
    cZ : ndarray, shape(N_coeffs,)
        Fourier-Zernike coefficients of Z
    cL : ndarray, shape(2M+1)*(2N+1)
        double Fourier coefficients of lambda
    bdry_ratio : float
        fraction in range [0,1] of the full non-axisymmetric boundary to use
    zernike_transform : ZernikeTransform
        zernike transform object for evaluating spectral coefficients on bdry
    lambda_idx : ndarray, shape(2M+1)*(2N+1)
        indices for lambda spectral basis,
        ie an array of [m,n] for each spectral coefficient
    bdryR : ndarray, shape(N_bdry_modes,)
        R coefficients of boundary shape
    bdryZ : ndarray, shape(N_bdry_modes,)
        Z coefficients of boundary shape
    bdryM : ndarray, shape(N_bdry_modes,)
        poloidal mode numbers
    bdryN : ndarray, shape(N_bdry_modes,)
        toroidal mode numbers
    NFP : int
        number of field periods
    sym : bool
         whether to assume stellarator symmetry. (Default value = False)

    Returns
    -------
    errR : ndarray, shape(N_bdry_pts,)
        vector of R errors in boundary spectral coeffs
    errZ : ndarray, shape(N_bdry_pts,)
        vector of Z errors in boundary spectral coeffs

    """

    # get grid for bdry eval
    vartheta = zernike_transform.nodes[1]
    zeta = zernike_transform.nodes[2]
    lamda = eval_double_fourier(cL, lambda_idx, NFP, vartheta, zeta)
    theta = vartheta - lamda
    phi = zeta

    # find values of R,Z at pts specified
    R = zernike_transform.transform(cR, 0, 0, 0).flatten()
    Z = zernike_transform.transform(cZ, 0, 0, 0).flatten()

    # interpolate R,Z to fourier basis in non sfl coords
    if sym:
        cos_idx = jnp.where(sign(bdryM) == sign(bdryN), True, False)
        sin_idx = jnp.where(sign(bdryM) != sign(bdryN), True, False)
        four_basis_R = jnp.where(cos_idx, double_fourier_basis(
            theta, phi, bdryM, bdryN, NFP), 0)
        four_basis_Z = jnp.where(sin_idx, double_fourier_basis(
            theta, phi, bdryM, bdryN, NFP), 0)
        four_basis_R_pinv = jnp.linalg.pinv(four_basis_R, rcond=1e-6)
        four_basis_Z_pinv = jnp.linalg.pinv(four_basis_Z, rcond=1e-6)
        cRb = jnp.matmul(four_basis_R_pinv, R)
        cZb = jnp.matmul(four_basis_Z_pinv, Z)
    else:
        cos_idx = jnp.ones_like(bdryM).astype(bool)
        sin_idx = jnp.ones_like(bdryN).astype(bool)

        four_basis = double_fourier_basis(theta, phi, bdryM, bdryN, NFP)
        four_basis_pinv = jnp.linalg.pinv(four_basis, rcond=1e-6)
        cRb, cZb = jnp.matmul(four_basis_pinv, jnp.array([R, Z]).T).T

    # ratio of non-axisymmetric boundary modes to use
    ratio = jnp.where(bdryN != 0, bdry_ratio, 1)

    # compute errors
    errR = jnp.where(cos_idx, cRb - (bdryR*ratio), 0)
    errZ = jnp.where(sin_idx, cZb - (bdryZ*ratio), 0)
    return errR, errZ


# TODO: Note that this method is stable but requires expensive Zernike evaluations
def compute_bdry_err_four_slow(cR, cZ, cL, bdry_ratio, zern_idx, lambda_idx, bdryR, bdryZ, bdryM, bdryN, NFP, sample=1.5):
    """Compute boundary error in (theta,phi) Fourier coefficients from uniform interpolation grid

    Parameters
    ----------
    cR : ndarray, shape(N_coeffs,)
        Fourier-Zernike coefficients of R
    cZ : ndarray, shape(N_coeffs,)
        Fourier-Zernike coefficients of Z
    cL : ndarray, shape(2M+1)*(2N+1)
        double Fourier coefficients of lambda
    bdry_ratio : float
        fraction in range [0,1] of the full non-axisymmetric boundary to use
    zern_idx : ndarray, shape(N_coeffs,3)
        indices for R,Z spectral basis, ie an
        array of [l,m,n] for each spectral coefficient
    lambda_idx : ndarray, shape(2M+1)*(2N+1)
        indices for lambda spectral basis,
        ie an array of [m,n] for each spectral coefficient
    bdryR : ndarray, shape(N_bdry_modes,)
        R coefficients of boundary shape
    bdryZ : ndarray, shape(N_bdry_modes,)
        Z coefficients of boundary shape
    bdryM : ndarray, shape(N_bdry_modes,)
        poloidal mode numbers
    bdryN : ndarray, shape(N_bdry_modes,)
        toroidal mode numbers
    NFP : int
        number of field periods
    sample : float
        sampling factor (eg, 1.0 would be no oversampling) (Default value = 1.5)

    Returns
    -------
    errR : ndarray, shape(N_bdry_pts,)
        vector of R errors in boundary spectral coeffs
    errZ : ndarray, shape(N_bdry_pts,)
        vector of Z errors in boundary spectral coeffs

    """

    # get grid for bdry eval
    dimFourM = 2*np.ceil(sample*np.max(np.abs(bdryM)))+1
    dimFourN = 2*np.ceil(sample*np.max(np.abs(bdryN)))+1
    dt = 2*np.pi/dimFourM
    dp = 2*np.pi/(NFP*dimFourN)
    theta = np.arange(0, 2*jnp.pi, dt)
    phi = np.arange(0, 2*jnp.pi/NFP, dp)
    theta, phi = np.meshgrid(theta, phi, indexing='ij')
    theta = theta.flatten()
    phi = phi.flatten()

    # find values of R,Z at pts specified
    rho = np.ones_like(theta)
    lamda = eval_double_fourier(cL, lambda_idx, NFP, theta, phi)
    vartheta = theta + lamda
    zeta = phi
    zern_basis = jnp.hstack([fourzern(
        rho, vartheta, zeta, lmn[0], lmn[1], lmn[2], NFP, 0, 0, 0) for lmn in zern_idx])
    R = jnp.matmul(zern_basis, cR).flatten()
    Z = jnp.matmul(zern_basis, cZ).flatten()

    four_basis = double_fourier_basis(theta, phi, bdryM, bdryN, NFP)
    cRb, cZb = jnp.linalg.lstsq(
        four_basis, jnp.array([R, Z]).T, rcond=1e-6)[0].T

    # ratio of non-axisymmetric boundary modes to use
    ratio = jnp.where(bdryN != 0, bdry_ratio, 1)

    # compute errors
    errR = cRb - bdryR*ratio
    errZ = cZb - bdryZ*ratio
    return errR, errZ


def compute_bdry_err_RZ(cR, cZ, cL, bdry_ratio, zern_idx, lambda_idx, bdryR, bdryZ, bdry_theta, bdry_phi, NFP):
    """Compute boundary error at discrete points

    Parameters
    ----------
    cR : ndarray, shape(N_coeffs,)
        Fourier-Zernike coefficients of R
    cZ : ndarray, shape(N_coeffs,)
        Fourier-Zernike coefficients of Z
    cL : ndarray, shape(2M+1)*(2N+1)
        double Fourier coefficients of lambda
    bdry_ratio : float
        fraction in range [0,1] of the full non-axisymmetric boundary to use
    zern_idx : ndarray, shape(N_coeffs,3)
        indices for R,Z spectral basis,
        ie an array of [l,m,n] for each spectral coefficient
    lambda_idx : ndarray, shape(2M+1)*(2N+1)
        indices for lambda spectral basis,
        ie an array of [m,n] for each spectral coefficient
    bdryR : ndarray, shape(N_bdry_pts,)
        R values of boundary shape
    bdryZ : ndarray, shape(N_bdry_pts,)
        Z values of boundary shape
    bdry_theta : ndarray, shape(N_bdry_pts,)
        real space poloidal coordinates where boundary is specified
    bdry_phi : ndarray, shape(N_bdry_pts,)
        real space toroidal coordinates where boundary is specified
    NFP : int
        number of field periods

    Returns
    -------
    errR : ndarray, shape(N_bdry_pts,)
        vector of R errors in boundary position at specified points
    errZ : ndarray, shape(N_bdry_pts,)
        vector of Z errors in boundary position at specified points

    """

    # find values of R,Z at pts specified
    rho = jnp.ones_like(bdry_theta)
    lamda = eval_double_fourier(cL, lambda_idx, NFP, bdry_theta, bdry_phi)
    vartheta = bdry_theta + lamda
    zeta = bdry_phi
    zern_bdry_interp = jnp.stack([fourzern(
        rho, vartheta, zeta, lmn[0], lmn[1], lmn[2], NFP, 0, 0, 0) for lmn in zern_idx])
    R = jnp.matmul(zern_bdry_interp, cR).flatten()
    Z = jnp.matmul(zern_bdry_interp, cZ).flatten()

    # compute errors
    errR = R-bdryR
    errZ = Z-bdryZ

    return errR, errZ


def compute_lambda_err(cL, idx, NFP):
    """Compute the error in sum(lambda_mn) to enforce
    vartheta(0,0) = 0

    Parameters
    ----------
    cL : ndarray, shape(2M+1)*(2N+1)
        double Fourier coefficients of lambda
    idx : ndarray, shape(2M+1)*(2N+1)
        indices for lambda spectral basis, ie an array of [m,n] for each spectral coefficient
    NFP : int
        number of field periods

    Returns
    -------
    lambda_err : float
        sum of lambda_mn where m,n>0

    """

    Lc = jnp.where(jnp.logical_and(idx[:, 0] >= 0, idx[:, 1] >= 0), cL, 0)
    errL = jnp.sum(Lc)

    return errL


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
