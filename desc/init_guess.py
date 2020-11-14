import numpy as np
from desc.backend import put, TextColors


def get_initial_guess_scale_bdry(axis, bdry, bdry_ratio, zern_idx, NFP, mode='spectral', rcond=1e-6):
    """Generate initial guess by scaling boundary shape

    Parameters
    ----------
    axis : ndarray, shape(Naxis,3)
        array of axis Fourier coeffs [n,Rcoeff, Zcoeff]
    bdry : ndarray, shape(Nbdry,4)
        array of boundary Fourier coeffs [m,n,Rcoeff, Zcoeff]
        OR
        array of real space coordinates, [theta,phi,R,Z]
    bdry_ratio : float
        fraction in range [0,1] of the full non-axisymmetric boundary to use
    zern_idx : ndarray, shape(N_coeffs,3)
        indices for spectral basis, ie an array of [l,m,n] for each spectral coefficient
    NFP : int
        number of field periods
    mode : str
        one of 'real', 'spectral' - which format is being used for bdryR,bdryZ,poloidal,toroidal (Default value = 'spectral')
    rcond : float
         relative limit on singular values for least squares fit to Zernike basis (Default value = 1e-6)

    Returns
    -------
    cR : ndarray, shape(N_coeffs,)
        Fourier-Zernike coefficients for R, following indexing given in zern_idx
    cZ : ndarray, shape(N_coeffs,)
        Fourier-Zernike coefficients for Z, following indexing given in zern_idx

    """

    if mode == 'spectral':
        dimZern = np.shape(zern_idx)[0]
        cR = np.zeros((dimZern,))
        cZ = np.zeros((dimZern,))

        for m, n, bR, bZ in bdry:

            bR *= np.clip(bdry_ratio+(n == 0), 0, 1)
            bZ *= np.clip(bdry_ratio+(n == 0), 0, 1)

            if m == 0:

                idx = np.where(axis[:, 0] == n)
                if idx[0].size == 0:
                    aR = bR
                    aZ = bZ
                else:
                    aR = axis[idx, 1][0, 0]
                    aZ = axis[idx, 2][0, 0]

                cR = put(cR, np.where(np.logical_and.reduce(
                    (zern_idx[:, 0] == 0, zern_idx[:, 1] == 0, zern_idx[:, 2] == n)))[0], (bR+aR)/2)
                cZ = put(cZ, np.where(np.logical_and.reduce(
                    (zern_idx[:, 0] == 0, zern_idx[:, 1] == 0, zern_idx[:, 2] == n)))[0], (bZ+aZ)/2)
                cR = put(cR, np.where(np.logical_and.reduce(
                    (zern_idx[:, 0] == 2, zern_idx[:, 1] == 0, zern_idx[:, 2] == n)))[0], (bR-aR)/2)
                cZ = put(cZ, np.where(np.logical_and.reduce(
                    (zern_idx[:, 0] == 2, zern_idx[:, 1] == 0, zern_idx[:, 2] == n)))[0], (bZ-aZ)/2)

            else:
                cR = put(cR, np.where(np.logical_and.reduce((zern_idx[:, 0] == np.absolute(
                    m), zern_idx[:, 1] == m, zern_idx[:, 2] == n)))[0], bR)
                cZ = put(cZ, np.where(np.logical_and.reduce((zern_idx[:, 0] == np.absolute(
                    m), zern_idx[:, 1] == m, zern_idx[:, 2] == n)))[0], bZ)

    else:
        raise ValueError(
            TextColors.FAIL + "Can't compute the initial guess in real space" + TextColors.ENDC)

    return cR, cZ
