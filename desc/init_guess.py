import numpy as np

from desc.backend import put
from desc.basis import FourierZernikeBasis


def get_initial_guess_scale_bdry(axis, bdry, bdry_ratio,
                 R_basis:FourierZernikeBasis, Z_basis:FourierZernikeBasis):
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
    R_basis : FourierZernikeBasis
        DESCRIPTION
    Z_basis : FourierZernikeBasis
        DESCRIPTION

    Returns
    -------
    cR : ndarray, shape(N_coeffs,)
        Fourier-Zernike coefficients for R, following indexing given in zern_idx
    cZ : ndarray, shape(N_coeffs,)
        Fourier-Zernike coefficients for Z, following indexing given in zern_idx

    """
    modes_R = R_basis.modes
    modes_Z = Z_basis.modes

    cR = np.zeros((R_basis.num_modes,))
    cZ = np.zeros((Z_basis.num_modes,))

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
                (modes_R[:, 0] == 0, modes_R[:, 1] == 0, modes_R[:, 2] == n)))[0], (bR+aR)/2)
            cZ = put(cZ, np.where(np.logical_and.reduce(
                (modes_Z[:, 0] == 0, modes_Z[:, 1] == 0, modes_Z[:, 2] == n)))[0], (bZ+aZ)/2)
            cR = put(cR, np.where(np.logical_and.reduce(
                (modes_R[:, 0] == 2, modes_R[:, 1] == 0, modes_R[:, 2] == n)))[0], (bR-aR)/2)
            cZ = put(cZ, np.where(np.logical_and.reduce(
                (modes_Z[:, 0] == 2, modes_Z[:, 1] == 0, modes_Z[:, 2] == n)))[0], (bZ-aZ)/2)

        else:
            cR = put(cR, np.where(np.logical_and.reduce((modes_R[:, 0] == np.absolute(
                m), modes_R[:, 1] == m, modes_R[:, 2] == n)))[0], bR)
            cZ = put(cZ, np.where(np.logical_and.reduce((modes_Z[:, 0] == np.absolute(
                    m), modes_Z[:, 1] == m, modes_Z[:, 2] == n)))[0], bZ)

    return cR, cZ
