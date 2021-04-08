import numpy as np

from desc.utils import sign
from desc.basis import jacobi


def ptolemy_identity_fwd(m_0, n_0, s, c):
    """Convert from double-angle to double-Fourier form using Ptolemy's identity.

    Converts from the double-angle form:
        s*sin(m*theta-n*phi) + c*cos(m*theta-n*phi)
    to a double Fourier series of the form:
        ss*sin(m*theta)*sin(n*phi) + sc*sin(m*theta)*cos(n*phi) +
        cs*cos(m*theta)*sin(n*phi) + cc*cos(m*theta)*cos(n*phi)
    using Ptolemy's sum and difference formulas.

    Parameters
    ----------
    m_0 : ndarray
        Poloidal mode numbers of the double-angle Fourier basis.
    n_0 : ndarray
        Toroidal mode numbers of the double-angle Fourier basis.
    s : ndarray, shape(surfs,num_modes), optional
        Coefficients of sin(m*theta-n*phi) terms.
        Each row is a separate flux surface.
    c : ndarray, shape(surfs,num_modes), optional
        Coefficients of cos(m*theta-n*phi) terms.
        Each row is a separate flux surface.

    Returns
    -------
    m_1 : ndarray, shape(num_modes,)
        Poloidal mode numbers of the double Fourier basis.
    n_1 : ndarray, shape(num_modes,)
        Toroidal mode numbers of the double Fourier basis.
    x : ndarray, shape(surfs,num_modes,)
        Spectral coefficients in the double Fourier basis.

    """
    s = np.atleast_2d(s)
    c = np.atleast_2d(c)

    M = int(np.max(np.abs(m_0)))
    N = int(np.max(np.abs(n_0)))

    mn_1 = np.array(
        [[m - M, n - N] for m in range(2 * M + 1) for n in range(2 * N + 1)]
    )
    m_1 = mn_1[:, 0]
    n_1 = mn_1[:, 1]
    x = np.zeros((s.shape[0], m_1.size))

    for k in range(len(m_0)):
        # sin(m*theta)*cos(n*phi)
        sin_mn_1 = np.where((mn_1 == [-np.abs(m_0[k]), np.abs(n_0[k])]).all(axis=1))[0][
            0
        ]
        # cos(m*theta)*sin(n*phi)
        sin_mn_2 = np.where((mn_1 == [np.abs(m_0[k]), -np.abs(n_0[k])]).all(axis=1))[0][
            0
        ]
        # cos(m*theta)*cos(n*phi)
        cos_mn_1 = np.where((mn_1 == [np.abs(m_0[k]), np.abs(n_0[k])]).all(axis=1))[0][
            0
        ]
        # sin(m*theta)*sin(n*phi)
        cos_mn_2 = np.where((mn_1 == [-np.abs(m_0[k]), -np.abs(n_0[k])]).all(axis=1))[
            0
        ][0]

        if np.sign(m_0[k]) != 0:
            x[:, sin_mn_1] += s[:, k]
        x[:, cos_mn_1] += c[:, k]
        if np.sign(n_0[k]) > 0:
            x[:, sin_mn_2] -= s[:, k]
            if np.sign(m_0[k]) != 0:
                x[:, cos_mn_2] += c[:, k]
        elif np.sign(n_0[k]) < 0:
            x[:, sin_mn_2] += s[:, k]
            if np.sign(m_0[k]) != 0:
                x[:, cos_mn_2] -= c[:, k]

    return m_1, n_1, x


def ptolemy_identity_rev(m_1, n_1, x):
    """Convert from double-Fourier to double-angle form using Ptolemy's identity.

    Converts from a double Fourier series of the form:
        ss*sin(m*theta)*sin(n*phi) + sc*sin(m*theta)*cos(n*phi) +
        cs*cos(m*theta)*sin(n*phi) + cc*cos(m*theta)*cos(n*phi)
    to the double-angle form:
        s*sin(m*theta-n*phi) + c*cos(m*theta-n*phi)
    using Ptolemy's sum and difference formulas.

    Parameters
    ----------
    m_1 : ndarray, shape(num_modes,)
        Poloidal mode numbers of the double Fourier basis.
    n_1 : ndarray, shape(num_modes,)
        Toroidal mode numbers of the double Fourier basis.
    x : ndarray, shape(surfs,num_modes,)
        Spectral coefficients in the double Fourier basis.

    Returns
    -------
    m_0 : ndarray
        Poloidal mode numbers of the double-angle Fourier basis.
    n_0 : ndarray
        Toroidal mode numbers of the double-angle Fourier basis.
    s : ndarray, shape(surfs,num_modes)
        Coefficients of sin(m*theta-n*phi) terms.
        Each row is a separate flux surface.
    c : ndarray, shape(surfs,num_modes)
        Coefficients of cos(m*theta-n*phi) terms.
        Each row is a separate flux surface.

    """
    x = np.atleast_2d(x)

    M = int(np.max(np.abs(m_1)))
    N = int(np.max(np.abs(n_1)))

    mn_0 = np.array([[m, n - N] for m in range(M + 1) for n in range(2 * N + 1)])
    mn_0 = mn_0[N:, :]
    m_0 = mn_0[:, 0]
    n_0 = mn_0[:, 1]

    s = np.zeros((x.shape[0], m_0.size))
    c = np.zeros_like(s)

    for k in range(len(m_1)):
        # (|m|*theta + |n|*phi)
        idx_pos = np.where((mn_0 == [np.abs(m_1[k]), -np.abs(n_1[k])]).all(axis=1))[0]
        # (|m|*theta - |n|*phi)
        idx_neg = np.where((mn_0 == [np.abs(m_1[k]), np.abs(n_1[k])]).all(axis=1))[0]

        # if m == 0 and n != 0, p = 0; otherwise p = 1
        p = int(bool(m_1[k])) ** int(bool(n_1[k]))

        if sign(m_1[k]) * sign(n_1[k]) < 0:
            # sin_mn terms
            if idx_pos.size:
                s[:, idx_pos[0]] += x[:, k] / (2 ** p)
            if idx_neg.size:
                s[:, idx_neg[0]] += x[:, k] / (2 ** p) * sign(n_1[k])
        else:
            # cos_mn terms
            if idx_pos.size:
                c[:, idx_pos[0]] += x[:, k] / (2 ** p) * sign(n_1[k])
            if idx_neg.size:
                c[:, idx_neg[0]] += x[:, k] / (2 ** p)

    return m_0, n_0, s, c


def fourier_to_zernike(m, n, x_mn, basis):
    """Convert from a double Fourier series to a Fourier-Zernike basis.

    Parameters
    ----------
    m : ndarray, shape(num_modes,)
        Poloidal mode numbers.
    n : ndarray, shape(num_modes,)
        Toroidal mode numbers.
    x_mn : ndarray, shape(surfs,num_modes)
        Spectral coefficients in the double Fourier basis.
        Each row is a separate flux surface, increasing from the magnetic
        axis to the boundary.
    basis : FourierZernikeBasis
        Basis set for x_lmn

    Returns
    -------
    x_lmn : ndarray, shape(num_modes,)
        Fourier-Zernike spectral coefficients.

    """
    x_lmn = np.zeros((basis.num_modes,))
    surfs = x_mn.shape[0]
    rho = np.sqrt(np.linspace(0, 1, surfs))

    for k in range(len(m)):
        idx = np.where((basis.modes[:, 1:] == [m[k], n[k]]).all(axis=1))[0]
        if len(idx):
            A = jacobi(rho, basis.modes[idx, 0], basis.modes[idx, 1])
            c = np.linalg.lstsq(A, x_mn[:, k], rcond=None)[0]
            x_lmn[idx] = c

    return x_lmn


def zernike_to_fourier(x_lmn, basis, rho):
    """Convert from a Fourier-Zernike basis to a double Fourier series.

    Parameters
    ----------
    x_lmn : ndarray, shape(num_modes,)
        Fourier-Zernike spectral coefficients.
    basis : FourierZernikeBasis
        Basis set for x_lmn.
    rho : ndarray
        Radial coordinates of flux surfaces, rho = sqrt(psi).

    Returns
    -------
    m : ndarray, shape(num_modes,)
        Poloidal mode numbers.
    n : ndarray, shape(num_modes,)
        Toroidal mode numbers.
    x_mn : ndarray, shape(surfs,num_modes)
        Spectral coefficients in the double Fourier basis.
        Each row is a separate flux surface, increasing from the magnetic
        axis to the boundary.

    """
    M = basis.M
    N = basis.N

    mn = np.array([[m - M, n - N] for m in range(2 * M + 1) for n in range(2 * N + 1)])
    m = mn[:, 0]
    n = mn[:, 1]

    x_mn = np.zeros((rho.size, m.size))
    for k in range(len(m)):
        idx = np.where((basis.modes[:, 1:] == [m[k], n[k]]).all(axis=1))[0]
        if len(idx):
            A = jacobi(rho, basis.modes[idx, 0], basis.modes[idx, 1])
            x_mn[:, k] = np.matmul(A, x_lmn[idx])

    return m, n, x_mn
