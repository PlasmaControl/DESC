"""Utility functions needed for converting VMEC inputs/outputs."""

import numpy as np
from scipy.linalg import block_diag, null_space

from desc.backend import sign
from desc.basis import zernike_radial


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
    s, c = map(np.atleast_2d, (s, c))
    m_0, n_0 = map(np.atleast_1d, (m_0, n_0))
    vmec_modes, x = _mnsc_to_modes_x(m_0, n_0, s, c)
    desc_modes = _desc_modes_from_vmec_modes(vmec_modes)
    A, _ = ptolemy_linear_transform(desc_modes, vmec_modes)
    y = np.linalg.solve(A, x.T).T
    return desc_modes[:, 1], desc_modes[:, 2], y


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
    m_1, n_1 = map(np.atleast_1d, (m_1, n_1))
    desc_modes = np.vstack([np.zeros_like(m_1), m_1, n_1]).T
    A, vmec_modes = ptolemy_linear_transform(desc_modes)
    y = (A @ x.T).T
    xm, xn, s, c = _modes_x_to_mnsc(vmec_modes, y)
    return xm, xn, s, c


def _mnsc_to_modes_x(xm, xn, s, c):
    """Convert from arrays of m, n, smn, cmn to [cos/sin, m, n] and x coeffs."""
    cmodes = np.vstack([np.ones_like(xm), xm, xn]).T

    mode_idx_00 = np.where(np.logical_and(xm == 0, xn == 0))
    if mode_idx_00[0].size:  # there is a 00 mode, get rid of it for the sin
        xm_no_0 = np.delete(xm, mode_idx_00[0][0])
        xn_no_0 = np.delete(xn, mode_idx_00[0][0])

        smodes = np.vstack(
            [-np.ones_like(xm_no_0), xm_no_0, xn_no_0]
        ).T  # index out m=n=0
        s = np.atleast_2d(np.delete(s.T, mode_idx_00[0][0], axis=0).T)
    else:  # no need to index out m=n=0 bc is not in the basis
        smodes = np.vstack([-np.ones_like(xm), xm, xn]).T

    vmec_modes = np.vstack([cmodes, smodes])
    idx = np.lexsort(vmec_modes.T[np.array([0, 2, 1])])
    x = np.concatenate([c.T, s.T]).T
    vmec_modes = vmec_modes[idx]
    x = (x.T[idx]).T
    return vmec_modes, x


def _modes_x_to_mnsc(vmec_modes, x):
    """Convert from [cos/sin, m, n] and x coeffs to arrays of m, n, smn, cmn."""
    cmask = vmec_modes[:, 0] == 1
    smask = vmec_modes[:, 0] == -1
    _, xm, xn = vmec_modes[cmask].T
    if not np.any(cmask):  #  there are no cos modes, so use mask to get mode numbers
        _, xm, xn = vmec_modes[smask].T
        # concatenate the 0,0 mode
        xm = np.insert(xm, 0, 0)
        xn = np.insert(xn, 0, 0)

    c = (x.T[cmask]).T
    s = (x.T[smask]).T
    if not len(s.T):
        s = np.zeros_like(c)
    elif len(s.T):  # if there are sin terms, add a zero for the m=n=0 mode
        s = np.concatenate([np.zeros_like(s.T[:1]), s.T]).T
    if not len(c.T):
        c = np.zeros_like(s)
    assert len(s.T) == len(c.T)
    return xm, xn, s, c


def _vmec_modes_from_desc_modes(desc_modes):
    """Finds the VMEC modes corresponding to a given set of DESC modes.

    input order: [l,m,n]
    output order : [+1 for cos/-1 for sin, m, n]
    """
    vmec_modes = np.vstack(
        [
            sign(desc_modes[:, 2]) * sign(desc_modes[:, 1]),
            abs(desc_modes[:, 1]),
            desc_modes[:, 2],
        ]
    ).T
    vmec_modes[vmec_modes[:, 1] == 0, 2] = abs(vmec_modes[vmec_modes[:, 1] == 0, 2])
    vmec_modes = vmec_modes[np.lexsort(vmec_modes.T[np.array([0, 2, 1])])]
    return vmec_modes


def _desc_modes_from_vmec_modes(vmec_modes):
    """Finds the DESC modes corresponding to a given set of VMEC modes.

    input order: [+1 for cos/-1 for sin, m, n]
    output order : [l,m,n]
    """
    desc_modes = np.vstack(
        [
            np.zeros(len(vmec_modes)),
            sign(vmec_modes[:, 2]) * vmec_modes[:, 0] * vmec_modes[:, 1],
            vmec_modes[:, 2],
        ]
    ).T
    desc_modes[desc_modes[:, 1] == 0, 2] *= vmec_modes[desc_modes[:, 1] == 0, 0]
    desc_modes = desc_modes[np.lexsort(desc_modes.T[np.array([1, 0, 2])])]
    return desc_modes


def ptolemy_linear_transform(desc_modes, vmec_modes=None, helicity=None, NFP=None):
    """Compute linear transformation matrix equivalent to reverse Ptolemy's identity.

    Parameters
    ----------
    desc_modes : ndarray, shape(num_modes, 3)
        Mode numbers [l,m,n] of the double-Fourier series.
    vmec_modes : ndarray, shape(num_modes,3), optional
        Desired order of modes of the double-angle basis.
        First column: +1/-1 for cos/sin term.
        Second column: poloidal mode number m (range 0 to M).
        Third column: toroidal mode number n (range -N to N).
        If None, determined automatically.
    helicity : tuple, optional
        Type of quasi-symmetry, specified as (M, N).
    NFP: int
        Number of field periods for helicity.

    Returns
    -------
    matrix : ndarray
        Transform matrix such that M*a=b, where a are the double-Fourier coefficients
        and b are the double-angle coefficients.
    vmec_modes : ndarray, shape(num_modes,3)
        Modes of the double-angle basis. First column: +1/-1 for cos/sin term.
        Second column: poloidal mode number m (range 0 to M).
        Third column: toroidal mode number n (range -N to N).
    idx : ndarray
        The indices of the rows of `modes` that correspond to non-quasi-symmetric modes.
        Only returned if helicity is specified.

    """
    if vmec_modes is None:
        vmec_modes = _vmec_modes_from_desc_modes(desc_modes)

    cs, m1, n1 = vmec_modes.T
    _, m2, n2 = desc_modes.T

    # some logical masking for different patterns of m,n
    idx_smn_m1 = (cs == -1) * (n1 < 0) * (m1 == m2[:, None]) * (n1 == n2[:, None])
    idx_smn_m2 = (cs == -1) * (n1 < 0) * (m1 == -m2[:, None]) * (n1 == -n2[:, None])
    idx_smn_p1 = (cs == -1) * (n1 > 0) * (m1 == m2[:, None]) * (n1 == -n2[:, None])
    idx_smn_p2 = (cs == -1) * (n1 > 0) * (m1 == -m2[:, None]) * (n1 == n2[:, None])
    idx_cmn_m1 = (cs == 1) * (n1 < 0) * (m1 == -m2[:, None]) * (n1 == n2[:, None])
    idx_cmn_m2 = (cs == 1) * (n1 < 0) * (m1 == m2[:, None]) * (n1 == -n2[:, None])
    idx_cmn_p1 = (cs == 1) * (n1 > 0) * (m1 == -m2[:, None]) * (n1 == -n2[:, None])
    idx_cmn_p2 = (cs == 1) * (n1 > 0) * (m1 == m2[:, None]) * (n1 == n2[:, None])
    m_zero = (m1 == 0) * (m2[:, None] == 0)
    n_zero = (n1 == 0) * (n2[:, None] == 0)
    both_zero = m_zero * n_zero
    either_zero = m_zero + n_zero

    mat = np.zeros((len(desc_modes), len(vmec_modes)))
    # pattern for m!=0, n!=0:
    # vmec smn- = 1/2 desc m,n + 1/2 desc -m,-n
    # vmec smn+ = -1/2 desc m,-n + 1/2 desc -m,n
    # vmec cmn- = -1/2 desc -m,n + 1/2 desc m,-n
    # vmec cmn+ = 1/2 desc -m,-n + 1/2 desc m,n
    mat[idx_smn_m1] = 0.5
    mat[idx_smn_m2] = 0.5
    mat[idx_smn_p1] = -0.5
    mat[idx_smn_p2] = 0.5
    mat[idx_cmn_m1] = -0.5
    mat[idx_cmn_m2] = 0.5
    mat[idx_cmn_p1] = 0.5
    mat[idx_cmn_p2] = 0.5
    # above stuff is wrong when m or n is 0 so reset those
    mat[either_zero] = 0
    # for m=0, cos terms get +1 where n1==n2
    mat[m_zero * (n1 == n2[:, None]) * (cs == 1)] = 1
    # and sin terms get -1 where n1==-n2
    mat[m_zero * (n1 == -n2[:, None]) * (cs == -1)] = -1
    # for n=0, sin terms get +1 where n1==-n2
    mat[n_zero * (m1 == -m2[:, None]) * (cs == -1)] = 1
    # and cos terms get 1 where n1==n2
    mat[n_zero * (m1 == m2[:, None]) * (cs == 1)] = 1
    # m=n=0 is always 1
    mat[both_zero] = 1
    matrix = mat.T

    # indices of non-quasi-symmetric modes
    if helicity is not None:
        assert NFP is not None, "NFP must be supplied when specifying helicity"
        assert isinstance(helicity, tuple) and len(helicity) == 2
        M = np.abs(helicity[0])
        N = np.abs(helicity[1]) / NFP * sign(np.prod(helicity))
        idx = np.ones((vmec_modes.shape[0],), bool)
        idx[0] = False  # m=0,n=0 mode
        if N == 0:
            idx_MN = np.nonzero(vmec_modes[:, 2] == 0)[0]
        else:
            idx_MN = np.nonzero(vmec_modes[:, 1] * N == vmec_modes[:, 2] * M)[0]
        idx[idx_MN] = False
        idx = np.nonzero(idx)[0]
        return matrix, vmec_modes, idx

    return matrix, vmec_modes


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
            A = zernike_radial(
                rho[:, np.newaxis], basis.modes[idx, 0], basis.modes[idx, 1]
            )
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
    # FIXME: this always returns the full double Fourier basis regardless of symmetry
    M = basis.M
    N = basis.N

    mn = np.array([[m - M, n - N] for m in range(2 * M + 1) for n in range(2 * N + 1)])
    m = mn[:, 0]
    n = mn[:, 1]

    x_mn = np.zeros((rho.size, m.size))
    for k in range(len(m)):
        idx = np.where((basis.modes[:, 1:] == [m[k], n[k]]).all(axis=1))[0]
        if len(idx):
            A = zernike_radial(
                rho[:, np.newaxis], basis.modes[idx, 0], basis.modes[idx, 1]
            )
            x_mn[:, k] = np.matmul(A, x_lmn[idx])

    return m, n, x_mn


def vmec_boundary_subspace(eq, RBC=None, ZBS=None, RBS=None, ZBC=None):  # noqa: C901
    """Get optimization subspace corresponding to VMEC boundary modes.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to perturb.
    RBC : ndarray of int, size(num_modes,2), optional
        R boundary modes of cos(m*theta-n*NFP*phi) to set as optimization parameters.
        Each row specifies the (n,m) mode numbers of a boundary coefficient.
    ZBS : ndarray of int, size(num_modes,2), optional
        Z boundary modes of sin(m*theta-n*NFP*phi) to set as optimization parameters.
        Each row specifies the (n,m) mode numbers of a boundary coefficient.
    RBS : ndarray of int, size(num_modes,2), optional
        R boundary modes of sin(m*theta-n*NFP*phi) to set as optimization parameters.
        Each row specifies the (n,m) mode numbers of a boundary coefficient.
    ZBC : ndarray of int, size(num_modes,2), optional
        Z boundary modes of cos(m*theta-n*NFP*phi) to set as optimization parameters.
        Each row specifies the (n,m) mode numbers of a boundary coefficient.

    Returns
    -------
    opt_subspace: ndarray
        Transform matrix to give a subspace from the full optimization parameter space.
        Can be used to enforce custom optimization constraints.

    """
    idxRb = np.array([], dtype=int)
    idxZb = np.array([], dtype=int)
    Rb_subspace = np.array([])
    Zb_subspace = np.array([])

    if RBC is not None:
        RBC = np.atleast_2d(RBC)
        Rb_subspace = np.eye(eq.Rb_lmn.size)
        for k, (n, m) in enumerate(RBC):
            if m < 0:
                raise ValueError("VMEC m boundary modes cannot be negative.")
            idxRcc = eq.surface.R_basis.get_idx(M=m, N=np.abs(n))
            idxRb = np.append(idxRb, idxRcc)
            if m * n:
                idxRss = eq.surface.R_basis.get_idx(M=-m, N=-np.abs(n))
                idxRb = np.append(idxRb, idxRss)
            if not np.where((RBC == [-n, m]).all(axis=1))[0].size and m * n:
                Rb_constraint = np.zeros((eq.Rb_lmn.size,))
                Rb_constraint[idxRcc] = 1
                Rb_constraint[idxRss] = -sign(n)
                Rb_subspace = np.vstack((Rb_subspace, Rb_constraint))

    if ZBS is not None:
        ZBS = np.atleast_2d(ZBS)
        Zb_subspace = np.eye(eq.Zb_lmn.size)
        for k, (n, m) in enumerate(ZBS):
            if m < 0:
                raise ValueError("VMEC m boundary modes cannot be negative.")
            if m:
                idxZsc = eq.surface.Z_basis.get_idx(M=-m, N=np.abs(n))
                idxZb = np.append(idxZb, idxZsc)
            if n:
                idxZcs = eq.surface.Z_basis.get_idx(M=m, N=-np.abs(n))
                idxZb = np.append(idxZb, idxZcs)
            if not np.where((ZBS == [-n, m]).all(axis=1))[0].size:
                Zb_constraint = np.zeros((eq.Zb_lmn.size,))
                if m:
                    Zb_constraint[idxZsc] = 1
                if n:
                    Zb_constraint[idxZcs] = sign(n)
                Zb_subspace = np.vstack((Zb_subspace, Zb_constraint))

    if RBS is not None:
        RBS = np.atleast_2d(RBS)
        if not Rb_subspace.size:
            Rb_subspace = np.eye(eq.Rb_lmn.size)
        for k, (n, m) in enumerate(RBS):
            if m < 0:
                raise ValueError("VMEC m boundary modes cannot be negative.")
            if m:
                idxRsc = eq.surface.R_basis.get_idx(M=-m, N=np.abs(n))
                idxRb = np.append(idxRb, idxRsc)
            if n:
                idxRcs = eq.surface.R_basis.get_idx(M=m, N=-np.abs(n))
                idxRb = np.append(idxRb, idxRcs)
            if not np.where((RBS == [-n, m]).all(axis=1))[0].size:
                Rb_constraint = np.zeros((eq.Rb_lmn.size,))
                if m:
                    Rb_constraint[idxRsc] = 1
                if n:
                    Rb_constraint[idxRcs] = sign(n)
                Rb_subspace = np.vstack((Rb_subspace, Rb_constraint))

    if ZBC is not None:
        ZBC = np.atleast_2d(ZBC)
        if not Zb_subspace.size:
            Zb_subspace = np.eye(eq.Zb_lmn.size)
        for k, (n, m) in enumerate(ZBC):
            if m < 0:
                raise ValueError("VMEC m boundary modes cannot be negative.")
            idxZcc = eq.surface.Z_basis.get_idx(M=m, N=np.abs(n))
            idxZb = np.append(idxZb, idxZcc)
            if m * n:
                idxZss = eq.surface.Z_basis.get_idx(M=-m, N=-np.abs(n))
                idxZb = np.append(idxZb, idxZss)
            if not np.where((ZBC == [-n, m]).all(axis=1))[0].size and m * n:
                Zb_constraint = np.zeros((eq.Zb_lmn.size,))
                Zb_constraint[idxZcc] = 1
                Zb_constraint[idxZss] = -sign(n)
                Zb_subspace = np.vstack((Zb_subspace, Zb_constraint))

    Rb_subspace = np.delete(Rb_subspace, idxRb, 0)
    Zb_subspace = np.delete(Zb_subspace, idxZb, 0)

    boundary_subspace = block_diag(Rb_subspace, Zb_subspace)
    opt_subspace = null_space(boundary_subspace)
    return opt_subspace
