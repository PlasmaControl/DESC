"""Utility functions needed for converting VMEC inputs/outputs."""

import numpy as np
from netCDF4 import Dataset, stringtochar
from scipy.constants import mu_0
from scipy.linalg import null_space

from desc.backend import block_diag, jnp, sign
from desc.basis import DoubleFourierSeries, zernike_radial
from desc.compute import get_transforms
from desc.grid import LinearGrid
from desc.utils import Timer, warnif


def ptolemy_identity_fwd(m_0, n_0, s, c):
    """Convert from double-angle to double-Fourier form using Ptolemy's identity.

    Converts from the double-angle form:
        s * sin(m𝛉-n𝛟) + c * cos(m𝛉-n𝛟)
    to a double Fourier series of the form:
        ss * sin(m𝛉) * sin(n𝛟) + sc * sin(m𝛉) * cos(n𝛟) +
        cs * cos(m𝛉) * sin(n𝛟) + cc * cos(m𝛉) * cos(n𝛟)
    using Ptolemy's sum and difference formulas.

    Parameters
    ----------
    m_0 : ndarray
        Poloidal mode numbers of the double-angle Fourier basis.
    n_0 : ndarray
        Toroidal mode numbers of the double-angle Fourier basis.
    s : ndarray, shape(surfs,num_modes), optional
        Coefficients of sin(m𝛉-n𝛟) terms.
        Each row is a separate flux surface.
    c : ndarray, shape(surfs,num_modes), optional
        Coefficients of cos(m𝛉-n𝛟) terms.
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
        ss * sin(m𝛉) * sin(n𝛟) + sc * sin(m𝛉) * cos(n𝛟) +
        cs * cos(m𝛉) * sin(n𝛟) + cc * cos(m𝛉) * cos(n𝛟)
    to the double-angle form:
        s * sin(m𝛉-n𝛟) + c * cos(m𝛉-n𝛟)
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
        Coefficients of sin(m𝛉-n𝛟) terms.
        Each row is a separate flux surface.
    c : ndarray, shape(surfs,num_modes)
        Coefficients of cos(m𝛉-n𝛟) terms.
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

    As = zernike_radial(rho[:, np.newaxis], basis.modes[:, 0], basis.modes[:, 1])
    for k in range(len(m)):
        idx = np.where((basis.modes[:, 1:] == [m[k], n[k]]).all(axis=1))[0]
        if len(idx):
            A = As[:, idx]
            c = np.linalg.lstsq(A, x_mn[:, k], rcond=None)[0]
            x_lmn[idx] = c

    return x_lmn


def zernike_to_fourier(x_lmn, basis, rho, sym=False):
    """Convert from a Fourier-Zernike basis to a double Fourier series.

    Parameters
    ----------
    x_lmn : ndarray, shape(num_modes,)
        Fourier-Zernike spectral coefficients.
    basis : FourierZernikeBasis
        Basis set for x_lmn.
    rho : ndarray
        Radial coordinates of flux surfaces, rho = sqrt(psi).
    sym : bool
        whether or not to return the full double Fourier basis, if False
        will instead only return the Fourier basis corresponding to the
        input FourierZernike basis (with the same symmetry)
        defaults to True.

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
    if sym:
        fourier_basis = DoubleFourierSeries(M=M, N=N, sym=basis.sym, NFP=basis.NFP)
        mn = fourier_basis.modes[:, 1:]
    else:
        mn = np.array(
            [[m - M, n - N] for m in range(2 * M + 1) for n in range(2 * N + 1)]
        )
    m = mn[:, 0]
    n = mn[:, 1]

    x_mn = np.zeros((rho.size, m.size))
    As = zernike_radial(rho[:, np.newaxis], basis.modes[:, 0], basis.modes[:, 1])
    for k in range(len(m)):
        idx = np.where((basis.modes[:, 1:] == [m[k], n[k]]).all(axis=1))[0]
        if len(idx):
            A = As[:, idx]
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


def make_boozmn_output(  # noqa: C901
    eq, path, surfs=128, M_booz=None, N_booz=None, verbose=0
):
    """Create and save a booz_xform-style .nc output file.

    based strongly off of https://github.com/hiddenSymmetries/booz_xform/tree/main

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to save.
    path : str
        File path of output data.
    surfs: int
        Number of flux surfaces to calculate Boozer transform at (Default = 128).
        NOTE: because this is performed on the so-called "half-grid", this will
        result in an output of size surfs-1, since for surfs number of surfaces
        there is only surfs-1 surfaces on the half-grid, the first one being
        the surface at s = 0.5 / surfs
        where s = rho**2 is the normalized toroidal flux coordinate
    M_booz : int, optional
        poloidal resolution to use for Boozer transform. Default is `2 * eq.M`.
    N_booz : int, optional
        toroidal resolution to use for Boozer transform.Default is `2 * eq.N`.
    verbose: int
        Level of output (Default = 1).
        * 0: no output
        * 1: status of quantities computed
        * 2: as above plus timing information

    Returns
    -------
    None

    """
    timer = Timer()
    timer.start("Total time")
    warnif(
        not eq.sym,
        UserWarning,
        "Equilibrium is asymmetric, note that the `numnc` saved with this DESC "
        "function is opposite sign of the `numnc` saved from hidden symmetries"
        " booz_xform and thus not verified, though every other quantity "
        "is tested and agrees with the hidden symmetries booz_xform.",
    )
    Psi = eq.Psi
    NFP = eq.NFP
    if M_booz is None:
        M_booz = 2 * eq.M
    if N_booz is None:
        N_booz = 2 * eq.N

    # calculations are done on a "half-grid" of size surfs-1
    # which starts at s =  psi = 0.5/(surfs-1)
    # and increments by 1 / (surfs-1)
    # until it reaches 1-0.5/(surfs-1)
    # VMEC radial coordinate: s = rho^2 = Psi / Psi(LCFS)
    s_full = np.linspace(0, 1, surfs)
    hs = 1 / (surfs - 1)
    s_half = s_full[0:-1] + hs / 2
    r_full = np.sqrt(s_full)
    r_half = np.sqrt(s_half)

    grid = LinearGrid(M=2 * M_booz, N=2 * N_booz, NFP=eq.NFP, rho=r_half, sym=False)

    transforms = get_transforms(
        "|B|_mn_B",
        obj=eq,
        grid=grid,
        M_booz=M_booz - 1,
        N_booz=N_booz,
    )
    basis = transforms["B"].basis

    matrix, modes = ptolemy_linear_transform(basis.modes)
    # if sym is False, then the number of modes is double what each individual mode
    # array should be, since it has both sin(mt - nz) and cos(mt-nz) modes in it
    # while num_modes is the number of modes for a single sin or cos series
    num_modes = modes.shape[0] if eq.sym else int((modes.shape[0] + 1) / 2)

    if eq.sym:  # need a separate sin basis for Z and nu
        transforms_sin = get_transforms(
            "|B|_mn_B",
            obj=eq,
            grid=grid,
            M_booz=M_booz - 1,
            N_booz=N_booz,
            sym="sin",
        )
        basis_sin = transforms_sin["B"].basis
        matrix_sin, modes_sin = ptolemy_linear_transform(basis_sin.modes)

    else:
        matrix_sin = matrix
        modes_sin = modes
        transforms_sin = transforms
        basis_sin = basis

    timer.start("Boozer Transform")

    # precompute the needed data for the boozer surface computations
    # and for saving the boozmn.nc file
    keys = [
        "|B|",
        "<|B|^2>",
        "p",
        "R",
        "Z",
        "sqrt(g)",
        "rho",
        "psi_r",
        "lambda",
        "B_zeta",
        "B_theta",
        "G",
        "I",
        "lambda_t",
        "lambda_z",
        "psi_r",
        "R0/a",
        # need to precompute these so that when eq.sym==True,
        # the sin-symmetric coefficients have the correct values,
        # since these quantities below rely on cos-symmetric transforms
        "w_Boozer",
        "nu",
        "sqrt(g)_Boozer_DESC",
    ]

    data_keys = ["|B|_mn_B", "R_mn_B", "sqrt(g)_Boozer_mn"] + keys
    data_keys = data_keys + ["Z_mn_B", "nu_B_mn"] if not eq.sym else data_keys
    data_keys_sin = ["Z_mn_B", "nu_B_mn"]

    data = eq.compute(data_keys, grid=grid, transforms=transforms)
    # sin-symmetric data needs different transform symmetry than cos-symmetric, so
    # separate out the computation
    if eq.sym:
        # remove the modes norm from the cos-symmetric transforms
        # so the sin-symmetric term can be correctly calculated
        data.pop("Boozer transform modes norm")
        data_sin = eq.compute(
            data_keys_sin, grid=grid, transforms=transforms_sin, data=data
        )
    m_neg_inds = jnp.where(transforms["B"].basis.modes[:, 1] < 0)

    b_mn = data["|B|_mn_B"].reshape((grid.num_rho, -1))
    mask = jnp.zeros_like(b_mn)
    mask = mask.at[:, m_neg_inds].set(True)
    b_mn = jnp.where(mask, -b_mn, b_mn)
    B_mn = jnp.atleast_2d(matrix @ b_mn.T).T

    r_mn = data["R_mn_B"].reshape((grid.num_rho, -1))
    r_mn = jnp.where(mask, -r_mn, r_mn)
    R_mn = jnp.atleast_2d(matrix @ r_mn.T).T

    # must divide by dpsi/drho so that the jacobian is
    # for (psi,theta_B, zeta_B) -> (R,phi,Z)
    # instead of (rho,theta_B, zeta_B)
    sqrt_g_B_mn = (
        data["sqrt(g)_Boozer_mn"].reshape((grid.num_rho, -1))
        / grid.compress(data["psi_r"])[:, None]
    )
    sqrt_g_B_mn = np.where(mask, -sqrt_g_B_mn, sqrt_g_B_mn)
    Sqrt_g_B_mn = np.atleast_2d(matrix @ sqrt_g_B_mn.T).T
    if not eq.sym:
        z_mn = data["Z_mn_B"].reshape((grid.num_rho, -1))
        z_mn = jnp.where(mask, -z_mn, z_mn)
        Z_mn = np.atleast_2d(matrix @ z_mn.T).T

        nu_B_mn = data["nu_B_mn"].reshape((grid.num_rho, -1))
        nu_B_mn = jnp.where(mask, -nu_B_mn, nu_B_mn)
        nu_B_mn = np.atleast_2d(matrix @ nu_B_mn.T).T
    else:
        z_mn = data_sin["Z_mn_B"].reshape((grid.num_rho, -1))
        mask = jnp.zeros_like(z_mn)
        m_neg_inds = jnp.where(transforms_sin["B"].basis.modes[:, 1] < 0)
        mask = mask.at[:, m_neg_inds].set(True)
        z_mn = jnp.where(mask, -z_mn, z_mn)
        Z_mn = np.atleast_2d(matrix_sin @ z_mn.T).T

        nu_B_mn = data_sin["nu_B_mn"].reshape((grid.num_rho, -1))
        nu_B_mn = jnp.where(mask, -nu_B_mn, nu_B_mn)
        nu_B_mn = np.atleast_2d(matrix_sin @ nu_B_mn.T).T

    timer.stop("Boozer Transform")
    if verbose > 1:
        timer.disp("Boozer Transform")

    file = Dataset(path, mode="w", format="NETCDF3_64BIT_OFFSET")
    # dimensions
    # a few of these are redundant, but are included for sake
    # of matching the convention of the original booz_xform outputs
    # and the hidden symmetries implementation:
    # https://github.com/hiddenSymmetries/booz_xform/blob/main/src/_booz_xform/write_boozmn.cpp # noqa:E501
    file.createDimension("radius", s_full.size)  # number of flux surfaces plus 1
    file.createDimension("comput_surfs", surfs - 1)  # number of flux surfaces
    file.createDimension("pack_rad", surfs)  # number of flux surfaces
    file.createDimension("mn_mode", num_modes)  # number of Fourier modes
    file.createDimension("mn_modes", num_modes)  # number of Fourier modes
    file.createDimension("preset", 21)  # dimension of profile inputs
    file.createDimension("ndfmax", 101)  # used for am_aux & ai_aux
    file.createDimension("time", 100)  # used for fsq* & wdot
    file.createDimension("dim_00001", 1)
    file.createDimension("dim_00020", 20)
    file.createDimension("dim_00100", 100)
    file.createDimension("dim_00200", 200)

    version_ = file.createVariable("version", "S1", ("dim_00100",))
    version_str = "DESC Python implementation of booz_xform"
    version_[:] = stringtochar(
        np.array(
            [" " * (100 - len(version_str))],
            "S" + str(file.dimensions["dim_00100"].size),
        )
    )

    nfp = file.createVariable("nfp_b", np.int32)
    nfp.long_name = "number of field periods"
    nfp[:] = NFP

    lasym = file.createVariable("lasym__logical__", np.int32)
    lasym.long_name = "0 if the configuration is stellarator-symmetric, 1 if not"
    lasym[:] = not eq.sym

    ns = file.createVariable("ns_b", np.int32)
    ns.long_name = "Number of radial surfaces at which data is outputted minus 1"
    ns[:] = surfs

    aspect = file.createVariable("aspect_b", np.float64)
    aspect.long_name = "Aspect Ratio"
    aspect[:] = eq.compute("R0/a")["R0/a"]

    Rs = data["R"]
    Rmax = file.createVariable("rmax_b", np.int32)
    Rmax.long_name = "Maximum Radius"
    Rmax[:] = np.max(Rs)

    Rmin = file.createVariable("rmin_b", np.int32)
    Rmin.long_name = "Minimum Radius"
    Rmin[:] = np.min(Rs)

    # betaxis = beta_vol at the axis?
    grid_axis = LinearGrid(M=M_booz, N=N_booz, rho=np.array([0.0]), NFP=NFP)
    data_axis = eq.compute(["p", "<|B|^2>"], grid=grid_axis)
    betaxis = file.createVariable("betaxis_b", np.float64)
    betaxis.long_name = "2 * mu_0 * pressure / <|B|^2> on the magnetic axis"
    betaxis.units = "None"
    betaxis[:] = 2 * mu_0 * data_axis["p"][0] / data_axis["<|B|^2>"][0]

    mboz = file.createVariable("mboz_b", np.int32)
    mboz.long_name = (
        "Maximum poloidal mode number m for which the Fourier"
        "amplitudes rmnc, bmnc etc are stored"
    )
    mboz[:] = M_booz

    nboz = file.createVariable("nboz_b", np.int32)
    nboz.long_name = (
        "Maximum toloidal mode number m for which"
        "the Fourier amplitudes rmnc, bmnc etc are stored"
    )
    nboz[:] = N_booz

    mnboz = file.createVariable("mnboz_b", np.int32)
    mnboz.long_name = (
        "The total number of (m,n) pairs for which Fourier amplitudes"
        "rmnc, bmnc etc are stored."
    )
    mnboz[:] = num_modes

    # make indicial arrays for the sin and cos modes
    inds_cos = np.where(modes[:, 0] == 1)
    inds_sin = (
        np.where(modes_sin[:, 0] == -1) if eq.sym else np.where(modes[:, 0] == -1)
    )
    # add the (0,0) mode to modes_sin if eq.sym to abide by booz xform convention
    # even though the mode is trivially zero for sin(mt-nz)
    if eq.sym:
        modes_sin = np.insert(modes_sin, 0, [-1, 0, 0], axis=0)

    ## 1D Arrays
    # these should be compressed to just the radial profiles
    jlist = file.createVariable("jlist", np.int32, ("comput_surfs",))
    jlist.long_name = (
        "1-based radial indices of the surfaces for which the "
        "transformation to Boozer coordinates was computed. 2 corresponds to the "
        "first half-grid point."
    )
    jlist.units = "None"
    jlist[:] = np.arange(1, s_half.size + 1) + 1

    ixm_b = file.createVariable("ixm_b", np.int32, ("mn_modes",))
    ixm_b.long_name = (
        "Poloidal mode numbers m for which the Fourier amplitudes rmnc,"
        "bmnc etc are stored"
    )
    ixm_b.units = "None"
    ixm_b[:] = modes[:, 1] if eq.sym else modes[inds_cos, 1]

    ixn_b = file.createVariable("ixn_b", np.int32, ("mn_modes",))
    ixn_b.long_name = (
        "Toroidal mode numbers n for which the Fourier amplitudes"
        "rmnc, bmnc etc are stored"
    )
    ixn_b.units = "None"
    ixn_b[:] = modes_sin[:, 2] * eq.NFP if eq.sym else modes[inds_cos, 2] * eq.NFP

    iotas = file.createVariable("iota_b", np.float64, ("radius",))
    iotas.long_name = (
        "Rotational transform. The radial grid corresponds to the"
        "requested surfaces, and a 0 is prepended"
    )
    iotas.units = "None"

    iotas[0:] = np.insert(-grid.compress(data["iota"]), 0, 0)

    buco_b = file.createVariable("buco_b", np.float64, ("radius",))
    buco_b.long_name = (
        "Coefficient multiplying grad theta_Boozer in the covariant"
        "representation of the magnetic field vector, often denoted I(psi)."
    )
    buco_b.units = "None"
    buco_b[0] = 0
    buco_b[1:] = -grid.compress(data["I"])

    bvco_b = file.createVariable("bvco_b", np.float64, ("radius",))
    bvco_b.long_name = (
        "Coefficient multiplying grad zeta_Boozer in the covariant"
        "representation of the magnetic field vector, often denoted G(psi)."
    )
    bvco_b.units = "None"
    bvco_b[0] = 0
    bvco_b[1:] = grid.compress(data["G"])

    presf = file.createVariable("pres_b", np.float64, ("radius",))
    presf.long_name = "pressure on full mesh"
    presf.units = "Pa"
    presf[:] = eq.compute("p", grid=LinearGrid(rho=r_full, theta=0, zeta=0))["p"]

    beta = file.createVariable("beta_b", np.float64, ("radius",))
    beta.long_name = "2 * mu_0 * pressure / <|B|^2>, on half mesh"
    beta.units = "None"
    beta[1:] = 2 * mu_0 * grid.compress(data["p"]) / grid.compress(data["<|B|^2>"])
    beta[0] = 0.0

    phipf = file.createVariable("phip_b", np.float64, ("radius",))
    phipf.long_name = "d(phi)/ds: toroidal flux derivative, not normalized by 2pi"
    phipf[:] = Psi * np.ones((surfs,))

    phi = file.createVariable("phi_b", np.float64, ("radius",))
    phi.long_name = "toroidal flux, on full grid"
    phi.units = "Wb"
    phi[:] = np.linspace(0, Psi, surfs)

    chi = file.createVariable("chi_b", np.float64, ("radius",))
    chi.long_name = "Uniformly spaced grid going from 0 to"
    "the boundary poloidal flux (not divided by (2*pi))."
    chi.units = "Wb"
    chi[:] = np.linspace(
        0,
        eq.compute("chi", grid=LinearGrid(L=eq.L_grid, M=M_booz, N=N_booz, NFP=eq.NFP))[
            "chi"
        ][-1]
        * 2
        * np.pi,
        surfs,
    )

    # multi-dim arrays

    # |B|

    bmnc = file.createVariable("bmnc_b", np.float64, ("comput_surfs", "mn_mode"))
    bmnc.long_name = "cos(m*t_Boozer-n*p_Boozer) component of |B|, on half mesh"
    bmnc.units = "T"

    bmnc[0:, :] = B_mn[:, inds_cos].squeeze()

    if not eq.sym:
        bmns = file.createVariable("bmns_b", np.float64, ("comput_surfs", "mn_mode"))
        bmns.long_name = "sin(m*t_Boozer-n*p_Boozer) component of |B|, on half mesh"
        bmns.units = "T"
        # have to insert the 0,0 mode as the booz xform convention
        # expects it, even though it is trivially zero for sin(mt-nz)
        bmns[0:, :] = np.insert(B_mn[:, inds_sin].squeeze(), 0, 0, axis=1)

    # R
    rmnc = file.createVariable("rmnc_b", np.float64, ("comput_surfs", "mn_mode"))
    rmnc.long_name = (
        "cos(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes of the"
        "major radius R"
    )
    rmnc.units = "m"
    rmnc[0:, :] = R_mn[:, inds_cos].squeeze()
    if not eq.sym:
        rmns = file.createVariable("rmns_b", np.float64, ("comput_surfs", "mn_mode"))
        rmns.long_name = (
            "sin(m * theta_Boozer - n * zeta_Boozer) Fourier"
            "amplitudes of the major radius R"
        )
        rmns.units = "m"
        rmns[0:, :] = np.insert(R_mn[:, inds_sin].squeeze(), 0, 0, axis=1)

    # Z
    zmns = file.createVariable("zmns_b", np.float64, ("comput_surfs", "mn_mode"))
    zmns.long_name = (
        "sin(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes"
        "of the vertical coordinate Z"
    )

    zmns.units = "m"
    zmns[0:, :] = np.insert(Z_mn[:, inds_sin].squeeze(), 0, 0, axis=1)
    if not eq.sym:
        zmnc = file.createVariable("zmnc_b", np.float64, ("comput_surfs", "mn_mode"))
        zmnc.long_name = (
            "cos(m * theta_Boozer - n * zeta_Boozer) Fourier"
            "amplitudes of the vertical coordinate Z"
        )
        zmnc.units = "m"
        zmnc[0:, :] = Z_mn[:, inds_cos].squeeze()

    # nu
    nums = file.createVariable("pmns_b", np.float64, ("comput_surfs", "mn_mode"))
    nums.long_name = (
        "sin(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes"
        "of the angle difference zeta_DESC - zeta_Boozer"
    )
    nums.units = "m"
    # we negate here because although nu is defined as zeta_B - zeta_DESC,
    # in the original fortran there is a negative sign so it is
    # actually zeta_DESC - zeta_B
    nums[0:, :] = -np.insert(nu_B_mn[:, inds_sin].squeeze(), 0, 0, axis=1)
    if not eq.sym:
        numc = file.createVariable("pmnc_b", np.float64, ("comput_surfs", "mn_mode"))
        numc.long_name = (
            "cos(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes"
            "of the angle difference zeta_DESC - zeta_Boozer"
        )
        numc.units = "None"
        numc[0:, :] = -nu_B_mn[:, inds_cos].squeeze()

    # calculate sqrt(g)
    gmn = file.createVariable("gmn_b", np.float64, ("comput_surfs", "mn_mode"))
    gmn.long_name = (
        "cos(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes of"
        "the Boozer coordinate Jacobian (G + iota * I) / B^2"
    )
    gmn.units = "m/T"
    gmn[0:, :] = Sqrt_g_B_mn[:, inds_cos].squeeze()
    if not eq.sym:
        gmns = file.createVariable("gmns_b", np.float64, ("comput_surfs", "mn_mode"))
        gmns.long_name = (
            "sin(m * theta_Boozer - n * zeta_Boozer) Fourier amplitudes"
            "of the Boozer coordinate Jacobian (G + iota * I) / B^2"
        )
        gmns.units = "m"
        gmns[0:, :] = np.insert(Sqrt_g_B_mn[:, inds_sin].squeeze(), 0, 0, axis=1)
    file.close()

    timer.stop("Total time")
    if verbose > 1:
        timer.disp("Total time")

    return None
