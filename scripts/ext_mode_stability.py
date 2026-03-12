import numpy as np
from desc.backend import jnp, put
from desc.utils import Index
from desc.nestor import (
    copy_vector_periods,
    eval_surface_geometry,
    compute_normal,
    compute_jacobian,
    compute_T_S,
)
from desc.grid import LinearGrid
from desc.transform import Transform
import jax


def precompute_quantities(M, N, ntheta, nzeta, NFP):
    sym = False  # hard-coded for now, can be generalized later
    ntheta_sym = ntheta // 2 + 1 if sym else ntheta

    weights = 2 * np.ones((ntheta_sym, nzeta)) / (ntheta * nzeta)
    weights[0] /= 2.0
    weights[-1] /= 2.0
    weights = weights.flatten()

    # pre-computable quantities and arrays
    # tanu, tanv
    epstan = 2.22e-16
    bigno = 1.0e50  # allows proper comparison against implementation used in VMEC
    # setting bigno = np.inf allows proper plotting

    tanu = 2.0 * np.tan(np.pi * np.arange(2 * ntheta) / ntheta)
    # mask explicit singularities at tan(pi/2), tan(3/2 pi)
    tanu = np.where(
        (np.arange(2 * ntheta) / ntheta - 0.5) % 1 < epstan,
        bigno,
        tanu,
    )

    if nzeta == 1:
        # Tokamak: need NFP_eff toroidal grid points
        NFP_eff = 64
        argv = np.arange(NFP_eff) / NFP_eff
    else:
        # Stellarator: need nzeta toroidal grid points
        argv = np.arange(nzeta) / nzeta

    tanv = 2.0 * np.tan(np.pi * argv)
    # mask explicit singularities at tan(pi/2)
    tanv = np.where((argv - 0.5) % 1 < epstan, bigno, tanv)

    cmn = np.zeros([M + N + 1, M + 1, N + 1])
    for m in range(M + 1):
        for n in range(N + 1):
            jmn = m + n
            imn = m - n
            kmn = abs(imn)
            smn = (jmn + kmn) / 2
            f1 = 1
            f2 = 1
            f3 = 1
            for i in range(1, kmn + 1):
                f1 *= smn - (i - 1)
                f2 *= i
            for l in range(kmn, jmn + 1, 2):
                cmn[l, m, n] = f1 / (f2 * f3) * ((-1) ** ((l - imn) / 2))
                f1 *= (jmn + l + 2) * (jmn - l) / 4
                f2 *= (l + 2 + kmn) / 2
                f3 *= (l + 2 - kmn) / 2

    # toroidal extent of one module
    dPhi_per = 2.0 * np.pi / NFP
    # cmns from cmn
    cmns = np.zeros([M + N + 1, M + 1, N + 1])
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            cmns[:, m, n] = (
                0.5
                * dPhi_per
                * (
                    cmn[:, m, n]
                    + cmn[:, m - 1, n]
                    + cmn[:, m, n - 1]
                    + cmn[:, m - 1, n - 1]
                )
            )
    cmns[:, 1 : M + 1, 0] = 0.5 * dPhi_per * (cmn[:, 1 : M + 1, 0] + cmn[:, 0:M, 0])
    cmns[:, 0, 1 : N + 1] = 0.5 * dPhi_per * (cmn[:, 0, 1 : N + 1] + cmn[:, 0, 0:N])
    cmns[:, 0, 0] = 0.5 * dPhi_per * (cmn[:, 0, 0] + cmn[:, 0, 0])
    cmns = jnp.asarray(cmns)
    tanu = jnp.asarray(tanu)
    tanv = jnp.asarray(tanv)

    return cmns, tanu, tanv, weights


def regularized_kernel(
    coords,
    normal,
    jacobian,
    tan_theta,
    tan_zeta,
    M,
    N,
    ntheta,
    nzeta,
    NFP,
    sym,
):
    """Computes regularized part of fourier transformed kernel and source term.

    Parameters
    ----------
    coords : dict of ndarray
        coordinates and derivatives on plasma surface
    normal : dict of ndarray
        cylindrical components of normal vector to surface
    jacobian : dict of ndarray
        jacobian elements on plasma surface
    B_field : dict of ndarray
        external magnetic field
    tan_theta, tan_zeta : ndarray
        tangent of theta, zeta with singularities masked
    M, N : integer
        maximum poloidal and toroidal mode numbers
    ntheta, nzeta : integer
        number of grid points in poloidal, toroidal directions
    NFP : integer
        number of field periods
    weights : ndarray
        quadrature weights for integration over surface

    Returns
    -------
    g_mntz : ndarray
        regularized part of greens function kernel, indexed by m, n, theta, zeta
    h_mn : ndarray
        regularized part of source term, indexed by m, n
    """
    ntheta_sym = ntheta // 2 + 1 if sym else ntheta
    NFP_eff = 64 if (nzeta == 1) else NFP
    zeta_fp = 2.0 * jnp.pi / NFP_eff * jnp.arange(NFP_eff)

    # indices over regular and primed arrays
    kt_ip, kz_ip, kt_i, kz_i = jnp.meshgrid(
        jnp.arange(ntheta_sym),
        jnp.arange(nzeta),
        jnp.arange(ntheta),
        jnp.arange(nzeta),
        indexing="ij",
    )
    ip = kt_ip * nzeta + kz_ip  # linear index over primed grid
    ip5 = (kt_ip * nzeta + kz_ip)[..., jnp.newaxis]  # linear index over primed grid
    i = kt_i * nzeta + kz_i  # linear index over primed grid
    izoff0 = ntheta * nzeta - ip5
    itoff = nzeta * (ntheta - kt_ip)[..., jnp.newaxis]

    # field-period invariant vectors
    r_squared = (coords["R_full"] ** 2 + coords["Z_full"] ** 2).reshape((-1, nzeta))
    gsave = (
        r_squared[kt_ip, kz_ip]
        + r_squared
        - 2.0
        * coords["Z"][ip].reshape(kt_ip.shape)
        * coords["Z_full"].reshape((-1, nzeta))
    )
    drv = -(coords["R"] * normal["R_n"] + coords["Z"] * normal["Z_n"])
    dsave = (
        drv[ip]
        + coords["Z_full"].reshape((-1, nzeta))
        * normal["Z_n"].reshape((ntheta_sym, nzeta))[kt_ip, kz_ip]
    )

    # copy cartesian coordinates in first field period to full domain
    X_full, Y_full = copy_vector_periods(
        jnp.array(
            [
                coords["X"].reshape((-1, nzeta))[kt_ip, kz_ip],
                coords["Y"].reshape((-1, nzeta))[kt_ip, kz_ip],
            ]
        ),
        zeta_fp,
    )
    # cartesian components of surface normal on full domain
    X_n = (normal["R_n"][ip5] * X_full - normal["phi_n"][ip5] * Y_full) / coords["R"][
        ip5
    ]
    Y_n = (normal["R_n"][ip5] * Y_full + normal["phi_n"][ip5] * X_full) / coords["R"][
        ip5
    ]

    # greens functions for kernel and source
    # theta', zeta', theta, zeta, period
    source = jnp.zeros([ntheta_sym, nzeta, ntheta, nzeta, NFP_eff])
    kernel = jnp.zeros([ntheta_sym, nzeta, ntheta, nzeta, NFP_eff])
    # full part, including singularity
    ftemp = (
        gsave[:, :, :, :, jnp.newaxis]
        - 2
        * X_full
        * coords["X"].reshape((-1, nzeta))[jnp.newaxis, jnp.newaxis, :, :, jnp.newaxis]
        - 2
        * Y_full
        * coords["Y"].reshape((-1, nzeta))[jnp.newaxis, jnp.newaxis, :, :, jnp.newaxis]
    )
    ftemp = 1 / jnp.where(ftemp <= 0, 1, ftemp)
    htemp = jnp.sqrt(ftemp)
    gtemp = (
        coords["X"].reshape((-1, nzeta))[jnp.newaxis, jnp.newaxis :, :, jnp.newaxis]
        * X_n
        + coords["Y"].reshape((-1, nzeta))[jnp.newaxis, jnp.newaxis :, :, jnp.newaxis]
        * Y_n
        + dsave[:, :, :, :, jnp.newaxis]
    )
    kernel_update = ftemp * htemp * gtemp
    source_update = htemp
    mask = ~((zeta_fp == 0) | (nzeta == 1)).reshape(
        (
            1,
            1,
            1,
            1,
            -1,
        )
    )
    kernel = jnp.where(mask, kernel + kernel_update, kernel)
    source = jnp.where(mask, source + source_update, source)

    kp = jnp.arange(NFP_eff)
    izoff = izoff0 + 2 * ntheta * kp.reshape((1, 1, 1, 1, -1))
    i_itoff = i[..., jnp.newaxis] + itoff
    i_izoff = i[..., jnp.newaxis] + izoff
    if nzeta == 1:
        # Tokamak: NFP_eff toroidal "modules"
        delta_kt = i_itoff % (2 * ntheta)
        delta_kz = i_izoff // (2 * ntheta)
    else:
        # Stellarator: nv toroidal grid points
        delta_kt = i_itoff // nzeta
        delta_kz = i_izoff % nzeta

    # subtract out singular part of the kernels
    tant = tan_theta[(delta_kt,)]
    tanz = tan_zeta[(delta_kz,)]
    ga1 = (
        tant * (jacobian["g_tt"][(ip5,)] * tant + 2 * jacobian["g_tz"][(ip5,)] * tanz)
        + jacobian["g_zz"][(ip5,)] * tanz**2
    )
    ga2 = (
        tant * (jacobian["a_tt"][(ip5,)] * tant + jacobian["a_tz"][(ip5,)] * tanz)
        + jacobian["a_zz"][(ip5,)] * tanz**2
    )

    kernel_sing = -(ga2 / ga1 * 1 / jnp.sqrt(ga1))
    source_sing = -1 / jnp.sqrt(ga1)
    mask = ((kt_ip != kt_i) | (kz_ip != kz_i) | (nzeta == 1 and kp > 0))[
        :, :, :, :, jnp.newaxis
    ] & ((zeta_fp == 0) | (nzeta == 1))
    kernel = jnp.where(mask, kernel + kernel_update + kernel_sing, kernel)
    source = jnp.where(mask, source + source_update + source_sing, source)

    if nzeta == 1:
        # Tokamak: need to do toroidal average / integral:
        # normalize by number of toroidal "modules"
        kernel /= NFP_eff
        source /= NFP_eff

    # summing over field periods
    kernel = jnp.sum(kernel, -1)
    source = jnp.sum(source, -1)

    # greens function kernel, indexed by theta,zeta,theta',zeta'
    # becomes g_mnm'n' from Merkel 1986
    # step 1: "fold over" contribution from (pi ... 2pi)
    # stellarator-symmetric first half-module is copied directly
    # the other half of the first module is "folded over" according to odd symmetry
    # under the stellarator-symmetry operation
    kt, kz = jnp.meshgrid(jnp.arange(ntheta_sym), jnp.arange(nzeta), indexing="ij")
    # anti-symmetric part from stellarator-symmetric half in second half of first
    # toroidal module
    kernel = kernel[:, :, kt, kz] - kernel[:, :, -kt, -kz]
    kernel = kernel * 1 / NFP * (2 * jnp.pi) / ntheta * (2.0 * jnp.pi) / nzeta
    kernel = put(
        kernel, Index[:, :, 0, :], 0.5 * kernel[:, :, 0, :]
    )  # scale endpoints by half (same pt in physical space)
    kernel = put(kernel, Index[:, :, -1, :], 0.5 * kernel[:, :, -1, :])
    kernel = jnp.pad(kernel, ((0, 0), (0, 0), (0, ntheta - ntheta_sym), (0, 0)))

    g_tzmn = jnp.fft.ifft(kernel, axis=2) * ntheta
    g_tzmn = jnp.fft.fft(g_tzmn, axis=3)
    g_mntz = jnp.concatenate(
        [
            g_tzmn[:ntheta_sym, :nzeta, : M + 1, : N + 1].imag,
            g_tzmn[:ntheta_sym, :nzeta, : M + 1, -N:].imag,
        ],
        axis=-1,
    ).transpose((2, 3, 0, 1))
    return g_mntz, source, kt, kz


def regularized_source(source, Bn, weights, M, N, ntheta, nzeta, NFP, sym, kt, kz):
    ntheta_sym = ntheta // 2 + 1 if sym else ntheta

    bexni = -weights * Bn * 4.0 * jnp.pi * jnp.pi
    h_tz = jnp.sum(
        bexni.reshape((ntheta_sym, nzeta, 1, 1)) * source[:ntheta_sym, :, :, :],
        axis=(0, 1),
    )
    # first step: "fold over" upper half of gsource to make use of stellarator symmetry
    # anti-symmetric part from stellarator-symmetric half in second half of first
    # toroidal module
    h_tz = h_tz[kt, kz] - h_tz[-kt, -kz]
    h_tz = h_tz * 1 / NFP * (2 * jnp.pi) / ntheta * (2.0 * jnp.pi) / nzeta
    h_tz = put(h_tz, Index[0, :], 0.5 * h_tz[0, :])
    h_tz = put(h_tz, Index[-1, :], 0.5 * h_tz[-1, :])
    h_tz = jnp.pad(h_tz, ((0, ntheta - ntheta_sym), (0, 0)))
    h_mn = jnp.fft.ifft(h_tz, axis=0) * ntheta
    h_mn = jnp.fft.fft(h_mn, axis=1)
    h_mn = jnp.concatenate(
        [h_mn[: M + 1, : N + 1].imag, h_mn[: M + 1, -N:].imag], axis=1
    )

    return h_mn.flatten()


def analytic_kernel(TS, M, N, ntheta, nzeta, cmns, sym):
    """Compute analytic integral of singular part of greens function kernels.

    Parameters
    ----------
    normal : dict of ndarray
        cylindrical components of normal vector to surface
    jacobian : dict of ndarray
        jacobian elements on plasma surface
    TS : dict of ndarray
        T^plus, T^minus, S^plus, S^minus
    B_field : dict of ndarray
        external magnetic field
    M, N : integer
        maximum poloidal and toroidal mode numbers
    ntheta, nzeta : integer
        number of grid points in poloidal, toroidal directions
    cmns : ndarray
        precomputed coefficients for power series expansion
    weights : ndarray
        quadrature weights for integration

    Returns
    -------
    I_mn : ndarray
        singular part of source term, indexed by m, n
    K_mntz : ndarray
        singular part of greens function kernel, indexed by m, n, theta, zeta
    """
    ntheta_sym = ntheta // 2 + 1 if sym else ntheta

    kt, kz = jnp.meshgrid(jnp.arange(ntheta_sym), jnp.arange(nzeta))
    i = nzeta * kt + kz

    num_four = M + N + 1
    S_p_4d = jnp.zeros([num_four, ntheta_sym, nzeta, ntheta_sym * nzeta])
    S_m_4d = jnp.zeros([num_four, ntheta_sym, nzeta, ntheta_sym * nzeta])

    S_p_4d = put(
        S_p_4d,
        Index[:, kt, kz, i],
        TS["S_p_l"].reshape(num_four, ntheta_sym, nzeta)[:, kt, kz],
    )
    S_m_4d = put(
        S_m_4d,
        Index[:, kt, kz, i],
        TS["S_m_l"].reshape(num_four, ntheta_sym, nzeta)[:, kt, kz],
    )

    S_p_4d = jnp.pad(S_p_4d, ((0, 0), (0, ntheta - ntheta_sym), (0, 0), (0, 0)))
    ft_S_p = jnp.fft.ifft(S_p_4d, axis=1) * ntheta
    ft_S_p = jnp.fft.fft(ft_S_p, axis=2)

    S_m_4d = jnp.pad(S_m_4d, ((0, 0), (0, ntheta - ntheta_sym), (0, 0), (0, 0)))
    ft_S_m = jnp.fft.ifft(S_m_4d, axis=1) * ntheta
    ft_S_m = jnp.fft.fft(ft_S_m, axis=2)

    m, n = jnp.meshgrid(
        jnp.arange(M + 1),
        jnp.concatenate([jnp.arange(N + 1), jnp.arange(-N, 0)]),
        indexing="ij",
    )

    K_mntz = jnp.zeros([M + 1, 2 * N + 1, ntheta_sym * nzeta])
    K_mntz = jnp.where(
        jnp.logical_or(m == 0, n == 0)[:, :, jnp.newaxis],
        jnp.sum(
            cmns[:, m, n, jnp.newaxis]
            * (ft_S_p[:, m, n, :].imag + ft_S_m[:, m, n, :].imag),
            axis=0,
        ),
        K_mntz,
    )
    K_mntz = jnp.where(
        jnp.logical_and(m != 0, n > 0)[:, :, jnp.newaxis],
        jnp.sum(cmns[:, m, n, jnp.newaxis] * ft_S_p[:, m, n, :].imag, axis=0),
        K_mntz,
    )
    K_mntz = jnp.where(
        jnp.logical_and(m != 0, n < 0)[:, :, jnp.newaxis],
        jnp.sum(cmns[:, m, -n, jnp.newaxis] * ft_S_m[:, m, n, :].imag, axis=0),
        K_mntz,
    )
    K_mntz = K_mntz.reshape(M + 1, 2 * N + 1, ntheta_sym, nzeta)
    return K_mntz


def analytic_source(TS, M, N, Bn, weights, cmns, sym, ntheta, nzeta):
    m, n = jnp.meshgrid(
        jnp.arange(M + 1),
        jnp.concatenate([jnp.arange(N + 1), jnp.arange(-N, 0)]),
        indexing="ij",
    )
    ntheta_sym = ntheta // 2 + 1 if sym else ntheta

    bexni = -weights * Bn * 4.0 * jnp.pi * jnp.pi
    T_p = (TS["T_p_l"] * bexni).reshape(-1, ntheta_sym, nzeta)
    T_m = (TS["T_m_l"] * bexni).reshape(-1, ntheta_sym, nzeta)

    T_p = jnp.pad(T_p, ((0, 0), (0, ntheta - ntheta_sym), (0, 0)))
    ft_T_p = jnp.fft.ifft(T_p, axis=1) * ntheta
    ft_T_p = jnp.fft.fft(ft_T_p, axis=2)

    T_m = jnp.pad(T_m, ((0, 0), (0, ntheta - ntheta_sym), (0, 0)))
    ft_T_m = jnp.fft.ifft(T_m, axis=1) * ntheta
    ft_T_m = jnp.fft.fft(ft_T_m, axis=2)

    I_mn = jnp.zeros([M + 1, 2 * N + 1])
    I_mn = jnp.where(
        jnp.logical_or(m == 0, n == 0),
        (n >= 0)
        * jnp.sum(
            cmns[:, m, n] * (ft_T_p[:, m, n].imag + ft_T_m[:, m, n].imag), axis=0
        ),
        I_mn,
    )
    I_mn = jnp.where(
        jnp.logical_and(m != 0, n > 0),
        jnp.sum(cmns[:, m, n] * ft_T_p[:, m, n].imag, axis=0),
        I_mn,
    )
    I_mn = jnp.where(
        jnp.logical_and(m != 0, n < 0),
        jnp.sum(cmns[:, m, -n] * ft_T_m[:, m, n].imag, axis=0),
        I_mn,
    )

    return I_mn.flatten()


def regularized_source_matrix(sym, source, weights, M, N, kt, kz, ntheta, nzeta, NFP):
    """Returns matrix A of shape (n_imn, n_theta*n_zeta) such that A @ Bn = I_mn."""
    ntheta_sym = ntheta // 2 + 1 if sym else ntheta

    def apply_to_Bn(Bn):
        return regularized_source(
            source, Bn, weights, M, N, ntheta, nzeta, NFP, sym, kt, kz
        )

    # Build identity matrix over spatial points
    n_in = ntheta_sym * nzeta
    eye = jnp.eye(n_in)  # (n_in, n_in)

    # vmap over rows of eye (each is a basis vector)
    A = jax.vmap(apply_to_Bn)(eye)  # (n_in, n_out)
    # A = put(A, Index[N + 1:2*N+1, :], 0.0)

    return A.T  # (n_out, n_in) so that A @ Bn = I_mn.flatten()


def analytic_source_matrix(
    TS, M, N, weights, cmns, sym, ntheta, nzeta
):  # , source, kt, kz):
    """Returns matrix A of shape (n_imn, n_theta*n_zeta) such that A @ Bn = I_mn."""
    ntheta_sym = ntheta // 2 + 1 if sym else ntheta

    def apply_to_Bn(Bn):
        return analytic_source(TS, M, N, Bn, weights, cmns, sym, ntheta, nzeta)

    # Build identity matrix over spatial points
    n_in = ntheta_sym * nzeta
    eye = jnp.eye(n_in)  # (n_in, n_in)

    # vmap over rows of eye (each is a basis vector)
    A = jax.vmap(apply_to_Bn)(eye)  # (n_in, n_out)
    # A = put(A, Index[N + 1:2*N+1, :], 0.0)

    return A.T  # (n_out, n_in) so that A @ Bn = I_mn.flatten()


def source_matrix(TS, M, N, weights, cmns, sym, source, kt, kz, ntheta, nzeta, NFP):
    A = regularized_source_matrix(
        sym, source, weights, M, N, kt, kz, ntheta, nzeta, NFP
    ) + analytic_source_matrix(TS, M, N, weights, cmns, sym, ntheta, nzeta)
    A = put(A, Index[N + 1 : 2 * N + 1, :], 0.0)
    return A  # -A  # flip the sign for exterior neumann problem


def ifft_matrix(M, N, ntheta, nzeta):
    """Converts phi_mn as returned by G^{-1}HB_n to phi on uniform theta, zeta grid
    theta = 0, 2pi/ntheta, ..., 2pi*(ntheta-1)/ntheta
    zeta = 0, 2pi/NFP * nzeta, ..., 2pi*(nzeta-1)/NFP * nzeta

    --> phi = modes_matrix @ phi_mn
    """
    m = jnp.arange(M + 1)
    n = jnp.concatenate([jnp.arange(N + 1), jnp.arange(-N, 0)])
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)  # (ntheta, )
    zeta = np.linspace(0, 2 * np.pi, nzeta, endpoint=False)  # (ntheta, )
    m_theta = m[:, None, None, None] * theta[None, None, :, None]
    n_zeta = n[None, :, None, None] * zeta[None, None, None, :]
    modes_matrix = (
        jnp.sin(m_theta - n_zeta).reshape((M + 1) * (2 * N + 1), ntheta * nzeta).T
    )  # (M+1, 2N+1, ntheta*nzeta)
    return modes_matrix


def get_matrices(eq, M, N, ntheta, nzeta):
    """G phi = H B_n"""
    # Precompute quantities
    signgs = np.sign(np.mean(eq.compute("sqrt(g)")["sqrt(g)"]))
    cmns, tanu, tanv, weights = precompute_quantities(M, N, ntheta, nzeta, eq.NFP)

    # Get boundary transfrom
    bdry_grid = LinearGrid(rho=1, theta=ntheta, zeta=nzeta, NFP=eq.NFP)
    _Rb_transform = Transform(bdry_grid, eq.R_basis, derivs=2)
    _Zb_transform = Transform(bdry_grid, eq.Z_basis, derivs=2)
    surface_coords = eval_surface_geometry(
        eq.R_lmn,
        eq.Z_lmn,
        _Rb_transform,
        _Zb_transform,
        ntheta,
        nzeta,
        eq.NFP,
        sym=False,
    )
    normal = compute_normal(surface_coords, signgs)
    jacobian = compute_jacobian(surface_coords, normal, eq.NFP)
    sym = False
    TS = compute_T_S(jacobian, M, N, ntheta, nzeta, sym)

    # Compute kernel matrix
    g_mntz, source, kt, kz = regularized_kernel(
        coords=surface_coords,
        normal=normal,
        jacobian=jacobian,
        tan_theta=tanu,
        tan_zeta=tanv,
        M=M,
        N=N,
        ntheta=ntheta,
        nzeta=nzeta,
        NFP=eq.NFP,
        sym=sym,
    )

    K_mntz = analytic_kernel(
        TS=TS, M=M, N=N, ntheta=ntheta, nzeta=nzeta, cmns=cmns, sym=sym
    )

    g_mntz = (  # -(
        g_mntz + K_mntz
    )  # flip the sign for exterior neumann problem; see merkel 1986

    # compute Fourier transform of grpmn to arrive at gmatrix
    ntheta_sym = ntheta // 2 + 1 if sym else ntheta
    g_mntz = g_mntz * weights.reshape([1, 1, ntheta_sym, nzeta])
    g_mntz = jnp.pad(g_mntz, ((0, 0), (0, 0), (0, ntheta - ntheta_sym), (0, 0)))
    g_mnmn = jnp.fft.ifft(g_mntz, axis=2) * ntheta
    g_mnmn = jnp.fft.fft(g_mnmn, axis=3)

    gmatrix_4d = jnp.concatenate(
        [g_mnmn[:, :, : M + 1, : N + 1].imag, g_mnmn[:, :, : M + 1, -N:].imag],
        axis=-1,
    )
    # scale gmatrix by (2 pi)^2, copied from fortran
    gmatrix_4d *= (2.0 * jnp.pi) ** 2
    m, n = jnp.meshgrid(jnp.arange(M + 1), jnp.arange(2 * N + 1), indexing="ij")
    # zero out (m=0, n<0, m', n') modes for all m', n', from fortran
    gmatrix_4d = jnp.where(
        jnp.logical_and(m == 0, n > N)[:, :, jnp.newaxis, jnp.newaxis], 0, gmatrix_4d
    )
    # add diagonal terms, copied from fortran
    gmatrix_4d = put(
        gmatrix_4d, Index[m, n, m, n], gmatrix_4d[m, n, m, n] + 4.0 * jnp.pi**3
    )

    gmatrix = gmatrix_4d.reshape([(M + 1) * (2 * N + 1), (M + 1) * (2 * N + 1)])

    # compute source matrix
    hmatrix = source_matrix(
        TS, M, N, weights, cmns, sym, source, kt, kz, ntheta, nzeta, eq.NFP
    )

    return gmatrix, hmatrix
