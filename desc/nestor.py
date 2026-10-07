"""Neumann Solver for Toroidal Systems.

Original python implementation by Jonathan Schilling (jonathan.schilling@ipp.mpg.de)
Rewritten to use JAX by DESC team.
"""

import numpy as np
from scipy.constants import mu_0

from desc.backend import fori_loop, jnp, put
from desc.grid import LinearGrid
from desc.io import IOAble
from desc.transform import Transform
from desc.utils import Index, setdefault


def copy_vector_periods(vec, zetas):
    """Copies a vector into each field period by rotation.

    Parameters
    ----------
    vec : ndarray, shape(3,...)
        vector(s) to rotate
    zetas : ndarray
        angles to rotate by (eg start of each field period)

    Returns
    -------
    vec : ndarray, shape(3,...,nzeta)
        vector(s) repeated and rotated by angle zeta
    """
    if vec.shape[0] == 3:
        x, y, z = vec
    else:
        x, y = vec
    shp = x.shape
    xx = x.reshape((*shp, 1)) * jnp.cos(zetas) - y.reshape((*shp, 1)) * jnp.sin(zetas)
    yy = y.reshape((*shp, 1)) * jnp.cos(zetas) + x.reshape((*shp, 1)) * jnp.sin(zetas)
    if vec.shape[0] == 3:
        zz = jnp.broadcast_to(z.reshape((*shp, 1)), (*shp, zetas.size))
        return jnp.array((xx, yy, zz))
    return jnp.array((xx, yy))


def eval_surface_geometry(
    R_lmn, Z_lmn, Rb_transform, Zb_transform, ntheta, nzeta, NFP, sym
):
    """Evaluate coordinates and derivatives on the surface.

    Parameters
    ----------
    R_lmn : ndarray
        spectral coefficients for R
    Z_lmn : ndarray
        spectral coefficients for Z
    Rb_transform : Transform
        object to transform R from spectral to real space
    Zb_transform : Transform
        object to transform Z from spectral to real space
    ntheta, nzeta : int
        number of grid points in poloidal, toroidal directions
    NFP : integer
        number of field periods
    sym : bool
        whether to assume stellarator symmetry

    Returns
    -------
    coords :dict of ndarray
        dictionary of arrays of coordinates R,Z and derivatives on a regular grid
        in theta, zeta
    """
    ntheta_sym = ntheta // 2 + 1 if sym else ntheta
    phi = jnp.linspace(0, 2 * jnp.pi, nzeta, endpoint=False) / NFP

    R_2d = Rb_transform.transform(R_lmn, dt=0, dz=0).reshape((nzeta, ntheta)).T
    R_t_2d = Rb_transform.transform(R_lmn, dt=1, dz=0).reshape((nzeta, ntheta)).T
    R_z_2d = Rb_transform.transform(R_lmn, dt=0, dz=1).reshape((nzeta, ntheta)).T
    R_tt_2d = Rb_transform.transform(R_lmn, dt=2, dz=0).reshape((nzeta, ntheta)).T
    R_tz_2d = Rb_transform.transform(R_lmn, dt=1, dz=1).reshape((nzeta, ntheta)).T
    R_zz_2d = Rb_transform.transform(R_lmn, dt=0, dz=2).reshape((nzeta, ntheta)).T
    Z_2d = Zb_transform.transform(Z_lmn, dt=0, dz=0).reshape((nzeta, ntheta)).T
    Z_t_2d = Zb_transform.transform(Z_lmn, dt=1, dz=0).reshape((nzeta, ntheta)).T
    Z_z_2d = Zb_transform.transform(Z_lmn, dt=0, dz=1).reshape((nzeta, ntheta)).T
    Z_tt_2d = Zb_transform.transform(Z_lmn, dt=2, dz=0).reshape((nzeta, ntheta)).T
    Z_tz_2d = Zb_transform.transform(Z_lmn, dt=1, dz=1).reshape((nzeta, ntheta)).T
    Z_zz_2d = Zb_transform.transform(Z_lmn, dt=0, dz=2).reshape((nzeta, ntheta)).T

    coords = {}
    coords["R_full"] = R_2d.flatten()
    coords["Z_full"] = Z_2d.flatten()
    coords["R"] = R_2d[:ntheta_sym, :].flatten()
    coords["Z"] = Z_2d[:ntheta_sym, :].flatten()
    coords["R_t"] = R_t_2d[:ntheta_sym, :].flatten()
    coords["R_z"] = R_z_2d[:ntheta_sym, :].flatten()
    coords["R_tt"] = R_tt_2d[:ntheta_sym, :].flatten()
    coords["R_tz"] = R_tz_2d[:ntheta_sym, :].flatten()
    coords["R_zz"] = R_zz_2d[:ntheta_sym, :].flatten()
    coords["Z_t"] = Z_t_2d[:ntheta_sym, :].flatten()
    coords["Z_z"] = Z_z_2d[:ntheta_sym, :].flatten()
    coords["Z_tt"] = Z_tt_2d[:ntheta_sym, :].flatten()
    coords["Z_tz"] = Z_tz_2d[:ntheta_sym, :].flatten()
    coords["Z_zz"] = Z_zz_2d[:ntheta_sym, :].flatten()

    coords["phi"] = jnp.broadcast_to(phi, (ntheta_sym, nzeta)).flatten()
    coords["X"] = (R_2d * jnp.cos(phi)).flatten()
    coords["Y"] = (R_2d * jnp.sin(phi)).flatten()

    return coords


def eval_axis_geometry(R_lmn, Z_lmn, Ra_transform, Za_transform, nzeta, NFP):
    """Evaluate coordinates and derivatives on the axis.

    Parameters
    ----------
    R_lmn : ndarray
        spectral coefficients for R
    Z_lmn : ndarray
        spectral coefficients for Z
    Ra_transform : Transform
        object to transform R from spectral to real space
    Za_transform : Transform
        object to transform Z from spectral to real space
    nzeta : int
        number of grid points in toroidal directions
    NFP : integer
        number of field periods

    Returns
    -------
    axis : dict of ndarray
        dictionary of arrays of cylindrical coordinates of axis
    """
    NFP_eff = 64 if (nzeta == 1) else NFP
    zeta_fp = 2.0 * jnp.pi / NFP_eff * jnp.arange(NFP_eff)

    raxis = Ra_transform.transform(R_lmn)
    zaxis = Za_transform.transform(Z_lmn)
    phiaxis = jnp.linspace(0, 2 * jnp.pi, nzeta, endpoint=False) / NFP
    axis = jnp.array([raxis * jnp.cos(phiaxis), raxis * jnp.sin(phiaxis), zaxis])

    axis = jnp.moveaxis(copy_vector_periods(axis, zeta_fp), -1, 1).reshape((3, -1))
    coords = {}
    coords["R"] = jnp.hypot(axis[0], axis[1])
    coords["phi"] = jnp.arctan2(axis[1], axis[0])
    coords["Z"] = axis[2]
    return coords


def compute_normal(coords, signgs):
    """Compute the outward normal vector to the plasma surface.

    Parameters
    ----------
    coords : dict of ndarray
        coordinates and derivatives on plasma surface
    signgs : integer
        sign of the coordinate jacobian (+1 for right handed coordinates, -1 for left)

    Returns
    -------
    normal : dict of ndarray
        R, phi, Z components of normal vector on regular grid in theta, zeta
    """
    normal = {}
    normal["R_n"] = signgs * (coords["R"] * coords["Z_t"])
    normal["phi_n"] = signgs * (
        coords["R_t"] * coords["Z_z"] - coords["R_z"] * coords["Z_t"]
    )
    normal["Z_n"] = -signgs * (coords["R"] * coords["R_t"])
    return normal


def compute_jacobian(coords, normal, NFP):
    """Compute the surface jacobian elements.

    Parameters
    ----------
    coords : dict of ndarray
        cylindrical coordinates and derivatives on the surface
    normal : dict of ndarray
        cylindrical components of normal vector to surface
    NFP : int
        number of field periods

    Returns
    -------
    jacobian : dict of ndarray
        jacobian elements on the surface on regular grid in theta, zeta
    """
    jacobian = {}
    # a, b, c in NESTOR article: dot-products of first-order derivatives of surface
    jacobian["g_tt"] = coords["R_t"] * coords["R_t"] + coords["Z_t"] * coords["Z_t"]
    jacobian["g_tz"] = (
        coords["R_t"] * coords["R_z"] + coords["Z_t"] * coords["Z_z"]
    ) / NFP
    jacobian["g_zz"] = (
        coords["R_z"] * coords["R_z"]
        + coords["Z_z"] * coords["Z_z"]
        + coords["R"] * coords["R"]
    ) / NFP**2
    # A, B and C in NESTOR article: surface normal dotted with second-order
    # derivative of surface
    jacobian["a_tt"] = 0.5 * (
        normal["R_n"] * coords["R_tt"] + normal["Z_n"] * coords["Z_tt"]
    )
    jacobian["a_tz"] = (
        normal["R_n"] * coords["R_tz"]
        + normal["phi_n"] * coords["R_t"]
        + normal["Z_n"] * coords["Z_tz"]
    ) / NFP
    jacobian["a_zz"] = (
        normal["phi_n"] * coords["R_z"]
        + 0.5
        * (
            normal["R_n"] * (coords["R_zz"] - coords["R"])
            + normal["Z_n"] * coords["Z_zz"]
        )
    ) / NFP**2
    return jacobian


def biot_savart(eval_pts, coil_pts, current):
    """Biot-Savart law following [1].

    Parameters
    ----------
    eval_pts : array-like shape(3,n)
        evaluation points in cartesian coordinates
    coil_pts : array-like shape(3,m)
        points in cartesian space defining coil
    current : float
        current through the coil

    Returns
    -------
    B : ndarray, shape(3,k)
        magnetic field in cartesian components at specified points

    [1] Hanson & Hirshman, "Compact expressions for the Biot-Savart fields of a
    filamentary segment" (2002)
    """
    dvec = jnp.diff(coil_pts, axis=1)
    L = jnp.linalg.norm(dvec, axis=0)

    Ri_vec = eval_pts[:, :, jnp.newaxis] - coil_pts[:, jnp.newaxis, :-1]
    Ri = jnp.linalg.norm(Ri_vec, axis=0)
    Rf = jnp.linalg.norm(
        eval_pts[:, :, jnp.newaxis] - coil_pts[:, jnp.newaxis, 1:], axis=0
    )
    Ri_p_Rf = Ri + Rf

    Bmag = (
        mu_0
        / (4 * jnp.pi)
        * current
        * 2.0
        * Ri_p_Rf
        / (Ri * Rf * (Ri_p_Rf * Ri_p_Rf - L * L))
    )

    # cross product of L*hat(eps)==dvec with Ri_vec, scaled by Bmag
    vec = jnp.cross(dvec, Ri_vec, axis=0)
    return jnp.sum(Bmag * vec, axis=-1)


def modelNetToroidalCurrent(axis, current, coords, normal):
    """Compute field due to net toroidal current.

    Models the current as a filament along the magnetic axis and computes field on
    the boundary

    Parameters
    ----------
    axis : ndarray, shape(3,k)
        cylindrical (R,phi,Z) coordinates of magnetic axis covering the full torus
    current : float
        net toroidal plasma current in Amps
    coords : dict of ndarray
        coordinates and derivatives on plasma surface in first field period
    normal : dict of ndarray
        cylindrical components of outward normal vector on surface in first field period

    Returns
    -------
    B_j : dict of ndarray
        field on the boundary due to net current at magnetic axis, in cartesian and
        cylindrical components
    """
    axis = jnp.array(
        [axis["R"] * jnp.cos(axis["phi"]), axis["R"] * jnp.sin(axis["phi"]), axis["Z"]]
    )
    # first point == last point for periodicity
    axis = jnp.hstack([axis[:, -1:], axis])

    eval_pts = jnp.array(
        [
            coords["R"] * jnp.cos(coords["phi"]),
            coords["R"] * jnp.sin(coords["phi"]),
            coords["Z"],
        ]
    )

    B = biot_savart(eval_pts, axis, current)

    # convert to cylindrical components
    B_j = {}
    B_j["BX"] = B[0]
    B_j["BY"] = B[1]
    B_j["BZ"] = B[2]
    B_j["BR"] = B[0] * jnp.cos(coords["phi"]) + B[1] * jnp.sin(coords["phi"])
    B_j["Bphi"] = -B[0] * jnp.sin(coords["phi"]) + B[1] * jnp.cos(coords["phi"])
    B_j["Bn"] = (
        normal["R_n"] * B_j["BR"]
        + normal["phi_n"] * B_j["Bphi"]
        + normal["Z_n"] * B_j["BZ"]
    )

    return B_j


def compute_T_S(jacobian, mf, nf, ntheta, nzeta, sym):
    """Compute T and S functions needed for analytic integrals by recurrence relation.

    Parameters
    ----------
    jacobian : dict of ndarray
        jacobian elements on plasma surface
    mf, nf : integer
        maximum poloidal and toroidal mode numbers
    ntheta, nzeta : integer
        number of grid points in poloidal, toroidal directions

    Returns
    -------
    TS : dict of ndarray
        T^plus, T^minus, S^plus, S^minus
    """
    ntheta_sym = ntheta // 2 + 1 if sym else ntheta

    a = jacobian["g_tt"]
    b = jacobian["g_tz"]
    c = jacobian["g_zz"]
    ap = a + 2 * b + c
    am = a - 2 * b + c
    cma = c - a

    sqrt_a = jnp.sqrt(a)
    sqrt_c = jnp.sqrt(c)
    sqrt_ap = jnp.sqrt(ap)
    sqrt_am = jnp.sqrt(am)

    delt1u = ap * am - cma * cma
    azp1u = jacobian["a_tt"] + jacobian["a_tz"] + jacobian["a_zz"]
    azm1u = jacobian["a_tt"] - jacobian["a_tz"] + jacobian["a_zz"]
    cma11u = jacobian["a_zz"] - jacobian["a_tt"]
    r1p = (azp1u * (delt1u - cma * cma) / ap - azm1u * ap + 2.0 * cma11u * cma) / delt1u
    r1m = (azm1u * (delt1u - cma * cma) / am - azp1u * am + 2.0 * cma11u * cma) / delt1u
    r0p = (-azp1u * am * cma / ap - azm1u * cma + 2.0 * cma11u * am) / delt1u
    r0m = (-azm1u * ap * cma / am - azp1u * cma + 2.0 * cma11u * ap) / delt1u
    ra1p = azp1u / ap
    ra1m = azm1u / am

    # compute T^{\pm}_l, S^{\pm}_l
    T_p_l = jnp.zeros([mf + nf + 1, ntheta_sym * nzeta])  # T^{+}_l
    T_m_l = jnp.zeros([mf + nf + 1, ntheta_sym * nzeta])  # T^{-}_l
    S_p_l = jnp.zeros([mf + nf + 1, ntheta_sym * nzeta])  # S^{+}_l
    S_m_l = jnp.zeros([mf + nf + 1, ntheta_sym * nzeta])  # S^{-}_l

    T_p_l = put(
        T_p_l,
        Index[0, :],
        1.0
        / sqrt_ap
        * jnp.log(
            (sqrt_ap * 2 * sqrt_c + ap + cma) / (sqrt_ap * 2 * sqrt_a - ap + cma)
        ),
    )
    T_m_l = put(
        T_m_l,
        Index[0, :],
        1.0
        / sqrt_am
        * jnp.log(
            (sqrt_am * 2 * sqrt_c + am + cma) / (sqrt_am * 2 * sqrt_a - am + cma)
        ),
    )
    S_p_l = put(
        S_p_l,
        Index[0, :],
        ra1p * T_p_l[0, :] - (r1p + r0p) / (2 * sqrt_c) + (r0p - r1p) / (2 * sqrt_a),
    )
    S_m_l = put(
        S_m_l,
        Index[0, :],
        ra1m * T_m_l[0, :] - (r1m + r0m) / (2 * sqrt_c) + (r0m - r1m) / (2 * sqrt_a),
    )

    T_p_l = put(
        T_p_l,
        Index[1, :],
        ((2 * sqrt_c + (-1) * 2 * sqrt_a) - (1.0) * cma * T_p_l[0, :]) / (ap),
    )
    T_m_l = put(
        T_m_l,
        Index[1, :],
        ((2 * sqrt_c + (-1) * 2 * sqrt_a) - (1.0) * cma * T_m_l[0, :]) / (am),
    )
    S_p_l = put(
        S_p_l,
        Index[1, :],
        (r1p + ra1p) * T_p_l[1, :]
        + r0p * T_p_l[0, :]
        - (r1p + r0p) / (2 * sqrt_c)
        + (-1) * (r0p - r1p) / (2 * sqrt_a),
    )
    S_m_l = put(
        S_m_l,
        Index[1, :],
        (r1m + ra1m) * T_m_l[1, :]
        + r0m * T_m_l[0, :]
        - (r1m + r0m) / (2 * sqrt_c)
        + (-1) * (r0m - r1m) / (2 * sqrt_a),
    )

    arrs = {
        "T_p_l": T_p_l,
        "T_m_l": T_m_l,
        "S_p_l": S_p_l,
        "S_m_l": S_m_l,
    }
    # now use recurrence relation for l > 0

    def body_fun(l, arrs):
        # compute T^{\pm}_l
        arrs["T_p_l"] = put(
            arrs["T_p_l"],
            Index[l, :],
            (
                (2 * sqrt_c + (-1) ** l * 2 * sqrt_a)
                - (2.0 * l - 1.0) * cma * arrs["T_p_l"][l - 1, :]
                - (l - 1) * am * arrs["T_p_l"][l - 2, :]
            )
            / (ap * l),
        )
        arrs["T_m_l"] = put(
            arrs["T_m_l"],
            Index[l, :],
            (
                (2 * sqrt_c + (-1) ** l * 2 * sqrt_a)
                - (2.0 * l - 1.0) * cma * arrs["T_m_l"][l - 1, :]
                - (l - 1) * ap * arrs["T_m_l"][l - 2, :]
            )
            / (am * l),
        )

        # compute S^{\pm}_l based on T^{\pm}_l and T^{\pm}_{l-1}
        arrs["S_p_l"] = put(
            arrs["S_p_l"],
            Index[l, :],
            (r1p * l + ra1p) * arrs["T_p_l"][l, :]
            + r0p * l * arrs["T_p_l"][l - 1, :]
            - (r1p + r0p) / (2 * sqrt_c)
            + (-1) ** l * (r0p - r1p) / (2 * sqrt_a),
        )
        arrs["S_m_l"] = put(
            arrs["S_m_l"],
            Index[l, :],
            (r1m * l + ra1m) * arrs["T_m_l"][l, :]
            + r0m * l * arrs["T_m_l"][l - 1, :]
            - (r1m + r0m) / (2 * sqrt_c)
            + (-1) ** l * (r0m - r1m) / (2 * sqrt_a),
        )

        return arrs

    arrs = fori_loop(2, mf + nf + 1, body_fun, arrs)

    return arrs


def compute_analytic_integrals(
    normal, jacobian, TS, B_field, mf, nf, ntheta, nzeta, cmns, weights, sym
):
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
    mf, nf : integer
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
    bexni = -weights * B_field["Bn"] * 4.0 * jnp.pi * jnp.pi
    T_p = (TS["T_p_l"] * bexni).reshape(-1, ntheta_sym, nzeta)
    T_m = (TS["T_m_l"] * bexni).reshape(-1, ntheta_sym, nzeta)

    T_p = jnp.pad(T_p, ((0, 0), (0, ntheta - ntheta_sym), (0, 0)))
    ft_T_p = jnp.fft.ifft(T_p, axis=1) * ntheta
    ft_T_p = jnp.fft.fft(ft_T_p, axis=2)

    T_m = jnp.pad(T_m, ((0, 0), (0, ntheta - ntheta_sym), (0, 0)))
    ft_T_m = jnp.fft.ifft(T_m, axis=1) * ntheta
    ft_T_m = jnp.fft.fft(ft_T_m, axis=2)

    kt, kz = jnp.meshgrid(jnp.arange(ntheta_sym), jnp.arange(nzeta))
    i = nzeta * kt + kz

    num_four = mf + nf + 1
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
        jnp.arange(mf + 1),
        jnp.concatenate([jnp.arange(nf + 1), jnp.arange(-nf, 0)]),
        indexing="ij",
    )

    I_mn = jnp.zeros([mf + 1, 2 * nf + 1])
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

    K_mntz = jnp.zeros([mf + 1, 2 * nf + 1, ntheta_sym * nzeta])
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
    K_mntz = K_mntz.reshape(mf + 1, 2 * nf + 1, ntheta_sym, nzeta)
    return I_mn, K_mntz


def regularizedFourierTransforms(
    coords,
    normal,
    jacobian,
    B_field,
    tan_theta,
    tan_zeta,
    mf,
    nf,
    ntheta,
    nzeta,
    NFP,
    weights,
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
    mf, nf : integer
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
            g_tzmn[:ntheta_sym, :nzeta, : mf + 1, : nf + 1].imag,
            g_tzmn[:ntheta_sym, :nzeta, : mf + 1, -nf:].imag,
        ],
        axis=-1,
    ).transpose((2, 3, 0, 1))

    # source term for integral equation, ie h_mn from Merkel 1986
    bexni = -weights * B_field["Bn"] * 4.0 * jnp.pi * jnp.pi
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
        [h_mn[: mf + 1, : nf + 1].imag, h_mn[: mf + 1, -nf:].imag], axis=1
    )

    return g_mntz, h_mn


def compute_scalar_magnetic_potential(
    I_mn, K_mntz, g_mntz, h_mn, mf, nf, ntheta, nzeta, weights, sym
):
    """Computes the magnetic scalar potential to cancel the normal field on the surface.

    Parameters
    ----------
    I_mn : ndarray
        singular part of source term, indexed by m, n
    K_mntz : ndarray
        singular part of greens function kernel, indexed by m, n, theta, zeta
    g_mntz : ndarray
        regularized part of greens function kernel, indexed by m, n, theta, zeta
    h_mn : ndarray
        regularized part of source term, indexed by m, n
    mf, nf : integer
        maximum poloidal and toroidal mode numbers
    ntheta, nzeta : integer
        number of grid points in poloidal, toroidal directions
    weights : ndarray
        quadrature weights for integration

    Returns
    -------
    phi_mn : ndarray
        scalar magnetic potential, indexed by m, n
    """
    ntheta_sym = ntheta // 2 + 1 if sym else ntheta

    # add in analytic part to get full kernel
    g_mntz = g_mntz + K_mntz
    # compute Fourier transform of grpmn to arrive at amatrix
    g_mntz = g_mntz * weights.reshape([1, 1, ntheta_sym, nzeta])
    g_mntz = jnp.pad(g_mntz, ((0, 0), (0, 0), (0, ntheta - ntheta_sym), (0, 0)))
    g_mnmn = jnp.fft.ifft(g_mntz, axis=2) * ntheta
    g_mnmn = jnp.fft.fft(g_mnmn, axis=3)

    amatrix_4d = jnp.concatenate(
        [g_mnmn[:, :, : mf + 1, : nf + 1].imag, g_mnmn[:, :, : mf + 1, -nf:].imag],
        axis=-1,
    )
    # scale amatrix by (2 pi)^2, copied from fortran
    amatrix_4d *= (2.0 * jnp.pi) ** 2
    m, n = jnp.meshgrid(jnp.arange(mf + 1), jnp.arange(2 * nf + 1), indexing="ij")
    # zero out (m=0, n<0, m', n') modes for all m', n', from fortran
    amatrix_4d = jnp.where(
        jnp.logical_and(m == 0, n > nf)[:, :, jnp.newaxis, jnp.newaxis], 0, amatrix_4d
    )
    # add diagonal terms, copied from fortran
    amatrix_4d = put(
        amatrix_4d, Index[m, n, m, n], amatrix_4d[m, n, m, n] + 4.0 * jnp.pi**3
    )

    amatrix = amatrix_4d.reshape([(mf + 1) * (2 * nf + 1), (mf + 1) * (2 * nf + 1)])

    # combine with contribution from analytic integral; available here in I_mn
    bvec = h_mn + I_mn
    # final fixup from fouri: zero out (m=0, n<0) components, from fortran
    bvec = put(bvec, Index[0, nf + 1 :], 0.0).flatten()

    phi_mn = jnp.linalg.solve(amatrix, bvec).reshape([mf + 1, 2 * nf + 1])
    return phi_mn


def compute_vacuum_magnetic_field(
    coords, normal, jacobian, B_field, phi_mn, mf, nf, ntheta, nzeta, NFP, sym
):
    """Computes vacuum magnetic field on plasma boundary.

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
    phi_mn : ndarray
        scalar magnetic potential, indexed by m, n
    mf, nf : integer
        maximum poloidal and toroidal mode numbers
    ntheta, nzeta : integer
        number of grid points in poloidal, toroidal directions
    NFP : integer
        number of field periods

    Returns
    -------
    Btot : dict of ndarray
        total field on plasma boundary from coils, plasma current, and scalar potential
    """
    ntheta_sym = ntheta // 2 + 1 if sym else ntheta

    potvac = phi_mn
    m_potvac = jnp.zeros([ntheta, nzeta])  # m*potvac --> for poloidal derivative
    n_potvac = jnp.zeros([ntheta, nzeta])  # n*potvac --> for toroidal derivative
    m, n = jnp.meshgrid(jnp.arange(mf + 1), jnp.arange(nf + 1), indexing="ij")

    m_potvac = put(m_potvac, Index[m, n], m * potvac[m, n])
    n_potvac = put(n_potvac, Index[m, n], n * potvac[m, n])
    m_potvac = put(m_potvac, Index[m, -n], m * potvac[m, -n])
    n_potvac = put(n_potvac, Index[m, -n], -n * potvac[m, -n])

    Bpot_theta = jnp.fft.ifft(m_potvac, axis=0) * ntheta
    Bpot_theta = (jnp.fft.fft(Bpot_theta, axis=1).real[:ntheta_sym, :]).flatten()

    Bpot_zeta = jnp.fft.ifft(n_potvac, axis=0) * ntheta
    Bpot_zeta = -(jnp.fft.fft(Bpot_zeta, axis=1).real[:ntheta_sym, :] * NFP).flatten()

    Btot = {}
    Btot["Bpot_theta"] = Bpot_theta
    Btot["Bpot_zeta"] = Bpot_zeta
    # compute covariant magnetic field components: B_u, B_v
    Btot["Bex_theta"] = coords["R_t"] * B_field["BR"] + coords["Z_t"] * B_field["BZ"]
    Btot["Bex_zeta"] = (
        coords["R_z"] * B_field["BR"]
        + coords["R"] * B_field["Bphi"]
        + coords["Z_z"] * B_field["BZ"]
    )

    Btot["B_theta"] = Btot["Bpot_theta"] + Btot["Bex_theta"]
    Btot["B_zeta"] = Btot["Bpot_zeta"] + Btot["Bex_zeta"]

    h_tz = NFP * jacobian["g_tz"]
    h_zz = jacobian["g_zz"] * NFP**2
    det = 1.0 / (jacobian["g_tt"] * h_zz - h_tz**2)

    Btot["B^theta"] = (h_zz * Btot["B_theta"] - h_tz * Btot["B_zeta"]) * det
    Btot["B^zeta"] = (-h_tz * Btot["B_theta"] + jacobian["g_tt"] * Btot["B_zeta"]) * det
    Btot["|B|^2"] = Btot["B_theta"] * Btot["B^theta"] + Btot["B_zeta"] * Btot["B^zeta"]
    Btot["Bex^theta"] = (h_zz * Btot["Bex_theta"] - h_tz * Btot["Bex_zeta"]) * det
    Btot["Bex^zeta"] = (
        -h_tz * Btot["Bex_theta"] + jacobian["g_tt"] * Btot["Bex_zeta"]
    ) * det
    Btot["|Bex|^2"] = (
        Btot["Bex_theta"] * Btot["Bex^theta"] + Btot["Bex_zeta"] * Btot["Bex^zeta"]
    )
    Btot["Bpot^theta"] = (h_zz * Btot["Bpot_theta"] - h_tz * Btot["Bpot_zeta"]) * det
    Btot["Bpot^zeta"] = (
        -h_tz * Btot["Bpot_theta"] + jacobian["g_tt"] * Btot["Bpot_zeta"]
    ) * det
    Btot["|Bpot|^2"] = (
        Btot["Bpot_theta"] * Btot["Bpot^theta"] + Btot["Bpot_zeta"] * Btot["Bpot^zeta"]
    )

    # compute cylindrical components B^R, B^\phi, B^Z
    Btot["BR"] = coords["R_t"] * Btot["B^theta"] + coords["R_z"] * Btot["B^zeta"]
    Btot["Bphi"] = coords["R"] * Btot["B^zeta"]
    Btot["BZ"] = coords["Z_t"] * Btot["B^theta"] + coords["Z_z"] * Btot["B^zeta"]
    Btot["BX"] = Btot["BR"] * jnp.cos(coords["phi"]) - Btot["Bphi"] * jnp.sin(
        coords["phi"]
    )
    Btot["BY"] = (
        Btot["BR"] * jnp.sin(coords["phi"]) + Btot["Bphi"] * jnp.cos(coords["phi"]),
    )
    Btot["Bn"] = (
        normal["R_n"] * Btot["BR"]
        + normal["phi_n"] * Btot["Bphi"]
        + normal["Z_n"] * Btot["BZ"]
    )

    return Btot


class Nestor(IOAble):
    """Neumann Solver for Toroidal Systems.

    Parameters
    ----------
    equil : Equilibrium
        equilibrium being optimized
    ext_field : MagneticField
        external field object, either splined or coils
    M, N : integer
        maximum poloidal and toroidal mode numbers to use
    ntheta, nzeta : int
        number of grid points in poloidal, toroidal directions to use
    field_grid : Grid, optional
        Grid used to discretize external field.
    """

    _static_attrs = ["M", "N", "ntheta", "nzeta", "sym", "NFP"]

    def __init__(
        self, equil, ext_field, M=None, N=None, ntheta=None, nzeta=None, field_grid=None
    ):

        M = setdefault(M, equil.M + 1)
        N = setdefault(N, equil.N)
        ntheta = setdefault(ntheta, 2 * M + 6)
        nzeta = setdefault(nzeta, 2 * M + 6)

        self.ext_field = ext_field
        self.field_grid = field_grid
        self.signgs = np.sign(np.mean(equil.compute("sqrt(g)")["sqrt(g)"]))
        self.M = M
        self.N = N
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.NFP = int(equil.NFP)
        self.sym = False  # hard-coded for now, can be generalized later
        ntheta_sym = self.ntheta // 2 + 1 if self.sym else self.ntheta

        bdry_grid = LinearGrid(rho=1, theta=ntheta, zeta=nzeta, NFP=self.NFP)
        axis_grid = LinearGrid(rho=0, theta=0, zeta=nzeta, NFP=self.NFP)
        self._Ra_transform = Transform(axis_grid, equil.R_basis)
        self._Za_transform = Transform(axis_grid, equil.Z_basis)
        self._Rb_transform = Transform(bdry_grid, equil.R_basis, derivs=2)
        self._Zb_transform = Transform(bdry_grid, equil.Z_basis, derivs=2)

        weights = 2 * np.ones((ntheta_sym, self.nzeta)) / (self.ntheta * self.nzeta)
        weights[0] /= 2.0
        weights[-1] /= 2.0
        self.weights = weights.flatten()

        # pre-computable quantities and arrays
        # tanu, tanv
        epstan = 2.22e-16
        bigno = 1.0e50  # allows proper comparison against implementation used in VMEC
        # setting bigno = np.inf allows proper plotting

        self.tanu = 2.0 * np.tan(np.pi * np.arange(2 * self.ntheta) / self.ntheta)
        # mask explicit singularities at tan(pi/2), tan(3/2 pi)
        self.tanu = np.where(
            (np.arange(2 * self.ntheta) / self.ntheta - 0.5) % 1 < epstan,
            bigno,
            self.tanu,
        )

        if self.nzeta == 1:
            # Tokamak: need NFP_eff toroidal grid points
            NFP_eff = 64
            argv = np.arange(NFP_eff) / NFP_eff
        else:
            # Stellarator: need nzeta toroidal grid points
            argv = np.arange(self.nzeta) / self.nzeta

        self.tanv = 2.0 * np.tan(np.pi * argv)
        # mask explicit singularities at tan(pi/2)
        self.tanv = np.where((argv - 0.5) % 1 < epstan, bigno, self.tanv)

        cmn = np.zeros([self.M + self.N + 1, self.M + 1, self.N + 1])
        for m in range(self.M + 1):
            for n in range(self.N + 1):
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
        dPhi_per = 2.0 * np.pi / self.NFP
        # cmns from cmn
        self.cmns = np.zeros([self.M + self.N + 1, self.M + 1, self.N + 1])
        for m in range(1, self.M + 1):
            for n in range(1, self.N + 1):
                self.cmns[:, m, n] = (
                    0.5
                    * dPhi_per
                    * (
                        cmn[:, m, n]
                        + cmn[:, m - 1, n]
                        + cmn[:, m, n - 1]
                        + cmn[:, m - 1, n - 1]
                    )
                )
        self.cmns[:, 1 : self.M + 1, 0] = (
            0.5 * dPhi_per * (cmn[:, 1 : self.M + 1, 0] + cmn[:, 0 : self.M, 0])
        )
        self.cmns[:, 0, 1 : self.N + 1] = (
            0.5 * dPhi_per * (cmn[:, 0, 1 : self.N + 1] + cmn[:, 0, 0 : self.N])
        )
        self.cmns[:, 0, 0] = 0.5 * dPhi_per * (cmn[:, 0, 0] + cmn[:, 0, 0])
        self.cmns = jnp.asarray(self.cmns)
        self.tanu = jnp.asarray(self.tanu)
        self.tanv = jnp.asarray(self.tanv)

    def eval_external_field(self, coords, normal, params=None):
        """Wrapper for handling fields from different coil types."""
        surf_coords = jnp.array([coords["R"], coords["phi"], coords["Z"]]).T
        B = self.ext_field.compute_magnetic_field(
            surf_coords, params=params, source_grid=self.field_grid
        ).T
        B_ex = {}
        B_ex["BR"] = B[0]
        B_ex["Bphi"] = B[1]
        B_ex["BZ"] = B[2]
        B_ex["BX"] = B[0] * jnp.cos(coords["phi"]) - B[1] * jnp.sin(coords["phi"])
        B_ex["BY"] = B[0] * jnp.sin(coords["phi"]) + B[1] * jnp.cos(coords["phi"])
        B_ex["Bn"] = (
            normal["R_n"] * B_ex["BR"]
            + normal["phi_n"] * B_ex["Bphi"]
            + normal["Z_n"] * B_ex["BZ"]
        )

        return B_ex

    def compute(self, R_lmn, Z_lmn, current, field_params=None):
        """Compute B^2 in the vacuum region and the scalar potential."""
        surface_coords = eval_surface_geometry(
            R_lmn,
            Z_lmn,
            self._Rb_transform,
            self._Zb_transform,
            self.ntheta,
            self.nzeta,
            self.NFP,
            self.sym,
        )
        axis_coords = eval_axis_geometry(
            R_lmn, Z_lmn, self._Ra_transform, self._Za_transform, self.nzeta, self.NFP
        )
        normal = compute_normal(surface_coords, self.signgs)
        jacobian = compute_jacobian(surface_coords, normal, self.NFP)
        B_extern = self.eval_external_field(surface_coords, normal, field_params)
        B_plasma = modelNetToroidalCurrent(
            axis_coords,
            current,
            surface_coords,
            normal,
        )
        B_field = {key: (B_extern[key] + B_plasma[key]) for key in B_extern.keys()}
        TS = compute_T_S(jacobian, self.M, self.N, self.ntheta, self.nzeta, self.sym)
        I_mn, K_mntz = compute_analytic_integrals(
            normal,
            jacobian,
            TS,
            B_field,
            self.M,
            self.N,
            self.ntheta,
            self.nzeta,
            self.cmns,
            self.weights,
            self.sym,
        )
        g_mntz, h_mn = regularizedFourierTransforms(
            surface_coords,
            normal,
            jacobian,
            B_field,
            self.tanu,
            self.tanv,
            self.M,
            self.N,
            self.ntheta,
            self.nzeta,
            self.NFP,
            self.weights,
            self.sym,
        )
        phi_mn = compute_scalar_magnetic_potential(
            I_mn,
            K_mntz,
            g_mntz,
            h_mn,
            self.M,
            self.N,
            self.ntheta,
            self.nzeta,
            self.weights,
            self.sym,
        )
        Btot = compute_vacuum_magnetic_field(
            surface_coords,
            normal,
            jacobian,
            B_field,
            phi_mn,
            self.M,
            self.N,
            self.ntheta,
            self.nzeta,
            self.NFP,
            self.sym,
        )
        return phi_mn, Btot
