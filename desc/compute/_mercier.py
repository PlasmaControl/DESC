"""Compute functions for Mercier stability objectives."""

from scipy.constants import mu_0

from desc.backend import jnp
from desc.compute import (
    compute_contravariant_current_density,
    compute_contravariant_magnetic_field,
    compute_contravariant_metric_coefficients,
    compute_pressure,
    compute_rotational_transform,
    compute_toroidal_flux,
    dot,
)
from desc.compute.utils import (
    compress,
    enclosed_volumes,
    expand,
    surface_averages,
    surface_integrals,
)
from desc.compute._core import check_derivs


def compute_DMerc(
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    c_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    pressure,
    iota,
    current,
    data=None,
):
    """Compute the Mercier stability criterion.

    See equation 4.16 in
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
    p_l : ndarray
        Spectral coefficients of p(rho) -- pressure profile.
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    c_l : ndarray
        Spectral coefficients of I(rho) -- toroidal current profile.
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface (Wb).
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.
    L_transform : Transform
        Transforms L_lmn coefficients to real space.
    pressure : Profile
        Transforms p_l coefficients to real space.
    iota : Profile
        Transforms i_l coefficients to real space.
    current : Profile
        Transforms c_l coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of Mercier criterion terms.
    """
    if check_derivs("DMerc", R_transform, Z_transform, L_transform):
        data = compute_DShear(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            current,
            data,
        )
        data = compute_DCurr(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            current,
            data,
        )
        data = compute_DWell(
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            c_l,
            Psi,
            R_transform,
            Z_transform,
            L_transform,
            pressure,
            iota,
            current,
            data,
        )
        data = compute_DGeod(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            current,
            data,
        )
        data["DMerc"] = data["DShear"] + data["DCurr"] + data["DWell"] + data["DGeod"]
    return data


def compute_DShear(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    c_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    current,
    data=None,
):
    """Compute the Mercier stability criterion magnetic sheer term.

    See equation 4.17 in
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    c_l : ndarray
        Spectral coefficients of I(rho) -- toroidal current profile.
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface (Wb).
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.
    L_transform : Transform
        Transforms L_lmn coefficients to real space.
    iota : Profile
        Transforms i_l coefficients to real space.
    current : Profile
        Transforms c_l coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of Mercier criterion magnetic sheer term.
    """
    if check_derivs("DShear", R_transform, Z_transform, L_transform):
        data = compute_toroidal_flux(Psi, R_transform.grid, data=data)
        data = compute_rotational_transform(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            current,
            data=data,
        )
        data["DShear"] = jnp.square(data["iota_r"] / data["psi_r"]) / (16 * jnp.pi ** 2)
    return data


def compute_DCurr(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    c_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    current,
    data=None,
):
    """Compute the Mercier stability criterion toroidal current term.

    See equation 4.18 in
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    c_l : ndarray
        Spectral coefficients of I(rho) -- toroidal current profile.
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface (Wb).
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.
    L_transform : Transform
        Transforms L_lmn coefficients to real space.
    iota : Profile
        Transforms i_l coefficients to real space.
    current : Profile
        Transforms c_l coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of Mercier criterion toroidal current term.
    """
    if check_derivs("DCurr", R_transform, Z_transform, L_transform):
        grid = R_transform.grid
        data = compute_contravariant_metric_coefficients(
            R_lmn, Z_lmn, R_transform, Z_transform, data
        )
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            current,
            data,
        )

        # grad(psi) = grad(rho) * dpsi/drho
        # A * dtdz = |ds / grad(psi)^3| = |sqrt(g) * grad(rho) / grad(psi)^3| * dtdz
        A = jnp.abs(data["sqrt(g)"] / data["psi_r"] ** 3) / data["g^rr"]
        dI_dpsi = surface_averages(
            grid,
            data["B_theta_r"] / data["psi_r"],
            match_grid=True,
        )
        xi = mu_0 * data["J"] - jnp.atleast_2d(dI_dpsi).T * data["B"]
        # sign of poloidal current, data["G"]
        sign_G = jnp.sign(surface_integrals(grid, data["B_zeta"], match_grid=True))

        data["DCurr"] = (
            -sign_G
            / (16 * jnp.pi ** 4)
            * data["iota_r"]
            / data["psi_r"]
            * surface_integrals(grid, A * dot(xi, data["B"]), match_grid=True)
        )
    return data


def compute_DWell(
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    c_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    pressure,
    iota,
    current,
    data=None,
):
    """Compute the Mercier stability criterion magnetic well term.

    See equation 4.19 in
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
    p_l : ndarray
        Spectral coefficients of p(rho) -- pressure profile.
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    c_l : ndarray
        Spectral coefficients of I(rho) -- toroidal current profile.
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface (Wb).
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.
    L_transform : Transform
        Transforms L_lmn coefficients to real space.
    pressure : Profile
        Transforms p_l coefficients to real space.
    iota : Profile
        Transforms i_l coefficients to real space.
    current : Profile
        Transforms c_l coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of Mercier criterion magnetic well term.
    """
    if check_derivs("DWell", R_transform, Z_transform, L_transform):
        grid = R_transform.grid
        data = compute_pressure(p_l, pressure, data)
        data = compute_contravariant_metric_coefficients(
            R_lmn, Z_lmn, R_transform, Z_transform, data
        )
        data = compute_contravariant_magnetic_field(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            current,
            data,
        )

        psi_r_sq = jnp.square(data["psi_r"])
        grad_psi_sq = data["g^rr"] * psi_r_sq  # grad(psi) = grad(rho) * dpsi/drho
        # A * dtdz = |ds / grad(psi)| = |sqrt(g) * grad(rho) / grad(psi)| * dtdz
        A = jnp.abs(data["sqrt(g)"] / data["psi_r"])

        dv_drho = enclosed_volumes(grid, data, dr=1, match_grid=True)
        d2v_drho2 = enclosed_volumes(grid, data, dr=2, match_grid=True)
        d2v_dpsi2 = (d2v_drho2 - dv_drho * data["psi_rr"] / data["psi_r"]) / psi_r_sq
        dp_dpsi = data["p_r"] / data["psi_r"]
        Bsq = dot(data["B"], data["B"])

        data["DWell"] = (
            mu_0
            / (64 * jnp.pi ** 6)
            * dp_dpsi
            * (
                jnp.sign(data["psi"]) * d2v_dpsi2
                - mu_0 * dp_dpsi * surface_integrals(grid, A / Bsq, match_grid=True)
            )
            * surface_integrals(grid, A / grad_psi_sq * Bsq, match_grid=True)
        )
    return data


def compute_DGeod(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    c_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    current,
    data=None,
):
    """Compute the Mercier stability criterion geodesic curvature term.

    See equation 4.20 in
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    c_l : ndarray
        Spectral coefficients of I(rho) -- toroidal current profile.
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface (Wb).
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.
    L_transform : Transform
        Transforms L_lmn coefficients to real space.
    iota : Profile
        Transforms i_l coefficients to real space.
    current : Profile
        Transforms c_l coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of Mercier criterion geodesic curvature term.
    """
    if check_derivs("DGeod", R_transform, Z_transform, L_transform):
        grid = R_transform.grid
        data = compute_contravariant_metric_coefficients(
            R_lmn, Z_lmn, R_transform, Z_transform, data
        )
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            current,
            data,
        )

        # grad(psi) = grad(rho) * dpsi/drho
        # A * dtdz = |ds / grad(psi)^3| = |sqrt(g) * grad(rho) / grad(psi)^3| * dtdz
        A = jnp.abs(data["sqrt(g)"] / data["psi_r"] ** 3) / data["g^rr"]
        j_dot_b = mu_0 * dot(data["J"], data["B"])
        Bsq = dot(data["B"], data["B"])

        data["DGeod"] = (
            expand(
                grid,
                jnp.square(surface_integrals(grid, A * j_dot_b))
                - surface_integrals(grid, A * Bsq)
                * surface_integrals(grid, A * jnp.square(j_dot_b) / Bsq),
            )
            / (64 * jnp.pi ** 6)
        )
    return data


def compute_AltWell(
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    c_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    pressure,
    iota,
    current,
    data=None,
):
    """Compute an alternative magnetic well parameter.

    See equation 3.2 in
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
    p_l : ndarray
        Spectral coefficients of p(rho) -- pressure profile.
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    c_l : ndarray
        Spectral coefficients of I(rho) -- toroidal current profile.
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface (Wb).
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.
    L_transform : Transform
        Transforms L_lmn coefficients to real space.
    pressure : Profile
        Transforms p_l coefficients to real space.
    iota : Profile
        Transforms i_l coefficients to real space.
    current : Profile
        Transforms c_l coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of the magnetic well parameter.
    """
    if check_derivs("AltWell", R_transform, Z_transform, L_transform):
        grid = R_transform.grid
        data = compute_pressure(p_l, pressure, data)
        data = compute_contravariant_magnetic_field(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            R_transform,
            Z_transform,
            L_transform,
            iota,
            current,
            data,
        )

        dv_drho = enclosed_volumes(grid, data, dr=1)
        sqrtg = jnp.abs(data["sqrt(g)"])
        sqrtg_r = jnp.abs(data["sqrt(g)_r"])
        Bsq = dot(data["B"], data["B"])
        Bsq_av = surface_averages(grid, Bsq, sqrtg, denominator=dv_drho)

        # pressure = thermal + magnetic
        # The flux surface average function is an additive homomorphism.
        # This means average(a + b) = average(a) + average(b).
        # Thermal pressure is constant over a rho surface.
        # Therefore average(thermal) = thermal.
        dthermal_drho = 2 * mu_0 * compress(grid, data["p_r"])
        dmagnetic_av_drho = (
            surface_integrals(
                grid, sqrtg_r * Bsq + sqrtg * 2 * dot(data["B"], data["B_r"])
            )
            - surface_integrals(grid, sqrtg_r) * Bsq_av
        ) / dv_drho

        V = enclosed_volumes(grid, data)
        data["AltWell"] = expand(
            grid, V * (dthermal_drho + dmagnetic_av_drho) / dv_drho / Bsq_av
        )
        # alternative that avoids computing the volume
        # matches above implementation to a scale factor
        # data["Well"] = data["rho"] * expand(grid, (dthermal_drho + dmagnetic_av_drho) / Bsq_av)
    return data
