"""Compute functions for Mercier stability objectives."""

from scipy.constants import mu_0

from desc.backend import jnp
from .utils import (
    check_derivs,
    dot,
    surface_averages,
    surface_integrals,
)
from ._core import compute_pressure, compute_toroidal_flux_gradient, compute_geometry
from ._field import (
    compute_magnetic_field_magnitude,
    compute_boozer_magnetic_field,
    compute_contravariant_current_density,
)


def compute_mercier_stability(
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    pressure,
    iota,
    data=None,
):
    """Compute the Mercier stability criterion.

    Notes
    -----
        Implements equations 4.16 through 4.20 in M. Landreman & R. Jorge (2020)
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

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of Mercier criterion terms.
        Keys are 'D_shear', 'D_current', 'D_well', 'D_geodesic', and 'D_Mercier'.

    """
    grid = R_transform.grid
    data = compute_pressure(p_l, pressure, data=data)
    data = compute_toroidal_flux_gradient(
        R_lmn, Z_lmn, Psi, R_transform, Z_transform, data=data
    )
    data = compute_geometry(R_lmn, Z_lmn, R_transform, Z_transform, data=data)
    data = compute_magnetic_field_magnitude(
        R_lmn,
        Z_lmn,
        L_lmn,
        i_l,
        Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        data=data,
    )
    data = compute_boozer_magnetic_field(
        R_lmn,
        Z_lmn,
        L_lmn,
        i_l,
        Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        data,
    )
    data = compute_contravariant_current_density(
        R_lmn,
        Z_lmn,
        L_lmn,
        i_l,
        Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        data,
    )

    dS = jnp.abs(data["sqrt(g)"]) * data["|grad(rho)|"]
    grad_psi_3 = data["|grad(psi)|"] ** 3

    if check_derivs("D_shear", R_transform, Z_transform, L_transform):
        data["D_shear"] = (data["iota_r"] / (4 * jnp.pi * data["psi_r"])) ** 2

    if check_derivs("D_current", R_transform, Z_transform, L_transform):
        Xi = (
            mu_0 * data["J"] - jnp.atleast_2d(data["I_r"] / data["psi_r"]).T * data["B"]
        )
        data["D_current"] = (
            -jnp.sign(data["G"])
            / (2 * jnp.pi) ** 4
            * data["iota_r"]
            / data["psi_r"]
            * surface_integrals(grid, dS / grad_psi_3 * dot(Xi, data["B"]))
        )

    if check_derivs("D_well", R_transform, Z_transform, L_transform):
        dp_dpsi = mu_0 * data["p_r"] / data["psi_r"]
        d2V_dpsi2 = (
            data["V_rr(r)"] - data["V_r(r)"] * data["psi_rr"] / data["psi_r"]
        ) / data["psi_r"] ** 2
        data["D_well"] = (
            dp_dpsi
            * (
                jnp.sign(data["psi"]) * d2V_dpsi2
                - dp_dpsi
                * surface_integrals(grid, dS / (data["|B|^2"] * data["|grad(psi)|"]))
            )
            * surface_integrals(grid, dS * data["|B|^2"] / grad_psi_3)
            / (2 * jnp.pi) ** 6
        )

    if check_derivs("D_geodesic", R_transform, Z_transform, L_transform):
        J_dot_B = mu_0 * dot(data["J"], data["B"])
        data["D_geodesic"] = (
            surface_integrals(grid, dS * J_dot_B / grad_psi_3) ** 2
            - surface_integrals(grid, dS * data["|B|^2"] / grad_psi_3)
            * surface_integrals(grid, dS * J_dot_B ** 2 / (data["|B|^2"] * grad_psi_3))
        ) / (2 * jnp.pi) ** 6

    if check_derivs("D_Mercier", R_transform, Z_transform, L_transform):
        data["D_Mercier"] = (
            data["D_shear"] + data["D_current"] + data["D_well"] + data["D_geodesic"]
        )

    return data


def compute_magnetic_well(
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    pressure,
    iota,
    data=None,
):
    """Compute the magnetic well proxy for MHD stability.

    Notes
    -----
        Implements equation 3.2 in M. Landreman & R. Jorge (2020)
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

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of the magnetic well parameter.

    """
    grid = R_transform.grid
    data = compute_pressure(p_l, pressure, data=data)
    data = compute_geometry(R_lmn, Z_lmn, R_transform, Z_transform, data=data)
    data = compute_magnetic_field_magnitude(
        R_lmn,
        Z_lmn,
        L_lmn,
        i_l,
        Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        data=data,
    )

    if check_derivs("magnetic well", R_transform, Z_transform, L_transform):
        B2_avg = surface_averages(
            grid,
            data["|B|^2"],
            jnp.abs(data["sqrt(g)"]),
            denominator=data["V_r(r)"],
        )
        # pressure = thermal + magnetic = 2 mu_0 p + B^2
        # The surface average operation is an additive homomorphism.
        # Thermal pressure is constant over a rho surface.
        # surface average(pressure) = thermal + surface average(magnetic)
        dp_drho = 2 * mu_0 * data["p_r"]
        dB2_drho_avg = (
            surface_integrals(
                grid,
                jnp.abs(data["sqrt(g)_r"]) * data["|B|^2"]
                + jnp.abs(data["sqrt(g)"]) * 2 * dot(data["B"], data["B_r"]),
            )
            - surface_integrals(grid, jnp.abs(data["sqrt(g)_r"])) * B2_avg
        ) / data["V_r(r)"]
        data["magnetic well"] = (
            data["V(r)"] * (dp_drho + dB2_drho_avg) / (data["V_r(r)"] * B2_avg)
        )

        # equivalent method (besides scaling factor) that avoids computing the volume
        # data["magnetic well"] = data["rho"] * (dp_drho + dB2_drho_avg) / B2_avg

    return data
