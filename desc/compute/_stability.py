"""Compute functions for Mercier stability objectives."""

from scipy.constants import mu_0

from desc.backend import jnp

from ._core import compute_geometry, compute_pressure, compute_toroidal_flux_gradient
from ._field import (
    compute_boozer_magnetic_field,
    compute_contravariant_current_density,
    compute_magnetic_field_magnitude,
)
from .utils import check_derivs, dot, surface_averages, surface_integrals


def compute_mercier_stability(params, transforms, profiles, data=None, **kwargs):
    """Compute the Mercier stability criterion.

    Notes
    -----
        Implements equations 4.16 through 4.20 in M. Landreman & R. Jorge (2020)
        doi:10.1017/S002237782000121X.
    """
    grid = transforms["R"].grid
    data = compute_pressure(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_toroidal_flux_gradient(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_geometry(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_magnetic_field_magnitude(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_boozer_magnetic_field(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_contravariant_current_density(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    dS = jnp.abs(data["sqrt(g)"]) * data["|grad(rho)|"]
    grad_psi_3 = data["|grad(psi)|"] ** 3

    if check_derivs("D_shear", transforms["R"], transforms["Z"], transforms["L"]):
        data["D_shear"] = (data["iota_r"] / (4 * jnp.pi * data["psi_r"])) ** 2

    if check_derivs("D_current", transforms["R"], transforms["Z"], transforms["L"]):
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

    if check_derivs("D_well", transforms["R"], transforms["Z"], transforms["L"]):
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

    if check_derivs("D_geodesic", transforms["R"], transforms["Z"], transforms["L"]):
        J_dot_B = mu_0 * dot(data["J"], data["B"])
        data["D_geodesic"] = (
            surface_integrals(grid, dS * J_dot_B / grad_psi_3) ** 2
            - surface_integrals(grid, dS * data["|B|^2"] / grad_psi_3)
            * surface_integrals(grid, dS * J_dot_B**2 / (data["|B|^2"] * grad_psi_3))
        ) / (2 * jnp.pi) ** 6

    if check_derivs("D_Mercier", transforms["R"], transforms["Z"], transforms["L"]):
        data["D_Mercier"] = (
            data["D_shear"] + data["D_current"] + data["D_well"] + data["D_geodesic"]
        )

    return data


def compute_magnetic_well(params, transforms, profiles, data=None, **kwargs):
    """Compute the magnetic well proxy for MHD stability.

    Notes
    -----
        Implements equation 3.2 in M. Landreman & R. Jorge (2020)
        doi:10.1017/S002237782000121X.
    """
    grid = transforms["R"].grid
    data = compute_pressure(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_geometry(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_magnetic_field_magnitude(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    if check_derivs("magnetic well", transforms["R"], transforms["Z"], transforms["L"]):
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
        dB2_avg_drho = (
            surface_integrals(
                grid,
                data["sqrt(g)_r"] * jnp.sign(data["sqrt(g)"]) * data["|B|^2"]
                + jnp.abs(data["sqrt(g)"]) * 2 * dot(data["B"], data["B_r"]),
            )
            - data["V_rr(r)"] * B2_avg
        ) / data["V_r(r)"]
        data["magnetic well"] = (
            data["V(r)"] * (dp_drho + dB2_avg_drho) / (data["V_r(r)"] * B2_avg)
        )

    return data
