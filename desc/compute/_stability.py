"""Compute functions for Mercier stability objectives."""

from scipy.constants import mu_0

from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import dot, surface_integrals


@register_compute_fun(
    name="D_shear",
    label="D_{shear}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion magnetic shear term",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["iota_r", "psi_r"],
)
def _D_shear(params, transforms, profiles, data, **kwargs):
    # Implements equations 4.16 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    data["D_shear"] = (data["iota_r"] / (4 * jnp.pi * data["psi_r"])) ** 2
    return data


@register_compute_fun(
    name="D_current",
    label="D_{current}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion toroidal current term",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["psi_r", "iota_r", "B", "J", "G", "I_r", "|grad(psi)|", "|e_theta x e_zeta|"],
)
def _D_current(params, transforms, profiles, data, **kwargs):
    # Implements equations 4.17 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    Xi = mu_0 * data["J"] - jnp.atleast_2d(data["I_r"] / data["psi_r"]).T * data["B"]
    data["D_current"] = (
        -jnp.sign(data["G"])
        / (2 * jnp.pi) ** 4
        * data["iota_r"]
        / data["psi_r"]
        * surface_integrals(
            transforms["grid"],
            data["|e_theta x e_zeta|"] / data["|grad(psi)|"] ** 3 * dot(Xi, data["B"]),
        )
    )
    return data


@register_compute_fun(
    name="D_well",
    label="D_{well}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion magnetic well term",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "p_r",
        "psi",
        "psi_r",
        "psi_rr",
        "V_rr(r)",
        "V_r(r)",
        "|B|^2",
        "|grad(psi)|",
        "|e_theta x e_zeta|",
    ],
)
def _D_well(params, transforms, profiles, data, **kwargs):
    # Implements equations 4.18 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    dp_dpsi = mu_0 * data["p_r"] / data["psi_r"]
    d2V_dpsi2 = (
        data["V_rr(r)"] - data["V_r(r)"] * data["psi_rr"] / data["psi_r"]
    ) / data["psi_r"] ** 2
    data["D_well"] = (
        dp_dpsi
        * (
            jnp.sign(data["psi"]) * d2V_dpsi2
            - dp_dpsi
            * surface_integrals(
                transforms["grid"],
                data["|e_theta x e_zeta|"] / (data["|B|^2"] * data["|grad(psi)|"]),
            )
        )
        * surface_integrals(
            transforms["grid"],
            data["|e_theta x e_zeta|"] * data["|B|^2"] / data["|grad(psi)|"] ** 3,
        )
        / (2 * jnp.pi) ** 6
    )
    return data


@register_compute_fun(
    name="D_geodesic",
    label="D_{geodesic}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion geodesic curvature term",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|grad(psi)|", "J*B", "|B|^2", "|e_theta x e_zeta|"],
)
def _D_geodesic(params, transforms, profiles, data, **kwargs):
    # Implements equations 4.19 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    data["D_geodesic"] = (
        surface_integrals(
            transforms["grid"],
            data["|e_theta x e_zeta|"] * mu_0 * data["J*B"] / data["|grad(psi)|"] ** 3,
        )
        ** 2
        - surface_integrals(
            transforms["grid"],
            data["|e_theta x e_zeta|"] * data["|B|^2"] / data["|grad(psi)|"] ** 3,
        )
        * surface_integrals(
            transforms["grid"],
            data["|e_theta x e_zeta|"]
            * mu_0**2
            * data["J*B"] ** 2
            / (data["|B|^2"] * data["|grad(psi)|"] ** 3),
        )
    ) / (2 * jnp.pi) ** 6
    return data


@register_compute_fun(
    name="D_Mercier",
    label="D_{Mercier}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion (positive/negative value "
    + "denotes stability/instability)",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["D_shear", "D_current", "D_well", "D_geodesic"],
)
def _D_Mercier(params, transforms, profiles, data, **kwargs):
    # Implements equations 4.20 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    data["D_Mercier"] = (
        data["D_shear"] + data["D_current"] + data["D_well"] + data["D_geodesic"]
    )
    return data


@register_compute_fun(
    name="magnetic well",
    label="Magnetic Well",
    units="~",
    units_long="None",
    description="Magnetic well proxy for MHD stability (positive/negative value "
    + "denotes stability/instability)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "V(r)",
        "V_r(r)",
        "p_r",
        "<B^2>",
        "<B^2>_r",
    ],
)
def _magnetic_well(params, transforms, profiles, data, **kwargs):
    # Implements equation 3.2 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    # pressure = thermal + magnetic = 2 mu_0 p + B^2
    # The surface average operation is an additive homomorphism.
    # Thermal pressure is constant over a rho surface.
    # surface average(pressure) = thermal + surface average(magnetic)
    data["magnetic well"] = (
        data["V(r)"]
        * (2 * mu_0 * data["p_r"] + data["<B^2>_r"])
        / (data["V_r(r)"] * data["<B^2>"])
    )
    return data
