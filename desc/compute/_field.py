"""Compute functions for magnetic field quantities."""

from scipy.constants import mu_0

from desc.backend import jnp

from ._core import (
    compute_contravariant_metric_coefficients,
    compute_covariant_metric_coefficients,
    compute_jacobian,
    compute_lambda,
    compute_rotational_transform,
    compute_toroidal_flux,
)
from .utils import dot, has_dependencies, surface_averages


def compute_contravariant_magnetic_field(  # noqa: C901
    params,
    transforms,
    profiles,
    data=None,
    **kwargs,
):
    """Compute contravariant magnetic field components."""
    data = compute_toroidal_flux(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_lambda(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_jacobian(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_rotational_transform(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    # 0th order terms
    if has_dependencies("B0", params, transforms, profiles, data):
        data["B0"] = data["psi_r"] / data["sqrt(g)"]
    if has_dependencies("B^rho", params, transforms, profiles, data):
        data["B^rho"] = data["0"]
    if has_dependencies("B^theta", params, transforms, profiles, data):
        data["B^theta"] = data["B0"] * (data["iota"] - data["lambda_z"])
    if has_dependencies("B^zeta", params, transforms, profiles, data):
        data["B^zeta"] = data["B0"] * (1 + data["lambda_t"])
    if has_dependencies("B", params, transforms, profiles, data):
        data["B"] = (
            data["B^theta"] * data["e_theta"].T + data["B^zeta"] * data["e_zeta"].T
        ).T
        data["B_R"] = data["B"][:, 0]
        data["B_phi"] = data["B"][:, 1]
        data["B_Z"] = data["B"][:, 2]

    # 1st order derivatives
    if has_dependencies("B0_r", params, transforms, profiles, data):
        data["B0_r"] = (
            data["psi_rr"] / data["sqrt(g)"]
            - data["psi_r"] * data["sqrt(g)_r"] / data["sqrt(g)"] ** 2
        )
    if has_dependencies("B^theta_r", params, transforms, profiles, data):
        data["B^theta_r"] = data["B0_r"] * (data["iota"] - data["lambda_z"]) + data[
            "B0"
        ] * (data["iota_r"] - data["lambda_rz"])
    if has_dependencies("B^zeta_r", params, transforms, profiles, data):
        data["B^zeta_r"] = (
            data["B0_r"] * (1 + data["lambda_t"]) + data["B0"] * data["lambda_rt"]
        )
    if has_dependencies("B_r", params, transforms, profiles, data):
        data["B_r"] = (
            data["B^theta_r"] * data["e_theta"].T
            + data["B^theta"] * data["e_theta_r"].T
            + data["B^zeta_r"] * data["e_zeta"].T
            + data["B^zeta"] * data["e_zeta_r"].T
        ).T
    if has_dependencies("B0_t", params, transforms, profiles, data):
        data["B0_t"] = -data["psi_r"] * data["sqrt(g)_t"] / data["sqrt(g)"] ** 2
    if has_dependencies("B^theta_t", params, transforms, profiles, data):
        data["B^theta_t"] = (
            data["B0_t"] * (data["iota"] - data["lambda_z"])
            - data["B0"] * data["lambda_tz"]
        )
    if has_dependencies("B^zeta_t", params, transforms, profiles, data):
        data["B^zeta_t"] = (
            data["B0_t"] * (1 + data["lambda_t"]) + data["B0"] * data["lambda_tt"]
        )
    if has_dependencies("B_t", params, transforms, profiles, data):
        data["B_t"] = (
            data["B^theta_t"] * data["e_theta"].T
            + data["B^theta"] * data["e_theta_t"].T
            + data["B^zeta_t"] * data["e_zeta"].T
            + data["B^zeta"] * data["e_zeta_t"].T
        ).T
    if has_dependencies("B0_z", params, transforms, profiles, data):
        data["B0_z"] = -data["psi_r"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 2
    if has_dependencies("B^theta_z", params, transforms, profiles, data):
        data["B^theta_z"] = (
            data["B0_z"] * (data["iota"] - data["lambda_z"])
            - data["B0"] * data["lambda_zz"]
        )
    if has_dependencies("B^zeta_z", params, transforms, profiles, data):
        data["B^zeta_z"] = (
            data["B0_z"] * (1 + data["lambda_t"]) + data["B0"] * data["lambda_tz"]
        )
    if has_dependencies("B_z", params, transforms, profiles, data):
        data["B_z"] = (
            data["B^theta_z"] * data["e_theta"].T
            + data["B^theta"] * data["e_theta_z"].T
            + data["B^zeta_z"] * data["e_zeta"].T
            + data["B^zeta"] * data["e_zeta_z"].T
        ).T

    # 2nd order derivatives
    if has_dependencies("B0_tt", params, transforms, profiles, data):
        data["B0_tt"] = -(
            data["psi_r"]
            / data["sqrt(g)"] ** 2
            * (data["sqrt(g)_tt"] - 2 * data["sqrt(g)_t"] ** 2 / data["sqrt(g)"])
        )
    if has_dependencies("B^theta_tt", params, transforms, profiles, data):
        data["B^theta_tt"] = (
            data["B0_tt"] * (data["iota"] - data["lambda_z"])
            - 2 * data["B0_t"] * data["lambda_tz"]
            - data["B0"] * data["lambda_ttz"]
        )
    if has_dependencies("B^zeta_tt", params, transforms, profiles, data):
        data["B^zeta_tt"] = (
            data["B0_tt"] * (1 + data["lambda_t"])
            + 2 * data["B0_t"] * data["lambda_tt"]
            + data["B0"] * data["lambda_ttt"]
        )
    if has_dependencies("B0_zz", params, transforms, profiles, data):
        data["B0_zz"] = -(
            data["psi_r"]
            / data["sqrt(g)"] ** 2
            * (data["sqrt(g)_zz"] - 2 * data["sqrt(g)_z"] ** 2 / data["sqrt(g)"])
        )
    if has_dependencies("B^theta_zz", params, transforms, profiles, data):
        data["B^theta_zz"] = (
            data["B0_zz"] * (data["iota"] - data["lambda_z"])
            - 2 * data["B0_z"] * data["lambda_zz"]
            - data["B0"] * data["lambda_zzz"]
        )
    if has_dependencies("B^zeta_zz", params, transforms, profiles, data):
        data["B^zeta_zz"] = (
            data["B0_zz"] * (1 + data["lambda_t"])
            + 2 * data["B0_z"] * data["lambda_tz"]
            + data["B0"] * data["lambda_tzz"]
        )
    if has_dependencies("B0_tz", params, transforms, profiles, data):
        data["B0_tz"] = -(
            data["psi_r"]
            / data["sqrt(g)"] ** 2
            * (
                data["sqrt(g)_tz"]
                - 2 * data["sqrt(g)_t"] * data["sqrt(g)_z"] / data["sqrt(g)"]
            )
        )
    if has_dependencies("B^theta_tz", params, transforms, profiles, data):
        data["B^theta_tz"] = (
            data["B0_tz"] * (data["iota"] - data["lambda_z"])
            - data["B0_t"] * data["lambda_zz"]
            - data["B0_z"] * data["lambda_tz"]
            - data["B0"] * data["lambda_tzz"]
        )
    if has_dependencies("B^zeta_tz", params, transforms, profiles, data):
        data["B^zeta_tz"] = (
            data["B0_tz"] * (1 + data["lambda_t"])
            + data["B0_t"] * data["lambda_tz"]
            + data["B0_z"] * data["lambda_tt"]
            + data["B0"] * data["lambda_ttz"]
        )

    return data


def compute_covariant_magnetic_field(
    params,
    transforms,
    profiles,
    data=None,
    **kwargs,
):
    """Compute covariant magnetic field components."""
    data = compute_contravariant_magnetic_field(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    # 0th order terms
    if has_dependencies("B_rho", params, transforms, profiles, data):
        data["B_rho"] = dot(data["B"], data["e_rho"])
    if has_dependencies("B_theta", params, transforms, profiles, data):
        data["B_theta"] = dot(data["B"], data["e_theta"])
    if has_dependencies("B_zeta", params, transforms, profiles, data):
        data["B_zeta"] = dot(data["B"], data["e_zeta"])

    # 1st order derivatives
    if has_dependencies("B_rho_r", params, transforms, profiles, data):
        data["B_rho_r"] = dot(data["B_r"], data["e_rho"]) + dot(
            data["B"], data["e_rho_r"]
        )
    if has_dependencies("B_theta_r", params, transforms, profiles, data):
        data["B_theta_r"] = dot(data["B_r"], data["e_theta"]) + dot(
            data["B"], data["e_theta_r"]
        )
    if has_dependencies("B_zeta_r", params, transforms, profiles, data):
        data["B_zeta_r"] = dot(data["B_r"], data["e_zeta"]) + dot(
            data["B"], data["e_zeta_r"]
        )
    if has_dependencies("B_rho_t", params, transforms, profiles, data):
        data["B_rho_t"] = dot(data["B_t"], data["e_rho"]) + dot(
            data["B"], data["e_rho_t"]
        )
    if has_dependencies("B_theta_t", params, transforms, profiles, data):
        data["B_theta_t"] = dot(data["B_t"], data["e_theta"]) + dot(
            data["B"], data["e_theta_t"]
        )
    if has_dependencies("B_zeta_t", params, transforms, profiles, data):
        data["B_zeta_t"] = dot(data["B_t"], data["e_zeta"]) + dot(
            data["B"], data["e_zeta_t"]
        )
    if has_dependencies("B_rho_z", params, transforms, profiles, data):
        data["B_rho_z"] = dot(data["B_z"], data["e_rho"]) + dot(
            data["B"], data["e_rho_z"]
        )
    if has_dependencies("B_theta_z", params, transforms, profiles, data):
        data["B_theta_z"] = dot(data["B_z"], data["e_theta"]) + dot(
            data["B"], data["e_theta_z"]
        )
    if has_dependencies("B_zeta_z", params, transforms, profiles, data):
        data["B_zeta_z"] = dot(data["B_z"], data["e_zeta"]) + dot(
            data["B"], data["e_zeta_z"]
        )

    return data


def compute_magnetic_field_magnitude(
    params,
    transforms,
    profiles,
    data=None,
    **kwargs,
):
    """Compute magnetic field magnitude."""
    data = compute_contravariant_magnetic_field(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_covariant_metric_coefficients(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    # TODO: would it be simpler to compute this as B^theta*B_theta+B^zeta*B_zeta?

    # 0th order term
    if has_dependencies("|B|^2", params, transforms, profiles, data):
        data["|B|^2"] = (
            data["B^theta"] ** 2 * data["g_tt"]
            + data["B^zeta"] ** 2 * data["g_zz"]
            + 2 * data["B^theta"] * data["B^zeta"] * data["g_tz"]
        )
        data["|B|"] = jnp.sqrt(data["|B|^2"])

    # 1st order derivatives
    # TODO: |B|_r
    if has_dependencies("|B|_t", params, transforms, profiles, data):
        data["|B|_t"] = (
            data["B^theta"]
            * (
                data["B^zeta_t"] * data["g_tz"]
                + data["B^theta_t"] * data["g_tt"]
                + data["B^theta"] * dot(data["e_theta_t"], data["e_theta"])
            )
            + data["B^zeta"]
            * (
                data["B^theta_t"] * data["g_tz"]
                + data["B^zeta_t"] * data["g_zz"]
                + data["B^zeta"] * dot(data["e_zeta_t"], data["e_zeta"])
            )
            + data["B^theta"]
            * data["B^zeta"]
            * (
                dot(data["e_theta_t"], data["e_zeta"])
                + dot(data["e_zeta_t"], data["e_theta"])
            )
        ) / data["|B|"]
    if has_dependencies("|B|_z", params, transforms, profiles, data):
        data["|B|_z"] = (
            data["B^theta"]
            * (
                data["B^zeta_z"] * data["g_tz"]
                + data["B^theta_z"] * data["g_tt"]
                + data["B^theta"] * dot(data["e_theta_z"], data["e_theta"])
            )
            + data["B^zeta"]
            * (
                data["B^theta_z"] * data["g_tz"]
                + data["B^zeta_z"] * data["g_zz"]
                + data["B^zeta"] * dot(data["e_zeta_z"], data["e_zeta"])
            )
            + data["B^theta"]
            * data["B^zeta"]
            * (
                dot(data["e_theta_z"], data["e_zeta"])
                + dot(data["e_zeta_z"], data["e_theta"])
            )
        ) / data["|B|"]

    # 2nd order derivatives
    # TODO: |B|_rr
    if has_dependencies("|B|_tt", params, transforms, profiles, data):
        data["|B|_tt"] = (
            data["B^theta_t"]
            * (
                data["B^zeta_t"] * data["g_tz"]
                + data["B^theta_t"] * data["g_tt"]
                + data["B^theta"] * dot(data["e_theta_t"], data["e_theta"])
            )
            + data["B^theta"]
            * (
                data["B^zeta_tt"] * data["g_tz"]
                + data["B^theta_tt"] * data["g_tt"]
                + data["B^theta_t"] * dot(data["e_theta_t"], data["e_theta"])
            )
            + data["B^theta"]
            * (
                data["B^zeta_t"]
                * (
                    dot(data["e_theta_t"], data["e_zeta"])  # TODO: this is just g_tz_t
                    + dot(data["e_theta"], data["e_zeta_t"])
                )
                + 2 * data["B^theta_t"] * dot(data["e_theta_t"], data["e_theta"])
                + data["B^theta"]
                * (
                    dot(data["e_theta_tt"], data["e_theta"])
                    + dot(data["e_theta_t"], data["e_theta_t"])
                )
            )
            + data["B^zeta_t"]
            * (
                data["B^theta_t"] * data["g_tz"]
                + data["B^zeta_t"] * data["g_zz"]
                + data["B^zeta"] * dot(data["e_zeta_t"], data["e_zeta"])
            )
            + data["B^zeta"]
            * (
                data["B^theta_tt"] * data["g_tz"]
                + data["B^zeta_tt"] * data["g_zz"]
                + data["B^zeta_t"] * dot(data["e_zeta_t"], data["e_zeta"])
            )
            + data["B^zeta"]
            * (
                data["B^theta_t"]
                * (
                    dot(data["e_theta_t"], data["e_zeta"])
                    + dot(data["e_theta"], data["e_zeta_t"])
                )
                + 2 * data["B^zeta_t"] * dot(data["e_zeta_t"], data["e_zeta"])
                + data["B^zeta"]
                * (
                    dot(data["e_zeta_tt"], data["e_zeta"])
                    + dot(data["e_zeta_t"], data["e_zeta_t"])
                )
            )
            + (data["B^theta_t"] * data["B^zeta"] + data["B^theta"] * data["B^zeta_t"])
            * (
                dot(data["e_theta_t"], data["e_zeta"])
                + dot(data["e_zeta_t"], data["e_theta"])
            )
            + data["B^theta"]
            * data["B^zeta"]
            * (
                dot(data["e_theta_tt"], data["e_zeta"])
                + dot(data["e_zeta_tt"], data["e_theta"])
                + 2 * dot(data["e_zeta_t"], data["e_theta_t"])
            )
        ) / data["|B|"] - data["|B|_t"] ** 2 / data["|B|"]
    if has_dependencies("|B|_zz", params, transforms, profiles, data):
        data["|B|_zz"] = (
            data["B^theta_z"]
            * (
                data["B^zeta_z"] * data["g_tz"]
                + data["B^theta_z"] * data["g_tt"]
                + data["B^theta"] * dot(data["e_theta_z"], data["e_theta"])
            )
            + data["B^theta"]
            * (
                data["B^zeta_zz"] * data["g_tz"]
                + data["B^theta_zz"] * data["g_tt"]
                + data["B^theta_z"] * dot(data["e_theta_z"], data["e_theta"])
            )
            + data["B^theta"]
            * (
                data["B^zeta_z"]
                * (
                    dot(data["e_theta_z"], data["e_zeta"])
                    + dot(data["e_theta"], data["e_zeta_z"])
                )
                + 2 * data["B^theta_z"] * dot(data["e_theta_z"], data["e_theta"])
                + data["B^theta"]
                * (
                    dot(data["e_theta_zz"], data["e_theta"])
                    + dot(data["e_theta_z"], data["e_theta_z"])
                )
            )
            + data["B^zeta_z"]
            * (
                data["B^theta_z"] * data["g_tz"]
                + data["B^zeta_z"] * data["g_zz"]
                + data["B^zeta"] * dot(data["e_zeta_z"], data["e_zeta"])
            )
            + data["B^zeta"]
            * (
                data["B^theta_zz"] * data["g_tz"]
                + data["B^zeta_zz"] * data["g_zz"]
                + data["B^zeta_z"] * dot(data["e_zeta_z"], data["e_zeta"])
            )
            + data["B^zeta"]
            * (
                data["B^theta_z"]
                * (
                    dot(data["e_theta_z"], data["e_zeta"])
                    + dot(data["e_theta"], data["e_zeta_z"])
                )
                + 2 * data["B^zeta_z"] * dot(data["e_zeta_z"], data["e_zeta"])
                + data["B^zeta"]
                * (
                    dot(data["e_zeta_zz"], data["e_zeta"])
                    + dot(data["e_zeta_z"], data["e_zeta_z"])
                )
            )
            + (data["B^theta_z"] * data["B^zeta"] + data["B^theta"] * data["B^zeta_z"])
            * (
                dot(data["e_theta_z"], data["e_zeta"])
                + dot(data["e_zeta_z"], data["e_theta"])
            )
            + data["B^theta"]
            * data["B^zeta"]
            * (
                dot(data["e_theta_zz"], data["e_zeta"])
                + dot(data["e_zeta_zz"], data["e_theta"])
                + 2 * dot(data["e_theta_z"], data["e_zeta_z"])
            )
        ) / data["|B|"] - data["|B|_z"] ** 2 / data["|B|"]
    # TODO: |B|_rt
    # TODO: |B|_rz
    if has_dependencies("|B|_tz", params, transforms, profiles, data):
        data["|B|_tz"] = (
            data["B^theta_z"]
            * (
                data["B^zeta_t"] * data["g_tz"]
                + data["B^theta_t"] * data["g_tt"]
                + data["B^theta"] * dot(data["e_theta_t"], data["e_theta"])
            )
            + data["B^theta"]
            * (
                data["B^zeta_tz"] * data["g_tz"]
                + data["B^theta_tz"] * data["g_tt"]
                + data["B^theta_z"] * dot(data["e_theta_t"], data["e_theta"])
            )
            + data["B^theta"]
            * (
                data["B^zeta_t"]
                * (
                    dot(data["e_theta_z"], data["e_zeta"])
                    + dot(data["e_theta"], data["e_zeta_z"])
                )
                + 2 * data["B^theta_t"] * dot(data["e_theta_z"], data["e_theta"])
                + data["B^theta"]
                * (
                    dot(data["e_theta_tz"], data["e_theta"])
                    + dot(data["e_theta_t"], data["e_theta_z"])
                )
            )
            + data["B^zeta_z"]
            * (
                data["B^theta_t"] * data["g_tz"]
                + data["B^zeta_t"] * data["g_zz"]
                + data["B^zeta"] * dot(data["e_zeta_t"], data["e_zeta"])
            )
            + data["B^zeta"]
            * (
                data["B^theta_tz"] * data["g_tz"]
                + data["B^zeta_tz"] * data["g_zz"]
                + data["B^zeta_z"] * dot(data["e_zeta_t"], data["e_zeta"])
            )
            + data["B^zeta"]
            * (
                data["B^theta_t"]
                * (
                    dot(data["e_theta_z"], data["e_zeta"])
                    + dot(data["e_theta"], data["e_zeta_z"])
                )
                + 2 * data["B^zeta_t"] * dot(data["e_zeta_z"], data["e_zeta"])
                + data["B^zeta"]
                * (
                    dot(data["e_zeta_tz"], data["e_zeta"])
                    + dot(data["e_zeta_t"], data["e_zeta_z"])
                )
            )
            + (data["B^theta_z"] * data["B^zeta"] + data["B^theta"] * data["B^zeta_z"])
            * (
                dot(data["e_theta_t"], data["e_zeta"])
                + dot(data["e_zeta_t"], data["e_theta"])
            )
            + data["B^theta"]
            * data["B^zeta"]
            * (
                dot(data["e_theta_tz"], data["e_zeta"])
                + dot(data["e_zeta_tz"], data["e_theta"])
                + dot(data["e_theta_t"], data["e_zeta_z"])
                + dot(data["e_zeta_t"], data["e_theta_z"])
            )
        ) / data["|B|"] - data["|B|_t"] * data["|B|_z"] / data["|B|"]

    return data


def compute_magnetic_pressure_gradient(
    params, transforms, profiles, data=None, **kwargs
):
    """Compute magnetic pressure gradient."""
    data = compute_covariant_magnetic_field(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_contravariant_metric_coefficients(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    # covariant components
    if has_dependencies("grad(|B|^2)_rho", params, transforms, profiles, data):
        data["grad(|B|^2)_rho"] = (
            data["B^theta"] * data["B_theta_r"]
            + data["B_theta"] * data["B^theta_r"]
            + data["B^zeta"] * data["B_zeta_r"]
            + data["B_zeta"] * data["B^zeta_r"]
        )
    if has_dependencies("grad(|B|^2)_theta", params, transforms, profiles, data):
        data["grad(|B|^2)_theta"] = (
            data["B^theta"] * data["B_theta_t"]
            + data["B_theta"] * data["B^theta_t"]
            + data["B^zeta"] * data["B_zeta_t"]
            + data["B_zeta"] * data["B^zeta_t"]
        )
    if has_dependencies("grad(|B|^2)_zeta", params, transforms, profiles, data):
        data["grad(|B|^2)_zeta"] = (
            data["B^theta"] * data["B_theta_z"]
            + data["B_theta"] * data["B^theta_z"]
            + data["B^zeta"] * data["B_zeta_z"]
            + data["B_zeta"] * data["B^zeta_z"]
        )

    # gradient vector
    if has_dependencies("grad(|B|^2)", params, transforms, profiles, data):
        data["grad(|B|^2)"] = (
            data["grad(|B|^2)_rho"] * data["e^rho"].T
            + data["grad(|B|^2)_theta"] * data["e^theta"].T
            + data["grad(|B|^2)_zeta"] * data["e^zeta"].T
        ).T

    # magnitude
    if has_dependencies("|grad(|B|^2)|/2mu0", params, transforms, profiles, data):
        data["|grad(|B|^2)|/2mu0"] = (
            jnp.sqrt(
                data["grad(|B|^2)_rho"] ** 2 * data["g^rr"]
                + data["grad(|B|^2)_theta"] ** 2 * data["g^tt"]
                + data["grad(|B|^2)_zeta"] ** 2 * data["g^zz"]
                + 2 * data["grad(|B|^2)_rho"] * data["grad(|B|^2)_theta"] * data["g^rt"]
                + 2 * data["grad(|B|^2)_rho"] * data["grad(|B|^2)_zeta"] * data["g^rz"]
                + 2
                * data["grad(|B|^2)_theta"]
                * data["grad(|B|^2)_zeta"]
                * data["g^tz"]
            )
            / 2
            / mu_0
        )

    return data


def compute_magnetic_tension(params, transforms, profiles, data=None, **kwargs):
    """Compute magnetic tension."""
    data = compute_contravariant_current_density(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_magnetic_pressure_gradient(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    if has_dependencies("(curl(B)xB)_rho", params, transforms, profiles, data):
        data["(curl(B)xB)_rho"] = (
            mu_0
            * data["sqrt(g)"]
            * (data["B^zeta"] * data["J^theta"] - data["B^theta"] * data["J^zeta"])
        )
    if has_dependencies("(curl(B)xB)_theta", params, transforms, profiles, data):
        data["(curl(B)xB)_theta"] = (
            -mu_0 * data["sqrt(g)"] * data["B^zeta"] * data["J^rho"]
        )
    if has_dependencies("(curl(B)xB)_zeta", params, transforms, profiles, data):
        data["(curl(B)xB)_zeta"] = (
            mu_0 * data["sqrt(g)"] * data["B^theta"] * data["J^rho"]
        )
    if has_dependencies("curl(B)xB", params, transforms, profiles, data):
        data["curl(B)xB"] = (
            data["(curl(B)xB)_rho"] * data["e^rho"].T
            + data["(curl(B)xB)_theta"] * data["e^theta"].T
            + data["(curl(B)xB)_zeta"] * data["e^zeta"].T
        ).T

    # tension vector
    if has_dependencies("(B*grad)B", params, transforms, profiles, data):
        data["(B*grad)B"] = data["curl(B)xB"] + data["grad(|B|^2)"] / 2
        data["((B*grad)B)_rho"] = dot(data["(B*grad)B"], data["e_rho"])
        data["((B*grad)B)_theta"] = dot(data["(B*grad)B"], data["e_theta"])
        data["((B*grad)B)_zeta"] = dot(data["(B*grad)B"], data["e_zeta"])
        data["|(B*grad)B|"] = jnp.sqrt(
            data["((B*grad)B)_rho"] ** 2 * data["g^rr"]
            + data["((B*grad)B)_theta"] ** 2 * data["g^tt"]
            + data["((B*grad)B)_zeta"] ** 2 * data["g^zz"]
            + 2 * data["((B*grad)B)_rho"] * data["((B*grad)B)_theta"] * data["g^rt"]
            + 2 * data["((B*grad)B)_rho"] * data["((B*grad)B)_zeta"] * data["g^rz"]
            + 2 * data["((B*grad)B)_theta"] * data["((B*grad)B)_zeta"] * data["g^tz"]
        )

    return data


def compute_B_dot_gradB(params, transforms, profiles, data=None, **kwargs):
    """Compute the quantity B*grad(|B|) and its partial derivatives."""
    data = compute_magnetic_field_magnitude(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    # 0th order term
    if has_dependencies("B*grad(|B|)", params, transforms, profiles, data):
        data["B*grad(|B|)"] = (
            data["B^theta"] * data["|B|_t"] + data["B^zeta"] * data["|B|_z"]
        )

    # 1st order derivatives
    # TODO: (B*grad(|B|))_r
    if has_dependencies("(B*grad(|B|))_t", params, transforms, profiles, data):
        data["(B*grad(|B|))_t"] = (
            data["B^theta_t"] * data["|B|_t"]
            + data["B^zeta_t"] * data["|B|_z"]
            + data["B^theta"] * data["|B|_tt"]
            + data["B^zeta"] * data["|B|_tz"]
        )
    if has_dependencies("(B*grad(|B|))_z", params, transforms, profiles, data):
        data["(B*grad(|B|))_z"] = (
            data["B^theta_z"] * data["|B|_t"]
            + data["B^zeta_z"] * data["|B|_z"]
            + data["B^theta"] * data["|B|_tz"]
            + data["B^zeta"] * data["|B|_zz"]
        )

    return data


def compute_boozer_magnetic_field(
    params,
    transforms,
    profiles,
    data=None,
    **kwargs,
):
    """Compute covariant magnetic field components in Boozer coordinates."""
    data = compute_covariant_magnetic_field(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    if has_dependencies("I", params, transforms, profiles, data):
        data["I"] = surface_averages(transforms["grid"], data["B_theta"])
        data["current"] = 2 * jnp.pi / mu_0 * data["I"]
    if has_dependencies("I_r", params, transforms, profiles, data):
        data["I_r"] = surface_averages(transforms["grid"], data["B_theta_r"])
        data["current_r"] = 2 * jnp.pi / mu_0 * data["I_r"]
    if has_dependencies("G", params, transforms, profiles, data):
        data["G"] = surface_averages(transforms["grid"], data["B_zeta"])
    if has_dependencies("G_r", params, transforms, profiles, data):
        data["G_r"] = surface_averages(transforms["grid"], data["B_zeta_r"])

    # TODO: add K(rho,theta,zeta)*grad(rho) term
    return data


def compute_contravariant_current_density(
    params,
    transforms,
    profiles,
    data=None,
    **kwargs,
):
    """Compute contravariant current density components."""
    data = compute_magnetic_field_magnitude(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    # TODO: can remove this call if compute_|B| changed to use B_covariant
    data = compute_covariant_magnetic_field(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    if has_dependencies("J^rho", params, transforms, profiles, data):
        data["J^rho"] = (data["B_zeta_t"] - data["B_theta_z"]) / (
            mu_0 * data["sqrt(g)"]
        )
    if has_dependencies("J^theta", params, transforms, profiles, data):
        data["J^theta"] = (data["B_rho_z"] - data["B_zeta_r"]) / (
            mu_0 * data["sqrt(g)"]
        )
    if has_dependencies("J^zeta", params, transforms, profiles, data):
        data["J^zeta"] = (data["B_theta_r"] - data["B_rho_t"]) / (
            mu_0 * data["sqrt(g)"]
        )
    if has_dependencies("J", params, transforms, profiles, data):
        data["J"] = (
            data["J^rho"] * data["e_rho"].T
            + data["J^theta"] * data["e_theta"].T
            + data["J^zeta"] * data["e_zeta"].T
        ).T
        data["J_R"] = data["J"][:, 0]
        data["J_phi"] = data["J"][:, 1]
        data["J_Z"] = data["J"][:, 2]
    if has_dependencies("|J|", params, transforms, profiles, data):
        data["|J|"] = jnp.sqrt(
            data["J^rho"] ** 2 * data["g_rr"]
            + data["J^theta"] ** 2 * data["g_tt"]
            + data["J^zeta"] ** 2 * data["g_zz"]
            + 2 * data["J^rho"] * data["J^theta"] * data["g_rt"]
            + 2 * data["J^rho"] * data["J^zeta"] * data["g_rz"]
            + 2 * data["J^theta"] * data["J^zeta"] * data["g_tz"]
        )
    if has_dependencies("J_parallel", params, transforms, profiles, data):
        data["J_parallel"] = (
            data["J^rho"] * data["B_rho"]
            + data["J^theta"] * data["B_theta"]
            + data["J^zeta"] * data["B_zeta"]
        ) / data["|B|"]

    return data
