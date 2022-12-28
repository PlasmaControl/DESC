"""Compute functions for equilibrium objectives, ie Force and MHD energy."""

from scipy.constants import mu_0

from desc.backend import jnp

from ._core import (
    compute_contravariant_metric_coefficients,
    compute_geometry,
    compute_pressure,
    compute_pressure_anisotropy,
    compute_pressure_gradient,
)
from ._field import (
    compute_contravariant_current_density,
    compute_magnetic_field_magnitude,
    compute_magnetic_pressure_gradient,
)
from .utils import cross, dot, has_dependencies


def compute_force_error(params, transforms, profiles, data=None, **kwargs):
    """Compute force error components."""
    data = compute_pressure(
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
    data = compute_contravariant_metric_coefficients(
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
    if has_dependencies("F_rho", params, transforms, profiles, data):
        data["F_rho"] = (-data["p_r"] - data["p_t"] - data["p_z"]) + data["sqrt(g)"] * (
            data["B^zeta"] * data["J^theta"] - data["B^theta"] * data["J^zeta"]
        )
    if has_dependencies("F_theta", params, transforms, profiles, data):
        data["F_theta"] = -data["sqrt(g)"] * data["B^zeta"] * data["J^rho"]
    if has_dependencies("F_zeta", params, transforms, profiles, data):
        data["F_zeta"] = data["sqrt(g)"] * data["B^theta"] * data["J^rho"]
    if has_dependencies("F_beta", params, transforms, profiles, data):
        data["F_beta"] = data["sqrt(g)"] * data["J^rho"]
    if has_dependencies("F", params, transforms, profiles, data):
        data["F"] = (
            data["F_rho"] * data["e^rho"].T
            + data["F_theta"] * data["e^theta"].T
            + data["F_zeta"] * data["e^zeta"].T
        ).T
    if has_dependencies("|F|", params, transforms, profiles, data):
        data["|F|"] = jnp.sqrt(
            data["F_rho"] ** 2 * data["g^rr"]
            + data["F_theta"] ** 2 * data["g^tt"]
            + data["F_zeta"] ** 2 * data["g^zz"]
            + 2 * data["F_rho"] * data["F_theta"] * data["g^rt"]
            + 2 * data["F_rho"] * data["F_zeta"] * data["g^rz"]
            + 2 * data["F_theta"] * data["F_zeta"] * data["g^tz"]
        )
    if has_dependencies("div(J_perp)", params, transforms, profiles, data):
        data["div(J_perp)"] = (mu_0 * data["J^rho"] * data["p_r"]) / data["|B|"] ** 2

    if has_dependencies("|beta|", params, transforms, profiles, data):
        data["|beta|"] = jnp.sqrt(
            data["B^zeta"] ** 2 * data["g^tt"]
            + data["B^theta"] ** 2 * data["g^zz"]
            - 2 * data["B^theta"] * data["B^zeta"] * data["g^tz"]
        )
    if has_dependencies("<|F|>_vol", params, transforms, profiles, data):
        data["<|F|>_vol"] = (
            jnp.sum(data["|F|"] * jnp.abs(data["sqrt(g)"]) * transforms["grid"].weights)
            / data["V"]
        )

    return data


def compute_force_error_anisotropic(params, transforms, profiles, data=None, **kwargs):
    """Compute force error for anisotropic pressure equilibrium."""
    data = compute_pressure(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_pressure_gradient(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_pressure_anisotropy(
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
    data = compute_magnetic_pressure_gradient(
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
    if has_dependencies("grad(d)", params, transforms, profiles, data):
        data["grad(d)"] = (
            data["d_r"] * data["e^rho"].T
            + data["d_t"] * data["e^theta"].T
            + data["d_z"] * data["e^zeta"].T
        ).T

    if has_dependencies("F_anisotropic", params, transforms, profiles, data):
        data["F_anisotropic"] = (
            (1 - data["d"]) * cross(data["J"], data["B"]).T
            - dot(data["B"], data["grad(d)"]) * data["B"].T
            - data["d"] * data["grad(|B|^2)"].T / (2 * mu_0)
            + data["grad(p)"].T
        ).T

    return data


def compute_energy(params, transforms, profiles, data=None, **kwargs):
    """Compute MHD energy. W = integral( B^2 / (2*mu0) + p / (gamma - 1) ) dV  (J)."""
    data = compute_pressure(
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

    if has_dependencies("W_B", params, transforms, profiles, data):
        data["W_B"] = jnp.sum(
            data["|B|"] ** 2 * jnp.abs(data["sqrt(g)"]) * transforms["grid"].weights
        ) / (2 * mu_0)
    if has_dependencies("W_p", params, transforms, profiles, data):
        data["W_p"] = jnp.sum(
            data["p"] * jnp.abs(data["sqrt(g)"]) * transforms["grid"].weights
        ) / (kwargs.get("gamma", 0) - 1)
    if has_dependencies("W", params, transforms, profiles, data):
        data["W"] = data["W_B"] + data["W_p"]

    return data
