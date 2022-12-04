"""Compute functions for equilibrium objectives, ie Force and MHD energy."""

from scipy.constants import mu_0

from desc.backend import jnp

from ._core import (
    compute_contravariant_metric_coefficients,
    compute_geometry,
    compute_pressure,
)
from ._field import (
    compute_contravariant_current_density,
    compute_magnetic_field_magnitude,
)
from .utils import check_derivs


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

    if check_derivs("F_rho", transforms["R"], transforms["Z"], transforms["L"]):
        data["F_rho"] = -data["p_r"] + data["sqrt(g)"] * (
            data["B^zeta"] * data["J^theta"] - data["B^theta"] * data["J^zeta"]
        )
    if check_derivs("F_theta", transforms["R"], transforms["Z"], transforms["L"]):
        data["F_theta"] = -data["sqrt(g)"] * data["B^zeta"] * data["J^rho"]
    if check_derivs("F_zeta", transforms["R"], transforms["Z"], transforms["L"]):
        data["F_zeta"] = data["sqrt(g)"] * data["B^theta"] * data["J^rho"]
    if check_derivs("F_beta", transforms["R"], transforms["Z"], transforms["L"]):
        data["F_beta"] = data["sqrt(g)"] * data["J^rho"]
    if check_derivs("F", transforms["R"], transforms["Z"], transforms["L"]):
        data["F"] = (
            data["F_rho"] * data["e^rho"].T
            + data["F_theta"] * data["e^theta"].T
            + data["F_zeta"] * data["e^zeta"].T
        ).T
    if check_derivs("|F|", transforms["R"], transforms["Z"], transforms["L"]):
        data["|F|"] = jnp.sqrt(
            data["F_rho"] ** 2 * data["g^rr"]
            + data["F_theta"] ** 2 * data["g^tt"]
            + data["F_zeta"] ** 2 * data["g^zz"]
            + 2 * data["F_rho"] * data["F_theta"] * data["g^rt"]
            + 2 * data["F_rho"] * data["F_zeta"] * data["g^rz"]
            + 2 * data["F_theta"] * data["F_zeta"] * data["g^tz"]
        )
        data["div(J_perp)"] = (mu_0 * data["J^rho"] * data["p_r"]) / data["|B|"] ** 2

    if check_derivs("|grad(p)|", transforms["R"], transforms["Z"], transforms["L"]):
        data["|grad(p)|"] = jnp.sqrt(data["p_r"] ** 2) * data["|grad(rho)|"]
        data["<|grad(p)|>_vol"] = (
            jnp.sum(
                data["|grad(p)|"]
                * jnp.abs(data["sqrt(g)"])
                * transforms["R"].grid.weights
            )
            / data["V"]
        )
    if check_derivs("|beta|", transforms["R"], transforms["Z"], transforms["L"]):
        data["|beta|"] = jnp.sqrt(
            data["B^zeta"] ** 2 * data["g^tt"]
            + data["B^theta"] ** 2 * data["g^zz"]
            - 2 * data["B^theta"] * data["B^zeta"] * data["g^tz"]
        )
    if check_derivs("<|F|>_vol", transforms["R"], transforms["Z"], transforms["L"]):
        data["<|F|>_vol"] = (
            jnp.sum(
                data["|F|"] * jnp.abs(data["sqrt(g)"]) * transforms["R"].grid.weights
            )
            / data["V"]
        )

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

    if check_derivs("W_B", transforms["R"], transforms["Z"], transforms["L"]):
        data["W_B"] = jnp.sum(
            data["|B|"] ** 2 * jnp.abs(data["sqrt(g)"]) * transforms["R"].grid.weights
        ) / (2 * mu_0)
    if check_derivs("W_p", transforms["R"], transforms["Z"], transforms["L"]):
        data["W_p"] = jnp.sum(
            data["p"] * jnp.abs(data["sqrt(g)"]) * transforms["R"].grid.weights
        ) / (kwargs["gamma"] - 1)
    if check_derivs("W", transforms["R"], transforms["Z"], transforms["L"]):
        data["W"] = data["W_B"] + data["W_p"]

    return data
