"""Compute functions for equilibrium objectives, ie Force and MHD energy."""

from desc.backend import jnp
from scipy.constants import mu_0

from ._core import (
    check_derivs,
    compute_pressure,
    compute_contravariant_metric_coefficients,
)
from ._field import (
    compute_contravariant_current_density,
    compute_magnetic_field_magnitude,
)


def compute_force_error(
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
    **kwargs,
):
    """Compute force error components.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
    p_l : ndarray
        Spectral coefficients of p(rho) -- pressure profile.
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface, in Webers.
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
        Dictionary of ndarray, shape(num_nodes,) of force error components.
        Keys are of the form 'F_x', meaning the x covariant (F_x) component of the
        force error.

    """
    data = compute_pressure(p_l, pressure, data=data)
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
        data=data,
    )
    data = compute_contravariant_metric_coefficients(
        R_lmn, Z_lmn, R_transform, Z_transform, data=data
    )

    if check_derivs("F_rho", R_transform, Z_transform, L_transform):
        data["F_rho"] = -data["p_r"] + data["sqrt(g)"] * (
            data["B^zeta"] * data["J^theta"] - data["B^theta"] * data["J^zeta"]
        )
    if check_derivs("F_theta", R_transform, Z_transform, L_transform):
        data["F_theta"] = -data["sqrt(g)"] * data["B^zeta"] * data["J^rho"]
    if check_derivs("F_zeta", R_transform, Z_transform, L_transform):
        data["F_zeta"] = data["sqrt(g)"] * data["B^theta"] * data["J^rho"]
    if check_derivs("F_beta", R_transform, Z_transform, L_transform):
        data["F_beta"] = data["sqrt(g)"] * data["J^rho"]
    if check_derivs("F", R_transform, Z_transform, L_transform):
        data["F"] = (
            data["F_rho"] * data["e^rho"].T
            + data["F_theta"] * data["e^theta"].T
            + data["F_zeta"] * data["e^zeta"].T
        ).T
    if check_derivs("|F|", R_transform, Z_transform, L_transform):
        data["|F|"] = jnp.sqrt(
            data["F_rho"] ** 2 * data["g^rr"]
            + data["F_theta"] ** 2 * data["g^tt"]
            + data["F_zeta"] ** 2 * data["g^zz"]
            + 2 * data["F_rho"] * data["F_theta"] * data["g^rt"]
            + 2 * data["F_rho"] * data["F_zeta"] * data["g^rz"]
            + 2 * data["F_theta"] * data["F_zeta"] * data["g^tz"]
        )
        data["div_J_perp"] = (mu_0 * data["J^rho"] * data["p_r"]) / data["|B|"] ** 2

    if check_derivs("|grad(p)|", R_transform, Z_transform, L_transform):
        data["|grad(p)|"] = jnp.sqrt(data["p_r"] ** 2) * data["|grad(rho)|"]
    if check_derivs("|beta|", R_transform, Z_transform, L_transform):
        data["|beta|"] = jnp.sqrt(
            data["B^zeta"] ** 2 * data["g^tt"]
            + data["B^theta"] ** 2 * data["g^zz"]
            - 2 * data["B^theta"] * data["B^zeta"] * data["g^tz"]
        )

    return data


def compute_energy(
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    pressure,
    gamma=0,
    data=None,
    **kwargs,
):
    """Compute MHD energy. W = integral( B^2 / (2*mu0) + p / (gamma - 1) ) dV  (J).

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
    p_l : ndarray
        Spectral coefficients of p(rho) -- pressure profile.
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface, in Webers.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.
    L_transform : Transform
        Transforms L_lmn coefficients to real space.
    iota : Profile
        Transforms i_l coefficients to real space.
    pressure : Profile
        Transforms p_l coefficients to real space.
    gamma : float
        Adiabatic (compressional) index.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) with energy keys "W", "W_B", "W_p".

    """
    data = compute_pressure(p_l, pressure, data=data)
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

    if check_derivs("W_B", R_transform, Z_transform, L_transform):
        data["W_B"] = jnp.sum(
            data["|B|"] ** 2 * jnp.abs(data["sqrt(g)"]) * R_transform.grid.weights
        ) / (2 * mu_0)
    if check_derivs("W_p", R_transform, Z_transform, L_transform):
        data["W_p"] = jnp.sum(
            data["p"] * jnp.abs(data["sqrt(g)"]) * R_transform.grid.weights
        ) / (gamma - 1)
    if check_derivs("W", R_transform, Z_transform, L_transform):
        data["W"] = data["W_B"] + data["W_p"]

    return data
