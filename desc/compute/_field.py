"""Compute functions for magnetic field quantities."""

from desc.backend import jnp
from scipy.constants import mu_0

from ._core import (
    dot,
    check_derivs,
    compute_toroidal_flux,
    compute_rotational_transform,
    compute_lambda,
    compute_jacobian,
    compute_covariant_metric_coefficients,
    compute_contravariant_metric_coefficients,
)


def compute_contravariant_magnetic_field(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    data=None,
    **kwargs,
):
    """Compute contravariant magnetic field components.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
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

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of contravariant magnetic field
        components. Keys are of the form 'B^x_y', meaning the x contravariant (B^x)
        component of the magnetic field, differentiated wrt y.

    """
    data = compute_toroidal_flux(Psi, iota, data=data)
    data = compute_rotational_transform(i_l, iota, data=data)
    data = compute_lambda(L_lmn, L_transform, data=data)
    data = compute_jacobian(
        R_lmn,
        Z_lmn,
        R_transform,
        Z_transform,
        data=data,
    )

    # 0th order terms
    if check_derivs("B0", R_transform, Z_transform, L_transform):
        data["B0"] = data["psi_r"] / data["sqrt(g)"]
    if check_derivs("B^rho", R_transform, Z_transform, L_transform):
        data["B^rho"] = data["0"]
    if check_derivs("B^theta", R_transform, Z_transform, L_transform):
        data["B^theta"] = data["B0"] * (data["iota"] - data["lambda_z"])
    if check_derivs("B^zeta", R_transform, Z_transform, L_transform):
        data["B^zeta"] = data["B0"] * (1 + data["lambda_t"])
    if check_derivs("B", R_transform, Z_transform, L_transform):
        data["B"] = (
            data["B^theta"] * data["e_theta"].T + data["B^zeta"] * data["e_zeta"].T
        ).T
        data["B_R"] = data["B"][:, 0]
        data["B_phi"] = data["B"][:, 1]
        data["B_Z"] = data["B"][:, 2]

    # 1st order derivatives
    if check_derivs("B0_r", R_transform, Z_transform, L_transform):
        data["B0_r"] = (
            data["psi_rr"] / data["sqrt(g)"]
            - data["psi_r"] * data["sqrt(g)_r"] / data["sqrt(g)"] ** 2
        )
    if check_derivs("B^theta_r", R_transform, Z_transform, L_transform):
        data["B^theta_r"] = data["B0_r"] * (data["iota"] - data["lambda_z"]) + data[
            "B0"
        ] * (data["iota_r"] - data["lambda_rz"])
    if check_derivs("B^zeta_r", R_transform, Z_transform, L_transform):
        data["B^zeta_r"] = (
            data["B0_r"] * (1 + data["lambda_t"]) + data["B0"] * data["lambda_rt"]
        )
    if check_derivs("B_r", R_transform, Z_transform, L_transform):
        data["B_r"] = (
            data["B^theta_r"] * data["e_theta"].T
            + data["B^theta"] * data["e_theta_r"].T
            + data["B^zeta_r"] * data["e_zeta"].T
            + data["B^zeta"] * data["e_zeta_r"].T
        ).T
    if check_derivs("B0_t", R_transform, Z_transform, L_transform):
        data["B0_t"] = -data["psi_r"] * data["sqrt(g)_t"] / data["sqrt(g)"] ** 2
    if check_derivs("B^theta_t", R_transform, Z_transform, L_transform):
        data["B^theta_t"] = (
            data["B0_t"] * (data["iota"] - data["lambda_z"])
            - data["B0"] * data["lambda_tz"]
        )
    if check_derivs("B^zeta_t", R_transform, Z_transform, L_transform):
        data["B^zeta_t"] = (
            data["B0_t"] * (1 + data["lambda_t"]) + data["B0"] * data["lambda_tt"]
        )
    if check_derivs("B_t", R_transform, Z_transform, L_transform):
        data["B_t"] = (
            data["B^theta_t"] * data["e_theta"].T
            + data["B^theta"] * data["e_theta_t"].T
            + data["B^zeta_t"] * data["e_zeta"].T
            + data["B^zeta"] * data["e_zeta_t"].T
        ).T
    if check_derivs("B0_z", R_transform, Z_transform, L_transform):
        data["B0_z"] = -data["psi_r"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 2
    if check_derivs("B^theta_z", R_transform, Z_transform, L_transform):
        data["B^theta_z"] = (
            data["B0_z"] * (data["iota"] - data["lambda_z"])
            - data["B0"] * data["lambda_zz"]
        )
    if check_derivs("B^zeta_z", R_transform, Z_transform, L_transform):
        data["B^zeta_z"] = (
            data["B0_z"] * (1 + data["lambda_t"]) + data["B0"] * data["lambda_tz"]
        )
    if check_derivs("B_z", R_transform, Z_transform, L_transform):
        data["B_z"] = (
            data["B^theta_z"] * data["e_theta"].T
            + data["B^theta"] * data["e_theta_z"].T
            + data["B^zeta_z"] * data["e_zeta"].T
            + data["B^zeta"] * data["e_zeta_z"].T
        ).T

    # 2nd order derivatives
    if check_derivs("B0_tt", R_transform, Z_transform, L_transform):
        data["B0_tt"] = -(
            data["psi_r"]
            / data["sqrt(g)"] ** 2
            * (data["sqrt(g)_tt"] - 2 * data["sqrt(g)_t"] ** 2 / data["sqrt(g)"])
        )
    if check_derivs("B^theta_tt", R_transform, Z_transform, L_transform):
        data["B^theta_tt"] = data["B0_tt"] * (data["iota"] - data["lambda_z"])
        -2 * data["B0_t"] * data["lambda_tz"] - data["B0"] * data["lambda_ttz"]
    if check_derivs("B^zeta_tt", R_transform, Z_transform, L_transform):
        data["B^zeta_tt"] = data["B0_tt"] * (1 + data["lambda_t"])
        +2 * data["B0_t"] * data["lambda_tt"] + data["B0"] * data["lambda_ttt"]
    if check_derivs("B0_zz", R_transform, Z_transform, L_transform):
        data["B0_zz"] = -(
            data["psi_r"]
            / data["sqrt(g)"] ** 2
            * (data["sqrt(g)_zz"] - 2 * data["sqrt(g)_z"] ** 2 / data["sqrt(g)"])
        )
    if check_derivs("B^theta_zz", R_transform, Z_transform, L_transform):
        data["B^theta_zz"] = data["B0_zz"] * (data["iota"] - data["lambda_z"])
        -2 * data["B0_z"] * data["lambda_zz"] - data["B0"] * data["lambda_zzz"]
    if check_derivs("B^zeta_zz", R_transform, Z_transform, L_transform):
        data["B^zeta_zz"] = data["B0_zz"] * (1 + data["lambda_t"])
        +2 * data["B0_z"] * data["lambda_tz"] + data["B0"] * data["lambda_tzz"]
    if check_derivs("B0_tz", R_transform, Z_transform, L_transform):
        data["B0_tz"] = -(
            data["psi_r"]
            / data["sqrt(g)"] ** 2
            * (
                data["sqrt(g)_tz"]
                - 2 * data["sqrt(g)_t"] * data["sqrt(g)_z"] / data["sqrt(g)"]
            )
        )
    if check_derivs("B^theta_tz", R_transform, Z_transform, L_transform):
        data["B^theta_tz"] = data["B0_tz"] * (data["iota"] - data["lambda_z"])
        -data["B0_t"] * data["lambda_zz"] - data["B0_z"] * data["lambda_tz"]
        -data["B0"] * data["lambda_tzz"]
    if check_derivs("B^zeta_tz", R_transform, Z_transform, L_transform):
        data["B^zeta_tz"] = data["B0_tz"] * (1 + data["lambda_t"])
        (
            +data["B0_t"] * data["lambda_tz"]
            + data["B0_z"] * data["lambda_tt"]
            + data["B0"] * data["lambda_ttz"]
        )

    return data


def compute_covariant_magnetic_field(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    data=None,
    **kwargs,
):
    """Compute covariant magnetic field components.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
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

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of covariant magnetic field
        components. Keys are of the form 'B_x_y', meaning the x covariant (B_x)
        component of the magnetic field, differentiated wrt y.

    """
    data = compute_contravariant_magnetic_field(
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

    # 0th order terms
    if check_derivs("B_rho", R_transform, Z_transform, L_transform):
        data["B_rho"] = dot(data["B"], data["e_rho"])
    if check_derivs("B_theta", R_transform, Z_transform, L_transform):
        data["B_theta"] = dot(data["B"], data["e_theta"])
    if check_derivs("B_zeta", R_transform, Z_transform, L_transform):
        data["B_zeta"] = dot(data["B"], data["e_zeta"])

    # 1st order derivatives
    if check_derivs("B_rho_r", R_transform, Z_transform, L_transform):
        data["B_rho_r"] = dot(data["B_r"], data["e_rho"]) + dot(
            data["B"], data["e_rho_r"]
        )
    if check_derivs("B_theta_r", R_transform, Z_transform, L_transform):
        data["B_theta_r"] = dot(data["B_r"], data["e_theta"]) + dot(
            data["B"], data["e_theta_r"]
        )
    if check_derivs("B_zeta_r", R_transform, Z_transform, L_transform):
        data["B_zeta_r"] = dot(data["B_r"], data["e_zeta"]) + dot(
            data["B"], data["e_zeta_r"]
        )
    if check_derivs("B_rho_t", R_transform, Z_transform, L_transform):
        data["B_rho_t"] = dot(data["B_t"], data["e_rho"]) + dot(
            data["B"], data["e_rho_t"]
        )
    if check_derivs("B_theta_t", R_transform, Z_transform, L_transform):
        data["B_theta_t"] = dot(data["B_t"], data["e_theta"]) + dot(
            data["B"], data["e_theta_t"]
        )
    if check_derivs("B_zeta_t", R_transform, Z_transform, L_transform):
        data["B_zeta_t"] = dot(data["B_t"], data["e_zeta"]) + dot(
            data["B"], data["e_zeta_t"]
        )
    if check_derivs("B_rho_z", R_transform, Z_transform, L_transform):
        data["B_rho_z"] = dot(data["B_z"], data["e_rho"]) + dot(
            data["B"], data["e_rho_z"]
        )
    if check_derivs("B_theta_z", R_transform, Z_transform, L_transform):
        data["B_theta_z"] = dot(data["B_z"], data["e_theta"]) + dot(
            data["B"], data["e_theta_z"]
        )
    if check_derivs("B_zeta_z", R_transform, Z_transform, L_transform):
        data["B_zeta_z"] = dot(data["B_z"], data["e_zeta"]) + dot(
            data["B"], data["e_zeta_z"]
        )

    return data


def compute_magnetic_field_magnitude(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    data=None,
    **kwargs,
):
    """Compute magnetic field magnitude.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
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

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of magnetic field magnitude.
        Keys are of the form '|B|_x', meaning the x derivative of the
        magnetic field magnitude |B|.

    """
    data = compute_contravariant_magnetic_field(
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
    data = compute_covariant_metric_coefficients(
        R_lmn, Z_lmn, R_transform, Z_transform, data=data
    )

    # TODO: would it be simpler to compute this as B^theta*B_theta+B^zeta*B_zeta?

    # 0th order term
    if check_derivs("|B|", R_transform, Z_transform, L_transform):
        data["|B|"] = jnp.sqrt(
            data["B^theta"] ** 2 * data["g_tt"]
            + data["B^zeta"] ** 2 * data["g_zz"]
            + 2 * data["B^theta"] * data["B^zeta"] * data["g_tz"]
        )

    # 1st order derivatives
    # TODO: |B|_r
    if check_derivs("|B|_t", R_transform, Z_transform, L_transform):
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
    if check_derivs("|B|_z", R_transform, Z_transform, L_transform):
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
    if check_derivs("|B|_tt", R_transform, Z_transform, L_transform):
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
                    dot(data["e_theta_t"], data["e_zeta"])
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
    if check_derivs("|B|_zz", R_transform, Z_transform, L_transform):
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
    if check_derivs("|B|_tz", R_transform, Z_transform, L_transform):
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
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    data=None,
    **kwargs,
):
    """Compute magnetic pressure gradient.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
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

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of magnetic pressure gradient
        components and magnitude. Keys are of the form 'grad(|B|^2)_x', meaning the x
        covariant component of the magnetic pressure gradient grad(|B|^2).

    """
    data = compute_covariant_magnetic_field(
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

    # covariant components
    if check_derivs("grad(|B|^2)_rho", R_transform, Z_transform, L_transform):
        data["grad(|B|^2)_rho"] = (
            data["B^theta"] * data["B_theta_r"]
            + data["B_theta"] * data["B^theta_r"]
            + data["B^zeta"] * data["B_zeta_r"]
            + data["B_zeta"] * data["B^zeta_r"]
        )
    if check_derivs("grad(|B|^2)_theta", R_transform, Z_transform, L_transform):
        data["grad(|B|^2)_theta"] = (
            data["B^theta"] * data["B_theta_t"]
            + data["B_theta"] * data["B^theta_t"]
            + data["B^zeta"] * data["B_zeta_t"]
            + data["B_zeta"] * data["B^zeta_t"]
        )
    if check_derivs("grad(|B|^2)_zeta", R_transform, Z_transform, L_transform):
        data["grad(|B|^2)_zeta"] = (
            data["B^theta"] * data["B_theta_z"]
            + data["B_theta"] * data["B^theta_z"]
            + data["B^zeta"] * data["B_zeta_z"]
            + data["B_zeta"] * data["B^zeta_z"]
        )

    # gradient vector
    if check_derivs("grad(|B|^2)", R_transform, Z_transform, L_transform):
        data["grad(|B|^2)"] = (
            data["grad(|B|^2)_rho"] * data["e^rho"].T
            + data["grad(|B|^2)_theta"] * data["e^theta"].T
            + data["grad(|B|^2)_zeta"] * data["e^zeta"].T
        ).T

    # magnitude
    if check_derivs("|grad(|B|^2)|/2mu0", R_transform, Z_transform, L_transform):
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


def compute_magnetic_tension(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    data=None,
    **kwargs,
):
    """Compute magnetic tension.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
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

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of magnetic tension vector components
        and magnitude. Keys are of the form '((B*grad(|B|))B)^x', meaning the x
        contravariant component of the magnetic tension vector (B*grad(|B|))B.

    """
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
    data = compute_magnetic_pressure_gradient(
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

    if check_derivs("(curl(B)xB)_rho", R_transform, Z_transform, L_transform):
        data["(curl(B)xB)_rho"] = (
            mu_0
            * data["sqrt(g)"]
            * (data["B^zeta"] * data["J^theta"] - data["B^theta"] * data["J^zeta"])
        )
    if check_derivs("(curl(B)xB)_theta", R_transform, Z_transform, L_transform):
        data["(curl(B)xB)_theta"] = (
            -mu_0 * data["sqrt(g)"] * data["B^zeta"] * data["J^rho"]
        )
    if check_derivs("(curl(B)xB)_zeta", R_transform, Z_transform, L_transform):
        data["(curl(B)xB)_zeta"] = (
            mu_0 * data["sqrt(g)"] * data["B^theta"] * data["J^rho"]
        )
    if check_derivs("curl(B)xB", R_transform, Z_transform, L_transform):
        data["curl(B)xB"] = (
            data["(curl(B)xB)_rho"] * data["e^rho"].T
            + data["(curl(B)xB)_theta"] * data["e^theta"].T
            + data["(curl(B)xB)_zeta"] * data["e^zeta"].T
        ).T

    # tension vector
    if check_derivs("(B*grad)B", R_transform, Z_transform, L_transform):
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


def compute_B_dot_gradB(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    data=None,
    **kwargs,
):
    """Compute the quantity B*grad(|B|) and its partial derivatives.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
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
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.
    dt : int, optional
        Order of derivative wrt the poloidal coordinate, theta.
    dz : int, optional
        Order of derivative wrt the toroidal coordinate, zeta.
    drtz : int, optional
        Order of mixed derivatives wrt multiple coordinates.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of the quantity B*grad(|B|). Keys are
        of the form 'B*grad(|B|)_x', meaning the derivative of B*grad(|B|) wrt x.

    """
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

    # 0th order term
    if check_derivs("B*grad(|B|)", R_transform, Z_transform, L_transform):
        data["B*grad(|B|)"] = (
            data["B^theta"] * data["|B|_t"] + data["B^zeta"] * data["|B|_z"]
        )

    # 1st order derivatives
    # TODO: (B*grad(|B|))_r
    if check_derivs("(B*grad(|B|))_t", R_transform, Z_transform, L_transform):
        data["(B*grad(|B|))_t"] = (
            data["B^theta_t"] * data["|B|_t"]
            + data["B^zeta_t"] * data["|B|_z"]
            + data["B^theta"] * data["|B|_tt"]
            + data["B^zeta"] * data["|B|_tz"]
        )
    if check_derivs("(B*grad(|B|))_z", R_transform, Z_transform, L_transform):
        data["(B*grad(|B|))_z"] = (
            data["B^theta_z"] * data["|B|_t"]
            + data["B^zeta_z"] * data["|B|_z"]
            + data["B^theta"] * data["|B|_tz"]
            + data["B^zeta"] * data["|B|_zz"]
        )

    return data


def compute_contravariant_current_density(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    data=None,
    **kwargs,
):
    """Compute contravariant current density components.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
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

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of contravariant current density
        components. Keys are of the form 'J^x_y', meaning the x contravariant (J^x)
        component of the current density J, differentiated wrt y.

    """
    data = compute_covariant_magnetic_field(
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
    data = compute_covariant_metric_coefficients(
        R_lmn, Z_lmn, R_transform, Z_transform, data=data
    )

    if check_derivs("J^rho", R_transform, Z_transform, L_transform):
        data["J^rho"] = (data["B_zeta_t"] - data["B_theta_z"]) / (
            mu_0 * data["sqrt(g)"]
        )
    if check_derivs("J^theta", R_transform, Z_transform, L_transform):
        data["J^theta"] = (data["B_rho_z"] - data["B_zeta_r"]) / (
            mu_0 * data["sqrt(g)"]
        )
    if check_derivs("J^zeta", R_transform, Z_transform, L_transform):
        data["J^zeta"] = (data["B_theta_r"] - data["B_rho_t"]) / (
            mu_0 * data["sqrt(g)"]
        )
    if check_derivs("J", R_transform, Z_transform, L_transform):
        data["J"] = (
            data["J^rho"] * data["e_rho"].T
            + data["J^theta"] * data["e_theta"].T
            + data["J^zeta"] * data["e_zeta"].T
        ).T
        data["J_R"] = data["J"][:, 0]
        data["J_phi"] = data["J"][:, 1]
        data["J_Z"] = data["J"][:, 2]
        data["|J|"] = jnp.sqrt(
            data["J^rho"] ** 2 * data["g_rr"]
            + data["J^theta"] ** 2 * data["g_tt"]
            + data["J^zeta"] ** 2 * data["g_zz"]
            + 2 * data["J^rho"] * data["J^theta"] * data["g_rt"]
            + 2 * data["J^rho"] * data["J^zeta"] * data["g_rz"]
            + 2 * data["J^theta"] * data["J^zeta"] * data["g_tz"]
        )
        data["|B|"] = jnp.sqrt(
            data["B^theta"] ** 2 * data["g_tt"]
            + data["B^zeta"] ** 2 * data["g_zz"]
            + 2 * data["B^theta"] * data["B^zeta"] * data["g_tz"]
        )
        data["J_parallel"] = (
            data["J^rho"] * data["B_rho"]
            + data["J^theta"] * data["B_theta"]
            + data["J^zeta"] * data["B_zeta"]
        ) / data["|B|"]

    return data
