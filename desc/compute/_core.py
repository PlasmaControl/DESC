"""Core compute functions, for profiles, geometry, and basis vectors/jacobians."""

import numpy as np
import pickle
from scipy.constants import mu_0

from desc.backend import jnp
from desc.compute import data_index
from desc.compute.utils import dot, cross, surface_averages


def check_derivs(key, R_transform=None, Z_transform=None, L_transform=None):
    """Check if Transforms can compute required derivatives of R, Z, lambda.

    Parameters
    ----------
    key : str
        Key indicating a quantity from data_index.
    R_transform : Transform, optional
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform, optional
        Transforms Z_lmn coefficients to real space.
    L_transform : Transform, optional
        Transforms L_lmn coefficients to real space.

    Returns
    -------
    flag : bool
        True if the Transforms can compute requested derivatives, False otherwise.

    """
    if "R_derivs" not in data_index[key]:
        R_flag = True
        Z_flag = True
    else:
        R_flag = np.array(
            [d in R_transform.derivatives.tolist() for d in data_index[key]["R_derivs"]]
        ).all()
        Z_flag = np.array(
            [d in Z_transform.derivatives.tolist() for d in data_index[key]["R_derivs"]]
        ).all()

    if "L_derivs" not in data_index[key]:
        L_flag = True
    else:
        L_flag = np.array(
            [d in L_transform.derivatives.tolist() for d in data_index[key]["L_derivs"]]
        ).all()

    return R_flag and Z_flag and L_flag


def compute_flux_coords(
    grid,
    data=None,
    **kwargs,
):
    """Compute flux coordinates (rho,theta,zeta).

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of flux coordinates.

    """
    if data is None:
        data = {}

    data["rho"] = grid.nodes[:, 0]
    data["theta"] = grid.nodes[:, 1]
    data["zeta"] = grid.nodes[:, 2]

    return data


def compute_toroidal_flux(
    Psi,
    grid,
    data=None,
    **kwargs,
):
    """Compute toroidal magnetic flux profile.

    Parameters
    ----------
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface, in Webers.
    grid : Grid
        Collocation grid containing the nodes to evaluate at.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of toroidal magnetic flux profile.
        Keys are of the form 'X_y' meaning the derivative of X wrt y.

    """
    data = compute_flux_coords(grid, data=data)

    data["psi"] = Psi * data["rho"] ** 2 / (2 * jnp.pi)
    data["psi_r"] = 2 * Psi * data["rho"] / (2 * jnp.pi)
    data["psi_rr"] = 2 * Psi * jnp.ones_like(data["rho"]) / (2 * jnp.pi)

    return data


def compute_toroidal_coords(
    R_lmn,
    Z_lmn,
    R_transform,
    Z_transform,
    data=None,
    **kwargs,
):
    """Compute toroidal coordinates (R,phi,Z).

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt y.

    """
    if data is None:
        data = {}

    derivs = [
        "",
        "_r",
        "_t",
        "_z",
        "_rr",
        "_tt",
        "_zz",
        "_rt",
        "_rz",
        "_tz",
        "_rrr",
        "_ttt",
        "_zzz",
        "_rrt",
        "_rtt",
        "_rrz",
        "_rzz",
        "_ttz",
        "_tzz",
        "_rtz",
    ]

    for d in derivs:
        keyR = "R" + d
        keyZ = "Z" + d
        if check_derivs(keyR, R_transform, Z_transform):
            data[keyR] = R_transform.transform(R_lmn, *data_index[keyR]["R_derivs"][0])
            data[keyZ] = Z_transform.transform(Z_lmn, *data_index[keyZ]["R_derivs"][0])

    return data


def compute_cartesian_coords(
    R_lmn,
    Z_lmn,
    R_transform,
    Z_transform,
    data=None,
    **kwargs,
):
    """Compute Cartesian coordinates (X,Y,Z).

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of Cartesian coordinates.

    """
    data = compute_flux_coords(R_transform.grid, data=data)
    data = compute_toroidal_coords(R_lmn, Z_lmn, R_transform, Z_transform, data=data)

    data["phi"] = data["zeta"]
    data["X"] = data["R"] * jnp.cos(data["phi"])
    data["Y"] = data["R"] * jnp.sin(data["phi"])

    return data


def compute_lambda(
    L_lmn,
    L_transform,
    data=None,
    **kwargs,
):
    """Compute lambda such that theta* = theta + lambda is a sfl coordinate.

    Parameters
    ----------
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
    L_transform : Transform
        Transforms L_lmn coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of lambda values.
        Keys are of the form 'lambda_x' meaning the derivative of lambda wrt x.

    """
    if data is None:
        data = {}

    keys = [
        "lambda",
        "lambda_r",
        "lambda_t",
        "lambda_z",
        "lambda_rr",
        "lambda_tt",
        "lambda_zz",
        "lambda_rt",
        "lambda_rz",
        "lambda_tz",
        "lambda_rrr",
        "lambda_ttt",
        "lambda_zzz",
        "lambda_rrt",
        "lambda_rtt",
        "lambda_rrz",
        "lambda_rzz",
        "lambda_ttz",
        "lambda_tzz",
        "lambda_rtz",
    ]

    for key in keys:
        if check_derivs(key, L_transform=L_transform):
            data[key] = L_transform.transform(L_lmn, *data_index[key]["L_derivs"][0])

    return data


def compute_pressure(
    p_l,
    pressure,
    data=None,
    **kwargs,
):
    """Compute pressure profile.

    Parameters
    ----------
    p_l : ndarray
        Spectral coefficients of p(rho) -- pressure profile.
    pressure : Profile
        Transforms p_l coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of pressure profile.
        Keys are of the form 'X_y' meaning the derivative of X wrt y.

    """
    if data is None:
        data = {}

    data["p"] = pressure.compute(p_l, dr=0)
    data["p_r"] = pressure.compute(p_l, dr=1)

    return data


def compute_rotational_transform(
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
    current_from_fixed_iota=None,
    current_r_from_fixed_iota=None,
    **kwargs,
):
    """
    Compute rotational transform profile from the iota or the toroidal current profile.

    Currently, the rotational transform is first computed to satisfy a zero toroidal
    current condition based on the geometry of the device as discussed in
    S.P. Hirshman, J.T. Hogan, ORMEC: A three-dimensional MHD spectral inverse
    equilibrium code. Journal of Computational Physics, 63, 2, 1986, p. 334.
    doi:10.1016/0021-9991(86)90197-X.

    Note their inconsistent notation for covariant and contravariant forms.
    Specifically, their j_zeta (covariant notation) should be j^zeta (contravariant).
    The covariant notation is correct for the other quantities.
    Also note that the integrals to compute the flux surface averages should not be weighted
    by a factor of sqrt(g), meaning the sqrt(g) in the denominator will not be canceled.

    This condition comes from solving a linear system for 0 average poloidal
    magnetic field. Because the rotational transform is defined as
    d(poloidal flux)/d(toroidal flux), and the system yielding iota for zero toroidal
    current is linear, the rotational transform corresponding to any toroidal
    current is computed by adding the Δ(poloidal flux) generated by the toroidal
    current enclosed by the flux surface.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate.
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
    current_from_fixed_iota : ndarray
        Used to simplify debugging. Will be removed later.
        Current in tesla meters as computed by a fixed-iota equilibrium solve.
        ndarray of data["I"].
    current_r_from_fixed_iota : ndarray
        Used to simplify debugging. Will be removed later.
        1st rho derivative of current in tesla meters as computed by a fixed-iota equilibrium solve.
        ndarray of surface_averages(B_theta_r).

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of rotational transform profile.
        Keys are of the form 'X_y' meaning the derivative of X wrt y.

    """
    if data is None:
        data = {}

    if iota is not None:
        data["iota"] = iota.compute(i_l, dr=0)
        data["iota_r"] = iota.compute(i_l, dr=1)
        data["iota_rr"] = iota.compute(i_l, dr=2)

    elif current is not None:
        grid = R_transform.grid
        data = compute_jacobian(R_lmn, Z_lmn, R_transform, Z_transform, data)
        data = compute_toroidal_flux(Psi, grid, data)
        data = compute_lambda(L_lmn, L_transform, data)
        data = compute_covariant_metric_coefficients(
            R_lmn, Z_lmn, R_transform, Z_transform, data
        )

        if check_derivs("iota", R_transform, Z_transform, L_transform):
            # Δ(poloidal flux) generated by the toroidal current enclosed by the flux surface
            # is B_theta average = I in tesla-meters = mu_0 / 2pi * (I in amps)
            if current_from_fixed_iota is not None:
                # TODO: once debugged, this branch of the if statement can be removed,
                #   so that the else statement always executes.
                enclosed_toroidal_current = current_from_fixed_iota
            else:
                enclosed_toroidal_current = (
                    mu_0 / 2 / jnp.pi * current.compute(c_l, grid=grid, dr=0)
                )
                # with open("debug/heliotron current tesla-meter.pkl", "rb") as file:
                #     enclosed_toroidal_current = pickle.load(file)
            term1 = data["psi_r"] / data["sqrt(g)"]
            term2 = data["g_tt"] * data["lambda_z"] - data["g_tz"] * (
                1 + data["lambda_t"]
            )
            num = surface_averages(grid, term1 * term2, match_grid=True)
            den = surface_averages(grid, term1 * data["g_tt"], match_grid=True)
            data["iota"] = (enclosed_toroidal_current + num) / den

        if check_derivs("iota_r", R_transform, Z_transform, L_transform):
            if current_r_from_fixed_iota is not None:
                # TODO: once debugged, this branch of the if statement can be removed,
                #   so that the else statement always executes.
                enclosed_toroidal_current_r = current_r_from_fixed_iota
            else:
                enclosed_toroidal_current_r = (
                    mu_0 / 2 / jnp.pi * current.compute(c_l, grid=grid, dr=1)
                )
                # with open("debug/heliotron current_r tesla-meter.pkl", "rb") as file:
                #     enclosed_toroidal_current_r = pickle.load(file)
            g_tt_r = 2 * dot(data["e_theta"], data["e_theta_r"])
            g_tz_r = dot(data["e_theta_r"], data["e_zeta"]) + dot(
                data["e_theta"], data["e_zeta_r"]
            )
            term1_r = (data["psi_rr"] - term1 * data["sqrt(g)_r"]) / data["sqrt(g)"]
            term2_r = (
                g_tt_r * data["lambda_z"]
                + data["g_tt"] * data["lambda_rz"]
                - g_tz_r * (1 + data["lambda_t"])
                - data["g_tz"] * data["lambda_rt"]
            )
            num_r = surface_averages(
                grid,
                term1_r * term2 + term1 * term2_r,
                match_grid=True,
            )
            den_r = surface_averages(
                grid,
                term1_r * data["g_tt"] + term1 * g_tt_r,
                match_grid=True,
            )
            data["iota_r"] = (
                enclosed_toroidal_current_r + num_r - data["iota"] * den_r
            ) / den

        # if check_derivs("iota_rr", R_transform, Z_transform, L_transform):
        #     # FIXME: Daniel asks: Do we need this term?
        #     #     Kaya responds: I thought we needed iota_rr for magnetic shear derivative stuff.
        #     enclosed_toroidal_current_rr = mu_0 / 2 / jnp.pi * current.compute(c_l, dr=2)
        #     g_tt_rr = 2 * (
        #         dot(data["e_theta_r"], data["e_theta_r"])
        #         + dot(data["e_theta"], data["e_theta_rr"])
        #     )
        #     g_tz_rr = (
        #         dot(data["e_theta_rr"], data["e_zeta"])
        #         + 2 * dot(data["e_theta_r"], data["e_zeta_r"])
        #         + dot(data["e_theta"], data["e_zeta_rr"])
        #     )
        #     term1_rr = (
        #         2 * (term1 * data["sqrt(g)_r"] - data["psi_rr"]) * data["sqrt(g)_r"]
        #         - data["psi_r"] * data["sqrt(g)_rr"]
        #     ) / jnp.square(data["sqrt(g)"])
        #     term2_rr = (
        #         g_tt_rr * data["lambda_z"]
        #         + 2 * g_tt_r * data["lambda_rz"]
        #         + data["g_tt"] * data["lambda_rrz"]
        #         - g_tz_rr * (1 + data["lambda_t"])
        #         - 2 * g_tz_r * data["lambda_rt"]
        #         - data["g_tz"] * data["lambda_rrt"]
        #     )
        #     num_rr = surface_averages(
        #         grid,
        #         term1_rr * term2 + 2 * term1_r * term2_r + term1 * term2_rr,
        #         match_grid=True,
        #     )
        #     den_rr = surface_averages(
        #         grid,
        #         term1_rr * data["g_tt"] + 2 * term1_r * g_tt_r + term1 * g_tt_rr,
        #         match_grid=True,
        #     )
        #     data["iota_rr"] = (
        #         enclosed_toroidal_current_rr
        #         + num_rr
        #         - 2 * (enclosed_toroidal_current_r + num_r) * den_r / den
        #         + data["iota"] * (2 * jnp.square(den_r) / den - den_rr)
        #     ) / den

    return data


def compute_covariant_basis(
    R_lmn,
    Z_lmn,
    R_transform,
    Z_transform,
    data=None,
    **kwargs,
):
    """Compute covariant basis vectors.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in the x
        direction, differentiated wrt y.

    """
    data = compute_toroidal_coords(R_lmn, Z_lmn, R_transform, Z_transform, data=data)
    data["0"] = jnp.zeros(R_transform.num_nodes)

    # 0th order derivatives
    if check_derivs("e_rho", R_transform, Z_transform):
        data["e_rho"] = jnp.array([data["R_r"], data["0"], data["Z_r"]]).T
    if check_derivs("e_theta", R_transform, Z_transform):
        data["e_theta"] = jnp.array([data["R_t"], data["0"], data["Z_t"]]).T
    if check_derivs("e_zeta", R_transform, Z_transform):
        data["e_zeta"] = jnp.array([data["R_z"], data["R"], data["Z_z"]]).T

    # 1st order derivatives
    if check_derivs("e_rho_r", R_transform, Z_transform):
        data["e_rho_r"] = jnp.array([data["R_rr"], data["0"], data["Z_rr"]]).T
    if check_derivs("e_rho_t", R_transform, Z_transform):
        data["e_rho_t"] = jnp.array([data["R_rt"], data["0"], data["Z_rt"]]).T
    if check_derivs("e_rho_z", R_transform, Z_transform):
        data["e_rho_z"] = jnp.array([data["R_rz"], data["0"], data["Z_rz"]]).T
    if check_derivs("e_theta_r", R_transform, Z_transform):
        data["e_theta_r"] = jnp.array([data["R_rt"], data["0"], data["Z_rt"]]).T
    if check_derivs("e_theta_t", R_transform, Z_transform):
        data["e_theta_t"] = jnp.array([data["R_tt"], data["0"], data["Z_tt"]]).T
    if check_derivs("e_theta_z", R_transform, Z_transform):
        data["e_theta_z"] = jnp.array([data["R_tz"], data["0"], data["Z_tz"]]).T
    if check_derivs("e_zeta_r", R_transform, Z_transform):
        data["e_zeta_r"] = jnp.array([data["R_rz"], data["R_r"], data["Z_rz"]]).T
    if check_derivs("e_zeta_t", R_transform, Z_transform):
        data["e_zeta_t"] = jnp.array([data["R_tz"], data["R_t"], data["Z_tz"]]).T
    if check_derivs("e_zeta_z", R_transform, Z_transform):
        data["e_zeta_z"] = jnp.array([data["R_zz"], data["R_z"], data["Z_zz"]]).T

    # 2nd order derivatives
    if check_derivs("e_rho_rr", R_transform, Z_transform):
        data["e_rho_rr"] = jnp.array([data["R_rrr"], data["0"], data["Z_rrr"]]).T
    if check_derivs("e_rho_tt", R_transform, Z_transform):
        data["e_rho_tt"] = jnp.array([data["R_rtt"], data["0"], data["Z_rtt"]]).T
    if check_derivs("e_rho_zz", R_transform, Z_transform):
        data["e_rho_zz"] = jnp.array([data["R_rzz"], data["0"], data["Z_rzz"]]).T
    if check_derivs("e_rho_rt", R_transform, Z_transform):
        data["e_rho_rt"] = jnp.array([data["R_rrt"], data["0"], data["Z_rrt"]]).T
    if check_derivs("e_rho_rz", R_transform, Z_transform):
        data["e_rho_rz"] = jnp.array([data["R_rrz"], data["0"], data["Z_rrz"]]).T
    if check_derivs("e_rho_tz", R_transform, Z_transform):
        data["e_rho_tz"] = jnp.array([data["R_rtz"], data["0"], data["Z_rtz"]]).T
    if check_derivs("e_theta_rr", R_transform, Z_transform):
        data["e_theta_rr"] = jnp.array([data["R_rrt"], data["0"], data["Z_rrt"]]).T
    if check_derivs("e_theta_tt", R_transform, Z_transform):
        data["e_theta_tt"] = jnp.array([data["R_ttt"], data["0"], data["Z_ttt"]]).T
    if check_derivs("e_theta_zz", R_transform, Z_transform):
        data["e_theta_zz"] = jnp.array([data["R_tzz"], data["0"], data["Z_tzz"]]).T
    if check_derivs("e_theta_rt", R_transform, Z_transform):
        data["e_theta_rt"] = jnp.array([data["R_rtt"], data["0"], data["Z_rtt"]]).T
    if check_derivs("e_theta_rz", R_transform, Z_transform):
        data["e_theta_rz"] = jnp.array([data["R_rtz"], data["0"], data["Z_rtz"]]).T
    if check_derivs("e_theta_tz", R_transform, Z_transform):
        data["e_theta_tz"] = jnp.array([data["R_ttz"], data["0"], data["Z_ttz"]]).T
    if check_derivs("e_zeta_rr", R_transform, Z_transform):
        data["e_zeta_rr"] = jnp.array([data["R_rrz"], data["R_rr"], data["Z_rrz"]]).T
    if check_derivs("e_zeta_tt", R_transform, Z_transform):
        data["e_zeta_tt"] = jnp.array([data["R_ttz"], data["R_tt"], data["Z_ttz"]]).T
    if check_derivs("e_zeta_zz", R_transform, Z_transform):
        data["e_zeta_zz"] = jnp.array([data["R_zzz"], data["R_zz"], data["Z_zzz"]]).T
    if check_derivs("e_zeta_rt", R_transform, Z_transform):
        data["e_zeta_rt"] = jnp.array([data["R_rtz"], data["R_rt"], data["Z_rtz"]]).T
    if check_derivs("e_zeta_rz", R_transform, Z_transform):
        data["e_zeta_rz"] = jnp.array([data["R_rzz"], data["R_rz"], data["Z_rzz"]]).T
    if check_derivs("e_zeta_tz", R_transform, Z_transform):
        data["e_zeta_tz"] = jnp.array([data["R_tzz"], data["R_tz"], data["Z_tzz"]]).T

    return data


def compute_contravariant_basis(
    R_lmn,
    Z_lmn,
    R_transform,
    Z_transform,
    data=None,
    **kwargs,
):
    """Compute contravariant basis vectors.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of contravariant basis vectors.
        Keys are of the form 'e^x_y', meaning the contravariant basis vector in the x
        direction, differentiated wrt y.

    """
    if data is None or "sqrt(g)" not in data:
        data = compute_jacobian(
            R_lmn,
            Z_lmn,
            R_transform,
            Z_transform,
            data=data,
        )

    if check_derivs("e^rho", R_transform, Z_transform):
        data["e^rho"] = (cross(data["e_theta"], data["e_zeta"]).T / data["sqrt(g)"]).T
    if check_derivs("e^theta", R_transform, Z_transform):
        data["e^theta"] = (cross(data["e_zeta"], data["e_rho"]).T / data["sqrt(g)"]).T
    if check_derivs("e^zeta", R_transform, Z_transform):
        data["e^zeta"] = jnp.array([data["0"], 1 / data["R"], data["0"]]).T

    return data


def compute_jacobian(
    R_lmn,
    Z_lmn,
    R_transform,
    Z_transform,
    data=None,
    **kwargs,
):
    """Compute coordinate system Jacobian.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of coordinate system Jacobian.
        Keys are of the form 'sqrt(g)_x', meaning the x derivative of the coordinate
        system Jacobian sqrt(g).

    """
    data = compute_covariant_basis(
        R_lmn,
        Z_lmn,
        R_transform,
        Z_transform,
        data=data,
    )

    if check_derivs("sqrt(g)", R_transform, Z_transform):
        data["sqrt(g)"] = dot(data["e_rho"], cross(data["e_theta"], data["e_zeta"]))
        data["|e_theta x e_zeta|"] = jnp.linalg.norm(
            cross(data["e_theta"], data["e_zeta"]), axis=1
        )
        data["|e_zeta x e_rho|"] = jnp.linalg.norm(
            cross(data["e_zeta"], data["e_rho"]), axis=1
        )
        data["|e_rho x e_theta|"] = jnp.linalg.norm(
            cross(data["e_rho"], data["e_theta"]), axis=1
        )

    # 1st order derivatives
    if check_derivs("sqrt(g)_r", R_transform, Z_transform):
        data["sqrt(g)_r"] = (
            dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_r"]))
        )
    if check_derivs("sqrt(g)_t", R_transform, Z_transform):
        data["sqrt(g)_t"] = (
            dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_t"]))
        )
    if check_derivs("sqrt(g)_z", R_transform, Z_transform):
        data["sqrt(g)_z"] = (
            dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_z"]))
        )

    # 2nd order derivatives
    if check_derivs("sqrt(g)_rr", R_transform, Z_transform):
        data["sqrt(g)_rr"] = (
            dot(data["e_rho_rr"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_rr"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_rr"]))
            + 2 * dot(data["e_rho_r"], cross(data["e_theta_r"], data["e_zeta"]))
            + 2 * dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta_r"]))
            + 2 * dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta_r"]))
        )
    if check_derivs("sqrt(g)_tt", R_transform, Z_transform):
        data["sqrt(g)_tt"] = (
            dot(data["e_rho_tt"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_tt"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_tt"]))
            + 2 * dot(data["e_rho_t"], cross(data["e_theta_t"], data["e_zeta"]))
            + 2 * dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta_t"]))
            + 2 * dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta_t"]))
        )
    if check_derivs("sqrt(g)_zz", R_transform, Z_transform):
        data["sqrt(g)_zz"] = (
            dot(data["e_rho_zz"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_zz"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_zz"]))
            + 2 * dot(data["e_rho_z"], cross(data["e_theta_z"], data["e_zeta"]))
            + 2 * dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta_z"]))
            + 2 * dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta_z"]))
        )
    if check_derivs("sqrt(g)_tz", R_transform, Z_transform):
        data["sqrt(g)_tz"] = (
            dot(data["e_rho_tz"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho_z"], cross(data["e_theta_t"], data["e_zeta"]))
            + dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta_t"]))
            + dot(data["e_rho_t"], cross(data["e_theta_z"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_tz"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta_t"]))
            + dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta_z"]))
            + dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta_z"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_tz"]))
        )

    return data


def compute_covariant_metric_coefficients(
    R_lmn,
    Z_lmn,
    R_transform,
    Z_transform,
    data=None,
    **kwargs,
):
    """Compute metric coefficients.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of metric coefficients.
        Keys are of the form 'g_xy', meaning the metric coefficient defined by the dot
        product of the covariant basis vectors e_x and e_y.

    """
    data = compute_covariant_basis(
        R_lmn,
        Z_lmn,
        R_transform,
        Z_transform,
        data=data,
    )

    if check_derivs("g_rr", R_transform, Z_transform):
        data["g_rr"] = dot(data["e_rho"], data["e_rho"])
    if check_derivs("g_tt", R_transform, Z_transform):
        data["g_tt"] = dot(data["e_theta"], data["e_theta"])
    if check_derivs("g_zz", R_transform, Z_transform):
        data["g_zz"] = dot(data["e_zeta"], data["e_zeta"])
    if check_derivs("g_rt", R_transform, Z_transform):
        data["g_rt"] = dot(data["e_rho"], data["e_theta"])
    if check_derivs("g_rz", R_transform, Z_transform):
        data["g_rz"] = dot(data["e_rho"], data["e_zeta"])
    if check_derivs("g_tz", R_transform, Z_transform):
        data["g_tz"] = dot(data["e_theta"], data["e_zeta"])

    return data


def compute_contravariant_metric_coefficients(
    R_lmn,
    Z_lmn,
    R_transform,
    Z_transform,
    data=None,
    **kwargs,
):
    """Compute reciprocal metric coefficients.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of reciprocal metric coefficients.
        Keys are of the form 'g^xy', meaning the metric coefficient defined by the dot
        product of the contravariant basis vectors e^x and e^y.

    """
    data = compute_contravariant_basis(
        R_lmn,
        Z_lmn,
        R_transform,
        Z_transform,
        data=data,
    )

    if check_derivs("g^rr", R_transform, Z_transform):
        data["g^rr"] = dot(data["e^rho"], data["e^rho"])
    if check_derivs("g^tt", R_transform, Z_transform):
        data["g^tt"] = dot(data["e^theta"], data["e^theta"])
    if check_derivs("g^zz", R_transform, Z_transform):
        data["g^zz"] = dot(data["e^zeta"], data["e^zeta"])
    if check_derivs("g^rt", R_transform, Z_transform):
        data["g^rt"] = dot(data["e^rho"], data["e^theta"])
    if check_derivs("g^rz", R_transform, Z_transform):
        data["g^rz"] = dot(data["e^rho"], data["e^zeta"])
    if check_derivs("g^tz", R_transform, Z_transform):
        data["g^tz"] = dot(data["e^theta"], data["e^zeta"])

    if check_derivs("|grad(rho)|", R_transform, Z_transform):
        data["|grad(rho)|"] = jnp.sqrt(data["g^rr"])
    if check_derivs("|grad(theta)|", R_transform, Z_transform):
        data["|grad(theta)|"] = jnp.sqrt(data["g^tt"])
    if check_derivs("|grad(zeta)|", R_transform, Z_transform):
        data["|grad(zeta)|"] = jnp.sqrt(data["g^zz"])

    return data


def compute_geometry(
    R_lmn,
    Z_lmn,
    R_transform,
    Z_transform,
    data=None,
    **kwargs,
):
    """Compute plasma volume and other geometric quantities such as effective minor/major radius and aspect ratio.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) with volume key "V", cross-sectional area "A",
        minor radius "a", major radius "R0", and aspect ration "R0/a".

    """
    data = compute_jacobian(R_lmn, Z_lmn, R_transform, Z_transform, data=data)

    # Poincare cross-section weights
    xs_weights = jnp.prod(R_transform.grid.spacing[:, :-1], axis=1)
    # number of toroidal grid points
    N = R_transform.grid.num_zeta

    data["V"] = jnp.sum(jnp.abs(data["sqrt(g)"]) * R_transform.grid.weights)
    data["A"] = jnp.mean(
        jnp.sum(  # sqrt(g) / R * weight = dArea
            jnp.reshape(jnp.abs(data["sqrt(g)"] / data["R"]) * xs_weights, (N, -1)),
            axis=1,
        )
    )
    data["R0"] = data["V"] / (2 * jnp.pi * data["A"])
    data["a"] = jnp.sqrt(data["A"] / jnp.pi)
    data["R0/a"] = data["V"] / (2 * jnp.sqrt(jnp.pi * data["A"] ** 3))

    return data
