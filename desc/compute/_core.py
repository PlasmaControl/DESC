"""Core compute functions, for profiles, geometry, and basis vectors/jacobians."""

import numpy as np

from desc.backend import jnp
from desc.compute import data_index
from desc.grid import LinearGrid


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


def dot(a, b, axis=-1):
    """Batched vector dot product.

    Parameters
    ----------
    a : array-like
        First array of vectors.
    b : array-like
        Second array of vectors.
    axis : int
        Axis along which vectors are stored.

    Returns
    -------
    y : array-like
        y = sum(a*b, axis=axis)

    """
    return jnp.sum(a * b, axis=axis, keepdims=False)


def cross(a, b, axis=-1):
    """Batched vector cross product.

    Parameters
    ----------
    a : array-like
        First array of vectors.
    b : array-like
        Second array of vectors.
    axis : int
        Axis along which vectors are stored.

    Returns
    -------
    y : array-like
        y = a x b

    """
    return jnp.cross(a, b, axis=axis)


def compute_flux_coords(
    iota,
    data=None,
    **kwargs,
):
    """Compute flux coordinates (rho,theta,zeta).

    Parameters
    ----------
    iota : Profile
        Transforms i_l coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of flux coordinates.

    """
    if data is None:
        data = {}

    data["rho"] = iota.grid.nodes[:, 0]
    data["theta"] = iota.grid.nodes[:, 1]
    data["zeta"] = iota.grid.nodes[:, 2]

    return data


def compute_toroidal_flux(
    Psi,
    iota,
    data=None,
    **kwargs,
):
    """Compute toroidal magnetic flux profile.

    Parameters
    ----------
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface, in Webers.
    iota : Profile
        Transforms i_l coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of toroidal magnetic flux profile.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    data = compute_flux_coords(iota, data=data)

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
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of Cartesian coordinates.

    """
    data = compute_flux_coords(R_transform, data=data)
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
        Keys are of the form 'lambda_x' meaning the derivative of lambda wrt to x.

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
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    if data is None:
        data = {}

    data["p"] = pressure.compute(p_l, dr=0)
    data["p_r"] = pressure.compute(p_l, dr=1)

    return data


def compute_rotational_transform(
    i_l,
    iota,
    data=None,
    **kwargs,
):
    """Compute rotational transform profile.

    Parameters
    ----------
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    iota : Profile
        Transforms i_l coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of rotational transform profile.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    if data is None:
        data = {}

    data["iota"] = iota.compute(i_l, dr=0)
    data["iota_r"] = iota.compute(i_l, dr=1)
    data["iota_rr"] = iota.compute(i_l, dr=2)

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
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
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
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
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
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
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
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
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
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
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
    """Compute plasma volume.

    Parameters
    ----------
    R_lmn : ndarray
        Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
    Z_lmn : ndarray
        Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
    R_transform : Transform
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform
        Transforms Z_lmn coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) with volume key "V".

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
