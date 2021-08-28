import numpy as np
from scipy.constants import mu_0

from desc.backend import jnp, put
from desc.compute import data_index
from desc.grid import Grid


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
    data["psi_rr"] = 2 * Psi * np.ones_like(data["rho"]) / (2 * jnp.pi)

    return data


def compute_toroidal_coords(
    R_lmn,
    Z_lmn,
    R_transform,
    Z_transform,
    data=None,
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
    data["0"] = jnp.zeros_like(data["R"])

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
    if check_derivs("|grad(|B|^2)|", R_transform, Z_transform, L_transform):
        data["|grad(|B|^2)|"] = jnp.sqrt(
            data["grad(|B|^2)_rho"] ** 2 * data["g^rr"]
            + data["grad(|B|^2)_theta"] ** 2 * data["g^tt"]
            + data["grad(|B|^2)_zeta"] ** 2 * data["g^zz"]
            + 2 * data["grad(|B|^2)_rho"] * data["grad(|B|^2)_theta"] * data["g^rt"]
            + 2 * data["grad(|B|^2)_rho"] * data["grad(|B|^2)_zeta"] * data["g^rz"]
            + 2 * data["grad(|B|^2)_theta"] * data["grad(|B|^2)_zeta"] * data["g^tz"]
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

    return data


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

    if check_derivs("|grad(p)|", R_transform, Z_transform, L_transform):
        data["|grad(p)|"] = jnp.sqrt(data["p_r"] ** 2) * data["|grad(rho)|"]
    if check_derivs("|beta|", R_transform, Z_transform, L_transform):
        data["|beta|"] = jnp.sqrt(
            data["B^zeta"] ** 2 * data["g^tt"]
            + data["B^theta"] ** 2 * data["g^zz"]
            - 2 * data["B^theta"] * data["B^zeta"] * data["g^tz"]
        )

    return data


def compute_boozer_coords(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    B_transform,
    nu_transform,
    iota,
    data=None,
):
    """Compute Boozer coordinates.

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
    B_transform : Transform
        Transforms spectral coefficients of B(rho,theta,zeta) to real space.
    nu_transform : Transform
        Transforms spectral coefficients of nu(rho,theta,zeta) to real space.
    iota : Profile
        Transforms i_l coefficients to real space.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of Boozer harmonics.

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
    # TODO: can remove this call if compute_|B| changed to use B_covariant
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

    idx0 = jnp.where((B_transform.basis.modes == [0, 0, 0]).all(axis=1))[0]

    # covariant Boozer components: I = B_theta, G = B_zeta (in Boozer coordinates)
    if check_derivs("I", R_transform, Z_transform, L_transform):
        B_theta_mn = B_transform.fit(data["B_theta"])
        data["I"] = B_theta_mn[idx0]
    if check_derivs("G", R_transform, Z_transform, L_transform):
        B_zeta_mn = B_transform.fit(data["B_zeta"])
        data["G"] = B_zeta_mn[idx0]

    # QS Boozer harmonics
    lambda_mn = nu_transform.fit(data["lambda"])  # FIXME: evaluate this exactly!
    nu_mn = jnp.zeros_like(lambda_mn)
    for k, (l, m, n) in enumerate(nu_transform.basis.modes):
        if m != 0:
            idx = jnp.where((B_transform.basis.modes == [0, -m, n]).all(axis=1))[0]
            nu_mn = put(
                nu_mn,
                k,
                (
                    (B_theta_mn[idx] / m - data["I"] * lambda_mn[k])
                    / (data["G"] + data["iota"][0] * data["I"])
                )[0],
            )
        elif n != 0:
            idx = jnp.where((B_transform.basis.modes == [0, m, -n]).all(axis=1))[0]
            nu_mn = put(
                nu_mn,
                k,
                (
                    (B_zeta_mn[idx] / n - data["I"] * lambda_mn[k])
                    / (data["G"] + data["iota"][0] * data["I"])
                )[0],
            )
    data["nu"] = nu_transform.transform(nu_mn)
    data["Boozer modes"] = B_transform.basis.modes

    if check_derivs("|B|_mn", R_transform, Z_transform, L_transform):
        b_nodes = nu_transform.grid.nodes
        b_nodes[:, 2] = b_nodes[:, 2] - data["nu"]
        b_nodes[:, 1] = b_nodes[:, 1] - data["lambda"] - data["iota"] * data["nu"]
        b_grid = Grid(b_nodes)
        B_transform.grid = b_grid
        data["|B|_mn"] = B_transform.fit(data["|B|"])

    return data


def compute_quasisymmetry_error(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    helicity=(1, 0),
    data=None,
):
    """Compute quasi-symmetry errors.

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
    helicity : tuple, int
        Type of quasi-symmetry (M, N).

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of quasi-symmetry errors.
        Key "QS_FF" is the flux function metric, key "QS_TP" is the triple product.

    """
    data = compute_B_dot_gradB(
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
    # TODO: can remove this call if compute_|B| changed to use B_covariant
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

    M = helicity[0]
    N = helicity[1]

    # covariant Boozer components: I = B_theta, G = B_zeta (in Boozer coordinates)
    if check_derivs("I", R_transform, Z_transform, L_transform):
        data["I"] = jnp.mean(data["B_theta"] * data["sqrt(g)"]) / jnp.mean(
            data["sqrt(g)"]
        )
    if check_derivs("G", R_transform, Z_transform, L_transform):
        data["G"] = jnp.mean(data["B_zeta"] * data["sqrt(g)"]) / jnp.mean(
            data["sqrt(g)"]
        )

    # QS flux function (T^3)
    if check_derivs("QS_FF", R_transform, Z_transform, L_transform):
        data["QS_FF"] = (data["psi_r"] / data["sqrt(g)"]) * (
            data["B_zeta"] * data["|B|_t"] - data["B_theta"] * data["|B|_z"]
        ) - (M * data["G"] + N * data["I"]) / (M * data["iota"] - N) * data[
            "B*grad(|B|)"
        ]
    # QS triple product (T^4/m^2)
    if check_derivs("QS_TP", R_transform, Z_transform, L_transform):
        data["QS_TP"] = (data["psi_r"] / data["sqrt(g)"]) * (
            data["|B|_t"] * data["(B*grad(|B|))_z"]
            - data["|B|_z"] * data["(B*grad(|B|))_t"]
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


def compute_geometry(
    R_lmn,
    Z_lmn,
    R_transform,
    Z_transform,
    data=None,
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

    # FIXME: this is not differentiable with JAX -- reference grid.N instead?
    N = jnp.unique(R_transform.grid.nodes[:, -1]).size  # number of toroidal angles
    weights = R_transform.grid.weights / (2 * jnp.pi / N)  # remove toroidal weights

    data["V"] = jnp.sum(jnp.abs(data["sqrt(g)"]) * R_transform.grid.weights)
    data["A"] = jnp.mean(
        jnp.sum(  # sqrt(g) / R * weight = dArea
            jnp.reshape(jnp.abs(data["sqrt(g)"] / data["R"]) * weights, (N, -1)),
            axis=1,
        )
    )
    data["R0"] = data["V"] / (2 * jnp.pi * data["A"])
    data["a"] = jnp.sqrt(data["A"] / jnp.pi)
    data["R0/a"] = data["V"] / (2 * jnp.sqrt(jnp.pi * data["A"] ** 3))

    return data
