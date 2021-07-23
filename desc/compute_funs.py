import numpy as np
from desc.backend import jnp
from scipy.constants import mu_0

"""These functions perform the core calculations of physical quantities."""


def dot(a, b, axis):
    """Batched vector dot product.

    Parameters
    ----------
    a : array-like
        first array of vectors
    b : array-like
        second array of vectors
    axis : int
        axis along which vectors are stored

    Returns
    -------
    y : array-like
        y = sum(a*b, axis=axis)

    """
    return jnp.sum(a * b, axis=axis, keepdims=False)


def cross(a, b, axis):
    """Batched vector cross product.

    Parameters
    ----------
    a : array-like
        first array of vectors
    b : array-like
        second array of vectors
    axis : int
        axis along which vectors are stored

    Returns
    -------
    y : array-like
        y = a x b

    """
    return jnp.cross(a, b, axis=axis)


def compute_toroidal_flux(
    Psi, iota, dr=0, data=None,
):
    """Compute toroidal magnetic flux profile.

    Parameters
    ----------
    Psi : float
        Total toroidal magnetic flux within the last closed flux surface, in Webers.
    iota : Profile
        Transforms i_l coefficients to real space.
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of toroidal magnetic flux profile.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    if data is None:
        data = {}

    # toroidal flux (Wb) divided by 2 pi
    rho = iota.grid.nodes[:, 0]
    data["psi"] = Psi * rho ** 2 / (2 * jnp.pi)
    if dr > 0:
        data["psi_r"] = 2 * Psi * rho / (2 * jnp.pi)
    if dr > 1:
        data["psi_rr"] = 2 * Psi * np.ones_like(rho) / (2 * jnp.pi)
    return data


def compute_iota(
    i_l, iota, dr=0, data=None,
):
    """Compute rotational transform profile.

    Parameters
    ----------
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    iota : Profile
        Transforms i_l coefficients to real space.
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of rotational transform profile.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    if data is None:
        data = {}

    data["iota"] = iota.compute(i_l, dr=0)
    if dr > 0:
        data["iota_r"] = iota.compute(i_l, dr=1)
    if dr > 1:
        data["iota_rr"] = iota.compute(i_l, dr=2)

    return data


def compute_pressure(
    p_l, pressure, dr=0, data=None,
):
    """Compute pressure profile.

    Parameters
    ----------
    p_l : ndarray
        Spectral coefficients of p(rho) -- pressure profile.
    pressure : Profile
        Transforms p_l coefficients to real space.
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of pressure profile.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    if data is None:
        data = {}

    data["p"] = pressure.compute(p_l, dr=0)
    if dr > 0:
        data["p_r"] = pressure.compute(p_l, dr=1)

    return data


def compute_lambda(
    L_lmn, L_transform, dr=0, dt=0, dz=0, data=None,
):
    """Compute lambda such that theta* = theta + lambda is a sfl coordinate.

    Parameters
    ----------
    L_lmn : ndarray
        Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
    L_transform : Transform
        Transforms L_lmn coefficients to real space.
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.
    dt : int, optional
        Order of derivative wrt the poloidal coordinate, theta.
    dz : int, optional
        Order of derivative wrt the toroidal coordinate, zeta.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of lambda values.
        Keys are of the form 'lambda_x' meaning the derivative of lambda wrt to x.

    """
    if data is None:
        data = {}

    data["lambda"] = L_transform.transform(L_lmn)

    # 1st order derivatives
    if dr > 0:
        data["lambda_r"] = L_transform.transform(L_lmn, 1, 0, 0)
    if dt > 0:
        data["lambda_t"] = L_transform.transform(L_lmn, 0, 1, 0)
    if dz > 0:
        data["lambda_z"] = L_transform.transform(L_lmn, 0, 0, 1)

    # 2nd order derivatives
    if dr > 1:
        data["lambda_rr"] = L_transform.transform(L_lmn, 2, 0, 0)
    if dt > 1:
        data["lambda_tt"] = L_transform.transform(L_lmn, 0, 2, 0)
    if dz > 1:
        data["lambda_zz"] = L_transform.transform(L_lmn, 0, 0, 2)
    if dr > 0 and dt > 0 and (dr > 1 or dt > 1):
        data["lambda_rt"] = L_transform.transform(L_lmn, 1, 1, 0)
    if dr > 0 and dz > 0 and (dr > 1 or dz > 1):
        data["lambda_rz"] = L_transform.transform(L_lmn, 1, 0, 1)
    if dt > 0 and dz > 0 and (dt > 1 or dz > 1):
        data["lambda_tz"] = L_transform.transform(L_lmn, 0, 1, 1)

    # 3rd order derivatives
    if dr > 2:
        data["lambda_rrr"] = L_transform.transform(L_lmn, 3, 0, 0)
    if dt > 2:
        data["lambda_ttt"] = L_transform.transform(L_lmn, 0, 3, 0)
    if dz > 2:
        data["lambda_zzz"] = L_transform.transform(L_lmn, 0, 0, 3)
    if dr > 1 and dt > 0 and (dr > 2 or dt > 2):
        data["lambda_rrt"] = L_transform.transform(L_lmn, 2, 1, 0)
    if dr > 0 and dt > 1 and (dr > 2 or dt > 2):
        data["lambda_rtt"] = L_transform.transform(L_lmn, 1, 2, 0)
    if dr > 1 and dz > 0 and (dr > 2 or dz > 2):
        data["lambda_rrz"] = L_transform.transform(L_lmn, 2, 0, 1)
    if dr > 0 and dz > 1 and (dr > 2 or dz > 2):
        data["lambda_rzz"] = L_transform.transform(L_lmn, 1, 0, 2)
    if dt > 1 and dz > 0 and (dt > 2 or dz > 2):
        data["lambda_ttz"] = L_transform.transform(L_lmn, 0, 2, 1)
    if dt > 0 and dz > 1 and (dt > 2 or dz > 2):
        data["lambda_tzz"] = L_transform.transform(L_lmn, 0, 1, 2)
    if dr > 0 and dt > 0 and dz > 0 and (dr > 2 or dt > 2 or dz > 2):
        data["lambda_rtz"] = L_transform.transform(L_lmn, 1, 1, 1)

    return data


def compute_toroidal_coords(
    R_lmn, Z_lmn, R_transform, Z_transform, dr=0, dt=0, dz=0, data=None,
):
    """Compute toroidal coordinates (R, phi, Z).

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
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.
    dt : int, optional
        Order of derivative wrt the poloidal coordinate, theta.
    dz : int, optional
        Order of derivative wrt the toroidal coordinate, zeta.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt y.

    """
    if data is None:
        data = {}

    data["R"] = R_transform.transform(R_lmn)
    data["Z"] = Z_transform.transform(Z_lmn)
    data["0"] = jnp.zeros_like(data["R"])

    # 1st order derivatives
    if dr > 0:
        data["R_r"] = R_transform.transform(R_lmn, 1, 0, 0)
        data["Z_r"] = Z_transform.transform(Z_lmn, 1, 0, 0)
    if dt > 0:
        data["R_t"] = R_transform.transform(R_lmn, 0, 1, 0)
        data["Z_t"] = Z_transform.transform(Z_lmn, 0, 1, 0)
    if dz > 0:
        data["R_z"] = R_transform.transform(R_lmn, 0, 0, 1)
        data["Z_z"] = Z_transform.transform(Z_lmn, 0, 0, 1)

    # 2nd order derivatives
    if dr > 1:
        data["R_rr"] = R_transform.transform(R_lmn, 2, 0, 0)
        data["Z_rr"] = Z_transform.transform(Z_lmn, 2, 0, 0)
    if dt > 1:
        data["R_tt"] = R_transform.transform(R_lmn, 0, 2, 0)
        data["Z_tt"] = Z_transform.transform(Z_lmn, 0, 2, 0)
    if dz > 1:
        data["R_zz"] = R_transform.transform(R_lmn, 0, 0, 2)
        data["Z_zz"] = Z_transform.transform(Z_lmn, 0, 0, 2)
    if dr > 0 and dt > 0 and (dr > 1 or dt > 1):
        data["R_rt"] = R_transform.transform(R_lmn, 1, 1, 0)
        data["Z_rt"] = Z_transform.transform(Z_lmn, 1, 1, 0)
    if dr > 0 and dz > 0 and (dr > 1 or dz > 1):
        data["R_rz"] = R_transform.transform(R_lmn, 1, 0, 1)
        data["Z_rz"] = Z_transform.transform(Z_lmn, 1, 0, 1)
    if dt > 0 and dz > 0 and (dt > 1 or dz > 1):
        data["R_tz"] = R_transform.transform(R_lmn, 0, 1, 1)
        data["Z_tz"] = Z_transform.transform(Z_lmn, 0, 1, 1)

    # 3rd order derivatives
    if dr > 2:
        data["R_rrr"] = R_transform.transform(R_lmn, 3, 0, 0)
        data["Z_rrr"] = Z_transform.transform(Z_lmn, 3, 0, 0)
    if dt > 2:
        data["R_ttt"] = R_transform.transform(R_lmn, 0, 3, 0)
        data["Z_ttt"] = Z_transform.transform(Z_lmn, 0, 3, 0)
    if dz > 2:
        data["R_zzz"] = R_transform.transform(R_lmn, 0, 0, 3)
        data["Z_zzz"] = Z_transform.transform(Z_lmn, 0, 0, 3)
    if dr > 1 and dt > 0 and (dr > 2 or dt > 2):
        data["R_rrt"] = R_transform.transform(R_lmn, 2, 1, 0)
        data["Z_rrt"] = Z_transform.transform(Z_lmn, 2, 1, 0)
    if dr > 0 and dt > 1 and (dr > 2 or dt > 2):
        data["R_rtt"] = R_transform.transform(R_lmn, 1, 2, 0)
        data["Z_rtt"] = Z_transform.transform(Z_lmn, 1, 2, 0)
    if dr > 1 and dz > 0 and (dr > 2 or dz > 2):
        data["R_rrz"] = R_transform.transform(R_lmn, 2, 0, 1)
        data["Z_rrz"] = Z_transform.transform(Z_lmn, 2, 0, 1)
    if dr > 0 and dz > 1 and (dr > 2 or dz > 2):
        data["R_rzz"] = R_transform.transform(R_lmn, 1, 0, 2)
        data["Z_rzz"] = Z_transform.transform(Z_lmn, 1, 0, 2)
    if dt > 1 and dz > 0 and (dt > 2 or dz > 2):
        data["R_ttz"] = R_transform.transform(R_lmn, 0, 2, 1)
        data["Z_ttz"] = Z_transform.transform(Z_lmn, 0, 2, 1)
    if dt > 0 and dz > 1 and (dt > 2 or dz > 2):
        data["R_tzz"] = R_transform.transform(R_lmn, 0, 1, 2)
        data["Z_tzz"] = Z_transform.transform(Z_lmn, 0, 1, 2)
    if dr > 0 and dt > 0 and dz > 0 and (dr > 2 or dt > 2 or dz > 2):
        data["R_rtz"] = R_transform.transform(R_lmn, 1, 1, 1)
        data["Z_rtz"] = Z_transform.transform(Z_lmn, 1, 1, 1)

    return data


def compute_cartesian_coords(
    R_lmn, Z_lmn, R_transform, Z_transform, data=None,
):
    """Compute Cartesian coordinates (X, Y, Z).

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
    data = compute_toroidal_coords(R_lmn, Z_lmn, R_transform, Z_transform, data=data)

    phi = R_transform.grid.nodes[:, 2]
    data["X"] = data["R"] * np.cos(phi)
    data["Y"] = data["R"] * np.sin(phi)
    data["Z"] = data["Z"]

    return data


def compute_covariant_basis(
    R_lmn, Z_lmn, R_transform, Z_transform, dr=0, dt=0, dz=0, data=None,
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
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.
    dt : int, optional
        Order of derivative wrt the poloidal coordinate, theta.
    dz : int, optional
        Order of derivative wrt the toroidal coordinate, zeta.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in the x
        direction, differentiated wrt y.

    """
    data = compute_toroidal_coords(
        R_lmn,
        Z_lmn,
        R_transform,
        Z_transform,
        dr=dr + 1,
        dt=dt + 1,
        dz=dz + 1,
        data=data,
    )

    data["e_rho"] = jnp.array([data["R_r"], data["0"], data["Z_r"]])
    data["e_theta"] = jnp.array([data["R_t"], data["0"], data["Z_t"]])
    data["e_zeta"] = jnp.array([data["R_z"], data["R"], data["Z_z"]])

    # 1st order derivatives
    if dr > 0:
        data["e_rho_r"] = jnp.array([data["R_rr"], data["0"], data["Z_rr"]])
        data["e_theta_r"] = jnp.array([data["R_rt"], data["0"], data["Z_rt"]])
        data["e_zeta_r"] = jnp.array([data["R_rz"], data["R_r"], data["Z_rz"]])
    if dt > 0:
        data["e_rho_t"] = jnp.array([data["R_rt"], data["0"], data["Z_rt"]])
        data["e_theta_t"] = jnp.array([data["R_tt"], data["0"], data["Z_tt"]])
        data["e_zeta_t"] = jnp.array([data["R_tz"], data["R_t"], data["Z_tz"]])
    if dz > 0:
        data["e_rho_z"] = jnp.array([data["R_rz"], data["0"], data["Z_rz"]])
        data["e_theta_z"] = jnp.array([data["R_tz"], data["0"], data["Z_tz"]])
        data["e_zeta_z"] = jnp.array([data["R_zz"], data["R_z"], data["Z_zz"]])

    # 2nd order derivatives
    if dr > 1:
        data["e_rho_rr"] = jnp.array([data["R_rrr"], data["0"], data["Z_rrr"]])
        data["e_theta_rr"] = jnp.array([data["R_rrt"], data["0"], data["Z_rrt"]])
        data["e_zeta_rr"] = jnp.array([data["R_rrz"], data["R_rr"], data["Z_rrz"]])
    if dt > 1:
        data["e_rho_tt"] = jnp.array([data["R_rtt"], data["0"], data["Z_rtt"]])
        data["e_theta_tt"] = jnp.array([data["R_ttt"], data["0"], data["Z_ttt"]])
        data["e_zeta_tt"] = jnp.array([data["R_ttz"], data["R_tt"], data["Z_ttz"]])
    if dz > 1:
        data["e_rho_zz"] = jnp.array([data["R_rzz"], data["0"], data["Z_rzz"]])
        data["e_theta_zz"] = jnp.array([data["R_tzz"], data["0"], data["Z_tzz"]])
        data["e_zeta_zz"] = jnp.array([data["R_zzz"], data["R_zz"], data["Z_zzz"]])
    if dr > 0 and dt > 0 and (dr > 1 or dt > 1):
        data["e_rho_rt"] = jnp.array([data["R_rrt"], data["0"], data["Z_rrt"]])
        data["e_theta_rt"] = jnp.array([data["R_rtt"], data["0"], data["Z_rtt"]])
        data["e_zeta_rt"] = jnp.array([data["R_rtz"], data["R_rt"], data["Z_rtz"]])
    if dr > 0 and dz > 0 and (dr > 1 or dz > 1):
        data["e_rho_rz"] = jnp.array([data["R_rrz"], data["0"], data["Z_rrz"]])
        data["e_theta_rz"] = jnp.array([data["R_rtz"], data["0"], data["Z_rtz"]])
        data["e_zeta_rz"] = jnp.array([data["R_rzz"], data["R_rz"], data["Z_rzz"]])
    if dt > 0 and dz > 0 and (dt > 1 or dz > 1):
        data["e_rho_tz"] = jnp.array([data["R_rtz"], data["0"], data["Z_rtz"]])
        data["e_theta_tz"] = jnp.array([data["R_ttz"], data["0"], data["Z_ttz"]])
        data["e_zeta_tz"] = jnp.array([data["R_tzz"], data["R_tz"], data["Z_tzz"]])

    return data


def compute_jacobian(
    R_lmn, Z_lmn, R_transform, Z_transform, dr=0, dt=0, dz=0, data=None,
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
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.
    dt : int, optional
        Order of derivative wrt the poloidal coordinate, theta.
    dz : int, optional
        Order of derivative wrt the toroidal coordinate, zeta.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of coordinate system Jacobian.
        Keys are of the form 'sqrt(g)_x', meaning the x derivative of the coordinate
        system Jacobian sqrt(g).

    """
    data = compute_covariant_basis(
        R_lmn, Z_lmn, R_transform, Z_transform, dr=dr, dt=dt, dz=dz, data=data
    )

    data["sqrt(g)"] = dot(data["e_rho"], cross(data["e_theta"], data["e_zeta"], 0), 0)

    # 1st order derivatives
    if dr > 0:
        data["sqrt(g)_r"] = (
            dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta"], 0), 0)
            + dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta"], 0), 0)
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_r"], 0), 0)
        )
    if dt > 0:
        data["sqrt(g)_t"] = (
            dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta"], 0), 0)
            + dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta"], 0), 0)
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_t"], 0), 0)
        )
    if dz > 0:
        data["sqrt(g)_z"] = (
            dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta"], 0), 0)
            + dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta"], 0), 0)
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_z"], 0), 0)
        )

    # 2nd order derivatives
    if dr > 1:
        data["sqrt(g)_rr"] = (
            dot(data["e_rho_rr"], cross(data["e_theta"], data["e_zeta"], 0), 0,)
            + dot(data["e_rho"], cross(data["e_theta_rr"], data["e_zeta"], 0), 0,)
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_rr"], 0), 0,)
            + 2 * dot(data["e_rho_r"], cross(data["e_theta_r"], data["e_zeta"], 0), 0,)
            + 2 * dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta_r"], 0), 0,)
            + 2 * dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta_r"], 0), 0,)
        )
    if dt > 1:
        data["sqrt(g)_tt"] = (
            dot(data["e_rho_tt"], cross(data["e_theta"], data["e_zeta"], 0), 0,)
            + dot(data["e_rho"], cross(data["e_theta_tt"], data["e_zeta"], 0), 0,)
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_tt"], 0), 0,)
            + 2 * dot(data["e_rho_t"], cross(data["e_theta_t"], data["e_zeta"], 0), 0,)
            + 2 * dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta_t"], 0), 0,)
            + 2 * dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta_t"], 0), 0,)
        )
    if dz > 1:
        data["sqrt(g)_zz"] = (
            dot(data["e_rho_zz"], cross(data["e_theta"], data["e_zeta"], 0), 0,)
            + dot(data["e_rho"], cross(data["e_theta_zz"], data["e_zeta"], 0), 0,)
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_zz"], 0), 0,)
            + 2 * dot(data["e_rho_z"], cross(data["e_theta_z"], data["e_zeta"], 0), 0,)
            + 2 * dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta_z"], 0), 0,)
            + 2 * dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta_z"], 0), 0,)
        )
    if dr > 0 and dt > 0 and (dr > 1 or dt > 1):
        raise NotImplementedError
    if dr > 0 and dz > 0 and (dr > 1 or dz > 1):
        raise NotImplementedError
    if dt > 0 and dz > 0 and (dt > 1 or dz > 1):
        data["sqrt(g)_tz"] = (
            dot(data["e_rho_tz"], cross(data["e_theta"], data["e_zeta"], 0), 0,)
            + dot(data["e_rho_z"], cross(data["e_theta_t"], data["e_zeta"], 0), 0,)
            + dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta_t"], 0), 0,)
            + dot(data["e_rho_t"], cross(data["e_theta_z"], data["e_zeta"], 0), 0,)
            + dot(data["e_rho"], cross(data["e_theta_tz"], data["e_zeta"], 0), 0,)
            + dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta_t"], 0), 0,)
            + dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta_z"], 0), 0,)
            + dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta_z"], 0), 0,)
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_tz"], 0), 0,)
        )

    return data


def compute_contravariant_basis(
    R_lmn, Z_lmn, R_transform, Z_transform, dr=0, dt=0, dz=0, data=None,
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
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.
    dt : int, optional
        Order of derivative wrt the poloidal coordinate, theta.
    dz : int, optional
        Order of derivative wrt the toroidal coordinate, zeta.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of contravariant basis vectors.
        Keys are of the form 'e^x_y', meaning the contravariant basis vector in the x
        direction, differentiated wrt y.

    """
    if data is None or "sqrt(g)" not in data:
        data = compute_jacobian(
            R_lmn, Z_lmn, R_transform, Z_transform, dr=dr, dt=dt, dz=dz, data=data,
        )

    data["e^rho"] = cross(data["e_theta"], data["e_zeta"], 0) / data["sqrt(g)"]
    data["e^theta"] = cross(data["e_zeta"], data["e_rho"], 0) / data["sqrt(g)"]
    data["e^zeta"] = jnp.array([data["0"], 1 / data["R"], data["0"]])

    return data


def compute_covariant_metric_coefficients(
    R_lmn, Z_lmn, R_transform, Z_transform, data=None,
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
    if data is None or "e_rho" not in data:
        data = compute_covariant_basis(
            R_lmn, Z_lmn, R_transform, Z_transform, dr=0, dt=0, dz=0, data=data,
        )

    data["g_rr"] = dot(data["e_rho"], data["e_rho"], 0)
    data["g_tt"] = dot(data["e_theta"], data["e_theta"], 0)
    data["g_zz"] = dot(data["e_zeta"], data["e_zeta"], 0)
    data["g_rt"] = dot(data["e_rho"], data["e_theta"], 0)
    data["g_rz"] = dot(data["e_rho"], data["e_zeta"], 0)
    data["g_tz"] = dot(data["e_theta"], data["e_zeta"], 0)

    return data


def compute_contravariant_metric_coefficients(
    R_lmn, Z_lmn, R_transform, Z_transform, data=None,
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
    if data is None or "e^rho" not in data:
        data = compute_contravariant_basis(
            R_lmn, Z_lmn, R_transform, Z_transform, dr=0, dt=0, dz=0, data=data,
        )

    data["g^rr"] = dot(data["e^rho"], data["e^rho"], 0)
    data["g^tt"] = dot(data["e^theta"], data["e^theta"], 0)
    data["g^zz"] = dot(data["e^zeta"], data["e^zeta"], 0)
    data["g^rt"] = dot(data["e^rho"], data["e^theta"], 0)
    data["g^rz"] = dot(data["e^rho"], data["e^zeta"], 0)
    data["g^tz"] = dot(data["e^theta"], data["e^zeta"], 0)

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
    dr=0,
    dt=0,
    dz=0,
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
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.
    dt : int, optional
        Order of derivative wrt the poloidal coordinate, theta.
    dz : int, optional
        Order of derivative wrt the toroidal coordinate, zeta.

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of contravariant magnetic field
        components. Keys are of the form 'B^x_y', meaning the x contravariant (B^x)
        component of the magnetic field, differentiated wrt y.

    """
    data = compute_toroidal_flux(Psi, iota, dr=dr + 1, data=data)
    data = compute_iota(i_l, iota, dr=dr, data=data)
    data = compute_lambda(L_lmn, L_transform, dr=dr, dt=dt + 1, dz=dz + 1, data=data)
    data = compute_jacobian(
        R_lmn, Z_lmn, R_transform, Z_transform, dr=dr, dt=dt, dz=dz, data=data,
    )

    data["B0"] = data["psi_r"] / data["sqrt(g)"]
    data["B^rho"] = jnp.zeros_like(data["B0"])
    data["B^theta"] = data["B0"] * (data["iota"] - data["lambda_z"])
    data["B^zeta"] = data["B0"] * (1 + data["lambda_t"])
    data["B"] = data["B^theta"] * data["e_theta"] + data["B^zeta"] * data["e_zeta"]

    # 1st order derivatives
    if dr > 0:
        data["B0_r"] = (
            data["psi_rr"] / data["sqrt(g)"]
            - data["psi_r"] * data["sqrt(g)_r"] / data["sqrt(g)"] ** 2
        )
        data["B^theta_r"] = data["B0_r"] * (data["iota"] - data["lambda_z"]) + data[
            "B0"
        ] * (data["iota_r"] - data["lambda_rz"])
        data["B^zeta_r"] = (
            data["B0_r"] * (1 + data["lambda_t"]) + data["B0"] * data["lambda_rt"]
        )
        data["B_r"] = (
            data["B^theta_r"] * data["e_theta"]
            + data["B^theta"] * data["e_theta_r"]
            + data["B^zeta_r"] * data["e_zeta"]
            + data["B^zeta"] * data["e_zeta_r"]
        )
    if dt > 0:
        data["B0_t"] = -data["psi_r"] * data["sqrt(g)_t"] / data["sqrt(g)"] ** 2
        data["B^theta_t"] = (
            data["B0_t"] * (data["iota"] - data["lambda_z"])
            - data["B0"] * data["lambda_tz"]
        )
        data["B^zeta_t"] = (
            data["B0_t"] * (1 + data["lambda_t"]) + data["B0"] * data["lambda_tt"]
        )
        data["B_t"] = (
            data["B^theta_t"] * data["e_theta"]
            + data["B^theta"] * data["e_theta_t"]
            + data["B^zeta_t"] * data["e_zeta"]
            + data["B^zeta"] * data["e_zeta_t"]
        )
    if dz > 0:
        data["B0_z"] = -data["psi_r"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 2
        data["B^theta_z"] = (
            data["B0_z"] * (data["iota"] - data["lambda_z"])
            - data["B0"] * data["lambda_zz"]
        )
        data["B^zeta_z"] = (
            data["B0_z"] * (1 + data["lambda_t"]) + data["B0"] * data["lambda_tz"]
        )
        data["B_z"] = (
            data["B^theta_z"] * data["e_theta"]
            + data["B^theta"] * data["e_theta_z"]
            + data["B^zeta_z"] * data["e_zeta"]
            + data["B^zeta"] * data["e_zeta_z"]
        )

    # 2nd order derivatives
    if dr > 1:
        raise NotImplementedError
    if dt > 1:
        data["B0_tt"] = -(
            data["psi_r"]
            / data["sqrt(g)"] ** 2
            * (data["sqrt(g)_tt"] - 2 * data["sqrt(g)_t"] ** 2 / data["sqrt(g)"])
        )
        data["B^theta_tt"] = data["B0_tt"] * (data["iota"] - data["lambda_z"])
        -2 * data["B0_t"] * data["lambda_tz"] - data["B0"] * data["lambda_ttz"]
        data["B^zeta_tt"] = data["B0_tt"] * (1 + data["lambda_t"])
        +2 * data["B0_t"] * data["lambda_tt"] + data["B0"] * data["lambda_ttt"]
    if dz > 1:
        data["B0_zz"] = -(
            data["psi_r"]
            / data["sqrt(g)"] ** 2
            * (data["sqrt(g)_zz"] - 2 * data["sqrt(g)_z"] ** 2 / data["sqrt(g)"])
        )
        data["B^theta_zz"] = data["B0_zz"] * (data["iota"] - data["lambda_z"])
        -2 * data["B0_z"] * data["lambda_zz"] - data["B0"] * data["lambda_zzz"]
        data["B^zeta_zz"] = data["B0_zz"] * (1 + data["lambda_t"])
        +2 * data["B0_z"] * data["lambda_tz"] + data["B0"] * data["lambda_tzz"]
    if dr > 0 and dt > 0 and (dr > 1 or dt > 1):
        raise NotImplementedError
    if dr > 0 and dz > 0 and (dr > 1 or dz > 1):
        raise NotImplementedError
    if dt > 0 and dz > 0 and (dt > 1 or dz > 1):
        data["B0_tz"] = -(
            data["psi_r"]
            / data["sqrt(g)"] ** 2
            * (
                data["sqrt(g)_tz"]
                - 2 * data["sqrt(g)_t"] * data["sqrt(g)_z"] / data["sqrt(g)"]
            )
        )
        data["B^theta_tz"] = data["B0_tz"] * (data["iota"] - data["lambda_z"])
        -data["B0_t"] * data["lambda_zz"] - data["B0_z"] * data["lambda_tz"]
        -data["B0"] * data["lambda_tzz"]
        data["B^zeta_tz"] = data["B0_tz"] * (1 + data["lambda_t"])
        +data["B0_t"] * data["lambda_tz"] + data["B0_z"] * data["lambda_tt"] + data[
            "B0"
        ] * data["lambda_ttz"]

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
    dr=0,
    dt=0,
    dz=0,
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
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.
    dt : int, optional
        Order of derivative wrt the poloidal coordinate, theta.
    dz : int, optional
        Order of derivative wrt the toroidal coordinate, zeta.

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
        dr=dr,
        dt=dt,
        dz=dz,
        data=data,
    )

    data["B_rho"] = dot(data["B"], data["e_rho"], 0)
    data["B_theta"] = dot(data["B"], data["e_theta"], 0)
    data["B_zeta"] = dot(data["B"], data["e_zeta"], 0)

    # 1st order derivatives
    if dr > 0:
        data["B_theta_r"] = dot(data["B_r"], data["e_theta"], 0) + dot(
            data["B"], data["e_theta_r"], 0
        )
        data["B_zeta_r"] = dot(data["B_r"], data["e_zeta"], 0) + dot(
            data["B"], data["e_zeta_r"], 0
        )
    if dt > 0:
        data["B_rho_t"] = dot(data["B_t"], data["e_rho"], 0) + dot(
            data["B"], data["e_rho_t"], 0
        )
        data["B_zeta_t"] = dot(data["B_t"], data["e_zeta"], 0) + dot(
            data["B"], data["e_zeta_t"], 0
        )
    if dz > 0:
        data["B_rho_z"] = dot(data["B_z"], data["e_rho"], 0) + dot(
            data["B"], data["e_rho_z"], 0
        )
        data["B_theta_z"] = dot(data["B_z"], data["e_theta"], 0) + dot(
            data["B"], data["e_theta_z"], 0
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
    dr=0,
    dt=0,
    dz=0,
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
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.
    dt : int, optional
        Order of derivative wrt the poloidal coordinate, theta.
    dz : int, optional
        Order of derivative wrt the toroidal coordinate, zeta.

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
        dr=dr,
        dt=dt,
        dz=dz,
        data=data,
    )
    data = compute_covariant_metric_coefficients(
        R_lmn, Z_lmn, R_transform, Z_transform, data=data
    )

    # TODO: would it be simpler to compute this as B^theta*B_theta+B^zeta*B_zeta?
    data["|B|"] = jnp.sqrt(
        data["B^theta"] ** 2 * data["g_tt"]
        + data["B^zeta"] ** 2 * data["g_zz"]
        + 2 * data["B^theta"] * data["B^zeta"] * data["g_tz"]
    )

    # 1st order derivatives
    if dr > 0:
        raise NotImplementedError
    if dt > 0:
        data["|B|_t"] = (
            data["B^theta"]
            * (
                data["B^zeta_t"] * data["g_tz"]
                + data["B^theta_t"] * data["g_tt"]
                + data["B^theta"] * dot(data["e_theta_t"], data["e_theta"], 0)
            )
            + data["B^zeta"]
            * (
                data["B^theta_t"] * data["g_tz"]
                + data["B^zeta_t"] * data["g_zz"]
                + data["B^zeta"] * dot(data["e_zeta_t"], data["e_zeta"], 0)
            )
            + data["B^theta"]
            * data["B^zeta"]
            * (
                dot(data["e_theta_t"], data["e_zeta"], 0)
                + dot(data["e_zeta_t"], data["e_theta"], 0)
            )
        ) / data["|B|"]
    if dz > 0:
        data["|B|_z"] = (
            data["B^theta"]
            * (
                data["B^zeta_z"] * data["g_tz"]
                + data["B^theta_z"] * data["g_tt"]
                + data["B^theta"] * dot(data["e_theta_z"], data["e_theta"], 0)
            )
            + data["B^zeta"]
            * (
                data["B^theta_z"] * data["g_tz"]
                + data["B^zeta_z"] * data["g_zz"]
                + data["B^zeta"] * dot(data["e_zeta_z"], data["e_zeta"], 0)
            )
            + data["B^theta"]
            * data["B^zeta"]
            * (
                dot(data["e_theta_z"], data["e_zeta"], 0)
                + dot(data["e_zeta_z"], data["e_theta"], 0)
            )
        ) / data["|B|"]

    # 2nd order derivatives
    if dr > 1:
        raise NotImplementedError
    if dt > 1:
        data["|B|_tt"] = (
            data["B^theta_t"]
            * (
                data["B^zeta_t"] * data["g_tz"]
                + data["B^theta_t"] * data["g_tt"]
                + data["B^theta"] * dot(data["e_theta_t"], data["e_theta"], 0)
            )
            + data["B^theta"]
            * (
                data["B^zeta_tt"] * data["g_tz"]
                + data["B^theta_tt"] * data["g_tt"]
                + data["B^theta_t"] * dot(data["e_theta_t"], data["e_theta"], 0)
            )
            + data["B^theta"]
            * (
                data["B^zeta_t"]
                * (
                    dot(data["e_theta_t"], data["e_zeta"], 0)
                    + dot(data["e_theta"], data["e_zeta_t"], 0)
                )
                + 2 * data["B^theta_t"] * dot(data["e_theta_t"], data["e_theta"], 0)
                + data["B^theta"]
                * (
                    dot(data["e_theta_tt"], data["e_theta"], 0)
                    + dot(data["e_theta_t"], data["e_theta_t"], 0)
                )
            )
            + data["B^zeta_t"]
            * (
                data["B^theta_t"] * data["g_tz"]
                + data["B^zeta_t"] * data["g_zz"]
                + data["B^zeta"] * dot(data["e_zeta_t"], data["e_zeta"], 0)
            )
            + data["B^zeta"]
            * (
                data["B^theta_tt"] * data["g_tz"]
                + data["B^zeta_tt"] * data["g_zz"]
                + data["B^zeta_t"] * dot(data["e_zeta_t"], data["e_zeta"], 0)
            )
            + data["B^zeta"]
            * (
                data["B^theta_t"]
                * (
                    dot(data["e_theta_t"], data["e_zeta"], 0)
                    + dot(data["e_theta"], data["e_zeta_t"], 0)
                )
                + 2 * data["B^zeta_t"] * dot(data["e_zeta_t"], data["e_zeta"], 0)
                + data["B^zeta"]
                * (
                    dot(data["e_zeta_tt"], data["e_zeta"], 0)
                    + dot(data["e_zeta_t"], data["e_zeta_t"], 0)
                )
            )
            + (data["B^theta_t"] * data["B^zeta"] + data["B^theta"] * data["B^zeta_t"])
            * (
                dot(data["e_theta_t"], data["e_zeta"], 0)
                + dot(data["e_zeta_t"], data["e_theta"], 0)
            )
            + data["B^theta"]
            * data["B^zeta"]
            * (
                dot(data["e_theta_tt"], data["e_zeta"], 0)
                + dot(data["e_zeta_tt"], data["e_theta"], 0)
                + 2 * dot(data["e_zeta_t"], data["e_theta_t"], 0)
            )
        ) / data["|B|"] - data["|B|_t"] ** 2 / data["|B|"]
    if dz > 1:
        data["|B|_zz"] = (
            data["B^theta_z"]
            * (
                data["B^zeta_z"] * data["g_tz"]
                + data["B^theta_z"] * data["g_tt"]
                + data["B^theta"] * dot(data["e_theta_z"], data["e_theta"], 0)
            )
            + data["B^theta"]
            * (
                data["B^zeta_zz"] * data["g_tz"]
                + data["B^theta_zz"] * data["g_tt"]
                + data["B^theta_z"] * dot(data["e_theta_z"], data["e_theta"], 0)
            )
            + data["B^theta"]
            * (
                data["B^zeta_z"]
                * (
                    dot(data["e_theta_z"], data["e_zeta"], 0)
                    + dot(data["e_theta"], data["e_zeta_z"], 0)
                )
                + 2 * data["B^theta_z"] * dot(data["e_theta_z"], data["e_theta"], 0)
                + data["B^theta"]
                * (
                    dot(data["e_theta_zz"], data["e_theta"], 0)
                    + dot(data["e_theta_z"], data["e_theta_z"], 0)
                )
            )
            + data["B^zeta_z"]
            * (
                data["B^theta_z"] * data["g_tz"]
                + data["B^zeta_z"] * data["g_zz"]
                + data["B^zeta"] * dot(data["e_zeta_z"], data["e_zeta"], 0)
            )
            + data["B^zeta"]
            * (
                data["B^theta_zz"] * data["g_tz"]
                + data["B^zeta_zz"] * data["g_zz"]
                + data["B^zeta_z"] * dot(data["e_zeta_z"], data["e_zeta"], 0)
            )
            + data["B^zeta"]
            * (
                data["B^theta_z"]
                * (
                    dot(data["e_theta_z"], data["e_zeta"], 0)
                    + dot(data["e_theta"], data["e_zeta_z"], 0)
                )
                + 2 * data["B^zeta_z"] * dot(data["e_zeta_z"], data["e_zeta"], 0)
                + data["B^zeta"]
                * (
                    dot(data["e_zeta_zz"], data["e_zeta"], 0)
                    + dot(data["e_zeta_z"], data["e_zeta_z"], 0)
                )
            )
            + (data["B^theta_z"] * data["B^zeta"] + data["B^theta"] * data["B^zeta_z"])
            * (
                dot(data["e_theta_z"], data["e_zeta"], 0)
                + dot(data["e_zeta_z"], data["e_theta"], 0)
            )
            + data["B^theta"]
            * data["B^zeta"]
            * (
                dot(data["e_theta_zz"], data["e_zeta"], 0)
                + dot(data["e_zeta_zz"], data["e_theta"], 0)
                + 2 * dot(data["e_theta_z"], data["e_zeta_z"], 0)
            )
        ) / data["|B|"] - data["|B|_z"] ** 2 / data["|B|"]
    if dr > 0 and dt > 0 and (dr > 1 or dt > 1):
        raise NotImplementedError
    if dr > 0 and dz > 0 and (dr > 1 or dz > 1):
        raise NotImplementedError
    if dt > 0 and dz > 0 and (dt > 1 or dz > 1):
        data["|B|_tz"] = (
            data["B^theta_z"]
            * (
                data["B^zeta_t"] * data["g_tz"]
                + data["B^theta_t"] * data["g_tt"]
                + data["B^theta"] * dot(data["e_theta_t"], data["e_theta"], 0)
            )
            + data["B^theta"]
            * (
                data["B^zeta_tz"] * data["g_tz"]
                + data["B^theta_tz"] * data["g_tt"]
                + data["B^theta_z"] * dot(data["e_theta_t"], data["e_theta"], 0)
            )
            + data["B^theta"]
            * (
                data["B^zeta_t"]
                * (
                    dot(data["e_theta_z"], data["e_zeta"], 0)
                    + dot(data["e_theta"], data["e_zeta_z"], 0)
                )
                + 2 * data["B^theta_t"] * dot(data["e_theta_z"], data["e_theta"], 0)
                + data["B^theta"]
                * (
                    dot(data["e_theta_tz"], data["e_theta"], 0)
                    + dot(data["e_theta_t"], data["e_theta_z"], 0)
                )
            )
            + data["B^zeta_z"]
            * (
                data["B^theta_t"] * data["g_tz"]
                + data["B^zeta_t"] * data["g_zz"]
                + data["B^zeta"] * dot(data["e_zeta_t"], data["e_zeta"], 0)
            )
            + data["B^zeta"]
            * (
                data["B^theta_tz"] * data["g_tz"]
                + data["B^zeta_tz"] * data["g_zz"]
                + data["B^zeta_z"] * dot(data["e_zeta_t"], data["e_zeta"], 0)
            )
            + data["B^zeta"]
            * (
                data["B^theta_t"]
                * (
                    dot(data["e_theta_z"], data["e_zeta"], 0)
                    + dot(data["e_theta"], data["e_zeta_z"], 0)
                )
                + 2 * data["B^zeta_t"] * dot(data["e_zeta_z"], data["e_zeta"], 0)
                + data["B^zeta"]
                * (
                    dot(data["e_zeta_tz"], data["e_zeta"], 0)
                    + dot(data["e_zeta_t"], data["e_zeta_z"], 0)
                )
            )
            + (data["B^theta_z"] * data["B^zeta"] + data["B^theta"] * data["B^zeta_z"])
            * (
                dot(data["e_theta_t"], data["e_zeta"], 0)
                + dot(data["e_zeta_t"], data["e_theta"], 0)
            )
            + data["B^theta"]
            * data["B^zeta"]
            * (
                dot(data["e_theta_tz"], data["e_zeta"], 0)
                + dot(data["e_zeta_tz"], data["e_theta"], 0)
                + dot(data["e_theta_t"], data["e_zeta_z"], 0)
                + dot(data["e_zeta_t"], data["e_theta_z"], 0)
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
        dr=1,
        dt=1,
        dz=1,
        data=data,
    )
    data = compute_contravariant_metric_coefficients(
        R_lmn, Z_lmn, R_transform, Z_transform, data=data
    )

    # covariant components
    data["grad(|B|^2)_rho"] = (
        data["B^theta"] * data["B_theta_r"]
        + data["B_theta"] * data["B^theta_r"]
        + data["B^zeta"] * data["B_zeta_r"]
        + data["B_zeta"] * data["B^zeta_r"]
    )
    data["grad(|B|^2)_theta"] = (
        data["B^theta"] * data["B_theta_t"]
        + data["B_theta"] * data["B^theta_t"]
        + data["B^zeta"] * data["B_zeta_t"]
        + data["B_zeta"] * data["B^zeta_t"]
    )
    data["grad(|B|^2)_zeta"] = (
        data["B^theta"] * data["B_theta_z"]
        + data["B_theta"] * data["B^theta_z"]
        + data["B^zeta"] * data["B_zeta_z"]
        + data["B_zeta"] * data["B^zeta_z"]
    )

    # gradient vector
    data["grad(|B|^2)"] = (
        data["grad(|B|^2)_rho"] * data["e^rho"]
        + data["grad(|B|^2)_theta"] * data["e^theta"]
        + data["grad(|B|^2)_zeta"] * data["e^zeta"]
    )

    # magnitude
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
        dr=0,
        dt=1,
        dz=1,
        data=data,
    )
    data = compute_contravariant_basis(
        R_lmn, Z_lmn, R_transform, Z_transform, dr=0, dt=0, dz=0, data=data
    )
    data = compute_covariant_metric_coefficients(
        R_lmn, Z_lmn, R_transform, Z_transform, data=data
    )

    # tension vector
    data["(B*grad(|B|))B"] = (
        (data["B^theta"] * data["B^theta_t"] + data["B^zeta"] * data["B^theta_z"])
        * data["e_theta"]
        + (data["B^theta"] * data["B^zeta_t"] + data["B^zeta"] * data["B^zeta_z"])
        * data["e_zeta"]
        + data["B^theta"] ** 2 * data["e_theta_t"]
        + data["B^zeta"] ** 2 * data["e_zeta_z"]
        + data["B^theta"] * data["B^zeta"] * (data["e_theta_z"] + data["e_zeta_t"])
    )

    # contravariant components
    data["((B*grad(|B|))B)^rho"] = dot(data["(B*grad(|B|))B"], data["e^rho"], 0)
    data["((B*grad(|B|))B)^theta"] = dot(data["(B*grad(|B|))B"], data["e^theta"], 0)
    data["((B*grad(|B|))B)^zeta"] = dot(data["(B*grad(|B|))B"], data["e^zeta"], 0)

    # magnitude
    data["|(B*grad(|B|))B|"] = jnp.sqrt(
        data["((B*grad(|B|))B)^rho"] ** 2 * data["g_rr"]
        + data["((B*grad(|B|))B)^theta"] ** 2 * data["g_tt"]
        + data["((B*grad(|B|))B)^zeta"] ** 2 * data["g_zz"]
        + 2
        * data["((B*grad(|B|))B)^rho"]
        * data["((B*grad(|B|))B)^theta"]
        * data["g_rt"]
        + 2
        * data["((B*grad(|B|))B)^rho"]
        * data["((B*grad(|B|))B)^zeta"]
        * data["g_rz"]
        + 2
        * data["((B*grad(|B|))B)^theta"]
        * data["((B*grad(|B|))B)^zeta"]
        * data["g_tz"]
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
    dr=0,
    dt=0,
    dz=0,
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
        dr=0,
        dt=dt + 1,
        dz=dz + 1,
        data=data,
    )

    data["B*grad(|B|)"] = (
        data["B^theta"] * data["|B|_t"] + data["B^zeta"] * data["|B|_z"]
    )

    # 1st order derivatives
    if dr > 0:
        raise NotImplementedError
    if dt > 0:
        data["B*grad(|B|)_t"] = (
            data["B^theta_t"] * data["|B|_t"]
            + data["B^zeta_t"] * data["|B|_z"]
            + data["B^theta"] * data["|B|_tt"]
            + data["B^zeta"] * data["|B|_tz"]
        )
    if dz > 0:
        data["B*grad(|B|)_z"] = (
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
    dr=0,
    dt=0,
    dz=0,
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
    dr : int, optional
        Order of derivative wrt the radial coordinate, rho.
    dt : int, optional
        Order of derivative wrt the poloidal coordinate, theta.
    dz : int, optional
        Order of derivative wrt the toroidal coordinate, zeta.

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
        dr=dr + 1,
        dt=dt + 1,
        dz=dz + 1,
        data=data,
    )

    data["J^rho"] = (data["B_zeta_t"] - data["B_theta_z"]) / (mu_0 * data["sqrt(g)"])
    data["J^theta"] = (data["B_rho_z"] - data["B_zeta_r"]) / (mu_0 * data["sqrt(g)"])
    data["J^zeta"] = (data["B_theta_r"] - data["B_rho_t"]) / (mu_0 * data["sqrt(g)"])
    data["J"] = (
        data["J^rho"] * data["e_rho"]
        + data["J^theta"] * data["e_theta"]
        + data["J^zeta"] * data["e_zeta"]
    )

    return data


def compute_force_error(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    p_l,
    Psi,
    R_transform,
    Z_transform,
    L_transform,
    iota,
    pressure,
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
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    p_l : ndarray
        Spectral coefficients of p(rho) -- pressure profile.
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

    Returns
    -------
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of force error components.
        Keys are of the form 'F_x', meaning the x covariant (F_x) component of the
        force error.

    """
    data = compute_pressure(p_l, pressure, dr=1, data=data)
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
        dr=0,
        dt=0,
        dz=0,
        data=data,
    )
    data = compute_contravariant_metric_coefficients(
        R_lmn, Z_lmn, R_transform, Z_transform, data=data
    )

    data["F_rho"] = -data["p_r"] + data["sqrt(g)"] * (
        data["B^zeta"] * data["J^theta"] - data["B^theta"] * data["J^zeta"]
    )
    data["F_theta"] = data["sqrt(g)"] * data["B^zeta"] * data["J^rho"]
    data["F_zeta"] = -data["sqrt(g)"] * data["B^theta"] * data["J^rho"]
    data["F_beta"] = data["sqrt(g)"] * data["J^rho"]
    data["F"] = (
        data["F_rho"] * data["e^rho"]
        + data["F_theta"] * data["e^theta"]
        + data["F_zeta"] * data["e^zeta"]
    )

    data["|F|"] = jnp.sqrt(
        data["F_rho"] ** 2 * data["g^rr"]
        + data["F_theta"] ** 2 * data["g^tt"]
        + data["F_zeta"] ** 2 * data["g^zz"]
        + 2 * data["F_rho"] * data["F_theta"] * data["g^rt"]
        + 2 * data["F_rho"] * data["F_zeta"] * data["g^rz"]
        + 2 * data["F_theta"] * data["F_zeta"] * data["g^tz"]
    )
    data["|grad(rho)|"] = jnp.sqrt(data["g^rr"])
    data["|beta|"] = jnp.sqrt(
        data["B^zeta"] ** 2 * data["g^tt"]
        + data["B^theta"] ** 2 * data["g^zz"]
        - 2 * data["B^theta"] * data["B^zeta"] * data["g^tz"]
    )
    data["|grad(p)|"] = jnp.sqrt(data["p_r"] ** 2) * data["|grad(rho)|"]

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
        dr=0,
        dt=1,
        dz=1,
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
    data["I"] = jnp.mean(data["B_theta"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
    data["G"] = jnp.mean(data["B_zeta"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])

    # QS flux function (T^3)
    data["QS_FF"] = (data["psi_r"] / data["sqrt(g)"]) * (
        data["B_zeta"] * data["|B|_t"] - data["B_theta"] * data["|B|_z"]
    ) - data["B*grad(|B|)"] * (M * data["G"] + N * data["I"]) / (M * data["iota"] - N)
    # QS triple product (T^4/m^2)
    data["QS_TP"] = (data["psi_r"] / data["sqrt(g)"]) * (
        data["|B|_t"] * data["B*grad(|B|)_z"] - data["|B|_z"] * data["B*grad(|B|)_t"]
    )

    return data


def compute_volume(
    R_lmn, Z_lmn, R_transform, Z_transform, data=None,
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
    data["V"] = jnp.sum(jnp.abs(data["sqrt(g)"]) * R_transform.grid.weights)
    return data


def compute_energy(
    R_lmn,
    Z_lmn,
    L_lmn,
    i_l,
    p_l,
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
    i_l : ndarray
        Spectral coefficients of iota(rho) -- rotational transform profile.
    p_l : ndarray
        Spectral coefficients of p(rho) -- pressure profile.
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

    data["W_B"] = jnp.sum(
        data["|B|"] ** 2 * jnp.abs(data["sqrt(g)"]) * R_transform.grid.weights
    ) / (2 * mu_0)
    data["W_p"] = jnp.sum(
        data["p"] * jnp.abs(data["sqrt(g)"]) * R_transform.grid.weights
    ) / (gamma - 1)
    data["W"] = data["W_B"] + data["W_p"]

    return data
