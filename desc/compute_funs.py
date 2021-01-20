import numpy as np
import warnings
from termcolor import colored
from desc.backend import jnp, put
from desc.transform import Transform
from scipy import special


"""These functions perform the core calculations of physical quantities.
They are used as methods of the Configuration class, and also used to compute
quantities in the objective functions.
All of the functions in this file have the same call signature:

Parameters
----------
Psi : float
    total toroidal flux (in Webers) within the last closed flux surface
R_lmn : ndarray
    spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
Z_lmn : ndarray
    spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
L_lmn : ndarray
    spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
p_l : ndarray
    spectral coefficients of p(rho) -- pressure profile
i_l : ndarray
    spectral coefficients of iota(rho) -- rotational transform profile
R_transform : Transform
    transforms R_lmn coefficients to real space
Z_transform : Transform
    transforms Z_lmn coefficients to real space
L_transform : Transform
    transforms L_lmn coefficients to real space
p_transform : Transform
    transforms p_l coefficients to real space
i_transform : Transform
    transforms i_l coefficients to real space
zeta_ratio : float
    scale factor for zeta derivatives. Setting to zero effectively solves
    for individual tokamak solutions at each toroidal plane,
    setting to 1 solves for a stellarator. (Default value = 1.0)

"""


def dot(a, b, axis):
    """Batched vector dot product

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
    """Batched vector cross product

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


def compute_profiles(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes magnetic flux, pressure, and rotational transform profiles.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    profiles = {}

    # toroidal flux
    rho = p_transform.grid.nodes[:, 0]
    profiles["psi"] = Psi * rho ** 2
    profiles["psi_r"] = 2 * Psi * rho
    profiles["psi_rr"] = 2 * Psi * np.ones_like(rho)

    # pressure
    profiles["p"] = p_transform.transform(p_l, 0)
    profiles["p_r"] = p_transform.transform(p_l, 1)

    # rotational transform
    profiles["iota"] = i_transform.transform(i_l, 0)
    profiles["iota_r"] = i_transform.transform(i_l, 1)

    return profiles


def compute_toroidal_coords(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Transforms toroidal coordinates to real space.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    coordinates : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    toroidal_coords = {}
    toroidal_coords["R"] = R_transform.transform(R_lmn)
    toroidal_coords["Z"] = Z_transform.transform(Z_lmn)
    toroidal_coords["lambda"] = L_transform.transform(L_lmn)
    toroidal_coords["0"] = jnp.zeros_like(toroidal_coords["R"])

    return toroidal_coords


def compute_cartesian_coords(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes cartesian coordinates from toroidal coordinates.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    cartesian_coords : dict
        dictionary of ndarray, shape(num_nodes,) of cartesian coordinates
        evaluated at grid nodes.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    toroidal_coords = compute_toroidal_coords(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    cartesian_coords = {}
    phi = R_transform.grid.nodes[:, 2]
    cartesian_coords["X"] = toroidal_coords["R"] * np.cos(phi)
    cartesian_coords["Y"] = toroidal_coords["R"] * np.sin(phi)
    cartesian_coords["Z"] = toroidal_coords["Z"]

    return cartesian_coords, toroidal_coords


def compute_covariant_basis(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes covariant basis vectors.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in
        the x direction, differentiated wrt to y.
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    toroidal_coords = compute_toroidal_coords(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    # toroidal coordinate 1st derivatives
    toroidal_coords["R_r"] = R_transform.transform(R_lmn, 1, 0, 0)
    toroidal_coords["Z_r"] = Z_transform.transform(Z_lmn, 1, 0, 0)
    toroidal_coords["R_t"] = R_transform.transform(R_lmn, 0, 1, 0)
    toroidal_coords["Z_t"] = Z_transform.transform(Z_lmn, 0, 1, 0)
    toroidal_coords["R_z"] = R_transform.transform(R_lmn, 0, 0, 1) * zeta_ratio
    toroidal_coords["Z_z"] = Z_transform.transform(Z_lmn, 0, 0, 1) * zeta_ratio

    cov_basis = {}
    cov_basis["e_rho"] = jnp.array(
        [toroidal_coords["R_r"], toroidal_coords["0"], toroidal_coords["Z_r"]]
    )
    cov_basis["e_theta"] = jnp.array(
        [toroidal_coords["R_t"], toroidal_coords["0"], toroidal_coords["Z_t"]]
    )
    cov_basis["e_zeta"] = jnp.array(
        [toroidal_coords["R_z"], toroidal_coords["R"], toroidal_coords["Z_z"]]
    )

    return cov_basis, toroidal_coords


def compute_jacobian(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes coordinate system jacobian.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate system jacobian.
        Keys are of the form 'g_x' meaning the x derivative of the coordinate
        system jacobian g.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in
        the x direction, differentiated wrt to y.
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    cov_basis, toroidal_coords = compute_covariant_basis(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    jacobian = {}
    jacobian["g"] = dot(
        cov_basis["e_rho"], cross(cov_basis["e_theta"], cov_basis["e_zeta"], 0), 0
    )

    return jacobian, cov_basis, toroidal_coords


def compute_contravariant_basis(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes contravariant basis vectors.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    con_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of contravariant basis vectors.
        Keys are of the form 'e^x_y', meaning the contravariant basis vector
        in the x direction, differentiated wrt to y.
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate system jacobian.
        Keys are of the form 'g_x' meaning the x derivative of the coordinate
        system jacobian g.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in
        the x direction, differentiated wrt to y.
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    jacobian, cov_basis, toroidal_coords = compute_jacobian(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    con_basis = {}
    con_basis["e^rho"] = (
        cross(cov_basis["e_theta"], cov_basis["e_zeta"], 0) / jacobian["g"]
    )
    con_basis["e^theta"] = (
        cross(cov_basis["e_zeta"], cov_basis["e_rho"], 0) / jacobian["g"]
    )
    con_basis["e^zeta"] = jnp.array(
        [toroidal_coords["0"], 1 / toroidal_coords["R"], toroidal_coords["0"]]
    )

    return con_basis, jacobian, cov_basis, toroidal_coords


def compute_magnetic_field(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes magnetic field components.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    magnetic_field: dict
        dictionary of ndarray, shape(num_nodes,) of magnetic field components.
        Keys are of the form 'B_x_y' or 'B^x_y', meaning the covariant (B_x)
        or contravariant (B^x) component of the magnetic field, with the
        derivative wrt to y.
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate system jacobian.
        Keys are of the form 'g_x' meaning the x derivative of the coordinate
        system jacobian g.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in
        the x direction, differentiated wrt to y.
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    profiles = compute_profiles(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )
    jacobian, cov_basis, toroidal_coords = compute_jacobian(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    # lambda derivatives
    toroidal_coords["lambda_t"] = L_transform.transform(L_lmn, 0, 1, 0)
    toroidal_coords["lambda_z"] = L_transform.transform(L_lmn, 0, 0, 1) * zeta_ratio

    magnetic_field = {}
    magnetic_field["B0"] = profiles["psi_r"] / (2 * jnp.pi * jacobian["g"])

    # contravariant components
    magnetic_field["B^rho"] = jnp.zeros_like(magnetic_field["B0"])
    magnetic_field["B^theta"] = magnetic_field["B0"] * (
        profiles["iota"] - toroidal_coords["lambda_z"]
    )
    magnetic_field["B^zeta"] = magnetic_field["B0"] * (1 + toroidal_coords["lambda_t"])
    magnetic_field["B_con"] = (
        magnetic_field["B^theta"] * cov_basis["e_theta"]
        + magnetic_field["B^zeta"] * cov_basis["e_zeta"]
    )

    # covariant components
    magnetic_field["B_rho"] = dot(magnetic_field["B_con"], cov_basis["e_rho"], 0)
    magnetic_field["B_theta"] = dot(magnetic_field["B_con"], cov_basis["e_theta"], 0)
    magnetic_field["B_zeta"] = dot(magnetic_field["B_con"], cov_basis["e_zeta"], 0)

    return magnetic_field, jacobian, cov_basis, toroidal_coords, profiles


def compute_magnetic_field_axis(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes magnetic field components; can handle nodes at the magnetic axis.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    magnetic_field: dict
        dictionary of ndarray, shape(num_nodes,) of magnetic field components.
        Keys are of the form 'B_x_y' or 'B^x_y', meaning the covariant (B_x)
        or contravariant (B^x) component of the magnetic field, with the
        derivative wrt to y.
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate system jacobian.
        Keys are of the form 'g_x' meaning the x derivative of the coordinate
        system jacobian g.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in
        the x direction, differentiated wrt to y.
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    profiles = compute_profiles(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )
    jacobian, cov_basis, toroidal_coords = compute_jacobian(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    axis = i_transform.grid.axis

    # lambda derivatives
    toroidal_coords["lambda_t"] = L_transform.transform(L_lmn, 0, 1, 0)
    toroidal_coords["lambda_z"] = L_transform.transform(L_lmn, 0, 0, 1) * zeta_ratio

    # toroidal coordinate 2nd derivatives
    toroidal_coords["R_rr"] = R_transform.transform(R_lmn, 2, 0, 0)
    toroidal_coords["Z_rr"] = Z_transform.transform(Z_lmn, 2, 0, 0)
    toroidal_coords["R_rt"] = R_transform.transform(R_lmn, 1, 1, 0)
    toroidal_coords["Z_rt"] = Z_transform.transform(Z_lmn, 1, 1, 0)
    toroidal_coords["R_rz"] = R_transform.transform(R_lmn, 1, 0, 1)
    toroidal_coords["Z_rz"] = Z_transform.transform(Z_lmn, 1, 0, 1)

    # covariant basis derivatives
    cov_basis["e_rho_r"] = jnp.array(
        [toroidal_coords["R_rr"], toroidal_coords["0"], toroidal_coords["Z_rr"]]
    )
    cov_basis["e_theta_r"] = jnp.array(
        [toroidal_coords["R_rt"], toroidal_coords["0"], toroidal_coords["Z_rt"]]
    )
    cov_basis["e_zeta_r"] = jnp.array(
        [toroidal_coords["R_rz"], toroidal_coords["R_r"], toroidal_coords["Z_rz"]]
    )

    # jacobian derivatives
    jacobian["g_r"] = (
        dot(
            cov_basis["e_rho_r"], cross(cov_basis["e_theta"], cov_basis["e_zeta"], 0), 0
        )
        + dot(
            cov_basis["e_rho"], cross(cov_basis["e_theta_r"], cov_basis["e_zeta"], 0), 0
        )
        + dot(
            cov_basis["e_rho"], cross(cov_basis["e_theta"], cov_basis["e_zeta_r"], 0), 0
        )
    )

    magnetic_field = {}
    magnetic_field["B0"] = profiles["psi_r"] / (2 * jnp.pi * jacobian["g"])
    magnetic_field["B0"] = put(
        magnetic_field["B0"],
        axis,
        profiles["psi_rr"][axis] / (2 * jnp.pi * jacobian["g_r"][axis]),
    )

    # contravariant components
    magnetic_field["B^rho"] = jnp.zeros_like(magnetic_field["B0"])
    magnetic_field["B^theta"] = magnetic_field["B0"] * (
        profiles["iota"] - toroidal_coords["lambda_z"]
    )
    magnetic_field["B^zeta"] = magnetic_field["B0"] * (1 + toroidal_coords["lambda_t"])
    magnetic_field["B_con"] = (
        magnetic_field["B^theta"] * cov_basis["e_theta"]
        + magnetic_field["B^zeta"] * cov_basis["e_zeta"]
    )

    # covariant components
    magnetic_field["B_rho"] = dot(magnetic_field["B_con"], cov_basis["e_rho"], 0)
    magnetic_field["B_theta"] = dot(magnetic_field["B_con"], cov_basis["e_theta"], 0)
    magnetic_field["B_zeta"] = dot(magnetic_field["B_con"], cov_basis["e_zeta"], 0)

    return magnetic_field, jacobian, cov_basis, toroidal_coords, profiles


def compute_magnetic_field_magnitude(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes magnetic field magnitude.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    magnetic_field: dict
        dictionary of ndarray, shape(num_nodes,) of magnetic field components.
        Keys are of the form 'B_x_y' or 'B^x_y', meaning the covariant (B_x)
        or contravariant (B^x) component of the magnetic field, with the
        derivative wrt to y.
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate system jacobian.
        Keys are of the form 'g_x' meaning the x derivative of the coordinate
        system jacobian g.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in
        the x direction, differentiated wrt to y.
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    (
        magnetic_field,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    ) = compute_magnetic_field(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    magnetic_field["|B|"] = jnp.sqrt(
        magnetic_field["B^theta"] ** 2
        * dot(cov_basis["e_theta"], cov_basis["e_theta"], 0)
        + magnetic_field["B^zeta"] ** 2
        * dot(cov_basis["e_zeta"], cov_basis["e_zeta"], 0)
        + 2
        * magnetic_field["B^theta"]
        * magnetic_field["B^zeta"]
        * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
    )

    return magnetic_field, jacobian, cov_basis, toroidal_coords, profiles


def compute_magnetic_field_magnitude_axis(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes magnetic field magnitude; can handle nodes at the magnetic axis.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    magnetic_field: dict
        dictionary of ndarray, shape(num_nodes,) of magnetic field components.
        Keys are of the form 'B_x_y' or 'B^x_y', meaning the covariant (B_x)
        or contravariant (B^x) component of the magnetic field, with the
        derivative wrt to y.
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate system jacobian.
        Keys are of the form 'g_x' meaning the x derivative of the coordinate
        system jacobian g.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in
        the x direction, differentiated wrt to y.
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    (
        magnetic_field,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    ) = compute_magnetic_field_axis(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    magnetic_field["|B|"] = jnp.sqrt(
        magnetic_field["B^theta"] ** 2
        * dot(cov_basis["e_theta"], cov_basis["e_theta"], 0)
        + magnetic_field["B^zeta"] ** 2
        * dot(cov_basis["e_zeta"], cov_basis["e_zeta"], 0)
        + 2
        * magnetic_field["B^theta"]
        * magnetic_field["B^zeta"]
        * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
    )

    return magnetic_field, jacobian, cov_basis, toroidal_coords, profiles


def compute_current_density(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes current density field components.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    current_density : dict
        dictionary of ndarray, shape(num_nodes,), of current density components.
        Keys are of the form 'J^x_y' meaning the contravariant (J^x)
        component of the current, with the derivative wrt to y.
    magnetic_field: dict
        dictionary of ndarray, shape(num_nodes,) of magnetic field components.
        Keys are of the form 'B_x_y' or 'B^x_y', meaning the covariant (B_x)
        or contravariant (B^x) component of the magnetic field, with the
        derivative wrt to y.
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate system jacobian.
        Keys are of the form 'g_x' meaning the x derivative of the coordinate
        system jacobian g.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in
        the x direction, differentiated wrt to y.
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    (
        magnetic_field,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    ) = compute_magnetic_field(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )
    mu0 = 4 * jnp.pi * 1e-7

    # toroidal coordinate 2nd derivatives
    toroidal_coords["R_rr"] = R_transform.transform(R_lmn, 2, 0, 0)
    toroidal_coords["Z_rr"] = Z_transform.transform(Z_lmn, 2, 0, 0)
    toroidal_coords["R_rt"] = R_transform.transform(R_lmn, 1, 1, 0)
    toroidal_coords["Z_rt"] = Z_transform.transform(Z_lmn, 1, 1, 0)
    toroidal_coords["R_rz"] = R_transform.transform(R_lmn, 1, 0, 1) * zeta_ratio
    toroidal_coords["Z_rz"] = Z_transform.transform(Z_lmn, 1, 0, 1) * zeta_ratio
    toroidal_coords["R_tt"] = R_transform.transform(R_lmn, 0, 2, 0)
    toroidal_coords["Z_tt"] = Z_transform.transform(Z_lmn, 0, 2, 0)
    toroidal_coords["R_tz"] = R_transform.transform(R_lmn, 0, 1, 1) * zeta_ratio
    toroidal_coords["Z_tz"] = Z_transform.transform(Z_lmn, 0, 1, 1) * zeta_ratio
    toroidal_coords["R_zz"] = R_transform.transform(R_lmn, 0, 0, 2) * zeta_ratio
    toroidal_coords["Z_zz"] = Z_transform.transform(Z_lmn, 0, 0, 2) * zeta_ratio

    # lambda derivatives
    toroidal_coords["lambda_rt"] = L_transform.transform(L_lmn, 1, 1, 0)
    toroidal_coords["lambda_rz"] = L_transform.transform(L_lmn, 1, 0, 1) * zeta_ratio
    toroidal_coords["lambda_tt"] = L_transform.transform(L_lmn, 0, 2, 0)
    toroidal_coords["lambda_tz"] = L_transform.transform(L_lmn, 0, 1, 1) * zeta_ratio
    toroidal_coords["lambda_zz"] = L_transform.transform(L_lmn, 0, 0, 2) * zeta_ratio

    # covariant basis derivatives
    cov_basis["e_rho_r"] = jnp.array(
        [toroidal_coords["R_rr"], toroidal_coords["0"], toroidal_coords["Z_rr"]]
    )
    cov_basis["e_rho_t"] = jnp.array(
        [toroidal_coords["R_rt"], toroidal_coords["0"], toroidal_coords["Z_rt"]]
    )
    cov_basis["e_rho_z"] = jnp.array(
        [toroidal_coords["R_rz"], toroidal_coords["0"], toroidal_coords["Z_rz"]]
    )

    cov_basis["e_theta_r"] = jnp.array(
        [toroidal_coords["R_rt"], toroidal_coords["0"], toroidal_coords["Z_rt"]]
    )
    cov_basis["e_theta_t"] = jnp.array(
        [toroidal_coords["R_tt"], toroidal_coords["0"], toroidal_coords["Z_tt"]]
    )
    cov_basis["e_theta_z"] = jnp.array(
        [toroidal_coords["R_tz"], toroidal_coords["0"], toroidal_coords["Z_tz"]]
    )

    cov_basis["e_zeta_r"] = jnp.array(
        [toroidal_coords["R_rz"], toroidal_coords["R_r"], toroidal_coords["Z_rz"]]
    )
    cov_basis["e_zeta_t"] = jnp.array(
        [toroidal_coords["R_tz"], toroidal_coords["R_t"], toroidal_coords["Z_tz"]]
    )
    cov_basis["e_zeta_z"] = jnp.array(
        [toroidal_coords["R_zz"], toroidal_coords["R_z"], toroidal_coords["Z_zz"]]
    )

    # jacobian derivatives
    jacobian["g_r"] = (
        dot(
            cov_basis["e_rho_r"], cross(cov_basis["e_theta"], cov_basis["e_zeta"], 0), 0
        )
        + dot(
            cov_basis["e_rho"], cross(cov_basis["e_theta_r"], cov_basis["e_zeta"], 0), 0
        )
        + dot(
            cov_basis["e_rho"], cross(cov_basis["e_theta"], cov_basis["e_zeta_r"], 0), 0
        )
    )
    jacobian["g_t"] = (
        dot(
            cov_basis["e_rho_t"], cross(cov_basis["e_theta"], cov_basis["e_zeta"], 0), 0
        )
        + dot(
            cov_basis["e_rho"], cross(cov_basis["e_theta_t"], cov_basis["e_zeta"], 0), 0
        )
        + dot(
            cov_basis["e_rho"], cross(cov_basis["e_theta"], cov_basis["e_zeta_t"], 0), 0
        )
    )
    jacobian["g_z"] = (
        dot(
            cov_basis["e_rho_z"], cross(cov_basis["e_theta"], cov_basis["e_zeta"], 0), 0
        )
        + dot(
            cov_basis["e_rho"], cross(cov_basis["e_theta_z"], cov_basis["e_zeta"], 0), 0
        )
        + dot(
            cov_basis["e_rho"], cross(cov_basis["e_theta"], cov_basis["e_zeta_z"], 0), 0
        )
    )

    # B contravariant component derivatives
    magnetic_field["B0_r"] = profiles["psi_rr"] / (2 * jnp.pi * jacobian["g"]) - (
        profiles["psi_r"] * jacobian["g_r"]
    ) / (2 * jnp.pi * jacobian["g"] ** 2)
    magnetic_field["B0_t"] = -(profiles["psi_r"] * jacobian["g_t"]) / (
        2 * jnp.pi * jacobian["g"] ** 2
    )
    magnetic_field["B0_z"] = -(profiles["psi_r"] * jacobian["g_z"]) / (
        2 * jnp.pi * jacobian["g"] ** 2
    )
    magnetic_field["B^theta_r"] = magnetic_field["B0_r"] * (
        profiles["iota"] - toroidal_coords["lambda_z"]
    ) + magnetic_field["B0"] * (profiles["iota_r"] - toroidal_coords["lambda_rz"])
    magnetic_field["B^theta_t"] = (
        magnetic_field["B0_t"] * (profiles["iota"] - toroidal_coords["lambda_z"])
        - magnetic_field["B0"] * toroidal_coords["lambda_tz"]
    )
    magnetic_field["B^theta_z"] = (
        magnetic_field["B0_z"] * (profiles["iota"] - toroidal_coords["lambda_z"])
        - magnetic_field["B0"] * toroidal_coords["lambda_zz"]
    )
    magnetic_field["B^zeta_r"] = (
        magnetic_field["B0_r"] * (1 + toroidal_coords["lambda_t"])
        + magnetic_field["B0"] * toroidal_coords["lambda_rt"]
    )
    magnetic_field["B^zeta_t"] = (
        magnetic_field["B0_t"] * (1 + toroidal_coords["lambda_t"])
        + magnetic_field["B0"] * toroidal_coords["lambda_tt"]
    )
    magnetic_field["B^zeta_z"] = (
        magnetic_field["B0_z"] * (1 + toroidal_coords["lambda_t"])
        + magnetic_field["B0"] * toroidal_coords["lambda_tz"]
    )
    magnetic_field["B_con_r"] = (
        magnetic_field["B^theta_r"] * cov_basis["e_theta"]
        + magnetic_field["B^theta"] * cov_basis["e_theta_r"]
        + magnetic_field["B^zeta_r"] * cov_basis["e_zeta"]
        + magnetic_field["B^zeta"] * cov_basis["e_zeta_r"]
    )
    magnetic_field["B_con_t"] = (
        magnetic_field["B^theta_t"] * cov_basis["e_theta"]
        + magnetic_field["B^theta"] * cov_basis["e_theta_t"]
        + magnetic_field["B^zeta_t"] * cov_basis["e_zeta"]
        + magnetic_field["B^zeta"] * cov_basis["e_zeta_t"]
    )
    magnetic_field["B_con_z"] = (
        magnetic_field["B^theta_z"] * cov_basis["e_theta"]
        + magnetic_field["B^theta"] * cov_basis["e_theta_z"]
        + magnetic_field["B^zeta_z"] * cov_basis["e_zeta"]
        + magnetic_field["B^zeta"] * cov_basis["e_zeta_z"]
    )

    # B covariant component derivatives
    magnetic_field["B_rho_t"] = dot(
        magnetic_field["B_con_t"], cov_basis["e_rho"], 0
    ) + dot(magnetic_field["B_con"], cov_basis["e_rho_t"], 0)
    magnetic_field["B_rho_z"] = dot(
        magnetic_field["B_con_z"], cov_basis["e_rho"], 0
    ) + dot(magnetic_field["B_con"], cov_basis["e_rho_z"], 0)
    magnetic_field["B_theta_r"] = dot(
        magnetic_field["B_con_r"], cov_basis["e_theta"], 0
    ) + dot(magnetic_field["B_con"], cov_basis["e_theta_r"], 0)
    magnetic_field["B_theta_z"] = dot(
        magnetic_field["B_con_z"], cov_basis["e_theta"], 0
    ) + dot(magnetic_field["B_con"], cov_basis["e_theta_z"], 0)
    magnetic_field["B_zeta_r"] = dot(
        magnetic_field["B_con_r"], cov_basis["e_zeta"], 0
    ) + dot(magnetic_field["B_con"], cov_basis["e_zeta_r"], 0)
    magnetic_field["B_zeta_t"] = dot(
        magnetic_field["B_con_t"], cov_basis["e_zeta"], 0
    ) + dot(magnetic_field["B_con"], cov_basis["e_zeta_t"], 0)

    current_density = {}

    # J contravariant components
    current_density["J^rho"] = (
        magnetic_field["B_zeta_t"] - magnetic_field["B_theta_z"]
    ) / (mu0 * jacobian["g"])
    current_density["J^theta"] = (
        magnetic_field["B_rho_z"] - magnetic_field["B_zeta_r"]
    ) / (mu0 * jacobian["g"])
    current_density["J^zeta"] = (
        magnetic_field["B_theta_r"] - magnetic_field["B_rho_t"]
    ) / (mu0 * jacobian["g"])

    return (
        current_density,
        magnetic_field,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    )


def compute_force_error(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes force error components.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    force_error : dict
        dictionary of ndarray, shape(num_nodes,), of force error components.
        Keys are of the form 'F_x' meaning the covariant (F_x) component of the
        force error.
    current_density : dict
        dictionary of ndarray, shape(num_nodes,), of current density components.
        Keys are of the form 'J^x_y' meaning the contravariant (J^x)
        component of the current, with the derivative wrt to y.
    magnetic_field: dict
        dictionary of ndarray, shape(num_nodes,) of magnetic field components.
        Keys are of the form 'B_x_y' or 'B^x_y', meaning the covariant (B_x)
        or contravariant (B^x) component of the magnetic field, with the
        derivative wrt to y.
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate system jacobian.
        Keys are of the form 'g_x' meaning the x derivative of the coordinate
        system jacobian g.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in
        the x direction, differentiated wrt to y.
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    (
        current_density,
        magnetic_field,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    ) = compute_current_density(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    force_error = {}
    force_error["F_rho"] = -profiles["p_r"] + jacobian["g"] * (
        current_density["J^theta"] * magnetic_field["B^zeta"]
        - current_density["J^zeta"] * magnetic_field["B^theta"]
    )
    force_error["F_beta"] = jacobian["g"] * current_density["J^rho"]
    force_error["F_theta"] = force_error["F_beta"] * magnetic_field["B^zeta"]
    force_error["F_zeta"] = -force_error["F_beta"] * magnetic_field["B^theta"]

    return (
        force_error,
        current_density,
        magnetic_field,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    )


def compute_force_error_magnitude(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes force error magnitude.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    force_error : dict
        dictionary of ndarray, shape(num_nodes,), of force error components.
        Keys are of the form 'F_x' meaning the covariant (F_x) component of the
        force error.
    current_density : dict
        dictionary of ndarray, shape(num_nodes,), of current density components.
        Keys are of the form 'J^x_y' meaning the contravariant (J^x)
        component of the current, with the derivative wrt to y.
    magnetic_field: dict
        dictionary of ndarray, shape(num_nodes,) of magnetic field components.
        Keys are of the form 'B_x_y' or 'B^x_y', meaning the covariant (B_x)
        or contravariant (B^x) component of the magnetic field, with the
        derivative wrt to y.
    con_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of contravariant basis vectors.
        Keys are of the form 'e^x_y', meaning the contravariant basis vector
        in the x direction, differentiated wrt to y.
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate system jacobian.
        Keys are of the form 'g_x' meaning the x derivative of the coordinate
        system jacobian g.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in
        the x direction, differentiated wrt to y.
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    (
        force_error,
        current_density,
        magnetic_field,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    ) = compute_force_error(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    # contravariant basis vectors
    con_basis = {}
    con_basis["e^rho"] = (
        cross(cov_basis["e_theta"], cov_basis["e_zeta"], 0) / jacobian["g"]
    )
    con_basis["e^theta"] = (
        cross(cov_basis["e_zeta"], cov_basis["e_rho"], 0) / jacobian["g"]
    )
    con_basis["e^zeta"] = jnp.array(
        [toroidal_coords["0"], 1 / toroidal_coords["R"], toroidal_coords["0"]]
    )

    force_error["beta"] = (
        magnetic_field["B^zeta"] * con_basis["e^theta"]
        - magnetic_field["B^theta"] * con_basis["e^zeta"]
    )

    force_error["|grad(rho)|"] = jnp.sqrt(
        dot(con_basis["e^rho"], con_basis["e^rho"], 0)
    )
    force_error["|grad(p)|"] = jnp.sqrt(
        profiles["p_r"] ** 2 * force_error["|grad(rho)|"] ** 2
    )
    force_error["|beta|"] = jnp.sqrt(
        magnetic_field["B^zeta"] ** 2
        * dot(con_basis["e^theta"], con_basis["e^theta"], 0)
        + magnetic_field["B^theta"] ** 2
        * dot(con_basis["e^zeta"], con_basis["e^zeta"], 0)
        - 2
        * magnetic_field["B^theta"]
        * magnetic_field["B^zeta"]
        * dot(con_basis["e^theta"], con_basis["e^zeta"], 0)
    )

    force_error["|F|"] = jnp.sqrt(
        force_error["F_rho"] ** 2 * dot(con_basis["e^rho"], con_basis["e^rho"], 0)
        + force_error["F_beta"] ** 2 * force_error["|beta|"] ** 2
    )

    return (
        force_error,
        current_density,
        magnetic_field,
        con_basis,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    )


def compute_energy(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform: Transform,
    Z_transform: Transform,
    L_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes MHD energy by quadrature sum. **REQUIRES 'quad' grid for correct results**

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R_lmn : ndarray
        spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate
    Z_lmn : ndarray
        spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante
    L_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    energy: float
        The scalar value of the integral of (B^2/(2*mu0) - p) over the plasma volume.
    magnetic_field: dict
        dictionary of ndarray, shape(num_nodes,) of magnetic field components.
        Keys are of the form 'B_x_y' or 'B^x_y', meaning the covariant (B_x)
        or contravariant (B^x) component of the magnetic field, with the
        derivative wrt to y.
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate system jacobian.
        Keys are of the form 'g_x' meaning the x derivative of the coordinate
        system jacobian g.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), of covariant basis vectors.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in
        the x direction, differentiated wrt to y.
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    (
        magnetic_field,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    ) = compute_magnetic_field_magnitude(
        Psi,
        R_lmn,
        Z_lmn,
        L_lmn,
        p_l,
        i_l,
        R_transform,
        Z_transform,
        L_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    mu0 = 4 * jnp.pi * 1e-7

    pressure = profiles["p"]
    g_abs = jnp.abs(jacobian["g"])
    mag_B_sq = magnetic_field["|B|"] ** 2

    rho = R_transform.grid.nodes[:, 0]
    theta = R_transform.grid.nodes[:, 1]
    zeta = R_transform.grid.nodes[:, 2]

    NFP = R_transform.grid.NFP
    N_radial_roots = len(jnp.unique(rho))
    N_theta = len(jnp.unique(theta))
    N_zeta = len(jnp.unique(zeta))

    volumes = R_transform.grid.volumes
    dr = volumes[:, 0]
    dt = volumes[:, 1]
    dz = volumes[:, 2]

    roots, weights = special.js_roots(N_radial_roots, 2, 2)
    if not np.all(np.unique(rho) == roots):
        warnings.warn(
            colored(
                "Quadrature energy integration method requires 'quad' pattern nodes, MHD energy calculated will be incorrect",
                "yellow",
            )
        )
    energy = {}

    W_p = jnp.sum(pressure * dr * dt * dz * g_abs) * NFP
    W_B = jnp.sum(mag_B_sq * dr * dt * dz * g_abs) / 2 / mu0 * NFP

    if R_transform.grid.sym:  # double to account for symmetric grid being used
        W_p = 2 * W_p
        W_B = 2 * W_B

    energy["W_p"] = -W_p
    energy["W_B"] = W_B
    W = W_B - W_p
    energy["W"] = W

    return (
        energy,
        magnetic_field,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    )
