import numpy as np

from desc.backend import jnp
from desc.utils import dot, cross
from desc.transform import Transform


"""These functions perform the core calculations of physical quantities.
They are used as methods of the Configuration class, and also used to compute
quantities in the objective functions.
All of the functions in this file have the same call signature:

Parameters
----------
Psi : float
    total toroidal flux (in Webers) within the last closed flux surface
R0_n : ndarray
    spectral coefficients of R0(zeta) -- magnetic axis
Z0_n : ndarray
    spectral coefficients of Z0(zeta) -- magnetic axis
r_lmn : ndarray
    spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
l_lmn : ndarray
    spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
R1_mn : ndarray
    spectral coefficients of R1(theta,zeta) -- last closed flux surface
Z1_mn : ndarray
    spectral coefficients of Z1(theta,zeta) -- last closed flux surface
p_l : ndarray
    spectral coefficients of p(rho) -- pressure profile
i_l : ndarray
    spectral coefficients of iota(rho) -- rotational transform profile
R0_transform : Transform
    transforms R0_n coefficients to real space
Z0_transform : Transform
    transforms Z0_n coefficients to real space
r_transform : Transform
    transforms r_lmn coefficients to real space
l_transform : Transform
    transforms l_lmn coefficients to real space
R1_transform : Transform
    transforms R1_mn coefficients to real space
Z1_transform : Transform
    transforms Z1_mn coefficients to real space
p_transform : Transform
    transforms p_l coefficients to real space
i_transform : Transform
    transforms i_l coefficients to real space
zeta_ratio : float
    scale factor for zeta derivatives. Setting to zero effectively solves
    for individual tokamak solutions at each toroidal plane,
    setting to 1 solves for a stellarator. (Default value = 1.0)

"""


def compute_polar_coords(
    Psi,
    R0_n,
    Z0_n,
    r_lmn,
    l_lmn,
    R1_mn,
    Z1_mn,
    p_l,
    i_l,
    R0_transform: Transform,
    Z0_transform: Transform,
    r_transform: Transform,
    l_transform: Transform,
    R1_transform: Transform,
    Z1_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Transforms spectral coefficients of polar coordinates to real space.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R0_n : ndarray
        spectral coefficients of R0(zeta) -- magnetic axis
    Z0_n : ndarray
        spectral coefficients of Z0(zeta) -- magnetic axis
    r_lmn : ndarray
        spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
    l_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    R1_mn : ndarray
        spectral coefficients of R1(theta,zeta) -- last closed flux surface
    Z1_mn : ndarray
        spectral coefficients of Z1(theta,zeta) -- last closed flux surface
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
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
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    polar_coords = {}

    polar_coords["rho"] = R0_transform.grid.nodes[:, 0]
    polar_coords["theta"] = R0_transform.grid.nodes[:, 1]
    polar_coords["zeta"] = R0_transform.grid.nodes[:, 2]
    polar_coords["0"] = jnp.zeros_like(polar_coords["rho"])

    polar_coords["R0"] = R0_transform.transform(R0_n)
    polar_coords["Z0"] = Z0_transform.transform(Z0_n)
    polar_coords["r"] = r_transform.transform(r_lmn)
    polar_coords["lambda"] = l_transform.transform(l_lmn)
    polar_coords["R1"] = R1_transform.transform(R1_mn)
    polar_coords["Z1"] = Z1_transform.transform(Z1_mn)

    return polar_coords


def compute_profiles(
    Psi,
    R0_n,
    Z0_n,
    r_lmn,
    l_lmn,
    R1_mn,
    Z1_mn,
    p_l,
    i_l,
    R0_transform: Transform,
    Z0_transform: Transform,
    r_transform: Transform,
    l_transform: Transform,
    R1_transform: Transform,
    Z1_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes magnetic flux, pressure, and rotational transform profiles.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R0_n : ndarray
        spectral coefficients of R0(zeta) -- magnetic axis
    Z0_n : ndarray
        spectral coefficients of Z0(zeta) -- magnetic axis
    r_lmn : ndarray
        spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
    l_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    R1_mn : ndarray
        spectral coefficients of R1(theta,zeta) -- last closed flux surface
    Z1_mn : ndarray
        spectral coefficients of Z1(theta,zeta) -- last closed flux surface
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
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
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    polar_coords = compute_polar_coords(
        Psi,
        R0_n,
        Z0_n,
        r_lmn,
        l_lmn,
        p_l,
        i_l,
        R0_transform,
        Z0_transform,
        r_transform,
        l_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    profiles = {}

    # toroidal flux
    profiles["psi"] = Psi * polar_coords["rho"] ** 2

    # pressure
    profiles["p"] = p_transform.transform(p_l, 0)

    # rotational transform
    profiles["iota"] = i_transform.transform(i_l, 0)

    return profiles, polar_coords


def compute_toroidal_coords(
    Psi,
    R0_n,
    Z0_n,
    r_lmn,
    l_lmn,
    R1_mn,
    Z1_mn,
    p_l,
    i_l,
    R0_transform: Transform,
    Z0_transform: Transform,
    r_transform: Transform,
    l_transform: Transform,
    R1_transform: Transform,
    Z1_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes toroidal coordinates from polar coordinates.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R0_n : ndarray
        spectral coefficients of R0(zeta) -- magnetic axis
    Z0_n : ndarray
        spectral coefficients of Z0(zeta) -- magnetic axis
    r_lmn : ndarray
        spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
    l_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    R1_mn : ndarray
        spectral coefficients of R1(theta,zeta) -- last closed flux surface
    Z1_mn : ndarray
        spectral coefficients of Z1(theta,zeta) -- last closed flux surface
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
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
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    polar_coords = compute_polar_coords(
        Psi,
        R0_n,
        Z0_n,
        r_lmn,
        l_lmn,
        R1_mn,
        Z1_mn,
        p_l,
        i_l,
        R0_transform,
        Z0_transform,
        r_transform,
        l_transform,
        R1_transform,
        Z1_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    toroidal_coords = {}
    toroidal_coords["0"] = polar_coords["0"]
    toroidal_coords["R"] = polar_coords["R0"] + polar_coords["r"] * (
        polar_coords["R1"] - polar_coords["R0"]
    )
    toroidal_coords["Z"] = polar_coords["Z0"] + polar_coords["r"] * (
        polar_coords["Z1"] - polar_coords["Z0"]
    )
    toroidal_coords["phi"] = polar_coords["zeta"]  # phi = zeta

    return toroidal_coords, polar_coords


def compute_cartesian_coords(
    Psi,
    R0_n,
    Z0_n,
    r_lmn,
    l_lmn,
    R1_mn,
    Z1_mn,
    p_l,
    i_l,
    R0_transform: Transform,
    Z0_transform: Transform,
    r_transform: Transform,
    l_transform: Transform,
    R1_transform: Transform,
    Z1_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes cartesian coordinates from toroidal coordinates.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R0_n : ndarray
        spectral coefficients of R0(zeta) -- magnetic axis
    Z0_n : ndarray
        spectral coefficients of Z0(zeta) -- magnetic axis
    r_lmn : ndarray
        spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
    l_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    R1_mn : ndarray
        spectral coefficients of R1(theta,zeta) -- last closed flux surface
    Z1_mn : ndarray
        spectral coefficients of Z1(theta,zeta) -- last closed flux surface
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
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
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    toroidal_coords, polar_coords = compute_toroidal_coords(
        Psi,
        R0_n,
        Z0_n,
        r_lmn,
        l_lmn,
        R1_mn,
        Z1_mn,
        p_l,
        i_l,
        R0_transform,
        Z0_transform,
        r_transform,
        l_transform,
        R1_transform,
        Z1_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    cartesian_coords = {}
    cartesian_coords["X"] = toroidal_coords["R"] * np.cos(toroidal_coords["phi"])
    cartesian_coords["Y"] = toroidal_coords["R"] * np.sin(toroidal_coords["phi"])
    cartesian_coords["Z"] = toroidal_coords["Z"]
    cartesian_coords["0"] = toroidal_coords["0"]

    return cartesian_coords, toroidal_coords, polar_coords


def compute_covariant_basis(
    Psi,
    R0_n,
    Z0_n,
    r_lmn,
    l_lmn,
    R1_mn,
    Z1_mn,
    p_l,
    i_l,
    R0_transform: Transform,
    Z0_transform: Transform,
    r_transform: Transform,
    l_transform: Transform,
    R1_transform: Transform,
    Z1_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes covariant basis vectors.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R0_n : ndarray
        spectral coefficients of R0(zeta) -- magnetic axis
    Z0_n : ndarray
        spectral coefficients of Z0(zeta) -- magnetic axis
    r_lmn : ndarray
        spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
    l_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    R1_mn : ndarray
        spectral coefficients of R1(theta,zeta) -- last closed flux surface
    Z1_mn : ndarray
        spectral coefficients of Z1(theta,zeta) -- last closed flux surface
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
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
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    toroidal_coords, polar_coords = compute_toroidal_coords(
        Psi,
        R0_n,
        Z0_n,
        r_lmn,
        l_lmn,
        R1_mn,
        Z1_mn,
        p_l,
        i_l,
        R0_transform,
        Z0_transform,
        r_transform,
        l_transform,
        R1_transform,
        Z1_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    # polar coordinate 1st derivatives
    polar_coords["R0_z"] = R0_transform.transform(R0_n, 0, 0, 1) * zeta_ratio
    polar_coords["Z0_z"] = Z0_transform.transform(Z0_n, 0, 0, 1) * zeta_ratio
    polar_coords["r_r"] = r_transform.transform(r_lmn, 1, 0, 0)
    polar_coords["r_t"] = r_transform.transform(r_lmn, 0, 1, 0)
    polar_coords["r_z"] = r_transform.transform(r_lmn, 0, 0, 1) * zeta_ratio
    polar_coords["R1_t"] = R1_transform.transform(R1_mn, 0, 1, 0)
    polar_coords["Z1_t"] = Z1_transform.transform(Z1_mn, 0, 1, 0)
    polar_coords["R1_z"] = R1_transform.transform(R1_mn, 0, 0, 1) * zeta_ratio
    polar_coords["Z1_z"] = Z1_transform.transform(Z1_mn, 0, 0, 1) * zeta_ratio

    # toroidal coordinate 1st derivatives
    toroidal_coords["R_r"] = polar_coords["r_r"] * (
        polar_coords["R1"] - polar_coords["R0"]
    )
    toroidal_coords["Z_r"] = polar_coords["r_r"] * (
        polar_coords["Z1"] - polar_coords["Z0"]
    )
    toroidal_coords["R_t"] = (
        polar_coords["r_t"] * (polar_coords["R1"] - polar_coords["R0"])
        + polar_coords["r"] * polar_coords["R1_t"]
    )
    toroidal_coords["Z_t"] = (
        polar_coords["r_t"] * (polar_coords["Z1"] - polar_coords["Z0"])
        + polar_coords["r"] * polar_coords["Z1_t"]
    )
    toroidal_coords["R_z"] = (
        polar_coords["R0_z"]
        + +polar_coords["r_z"] * (polar_coords["R1"] - polar_coords["R0"])
        + polar_coords["r"] * (polar_coords["R1_z"] - polar_coords["R0_z"])
    )
    toroidal_coords["Z_z"] = (
        polar_coords["Z0_z"]
        + +polar_coords["r_z"] * (polar_coords["Z1"] - polar_coords["Z0"])
        + polar_coords["r"] * (polar_coords["Z1_z"] - polar_coords["Z0_z"])
    )

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

    return cov_basis, toroidal_coords, polar_coords


def compute_jacobian(
    Psi,
    R0_n,
    Z0_n,
    r_lmn,
    l_lmn,
    R1_mn,
    Z1_mn,
    p_l,
    i_l,
    R0_transform: Transform,
    Z0_transform: Transform,
    r_transform: Transform,
    l_transform: Transform,
    R1_transform: Transform,
    Z1_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes coordinate system jacobian.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R0_n : ndarray
        spectral coefficients of R0(zeta) -- magnetic axis
    Z0_n : ndarray
        spectral coefficients of Z0(zeta) -- magnetic axis
    r_lmn : ndarray
        spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
    l_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    R1_mn : ndarray
        spectral coefficients of R1(theta,zeta) -- last closed flux surface
    Z1_mn : ndarray
        spectral coefficients of Z1(theta,zeta) -- last closed flux surface
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
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
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    cov_basis, toroidal_coords, polar_coords = compute_covariant_basis(
        Psi,
        R0_n,
        Z0_n,
        r_lmn,
        l_lmn,
        R1_mn,
        Z1_mn,
        p_l,
        i_l,
        R0_transform,
        Z0_transform,
        r_transform,
        l_transform,
        R1_transform,
        Z1_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    jacobian = {}
    jacobian["g"] = dot(
        cov_basis["e_rho"], cross(cov_basis["e_theta"], cov_basis["e_zeta"], 0), 0
    )

    return jacobian, cov_basis, toroidal_coords, polar_coords


def compute_contravariant_basis(
    Psi,
    R0_n,
    Z0_n,
    r_lmn,
    l_lmn,
    R1_mn,
    Z1_mn,
    p_l,
    i_l,
    R0_transform: Transform,
    Z0_transform: Transform,
    r_transform: Transform,
    l_transform: Transform,
    R1_transform: Transform,
    Z1_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes contravariant basis vectors.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R0_n : ndarray
        spectral coefficients of R0(zeta) -- magnetic axis
    Z0_n : ndarray
        spectral coefficients of Z0(zeta) -- magnetic axis
    r_lmn : ndarray
        spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
    l_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    R1_mn : ndarray
        spectral coefficients of R1(theta,zeta) -- last closed flux surface
    Z1_mn : ndarray
        spectral coefficients of Z1(theta,zeta) -- last closed flux surface
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
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
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
    """
    # prerequisites
    jacobian, cov_basis, toroidal_coords, polar_coords = compute_jacobian(
        Psi,
        R0_n,
        Z0_n,
        r_lmn,
        l_lmn,
        R1_mn,
        Z1_mn,
        p_l,
        i_l,
        R0_transform,
        Z0_transform,
        r_transform,
        l_transform,
        R1_transform,
        Z1_transform,
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

    return con_basis, jacobian, cov_basis, toroidal_coords, polar_coords


def compute_magnetic_field(
    Psi,
    R0_n,
    Z0_n,
    r_lmn,
    l_lmn,
    R1_mn,
    Z1_mn,
    p_l,
    i_l,
    R0_transform: Transform,
    Z0_transform: Transform,
    r_transform: Transform,
    l_transform: Transform,
    R1_transform: Transform,
    Z1_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes magnetic field components.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R0_n : ndarray
        spectral coefficients of R0(zeta) -- magnetic axis
    Z0_n : ndarray
        spectral coefficients of Z0(zeta) -- magnetic axis
    r_lmn : ndarray
        spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
    l_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    R1_mn : ndarray
        spectral coefficients of R1(theta,zeta) -- last closed flux surface
    Z1_mn : ndarray
        spectral coefficients of Z1(theta,zeta) -- last closed flux surface
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
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
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
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
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    jacobian, cov_basis, toroidal_coords, polar_coords = compute_jacobian(
        Psi,
        R0_n,
        Z0_n,
        r_lmn,
        l_lmn,
        R1_mn,
        Z1_mn,
        p_l,
        i_l,
        R0_transform,
        Z0_transform,
        r_transform,
        l_transform,
        R1_transform,
        Z1_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    # profiles
    profiles = {}
    profiles["iota"] = i_transform.transform(i_l, 0)
    profiles["psi"] = Psi * polar_coords["rho"] ** 2
    profiles["psi_r"] = 2 * Psi * polar_coords["rho"]

    # lambda derivatives
    polar_coords["lambda_t"] = l_transform.transform(l_lmn, 0, 1, 0)
    polar_coords["lambda_z"] = l_transform.transform(l_lmn, 0, 0, 1) * zeta_ratio

    magnetic_field = {}
    magnetic_field["B0"] = profiles["psi_r"] / (2 * jnp.pi * jacobian["g"])

    # contravariant components
    magnetic_field["B^rho"] = jnp.zeros_like(magnetic_field["B0"])
    magnetic_field["B^theta"] = magnetic_field["B0"] * (
        profiles["iota"] - polar_coords["lambda_z"]
    )
    magnetic_field["B^zeta"] = magnetic_field["B0"] * (1 + polar_coords["lambda_t"])
    magnetic_field["B_con"] = (
        magnetic_field["B^theta"] * cov_basis["e_theta"]
        + magnetic_field["B^zeta"] * cov_basis["e_zeta"]
    )

    # covariant components
    magnetic_field["B_rho"] = dot(magnetic_field["B_con"], cov_basis["e_rho"], 0)
    magnetic_field["B_theta"] = dot(magnetic_field["B_con"], cov_basis["e_theta"], 0)
    magnetic_field["B_zeta"] = dot(magnetic_field["B_con"], cov_basis["e_zeta"], 0)

    return magnetic_field, profiles, jacobian, cov_basis, toroidal_coords, polar_coords


def compute_magnetic_field_magnitude(
    Psi,
    R0_n,
    Z0_n,
    r_lmn,
    l_lmn,
    R1_mn,
    Z1_mn,
    p_l,
    i_l,
    R0_transform: Transform,
    Z0_transform: Transform,
    r_transform: Transform,
    l_transform: Transform,
    R1_transform: Transform,
    Z1_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes magnetic field magnitude.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R0_n : ndarray
        spectral coefficients of R0(zeta) -- magnetic axis
    Z0_n : ndarray
        spectral coefficients of Z0(zeta) -- magnetic axis
    r_lmn : ndarray
        spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
    l_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    R1_mn : ndarray
        spectral coefficients of R1(theta,zeta) -- last closed flux surface
    Z1_mn : ndarray
        spectral coefficients of Z1(theta,zeta) -- last closed flux surface
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
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
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
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
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    (
        magnetic_field,
        profiles,
        jacobian,
        cov_basis,
        toroidal_coords,
        polar_coords,
    ) = compute_magnetic_field(
        Psi,
        R0_n,
        Z0_n,
        r_lmn,
        l_lmn,
        R1_mn,
        Z1_mn,
        p_l,
        i_l,
        R0_transform,
        Z0_transform,
        r_transform,
        l_transform,
        R1_transform,
        Z1_transform,
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

    return magnetic_field, profiles, jacobian, cov_basis, toroidal_coords, polar_coords


def compute_current_density(
    Psi,
    R0_n,
    Z0_n,
    r_lmn,
    l_lmn,
    R1_mn,
    Z1_mn,
    p_l,
    i_l,
    R0_transform: Transform,
    Z0_transform: Transform,
    r_transform: Transform,
    l_transform: Transform,
    R1_transform: Transform,
    Z1_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes current density field components.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R0_n : ndarray
        spectral coefficients of R0(zeta) -- magnetic axis
    Z0_n : ndarray
        spectral coefficients of Z0(zeta) -- magnetic axis
    r_lmn : ndarray
        spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
    l_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    R1_mn : ndarray
        spectral coefficients of R1(theta,zeta) -- last closed flux surface
    Z1_mn : ndarray
        spectral coefficients of Z1(theta,zeta) -- last closed flux surface
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
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
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
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
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    (
        magnetic_field,
        profiles,
        jacobian,
        cov_basis,
        toroidal_coords,
        polar_coords,
    ) = compute_magnetic_field(
        Psi,
        R0_n,
        Z0_n,
        r_lmn,
        l_lmn,
        R1_mn,
        Z1_mn,
        p_l,
        i_l,
        R0_transform,
        Z0_transform,
        r_transform,
        l_transform,
        R1_transform,
        Z1_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )
    mu0 = 4 * jnp.pi * 1e-7

    # polar coordinate derivatives
    polar_coords["R0_zz"] = R0_transform.transform(R0_n, 0, 0, 2) * zeta_ratio
    polar_coords["Z0_zz"] = Z0_transform.transform(Z0_n, 0, 0, 2) * zeta_ratio
    polar_coords["r_rr"] = r_transform.transform(r_lmn, 2, 0, 0)
    polar_coords["r_rt"] = r_transform.transform(r_lmn, 1, 1, 0)
    polar_coords["r_rz"] = r_transform.transform(r_lmn, 1, 0, 1) * zeta_ratio
    polar_coords["r_tt"] = r_transform.transform(r_lmn, 0, 2, 0)
    polar_coords["r_tz"] = r_transform.transform(r_lmn, 0, 1, 1) * zeta_ratio
    polar_coords["r_zz"] = r_transform.transform(r_lmn, 0, 0, 2) * zeta_ratio
    polar_coords["R1_tt"] = R1_transform.transform(R1_mn, 0, 2, 0)
    polar_coords["Z1_tt"] = Z1_transform.transform(Z1_mn, 0, 2, 0)
    polar_coords["R1_tz"] = R1_transform.transform(R1_mn, 0, 1, 1) * zeta_ratio
    polar_coords["Z1_tz"] = Z1_transform.transform(Z1_mn, 0, 1, 1) * zeta_ratio
    polar_coords["R1_zz"] = R1_transform.transform(R1_mn, 0, 0, 2) * zeta_ratio
    polar_coords["Z1_zz"] = Z1_transform.transform(Z1_mn, 0, 0, 2) * zeta_ratio

    # lambda derivatives
    polar_coords["lambda_rt"] = l_transform.transform(l_lmn, 1, 1, 0)
    polar_coords["lambda_rz"] = l_transform.transform(l_lmn, 1, 0, 1) * zeta_ratio
    polar_coords["lambda_tt"] = l_transform.transform(l_lmn, 0, 2, 0)
    polar_coords["lambda_tz"] = l_transform.transform(l_lmn, 0, 1, 1) * zeta_ratio
    polar_coords["lambda_zz"] = l_transform.transform(l_lmn, 0, 0, 2) * zeta_ratio

    # profile derivatives
    profiles["iota_r"] = i_transform.transform(i_l, 1)
    profiles["psi_rr"] = 2 * Psi * jnp.ones_like(profiles["psi"])

    # toroidal coordinate 2nd derivatives
    toroidal_coords["R_rr"] = polar_coords["r_rr"] * (
        polar_coords["R1"] - polar_coords["R0"]
    )
    toroidal_coords["Z_rr"] = polar_coords["r_rr"] * (
        polar_coords["Z1"] - polar_coords["Z0"]
    )
    toroidal_coords["R_rt"] = (
        polar_coords["r_rt"] * (polar_coords["R1"] - polar_coords["R0"])
        + polar_coords["r_r"] * polar_coords["R1_t"]
    )
    toroidal_coords["Z_rt"] = (
        polar_coords["r_rt"] * (polar_coords["Z1"] - polar_coords["Z0"])
        + polar_coords["r_r"] * polar_coords["Z1_t"]
    )
    toroidal_coords["R_rz"] = polar_coords["r_rz"] * (
        polar_coords["R1"] - polar_coords["R0"]
    ) + polar_coords["r_r"] * (polar_coords["R1_z"] - polar_coords["R0_z"])
    toroidal_coords["Z_rz"] = polar_coords["r_rz"] * (
        polar_coords["Z1"] - polar_coords["Z0"]
    ) + polar_coords["r_r"] * (polar_coords["Z1_z"] - polar_coords["Z0_z"])
    toroidal_coords["R_tt"] = (
        polar_coords["r_tt"] * (polar_coords["R1"] - polar_coords["R0"])
        + 2 * polar_coords["r_t"] * polar_coords["R1_t"]
        + polar_coords["r"] * polar_coords["R1_tt"]
    )
    toroidal_coords["Z_tt"] = (
        polar_coords["r_tt"] * (polar_coords["Z1"] - polar_coords["Z0"])
        + 2 * polar_coords["r_t"] * polar_coords["Z1_t"]
        + polar_coords["r"] * polar_coords["Z1_tt"]
    )
    toroidal_coords["R_tz"] = (
        polar_coords["r_tz"] * (polar_coords["R1"] - polar_coords["R0"])
        + polar_coords["r_z"] * polar_coords["R1_t"]
        + polar_coords["r_t"] * (polar_coords["R1_z"] - polar_coords["R0_z"])
        + polar_coords["r"] * polar_coords["R1_tz"]
    )
    toroidal_coords["Z_tz"] = (
        polar_coords["r_tz"] * (polar_coords["Z1"] - polar_coords["Z0"])
        + polar_coords["r_z"] * polar_coords["Z1_t"]
        + polar_coords["r_t"] * (polar_coords["Z1_z"] - polar_coords["Z0_z"])
        + polar_coords["r"] * polar_coords["Z1_tz"]
    )
    toroidal_coords["R_zz"] = (
        polar_coords["R0_zz"]
        + polar_coords["r_zz"] * (polar_coords["R1"] - polar_coords["R0"])
        + 2 * polar_coords["r_z"] * (polar_coords["R1_z"] - polar_coords["R0_z"])
        + polar_coords["r"] * (polar_coords["R1_zz"] - polar_coords["R0_zz"])
    )
    toroidal_coords["Z_zz"] = (
        polar_coords["Z0_zz"]
        + polar_coords["r_zz"] * (polar_coords["Z1"] - polar_coords["Z0"])
        + 2 * polar_coords["r_z"] * (polar_coords["Z1_z"] - polar_coords["Z0_z"])
        + polar_coords["r"] * (polar_coords["Z1_zz"] - polar_coords["Z0_zz"])
    )

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
        profiles["iota"] - polar_coords["lambda_z"]
    ) + magnetic_field["B0"] * (profiles["iota_r"] - polar_coords["lambda_rz"])
    magnetic_field["B^theta_t"] = (
        magnetic_field["B0_t"] * (profiles["iota"] - polar_coords["lambda_z"])
        - magnetic_field["B0"] * polar_coords["lambda_tz"]
    )
    magnetic_field["B^theta_z"] = (
        magnetic_field["B0_z"] * (profiles["iota"] - polar_coords["lambda_z"])
        - magnetic_field["B0"] * polar_coords["lambda_zz"]
    )
    magnetic_field["B^zeta_r"] = (
        magnetic_field["B0_r"] * (1 + polar_coords["lambda_t"])
        + magnetic_field["B0"] * polar_coords["lambda_rt"]
    )
    magnetic_field["B^zeta_t"] = (
        magnetic_field["B0_t"] * (1 + polar_coords["lambda_t"])
        + magnetic_field["B0"] * polar_coords["lambda_tt"]
    )
    magnetic_field["B^zeta_z"] = (
        magnetic_field["B0_z"] * (1 + polar_coords["lambda_t"])
        + magnetic_field["B0"] * polar_coords["lambda_tz"]
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
        profiles,
        jacobian,
        cov_basis,
        toroidal_coords,
        polar_coords,
    )


def compute_force_error(
    Psi,
    R0_n,
    Z0_n,
    r_lmn,
    l_lmn,
    R1_mn,
    Z1_mn,
    p_l,
    i_l,
    R0_transform: Transform,
    Z0_transform: Transform,
    r_transform: Transform,
    l_transform: Transform,
    R1_transform: Transform,
    Z1_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes force error components.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R0_n : ndarray
        spectral coefficients of R0(zeta) -- magnetic axis
    Z0_n : ndarray
        spectral coefficients of Z0(zeta) -- magnetic axis
    r_lmn : ndarray
        spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
    l_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    R1_mn : ndarray
        spectral coefficients of R1(theta,zeta) -- last closed flux surface
    Z1_mn : ndarray
        spectral coefficients of Z1(theta,zeta) -- last closed flux surface
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
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
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
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
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    (
        current_density,
        magnetic_field,
        profiles,
        jacobian,
        cov_basis,
        toroidal_coords,
        polar_coords,
    ) = compute_current_density(
        Psi,
        R0_n,
        Z0_n,
        r_lmn,
        l_lmn,
        R1_mn,
        Z1_mn,
        p_l,
        i_l,
        R0_transform,
        Z0_transform,
        r_transform,
        l_transform,
        R1_transform,
        Z1_transform,
        p_transform,
        i_transform,
        zeta_ratio,
    )

    # profile derivatives
    profiles["p_r"] = p_transform.transform(p_l, 1)

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
        profiles,
        jacobian,
        cov_basis,
        toroidal_coords,
        polar_coords,
    )


def compute_force_error_magnitude(
    Psi,
    R0_n,
    Z0_n,
    r_lmn,
    l_lmn,
    R1_mn,
    Z1_mn,
    p_l,
    i_l,
    R0_transform: Transform,
    Z0_transform: Transform,
    r_transform: Transform,
    l_transform: Transform,
    R1_transform: Transform,
    Z1_transform: Transform,
    p_transform: Transform,
    i_transform: Transform,
    zeta_ratio=1.0,
):
    """Computes force error magnitude.

    Parameters
    ----------
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    R0_n : ndarray
        spectral coefficients of R0(zeta) -- magnetic axis
    Z0_n : ndarray
        spectral coefficients of Z0(zeta) -- magnetic axis
    r_lmn : ndarray
        spectral coefficients of r(rho,theta,zeta) -- flux surface shapes
    l_lmn : ndarray
        spectral coefficients of lambda(rho,theta,zeta) -- sfl coordinate map
    R1_mn : ndarray
        spectral coefficients of R1(theta,zeta) -- last closed flux surface
    Z1_mn : ndarray
        spectral coefficients of Z1(theta,zeta) -- last closed flux surface
    p_l : ndarray
        spectral coefficients of p(rho) -- pressure profile
    i_l : ndarray
        spectral coefficients of iota(rho) -- rotational transform profile
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
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
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.
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
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # prerequisites
    (
        force_error,
        current_density,
        magnetic_field,
        profiles,
        jacobian,
        cov_basis,
        toroidal_coords,
        polar_coords,
    ) = compute_force_error(
        Psi,
        R0_n,
        Z0_n,
        r_lmn,
        l_lmn,
        R1_mn,
        Z1_mn,
        p_l,
        i_l,
        R0_transform,
        Z0_transform,
        r_transform,
        l_transform,
        R1_transform,
        Z1_transform,
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
        profiles,
        con_basis,
        jacobian,
        cov_basis,
        toroidal_coords,
        polar_coords,
    )
