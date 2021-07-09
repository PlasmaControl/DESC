import numpy as np
from desc.backend import jnp, put
from scipy.constants import mu_0

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
p_profile : Profile
    transforms p_l coefficients to real space
i_profile : Profile
    transforms i_l coefficients to real space

"""


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


def compute_profiles(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute magnetic flux, pressure, and rotational transform profiles.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

    Returns
    -------
    profiles : dict
        dictionary of ndarray, shape(num_nodes,) of profiles.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    profiles = {}

    # toroidal flux (Wb) divided by 2 pi
    rho = p_profile.grid.nodes[:, 0]
    profiles["psi"] = Psi * rho ** 2 / (2 * jnp.pi)
    profiles["psi_r"] = 2 * Psi * rho / (2 * jnp.pi)
    profiles["psi_rr"] = 2 * Psi * np.ones_like(rho) / (2 * jnp.pi)

    # pressure (Pa)
    profiles["p"] = p_profile.compute(p_l, dr=0)
    profiles["p_r"] = p_profile.compute(p_l, dr=1)

    # rotational transform
    profiles["iota"] = i_profile.compute(i_l, dr=0)
    profiles["iota_r"] = i_profile.compute(i_l, dr=1)

    return profiles


def compute_toroidal_coords(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Transform toroidal coordinates to real space.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute cartesian coordinates from toroidal coordinates.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
        p_profile,
        i_profile,
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
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute covariant basis vectors.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
        p_profile,
        i_profile,
    )

    # toroidal coordinate 1st derivatives
    toroidal_coords["R_r"] = R_transform.transform(R_lmn, 1, 0, 0)
    toroidal_coords["Z_r"] = Z_transform.transform(Z_lmn, 1, 0, 0)
    toroidal_coords["R_t"] = R_transform.transform(R_lmn, 0, 1, 0)
    toroidal_coords["Z_t"] = Z_transform.transform(Z_lmn, 0, 1, 0)
    toroidal_coords["R_z"] = R_transform.transform(R_lmn, 0, 0, 1)
    toroidal_coords["Z_z"] = Z_transform.transform(Z_lmn, 0, 0, 1)

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
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute coordinate system jacobian.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
        p_profile,
        i_profile,
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
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute contravariant basis vectors.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
        p_profile,
        i_profile,
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
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute magnetic field components.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
        p_profile,
        i_profile,
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
        p_profile,
        i_profile,
    )

    # lambda derivatives
    toroidal_coords["lambda_t"] = L_transform.transform(L_lmn, 0, 1, 0)
    toroidal_coords["lambda_z"] = L_transform.transform(L_lmn, 0, 0, 1)

    magnetic_field = {}
    magnetic_field["B0"] = profiles["psi_r"] / jacobian["g"]

    # contravariant components
    magnetic_field["B^rho"] = jnp.zeros_like(magnetic_field["B0"])
    magnetic_field["B^theta"] = magnetic_field["B0"] * (
        profiles["iota"] - toroidal_coords["lambda_z"]
    )
    magnetic_field["B^zeta"] = magnetic_field["B0"] * (1 + toroidal_coords["lambda_t"])
    magnetic_field["B"] = (
        magnetic_field["B^theta"] * cov_basis["e_theta"]
        + magnetic_field["B^zeta"] * cov_basis["e_zeta"]
    )

    # covariant components
    magnetic_field["B_rho"] = dot(magnetic_field["B"], cov_basis["e_rho"], 0)
    magnetic_field["B_theta"] = dot(magnetic_field["B"], cov_basis["e_theta"], 0)
    magnetic_field["B_zeta"] = dot(magnetic_field["B"], cov_basis["e_zeta"], 0)

    # cylindrical components (R,Z,phi)
    magnetic_field["B_R"] = magnetic_field['B^theta']*cov_basis['e_theta'][0,:] + magnetic_field['B^zeta']*cov_basis['e_zeta'][0,:]
    magnetic_field["B_Z"] = magnetic_field['B^theta']*cov_basis['e_theta'][2,:] + magnetic_field['B^zeta']*cov_basis['e_zeta'][2,:]
    magnetic_field["B_phi"] = toroidal_coords["R"]*magnetic_field['B^zeta']

    return magnetic_field, jacobian, cov_basis, toroidal_coords, profiles


def compute_magnetic_field_axis(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute magnetic field components; can handle nodes at the magnetic axis.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
        p_profile,
        i_profile,
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
        p_profile,
        i_profile,
    )

    axis = i_profile.grid.axis

    # lambda derivatives
    toroidal_coords["lambda_t"] = L_transform.transform(L_lmn, 0, 1, 0)
    toroidal_coords["lambda_z"] = L_transform.transform(L_lmn, 0, 0, 1)

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
    magnetic_field["B0"] = profiles["psi_r"] / jacobian["g"]
    magnetic_field["B0"] = put(
        magnetic_field["B0"], axis, profiles["psi_rr"][axis] / jacobian["g_r"][axis],
    )

    # contravariant components
    magnetic_field["B^rho"] = jnp.zeros_like(magnetic_field["B0"])
    magnetic_field["B^theta"] = magnetic_field["B0"] * (
        profiles["iota"] - toroidal_coords["lambda_z"]
    )
    magnetic_field["B^zeta"] = magnetic_field["B0"] * (1 + toroidal_coords["lambda_t"])
    magnetic_field["B"] = (
        magnetic_field["B^theta"] * cov_basis["e_theta"]
        + magnetic_field["B^zeta"] * cov_basis["e_zeta"]
    )

    # covariant components
    magnetic_field["B_rho"] = dot(magnetic_field["B"], cov_basis["e_rho"], 0)
    magnetic_field["B_theta"] = dot(magnetic_field["B"], cov_basis["e_theta"], 0)
    magnetic_field["B_zeta"] = dot(magnetic_field["B"], cov_basis["e_zeta"], 0)

    # cylindrical components (R,Z,phi)
    magnetic_field["B_R"] = magnetic_field['B^theta']*cov_basis['e_theta'][0,:] + magnetic_field['B^zeta']*cov_basis['e_zeta'][0,:]
    magnetic_field["B_Z"] = magnetic_field['B^theta']*cov_basis['e_theta'][2,:] + magnetic_field['B^zeta']*cov_basis['e_zeta'][2,:]
    magnetic_field["B_phi"] = toroidal_coords["R"]*magnetic_field['B^zeta']

    return magnetic_field, jacobian, cov_basis, toroidal_coords, profiles


def compute_magnetic_field_magnitude(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute magnetic field magnitude.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
        p_profile,
        i_profile,
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
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute magnetic field magnitude; can handle nodes at the magnetic axis.

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
    p_profile : Transform
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
        p_profile,
        i_profile,
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
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute current density field components.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
        p_profile,
        i_profile,
    )

    # toroidal coordinate 2nd derivatives
    toroidal_coords["R_rr"] = R_transform.transform(R_lmn, 2, 0, 0)
    toroidal_coords["Z_rr"] = Z_transform.transform(Z_lmn, 2, 0, 0)
    toroidal_coords["R_rt"] = R_transform.transform(R_lmn, 1, 1, 0)
    toroidal_coords["Z_rt"] = Z_transform.transform(Z_lmn, 1, 1, 0)
    toroidal_coords["R_rz"] = R_transform.transform(R_lmn, 1, 0, 1)
    toroidal_coords["Z_rz"] = Z_transform.transform(Z_lmn, 1, 0, 1)
    toroidal_coords["R_tt"] = R_transform.transform(R_lmn, 0, 2, 0)
    toroidal_coords["Z_tt"] = Z_transform.transform(Z_lmn, 0, 2, 0)
    toroidal_coords["R_tz"] = R_transform.transform(R_lmn, 0, 1, 1)
    toroidal_coords["Z_tz"] = Z_transform.transform(Z_lmn, 0, 1, 1)
    toroidal_coords["R_zz"] = R_transform.transform(R_lmn, 0, 0, 2)
    toroidal_coords["Z_zz"] = Z_transform.transform(Z_lmn, 0, 0, 2)

    # lambda derivatives
    toroidal_coords["lambda_rt"] = L_transform.transform(L_lmn, 1, 1, 0)
    toroidal_coords["lambda_rz"] = L_transform.transform(L_lmn, 1, 0, 1)
    toroidal_coords["lambda_tt"] = L_transform.transform(L_lmn, 0, 2, 0)
    toroidal_coords["lambda_tz"] = L_transform.transform(L_lmn, 0, 1, 1)
    toroidal_coords["lambda_zz"] = L_transform.transform(L_lmn, 0, 0, 2)

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
    magnetic_field["B0_r"] = (
        profiles["psi_rr"] / jacobian["g"]
        - (profiles["psi_r"] * jacobian["g_r"]) / jacobian["g"] ** 2
    )
    magnetic_field["B0_t"] = -(profiles["psi_r"] * jacobian["g_t"]) / jacobian["g"] ** 2
    magnetic_field["B0_z"] = -(profiles["psi_r"] * jacobian["g_z"]) / jacobian["g"] ** 2
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

    # B vector partial derivatives
    magnetic_field["B_r"] = (
        magnetic_field["B^theta_r"] * cov_basis["e_theta"]
        + magnetic_field["B^theta"] * cov_basis["e_theta_r"]
        + magnetic_field["B^zeta_r"] * cov_basis["e_zeta"]
        + magnetic_field["B^zeta"] * cov_basis["e_zeta_r"]
    )
    magnetic_field["B_t"] = (
        magnetic_field["B^theta_t"] * cov_basis["e_theta"]
        + magnetic_field["B^theta"] * cov_basis["e_theta_t"]
        + magnetic_field["B^zeta_t"] * cov_basis["e_zeta"]
        + magnetic_field["B^zeta"] * cov_basis["e_zeta_t"]
    )
    magnetic_field["B_z"] = (
        magnetic_field["B^theta_z"] * cov_basis["e_theta"]
        + magnetic_field["B^theta"] * cov_basis["e_theta_z"]
        + magnetic_field["B^zeta_z"] * cov_basis["e_zeta"]
        + magnetic_field["B^zeta"] * cov_basis["e_zeta_z"]
    )

    # B covariant component derivatives
    magnetic_field["B_rho_t"] = dot(magnetic_field["B_t"], cov_basis["e_rho"], 0) + dot(
        magnetic_field["B"], cov_basis["e_rho_t"], 0
    )
    magnetic_field["B_rho_z"] = dot(magnetic_field["B_z"], cov_basis["e_rho"], 0) + dot(
        magnetic_field["B"], cov_basis["e_rho_z"], 0
    )
    magnetic_field["B_theta_r"] = dot(
        magnetic_field["B_r"], cov_basis["e_theta"], 0
    ) + dot(magnetic_field["B"], cov_basis["e_theta_r"], 0)
    magnetic_field["B_theta_z"] = dot(
        magnetic_field["B_z"], cov_basis["e_theta"], 0
    ) + dot(magnetic_field["B"], cov_basis["e_theta_z"], 0)
    magnetic_field["B_zeta_r"] = dot(
        magnetic_field["B_r"], cov_basis["e_zeta"], 0
    ) + dot(magnetic_field["B"], cov_basis["e_zeta_r"], 0)
    magnetic_field["B_zeta_t"] = dot(
        magnetic_field["B_t"], cov_basis["e_zeta"], 0
    ) + dot(magnetic_field["B"], cov_basis["e_zeta_t"], 0)

    current_density = {}

    # J contravariant components
    current_density["J^rho"] = (
        magnetic_field["B_zeta_t"] - magnetic_field["B_theta_z"]
    ) / (mu_0 * jacobian["g"])
    current_density["J^theta"] = (
        magnetic_field["B_rho_z"] - magnetic_field["B_zeta_r"]
    ) / (mu_0 * jacobian["g"])
    current_density["J^zeta"] = (
        magnetic_field["B_theta_r"] - magnetic_field["B_rho_t"]
    ) / (mu_0 * jacobian["g"])

    return (
        current_density,
        magnetic_field,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    )


def compute_magnetic_pressure_gradient(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute magnetic pressure gradient components and its magnitude.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

    Returns
    -------
    magnetic_pressure : dict
        dictionary of ndarray, shape(num_nodes,), of magnetic pressure gradient components.
        Keys are of the form 'grad(B)^x' meaning the contravariant (grad(B)^x)
        component of the magnetic pressure gradient.
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
        p_profile,
        i_profile,
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

    magnetic_pressure = {}

    # B covariant component derivatives
    magnetic_field["B_theta_t"] = dot(
        magnetic_field["B_t"], cov_basis["e_theta"], 0
    ) + dot(magnetic_field["B"], cov_basis["e_theta_t"], 0)
    magnetic_field["B_zeta_z"] = dot(
        magnetic_field["B_z"], cov_basis["e_zeta"], 0
    ) + dot(magnetic_field["B"], cov_basis["e_zeta_z"], 0)

    # magnetic pressure gradient covariant components
    magnetic_pressure["grad(|B|^2)_rho"] = (
        magnetic_field["B^theta"] * magnetic_field["B_theta_r"]
        + magnetic_field["B_theta"] * magnetic_field["B^theta_r"]
        + magnetic_field["B^zeta"] * magnetic_field["B_zeta_r"]
        + magnetic_field["B_zeta"] * magnetic_field["B^zeta_r"]
    )
    magnetic_pressure["grad(|B|^2)_theta"] = (
        magnetic_field["B^theta"] * magnetic_field["B_theta_t"]
        + magnetic_field["B_theta"] * magnetic_field["B^theta_t"]
        + magnetic_field["B^zeta"] * magnetic_field["B_zeta_t"]
        + magnetic_field["B_zeta"] * magnetic_field["B^zeta_t"]
    )
    magnetic_pressure["grad(|B|^2)_zeta"] = (
        magnetic_field["B^theta"] * magnetic_field["B_theta_z"]
        + magnetic_field["B_theta"] * magnetic_field["B^theta_z"]
        + magnetic_field["B^zeta"] * magnetic_field["B_zeta_z"]
        + magnetic_field["B_zeta"] * magnetic_field["B^zeta_z"]
    )

    # magnetic pressure gradient
    magnetic_pressure["grad(|B|^2)"] = (
        magnetic_pressure["grad(|B|^2)_rho"] * con_basis["e^rho"]
        + magnetic_pressure["grad(|B|^2)_theta"] * con_basis["e^theta"]
        + magnetic_pressure["grad(|B|^2)_zeta"] * con_basis["e^zeta"]
    )

    # magnetic pressure gradient magnitude
    magnetic_pressure["|grad(|B|^2)|"] = jnp.sqrt(
        magnetic_pressure["grad(|B|^2)_rho"] ** 2
        * dot(con_basis["e^rho"], con_basis["e^rho"], 0)
        + magnetic_pressure["grad(|B|^2)_theta"] ** 2
        * dot(con_basis["e^theta"], con_basis["e^theta"], 0)
        + magnetic_pressure["grad(|B|^2)_zeta"] ** 2
        * dot(con_basis["e^zeta"], con_basis["e^zeta"], 0)
    )

    # scaled magnetic pressure
    magnetic_pressure["Bpressure"] = magnetic_pressure["|grad(|B|^2)|"] / (2 * mu_0)

    return (
        magnetic_pressure,
        current_density,
        magnetic_field,
        con_basis,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    )


def compute_magnetic_tension(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute magnetic tension vector and its magnitude.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

    Returns
    -------
    magnetic_tension : dict
        dictionary of ndarray, shape(num_nodes,), of magnetic tension vector.
        Keys are of the form 'grad(B)' for the vector form and '|grad(B)|' for its
        magnitude.
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
        p_profile,
        i_profile,
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

    magnetic_tension = {}

    # magnetic tension contravariant vector
    magnetic_tension["(B*grad(|B|))B"] = (
        (
            magnetic_field["B^theta"] * magnetic_field["B^theta_t"]
            + magnetic_field["B^zeta"] * magnetic_field["B^theta_z"]
        )
        * cov_basis["e_theta"]
        + (
            magnetic_field["B^theta"] * magnetic_field["B^zeta_t"]
            + magnetic_field["B^zeta"] * magnetic_field["B^zeta_z"]
        )
        * cov_basis["e_zeta"]
        + magnetic_field["B^theta"] ** 2 * cov_basis["e_theta_t"]
        + magnetic_field["B^zeta"] ** 2 * cov_basis["e_zeta_z"]
        + magnetic_field["B^theta"]
        * magnetic_field["B^zeta"]
        * (cov_basis["e_theta_z"] + cov_basis["e_zeta_t"])
    )

    # magnetic tension contravariant components
    magnetic_tension["((B*grad(|B|))B)^rho"] = dot(
        magnetic_tension["(B*grad(|B|))B"], con_basis["e^rho"], 0
    )
    magnetic_tension["((B*grad(|B|))B)^theta"] = dot(
        magnetic_tension["(B*grad(|B|))B"], con_basis["e^theta"], 0
    )
    magnetic_tension["((B*grad(|B|))B)^zeta"] = dot(
        magnetic_tension["(B*grad(|B|))B"], con_basis["e^zeta"], 0
    )

    # magnetic tension magnitude
    magnetic_tension["|(B*grad(|B|))B|"] = jnp.sqrt(
        magnetic_tension["((B*grad(|B|))B)^rho"] ** 2
        * dot(cov_basis["e_rho"], cov_basis["e_rho"], 0)
        + magnetic_tension["((B*grad(|B|))B)^theta"] ** 2
        * dot(cov_basis["e_theta"], cov_basis["e_theta"], 0)
        + magnetic_tension["((B*grad(|B|))B)^zeta"] ** 2
        * dot(cov_basis["e_zeta"], cov_basis["e_zeta"], 0)
    )

    # scaled magnetic tension
    magnetic_tension["Btension"] = magnetic_tension["|(B*grad(|B|))B|"] / (2 * mu_0)

    return (
        magnetic_tension,
        current_density,
        magnetic_field,
        con_basis,
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
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute force error components.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
        p_profile,
        i_profile,
    )

    force_error = {}
    force_error["F_rho"] = -profiles["p_r"] + jacobian["g"] * (
        current_density["J^theta"] * magnetic_field["B^zeta"]
        - current_density["J^zeta"] * magnetic_field["B^theta"]
    )
    force_error["F_theta"] = (
        jacobian["g"] * current_density["J^rho"] * magnetic_field["B^zeta"]
    )
    force_error["F_zeta"] = -profiles["iota"] * force_error["F_theta"]
    force_error["F_beta"] = force_error["F_theta"]

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
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute force error magnitude.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
        p_profile,
        i_profile,
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

    force_error["F"] = (
        force_error["F_rho"] * con_basis["e^rho"]
        + force_error["F_theta"] * con_basis["e^theta"]
        + force_error["F_zeta"] * con_basis["e^zeta"]
    )
    force_error["beta"] = con_basis["e^theta"] - profiles["iota"] * con_basis["e^zeta"]

    force_error["|grad(rho)|"] = jnp.sqrt(
        dot(con_basis["e^rho"], con_basis["e^rho"], 0)
    )
    force_error["|grad(p)|"] = jnp.sqrt(
        profiles["p_r"] ** 2 * force_error["|grad(rho)|"] ** 2
    )
    force_error["|beta|"] = jnp.sqrt(
        dot(con_basis["e^theta"], con_basis["e^theta"], 0)
        + profiles["iota"] ** 2 * dot(con_basis["e^zeta"], con_basis["e^zeta"], 0)
        - 2 * profiles["iota"] * dot(con_basis["e^theta"], con_basis["e^zeta"], 0)
    )

    force_error["|F|"] = jnp.sqrt(
        force_error["F_rho"] ** 2 * force_error["|grad(rho)|"] ** 2
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
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute MHD energy by quadrature sum.
    :math:`\int_V dV(\\frac{B^2}{2\mu_0} + \\frac{p}{\gamma - 1})`
    where DESC assumes :math:`\gamma=0`
    **REQUIRES 'quad' grid for correct results**

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

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
        p_profile,
        i_profile,
    )

    NFP = R_transform.grid.NFP
    weights = R_transform.grid.weights

    energy = {}
    energy["W_p"] = -jnp.sum(profiles["p"] * jnp.abs(jacobian["g"]) * weights) * NFP
    energy["W_B"] = (
        jnp.sum(magnetic_field["|B|"] ** 2 * jnp.abs(jacobian["g"]) * weights)
        / (2 * mu_0)
        * NFP
    )
    energy["W"] = energy["W_B"] + energy["W_p"]
    energy["beta"] = jnp.abs(energy["W_p"] / energy["W_B"])

    return (energy, magnetic_field, jacobian, cov_basis, toroidal_coords, profiles)


def compute_quasisymmetry(
    Psi,
    R_lmn,
    Z_lmn,
    L_lmn,
    p_l,
    i_l,
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
):
    """Compute quasisymmetry metrics.

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
    p_profile : Profile
        transforms p_l coefficients to real space
    i_profile : Profile
        transforms i_l coefficients to real space

    Returns
    -------
    quasisymmetry: dict
        dictionary of ndarray, shape(num_nodes,), of quasisymmetry components.
        The triple product metric has the key 'QS_TP',
        and the flux function metric has the key 'QS_FF'.
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
        p_profile,
        i_profile,
    )

    axis = i_profile.grid.axis

    # toroidal coordinate 2nd derivatives
    toroidal_coords["R_rtt"] = R_transform.transform(R_lmn, 1, 2, 0)
    toroidal_coords["Z_rtt"] = Z_transform.transform(Z_lmn, 1, 2, 0)
    toroidal_coords["R_rtz"] = R_transform.transform(R_lmn, 1, 1, 1)
    toroidal_coords["Z_rtz"] = Z_transform.transform(Z_lmn, 1, 1, 1)
    toroidal_coords["R_rzz"] = R_transform.transform(R_lmn, 1, 0, 2)
    toroidal_coords["Z_rzz"] = Z_transform.transform(Z_lmn, 1, 0, 2)
    toroidal_coords["R_ttt"] = R_transform.transform(R_lmn, 0, 3, 0)
    toroidal_coords["Z_ttt"] = Z_transform.transform(Z_lmn, 0, 3, 0)
    toroidal_coords["R_ttz"] = R_transform.transform(R_lmn, 0, 2, 1)
    toroidal_coords["Z_ttz"] = Z_transform.transform(Z_lmn, 0, 2, 1)
    toroidal_coords["R_tzz"] = R_transform.transform(R_lmn, 0, 1, 2)
    toroidal_coords["Z_tzz"] = Z_transform.transform(Z_lmn, 0, 1, 2)
    toroidal_coords["R_zzz"] = R_transform.transform(R_lmn, 0, 0, 3)
    toroidal_coords["Z_zzz"] = Z_transform.transform(Z_lmn, 0, 0, 3)

    # lambda derivatives
    toroidal_coords["lambda_ttt"] = L_transform.transform(L_lmn, 0, 3, 0)
    toroidal_coords["lambda_ttz"] = L_transform.transform(L_lmn, 0, 2, 1)
    toroidal_coords["lambda_tzz"] = L_transform.transform(L_lmn, 0, 1, 2)
    toroidal_coords["lambda_zzz"] = L_transform.transform(L_lmn, 0, 0, 3)

    # covariant basis derivatives
    cov_basis["e_rho_tt"] = jnp.array(
        [toroidal_coords["R_rtt"], toroidal_coords["0"], toroidal_coords["Z_rtt"]]
    )
    cov_basis["e_theta_tt"] = jnp.array(
        [toroidal_coords["R_ttt"], toroidal_coords["0"], toroidal_coords["Z_ttt"]]
    )
    cov_basis["e_zeta_tt"] = jnp.array(
        [toroidal_coords["R_ttz"], toroidal_coords["R_tt"], toroidal_coords["Z_ttz"]]
    )
    cov_basis["e_rho_zz"] = jnp.array(
        [toroidal_coords["R_rzz"], toroidal_coords["0"], toroidal_coords["Z_rzz"]]
    )
    cov_basis["e_theta_zz"] = jnp.array(
        [toroidal_coords["R_tzz"], toroidal_coords["0"], toroidal_coords["Z_tzz"]]
    )
    cov_basis["e_zeta_zz"] = jnp.array(
        [toroidal_coords["R_zzz"], toroidal_coords["R_zz"], toroidal_coords["Z_zzz"]]
    )
    cov_basis["e_rho_tz"] = jnp.array(
        [toroidal_coords["R_rtz"], toroidal_coords["0"], toroidal_coords["Z_rtz"]]
    )
    cov_basis["e_theta_tz"] = jnp.array(
        [toroidal_coords["R_ttz"], toroidal_coords["0"], toroidal_coords["Z_ttz"]]
    )
    cov_basis["e_zeta_tz"] = jnp.array(
        [toroidal_coords["R_tzz"], toroidal_coords["R_tz"], toroidal_coords["Z_tzz"]]
    )

    # jacobian derivatives
    jacobian["g_tt"] = (
        dot(
            cov_basis["e_rho_tt"],
            cross(cov_basis["e_theta"], cov_basis["e_zeta"], 0),
            0,
        )
        + dot(
            cov_basis["e_rho"],
            cross(cov_basis["e_theta_tt"], cov_basis["e_zeta"], 0),
            0,
        )
        + dot(
            cov_basis["e_rho"],
            cross(cov_basis["e_theta"], cov_basis["e_zeta_tt"], 0),
            0,
        )
        + 2
        * dot(
            cov_basis["e_rho_t"],
            cross(cov_basis["e_theta_t"], cov_basis["e_zeta"], 0),
            0,
        )
        + 2
        * dot(
            cov_basis["e_rho_t"],
            cross(cov_basis["e_theta"], cov_basis["e_zeta_t"], 0),
            0,
        )
        + 2
        * dot(
            cov_basis["e_rho"],
            cross(cov_basis["e_theta_t"], cov_basis["e_zeta_t"], 0),
            0,
        )
    )
    jacobian["g_zz"] = (
        dot(
            cov_basis["e_rho_zz"],
            cross(cov_basis["e_theta"], cov_basis["e_zeta"], 0),
            0,
        )
        + dot(
            cov_basis["e_rho"],
            cross(cov_basis["e_theta_zz"], cov_basis["e_zeta"], 0),
            0,
        )
        + dot(
            cov_basis["e_rho"],
            cross(cov_basis["e_theta"], cov_basis["e_zeta_zz"], 0),
            0,
        )
        + 2
        * dot(
            cov_basis["e_rho_z"],
            cross(cov_basis["e_theta_z"], cov_basis["e_zeta"], 0),
            0,
        )
        + 2
        * dot(
            cov_basis["e_rho_z"],
            cross(cov_basis["e_theta"], cov_basis["e_zeta_z"], 0),
            0,
        )
        + 2
        * dot(
            cov_basis["e_rho"],
            cross(cov_basis["e_theta_z"], cov_basis["e_zeta_z"], 0),
            0,
        )
    )
    jacobian["g_tz"] = (
        dot(
            cov_basis["e_rho_tz"],
            cross(cov_basis["e_theta"], cov_basis["e_zeta"], 0),
            0,
        )
        + dot(
            cov_basis["e_rho_z"],
            cross(cov_basis["e_theta_t"], cov_basis["e_zeta"], 0),
            0,
        )
        + dot(
            cov_basis["e_rho_z"],
            cross(cov_basis["e_theta"], cov_basis["e_zeta_t"], 0),
            0,
        )
        + dot(
            cov_basis["e_rho_t"],
            cross(cov_basis["e_theta_z"], cov_basis["e_zeta"], 0),
            0,
        )
        + dot(
            cov_basis["e_rho"],
            cross(cov_basis["e_theta_tz"], cov_basis["e_zeta"], 0),
            0,
        )
        + dot(
            cov_basis["e_rho"],
            cross(cov_basis["e_theta_z"], cov_basis["e_zeta_t"], 0),
            0,
        )
        + dot(
            cov_basis["e_rho_t"],
            cross(cov_basis["e_theta"], cov_basis["e_zeta_z"], 0),
            0,
        )
        + dot(
            cov_basis["e_rho"],
            cross(cov_basis["e_theta_t"], cov_basis["e_zeta_z"], 0),
            0,
        )
        + dot(
            cov_basis["e_rho"],
            cross(cov_basis["e_theta"], cov_basis["e_zeta_tz"], 0),
            0,
        )
    )

    # B covariant component derivatives
    magnetic_field["B_theta_t"] = dot(
        magnetic_field["B_t"], cov_basis["e_theta"], 0
    ) + dot(magnetic_field["B"], cov_basis["e_theta_t"], 0)
    magnetic_field["B_zeta_z"] = dot(
        magnetic_field["B_z"], cov_basis["e_zeta"], 0
    ) + dot(magnetic_field["B"], cov_basis["e_zeta_z"], 0)

    # B contravariant component derivatives
    magnetic_field["B0_tt"] = -(
        profiles["psi_r"]
        / jacobian["g"] ** 2
        * (jacobian["g_tt"] - 2 * jacobian["g_t"] ** 2 / jacobian["g"])
    )
    magnetic_field["B0_zz"] = -(
        profiles["psi_r"]
        / jacobian["g"] ** 2
        * (jacobian["g_zz"] - 2 * jacobian["g_z"] ** 2 / jacobian["g"])
    )
    magnetic_field["B0_tz"] = -(
        profiles["psi_r"]
        / jacobian["g"] ** 2
        * (jacobian["g_tz"] - 2 * jacobian["g_t"] * jacobian["g_z"] / jacobian["g"])
    )
    magnetic_field["B^theta_tt"] = magnetic_field["B0_tt"] * (
        profiles["iota"] - toroidal_coords["lambda_z"]
    )
    -2 * magnetic_field["B0_t"] * toroidal_coords["lambda_tz"]
    -magnetic_field["B0"] * toroidal_coords["lambda_ttz"]
    magnetic_field["B^theta_zz"] = magnetic_field["B0_zz"] * (
        profiles["iota"] - toroidal_coords["lambda_z"]
    )
    -2 * magnetic_field["B0_z"] * toroidal_coords["lambda_zz"]
    -magnetic_field["B0"] * toroidal_coords["lambda_zzz"]
    magnetic_field["B^theta_tz"] = magnetic_field["B0_tz"] * (
        profiles["iota"] - toroidal_coords["lambda_z"]
    )
    -magnetic_field["B0_t"] * toroidal_coords["lambda_zz"]
    -magnetic_field["B0_z"] * toroidal_coords["lambda_tz"]
    -magnetic_field["B0"] * toroidal_coords["lambda_tzz"]
    magnetic_field["B^zeta_tt"] = magnetic_field["B0_tt"] * (
        1 + toroidal_coords["lambda_t"]
    )
    +2 * magnetic_field["B0_t"] * toroidal_coords["lambda_tt"]
    +magnetic_field["B0"] * toroidal_coords["lambda_ttt"]
    magnetic_field["B^zeta_zz"] = magnetic_field["B0_zz"] * (
        1 + toroidal_coords["lambda_t"]
    )
    +2 * magnetic_field["B0_z"] * toroidal_coords["lambda_tz"]
    +magnetic_field["B0"] * toroidal_coords["lambda_tzz"]
    magnetic_field["B^zeta_tz"] = magnetic_field["B0_tz"] * (
        1 + toroidal_coords["lambda_t"]
    )
    +magnetic_field["B0_t"] * toroidal_coords["lambda_tz"]
    +magnetic_field["B0_z"] * toroidal_coords["lambda_tt"]
    +magnetic_field["B0"] * toroidal_coords["lambda_ttz"]

    # |B|
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

    # |B| derivatives
    magnetic_field["|B|_t"] = (
        magnetic_field["B^theta"]
        * (
            magnetic_field["B^zeta_t"]
            * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^theta_t"]
            * dot(cov_basis["e_theta"], cov_basis["e_theta"], 0)
            + magnetic_field["B^theta"]
            * dot(cov_basis["e_theta_t"], cov_basis["e_theta"], 0)
        )
        + magnetic_field["B^zeta"]
        * (
            magnetic_field["B^theta_t"]
            * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta_t"]
            * dot(cov_basis["e_zeta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta"]
            * dot(cov_basis["e_zeta_t"], cov_basis["e_zeta"], 0)
        )
        + magnetic_field["B^theta"]
        * magnetic_field["B^zeta"]
        * (
            dot(cov_basis["e_theta_t"], cov_basis["e_zeta"], 0)
            + dot(cov_basis["e_zeta_t"], cov_basis["e_theta"], 0)
        )
    ) / magnetic_field["|B|"]
    magnetic_field["|B|_z"] = (
        magnetic_field["B^theta"]
        * (
            magnetic_field["B^zeta_z"]
            * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^theta_z"]
            * dot(cov_basis["e_theta"], cov_basis["e_theta"], 0)
            + magnetic_field["B^theta"]
            * dot(cov_basis["e_theta_z"], cov_basis["e_theta"], 0)
        )
        + magnetic_field["B^zeta"]
        * (
            magnetic_field["B^theta_z"]
            * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta_z"]
            * dot(cov_basis["e_zeta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta"]
            * dot(cov_basis["e_zeta_z"], cov_basis["e_zeta"], 0)
        )
        + magnetic_field["B^theta"]
        * magnetic_field["B^zeta"]
        * (
            dot(cov_basis["e_theta_z"], cov_basis["e_zeta"], 0)
            + dot(cov_basis["e_zeta_z"], cov_basis["e_theta"], 0)
        )
    ) / magnetic_field["|B|"]
    magnetic_field["|B|_tt"] = (
        magnetic_field["B^theta_t"]
        * (
            magnetic_field["B^zeta_t"]
            * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^theta_t"]
            * dot(cov_basis["e_theta"], cov_basis["e_theta"], 0)
            + magnetic_field["B^theta"]
            * dot(cov_basis["e_theta_t"], cov_basis["e_theta"], 0)
        )
        + magnetic_field["B^theta"]
        * (
            magnetic_field["B^zeta_tt"]
            * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^theta_tt"]
            * dot(cov_basis["e_theta"], cov_basis["e_theta"], 0)
            + magnetic_field["B^theta_t"]
            * dot(cov_basis["e_theta_t"], cov_basis["e_theta"], 0)
        )
        + magnetic_field["B^theta"]
        * (
            magnetic_field["B^zeta_t"]
            * (
                dot(cov_basis["e_theta_t"], cov_basis["e_zeta"], 0)
                + dot(cov_basis["e_theta"], cov_basis["e_zeta_t"], 0)
            )
            + 2
            * magnetic_field["B^theta_t"]
            * dot(cov_basis["e_theta_t"], cov_basis["e_theta"], 0)
            + magnetic_field["B^theta"]
            * (
                dot(cov_basis["e_theta_tt"], cov_basis["e_theta"], 0)
                + dot(cov_basis["e_theta_t"], cov_basis["e_theta_t"], 0)
            )
        )
        + magnetic_field["B^zeta_t"]
        * (
            magnetic_field["B^theta_t"]
            * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta_t"]
            * dot(cov_basis["e_zeta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta"]
            * dot(cov_basis["e_zeta_t"], cov_basis["e_zeta"], 0)
        )
        + magnetic_field["B^zeta"]
        * (
            magnetic_field["B^theta_tt"]
            * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta_tt"]
            * dot(cov_basis["e_zeta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta_t"]
            * dot(cov_basis["e_zeta_t"], cov_basis["e_zeta"], 0)
        )
        + magnetic_field["B^zeta"]
        * (
            magnetic_field["B^theta_t"]
            * (
                dot(cov_basis["e_theta_t"], cov_basis["e_zeta"], 0)
                + dot(cov_basis["e_theta"], cov_basis["e_zeta_t"], 0)
            )
            + 2
            * magnetic_field["B^zeta_t"]
            * dot(cov_basis["e_zeta_t"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta"]
            * (
                dot(cov_basis["e_zeta_tt"], cov_basis["e_zeta"], 0)
                + dot(cov_basis["e_zeta_t"], cov_basis["e_zeta_t"], 0)
            )
        )
        + (
            magnetic_field["B^theta_t"] * magnetic_field["B^zeta"]
            + magnetic_field["B^theta"] * magnetic_field["B^zeta_t"]
        )
        * (
            dot(cov_basis["e_theta_t"], cov_basis["e_zeta"], 0)
            + dot(cov_basis["e_zeta_t"], cov_basis["e_theta"], 0)
        )
        + magnetic_field["B^theta"]
        * magnetic_field["B^zeta"]
        * (
            dot(cov_basis["e_theta_tt"], cov_basis["e_zeta"], 0)
            + dot(cov_basis["e_zeta_tt"], cov_basis["e_theta"], 0)
            + 2 * dot(cov_basis["e_zeta_t"], cov_basis["e_theta_t"], 0)
        )
    ) / magnetic_field["|B|"] - magnetic_field["|B|_t"] ** 2 / magnetic_field["|B|"]
    magnetic_field["|B|_zz"] = (
        magnetic_field["B^theta_z"]
        * (
            magnetic_field["B^zeta_z"]
            * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^theta_z"]
            * dot(cov_basis["e_theta"], cov_basis["e_theta"], 0)
            + magnetic_field["B^theta"]
            * dot(cov_basis["e_theta_z"], cov_basis["e_theta"], 0)
        )
        + magnetic_field["B^theta"]
        * (
            magnetic_field["B^zeta_zz"]
            * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^theta_zz"]
            * dot(cov_basis["e_theta"], cov_basis["e_theta"], 0)
            + magnetic_field["B^theta_z"]
            * dot(cov_basis["e_theta_z"], cov_basis["e_theta"], 0)
        )
        + magnetic_field["B^theta"]
        * (
            magnetic_field["B^zeta_z"]
            * (
                dot(cov_basis["e_theta_z"], cov_basis["e_zeta"], 0)
                + dot(cov_basis["e_theta"], cov_basis["e_zeta_z"], 0)
            )
            + 2
            * magnetic_field["B^theta_z"]
            * dot(cov_basis["e_theta_z"], cov_basis["e_theta"], 0)
            + magnetic_field["B^theta"]
            * (
                dot(cov_basis["e_theta_zz"], cov_basis["e_theta"], 0)
                + dot(cov_basis["e_theta_z"], cov_basis["e_theta_z"], 0)
            )
        )
        + magnetic_field["B^zeta_z"]
        * (
            magnetic_field["B^theta_z"]
            * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta_z"]
            * dot(cov_basis["e_zeta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta"]
            * dot(cov_basis["e_zeta_z"], cov_basis["e_zeta"], 0)
        )
        + magnetic_field["B^zeta"]
        * (
            magnetic_field["B^theta_zz"]
            * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta_zz"]
            * dot(cov_basis["e_zeta"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta_z"]
            * dot(cov_basis["e_zeta_z"], cov_basis["e_zeta"], 0)
        )
        + magnetic_field["B^zeta"]
        * (
            magnetic_field["B^theta_z"]
            * (
                dot(cov_basis["e_theta_z"], cov_basis["e_zeta"], 0)
                + dot(cov_basis["e_theta"], cov_basis["e_zeta_z"], 0)
            )
            + 2
            * magnetic_field["B^zeta_z"]
            * dot(cov_basis["e_zeta_z"], cov_basis["e_zeta"], 0)
            + magnetic_field["B^zeta"]
            * (
                dot(cov_basis["e_zeta_zz"], cov_basis["e_zeta"], 0)
                + dot(cov_basis["e_zeta_z"], cov_basis["e_zeta_z"], 0)
            )
        )
        + (
            magnetic_field["B^theta_z"] * magnetic_field["B^zeta"]
            + magnetic_field["B^theta"] * magnetic_field["B^zeta_z"]
        )
        * (
            dot(cov_basis["e_theta_z"], cov_basis["e_zeta"], 0)
            + dot(cov_basis["e_zeta_z"], cov_basis["e_theta"], 0)
        )
        + magnetic_field["B^theta"]
        * magnetic_field["B^zeta"]
        * (
            dot(cov_basis["e_theta_zz"], cov_basis["e_zeta"], 0)
            + dot(cov_basis["e_zeta_zz"], cov_basis["e_theta"], 0)
            + 2 * dot(cov_basis["e_theta_z"], cov_basis["e_zeta_z"], 0)
        )
    ) / magnetic_field["|B|"] - magnetic_field["|B|_z"] ** 2 / magnetic_field["|B|"]
    magnetic_field["|B|_tz"] = (
        (
            magnetic_field["B^theta_z"]
            * (
                magnetic_field["B^zeta_t"]
                * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
                + magnetic_field["B^theta_t"]
                * dot(cov_basis["e_theta"], cov_basis["e_theta"], 0)
                + magnetic_field["B^theta"]
                * dot(cov_basis["e_theta_t"], cov_basis["e_theta"], 0)
            )
            + magnetic_field["B^theta"]
            * (
                magnetic_field["B^zeta_tz"]
                * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
                + magnetic_field["B^theta_tz"]
                * dot(cov_basis["e_theta"], cov_basis["e_theta"], 0)
                + magnetic_field["B^theta_z"]
                * dot(cov_basis["e_theta_t"], cov_basis["e_theta"], 0)
            )
            + magnetic_field["B^theta"]
            * (
                magnetic_field["B^zeta_t"]
                * (
                    dot(cov_basis["e_theta_z"], cov_basis["e_zeta"], 0)
                    + dot(cov_basis["e_theta"], cov_basis["e_zeta_z"], 0)
                )
                + 2
                * magnetic_field["B^theta_t"]
                * dot(cov_basis["e_theta_z"], cov_basis["e_theta"], 0)
                + magnetic_field["B^theta"]
                * (
                    dot(cov_basis["e_theta_tz"], cov_basis["e_theta"], 0)
                    + dot(cov_basis["e_theta_t"], cov_basis["e_theta_z"], 0)
                )
            )
            + magnetic_field["B^zeta_z"]
            * (
                magnetic_field["B^theta_t"]
                * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
                + magnetic_field["B^zeta_t"]
                * dot(cov_basis["e_zeta"], cov_basis["e_zeta"], 0)
                + magnetic_field["B^zeta"]
                * dot(cov_basis["e_zeta_t"], cov_basis["e_zeta"], 0)
            )
            + magnetic_field["B^zeta"]
            * (
                magnetic_field["B^theta_tz"]
                * dot(cov_basis["e_theta"], cov_basis["e_zeta"], 0)
                + magnetic_field["B^zeta_tz"]
                * dot(cov_basis["e_zeta"], cov_basis["e_zeta"], 0)
                + magnetic_field["B^zeta_z"]
                * dot(cov_basis["e_zeta_t"], cov_basis["e_zeta"], 0)
            )
            + magnetic_field["B^zeta"]
            * (
                magnetic_field["B^theta_t"]
                * (
                    dot(cov_basis["e_theta_z"], cov_basis["e_zeta"], 0)
                    + dot(cov_basis["e_theta"], cov_basis["e_zeta_z"], 0)
                )
                + 2
                * magnetic_field["B^zeta_t"]
                * dot(cov_basis["e_zeta_z"], cov_basis["e_zeta"], 0)
                + magnetic_field["B^zeta"]
                * (
                    dot(cov_basis["e_zeta_tz"], cov_basis["e_zeta"], 0)
                    + dot(cov_basis["e_zeta_t"], cov_basis["e_zeta_z"], 0)
                )
            )
            + (
                magnetic_field["B^theta_z"] * magnetic_field["B^zeta"]
                + magnetic_field["B^theta"] * magnetic_field["B^zeta_z"]
            )
            * (
                dot(cov_basis["e_theta_t"], cov_basis["e_zeta"], 0)
                + dot(cov_basis["e_zeta_t"], cov_basis["e_theta"], 0)
            )
            + magnetic_field["B^theta"]
            * magnetic_field["B^zeta"]
            * (
                dot(cov_basis["e_theta_tz"], cov_basis["e_zeta"], 0)
                + dot(cov_basis["e_zeta_tz"], cov_basis["e_theta"], 0)
                + dot(cov_basis["e_theta_t"], cov_basis["e_zeta_z"], 0)
                + dot(cov_basis["e_zeta_t"], cov_basis["e_theta_z"], 0)
            )
        )
        / magnetic_field["|B|"]
        - magnetic_field["|B|_t"] * magnetic_field["|B|_z"] / magnetic_field["|B|"]
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

    quasisymmetry = {}

    quasisymmetry["|grad(psi)|"] = jnp.sqrt(
        profiles["psi_r"] ** 2 * dot(con_basis["e^rho"], con_basis["e^rho"], 0)
    )

    # B * grad(|B|) and derivatives
    quasisymmetry["B*grad(|B|)"] = (
        magnetic_field["B^theta"] * magnetic_field["|B|_t"]
        + magnetic_field["B^zeta"] * magnetic_field["|B|_z"]
    )
    quasisymmetry["B*grad(|B|)_t"] = (
        magnetic_field["B^theta_t"] * magnetic_field["|B|_t"]
        + magnetic_field["B^zeta_t"] * magnetic_field["|B|_z"]
        + magnetic_field["B^theta"] * magnetic_field["|B|_tt"]
        + magnetic_field["B^zeta"] * magnetic_field["|B|_tz"]
    )
    quasisymmetry["B*grad(|B|)_z"] = (
        magnetic_field["B^theta_z"] * magnetic_field["|B|_t"]
        + magnetic_field["B^zeta_z"] * magnetic_field["|B|_z"]
        + magnetic_field["B^theta"] * magnetic_field["|B|_tz"]
        + magnetic_field["B^zeta"] * magnetic_field["|B|_zz"]
    )

    return (
        quasisymmetry,
        current_density,
        magnetic_field,
        con_basis,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    )
