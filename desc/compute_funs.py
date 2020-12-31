import numpy as np

from desc.backend import jnp, put
from desc.utils import opsindex
from desc.utils import dot, cross
from desc.transform import Transform


def compute_polar_coords(R0_n, Z0_n, r_lmn, l_lmn,
                         R0_transform:Transform, Z0_transform:Transform,
                         r_transform:Transform, l_transform:Transform):
    """Transforms spectral coefficients of polar coordinates to real space.

    Parameters
    ----------
    R0_n : ndarray
        spectral coefficients of R0
    Z0_n : ndarray
        spectral coefficients of Z0
    r_lmn : ndarray
        spectral coefficients of r
    l_lmn : ndarray
        spectral coefficients of lambda
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space

    Returns
    -------
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates
        evaluated at grid nodes.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    polar_coords = {}

    polar_coords['rho']   = R0_transform.grid.nodes[:, 0]
    polar_coords['theta'] = R0_transform.grid.nodes[:, 1]
    polar_coords['zeta']  = R0_transform.grid.nodes[:, 2]
    polar_coords['0'] = jnp.zeros_like(polar_coords['rho'])

    polar_coords['R0'] = R0_transform.transfrom(R0_n)
    polar_coords['Z0'] = Z0_transform.transfrom(Z0_n)
    polar_coords['r'] = r_transform.transform(r_lmn)
    polar_coords['lambda'] = l_transform.transform(l_lmn)

    return polar_coords


def compute_polar_coords_force(R0_n, Z0_n, r_lmn, l_lmn,
                               R0_transform:Transform, Z0_transform:Transform,
                               r_transform:Transform, l_transform:Transform):
    """Transforms spectral coefficients of polar coordinates to real space.
    Also computes partial derivatives needed for force balance without the
    magnetic axis.

    Parameters
    ----------
    R0_n : ndarray
        spectral coefficients of R0
    Z0_n : ndarray
        spectral coefficients of Z0
    r_lmn : ndarray
        spectral coefficients of r
    l_lmn : ndarray
        spectral coefficients of lambda
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space

    Returns
    -------
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates
        evaluated at grid nodes.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    polar_coords = {}
    polar_coords['rho']   = R0_transform.grid.nodes[:, 0]
    polar_coords['theta'] = R0_transform.grid.nodes[:, 1]
    polar_coords['0'] = jnp.zeros_like(polar_coords['rho'])

    polar_coords['R0'] = R0_transform.transfrom(R0_n)
    polar_coords['Z0'] = Z0_transform.transfrom(Z0_n)
    polar_coords['r'] = r_transform.transform(r_lmn)
    polar_coords['lambda'] = l_transform.transform(l_lmn)

    polar_coords['R0_z'] = R0_transform.transfrom(R0_n, 0, 0, 1)
    polar_coords['Z0_z'] = R0_transform.transfrom(Z0_n, 0, 0, 1)

    polar_coords['R0_zz'] = R0_transform.transfrom(R0_n, 0, 0, 2)
    polar_coords['Z0_zz'] = R0_transform.transfrom(Z0_n, 0, 0, 2)

    polar_coords['r_r'] = r_transform.transform(r_lmn, 1, 0, 0)
    polar_coords['r_t'] = r_transform.transform(r_lmn, 0, 1, 0)
    polar_coords['r_z'] = r_transform.transform(r_lmn, 0, 0, 1)

    polar_coords['r_rr'] = r_transform.transform(r_lmn, 2, 0, 0)
    polar_coords['r_rt'] = r_transform.transform(r_lmn, 1, 1, 0)
    polar_coords['r_rz'] = r_transform.transform(r_lmn, 1, 0, 1)
    polar_coords['r_tt'] = r_transform.transform(r_lmn, 0, 2, 0)
    polar_coords['r_tz'] = r_transform.transform(r_lmn, 0, 1, 1)
    polar_coords['r_zz'] = r_transform.transform(r_lmn, 0, 0, 2)

    polar_coords['lambda_r'] = l_transform.transform(l_lmn, 1, 0, 0)
    polar_coords['lambda_t'] = l_transform.transform(l_lmn, 0, 1, 0)
    polar_coords['lambda_z'] = l_transform.transform(l_lmn, 0, 0, 1)

    polar_coords['lambda_r'] = l_transform.transform(l_lmn, 1, 0, 0)
    polar_coords['lambda_t'] = l_transform.transform(l_lmn, 0, 1, 0)
    polar_coords['lambda_z'] = l_transform.transform(l_lmn, 0, 0, 1)

    polar_coords['lambda_rt'] = l_transform.transform(l_lmn, 1, 1, 0)
    polar_coords['lambda_rz'] = l_transform.transform(l_lmn, 1, 0, 1)
    polar_coords['lambda_tt'] = l_transform.transform(l_lmn, 0, 2, 0)
    polar_coords['lambda_tz'] = l_transform.transform(l_lmn, 0, 1, 1)
    polar_coords['lambda_zz'] = l_transform.transform(l_lmn, 0, 0, 2)

    return polar_coords


def compute_toroidal_coords(polar_coords):
    """Computes toroidal coordinates from polar coordinates.

    Parameters
    ----------
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates
        evaluated at grid nodes.

    Returns
    -------
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates
        evaluated at grid nodes.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    toroidal_coords = {}
    toroidal_coords['R'] = polar_coords['R0'] + polar_coords['r']*jnp.cos(polar_coords['theta'])
    toroidal_coords['Z'] = polar_coords['Z0'] + polar_coords['r']*jnp.sin(polar_coords['theta'])
    toroidal_coords['phi'] = polar_coords['zeta']   # phi = zeta
    toroidal_coords['X'] = toroidal_coords['R']*np.cos(toroidal_coords['phi'])
    toroidal_coords['Y'] = toroidal_coords['R']*np.sin(toroidal_coords['phi'])
    toroidal_coords['0'] = polar_coords['0']

    return toroidal_coords


def compute_toroidal_coords_force(polar_coords, zeta_ratio=1.0):
    """Computes toroidal coordinates from polar coordinates.
    Also computes partial derivatives needed for force balance without the
    magnetic axis.

    Parameters
    ----------
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates
        evaluated at grid nodes.
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero effectively solves
        for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates
        evaluated at grid nodes.
        Keys are of the form 'X_y' meaning the derivative of X wrt to y.

    """
    # notation: X_y means derivative of X wrt y
    toroidal_coords = {}

    # 0th order
    toroidal_coords['R'] = polar_coords['R0'] + polar_coords['r']*jnp.cos(polar_coords['theta'])
    toroidal_coords['Z'] = polar_coords['Z0'] + polar_coords['r']*jnp.sin(polar_coords['theta'])
    toroidal_coords['0'] = polar_coords['0']

    # 1st order
    toroidal_coords['R_r'] = polar_coords['r_r']*jnp.cos(polar_coords['theta'])
    toroidal_coords['Z_r'] = polar_coords['r_r']*jnp.sin(polar_coords['theta'])
    toroidal_coords['R_v'] = polar_coords['r_t']*jnp.cos(polar_coords['theta']) - polar_coords['r']*jnp.sin(polar_coords['theta'])
    toroidal_coords['Z_v'] = polar_coords['r_t']*jnp.sin(polar_coords['theta']) + polar_coords['r']*jnp.cos(polar_coords['theta'])
    toroidal_coords['R_z'] = polar_coords['R0_z'] + polar_coords['r_z']*jnp.cos(polar_coords['theta'])
    toroidal_coords['Z_z'] = polar_coords['Z0_z'] + polar_coords['r_z']*jnp.sin(polar_coords['theta'])

    # 2nd order
    toroidal_coords['R_rr'] = polar_coords['r_rr']*jnp.cos(polar_coords['theta'])
    toroidal_coords['Z_rr'] = polar_coords['r_rr']*jnp.sin(polar_coords['theta'])
    toroidal_coords['R_rt'] = polar_coords['r_rt']*jnp.cos(polar_coords['theta']) - polar_coords['r_r']*jnp.sin(polar_coords['theta'])
    toroidal_coords['Z_rt'] = polar_coords['r_rt']*jnp.sin(polar_coords['theta']) + polar_coords['r_r']*jnp.cos(polar_coords['theta'])
    toroidal_coords['R_rz'] = polar_coords['r_rz']*jnp.cos(polar_coords['theta'])
    toroidal_coords['Z_rz'] = polar_coords['r_rz']*jnp.sin(polar_coords['theta'])
    toroidal_coords['R_tt'] = polar_coords['r_tt']*jnp.cos(polar_coords['theta']) - 2*polar_coords['r_t']*jnp.sin(polar_coords['theta']) - polar_coords['r']*jnp.cos(polar_coords['theta'])
    toroidal_coords['Z_tt'] = polar_coords['r_tt']*jnp.sin(polar_coords['theta']) + 2*polar_coords['r_t']*jnp.cos(polar_coords['theta']) - polar_coords['r']*jnp.sin(polar_coords['theta'])
    toroidal_coords['R_tz'] = polar_coords['r_tz']*jnp.cos(polar_coords['theta']) - polar_coords['r_z']*jnp.sin(polar_coords['theta'])
    toroidal_coords['Z_tz'] = polar_coords['r_tz']*jnp.sin(polar_coords['theta']) + polar_coords['r_z']*jnp.cos(polar_coords['theta'])
    toroidal_coords['R_zz'] = polar_coords['R0_zz'] + polar_coords['r_zz']*jnp.cos(polar_coords['theta'])
    toroidal_coords['Z_zz'] = polar_coords['Z0_zz'] + polar_coords['r_zz']*jnp.sin(polar_coords['theta'])

    # apply zeta ratio
    toroidal_coords['R_z'] *= zeta_ratio
    toroidal_coords['Z_z'] *= zeta_ratio
    toroidal_coords['R_rz'] *= zeta_ratio
    toroidal_coords['Z_rz'] *= zeta_ratio
    toroidal_coords['R_tz'] *= zeta_ratio
    toroidal_coords['Z_tz'] *= zeta_ratio
    toroidal_coords['R_zz'] *= zeta_ratio
    toroidal_coords['Z_zz'] *= zeta_ratio

    return toroidal_coords


def compute_covariant_basis_force(toroidal_coords):
    """Computes covariant basis vectors needed for force balance without the
    magnetic axis.

    Parameters
    ----------
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates
        evaluated at grid nodes.

    coord_der : dict
        dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations.
        keys are of the form 'X_y' meaning the derivative of X wrt to y

    Returns
    -------
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), containing covariant basis
        vectors and derivatives at grid nodes.
        Keys are of the form 'e_x_y', meaning the covariant basis vector in
        the x direction, differentiated wrt to y.

    """
    # notation: subscript word is direction of unit vector, subscript letters denote partial derivatives
    # eg, e_rho_t is the theta derivative of the covariant basis vector in the rho direction
    cov_basis = {}

    cov_basis['e_rho'] = jnp.array(
        [toroidal_coords['R_r'],  toroidal_coords['0'],   toroidal_coords['Z_r']])
    cov_basis['e_theta'] = jnp.array(
        [toroidal_coords['R_t'],  toroidal_coords['0'],   toroidal_coords['Z_t']])
    cov_basis['e_zeta'] = jnp.array(
        [toroidal_coords['R_z'],  toroidal_coords['R'],   toroidal_coords['Z_z']])

    cov_basis['e_rho_r'] = jnp.array(
        [toroidal_coords['R_rr'], toroidal_coords['0'],   toroidal_coords['Z_rr']])
    cov_basis['e_rho_t'] = jnp.array(
        [toroidal_coords['R_rt'], toroidal_coords['0'],   toroidal_coords['Z_rt']])
    cov_basis['e_rho_z'] = jnp.array(
        [toroidal_coords['R_rz'], toroidal_coords['0'],   toroidal_coords['Z_rz']])

    cov_basis['e_theta_r'] = jnp.array(
        [toroidal_coords['R_rt'], toroidal_coords['0'],   toroidal_coords['Z_rt']])
    cov_basis['e_theta_t'] = jnp.array(
        [toroidal_coords['R_tt'], toroidal_coords['0'],   toroidal_coords['Z_tt']])
    cov_basis['e_theta_z'] = jnp.array(
        [toroidal_coords['R_tz'], toroidal_coords['0'],   toroidal_coords['Z_tz']])

    cov_basis['e_zeta_r'] = jnp.array(
        [toroidal_coords['R_rz'], toroidal_coords['R_r'], toroidal_coords['Z_rz']])
    cov_basis['e_zeta_t'] = jnp.array(
        [toroidal_coords['R_tz'], toroidal_coords['R_t'], toroidal_coords['Z_tz']])
    cov_basis['e_zeta_z'] = jnp.array(
        [toroidal_coords['R_zz'], toroidal_coords['R_z'], toroidal_coords['Z_zz']])

    return cov_basis


def compute_contravariant_basis(toroidal_coords, cov_basis, jacobian):
    """Computes contravariant basis vectors.

    Parameters
    ----------
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates
        evaluated at grid nodes.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), containing covariant basis
        vectors and derivatives at grid nodes.
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate jacobian and
        partial derivatives.

    Returns
    -------
    con_basis : dict
        dictionary of ndarray, shape(3,num_nodes), containing contravariant
        basis vectors and derivatives at grid nodes.
        Keys are of the form 'e^x_y', meaning the contravariant basis vector
        in the x direction, differentiated wrt to y.

    """
    # subscripts (superscripts) denote covariant (contravariant) basis vectors
    con_basis = {}

    # contravariant basis vectors
    con_basis['e^rho'] = cross(
        cov_basis['e_theta'], cov_basis['e_zeta'], 0) / jacobian['g']
    con_basis['e^theta'] = cross(
        cov_basis['e_zeta'], cov_basis['e_rho'], 0) / jacobian['g']
    con_basis['e^zeta'] = jnp.array(
        [toroidal_coords['0'], 1/toroidal_coords['R'], toroidal_coords['0']])

    return con_basis


def compute_jacobian_force(toroidal_coords, cov_basis):
    """Computes coordinate jacobian and derivatives needed for force balance
    without the magnetic axis.

    Parameters
    ----------
    toroidal_coords : dict
        dictionary of ndarray, shape(num_nodes,) of toroidal coordinates
        evaluated at grid nodes.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), containing covariant basis
        vectors and derivatives at grid nodes.

    Returns
    -------
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate jacobian and
        partial derivatives.
        Keys are of the form 'g_x' meaning the x derivative of the coordinate
        jacobian g.

    """
    # notation: subscripts denote partial derivatives
    jacobian = {}

    jacobian['g'] = dot(cov_basis['e_rho'],
                    cross(cov_basis['e_theta'], cov_basis['e_zeta'], 0), 0)
    jacobian['g_r'] = dot(cov_basis['e_rho_r'],
                      cross(cov_basis['e_theta'], cov_basis['e_zeta'], 0), 0) \
                    + dot(cov_basis['e_rho'],
                      cross(cov_basis['e_theta_r'], cov_basis['e_zeta'], 0), 0) \
                    + dot(cov_basis['e_rho'],
                      cross(cov_basis['e_theta'], cov_basis['e_zeta_r'], 0), 0)
    jacobian['g_v'] = dot(cov_basis['e_rho_t'],
                      cross(cov_basis['e_theta'], cov_basis['e_zeta'], 0), 0) \
                    + dot(cov_basis['e_rho'],
                      cross(cov_basis['e_theta_t'], cov_basis['e_zeta'], 0), 0) \
                    + dot(cov_basis['e_rho'],
                      cross(cov_basis['e_theta'], cov_basis['e_zeta_t'], 0), 0)

    return jacobian


def compute_magnetic_field_force(polar_coords, cov_basis, jacobian, Psi, i_l, i_transform:Transform):
    """Computes magnetic field components needed for force balance without the
    magnetic axis.

    Parameters
    ----------
    polar_coords : dict
        dictionary of ndarray, shape(num_nodes,) of polar coordinates
        evaluated at grid nodes.
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), containing covariant basis
        vectors and derivatives at grid nodes.
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate jacobian and
        partial derivatives.
    Psi : float
        total toroidal flux (in Webers) within the last closed flux surface
    i_l : ndarray
        spectral coefficients of iota
    i_transform : Transform
        transforms i_l coefficients to real space

    Returns
    -------
    magnetic_field: dict
        dictionary of ndarray, shape(num_nodes,) of magnetic field components
        and derivatives.
        Keys are of the form 'B_x_y' or 'B^x_y', meaning the covariant (B_x)
        or contravariant (B^x) component of the magnetic field, with the
        derivative wrt to y.

    """
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    magnetic_field = {}

    # rotational transform
    iota = i_transform.transform(i_l, 0)
    iota_r = i_transform.transform(i_l, 1)

    # toroidal flux
    magnetic_field['psi'] = Psi*polar_coords['rho']**2
    magnetic_field['psi_r'] = 2*Psi*polar_coords['rho']
    magnetic_field['psi_rr'] = 2*Psi*jnp.ones_like(polar_coords['rho'])

    # note: B^rho components are ignored since they are all 0

    # contravariant components
    B0 = magnetic_field['psi_r'] / (2*jnp.pi*jacobian['g'])
    magnetic_field['B^theta'] = B0*(iota - polar_coords['lambda_z'])
    magnetic_field['B^zeta']  = B0*(1    + polar_coords['lambda_t'])
    magnetic_field['B_con'] = magnetic_field['B^theta']*cov_basis['e_theta'] \
                            + magnetic_field['B^zeta']* cov_basis['e_zeta']

    # contravariant component derivatives
    B0_r = magnetic_field['psi_rr'] / (2*jnp.pi*jacobian['g']) \
         - (magnetic_field['psi_r']*jacobian['g_r']) / (2*jnp.pi*jacobian['g']**2)
    B0_t = -(magnetic_field['psi_r']*jacobian['g_v']) / (2*jnp.pi*jacobian['g']**2)
    B0_z = -(magnetic_field['psi_r']*jacobian['g_z']) / (2*jnp.pi*jacobian['g']**2)
    magnetic_field['B^theta_r'] = B0_r*(iota - polar_coords['lambda_z']) \
                                + B0*(iota_r - polar_coords['lambda_rz'])
    magnetic_field['B^theta_t'] = B0_t*(iota - polar_coords['lambda_z']) \
                                - B0*polar_coords['lambda_tz']
    magnetic_field['B^theta_z'] = B0_z*(iota - polar_coords['lambda_z']) \
                                - B0*polar_coords['lambda_zz']
    magnetic_field['B^zeta_r']  = B0_r*(1 + polar_coords['lambda_t']) \
                                + B0*polar_coords['lambda_rt']
    magnetic_field['B^zeta_t']  = B0_t*(1 + polar_coords['lambda_t']) \
                                + B0*polar_coords['lambda_tt']
    magnetic_field['B^zeta_z']  = B0_z*(1 + polar_coords['lambda_t']) \
                                + B0*polar_coords['lambda_tz']
    magnetic_field['B_con_r'] = magnetic_field['B^theta_r']*cov_basis['e_theta'] \
                              + magnetic_field['B^theta']*  cov_basis['e_theta_r'] \
                              + magnetic_field['B^zeta_r']* cov_basis['e_zeta'] \
                              + magnetic_field['B^zeta']*   cov_basis['e_zeta_r']
    magnetic_field['B_con_t'] = magnetic_field['B^theta_t']*cov_basis['e_theta'] \
                              + magnetic_field['B^theta']*  cov_basis['e_theta_t'] \
                              + magnetic_field['B^zeta_t']* cov_basis['e_zeta'] \
                              + magnetic_field['B^zeta']*   cov_basis['e_zeta_t']
    magnetic_field['B_con_z'] = magnetic_field['B^theta_z']*cov_basis['e_theta'] \
                              + magnetic_field['B^theta']*  cov_basis['e_theta_z'] \
                              + magnetic_field['B^zeta_z']* cov_basis['e_zeta'] \
                              + magnetic_field['B^zeta']*   cov_basis['e_zeta_z']

    # covariant components
    magnetic_field['B_rho']   = dot(magnetic_field['B_con'], cov_basis['e_rho'], 0)
    magnetic_field['B_theta'] = dot(magnetic_field['B_con'], cov_basis['e_theta'], 0)
    magnetic_field['B_zeta']  = dot(magnetic_field['B_con'], cov_basis['e_zeta'], 0)

    # covariant component derivatives
    magnetic_field['B_rho_t']   = dot(magnetic_field['B_con_t'], cov_basis['e_rho'], 0) \
                                + dot(magnetic_field['B_con'],   cov_basis['e_rho_t'], 0)
    magnetic_field['B_rho_z']   = dot(magnetic_field['B_con_z'], cov_basis['e_rho'], 0) \
                                + dot(magnetic_field['B_con'],   cov_basis['e_rho_z'], 0)
    magnetic_field['B_theta_r'] = dot(magnetic_field['B_con_r'], cov_basis['e_theta'], 0) \
                                + dot(magnetic_field['B_con'],   cov_basis['e_theta_r'], 0)
    magnetic_field['B_theta_z'] = dot(magnetic_field['B_con_z'], cov_basis['e_theta'], 0) \
                                + dot(magnetic_field['B_con'],   cov_basis['e_theta_z'], 0)
    magnetic_field['B_zeta_r']  = dot(magnetic_field['B_con_r'], cov_basis['e_zeta'], 0) \
                                + dot(magnetic_field['B_con'],   cov_basis['e_zeta_r'], 0)
    magnetic_field['B_zeta_t']  = dot(magnetic_field['B_con_t'], cov_basis['e_zeta'], 0) \
                                + dot(magnetic_field['B_con'],   cov_basis['e_zeta_t'], 0)

    return magnetic_field


def compute_plasma_current_force(cov_basis, jacobian, magnetic_field):
    """Computes current density field components needed for force balance
    without the magnetic axis.

    Parameters
    ----------
    cov_basis : dict
        dictionary of ndarray, shape(3,num_nodes), containing covariant basis
        vectors and derivatives at grid nodes.
    jacobian : dict
        dictionary of ndarray, shape(num_nodes,), of coordinate jacobian and
        partial derivatives.
    magnetic_field: dict
        dictionary of ndarray, shape(num_nodes,) of magnetic field components
        and derivatives.

    Returns
    -------
    plasma_current : dict
        dictionary of ndarray, shape(num_nodes,), of plasma current field.
        Keys are of the form 'J^x_y' meaning the contravariant (J^x)
        component of the current, with the derivative wrt to y.

    """
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    plasma_current = {}
    mu0 = 4*jnp.pi*1e-7

    # contravariant components
    plasma_current['J^rho']   = (magnetic_field['B_zeta_t']  - magnetic_field['B_theta_z']) \
                              / (mu0*jacobian['g'])
    plasma_current['J^theta'] = (magnetic_field['B_rho_z']   - magnetic_field['B_zeta_r']) \
                              / (mu0*jacobian['g'])
    plasma_current['J^zeta']  = (magnetic_field['B_theta_r'] - magnetic_field['B_rho_t']) \
                              / (mu0*jacobian['g'])

    plasma_current['J_con'] = plasma_current['J^rho']*  cov_basis['e_rho'] \
                            + plasma_current['J^theta']*cov_basis['e_theta'] \
                            + plasma_current['J^zeta']* cov_basis['e_zeta']

    return plasma_current


def compute_magnetic_field_magnitude(cov_basis, magnetic_field, cI, I_transform:Transform, derivs='force'):
    """Computes magnetic field magnitude at node locations

    Parameters
    ----------
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    magnetic_field : dict
        dictionary of ndarray containing magnetic field and derivatives,
        such as computed by ``compute_magnetic_field``.
    cI : ndarray
        coefficients to pass to rotational transform function
    I_transform : Transform
        object with transform method to go from spectral to physical space with derivatives
    derivs : str
        type of calculation being performed
        ``'force'``: all of the derivatives needed to calculate an
        equilibrium from the force balance equations
        ``'qs'``: all of the derivatives needed to calculate quasi-
        symmetry from the triple-product equation

    Returns
    -------
    magnetic_field_mag : dict
        dictionary of ndarray, shape(N_nodes,) of magnetic field magnitude and derivatives

    """
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    magnetic_field_mag = {}
    iota = I_transform.transform(cI, 0)

    magnetic_field_mag['|B|'] = jnp.abs(magnetic_field['B^zeta'])*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0) +
                                                              2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0) + dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))

    magnetic_field_mag['|B|_v'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_v']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0)) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)))

    magnetic_field_mag['|B|_z'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_z']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)))

    # QS terms
    if derivs == 'qs':

        magnetic_field_mag['|B|_vv'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_vv']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
            + jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_v']*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0)) \
            / jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
            + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_v'], cov_basis['e_theta_v'], 0)+dot(cov_basis['e_theta'], cov_basis['e_theta_vv'], 0))+2*iota*(dot(cov_basis['e_theta_vv'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_vv'], 0)+2*dot(cov_basis['e_theta_v'], cov_basis['e_zeta_v'], 0))+2*(dot(cov_basis['e_zeta_v'], cov_basis['e_zeta_v'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta_vv'], 0))) \
            / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))) \
            + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0))**2 \
            / (2*(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))**(3/2))

        magnetic_field_mag['|B|_zz'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_zz']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
            + jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_z']*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
            / jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
            + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_z'], cov_basis['e_theta_z'], 0)+dot(cov_basis['e_theta'], cov_basis['e_theta_zz'], 0))+2*iota*(dot(cov_basis['e_theta_zz'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_zz'], 0)+2*dot(cov_basis['e_theta_z'], cov_basis['e_zeta_z'], 0))+2*(dot(cov_basis['e_zeta_z'], cov_basis['e_zeta_z'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta_vz'], 0))) \
            / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))) \
            + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0))**2 \
            / (2*(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))**(3/2))

        magnetic_field_mag['|B|_vz'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_vz']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
            + jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_v']*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
            / jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
            + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_z'], cov_basis['e_theta_v'], 0)+dot(cov_basis['e_theta'], cov_basis['e_theta_vz'], 0))+2*iota*(dot(cov_basis['e_theta_vz'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta_v'], cov_basis['e_zeta_z'], 0)+dot(cov_basis['e_theta_z'], cov_basis['e_zeta_v'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_vz'], 0))+2*(dot(cov_basis['e_zeta_z'], cov_basis['e_zeta_v'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta_vz'], 0))) \
            / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))) \
            + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0))*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
            / (2*(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))**(3/2))

    return magnetic_field_mag


def compute_force_magnitude(coord_der, cov_basis, con_basis, jacobian, magnetic_field, plasma_current, cP, P_transform:Transform):
    """Computes force error magnitude at node locations

    Parameters
    ----------
    coord_der : dict
        dictionary of ndarray containing of coordinate
        derivatives evaluated at node locations, such as computed by ``compute_coordinate_derivatives``.
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    con_basis : dict
        dictionary of ndarray containing contravariant basis
        vectors and metric elements at each node, such as computed by ``compute_contravariant_basis``.
    jacobian : dict
        dictionary of ndarray containing coordinate jacobian
        and partial derivatives, such as computed by ``compute_jacobian``.
    magnetic_field : dict
        dictionary of ndarray containing magnetic field and derivatives,
        such as computed by ``compute_magnetic_field``.
    plasma_current : dict
        dictionary of ndarray containing current and derivatives,
        such as computed by ``compute_plasma_current``.
    cP : ndarray
        parameters to pass to pressure function
    Psi_lcfs : float
        total toroidal flux (in Webers) within LCFS
    P_transform : Transform
        object with transform method to go from spectral to physical space with derivatives

    Returns
    -------
    force_mag : dict
        dictionary of ndarray, shape(N_nodes,) of force magnitudes

    """
    force_mag = {}
    mu0 = 4*jnp.pi*1e-7
    axis = P_transform.grid.axis
    pres_r = P_transform.transform(cP, 1)

    # force balance error covariant components
    F_rho = jacobian['g']*(plasma_current['J^theta']*magnetic_field['B^zeta'] -
                           plasma_current['J^zeta']*magnetic_field['B^theta']) - pres_r
    F_theta = jacobian['g']*plasma_current['J^rho']*magnetic_field['B^zeta']
    F_zeta = -jacobian['g']*plasma_current['J^rho']*magnetic_field['B^theta']

    # axis terms
    if len(axis):
        Jsup_theta = (magnetic_field['B_rho_z'] -
                      magnetic_field['B_zeta_r']) / mu0
        Jsup_zeta = (magnetic_field['B_theta_r'] -
                     magnetic_field['B_rho_v']) / mu0
        F_rho = put(F_rho, axis, Jsup_theta[axis]*magnetic_field['B^zeta']
                    [axis] - Jsup_zeta[axis]*magnetic_field['B^theta'][axis])
        grad_theta = cross(cov_basis['e_zeta'], cov_basis['e_rho'], 0)
        gsup_vv = dot(grad_theta, grad_theta, 0)
        gsup_rv = dot(con_basis['e^rho'], grad_theta, 0)
        gsup_vz = dot(grad_theta, con_basis['e^zeta'], 0)
        F_theta = put(
            F_theta, axis, plasma_current['J^rho'][axis]*magnetic_field['B^zeta'][axis])
        F_zeta = put(F_zeta, axis, -plasma_current['J^rho']
                     [axis]*magnetic_field['B^theta'][axis])
        con_basis['g^vv'] = put(con_basis['g^vv'], axis, gsup_vv[axis])
        con_basis['g^rv'] = put(con_basis['g^rv'], axis, gsup_rv[axis])
        con_basis['g^vz'] = put(con_basis['g^vz'], axis, gsup_vz[axis])

    # F_i*F_j*g^ij terms
    Fg_rr = F_rho * F_rho * con_basis['g^rr']
    Fg_vv = F_theta*F_theta*con_basis['g^vv']
    Fg_zz = F_zeta * F_zeta * con_basis['g^zz']
    Fg_rv = F_rho * F_theta*con_basis['g^rv']
    Fg_rz = F_rho * F_zeta * con_basis['g^rz']
    Fg_vz = F_theta*F_zeta * con_basis['g^vz']

    # magnitudes
    force_mag['|F|'] = jnp.sqrt(Fg_rr + Fg_vv + Fg_zz + 2*Fg_rv + 2*Fg_rz + 2*Fg_vz)
    force_mag['|grad(p)|'] = jnp.sqrt(pres_r*pres_r*con_basis['g^rr'])

    return force_mag
