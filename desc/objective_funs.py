import numpy as np
import matplotlib.pyplot
from desc.field_components import compute_coordinate_derivatives, compute_covariant_basis
from desc.field_components import compute_contravariant_basis, compute_jacobian
from desc.field_components import compute_magnetic_field, compute_plasma_current, compute_magnetic_field_magnitude
from desc.boundary_conditions import compute_bdry_err_RZ, compute_bdry_err_four, compute_lambda_err
from desc.zernike import symmetric_x, double_fourier_basis, fourzern
from desc.backend import jnp, put, cross, dot, presfun, iotafun, unpack_x, rms, TextColors


def get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M, N, NFP, zernike_transform, bdry_zernike_transform, zern_idx, lambda_idx, bdryM, bdryN, scalar=False):
    """Gets the equilibrium objective function

    Args:
        stell_sym (bool): True if stellarator symmetry is enforced
        error_mode (string): 'force' or 'accel'
        bdry_mode (string): 'real' or 'spectral'
        M (int): maximum poloidal resolution
        N (int): maximum toroidal resolution
        NFP (int): number of field periods
        zernike_transform (ZernikeTransform): zernike transform object for force balance
        bdry_zernike_transform (ZernikeTransform): zernike transform object for boundary conditions
        zern_idx (ndarray of int, shape(Nc,3)): mode numbers for Zernike basis
        lambda_idx (ndarray of int, shape(Nc,2)): mode numbers for Fourier basis
        bdryM (ndarray of int): poloidal mode numbers for boundary
        bdryN (ndarray of int): toroidal mode numbers for boundary

    Returns:
        equil_obj (function): equilibrium objective function
        callback (function): function that prints equilibrium errors
    """

    # stellarator symmetry
    if stell_sym:
        sym_mat = symmetric_x(zern_idx, lambda_idx)
    else:
        sym_mat = np.eye(2*zern_idx.shape[0] + lambda_idx.shape[0])

    if errr_mode == 'force':
        equil_fun = compute_force_error_nodes
    elif errr_mode == 'accel':
        equil_fun = compute_accel_error_spectral

    if bdry_mode == 'real':
        raise ValueError(TextColors.FAIL + "evaluating bdry error in real space coordinates is currently broken." +
                         " Please yell at one of the developers and we will fix it" + TextColors.ENDC)
        bdry_fun = compute_bdry_err_RZ
    elif bdry_mode == 'spectral':
        bdry_fun = compute_bdry_err_four

    def equil_obj(x, bdryR, bdryZ, cP, cI, Psi_lcfs, bdry_ratio=1.0, pres_ratio=1.0, zeta_ratio=1.0, errr_ratio=1.0):

        cR, cZ, cL = unpack_x(jnp.matmul(sym_mat, x), len(zern_idx))
        errRf, errZf = equil_fun(
            cR, cZ, cP, cI, Psi_lcfs, pres_ratio, zeta_ratio, zernike_transform)
        errRb, errZb = bdry_fun(
            cR, cZ, cL, bdry_ratio, bdry_zernike_transform, lambda_idx, bdryR, bdryZ, bdryM, bdryN, NFP, stell_sym)

        residual = jnp.concatenate([errRf.flatten(),
                                    errZf.flatten(),
                                    errRb.flatten()/errr_ratio,
                                    errZb.flatten()/errr_ratio])

        if not stell_sym:
            errL0 = compute_lambda_err(cL, lambda_idx, NFP)
            residual = jnp.concatenate([residual, errL0.flatten()/errr_ratio])

        if scalar:
            residual = jnp.log1p(jnp.sum(residual**2))
        return residual

    def callback(x, bdryR, bdryZ, cP, cI, Psi_lcfs, bdry_ratio=1.0, pres_ratio=1.0, zeta_ratio=1.0, errr_ratio=1.0):

        cR, cZ, cL = unpack_x(jnp.matmul(sym_mat, x), len(zern_idx))
        errRf, errZf = equil_fun(
            cR, cZ, cP, cI, Psi_lcfs, pres_ratio, zeta_ratio, zernike_transform)
        errRb, errZb = bdry_fun(
            cR, cZ, cL, bdry_ratio, bdry_zernike_transform, lambda_idx, bdryR, bdryZ, bdryM, bdryN, NFP, stell_sym)
        errL0 = compute_lambda_err(cL, lambda_idx, NFP)

        errRf_rms = jnp.sqrt(jnp.sum(errRf**2))
        errZf_rms = jnp.sqrt(jnp.sum(errZf**2))
        errRb_rms = jnp.sqrt(jnp.sum(errRb**2))
        errZb_rms = jnp.sqrt(jnp.sum(errZb**2))
        errL0_rms = jnp.sqrt(jnp.sum(errL0**2))

        residual = jnp.concatenate([errRf.flatten(),
                                    errZf.flatten(),
                                    errRb.flatten()/errr_ratio,
                                    errZb.flatten()/errr_ratio,
                                    errL0.flatten()/errr_ratio])
        resid_rms = 1/2*jnp.sum(residual**2)
        if errr_mode == 'force':
            print('Weighted Loss: {:10.3e}  errFrho: {:10.3e}  errFbeta: {:10.3e}  errRb: {:10.3e}  errZb: {:10.3e}  errL0: {:10.3e}'.format(
                resid_rms, errRf_rms, errZf_rms, errRb_rms, errZb_rms, errL0_rms))
        elif errr_mode == 'accel':
            print('Weighted Loss: {:10.3e}  errRf: {:10.3e}  errZf: {:10.3e}  errRb: {:10.3e}  errZb: {:10.3e}  errL0: {:10.3e}'.format(
                resid_rms, errRf_rms, errZf_rms, errRb_rms, errZb_rms, errL0_rms))

    return equil_obj, callback


def get_qisym_obj_fun(stell_sym, M, N, NFP, zernike_transform, zern_idx, lambda_idx, modes_pol, modes_tor):
    """Gets the quasisymmetry objective function

    Args:
        M (int): maximum poloidal resolution
        N (int): maximum toroidal resolution
        zern_idx (ndarray of int, shape(Nc,3)): mode numbers for Zernike basis
        lambda_idx (ndarray of int, shape(Nc,2)): mode numbers for Fourier basis
        stell_sym (bool): True if stellarator symmetry is enforced

    Returns:
        qsym_obj (function): quasisymmetry objective function
    """

    # stellarator symmetry
    if stell_sym:
        sym_mat = symmetric_x(zern_idx, lambda_idx)
    else:
        sym_mat = np.eye(2*zern_idx.shape[0] + lambda_idx.shape[0])

    def qisym_obj(x, cI, Psi_total):

        cR, cZ, cL = unpack_x(jnp.matmul(sym_mat, x), len(zern_idx))
        errQS = compute_qs_error_spectral(
            cR, cZ, cI, Psi_total, NFP, zernike_transform, modes_pol, modes_tor, 1.0)

        # normalize weighting by numper of nodes
        residual = errQS.flatten()/jnp.sqrt(errQS.size)
        return residual

    return qisym_obj


def curve_self_intersects(x, y):
    """Checks if a curve intersects itself

    Args:
        x,y (ndarray): x and y coordinates of points along the curve

    Returns:
        (bool): whether the curve intersects itself
    """

    pts = np.array([x, y])
    pts1 = pts[:, 0:-1]
    pts2 = pts[:, 1:]

    # [start/stop, x/y, segment]
    segments = np.array([pts1, pts2])
    s1, s2 = np.meshgrid(np.arange(len(x)-1), np.arange(len(y)-1))
    idx = np.array([s1.flatten(), s2.flatten()])
    a, b = segments[:, :, idx[0, :]]
    c, d = segments[:, :, idx[1, :]]

    def signed_2d_area(a, b, c): return (
        a[0] - c[0])*(b[1] - c[1]) - (a[1] - c[1])*(b[0] - c[0])

    # signs of areas correspond to which side of ab points c and d are
    a1 = signed_2d_area(a, b, d)  # Compute winding of abd (+ or -)
    a2 = signed_2d_area(a, b, c)  # To intersect, must have sign opposite of a1
    a3 = signed_2d_area(c, d, a)  # Compute winding of cda (+ or -)
    a4 = a3 + a2 - a1  # Since area is constant a1 - a2 = a3 - a4, or a4 = a3 + a2 - a1

    return np.any(np.where(np.logical_and(a1*a2 < 0, a3*a4 < 0), True, False))


def is_nested(cR, cZ, zern_idx, NFP, nsurfs=10, zeta=0, Nt=361):
    """Checks that an equilibrium has properly nested flux surfaces
    in a given toroidal plane

    Args:
        cR (ndarray, shape(N,)): R coefficients
        cZ (ndarray, shape(N,)): Z coefficients
        zern_idx (ndarray, shape(N,3)): zernike basis mode numbers
        NFP (int): number of field periods
        nsurfs (int): number of surfaces to check
        zeta (float): toroidal plane to check
        Nt (int): number of theta points to use for the test

    Returns:
        (bool): whether or not the surfaces are nested
    """

    surfs = np.linspace(0, 1, nsurfs)[::-1]
    t = np.tile(np.linspace(0, 2*np.pi, Nt), [nsurfs, 1])
    r = surfs[:, np.newaxis]*np.ones_like(t)
    z = zeta*np.ones_like(t)

    bdry_interp = fourzern(r.flatten(), t.flatten(), z.flatten(),
                           zern_idx[:, 0], zern_idx[:, 1], zern_idx[:, 2],
                           NFP, 0, 0, 0)

    Rs = np.matmul(bdry_interp, cR).reshape((nsurfs, -1))
    Zs = np.matmul(bdry_interp, cZ).reshape((nsurfs, -1))

    p = [matplotlib.path.Path(np.stack([R, Z]).T, closed=True)
         for R, Z in zip(Rs, Zs)]
    nested = np.all([p[i].contains_path(p[i+1]) for i in range(len(p)-1)])
    intersects = np.any([curve_self_intersects(R, Z) for R, Z in zip(Rs, Zs)])
    return nested and not intersects


def compute_force_error_nodes(cR, cZ, cP, cI, Psi_total, pres_ratio, zeta_ratio, zernike_transform):
    """Computes force balance error at each node

    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        cP (array-like): parameters to pass to pressure function
        cI (array-like): parameters to pass to rotational transform function
        Psi_lcfs (float): total toroidal flux within LCFS
        pres_ratio (double): fraction in range [0,1] of the full pressure profile to use
        zeta_ratio (double): fraction in range [0,1] of the full toroidal (zeta) derivatives to use
        zernike_transform (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes

    Returns:
        F_rho (ndarray, shape(N_nodes,)): radial force balance error at each node
        F_beta (ndarray, shape(N_nodes,)): helical force balance error at each node
    """

    mu0 = 4*jnp.pi*1e-7
    nodes = zernike_transform.nodes
    volumes = zernike_transform.volumes
    axn = zernike_transform.axn
    r = nodes[0]
    pres_r = presfun(r, 1, cP) * pres_ratio

    # compute fields components
    coord_der = compute_coordinate_derivatives(
        cR, cZ, zernike_transform, zeta_ratio)
    cov_basis = compute_covariant_basis(coord_der, zernike_transform)
    jacobian = compute_jacobian(coord_der, cov_basis, zernike_transform)
    con_basis = compute_contravariant_basis(
        coord_der, cov_basis, jacobian, zernike_transform)
    magnetic_field = compute_magnetic_field(cov_basis, jacobian, cI,
                                            Psi_total, zernike_transform)
    plasma_current = compute_plasma_current(coord_der, cov_basis,
                                            jacobian, magnetic_field, cI, Psi_total, zernike_transform)

    # force balance error components
    F_rho = jacobian['g']*(plasma_current['J^theta']*magnetic_field['B^zeta'] -
                           plasma_current['J^zeta']*magnetic_field['B^theta']) - pres_r
    F_beta = jacobian['g']*plasma_current['J^rho']

    # radial and helical directions
    beta = magnetic_field['B^zeta']*con_basis['e^theta'] - \
        magnetic_field['B^theta']*con_basis['e^zeta']
    radial = jnp.sqrt(
        con_basis['g^rr']) * jnp.sign(dot(con_basis['e^rho'], cov_basis['e_rho'], 0))
    helical = jnp.sqrt(con_basis['g^vv']*magnetic_field['B^zeta']**2 + con_basis['g^zz']*magnetic_field['B^theta']**2 - 2*con_basis['g^vz']*magnetic_field['B^theta']*magnetic_field['B^zeta']) \
        * jnp.sign(dot(beta, cov_basis['e_theta'], 0)) * jnp.sign(dot(beta, cov_basis['e_zeta'], 0))

    # axis terms
    if len(axn):
        Jsup_theta = (magnetic_field['B_rho_z'] -
                      magnetic_field['B_zeta_r']) / mu0
        Jsup_zeta = (magnetic_field['B_theta_r'] -
                     magnetic_field['B_rho_v']) / mu0
        F_rho = put(F_rho, axn, Jsup_theta[axn]*magnetic_field['B^zeta']
                    [axn] - Jsup_zeta[axn]*magnetic_field['B^theta'][axn])
        grad_theta = cross(cov_basis['e_zeta'], cov_basis['e_rho'], 0)
        gsup_vv = dot(grad_theta, grad_theta, 0)
        F_beta = put(F_beta, axn, plasma_current['J^rho'][axn])
        helical = put(helical, axn, jnp.sqrt(
            gsup_vv[axn]*magnetic_field['B^zeta'][axn]**2) * jnp.sign(magnetic_field['B^zeta'][axn]))

    # scalar errors
    f_rho = F_rho * radial
    f_beta = F_beta*helical

    # weight by local volume
    vol = jacobian['g']*volumes[0]*volumes[1]*volumes[2]
    if len(axn):
        r1 = jnp.min(r[r != 0])  # value of r one step out from axis
        r1_g = jnp.where(r == r1,jacobian['g'],0)
        cnt = jnp.count_nonzero(r1_g)
        # volume of axis is zero, but we want to account for nonzero volume in cell around axis
        vol = put(vol, axn, jnp.sum(r1_g/2*volumes[0,:]*volumes[1,:]*volumes[2,:])/cnt)
    f_rho = f_rho * vol
    f_beta = f_beta*vol

    return f_rho, f_beta


def compute_force_error_RphiZ(cR, cZ, zernike_transform, cP, cI, Psi_total):
    """Computes force balance error at each node

    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zernike_transform (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        cP (array-like): coefficients to pass to pressure function
        cI (array-like): coefficients to pass to rotational transform function
        Psi_lcfs (float): total toroidal flux within LCFS

    Returns:
        F_err (ndarray, shape(3,N_nodes,)): F_R, F_phi, F_Z at each node
    """

    nodes = zernike_transform.nodes
    volumes = zernike_transform.volumes
    axn = zernike_transform.axn
    r = nodes[0]
    # value of r one step out from axis

    mu0 = 4*jnp.pi*1e-7
    presr = presfun(r, 1, cP)

    # compute fields components
    coord_der = compute_coordinate_derivatives(cR, cZ, zernike_transform)
    cov_basis = compute_covariant_basis(coord_der, zernike_transform)
    jacobian = compute_jacobian(coord_der, cov_basis, zernike_transform)
    con_basis = compute_contravariant_basis(
        coord_der, cov_basis, jacobian, zernike_transform)
    magnetic_field = compute_magnetic_field(cov_basis, jacobian, cI,
                                            Psi_total, zernike_transform)
    plasma_current = compute_plasma_current(coord_der, cov_basis,
                                            jacobian, magnetic_field, cI, Psi_total, zernike_transform)

    # helical basis vector
    beta = magnetic_field['B^zeta']*con_basis['e^theta'] - \
        magnetic_field['B^theta']*con_basis['e^zeta']

    # force balance error in radial and helical direction
    f_rho = mu0*(plasma_current['J^theta']*magnetic_field['B^zeta'] -
                 plasma_current['J^zeta']*magnetic_field['B^theta']) - mu0*presr
    f_beta = mu0*plasma_current['J^rho']

    F_err = f_rho * con_basis['grad_rho'] + f_beta * beta

    # weight by local volume
    vol = jacobian['g']*volumes[0]*volumes[1]*volumes[2]
    if len(axn):
        r1 = jnp.min(r[r != 0])
        r1idx = jnp.where(r == r1)[0]
        vol = put(vol, axn, jnp.mean(
            jacobian['g'][r1idx])/2*volumes[0, axn]*volumes[1, axn]*volumes[2, axn])
    F_err = F_err*vol

    return F_err


def compute_force_error_RddotZddot(cR, cZ, zernike_transform, cP, cI, Psi_total):
    """Computes force balance error at each node

    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zernike_transform (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        cP (array-like): coefficients to pass to pressure function
        cI (array-like): coefficients to pass to rotational transform function
        Psi_lcfs (float): total toroidal flux within LCFS

    Returns:
        cRddot (ndarray, shape(Ncoeffs,)): spectral coefficients for d^2R/dt^2
        cZddot (ndarray, shape(Ncoeffs,)): spectral coefficients for d^2Z/dt^2
    """

    coord_der = compute_coordinate_derivatives(cR, cZ, zernike_transform)
    F_err = compute_force_error_RphiZ(
        cR, cZ, zernike_transform, cP, cI, Psi_total)
    num_nodes = len(zernike_transform.nodes[0])

    AR = jnp.stack([jnp.ones(num_nodes), -coord_der['R_z'],
                    jnp.zeros(num_nodes)], axis=1)
    AZ = jnp.stack([jnp.zeros(num_nodes), -coord_der['Z_z'],
                    jnp.ones(num_nodes)], axis=1)
    A = jnp.stack([AR, AZ], axis=1)
    Rddot, Zddot = jnp.squeeze(jnp.matmul(A, F_err.T[:, :, jnp.newaxis])).T

    cRddot, cZddot = zernike_transform.fit(jnp.array([Rddot, Zddot]).T).T

    return cRddot, cZddot


def compute_accel_error_spectral(cR, cZ, cP, cI, Psi_total, pres_ratio, zeta_ratio, zernike_transform):
    """Computes acceleration error in spectral space

    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        cP (array-like): parameters to pass to pressure function
        cI (array-like): parameters to pass to rotational transform function
        Psi_lcfs (float): total toroidal flux within LCFS
        pres_ratio (double): fraction in range [0,1] of the full pressure profile to use
        zeta_ratio (double): fraction in range [0,1] of the full toroidal (zeta) derivatives to use
        zernike_transform (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes

    Returns:
        cR_zz_err (ndarray, shape(N_nodes,)): error in cR_zz
        cZ_zz_err (ndarray, shape(N_nodes,)): error in cZ_zz
    """

    mu0 = 4*jnp.pi*1e-7
    nodes = zernike_transform.nodes
    axn = zernike_transform.axn
    r = nodes[0]
    presr = presfun(r, 1, cP) * pres_ratio
    iota = iotafun(r, 0, cI)
    iotar = iotafun(r, 1, cI)

    coord_der = compute_coordinate_derivatives(
        cR, cZ, zernike_transform, zeta_ratio)

    R_zz = -(Psi_total**2*coord_der['R_r']**2*coord_der['Z_v']**2*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_z']**2*r**2 - Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['Z_v']**2*r + Psi_total**2*coord_der['R_r']**2*coord_der['Z_v']**4*r**2*iota**2 + Psi_total**2*coord_der['R']**3*coord_der['R_rr']*coord_der['Z_v']**2*r**2 + Psi_total**2*coord_der['R']**3*coord_der['R_vv']*coord_der['Z_r']**2*r**2 - Psi_total**2*coord_der['R']**2*coord_der['R_r']**2*coord_der['Z_v']**2*r**2 - Psi_total**2*coord_der['R']**2*coord_der['R_v']**2*coord_der['Z_r']**2*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**4*r*iota**2 + Psi_total**2*coord_der['R']**3*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*r + Psi_total**2*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_v']**2*r**2*iota**2 - coord_der['R']**3*coord_der['R_r']**3*coord_der['Z_v']**4*mu0*jnp.pi**2*presr + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['Z_v']**4*r**2*iota**2 + 2*Psi_total**2*coord_der['R_r']**2*coord_der['Z_v']**3*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']**2*coord_der['Z_v']**2*r - Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_vv']*r**2 + Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['Z_rv']*coord_der['Z_v']*r**2 - 2*Psi_total**2*coord_der['R']**3*coord_der['R_rv']*coord_der['Z_r']*coord_der['Z_v']*r**2 + Psi_total**2*coord_der['R']**3*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_rv']*r**2 - Psi_total**2*coord_der['R']**3*coord_der['R_v']*coord_der['Z_rr']*coord_der['Z_v']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**2*coord_der['Z_z']**2*r + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_z']**2*coord_der['Z_v']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_vv']*coord_der['R_z']**2*coord_der['Z_r']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['Z_v']**2*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_z']**2*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**3*coord_der['Z_z']*r*iota + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']**2*coord_der['Z_v']**2*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_vv']*coord_der['Z_v']**2*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_v']**2*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**3*coord_der['Z_z']*r**2*iotar + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**3*r*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['Z_r']*coord_der['Z_v']*r*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**3*coord_der['Z_rz']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['Z_v']**3*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['Z_r']*coord_der['Z_rz']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_r']*coord_der['Z_v']*r + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']**2*r + coord_der['R']**3*coord_der['R_v']**3*coord_der['Z_r']**3*coord_der['Z_v']*mu0*jnp.pi**2*presr - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**4*r**2*iota*iotar - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_v']**2*r*iota**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_vz']*coord_der['Z_v']**2*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['Z_r']*coord_der['Z_v']**3*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_rr']*coord_der['Z_v']**3*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['Z_rr']*coord_der['Z_v']*r**2*iota**2 - 2*Psi_total**2*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**3*r**2*iota**2 + 2*Psi_total**2*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']*coord_der['R_rz']*coord_der['Z_v']**2*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']*coord_der['R_vz']*coord_der['Z_r']**2*r**2 + 2*Psi_total**2*coord_der['R']**2*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']**2*coord_der['Z_r']*coord_der['Z_vv']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']**2*coord_der['Z_rv']*coord_der['Z_v']*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_z']**2*coord_der['Z_r']*coord_der['Z_v']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_r']*coord_der['Z_rv']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_rr']*coord_der['Z_v']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_rz']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_z']*coord_der['Z_v']*coord_der['Z_vz']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_vv']*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_rv']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_z']**2*r**2 - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_rr']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**2*coord_der['Z_z']*coord_der['Z_rz']*r**2 -
             Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']**2*coord_der['Z_z']*coord_der['Z_vz']*r**2 - 2*Psi_total**2*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_v']**2*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**3*r**2*iota*iotar + Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['Z_r']*coord_der['Z_v']*r**2*iota*iotar + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_rv']*coord_der['Z_v']*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_vv']*r**2*iota**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_v']**2*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']**2*coord_der['Z_v']*coord_der['Z_vv']*r**2*iota**2 - 3*coord_der['R']**3*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_v']**2*mu0*jnp.pi**2*presr - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']**2*r*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*r*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_z']*r*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_v']**2*r**2*iota*iotar - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']**2*r**2*iotar + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*r**2*iotar + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iotar - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_z']*coord_der['Z_v']**2*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_rz']*coord_der['Z_v']**2*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']**2*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_vz']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_rv']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_v']*coord_der['Z_rz']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_rr']*coord_der['Z_v']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_rz']*coord_der['Z_r']*coord_der['Z_v']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_vz']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_z']*coord_der['Z_v']*coord_der['Z_vv']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_vz']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_rv']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota - 4*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_rz']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_rr']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']**2*coord_der['Z_v']*coord_der['Z_vz']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']**2*coord_der['Z_vv']*coord_der['Z_z']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - 4*Psi_total**2*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_vz']*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']*coord_der['R_vz']*coord_der['Z_r']*coord_der['Z_v']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']*coord_der['Z_rz']*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']*coord_der['R_rz']*coord_der['Z_r']*coord_der['Z_v']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*coord_der['Z_vz']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*coord_der['Z_rz']*r**2 + 3*coord_der['R']**3*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**3*mu0*jnp.pi**2*presr - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_vv']*r**2*iota + 3*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_rv']*coord_der['Z_v']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_vz']*coord_der['Z_r']*coord_der['Z_v']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_vv']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_vv']*coord_der['Z_z']*r**2*iota + 3*Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota) / (Psi_total**2*coord_der['R']*r**2*(coord_der['R_r']*coord_der['Z_v'] - coord_der['R_v']*coord_der['Z_r'])**2)

    Z_zz = (Psi_total**2*coord_der['R']**3*coord_der['R_v']**2*coord_der['Z_r']*r - Psi_total**2*coord_der['R']**3*coord_der['R_v']**2*coord_der['Z_rr']*r**2 - Psi_total**2*coord_der['R']**3*coord_der['R_r']**2*coord_der['Z_vv']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']**4*coord_der['Z_r']*r*iota**2 - Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']*r + coord_der['R']**3*coord_der['R_v']**4*coord_der['Z_r']**3*mu0*jnp.pi**2*presr - Psi_total**2*coord_der['R']*coord_der['R_v']**4*coord_der['Z_rr']*r**2*iota**2 + Psi_total**2*coord_der['R_r']**2*coord_der['R_z']*coord_der['Z_v']**3*r**2*iota + Psi_total**2*coord_der['R_v']**3*coord_der['Z_r']**2*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['R_rv']*coord_der['Z_v']*r**2 + 2*Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['R_v']*coord_der['Z_rv']*r**2 + Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['R_vv']*coord_der['Z_r']*r**2 - Psi_total**2*coord_der['R']**3*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_r']*r**2 + Psi_total**2*coord_der['R']**3*coord_der['R_rr']*coord_der['R_v']*coord_der['Z_v']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']**2*coord_der['Z_r']*r + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_z']**2*r + Psi_total**2*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_v']**3*r**2*iota**2 + Psi_total**2*coord_der['R_v']**3*coord_der['Z_r']**2*coord_der['Z_v']*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']**2*coord_der['Z_rr']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_z']**2*coord_der['Z_vv']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_rr']*coord_der['Z_z']**2*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['Z_vv']*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R_r']**2*coord_der['R_z']*coord_der['Z_v']**2*coord_der['Z_z']*r**2 + Psi_total**2*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']**2*coord_der['Z_z']*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['R_z']*coord_der['Z_r']*r*iota - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']**2*coord_der['Z_vv']*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_rr']*coord_der['Z_v']**2*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_vv']*r**2*iota**2 - 2*Psi_total**2*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['R_z']*coord_der['Z_r']*r**2*iotar - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**3*r*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**3*coord_der['Z_v']*r*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rz']*coord_der['Z_v']**3*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['R_z']*coord_der['Z_rr']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['R_rz']*coord_der['Z_r']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_v']*r - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_z']**2*r - coord_der['R']**3*coord_der['R_r']**3*coord_der['R_v']*coord_der['Z_v']**3*mu0*jnp.pi**2*presr + Psi_total**2*coord_der['R']*coord_der['R_v']**4*coord_der['Z_r']*r**2*iota*iotar + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**3*coord_der['Z_rv']*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['Z_v']**3*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']**3*coord_der['Z_v']*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']**2*r*iota**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_vz']*r**2*iota + Psi_total**2*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota + Psi_total**2*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']**2*coord_der['Z_v']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_z']**2*coord_der['Z_v']*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_rv']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_vv']*coord_der['R_z']**2*coord_der['Z_r']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_r']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_v']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['R_rz']*coord_der['Z_r']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_z']*coord_der['R_vz']*coord_der['Z_v']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_rv']*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_vv']*coord_der['Z_r']*coord_der['Z_z']**2*r**2 - Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rz']*coord_der['Z_v']**2*coord_der['Z_z']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_vz']*coord_der['Z_r']**2*coord_der['Z_z']*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_z']*coord_der['Z_rz']*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['Z_v']*coord_der['Z_z'] *
            coord_der['Z_vz']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**3*r**2*iota*iotar - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**3*coord_der['Z_v']*r**2*iota*iotar - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_v']**2*coord_der['Z_v']*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['R_vv']*coord_der['Z_v']*r**2*iota**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_v']*r**2*iota**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_v']*r**2*iota**2 + 3*coord_der['R']**3*coord_der['R_r']**2*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']**2*mu0*jnp.pi**2*presr - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_v']*r*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_z']*r*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r*iota + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota*iotar - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_v']*r**2*iotar - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iotar + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2*iotar + 4*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_rv']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_vz']*coord_der['Z_r']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_rz']*coord_der['Z_v']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_v']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['R_z']*coord_der['Z_vv']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['R_vz']*coord_der['Z_v']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_vv']*coord_der['R_z']*coord_der['Z_v']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_vz']*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_rz']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_rz']*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_z']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_vz']*coord_der['Z_r']**2*coord_der['Z_v']*r**2*iota - 2*Psi_total**2*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_z']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_rz']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_rr']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - 2*Psi_total**2*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['R_vz']*coord_der['Z_r']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['R_rz']*coord_der['Z_v']*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_z']*coord_der['Z_vz']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_vz']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_z']*coord_der['Z_rz']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_rz']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2 - 2*Psi_total**2*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2 - 3*coord_der['R']**3*coord_der['R_r']*coord_der['R_v']**3*coord_der['Z_r']**2*coord_der['Z_v']*mu0*jnp.pi**2*presr - 3*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_vv']*coord_der['R_z']*coord_der['Z_r']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_vz']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_vv']*coord_der['Z_z']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_rv']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_vv']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - 3*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota) / (Psi_total**2*coord_der['R']*r**2*(coord_der['R_r']*coord_der['Z_v'] - coord_der['R_v']*coord_der['Z_r'])**2)

    if len(axn):
        R_zz = put(R_zz, axn, (24*Psi_total**2*coord_der['R_rv'][axn]**2*coord_der['Z_r'][axn]**2*coord_der['R'][axn]**2 - 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]**2 - 24*Psi_total**2*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]**2*coord_der['Z_r'][axn]**2 + 24*Psi_total**2*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]**2*coord_der['R'][axn]**2 - 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R'][axn]**3 - 12*Psi_total**2*coord_der['R_rrvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn] - 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]**2 - 12*Psi_total**2*coord_der['R_rrvv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn] - 72*Psi_total**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn]**2 + 72*Psi_total**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn]*coord_der['R'][axn]**2 + 12*Psi_total**2*coord_der['Z_rrvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_rr'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn]**3 - 12*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 - 48*Psi_total**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['Z_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 - 12*Psi_total**2*coord_der['Z_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 - 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]**2*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]**2 - 12*Psi_total**2*coord_der['R_rrvv'][axn]*coord_der['R_z'][axn]**2*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**3*coord_der['R_r'][axn]*coord_der['R'][axn] + 12*Psi_total**2*coord_der['Z_rrvv'][axn]*coord_der['R_z'][axn]**2*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]*coord_der['Z_rr'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn] + 12*Psi_total**2*coord_der['Z_rrvv'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]**2*coord_der['R']
                               [axn] + 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rvz'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] - 48*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**2 + 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['Z_rz'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**2*coord_der['Z_rz'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['Z_rv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**2*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rz'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rz'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] - 48*Psi_total**2*iota[axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]) / (24*Psi_total**2*(coord_der['R_rv'][axn]*coord_der['Z_r'][axn] - coord_der['Z_rv'][axn]*coord_der['R_r'][axn])**2*coord_der['R'][axn]))

        Z_zz = put(Z_zz, axn, (24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn] - 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**3 - 24*Psi_total**2*coord_der['R_rv'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R'][axn]**3 - 72*Psi_total**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**3*coord_der['R'][axn]**2 - 12*Psi_total**2*coord_der['Z_rrvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn]**3 - 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**3 - 12*Psi_total**2*coord_der['Z_rrvv'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 72*Psi_total**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn]*coord_der['R'][axn]**2 + 12*Psi_total**2*coord_der['R_rrvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 - 48*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 - 12*Psi_total**2*coord_der['R_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 - 12*Psi_total**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 - 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]**2 + 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]*coord_der['Z_r'][axn]**2 - 12*Psi_total**2*coord_der['Z_rrvv'][axn]*coord_der['R_z'][axn]**2*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**3*coord_der['Z_r'][axn]*coord_der['R'][axn] + 12*Psi_total**2*coord_der['R_rrvv'][axn]*coord_der['R_z'][axn]**2*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 12*Psi_total**2*coord_der['R_rrvv'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_rr'][axn]
                               * coord_der['Z_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['R_rvz'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 48*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] - 48*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['R_rz'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['Z_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn] - 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rz'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]) / (24*Psi_total**2*(coord_der['R_rv'][axn]*coord_der['Z_r'][axn] - coord_der['Z_rv'][axn]*coord_der['R_r'][axn])**2*coord_der['R'][axn]))

    R_zz_err = coord_der['R_zz'] - R_zz
    Z_zz_err = coord_der['Z_zz'] - Z_zz

    cR_zz_err = zernike_transform.fit(R_zz_err)
    cZ_zz_err = zernike_transform.fit(Z_zz_err)

    return cR_zz_err, cZ_zz_err


def compute_qs_error_spectral(cR, cZ, cI, Psi_total, NFP, zernike_transform, modes_pol, modes_tor, zeta_ratio):
    """Computes quasisymmetry error in spectral space

    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        cI (array-like): coefficients to pass to rotational transform function
        Psi_lcfs (float): total toroidal flux within LCFS
        NFP (int): number of field periods
        zernike_transform (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        modes_pol (ndarray, shape(N_modes)): poloidal Fourier mode numbers
        modes_tor (ndarray, shape(N_modes)): toroidal Fourier mode numbers
        zeta_ratio (float): fraction in range [0,1] of the full toroidal (zeta) derivatives to use

    Returns:
        cQS (ndarray, shape(N_modes,)): quasisymmetry error Fourier coefficients
    """

    nodes = zernike_transform.nodes
    r = nodes[0]
    iota = iotafun(r, 0, cI)

    coord_der = compute_coordinate_derivatives(
        cR, cZ, zernike_transform, zeta_ratio, mode='qs')
    cov_basis = compute_covariant_basis(
        coord_der, zernike_transform, mode='qs')
    jacobian = compute_jacobian(
        coord_der, cov_basis, zernike_transform, mode='qs')
    magnetic_field = compute_magnetic_field(cov_basis, jacobian, cI,
                                            Psi_total, zernike_transform, mode='qs')
    B_mag = compute_magnetic_field_magnitude(
        cov_basis, magnetic_field, cI, zernike_transform)

    # B-tilde derivatives
    Bt_v = magnetic_field['B^zeta_v']*(iota*B_mag['|B|_v']+B_mag['|B|_z']) + \
        magnetic_field['B^zeta']*(iota*B_mag['|B|_vv']+B_mag['|B|_vz'])
    Bt_z = magnetic_field['B^zeta_z']*(iota*B_mag['|B|_v']+B_mag['|B|_z']) + \
        magnetic_field['B^zeta']*(iota*B_mag['|B|_vz']+B_mag['|B|_zz'])

    # quasisymmetry
    QS = B_mag['|B|_v']*Bt_z - B_mag['|B|_z']*Bt_v

    theta = nodes[1]
    zeta = nodes[2]

    four_interp = double_fourier_basis(theta, zeta, modes_pol, modes_tor, NFP)
    cQS = jnp.linalg.lstsq(four_interp, jnp.array([QS]).T, rcond=None)[0].T

    return cQS
