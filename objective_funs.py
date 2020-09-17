import numpy as np
from field_components import compute_coordinate_derivatives, compute_covariant_basis, compute_contravariant_basis, compute_jacobian, compute_B_field, compute_J_field, compute_B_magnitude
from boundary_conditions import compute_bc_err_RZ, compute_bc_err_four, compute_lambda_err
from zernike import symmetric_x, double_fourier_basis
from backend import jnp, put, cross, dot, presfun, iotafun, unpack_x, rms


def get_equil_obj_fun(M,N,zern_idx,lambda_idx,stell_sym,error_mode,bdry_mode):
    """Gets the equilibrium objective function
    
    Args:
        M (int): maximum poloidal resolution
        N (int): maximum toroidal resolution
        zern_idx (ndarray of int, shape(Nc,3)): mode numbers for Zernike basis
        lambda_idx (ndarray of int, shape(Nc,2)): mode numbers for Fourier basis
        stell_sym (bool): True if stellarator symmetry is enforced
        error_mode (string): 'force' or 'accel'
        bdry_mode (string): 'real' or 'spectral'
        
    Returns:
        equil_obj (function): equilibrium objective function
        callback (function): function that prints equilibrium errors
    """
    
    if stell_sym:
        sym_mat = symmetric_x(M,N)
    else:
        sym_mat = np.eye(2*len(zern_idx) + len(lambda_idx))
    
    if error_mode == 'force':
        equil_fun = compute_force_error_nodes
    elif error_mode == 'accel':
        equil_fun = compute_accel_error_spectral
    
    if bdry_mode == 'real':
        bdry_fun = compute_bc_err_RZ
    elif bdry_mode == 'spectral':
        bdry_fun = compute_bc_err_four
    
    def equil_obj(x,bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio,pres_ratio,zeta_ratio,error_ratio,
                  NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights):
        
        cR,cZ,cL = unpack_x(jnp.matmul(sym_mat,x),len(zern_idx))
        errRf,errZf = equil_fun(cR,cZ,cP,cI,Psi_total,pres_ratio,zeta_ratio,zernt,nodes,volumes)
        errRb,errZb = bdry_fun(cR,cZ,cL,bdry_ratio,zern_idx,lambda_idx,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,NFP)
        errL0 = compute_lambda_err(cL,lambda_idx,NFP)
        
        # normalize weighting by numper of nodes
        residual = jnp.concatenate([weights['F']*errRf.flatten()/errRf.size*error_ratio,
                                    weights['F']*errZf.flatten()/errZf.size*error_ratio,
                                    weights['B']*errRb.flatten()/errRb.size,
                                    weights['B']*errZb.flatten()/errZb.size,
                                    weights['L']*errL0.flatten()/errL0.size])
        return residual
    
    def callback(x,bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio,pres_ratio,zeta_ratio,error_ratio,
                 NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights):
        
        cR,cZ,cL = unpack_x(jnp.matmul(sym_mat,x),len(zern_idx))
        errRf,errZf = equil_fun(cR,cZ,cP,cI,Psi_total,pres_ratio,zeta_ratio,zernt,nodes,volumes)
        errRb,errZb = bdry_fun(cR,cZ,cL,bdry_ratio,zern_idx,lambda_idx,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,NFP)
        errL0 = compute_lambda_err(cL,lambda_idx,NFP)
        
        errRf_rms = rms(errRf)
        errZf_rms = rms(errZf)
        errRb_rms = rms(errRb)
        errZb_rms = rms(errZb)
        errL0_rms = rms(errL0)
        
        # normalize weighting by numper of nodes
        residual = np.concatenate([weights['F']*errRf.flatten(),
                               weights['F']*errZf.flatten()*error_ratio,
                               weights['B']*errRb.flatten()*error_ratio,
                               weights['B']*errZb.flatten(),
                               weights['L']*errL0.flatten()])
        resid_rms = jnp.sum(residual**2)
        
        print('Weighted Loss: {:10.3e}  errRf: {:10.3e}  errZf: {:10.3e}  errRb: {:10.3e}  errZb: {:10.3e}  errL0: {:10.3e}'.format(
        resid_rms,errRf_rms,errZf_rms,errRb_rms,errZb_rms,errL0_rms))
        
    return equil_obj, callback


def compute_force_error_nodes(cR,cZ,cP,cI,Psi_total,pres_ratio,zeta_ratio,zernt,nodes,volumes):
    """Computes force balance error at each node
    
    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        cP (array-like): parameters to pass to pressure function
        cI (array-like): parameters to pass to rotational transform function
        Psi_total (float): total toroidal flux within LCFS
        pres_ratio (float): fraction in range [0,1] of the full pressure profile to use
        zeta_ratio (float): fraction in range [0,1] of the full toroidal (zeta) derivatives to use
        zernt (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        nodes (ndarray, shape(3,N_nodes)): coordinates (r,v,z) of the collocation points
        volumes (ndarray, shape(3,N_nodes)): arc length (dr,dv,dz) along each coordinate at each node, for computing volume
        
    Returns:
        F_rho (ndarray, shape(N_nodes,)): radial force balance error at each node
        F_beta (ndarray, shape(N_nodes,)): helical force balance error at each node
    """
    
    mu0 = 4*jnp.pi*1e-7
    r = nodes[0]
    axn = jnp.where(r == 0)[0]
    r1 = jnp.min(r[r != 0]) # value of r one step out from axis
    r1_idx = jnp.where(r == r1)[0]
    pres_r = presfun(r,1, cP) * pres_ratio
    
    # compute fields components
    coord_der = compute_coordinate_derivatives(cR,cZ,zernt,zeta_ratio)
    cov_basis = compute_covariant_basis(coord_der)
    jacobian  = compute_jacobian(coord_der,cov_basis)
    con_basis = compute_contravariant_basis(coord_der,cov_basis,jacobian,nodes)
    B_field   = compute_B_field(cov_basis,jacobian,cI,Psi_total,nodes)
    J_field   = compute_J_field(coord_der,cov_basis,jacobian,B_field,cI,Psi_total,nodes)

    # force balance error components
    F_rho  = jacobian['g']*(J_field['J^theta']*B_field['B^zeta'] - J_field['J^zeta']*B_field['B^theta']) - pres_r
    F_beta = jacobian['g']*J_field['J^rho']
    
    # radial and helical directions
    beta = B_field['B^zeta']*con_basis['e^theta'] - B_field['B^theta']*con_basis['e^zeta']
    radial  = jnp.sqrt(con_basis['g^rr']) * jnp.sign(dot(con_basis['e^rho'],cov_basis['e_rho'],0))
    helical = jnp.sqrt(con_basis['g^vv']*B_field['B^zeta']**2 + con_basis['g^zz']*B_field['B^theta']**2 - 2*con_basis['g^vz']*B_field['B^theta']*B_field['B^zeta']) \
            * jnp.sign(dot(beta,cov_basis['e_theta'],0)) * jnp.sign(dot(beta,cov_basis['e_zeta'],0))
    
    # axis terms
    Jsup_theta = (B_field['B_rho_z']   - B_field['B_zeta_r']) / mu0
    Jsup_zeta  = (B_field['B_theta_r'] - B_field['B_rho_v'])  / mu0
    F_rho      = put(F_rho, axn, Jsup_theta[axn]*B_field['B^zeta'][axn] - Jsup_zeta[axn]*B_field['B^theta'][axn])
    grad_theta = cross(cov_basis['e_zeta'],cov_basis['e_rho'],0)
    gsup_vv    = dot(grad_theta,grad_theta,0)
    F_beta     = put(F_beta, axn, J_field['J^rho'][axn])
    helical    = put(helical,axn, jnp.sqrt(gsup_vv[axn]*B_field['B^zeta'][axn]**2) * jnp.sign(B_field['B^zeta'][axn]))
    
    # scalar errors
    f_rho  = F_rho *radial
    f_beta = F_beta*helical
    
    # weight by local volume
    if volumes is not None:
        vol = jacobian['g']*volumes[0]*volumes[1]*volumes[2];
        vol = put(vol, axn, jnp.mean(jacobian['g'][r1_idx])/2*volumes[0,axn]*volumes[1,axn]*volumes[2,axn])
        f_rho  = f_rho *vol
        f_beta = f_beta*vol
    
    return f_rho, f_beta


def compute_force_error_RphiZ(cR,cZ,zernt,nodes,cP,cI,Psi_total,volumes):
    """Computes force balance error at each node
    
    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zernt (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        cP (array-like): coefficients to pass to pressure function
        cI (array-like): coefficients to pass to rotational transform function
        Psi_total (float): total toroidal flux within LCFS
        volumes (ndarray, shape(3,N_nodes)): arc length (dr,dv,dz) along each coordinate at each node, for computing volume.
        
    Returns:
        F_err (ndarray, shape(3,N_nodes,)): F_R, F_phi, F_Z at each node
    """
    
    r = nodes[0]
    axn = jnp.where(r == 0)[0]
    # value of r one step out from axis
    r1 = jnp.min(r[r != 0])
    r1idx = jnp.where(r == r1)[0]
    
    mu0 = 4*jnp.pi*1e-7
    presr = presfun(r,1, cP)
    
    # compute fields components
    coord_der = compute_coordinate_derivatives(cR,cZ,zernt)
    cov_basis = compute_covariant_basis(coord_der)
    jacobian  = compute_jacobian(coord_der,cov_basis)
    con_basis = compute_contravariant_basis(coord_der,cov_basis,jacobian,nodes)
    B_field   = compute_B_field(cov_basis,jacobian,cI,Psi_total,nodes)
    J_field   = compute_J_field(coord_der,cov_basis,jacobian,B_field,cI,Psi_total,nodes)
    
    # helical basis vector
    beta = B_field['B^zeta']*con_basis['e^theta'] - B_field['B^theta']*con_basis['e^zeta']
    
    # force balance error in radial and helical direction
    f_rho = mu0*(J_field['J^theta']*B_field['B^zeta'] - J_field['J^zeta']*B_field['B^theta']) - mu0*presr
    f_beta = mu0*J_field['J^rho']
    
    F_err = f_rho * con_basis['grad_rho'] + f_beta * beta
    
    # weight by local volume
    if volumes is not None:
        vol = jacobian['g']*volumes[0]*volumes[1]*volumes[2];
        vol = put(vol, axn, jnp.mean(jacobian['g'][r1idx])/2*volumes[0,axn]*volumes[1,axn]*volumes[2,axn])
        F_err = F_err*vol
    
    return F_err


def compute_force_error_RddotZddot(cR,cZ,zernt,nodes,cP,cI,Psi_total,volumes):
    """Computes force balance error at each node
    
    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zernt (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        cP (array-like): coefficients to pass to pressure function
        cI (array-like): coefficients to pass to rotational transform function
        Psi_total (float): total toroidal flux within LCFS
        volumes (ndarray, shape(3,N_nodes)): arc length (dr,dv,dz) along each coordinate at each node, for computing volume.

    Returns:
        cRddot (ndarray, shape(Ncoeffs,)): spectral coefficients for d^2R/dt^2
        cZddot (ndarray, shape(Ncoeffs,)): spectral coefficients for d^2Z/dt^2
    """
    
    coord_der = compute_coordinate_derivatives(cR,cZ,zernt)
    F_err = compute_force_error_RphiZ(cR,cZ,zernt,nodes,cP,cI,Psi_total,volumes)
    num_nodes = len(nodes[0])
    
    AR = jnp.stack([jnp.ones(num_nodes),-coord_der['R_z'],jnp.zeros(num_nodes)],axis=1)
    AZ = jnp.stack([jnp.zeros(num_nodes),-coord_der['Z_z'],jnp.ones(num_nodes)],axis=1)
    A = jnp.stack([AR,AZ],axis=1)
    Rddot, Zddot = jnp.squeeze(jnp.matmul(A,F_err.T[:,:,jnp.newaxis])).T
    
    cRddot, cZddot = zernt.fit(jnp.array([Rddot,Zddot]).T,1e-6).T
    
    return cRddot, cZddot


def compute_accel_error_spectral(cR,cZ,cP,cI,Psi_total,pres_ratio,zeta_ratio,zernt,nodes,volumes):
    """Computes acceleration error in spectral space
    
    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        cP (array-like): parameters to pass to pressure function
        cI (array-like): parameters to pass to rotational transform function
        Psi_total (float): total toroidal flux within LCFS
        pres_ratio (float): fraction in range [0,1] of the full pressure profile to use
        zeta_ratio (float): fraction in range [0,1] of the full toroidal (zeta) derivatives to use
        zernt (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        nodes (ndarray, shape(3,N_nodes)): coordinates (r,v,z) of the collocation points
        volumes (ndarray, shape(3,N_nodes)): arc length (dr,dv,dz) along each coordinate at each node, for computing volume
        
    Returns:
        cR_zz_err (ndarray, shape(N_nodes,)): error in cR_zz
        cZ_zz_err (ndarray, shape(N_nodes,)): error in cZ_zz
    """
    
    mu0 = 4*jnp.pi*1e-7
    r = nodes[0]
    axn = jnp.where(r == 0)[0]
    presr = presfun(r,1, cP) * pres_ratio
    iota  = iotafun(r,0, cI)
    iotar = iotafun(r,1, cI)
    
    coord_der = compute_coordinate_derivatives(cR,cZ,zernt,zeta_ratio)
    
    R_zz = -(Psi_total**2*coord_der['R_r']**2*coord_der['Z_v']**2*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_z']**2*r**2 - Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['Z_v']**2*r + Psi_total**2*coord_der['R_r']**2*coord_der['Z_v']**4*r**2*iota**2 + Psi_total**2*coord_der['R']**3*coord_der['R_rr']*coord_der['Z_v']**2*r**2 + Psi_total**2*coord_der['R']**3*coord_der['R_vv']*coord_der['Z_r']**2*r**2 - Psi_total**2*coord_der['R']**2*coord_der['R_r']**2*coord_der['Z_v']**2*r**2 - Psi_total**2*coord_der['R']**2*coord_der['R_v']**2*coord_der['Z_r']**2*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**4*r*iota**2 + Psi_total**2*coord_der['R']**3*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*r + Psi_total**2*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_v']**2*r**2*iota**2 - coord_der['R']**3*coord_der['R_r']**3*coord_der['Z_v']**4*mu0*jnp.pi**2*presr + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['Z_v']**4*r**2*iota**2 + 2*Psi_total**2*coord_der['R_r']**2*coord_der['Z_v']**3*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']**2*coord_der['Z_v']**2*r - Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_vv']*r**2 + Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['Z_rv']*coord_der['Z_v']*r**2 - 2*Psi_total**2*coord_der['R']**3*coord_der['R_rv']*coord_der['Z_r']*coord_der['Z_v']*r**2 + Psi_total**2*coord_der['R']**3*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_rv']*r**2 - Psi_total**2*coord_der['R']**3*coord_der['R_v']*coord_der['Z_rr']*coord_der['Z_v']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**2*coord_der['Z_z']**2*r + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_z']**2*coord_der['Z_v']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_vv']*coord_der['R_z']**2*coord_der['Z_r']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['Z_v']**2*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_z']**2*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**3*coord_der['Z_z']*r*iota + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']**2*coord_der['Z_v']**2*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_vv']*coord_der['Z_v']**2*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_v']**2*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**3*coord_der['Z_z']*r**2*iotar + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**3*r*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['Z_r']*coord_der['Z_v']*r*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**3*coord_der['Z_rz']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['Z_v']**3*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['Z_r']*coord_der['Z_rz']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_r']*coord_der['Z_v']*r + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']**2*r + coord_der['R']**3*coord_der['R_v']**3*coord_der['Z_r']**3*coord_der['Z_v']*mu0*jnp.pi**2*presr - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**4*r**2*iota*iotar - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_v']**2*r*iota**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_vz']*coord_der['Z_v']**2*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['Z_r']*coord_der['Z_v']**3*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_rr']*coord_der['Z_v']**3*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['Z_rr']*coord_der['Z_v']*r**2*iota**2 - 2*Psi_total**2*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**3*r**2*iota**2 + 2*Psi_total**2*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']*coord_der['R_rz']*coord_der['Z_v']**2*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']*coord_der['R_vz']*coord_der['Z_r']**2*r**2 + 2*Psi_total**2*coord_der['R']**2*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']**2*coord_der['Z_r']*coord_der['Z_vv']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']**2*coord_der['Z_rv']*coord_der['Z_v']*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_z']**2*coord_der['Z_r']*coord_der['Z_v']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_r']*coord_der['Z_rv']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_rr']*coord_der['Z_v']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_rz']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_z']*coord_der['Z_v']*coord_der['Z_vz']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_vv']*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_rv']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_z']**2*r**2 - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_rr']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**2*coord_der['Z_z']*coord_der['Z_rz']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']**2*coord_der['Z_z']*coord_der['Z_vz']*r**2 - 2*Psi_total**2*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_v']**2*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**3*r**2*iota*iotar + Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['Z_r']*coord_der['Z_v']*r**2*iota*iotar + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_rv']*coord_der['Z_v']*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_vv']*r**2*iota**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_v']**2*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']**2*coord_der['Z_v']*coord_der['Z_vv']*r**2*iota**2 - 3*coord_der['R']**3*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_v']**2*mu0*jnp.pi**2*presr - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']**2*r*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*r*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_z']*r*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_v']**2*r**2*iota*iotar - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']**2*r**2*iotar + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*r**2*iotar + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iotar - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_z']*coord_der['Z_v']**2*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_rz']*coord_der['Z_v']**2*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']**2*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_vz']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_rv']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_v']*coord_der['Z_rz']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_rr']*coord_der['Z_v']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_rz']*coord_der['Z_r']*coord_der['Z_v']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_vz']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_z']*coord_der['Z_v']*coord_der['Z_vv']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_vz']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_rv']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota - 4*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_rz']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_rr']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']**2*coord_der['Z_v']*coord_der['Z_vz']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']**2*coord_der['Z_vv']*coord_der['Z_z']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - 4*Psi_total**2*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_vz']*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']*coord_der['R_vz']*coord_der['Z_r']*coord_der['Z_v']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']*coord_der['Z_rz']*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']*coord_der['R_rz']*coord_der['Z_r']*coord_der['Z_v']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*coord_der['Z_vz']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*coord_der['Z_rz']*r**2 + 3*coord_der['R']**3*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**3*mu0*jnp.pi**2*presr - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_vv']*r**2*iota + 3*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_rv']*coord_der['Z_v']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_vz']*coord_der['Z_r']*coord_der['Z_v']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_vv']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_vv']*coord_der['Z_z']*r**2*iota + 3*Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota) / (Psi_total**2*coord_der['R']*r**2*(coord_der['R_r']*coord_der['Z_v'] - coord_der['R_v']*coord_der['Z_r'])**2)
    Z_zz = (Psi_total**2*coord_der['R']**3*coord_der['R_v']**2*coord_der['Z_r']*r - Psi_total**2*coord_der['R']**3*coord_der['R_v']**2*coord_der['Z_rr']*r**2 - Psi_total**2*coord_der['R']**3*coord_der['R_r']**2*coord_der['Z_vv']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']**4*coord_der['Z_r']*r*iota**2 - Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']*r + coord_der['R']**3*coord_der['R_v']**4*coord_der['Z_r']**3*mu0*jnp.pi**2*presr - Psi_total**2*coord_der['R']*coord_der['R_v']**4*coord_der['Z_rr']*r**2*iota**2 + Psi_total**2*coord_der['R_r']**2*coord_der['R_z']*coord_der['Z_v']**3*r**2*iota + Psi_total**2*coord_der['R_v']**3*coord_der['Z_r']**2*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['R_rv']*coord_der['Z_v']*r**2 + 2*Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['R_v']*coord_der['Z_rv']*r**2 + Psi_total**2*coord_der['R']**3*coord_der['R_r']*coord_der['R_vv']*coord_der['Z_r']*r**2 - Psi_total**2*coord_der['R']**3*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_r']*r**2 + Psi_total**2*coord_der['R']**3*coord_der['R_rr']*coord_der['R_v']*coord_der['Z_v']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']**2*coord_der['Z_r']*r + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_z']**2*r + Psi_total**2*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_v']**3*r**2*iota**2 + Psi_total**2*coord_der['R_v']**3*coord_der['Z_r']**2*coord_der['Z_v']*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']**2*coord_der['Z_rr']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_z']**2*coord_der['Z_vv']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_rr']*coord_der['Z_z']**2*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['Z_vv']*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R_r']**2*coord_der['R_z']*coord_der['Z_v']**2*coord_der['Z_z']*r**2 + Psi_total**2*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']**2*coord_der['Z_z']*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['R_z']*coord_der['Z_r']*r*iota - Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']**2*coord_der['Z_vv']*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_rr']*coord_der['Z_v']**2*r**2*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_vv']*r**2*iota**2 - 2*Psi_total**2*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['R_z']*coord_der['Z_r']*r**2*iotar - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**3*r*iota**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**3*coord_der['Z_v']*r*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rz']*coord_der['Z_v']**3*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['R_z']*coord_der['Z_rr']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_v']**3*coord_der['R_rz']*coord_der['Z_r']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_v']*r - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_z']**2*r - coord_der['R']**3*coord_der['R_r']**3*coord_der['R_v']*coord_der['Z_v']**3*mu0*jnp.pi**2*presr + Psi_total**2*coord_der['R']*coord_der['R_v']**4*coord_der['Z_r']*r**2*iota*iotar + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**3*coord_der['Z_rv']*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['Z_v']**3*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']**3*coord_der['Z_v']*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']**2*r*iota**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_vz']*r**2*iota + Psi_total**2*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota + Psi_total**2*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']**2*coord_der['Z_v']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_z']**2*coord_der['Z_v']*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_rv']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_vv']*coord_der['R_z']**2*coord_der['Z_r']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_r']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_v']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['R_rz']*coord_der['Z_r']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_z']*coord_der['R_vz']*coord_der['Z_v']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_rv']*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_vv']*coord_der['Z_r']*coord_der['Z_z']**2*r**2 - Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rz']*coord_der['Z_v']**2*coord_der['Z_z']*r**2 + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_vz']*coord_der['Z_r']**2*coord_der['Z_z']*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_z']*coord_der['Z_rz']*r**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['Z_v']*coord_der['Z_z']*coord_der['Z_vz']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**3*r**2*iota*iotar - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**3*coord_der['Z_v']*r**2*iota*iotar - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_v']**2*coord_der['Z_v']*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['R_vv']*coord_der['Z_v']*r**2*iota**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota**2 + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_v']*r**2*iota**2 + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_v']*r**2*iota**2 + 3*coord_der['R']**3*coord_der['R_r']**2*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']**2*mu0*jnp.pi**2*presr - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_v']*r*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_z']*r*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r*iota + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota*iotar - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_v']*r**2*iotar - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iotar + Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2*iotar + 4*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_rv']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_vz']*coord_der['Z_r']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_rz']*coord_der['Z_v']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_v']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['R_z']*coord_der['Z_vv']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['R_vz']*coord_der['Z_v']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_vv']*coord_der['R_z']*coord_der['Z_v']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_vz']*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_rz']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_rz']*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_z']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_vz']*coord_der['Z_r']**2*coord_der['Z_v']*r**2*iota - 2*Psi_total**2*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_z']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_rz']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_rr']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - 2*Psi_total**2*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['R_vz']*coord_der['Z_r']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['R_rz']*coord_der['Z_v']*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_z']*coord_der['Z_vz']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_vz']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2 - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_z']*coord_der['Z_rz']*r**2 - Psi_total**2*coord_der['R']*coord_der['R_v']*coord_der['R_rz']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2 - 2*Psi_total**2*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2 - 3*coord_der['R']**3*coord_der['R_r']*coord_der['R_v']**3*coord_der['Z_r']**2*coord_der['Z_v']*mu0*jnp.pi**2*presr - 3*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_vv']*coord_der['R_z']*coord_der['Z_r']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_vz']*r**2*iota - 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_vv']*coord_der['Z_z']*r**2*iota + 2*Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_rv']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota + Psi_total**2*coord_der['R']*coord_der['R_r']*coord_der['R_vv']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - 3*Psi_total**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota) / (Psi_total**2*coord_der['R']*r**2*(coord_der['R_r']*coord_der['Z_v'] - coord_der['R_v']*coord_der['Z_r'])**2)
    
    put(R_zz,axn,(24*Psi_total**2*coord_der['R_rv'][axn]**2*coord_der['Z_r'][axn]**2*coord_der['R'][axn]**2 - 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]**2 - 24*Psi_total**2*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]**2*coord_der['Z_r'][axn]**2 + 24*Psi_total**2*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]**2*coord_der['R'][axn]**2 - 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R'][axn]**3 - 12*Psi_total**2*coord_der['R_rrvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn] - 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]**2 - 12*Psi_total**2*coord_der['R_rrvv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn] - 72*Psi_total**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn]**2 + 72*Psi_total**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn]*coord_der['R'][axn]**2 + 12*Psi_total**2*coord_der['Z_rrvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_rr'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn]**3 - 12*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 - 48*Psi_total**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['Z_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 - 12*Psi_total**2*coord_der['Z_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 - 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]**2*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]**2 - 12*Psi_total**2*coord_der['R_rrvv'][axn]*coord_der['R_z'][axn]**2*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**3*coord_der['R_r'][axn]*coord_der['R'][axn] + 12*Psi_total**2*coord_der['Z_rrvv'][axn]*coord_der['R_z'][axn]**2*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]*coord_der['Z_rr'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn] + 12*Psi_total**2*coord_der['Z_rrvv'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rvz'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] - 48*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**2 + 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['Z_rz'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**2*coord_der['Z_rz'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['Z_rv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**2*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rz'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rz'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] - 48*Psi_total**2*iota[axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]) / (24*Psi_total**2*(coord_der['R_rv'][axn]*coord_der['Z_r'][axn] - coord_der['Z_rv'][axn]*coord_der['R_r'][axn])**2*coord_der['R'][axn]))
    put(Z_zz,axn,(24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn] - 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**3 - 24*Psi_total**2*coord_der['R_rv'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R'][axn]**3 - 72*Psi_total**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**3*coord_der['R'][axn]**2 - 12*Psi_total**2*coord_der['Z_rrvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn]**3 - 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**3 - 12*Psi_total**2*coord_der['Z_rrvv'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 72*Psi_total**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn]*coord_der['R'][axn]**2 + 12*Psi_total**2*coord_der['R_rrvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 - 48*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 - 12*Psi_total**2*coord_der['R_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_total**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 - 12*Psi_total**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 - 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]**2 + 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]*coord_der['Z_r'][axn]**2 - 12*Psi_total**2*coord_der['Z_rrvv'][axn]*coord_der['R_z'][axn]**2*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**3*coord_der['Z_r'][axn]*coord_der['R'][axn] + 12*Psi_total**2*coord_der['R_rrvv'][axn]*coord_der['R_z'][axn]**2*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['R_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 12*Psi_total**2*coord_der['R_rrvv'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_rr'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 12*Psi_total**2*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['R_rvz'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 48*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] - 48*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['R_rz'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*coord_der['Z_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn] - 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rz'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_total**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_total**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_total**2*iota[axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]) / (24*Psi_total**2*(coord_der['R_rv'][axn]*coord_der['Z_r'][axn] - coord_der['Z_rv'][axn]*coord_der['R_r'][axn])**2*coord_der['R'][axn]))
    
    R_zz_err = coord_der['R_zz'] - R_zz
    Z_zz_err = coord_der['Z_zz'] - Z_zz
    
    cR_zz_err = zernt.fit(R_zz_err,1e-6)
    cZ_zz_err = zernt.fit(Z_zz_err,1e-6)
    
    return cR_zz_err,cZ_zz_err


def compute_qs_error_spectral(cR,cZ,zernt,nodes,modesM,modesN,cI,Psi_total,NFP,zeta_ratio):
    """Computes quasi-symmetry error in spectral space
    
    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zernt (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        nodes (ndarray, shape(3,N_nodes)): coordinates (r,v,z) of the collocation points
        modesM (ndarray, shape(N_modes)): poloidal Fourier modes
        modesN (ndarray, shape(N_modes)): toroidal Fourier modes
        cI (array-like): coefficients to pass to rotational transform function
        Psi_total (float): total toroidal flux within LCFS
        NFP (int): number of field periods
        
    Returns:
        cQS (ndarray, shape(N_modes,)): quasisymmetry error Fourier coefficients
    """
    
    r = nodes[0]
    iota = iotafun(r,0, cI)
    
    coord_der = compute_coordinate_derivatives(cR,cZ,zernt,zeta_ratio)
    cov_basis = compute_covariant_basis(coord_der)
    jacobian  = compute_jacobian(coord_der,cov_basis)
    B_field   = compute_B_field(cov_basis,jacobian,cI,Psi_total,nodes)
    B_mag     = compute_B_magnitude(nodes,cI,cov_basis,B_field)
    
    # B-tilde derivatives
    Bt_v = B_field['B^zeta_v']*(iota*B_mag['|B|_v']+B_mag['|B|_z']) + B_field['B^zeta']*(iota*B_mag['|B|_vv']+B_mag['|B|_vz'])
    Bt_z = B_field['B^zeta_z']*(iota*B_mag['|B|_v']+B_mag['|B|_z']) + B_field['B^zeta']*(iota*B_mag['|B|_vz']+B_mag['|B|_zz'])
    
    # quasisymmetry
    QS = B_mag['|B|_v']*Bt_z - B_mag['|B|_z']*Bt_v
    
    theta = nodes[1]
    zeta  = nodes[2]
    four_bdry_interp = jnp.stack([double_fourier_basis(theta,zeta,m,n,NFP) for m,n in zip(modesM,modesN)]).T
    cQS = jnp.linalg.lstsq(four_bdry_interp,jnp.array([QS]).T,rcond=None)[0].T
    
    return cQS