import functools
from backend import jnp, conditional_decorator, jit, use_jax, fori_loop, put, cross, dot, sign, presfun, iotafun


def compute_coordinate_derivatives(cR,cZ,zernt):
    """Converts from spectral to real space and evaluates derivatives of R,Z wrt to SFL coords
    
    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zernt (ZernikeTransform): object with transform method to go from spectral to physical space with derivatives
    
    Returns:
        coord_der (dict): dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations
    """
    # notation: X_y means derivative of X wrt y
    coord_der = {}
    coord_der['R'] = zernt.transform(cR,0,0,0)
    coord_der['Z'] = zernt.transform(cZ,0,0,0)
    coord_der['0'] = jnp.zeros_like(coord_der['R'])
    
    coord_der['R_r'] = zernt.transform(cR,1,0,0)
    coord_der['Z_r'] = zernt.transform(cZ,1,0,0)
    coord_der['R_v'] = zernt.transform(cR,0,1,0)
    coord_der['Z_v'] = zernt.transform(cZ,0,1,0)
    coord_der['R_z'] = zernt.transform(cR,0,0,1)
    coord_der['Z_z'] = zernt.transform(cZ,0,0,1)

    coord_der['R_rr'] = zernt.transform(cR,2,0,0)
    coord_der['Z_rr'] = zernt.transform(cZ,2,0,0)
    coord_der['R_rv'] = zernt.transform(cR,1,1,0)
    coord_der['Z_rv'] = zernt.transform(cZ,1,1,0)
    coord_der['R_rz'] = zernt.transform(cR,1,0,1)
    coord_der['Z_rz'] = zernt.transform(cZ,1,0,1)

    coord_der['R_vv'] = zernt.transform(cR,0,2,0)
    coord_der['Z_vv'] = zernt.transform(cZ,0,2,0)
    coord_der['R_vz'] = zernt.transform(cR,0,1,1)
    coord_der['Z_vz'] = zernt.transform(cZ,0,1,1)

    coord_der['R_zz'] = zernt.transform(cR,0,0,2)
    coord_der['Z_zz'] = zernt.transform(cZ,0,0,2)

    coord_der['R_rrv'] = zernt.transform(cR,2,1,0)
    coord_der['Z_rrv'] = zernt.transform(cZ,2,1,0)
    coord_der['R_rvv'] = zernt.transform(cR,1,2,0)
    coord_der['Z_rvv'] = zernt.transform(cZ,1,2,0)
    coord_der['R_rvz'] = zernt.transform(cR,1,1,1)
    coord_der['Z_rvz'] = zernt.transform(cZ,1,1,1)

    coord_der['R_rrvv'] = zernt.transform(cR,2,2,0)
    coord_der['Z_rrvv'] = zernt.transform(cZ,2,2,0)

    return coord_der


def compute_covariant_basis(coord_der):
    """Computes covariant basis vectors at grid points
    
    Args:
        coord_der (dict): dictionary of ndarray, shape(N_nodes,) of the coordinate derivatives at each node
        
    Returns:
        cov_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node
    """
    # notation: subscript word is direction of unit vector, subscript letters denote partial derivatives
    # eg, e_rho_v is the v derivative of the covariant basis vector in the rho direction
    cov_basis = {}
    cov_basis['e_rho']      = jnp.array([coord_der['R_r'],  coord_der['0'],   coord_der['Z_r']])
    cov_basis['e_theta']    = jnp.array([coord_der['R_v'],  coord_der['0'],   coord_der['Z_v']])
    cov_basis['e_zeta']     = jnp.array([coord_der['R_z'], -coord_der['R'],   coord_der['Z_z']])

    cov_basis['e_rho_r']    = jnp.array([coord_der['R_rr'], coord_der['0'],   coord_der['Z_rr']])
    cov_basis['e_rho_v']    = jnp.array([coord_der['R_rv'], coord_der['0'],   coord_der['Z_rv']])
    cov_basis['e_rho_z']    = jnp.array([coord_der['R_rz'], coord_der['0'],   coord_der['Z_rz']])

    cov_basis['e_theta_r']  = jnp.array([coord_der['R_rv'], coord_der['0'],   coord_der['Z_rv']])
    cov_basis['e_theta_v']  = jnp.array([coord_der['R_vv'], coord_der['0'],   coord_der['Z_vv']])
    cov_basis['e_theta_z']  = jnp.array([coord_der['R_vz'], coord_der['0'],   coord_der['Z_vz']])

    cov_basis['e_zeta_r']  = jnp.array([coord_der['R_rz'], -coord_der['R_r'], coord_der['Z_rz']])
    cov_basis['e_zeta_v']  = jnp.array([coord_der['R_vz'], -coord_der['R_v'], coord_der['Z_vz']])
    cov_basis['e_zeta_z']  = jnp.array([coord_der['R_zz'], -coord_der['R_z'], coord_der['Z_zz']])

    cov_basis['e_rho_vv']  = jnp.array([coord_der['R_rvv'], coord_der['0'],   coord_der['Z_rvv']])
    cov_basis['e_rho_vz']  = jnp.array([coord_der['R_rvz'], coord_der['0'],   coord_der['Z_rvz']])
    cov_basis['e_zeta_rv'] = jnp.array([coord_der['R_rvz'],-coord_der['R_rv'],coord_der['Z_rvz']])
    
    return cov_basis


def compute_jacobian(coord_der,cov_basis):
    """Computes coordinate jacobian and derivatives
    
    Args:
        coord_der (dict): dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations
        cov_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node 
        
    Returns:
        jacobian (dict): dictionary of ndarray, shape(N_nodes,) of coordinate jacobian and partial derivatives
    """
    # notation: subscripts denote partial derivatives
    jacobian = {}    
    jacobian['g'] = dot(cov_basis['e_rho'] , cross(cov_basis['e_theta'],cov_basis['e_zeta'],0),0)

    jacobian['g_r'] = dot(cov_basis['e_rho_r'],cross(cov_basis['e_theta'],cov_basis['e_zeta'],0),0) \
                      + dot(cov_basis['e_rho'],cross(cov_basis['e_rho_v'],cov_basis['e_zeta'],0),0) \
                      + dot(cov_basis['e_rho'],cross(cov_basis['e_theta'],cov_basis['e_zeta_r'],0),0)
    jacobian['g_v'] = dot(cov_basis['e_rho_v'],cross(cov_basis['e_theta'],cov_basis['e_zeta'],0),0) \
                      + dot(cov_basis['e_rho'],cross(cov_basis['e_theta_v'],cov_basis['e_zeta'],0),0) \
                      + dot(cov_basis['e_rho'],cross(cov_basis['e_theta'],cov_basis['e_zeta_v'],0),0)
    jacobian['g_z'] = dot(cov_basis['e_rho_z'],cross(cov_basis['e_theta'],cov_basis['e_zeta'],0),0) \
                      + dot(cov_basis['e_rho'],cross(cov_basis['e_theta_z'],cov_basis['e_zeta'],0),0) \
                      + dot(cov_basis['e_rho'],cross(cov_basis['e_theta'],cov_basis['e_zeta_z'],0),0)
    # need these later for rho=0
    jacobian['g_rr']  = coord_der['R']*(coord_der['R_r']*coord_der['Z_rrv'] - coord_der['Z_r']*coord_der['R_rrv']
                                        + 2*coord_der['R_rr']*coord_der['Z_rv'] - 2*coord_der['R_rv']*coord_der['Z_rr']) \
                                        + 2*coord_der['R_r']*(coord_der['Z_rv']*coord_der['R_r'] - coord_der['R_rv']*coord_der['Z_r'])
    jacobian['g_rv']  = coord_der['R']*(coord_der['Z_rvv']*coord_der['R_r'] - coord_der['R_rvv']*coord_der['Z_r'])
    jacobian['g_rz']  = coord_der['R_z']*(coord_der['R_r']*coord_der['Z_rv'] - coord_der['R_rv']*coord_der['Z_r']) \
                                          + coord_der['R']*(coord_der['R_rz']*coord_der['Z_rv'] + coord_der['R_r']*coord_der['Z_rvz']
                                          - coord_der['R_rvz']*coord_der['Z_r'] - coord_der['R_rv']*coord_der['Z_rz'])
    jacobian['g_rrv'] = 2*coord_der['R_rv']*(coord_der['Z_rv']*coord_der['R_r'] - coord_der['R_rv']*coord_der['Z_r']) \
                            + 2*coord_der['R_r']*(coord_der['Z_rvv']*coord_der['R_r'] - coord_der['R_rvv']*coord_der['Z_r']) \
                            + coord_der['R']*(coord_der['R_r']*coord_der['Z_rrvv'] - coord_der['Z_r']*coord_der['R_rrvv']
                                              + 2*coord_der['R_rr']*coord_der['Z_rvv'] - coord_der['R_rv']*coord_der['Z_rrv']
                                              - 2*coord_der['Z_rr']*coord_der['R_rvv'] + coord_der['Z_rv']*coord_der['R_rrv'])
    for key, val in jacobian.items():
        jacobian[key] = val.flatten()
    
    return jacobian


def compute_B_field(Psi_total, jacobian, nodes, axn, cov_basis, iotafun_params):
    """Computes magnetic field at node locations
    
    Args:
        Psi_total (float): total toroidal flux within LCFS
        jacobian (dict): dictionary of ndarray, shape(N_nodes,) of coordinate jacobian and partial derivatives
        nodes (ndarray, shape(3,N_nodes)): array of node locations in rho, vartheta, zeta coordinates
        axn (array-like): indices of nodes at the magnetic axis
        cov_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node 
        iotafun_params (array-like): parameters to pass to rotational transform function   
    Return:
        B_field (dict): dictionary of ndarray, shape(N_nodes,) of magnetic field and derivatives
    """
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi/dr^2
    # word sub or superscripts denote co and contravariant components of the field
    r = nodes[0]   
    iota = iotafun(r,0, iotafun_params)
    iotar = iotafun(r,1, iotafun_params)
    
    B_field = {}
    # B field
    B_field['psi'] = Psi_total*r**2 # could instead make Psi(r) an arbitrary function?
    B_field['psi_r']  = 2*Psi_total*r
    B_field['psi_rr'] = 2*Psi_total*jnp.ones_like(r)
    B_field['B^rho'] = jnp.zeros_like(r)
    B_field['B^zeta'] = B_field['psi_r'] / (2*jnp.pi*jacobian['g'])
    B_field['B^theta'] = iota * B_field['B^zeta']

    B_field['B_RphiZ'] = B_field['B^rho']*cov_basis['e_rho'] \
                        + B_field['B^theta']*cov_basis['e_theta'] \
                        + B_field['B^zeta']*cov_basis['e_zeta']
    
    
    # B^{zeta} derivatives
    B_field['B^zeta_r'] = B_field['psi_rr'] / (2*jnp.pi*jacobian['g']) - (B_field['psi_r']*jacobian['g_r']) / (2*jnp.pi*jacobian['g']**2)
    B_field['B^zeta_v'] = - (B_field['psi_r']*jacobian['g_v']) / (2*jnp.pi*jacobian['g']**2)
    B_field['B^zeta_z'] = - (B_field['psi_r']*jacobian['g_z']) / (2*jnp.pi*jacobian['g']**2)
    # rho=0 terms only
    B_field['B^zeta_rv'] = B_field['psi_rr']*(2*jacobian['g_rr']*jacobian['g_rv'] 
                                              - jacobian['g_r']*jacobian['g_rrv']) / (4*jnp.pi*jacobian['g_r']**3)

    # magnetic axis
    B_field['B^zeta'] = put(B_field['B^zeta'], axn, Psi_total / (jnp.pi*jacobian['g_r'][axn]))
    B_field['B^theta'] = put(B_field['B^theta'], axn, Psi_total*iota[axn] / (jnp.pi*jacobian['g_r'][axn]))
    B_field['B^zeta_r'] = put(B_field['B^zeta_r'], axn, -(B_field['psi_rr'][axn]*jacobian['g_rr'][axn]) / (4*jnp.pi*jacobian['g_r'][axn]**2))
    B_field['B^zeta_v'] = put(B_field['B^zeta_v'], axn, 0)
    B_field['B^zeta_z'] = put(B_field['B^zeta_z'], axn, -(B_field['psi_rr'][axn]*jacobian['g_rz'][axn]) / (2*jnp.pi*jacobian['g_r'][axn]**2))

    # covariant B-component derivatives
    B_field['B_theta_r'] = B_field['B^zeta_r']*dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'],cov_basis['e_theta'],0) \
                            + B_field['B^zeta']*dot(iotar*cov_basis['e_theta']+iota*cov_basis['e_rho_v']
                                                    +cov_basis['e_zeta_r'],cov_basis['e_theta'],0) \
                            + B_field['B^zeta']*dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_rho_v'],0)
    
    B_field['B_zeta_r'] = B_field['B^zeta_r']*dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_zeta'],0) \
                            + B_field['B^zeta']*dot(iotar*cov_basis['e_theta']+iota*cov_basis['e_rho_v']
                                                    +cov_basis['e_zeta_r'],cov_basis['e_zeta'],0) \
                            + B_field['B^zeta']*dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_zeta_r'],0)
    
    B_field['B_rho_v'] = B_field['B^zeta_v']*dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_rho'],0) \
                        + B_field['B^zeta']*dot(iota*cov_basis['e_theta_v'] + cov_basis['e_zeta_v'], cov_basis['e_rho'],0) \
                        + B_field['B^zeta']*dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_rho_v'],0)
    
    B_field['B_zeta_v'] = B_field['B^zeta_v']*dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_zeta'],0) \
                            + B_field['B^zeta']*dot(iota*cov_basis['e_theta_v'] + cov_basis['e_zeta_v'], cov_basis['e_zeta'],0) \
                            + B_field['B^zeta']*dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_zeta_v'],0)
    
    B_field['B_rho_z'] = B_field['B^zeta_z']*dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_rho'],0) \
                            + B_field['B^zeta']*dot(iota*cov_basis['e_theta_z'] + cov_basis['e_zeta_z'], cov_basis['e_rho'],0) \
                            + B_field['B^zeta']*dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_rho_z'],0)
    
    B_field['B_theta_z'] = B_field['B^zeta_z']*dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_theta'],0) \
                            + B_field['B^zeta']*dot(iota*cov_basis['e_theta_z'] + cov_basis['e_zeta_z'], cov_basis['e_theta'],0) \
                            + B_field['B^zeta']*dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_theta_z'],0)
    
    # need these later to evaluate axis terms
    B_field['B_zeta_rv'] = B_field['B^zeta_rv']*dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0) \
                            + B_field['B^zeta']*dot(iota*cov_basis['e_rho_vv'] + 2*cov_basis['e_zeta_rv'], cov_basis['e_zeta'],0)
    B_field['B_theta_rz'] = B_field['B^zeta_z']*dot(cov_basis['e_zeta'],cov_basis['e_rho_v'],0) \
                            + B_field['B^zeta']*(dot(cov_basis['e_zeta_z'],cov_basis['e_rho_v'],0) 
                                                 + dot(cov_basis['e_zeta'],cov_basis['e_rho_vz'],0))


    return B_field


def compute_J_field(B_field, jacobian, cov_basis, nodes, axn):
    """Computes J from B
    
    Args:
        B_field (dict): dictionary of ndarray, shape(N_nodes,) of magnetic field and derivatives    
        jacobian (dict): dictionary of ndarray, shape(N_nodes,) of coordinate jacobian and partial derivatives
        cov_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node 
        nodes (ndarray, shape(3,N_nodes)): array of node locations in rho, vartheta, zeta coordinates
        axn (array-like): indices of nodes at the magnetic axis
    
    Returns:
        J_field (dict): dictionary of ndarray, shape(N_nodes,) of current density vector at each node
    """
    # notation: superscript denotes contravariant component
    J_field = {}
    mu0 = 4*jnp.pi*1e-7
    # contravariant J-components
    J_field['J^rho'] = (B_field['B_zeta_v'] - B_field['B_theta_z'])/mu0
    J_field['J^theta'] = (B_field['B_rho_z'] - B_field['B_zeta_r'])/mu0
    J_field['J^zeta'] = (B_field['B_theta_r'] - B_field['B_rho_v'])/mu0

    # axis terms
    J_field['J^rho'] = put(J_field['J^rho'], axn, (B_field['B_zeta_rv'][axn] - B_field['B_theta_rz'][axn]) / (jacobian['g_r'][axn]))
    
    J_field['J_RphiZ'] = J_field['J^rho']*cov_basis['e_rho'] \
                        + J_field['J^theta']*cov_basis['e_theta'] \
                        + J_field['J^zeta']*cov_basis['e_zeta']
    
    
    return J_field


def compute_contravariant_basis(coord_der, cov_basis, jacobian, nodes, axn):
    """Computes contravariant basis vectors and jacobian elements
    
    Args:
        coord_der (dict): dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations
        cov_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node 
        jacobian (dict): dictionary of ndarray, shape(N_nodes,) of coordinate jacobian and partial derivatives
        nodes (ndarray, shape(3,N_nodes)): array of node locations in rho, vartheta, zeta coordinates
        axn (array-like): indices of nodes at the magnetic axis

    Returns:
        con_basis (dict): dictionary of ndarray, shape(N_nodes,) of contravariant basis vectors and jacobian elements
    
    """
    
    # notation: grad_x denotes gradient of x
    # superscript denotes contravariant component
    N_nodes = nodes[0].size
    r = nodes[0]
    
    con_basis = {}
    # contravariant basis vectors
    con_basis['grad_rho'] = cross(cov_basis['e_theta'],
                                            cov_basis['e_zeta'],0)/jacobian['g']  
    con_basis['grad_theta'] = cross(cov_basis['e_zeta'],
                                              cov_basis['e_rho'],0)/jacobian['g']  
    con_basis['grad_zeta'] = jnp.array([coord_der['0'],
                                                 -1/coord_der['R'],
                                                 coord_der['0']])

    # axis terms. need some weird indexing because we're indexing into a 2d array with 
    # a 1d array of columns where we want to overwrite stuff
    # basically this gets the linear (flattened) indices we want to overwrite
    idx0 = jnp.ones((3,axn.size))
    idx1 = jnp.ones((3,axn.size))
    idx0 = (idx0*jnp.array([[0,1,2]]).T).flatten().astype(jnp.int32)
    idx1 = (idx1*axn).flatten().astype(jnp.int32)
    con_basis['grad_rho'] = put(con_basis['grad_rho'], (idx0,idx1), (cross(cov_basis['e_rho_v'][:,axn],
                                                 cov_basis['e_zeta'][:,axn],0) / jacobian['g_r'][axn]).flatten())
    con_basis['grad_theta'] = put(con_basis['grad_theta'], (idx0,idx1), (cross(cov_basis['e_zeta'][:,axn],
                                                   cov_basis['e_rho'][:,axn],0)).flatten())

    # just different names for the same thing
    con_basis['e^rho'] = con_basis['grad_rho']
    con_basis['e^theta'] = con_basis['grad_theta']
    con_basis['e^zeta'] = con_basis['grad_zeta']
    # metric coefficients
    con_basis['g^rr'] = dot(con_basis['grad_rho'],con_basis['grad_rho'],0)
    con_basis['g^vv'] = dot(con_basis['grad_theta'],con_basis['grad_theta'],0)  
    con_basis['g^zz'] = dot(con_basis['grad_zeta'],con_basis['grad_zeta'],0)  
    con_basis['g^vz'] = dot(con_basis['grad_theta'],con_basis['grad_zeta'],0)   
    
    return con_basis


def compute_force_error_nodes(cR,cZ,zernt,nodes,presfun_params,iotafun_params,Psi_total,volumes):
    """Computes force balance error at each node
    
    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zernt (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        presfun_params (array-like): parameters to pass to pressure function
        iotafun_params (array-like): parameters to pass to rotational transform function
        Psi_total (float): total toroidal flux within LCFS
        volumes (ndarray, shape(3,N_nodes)): arc length (dr,dv,dz) along each coordinate at each node, for computing volume.
        
    Returns:
        F_rho (ndarray, shape(N_nodes,)): Radial force balance error at each node
        F_beta (ndarray, shape(N_nodes,)): Helical force balance error at each node
    """
    N_nodes = nodes[0].size
    r = nodes[0]
    axn = jnp.where(r == 0)[0]
    # value of r one step out from axis
    r1 = jnp.min(r[r != 0])
    r1idx = jnp.where(r == r1)[0]
    
    mu0 = 4*jnp.pi*1e-7
    presr = presfun(r,1, presfun_params)

    # compute coordinates, fields etc.
    coord_der = compute_coordinate_derivatives(cR,cZ,zernt)
    cov_basis = compute_covariant_basis(coord_der)
    jacobian = compute_jacobian(coord_der,cov_basis)
    B_field = compute_B_field(Psi_total, jacobian, nodes, axn, cov_basis, iotafun_params)
    J_field = compute_J_field(B_field, jacobian, cov_basis, nodes, axn)
    con_basis = compute_contravariant_basis(coord_der, cov_basis, jacobian, nodes, axn)

    # helical basis vector
    beta = B_field['B^zeta']*con_basis['e^theta'] - B_field['B^theta']*con_basis['e^zeta']

    # force balance error in radial and helical direction
    f_rho = mu0*(J_field['J^theta']*B_field['B^zeta'] - J_field['J^zeta']*B_field['B^theta']) - mu0*presr
    f_beta = mu0*J_field['J^rho']
    
    radial  = jnp.sqrt(con_basis['g^rr']) * jnp.sign(dot(con_basis['e^rho'],cov_basis['e_rho'],0));
    helical = jnp.sqrt(con_basis['g^vv']*B_field['B^zeta']**2 + con_basis['g^zz']*B_field['B^theta']**2 \
               - 2*con_basis['g^vz']*B_field['B^theta']*B_field['B^zeta']) * jnp.sign(
            dot(beta,cov_basis['e_theta'],0))*jnp.sign(dot(beta,cov_basis['e_zeta'],0));
    put(helical,axn,jnp.sqrt(con_basis['g^vv'][axn]*B_field['B^zeta'][axn]**2) * jnp.sign(B_field['B^zeta'][axn]))
    
    F_rho = f_rho * radial
    F_beta = f_beta * helical
    
    # weight by local volume
    if volumes is not None:
        vol = jacobian['g']*volumes[0]*volumes[1]*volumes[2];
        vol = put(vol, axn, jnp.mean(jacobian['g'][r1idx])/2*volumes[0,axn]*volumes[1,axn]*volumes[2,axn])
        F_rho = F_rho*vol
        F_beta = F_beta*vol
    
    return F_rho,F_beta


def compute_force_error_RphiZ(cR,cZ,zernt,nodes,presfun_params,iotafun_params,Psi_total,volumes):
    """Computes force balance error at each node
    
    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zernt (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        presfun_params (array-like): parameters to pass to pressure function
        iotafun_params (array-like): parameters to pass to rotational transform function
        Psi_total (float): total toroidal flux within LCFS
        volumes (ndarray, shape(3,N_nodes)): arc length (dr,dv,dz) along each coordinate at each node, for computing volume.
        
    Returns:
        F_err (ndarray, shape(3,N_nodes,)): F_R, F_phi, F_Z at each node
    """
    N_nodes = nodes[0].size
    r = nodes[0]
    axn = jnp.where(r == 0)[0]
    # value of r one step out from axis
    r1 = jnp.min(r[r != 0])
    r1idx = jnp.where(r == r1)[0]
    
    mu0 = 4*jnp.pi*1e-7
    presr = presfun(r,1, presfun_params)

    # compute coordinates, fields etc.
    coord_der = compute_coordinate_derivatives(cR,cZ,zernt)
    cov_basis = compute_covariant_basis(coord_der)
    jacobian = compute_jacobian(coord_der,cov_basis)
    B_field = compute_B_field(Psi_total, jacobian, nodes, axn, cov_basis, iotafun_params)
    J_field = compute_J_field(B_field, jacobian, cov_basis, nodes, axn)
    con_basis = compute_contravariant_basis(coord_der, cov_basis, jacobian, nodes, axn)

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


def compute_force_error_RddotZddot(cR,cZ,zernt,nodes,presfun_params,iotafun_params,Psi_total,volumes):
    """Computes force balance error at each node

    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zernt (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        presfun_params (array-like): parameters to pass to pressure function
        iotafun_params (array-like): parameters to pass to rotational transform function
        Psi_total (float): total toroidal flux within LCFS
        volumes (ndarray, shape(3,N_nodes)): arc length (dr,dv,dz) along each coordinate at each node, for computing volume.

    Returns:
        cRddot (ndarray, shape(Ncoeffs,)): spectral coefficients for d^2R/dt^2
        cZddot (ndarray, shape(Ncoeffs,)): spectral coefficients for d^2Z/dt^2
    """

    coord_der = compute_coordinate_derivatives(cR,cZ,zernt)
    F_err = compute_force_error_RphiZ(cR,cZ,zernt,nodes,presfun_params,iotafun_params,Psi_total,volumes)
    num_nodes = len(nodes[0])

    AR = jnp.stack([jnp.ones(num_nodes),-coord_der['R_z'],jnp.zeros(num_nodes)],axis=1)
    AZ = jnp.stack([jnp.zeros(num_nodes),-coord_der['Z_z'],jnp.ones(num_nodes)],axis=1)
    A = jnp.stack([AR,AZ],axis=1)
    Rddot, Zddot = jnp.squeeze(jnp.matmul(A,F_err.T[:,:,jnp.newaxis])).T
    
    cRddot, cZddot = zernt.fit(jnp.array([Rddot,Zddot]).T,1e-6).T
    
    return cRddot, cZddot
