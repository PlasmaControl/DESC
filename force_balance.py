import functools
from backend import jnp, conditional_decorator, jit, use_jax, fori_loop, put, cross, dot, sign, pressfun, iotafun

@conditional_decorator(functools.partial(jit,static_argnums=(2)), use_jax)
def compute_coordinate_derivatives(cR,cZ,zernt):
    """Converts from spectral to real space and evaluates derivatives of R,Z wrt to SFL coords
    
    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zernt (ZernikeTransform): object with transform method to go from spectral to physical space with derivatives
    
    Returns:
        coordinate_derivatives (dict): dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations
    """
    # notation: X_y means derivative of X wrt y
    coordinate_derivatives = {}
    coordinate_derivatives['R'] = zernt.transform(cR,0,0,0)
    coordinate_derivatives['Z'] = zernt.transform(cZ,0,0,0)
    coordinate_derivatives['0'] = jnp.zeros_like(coordinate_derivatives['R'])
    
    coordinate_derivatives['R_r'] = zernt.transform(cR,1,0,0)
    coordinate_derivatives['Z_r'] = zernt.transform(cZ,1,0,0)
    coordinate_derivatives['R_v'] = zernt.transform(cR,0,1,0)
    coordinate_derivatives['Z_v'] = zernt.transform(cZ,0,1,0)
    coordinate_derivatives['R_z'] = zernt.transform(cR,0,0,1)
    coordinate_derivatives['Z_z'] = zernt.transform(cZ,0,0,1)

    coordinate_derivatives['R_rr'] = zernt.transform(cR,2,0,0)
    coordinate_derivatives['Z_rr'] = zernt.transform(cZ,2,0,0)
    coordinate_derivatives['R_rv'] = zernt.transform(cR,1,1,0)
    coordinate_derivatives['Z_rv'] = zernt.transform(cZ,1,1,0)
    coordinate_derivatives['R_rz'] = zernt.transform(cR,1,0,1)
    coordinate_derivatives['Z_rz'] = zernt.transform(cZ,1,0,1)

    coordinate_derivatives['R_vv'] = zernt.transform(cR,0,2,0)
    coordinate_derivatives['Z_vv'] = zernt.transform(cZ,0,2,0)
    coordinate_derivatives['R_vz'] = zernt.transform(cR,0,1,1)
    coordinate_derivatives['Z_vz'] = zernt.transform(cZ,0,1,1)
    coordinate_derivatives['R_zz'] = zernt.transform(cR,0,0,2)
    coordinate_derivatives['Z_zz'] = zernt.transform(cZ,0,0,2)

    coordinate_derivatives['R_rrv'] = zernt.transform(cR,2,1,0)
    coordinate_derivatives['Z_rrv'] = zernt.transform(cZ,2,1,0)
    coordinate_derivatives['R_rvv'] = zernt.transform(cR,1,2,0)
    coordinate_derivatives['Z_rvv'] = zernt.transform(cZ,1,2,0)
    coordinate_derivatives['R_rvz'] = zernt.transform(cR,1,1,1)
    coordinate_derivatives['Z_rvz'] = zernt.transform(cZ,1,1,1)

    coordinate_derivatives['R_rrvv'] = zernt.transform(cR,2,2,0)
    coordinate_derivatives['Z_rrvv'] = zernt.transform(cZ,2,2,0)

    return coordinate_derivatives


@conditional_decorator(functools.partial(jit), use_jax)
def compute_covariant_basis(coordinate_derivatives):
    """Computes covariant basis vectors at grid points
    
    Args:
        coordinate_derivatives (dict): dictionary of ndarray, shape(N_nodes,) of the coordinate derivatives at each node
        
    Returns:
        covariant_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node
    """
    # notation: subscript word is direction of unit vector, subscript letters denote partial derivatives
    # eg, e_rho_v is the v derivative of the covariant basis vector in the rho direction
    cov_basis = {}
    cov_basis['e_rho']      = jnp.array([coordinate_derivatives['R_r'],  coordinate_derivatives['0'],   coordinate_derivatives['Z_r']])
    cov_basis['e_theta']    = jnp.array([coordinate_derivatives['R_v'],  coordinate_derivatives['0'],   coordinate_derivatives['Z_v']])
    cov_basis['e_zeta']     = jnp.array([coordinate_derivatives['R_z'], -coordinate_derivatives['R'],   coordinate_derivatives['Z_z']])

    cov_basis['e_rho_r']    = jnp.array([coordinate_derivatives['R_rr'], coordinate_derivatives['0'],   coordinate_derivatives['Z_rr']])
    cov_basis['e_rho_v']    = jnp.array([coordinate_derivatives['R_rv'], coordinate_derivatives['0'],   coordinate_derivatives['Z_rv']])
    cov_basis['e_rho_z']    = jnp.array([coordinate_derivatives['R_rz'], coordinate_derivatives['0'],   coordinate_derivatives['Z_rz']])

    cov_basis['e_theta_r']  = jnp.array([coordinate_derivatives['R_rz'], coordinate_derivatives['0'],   coordinate_derivatives['Z_rv']])
    cov_basis['e_theta_v']  = jnp.array([coordinate_derivatives['R_vv'], coordinate_derivatives['0'],   coordinate_derivatives['Z_vv']])
    cov_basis['e_theta_z']  = jnp.array([coordinate_derivatives['R_vz'], coordinate_derivatives['0'],   coordinate_derivatives['Z_vz']])

    cov_basis['e_zeta_r']  = jnp.array([coordinate_derivatives['R_rz'], -coordinate_derivatives['R_r'], coordinate_derivatives['Z_rz']])
    cov_basis['e_zeta_v']  = jnp.array([coordinate_derivatives['R_vz'], -coordinate_derivatives['R_v'], coordinate_derivatives['Z_vz']])
    cov_basis['e_zeta_z']  = jnp.array([coordinate_derivatives['R_zz'], -coordinate_derivatives['R_z'], coordinate_derivatives['Z_zz']])

    cov_basis['e_rho_vv']  = jnp.array([coordinate_derivatives['R_rvv'], coordinate_derivatives['0'],   coordinate_derivatives['Z_rvv']])
    cov_basis['e_rho_vz']  = jnp.array([coordinate_derivatives['R_rvz'], coordinate_derivatives['0'],   coordinate_derivatives['Z_rvz']])
    cov_basis['e_zeta_rv'] = jnp.array([coordinate_derivatives['R_rvz'],-coordinate_derivatives['R_rv'],coordinate_derivatives['Z_rvz']])
    
    return cov_basis


@conditional_decorator(functools.partial(jit), use_jax)
def compute_jacobian(coordinate_derivatives,covariant_basis):
    """Computes coordinate jacobian and derivatives
    
    Args:
        coordinate_derivatives (dict): dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations
        covariant_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node 
        
    Returns:
        jacobian (dict): dictionary of ndarray, shape(N_nodes,) of coordinate jacobian and partial derivatives
    """
    # notation: subscripts denote partial derivatives
    jacobian = {}    
    jacobian['g'] = dot(covariant_basis['e_rho'] , cross(covariant_basis['e_theta'],covariant_basis['e_zeta'],0),0)

    jacobian['g_r'] = dot(covariant_basis['e_rho_r'],cross(covariant_basis['e_theta'],covariant_basis['e_zeta'],0),0) \
                      + dot(covariant_basis['e_rho'],cross(covariant_basis['e_rho_v'],covariant_basis['e_zeta'],0),0) \
                      + dot(covariant_basis['e_rho'],cross(covariant_basis['e_theta'],covariant_basis['e_zeta_r'],0),0)
    jacobian['g_v'] = dot(covariant_basis['e_rho_v'],cross(covariant_basis['e_theta'],covariant_basis['e_zeta'],0),0) \
                      + dot(covariant_basis['e_rho'],cross(covariant_basis['e_theta_v'],covariant_basis['e_zeta'],0),0) \
                      + dot(covariant_basis['e_rho'],cross(covariant_basis['e_theta'],covariant_basis['e_zeta_v'],0),0)
    jacobian['g_z'] = dot(covariant_basis['e_rho_z'],cross(covariant_basis['e_theta'],covariant_basis['e_zeta'],0),0) \
                      + dot(covariant_basis['e_rho'],cross(covariant_basis['e_theta_z'],covariant_basis['e_zeta'],0),0) \
                      + dot(covariant_basis['e_rho'],cross(covariant_basis['e_theta'],covariant_basis['e_zeta_z'],0),0)
    # need these later for rho=0
    jacobian['g_rr']  = coordinate_derivatives['R']*(coordinate_derivatives['R_r']*coordinate_derivatives['Z_rrv'] 
                                                     - coordinate_derivatives['Z_r']*coordinate_derivatives['R_rrv']
                                                     + 2*coordinate_derivatives['R_rr']*coordinate_derivatives['Z_rv']
                                                     - 2*coordinate_derivatives['R_rv']*coordinate_derivatives['Z_rr']) \
                                        + 2*coordinate_derivatives['R_r']*(coordinate_derivatives['Z_rv']*coordinate_derivatives['R_r']
                                                                           - coordinate_derivatives['R_rv']*coordinate_derivatives['Z_r'])
    jacobian['g_rv']  = coordinate_derivatives['R']*(coordinate_derivatives['Z_rvv']*coordinate_derivatives['R_r']
                                                     - coordinate_derivatives['R_rvv']*coordinate_derivatives['Z_r'])
    jacobian['g_zr']  = coordinate_derivatives['R_z']*(coordinate_derivatives['R_r']*coordinate_derivatives['Z_rv'] 
                                                       - coordinate_derivatives['R_rv']*coordinate_derivatives['Z_r']) \
                            + coordinate_derivatives['R']*(coordinate_derivatives['R_rz']*coordinate_derivatives['Z_rv'] 
                                                           + coordinate_derivatives['R_r']*coordinate_derivatives['Z_rvz']
                                                           - coordinate_derivatives['R_rvz']*coordinate_derivatives['Z_r']
                                                           - coordinate_derivatives['R_rv']*coordinate_derivatives['Z_rz'])
    jacobian['g_rrv'] = 2*coordinate_derivatives['R_rv']*(coordinate_derivatives['Z_rv']*coordinate_derivatives['R_r']
                                                          - coordinate_derivatives['R_rv']*coordinate_derivatives['Z_r']) \
                            + 2*coordinate_derivatives['R_r']*(coordinate_derivatives['Z_rvv']*coordinate_derivatives['R_r']
                                                               - coordinate_derivatives['R_rvv']*coordinate_derivatives['Z_r']) \
                            + coordinate_derivatives['R']*(coordinate_derivatives['R_r']*coordinate_derivatives['Z_rrvv']
                                                           - coordinate_derivatives['Z_r']*coordinate_derivatives['R_rrvv']
                                                           + 2*coordinate_derivatives['R_rr']*coordinate_derivatives['Z_rvv']
                                                           - coordinate_derivatives['R_rv']*coordinate_derivatives['Z_rrv']
                                                           - 2*coordinate_derivatives['Z_rr']*coordinate_derivatives['R_rvv']
                                                           + coordinate_derivatives['Z_rv']*coordinate_derivatives['R_rrv'])
    for key, val in jacobian.items():
        jacobian[key] = val.flatten()
    
    return jacobian


@conditional_decorator(functools.partial(jit,static_argnums=(0,2,3,5)), use_jax)
def compute_B_field(Psi_total, jacobian, nodes, axn, covariant_basis, iotafun_params):
    """Computes magnetic field at node locations
    
    Args:
        Psi_total (float): total toroidal flux within LCFS
        jacobian (dict): dictionary of ndarray, shape(N_nodes,) of coordinate jacobian and partial derivatives
        nodes (ndarray, shape(3,N_nodes)): array of node locations in rho, vartheta, zeta coordinates
        axn (array-like): indices of nodes at the magnetic axis
        covariant_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node 
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
    B_field['B^zeta_z'] = put(B_field['B^zeta_z'], axn, -(B_field['psi_rr'][axn]*jacobian['g_zr'][axn]) / (2*jnp.pi*jacobian['g_r'][axn]**2))

    # covariant B-component derivatives
    B_field['B_theta_r'] = B_field['B^zeta_r']*dot(iota*covariant_basis['e_theta']
                                                   +covariant_basis['e_zeta'],covariant_basis['e_theta'],0) \
                            + B_field['B^zeta']*dot(iotar*covariant_basis['e_theta']+iota*covariant_basis['e_rho_v']
                                                    +covariant_basis['e_zeta_r'],covariant_basis['e_theta'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis['e_theta']+covariant_basis['e_zeta'],
                                                    covariant_basis['e_rho_v'],0)
    B_field['B_zeta_r'] = B_field['B^zeta_r']*dot(iota*covariant_basis['e_theta']+covariant_basis['e_zeta'],
                                                  covariant_basis['e_zeta'],0) \
                            + B_field['B^zeta']*dot(iotar*covariant_basis['e_theta']+iota*covariant_basis['e_rho_v']
                                                    +covariant_basis['e_zeta_r'],covariant_basis['e_zeta'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis['e_theta']+covariant_basis['e_zeta'],
                                                    covariant_basis['e_zeta_r'],0)
    B_field['B_rho_v'] = B_field['B^zeta_v']*dot(iota*covariant_basis['e_theta']+covariant_basis['e_zeta'],
                                                 covariant_basis['e_rho'],0) \
                        + B_field['B^zeta']*dot(iota*covariant_basis['e_theta_v']+covariant_basis['e_zeta_v'],
                                                covariant_basis['e_rho'],0) \
                        + B_field['B^zeta']*dot(iota*covariant_basis['e_theta']+covariant_basis['e_zeta'],
                                                covariant_basis['e_rho_v'],0)
    B_field['B_zeta_v'] = B_field['B^zeta_v']*dot(iota*covariant_basis['e_theta']+covariant_basis['e_zeta'],
                                                  covariant_basis['e_zeta'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis['e_theta_v']+covariant_basis['e_zeta_v'],
                                                    covariant_basis['e_zeta'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis['e_theta']+covariant_basis['e_zeta'],
                                                    covariant_basis['e_zeta_v'],0)
    B_field['B_rho_z'] = B_field['B^zeta_z']*dot(iota*covariant_basis['e_theta']+covariant_basis['e_zeta'],
                                                 covariant_basis['e_rho'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis['e_theta_z']+covariant_basis['e_zeta_z'],
                                                    covariant_basis['e_rho'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis['e_theta']+covariant_basis['e_zeta'],
                                                    covariant_basis['e_rho_z'],0)
    B_field['B_theta_z'] = B_field['B^zeta_z']*dot(iota*covariant_basis['e_theta']+covariant_basis['e_zeta'],
                                                   covariant_basis['e_theta'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis['e_theta_z']+covariant_basis['e_zeta_z'],
                                                    covariant_basis['e_theta'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis['e_theta']+covariant_basis['e_zeta'],
                                                    covariant_basis['e_theta_z'],0)
    # need these later to evaluate axis terms
    B_field['B_zeta_rv'] = B_field['B^zeta_rv']*dot(covariant_basis['e_zeta'],covariant_basis['e_zeta'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis['e_rho_vv']+2*covariant_basis['e_zeta_rv'],
                                                    covariant_basis['e_zeta'],0)
    B_field['B_theta_zr'] = B_field['B^zeta_z']*dot(covariant_basis['e_zeta'],covariant_basis['e_rho_v'],0) \
                            + B_field['B^zeta']*(dot(covariant_basis['e_zeta_z'],covariant_basis['e_rho_v'],0) \
                                                 + dot(covariant_basis['e_zeta'],covariant_basis['e_rho_vz'],0))

    for key, val in B_field.items():
        B_field[key] = val.flatten()

    return B_field


@conditional_decorator(functools.partial(jit,static_argnums=(2,3)), use_jax)
def compute_J_field(B_field, jacobian, nodes, axn):
    """Computes J from B
    (note it actually just computes curl(B), ie mu0*J)
    
    Args:
        B_field (dict): dictionary of ndarray, shape(N_nodes,) of magnetic field and derivatives    
        jacobian (dict): dictionary of ndarray, shape(N_nodes,) of coordinate jacobian and partial derivatives
        nodes (ndarray, shape(3,N_nodes)): array of node locations in rho, vartheta, zeta coordinates
        axn (array-like): indices of nodes at the magnetic axis
    
    Returns:
        J_field (dict): dictionary of ndarray, shape(N_nodes,) of current density vector at each node
    """
    # notation: superscript denotes contravariant component
    J_field = {}
    # contravariant J-components
    J_field['J^rho'] = (B_field['B_zeta_v'] - B_field['B_theta_z'])
    J_field['J^theta'] = (B_field['B_rho_z'] - B_field['B_zeta_r'])
    J_field['J^zeta'] = (B_field['B_theta_r'] - B_field['B_rho_v'])

    # axis terms
    J_field['J^rho'] = put(J_field['J^rho'], axn, (B_field['B_zeta_rv'][axn] - B_field['B_theta_zr'][axn]) / (jacobian['g_r'][axn]))
    
    for key, val in J_field.items():
        J_field[key] = val.flatten()
    
    return J_field


@conditional_decorator(functools.partial(jit,static_argnums=(3,4)), use_jax)
def compute_contravariant_basis(coordinate_derivatives, covariant_basis, jacobian, nodes, axn):
    """Computes contravariant basis vectors and jacobian elements
    
    Args:
        coordinate_derivatives (dict): dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations
        covariant_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node 
        jacobian (dict): dictionary of ndarray, shape(N_nodes,) of coordinate jacobian and partial derivatives
        nodes (ndarray, shape(3,N_nodes)): array of node locations in rho, vartheta, zeta coordinates
        axn (array-like): indices of nodes at the magnetic axis

    Returns:
        contravariant_basis (dict): dictionary of ndarray, shape(N_nodes,) of contravariant basis vectors and jacobian elements
    
    """
    
    # notation: grad_x denotes gradient of x
    # superscript denotes contravariant component
    N_nodes = nodes[0].size
    r = nodes[0]
    
    contravariant_basis = {}
    # contravariant basis vectors
    contravariant_basis['grad_rho'] = cross(covariant_basis['e_theta'],
                                            covariant_basis['e_zeta'],0)/jacobian['g']  
    contravariant_basis['grad_theta'] = cross(covariant_basis['e_zeta'],
                                              covariant_basis['e_rho'],0)/jacobian['g']  
    contravariant_basis['grad_zeta'] = jnp.array([coordinate_derivatives['0'],
                                                 -1/coordinate_derivatives['R'],
                                                 coordinate_derivatives['0']])

    # axis terms. need some weird indexing because we're indexing into a 2d array with 
    # a 1d array of columns where we want to overwrite stuff
    # basically this gets the linear (flattened) indices we want to overwrite
    idx0 = jnp.ones((3,axn.size))
    idx1 = jnp.ones((3,axn.size))
    idx0 = (idx0*jnp.array([[0,1,2]]).T).flatten().astype(jnp.int32)
    idx1 = (idx1*axn).flatten().astype(jnp.int32)
    contravariant_basis['grad_rho'] = put(contravariant_basis['grad_rho'], (idx0,idx1), 
                                          (cross(covariant_basis['e_rho_v'][:,axn],
                                                 covariant_basis['e_zeta'][:,axn],0) / jacobian['g_r'][axn]).flatten())
    contravariant_basis['grad_theta'] = put(contravariant_basis['grad_theta'], (idx0,idx1), 
                                            (cross(covariant_basis['e_zeta'][:,axn],
                                                   covariant_basis['e_rho'][:,axn],0)).flatten())

    # just different names for the same thing
    contravariant_basis['e^rho'] = contravariant_basis['grad_rho']
    contravariant_basis['e^theta'] = contravariant_basis['grad_theta']
    contravariant_basis['e^zeta'] = contravariant_basis['grad_zeta']
    # metric coefficients
    contravariant_basis['g^rr'] = dot(contravariant_basis['grad_rho'],contravariant_basis['grad_rho'],0)
    contravariant_basis['g^vv'] = dot(contravariant_basis['grad_theta'],contravariant_basis['grad_theta'],0)  
    contravariant_basis['g^zz'] = dot(contravariant_basis['grad_zeta'],contravariant_basis['grad_zeta'],0)  
    contravariant_basis['g^vz'] = dot(contravariant_basis['grad_theta'],contravariant_basis['grad_zeta'],0)   
    
    return contravariant_basis


@conditional_decorator(functools.partial(jit,static_argnums=(2,3,4,5,6,7)), use_jax)
def compute_force_error_nodes(cR,cZ,zernt,nodes,pressfun_params,iotafun_params,Psi_total,volumes):
    """Computes force balance error at each node
    
    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zernt (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        nodes (ndarray, shape(3,N_nodes)): coordinates (r,v,z) of the collocation points
        pressfun_params (array-like): parameters to pass to pressure function
        iotafun_params (array-like): parameters to pass to rotational transform function
        Psi_total (float): total toroidal flux within LCFS
        volumes (ndarray, shape(3,N_nodes)): arc length (dr,dv,dz) along each coordinate at each node, for computing volume.
        
    Returns:
        F_rho (ndarray, shape(N_nodes,)): radial component of force balance error at each grid point (in Newtons)
        F_beta (ndarray, shape(N_nodes,)): helical component of force balance error at each grid point (in Newtons)
    """
    
    r = nodes[0]
    axn = jnp.where(r == 0)[0]
    # value of r one step out from axis
    r1 = jnp.min(r[r != 0])
    r1idx = jnp.where(r == r1)[0]
    
    mu0 = 4*jnp.pi*1e-7
    presr = pressfun(r,1, pressfun_params)
    
    # compute coordinates, fields etc.
    coordinate_derivatives = compute_coordinate_derivatives(cR,cZ,zernt)
    covariant_basis = compute_covariant_basis(coordinate_derivatives)
    jacobian = compute_jacobian(coordinate_derivatives,covariant_basis)
    B_field = compute_B_field(Psi_total, jacobian, nodes, axn, covariant_basis, iotafun_params)
    J_field = compute_J_field(B_field, jacobian, nodes, axn)
    contravariant_basis = compute_contravariant_basis(coordinate_derivatives, covariant_basis, jacobian, nodes, axn)
    
    # helical basis vector
    beta = B_field['B^zeta']*contravariant_basis['e^theta'] - B_field['B^theta']*contravariant_basis['e^zeta']
    
    # force balance error in radial and helical direction
    f_rho = (J_field['J^theta']*B_field['B^zeta'] - J_field['J^zeta']*B_field['B^theta']) - mu0*presr
    f_beta = J_field['J^rho']
    
    radial  = jnp.sqrt(contravariant_basis['g^rr']) * jnp.sign(dot(contravariant_basis['e^rho'],covariant_basis['e_rho'],0));
    helical = jnp.sqrt(contravariant_basis['g^vv']*B_field['B^zeta']**2 + contravariant_basis['g^zz']*B_field['B^theta']**2 \
               - 2*contravariant_basis['g^vz']*B_field['B^theta']*B_field['B^zeta']) * jnp.sign(
            dot(beta,covariant_basis['e_theta'],0))*jnp.sign(dot(beta,covariant_basis['e_zeta'],0));
    put(helical,axn,jnp.sqrt(contravariant_basis['g^vv'][axn]*B_field['B^zeta'][axn]**2) * jnp.sign(B_field['B^zeta'][axn]))
    
    F_rho = f_rho * radial
    F_beta = f_beta * helical
    
    # weight by local volume
    if volumes is not None:
        vol = jacobian['g']*volumes[0]*volumes[1]*volumes[2];
        vol = put(vol, axn, jnp.mean(jacobian['g'][r1idx])/2*volumes[0,axn]*volumes[1,axn]*volumes[2,axn])
        F_rho = F_rho*vol
        F_beta = F_beta*vol
    
    return F_rho,F_beta