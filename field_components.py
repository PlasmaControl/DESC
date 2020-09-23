from backend import jnp, put, cross, dot, presfun, iotafun


def compute_coordinate_derivatives(cR,cZ,zernt,zeta_ratio=1.0):
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
    coord_der['R_z'] = zernt.transform(cR,0,0,1) * zeta_ratio
    coord_der['Z_z'] = zernt.transform(cZ,0,0,1) * zeta_ratio
    
    coord_der['R_rr'] = zernt.transform(cR,2,0,0)
    coord_der['Z_rr'] = zernt.transform(cZ,2,0,0)
    coord_der['R_rv'] = zernt.transform(cR,1,1,0)
    coord_der['Z_rv'] = zernt.transform(cZ,1,1,0)
    coord_der['R_rz'] = zernt.transform(cR,1,0,1) * zeta_ratio
    coord_der['Z_rz'] = zernt.transform(cZ,1,0,1) * zeta_ratio
    coord_der['R_vv'] = zernt.transform(cR,0,2,0)
    coord_der['Z_vv'] = zernt.transform(cZ,0,2,0)
    coord_der['R_vz'] = zernt.transform(cR,0,1,1) * zeta_ratio
    coord_der['Z_vz'] = zernt.transform(cZ,0,1,1) * zeta_ratio
    coord_der['R_zz'] = zernt.transform(cR,0,0,2) * zeta_ratio
    coord_der['Z_zz'] = zernt.transform(cZ,0,0,2) * zeta_ratio
    
    coord_der['R_rrr'] = zernt.transform(cR,3,0,0)
    coord_der['Z_rrr'] = zernt.transform(cZ,3,0,0)
    coord_der['R_rrv'] = zernt.transform(cR,2,1,0)
    coord_der['Z_rrv'] = zernt.transform(cZ,2,1,0)
    coord_der['R_rrz'] = zernt.transform(cR,2,0,1) * zeta_ratio
    coord_der['Z_rrz'] = zernt.transform(cZ,2,0,1) * zeta_ratio
    coord_der['R_rvv'] = zernt.transform(cR,1,2,0)
    coord_der['Z_rvv'] = zernt.transform(cZ,1,2,0)
    coord_der['R_rvz'] = zernt.transform(cR,1,1,1) * zeta_ratio
    coord_der['Z_rvz'] = zernt.transform(cZ,1,1,1) * zeta_ratio
    coord_der['R_rzz'] = zernt.transform(cR,1,0,2) * zeta_ratio
    coord_der['Z_rzz'] = zernt.transform(cZ,1,0,2) * zeta_ratio
    coord_der['R_vvv'] = zernt.transform(cR,0,3,0)
    coord_der['Z_vvv'] = zernt.transform(cZ,0,3,0)
    coord_der['R_vvz'] = zernt.transform(cR,0,2,1) * zeta_ratio
    coord_der['Z_vvz'] = zernt.transform(cZ,0,2,1) * zeta_ratio
    coord_der['R_vzz'] = zernt.transform(cR,0,1,2) * zeta_ratio
    coord_der['Z_vzz'] = zernt.transform(cZ,0,1,2) * zeta_ratio
    coord_der['R_zzz'] = zernt.transform(cR,0,0,3) * zeta_ratio
    coord_der['Z_zzz'] = zernt.transform(cZ,0,0,3) * zeta_ratio
    
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
    cov_basis['e_zeta']     = jnp.array([coord_der['R_z'],  coord_der['R'],   coord_der['Z_z']])
    
    cov_basis['e_rho_r']    = jnp.array([coord_der['R_rr'], coord_der['0'],   coord_der['Z_rr']])
    cov_basis['e_rho_v']    = jnp.array([coord_der['R_rv'], coord_der['0'],   coord_der['Z_rv']])
    cov_basis['e_rho_z']    = jnp.array([coord_der['R_rz'], coord_der['0'],   coord_der['Z_rz']])
    
    cov_basis['e_theta_r']  = jnp.array([coord_der['R_rv'], coord_der['0'],   coord_der['Z_rv']])
    cov_basis['e_theta_v']  = jnp.array([coord_der['R_vv'], coord_der['0'],   coord_der['Z_vv']])
    cov_basis['e_theta_z']  = jnp.array([coord_der['R_vz'], coord_der['0'],   coord_der['Z_vz']])
    
    cov_basis['e_zeta_r']   = jnp.array([coord_der['R_rz'], coord_der['R_r'], coord_der['Z_rz']])
    cov_basis['e_zeta_v']   = jnp.array([coord_der['R_vz'], coord_der['R_v'], coord_der['Z_vz']])
    cov_basis['e_zeta_z']   = jnp.array([coord_der['R_zz'], coord_der['R_z'], coord_der['Z_zz']])
    
    cov_basis['e_rho_rr']   = jnp.array([coord_der['R_rrr'],coord_der['0'],   coord_der['Z_rrr']])
    cov_basis['e_rho_rv']   = jnp.array([coord_der['R_rrv'],coord_der['0'],   coord_der['Z_rrv']])
    cov_basis['e_rho_rz']   = jnp.array([coord_der['R_rrz'],coord_der['0'],   coord_der['Z_rrz']])
    cov_basis['e_rho_vv']   = jnp.array([coord_der['R_rvv'],coord_der['0'],   coord_der['Z_rvv']])
    cov_basis['e_rho_vz']   = jnp.array([coord_der['R_rvz'],coord_der['0'],   coord_der['Z_rvz']])
    cov_basis['e_rho_zz']   = jnp.array([coord_der['R_rzz'],coord_der['0'],   coord_der['Z_rzz']])
    
    cov_basis['e_theta_rr'] = jnp.array([coord_der['R_rrv'],coord_der['0'],   coord_der['Z_rrv']])
    cov_basis['e_theta_rv'] = jnp.array([coord_der['R_rvv'],coord_der['0'],   coord_der['Z_rvv']])
    cov_basis['e_theta_rz'] = jnp.array([coord_der['R_rvz'],coord_der['0'],   coord_der['Z_rvz']])
    cov_basis['e_theta_vv'] = jnp.array([coord_der['R_vvv'],coord_der['0'],   coord_der['Z_vvv']])
    cov_basis['e_theta_vz'] = jnp.array([coord_der['R_vvz'],coord_der['0'],   coord_der['Z_vvz']])
    cov_basis['e_theta_zz'] = jnp.array([coord_der['R_vzz'],coord_der['0'],   coord_der['Z_vzz']])
    
    cov_basis['e_zeta_rr']  = jnp.array([coord_der['R_rrz'],coord_der['R_rr'],coord_der['Z_rrz']])
    cov_basis['e_zeta_rv']  = jnp.array([coord_der['R_rvz'],coord_der['R_rv'],coord_der['Z_rvz']])
    cov_basis['e_zeta_rz']  = jnp.array([coord_der['R_rzz'],coord_der['R_rz'],coord_der['Z_rzz']])
    cov_basis['e_zeta_vv']  = jnp.array([coord_der['R_vvz'],coord_der['R_vv'],coord_der['Z_vvz']])
    cov_basis['e_zeta_vz']  = jnp.array([coord_der['R_vzz'],coord_der['R_vz'],coord_der['Z_vzz']])
    cov_basis['e_zeta_zz']  = jnp.array([coord_der['R_zzz'],coord_der['R_zz'],coord_der['Z_zzz']])
    
    return cov_basis


def compute_contravariant_basis(coord_der,cov_basis,jacobian,nodes):
    """Computes contravariant basis vectors and jacobian elements
    
    Args:
        coord_der (dict): dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations
        cov_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node
        jacobian (dict): dictionary of ndarray, shape(N_nodes,) of coordinate jacobian and partial derivatives
        nodes (ndarray, shape(3,N_nodes)): array of node locations in rho, vartheta, zeta coordinates
    
    Returns:
        con_basis (dict): dictionary of ndarray, shape(N_nodes,) of contravariant basis vectors and jacobian elements
    
    """
    
    # subscripts (superscripts) denote covariant (contravariant) basis vectors
    con_basis = {}
    r = nodes[0]
    axn = jnp.where(r == 0)[0]
    
    # contravariant basis vectors
    con_basis['e^rho']   = cross(cov_basis['e_theta'],cov_basis['e_zeta'],0)/jacobian['g']
    con_basis['e^theta'] = cross(cov_basis['e_zeta'],cov_basis['e_rho'],0)/jacobian['g']
    con_basis['e^zeta']  = jnp.array([coord_der['0'],1/coord_der['R'],coord_der['0']])
    
    # axis terms
    idx0 = jnp.ones((3,axn.size)) # indexing into a 2D array with a 1D array of columns where we want to put axis terms
    idx1 = jnp.ones((3,axn.size))
    idx0 = (idx0*jnp.array([[0,1,2]]).T).flatten().astype(jnp.int32) # this gets the flattened indices we want to overwrite
    idx1 = (idx1*axn).flatten().astype(jnp.int32)
    con_basis['e^rho'] = put(con_basis['e^rho'], (idx0,idx1), (cross(cov_basis['e_theta_r'][:,axn],cov_basis['e_zeta'][:,axn],0)/jacobian['g_r'][axn]).flatten())
    # e^theta = infinite at the axis
    
    # metric coefficients
    con_basis['g^rr'] = dot(con_basis['e^rho'],  con_basis['e^rho'],  0)
    con_basis['g^rv'] = dot(con_basis['e^rho'],  con_basis['e^theta'],0)
    con_basis['g^rz'] = dot(con_basis['e^rho'],  con_basis['e^zeta'], 0)
    con_basis['g^vv'] = dot(con_basis['e^theta'],con_basis['e^theta'],0)
    con_basis['g^vz'] = dot(con_basis['e^theta'],con_basis['e^zeta'], 0)
    con_basis['g^zz'] = dot(con_basis['e^zeta'], con_basis['e^zeta'], 0)
    
    return con_basis


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
    
    jacobian['g_r'] = dot(cov_basis['e_rho_r'],cross(cov_basis['e_theta'],  cov_basis['e_zeta'],0),0) \
                    + dot(cov_basis['e_rho'],  cross(cov_basis['e_theta_r'],cov_basis['e_zeta'],0),0) \
                    + dot(cov_basis['e_rho'],  cross(cov_basis['e_theta'],  cov_basis['e_zeta_r'],0),0)
    jacobian['g_v'] = dot(cov_basis['e_rho_v'],cross(cov_basis['e_theta'],  cov_basis['e_zeta'],0),0) \
                    + dot(cov_basis['e_rho'],  cross(cov_basis['e_theta_v'],cov_basis['e_zeta'],0),0) \
                    + dot(cov_basis['e_rho'],  cross(cov_basis['e_theta'],  cov_basis['e_zeta_v'],0),0)
    jacobian['g_z'] = dot(cov_basis['e_rho_z'],cross(cov_basis['e_theta'],  cov_basis['e_zeta'],0),0) \
                    + dot(cov_basis['e_rho'],  cross(cov_basis['e_theta_z'],cov_basis['e_zeta'],0),0) \
                    + dot(cov_basis['e_rho'],  cross(cov_basis['e_theta'],  cov_basis['e_zeta_z'],0),0)
    
    jacobian['g_rr'] = dot(cov_basis['e_rho_rr'],cross(cov_basis['e_theta'],   cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta_r'], cov_basis['e_zeta'],0),0)*2 \
                     + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_r'],0),0)*2 \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_rr'],cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_r'], cov_basis['e_zeta_r'],0),0)*2 \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],   cov_basis['e_zeta_rr'],0),0)
    jacobian['g_rv'] = dot(cov_basis['e_rho_rv'],cross(cov_basis['e_theta'],   cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta_v'], cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_v'],0),0) \
                     + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta_r'], cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_rv'],cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_r'], cov_basis['e_zeta_v'],0),0) \
                     + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_r'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_v'], cov_basis['e_zeta_r'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],   cov_basis['e_zeta_rv'],0),0)
    jacobian['g_rz'] = dot(cov_basis['e_rho_rz'],cross(cov_basis['e_theta'],   cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta_z'], cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_z'],0),0) \
                     + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta_r'], cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_rz'],cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_r'], cov_basis['e_zeta_z'],0),0) \
                     + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_r'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_z'], cov_basis['e_zeta_r'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],   cov_basis['e_zeta_rz'],0),0)
    
    jacobian['g_vv'] = dot(cov_basis['e_rho_vv'],cross(cov_basis['e_theta'],   cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta_v'], cov_basis['e_zeta'],0),0)*2 \
                     + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_v'],0),0)*2 \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_vv'],cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_v'], cov_basis['e_zeta_v'],0),0)*2 \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],   cov_basis['e_zeta_vv'],0),0)
    jacobian['g_vz'] = dot(cov_basis['e_rho_vz'],cross(cov_basis['e_theta'],   cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta_z'], cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_z'],0),0) \
                     + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta_v'], cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_vz'],cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_v'], cov_basis['e_zeta_z'],0),0) \
                     + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_v'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_z'], cov_basis['e_zeta_v'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],   cov_basis['e_zeta_vz'],0),0)
    jacobian['g_zz'] = dot(cov_basis['e_rho_zz'],cross(cov_basis['e_theta'],   cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta_z'], cov_basis['e_zeta'],0),0)*2 \
                     + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_z'],0),0)*2 \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_zz'],cov_basis['e_zeta'],0),0) \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_z'], cov_basis['e_zeta_z'],0),0)*2 \
                     + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],   cov_basis['e_zeta_zz'],0),0)
    
    for key,val in jacobian.items():
        jacobian[key] = val.flatten()
    
    return jacobian


def compute_B_field(cov_basis,jacobian,cI,Psi_total,nodes):
    """Computes magnetic field components at node locations
    
    Args:
        cov_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node
        jacobian (dict): dictionary of ndarray, shape(N_nodes,) of coordinate jacobian and partial derivatives
        cI (array-like): coefficients to pass to rotational transform function
        Psi_total (float): total toroidal flux within LCFS
        nodes (ndarray, shape(3,N_nodes)): array of node locations in rho, vartheta, zeta coordinates
    
    Return:
        B_field (dict): dictionary of ndarray, shape(N_nodes,) of magnetic field and derivatives
    """
    
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    B_field = {}
    r = nodes[0]
    axn = jnp.where(r == 0)[0]
    iota   = iotafun(r,0,cI)
    iota_r = iotafun(r,1,cI)
    
    # toroidal flux
    B_field['psi']    = Psi_total*r**2
    B_field['psi_r']  = 2*Psi_total*r
    B_field['psi_rr'] = 2*Psi_total*jnp.ones_like(r)
    
    # contravariant B components
    B_field['B^rho']   = jnp.zeros_like(r)
    B_field['B^zeta']  = B_field['psi_r'] / (2*jnp.pi*jacobian['g'])
    B_field['B^zeta']  = put(B_field['B^zeta'], axn, B_field['psi_rr'][axn] / (2*jnp.pi*jacobian['g_r'][axn]))
    B_field['B^theta'] = iota * B_field['B^zeta']
    B_field['B_con'] = B_field['B^rho']*cov_basis['e_rho'] + B_field['B^theta']*cov_basis['e_theta'] + B_field['B^zeta']*cov_basis['e_zeta']
    
    # covariant B components
    B_field['B_rho']   = B_field['B^zeta']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'],cov_basis['e_rho'],0)
    B_field['B_theta'] = B_field['B^zeta']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'],cov_basis['e_theta'],0)
    B_field['B_zeta']  = B_field['B^zeta']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'],cov_basis['e_zeta'],0)
    
    # B^{zeta} derivatives
    B_field['B^zeta_r'] = B_field['psi_rr'] / (2*jnp.pi*jacobian['g']) - (B_field['psi_r']*jacobian['g_r']) / (2*jnp.pi*jacobian['g']**2)
    B_field['B^zeta_v'] = - (B_field['psi_r']*jacobian['g_v']) / (2*jnp.pi*jacobian['g']**2)
    B_field['B^zeta_z'] = - (B_field['psi_r']*jacobian['g_z']) / (2*jnp.pi*jacobian['g']**2)
    B_field['B^zeta_vv'] = - (B_field['psi_r']*jacobian['g_vv']) / (2*jnp.pi*jacobian['g']**2) \
                           + (B_field['psi_r']*jacobian['g_v']**2) / (jnp.pi*jacobian['g']**3)
    B_field['B^zeta_vz'] = - (B_field['psi_r']*jacobian['g_vz']) / (2*jnp.pi*jacobian['g']**2) \
                           + (B_field['psi_r']*jacobian['g_v']*jacobian['g_z']) / (jnp.pi*jacobian['g']**3)
    B_field['B^zeta_zz'] = - (B_field['psi_r']*jacobian['g_zz']) / (2*jnp.pi*jacobian['g']**2) \
                           + (B_field['psi_r']*jacobian['g_z']**2) / (jnp.pi*jacobian['g']**3)
    
    # axis values
    B_field['B^zeta_r'] = put(B_field['B^zeta_r'], axn, -(B_field['psi_rr'][axn]*jacobian['g_rr'][axn]) / (4*jnp.pi*jacobian['g_r'][axn]**2))
    B_field['B^zeta_v'] = put(B_field['B^zeta_v'], axn, 0)
    B_field['B^zeta_z'] = put(B_field['B^zeta_z'], axn, -(B_field['psi_rr'][axn]*jacobian['g_rz'][axn]) / (2*jnp.pi*jacobian['g_r'][axn]**2))
    
    # covariant B component derivatives
    B_field['B_theta_r'] = B_field['B^zeta_r']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_theta'],0) \
                         + B_field['B^zeta']*(dot(iota_r*cov_basis['e_theta']+iota*cov_basis['e_rho_v']+cov_basis['e_zeta_r'], cov_basis['e_theta'],0) \
                         + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho_v'],0))
    B_field['B_zeta_r']  = B_field['B^zeta_r']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_zeta'],0) \
                         + B_field['B^zeta']*(dot(iota_r*cov_basis['e_theta']+iota*cov_basis['e_rho_v']+cov_basis['e_zeta_r'], cov_basis['e_zeta'],0) \
                         + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_zeta_r'],0))
    B_field['B_rho_v']   = B_field['B^zeta_v']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho'],0) \
                         + B_field['B^zeta']*(dot(iota*cov_basis['e_theta_v']+cov_basis['e_zeta_v'], cov_basis['e_rho'],0) \
                         + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho_v'],0))
    B_field['B_zeta_v']  = B_field['B^zeta_v']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_zeta'],0) \
                         + B_field['B^zeta']*(dot(iota*cov_basis['e_theta_v']+cov_basis['e_zeta_v'], cov_basis['e_zeta'],0) \
                         + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_zeta_v'],0))
    B_field['B_rho_z']   = B_field['B^zeta_z']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho'],0) \
                         + B_field['B^zeta']*(dot(iota*cov_basis['e_theta_z']+cov_basis['e_zeta_z'], cov_basis['e_rho'],0) \
                         + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho_z'],0))
    B_field['B_theta_z'] = B_field['B^zeta_z']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_theta'],0) \
                         + B_field['B^zeta']*(dot(iota*cov_basis['e_theta_z']+cov_basis['e_zeta_z'], cov_basis['e_theta'],0) \
                         + dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_theta_z'],0))
    
    for key,val in B_field.items():
        B_field[key] = val.flatten()
    
    return B_field


def compute_J_field(coord_der,cov_basis,jacobian,B_field,cI,Psi_total,nodes):
    """Computes current density field at node locations
    
    Args:
        coord_der (dict): dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations
        cov_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node
        jacobian (dict): dictionary of ndarray, shape(N_nodes,) of coordinate jacobian and partial derivatives
        B_field (dict): dictionary of ndarray, shape(N_nodes,) of magnetic field and derivatives
        cI (array-like): coefficients to pass to rotational transform function
        Psi_total (float): total toroidal flux within LCFS
        nodes (ndarray, shape(3,N_nodes)): array of node locations in rho, vartheta, zeta coordinates
    
    Return:
        J_field (dict): dictionary of ndarray, shape(N_nodes,) of current field
    """
    
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    J_field = {}
    mu0 = 4*jnp.pi*1e-7
    r = nodes[0]
    axn = jnp.where(r == 0)[0]
    iota = iotafun(r,0,cI)
    
    # axis quantities
    g_rrv = 2*coord_der['R_rv']*(coord_der['Z_r']*coord_der['R_rv'] - coord_der['R_r']*coord_der['Z_rv']) \
                      + 2*coord_der['R_r']*(coord_der['Z_r']*coord_der['R_rvv'] - coord_der['R_r']*coord_der['Z_rvv']) \
                      + coord_der['R']*(2*coord_der['Z_rr']*coord_der['R_rvv'] - 2*coord_der['R_rr']*coord_der['Z_rvv'] \
                                        + coord_der['R_rv']*coord_der['Z_rrv'] - coord_der['Z_rv']*coord_der['R_rrv'] \
                                        + coord_der['Z_r']*coord_der['R_rrvv'] - coord_der['R_r']*coord_der['Z_rrvv'])
    Bsup_zeta_rv = B_field['psi_rr']*(2*jacobian['g_rr']*jacobian['g_rv'] - jacobian['g_r']*g_rrv) / (4*jnp.pi*jacobian['g_r']**3)
    Bsub_zeta_rv = Bsup_zeta_rv*dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0) + B_field['B^zeta']*dot(iota*cov_basis['e_rho_vv'] + 2*cov_basis['e_zeta_rv'], cov_basis['e_zeta'],0)
    Bsub_theta_rz = B_field['B^zeta_z']*dot(cov_basis['e_zeta'],cov_basis['e_rho_v'],0) + B_field['B^zeta']*(dot(cov_basis['e_zeta_z'],cov_basis['e_rho_v'],0) + dot(cov_basis['e_zeta'],cov_basis['e_rho_vz'],0))
    
    # contravariant J components
    J_field['J^rho']   = (B_field['B_zeta_v']  - B_field['B_theta_z']) / (mu0*jacobian['g'])
    J_field['J^theta'] = (B_field['B_rho_z']   - B_field['B_zeta_r'])  / (mu0*jacobian['g'])
    J_field['J^zeta']  = (B_field['B_theta_r'] - B_field['B_rho_v'])   / (mu0*jacobian['g'])
    
    # axis values
    J_field['J^rho'] = put(J_field['J^rho'], axn, (Bsub_zeta_rv[axn] - Bsub_theta_rz[axn]) / (jacobian['g_r'][axn]))
    
    J_field['J_con'] = J_field['J^rho']*cov_basis['e_rho'] + J_field['J^theta']*cov_basis['e_theta'] + J_field['J^zeta']*cov_basis['e_zeta']
    
    for key,val in J_field.items():
        J_field[key] = val.flatten()
    
    return J_field


def compute_B_magnitude(cov_basis,B_field,cI,nodes):
    """Computes magnetic field magnitude at node locations
    
    Args:
        cov_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node
        B_field (dict): dictionary of ndarray, shape(N_nodes,) of magnetic field and derivatives
        cI (array-like): coefficients to pass to rotational transform function
        nodes (ndarray, shape(3,N_nodes)): array of node locations in rho, vartheta, zeta coordinates
    Return:
        B_mag (dict): dictionary of ndarray, shape(N_nodes,) of magnetic field magnitude and derivatives
    """
    
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    B_mag = {}
    r = nodes[0]
    axn = jnp.where(r == 0)[0]
    iota = iotafun(r,0,cI)
    
    B_mag['|B|'] = jnp.abs(B_field['B^\zeta'])*jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta']) + 2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta']) + dot(cov_basis['e_zeta'],cov_basis['e_zeta']))
    
    B_mag['|B|_v'] = jnp.sign(B_field['B^zeta'])*B_field['B^zeta_v']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0)) \
                         + jnp.abs(B_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta_v'],0)+2*iota*(dot(cov_basis['e_theta_v'],cov_basis['e_zeta'],0)+dot(cov_basis['e_theta'],cov_basis['e_zeta_v'],0))+2*dot(cov_basis['e_zeta'],cov_basis['e_zeta_v'],0)) \
                         / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0)))
    
    B_mag['|B|_z']  = jnp.sign(B_field['B^zeta'])*B_field['B^zeta_z']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0)) \
          + jnp.abs(B_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta_z'],0)+2*iota*(dot(cov_basis['e_theta_z'],cov_basis['e_zeta'],0)+dot(cov_basis['e_theta'],cov_basis['e_zeta_z'],0))+2*dot(cov_basis['e_zeta'],cov_basis['e_zeta_z'],0)) \
          / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0)))
    
    B_mag['|B|_vv'] = jnp.sign(B_field['B^zeta'])*B_field['B^zeta_vv']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0)) \
          + jnp.sign(B_field['B^zeta'])*B_field['B^zeta_v']*(2*iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta_v'],0)+2*iota*(dot(cov_basis['e_theta_v'],cov_basis['e_zeta'],0)+dot(cov_basis['e_theta'],cov_basis['e_zeta_v'],0))+2*dot(cov_basis['e_zeta'],cov_basis['e_zeta_v'],0)) \
          / jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0)) \
          + jnp.abs(B_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_v'],cov_basis['e_theta_v'],0)+dot(cov_basis['e_theta'],cov_basis['e_theta_vv'],0))+2*iota*(dot(cov_basis['e_theta_vv'],cov_basis['e_zeta'],0)+dot(cov_basis['e_theta'],cov_basis['e_zeta_vv'],0)+2*dot(cov_basis['e_theta_v'],cov_basis['e_zeta_v'],0))+2*(dot(cov_basis['e_zeta_v'],cov_basis['e_zeta_v'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta_vv'],0))) \
          / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0))) \
          + jnp.abs(B_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta_v'],0)+2*iota*(dot(cov_basis['e_theta_v'],cov_basis['e_zeta'],0)+dot(cov_basis['e_theta'],cov_basis['e_zeta_v'],0))+2*dot(cov_basis['e_zeta'],cov_basis['e_zeta_v'],0))**2 \
          / (2*(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0))**(3/2))
    
    B_mag['|B|_zz'] = jnp.sign(B_field['B^zeta'])*B_field['B^zeta_zz']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0)) \
          + jnp.sign(B_field['B^zeta'])*B_field['B^zeta_z']*(2*iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta_z'],0)+2*iota*(dot(cov_basis['e_theta_z'],cov_basis['e_zeta'],0)+dot(cov_basis['e_theta'],cov_basis['e_zeta_z'],0))+2*dot(cov_basis['e_zeta'],cov_basis['e_zeta_z'],0)) \
          / jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0)) \
          + jnp.abs(B_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_z'],cov_basis['e_theta_z'],0)+dot(cov_basis['e_theta'],cov_basis['e_theta_zz'],0))+2*iota*(dot(cov_basis['e_theta_zz'],cov_basis['e_zeta'],0)+dot(cov_basis['e_theta'],cov_basis['e_zeta_zz'],0)+2*dot(cov_basis['e_theta_z'],cov_basis['e_zeta_z'],0))+2*(dot(cov_basis['e_zeta_z'],cov_basis['e_zeta_z'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta_vz'],0))) \
          / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0))) \
          + jnp.abs(B_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta_z'],0)+2*iota*(dot(cov_basis['e_theta_z'],cov_basis['e_zeta'],0)+dot(cov_basis['e_theta'],cov_basis['e_zeta_z'],0))+2*dot(cov_basis['e_zeta'],cov_basis['e_zeta_z'],0))**2 \
          / (2*(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0))**(3/2))
    
    B_mag['|B|_vz'] = jnp.sign(B_field['B^zeta'])*B_field['B^zeta_vz']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0)) \
          + jnp.sign(B_field['B^zeta'])*B_field['B^zeta_v']*(2*iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta_z'],0)+2*iota*(dot(cov_basis['e_theta_z'],cov_basis['e_zeta'],0)+dot(cov_basis['e_theta'],cov_basis['e_zeta_z'],0))+2*dot(cov_basis['e_zeta'],cov_basis['e_zeta_z'],0)) \
          / jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0)) \
          + jnp.abs(B_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_z'],cov_basis['e_theta_v'],0)+dot(cov_basis['e_theta'],cov_basis['e_theta_vz'],0))+2*iota*(dot(cov_basis['e_theta_vz'],cov_basis['e_zeta'],0)+dot(cov_basis['e_theta_v'],cov_basis['e_zeta_z'],0)+dot(cov_basis['e_theta_z'],cov_basis['e_zeta_v'],0)+dot(cov_basis['e_theta'],cov_basis['e_zeta_vz'],0))+2*(dot(cov_basis['e_zeta_z'],cov_basis['e_zeta_v'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta_vz'],0))) \
          / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0))) \
          + jnp.abs(B_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta_v'],0)+2*iota*(dot(cov_basis['e_theta_v'],cov_basis['e_zeta'],0)+dot(cov_basis['e_theta'],cov_basis['e_zeta_v'],0))+2*dot(cov_basis['e_zeta'],cov_basis['e_zeta_v'],0))*(2*iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta_z'],0)+2*iota*(dot(cov_basis['e_theta_z'],cov_basis['e_zeta'],0)+dot(cov_basis['e_theta'],cov_basis['e_zeta_z'],0))+2*dot(cov_basis['e_zeta'],cov_basis['e_zeta_z'],0)) \
          / (2*(iota**2*dot(cov_basis['e_theta'],cov_basis['e_theta'],0)+2*iota*dot(cov_basis['e_theta'],cov_basis['e_zeta'],0)+dot(cov_basis['e_zeta'],cov_basis['e_zeta'],0))**(3/2))
    
    # TODO: axis values
    
    for key,val in B_mag.items():
        B_mag[key] = val.flatten()
    
    return B_mag


def compute_F_magnitude(coord_der,cov_basis,con_basis,jacobian,B_field,J_field,cP,cI,Psi_total,nodes):
    """Computes force error magnitude at node locations
    
    Args:
        coord_der (dict): dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations
        cov_basis (dict): dictionary of ndarray, shape(N_nodes,) of covariant basis vectors and derivatives at each node
        con_basis (dict): dictionary of ndarray, shape(N_nodes,) of contravariant basis vectors and metric elements at each node
        jacobian (dict): dictionary of ndarray, shape(N_nodes,) of coordinate jacobian and partial derivatives
        B_field (dict): dictionary of ndarray, shape(N_nodes,) of magnetic field and derivatives
        J_field (dict): dictionary of ndarray, shape(N_nodes,) of current field
        cP (array-like): parameters to pass to pressure function
        cI (array-like): parameters to pass to rotational transform function
        Psi_total (float): total toroidal flux within LCFS
        nodes (ndarray, shape(3,N_nodes)): array of node locations in rho, vartheta, zeta coordinates
        
    Return:
        F (ndarray, shape(N_nodes,)): force error magnitudes
    """
    
    mu0 = 4*jnp.pi*1e-7
    r = nodes[0]
    axn = jnp.where(r == 0)[0]
    pres_r = presfun(r,1, cP)
    
    # force balance error covariant components
    F_rho   = jacobian['g']*(J_field['J^theta']*B_field['B^zeta'] - J_field['J^zeta']*B_field['B^theta']) - pres_r
    F_theta = jacobian['g']*J_field['J^rho']*B_field['B^zeta']
    F_zeta  = -jacobian['g']*J_field['J^rho']*B_field['B^theta']
    
    # axis terms
    Jsup_theta = (B_field['B_rho_z']   - B_field['B_zeta_r']) / mu0
    Jsup_zeta  = (B_field['B_theta_r'] - B_field['B_rho_v'])  / mu0
    F_rho      = put(F_rho,axn, Jsup_theta[axn]*B_field['B^zeta'][axn] - Jsup_zeta[axn]*B_field['B^theta'][axn])
    grad_theta = cross(cov_basis['e_zeta'],cov_basis['e_rho'],0)
    gsup_vv    = dot(grad_theta,grad_theta,0)
    gsup_rv    = dot(con_basis['e^rho'],grad_theta,0)
    gsup_vz    = dot(grad_theta,con_basis['e^zeta'],0)
    F_theta    = put(F_theta,axn, J_field['J^rho'][axn]*B_field['B^zeta'][axn])
    F_zeta     = put(F_zeta,axn, -J_field['J^rho'][axn]*B_field['B^theta'][axn])
    con_basis['g^vv'] = put(con_basis['g^vv'],axn, gsup_vv[axn])
    con_basis['g^rv'] = put(con_basis['g^rv'],axn, gsup_rv[axn])
    con_basis['g^vz'] = put(con_basis['g^vz'],axn, gsup_vz[axn])
    
    # F_i*F_j*g^ij terms
    Fg_rr = F_rho*  F_rho*  con_basis['g^rr']
    Fg_vv = F_theta*F_theta*con_basis['g^vv']
    Fg_zz = F_zeta* F_zeta* con_basis['g^zz']
    Fg_rv = F_rho*  F_theta*con_basis['g^rv']
    Fg_rz = F_rho*  F_zeta* con_basis['g^rz']
    Fg_vz = F_theta*F_zeta* con_basis['g^vz']
    
    # magnitudes
    F_mag = jnp.sqrt(Fg_rr + Fg_vv + Fg_zz + 2*Fg_rv + 2*Fg_rz + 2*Fg_vz)
    p_mag = jnp.sqrt(pres_r*pres_r*con_basis['g^rr'])
    
    return F_mag, p_mag
