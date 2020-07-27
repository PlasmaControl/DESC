import numpy as np
from utils import cross, dot


def force_error_nodes(cR,cZ,zernt,nodes,pressfun,iotafun,Psi_total,dr=None,dv=None,dz=None):
    """Computes force balance error at each node
    
    Args:
        cR (array-like): spectral coefficients of R
        cZ (array-like): spectral coefficients of Z
        zernt (ZernikeTransform): object with tranform method to convert from spectral basis to physical basis at nodes
        pressfun (callable): mu0*pressure = pressfun(x,nu). First argument, x, is points to evaluate pressure at. 
            Second argument, nu, is the order of derivative to evaluate. Should return pressure scaled by mu0.
        iotafun (callable): rotational transform = iotafun(x,nu). First argument, x, is points to evaluate rotational transform at. 
            Second argument, nu, is the order of derivative to evaluate.
        Psi_total (float): total toroidal flux within LCFS
        dr,dv,dz (array-like): arc length along each coordinate at each node, for computing volume. If None, no volume weighting is done.
        
    Returns:
        F_err (array-like): R,phi,Z components of force balance error at each grid point
    """
    N_nodes = nodes[0].size
    r = nodes[0]
    axn = np.where(r == 0)[0]
    # value of r one step out from axis
    r1 = np.min(r[r != 0])
    r1idx = np.where(r == r1)[0]
    
    pres = pressfun(r)
    presr = pressfun(r,nu=1)

    # compute coordinates, fields etc.
    coordinate_derivatives = compute_coordinate_derivatives(cR,cZ,zernt)
    covariant_basis_vectors = compute_covariant_basis_vectors(coordinate_derivatives)
    jacobian = compute_jacobian(coordinate_derivatives,covariant_basis_vectors)
    B_field = compute_B_field(Psi_total, jacobian, nodes, covariant_basis_vectors, iotafun)
    J_field = compute_J_field(B_field, jacobian, nodes)
    contravariant_basis = compute_contravariant_basis(covariant_basis_vectors, jacobian, nodes)

    # helical basis vector
    beta = B_field['B^zeta']*contravariant_basis['e^theta'] - B_field['B^theta']*contravariant_basis['e^zeta']

    # force balance error in radial and helical direction
    Frho = (J_field['J^theta']*B_field['B^zeta'] - J_field['J^zeta']*B_field['B^theta']) - presr
    Fbeta = J_field['J^rho']
    
    # force balance error in R,phi,Z
    F_err = Frho*contravariant_basis['grad_rho'] + Fbeta*beta
    
    # weight by local volume
    if dr is not None and dv is not None and dz is not None:
        vol = jacobian['g']*dr*dv*dz;
        if axn.size:
            vol[axn] = np.mean(jacobian['g'][r1idx])/2*dr[axn]*dv[axn]*dz[axn];
        F_err = F_err*vol
        
    return F_err


def compute_coordinate_derivatives(cR,cZ,zernt):
    """Converts from spectral to real space and evaluates derivatives of R,Z wrt to SFL coords
    
    Args:
        cR (array-like): spectral coefficients of R
        cZ (array-like): spectral coefficients of Z
        zernt (ZernikeTransform): object with transform method to go from spectral to physical space with derivatives
    
    Returns:
        coordinate_derivatives (dict): dictionary of arrays of coordinate derivatives evaluated at node locations
    """
    
    # notation: X_y means derivative of X wrt y
    coordinate_derivatives = {}
    coordinate_derivatives['R'] = zernt.transform(cR)
    coordinate_derivatives['Z'] = zernt.transform(cZ)
    coordinate_derivatives['0'] = np.zeros_like(coordinate_derivatives['R'])
    
    coordinate_derivatives['R_r'] = zernt.transform(cR,dr=1)
    coordinate_derivatives['Z_r'] = zernt.transform(cZ,dr=1)
    coordinate_derivatives['R_v'] = zernt.transform(cR,dv=1)
    coordinate_derivatives['Z_v'] = zernt.transform(cZ,dv=1)
    coordinate_derivatives['R_z'] = zernt.transform(cR,dz=1)
    coordinate_derivatives['Z_z'] = zernt.transform(cZ,dz=1)

    coordinate_derivatives['R_rr'] = zernt.transform(cR,dr=2)
    coordinate_derivatives['Z_rr'] = zernt.transform(cZ,dr=2)
    coordinate_derivatives['R_rv'] = zernt.transform(cR,dr=1,dv=1)
    coordinate_derivatives['Z_rv'] = zernt.transform(cZ,dr=1,dv=1)
    coordinate_derivatives['R_rz'] = zernt.transform(cR,dr=1,dz=1)
    coordinate_derivatives['Z_rz'] = zernt.transform(cZ,dr=1,dz=1)

    coordinate_derivatives['R_vr'] = zernt.transform(cR,dr=1,dv=1)
    coordinate_derivatives['Z_vr'] = zernt.transform(cZ,dr=1,dv=1)
    coordinate_derivatives['R_vv'] = zernt.transform(cR,dv=2)
    coordinate_derivatives['Z_vv'] = zernt.transform(cZ,dv=2)
    coordinate_derivatives['R_vz'] = zernt.transform(cR,dv=1,dz=1)
    coordinate_derivatives['Z_vz'] = zernt.transform(cZ,dv=1,dz=1)

    coordinate_derivatives['R_zr'] = zernt.transform(cR,dr=1,dz=1)
    coordinate_derivatives['Z_zr'] = zernt.transform(cZ,dr=1,dz=1)
    coordinate_derivatives['R_zv'] = zernt.transform(cR,dv=1,dz=1)
    coordinate_derivatives['Z_zv'] = zernt.transform(cZ,dv=1,dz=1)
    coordinate_derivatives['R_zz'] = zernt.transform(cR,dz=2)
    coordinate_derivatives['Z_zz'] = zernt.transform(cZ,dz=2)

    coordinate_derivatives['R_rrv'] = zernt.transform(cR,dr=2,dv=1)
    coordinate_derivatives['Z_rrv'] = zernt.transform(cZ,dr=2,dv=1)
    coordinate_derivatives['R_rvv'] = zernt.transform(cR,dr=1,dv=2)
    coordinate_derivatives['Z_rvv'] = zernt.transform(cZ,dr=1,dv=2)
    coordinate_derivatives['R_zrv'] = zernt.transform(cR,dr=1,dv=1,dz=1)
    coordinate_derivatives['Z_zrv'] = zernt.transform(cZ,dr=1,dv=1,dz=1)

    coordinate_derivatives['R_rrvv'] = zernt.transform(cR,dr=2,dv=2)
    coordinate_derivatives['Z_rrvv'] = zernt.transform(cZ,dr=2,dv=2)
    
    return coordinate_derivatives



def compute_covariant_basis_vectors(coordinate_derivatives):
    """Computes covariant basis vectors at grid points
    
    Args:
        coordinate_derivatives (dict): dictionary of arrays of the coordinate derivatives at each node
        
    Returns:
        covariant_basis (dict): dictionary of arrays of covariant basis vectors and derivatives at each node
    """
    # notation: first subscript is direction of unit vector, others denote partial derivatives
    # eg, e_rv is the v derivative of the covariant basis vector in the r direction
    cov_basis = {}
    cov_basis['e_r'] = np.array([coordinate_derivatives['R_r'],coordinate_derivatives['0'],coordinate_derivatives['Z_r']])
    cov_basis['e_v'] = np.array([coordinate_derivatives['R_v'],coordinate_derivatives['0'],coordinate_derivatives['Z_v']])
    cov_basis['e_z'] = np.array([coordinate_derivatives['R_z'],coordinate_derivatives['R'],coordinate_derivatives['Z_z']])

    cov_basis['e_rr'] = np.array([coordinate_derivatives['R_rr'],coordinate_derivatives['0'],coordinate_derivatives['Z_rr']])
    cov_basis['e_rv'] = np.array([coordinate_derivatives['R_rv'],coordinate_derivatives['0'],coordinate_derivatives['Z_rv']])
    cov_basis['e_rz'] = np.array([coordinate_derivatives['R_rz'],coordinate_derivatives['0'],coordinate_derivatives['Z_rz']])

    cov_basis['e_vr'] = np.array([coordinate_derivatives['R_vr'],coordinate_derivatives['0'],coordinate_derivatives['Z_vr']])
    cov_basis['e_vv'] = np.array([coordinate_derivatives['R_vv'],coordinate_derivatives['0'],coordinate_derivatives['Z_vv']])
    cov_basis['e_vz'] = np.array([coordinate_derivatives['R_vz'],coordinate_derivatives['0'],coordinate_derivatives['Z_vz']])

    cov_basis['e_zr'] = np.array([coordinate_derivatives['R_zr'],coordinate_derivatives['R_r'],coordinate_derivatives['Z_zr']])
    cov_basis['e_zv'] = np.array([coordinate_derivatives['R_zv'],coordinate_derivatives['R_v'],coordinate_derivatives['Z_zv']])
    cov_basis['e_zz'] = np.array([coordinate_derivatives['R_zz'],coordinate_derivatives['R_z'],coordinate_derivatives['Z_zz']])

    cov_basis['e_rvv'] = np.array([coordinate_derivatives['R_rvv'],coordinate_derivatives['0'],coordinate_derivatives['Z_rvv']])
    cov_basis['e_rvz'] = np.array([coordinate_derivatives['R_zrv'],coordinate_derivatives['0'],coordinate_derivatives['Z_zrv']])
    cov_basis['e_zrv'] = np.array([coordinate_derivatives['R_zrv'],-coordinate_derivatives['R_rv'],coordinate_derivatives['Z_zrv']])
    
    return cov_basis


def compute_jacobian(coordinate_derivatives,covariant_basis_vectors):
    """Computes coordinate jacobian and derivatives
    
    Args:
        coordinate_derivatives (dict): dictionary of arrays of coordinate derivatives evaluated at node locations
        covariant_basis_vectors (dict): dictionary of arrays of covariant basis vectors and derivatives at each node 
        
    Returns:
        jacobian (dict): dictionary of arrays of coordinate jacobian and partial derivatives
    """
    # notation: subscripts denote partial derivatives
    jacobian = {}    
    jacobian['g'] = dot(covariant_basis_vectors['e_r'] , cross(covariant_basis_vectors['e_v'],covariant_basis_vectors['e_z'],0),0)

    jacobian['g_r'] = dot(covariant_basis_vectors['e_rr'],cross(covariant_basis_vectors['e_v'],covariant_basis_vectors['e_z'],0),0) \
                      + dot(covariant_basis_vectors['e_r'],cross(covariant_basis_vectors['e_rv'],covariant_basis_vectors['e_z'],0),0) \
                      + dot(covariant_basis_vectors['e_r'],cross(covariant_basis_vectors['e_v'],covariant_basis_vectors['e_zr'],0),0)
    jacobian['g_v'] = dot(covariant_basis_vectors['e_rv'],cross(covariant_basis_vectors['e_v'],covariant_basis_vectors['e_z'],0),0) \
                      + dot(covariant_basis_vectors['e_r'],cross(covariant_basis_vectors['e_vv'],covariant_basis_vectors['e_z'],0),0) \
                      + dot(covariant_basis_vectors['e_r'],cross(covariant_basis_vectors['e_v'],covariant_basis_vectors['e_zv'],0),0)
    jacobian['g_z'] = dot(covariant_basis_vectors['e_rz'],cross(covariant_basis_vectors['e_v'],covariant_basis_vectors['e_z'],0),0) \
                      + dot(covariant_basis_vectors['e_r'],cross(covariant_basis_vectors['e_vz'],covariant_basis_vectors['e_z'],0),0) \
                      + dot(covariant_basis_vectors['e_r'],cross(covariant_basis_vectors['e_v'],covariant_basis_vectors['e_zz'],0),0)
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
                            + coordinate_derivatives['R']*(coordinate_derivatives['R_zr']*coordinate_derivatives['Z_rv'] 
                                                           + coordinate_derivatives['R_r']*coordinate_derivatives['Z_zrv']
                                                           - coordinate_derivatives['R_zrv']*coordinate_derivatives['Z_r']
                                                           - coordinate_derivatives['R_rv']*coordinate_derivatives['Z_zr'])
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


def compute_B_field(Psi_total, jacobian, nodes, covariant_basis_vectors, iotafun):
    """Computes magnetic field at node locations
    
    Args:
        Psi_total (float): total toroidal flux within LCFS
        jacobian (dict): dictionary of arrays of coordinate jacobian and partial derivatives
        nodes (array-like): array of node locations in rho, vartheta, zeta coordinates
        covariant_basis_vectors (dict): dictionary of arrays of covariant basis vectors and derivatives at each node 
        iotafun (callable): rotational transform = iotafun(x,nu). First argument, x, is points to evaluate rotational transform at. 
            Second argument, nu, is the order of derivative to evaluate.
    
    Return:
        B_field (dict): dictionary of values of magnetic field and derivatives
    """
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi/dr^2
    # word sub or superscripts denote co and contravariant components of the field
    N_nodes = nodes[0].size
    r = nodes[0]   
    axn = np.where(r == 0)[0];

    iota = iotafun(r,nu=0)
    iotar = iotafun(r,nu=1)
    B_field = {}
    # B field
    B_field['psi'] = Psi_total*r**2 # could instead make Psi(r) an arbitrary function?
    B_field['psi_r']  = 2*Psi_total*r
    B_field['psi_rr'] = 2*Psi_total*np.ones_like(r)
    B_field['B^zeta'] = B_field['psi_r'] / (2*np.pi*jacobian['g'])
    B_field['B^theta'] = iota * B_field['B^zeta']

    # B^{zeta} derivatives
    B_field['B^zeta_r'] = B_field['psi_rr'] / (2*np.pi*jacobian['g']) - (B_field['psi_r']*jacobian['g_r']) / (2*np.pi*jacobian['g']**2)
    B_field['B^zeta_v'] = - (B_field['psi_r']*jacobian['g_v']) / (2*np.pi*jacobian['g']**2)
    B_field['B^zeta_z'] = - (B_field['psi_r']*jacobian['g_z']) / (2*np.pi*jacobian['g']**2)
    # rho=0 terms only
    B_field['B^zeta_rv'] = B_field['psi_rr']*(2*jacobian['g_rr']*jacobian['g_rv'] 
                                              - jacobian['g_r']*jacobian['g_rrv']) / (4*np.pi*jacobian['g_r']**3)

    # magnetic axis
    if axn.size:
        B_field['B^zeta'][axn]  = Psi_total / (np.pi*jacobian['g_r'][axn])
        B_field['B^theta'][axn]  = Psi_total*iota[axn] / (np.pi*jacobian['g_r'][axn])
        B_field['B^zeta_r'][axn] = - (B_field['psi_rr'][axn]*jacobian['g_rr'][axn]) / (4*np.pi*jacobian['g_r'][axn]**2)
        B_field['B^zeta_v'][axn] = 0
        B_field['B^zeta_z'][axn] = - (B_field['psi_rr'][axn]*jacobian['g_zr'][axn]) / (2*np.pi*jacobian['g_r'][axn]**2)

    # covariant B-component derivatives
    B_field['B_theta_r'] = B_field['B^zeta_r']*dot(iota*covariant_basis_vectors['e_v']
                                                   +covariant_basis_vectors['e_z'],covariant_basis_vectors['e_v'],0) \
                            + B_field['B^zeta']*dot(iotar*covariant_basis_vectors['e_v']+iota*covariant_basis_vectors['e_rv']
                                                    +covariant_basis_vectors['e_zr'],covariant_basis_vectors['e_v'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis_vectors['e_v']+covariant_basis_vectors['e_z'],
                                                    covariant_basis_vectors['e_rv'],0)
    B_field['B_zeta_r'] = B_field['B^zeta_r']*dot(iota*covariant_basis_vectors['e_v']+covariant_basis_vectors['e_z'],
                                                  covariant_basis_vectors['e_z'],0) \
                            + B_field['B^zeta']*dot(iotar*covariant_basis_vectors['e_v']+iota*covariant_basis_vectors['e_rv']
                                                    +covariant_basis_vectors['e_zr'],covariant_basis_vectors['e_z'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis_vectors['e_v']+covariant_basis_vectors['e_z'],
                                                    covariant_basis_vectors['e_zr'],0)
    B_field['B_rho_v'] = B_field['B^zeta_v']*dot(iota*covariant_basis_vectors['e_v']+covariant_basis_vectors['e_z'],
                                                 covariant_basis_vectors['e_r'],0) \
                        + B_field['B^zeta']*dot(iota*covariant_basis_vectors['e_vv']+covariant_basis_vectors['e_zv'],
                                                covariant_basis_vectors['e_r'],0) \
                        + B_field['B^zeta']*dot(iota*covariant_basis_vectors['e_v']+covariant_basis_vectors['e_z'],
                                                covariant_basis_vectors['e_rv'],0)
    B_field['B_zeta_v'] = B_field['B^zeta_v']*dot(iota*covariant_basis_vectors['e_v']+covariant_basis_vectors['e_z'],
                                                  covariant_basis_vectors['e_z'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis_vectors['e_vv']+covariant_basis_vectors['e_zv'],
                                                    covariant_basis_vectors['e_z'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis_vectors['e_v']+covariant_basis_vectors['e_z'],
                                                    covariant_basis_vectors['e_zv'],0)
    B_field['B_rho_z'] = B_field['B^zeta_z']*dot(iota*covariant_basis_vectors['e_v']+covariant_basis_vectors['e_z'],
                                                 covariant_basis_vectors['e_r'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis_vectors['e_vz']+covariant_basis_vectors['e_zz'],
                                                    covariant_basis_vectors['e_r'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis_vectors['e_v']+covariant_basis_vectors['e_z'],
                                                    covariant_basis_vectors['e_rz'],0)
    B_field['B_theta_z'] = B_field['B^zeta_z']*dot(iota*covariant_basis_vectors['e_v']+covariant_basis_vectors['e_z'],
                                                   covariant_basis_vectors['e_v'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis_vectors['e_vz']+covariant_basis_vectors['e_zz'],
                                                    covariant_basis_vectors['e_v'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis_vectors['e_v']+covariant_basis_vectors['e_z'],
                                                    covariant_basis_vectors['e_vz'],0)
    # need these later to evaluate axis terms
    B_field['B_zeta_rv'] = B_field['B^zeta_rv']*dot(covariant_basis_vectors['e_z'],covariant_basis_vectors['e_z'],0) \
                            + B_field['B^zeta']*dot(iota*covariant_basis_vectors['e_rvv']+2*covariant_basis_vectors['e_zrv'],
                                                    covariant_basis_vectors['e_z'],0)
    B_field['B_theta_zr'] = B_field['B^zeta_z']*dot(covariant_basis_vectors['e_z'],covariant_basis_vectors['e_rv'],0) \
                            + B_field['B^zeta']*(dot(covariant_basis_vectors['e_zz'],covariant_basis_vectors['e_rv'],0) \
                                                 + dot(covariant_basis_vectors['e_z'],covariant_basis_vectors['e_rvz'],0))

    for key, val in B_field.items():
        B_field[key] = val.flatten()

    return B_field


def compute_J_field(B_field, jacobian, nodes):
    """Computes J from B
    (note it actually just computes curl(B), ie mu0*J)
    
    Args:
        B_field (dict): dictionary of values of magnetic field and derivatives    
        jacobian (dict): dictionary of arrays of coordinate jacobian and partial derivatives
        nodes (array-like): array of node locations in rho, vartheta, zeta coordinates
    
    Returns:
        J_field (dict): dictionary of arrays of current density vector at each node
    """
    # notation: superscript denotes contravariant component
    N_nodes = nodes[0].size
    r = nodes[0]
    axn = np.where(r == 0)[0]
    
    J_field = {}
    # contravariant J-components
    J_field['J^rho'] = (B_field['B_zeta_v'] - B_field['B_theta_z'])
    J_field['J^theta'] = (B_field['B_rho_z'] - B_field['B_zeta_r'])
    J_field['J^zeta'] = (B_field['B_theta_r'] - B_field['B_rho_v'])

    if axn.size:
        J_field['J^rho'][axn] = (B_field['B_zeta_rv'][axn] - B_field['B_theta_zr'][axn]) / (jacobian['g_r'][axn])
    
    for key, val in J_field.items():
        J_field[key] = val.flatten()
    
    return J_field


def compute_contravariant_basis(covariant_basis_vectors, jacobian, nodes):
    """Computes contravariant basis vectors and jacobian elements
    
    Args:
        covariant_basis_vectors (dict): dictionary of arrays of covariant basis vectors and derivatives at each node 
        jacobian (dict): dictionary of arrays of coordinate jacobian and partial derivatives
        nodes (array-like): array of node locations in rho, vartheta, zeta coordinates

    Returns:
        contravariant_basis (dict): dictionary of arrays of contravariant basis vectors and jacobian elements
    
    """
    
    # notation: grad_x denotes gradient of x
    # superscript denotes contravariant component
    N_nodes = nodes[0].size
    r = nodes[0]
    axn = np.where(r == 0)[0]
    
    contravariant_basis = {}
    # contravariant basis vectors
    contravariant_basis['grad_rho'] = cross(covariant_basis_vectors['e_v'],
                                            covariant_basis_vectors['e_z'],0)/jacobian['g']  
    contravariant_basis['grad_theta'] = cross(covariant_basis_vectors['e_z'],
                                              covariant_basis_vectors['e_r'],0)/jacobian['g']  
    contravariant_basis['grad_zeta'] = cross(covariant_basis_vectors['e_r'],
                                             covariant_basis_vectors['e_v'],0)/jacobian['g']
    if axn.size:
        contravariant_basis['grad_rho'][:,axn] = cross(covariant_basis_vectors['e_rv'][:,axn],
                                                     covariant_basis_vectors['e_z'][:,axn],0) / jacobian['g_r'][axn]
        contravariant_basis['grad_theta'][:,axn] = cross(covariant_basis_vectors['e_z'][:,axn],
                                                       covariant_basis_vectors['e_r'][:,axn],0)
        contravariant_basis['grad_zeta'][:,axn] = cross(covariant_basis_vectors['e_r'][:,axn],
                                                      covariant_basis_vectors['e_v'][:,axn],0)

    contravariant_basis['e^rho'] = contravariant_basis['grad_rho']
    contravariant_basis['e^theta'] = contravariant_basis['grad_theta']
    contravariant_basis['e^zeta'] = contravariant_basis['grad_zeta']
    # metric coefficients
    contravariant_basis['g^rr'] = dot(contravariant_basis['grad_rho'],contravariant_basis['grad_rho'],0)
    contravariant_basis['g^vv'] = dot(contravariant_basis['grad_theta'],contravariant_basis['grad_theta'],0)  
    contravariant_basis['g^zz'] = dot(contravariant_basis['grad_zeta'],contravariant_basis['grad_zeta'],0)  
    contravariant_basis['g^vz'] = dot(contravariant_basis['grad_theta'],contravariant_basis['grad_zeta'],0)   
    
    return contravariant_basis