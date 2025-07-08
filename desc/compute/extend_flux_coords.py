import numpy as np
from desc.basis import PowerDoubleFourierBasis, PowerSeries
from desc.transform import Transform
from desc.grid import LinearGrid
from desc.utils import safenorm

def _compute_s_edge(
    eq,
    res = 24,
    ds = 0.01,
    dist = 0.5):

    edge_s_idx = np.zeros(0)
    
    # Compute R and Z at a fixed grid of rho, theta, and zeta
    compute_grid = LinearGrid(rho=np.linspace(0,1,3),M=res,N=res,NFP=eq.NFP)
    data = eq.compute(['R','Z'],compute_grid)

    # Fit a quadratic R(rho,theta,zeta) = r0(theta,zeta) + r1(theta,zeta)rho + r2(theta,zeta)rho^2
    basis = PowerDoubleFourierBasis(2,res,res,sym=False, NFP = eq.NFP)
    transform = Transform(compute_grid, basis, build=True, build_pinv=True, derivs = 1)
    R_c = transform.fit(data['R'])
    Z_c = transform.fit(data['Z'])

    # Get an array of all angles
    lcfs_mask = np.isclose(compute_grid.nodes[:,0],1)
    angles_2D = compute_grid.nodes[lcfs_mask]
    TT,ZZ = (arr.reshape(compute_grid.num_zeta,compute_grid.num_theta) for arr in (angles_2D[:,1],angles_2D[:,2]))
    angles = np.stack([np.ones_like(TT),TT,ZZ]) # shape: 3, zeta, theta

    # Get the R and Z coordinates of the plasma boundary
    R0, Z0 = data['R'][lcfs_mask], data['Z'][lcfs_mask]
    lcfs = np.stack([R0,Z0]).reshape(2,compute_grid.num_zeta,compute_grid.num_theta) # shape 2, zeta, theta
    lcfs = lcfs[:,np.newaxis,:,np.newaxis,:] # shape 2,1,zeta,1,theta

    # Determine r0, r1, r2 as a function of theta and zeta
    A = basis.evaluate(angles_2D)
    masks = (basis.modes[:,0][:,np.newaxis] == np.arange(3)[np.newaxis,:]).T

    r0 = A[:,masks[0]] @ R_c[masks[0]]
    r1 = A[:,masks[1]] @ R_c[masks[1]]
    r2 = A[:,masks[2]] @ R_c[masks[2]]

    z0 = A[:,masks[0]] @ Z_c[masks[0]]
    z1 = A[:,masks[1]] @ Z_c[masks[1]]
    z2 = A[:,masks[2]] @ Z_c[masks[2]]

    # Extrapolate R and Z past the plasma boundary
    r0,r1,r2,z0,z1,z2 = (arr.reshape(TT.shape) for arr in (r0,r1,r2,z0,z1,z2)) # shape: zeta, theta

    s_max = 9
    while edge_s_idx.size!=TT.size:
        s = np.arange(1,s_max,ds).reshape(-1,1,1)
        rays = np.stack([r0+r1*s+r2*s**2,z0+z1*s+z2*s**2]) # shape 2, rho, zeta, theta
        rays = rays[:,:,:,:,np.newaxis]

        # Find the distance of each point from the plasma boundary
        dists = rays - lcfs # shape 2, s(ray), zeta, theta(ray), theta(lcfs)
        dists = safenorm(dists,axis=0).min(axis=-1) # min distance for each ray from lcfs (shape: s,zeta,theta)
        dists = dists.reshape(-1,TT.size)

        # For each ray, determine at which s the ray is >dist away from the LCFS 
        axis_0,axis_1 = np.where(dists>dist)
        edge_s_idx = axis_0[np.unique(axis_1,return_index=True)[1]]
        s_max+=1
    s_edge = np.squeeze(s)[edge_s_idx].reshape(TT.shape) # shape zeta, theta

    return s_edge, r0, r1, r2, z0, z1, z2

def _rescale_rho(
    eq, s_edge, r0, r1, r2, z0, z1, z2,
    res = 24,
):
    # Define spectral transformation
    s_tilde_basis = PowerDoubleFourierBasis(res,res,res,eq.NFP)
    grid = LinearGrid(L=res,M=res,N=res,NFP=eq.NFP)
    transform = Transform(grid,s_tilde_basis,build=True,build_pinv=True,derivs=1)

    # Define s (the naively extended radial coordinate) from 1 to the edge of the domain
    s_tilde = grid.nodes[:,0].reshape(grid.num_zeta,grid.num_rho,grid.num_theta)
    s = s_tilde*(s_edge[:,np.newaxis,:]-1)+1

    # Extend R and Z in the vacuum region
    R_v = r2[:,np.newaxis,:]*s**2+r1[:,np.newaxis,:]*s+r0[:,np.newaxis,:]
    Z_v = z2[:,np.newaxis,:]*s**2+z1[:,np.newaxis,:]*s+z0[:,np.newaxis,:]
    R_v = R_v.flatten()
    Z_v = Z_v.flatten()

    # Spectrally fit R_v and Z_v to s_tilde, theta, zeta     
    R_v_c = transform.fit(R_v)
    Z_v_c = transform.fit(Z_v)

    # Calculate the Jacobian of (x,y,z) --> (s_tilde,theta,zeta) at every point in the vacuum
    dRds = transform.transform(R_v_c,dr=1)
    dZds = transform.transform(Z_v_c,dr=1)
    dRdt = transform.transform(R_v_c,dt=1)
    dZdt = transform.transform(Z_v_c,dt=1)
    sqrt_g = R_v * (dZds*dRdt-dRds*dZdt)

    # Find the volume between surfaces of constant s_tilde (not surfaces of constant s!)
    dV = (sqrt_g * grid.weights)
    dV = dV.reshape(grid.num_zeta,grid.num_rho,grid.num_theta)
    dV = dV.sum(axis=2).sum(axis=0)

    # Add volume of plasma to get the total volume within surfaces of constant s_tilde
    V_plasma = eq.compute('V')['V']
    V = np.cumsum(dV)+V_plasma

    # Define rho to be proportional the volume contained within a constant rho surface
    rho = np.sqrt(V/V[0])

    # Resample rho
    basis_1D = PowerSeries(res, sym=False)
    s_tilde_transform = Transform(LinearGrid(rho=rho),basis_1D, build_pinv=True,build=True,derivs=1)
    s_tilde_c = s_tilde_transform.fit(s_tilde[0,:,0])

    # R = s_tilde_basis.evaluate(s_tilde,theta,zeta) @ R_c = s_tilde_basis.evaluate(basis_1D.evaluate(rho)@s_tilde_c,theta,zeta) @ R_c
    return R_v_c, Z_v_c, s_tilde_c, s_tilde_basis, basis_1D, rho.max()