import numpy as np
from desc.utils import errorif, safenorm
from desc.grid import LinearGrid


def winding(curve,points):
    """
    Compute the winding number.
    
    Parameters
    ----------
    curve : array-like, shape(n_curves,M,2)
        X,Y coordinates of closed curves to be evaluated.
    points : array-like, shape(n_curves,N,2)
        X,Y coordinates of points to be evaluated. 

    Returns
    -------
    winding : array-like, shape(n_curves,N)
        Winding numbers
    """
    errorif(curve.shape[0]!=points.shape[0])
    if not np.allclose(curve[:,0],curve[:,-1]):
        curve = np.concatenate([curve,curve[:,0:1]],axis=1)
    z = points[:,:,None,:] - curve[:,None,:-1,:]
    z_next = points[:,:,None,:] - curve[:,None,1:,:]
    z = z[..., 0] + 1j * z[...,1]
    z_next = z_next[...,0] + 1j * z_next[...,1]
    angles = np.angle(z_next/z)
    winding = np.sum(angles,axis=2)
    return winding

def in_plasma(points,eq):
    """
    Determine if an array of points in cylindrical coordinates is inside the plasma boundary.
    
    Parameters
    ----------
    points : array-like shape(n_r,n_phi,n_z,3)
        R,phi,Z coordinates of points to be evaluated. points[:,idx,:,1] should be constant.
    eq : Equilibrium
        Equilibrium with the desired plasma boundary.

    Returns
    -------
    out : array-like shape(n_r,n_phi,n_z)
        Boolean array indicating whether each point is inside the plasma boundary.
    """
    phi = np.unique(points[...,1])
    grid = LinearGrid(rho=[1.0],M=24,zeta=phi,NFP=eq.NFP)
    data = eq.compute(['R','Z'],grid=grid)
    R_plasma = data['R'].reshape(grid.num_zeta,-1)
    Z_plasma = data['Z'].reshape(grid.num_zeta,-1)
    pts = points[...,[0,2]].transpose(1,0,2,3).reshape(points.shape[1],-1,2)
    curve = np.stack([R_plasma,Z_plasma],axis=-1)

    out = np.isclose(np.abs(winding(curve,pts)),2*np.pi)
    out = out.reshape(points.shape[1],points.shape[0],points.shape[2]).transpose(1,0,2)
    return out

def plasma_dist(points,eq):
    """
    Determine distance of array of points in cylindrical coordinates from the plasma boundary.
    
    Parameters
    ----------
    points : array-like shape(n_r,n_phi,n_z,3)
        R,phi,Z coordinates of points to be evaluated. points[:,idx,:,1] should be constant.
    eq : Equilibrium
        Equilibrium with the desired plasma boundary.

    Returns
    -------
    out : array-like shape(n_r,n_phi,n_z)
        Minimum distance of each point in points from the plasma. Assumes ~ axisymmetry, i.e.
        assumes nearest point is in the same phi plane for improved computational performance.
    """
    phi = np.unique(points[...,1])
    grid = LinearGrid(rho=[1.0],M=24,zeta=phi,NFP=eq.NFP)
    data = eq.compute(['R','Z'],grid=grid)
    R_plasma = data['R'].reshape(grid.num_zeta,-1)
    Z_plasma = data['Z'].reshape(grid.num_zeta,-1)
    pts = points[...,[0,2]].transpose(1,0,2,3).reshape(points.shape[1],-1,2)
    curve = np.stack([R_plasma,Z_plasma],axis=-1)
    out = safenorm(pts[:,:,None,:] - curve[:,None,:,:],axis=-1).min(axis=-1)
    out = out.reshape(points.shape[1],points.shape[0],points.shape[2]).transpose(1,0,2)
    return out
