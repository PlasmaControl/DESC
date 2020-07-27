import numpy as np
from zernike import ZernikeTransform

def get_initial_guess_scale_bdry(bdryR,bdryZ,bdry_theta,bdry_phi,M,N,NFP,nr=20,rcond=1e-1, return_rvz=False):
    """Generate initial guess by scaling boundary shape
    
    Args:
        bdryR (array-like): R coordinates of boundary
        bdryZ (array-like): Z coordinates of boundary
        bdry_theta (array-like): poloidal coordinates where bdryR,bdryZ are given
        bdry_phi (array-like): toroidal coordinates where bdryR,bdryZ are given
        M (int): maximum poloidal mode number
        N (int): maximum toroidal mode number
        NFP (int): number of field periods
        nr (int): number of radial points to use when generating guess
        rcond (float): relative limit on singular values for least squares fit to Zernike basis
        return_rvz (bool): whether to also return the grid used to generate the initial guess
        
    Returns:
        cR (array-like): Fourier-Zernike coefficients for R, indexed as (lm,n) flattened in row major order
        cZ (array-like): Fourier-Zernike coefficients for Z, indexed as (lm,n) flattened in row major order
        cL (array-like): double Fourier series coefficients for lambda, indexed as (m,n) flattened in row major order
        Rinit (array-like): R coordinates of initial flux surface guess
        Zinit (array-like): Z coordinates of initial flux surface guess
        
        if return_rvz:
        rr,vv,zz (array-like): grid used to generate initial guess
    """
    
    
    r = np.linspace(1e-2,1,nr)
    rr,tt = np.meshgrid(r,bdry_theta,indexing='ij')
    rr,pp = np.meshgrid(r,bdry_phi,indexing='ij')
    rr = rr.flatten()
    tt = tt.flatten()
    pp = pp.flatten()
    vv = np.pi - tt
    zz = -pp

    zernt = ZernikeTransform([rr,vv,zz],M,N,NFP)
    R0_est = (np.max(bdryR) + np.min(bdryR))/2
    Z0_est = (np.max(bdryZ) + np.min(bdryZ))/2

    Rinit = (r[:,np.newaxis]*(bdryR[np.newaxis,:]-R0_est) + R0_est).flatten()
    Zinit = (r[:,np.newaxis]*(bdryZ[np.newaxis,:]-Z0_est) + Z0_est).flatten()

    cL = np.zeros((2*M+1)*(2*N+1))
    cR = zernt.fit(Rinit,rcond=rcond).flatten()
    cZ = zernt.fit(Zinit,rcond=rcond).flatten()
    if return_rvz:
        return cR, cZ, cL, Rinit, Zinit, rr, vv, zz
    else:
        return cR, cZ, cL, Rinit, Zinit