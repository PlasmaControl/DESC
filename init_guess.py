import numpy as np
from zernike import ZernikeTransform, eval_double_fourier

def get_initial_guess_scale_bdry(bdryR,bdryZ,poloidal,toroidal,zern_idx,NFP,mode,nr=20,rcond=1e-6):
    """Generate initial guess by scaling boundary shape
    
    Args:
        bdryR (ndarray, shape(N_bdry_vals,)): R coordinates of boundary, or spectral coefficients of boundary R shape
        bdryZ (ndarray, shape(N_bdry_vals,)): Z coordinates of boundary, or spectral coefficients of boundary Z shape
        poloidal (ndarray, shape(N_bdry_vals,)): poloidal coordinates where bdryR,bdryZ are given, or poloidal mode numbers
        toroidal (ndarray, shape(N_bdry_vals,)): toroidal coordinates where bdryR,bdryZ are given, or toroidal mode numbers
        zern_idx (ndarray, shape(Nc,3)): indices for spectral basis, ie an array of [l,m,n] for each spectral coefficient
        NFP (int): number of field periods
        mode (str): one of 'real', 'spectral' - which format is being used for bdryR,bdryZ,poloidal,toroidal
        nr (int): number of radial points to use when generating guess
        rcond (float): relative limit on singular values for least squares fit to Zernike basis
    Returns:
        cR (ndarray, shape(N_coeffs,)): Fourier-Zernike coefficients for R, following indexing given in zern_idx
        cZ (ndarray, shape(N_coeffs,)): Fourier-Zernike coefficients for Z, following indexing given in zern_idx
    """
    if mode == 'spectral':
        # convert to real space by evaluating spectral coefficients on grid in theta, phi
        dimFourN = 2*np.max(np.abs(toroidal))+1
        dv = np.pi/64
        dz = 2*np.pi/(NFP*dimFourN)
        bdry_theta = np.arange(0,2*np.pi,dv)
        bdry_phi = np.arange(0,2*np.pi/NFP,dz)
        bdry_theta, bdry_phi = np.meshgrid(bdry_theta,bdry_phi,indexing='ij')
        bdry_theta = bdry_theta.flatten()
        bdry_phi = bdry_phi.flatten()
        bdry_idx = np.stack([poloidal,toroidal]).T
        bdryR = eval_double_fourier(bdryR,bdry_idx,NFP,bdry_theta,bdry_phi)
        bdryZ = eval_double_fourier(bdryZ,bdry_idx,NFP,bdry_theta,bdry_phi)
    else:
        bdry_theta = poloidal
        bdry_phi = toroidal
        
    # set up grid for zernike basis
    r = np.linspace(1e-2,1,nr)
    rr,tt = np.meshgrid(r,bdry_theta,indexing='ij')
    rr,pp = np.meshgrid(r,bdry_phi,indexing='ij')
    rr = rr.flatten()
    tt = tt.flatten()
    pp = pp.flatten()
    vv = np.pi - tt
    zz = -pp
    zernt = ZernikeTransform([rr,vv,zz],zern_idx,NFP,derivatives=[0,0,0])
    
    # estimate axis location as center of bdry
    R0_est = (np.max(bdryR) + np.min(bdryR))/2
    Z0_est = (np.max(bdryZ) + np.min(bdryZ))/2

    # scale boundary
    Rinit = (r[:,np.newaxis]*(bdryR[np.newaxis,:]-R0_est) + R0_est).flatten()
    Zinit = (r[:,np.newaxis]*(bdryZ[np.newaxis,:]-Z0_est) + Z0_est).flatten()

    # fit to zernike basis for initial guess
    cR = zernt.fit(Rinit,rcond).flatten()
    cZ = zernt.fit(Zinit,rcond).flatten()
    
    return cR, cZ