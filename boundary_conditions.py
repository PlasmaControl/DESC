import numpy as np
import functools
from backend import jnp, conditional_decorator, jit, use_jax, fori_loop
from zernike import fourzern, double_fourier_basis, eval_double_fourier

@conditional_decorator(functools.partial(jit,static_argnums=(3,4,7,8,9)), use_jax)
def compute_bc_err_four(cR,cZ,cL,zern_idx,lambda_idx,bdryR,bdryZ,bdryM,bdryN,NFP):
    """Compute boundary error in fourier coefficients
    
    Args:
        cR (ndarray, shape(N_coeffs,)): Fourier-Zernike coefficients of R
        cZ (ndarray, shape(N_coeffs,)): Fourier-Zernike coefficients of Z
        cL (ndarray, shape(2M+1)*(2N+1)): double Fourier coefficients of lambda
        zern_idx (ndarray, shape(Nc,3)): indices for R,Z spectral basis, ie an 
            array of [l,m,n] for each spectral coefficient
        lambda_idx (ndarray, shape(Nlambda,2)): indices for lambda spectral basis, 
            ie an array of [m,n] for each spectral coefficient
        bdryR (ndarray, shape(N_bdry_modes,)): R coefficients of boundary shape
        bdryZ (ndarray, shape(N_bdry_modes,)): Z coefficients of boundary shape
        bdryM (ndarray, shape(N_bdry_modes,)): poloidal mode numbers
        bdryN (ndarray, shape(N_bdry_modes,)): toroidal mode numbers
        NFP (int): number of field periods   
        
    Returns:
        errR ((ndarray, shape(N_bdry_pts,))): vector of R errors in boundary spectral coeffs
        errZ ((ndarray, shape(N_bdry_pts,))): vector of Z errors in boundary spectral coeffs
    """
    
    # get grid for bdry eval
    dimFourN = 2*jnp.max(np.abs(bdryN))+1
    dv = jnp.pi/64
    dz = 2*jnp.pi/(NFP*dimFourN)
    bdry_theta = jnp.arange(0,2*jnp.pi,dv)
    bdry_phi = jnp.arange(0,2*jnp.pi/NFP,dz)
    bdry_theta, bdry_phi = jnp.meshgrid(bdry_theta,bdry_phi,indexing='ij')
    bdry_theta = bdry_theta.flatten()
    bdry_phi = bdry_phi.flatten()

    L = eval_double_fourier(cL,lambda_idx,NFP,bdry_theta,bdry_phi)

    # find values of R,Z at pts specified
    rho = jnp.ones_like(bdry_theta)
    vartheta = jnp.pi - bdry_theta - L
    zeta = -bdry_phi
    zern_bdry_interp = jnp.stack([fourzern(rho,vartheta,zeta,lmn[0],lmn[1],lmn[2],NFP,0,0,0) for lmn in zern_idx]).T
    R = jnp.matmul(zern_bdry_interp,cR).flatten()
    Z = jnp.matmul(zern_bdry_interp,cZ).flatten()

    four_bdry_interp = jnp.stack([double_fourier_basis(bdry_theta,bdry_phi,m,n,NFP) for m, n in zip(bdryM,bdryN)]).T

    cRb, cZb = jnp.linalg.lstsq(four_bdry_interp,jnp.array([R,Z]).T,rcond=None)[0].T

    # compute errors
    errR = cRb - bdryR
    errZ = cZb - bdryZ
    return errR,errZ


@conditional_decorator(functools.partial(jit,static_argnums=(3,4,7,8,9)), use_jax)
def compute_bc_err_RZ(cR,cZ,cL,zern_idx,lambda_idx,bdryR,bdryZ,bdry_theta,bdry_phi,NFP):
    """Compute boundary error at discrete points
    
    Args:
        cR (ndarray, shape(N_coeffs,)): Fourier-Zernike coefficients of R
        cZ (ndarray, shape(N_coeffs,)): Fourier-Zernike coefficients of Z
        cL (ndarray, shape(2M+1)*(2N+1)): double Fourier coefficients of lambda
        zern_idx (ndarray, shape(Nc,3)): indices for R,Z spectral basis, 
            ie an array of [l,m,n] for each spectral coefficient
        lambda_idx (ndarray, shape(Nlambda,2)): indices for lambda spectral basis, 
            ie an array of [m,n] for each spectral coefficient
        bdryR (ndarray, shape(N_bdry_pts,)): R values of boundary shape
        bdryZ (ndarray, shape(N_bdry_pts,)): Z values of boundary shape
        bdry_theta (ndarray, shape(N_bdry_pts,)): real space poloidal coordinates where boundary is specified
        bdry_phi (ndarray, shape(N_bdry_pts,)): real space toroidal coordinates where boundary is specified
        NFP (int): number of field periods   
        
    Returns:
        errR ((ndarray, shape(N_bdry_pts,))): vector of R errors in boundary position at specified points
        errZ ((ndarray, shape(N_bdry_pts,))): vector of Z errors in boundary position at specified points
    """


    L = eval_double_fourier(cL,lambda_idx,NFP,bdry_theta,bdry_phi)

    # find values of R,Z at pts specified
    rho = jnp.ones_like(bdry_theta)
    vartheta = jnp.pi - bdry_theta - L
    zeta = -bdry_phi
    zern_bdry_interp = jnp.stack([fourzern(rho,vartheta,zeta,lmn[0],lmn[1],lmn[2],NFP,0,0,0) for lmn in zern_idx]).T
    R = jnp.matmul(zern_bdry_interp,cR).flatten()
    Z = jnp.matmul(zern_bdry_interp,cZ).flatten()

    # compute errors
    errR = R-bdryR
    errZ = Z-bdryZ
    
    return errR,errZ


@conditional_decorator(functools.partial(jit,static_argnums=(1,2)), use_jax)
def compute_lambda_err(cL,idx,NFP):
    """Compute the error in sum(lambda_mn) to enforce 
    vartheta(0,0) = 0
    
    Args:
        cL (ndarray, shape(2M+1)*(2N+1)): double Fourier coefficients of lambda
        idx (ndarray, shape(Nlambda,2)): indices for lambda spectral basis, ie an array of [m,n] for each spectral coefficient
        NFP (int): number of field periods 
        
    Returns:
        errL (float): sum of lambda_mn where m,n>0
    """
    
    mn_pos = jnp.where(jnp.logical_and(idx[:,0]>=0, idx[:,1]>=0))[0]
    errL = jnp.sum(cL[mn_pos])
            
    return errL


def get_lambda_constraint_matrix(zern_idx,lambda_idx):
    """Computes a linear constraint matrix to enforce vartheta(0,0) = 0
    We require sum(lambda_mn) = 0, is Cx = 0
    
    Args:
        zern_idx (ndarray, shape(Nc,3)): indices for R,Z spectral basis, 
            ie an array of [l,m,n] for each spectral coefficient
        lambda_idx (ndarray, shape(Nlambda,2)): indices for lambda spectral basis, 
            ie an array of [m,n] for each spectral coefficient
        
    Returns:
        C (ndarray, shape(2*N_coeffs + (2M+1)*(2N+1))): linear constraint matrix, 
            so Cx is the error in the lambda constraint
    """
    
    # assumes x = [cR, cZ, cL]
    offset = 2*len(zern_idx)
    mn_pos = np.where(np.logical_and(lambda_idx[:,0]>=0, lambda_idx[:,1]>=0))[0]
    C = np.zeros(offset + len(lambda_idx))
    C[offset+mn_pos] = 1
            
    return C
