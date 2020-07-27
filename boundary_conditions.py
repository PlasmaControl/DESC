import numpy as np
from zernike import fourzern, fringe_to_lm, F_mn

def bc_err_RZ(cR,cZ,cL,bdryR,bdryZ,bdry_theta,bdry_phi,M,N,NFP):
    """Compute boundary error at discrete R,Z points
    
    Args:
        cR (array-like): Fourier-Zernike coefficients of R
        cZ (array-like): Fourier-Zernike coefficients of Z
        cL (array-like): double Fourier coefficients of lambda
        bdryR (array-like): R values of boundary shape
        bdryZ (array-like): Z values of boundary shape
        bdry_theta (array-like): real space poloidal coordinates where boundary is specified
        bdry_phi (array-like): real space toroidal coordinates where boundary is specified
        M (int): maximum poloidal mode number
        N (int): maximum toroidal mode number
        NFP (int): number of field periods   
        
    Returns:
        errR (array-like): vector of R errors in boundary position at specified points
        errZ (array-like): vector of Z errors in boundary position at specified points
        errL (float): sum of lambda_mn
    """
    num_lm_modes = (M+1)**2
    num_four = 2*N+1    
    # compute lambda, theres probably a faster vectorized way to do this
    cL = cL.reshape((2*M+1,2*N+1))
    L = np.zeros_like(bdry_theta)
    for i in range(cL.shape[0]):
        for j in range(cL.shape[1]):
            m = -M + i
            n = -N + j
            L += cL[i,j]*F_mn(bdry_theta,bdry_phi,m,n,NFP)

    # find values of R,Z at pts specified
    vartheta = np.pi - bdry_theta + L
    zeta = -bdry_phi
    bdry_interp = np.stack([fourzern(np.ones_like(bdry_theta),vartheta,zeta,*fringe_to_lm(i),n-N,NFP) for i in range(num_lm_modes) 
                            for n in range(num_four)]).T
    R = np.matmul(bdry_interp,cR).flatten()
    Z = np.matmul(bdry_interp,cZ).flatten()

    # compute errors
    errR = R-bdryR
    errZ = Z-bdryZ
    errL = np.sum(cL[M:,N:])
    return errR,errZ,errL