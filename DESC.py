import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from init_guess import get_initial_guess_scale_bdry
from zernike import ZernikeTransform
from force_balance import force_error_nodes
from boundary_conditions import bc_err_RZ
from utils import rms, plotter


def wrapper(x,M,N,NFP,zernt,nodes,pressfun,iotafun,Psi_total,dr,dv,dz,bdryR,bdryZ,bdry_theta,bdry_phi,weights):
    nRZ = (M+1)**2*(2*N+1)
    nL = (2*M+1)*(2*N+1)
    cR = x[:nRZ]
    cZ = x[nRZ:2*nRZ]
    cL = x[2*nRZ:]
    
    errF = force_error_nodes(cR,cZ,zernt,nodes,pressfun,iotafun,Psi_total,dr,dv,dz)
    errR,errZ,errL = bc_err_RZ(cR,cZ,cL,bdryR,bdryZ,bdry_theta,bdry_phi,M,N,NFP)

    # divide through by size of the array so weighting isn't thrown off by more points
    loss = np.concatenate([weights['F']*errF.flatten()/errF.size,   
                           weights['R']*errR.flatten()/errR.size,
                           weights['Z']*errZ.flatten()/errZ.size,
                           weights['L']*errL.flatten()/errL.size])
    return loss

def iotafun(rho,nu=0, params=None):
    """Rotational transform
    
    Args:
        rho (array-like): coordinates at which to evaluate
        nu (int): order of derivative (for compatibility with scipy spline routines)
    """
    if nu==0:
        return .7*(1-rho**2)
    elif nu==1:
        return -.7*2*rho
    else:
        raise NotImplementedError
        
def pressfun(rho,nu=0, params=None):
    """Plasma pressure * mu0
    
    Args:
        rho (array-like): coordinates at which to evaluate
        nu (int): order of derivative (for compatibility with scipy spline routines)
    """
    mu0 = 4*np.pi*1e-7
    p0 = 1e4
    if nu==0:
        return mu0*p0*(1-rho**2)
    elif nu==1:
        return  -mu0*p0*2*rho
    else:
        raise NotImplementedError
        
        
# Inputs
Psi_total = 1
M = 6
N = 0
NFP = 1


# Node locations
r = np.linspace(0,1,100)
dr = np.diff(r)[0]
v = np.linspace(0,2*np.pi,100)
dv = np.diff(v)[0]
dz = 2*np.pi/NFP
rr,vv = np.meshgrid(r,v,indexing='ij')
rr = rr.flatten()
vv = vv.flatten()
zz = np.zeros_like(rr)
nodes = np.stack([rr,vv,zz])
dr = dr*np.ones_like(rr)
dv = dv*np.ones_like(vv)
dz = dz*np.ones_like(zz)


# Boundary Shape
bdry_theta = np.linspace(0,2*np.pi,100)
bdry_phi = np.zeros_like(bdry_theta)
b = 1
a = 2
e = np.sqrt(1-b**2/a**2)
R0 = 2
Z0 = 0
bdryR = R0 + b*np.cos(bdry_theta)
bdryZ = Z0 + a*np.sin(bdry_theta)


# interpolator
zernt = ZernikeTransform(nodes,M,N,NFP)

# initial guess
cR_init,cZ_init,cL_init, Rinit, Zinit = get_initial_guess_scale_bdry(bdryR,bdryZ,bdry_theta,bdry_phi,M,N,NFP)
x_init = np.concatenate([cR_init,cZ_init,cL_init])

# weights
weights = {'F':1e4,     # force balance error
           'R':1e6,     # error in R component of bdry
           'Z':1e6,     # error in Z component of bdry
           'L':1e8}     # error in sum lambda_mn


args = (M,N,NFP,zernt,nodes,pressfun,iotafun,Psi_total,dr,dv,dz,bdryR,bdryZ,bdry_theta,bdry_phi,weights)

sol = scipy.optimize.least_squares(wrapper,
                                   x_init,
                                   args=args,
                                   x_scale='jac',
                                   ftol=1e-4, 
                                   xtol=1e-4, 
                                   gtol=1e-4, 
                                   max_nfev=100, 
                                   verbose=2)


x = sol['x']
nRZ = (M+1)**2*(2*N+1)
nL = (2*M+1)*(2*N+1)
cR = x[:nRZ]
cZ = x[nRZ:2*nRZ]
cL = x[2*nRZ:]

fig, ax = plt.subplots(1,2,figsize=(6,3))
plotter(cR_init,cZ_init,M,N,NFP,bdryR=bdryR,bdryZ=bdryZ,ax=ax[0],nr=10,ntheta=10)
plotter(cR,cZ,M,N,NFP,bdryR=bdryR,bdryZ=bdryZ,ax=ax[1],nr=10,ntheta=10)
plt.show()