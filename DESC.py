import numpy as np
import functools
import scipy.optimize
import matplotlib.pyplot as plt
from init_guess import get_initial_guess_scale_bdry
from zernike import ZernikeTransform, get_zern_basis_idx_dense, get_double_four_basis_idx_dense, axis_posn
from force_balance import compute_force_error_nodes
from boundary_conditions import compute_bc_err_RZ, compute_bc_err_four, compute_lambda_err
from backend import jnp, conditional_decorator, jit, use_jax, fori_loop, put, pressfun, iotafun
from backend import get_needed_derivatives, unpack_x, rms, jacfwd, jacrev
from plotting import plot_IC, plot_coord_surfaces, plot_coeffs, plot_fb_err

# Inputs
Psi_total = 1
M = 6
N = 0
NFP = 1
zern_idx = get_zern_basis_idx_dense(M,N)
lambda_idx = get_double_four_basis_idx_dense(M,N)

# profiles, polynomial basis, coeffs in ascending powers of x
p0 = 1e4
pressfun_params = (p0,0,-p0)
iota0 = 1.618
iotafun_params = (iota0,0,-iota0)


# Node locations
r = np.linspace(0.,1,20)
dr = np.diff(r)[0]
v = np.linspace(0,2*jnp.pi,21)[:-1]
dv = np.diff(v)[0]
# z = np.linspace(0,2*np.pi/NFP,N*)
dz = 2*np.pi/NFP
rr,vv = np.meshgrid(r,v,indexing='ij')
rr = rr.flatten()
vv = vv.flatten()
zz = np.zeros_like(rr)
nodes = np.stack([rr,vv,zz])
dr = dr*np.ones_like(rr)
dv = dv*np.ones_like(vv)
dz = dz*np.ones_like(zz)
node_volume = np.stack([dr,dv,dz])
axn = np.where(rr == 0)[0]

# interpolator
print('precomputing Fourier-Zernike basis')
derivatives = get_needed_derivatives('force')
zernt = ZernikeTransform(nodes,zern_idx,NFP,derivatives)


# Boundary Shape as fourier series
bdry_mode = 'spectral'
bdryM = jnp.arange(-M,M+1)
bdryN = jnp.arange(-N,N+1)
bdryM, bdryN = jnp.meshgrid(bdryM, bdryN, indexing='ij')
bdryM = bdryM.flatten()
bdryN = bdryN.flatten()
bdryR = jnp.zeros(len(bdryM),dtype=jnp.float64)
bdryZ = jnp.zeros(len(bdryM),dtype=jnp.float64)

"""D shaped tokamak"""
# bndryR = [0.000; 0.00; 3.51; -1.00; 0.106];
bdryR = put(bdryR, jnp.where(jnp.logical_and(bdryM == 0, bdryN == 0))[0], 3.51)
bdryR = put(bdryR, jnp.where(jnp.logical_and(bdryM == 1, bdryN == 0))[0], -1.0)
bdryR = put(bdryR, jnp.where(jnp.logical_and(bdryM == 2, bdryN == 0))[0], 0.106)
#bndryZ = [0.160; 1.47; 0.00;  0.00; 0.000];
bdryZ = put(bdryZ, jnp.where(jnp.logical_and(bdryM == -2, bdryN == 0))[0], 0.160)
bdryZ = put(bdryZ, jnp.where(jnp.logical_and(bdryM == -1, bdryN == 0))[0], 1.47)

bdry_poloidal = bdryM
bdry_toroidal = bdryN


# # Boundary shape in real space
# bdry_mode = 'real'
# bdry_theta = np.linspace(0,2*np.pi,101)
# bdry_psi = np.zeros_like(bdry_theta)
"""circular tokamak"""
# b = 1
# a = 1
# R0 = 2
# Z0 = 0
# bdryR = R0 + b*np.cos(bdry_theta)
# bdryZ = Z0 + a*np.sin(bdry_theta)
# bdry_poloidal = bdry_theta
# bdry_toroidal = bdry_psi




print('computing initial guess')
# initial guess
cR_init,cZ_init = get_initial_guess_scale_bdry(bdryR,bdryZ,bdry_poloidal,bdry_toroidal,zern_idx,NFP,mode=bdry_mode,nr=20,rcond=1e-6)
cL_init = jnp.zeros(len(lambda_idx))
x_init = jnp.concatenate([cR_init,cZ_init,cL_init])

# weights
weights = {'F':1e6,     # force balance error
           'R':1e4,     # error in R component of bdry
           'Z':1e4,     # error in Z component of bdry
           'L':1e4}     # error in sum lambda_mn


nodes = jnp.asarray(nodes)
node_volume = jnp.asarray(node_volume)

args = (zern_idx,lambda_idx,NFP,zernt,nodes,pressfun_params,iotafun_params,Psi_total,
        node_volume,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,weights)

fig, ax = plot_IC(cR_init, cZ_init, zern_idx, NFP, nodes, pressfun_params, iotafun_params)
plt.show()

if bdry_mode == 'real':
    bdry_fun = compute_bc_err_RZ
elif bdry_mode == 'spectral':
    bdry_fun = compute_bc_err_four

@conditional_decorator(functools.partial(jit,static_argnums=np.arange(1,15)), use_jax)
def lstsq_obj(x,zern_idx,lambda_idx,NFP,zernt,nodes,pressfun_params,iotafun_params,
              Psi_total,node_volume,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,weights):
    
    cR,cZ,cL = unpack_x(x,zern_idx)
    errF = compute_force_error_nodes(cR,cZ,zernt,nodes,pressfun_params,iotafun_params,Psi_total,node_volume)
    errR,errZ = bdry_fun(cR,cZ,cL,zern_idx,lambda_idx,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,NFP)
    errL = compute_lambda_err(cL,lambda_idx,NFP)
    # divide through by size of the array so weighting isn't thrown off by more points
    loss = jnp.concatenate([weights['F']*errF.flatten()/errF.size,   
                           weights['R']*errR.flatten()/errR.size,
                           weights['Z']*errZ.flatten()/errZ.size,
                           weights['L']*errL.flatten()/errL.size])
    return loss


def callback(x,zern_idx,lambda_idx,NFP,zernt,nodes,pressfun_params,iotafun_params,
             Psi_total,node_volume,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,weights):
    
    cR,cZ,cL = unpack_x(x,zern_idx)
    errF = compute_force_error_nodes(cR,cZ,zernt,nodes,pressfun_params,iotafun_params,Psi_total,node_volume)
    errR,errZ = bdry_fun(cR,cZ,cL,zern_idx,lambda_idx,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,NFP)
    errL = compute_lambda_err(cL,lambda_idx,NFP)

    errFrms = rms(errF)
    errRrms = rms(errR)
    errZrms = rms(errZ)    
    errLrms = rms(errL)
    
    loss = np.concatenate([weights['F']*errF.flatten(),
                           weights['R']*errR.flatten(),
                           weights['Z']*errZ.flatten(),
                           weights['L']*errL.flatten()])
    lossrms = rms(loss)    
    print('Weighted Loss: {:10.3e}  errF: {:10.3e}  errBdryR: {:10.3e}  errBdryZ: {:10.3e}  errLambda: {:10.3e}'.format(
    lossrms,errFrms,errRrms,errZrms,errLrms))
    

if use_jax:
    lstsq_obj = jit(lstsq_obj, static_argnums=np.arange(len(args))+1)
    jac = jit(jacfwd(lstsq_obj),static_argnums=np.arange(len(args))+1)
    print('compiling')
    foo = lstsq_obj(x_init,*args).block_until_ready() 
    foo = jac(x_init,*args).block_until_ready() 
else:
    jac = None

print('starting optimization')
out = scipy.optimize.least_squares(lstsq_obj,
                                   x_init,
                                   args=args,
                                   jac=jac,
                                   x_scale='jac',
                                   ftol=1e-10, 
                                   xtol=1e-8, 
                                   gtol=1e-8, 
                                   max_nfev=1000, 
                                   verbose=2)
x = out['x']
print('Initial')
callback(x_init, *args)
print('Final')
callback(x, *args)

cR,cZ,cL = unpack_x(x,zern_idx)
print('(R0,Z0) = ({:.4e},{:.4e})'.format(*axis_posn(cR,cZ,zern_idx,NFP)))

print('Fourier-Zernike coefficients:')
for k, lmn in enumerate(zern_idx):
    print('l: {:3d}, m: {:3d}, n: {:3d}, cR: {:10.3e}, cZ: {:10.3e}'.format(lmn[0],lmn[1],lmn[2],cR[k],cZ[k]))

print('Lambda coefficients')
for k, mn in enumerate(lambda_idx):
    print('m: {:3d}, n: {:3d}, cL: {:10.3e}'.format(mn[0],mn[1],cL[k]))

fig, ax = plt.subplots(1,2,figsize=(6,3))
plot_coord_surfaces(cR_init,cZ_init,zern_idx,NFP,nr=10,ntheta=10,ax=ax[0])
plot_coord_surfaces(cR,cZ,zern_idx,NFP,nr=10,ntheta=10,ax=ax[1])
ax[0].set_title('Initial')
ax[1].set_title('Final')
plt.show()

fig, ax = plt.subplots()
ax, im = plot_fb_err(cR,cZ,cL,zern_idx,lambda_idx,NFP,iotafun_params, pressfun_params, Psi_total,
                domain='sfl', normalize='global', ax=ax, log=False, cmap='plasma')
plt.colorbar(im)
plt.show()

plot_coeffs(cR,cZ,cL,zern_idx,lambda_idx,cR_init,cZ_init,cL_init);
plt.show()
