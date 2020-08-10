import numpy as np
import functools
import scipy.optimize
import matplotlib.pyplot as plt
from init_guess import get_initial_guess_scale_bdry
from nodes import compute_nodes
from zernike import ZernikeTransform, get_zern_basis_idx_dense, get_double_four_basis_idx_dense, axis_posn, symmetric_x
from force_balance import compute_force_error_nodes
from equilibrium_dynamics import compute_accel_error_spectral
from boundary_conditions import compute_bc_err_RZ, compute_bc_err_four, compute_lambda_err
from backend import jnp, conditional_decorator, jit, use_jax, fori_loop, put, pressfun, iotafun
from backend import get_needed_derivatives, unpack_x, rms, jacfwd, jacrev
from plotting import plot_IC, plot_coord_surfaces, plot_coeffs, plot_fb_err, plot_accel_err

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

# options
sym = True
error_mode = 'accel'
bdry_mode = 'spectral'

# Node locations
nodes,volumes = compute_nodes(2*M,N,NFP,surfs='cheb2')

# interpolator
print('precomputing Fourier-Zernike basis')
derivatives = get_needed_derivatives('force')
zernt = ZernikeTransform(nodes,zern_idx,NFP,derivatives)


# Boundary Shape as fourier series
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

# TODO: make this implementation more efficient for non-symmetric case
if sym:
    sym_mat = symmetric_x(M,N)
else:
    sym_mat = np.diag(np.ones(zern_idx.size+lambda_idx.size),k=0)

if error_mode == 'force':
    equil_fun = compute_force_error_nodes
elif error_mode == 'accel':
    equil_fun = compute_accel_error_spectral
if bdry_mode == 'real':
    bdry_fun = compute_bc_err_RZ
elif bdry_mode == 'spectral':
    bdry_fun = compute_bc_err_four


print('computing initial guess')
# initial guess
cR_init,cZ_init = get_initial_guess_scale_bdry(bdryR,bdryZ,bdry_poloidal,bdry_toroidal,zern_idx,NFP,mode=bdry_mode,nr=20,rcond=1e-6)
cL_init = jnp.zeros(len(lambda_idx))
x_init = jnp.matmul(sym_mat.T,jnp.concatenate([cR_init,cZ_init,cL_init]))

# weights
weights = {'F':1e0,     # equilibrium force balance
           'B':1e2,     # fixed boundary conditions
           'L':1e1}     # gauge constraint on lambda

nodes = jnp.asarray(nodes)
volumes = jnp.asarray(volumes)

args = (zern_idx,lambda_idx,NFP,zernt,nodes,pressfun_params,iotafun_params,Psi_total,
        volumes,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,bdry_fun,weights,sym_mat)

fig, ax = plot_IC(cR_init, cZ_init, zern_idx, NFP, nodes, pressfun_params, iotafun_params)
plt.show()

@conditional_decorator(functools.partial(jit,static_argnums=np.arange(1,17)), use_jax)
def lstsq_obj(x,zern_idx,lambda_idx,NFP,zernt,nodes,pressfun_params,iotafun_params,
              Psi_total,volumes,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,bdry_fun,weights,sym_mat):
    
    cR,cZ,cL = unpack_x(jnp.matmul(sym_mat,x),zern_idx)
    errRf,errZf = equil_fun(cR,cZ,zernt,nodes,pressfun_params,iotafun_params,Psi_total,volumes)
    errRb,errZb = bdry_fun(cR,cZ,cL,zern_idx,lambda_idx,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,NFP)
    errL0 = compute_lambda_err(cL,lambda_idx,NFP)
    # divide through by size of the array so weighting isn't thrown off by more points
    loss = jnp.concatenate([weights['F']*errRf.flatten()/errRf.size,   
                            weights['F']*errZf.flatten()/errZf.size,
                            weights['B']*errRb.flatten()/errRb.size,
                            weights['B']*errZb.flatten()/errZb.size,
                            weights['L']*errL0.flatten()/errL0.size])
    return loss

def callback(x,zern_idx,lambda_idx,NFP,zernt,nodes,pressfun_params,iotafun_params,
             Psi_total,volumes,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,bdry_fun,weights,sym_mat):
    
    cR,cZ,cL = unpack_x(jnp.matmul(sym_mat,x),zern_idx)
    errRf,errZf = equil_fun(cR,cZ,zernt,nodes,pressfun_params,iotafun_params,Psi_total,volumes)
    errRb,errZb = bdry_fun(cR,cZ,cL,zern_idx,lambda_idx,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,NFP)
    errL0 = compute_lambda_err(cL,lambda_idx,NFP)
    
    errRf_rms = rms(errRf)
    errZf_rms = rms(errZf)
    errRb_rms = rms(errRb)
    errZb_rms = rms(errZb)
    errL0_rms = rms(errL0)
    
    loss = np.concatenate([weights['F']*errRf.flatten(),
                           weights['F']*errZf.flatten(),
                           weights['B']*errRb.flatten(),
                           weights['B']*errZb.flatten(),
                           weights['L']*errL0.flatten()])
    loss_rms = rms(loss)
    print('Weighted Loss: {:10.3e}  errRf: {:10.3e}  errZf: {:10.3e}  errRb: {:10.3e}  errZb: {:10.3e}  errL0: {:10.3e}'.format(
    loss_rms,errRf_rms,errZf_rms,errRb_rms,errZb_rms,errL0_rms))


if use_jax:
    lstsq_obj = jit(lstsq_obj, static_argnums=np.arange(len(args))+1)
    jac = jit(jacfwd(lstsq_obj),static_argnums=np.arange(len(args))+1)
    print('compiling')
    foo = lstsq_obj(x_init,*args).block_until_ready() 
    foo = jac(x_init,*args).block_until_ready() 
else:
    jac = '2-point'

print('starting optimization')
out = scipy.optimize.least_squares(lstsq_obj,
                                   x_init,
                                   args=args,
                                   jac=jac,
                                   x_scale='jac',
                                   ftol=1e-6, 
                                   xtol=1e-4, 
                                   gtol=1e-4, 
                                   max_nfev=10, 
                                   verbose=2)
x = out['x']
print('Initial')
callback(x_init, *args)
print('Final')
callback(x, *args)

cR,cZ,cL = unpack_x(jnp.matmul(sym_mat,x),zern_idx)
print('(R0,Z0) = ({:.4e},{:.4e})'.format(*axis_posn(cR,cZ,zern_idx,NFP)))

print('Fourier-Zernike coefficients:')
for k, lmn in enumerate(zern_idx):
    print('l: {:3d}, m: {:3d}, n: {:3d}, cR: {:10.3e}, cZ: {:10.3e}'.format(lmn[0],lmn[1],lmn[2],cR[k],cZ[k]))

print('Lambda coefficients')
for k, mn in enumerate(lambda_idx):
    print('m: {:3d}, n: {:3d}, cL: {:10.3e}'.format(mn[0],mn[1],cL[k]))

# plot flux surfaces
fig, ax = plt.subplots(1,2,figsize=(6,3))
plot_coord_surfaces(cR_init,cZ_init,zern_idx,NFP,nr=10,nt=12,ax=ax[0])
plot_coord_surfaces(cR,cZ,zern_idx,NFP,nr=10,nt=12,ax=ax[1])
ax[0].set_title('Initial')
ax[1].set_title('Final')
plt.show()

# plot force balance error
fig, ax = plt.subplots()
ax, im = plot_fb_err(cR,cZ,cL,zern_idx,lambda_idx,NFP,iotafun_params,pressfun_params,Psi_total,
                domain='sfl', normalize='global', ax=ax, log=False, cmap='plasma')
plt.colorbar(im)
plt.show()

# plot acceleration error
ax,imR,imZ = plot_accel_err(cR,cZ,zernt,zern_idx,NFP,pressfun_params,iotafun_params,Psi_total,domain='sfl',log=False,cmap='plasma')

# plot solution coefficients
plot_coeffs(cR,cZ,cL,zern_idx,lambda_idx,cR_init,cZ_init,cL_init);
plt.show()
