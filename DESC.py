import numpy as np
import functools
import scipy.optimize
import matplotlib.pyplot as plt
from init_guess import get_initial_guess_scale_bdry
from nodes import get_nodes
from zernike import ZernikeTransform, get_zern_basis_idx_dense, get_double_four_basis_idx_dense, axis_posn, symmetric_x, eval_double_fourier
from force_balance import compute_force_error_nodes
from equilibrium_dynamics import compute_accel_error_spectral
from boundary_conditions import compute_bc_err_RZ, compute_bc_err_four, compute_lambda_err, format_bdry
from backend import jnp, conditional_decorator, jit, use_jax, fori_loop, put, presfun, iotafun
from backend import get_needed_derivatives, unpack_x, rms, jacfwd, jacrev
from plotting import plot_IC, plot_coord_surfaces, plot_coeffs, plot_fb_err, plot_accel_err, print_coeffs, plot_vmec_comparison
from input_output import read_input, output_to_file, read_vmec_output

filename = 'HELIOTRON'
inputs = read_input('benchmarks/input.'+filename)

stell_sym      = inputs['stell_sym']
NFP            = inputs['NFP']
Psi_total      = inputs['Psi_total']
M              = inputs['Mpol'][-1] # these are arrays for continuation method
N              = inputs['Ntor'][-1] # but just taking the final values for now
Mnodes         = inputs['Mnodes'][-1]
Nnodes         = inputs['Nnodes'][-1]
bdry_ratio     = inputs['bdry_ratio'][-1]
pres_ratio     = inputs['pres_ratio'][-1]
error_ratio    = inputs['errr_ratio'][-1]
error_mode     = inputs['errr_mode']
bdry_mode      = inputs['bdry_mode']
node_mode      = inputs['node_mode']
presfun_params = inputs['presfun_params']
iotafun_params = inputs['iotafun_params']
axis           = inputs['axis']
bdry           = inputs['bdry']

# weights
weights = {'F' : error_ratio,  # force balance error
           'B' : 1e0,          # error in bdry
           'L' : 1e0}          # error in sum lambda_mn

nodes,volumes = get_nodes(Mnodes,Nnodes,NFP,surfs=node_mode,nr=25,nt=25,nz=0)

# interpolator
print('precomputing Fourier-Zernike basis')
derivatives = get_needed_derivatives(error_mode)
zern_idx = get_zern_basis_idx_dense(M,N)
lambda_idx = get_double_four_basis_idx_dense(M,N)
zernt = ZernikeTransform(nodes,zern_idx,NFP,derivatives)

# format boundary shape
bdry_poloidal, bdry_toroidal, bdryR, bdryZ = format_bdry(M, N, NFP, bdry, bdry_mode, bdry_mode)

# initial guess
print('computing initial guess')
cR_init,cZ_init = get_initial_guess_scale_bdry(axis,bdry,zern_idx,NFP,mode=bdry_mode,rcond=1e-6)
cL_init = np.zeros(len(lambda_idx))
x_init = jnp.concatenate([cR_init,cZ_init,cL_init])

fig, ax = plot_IC(cR_init,cZ_init,zern_idx,NFP,nodes,presfun_params,iotafun_params)
plt.show()

if stell_sym:
    sym_mat = symmetric_x(M,N)
    x_init = jnp.matmul(sym_mat.T,x_init)
else:
    sym_mat = np.eye(2*len(zern_idx) + len(lambda_idx))

if error_mode == 'force':
    equil_fun = compute_force_error_nodes
elif error_mode == 'accel':
    equil_fun = compute_accel_error_spectral
if bdry_mode == 'real':
    bdry_fun = compute_bc_err_RZ
elif bdry_mode == 'spectral':
    bdry_fun = compute_bc_err_four

args = (presfun_params,iotafun_params,Psi_total,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,
        NFP,zernt,nodes,volumes,zern_idx,lambda_idx,weights,sym_mat)

def lstsq_obj(x,pressfun_params,iotafun_params,Psi_total,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,
              NFP,zernt,nodes,volumes,zern_idx,lambda_idx,weights,sym_mat):
    
    cR,cZ,cL = unpack_x(jnp.matmul(sym_mat,x),len(zern_idx))
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

def callback(x,pressfun_params,iotafun_params,Psi_total,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,
             NFP,zernt,nodes,volumes,zern_idx,lambda_idx,weights,sym_mat):
    
    cR,cZ,cL = unpack_x(jnp.matmul(sym_mat,x),len(zern_idx))
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
    loss_rms = jnp.sum(loss**2)
    print('Weighted Loss: {:10.3e}  errRf: {:10.3e}  errZf: {:10.3e}  errRb: {:10.3e}  errZb: {:10.3e}  errL0: {:10.3e}'.format(
    loss_rms,errRf_rms,errZf_rms,errRb_rms,errZb_rms,errL0_rms))

if use_jax:
    jac = jacfwd(lstsq_obj,argnums=0)
    press_jac = jacfwd(lstsq_obj,argnums=1)
    iota_jac = jacfwd(lstsq_obj,argnums=2)
    psi_jac = jacfwd(lstsq_obj,argnums=3)
    Rb_jac = jacfwd(lstsq_obj,argnums=4)
    Zb_jac = jacfwd(lstsq_obj,argnums=5)
    
    lstsq_obj_jit = jit(lstsq_obj, static_argnums=np.arange(len(args))+1)
    jac_jit = jit(jacfwd(lstsq_obj_jit), static_argnums=np.arange(len(args))+1)
    print('compiling')
    foo = lstsq_obj_jit(x_init,*args).block_until_ready() 
    print('compiled objective')
    foo = jac_jit(x_init,*args).block_until_ready() 
    print('compiled jacobian')
else:
    lstsq_obj_jit = lstsq_obj
    jac_jit = '2-point'

# normalize weights
loss = lstsq_obj_jit(x_init,*args)
loss_rms = jnp.sum(loss**2)
weights = {'F' : error_ratio/np.sqrt(loss_rms),
           'B' : 1.0/np.sqrt(loss_rms),
           'L' : 1.0/np.sqrt(loss_rms)}
args = (presfun_params,iotafun_params,Psi_total,bdryR,bdryZ,bdry_poloidal,bdry_toroidal,
        NFP,zernt,nodes,volumes,zern_idx,lambda_idx,weights,sym_mat)

print('starting optimization')
out = scipy.optimize.least_squares(lstsq_obj_jit,
                                   x0=x_init,
                                   args=args,
                                   jac=jac_jit,
                                   method='trf',
                                   x_scale='jac',
                                   ftol=1e-8,
                                   xtol=1e-8,
                                   gtol=1e-8,
                                   max_nfev=100,
                                   verbose=2)
x = out['x']
print('Initial')
callback(x_init, *args)
print('Final')
callback(x, *args)

cR,cZ,cL = unpack_x(np.matmul(sym_mat,x),len(zern_idx))
axis_init = axis_posn(cR_init,cZ_init,zern_idx,NFP)
axis_final = axis_posn(cR,cZ,zern_idx,NFP)
print('initial: R0 = {:.3f}, Z0 = {:.3f}'.format(axis_init[0],axis_init[1]))
print('final:   R0 = {:.3f}, Z0 = {:.3f}'.format(axis_final[0],axis_final[1]))

# print_coeffs(cR,cZ,cL,zern_idx,lambda_idx)
output_to_file('benchmarks/output.'+filename,np.matmul(sym_mat,x),zern_idx,lambda_idx,NFP,Psi_total,presfun_params,iotafun_params,bdry)

vmec_data = read_vmec_output('benchmarks/VMEC/wout_'+filename+'.nc')
plot_vmec_comparison(vmec_data,cR,cZ,zern_idx,NFP)

theta = np.linspace(0,2*np.pi,1000)
phi = np.zeros_like(theta)
Rlcfs = eval_double_fourier(bdryR,np.array([bdry_poloidal,bdry_toroidal]).T,NFP,theta,phi)
Zlcfs = eval_double_fourier(bdryZ,np.array([bdry_poloidal,bdry_toroidal]).T,NFP,theta,phi)

fig, ax = plt.subplots(1,3,figsize=(9,3))
plot_coord_surfaces(cR_init,cZ_init,zern_idx,NFP,nr=10,nt=20,ax=ax[0],bdryR=Rlcfs,bdryZ=Zlcfs, title='Initial');
plot_coord_surfaces(cR,cZ,zern_idx,NFP,nr=10,nt=20,ax=ax[1],bdryR=Rlcfs,bdryZ=Zlcfs,title='Solution');
ax[2], im = plot_fb_err(cR,cZ,cL,zern_idx,lambda_idx,NFP,iotafun_params, presfun_params, Psi_total,
                domain='real', normalize='global', ax=ax[2], log=False, cmap='plasma',levels=10)
plt.show()

# plot acceleration error
# ax,imR,imZ = plot_accel_err(cR,cZ,zernt,zern_idx,NFP,pressfun_params,iotafun_params,Psi_total,domain='sfl',log=False,cmap='plasma')
