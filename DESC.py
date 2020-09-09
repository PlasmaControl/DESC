import numpy as np
import functools
import scipy.optimize
import matplotlib.pyplot as plt
from init_guess import get_initial_guess_scale_bdry
from nodes import get_nodes
from zernike import ZernikeTransform, get_zern_basis_idx_dense, get_double_four_basis_idx_dense, axis_posn, symmetric_x, eval_double_fourier
from boundary_conditions import format_bdry
from backend import jnp, conditional_decorator, jit, use_jax, fori_loop, put, presfun, iotafun
from backend import get_needed_derivatives, unpack_x, jacfwd
from plotting import plot_IC, plot_coord_surfaces, plot_coeffs, plot_fb_err, plot_accel_err, print_coeffs, plot_vmec_comparison
from input_output import read_input, output_to_file, read_vmec_output
from objective_funs import get_equil_obj_fun

filename = 'SOLOVEV'
inputs = read_input('benchmarks/DESC/inputs/input.'+filename)

stell_sym  = inputs['stell_sym']
NFP        = inputs['NFP']
Psi_total  = inputs['Psi_total']
M          = inputs['Mpol'][-1] # these are arrays for continuation method
N          = inputs['Ntor'][-1] # but just taking the final values for now
Mnodes     = inputs['Mnodes'][-1]
Nnodes     = inputs['Nnodes'][-1]
bdry_ratio = inputs['bdry_ratio'][-1]
pres_ratio = inputs['pres_ratio'][-1]
zeta_ratio = inputs['zeta_ratio'][-1]
errr_ratio = inputs['errr_ratio'][-1]
errr_mode  = inputs['errr_mode']
bdry_mode  = inputs['bdry_mode']
node_mode  = inputs['node_mode']
cP         = inputs['presfun_params']
cI         = inputs['iotafun_params']
axis       = inputs['axis']
bdry       = inputs['bdry']

# weights
weights = {'F' : 1e0, # force balance error
           'B' : 1e0, # error in bdry
           'L' : 1e0} # error in sum lambda_mn

# interpolation nodes
nodes,volumes = get_nodes(Mnodes,Nnodes,NFP,surfs=node_mode,nr=25,nt=25,nz=0)

# interpolator
print('precomputing Fourier-Zernike basis')
derivatives = get_needed_derivatives('all') # changed option from error_mode
zern_idx = get_zern_basis_idx_dense(M,N)
lambda_idx = get_double_four_basis_idx_dense(M,N)
zernt = ZernikeTransform(nodes,zern_idx,NFP,derivatives)

# format boundary shape
bdry_poloidal, bdry_toroidal, bdryR, bdryZ = format_bdry(M, N, NFP, bdry, bdry_mode, bdry_mode)

# stellarator symmetry
if stell_sym:
    sym_mat = symmetric_x(M,N)
else:
    sym_mat = np.eye(2*len(zern_idx) + len(lambda_idx))

# initial guess
print('computing initial guess')
cR_init,cZ_init = get_initial_guess_scale_bdry(axis,bdry,zern_idx,NFP,mode=bdry_mode,rcond=1e-6)
cL_init = np.zeros(len(lambda_idx))
x_init = jnp.concatenate([cR_init,cZ_init,cL_init])
x_init = jnp.matmul(sym_mat.T,x_init)
fig, ax = plot_IC(cR_init,cZ_init,zern_idx,NFP,nodes,cP,cI)
plt.show()

# equilibrium objective function
equil_obj,callback = get_equil_obj_fun(M,N,zern_idx,lambda_idx,stell_sym,errr_mode,bdry_mode)
args = (bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio,pres_ratio,zeta_ratio,errr_ratio,NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights)

if use_jax:
    jac = jacfwd(equil_obj,argnums=0)
    pres_jac = jacfwd(equil_obj,argnums=1)
    iota_jac = jacfwd(equil_obj,argnums=2)
    psi_jac = jacfwd(equil_obj,argnums=3)
    Rb_jac = jacfwd(equil_obj,argnums=4)
    Zb_jac = jacfwd(equil_obj,argnums=5)
    
    equil_obj_jit = jit(equil_obj, static_argnums=np.arange(len(args))+1)
    jac_jit = jit(jacfwd(equil_obj_jit), static_argnums=np.arange(len(args))+1)
    print('compiling')
    foo = equil_obj_jit(x_init,*args).block_until_ready() 
    print('compiled objective')
    foo = jac_jit(x_init,*args).block_until_ready() 
    print('compiled jacobian')
else:
    equil_obj_jit = equil_obj
    jac_jit = '2-point'

# normalize weights
loss = equil_obj_jit(x_init,*args)
loss_rms = jnp.sum(loss**2)
weights = {'F' : 1.0/np.sqrt(loss_rms),
           'B' : 1.0/np.sqrt(loss_rms),
           'L' : 1.0/np.sqrt(loss_rms)}
args = (bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio,pres_ratio,zeta_ratio,errr_ratio,NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights)

print('starting optimization')
out = scipy.optimize.least_squares(equil_obj_jit,
                                   x0=x_init,
                                   args=args,
                                   jac=jac_jit,
                                   method='trf',
                                   x_scale='jac',
                                   ftol=1e-8,
                                   xtol=1e-8,
                                   gtol=1e-8,
                                   max_nfev=1000,
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
output_to_file('benchmarks/DESC/outputs/output.'+filename,np.matmul(sym_mat,x),zern_idx,lambda_idx,NFP,Psi_total,cP,cI,bdry)

vmec_data = read_vmec_output('benchmarks/VMEC/outputs/wout_'+filename+'.nc')
plot_vmec_comparison(vmec_data,cR,cZ,zern_idx,NFP)

theta = np.linspace(0,2*np.pi,1000)
phi = np.zeros_like(theta)
Rlcfs = eval_double_fourier(bdryR,np.array([bdry_poloidal,bdry_toroidal]).T,NFP,theta,phi)
Zlcfs = eval_double_fourier(bdryZ,np.array([bdry_poloidal,bdry_toroidal]).T,NFP,theta,phi)

fig, ax = plt.subplots(1,3,figsize=(9,3))
plot_coord_surfaces(cR_init,cZ_init,zern_idx,NFP,nr=10,nt=20,ax=ax[0],bdryR=Rlcfs,bdryZ=Zlcfs, title='Initial');
plot_coord_surfaces(cR,cZ,zern_idx,NFP,nr=10,nt=20,ax=ax[1],bdryR=Rlcfs,bdryZ=Zlcfs,title='Solution');
ax[2], im = plot_fb_err(cR,cZ,cL,zern_idx,lambda_idx,NFP,cI,cP,Psi_total,
                domain='real', normalize='global', ax=ax[2], log=False, cmap='plasma',levels=10)
plt.show()

# plot acceleration error
# ax,imR,imZ = plot_accel_err(cR,cZ,zernt,zern_idx,NFP,pressfun_params,iotafun_params,Psi_total,domain='sfl',log=False,cmap='plasma')
