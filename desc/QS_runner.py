import numpy as np
import functools
import scipy.optimize
import matplotlib.pyplot as plt
from nodes import get_nodes
from boundary_conditions import format_bdry
from zernike import ZernikeTransform, get_zern_basis_idx_dense, get_double_four_basis_idx_dense, symmetric_x
from objective_funs import get_equil_obj_fun
from backend import jnp, conditional_decorator, jit, use_jax, fori_loop, put, presfun, iotafun
from backend import get_needed_derivatives, unpack_x, rms, jacfwd, jacrev
from plotting import plot_comparison
from input_output import load_from_file

filename = 'SOLOVEV'
x,zern_idx,lambda_idx,NFP,Psi_total,cP,cI,bdry = load_from_file('benchmarks/DESC/outputs/output.'+filename)

stell_sym  = True
M          = np.amax(zern_idx[:,1])
N          = np.amax(zern_idx[:,2])
Mnodes     = 12
Nnodes     = 0
bdry_ratio = 1.0
pres_ratio = 1.0
zeta_ratio = 1.0
errr_ratio = 1.0
errr_mode  = 'force'
bdry_mode  = 'spectral'
node_mode  = 'cheb2'

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
bdry_poloidal,bdry_toroidal,bdryR,bdryZ = format_bdry(M,N,NFP,bdry,bdry_mode,bdry_mode)

# stellarator symmetry
if stell_sym:
    sym_mat = symmetric_x(M,N)
else:
    sym_mat = np.eye(2*len(zern_idx) + len(lambda_idx))

# equilibrium objective function
equil_obj,callback = get_equil_obj_fun(M,N,zern_idx,lambda_idx,stell_sym,errr_mode,bdry_mode)
args = (bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio,pres_ratio,zeta_ratio,errr_ratio,NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights)

# jacobian functions
jac      = jacfwd(equil_obj,argnums=0)
Rb_jac   = jacfwd(equil_obj,argnums=1)
Zb_jac   = jacfwd(equil_obj,argnums=2)
pres_jac = jacfwd(equil_obj,argnums=3)
iota_jac = jacfwd(equil_obj,argnums=4)
psi_jac  = jacfwd(equil_obj,argnums=5)

# jacobian evaluations
dF_dx  = jac(x,bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio,pres_ratio,zeta_ratio,errr_ratio,NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights)
dF_dRb = Rb_jac(x,bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio,pres_ratio,zeta_ratio,errr_ratio,NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights)
dF_dZb = Zb_jac(x,bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio,pres_ratio,zeta_ratio,errr_ratio,NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights)
dF_dP  = pres_jac(x,bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio,pres_ratio,zeta_ratio,errr_ratio,NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights)
dF_dI  = iota_jac(x,bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio,pres_ratio,zeta_ratio,errr_ratio,NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights)
