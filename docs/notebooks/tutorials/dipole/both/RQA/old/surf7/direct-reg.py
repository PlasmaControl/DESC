import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import os
import pickle
from mpi4py import MPI
import jax.lib.xla_bridge as xb

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure JAX uses CPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Initialize JAX distributed environment
jax.distributed.initialize()

from scipy.io import netcdf_file
import copy
#import matplotlib
#import matplotlib.pyplot as plt
import scipy
from scipy.constants import mu_0
import sys
import functools
import pickle

#import jax
import jax.numpy as np
from jax import jit, jacfwd

from netCDF4 import Dataset
import h5py

from desc.backend import put, fori_loop, np, sign

from desc.basis import FourierZernikeBasis, DoubleFourierSeries, FourierSeries

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import ConcentricGrid, LinearGrid, Grid, QuadratureGrid
from desc.io import InputReader, load
from desc.objectives import *
from desc.objectives.objective_funs import _Objective
#from desc.plotting import plot_1d, plot_2d, plot_3d, plot_section, plot_surfaces, plot_comparison

#from desc.plotting import *

from desc.transform import Transform
from desc.vmec import VMECIO
from desc.derivatives import Derivative
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import SplineProfile, PowerSeriesProfile

from desc.magnetic_fields import ( SplineMagneticField, 
                                  FourierCurrentPotentialField, ToroidalMagneticField,
                                  field_line_integrate)

from desc.backend import fori_loop, jit, np, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.utils import flatten_list, Timer, copy_coeffs, errorif, setdefault, svd_inv_null, warnif, cross, dot

import time

from numpy import ndarray

#from desc.fns_simp import (surf_int,
#                           _compute_magnetic_field_from_Current)

from find_dips import (iso_coords_interp,
                      bn_res,
                      B_dips,
                        B_sticks,
                      K_dips)

print("JAX devices:", jax.device_count(), jax.devices())
print("Global rank:", jax.process_index(), "of", jax.process_count())

# Plasma surfaces
eqname_QA = "/home/fcastro/LMStell/regcoil/rogerio/input.QA_final_output.h5" #vacuum equilibrium
eq_QA = desc.io.load(eqname_QA)[4]

eqname_QH = "/home/fcastro/LMStell/regcoil/rogerio/input.QH_final_output.h5" #vacuum equilibrium
eq_QH = desc.io.load(eqname_QH)[4]

eq = eq_QA    
###
surf_winding = load("iso_coords/surf.h5")

# Evaluate Green's function for flat tori
sgrid = LinearGrid(M = 100, N = 150, NFP = surf_winding.NFP)
#sdata = surf_winding.compute(["theta","zeta", "e^theta_s","e^zeta_s"], grid = sgrid)
name = 'iso_coords/'
#sdata = iso_coords_interp(name, sdata, sgrid, surf_winding)

# Load your data from npy files
w_vals = jnp.load("/scratch/gpfs/fcastro/w.npy")          # shape (N_points, ...)
w2_pol = jnp.load("/scratch/gpfs/fcastro/w2pol.npy")      # shape (total_points, ...)
w1_pol = jnp.load("/scratch/gpfs/fcastro/w1pol.npy")
w2_tor = jnp.load("/scratch/gpfs/fcastro/w2tor.npy")
w1_tor = jnp.load("/scratch/gpfs/fcastro/w1tor.npy")
        
l_vals = jnp.load("/scratch/gpfs/fcastro/l.npy")
l_u_vals = jnp.load("/scratch/gpfs/fcastro/l_u.npy")
l_v_vals = jnp.load("/scratch/gpfs/fcastro/l_v.npy")
    
e_u_vals = jnp.load("/scratch/gpfs/fcastro/e_u.npy")
e_v_vals = jnp.load("/scratch/gpfs/fcastro/e_v.npy")

du = jnp.load("/scratch/gpfs/fcastro/du.npy")
dv = jnp.load("/scratch/gpfs/fcastro/dv.npy")

jac_surf = jnp.load("/scratch/gpfs/fcastro/jac_surf.npy")
x_surf = jnp.load("/scratch/gpfs/fcastro/x_surf.npy")
coords = jnp.load("/scratch/gpfs/fcastro/coords.npy")

rhs = jnp.load('/scratch/gpfs/fcastro/rhs.npy')

# Convert to JAX arrays if not already
w_vals = jnp.asarray(w_vals)
w2_pol = jnp.asarray(w2_pol)
w1_pol = jnp.asarray(w1_pol)
w2_tor = jnp.asarray(w2_tor)
w1_tor = jnp.asarray(w1_tor)
    
l_vals = jnp.asarray(l_vals)
l_u_vals = jnp.asarray(l_u_vals)
l_v_vals = jnp.asarray(l_v_vals)
e_u_vals = jnp.asarray(e_u_vals)
e_v_vals = jnp.asarray(e_v_vals)
    
du = jnp.asarray(du)
dv = jnp.asarray(dv)

jac_surf = jnp.asarray(jac_surf)
x_surf = jnp.asarray(x_surf)
coords = jnp.asarray(coords)

rhs = jnp.asarray(rhs)

sdata = {
        "w": w_vals,
        "x": x_surf,
        "lambda_iso": l_vals,
        "lambda_u": l_u_vals,
        "lambda_v": l_v_vals,
        "e_u": e_u_vals,
        "e_v": e_v_vals,
        "|e_theta x e_zeta|": jac_surf,
    }
w2_pol = jnp.asarray( du_data["u_iso"] + 1j * du_data["v_iso"] )
    w1_pol = jnp.asarray( dd_data["u_iso"] + 1j * dd_data["v_iso"] )
    
    w2_tor = jnp.asarray( dr_data["u_iso"] + 1j * dr_data["v_iso"] )
    w1_tor = jnp.asarray( dl_data["u_iso"] + 1j * dl_data["v_iso"] )
eps = 1e-2
dt = eps
dz = eps
d0 = eps/3

grid_M = 10
grid_N = 10

egrid = LinearGrid(M = grid_M, N = grid_N, NFP = eq.NFP)
#edata = eq.compute(["n_rho","B"], grid = egrid)

N = 20 # Terms toa pproximate the infinite series

# Numer of dipoles
#sMv = jnp.asarray([5,10,15,20,])
sMv = jnp.asarray([15])
sNv = jnp.asarray([15])
#sNv = sMv

b_chi = []
Bn_chi = []
max_I = []
min_I = []

for i in range(0,len(sMv)):

    p_M = sMv[i]
    p_N = sNv[i]
    
    x = jnp.ones( ( p_M * 2 + 1 ) * ( p_N * 2 + 1 ) * 2 + 0 )
    
    fun_wrapped1 = lambda x: bn_res(w_vals,
                                       w2_pol, w1_pol, w2_tor, w1_tor,
                                       l_vals, l_u_vals, l_v_vals,
                                       e_u_vals, e_v_vals,
                                       du, dv,
                                       d0,
                                       sgrid, x_surf,jac_surf, coords)
    
    A = Derivative(fun_wrapped1, deriv_mode="looped").compute(x)
    # Force CPU device (optional)
    #A = jax.device_put(A, jax.devices("cpu")[0])

    alpha = 1e-15
    #A_inv, _ = svd_inv_null( A.T @ A + alpha * jnp.eye( A.shape[1] ) )
    
    # Find the regularized solution of dipoles
    #soln = A_inv @ ( rhs @ A)
    
    start = time.time()
    #A_pinv = jnp.linalg.pinv(A).block_until_ready()s
    soln = (jnp.linalg.pinv(A.T @ A + alpha * jnp.eye( A.shape[1] )).block_until_ready()) @ ( rhs @ A)
    print(f"Computed pseudo-inverse in {time.time() - start:.2f} seconds")
    #soln = jnp.linalg.pinv(A) @ rhs
    #soln = jnp.linalg.pinv(A + alpha * jnp.eye( A.shape[0],A.shape[1] )) @ rhs
    
    test = A@soln
    tsize = edata2['n_rho'].shape[0]
    B_d2 = jnp.column_stack((test[0:tsize],test[tsize:tsize*2],test[tsize*2:tsize*3]))
    
    K_d = K_dips(p_M, p_N,
                 sdata, sgrid, surf_winding,
                 soln,
                 dt,dz, N, d0)
    
    B0 = B_d2 - B_s2
    B_total = B_d2 + B_sec2
    Bn_total = dot(edata2["n_rho"],B_total)
    K0 = K_d
    
    error = surf_int( dot(B0,B0) ** 2, edata2, egrid2 )
    b_chi.append(error)
    Bn_chi.append( max(abs(Bn_total * dot(B_total,B_total)**(-1/2)))) 
    min_I.append(min(abs(soln)))
    max_I.append(max(abs(soln)))
    
    jnp.save('reg_soln_M_' + str(sMv[i]) + '_N_' + str(sNv[i]) + '.npy' ,soln)
    jnp.save('reg_error_M_' + str(sMv[i]) + '_N_' + str(sNv[i]) + '.npy' , error)