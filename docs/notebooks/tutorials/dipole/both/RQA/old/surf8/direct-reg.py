import os
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))

import numpy as np
import os

from scipy.io import netcdf_file
import copy
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.constants import mu_0
import sys
import functools
import pickle

import jax
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
from desc.plotting import plot_1d, plot_2d, plot_3d, plot_section, plot_surfaces, plot_comparison

from desc.plotting import *

from desc.transform import Transform
from desc.vmec import VMECIO
from desc.derivatives import Derivative
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import SplineProfile, PowerSeriesProfile

from desc.magnetic_fields import ( SplineMagneticField, 
                                  FourierCurrentPotentialField, ToroidalMagneticField,
                                  field_line_integrate)

import desc.examples

from desc.backend import fori_loop, jit, np, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.coils import *

from desc.utils import flatten_list, Timer, copy_coeffs, errorif, setdefault, svd_inv_null, warnif, cross, dot

from desc.optimize import lsqtr, lsq_auglag

from scipy.optimize import NonlinearConstraint 

import time

from numpy import ndarray

from desc.fns_simp import (plot_figure,
                           plot_figure2,
                           plot_xy,
                           surf_int,
                           _compute_magnetic_field_from_Current)

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

# Plot both surfaces
plot_comparison([surf_winding,eq_QA,eq_QH],labels=["surf","RQA","RQH"],theta=0,rho=jnp.array(1.0))


# Evaluate Green's function for flat tori
sgrid = LinearGrid(M = 100, N = 150, NFP = surf_winding.NFP)
sdata = surf_winding.compute(["theta","zeta", "e^theta_s","e^zeta_s"], grid = sgrid)
name = 'iso_coords/'
sdata = iso_coords_interp(name, sdata, sgrid, surf_winding)

G = jnp.load("iso_coords/G.npy")
K_sec = G * cross(sdata["n_rho"], sdata["e^u_s"])
K_sv = surf_winding.compute(["K"],grid = sgrid)["K"]

# Confirm that variables have the same shape
plot_figure2(sdata["u_iso"],sgrid,''r' $ u(\theta,\zeta)$ ')
plot_figure2(sdata["v_iso"],sgrid,''r' $ v(\theta,\zeta)$ ')
#plot_figure2(sdata["lambda_iso"],sgrid,''r' $ \lambda $ ')

eps = 1e-2
dt = eps
dz = eps
d0 = eps/3

grid_M = 10
grid_N = 10

egrid = LinearGrid(M = grid_M, N = grid_N, NFP = eq.NFP)
edata = eq.compute(["n_rho","B"], grid = egrid)

#B_s = edata["B"]
B_sec = _compute_magnetic_field_from_Current( sgrid, K_sec, surf_winding, eq, egrid, basis = "rpz" )
B_s = _compute_magnetic_field_from_Current( sgrid, K_sv, surf_winding, eq, egrid, basis = "rpz" )
#B_s = _compute_magnetic_field_from_Current( sgrid, K_sec, surf_winding, eq, egrid, basis = "rpz" )
rhs = jnp.concatenate((B_s[:,0],B_s[:,1],B_s[:,2]))
#- dot( edata["n_rho"],  B_sec )

plot_figure2( dot(B_s,B_s) ** (1/2), egrid,''r' $ | \mathbf{B_{s}} |$ ')

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

grid_M2 = grid_M #40
grid_N2 = grid_N #40

egrid2 = LinearGrid(M = grid_M2, N = grid_N2, NFP = eq.NFP)
edata2 = eq.compute(["n_rho"], grid = egrid2)

B_sec2 = jnp.load('iso_coords/B_sec_'+str(grid_M2)+'x'+str(grid_N2)+'.npy')
#_compute_magnetic_field_from_Current( sgrid, K_sec, surf_winding, eq, egrid2, basis = "rpz" )
B_s2 = _compute_magnetic_field_from_Current( sgrid, K_sv, surf_winding, eq, egrid2, basis = "rpz" )
#B_s2 = edata2["B"]

for i in range(0,len(sMv)):

    p_M = sMv[i]
    p_N = sNv[i]
    
    x = jnp.ones( ( p_M * 2 + 1 ) * ( p_N * 2 + 1 ) * 2 + 0 )
    
    fun_wrapped1 = lambda x: bn_res(p_M, p_N, # Dipole pairs in toroidal direction 
                                    sdata, sgrid, surf_winding, #winding_surf,
                                    x, 
                                    dt,dz, N, d0,
                                    eq, egrid)
    
    A = Derivative(fun_wrapped1, deriv_mode="looped").compute(x)
    # Force CPU device (optional)
    #A = jax.device_put(A, jax.devices("cpu")[0])

    alpha = 1e-15
    #A_inv, _ = svd_inv_null( A.T @ A + alpha * jnp.eye( A.shape[1] ) )
    
    # Find the regularized solution of dipoles
    #soln = A_inv @ ( rhs @ A)
    
    start = time.time()
    #A_pinv = jnp.linalg.pinv(A).block_until_ready()
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
    
    
    #B0 = B_d2 - B_s2
    B0 = B_d2 - B_s2
    B_total = B_d2 + B_sec2
    Bn_total = dot(edata2["n_rho"],B_total)
    K0 = K_d
    
    #res = A @ soln - rhs
    
    #plot_figure2( res, egrid, ''r' $ \mathbf{B \cdot n}$ ' )
    plot_figure2( Bn_total * dot(B_total,B_total)**(1/2), egrid2,
                 ''r' $ \frac{ (\mathbf{B_{sec} + B_{dip}}) \cdot \mathbf{n} }{| \mathbf{B_{sec} + B_{dip}} | }$ ' )
    
    plot_figure2( dot(B_d2,B_d2)**(1/2), egrid2, ''r' $ | \mathbf{B_{dip}} | $ ' )
    
    plot_figure2( dot(B0,B0)**(1/2) * dot(B_s2,B_s2)**(-1/2), 
                 egrid2, 
                 ''r' $ \frac{| \mathbf{B_s - B_{dip}}  |}{|\mathbf{B_s}|} $ ' )
    
    plot_figure2( dot(K0,K0)**(1/2), sgrid, ''r' $ | \mathbf{K} | $ ' )
    
    error = surf_int( dot(B0,B0) ** 2, edata2, egrid2 )
    b_chi.append(error)
    Bn_chi.append( max(abs(Bn_total * dot(B_total,B_total)**(-1/2)))) 
    min_I.append(min(abs(soln)))
    max_I.append(max(abs(soln)))
    
    jnp.save('reg_soln_M_' + str(sMv[i]) + '_N_' + str(sNv[i]) + '.npy' ,soln)
    jnp.save('reg_error_M_' + str(sMv[i]) + '_N_' + str(sNv[i]) + '.npy' , error)