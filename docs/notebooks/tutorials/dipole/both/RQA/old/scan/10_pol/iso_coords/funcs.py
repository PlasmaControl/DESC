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
import jax.numpy as jnpå
from jax import jit, jacfwd

from netCDF4 import Dataset
import h5py

from desc.backend import put, fori_loop, jnp, sign

from desc.basis import FourierZernikeBasis, DoubleFourierSeries, FourierSeries
#from desc.basis import DoubleChebyshevSeries

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

from desc.backend import fori_loop, jit, jnp, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.coils import *

from desc.utils import flatten_list
from desc.compute.utils import cross
from desc.compute.utils import dot

from desc.optimize import lsqtr, lsq_auglag

from scipy.optimize import NonlinearConstraint 

from desc.magnetic_fields import FourierCurrentPotentialField

import time

import numpy as np
from numpy.linalg import eig

from scipy.linalg import null_space
from numpy import ndarray

from desc.fns_simp import (plot_figure,
                           surf_int,
                           _compute_magnetic_field_from_Current,)

#from green import green_eval 

from desc.finite_diff2 import (first_derivative_t, first_derivative_z,
                              )

from desc.finite_diff4 import (first_derivative_t2, first_derivative_z2,
                              )

# Define function to find A and its respective derivatives
def eqn_residual(grid, data, x,):

    f_t = first_derivative_t(x,
                             data,
                             grid,)
    
    f_z = first_derivative_z(x,
                             data,
                             grid,)
    
    f_tt = first_derivative_t(f_t,
                              data,
                              grid,)
    
    f_zz = first_derivative_z(f_z,
                              data,
                              grid,)
    
    f_tz = first_derivative_z(f_t,
                              data,
                              grid,)
    
    ##################################################################################################################
    
    # Surface Laplacian of theta
    nabla_gamma_2_f = (dot(data["e^theta_gamma"],data["e^theta_gamma_t"])*f_t
                                   + dot(data["e^theta_gamma"],data["e^theta_gamma"])*f_tt
                                   + dot(data["e^theta_gamma"],data["e^zeta_gamma_t"])*f_z
                                   + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*f_tz
                                   + dot(data["e^zeta_gamma"],data["e^theta_gamma_z"])*f_t
                                   + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*f_tz
                                   + dot(data["e^zeta_gamma"],data["e^zeta_gamma_z"])*f_z
                                   + dot(data["e^zeta_gamma"],data["e^zeta_gamma"])*f_zz
                      )
    
    #nabla_gamma_2_f
    #jnp.asarray([surf_int(nabla_gamma_2_f**2,data,grid)])
    return nabla_gamma_2_f

# Define function to find A and its respective derivatives
def v_residual(grid, data, y,):

    f_t_ = first_derivative_t2(y,
                             data,
                             grid,)
    
    f_z_ = first_derivative_z2(y,
                             data,
                             grid,)
    
    grad_f_ = (f_t_*data["e^theta_gamma"].T 
              + f_z_*data["e^zeta_gamma"].T
             ).T
    
    return jnp.concatenate((grad_f_[:,0],grad_f_[:,1],grad_f_[:,2]))

# Define function to find A and its respective derivatives
def v_residual2(grid, data, yp,):

    y = yp[0:yp.shape[0]-1]
    
    f_t_ = first_derivative_t(y,
                             data,
                             grid,)
    
    f_z_ = first_derivative_z(y,
                             data,
                             grid,)
    
    grad_f_ = (yp[yp.shape[0]]*data["e^theta_gamma"]
               + (f_t_*data["e^theta_gamma"].T 
                  + f_z_*data["e^zeta_gamma"].T
                 ).T
              )
    
    return jnp.concatenate((grad_f_[:,0],grad_f_[:,1],grad_f_[:,2]))

def div_(grid, data, x,):
    
    ##################################################################################################################
    
    f_t = first_derivative_t(x,
                             data,
                             grid,)
    
    f_z = first_derivative_z(x,
                             data,
                             grid,)
    
    f_tt = first_derivative_t(f_t,
                              data,
                              grid,)
    
    f_zz = first_derivative_z(f_z,
                              data,
                              grid,)
    
    f_tz = first_derivative_z(f_t,
                              data,
                              grid,)
    
    grad_f = (f_t*data["e^theta_gamma"].T 
              + f_z*data["e^zeta_gamma"].T
             ).T
    
    # Surface Laplacian of theta
    div2 = (dot(data["e^theta_gamma"],data["e^theta_gamma_t"])*f_t
                                   + dot(data["e^theta_gamma"],data["e^theta_gamma"])*f_tt
                                   + dot(data["e^theta_gamma"],data["e^zeta_gamma_t"])*f_z
                                   + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*f_tz
                                   + dot(data["e^zeta_gamma"],data["e^theta_gamma_z"])*f_t
                                   + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*f_tz
                                   + dot(data["e^zeta_gamma"],data["e^zeta_gamma_z"])*f_z
                                   + dot(data["e^zeta_gamma"],data["e^zeta_gamma"])*f_zz
                      )
    
    
    return grad_f, div2, div2*dot(grad_f,grad_f)**(-1/2)

def div_2(grid, data, x,):
    
    ##################################################################################################################
    
    f_t = first_derivative_t2(x,
                             data,
                             grid,)
    
    f_z = first_derivative_z2(x,
                             data,
                             grid,)
    
    f_tt = first_derivative_t2(f_t,
                              data,
                              grid,)
    
    f_zz = first_derivative_z2(f_z,
                              data,
                              grid,)
    
    f_tz = first_derivative_z2(f_t,
                              data,
                              grid,)
    
    grad_f = (f_t*data["e^theta_gamma"].T 
              + f_z*data["e^zeta_gamma"].T
             ).T
    
    # Surface Laplacian of theta
    div2 = (dot(data["e^theta_gamma"],data["e^theta_gamma_t"])*f_t
                                   + dot(data["e^theta_gamma"],data["e^theta_gamma"])*f_tt
                                   + dot(data["e^theta_gamma"],data["e^zeta_gamma_t"])*f_z
                                   + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*f_tz
                                   + dot(data["e^zeta_gamma"],data["e^theta_gamma_z"])*f_t
                                   + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*f_tz
                                   + dot(data["e^zeta_gamma"],data["e^zeta_gamma_z"])*f_z
                                   + dot(data["e^zeta_gamma"],data["e^zeta_gamma"])*f_zz
                      )
    
    
    return grad_f, div2, div2*dot(grad_f,grad_f)**(-1/2)

# Constraint norm of modes
def norm_(x):
    return jnp.asarray([jnp.sum(x**2)])

def phi_eval(y, b_basis, data, grid):
 
    b_mn = y[0:b_basis.num_modes]
    
    trans = desc.transform.Transform(grid, b_basis, derivs=2, 
                                     rcond='auto', build=True, build_pinv=False, method='auto')
    
    # Compute V, rho and derivatives on the grid
    fs = {
        "b": desc.transform.Transform(grid = grid, basis = b_basis),
        # First derivatives of a
        "b_t": desc.transform.Transform(grid = grid, basis = b_basis),
        "b_z": desc.transform.Transform(grid = grid, basis = b_basis),
        # First derivatives of f
        "b_tt": desc.transform.Transform(grid = grid, basis = b_basis),
        "b_tz": desc.transform.Transform(grid = grid, basis = b_basis),
        "b_zz": desc.transform.Transform(grid = grid, basis = b_basis),
    }
    
    ##
    fs["b"] = trans.transform(b_mn, dt=0, dz=0)
    fs["b_t"] = trans.transform(b_mn, dt=1, dz=0)
    fs["b_z"] = trans.transform(b_mn, dt=0, dz=1)
    
    fs["b_tt"] = trans.transform(b_mn, dt=2, dz=0)
    fs["b_tz"] = trans.transform(b_mn, dt=1, dz=1)
    fs["b_zz"] = trans.transform(b_mn, dt=0, dz=2)
    
    # Define nabla_gamma_b
    nabla_gamma_b = ((y[b_basis.num_modes] + fs["b_t"])*data["e^theta_gamma"].T 
                     + (y[b_basis.num_modes+1] + fs["b_z"])*data["e^zeta_gamma"].T).T
    
    b_sol = (y[b_basis.num_modes]*data["theta"] 
             + y[b_basis.num_modes+1]*data["zeta"] 
             + fs["b"]) 
    
    # Surface Laplacian of phi
    laplace = (y[b_basis.num_modes]*data["nabla_gamma^2_theta"] 
               + y[b_basis.num_modes+1]*data["nabla_gamma^2_zeta"]
               + ( dot(data["e^theta_gamma"],data["e^theta_gamma_t"])*fs["b_t"] 
                  + dot(data["e^theta_gamma"],data["e^theta_gamma"])*fs["b_tt"]
                  + dot(data["e^theta_gamma"],data["e^zeta_gamma_t"])*fs["b_z"]
                  + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*fs["b_tz"]
                  + dot(data["e^zeta_gamma"],data["e^theta_gamma_z"])*fs["b_t"]
                  + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*fs["b_tz"]
                  + dot(data["e^zeta_gamma"],data["e^zeta_gamma_z"])*fs["b_z"]
                  + dot(data["e^zeta_gamma"],data["e^zeta_gamma"])*fs["b_zz"]
                 )
              )
    
    return b_sol, nabla_gamma_b, laplace

def b_inv(sdata, # Data on the winding surface
          sgrid, # Source grid (winding surface)
          surf_winding,
          eq,
          edata,
          egrid,
          y,# modes of current
          b_basis,
          transform1
         ):
    
    
    b_mn = y[0:b_basis.num_modes]
    # Compute V, rho and derivatives on the grid
    fs = {# a
        #"b": transform1.transform(b_mn),
        # First derivatives of rho
        "b_t": transform1.transform(b_mn, dt = 1),
        "b_z": transform1.transform(b_mn, dz = 1),
    }
    
    gradp = ((y[b_basis.num_modes] + fs["b_t"])*sdata["e^theta_gamma"].T 
             + (y[b_basis.num_modes+1] + fs["b_z"])*sdata["e^zeta_gamma"].T).T
    # Define nabla_gamma_b
    K_ = cross(sdata["n_rho"],gradp)
                    
    B_ = _compute_magnetic_field_from_Current(sgrid,
                                                 K_,
                                                 surf_winding,
                                                 eq,
                                                 egrid,
                                                 basis="rpz")
    
    
    return jnp.concatenate((B_[:,0],
                            B_[:,1],
                            B_[:,2])
                          )

def b_inv2(sdata, # Data on the winding surface
          sgrid, # Source grid (winding surface)
          surf_winding,
          eq,
          edata,
          egrid,
          y,# modes of current
          b_basis,
          transform1
         ):
    
    
    b_mn = y[0:b_basis.num_modes]
    # Compute V, rho and derivatives on the grid
    fs = {# a
        #"b": transform1.transform(b_mn),
        # First derivatives of rho
        "b_t": transform1.transform(b_mn, dt = 1),
        "b_z": transform1.transform(b_mn, dz = 1),
    }
    
    gradp = ((y[b_basis.num_modes] + fs["b_t"])*sdata["e^theta_gamma"].T 
             + (y[b_basis.num_modes+1] + fs["b_z"])*sdata["e^zeta_gamma"].T).T
    
    # Define nabla_gamma_b
    K_ = cross(sdata["n_rho"],gradp)
                    
    B_ = _compute_magnetic_field_from_Current(sgrid,
                                                 K_,
                                                 surf_winding,
                                                 eq,
                                                 egrid,
                                                 basis="rpz")
    
    
    return jnp.concatenate((dot(edata["e_theta"],B_),
                            dot(edata["e_zeta"],B_))
                          )

def b_inv3(sdata, # Data on the winding surface
          sgrid, # Source grid (winding surface)
          surf_winding,
          eq,
          edata,
          egrid,
          y,# modes of current
          b_basis,
          transform1
         ):
    
    
    b_mn = y[0:b_basis.num_modes]
    # Compute V, rho and derivatives on the grid
    fs = {# a
        #"b": transform1.transform(b_mn),
        # First derivatives of rho
        "b_t": transform1.transform(b_mn, dt = 1),
        "b_z": transform1.transform(b_mn, dz = 1),
    }
    
    gradp = ((y[b_basis.num_modes] + fs["b_t"])*sdata["e^theta_gamma"].T 
             + (y[b_basis.num_modes+1] + fs["b_z"])*sdata["e^zeta_gamma"].T).T
    
    # Define nabla_gamma_b
    K_ = cross(sdata["n_rho"],gradp)
                    
    B_ = _compute_magnetic_field_from_Current(sgrid,
                                                 K_,
                                                 surf_winding,
                                                 eq,
                                                 egrid,
                                                 basis="rpz")
    
    
    return jnp.concatenate((dot(edata["n_rho"],B_),
                            dot(edata["e_theta"],B_),
                            dot(edata["e_zeta"],B_))
                          )

def b_theta_inv(sdata, # Data on the winding surface
                  sgrid, # Source grid (winding surface)
                  surf_winding,
                  eq,
                  edata,
                  egrid,
                  y,# modes of current
                  b_basis,
                  transform1
                 ):
    
    
    b_mn = y[0:b_basis.num_modes]
    # Compute V, rho and derivatives on the grid
    fs = {# a
        #"b": transform1.transform(b_mn),
        # First derivatives of rho
        "b_t": transform1.transform(b_mn, dt = 1),
        "b_z": transform1.transform(b_mn, dz = 1),
    }
    
    gradp = ((y[b_basis.num_modes] + fs["b_t"])*sdata["e^theta_gamma"].T 
             + (y[b_basis.num_modes+1] + fs["b_z"])*sdata["e^zeta_gamma"].T).T
    
    # Define nabla_gamma_b
    K_ = cross(sdata["n_rho"],gradp)
                    
    B_ = _compute_magnetic_field_from_Current(sgrid,
                                                 K_,
                                                 surf_winding,
                                                 eq,
                                                 egrid,
                                                 basis="rpz")
    
    
    return dot(edata["e_theta"],B_)

def b_zeta_inv(sdata, # Data on the winding surface
                  sgrid, # Source grid (winding surface)
                  surf_winding,
                  eq,
                  edata,
                  egrid,
                  y,# modes of current
                  b_basis,
                  transform1
                 ):
    
    
    b_mn = y[0:b_basis.num_modes]
    # Compute V, rho and derivatives on the grid
    fs = {# a
        #"b": transform1.transform(b_mn),
        # First derivatives of rho
        "b_t": transform1.transform(b_mn, dt = 1),
        "b_z": transform1.transform(b_mn, dz = 1),
    }
    
    gradp = ((y[b_basis.num_modes] + fs["b_t"])*sdata["e^theta_gamma"].T 
             + (y[b_basis.num_modes+1] + fs["b_z"])*sdata["e^zeta_gamma"].T).T
    
    # Define nabla_gamma_b
    K_ = cross(sdata["n_rho"],gradp)
                    
    B_ = _compute_magnetic_field_from_Current(sgrid,
                                                 K_,
                                                 surf_winding,
                                                 eq,
                                                 egrid,
                                                 basis="rpz")
    
    
    return dot(edata["e_zeta"],B_)

def b_calc(sdata, # Data on the winding surface
          sgrid, # Source grid (winding surface)
          surf_winding,
          eq,
          edata,
          egrid,
          y,# modes of current
          b_basis,
          transform1
         ):
    
    
    b_mn = y[0:b_basis.num_modes]
    # Compute V, rho and derivatives on the grid
    fs = {# a
        #"b": transform1.transform(b_mn),
        # First derivatives of rho
        "b_t": transform1.transform(b_mn, dt = 1),
        "b_z": transform1.transform(b_mn, dz = 1),
    }
    
    gradp = ((y[b_basis.num_modes] + fs["b_t"])*sdata["e^theta_gamma"].T 
             + (y[b_basis.num_modes+1] + fs["b_z"])*sdata["e^zeta_gamma"].T).T
    # Define nabla_gamma_b
    K_ = cross(sdata["n_rho"],gradp)
                    
    B_ = _compute_magnetic_field_from_Current(sgrid,
                                                 K_,
                                                 surf_winding,
                                                 eq,
                                                 egrid,
                                                 basis="rpz")
    
    
    return B_

def pol_cont_int(x_vals,
                 f
                 ):
    
    
    #udote = dot(u,edata["e_theta"])
    
    # Points along integration contour
    #x_vals = egrid.nodes[:,1]
    
    # Initiate a variable to compute the line integral of K_s x n
    test = 0
    
    # For loop for trapezoidal rule as an approximation to the line integral
    for i in range(0,len(f)-1):
        
        test = (test + (x_vals[i+1] - x_vals[i])*1/2*(f[i+1] + f[i])
               )
    
        #test = (test + (x_vals[i+1] - x_vals[i])*1/2*(udote[i+1]*np.sqrt(intdata["g_zz"][i+1])
        #                                              + udote[i]*np.sqrt(intdata["g_zz"][i])
        #                                             )
        #       )
    
    
    return test

def metric_dervs(u_t,u_z,v_t,v_z, 
                 x,):
    
    rh = u_t.shape[0]
    fh = x[0:rh]
    gh = x[rh:2*rh]
    
    return jnp.concatenate((u_t*fh + u_z*gh,
                            v_t*fh + v_z*gh,))