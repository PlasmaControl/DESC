import numpy as np
import os
# os.environ["JAX_LOG_COMPILES"] = "True"
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
                                  #biot_savart_general, 
                                  FourierCurrentPotentialField, ToroidalMagneticField,
                                  field_line_integrate)

import desc.examples

from desc.backend import fori_loop, jit, jnp, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.utils import flatten_list
from desc.compute.utils import cross
from desc.compute.utils import dot

from desc.optimize import lsqtr, lsq_auglag


import numpy as np

from scipy.linalg import null_space
from numpy import ndarray

# Import my own functions
from desc.fns_simp import (data_eval,)

#from desc.phi2d_eval import phi_sv_eval

# Find normal component from solution to confirm it is actually zero
def phi_sv_eval(grid, field, basis):
    
    transform = Transform(grid, basis, derivs = 2)
    phi_trans = desc.transform.Transform(grid, basis, derivs=2, 
                                         rcond='auto', build=True, build_pinv=False, method='auto')
    
    
    phi_mn = field.Phi_mn

    fs = {#"phi": transform.transform(phi_mn),
          #"phi_t": transform.transform(phi_mn, dt = 1),
          #"phi_z": transform.transform(phi_mn, dz = 1),
          "phi_tt": transform.transform(phi_mn, dt = 2),
          "phi_tz": transform.transform(phi_mn, dt = 1, dz = 1),
          "phi_zz": transform.transform(phi_mn, dz = 2),
            }

    #fs["phi"] = phi_trans.transform(phi_mn, dr=0, dt=0, dz=0)
    
    #fs["phi_t"] = phi_trans.transform(phi_mn, dr=0, dt=1, dz=0)
    #fs["phi_z"] = phi_trans.transform(phi_mn, dr=0, dt=0, dz=1)
    
    fs["phi_tt"] = phi_trans.transform(phi_mn, dr=0, dt=2, dz=0)
    fs["phi_tz"] = phi_trans.transform(phi_mn, dr=0, dt=1, dz=1)
    fs["phi_zz"] = phi_trans.transform(phi_mn, dr=0, dt=0, dz=2)
                                   
    return fs

def contour_calc_c(phi_sv_basis,
                   kgrid,
                   winding_surf, 
                   surface_current_field,
                   K_0):

    #kdata = eq.compute(["e_zeta","g_zz"], grid = kgrid)
    kdata = data_eval(kgrid, winding_surf)
    
    phi_t = (K_0)**(-1)*surface_current_field.compute(["Phi_t"], grid=kgrid)["Phi_t"]
    phi_z = (K_0)**(-1)*surface_current_field.compute(["Phi_z"], grid=kgrid)["Phi_z"]

    nabla_s_psi = (phi_t*kdata["e^theta_gamma"].T
                   + phi_z*kdata["e^zeta_gamma"].T).T

    g = dot(kdata["n_rho"], cross(kdata["e^theta_gamma"],kdata["e^zeta_gamma"]))

    g_t = (dot(kdata["n_t"],cross(kdata["e^theta_gamma"],kdata["e^zeta_gamma"])) 
           + dot(kdata["n_rho"],(cross(kdata["e^theta_gamma_t"],kdata["e^zeta_gamma"]) 
                                 + cross(kdata["e^theta_gamma"],kdata["e^zeta_gamma_t"]))))

    g_z = (dot(kdata["n_z"],cross(kdata["e^theta_gamma"],kdata["e^zeta_gamma"])) 
           + dot(kdata["n_rho"],(cross(kdata["e^theta_gamma_z"],kdata["e^zeta_gamma"]) 
                                 + cross(kdata["e^theta_gamma"],kdata["e^zeta_gamma_z"]))))

    x = dot(nabla_s_psi,kdata["e^theta_gamma"])*g**(-1)

    d2_phi = phi_sv_eval(kgrid, surface_current_field,phi_sv_basis)
    phi_tt = (K_0)**(-1)*d2_phi["phi_tt"]
    phi_tz = (K_0)**(-1)*d2_phi["phi_tz"]
    phi_zz = (K_0)**(-1)*d2_phi["phi_zz"]

    # Surface Laplacian of theta
    laplace_phi = (dot(kdata["e^theta_gamma"],kdata["e^theta_gamma_t"])*phi_t
                   + dot(kdata["e^theta_gamma"],kdata["e^theta_gamma"])*phi_tt
                   + dot(kdata["e^theta_gamma"],kdata["e^zeta_gamma_t"])*phi_z
                   + dot(kdata["e^theta_gamma"],kdata["e^zeta_gamma"])*phi_tz
                   + dot(kdata["e^zeta_gamma"],kdata["e^theta_gamma_z"])*phi_t
                   + dot(kdata["e^theta_gamma"],kdata["e^zeta_gamma"])*phi_tz
                   + dot(kdata["e^zeta_gamma"],kdata["e^zeta_gamma_z"])*phi_z
                   + dot(kdata["e^zeta_gamma"],kdata["e^zeta_gamma"])*phi_zz
                                      )

    x_t = ((phi_tt*dot(kdata["e^theta_gamma"],kdata["e^theta_gamma"]) 
           + phi_t*2*dot(kdata["e^theta_gamma"],kdata["e^theta_gamma_t"]) 
           + phi_tz*dot(kdata["e^theta_gamma"],kdata["e^zeta_gamma"]) 
           + phi_z*(dot(kdata["e^theta_gamma_t"],kdata["e^zeta_gamma"]) 
                   + dot(kdata["e^theta_gamma"],kdata["e^zeta_gamma_t"]) 
                   ) 
          )*g**(-1) 
           - dot(nabla_s_psi,kdata["e^theta_gamma"])*g**(-2)*g_t
          )
    
    x_z = ((phi_tz*dot(kdata["e^theta_gamma"],kdata["e^theta_gamma"]) 
           + phi_t*2*dot(kdata["e^theta_gamma"],kdata["e^theta_gamma_z"]) 
           + phi_zz*dot(kdata["e^zeta_gamma"],kdata["e^theta_gamma"]) 
           + phi_z*(dot(kdata["e^theta_gamma_z"],kdata["e^zeta_gamma"]) 
                   + dot(kdata["e^theta_gamma"],kdata["e^zeta_gamma_z"]) 
                   ) 
          )*g**(-1) 
           - dot(nabla_s_psi,kdata["e^theta_gamma"])*g**(-2)*g_z
          )
    

    y = -dot(nabla_s_psi,kdata["e^zeta_gamma"])*g**(-1)


    y_t = -((phi_tt*dot(kdata["e^theta_gamma"],kdata["e^zeta_gamma"]) 
             + phi_t*(dot(kdata["e^theta_gamma_t"],kdata["e^zeta_gamma"]) 
                      + dot(kdata["e^theta_gamma"],kdata["e^zeta_gamma_t"]) 
                     )
             + phi_zz*dot(kdata["e^zeta_gamma"],kdata["e^zeta_gamma"]) 
             + phi_z*2*dot(kdata["e^zeta_gamma"],kdata["e^zeta_gamma_t"])  
            )*g**(-1) 
            - dot(nabla_s_psi,kdata["e^zeta_gamma"])*g**(-2)*g_t
           )
    
    y_z = -((phi_tz*dot(kdata["e^theta_gamma"],kdata["e^zeta_gamma"]) 
             + phi_t*(dot(kdata["e^theta_gamma_z"],kdata["e^zeta_gamma"]) 
                      + dot(kdata["e^theta_gamma"],kdata["e^zeta_gamma_z"]) 
                     )
             + phi_zz*dot(kdata["e^zeta_gamma"],kdata["e^zeta_gamma"]) 
             + phi_z*2*dot(kdata["e^zeta_gamma"],kdata["e^zeta_gamma_z"])  
            )*g**(-1) 
            - dot(nabla_s_psi,kdata["e^zeta_gamma"])*g**(-2)*g_z
           )

    
    z = x_t - y_z
    
    return x,y,z
#(x,y,z,x_t,x_z,y_t,y_z,g)