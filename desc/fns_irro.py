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

from desc.magnetic_fields import ( SplineMagneticField, biot_savart_general, 
                                  FourierCurrentPotentialField, ToroidalMagneticField,
                                  field_line_integrate)

import desc.examples

#from desc.geometry.utils import rpz2xyz, rpz2xyz_vec
#from desc.compute import rpz2xyz, rpz2xyz_vec
from desc.backend import fori_loop, jit, jnp, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.coils import *

from desc.utils import flatten_list
from desc.compute.utils import cross
from desc.compute.utils import dot

#from desc.regcoil import run_regcoil
from desc.optimize import lsqtr, lsq_auglag

from scipy.optimize import NonlinearConstraint 

from desc.magnetic_fields import FourierCurrentPotentialField

import time

import numpy as np
from numpy.linalg import eig

from scipy.linalg import null_space
from numpy import ndarray

def eqn_residual(data, 
                 b_basis,
                 transform1,
                 y,
                 nabla_gamma_psi
                ):

    
    b_mn = y[0:b_basis.num_modes]
    
    # Compute V, rho and derivatives on the grid
    fs = {# a
        "b": transform1.transform(b_mn),
        # First derivatives of rho
        "b_t": transform1.transform(b_mn, dt = 1),
        "b_z": transform1.transform(b_mn, dz = 1),
         }
    
    # Define nabla_gamma_b
    nabla_gamma_b = ((y[b_basis.num_modes]*data["theta"]**(0) + fs["b_t"])*data["e^theta_gamma"].T 
                         + (y[b_basis.num_modes + 1]*data["zeta"]**(0) + fs["b_z"])*data["e^zeta_gamma"].T).T
    
    return dot(nabla_gamma_b,nabla_gamma_psi)

# Find normal component from solution to confirm it is actually zero
#def phi_sv_eval(grid, field, basis):
    
    #phi_modes = psi_basis.num_modes
#    transform = Transform(grid, basis, derivs = 2)
#    phi_trans = desc.transform.Transform(grid, basis, derivs=2, 
#                                         rcond='auto', build=True, build_pinv=False, method='auto')
    
#    field_SV = field.copy()
#    field_SV.I = np.array(0.0)
#    field_SV.G = np.array(0.0)
    
#    phi_mn = field_SV.Phi_mn

#    fs = {#"phi": transform.transform(phi_mn),
#          "phi_t": transform.transform(phi_mn, dt = 1),
#          "phi_z": transform.transform(phi_mn, dz = 1),
#          "phi_tt": transform.transform(phi_mn, dt = 2),
#          "phi_tz": transform.transform(phi_mn, dt = 1, dz = 1),
#          "phi_zz": transform.transform(phi_mn, dz = 2),
#            }

    #fs["phi"] = phi_trans.transform(phi_mn, dr=0, dt=0, dz=0)
    
#    fs["phi_t"] = phi_trans.transform(phi_mn, dr=0, dt=1, dz=0)
#    fs["phi_z"] = phi_trans.transform(phi_mn, dr=0, dt=0, dz=1)
    
#    fs["phi_tt"] = phi_trans.transform(phi_mn, dr=0, dt=2, dz=0)
#    fs["phi_tz"] = phi_trans.transform(phi_mn, dr=0, dt=1, dz=1)
#    fs["phi_zz"] = phi_trans.transform(phi_mn, dr=0, dt=0, dz=2)
                                   
#    return fs

######### define a function to evaluate on different grids
def K_eval(data,current_field, grid,phi_sv_basis):

    def phi_sv_eval(grid, field, basis):
    
        #phi_modes = psi_basis.num_modes
        transform = Transform(grid, basis, derivs = 2)
        phi_trans = desc.transform.Transform(grid, basis, derivs=2, 
                                             rcond='auto', build=True, build_pinv=False, method='auto')

        field_SV = field.copy()
        field_SV.I = np.array(0.0)
        field_SV.G = np.array(0.0)

        phi_mn = field_SV.Phi_mn

        fs = {#"phi": transform.transform(phi_mn),
              "phi_t": transform.transform(phi_mn, dt = 1),
              "phi_z": transform.transform(phi_mn, dz = 1),
              "phi_tt": transform.transform(phi_mn, dt = 2),
              "phi_tz": transform.transform(phi_mn, dt = 1, dz = 1),
              "phi_zz": transform.transform(phi_mn, dz = 2),
                }

        #fs["phi"] = phi_trans.transform(phi_mn, dr=0, dt=0, dz=0)

        fs["phi_t"] = phi_trans.transform(phi_mn, dr=0, dt=1, dz=0)
        fs["phi_z"] = phi_trans.transform(phi_mn, dr=0, dt=0, dz=1)

        fs["phi_tt"] = phi_trans.transform(phi_mn, dr=0, dt=2, dz=0)
        fs["phi_tz"] = phi_trans.transform(phi_mn, dr=0, dt=1, dz=1)
        fs["phi_zz"] = phi_trans.transform(phi_mn, dr=0, dt=0, dz=2)

        return fs

    Ks = current_field.compute("K",grid = grid)["K"]

    K_sec = cross(data["n_rho"],(current_field.I/(2*np.pi)*data["e^theta"]
                                 + current_field.G/(2*np.pi)*data["e^zeta"] )
                 )

    K_sv = Ks - K_sec

    # Find phi scalar
    vphi = current_field.compute("Phi", grid = grid)["Phi"]

    Ksec_norm = max(jnp.sqrt(dot(K_sec,K_sec)))

    # Find a length scale
    Ks_norm = max(jnp.sqrt(dot(Ks,Ks)))

    psi0 = Ks_norm

    Ks_mod = Ks/Ks_norm # Normalized surface-current density

    K_sec = K_sec/Ks_norm
    K_sv = K_sv/Ks_norm

    nabla_gamma_psi = -cross(data["n_rho"],Ks_mod)

    nabla_gamma_phi_sv = -cross(data["n_rho"],K_sv)

    nabla_gamma_phi_sec = nabla_gamma_psi - nabla_gamma_phi_sv
    
    ts = phi_sv_eval(grid, current_field, phi_sv_basis)
    
    # Surface Laplacian of phi_sec
    laplace_phi_sec = ( (current_field.I/(2*np.pi*psi0))*data["nabla_gamma^2_theta"] 
                       + (current_field.G/(2*np.pi*psi0))*data["nabla_gamma^2_zeta"] 
                      )/Ks_norm
    
    # Surface Laplacian of phi_sv
    laplace_phi_sv = (dot(data["e^theta_gamma"],data["e^theta_gamma_t"])*ts["phi_t"] 
                      + dot(data["e^theta_gamma"],data["e^theta_gamma"])*ts["phi_tt"]
                      + dot(data["e^theta_gamma"],data["e^zeta_gamma_t"])*ts["phi_z"]
                      + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*ts["phi_tz"]
                      + dot(data["e^zeta_gamma"],data["e^theta_gamma_z"])*ts["phi_t"]
                      + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*ts["phi_tz"]
                      + dot(data["e^zeta_gamma"],data["e^zeta_gamma_z"])*ts["phi_z"]
                      + dot(data["e^zeta_gamma"],data["e^zeta_gamma"])*ts["phi_zz"]
                     )/Ks_norm
    
    return (K_sec, psi0, Ks_norm,
            nabla_gamma_phi_sec, nabla_gamma_phi_sv, Ks_mod, 
            nabla_gamma_psi, 
            laplace_phi_sec, laplace_phi_sv)

def sol_eval(grid, 
             data, 
             b_basis,           
             x
            ):

    b_mn = x[0:b_basis.num_modes]
    
    trans = desc.transform.Transform(grid, b_basis, derivs=0, 
                                     rcond='auto', build=True, build_pinv=False, method='auto')
    
    # Compute V, rho and derivatives on the grid
    fs = {
        "b": desc.transform.Transform(grid = grid, basis = b_basis),
    }
           
    ##
    fs["b"] = trans.transform(b_mn, dt=0, dz=0)
    
    rho = np.exp(x[b_basis.num_modes]*data["theta"] +
                 x[b_basis.num_modes + 1]*data["zeta"] +
                 fs["b"])
    
    return rho