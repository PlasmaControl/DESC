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

# Define a fuction to evaluate vectors on a given surface with a point grid
def data_eval(sgrid,surface):

    data = surface.compute(["n_rho",
                            "X","Y","Z",
                            "e^rho","e^theta","e^zeta",
                            "e^theta_r","e^theta_t","e^theta_z",
                            "e^zeta_r","e^zeta_t","e^zeta_z",
                            "e^theta_rr","e^theta_rt","e^theta_rz",
                            "e^theta_tt","e^theta_tz",
                            "e^theta_zz",
                            "e^zeta_rr","e^zeta_rt","e^zeta_rz",
                            "e^zeta_tt","e^zeta_tz",
                            "e^zeta_zz",
                            "e_rho","e_theta","e_zeta","n_rho",
                            "e_rho_r","e_rho_t","e_rho_z",
                            "e_theta_r","e_theta_t","e_theta_z",
                            "e_zeta_r","e_zeta_t","e_zeta_z",
                            "theta","zeta",
                            "theta_t","theta_z",
                            "zeta_t","zeta_z",
                            #"theta_tt",
                            ],
                           grid = sgrid)

    # Find derivatives of the norm of the normal unit vector
    data["|e_theta x e_zeta|_z"] = dot(
                                        (
                                        cross(data["e_theta_z"], data["e_zeta"])
                                        + cross(data["e_theta"], data["e_zeta_z"])
                                        ),
                                        cross(data["e_theta"], data["e_zeta"]),
                                        ) / (data["|e_theta x e_zeta|"])

    data["|e_theta x e_zeta|_t"] = dot(
                                        (
                                        cross(data["e_theta_t"], data["e_zeta"])
                                        + cross(data["e_theta"], data["e_zeta_t"])
                                        ),
                                        cross(data["e_theta"], data["e_zeta"]),
                                        ) / (data["|e_theta x e_zeta|"])

    # Find the first derivatives of the normal vector
    data["n_t"] = ( cross(data["e_theta_t"], data["e_zeta"]) + cross(data["e_theta"], data["e_zeta_t"])
                            )/ data["|e_theta x e_zeta|"][:, None] - data["n_rho"] / data[
                            "|e_theta x e_zeta|"][
                            :, None] * data[
                            "|e_theta x e_zeta|_t"][
                            :, None
                            ]

    data["n_z"] = ( cross(data["e_theta_z"], data["e_zeta"]) + cross(data["e_theta"], data["e_zeta_z"])
                            )/ data["|e_theta x e_zeta|"][:, None] - data["n_rho"] / data[
                            "|e_theta x e_zeta|"][
                            :, None] * data[
                            "|e_theta x e_zeta|_z"][
                            :, None
                            ]
    
    # Find the directional tangent vectors
    # e^theta_gamma
    data["e^theta_gamma"] = data["e^theta"] - (dot(data["n_rho"],data["e^theta"])*data["n_rho"].T).T
    # e^zeta_gamma
    data["e^zeta_gamma"] = data["e^zeta"] - (dot(data["n_rho"],data["e^zeta"])*data["n_rho"].T).T
    
    # t,z- derivatives of e^theta_gamma
    
    # d(e^theta_gamma)/dtheta
    data["e^theta_gamma_t"] = ( data["e^theta_t"] 
                               - ( (dot(data["n_t"],data["e^theta"]) 
                                    + dot(data["n_rho"],data["e^theta_t"]) )*data["n_rho"].T
                                 ).T
                               - ( dot(data["n_rho"],data["e^theta"])*data["n_t"].T ).T
                              )
    
    # d(e^theta_gamma)/dzeta
    data["e^theta_gamma_z"] = ( data["e^theta_z"]
                               - ( (dot(data["n_z"],data["e^theta"]) 
                                    + dot(data["n_rho"],data["e^theta_z"]) )*data["n_rho"].T
                                 ).T
                               - ( dot(data["n_rho"],data["e^theta"])*data["n_z"].T ).T
                              )
    
    # t,z- derivatives of e^zeta_gamma
    
    # d(e^zeta_gamma)/dtheta
    data["e^zeta_gamma_t"] = ( data["e^zeta_t"] 
                              - ( (dot(data["n_t"],data["e^zeta"]) 
                                   + dot(data["n_rho"],data["e^zeta_t"]) )*data["n_rho"].T
                                ).T
                              - ( dot(data["n_rho"],data["e^zeta"])*data["n_t"].T).T
                             )
    # d(e^zeta_gamma)/dzeta
    data["e^zeta_gamma_z"] = ( data["e^zeta_z"] 
                              - ( ( dot(data["n_z"],data["e^zeta"]) 
                                   + dot(data["n_rho"],data["e^zeta_z"]) )*data["n_rho"].T).T
                              - ( dot(data["n_rho"],data["e^zeta"] )*data["n_z"].T).T
                             )
    
        
    # Surface Laplacian of theta
    data["nabla_gamma^2_theta"] = (dot(data["e^theta_gamma"],data["e^theta_gamma_t"])*data["theta_t"] 
                                   #+ dot(data["e^theta_gamma"],data["e^theta_gamma"])*data["theta_tt"]
                                   + dot(data["e^theta_gamma"],data["e^zeta_gamma_t"])*data["theta_z"]
                                   #+ dot(data["e^theta_gamma"],data["e^zeta_gamma"])*data["theta_tz"]
                                   + dot(data["e^zeta_gamma"],data["e^theta_gamma_z"])*data["theta_t"]
                                   #+ dot(data["e^theta_gamma"],data["e^zeta_gamma"])*data["theta_tz"]
                                   + dot(data["e^zeta_gamma"],data["e^zeta_gamma_z"])*data["theta_z"]
                                   #+ dot(data["e^zeta_gamma"],data["e^zeta_gamma"])*data["theta_zz"]
                                  )
    
    # Surface Laplacian of zeta
    data["nabla_gamma^2_zeta"] = (dot(data["e^theta_gamma"],data["e^theta_gamma_t"])*data["zeta_t"] 
                                  #+ dot(data["e^theta_gamma"],data["e^theta_gamma"])*data["zeta_tt"]
                                  + dot(data["e^theta_gamma"],data["e^zeta_gamma_t"])*data["zeta_z"]
                                  #+ dot(data["e^theta_gamma"],data["e^zeta_gamma"])*data["zeta_tz"]
                                  + dot(data["e^zeta_gamma"],data["e^theta_gamma_z"])*data["zeta_t"]
                                  #+ dot(data["e^theta_gamma"],data["e^zeta_gamma"])*data["zeta_tz"]
                                  + dot(data["e^zeta_gamma"],data["e^zeta_gamma_z"])*data["zeta_z"]
                                  #+ dot(data["e^zeta_gamma"],data["e^zeta_gamma"])*data["zeta_zz"]
                                 )
    
    return data

######### define a function to evaluate on different grids
def K_eval(data, 
           grid,
           I,G,
          ):
    
    # Switched the convention for I and G since we mainly need a toroidal flux, which conventionally is generated by G
    K_sec = (I/(2*np.pi)*data["e^zeta_gamma"] 
             + G/(2*np.pi)*data["e^theta_gamma"]
            )

    Ksec_norm = max(jnp.sqrt(dot(K_sec,K_sec)))

    K_sec = K_sec/Ksec_norm
    
    # sigma_sec = 1e7 # 10 MS/m, this is a reference conductivity, 
    # could be changed to adjust the values of V_sec and nabla_gamma_V_sec

    sigma0 = G*(2*np.pi)**(-1) # Reference conductivity abs(G) (?)
    
    sigma_sec = sigma0*jnp.ones(data["theta"].shape)
    nabla_gamma_sigma_sec = 0*data["e^theta_gamma"]
    
    V_sec = data["theta"]#-(G/(2*np.pi)*sigma_sec**(-1))*int_kappa
    
    nabla_gamma_V_sec = data["e^theta_gamma"]
    
    nabla_gamma2_V_sec = data["nabla_gamma^2_theta"]
    
    nabla_gamma_dot_K_sec = dot(nabla_gamma_sigma_sec,nabla_gamma_V_sec) + sigma_sec*nabla_gamma2_V_sec
    
    return (K_sec, Ksec_norm,
            sigma_sec, 
            nabla_gamma_sigma_sec,
            V_sec,
            nabla_gamma_V_sec,nabla_gamma2_V_sec,
            nabla_gamma_dot_K_sec
           )

def _compute_magnetic_field_from_Current(Kgrid,
                                         K_at_grid, 
                                         surface, 
                                         eq,
                                         Bgrid,
                                         basis="rpz"):
    """Compute magnetic field at a set of points.

    Parameters
    ----------
    K_at_grid : ndarray, shape (num_nodes,3)
        Surface current evaluated at points on a grid, which you want to calculate
        B from, should be in cartesian ("xyz") or cylindrical ("rpz") specifiec
        by "basis" argument
    surface : FourierRZToroidalSurface
        surface object upon which the surface current K_at_grid lies
    coords : array-like shape(N,3) or Grid
        cylindrical or cartesian coordinates to evlauate B at
    grid : Grid,
        source grid upon which to evaluate the surface current density K
    basis : {"rpz", "xyz"}
        basis for input coordinates and returned magnetic field

    Returns
    -------
    field : ndarray, shape(N,3)
        magnetic field at specified points

    """

    Bdata = eq.compute(["R","phi","Z","n_rho"], grid = Bgrid)
    coords = np.vstack([Bdata["R"],Bdata["phi"],Bdata["Z"]]).T
    
    assert basis.lower() in ["rpz", "xyz"]
    if hasattr(coords, "nodes"):
        coords = coords.nodes
    coords = jnp.atleast_2d(coords)
    if basis == "rpz":
        coords = rpz2xyz(coords)
    else:
        K_at_grid = xyz2rpz_vec(K_at_grid, x=coords[:, 0], y=coords[:, 1])
    surface_grid = Kgrid

    # compute and store grid quantities
    # needed for integration
    # TODO: does this have to be xyz, or can it be computed in rpz as well?
    data = surface.compute(["x", "|e_theta x e_zeta|"], grid=surface_grid, basis="xyz")

    _rs = xyz2rpz(data["x"])
    _K = K_at_grid

    # surface element, must divide by NFP to remove the NFP multiple on the
    # surface grid weights, as we account for that when doing the for loop
    # over NFP
    _dV = surface_grid.weights * data["|e_theta x e_zeta|"] / surface_grid.NFP

    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi = (surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP) % (
            2 * jnp.pi
        )
        # new coords are just old R,Z at a new phi (bc of discrete NFP symmetry)
        rs = jnp.vstack((_rs[:, 0], phi, _rs[:, 2])).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_general(
            coords,
            rs,
            K,
            _dV,
        )
        f += fj
        return f
    
    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        
    return B

def plot_figure(var,   # variable to plot
                egrid, # grid to plot on
                title, # title for the figure
               ):
    
    plt.figure(figsize=(6,5)).set_tight_layout(False)
    plt.contourf(egrid.nodes[egrid.unique_zeta_idx,2],
                 egrid.nodes[egrid.unique_theta_idx,1],
                 (var).reshape(egrid.num_theta,
                                  egrid.num_zeta,order="F"))
    plt.ylabel(''r'$\theta$',fontsize = 20)
    plt.xlabel(''r'$\zeta$',fontsize = 20)
    plt.title(title,fontsize = 20)
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
              
    return None

# Define function to minimize Bn
def eqn_residual(s_basis,
                 V_basis,
                 y,
                 grid,
                 data,
                 surf_winding,
                 s_sec,
                 nabla_gamma_s_sec,
                 nabla_gamma_V_sec,
                 nabla_gamma2_V_sec,
                 nabla_gamma_dot_K_sec,
                 K_sec,
                 eq,
                 Bgrid,
                 Bdata,
                 Bn_sec,
                 alpha_div,
                 alpha_Bn,
                 alpha_k,
                ):
  
    transform1 = Transform(grid, s_basis, derivs = 2, build=True)
    
    s_mn = y[0:s_basis.num_modes]
    V_mn = y[s_basis.num_modes: s_basis.num_modes + V_basis.num_modes]
    
    # Compute V, rho and derivatives on the grid
    fs = {# sigma_sv
        "s": transform1.transform(s_mn),
        # First derivatives of V
        "s_t": transform1.transform(s_mn, dt = 1),
        "s_z": transform1.transform(s_mn, dz = 1),
        # V
        "V": transform1.transform(V_mn),
        # First derivatives of V
        "V_t": transform1.transform(V_mn, dt = 1),
        "V_z": transform1.transform(V_mn, dz = 1),
        # Second derivatives of V
        "V_tt": transform1.transform(V_mn, dt = 2),
        "V_tz": transform1.transform(V_mn, dt = 1,dz = 1),
        "V_zz": transform1.transform(V_mn, dz = 2),
         }
    
    # Define sigma
    s = (0
         #y[s_basis.num_modes + V_basis.num_modes]*data["theta"]
         #+ y[s_basis.num_modes + V_basis.num_modes + 1]*data["zeta"]
         + fs["s"])
    
    # Define nabla_gamma_V
    nabla_gamma_V = ((0
                      #y[s_basis.num_modes + V_basis.num_modes + 2]*data["theta"]**(0) 
                      + fs["V_t"]
                     )*data["e^theta_gamma"].T 
                     + (0
                        #y[s_basis.num_modes + V_basis.num_modes + 3]*data["zeta"]**(0) 
                        + fs["V_z"]
                       )*data["e^zeta_gamma"].T
                    ).T
    
    K_sv = ((s_sec + s)*nabla_gamma_V.T 
            + s*nabla_gamma_V_sec.T).T
    
    B_sv = _compute_magnetic_field_from_Current(grid,
                                                K_sv, 
                                                surf_winding,
                                                eq,
                                                Bgrid,
                                                basis="rpz")
    
    Bn_error = alpha_Bn*(dot(Bdata["n_rho"],B_sv) + Bn_sec)**(2)
    
    # Divergence of total current
    
    # Define nabla_gamma_sigma_sv
    nabla_gamma_s = ((0
                      #y[s_basis.num_modes + V_basis.num_modes]*data["theta"]**(0) 
                      + fs["s_t"]
                     )*data["e^theta_gamma"].T 
                     + (0
                        #y[s_basis.num_modes + V_basis.num_modes + 1]*data["zeta"]**(0) 
                        + fs["s_z"]
                       )*data["e^zeta_gamma"].T
                    ).T
    
    # Surface Laplacian of V_sv
    nabla_gamma2_V = (0 
                      #y[s_basis.num_modes + V_basis.num_modes+2]*data["nabla_gamma^2_theta"]
                      #+ y[s_basis.num_modes + V_basis.num_modes+3]*data["nabla_gamma^2_theta"]
                      + (dot(data["e^theta_gamma"],data["e^theta_gamma_t"])*fs["V_t"] 
                         + dot(data["e^theta_gamma"],data["e^theta_gamma"])*fs["V_tt"]
                         + dot(data["e^theta_gamma"],data["e^zeta_gamma_t"])*fs["V_z"]
                         + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*fs["V_tz"]
                         + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*fs["V_tz"]
                         + dot(data["e^zeta_gamma"],data["e^zeta_gamma_z"])*fs["V_z"]
                         + dot(data["e^zeta_gamma"],data["e^zeta_gamma"])*fs["V_zz"]
                        )
                     )
    
    nabla_gamma_dot_K = alpha_div*(dot(nabla_gamma_s_sec,nabla_gamma_V)
                                   + s_sec*nabla_gamma2_V 
                                   + dot(nabla_gamma_s,nabla_gamma_V_sec)
                                   + s*nabla_gamma2_V_sec
                                   + dot(nabla_gamma_s,nabla_gamma_V) 
                                   + s*nabla_gamma2_V
                                   + nabla_gamma_dot_K_sec # Divergence of secular current
                                  )**(2)

    K_total = K_sec + K_sv
    mag_K = alpha_k*dot(K_total,K_total)
    
    cost = jnp.concatenate([Bn_error,nabla_gamma_dot_K,mag_K])
    
    return cost

# Define a function to evaluate the solution
def sol_eval(y,
             s_basis,
             V_basis,
             grid,
             data,
             surf_winding,
             s_sec,
             nabla_gamma_s_sec,
             nabla_gamma_V_sec,
             nabla_gamma2_V_sec,
             nabla_gamma_dot_K_sec,
             eq,
             Bgrid,
             Bdata,
             B_sec,
            ):

    trans = desc.transform.Transform(grid, s_basis, derivs=2, 
                                     rcond='auto', build=True, build_pinv=False, method='auto')
    
    s_mn = y[0:s_basis.num_modes]
    V_mn = y[s_basis.num_modes: s_basis.num_modes + V_basis.num_modes]
    
    # Compute V, rho and derivatives on the grid
    fs = {# sigma_sv
        "s": trans.transform(s_mn),
        # First derivatives of V
        "s_t": trans.transform(s_mn, dt = 1),
        "s_z": trans.transform(s_mn, dz = 1),
        # V
        "V": trans.transform(V_mn),
        # First derivatives of V
        "V_t": trans.transform(V_mn, dt = 1),
        "V_z": trans.transform(V_mn, dz = 1),
        # Second derivatives of V
        "V_tt": trans.transform(V_mn, dt = 2),
        "V_tz": trans.transform(V_mn, dt = 1,dz = 1),
        "V_zz": trans.transform(V_mn, dz = 2),
         }
    
    # Define sigma
    s = (0
         #y[s_basis.num_modes + V_basis.num_modes]*data["theta"]
         #+ y[s_basis.num_modes + V_basis.num_modes + 1]*data["zeta"]
         + fs["s"])
    
    # Define V
    V = (0
         #y[s_basis.num_modes + V_basis.num_modes + 2]*data["theta"]
         #+ y[s_basis.num_modes + V_basis.num_modes + 3]*data["zeta"]
         + fs["V"])
    
    # Define nabla_gamma_V
    nabla_gamma_V = ((0
                      #y[s_basis.num_modes + V_basis.num_modes + 2]*data["theta"]**(0) 
                      + fs["V_t"]
                     )*data["e^theta_gamma"].T 
                     + (0
                        #y[s_basis.num_modes + V_basis.num_modes + 3]*data["zeta"]**(0) 
                        + fs["V_z"]
                       )*data["e^zeta_gamma"].T
                    ).T
    
    K_sv = ((s_sec + s)*nabla_gamma_V.T 
            + s*nabla_gamma_V_sec.T).T
    
    B_sv = _compute_magnetic_field_from_Current(grid,
                                                K_sv, 
                                                surf_winding,
                                                eq,
                                                Bgrid,
                                                basis="rpz")
    
    Btotal = B_sv + B_sec
    Bn_error = dot(Bdata["n_rho"],Btotal)
    
    # Divergence of total current
    
    # Define nabla_gamma_sigma_sv
    nabla_gamma_s = ((0
                      #y[s_basis.num_modes + V_basis.num_modes]*data["theta"]**(0) 
                      + fs["s_t"]
                     )*data["e^theta_gamma"].T 
                     + (0
                        #y[s_basis.num_modes + V_basis.num_modes + 1]*data["zeta"]**(0) 
                        + fs["s_z"]
                       )*data["e^zeta_gamma"].T
                    ).T
    
    # Surface Laplacian of V_sv
    nabla_gamma2_V = (0 
                      #y[s_basis.num_modes + V_basis.num_modes+2]*data["nabla_gamma^2_theta"]
                      #+ y[s_basis.num_modes + V_basis.num_modes+3]*data["nabla_gamma^2_theta"]
                      + (dot(data["e^theta_gamma"],data["e^theta_gamma_t"])*fs["V_t"] 
                         + dot(data["e^theta_gamma"],data["e^theta_gamma"])*fs["V_tt"]
                         + dot(data["e^theta_gamma"],data["e^zeta_gamma_t"])*fs["V_z"]
                         + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*fs["V_tz"]
                         + dot(data["e^theta_gamma"],data["e^zeta_gamma"])*fs["V_tz"]
                         + dot(data["e^zeta_gamma"],data["e^zeta_gamma_z"])*fs["V_z"]
                         + dot(data["e^zeta_gamma"],data["e^zeta_gamma"])*fs["V_zz"]
                        )
                     )
    
    nabla_gamma_dot_K = (dot(nabla_gamma_s_sec,nabla_gamma_V) 
                         + s_sec*nabla_gamma2_V 
                         + dot(nabla_gamma_s,nabla_gamma_V_sec)
                         + s*nabla_gamma2_V_sec
                         + dot(nabla_gamma_s,nabla_gamma_V) 
                         + s*nabla_gamma2_V
                         + nabla_gamma_dot_K_sec # Divergence of secular current
                        )
    
    return Bn_error,Btotal,nabla_gamma_dot_K, s, V, nabla_gamma_V