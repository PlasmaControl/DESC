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

from desc.magnetic_fields._core import (biot_savart_general, 
                                        biot_savart_general_reg, 
                                        #biot_savart_potential, 
                                        #biot_savart_t, surf_div_general,
                                        #vector_potential_general,
                                       #vector_potential_potential
                                       )

from desc.magnetic_fields import ( SplineMagneticField,
                                  FourierCurrentPotentialField, ToroidalMagneticField,
                                  field_line_integrate)

import desc.examples

from desc.backend import fori_loop, jit, jnp, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.coils import *

from desc.utils import flatten_list
from desc.utils import cross
from desc.utils import dot

from desc.magnetic_fields import FourierCurrentPotentialField

import numpy as np

from numpy import ndarray

# Define function to find A and its respective derivatives
def u_div_residual(grid, data, x,):

    f_t = first_derivative_t(x, data, grid,)
    
    f_z = first_derivative_z(x, data, grid,)
    
    f_tt = first_derivative_t(f_t, data, grid,)
    
    f_zz = first_derivative_z(f_z, data, grid,)
    
    f_tz = first_derivative_z(f_t, data, grid,)
    
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
    
    return nabla_gamma_2_f


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
    coords = jnp.vstack([Bdata["R"],Bdata["phi"],Bdata["Z"]]).T
    
    assert basis.lower() in ["rpz", "xyz"]
    if hasattr(coords, "nodes"):
        coords = coords.nodes
    coords = jnp.atleast_2d(coords)
    if basis == "rpz":
        coords = rpz2xyz(coords)
    else:
        K_at_grid = xyz2rpz_vec(K_at_grid, x=coords[:, 0], y=coords[:, 1],)
    
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
        fj = biot_savart_general( coords, rs, K, _dV,
        )
        f += fj
        return f
    
    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        
    return B

def _compute_magnetic_field_from_Current_reg(Kgrid,
                                         K_at_grid, 
                                         surface, 
                                         eq,
                                         Bgrid,
                                         basis="rpz",
                                            patch = 1e-3):
    
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
    coords = jnp.vstack([Bdata["R"],Bdata["phi"],Bdata["Z"]]).T
    
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
        fj = biot_savart_general_reg( coords, rs, K, _dV, patch = patch,
        )
        f += fj
        return f
    
    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        
    return B

def _compute_magnetic_field_from_Current_Contour(Kgrid,
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
    coords = jnp.vstack([Bdata["R"],Bdata["phi"],Bdata["Z"]]).T
    
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
    data = surface.compute(["x", 
                            'e_theta',
                            #"|e_theta x e_zeta|",
                           ], grid=surface_grid, basis="xyz")

    _rs = xyz2rpz(data["x"])
    _K = K_at_grid

    # surface element, must divide by NFP to remove the NFP multiple on the
    # surface grid weights, as we account for that when doing the for loop
    # over NFP
    #_dV = surface_grid.weights * data["|e_theta x e_zeta|"] / surface_grid.NFP
    #_dV = surface_grid.weights * 1 / surface_grid.NFP
    _dV = surface_grid.weights * dot(data['e_theta'],data['e_theta'])**(1/2) / surface_grid.NFP
    
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi = (surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP) % (2 * jnp.pi)
        
        # new coords are just old R,Z at a new phi (bc of discrete NFP symmetry)
        rs = jnp.vstack((_rs[:, 0], phi, _rs[:, 2])).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_general( coords, rs, K, _dV,
        )
        f += fj
        return f
    
    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        
    return B

def plot_xy(var, 
            grid, # grid to plot on
            title, # title for the figure
            x_axis, # title of x axis
           ):
    
    plt.scatter(grid, var)
    
    plt.xlabel(x_axis,fontsize = 20)
    plt.title(title,fontsize = 20)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
         
    return None

def scatter_xy(var, data, # grid to plot on
            title, # title for the figure
           ):

    plt.figure(figsize=(6,5)).set_tight_layout(False)
    plt.scatter(data["zeta"],data["theta"], c=var, cmap='RdYlBu')
    plt.colorbar()
    plt.title(title, fontsize = 20)
    plt.xlabel(''r'$\zeta$',fontsize = 20)
    plt.ylabel(''r'$\theta$',fontsize = 20)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    return None

def plot_figure(var,   # variable to plot
                grid, # grid to plot on
                title, # title for the figure
               ):
    
    levels = jnp.arange(min(var),
                        max(var)*1.05,
                        (1.01*max(var)-min(var))/500
                       )

    plt.figure(figsize=(6,5)).set_tight_layout(False)
    plt.contourf(grid.nodes[grid.unique_zeta_idx,2],
                 grid.nodes[grid.unique_theta_idx,1],
                 (var).reshape(grid.num_theta,
                                 grid.num_zeta,order="F"),

                 levels = levels,
                )
    plt.ylabel(''r'$\theta$',fontsize = 20)
    plt.xlabel(''r'$\zeta$',fontsize = 20)
    plt.title(title,fontsize = 20)
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
              
    return None

def plot_figure4(var, grid, title, N=None):

    Z = var.reshape(grid.num_theta, grid.num_zeta, order="F")

    plt.figure(figsize=(6,5)).set_tight_layout(False)

    cf = plt.contourf(
        grid.nodes[grid.unique_zeta_idx, 2],
        grid.nodes[grid.unique_theta_idx, 1],
        Z,
        #levels=[-0.01,0.25, 0.5, 1.01],
        #cmap="gray"
        levels = [-0.01, 0.1, 0.9, 1.01],
        colors = ["black", "orange", "white"]
    )

    plt.ylabel(r'$\theta$', fontsize=20)
    plt.xlabel(r'$\zeta$', fontsize=20)
    plt.title(title, fontsize=20)

    cbar = plt.colorbar(cf, ticks=[0,0.5, 1])
    cbar.ax.set_yticklabels(['0','0.5', '1'])
    cbar.ax.tick_params(labelsize=15)

    return None
    
def plot_figure3(var,   # variable to plot
                grid, # grid to plot on
                title, # title for the figure
                 N,
               ):
    
    levels = jnp.arange(min(var),
                        jnp.max(var)*1.05,
                        (1.01*jnp.max(var)-jnp.min(var))/N
                       )

    plt.figure(figsize=(6,5)).set_tight_layout(False)
    plt.contourf(grid.nodes[grid.unique_zeta_idx,2],
                 grid.nodes[grid.unique_theta_idx,1],
                 (var).reshape(grid.num_theta,
                                 grid.num_zeta,order="F"),

                 levels = levels,
                )
    plt.ylabel(''r'$\theta$',fontsize = 20)
    plt.xlabel(''r'$\zeta$',fontsize = 20)
    plt.title(title,fontsize = 20)
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
              
    return None

def plot_figure2(var,   # variable to plot
                grid, # grid to plot on
                title, # title for the figure
               ):
    
    plt.figure(figsize=(6,5)).set_tight_layout(False)
    plt.contourf(grid.nodes[grid.unique_zeta_idx,2],
                 grid.nodes[grid.unique_theta_idx,1],
                 (var).reshape(grid.num_theta,
                                 grid.num_zeta,order="F"),
                )
    plt.ylabel(''r'$\theta$',fontsize = 20)
    plt.xlabel(''r'$\zeta$',fontsize = 20)
    plt.title(title,fontsize = 20)
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
              
    return None

def plot_figure5(var,   # variable to plot
                grid, # grid to plot on
                title, # title for the figure
                 a,
                 b
               ):
    
    levels = jnp.arange(min(var),
                        max(var)*1.05,
                        (1.01*max(var)-min(var))/500
                       )

    alpha = jnp.sqrt((a-b)/(a+b)
                    )
    c = jnp.sqrt(a**2 - b**2)
    
    plt.figure(figsize=(6,5)).set_tight_layout(False)
    plt.contourf((grid.nodes[grid.unique_zeta_idx,2]  - jnp.pi),#*c,
                  2*1*jnp.arctan(alpha*jnp.tan((grid.nodes[grid.unique_theta_idx,1] - jnp.pi)/2
                                              )
                                ),
                 (var).reshape(grid.num_theta,
                                 grid.num_zeta,order="F"),

                 levels = levels,
                )

    plt.ylabel(''r'$v/b$',fontsize = 20)
    plt.xlabel(''r'$u/c$',fontsize = 20)
    plt.title(title,fontsize = 20)
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
              
    return None


def surf_int(f,data,grid):

    Q = f*data["|e_theta x e_zeta|"]
    
    integrand = grid.spacing[:, 1] * grid.spacing[:, 2] * Q
    desired_rho_surface = 1.0
    indices = jnp.where(grid.nodes[:, 0] == desired_rho_surface)[0]
    integrand = integrand[indices]
    
    return integrand.sum()

def sqr_int(f,data,grid):
    
    integrand = grid.spacing[:, 1] * grid.spacing[:, 2] * f
    desired_rho_surface = 1.0
    indices = jnp.where(grid.nodes[:, 0] == desired_rho_surface)[0]
    integrand = integrand[indices]
    
    return integrand.sum()


#Zeta line-integration:
def int_zeta_line(data,b,y):
    
    # Integrate the integrand along contours of constant zeta and build the array for V(theta,zeta)
    # Define the integrand. j indicates the contour on zeta
    
    n_size = int(np.sqrt(data["theta"].shape[0]))

    # Points along integration contour. Technically these are the same for all the integrals
    x_vals = (data["zeta"].reshape((n_size,n_size)))[:,0]
    
    V = jnp.zeros_like(x_vals)
    
    integrand = (jnp.exp(-b)*y)#[(j)*n_size:(j+1)*n_size]
    
    # Initiate a variable to compute the line integral of B along the toroidal contour
    integral_c = 0
    
    # For-loop for trapezoidal rule as an approximation to the line integral
    for i in range(0,len(x_vals[:])-1):
        
        integral_c = (integral_c 
                          + 0.5*(x_vals[i+1] - x_vals[i])*(integrand[i+1] + integrand[i] )
                         )
        # Assign the value of each integration to the value of V
        V = V.at[i + 1].set(integral_c)    

    return V

# Theta integration
def int_theta(data,b,y,V0):
    # Integrate the integrand along contours of constant zeta and build the array for V(theta,zeta)

    # Define the integrand. j indicates the contour on zeta

    #x_vals = kdata["theta"][(j)*n_size:(j+1)*n_size] # Cgrid.nodes[:,2]
    
    V = jnp.zeros_like(data["theta"])
    #V = V.reshape(-1)
    
    n_size = int(np.sqrt(data["theta"].shape[0]))
    
    #integral_c = None
    for j in range(0,n_size):

        # Points along integration contour. Technically these are the same for all the integrals
        x_vals = data["theta"][(j)*n_size:(j+1)*n_size]

        # Define the integrand for the contour
        integrand = (jnp.exp(-b)*y)[(j)*n_size:(j+1)*n_size]

        # Initiate a variable to compute the line integral of B along the toroidal contour
        V = V.at[j*n_size].set(V0[j])  
        
        integral_c = V0[j]

        # For-loop for trapezoidal rule as an approximation to the line integral
        for i in range(0,len(x_vals[:])-1):

            integral_c = (integral_c 
                          + 0.5*(x_vals[i+1] - x_vals[i])*(integrand[i+1] + integrand[i] )
                         )

            # Assign the value of each integration to the value of V
            V = V.at[j*n_size + i + 1].set(integral_c)    

    return V

######################################################################################################################
# Define function to find A and its respective derivatives
def u_div_residual(grid, data, x,):

    f_t = first_derivative_t(x, data, grid,)
    
    f_z = first_derivative_z(x, data, grid,)
    
    f_tt = first_derivative_t(f_t, data, grid,)
    
    f_zz = first_derivative_z(f_z, data, grid,)
    
    f_tz = first_derivative_z(f_t, data, grid,)
    
    ##################################################################################################################
    
    # Surface Laplacian of theta
    nabla_s_2_f = (dot(data["e^theta_s"],data["e^theta_s_t"])*f_t
                       + dot(data["e^theta_s"],data["e^theta_s"])*f_tt
                       + dot(data["e^theta_s"],data["e^zeta_s_t"])*f_z
                                   + dot(data["e^theta_s"],data["e^zeta_s"])*f_tz
                                   + dot(data["e^zeta_s"],data["e^theta_s_z"])*f_t
                                   + dot(data["e^theta_s"],data["e^zeta_s"])*f_tz
                                   + dot(data["e^zeta_s"],data["e^zeta_s_z"])*f_z
                                   + dot(data["e^zeta_s"],data["e^zeta_s"])*f_zz
                      )
    
    return nabla_s_2_f

############################################################
# Function to find the scalar that cancels the surface divergence
def find_phi(egrid,edata,):
    
    x = jnp.ones(edata["theta"].shape[0])
    fun_wrapped = lambda x: u_div_residual(egrid, edata, x)
    A = jax.jacfwd(fun_wrapped)(x)
    
    # We will cancel the surface divergence of theta on the surface
    rhs = edata["nabla_s^2_theta"]
    return jnp.linalg.pinv(A)@rhs

def div_(grid, data, x,):
    
    ##################################################################################################################
    
    f_t = first_derivative_t(x, data, grid,)
    
    f_z = first_derivative_z(x, data, grid,)
    
    f_tt = first_derivative_t(f_t, data, grid,)
    
    f_zz = first_derivative_z(f_z, data, grid,)
    
    f_tz = first_derivative_z(f_t, data, grid,)
    
    # Surface Laplacian of theta
    div2 = (dot(data["e^theta_s"],data["e^theta_s_t"])*f_t
                                   + dot(data["e^theta_s"],data["e^theta_s"])*f_tt
                                   + dot(data["e^theta_s"],data["e^zeta_s_t"])*f_z
                                   + dot(data["e^theta_s"],data["e^zeta_s"])*f_tz
                                   + dot(data["e^zeta_s"],data["e^theta_s_z"])*f_t
                                   + dot(data["e^theta_s"],data["e^zeta_s"])*f_tz
                                   + dot(data["e^zeta_s"],data["e^zeta_s_z"])*f_z
                                   + dot(data["e^zeta_s"],data["e^zeta_s"])*f_zz
                      )
    
    
    return div2

def grad_(grid, data, x,):
    
    ##################################################################################################################
    
    f_t = first_derivative_t(x, data, grid,)
    
    f_z = first_derivative_z(x, data, grid,)
    
    grad_f = (f_t*data["e^theta_s"].T 
              + f_z*data["e^zeta_s"].T
             ).T
    
    return grad_f

# Find isothermal field on the surface
def find_H1(egrid,edata):

    phi = find_phi(egrid,edata,)
    # Build a harmonic vector
    grad = grad_(egrid, edata,phi)
    H1 = edata["e^theta_s"] - grad
    #H2 = cross(edata["n_rho"],H1)
    
    return H1

################################################################
# Define own functions to calculate magnetic field
def _compute_magnetic_field_from_Current2(phi_scalar,
                                         surf_data, 
                                         Kgrid,
                                         eval_data,
                                         #coords, eq, Bgrid,
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

    #Bdata = eq.compute(["R","phi","Z","n_rho"], grid = Bgrid)
    coords = jnp.vstack([eval_data["R"],eval_data["phi"],eval_data["Z"]]).T
    
    phi_t = first_derivative_t2(phi_scalar, surf_data,
                                        2 * (Kgrid.M) + 1 , 2 * (Kgrid.N) + 1 )
    phi_z = first_derivative_z2(phi_scalar, surf_data,
                                2 * (Kgrid.M) + 1 , 2 * (Kgrid.N) + 1 )
    
    K_at_grid = ( ( - phi_z * (surf_data["|e_theta x e_zeta|"]) **(-1) ) * surf_data["e_theta"].T 
                 + ( phi_t * (surf_data["|e_theta x e_zeta|"]) **(-1) ) * surf_data["e_zeta"].T ) .T

    
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
    #data = surface.compute(["x", "|e_theta x e_zeta|"], grid=Kgrid, basis="xyz")

    _rs = xyz2rpz(surf_data["x"])
    _K = K_at_grid

    # surface element, must divide by NFP to remove the NFP multiple on the
    # surface grid weights, as we account for that when doing the for loop
    # over NFP
    _dV = Kgrid.weights * surf_data["|e_theta x e_zeta|"] / Kgrid.NFP

    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi = (Kgrid.nodes[:, 2] + j * 2 * jnp.pi / Kgrid.NFP) % (
            2 * jnp.pi
        )
        # new coords are just old R,Z at a new phi (bc of discrete NFP symmetry)
        rs = jnp.vstack((_rs[:, 0], phi, _rs[:, 2])).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_general( coords, rs, K, _dV,
        )
        f += fj
        return f
    
    B = fori_loop(0, Kgrid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        
    return B