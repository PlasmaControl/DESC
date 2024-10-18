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
                                        biot_savart_potential, 
                                        biot_savart_t, surf_div_general,
                                        vector_potential_general,
                                       vector_potential_potential)

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

#from desc.regcoil import run_regcoil
#from desc.optimize import lsqtr, lsq_auglag

#from scipy.optimize import NonlinearConstraint 

from desc.magnetic_fields import FourierCurrentPotentialField

#import time

import numpy as np
from numpy.linalg import eig

#from scipy.linalg import null_space
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

# Define a fuction to evaluate vectors on a given surface with a point grid
def data_eval(sgrid,surface):

    data = surface.compute(["n_rho",
                            "n_rho_t","n_rho_z",
                            #"x",
                            "X","Y","Z",
                            "R","phi",
                            "X_r","Y_r","Z_r",
                            "X_t","Y_t","Z_t",
                            "X_z","Y_z","Z_z",
                            "Z_tt","Z_zz","Z_tz",
                            "e^rho","e^theta","e^zeta",
                            "e^rho_r","e^rho_t","e^rho_z",
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
                            ##
                            "R_t","R_z",
                            "R_tt","R_zz","R_tz",
                            "phi_t","phi_z",
                            "phi_tt","phi_zz","phi_tz",
                            ],
                           grid = sgrid)
    
    
    data["X_tt"] = (data["R_tt"]*np.cos(data["phi"])
                - data["R_t"]*np.sin(data["phi"])*data["phi_t"]
                - data["R_t"]*np.sin(data["phi"])*data["phi_t"]
                - data["R"]*np.cos(data["phi"])*data["phi_t"]*data["phi_t"]
                - data["R"]*np.sin(data["phi"])*data["phi_tt"]
               )

    data["X_zz"] = (data["R_zz"]*np.cos(data["phi"])
                    - data["R_z"]*np.sin(data["phi"])*data["phi_z"]
                    - data["R_z"]*np.sin(data["phi"])*data["phi_z"]
                    - data["R"]*np.cos(data["phi"])*data["phi_z"]*data["phi_z"]
                    - data["R"]*np.sin(data["phi"])*data["phi_zz"]
                   )
    
    # Find derivatives of the norm of the normal unit vector
    #data["|e_theta x e_zeta|_z"] = dot(
    #                                    (
    #                                    cross(data["e_theta_z"], data["e_zeta"])
    #                                    + cross(data["e_theta"], data["e_zeta_z"])
    #                                    ),
    #                                    cross(data["e_theta"], data["e_zeta"]),
    #                                    ) / (data["|e_theta x e_zeta|"])

    #data["|e_theta x e_zeta|_t"] = dot(
    #                                    (
    #                                    cross(data["e_theta_t"], data["e_zeta"])
    #                                    + cross(data["e_theta"], data["e_zeta_t"])
    #                                    ),
    #                                    cross(data["e_theta"], data["e_zeta"]),
    #                                    ) / (data["|e_theta x e_zeta|"])

    data["n_t"] = data["n_rho_t"]
    data["n_z"] = data["n_rho_z"]
    
    # Radial derivatives must be found using n = e^rho/|e^rho|
    norm = jnp.sqrt(dot(data["e^rho"],data["e^rho"]))
    #norm2 = dot(data["e^rho"],data["e^rho"])
    
    data["n_r"] = ((1/norm)*(data["e^rho_r"] 
                             - (dot(data["n_rho"],data["e^rho_r"])*data["n_rho"].T ).T 
                            ).T).T 
    
    # First normal derivative of unit normal vector
    data["n_n"] = ((dot(data["n_rho"],data["e^rho"])*data["n_r"].T
                   + dot(data["n_rho"],data["e^theta"])*data["n_t"].T
                   + dot(data["n_rho"],data["e^zeta"])*data["n_z"].T).T )
    
    # Find the directional tangent vectors
    # e^theta_gamma
    data["e^theta_gamma"] = data["e^theta"] - (dot(data["n_rho"],data["e^theta"])*data["n_rho"].T).T
    # e^zeta_gamma
    data["e^zeta_gamma"] = data["e^zeta"] - (dot(data["n_rho"],data["e^zeta"])*data["n_rho"].T).T
    
    data["g_f"] = - dot(data["n_rho"],
                        cross(data["e^theta_gamma"],data["e^zeta_gamma"]))*(dot(data["e^theta_gamma"],
                                                                                 data["e^zeta_gamma"])**2 
                                                   - dot(data["e^theta_gamma"],
                                                         data["e^theta_gamma"])*dot(data["e^zeta_gamma"],
                                                                                    data["e^zeta_gamma"])
                                                                           )**(-1)
    
    # Find surface divergence of normal vector
    data["nabla_gamma_dot_n"] = (dot(data["e^theta"],data["n_t"]) 
                                 + dot(data["e^zeta"],data["n_z"]))
    
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
        fj = biot_savart_general( coords, rs, K, _dV,
        )
        f += fj
        return f
    
    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        
    return B

def _compute_magnetic_field_t_from_Current(Kgrid,
                                           K_at_grid, 
                                             surface, 
                                             eq,
                                             Bgrid,
                                           indicator,
                                             basis="rpz",
                                          ):
    
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

    Bdata = eq.compute(["R","phi","Z","n_rho",],
                       grid = Bgrid)
    
    coords = np.vstack([Bdata["R"],Bdata["phi"],Bdata["Z"]]).T
    
    Bdata = eq.compute(["R","phi","Z","n_rho","e_theta","e_zeta"],
                       grid = Bgrid, basis="xyz")
    
    assert basis.lower() in ["rpz", "xyz"]
    if hasattr(coords, "nodes"):
        coords = coords.nodes
    coords = jnp.atleast_2d(coords)
    #if basis == "rpz":
    #    coords = rpz2xyz(coords)
    #else:
    #    K_at_grid = xyz2rpz_vec(K_at_grid, x=coords[:, 0], y=coords[:, 1])
    
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
    
    if indicator == "t":
        out = Bdata["e_theta"]#rpz2xyz_vec(Bdata["e_theta"], phi = Bdata["phi"])
    
    if indicator == "z": 
        out = Bdata["e_zeta"]#rpz2xyz_vec(Bdata["e_zeta"], phi = Bdata["phi"])
    
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi = (surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP) % (
            2 * jnp.pi
        )
        # new coords are just old R,Z at a new phi (bc of discrete NFP symmetry)
        rs = jnp.vstack((_rs[:, 0], phi, _rs[:, 2])).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_t(coords,
                            rs,
                            K,
                            _dV,
                            out
                        )
        f += fj
        return f
    
    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        
    return B

def _compute_magnetic_field_from_potential(Kgrid,
                                             psi_at_grid, 
                                             surface,
                                           surface_eq,
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
    #else:
        #K_at_grid = xyz2rpz_vec(K_at_grid, x=coords[:, 0], y=coords[:, 1])
    
    surface_grid = Kgrid

    # compute and store grid quantities
    # needed for integration
    # TODO: does this have to be xyz, or can it be computed in rpz as well?
    data = surface.compute(["x", "|e_theta x e_zeta|",
                            "n_rho",], grid=surface_grid, basis="xyz")
    
    datas = surface_eq.compute(["e^theta","e^zeta",
                             "n_rho","n_rho_t","n_rho_z"], grid=surface_grid, basis="xyz")
    
    datas["e^theta_gamma"] = datas["e^theta"] - (dot(datas["e^theta"],datas["n_rho"])*datas["n_rho"].T).T
    datas["e^zeta_gamma"] = datas["e^zeta"] - (dot(datas["e^zeta"],datas["n_rho"])*datas["n_rho"].T).T
    
    datas["nabla_gamma_dot_n"] = (dot(datas["e^theta_gamma"],datas["n_rho_t"]) 
                                 + dot(datas["e^zeta_gamma"],datas["n_rho_z"])
                                )
    
    # Data on the evaluation surface
    datae = eq.compute(["e_rho","e_theta","e_zeta",
                        "e^rho","e^theta","e^zeta"], grid=Bgrid, basis="xyz")

    _rs = xyz2rpz(data["x"])
    #_K = K_at_grid

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
        #K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_potential(coords,
                                   rs,
                                   psi_at_grid,
                                   _dV,
                                   data["n_rho"],datas["nabla_gamma_dot_n"],
                                   datae["e^rho"], datae["e^theta"], datae["e^zeta"],
                                   datae["e_rho"], datae["e_theta"], datae["e_zeta"]
                                  )
        f += fj
        return f
    
    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        
    return B

def _compute_surface_divergence_from_Current(Kgrid,
                                             K_at_grid, 
                                             surface, 
                                             eq,
                                             Bgrid,
                                             ed,
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
    
    Bdata = eq.compute(["e_theta","e_zeta"], grid = Bgrid,basis = "xyz")
    r_t = Bdata["e_theta"]#rpz2xyz_vec(ed["e_theta"], phi=phi)
    r_z = Bdata["e_zeta"]
    
    #pos_t = r_t #np.vstack([ed["X_t"],ed["Y_t"],ed["Z_t"]]).T
    #pos_z = r_z #np.vstack([ed["X_z"],ed["Y_z"],ed["Z_z"]]).T
    etg = Bdata["e^theta_gamma"]
    ezg = Bdata["e^zeta_gamma"]
    n = Bdata["n_rho"]
    nt = Bdata["n_rho_t"]
    nz = Bdata["n_rho_z"]
    
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
        
        
        fj = surf_div_general(coords,
                              rs,
                              K,
                              _dV,
                              r_t, r_z,
                              etg, ezg,
                              n,
                              nt,nz,
                             )
        f += fj
        return f
    
    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        
    return B

def _compute_vector_potential_from_Current(Kgrid,
                                             K_at_grid, 
                                             surface, 
                                             eq,
                                             Bgrid,
                                             basis="rpz"):
    """Compute vector potential at a set of points.

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
        Vector potential at specified points

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
        fj = vector_potential_general(
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

def _compute_vector_potential_from_potential(Kgrid,
                                             K_at_grid, 
                                             surface, 
                                             eq,
                                             Bgrid,
                                             basis="rpz"):
    """Compute vector potential at a set of points.

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
        Vector potential at specified points

    """

    Bdata = eq.compute(["R","phi","Z"], grid = Bgrid)
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
        fj = vector_potential_potential(
            coords,
            rs,
            K,
            _dV,
        )
        f += fj
        return f
    
    A = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        A = xyz2rpz_vec(A, x=coords[:, 0], y=coords[:, 1])
        
    return A

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

def plot_figure3(var,   # variable to plot
                grid, # grid to plot on
                title, # title for the figure
                 N,
               ):
    
    levels = jnp.arange(min(var),
                        max(var)*1.05,
                        (1.01*max(var)-min(var))/N
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

def plot_figure4(var,   # variable to plot
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

# Define function to minimize Bn
def eqn_residual(data, 
                 b_basis,
                 f_basis,
                 transform1,
                 y,
                 K_s
                ):

    
    b_mn = y[0:b_basis.num_modes]
    f_mn = y[b_basis.num_modes: b_basis.num_modes + f_basis.num_modes]
    
    # Compute V, rho and derivatives on the grid
    fs = {# a
        "b": transform1.transform(b_mn),
        # First derivatives of rho
        "b_t": transform1.transform(b_mn, dt = 1),
        "b_z": transform1.transform(b_mn, dz = 1),
        # f
        "f": transform1.transform(f_mn),
        # First derivatives of rho
        "f_t": transform1.transform(f_mn, dt = 1),
        "f_z": transform1.transform(f_mn, dz = 1),
         }
    
    # Define nabla_gamma_b
    nabla_gamma_b = ((y[b_basis.num_modes 
                           + f_basis.num_modes]*data["theta"]**(0) 
                         + fs["b_t"])*data["e^theta_gamma"].T 
                         + (y[b_basis.num_modes 
                              + f_basis.num_modes + 1]*data["zeta"]**(0) 
                            + fs["b_z"])*data["e^zeta_gamma"].T).T
    
    # Define f
    f = (y[b_basis.num_modes + f_basis.num_modes + 2]*data["theta"] 
         + y[b_basis.num_modes + f_basis.num_modes + 3]*data["zeta"]
         + fs["f"])
    
    # Define nabla_gamma_f
    nabla_gamma_f = ((y[b_basis.num_modes 
                           + f_basis.num_modes + 2]*data["theta"]**(0) 
                         + fs["f_t"])*data["e^theta_gamma"].T 
                         + (y[b_basis.num_modes 
                              + f_basis.num_modes + 3]*data["zeta"]**(0) 
                            + fs["f_z"])*data["e^zeta_gamma"].T).T
    
    error_v = nabla_gamma_f - (f*nabla_gamma_b.T).T - K_s
    
    return dot(error_v,error_v)

def surf_int(f,data,grid):

    Q = f*data["|e_theta x e_zeta|"]
    
    integrand = grid.spacing[:, 1] * grid.spacing[:, 2] * Q
    desired_rho_surface = 1.0
    indices = np.where(grid.nodes[:, 0] == desired_rho_surface)[0]
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