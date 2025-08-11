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
from jax import jit, jacfwd, lax

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
from desc.utils import cross, dot

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
                           _compute_magnetic_field_from_Current,
                          )

from desc.finite_diff2 import (first_derivative_t, first_derivative_z)

from scipy.interpolate import griddata

def K_dips(p_M, p_N,
           sdata,
           sgrid,
           surface,
           y,
           dt, dz,
           N,
           d_0):

    theta = jnp.linspace(2 * jnp.pi * (1 / (p_M * 2 + 1)) * 1/2,
                         2 * jnp.pi * (1 - 1 / (p_M * 2 + 1) * 1/2),
                         p_M * 2 + 1)

    zeta = jnp.linspace(2 * jnp.pi / sgrid.NFP * (1 / (p_N * 2 + 1)) * 1/2,
                        2 * jnp.pi / sgrid.NFP * (1 - 1 / (p_N * 2 + 1) * 1/2),
                        p_N * 2 + 1)

    
    name = "iso_coords/"
    dl_data, dr_data, dd_data, du_data = shift_grid(theta, zeta, dt, dz, surface, name)

    du_data["u_iso"] = jnp.asarray(du_data["u_iso"])
    dd_data["u_iso"] = jnp.asarray(dd_data["u_iso"])
    du_data["v_iso"] = jnp.asarray(du_data["v_iso"])
    dd_data["v_iso"] = jnp.asarray(dd_data["v_iso"])
    dr_data["u_iso"] = jnp.asarray(dr_data["u_iso"])
    dr_data["v_iso"] = jnp.asarray(dr_data["v_iso"])
    dl_data["u_iso"] = jnp.asarray(dl_data["u_iso"])
    dl_data["v_iso"] = jnp.asarray(dl_data["v_iso"])

    #r = int(dl_data["theta"].shape[0])  # Make r a Python int for indexing
    
    assert (p_M * 2+1)*(p_N * 2+1) == dl_data["theta"].shape[0] , "Check that the number of dipole locations coincide with the number of dipoles"
    r = dl_data["theta"].shape[0]  # Make r a Python int for indexing

    def body_fun(i, carry):
        omega_total_real, omega_total_imag = carry
        i_r = i + r

        y_pol = lax.dynamic_index_in_dim(y, i, axis=0)
        y_tor = lax.dynamic_index_in_dim(y, i_r, axis=0)

        omega_pol1 = omega_pair(
            sdata,
            du_data["u_iso"][i], du_data["v_iso"][i],
            dd_data["u_iso"][i], dd_data["v_iso"][i],
            N, d_0
        )

        omega_tor1 = omega_pair(
            sdata,
            dr_data["u_iso"][i], dr_data["v_iso"][i],
            dl_data["u_iso"][i], dl_data["v_iso"][i],
            N, d_0
        )

        omega_total_real += y_pol * jnp.real(omega_pol1) + y_tor * jnp.real(omega_tor1)
        omega_total_imag += y_pol * jnp.imag(omega_pol1) + y_tor * jnp.imag(omega_tor1)
        return omega_total_real, omega_total_imag

    omega_total_real, omega_total_imag = fori_loop(0, r, body_fun, (jnp.zeros_like(sdata["theta"]),
                                                                    jnp.zeros_like(sdata["theta"]))
                                                  )

    #return ( ( ( sdata["lambda_iso"] ** (-1) )*( omega_total_imag * sdata["e_u"].T #- cross(sdata["n_rho"],sdata["e_u"]).T 
    #                                            + omega_total_real * sdata["e_v"].T#- cross(sdata["n_rho"],sdata["e_v"]).T
    #                                         )
    #         ).T
    #       )

    return ( ( ( sdata["lambda_iso"] ** (-1) )*( omega_total_imag * cross(sdata["n_rho"],sdata["e_u"]).T 
                                                    + omega_total_real * cross(sdata["n_rho"],sdata["e_v"]).T
                                                 )
                 ).T
               )#.flatten()

def B_dips(p_M, p_N,
                 sdata,
                 sgrid,
                 surface,
                 y,
                 dt,dz,
                 N,
                 d_0,
                 eq,
                 Bgrid,):


    return _compute_magnetic_field_from_Current(sgrid, 
                                                K_dips(p_M, p_N,
                                                       sdata, sgrid, surface, #w_surface,
                                                       y,
                                                       dt,dz, N, d_0), 
                                                surface, eq, Bgrid, basis="rpz")

def B_sticks(p_M, p_N,
                 sdata,
                 sgrid,
                 surface,
                 y,
             dt,dz,
                 eq,
                 Bgrid,
            ):

    #theta = jnp.linspace(2 * jnp.pi * (1 / (p_M * 2)) * 1/2,
    #                     2 * jnp.pi * (1 - 1 / (p_M * 2) * 1/2),
    #                     p_M * 2)
    
    #zeta = jnp.linspace(2 * jnp.pi / sgrid.NFP * (1 / (p_N * 2)) * 1/2,
    #                    2 * jnp.pi / sgrid.NFP * (1 - 1 / (p_N * 2) * 1/2),
    #                    p_N * 2,)

    theta = jnp.linspace(2 * jnp.pi * (1 / (p_M * 2 + 1)) * 1/2,
                         2 * jnp.pi * (1 - 1 / (p_M * 2 + 1) * 1/2),
                         p_M * 2 + 1)

    zeta = jnp.linspace(2 * jnp.pi / sgrid.NFP * (1 / (p_N * 2 + 1)) * 1/2,
                        2 * jnp.pi / sgrid.NFP * (1 - 1 / (p_N * 2 + 1) * 1/2),
                        p_N * 2 + 1)

    
    name = "iso_coords/"
    (dl_data, dr_data, dd_data, du_data) = shift_grid(theta, zeta, dt, dz, 
                                                      surface,
                                                      name)
    
    eq_surf = eq.surface
    pls_points = eq_surf.compute(["x"], grid = Bgrid, basis = 'xyz')["x"]
    
    r = dl_data["theta"].shape[0]
    
    b_sticks_total = 0
    for i in range(0,r):
    
        b_stick_pol = stick(du_data["x"][i], # u-pos of second dipole
                            dd_data["x"][i], # u-pos of first dipole
                            pls_points,
                            sgrid,
                            basis = "rpz",
                            )

    
        b_stick_tor = stick(dr_data["x"][i], # u-pos of second dipole
                            dl_data["x"][i], # u-pos of first dipole
                            pls_points,
                            sgrid,
                            basis = "rpz",
                            )

        b_sticks_total += y[i] * b_stick_pol + y[i+r] * b_stick_tor
    
    return b_sticks_total

def bn_res(p_M, p_N, sdata, sgrid, surface, #w_surface, 
           y, dt, dz, N, d_0, eq, Bgrid):
    
    B0_dips = B_dips(p_M, p_N, sdata, sgrid, surface, #w_surface, 
                y, dt,dz, N, d_0, eq, Bgrid)

    B0_sticks = B_sticks(p_M, p_N,
                         sdata,
                         sgrid,
                         surface,
                         y,
                         dt,dz,
                         eq,
                         Bgrid,)

    # Minus sign for sticks to change the polarity of the current in the sticks
    B0 = (B0_dips
          -B0_sticks)
    #B0_dips
    return jnp.concatenate((B0[:,0],B0[:,1],B0[:,2]))

def f_pair(data_or,
           u1_, v1_, # first dipole
           u2_, v2_, # second dipole
           N, d_0):
    
    gamma = data_or["du"] / (2*jnp.pi)
    
    # Evaluate the dipoles on the grid of vortices
    w1 = comp_loc(u1_, v1_,)
    w2 = comp_loc(u2_, v2_,)
    
    v_1_num = v1_eval(w1, N, d_0, data_or)
    v_1_den = v1_eval(w2, N, d_0, data_or)
    
    f1_reg = f_reg(w1, d_0, data_or)
    f2_reg = f_reg(w2, d_0, data_or)

    return ( jnp.log( v_1_num ) - jnp.log( v_1_den )
            - jnp.real( w1 - w2 ) / ( gamma * data_or["dv"] ) * data_or["w"] 
            - 1/2 * ( f1_reg - f2_reg )
           )

def omega_pair(data_or,
               u1_, v1_, # first dipole
               u2_, v2_, # second dipole
               N,
               d_0,
              ):
    
    w1 = comp_loc(u1_, v1_,)
    w2 = comp_loc(u2_, v2_,)
    
    v_1_num = v1_eval(w1, N,  d_0, data_or)
    v_1_den = v1_eval(w2, N, d_0, data_or) 
    v_1_num_prime = v1_prime_eval(w1, N, d_0, data_or)
    v_1_den_prime = v1_prime_eval(w2, N, d_0, data_or)
    
    chi_reg_1 = chi_reg(w1, d_0, data_or)
    chi_reg_2 = chi_reg(w2, d_0, data_or)
    
    gamma = data_or["du"] / ( 2 * jnp.pi )
    
    omega = ( ( v_1_num_prime / v_1_num - v_1_den_prime / v_1_den ) # Regularized near the vortex cores
             - jnp.real( w1 - w2 ) / ( gamma * data_or["dv"] ) 
             + 1 / 2 * ( chi_reg_1 - chi_reg_2 ) # Additional terms with regularization close to the vortex core
            )
    
    return omega

def stick(p2_, # second point of the stick
          p1_, # first point of the stick
          plasma_points_, # points on the plasma surface
          surface_grid,#Kgrid,
          basis = "rpz",
          ):

    #surface_grid = Kgrid
    #p2_ = xyz2rpz(p2)
    #p1_ = xyz2rpz(p1)
    #plasma_points_ = rpz2xyz(plasma_points)
    
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi2 = ( p2_[2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )
        phi1 = ( p1_[2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )
        #(surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )

        p2s = jnp.vstack((p2_[0], phi2, p2_[2])).T
        p2s = rpz2xyz(p2s)
        p1s = jnp.vstack((p1_[0], phi1, p1_[2])).T
        p1s = rpz2xyz(p1s)
        
        a_s = p2s - p1s
        
        b_s = p1s - plasma_points_
        
        c_s = p2s - plasma_points_
        
        c_sxa_s = cross(c_s, a_s)
    
        f += 1e-7 * ( ( jnp.sum(c_sxa_s*c_sxa_s, axis = 1) * jnp.sum(c_s * c_s, axis = 1) ** (1/2)
                       ) ** (-1) * (jnp.sum(a_s * c_s) - jnp.sum(a_s * b_s)
                                    ) * c_sxa_s.T 
                     ).T
        return f
    
    b_stick = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(plasma_points_))
    
    if basis == "rpz":
        b_stick = xyz2rpz_vec(b_stick, x=plasma_points_[:, 0], y=plasma_points_[:, 1])
        
    return b_stick

def v1_eval(w0, N, d_0, data_or):
    
    gamma = data_or["du"] / ( 2 * jnp.pi )
    p = jnp.exp( - data_or["dv"] / ( 2 * gamma ) )
    
    product_ = 0

    for n in range(0,N):

        product_ = ( product_ 
                    + ( ( ( (-1) ** n
                        )*( p ** ( n **2 + n )
                          )
                       ) * jnp.sin( ( 2 * n + 1 ) * ( data_or["w"] - w0 ) / ( 2 * gamma ) )
                      )
                   )
    
    return jnp.where(jnp.abs(data_or["w"] - w0) > d_0,
                     2*p**(1/4)*product_ ,
                     1 # Arbitraty value of 1 inside the circle around the vortex core
                    )

def chi_reg(w0,# location of the vortex
            d_0, data_or):
    
    return jnp.where( jnp.abs( data_or["w"] - w0 ) < d_0,
                     - ( data_or["lambda_u"] / data_or["lambda_iso"]) + ( data_or["lambda_v"] / data_or["lambda_iso"] ) * 1j,
                     0)

def f_reg(w0,# location of the vortex
          d_0, data_or):
    
    return jnp.where(jnp.abs(data_or["w"] - w0) < d_0,
                     jnp.log(data_or["lambda_iso"]),
                     0)

def v1_prime_eval(w0, N, d_0, data_or):

    gamma = data_or["du"] / ( 2 * jnp.pi )
    p = jnp.exp( - data_or["dv"] / ( 2 * gamma ) )
    
    _product = 0
    for n in range(0,N):

        _product = ( _product 
                    + ( ( ( (-1) ** n
                        ) * ( p ** ( n ** 2 + n )
                            )
                        ) * ( ( ( 2 * n + 1 ) / ( 2 * gamma )
                              ) * jnp.cos( ( 2 * n + 1 ) * ( data_or["w"] - w0 ) / ( 2 * gamma )
                                         )
                            )
                      )
                   )
    
    #return (p**(1/4)/1)*_product
    return jnp.where( abs( data_or["w"] - w0 ) > d_0, 
                     ( p ** ( 1 / 4 ) / 1 ) * _product, 
                     0 )

def comp_loc(theta_0_,phi_0_,):
    
    return theta_0_ + phi_0_*1j

# Load isothermal coordinates and interpolate on a different grid
def iso_coords_interp(name,_data, sgrid, eq):

    u = np.load(name + 'u.npy')
    v = np.load(name + 'v.npy')
    Phi = np.load(name + 'Phi.npy') #first_derivative_t(u, tdata, tgrid,)
    Psi = np.load(name + 'Psi.npy') #first_derivative_z(u, tdata, tgrid,)
    b0 = np.load(name + 'b.npy') #first_derivative_z(u, tdata, tgrid,)
    lamb_ratio = np.load(name + 'ratio.npy') #first_derivative_z(u, tdata, tgrid,)
    
    _data["du"] = 2 * jnp.pi#jnp.max(u.flatten()) - jnp.min(u.flatten()) #* sgrid.NFP
    _data["dv"] = jnp.max(v.flatten()) - jnp.min(v.flatten())
    
    lamb_u = np.load(name + 'lambda_u.npy')
    lamb_v = np.load(name + 'lambda_v.npy')

    # Temporary grid
    tgrid = LinearGrid(M = 60, N = 60)

    # Data on plasma surface
    tdata = eq.compute(["theta","zeta"], grid = tgrid)
    
    u_t = np.load(name + 'u_t.npy') 
    u_z = np.load(name + 'u_z.npy') 
    v_t = np.load(name + 'v_t.npy') 
    v_z = np.load(name + 'v_z.npy') 
    
    # Build new grids to allow interpolation between last grid points and theta = 2*pi or zeta = 2*pi
    m_size = tgrid.M * 2 + 1
    n_size = tgrid.N * 2 + 1

    # Rearrange variables
    # Add extra rows and columns to represent theta = 2pi or zeta = 2pi
    theta_mod = add_extra_coords(tdata["theta"], n_size,m_size,0)
    zeta_mod = add_extra_coords(tdata["zeta"], n_size,m_size,1)
    u_mod = zeta_mod - add_extra_periodic(Phi, n_size,m_size)
    v_mod = lamb_ratio * ( b0 * ( theta_mod - add_extra_periodic(Psi, n_size,m_size) ) )
    #- lamb_ratio * (u_mod + b0 * ( theta_mod - add_extra_periodic(Psi, n_size,m_size) ) )
    
    u_t_mod = add_extra_periodic(u_t, n_size,m_size)
    u_z_mod = add_extra_periodic(u_z, n_size,m_size)
    v_t_mod = add_extra_periodic(v_t, n_size,m_size)
    v_z_mod = add_extra_periodic(v_z, n_size,m_size)
    lamb_u_mod = add_extra_periodic(lamb_u, n_size,m_size)
    lamb_v_mod = add_extra_periodic(lamb_v, n_size,m_size)
    
    # Interpolate on theta_mod, zeta_mod
    points = np.array( (zeta_mod.flatten(), theta_mod.flatten()) ).T

    X0 = _data["zeta"].flatten()
    Y0 = _data["theta"].flatten()

    # Interpolate isothermal coordinates
    _data["u_iso"] = griddata( points, u_mod.flatten(), (X0,Y0), method='linear' )
    _data["v_iso"] = griddata( points, v_mod.flatten(), (X0,Y0), method='linear' )
    
    _data["lambda_u"] = griddata( points, lamb_u_mod.flatten(), (X0,Y0), method='linear' )
    _data["lambda_v"] = griddata( points, lamb_v_mod.flatten(), (X0,Y0), method='linear' )

    # Interpolate derivatives of isothermal coordinates
    u0_t = griddata( points, u_t_mod.flatten(), (X0,Y0), method='linear' )
    u0_z = griddata( points, u_z_mod.flatten(), (X0,Y0), method='linear' )
    v0_t = griddata( points, v_t_mod.flatten(), (X0,Y0), method='linear' )
    v0_z = griddata( points, v_z_mod.flatten(), (X0,Y0), method='linear' )
    
    # Build harmonic vectors with interpolated data
    grad1 = ( u0_t * _data["e^theta_s"].T + u0_z * _data["e^zeta_s"].T ).T
    grad2 = ( v0_t * _data["e^theta_s"].T + v0_z * _data["e^zeta_s"].T ).T
    
    _data["e^u_s"] = grad1
    _data["e^v_s"] = grad2
    
    _data["e_u"] = ( dot(grad1,grad1) ** (-1) * grad1.T ).T
    _data["e_v"] = ( dot(grad2,grad2) ** (-1) * grad2.T ).T

    # Define the parameter "lambda" according to the paper
    _data["lambda_iso"] = dot( _data["e_u"], _data["e_u"] ) ** ( 1 / 2 )

    _data["w"] = comp_loc( _data["u_iso"], _data["v_iso"] )
    
    return _data

def shift_grid(theta, zeta, dt, dz, w_surface, name):
    
    l_zeta = zeta - dz/2
    r_zeta = zeta + dz/2
    d_theta = theta - dt/2
    u_theta = theta + dt/2
    
    # Replace first negative entry of dl_zeta with the real position 2*pi-dz/2
    #l_zeta = l_zeta.at[0].set(2*jnp.pi - dz/2)
    #r_zeta = r_zeta.at[-1].set(0 + dz/2)
    
    # Replace first negative entry of dl_zeta with the real position 2*pi-dz/2
    #d_theta = d_theta.at[0].set(2*jnp.pi - dt/2)
    #u_theta = u_theta.at[-1].set(0 + dt/2)
    
    # Find grids for dipoles
    l_grid = alt_grid(theta,l_zeta)
    r_grid = alt_grid(theta,r_zeta)

    d_grid = alt_grid(d_theta,zeta)
    u_grid = alt_grid(u_theta,zeta)
    
    # Evaluate data on grids of dipoles
    l_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                               #"X","Y","Z",#"x",
                               ], grid = l_grid)
    r_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                               #"X","Y","Z",#"x",
                               ], grid = r_grid)
    d_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                               #"X","Y","Z",#"x",
                               ], grid = d_grid)
    u_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                               #"X","Y","Z",#"x",
                               ], grid = u_grid)
    
    return (iso_coords_interp(name, l_data, l_grid, w_surface),
            iso_coords_interp(name, r_data, r_grid, w_surface),
            iso_coords_interp(name, d_data, d_grid, w_surface),
            iso_coords_interp(name, u_data, u_grid, w_surface)
           )

def add_extra(data_, n_size,m_size):

    _mod = data_.reshape( (n_size, m_size) ).T
    _mod = jnp.column_stack( [_mod, _mod[0:m_size,0]] )
    _mod = jnp.vstack( [ _mod, 2 * jnp.pi * jnp.ones( _mod.shape[1] ) ] )
    
    return _mod

def add_extra_periodic(data_, n_size,m_size):

    _mod = data_.reshape( (n_size, m_size) ).T
    _mod = jnp.column_stack( [_mod, _mod[:,0]] )
    _mod = jnp.vstack( [ _mod, _mod[0,:] ] )
    
    return _mod

def add_extra_coords(data_, n_size,m_size,ind):

    _mod = data_.reshape( (n_size, m_size) ).T
    #_mod = jnp.vstack( [ _mod, _mod[0:m_size,0] ] )

    if ind == 0:
        _mod = jnp.column_stack( [_mod, _mod[:,0]] )
        _mod = jnp.vstack( [ _mod, 2 * jnp.pi * jnp.ones( _mod.shape[1] ) ] )

    if ind == 1:
        _mod = jnp.column_stack( [_mod, 2 * jnp.pi * jnp.ones_like(_mod[:,0])] )
        _mod = jnp.vstack( [ _mod, _mod[0,:] ] )
    
    return _mod

def alt_grid(theta,zeta):

    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta)
    theta_flat = theta_grid.flatten()
    zeta_flat = zeta_grid.flatten() 
    
    return Grid(jnp.stack((np.ones_like(theta_flat),
                           theta_flat,zeta_flat)).T)