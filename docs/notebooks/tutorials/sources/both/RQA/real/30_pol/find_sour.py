import numpy as np
import os

import copy
import scipy
import sys
import functools
import pickle

import jax
import jax.numpy as jnpå
from jax import jit, jacfwd, lax

from netCDF4 import Dataset
import h5py

from desc.backend import put, fori_loop, jnp, sign

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import ConcentricGrid, LinearGrid, Grid, QuadratureGrid
from desc.io import InputReader, load
from desc.objectives import *
from desc.objectives.objective_funs import _Objective
from desc.plotting import plot_1d, plot_2d, plot_3d, plot_section, plot_surfaces, plot_comparison

from desc.plotting import *

from desc.transform import Transform
from desc.derivatives import Derivative

from desc.backend import fori_loop, jit, jnp, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.coils import *

from desc.utils import flatten_list, cross, dot

import numpy as np

from numpy import ndarray

from desc.fns_simp import _compute_magnetic_field_from_Current, _compute_magnetic_field_from_Current_Contour

from scipy.interpolate import griddata


def bn_res(p_M, p_N, 
           sdata1,
           sdata2,
           sdata3,
           sgrid, surface, #w_surface,
           y, N, d_0, eq, Bgrid):
    
    B0 = B_sour(p_M, p_N,
                sdata1,
                sdata2,
                sdata3,
                sgrid, surface,
                y, N, d_0, eq, Bgrid)

    B_wire_cont = B_theta_contours(p_M, p_N,
                     sdata1,
                     sgrid,
                     surface,
                     y,
                     eq,
                     Bgrid,)

    B_sticks_ = B_sticks(p_M, p_N,
                         sgrid,
                         surface,
                         y,
                         eq,
                         Bgrid,
                        )

    B_total = (B0
               + B_wire_cont 
               + B_sticks_ 
              )
    
    return jnp.concatenate((B_total[:,0],B_total[:,1],B_total[:,2], # B field
                            jnp.asarray([jnp.sum(y)]) # Sum of sources and sinks
                           ))

def B_sour(p_M, p_N,
                 sdata1,
               sdata2,
               sdata3,
                 sgrid,
                 surface,
                 #w_surface,
                 y,
                 N,
                 d_0,
                 eq,
                 Bgrid,):
    
    return _compute_magnetic_field_from_Current(sgrid, 
                                                K_sour(p_M, p_N,
                                                       sdata1,
                                                       sdata2,
                                                       sdata3,
                                                       sgrid,
                                                       surface,
                                                       y,
                                                       #dt, dz,
                                                       N,
                                                       d_0), 
                                                surface, eq, Bgrid, basis="rpz")

def B_theta_contours(p_M, p_N,
                     sdata,
                     sgrid,
                     surface,
                     y,
                     eq,
                     Bgrid,):

    theta_coarse = jnp.linspace(2 * jnp.pi * (1 / (p_M * 2)) * 1/2,
                                2 * jnp.pi * (1 - 1 / (p_M * 2) * 1/2),
                                p_M * 2)

    zeta_coarse = jnp.linspace(2 * jnp.pi / surface.NFP * (1 / (p_N * 2)) * 1/2,
                               2 * jnp.pi / surface.NFP * (1 - 1 / (p_N * 2) * 1/2),
                               p_N * 2,)

    # Refine the grid to do the contour integration of the poloidal wires
    add_points = 5
    theta = densify_linspace(theta_coarse, points_per_interval = add_points)
    zeta = zeta_coarse # no need to refine zeta positions since these are fixed during the integration
    #zeta = densify_linspace(zeta_coarse, points_per_interval = add_points)

    assert (p_M * 2)*(p_N * 2) == theta_coarse.shape[0] * zeta_coarse.shape[0], "Check that the sources coincide with the number of sources/sinks"

    r = theta_coarse.shape[0] * zeta_coarse.shape[0] # ss_data["theta"].shape[0]

    r_t = theta_coarse.shape[0]
    r_z = zeta_coarse.shape[0]

    ss_grid = alt_grid_sticks(theta, zeta, sgrid)
    ss_data = surface.compute(['theta','zeta','e_theta'], grid = ss_grid)
    
    sign_vals = jnp.where(ss_data['theta'] < jnp.pi, -1, 0) + jnp.where(ss_data['theta'] > jnp.pi, 1, 0) 
    
    K_cont = 0
    for i in range(0,r_z):
        
        for j in range(0,r_t):
        
            k_fix = jnp.where((ss_data['zeta'] == zeta_coarse[i]) & (ss_data['theta'] > theta_coarse[j]), 1,0)
            K_cont += (y[i*r_t + j] * sign_vals * k_fix * dot(ss_data['e_theta'],ss_data['e_theta'])**(-1/2) * ss_data['e_theta'].T).T

    return _compute_magnetic_field_from_Current_Contour(ss_grid, 
                                                        K_cont,
                                                        surface, eq, Bgrid, basis="rpz")

def B_sticks(p_M, p_N,
                 sgrid,
                 surface,
                 y,
                 eq,
                 Bgrid,
            ):

    theta = jnp.linspace(2 * jnp.pi * (1 / (p_M * 2)) * 1/2,
                         2 * jnp.pi * (1 - 1 / (p_M * 2) * 1/2),
                         p_M * 2)

    # These are the sticks that are all located at the (theta = pi) cut
    theta = jnp.pi * jnp.ones_like(theta)
    
    zeta = jnp.linspace(2 * jnp.pi / surface.NFP * (1 / (p_N * 2)) * 1/2,
                        2 * jnp.pi / surface.NFP * (1 - 1 / (p_N * 2) * 1/2),
                        p_N * 2,)

    #zeta = jnp.linspace(2 * jnp.pi / 1 * (1 / (p_N * 2)) * 1/2,
    #                    2 * jnp.pi / 1 * (1 - 1 / (p_N * 2) * 1/2),
    #                    p_N * 2,)
    
    name = "iso_coords/"
    stick_grid = alt_grid_sticks(theta, zeta, sgrid)
    ss_data = surface.compute(['theta', 'x'], grid = stick_grid)
    #interp_grid(theta, zeta, surface, name)

    eq_surf = eq.surface
    pls_points = eq_surf.compute(["x"], grid = Bgrid, basis = 'xyz')["x"]
    
    assert (p_M * 2)*(p_N * 2) == ss_data["theta"].shape[0] , "Check that the sources coincide with the number of sources/sinks"

    r = ss_data["theta"].shape[0]  # Make r a Python int for indexing
    
    b_sticks_total = 0
    for i in range(0,r):
    
        b_stick_ = stick(ss_data["x"][i], # Location of the wire at the theta = pi cut, variable zeta position
                         0 * ss_data["x"][i], # All wires at the center go to the origin
                         pls_points,
                         sgrid, 
                         basis = "rpz",
                         )

        b_sticks_total +=  y[i] * b_stick_
    
    return b_sticks_total

def stick(p2_, # second point of the stick
          p1_, # first point of the stick
          plasma_points, # points on the plasma surface
          surface_grid, #Kgrid,
          basis = "rpz",
          ):

    #surface_grid = Kgrid
    #p2_ = xyz2rpz(p2)

    #print(p2_)
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi2 = ( p2_[2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )
        #phi1 = ( p1_[2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )
        #(surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )

        p2s = jnp.vstack((p2_[0], phi2, p2_[2])).T
        p2s = rpz2xyz(p2s)
        #p1s = jnp.vstack((p1_[0], phi1, p1_[2])).T
        #p1s = #rpz2xyz(p1s)
        
        a_s = p2s - p1_
        b_s = p1_ - plasma_points
        c_s = p2s - plasma_points
        
        c_sxa_s = cross(c_s, a_s)
    
        f += 1e-7 * ( ( jnp.sum(c_sxa_s*c_sxa_s, axis = 1) * jnp.sum(c_s * c_s, axis = 1) ** (1/2)
                           ) ** (-1) * (jnp.sum(a_s * c_s) - jnp.sum(a_s * b_s)
                                       ) * c_sxa_s.T 
                         ).T
        return f
    
    b_stick = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(plasma_points))
    
    if basis == "rpz":
        b_stick = xyz2rpz_vec(b_stick, x=plasma_points[:, 0], y=plasma_points[:, 1])
        
    return b_stick

def K_sour(p_M, p_N,
           sdata1,
           sdata2,
           sdata3,
           sgrid,
           surface,
           y,
           #dt, dz,
           N,
           d_0):

    theta = jnp.linspace(2 * jnp.pi * (1 / (p_M * 2)) * 1/2,
                         2 * jnp.pi * (1 - 1 / (p_M * 2) * 1/2),
                         p_M * 2)

    zeta = jnp.linspace(2 * jnp.pi / surface.NFP * (1 / (p_N * 2)) * 1/2,
                        2 * jnp.pi / surface.NFP * (1 - 1 / (p_N * 2) * 1/2),
                        p_N * 2,)

    #zeta = jnp.linspace(2 * jnp.pi / 1 * (1 / (p_N * 2)) * 1/2,
    #                    2 * jnp.pi / 1 * (1 - 1 / (p_N * 2) * 1/2),
    #                    p_N * 2,)

    name = "iso_coords/"
    ss_data = interp_grid(theta, zeta, surface, name)

    #r = int(ss_data["theta"].shape[0])  # Make r a Python int for indexing
    assert (p_M * 2)*(p_N * 2) == ss_data["theta"].shape[0] , "Check that the sources coincide with the number of sources/sinks"
    
    r = ss_data["theta"].shape[0]  # Make r a Python int for indexing

    ss_data["u_iso"] = jnp.asarray(ss_data["u_iso"])
    ss_data["v_iso"] = jnp.asarray(ss_data["v_iso"])
    
    def body_fun1(i, carry):
        omega_total_real, omega_total_imag = carry

        y_ = lax.dynamic_index_in_dim(y, i, axis=0)

        # Need to evlauate three omegas
        omega_s1 = omega_sour(sdata1,
                              ss_data["u_iso"][i], ss_data["v_iso"][i],
                              N, d_0
                              )

        omega_total_real += y_ * jnp.real(omega_s1)
        omega_total_imag += y_ * jnp.imag(omega_s1)
        return omega_total_real, omega_total_imag

    def body_fun2(i, carry):
        omega_total_real, omega_total_imag = carry

        y_ = lax.dynamic_index_in_dim(y, i, axis=0)

        # Need to evlauate three omegas
        omega_s2 = omega_sour(sdata2,
                              ss_data["u_iso"][i], ss_data["v_iso"][i],
                              N, d_0
                              )

        omega_total_real += y_ * jnp.real(omega_s2)
        omega_total_imag += y_ * jnp.imag(omega_s2)
        return omega_total_real, omega_total_imag

    def body_fun3(i, carry):
        omega_total_real, omega_total_imag = carry

        y_ = lax.dynamic_index_in_dim(y, i, axis=0)

        # Need to evlauate three omegas
        omega_s3 = omega_sour(sdata3,
                              ss_data["u_iso"][i], ss_data["v_iso"][i],
                              N, d_0
                              )

        omega_total_real += y_ * jnp.real(omega_s3)
        omega_total_imag += y_ * jnp.imag(omega_s3)
        return omega_total_real, omega_total_imag

    omega_total_real1, omega_total_imag1 = fori_loop(0, r, body_fun1, (jnp.zeros_like(sdata1["theta"]),
                                                                       jnp.zeros_like(sdata1["theta"]))
                                                    )

    omega_total_real2, omega_total_imag2 = fori_loop(0, r, body_fun2, (jnp.zeros_like(sdata2["theta"]),
                                                                       jnp.zeros_like(sdata2["theta"]))
                                                     )

    omega_total_real3, omega_total_imag3 = fori_loop(0, r, body_fun3, (jnp.zeros_like(sdata3["theta"]),
                                                                       jnp.zeros_like(sdata3["theta"]))
                                                     )

    # Assume periodicity on the values of the sources so
    return ( ( sdata1["lambda_iso"] ** (-1) ) * ( - omega_total_imag1 * sdata1["e_v"].T #- cross(sdata["n_rho"],sdata["e_u"]).T 
                                                 - omega_total_imag2 * sdata2["e_v"].T 
                                                - omega_total_imag2 * sdata3["e_v"].T 
                                                + omega_total_real1 * sdata1["e_u"].T#- cross(sdata["n_rho"],sdata["e_v"]).T
                                                + omega_total_real2 * sdata2["e_u"].T
                                                + omega_total_real2 * sdata3["e_u"].T
                                                )
            ).T

def f_sour(data_or,
           u1_, v1_, # first dipole
           N, d_0):
    
    gamma = data_or["du"] / (2*jnp.pi)
    
    # Evaluate the dipoles on the grid of vortices
    w1 = comp_loc(u1_, v1_,)
    
    v_1_num = v1_eval(w1, N, d_0, data_or)
    
    f1_reg = f_reg(w1, d_0, data_or)

    return ( jnp.log( v_1_num )
            - jnp.real( w1 ) / ( gamma * data_or["dv"] ) * data_or["w"] 
            - 1/2 * f1_reg
           )

def omega_sour(data_or,
               u1_, v1_, # first dipole
               N,
               d_0,
              ):
    
    w1 = comp_loc(u1_, v1_,)
    
    v_1_num = v1_eval(w1, N,  d_0, data_or)
    v_1_num_prime = v1_prime_eval(w1, N, d_0, data_or)
    
    chi_reg_1 = chi_reg(w1, d_0, data_or)
    
    gamma = data_or["du"] / ( 2 * jnp.pi )
    
    omega = ( v_1_num_prime / v_1_num # Regularized near the vortex cores
             - jnp.real( w1 ) / ( gamma * data_or["dv"] ) 
             + 1 / 2 * ( chi_reg_1 ) # Additional terms with regularization close to the vortex core
            )
    
    return omega

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
                     2 * p **(1/4)*product_ ,
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
    
    _data["du"] = 2*jnp.pi#jnp.max(u.flatten()) - jnp.min(u.flatten())
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
    v_mod = - lamb_ratio * (u_mod + b0 * ( theta_mod - add_extra_periodic(Psi, n_size,m_size) ) )
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

def interp_grid(theta, zeta, w_surface, name):
    
    # Find grids for dipoles
    s_grid = alt_grid(theta,zeta)
    
    # Evaluate data on grids of dipoles
    s_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s","x",
                               "e_theta", # extra vector needed for the poloidal wire contours
                               ], grid = s_grid)
    
    return iso_coords_interp(name, s_data, s_grid, w_surface)

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
    
    return Grid(jnp.stack( ( jnp.ones_like(theta_flat),
                            theta_flat, zeta_flat ) 
                          ).T,
                )

def alt_grid_sticks(theta,zeta, sgrid):

    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta)
    theta_flat = theta_grid.flatten()
    zeta_flat = zeta_grid.flatten() 

    return Grid(jnp.stack( ( jnp.ones_like(theta_flat),
                                 theta_flat, zeta_flat ) 
                              ).T,
                     weights = jnp.ones_like(theta_flat),
                     NFP = sgrid.NFP,
                     )

def densify_linspace(arr, points_per_interval=1):
    """
    Given a jnp.linspace array, return a new array with additional points
    between each pair of original points while keeping all original points.

    Args:
        arr (jnp.ndarray): Original 1D array (typically from jnp.linspace)
        points_per_interval (int): Number of points to insert between each pair

    Returns:
        jnp.ndarray: New array with original + additional interpolated points
    """
    if arr.ndim != 1:
        raise ValueError("Only 1D arrays supported")
    
    new_points = []
    for i in range(len(arr) - 1):
        start = arr[i]
        end = arr[i + 1]
        
        # Include original point
        new_points.append(start)

        # Generate internal points (excluding end to avoid duplication)
        if points_per_interval > 0:
            inter_points = jnp.linspace(start, end, points_per_interval + 2)[1:-1]
            new_points.append(inter_points)

    new_points.append(arr[-1])  # Don't forget the last point!

    return jnp.concatenate([jnp.atleast_1d(p) for p in new_points])