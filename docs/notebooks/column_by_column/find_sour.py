import numpy as jnp
import os

import copy
import scipy
import sys
import functools
import pickle

import jax
import jax.numpy as jnp
from jax import jit, jacfwd, lax

from netCDF4 import Dataset
import h5py

from desc.backend import put, fori_loop, jnp, sign

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import ConcentricGrid, LinearGrid, Grid, QuadratureGrid
from desc.io import InputReader, load

from desc.transform import Transform
from desc.derivatives import Derivative

from desc.backend import fori_loop, jit, jnp, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.utils import flatten_list, cross, dot

from numpy import ndarray

from desc.fns_simp import _compute_magnetic_field_from_Current, _compute_magnetic_field_from_Current_Contour

from scipy.interpolate import griddata

def K_sour(#p_M, p_N,
           sdata1, sdata2, sdata3,
           sgrid,
           surface,
           #y,
           ss_data,
           N, d_0, i,
    ):

    #theta = jnp.linspace(2 * jnp.pi * (1 / (p_M * 2+1)) * 1/2,
    #                     2 * jnp.pi * (1 - 1 / (p_M * 2+1) * 1/2),
    #                     p_M * 2+1)

    #zeta = jnp.linspace(2 * jnp.pi / surface.NFP * (1 / (p_N * 2+1)) * 1/2,
    #                    2 * jnp.pi / surface.NFP * (1 - 1 / (p_N * 2+1) * 1/2),
    #                    p_N * 2+1,)

    #name = "iso_coords/"
    #ss_data = interp_grid(theta, zeta, surface, name)

    #r = int(ss_data["theta"].shape[0])  # Make r a Python int for indexing
    #assert (p_M * 2+1)*(p_N * 2+1) == ss_data["theta"].shape[0] , "Check that the sources coincide with the number of sources/sinks"
    
    #r = ss_data["theta"].shape[0]  # Make r a Python int for indexing

    #omega_total = 0
    #for i in range(0,r):

        # Need to evlauate three omegas
    omega_total  = ( omega_sour(sdata1,
                              ss_data["u_iso"][i], ss_data["v_iso"][i],
                              N, d_0
                              ) 
                    + omega_sour(sdata2,
                              ss_data["u_iso"][i], ss_data["v_iso"][i],
                              N, d_0
                              ) 
                    + omega_sour(sdata3,
                              ss_data["u_iso"][i], ss_data["v_iso"][i],
                              N, d_0
                              )
                    )

    #omega_total = y[i] * (omega_s1 + omega_s2 + omega_s3) 

    # Assume periodicity on the values of the sources so
    return ( ( sdata1["lambda_iso"] ** (-1) ) * ( - jnp.imag(omega_total) * sdata1["e_v"].T #- cross(sdata["n_rho"],sdata["e_u"]).T 
                                                + jnp.real(omega_total) * sdata1["e_u"].T#- cross(sdata["n_rho"],sdata["e_v"]).T
                                                )
            ).T

def B_sour(#p_M, p_N,
                 sdata1,
               sdata2,
               sdata3,
                 sgrid,
                 surface,
                 ss_data,
                 #y,
                 N,
                 d_0,
                 eq,
                 Bgrid,
           index,):
    
    return _compute_magnetic_field_from_Current(sgrid, 
                                                K_sour(#p_M, p_N,
                                                       sdata1,
                                                       sdata2,
                                                       sdata3,
                                                       sgrid,
                                                       surface,
                                                       #y,
                                                       ss_data,
                                                       N,
                                                       d_0, 
                                                    index,), 
                                                surface, eq, Bgrid, basis="rpz")


def bn_res(#p_M, p_N, 
           sdata1,
           sdata2,
           sdata3,
           sgrid, surface, 
           ss_data,
           #y, 
           N, d_0, eq, Bgrid,
           #stick_data,contour_data, contour_grid,
           index,
          ):
    
    B_total = B_sour(#p_M, p_N,
                sdata1,
                sdata2,
                sdata3,
                sgrid, surface, ss_data,
                #y, 
                     N, d_0, eq, Bgrid, index)
    
    return jnp.concatenate((B_total[:,0],B_total[:,1],B_total[:,2], # B field
                            jnp.asarray([1]) # Sum of sources and sinks
                           ))

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

def v1_prime_eval(w0, N, d_0, data_or):

    gamma = data_or["du"] / ( 2 * jnp.pi )
    p = jnp.exp( - data_or["dv"] / ( 2 * gamma ) )
    
    #_product = 0
    #for n in range(0,N):

    #    _product = ( _product 
    #                + ( ( ( (-1) ** n
    #                    ) * ( p ** ( n ** 2 + n )
    #                        )
    #                    ) * ( ( ( 2 * n + 1 ) / ( 2 * gamma )
    #                          ) * jnp.cos( ( 2 * n + 1 ) * ( data_or["w"] - w0 ) / ( 2 * gamma )
    #                                     )
    #                        )
    #                  )
    #               )
    
    #return (p**(1/4)/1)*_product

    def body_fun(n, carry):
        _product = carry
        term = ( ((-1) ** n) * (p ** (n**2 + n)) ) * ( 
            ( (2 * n + 1) / gamma ) * jnp.cos((2 * n + 1) * ( data_or["w"][:,None] - w0[None,:]) / gamma
                                              ) 
            )
        return _product + term
    
    #test = fori_loop( 0, N, body_fun, jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) ) + jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) )*1j)
    
    #return test

    return jnp.where( jnp.abs( data_or["w"][:,None] - w0[None,:] ) > d_0, 
                     ( p ** ( 1 / 4 ) / 1 ) * fori_loop( 0, N, body_fun, jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) ) 
                                                        + jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) )*1j
                                                       ), 
                     0 )
    
def v1_eval(w0, N, d_0, data_or):
    
    gamma = data_or["du"] / ( 2 * jnp.pi )
    p = jnp.exp( - data_or["dv"] / ( 2 * gamma ) )
    
    #product_ = 0
    #for n in range(0,N):

    #    product_ = ( product_ 
    #                + ( ( ( (-1) ** n
    #                    )*( p ** ( n **2 + n )
    #                      )
    #                   ) * jnp.sin( ( 2 * n + 1 ) * ( data_or["w"] - w0 ) / ( 2 * gamma ) )
    #                  )
    #               )

    def body_fun(n,carry):
        product_ = carry
        term = product_ + (  ( ( (-1) ** n) * (p ** (n**2 + n) ) ) * jnp.sin( (2 * n + 1) * (data_or["w"][:,None] - w0[None,:]) / gamma) )
        
        return product_ + term
        
    return jnp.where(jnp.abs(data_or["w"][:,None] - w0[None,:]) > d_0,
                     2 * p **(1/4) * fori_loop(0, N, body_fun, 
                                     jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) ) + jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) )*1j ) ,
                     1 # Arbitraty value of 1 inside the circle around the vortex core
                    )

def chi_reg(w0, d_0, data_or):  # location of the vortex

    return jnp.where(
        jnp.abs(data_or["w"][:,None] - w0[None,:]) < d_0,
        (- ( data_or["lambda_u"] / data_or["lambda_iso"])
        + (data_or["lambda_v"] / data_or["lambda_iso"]) * 1j)[:,None],
        0,
    )

def f_reg(w0,# location of the vortex
          d_0, data_or):
    
    return jnp.where(jnp.abs(data_or["w"] - w0) < d_0,
                     jnp.log(data_or["lambda_iso"]),
                     0)

def comp_loc(theta_0_,phi_0_,):
    
    return theta_0_ + phi_0_*1j

# Load isothermal coordinates and interpolate on a different grid
def iso_coords_interp(name,_data, sgrid, eq):

    u = jnp.load(name + 'u.npy')
    v = jnp.load(name + 'v.npy')
    Phi = jnp.load(name + 'Phi.npy') #first_derivative_t(u, tdata, tgrid,)
    Psi = jnp.load(name + 'Psi.npy') #first_derivative_z(u, tdata, tgrid,)
    b0 = jnp.load(name + 'b.npy') #first_derivative_z(u, tdata, tgrid,)
    lamb_ratio = jnp.load(name + 'ratio.npy') #first_derivative_z(u, tdata, tgrid,)
    
    _data["du"] = 2*jnp.pi#jnp.max(u.flatten()) - jnp.min(u.flatten())
    _data["dv"] = jnp.max(v.flatten()) - jnp.min(v.flatten())

    #_data["omega_1"] = jnp.load(name + 'omega_1.npy')
    #_data["omega_2"] = jnp.load(name + 'omega_2.npy')

    #_data["tau"] = jnp.load(name + 'tau.npy')
    #_data["tau_1"] = jnp.load(name + 'tau_1.npy')
    #_data["tau_2"] = jnp.load(name + 'tau_2.npy')
    
    lamb_u = jnp.load(name + 'lambda_u.npy')
    lamb_v = jnp.load(name + 'lambda_v.npy')

    # Temporary grid
    tgrid = LinearGrid(M = 60, N = 60)

    # Data on plasma surface
    tdata = eq.compute(["theta","zeta"], grid = tgrid)
    
    u_t = jnp.load(name + 'u_t.npy') 
    u_z = jnp.load(name + 'u_z.npy') 
    v_t = jnp.load(name + 'v_t.npy') 
    v_z = jnp.load(name + 'v_z.npy') 
    
    # Build new grids to allow interpolation between last grid points and theta = 2*pi or zeta = 2*pi
    m_size = tgrid.M * 2 + 1
    n_size = tgrid.N * 2 + 1

    # Rearrange variables
    # Add extra rows and columns to represent theta = 2pi or zeta = 2pi
    #theta_mod = add_extra_coords(tdata["theta"], n_size,m_size,0)
    #zeta_mod = add_extra_coords(tdata["zeta"], n_size,m_size,1)
    #u_mod = zeta_mod - add_extra_periodic(Phi, n_size,m_size)
    #v_mod = lamb_ratio * ( theta_mod - add_extra_periodic(Psi, n_size,m_size) + b0 * u_mod )
    #u_t_mod = add_extra_periodic(u_t, n_size,m_size)
    #u_z_mod = add_extra_periodic(u_z, n_size,m_size)
    #v_t_mod = add_extra_periodic(v_t, n_size,m_size)
    #v_z_mod = add_extra_periodic(v_z, n_size,m_size)
    #lamb_u_mod = add_extra_periodic(lamb_u, n_size,m_size)
    #lamb_v_mod = add_extra_periodic(lamb_v, n_size,m_size)

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
    points = jnp.array( (zeta_mod.flatten(), theta_mod.flatten()) ).T

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
    
    return Grid(nodes = jnp.stack( ( jnp.ones_like(theta_flat),
                                    theta_flat, zeta_flat ) 
                                  ).T,
                jitable = True,
                )

def alt_grid_sticks(theta,zeta, sgrid):

    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta)
    theta_flat = theta_grid.flatten()
    zeta_flat = zeta_grid.flatten() 

    return Grid(nodes = jnp.stack( ( jnp.ones_like(theta_flat),
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