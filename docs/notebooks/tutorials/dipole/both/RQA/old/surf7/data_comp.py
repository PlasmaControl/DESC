import numpy as np
import os

from scipy.io import netcdf_file
import copy
import scipy
from scipy.constants import mu_0
import sys
import functools
import pickle

import jax
import jax.numpy as jnp
from jax import jit, jacfwd, lax, vmap

from netCDF4 import Dataset
import h5py

from desc.backend import put, fori_loop, jnp, sign

from desc.io import InputReader, load

from desc.grid import Grid, LinearGrid
from desc.transform import Transform
from desc.derivatives import Derivative
from desc.backend import fori_loop, jit, jnp, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.utils import flatten_list, cross, dot

import numpy as np
from numpy import ndarray

from desc.fns_simp import _compute_magnetic_field_from_Current

from scipy.interpolate import griddata

def data_calc(p_M,p_N,eps,sdata,surface):
#w, w2pol, w1pol, w2tor, w1tor, l, l_u, l_v, e_u, e_v = 
    return data_vectorized(p_M,p_N,eps,sdata,surface)
  
def data_vectorized(p_M,p_N,eps,sdata,surface):

    du_data, dd_data, dr_data, dl_data = iso_loc(p_M,p_N,surface,eps,eps)
    
    n_columns = du_data["theta"].shape[0]
    
    #w_vals = ( jnp.tile(sdata["u_iso"].reshape(sdata["theta"].shape[0], 1), n_columns) 
    #          + jnp.tile(sdata["v_iso"].reshape(sdata["theta"].shape[0], 1), n_columns) * 1j ).T

    w_vals = jnp.asarray( sdata["u_iso"] + sdata["v_iso"] * 1j )
    
    w_vals = jnp.asarray(w_vals)
    w2_pol = jnp.asarray( du_data["u_iso"] + 1j * du_data["v_iso"] )
    w1_pol = jnp.asarray( dd_data["u_iso"] + 1j * dd_data["v_iso"] )
    
    w2_tor = jnp.asarray( dr_data["u_iso"] + 1j * dr_data["v_iso"] )
    w1_tor = jnp.asarray( dl_data["u_iso"] + 1j * dl_data["v_iso"] )
    
    l_vals = jnp.asarray( sdata["lambda_iso"] )#( jnp.tile(sdata["lambda_iso"].reshape(sdata["theta"].shape[0], 1), n_columns) ).T
    l_u_vals = jnp.asarray( sdata["lambda_u"] )#( jnp.tile(sdata["lambda_u"].reshape(sdata["theta"].shape[0], 1), n_columns) ).T
    l_v_vals = jnp.asarray( sdata["lambda_v"] )#( jnp.tile(sdata["lambda_v"].reshape(sdata["theta"].shape[0], 1), n_columns) ).T

    #def repeat_fn(_,x):
    #    return x
    
    e_u_vals = jnp.asarray(sdata["e_u"]) #vmap(repeat_fn, in_axes=(0, None))(jnp.arange(n_columns), sdata["e_u"])
    e_v_vals = jnp.asarray(sdata["e_v"]) #vmap(repeat_fn, in_axes=(0, None))(jnp.arange(n_columns), sdata["e_v"])

    return w_vals, w2_pol, w1_pol, w2_tor, w1_tor, l_vals, l_u_vals, l_v_vals, e_u_vals, e_v_vals

# Function to find the isothermal location of the vortices
def iso_loc(p_M,p_N,surface,dt,dz):

    theta = jnp.linspace(2 * jnp.pi * (1 / (p_M * 2 + 1)) * 1/2,
                         2 * jnp.pi * (1 - 1 / (p_M * 2 + 1) * 1/2),
                         p_M * 2 + 1)

    zeta = jnp.linspace(2 * jnp.pi / surface.NFP * (1 / (p_N * 2 + 1)) * 1/2,
                        2 * jnp.pi / surface.NFP * (1 - 1 / (p_N * 2 + 1) * 1/2),
                        p_N * 2 + 1)

    name = "iso_coords/"
    #dt, dz = 1e-2/3, 1e-2/3
    dl_data, dr_data, dd_data, du_data = shift_grid(theta, zeta, dt, dz, surface, name)

    # Ensure arrays are JAX arrays
    for d in [du_data, dd_data, dr_data, dl_data]:
        d["u_iso"] = jnp.asarray(d["u_iso"])
        d["v_iso"] = jnp.asarray(d["v_iso"])

    return du_data, dd_data, dr_data, dl_data
    
def f_reg(w0,# location of the vortex
          d_0, data_or):
    
    return jnp.where(jnp.abs(data_or["w"] - w0) < d_0,
                     jnp.log(data_or["lambda_iso"]),
                     0)

#@jit
def comp_loc(theta_0_,phi_0_,):
    return jnp.asarray(theta_0_ + phi_0_*1j)

# Load isothermal coordinates and interpolate on a different grid
def iso_coords_interp(name,_data, sgrid, eq):

    u = jnp.load(name + 'u.npy')
    v = jnp.load(name + 'v.npy')
    Phi = jnp.load(name + 'Phi.npy') #first_derivative_t(u, tdata, tgrid,)
    Psi = jnp.load(name + 'Psi.npy') #first_derivative_z(u, tdata, tgrid,)
    b0 = jnp.load(name + 'b.npy') #first_derivative_z(u, tdata, tgrid,)
    lamb_ratio = jnp.load(name + 'ratio.npy') #first_derivative_z(u, tdata, tgrid,)
    
    _data["du"] = jnp.max(u.flatten()) - jnp.min(u.flatten())
    _data["dv"] = jnp.max(v.flatten()) - jnp.min(v.flatten())
    
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

def shift_grid(theta, zeta, dt, dz, w_surface, name):
    
    l_zeta = zeta - dz/2
    r_zeta = zeta + dz/2
    d_theta = theta - dt/2
    u_theta = theta + dt/2
    
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
    
    return Grid(jnp.stack((jnp.ones_like(theta_flat),
                           theta_flat,zeta_flat)).T)

# Load isothermal coordinates and interpolate on a different grid
def iso_coords_interp(name,_data, sgrid, eq):

    u = jnp.load(name + 'u.npy')
    v = jnp.load(name + 'v.npy')
    Phi = jnp.load(name + 'Phi.npy') #first_derivative_t(u, tdata, tgrid,)
    Psi = jnp.load(name + 'Psi.npy') #first_derivative_z(u, tdata, tgrid,)
    b0 = jnp.load(name + 'b.npy') #first_derivative_z(u, tdata, tgrid,)
    lamb_ratio = jnp.load(name + 'ratio.npy') #first_derivative_z(u, tdata, tgrid,)
    
    _data["du"] = jnp.max(u.flatten()) - jnp.min(u.flatten())
    _data["dv"] = jnp.max(v.flatten()) - jnp.min(v.flatten())
    
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
    
    return Grid(jnp.stack((jnp.ones_like(theta_flat),
                           theta_flat,zeta_flat)).T)