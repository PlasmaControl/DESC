import numpy as np
import os

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

from desc.backend import fori_loop, jit, jnp
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.utils import flatten_list, cross, dot

from numpy import ndarray

from scipy.interpolate import griddata

def G_w_pair(data_or,
               u1_, v1_, # first dipole: upper or right pole
               u2_, v2_, # second dipole: lower or left pole
               N,
               d_0,
              ):
    
    return G_w_regularized(data_or, u1_, v1_, N, d_0) - G_w_regularized(data_or, u2_, v2_, N, d_0)

def G_w_regularized(data_or, u1_, v1_, N, d_0):

    " Note: this definition of the Omega field is incomplete. The correct way to evaluate it is with neutral configurations Sum(q_n) = 0."
    "For our purposes, we are just evaluating it this way."
    w_n = comp_loc(u1_, v1_,)
    return ( - F_w_regularized(data_or, u1_, v1_,  N, d_0) / ( 2 * jnp.pi ) 
            - ( jnp.pi * jnp.imag( w_n[None,:] ) / data_or['omega_1'] ) / ( 2 * 1j * data_or['omega_1'] * jnp.pi * jnp.imag(data_or['tau'])
                                                                          ) # Additional term due to quadratic term from definition of Green's 
                                                                            # function
           )# / data_or["lambda_iso"][:,None]

def F_w_regularized(data_or, u1_, v1_, N, d_0):
    
    w_n = comp_loc(u1_, v1_,)

    return jnp.where(jnp.abs( data_or["w"][:,None] - w_n[None,:] ) > d_0,
                     Jacobi_theta_prime(w_n, N, d_0, data_or) / Jacobi_theta(w_n, N,  d_0, data_or),
                     1 / 2 * ( ( - data_or["lambda_u"]  + data_or["lambda_v"] * 1j ) / data_or["lambda_iso"] )[:,None]
                    )

def Jacobi_theta_prime(w0, N, d_0, data_or):

    p = jnp.exp(1j * jnp.pi * data_or["tau"])
    zeta = ( data_or["w"][:,None] - w0[None,:] ) / ( data_or["omega_1"] / jnp.pi )
    
    def body_fun(n, carry_):
        
        carry_ += (-1) ** n * p ** ( ( n + 1/2 ) ** 2 )  * ( 2 * n + 1 ) * jnp.pi / data_or["omega_1"] * jnp.cos( ( 2 * n + 1 ) * zeta )
        return carry_

    return 2 * fori_loop( 0, N, body_fun, 
                         jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) ) + jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) ) * 1j
                         )
    
def Jacobi_theta(w0, N, d_0, data_or):
    
    p = jnp.exp(1j * jnp.pi * data_or["tau"] )
    zeta = ( data_or["w"][:,None] - w0[None,:] ) / ( data_or["omega_1"] / jnp.pi )
    
    def body_fun2(n, carry):
        
        carry += ( (-1) ** n ) * ( p ** ( ( n + 1/2 ) ** 2 ) ) * jnp.sin( ( 2 * n + 1 ) * zeta )
        return carry
    
    return 2 * fori_loop( 0, N, body_fun2, 
                          jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) ) 
                          + jnp.zeros( (data_or["w"].shape[0], w0.shape[0]) ) * 1j
                          )

def G_Greens_Function_regularized(w0,N,d_0, data_or):

    "w0: location of the pole"
    return ( - jnp.real ( F_Complex_Potential_regularized(w0,N,d_0, data_or) ) / (2 * jnp.pi)
            + ( jnp.imag( (data_or["w"][:,None] - w0[None,:]) / (data_or["omega_1"] / jnp.pi) ) ) **(2) / ( 2 * jnp.pi ** 2 * jnp.imag(data_or['tau']) ) 
           ) 
    
def F_Complex_Potential_regularized(w0,N,d_0, data_or):

    "w0: location of the pole"
    return jnp.where(jnp.abs(data_or["w"][:,None] - w0[None,:]) > d_0,
                     jnp.log( Jacobi_theta(w0, N, d_0, data_or) ), # Logarithm term away from the pole
                     1 / 2 * jnp.log(data_or["lambda_iso"])[:,None] # Regularization close to the pole
                    )
    
def comp_loc(theta_0_,phi_0_,):
    return theta_0_ + phi_0_*1j

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

def add_extra_coords(data_, n_size,m_size, grid,ind):

    _mod = data_.reshape( (n_size, m_size) ).T
    #_mod = jnp.vstack( [ _mod, _mod[0:m_size,0] ] )

    # theta
    if ind == 0:
        _mod = jnp.column_stack( [_mod, _mod[:,0]] )
        _mod = jnp.vstack( [ _mod, 2 * jnp.pi * jnp.ones( _mod.shape[1] ) ] )

    # zeta
    if ind == 1:
        _mod = jnp.column_stack( [_mod, 2 * jnp.pi / grid.NFP * jnp.ones_like(_mod[:,0])] )
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

def stick(p2_,  # second point of the stick
          p1_,  # first point of the stick
          plasma_points,  # points on the plasma surface
          surface_grid,  # Kgrid,
          basis="rpz"
         ):
    
    """Computes the magnetic field on the plasma surface due to a unit current on the source wires.
    
        p2_: numpy.ndarray of dimension (N, 3)
        p1_: numpy.ndarray of dimension (N, 3)
        plasma_point: numpy.ndarray of dimension (M, 3)
    
    """
    
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi1 = (p1_[:, 1] + j * 2 * jnp.pi / surface_grid.NFP) % (2 * jnp.pi)
        phi2 = (p2_[:, 1] + j * 2 * jnp.pi / surface_grid.NFP) % (2 * jnp.pi)

        # TODO: Make sure p2s has the shape (N, 3)
        p1s = jnp.stack([p1_[:, 0], phi1, p1_[:, 2]], axis=1)
        p2s = jnp.stack([p2_[:, 0], phi2, p2_[:, 2]], axis=1)
        
        p1s = rpz2xyz(p1s)
        p2s = rpz2xyz(p2s)

        # a_s.shape = b_s.shape = c_s.shape = (N, M, 3)
        a_s = p2s[:, None, :] - p1s[:, None, :]
        b_s = p1s[:, None, :] - plasma_points[None, :, :]
        c_s = p2s[:, None, :] - plasma_points[None, :, :]

        # if c_s and a_s are (N, 3), will work fine
        c_sxa_s = cross(c_s, a_s)

        f += (
            1e-7
            * (
                (
                    jnp.clip(jnp.sum(c_sxa_s * c_sxa_s, axis=2), a_min=1e-8, a_max=None)
                    * jnp.sum(c_s * c_s, axis=2) ** (1 / 2)
                )
                ** (-1)
                * (jnp.sum(a_s * c_s, axis=2) - jnp.sum(a_s * b_s, axis=2)) )[:, :, None]
                * c_sxa_s
        ) # (N, M, 3)
        
        return f

    b_stick = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros((p1_.shape[0], plasma_points.shape[0], plasma_points.shape[1])))

    if basis == "rpz":
        b_stick = xyz2rpz_vec(b_stick, x=plasma_points[:, 0], y=plasma_points[:, 1])

    return b_stick
    
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

def iso_coords_interp(name,_data, sgrid, eq):

    u = jnp.load(name + 'u.npy')
    v = jnp.load(name + 'v.npy')
    Phi = jnp.load(name + 'Phi.npy') #first_derivative_t(u, tdata, tgrid,)
    Psi = jnp.load(name + 'Psi.npy') #first_derivative_z(u, tdata, tgrid,)
    b0 = jnp.load(name + 'b.npy') #first_derivative_z(u, tdata, tgrid,)
    lamb_ratio = jnp.load(name + 'ratio.npy') #first_derivative_z(u, tdata, tgrid,)
    
    _data["omega_1"] = jnp.load(name + 'omega_1.npy')
    _data["omega_2"] = jnp.load(name + 'omega_2.npy')

    _data["tau"] = jnp.load(name + 'tau.npy')
    
    lamb_u = jnp.load(name + 'lambda_u.npy')
    lamb_v = jnp.load(name + 'lambda_v.npy')

    # Temporary grid
    tgrid = LinearGrid(M = 60, N = 60, NFP = sgrid.NFP)

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
    theta_mod = add_extra_coords(tdata["theta"], n_size,m_size,tgrid,0)
    zeta_mod = add_extra_coords(tdata["zeta"], n_size,m_size,tgrid,1)
    v_mod = theta_mod - jnp.pi - add_extra_periodic(Psi, n_size,m_size)
    #lamb_ratio * ( theta_mod - add_extra_periodic(Psi, n_size,m_size) + b0 * u_mod )
    u_mod = lamb_ratio * (zeta_mod - add_extra_periodic(Phi, n_size,m_size) - jnp.pi + b0 * v_mod)
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
    _data["Area"] = jnp.abs( jnp.imag( jnp.conjugate(_data["omega_1"]) * _data["omega_2"]) )
    _data['b_iso'] = b0
    _data['lambda_ratio'] = lamb_ratio
    
    return _data

def segment_array(array_size, num_segments):
    """
    Divide an array of given size into specified number of segments.
    Returns a list of tuples containing (start_index, end_index) for each segment.
    """
    if array_size < 0 or num_segments <= 0:
        return []
    
    segment_size = array_size // num_segments
    remainder = array_size % num_segments
    segments = []
    current_index = 0
    
    for i in range(num_segments):
        # Calculate segment length, distributing remainder across initial segments
        current_segment_size = segment_size + (1 if i < remainder else 0)
        if current_segment_size > 0:  # Only add non-empty segments
            segments.append((current_index, current_index + current_segment_size - 1))
            current_index += current_segment_size
            
    return segments

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
    l_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",'x'], grid = l_grid)
    r_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",'x'], grid = r_grid)
    d_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",'x'], grid = d_grid)
    u_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",'x'], grid = u_grid)
    
    return (iso_coords_interp(name, l_data, l_grid, w_surface),
            iso_coords_interp(name, r_data, r_grid, w_surface),
            iso_coords_interp(name, d_data, d_grid, w_surface),
            iso_coords_interp(name, u_data, u_grid, w_surface)
           )