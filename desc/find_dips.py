import numpy as np
import os

from scipy.io import netcdf_file
import copy
import sys
import functools
import pickle

import jax
import jax.numpy as jnp
from jax import jit, jacfwd, lax, vmap

#from netCDF4 import Dataset
import h5py

from desc.backend import put, fori_loop, jnp, sign

from desc.io import InputReader, load

from desc.grid import Grid, LinearGrid
from desc.transform import Transform
from desc.derivatives import Derivative
from desc.backend import fori_loop, jit, jnp, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.utils import flatten_list, cross, dot
from scipy.interpolate import griddata

@jit
def B_tot(w, w2_pol, w1_pol, w2_tor, w1_tor,
          l, l_u, l_v, e_u, e_v, du, dv, d_0,
          Kgrid, x_surf, jac_surf, coords):
    
    #B_w2_pol = biot(w, w2_pol, l, l_u, l_v, e_u, e_v, du, dv, d_0, Kgrid, x_surf, jac_surf, coords)
    #B_w1_pol = biot(w, w1_pol, l, l_u, l_v, e_u, e_v, du, dv, d_0, Kgrid, x_surf, jac_surf, coords)
    #B_w2_tor = biot(w, w2_tor, l, l_u, l_v, e_u, e_v, du, dv, d_0, Kgrid, x_surf, jac_surf, coords)
    #B_w1_tor = biot(w, w1_tor, l, l_u, l_v, e_u, e_v, du, dv, d_0, Kgrid, x_surf, jac_surf, coords)
    
    return jnp.column_stack(((biot(w, w2_pol, l, l_u, l_v, e_u, e_v, du, dv, d_0, Kgrid, x_surf, jac_surf, coords) 
                              - biot(w, w1_pol, l, l_u, l_v, e_u, e_v, du, dv, d_0, Kgrid, x_surf, jac_surf, coords)
                             ).T, 
                             (biot(w, w2_tor, l, l_u, l_v, e_u, e_v, du, dv, d_0, Kgrid, x_surf, jac_surf, coords) 
                              - biot(w, w1_tor, l, l_u, l_v, e_u, e_v, du, dv, d_0, Kgrid, x_surf, jac_surf, coords)
                             ).T
                            )
                           )  # Shape (num_dipoles, 6*M)
    
# Apply Biot-Savart to each of the colums of the output of K_dip
@jit
def biot(w, w1,
         l, l_u, l_v,
         e_u, e_v,
         du, dv,
         #N,
         d_0,
         #Kgrid,# K_at_grid, 
         surface_grid_weights,surface_grid_NFP, surface_grid_nodes,
         x_surf, jac_surf,
         coords):

    K_at_grid = K_omega(w, w1,
                        l, l_u, l_v,
                        e_u, e_v,
                        du, dv,
                        #N,
                          d_0)
    
    # Define a wrapped version of biot_savart_general to fix rs and _dV and vary J:
    def biot_per_source(Ji):
        # Ji has shape (3321, 3) for a single source
        return _compute_magnetic_field_from_Current(surface_grid_weights,surface_grid_NFP, surface_grid_nodes,
                                                    Ji, 
                                                           x_surf, jac_surf,
                                                           coords,
                                                           basis="rpz")

    # vmap over 453 sources axis:
    B_per_source = vmap(biot_per_source)(K_at_grid)  # shape (453, ...)
    
    return B_per_source


@jit
def K_omega(w, w1, l, l_u, l_v, e_u, e_v, du, dv, d_0):
    
    omega = omega_single(w, w1, l, l_u, l_v, du, dv, d_0)
    #jnp.squeeze(omega_single(w, w1, l, l_u, l_v, du, dv, d_0).T)
    # Compute current density
    #return (jnp.imag(omega)[:, :, None] * ( ( l * e_u.T ).T )[None, :, :]
    #        + jnp.real(omega)[:, :, None] * ( ( l * e_u.T ).T )[None, :, :] )
    return  (jnp.imag(omega) * l * e_u.T +  jnp.real(omega) * l * e_v.T ).T
    
@jit
def omega_single(w, w1,
               l,l_u,l_v,
               du, dv,
               #N,
               d_0,
              ):
    
    v_1_num = v1_eval(w, w1, du, dv, 
                      #N, 
                      d_0)
    
    v_1_num_prime = v1_prime_eval(w, w1, du, dv, 
                                  #N, 
                                  d_0)
    
    chi_reg_1 = chi_reg(w, w1,# location of the vortex
                        l, l_u, l_v,
                        d_0,)
    
    gamma = du / ( 2 * jnp.pi )
    
    omega = ( ( v_1_num_prime / v_1_num) 
             - (jnp.real( w1 ) / ( gamma * dv ))
             + 1 / 2 * ( chi_reg_1 )
            )
    
    return omega

@jit
def v1_prime_eval(w, w0, du, dv,# N,
                  d_0):

    gamma = du / ( 2 * jnp.pi )
    p = jnp.exp( - dv / ( 2 * gamma ) )
    
    def fp_seq(n,w,w0):
        return ( ( ( (-1) ** n
                        ) * ( p ** ( n ** 2 + n )
                            )
                        ) * ( ( ( 2 * n + 1 ) / ( 2 * gamma )
                              ) * jnp.cos( ( 2 * n + 1 ) * ( w-w0 ) / ( 2 * gamma )
                                         )
                            )
                )

    #print("f_seq shapes:", getattr(w, 'shape', 'no shape'), getattr(w0, 'shape', 'no shape'))
    fp_vec = jax.vmap(fp_seq, in_axes=(0,None, None))(jnp.arange(20),w,w0)
    _product = jnp.sum(fp_vec, axis = 0)#[None,:,:]

    return jnp.where( jnp.abs( w-w0 ) > d_0, 
                     ( p ** ( 1 / 4 ) / 1 ) * _product, 
                     0 )
@jit  
def v1_eval(w, w0, du, dv,# N, 
            d_0):    
    
    #w0 = comp_loc(u_,v_)
    gamma = du / (2 * jnp.pi)
    p = jnp.exp(-dv / (2 * gamma))

    #w = w[:, None]   # (3321, 1)
    #w0 = w0[None, :] # (1, 453)

    #print("f_seq shapes:", getattr(w, 'shape', 'no shape'), getattr(w0, 'shape', 'no shape'))
    def f_seq(n,w,w0):
        return ((-1)**n) * (p**(n**2 + n)) * jnp.sin((2 * n + 1) * (w-w0) / (2 * gamma))

    f_vec = jax.vmap(f_seq, in_axes=(0,None, None))(jnp.arange(20),w,w0)

    product_ = jnp.sum(f_vec,axis = 0)#[None,:,:]

    return jnp.where(jnp.abs(w-w0) > d_0,
                     2 * p**(1/4) * product_,
                     1.0)

@jit
def chi_reg(w, w0,# location of the vortex
            l, l_u, l_v,
            d_0,):

    #print("f_seq shapes:", getattr(w, 'shape', 'no shape'), getattr(w0, 'shape', 'no shape'))
    # Compute condition mask of shape (N, M)
    cond = jnp.abs(w-w0) < d_0  # (N, M)

    # Broadcast l, l_u, l_v from (N,) to (N, M)
    l_b = l#[:, None]      # (N, 1)
    l_u_b = l_u#[:, None]  # (N, 1)
    l_v_b = l_v#[:, None]  # (N, 1)

    val = -l_u_b / l_b + (l_v_b / l_b) * 1j  # (N, 1), broadcasted to (N, M)
    return jnp.where(cond, val, 0.0)

def data_calc(p_M,p_N,eps,sdata,surface):
#w, w2pol, w1pol, w2tor, w1tor, l, l_u, l_v, e_u, e_v = 
 return data_vectorized(p_M,p_N,eps,sdata,surface)
    
def data_vectorized(p_M, p_N, eps, sdata, surface):

    du_data, dd_data, dr_data, dl_data = iso_loc(p_M,p_N,surface,eps,eps)
    
    n_columns = du_data["theta"].shape[0]
    
    w_vals = jnp.asarray( sdata["u_iso"] + sdata["v_iso"]* 1j )
    #( jnp.tile(sdata["u_iso"].reshape(sdata["theta"].shape[0], 1), n_columns) 
    #          + jnp.tile(sdata["v_iso"].reshape(sdata["theta"].shape[0], 1), n_columns) * 1j ).T
    
    w_vals = jnp.asarray(w_vals)
    w2_pol = jnp.asarray( du_data["u_iso"] + 1j * du_data["v_iso"] )
    w1_pol = jnp.asarray( dd_data["u_iso"] + 1j * dd_data["v_iso"] )
    
    w2_tor = jnp.asarray( dr_data["u_iso"] + 1j * dr_data["v_iso"] )
    w1_tor = jnp.asarray( dl_data["u_iso"] + 1j * dl_data["v_iso"] )
    
    l_vals = jnp.asarray(sdata["lambda_iso"])
    l_u_vals = jnp.asarray(sdata["lambda_u"])
    l_v_vals = jnp.asarray(sdata["lambda_v"])

    def repeat_fn(_,x):
        return x
    
    e_u_vals = jnp.asarray(sdata["e_u"])
    e_v_vals = jnp.asarray(sdata["e_v"])

    return w_vals, w2_pol, w1_pol, w2_tor, w1_tor, l_vals, l_u_vals, l_v_vals, e_u_vals, e_v_vals

# Function to find the isothermal location of the vortices
def iso_loc(p_M, p_N, surface, dt, dz):

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
    

#@jit
def comp_loc(theta_0_,phi_0_,):
    return jnp.asarray(theta_0_ + phi_0_*1j)

#@jit
def _compute_magnetic_field_from_Current(surface_grid_weights,surface_grid_NFP, surface_grid_nodes,
                                         K_at_grid, 
                                         #surface, 
                                         #data,
                                         x_surf,
                                         jac_surf,
                                         #eq,
                                         #Bgrid,
                                         coords,
                                         basis="rpz"):

    #Bdata = eq.compute(["R","phi","Z","n_rho"], grid = Bgrid)
    #coords = jnp.vstack([Bdata["R"],Bdata["phi"],Bdata["Z"]]).T
    
    assert basis.lower() in ["rpz", "xyz"]
    if hasattr(coords, "nodes"):
        coords = coords.nodes
    coords = jnp.atleast_2d(coords)
    if basis == "rpz":
        coords = rpz2xyz(coords)
    else:
        K_at_grid = xyz2rpz_vec(K_at_grid, x=coords[:, 0], y=coords[:, 1])

    coords = jnp.asarray(coords, dtype=jnp.float64)
    _rs = jnp.asarray(x_surf, dtype=jnp.float64)
    _K = jnp.asarray(K_at_grid, dtype=jnp.float64)
    _dV = jnp.asarray(surface_grid_weights * jac_surf / surface_grid_NFP, dtype=jnp.float64)
    surface_grid_nodes = jnp.asarray(surface_grid_nodes, dtype=jnp.float64)
    
    #surface_grid = Kgrid

    # compute and store grid quantities
    # needed for integration
    # TODO: does this have to be xyz, or can it be computed in rpz as well?
    #data = surface.compute(["x", "|e_theta x e_zeta|"], grid=surface_grid, basis="xyz")

    #_rs = xyz2rpz(data["x"])
    _rs = x_surf#xyz2rpz(x_surf)
    _K = K_at_grid

    # surface element, must divide by NFP to remove the NFP multiple on the
    # surface grid weights, as we account for that when doing the for loop
    # over NFP
    #_dV = surface_grid.weights * data["|e_theta x e_zeta|"] / surface_grid.NFP
    _dV = surface_grid_weights * jac_surf / surface_grid_NFP
    
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi = ( surface_grid_nodes[:, 2] + j * 2 * jnp.pi / surface_grid_NFP ) % ( 2 * jnp.pi )
        # new coords are just old R,Z at a new phi (bc of discrete NFP symmetry)
        rs = jnp.vstack( (_rs[:, 0], phi, _rs[:, 2]) ).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_general( coords, rs, K, _dV, )
        f += fj
        return f
    
    B = fori_loop(0, surface_grid_NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])

    return jnp.concatenate((B[:,0], B[:,1], B[:,2]))
    #return B

@jit
def biot_savart_general(re, rs, J, dV):

    #print("rs.shape =", rs.shape)
    #print("J.shape =", J.shape)
    
    #re, rs, J, dV = map(jnp.asarray, (re, rs, J, dV))
    #assert J.shape == rs.shape
    #JdV = J * dV[:, None]
    #B = jnp.zeros_like(re)

    re, rs, J, dV = map(lambda x: jnp.asarray(x, dtype=jnp.float64), (re, rs, J, dV))
    JdV = J * dV[:, None]
    B = jnp.zeros_like(re, dtype=jnp.float64)
    
    def body(i, B):
        r = re - rs[i, :]
        num = jnp.cross(JdV[i, :], r, axis=-1)
        den = jnp.linalg.norm(r, axis=-1) ** 3
        B = B + jnp.where(den[:, None] == 0, 0, num / den[:, None])
        return B

    #return 1e-7 * fori_loop(0, J.shape[0], body, B)
    return 1e-7 * fori_loop(0, rs.shape[0], body, B)

# Load isothermal coordinates and interpolate on a different grid
def iso_coords_interp(name,_data, sgrid, eq):

    u = np.load(name + 'u.npy')
    v = np.load(name + 'v.npy')
    Phi = np.load(name + 'Phi.npy') #first_derivative_t(u, tdata, tgrid,)
    Psi = np.load(name + 'Psi.npy') #first_derivative_z(u, tdata, tgrid,)
    b0 = np.load(name + 'b.npy') #first_derivative_z(u, tdata, tgrid,)
    lamb_ratio = np.load(name + 'ratio.npy') #first_derivative_z(u, tdata, tgrid,)
    
    _data["du"] = jnp.max(u.flatten()) - jnp.min(u.flatten())
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