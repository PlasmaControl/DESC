import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import os
import pickle
from mpi4py import MPI
import jax.lib.xla_bridge as xb

from scipy.io import netcdf_file
import copy
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.constants import mu_0
import sys
import functools
import pickle

#import jax
#import jax.numpy as jnp
from jax import jit, jacfwd, lax

from netCDF4 import Dataset
import h5py

from desc.backend import put, fori_loop, np, sign

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

from desc.backend import fori_loop, jit, np, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.coils import *

from desc.utils import flatten_list, cross, dot

import time

from numpy import ndarray

from desc.fns_simp import (plot_figure,
                           surf_int,
                           _compute_magnetic_field_from_Current,
                          )

from scipy.interpolate import griddata

# Helper function (unchanged)
def to_jax_dict(d):
    """Convert dictionary values to JAX arrays, ensuring compatibility."""
    return {k: jnp.array(v, copy=True) if isinstance(v, (np.ndarray, list, tuple)) else v
            for k, v in d.items()}

# omega_wrapper_jax (unchanged)
def omega_wrapper_jax(i, y, sdata, du_data, dd_data, dr_data, dl_data, N, d_0, r):
    i_r = i + r
    y_pol = y[i]
    y_tor = y[i_r]
    omega_pol1 = omega_pair(sdata, du_data["u_iso"][i], du_data["v_iso"][i],
                            dd_data["u_iso"][i], dd_data["v_iso"][i], N, d_0)
    omega_tor1 = omega_pair(sdata, dr_data["u_iso"][i], dr_data["v_iso"][i],
                            dl_data["u_iso"][i], dl_data["v_iso"][i], N, d_0)
    real_part = y_pol * jnp.real(omega_pol1) + y_tor * jnp.real(omega_tor1)
    imag_part = y_pol * jnp.imag(omega_pol1) + y_tor * jnp.imag(omega_tor1)
    return real_part, imag_part

# Modified K_dips for multi-node, multi-task CPU execution
def K_dips(p_M, p_N, sdata, sgrid, surface, y, dt, dz, N, d_0, output_dir="output"):
    # Compute theta and zeta (same across all ranks)
    theta = jnp.linspace(2 * jnp.pi * (1 / (p_M * 2 + 1)) * 0.5,
                        2 * jnp.pi * (1 - 1 / (p_M * 2 + 1) * 0.5), p_M * 2 + 1)
    zeta = jnp.linspace(2 * jnp.pi / sgrid.NFP * (1 / (p_N * 2 + 1)) * 0.5,
                       2 * jnp.pi / sgrid.NFP * (1 - 1 / (p_N * 2 + 1) * 0.5), p_N * 2 + 1)

    # Compute grid data (assumed to be lightweight; compute on all ranks)
    dl_data, dr_data, dd_data, du_data = shift_grid(theta, zeta, dt, dz, surface, "iso_coords/")
    
    # Convert to JAX arrays
    dl_data = to_jax_dict(dl_data)
    dr_data = to_jax_dict(dr_data)
    dd_data = to_jax_dict(dd_data)
    du_data = to_jax_dict(du_data)
    sdata = to_jax_dict(sdata)

    # Assert data types
    for key in ["u_iso", "v_iso"]:
        assert isinstance(du_data[key], jnp.ndarray), f"du_data[{key}] is not a JAX array"
        assert isinstance(dd_data[key], jnp.ndarray), f"dd_data[{key}] is not a JAX array"
        assert isinstance(dr_data[key], jnp.ndarray), f"dr_data[{key}] is not a JAX array"
        assert isinstance(dl_data[key], jnp.ndarray), f"dl_data[{key}] is not a JAX array"

    r = dl_data["theta"].shape[0]
    assert (p_M * 2 + 1) * (p_N * 2 + 1) == r, "Dipole location mismatch"

    # Distribute indices across all MPI ranks
    indices = jnp.arange(r)
    indices_per_rank = (r + size - 1) // size
    start = rank * indices_per_rank
    end = min((rank + 1) * indices_per_rank, r)
    indices_chunk = indices[start:end]

    # Safe omega wrapper for JAX
    def safe_omega_wrapper(i, y, sdata, du_data, dd_data, dr_data, dl_data, N, d_0, r):
        def true_fun():
            return omega_wrapper_jax(i, y, sdata, du_data, dd_data, dr_data, dl_data, N, d_0, r)
        def false_fun():
            output_shape = sdata["n_rho"].shape[0]
            return jnp.zeros(output_shape), jnp.zeros(output_shape)
        return jax.lax.cond(i >= 0, true_fun, false_fun)

    # Scan over indices chunk
    def scan_body(carry, i):
        real_sum, imag_sum = carry
        real_part, imag_part = safe_omega_wrapper(i, y, sdata, du_data, dd_data, dr_data, dl_data, N, d_0, r)
        return (real_sum + real_part, imag_sum + imag_part), None

    output_shape = sdata["n_rho"].shape[0]
    init_carry = (jnp.zeros(output_shape), jnp.zeros(output_shape))
    (omega_real, omega_imag), _ = lax.scan(scan_body, init_carry, indices_chunk)

    # Convert to NumPy for MPI communication
    omega_real_np = np.array(omega_real)
    omega_imag_np = np.array(omega_imag)

    # Aggregate results across nodes using MPI Allreduce
    omega_real_all = np.zeros(output_shape, dtype=np.float64)
    omega_imag_all = np.zeros(output_shape, dtype=np.float64)
    comm.Allreduce(omega_real_np, omega_real_all, op=MPI.SUM)
    comm.Allreduce(omega_imag_np, omega_imag_all, op=MPI.SUM)

    # Compute final result (on all ranks, but could be restricted to rank 0 if needed)
    result = ((sdata["lambda_iso"] ** (-1)) * (
        omega_imag_all * -cross(sdata["n_rho"], sdata["e_u"]).T
        + omega_real_all * -cross(sdata["n_rho"], sdata["e_v"]).T
    )).T

    # Optionally save partial results for debugging
    os.makedirs(output_dir, exist_ok=True)
    partial_file = os.path.join(output_dir, f"partial_{rank}.pkl")
    with open(partial_file, 'wb') as f:
        pickle.dump({'omega_real': omega_real_np, 'omega_imag': omega_imag_np}, f)

    return result

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


    return _compute_magnetic_field_from_Current(Kgrid,
                                         K_dips(p_M, p_N,
                                                       sdata, sgrid, surface, #w_surface,
                                                       y,
                                                       dt,dz, N, d_0), 
                                         #surface, 
                                         #data,
                                         x_surf,
                                         jac_surf,
                                         #eq,
                                         #Bgrid,
                                         coords,
                                         basis="rpz")

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
    pls_points = eq_surf.compute(["x"], grid = Bgrid)["x"]
    
    r = dl_data["theta"].shape[0]

    def body_fun_single(i, r, y, pls_points, sgrid, du_data, dd_data, dr_data, dl_data):
        i_r = i + r
    
        y_pol = lax.dynamic_index_in_dim(y, i, axis=0)
        y_tor = lax.dynamic_index_in_dim(y, i_r, axis=0)
    
        b_stick_pol = stick(du_data["x"][i], dd_data["x"][i], pls_points, sgrid, basis="rpz")
        b_stick_tor = stick(dr_data["x"][i], dl_data["x"][i], pls_points, sgrid, basis="rpz")
    
        return y_pol * b_stick_pol + y_tor * b_stick_tor

    # Wrap for multiprocessing (must take a single argument)
    def wrapper(args):
        return body_fun_single(*args)
    
    def parallel_compute(r, y, pls_points, sgrid, du_data, dd_data, dr_data, dl_data, num_workers=None):
        if num_workers is None:
            num_workers = min(cpu_count(), r)
    
        # Prepackage args
        task_args = [(i, r, y, pls_points, sgrid, du_data, dd_data, dr_data, dl_data) for i in range(r)]
    
        with Pool(num_workers) as pool:
            results = pool.map(wrapper, task_args)
    
        # Sum all partial results
        return jnp.sum(jnp.stack(results), axis=0)
    
    
    # Call the function like this:
    b_sticks_total = parallel_compute(
        r=r,
        y=y,
        pls_points=pls_points,
        sgrid=sgrid,
        du_data=du_data,
        dd_data=dd_data,
        dr_data=dr_data,
        dl_data=dl_data,
    )
    return b_sticks_total

def bn_res(p_M, p_N, sdata, sgrid, surface, #w_surface, 
           y, dt, dz, N, d_0, eq, Bgrid):
    
    B0_dips = B_dips(p_M, p_N, sdata, sgrid, surface, #w_surface, 
                y, dt,dz, N, d_0, eq, Bgrid)

    #B0_sticks = B_sticks(p_M, p_N,
    #                     sdata,
    #                     sgrid,
    #                     surface,
    #                     y,
    #                     dt,dz,
    #                     eq,
    #                     Bgrid,)

    # Minus sign for sticks to change the polarity of the current in the sticks
    B0 = B0_dips #+ B0_sticks
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

def stick(p2, # second point of the stick
          p1, # first point of the stick
          plasma_points, # points on the plasma surface
          Kgrid,
          basis = "rpz",
          ):

    surface_grid = Kgrid
    p2_ = xyz2rpz(p2)
    p1_ = xyz2rpz(p1)

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
        
        b_s = p1s - plasma_points
        
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

def _compute_magnetic_field_from_Current(Kgrid,
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
    
    surface_grid = Kgrid

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
    _dV = surface_grid.weights * jac_surf / surface_grid.NFP
    
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi = ( surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )
        # new coords are just old R,Z at a new phi (bc of discrete NFP symmetry)
        rs = jnp.vstack( (_rs[:, 0], phi, _rs[:, 2]) ).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_general( coords, rs, K, _dV, )
        f += fj
        return f
    
    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])

    return jnp.concatenate((B[:,0], B[:,1], B[:,2]))
    #return B