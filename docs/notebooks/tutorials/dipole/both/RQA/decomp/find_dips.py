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

from interpax import interp2d

def K_decomp(p_M, p_N,
           sdata1,
           #sdata2,
           #sdata3,
           sgrid,
           surface,
           #y,
           dt, dz,
           N,
           d_0):

    theta = jnp.linspace(2 * jnp.pi * (1 / (p_M * 2 + 1)) * 1/2,
                         2 * jnp.pi * (1 - 1 / (p_M * 2 + 1) * 1/2),
                         p_M * 2 + 1)

    zeta = jnp.linspace(2 * jnp.pi / surface.NFP * (1 / (p_N * 2 + 1)) * 1/2,
                        2 * jnp.pi / surface.NFP * (1 - 1 / (p_N * 2 + 1) * 1/2),
                        p_N * 2 + 1)

    
    name = "iso_coords/"
    dl_data, dr_data, dd_data, du_data = shift_grid(theta, zeta, dt, dz, surface, name)
    
    assert (p_M * 2+1)*(p_N * 2+1) == dl_data["theta"].shape[0] , "Check that the number of dipole locations coincide with the number of dipoles"
    r = dl_data["theta"].shape[0]  # Make r a Python int for indexing

    x_sol = jnp.zeros(r)
    for i in range(0,r):

        omega_pol1 = omega_pair(
            sdata1,
            du_data["u_iso"][i], du_data["v_iso"][i],
            dd_data["u_iso"][i], dd_data["v_iso"][i],
            N, d_0
        )

        omega_tor1 = omega_pair(
            sdata1,
            dr_data["u_iso"][i], dr_data["v_iso"][i],
            dl_data["u_iso"][i], dl_data["v_iso"][i],
            N, d_0
        )

        A = jnp.real(omega_pol1)
        B = jnp.real(omega_tor1)
        C = jnp.imag(omega_pol1)
        D = jnp.imag(omega_tor1)

        x_sol = x_sol.at[i].set( ( A * D - B * C) ** (-1) * sdata1['lambda_iso'] ** (-1) * ( D * sdata['K.e_u'] - B * sdata['K.e_v']) )
        x_sol =  x_sol.at[i + r].set( (A * D - B * C) ** (-1) * sdata1['lambda_iso'] ** (-1) * ( - C * sdata['K.e_u'] - A * sdata['K.e_v']) )

    return x_sol

def bn_res(p_M, p_N, sdata1,
           sdata2,
           sdata3,
           sgrid, surface, #w_surface, 
           y, dt, dz, N, d_0, eq, Bgrid):
    
    B0_dips = B_dips(p_M, p_N, 
                     sdata1,
                     sdata2,
                     sdata3,
                     sgrid, surface, #w_surface, 
                y, dt,dz, N, d_0, eq, Bgrid)

    B0_sticks = B_sticks(p_M, p_N,
                         sdata1,
                         sgrid,
                         surface,
                         y,
                         dt,dz,
                         eq,
                         Bgrid,)

    # Minus sign for sticks to change the polarity of the current in the sticks
    B0 = (B0_dips
          - B0_sticks)
    return jnp.concatenate((B0[:,0],B0[:,1],B0[:,2]))
    
def B_dips(p_M, p_N,
           sdata1,
           sdata2,
           sdata3,
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
                                                       sdata1,
                                                       sdata2,
                                                       sdata3,
                                                       sgrid, surface, #w_surface,
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
    
    du_data["u_iso"] = jnp.asarray(du_data["u_iso"])
    dd_data["u_iso"] = jnp.asarray(dd_data["u_iso"])
    du_data["v_iso"] = jnp.asarray(du_data["v_iso"])
    dd_data["v_iso"] = jnp.asarray(dd_data["v_iso"])
    dr_data["u_iso"] = jnp.asarray(dr_data["u_iso"])
    dr_data["v_iso"] = jnp.asarray(dr_data["v_iso"])
    dl_data["u_iso"] = jnp.asarray(dl_data["u_iso"])
    dl_data["v_iso"] = jnp.asarray(dl_data["v_iso"])
    
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

def K_dips(p_M, p_N,
           sdata1,
           sdata2,
           sdata3,
           sgrid,
           surface,
           y,
           dt, dz,
           N,
           d_0):

    theta = jnp.linspace(2 * jnp.pi * (1 / (p_M * 2 + 1)) * 1/2,
                         2 * jnp.pi * (1 - 1 / (p_M * 2 + 1) * 1/2),
                         p_M * 2 + 1)

    zeta = jnp.linspace(2 * jnp.pi / surface.NFP * (1 / (p_N * 2 + 1)) * 1/2,
                        2 * jnp.pi / surface.NFP * (1 - 1 / (p_N * 2 + 1) * 1/2),
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

    def body_fun1(i, carry):
        omega_total_real, omega_total_imag = carry
        i_r = i + r

        y_pol = lax.dynamic_index_in_dim(y, i, axis=0)
        y_tor = lax.dynamic_index_in_dim(y, i_r, axis=0)

        omega_pol1 = omega_pair(
            sdata1,
            du_data["u_iso"][i], du_data["v_iso"][i],
            dd_data["u_iso"][i], dd_data["v_iso"][i],
            N, d_0
        )

        omega_tor1 = omega_pair(
            sdata1,
            dr_data["u_iso"][i], dr_data["v_iso"][i],
            dl_data["u_iso"][i], dl_data["v_iso"][i],
            N, d_0
        )

        omega_total_real += y_pol * jnp.real(omega_pol1) + y_tor * jnp.real(omega_tor1)
        omega_total_imag += y_pol * jnp.imag(omega_pol1) + y_tor * jnp.imag(omega_tor1)
        return omega_total_real, omega_total_imag

    def body_fun2(i, carry):
        omega_total_real, omega_total_imag = carry
        i_r = i + r

        y_pol = lax.dynamic_index_in_dim(y, i, axis=0)
        y_tor = lax.dynamic_index_in_dim(y, i_r, axis=0)

        omega_pol2 = omega_pair(
            sdata2,
            du_data["u_iso"][i], du_data["v_iso"][i],
            dd_data["u_iso"][i], dd_data["v_iso"][i],
            N, d_0
        )

        omega_tor2 = omega_pair(
            sdata2,
            dr_data["u_iso"][i], dr_data["v_iso"][i],
            dl_data["u_iso"][i], dl_data["v_iso"][i],
            N, d_0
        )

        omega_total_real += y_pol * jnp.real(omega_pol2) + y_tor * jnp.real(omega_tor2)
        omega_total_imag += y_pol * jnp.imag(omega_pol2) + y_tor * jnp.imag(omega_tor2)
        return omega_total_real, omega_total_imag

    def body_fun3(i, carry):
        omega_total_real, omega_total_imag = carry
        i_r = i + r

        y_pol = lax.dynamic_index_in_dim(y, i, axis=0)
        y_tor = lax.dynamic_index_in_dim(y, i_r, axis=0)

        omega_pol3 = omega_pair(
            sdata3,
            du_data["u_iso"][i], du_data["v_iso"][i],
            dd_data["u_iso"][i], dd_data["v_iso"][i],
            N, d_0
        )

        omega_tor3 = omega_pair(
            sdata3,
            dr_data["u_iso"][i], dr_data["v_iso"][i],
            dl_data["u_iso"][i], dl_data["v_iso"][i],
            N, d_0
        )

        omega_total_real += y_pol * jnp.real(omega_pol3) + y_tor * jnp.real(omega_tor3)
        omega_total_imag += y_pol * jnp.imag(omega_pol3) + y_tor * jnp.imag(omega_tor3)
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

    #return ( ( ( sdata["lambda_iso"] ** (-1) )*( omega_total_imag * sdata["e_u"].T #- cross(sdata["n_rho"],sdata["e_u"]).T 
    #                                            + omega_total_real * sdata["e_v"].T#- cross(sdata["n_rho"],sdata["e_v"]).T
    #                                         )
    #         ).T
    #       )

    return ( ( ( sdata1["lambda_iso"] ** (-1) ) * ( omega_total_imag1 * cross(sdata1["n_rho"],sdata1["e_u"]).T
                                                   + omega_total_real1 * cross(sdata1["n_rho"],sdata1["e_v"]).T
                                                    + omega_total_imag2 * cross(sdata2["n_rho"],sdata2["e_u"]).T 
                                                    + omega_total_real2 * cross(sdata2["n_rho"],sdata2["e_v"]).T
                                                    + omega_total_imag3 * cross(sdata3["n_rho"],sdata3["e_u"]).T 
                                                    + omega_total_real3 * cross(sdata3["n_rho"],sdata3["e_v"]).T
                                                 )
              ).T
            )#.flatten()

def f_pair(data_or,
           u1_, v1_, # first dipole
           u2_, v2_, # second dipole
           N, d_0):
    
    # Evaluate the dipoles on the grid of vortices
    w1 = comp_loc(u1_, v1_,)
    w2 = comp_loc(u2_, v2_,)
    
    v_1_num = v1_eval(w1, N, d_0, data_or)
    v_1_den = v1_eval(w2, N, d_0, data_or)
    
    f1_reg = f_reg(w1, d_0, data_or)
    f2_reg = f_reg(w2, d_0, data_or)

    return ( jnp.log( v_1_num ) - jnp.log( v_1_den )
            - 2 * jnp.pi * jnp.real( w1 - w2 ) / ( data_or["omega_1"] ** 2 * data_or["tau_2"] ) * data_or["w"] 
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
    
    omega = ( ( v_1_num_prime / v_1_num - v_1_den_prime / v_1_den ) # Regularized near the vortex cores
             - 2 * jnp.pi * jnp.real( w1 - w2 ) / ( data_or["omega_1"] ** 2 * data_or["tau_2"]) 
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
    
    gamma = data_or["omega_1"] / jnp.pi
    p = data_or["tau"]
    
    product_ = 0

    for n in range(0,N):

        product_ = ( product_ 
                    + ( ( ( (-1) ** n
                        )*( p ** ( n **2 + n )
                          )
                       ) * jnp.sin( ( 2 * n + 1 ) * ( data_or["w"] - w0 ) / gamma )
                      )
                   )
    
    return jnp.where(jnp.abs(data_or["w"] - w0) > d_0,
                     2 * p **(1/4)*product_ ,
                     1 # Arbitraty value of 1 inside the circle around the vortex core
                    )

#@jax.jit
def chi_reg(w0,# location of the vortex
            d_0, data_or):
    
    return jnp.where( jnp.abs( data_or["w"] - w0 ) < d_0,
                     - ( data_or["lambda_u"] / data_or["lambda_iso"]) + ( data_or["lambda_v"] / data_or["lambda_iso"] ) * 1j,
                     0)

#@jax.jit

def f_reg(w0,# location of the vortex
          d_0, data_or):
    
    return jnp.where(jnp.abs(data_or["w"] - w0) < d_0,
                     jnp.log(data_or["lambda_iso"]),
                     0)


def v1_prime_eval(w0, N, d_0, data_or):

    gamma = data_or["omega_1"] / jnp.pi
    p = data_or["tau"]
    
    _product = 0
    for n in range(0,N):

        _product = ( _product 
                    + ( ( ( (-1) ** n
                        ) * ( p ** ( n ** 2 + n )
                            )
                        ) * ( ( ( 2 * n + 1 ) / gamma
                              ) * jnp.cos( ( 2 * n + 1 ) * ( data_or["w"] - w0 ) / gamma
                                         )
                            )
                      )
                   )
    
    return jnp.where( abs( data_or["w"] - w0 ) > d_0, 
                     ( p ** ( 1 / 4 ) / 1 ) * _product, 
                     0 )

def comp_loc(theta_0_,phi_0_,):
    
    return theta_0_ + phi_0_*1j

# Load isothermal coordinates and interpolate on a different grid
# Load isothermal coordinates and interpolate on a different grid
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
    _data["tau_1"] = jnp.load(name + 'tau_1.npy')
    _data["tau_2"] = jnp.load(name + 'tau_2.npy')
    
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
    v_mod = lamb_ratio * ( theta_mod - add_extra_periodic(Psi, n_size,m_size) + b0 * u_mod )
    u_t_mod = add_extra_periodic(u_t, n_size,m_size)
    u_z_mod = add_extra_periodic(u_z, n_size,m_size)
    v_t_mod = add_extra_periodic(v_t, n_size,m_size)
    v_z_mod = add_extra_periodic(v_z, n_size,m_size)
    lamb_u_mod = add_extra_periodic(lamb_u, n_size,m_size)
    lamb_v_mod = add_extra_periodic(lamb_v, n_size,m_size)
    
    # Interpolate on theta_mod, zeta_mod
    points = jnp.array( (zeta_mod.flatten(), theta_mod.flatten()) ).T
    
    # Interpolate isothermal coordinates
    _data["u_iso"] = interp2d(_data['theta'], _data['zeta'], theta_mod[:,0], zeta_mod[0,:], u_mod, method="cubic")
    _data["v_iso"] = interp2d(_data['theta'], _data['zeta'], theta_mod[:,0], zeta_mod[0,:], v_mod, method="cubic")
    
    _data["lambda_u"] = interp2d(_data['theta'], _data['zeta'], theta_mod[:,0], zeta_mod[0,:], lamb_u_mod, method="cubic")
    _data["lambda_v"] = interp2d(_data['theta'], _data['zeta'], theta_mod[:,0], zeta_mod[0,:], lamb_v_mod, method="cubic")

    # Interpolate derivatives of isothermal coordinates
    u0_t = interp2d(_data['theta'], _data['zeta'], theta_mod[:,0], zeta_mod[0,:], u_t_mod, method="cubic")
    u0_z = interp2d(_data['theta'], _data['zeta'], theta_mod[:,0], zeta_mod[0,:], u_z_mod, method="cubic")
    v0_t = interp2d(_data['theta'], _data['zeta'], theta_mod[:,0], zeta_mod[0,:], v_t_mod, method="cubic")
    v0_z = interp2d(_data['theta'], _data['zeta'], theta_mod[:,0], zeta_mod[0,:], v_z_mod, method="cubic")
    
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
                                "x",
                               ], grid = l_grid)
    r_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                                "x",
                               ], grid = r_grid)
    d_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                                "x",
                               ], grid = d_grid)
    u_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                                "x",
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