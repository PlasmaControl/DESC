"""Compute functions for isothermal coordinates and Harmonic Field.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""
from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import dot, surface_integrals_map

import numpy as np
from numpy import ndarray

import os

import copy
import sys
import functools
import pickle

import jax
import jax.numpy as jnpå
from jax import jit, jacfwd

from desc.backend import put, fori_loop, jnp, sign

from desc.grid import ConcentricGrid, LinearGrid, Grid, QuadratureGrid

from desc.transform import Transform

from desc.backend import fori_loop, jit, jnp, odeint, sign

from desc.utils import flatten_list

from .data_index import register_compute_fun

# Find a poloidal harmonic vector on a surface
@register_compute_fun(
    name="H_1",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta","zeta",
          "e^theta_s","e^zeta_s",
          "e^theta_s_t","e^theta_s_z",
          "e^zeta_s_t","e^zeta_s_z",
          "nabla_s^2_theta"],
)

def H1(params, transforms, profiles, data, **kwargs):

    phi = find_phi(data,)
    grad = grad_(phi,data,)
    data["H_1"] = data["e^theta_s"] - grad
    
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["H_1"] = rpz2xyz_vec(data["H_1"], phi=data["phi"])
        
    return data

# Function to find the scalar that cancels the surface divergence
def find_phi(data,):
    
    x = jnp.ones(data["theta"].shape[0])
    fun_wrapped = lambda x: u_div_residual(x,data,)
    A_ = jax.jacfwd(fun_wrapped)(x)
    
    return jnp.linalg.pinv(A_)@data["nabla_s^2_theta"]

def u_div_residual(y,data,):
#def u_div_residual(y,data,):

    f_t = first_derivative_t(y, data)
    f_z = first_derivative_z(y, data,)
    f_tt = first_derivative_t(f_t, data,)
    f_zz = first_derivative_z(f_z, data,)
    f_tz = first_derivative_z(f_t, data,)

    nabla_s_2_f = (jnp.sum(data["e^theta_s"]*data["e^theta_s_t"], axis=-1)*f_t
                   + jnp.sum(data["e^theta_s"]*data["e^theta_s"], axis=-1)*f_tt
                   + jnp.sum(data["e^theta_s"]*data["e^zeta_s_t"], axis=-1)*f_z
                   + jnp.sum(data["e^theta_s"]*data["e^zeta_s"], axis=-1)*f_tz
                   + jnp.sum(data["e^zeta_s"]*data["e^theta_s_z"], axis=-1)*f_t
                   + jnp.sum(data["e^theta_s"]*data["e^zeta_s"], axis=-1)*f_tz
                   + jnp.sum(data["e^zeta_s"]*data["e^zeta_s_z"], axis=-1)*f_z
                   + jnp.sum(data["e^zeta_s"]*data["e^zeta_s"], axis=-1)*f_zz
                  )
    
    return nabla_s_2_f

def grad_(y,data,):
    
    f_t_ = first_derivative_t(y, data,)
    f_z_ = first_derivative_z(y, data,)
    
    return (f_t_*data["e^theta_s"].T  + f_z_*data["e^zeta_s"].T).T


def first_derivative_t(a_mn,data,):
    
    # Expecting square grids
    n_size = int(np.sqrt(data["theta"].shape[0]))
    
    # Rearrange A as a matrix
    A1 = a_mn.reshape((n_size, n_size)).T
    
    # theta-step
    dt = data["theta"][1] - data["theta"][0]
                                     
    # d(sigma)/dt
    A_t = jnp.zeros_like(A1)
    # i = 0
    A_t = A_t.at[0, :].set( (A1[1, :] - A1[n_size-1, :]) * (2 * dt) ** (-1) )
    # i = n_size
    A_t = A_t.at[n_size-1, :].set( (A1[0, :] - A1[n_size-2, :]) * (2 * dt) ** (-1) )
    # Intermediate steps
    A_t = A_t.at[1:n_size - 1, :].set((A1[2:n_size, :] - A1[0:n_size - 2, :]) * (2 * dt) ** (-1))
    
    return (A_t.T).reshape(-1)#.flatten()

def first_derivative_z(a_mn,data,):

    # Expecting square grids
    n_size = int(np.sqrt(data["zeta"].shape[0]))
    
    # Rearrange A as a matrix
    A2 = a_mn.reshape((n_size, n_size)).T
    
    # dz-step
    dz = data["zeta"][n_size] - data["zeta"][0]
    # d(V)/dz
    A_z = jnp.zeros_like(A2)
    # at i = 0
    A_z = A_z.at[:, 0].set( (A2[:, 1] - A2[:, n_size - 1]) * (2 * dz) ** (-1) )
    # at i = n_size
    A_z = A_z.at[:, n_size - 1].set((A2[:, 0] - A2[:, n_size - 2]) * (2 * dz) ** (-1) )
    # Intermediate steps
    A_z = A_z.at[:, 1:n_size - 1].set((A2[:, 2:n_size] - A2[:, 0:n_size - 2]) * (2 * dz) ** (-1) )
    
    return (A_z.T).reshape(-1)#flatten()