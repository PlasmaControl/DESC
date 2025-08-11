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
#from .utils import surface_integrals_map

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

from desc.utils import flatten_list, cross, dot

from .data_index import register_compute_fun

from desc.derivatives import Derivative

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="v_iso_tt",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "u_iso_tt",
          "V_iso_tt",
          "b_iso",
          "lambda_ratio"
         ],
)

def v_iso_tt(params, transforms, profiles, data, **kwargs):
    
    data["v_iso_tt"] = data["lambda_ratio"] * ( data["u_iso_tt"] + data["b_iso"] * data["V_iso_tt"] ) 
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="v_iso_tz",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "u_iso_tz",
          "V_iso_tz",
          "b_iso",
          "lambda_ratio"
         ],
)

def v_iso_tt(params, transforms, profiles, data, **kwargs):
    
    data["v_iso_tz"] = data["lambda_ratio"] * ( data["u_iso_tz"] + data["b_iso"] * data["V_iso_tz"] ) 
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="v_iso_zz",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "u_iso_zz",
          "V_iso_zz",
          "b_iso",
          "lambda_ratio"
         ],
)

def v_iso_zz(params, transforms, profiles, data, **kwargs):
    
    data["v_iso_zz"] = data["lambda_ratio"] * ( data["u_iso_zz"] + data["b_iso"] * data["V_iso_zz"] ) 
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="v_iso_t",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "u_iso","u_iso_t",
          "V_iso","V_iso_t",
          "b_iso",
          "lambda_ratio"
         ],
)

def v_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["v_iso_t"] = data["lambda_ratio"] * ( data["u_iso_t"] + data["b_iso"] * data["V_iso_t"] ) 
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="v_iso_z",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "u_iso","u_iso_z",
          "V_iso","V_iso_z",
          "b_iso",
          "lambda_ratio"
         ],
)

def v_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["v_iso_z"] = data["lambda_ratio"] * ( data["u_iso_z"] + data["b_iso"] * data["V_iso_z"] ) 
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="v_iso",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "u_iso",
          "V_iso","b_iso",
          "lambda_ratio"
         ],
)

def v_iso(params, transforms, profiles, data, **kwargs):
    
    data["v_iso"] = data["lambda_ratio"] * ( data["u_iso"] + data["b_iso"] * data["V_iso"] ) 
                             
    return data

@register_compute_fun(
    name="u_iso_tt",
    label="\\u_{iso,t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal derivative of toroidal isothermal coordinate ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","u_iso_t"],
)

def u_iso_tt(params, transforms, profiles, data, **kwargs):
    
    data["u_iso_tt"] = - first_derivative_t(data["u_iso_t"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="u_iso_tz",
    label="\\u_{iso,tz}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal derivative of toroidal isothermal coordinate ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","u_iso_t"],
)

def u_iso_tz(params, transforms, profiles, data, **kwargs):
    
    data["u_iso_tz"] = - first_derivative_z(data["u_iso_t"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="u_iso_zz",
    label="\\u_{iso,zz}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal derivative of toroidal isothermal coordinate ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","u_iso_z"],
)

def u_iso_zz(params, transforms, profiles, data, **kwargs):
    
    data["u_iso_zz"] = - first_derivative_z(data["u_iso_z"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="u_iso_t",
    label="\\u_{iso,t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal derivative of toroidal isothermal coordinate ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","phi_iso"],
)

def u_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["u_iso_t"] = - first_derivative_t(data["phi_iso"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="u_iso_z",
    label="\\u_{iso,t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal derivative of toroidal isothermal coordinate ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","phi_iso"],
)

def u_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["u_iso_z"] = 1 - first_derivative_z(data["phi_iso"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="u_iso",
    label="\\u_{iso}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "phi_iso"],
)

def u_iso(params, transforms, profiles, data, **kwargs):
    
    data["u_iso"] = data["zeta"] - data["phi_iso"]
    
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="V_iso_tt",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "Psi_iso_tt",
         ],
)

def V_iso_tt(params, transforms, profiles, data, **kwargs):
    
    data["V_iso_tt"] = - data["Psi_iso_tt"]
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="V_iso_tz",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "Psi_iso_tz",
         ],
)

def V_iso_tz(params, transforms, profiles, data, **kwargs):
    
    data["V_iso_tz"] = - data["Psi_iso_tz"]
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="V_iso_zz",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "Psi_iso_zz",
         ],
)

def V_iso_zz(params, transforms, profiles, data, **kwargs):
    
    data["V_iso_zz"] = - data["Psi_iso_zz"]
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="V_iso_t",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "Psi_iso_t",
         ],
)

def V_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["V_iso_t"] = 1 - data["Psi_iso_t"]
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="V_iso_z",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "Psi_iso_z",
         ],
)

def V_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["V_iso_z"] = - data["Psi_iso_z"]
                             
    return data


# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="V_iso",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "u_iso",
          "Psi_iso",
          "H_1","H_2",
         ],
)

def V_iso(params, transforms, profiles, data, **kwargs):
    
    data["V_iso"] = data["theta"] - data["Psi_iso"]
                             
    return data

# Find a poloidal harmonic vector on a surface
@register_compute_fun(
    name="H_2",
    label="\\mathrm{H}_2",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta", "zeta",
          "e^theta_s", "e^zeta_s",
          "psi_iso",
          "H_1", "lambda_ratio",
          "nabla_s_V_iso"
         ],
)

def H2(params, transforms, profiles, data, **kwargs):
    
    # Normalize H_2 to match same magnitude of H_1 (?)
    data["H_2"] = data["lambda_ratio"] * ( data["H_1"] + data["b_iso"] * data["nabla_s_V_iso"] )
    
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["H_2"] = rpz2xyz_vec(data["H_2"], phi=data["phi"])
        
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="lambda_ratio",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "e^theta_s","e^zeta_s",
          "H_1", "nabla_s_V_iso",
          "b_iso",
         ],
)

def lambda_ratio(params, transforms, profiles, data, **kwargs):
    
    lambda_temp = data["H_1"] + data["b_iso"] * data["nabla_s_V_iso"]
    
    data["lambda_ratio"] = jnp.mean( jnp.sqrt( jnp.sum( data["H_1"] * data["H_1"] , axis=-1 
                                                      ) / jnp.sum( lambda_temp * lambda_temp , axis=-1 )
                                             )
                                   )
                             
    return data

@register_compute_fun(
    name="psi_iso",
    label="\\psi_{iso}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "Psi_iso","b_iso",
         ],
)

def psi_iso(params, transforms, profiles, data, **kwargs):
    
    data["psi_iso"] = data["b_iso"] * data["Psi_iso"]
    
    return data

# Find a poloidal harmonic vector on a surface
@register_compute_fun(
    name="b_iso",
    label="\\mathrm{H}_2",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "e^theta_s","e^zeta_s",
          "H_1",
          "nabla_s_V_iso",
         ],
)

def b_iso(params, transforms, profiles, data, **kwargs):

    data["b_iso"] = - jnp.mean( jnp.sum( data["H_1"] * data["H_1"] , axis=-1 
                                       ) / jnp.sum( data["nabla_s_V_iso"] * data["H_1"] , axis=-1 
                                                  ) 
                              )

    return data

# Find a poloidal harmonic vector on a surface
@register_compute_fun(
    name="nabla_s_V_iso",
    label="\\nabla_s V_{iso}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "e^theta_s","e^zeta_s",
          "Psi_iso",
          #"H_1",
         ],
)

def nabla_s_V_iso(params, transforms, profiles, data, **kwargs):

    data["nabla_s_V_iso"] = data["e^theta_s"] - grad_(data["Psi_iso"],
                                                      data,
                                                      2*(transforms["grid"].M)+1, 
                                                      2*(transforms["grid"].N)+1)
    
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["nabla_s_V_iso"] = rpz2xyz_vec(data["nabla_s_V_iso"], phi=data["phi"])
        
    return data

@register_compute_fun(
    name="Psi_iso_tt",
    label="\\Psi_{iso,tt}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","Psi_iso_t"
         ],
)

def Psi_iso_tt(params, transforms, profiles, data, **kwargs):
    
    data["Psi_iso_tt"] = first_derivative_t(data["Psi_iso_t"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="Psi_iso_tz",
    label="\\psi_{iso,tz}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","Psi_iso_t"
         ],
)

def Psi_iso_tz(params, transforms, profiles, data, **kwargs):
    
    data["Psi_iso_tz"] = first_derivative_z(data["Psi_iso_t"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="Psi_iso_zz",
    label="\\Psi_{iso,zz}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","Psi_iso_z"
         ],
)

def Psi_iso_zz(params, transforms, profiles, data, **kwargs):
    
    data["Psi_iso_zz"] = first_derivative_z(data["Psi_iso_z"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="Psi_iso_t",
    label="\\Psi_{iso,t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "psi_iso"
         ],
)

def Psi_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["Psi_iso_t"] = first_derivative_t(data["Psi_iso"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="Psi_iso_z",
    label="\\Psi_{iso,z}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
         ],
)

def Psi_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["Psi_iso_z"] = first_derivative_z(data["Psi_iso"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="Psi_iso",
    label="\\Psi_{iso}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "e^theta_s","e^zeta_s",
          "e^theta_s_t","e^theta_s_z",
          "e^zeta_s_t","e^zeta_s_z",
          "sqrt(g)",
          "sqrt(g)_t","sqrt(g)_z",
          "g^tt", "g^zz", "g^tz",
          "g^tt_t", "g^zz_z", 
          "g^tz_t",
          "g^tz_z",
          "nabla_s^2_theta"],
)

def Psi_iso(params, transforms, profiles, data, **kwargs):
    
    data["Psi_iso"] = find_phi(data, 
                               2*(transforms["grid"].M)+1,  
                               2*(transforms["grid"].N)+1, 
                               data["nabla_s^2_theta"]
                              )
    
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="H_1",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "e^theta_s","e^zeta_s",
          "phi_iso",
         ],
)

def H1(params, transforms, profiles, data, **kwargs):

    data["H_1"] = data["e^zeta_s"] - grad_(data["phi_iso"], 
                                           data,
                                           2*(transforms["grid"].M)+1, 
                                           2*(transforms["grid"].N)+1)
    
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["H_1"] = rpz2xyz_vec(data["H_1"], phi=data["phi"])
        
    return data

@register_compute_fun(
    name="phi_iso",
    label="\\phi_{iso}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "e^theta_s","e^zeta_s",
          "e^theta_s_t","e^theta_s_z",
          "e^zeta_s_t","e^zeta_s_z",
          "sqrt(g)",
          "sqrt(g)_t","sqrt(g)_z",
          "g^tt", "g^zz", "g^tz",
          "g^tt_t", "g^zz_z", 
          "g^tz_t",
          "g^tz_z",
          "nabla_s^2_zeta",
         ],
)

def phi_iso(params, transforms, profiles, data, **kwargs):
    
    data["phi_iso"] = find_phi(data, 
                               2*(transforms["grid"].M)+1,  
                               2*(transforms["grid"].N)+1, 
                               data["nabla_s^2_zeta"]
                              )
    
    return data

# Function to find the scalar that cancels the surface divergence
def find_phi(data,m_size,n_size, rhs):
    
    x = jnp.ones(data["theta"].shape[0])
    fun_wrapped = lambda x: u_div_residual(x,data,m_size,n_size)
    A_ = Derivative(fun_wrapped,deriv_mode="looped").compute(x)
    
    #A_ = jax.jacfwd(fun_wrapped)(x)
    #return jnp.linalg.pinv(A_)@rhs
    
    return jnp.linalg.pinv(A_)@rhs

def u_div_residual(y,data,m_size,n_size,):

    f_t = first_derivative_t(y, data,m_size,n_size,)
    f_z = first_derivative_z(y, data,m_size,n_size,)
    f_tt = first_derivative_t(f_t, data,m_size,n_size,)
    f_zz = first_derivative_z(f_z, data,m_size,n_size,)
    f_tz = first_derivative_z(f_t, data,m_size,n_size,)

    nabla_s_2_f = (jnp.sum(data["e^theta_s"]*data["e^theta_s_t"], axis=-1)*f_t
                   + jnp.sum(data["e^theta_s"]*data["e^theta_s"], axis=-1)*f_tt
                   + jnp.sum(data["e^theta_s"]*data["e^zeta_s_t"], axis=-1)*f_z
                   + jnp.sum(data["e^theta_s"]*data["e^zeta_s"], axis=-1)*f_tz
                   + jnp.sum(data["e^zeta_s"]*data["e^theta_s_z"], axis=-1)*f_t
                   + jnp.sum(data["e^theta_s"]*data["e^zeta_s"], axis=-1)*f_tz
                   + jnp.sum(data["e^zeta_s"]*data["e^zeta_s_z"], axis=-1)*f_z
                   + jnp.sum(data["e^zeta_s"]*data["e^zeta_s"], axis=-1)*f_zz
                  )
    
    #nabla_s_2_f = ( data["sqrt(g)_t"]*(data["g^tt"]*f_t + data["g^tz"]*f_z )
    #               + data["sqrt(g)"]*(data["g^tt_t"]*f_t + data["g^tt"]*f_tt 
    #                                  + data["g^tz_t"]*f_z + data["g^tz"]*f_tz
    #                                 ) 
    #               + data["sqrt(g)_z"]*(data["g^tz"]*f_t + data["g^zz"]*f_z )
    #               + data["sqrt(g)"]*(data["g^tz_z"]*f_t + data["g^tz"]*f_tz 
    #                                  + data["g^zz_z"]*f_z + data["g^zz"]*f_zz)
    #              )*data["sqrt(g)"]**(-1)
    
    return nabla_s_2_f

def grad_(y,data,m_size,n_size):
    
    f_t_ = first_derivative_t(y, data,m_size,n_size,)
    f_z_ = first_derivative_z(y, data,m_size,n_size,)
    
    return (f_t_*data["e^theta_s"].T  + f_z_*data["e^zeta_s"].T).T
    #return ((data["g^tt"]*f_t_ + data["g^tz"]*f_z_)*data["e_theta"].T
    #        + (data["g^tz"]*f_t_ + data["g^zz"]*f_z_)*data["e_zeta"].T
    #       ).T

##############################################################################################################################
# Finite difference derivatives #
##############################################################################################################################

# First derivatives for periodic functions
def first_derivative_t(a_mn,data,m_size,n_size):
    
    # Rearrange A as a matrix
    A1 = a_mn.reshape((n_size,m_size)).T
    
    # theta-step
    dt = data["theta"][1] - data["theta"][0]
    
    # d(sigma)/dt
    A_t = jnp.zeros_like(A1)
    # i = 0
    A_t = A_t.at[0, :].set( (A1[1, :] - A1[m_size-1, :]) * (2 * dt) ** (-1) )
    # i = n_size
    A_t = A_t.at[m_size-1, :].set( (A1[0, :] - A1[m_size-2, :]) * (2 * dt) ** (-1) )
    # Intermediate steps
    A_t = A_t.at[1:m_size - 1, :].set((A1[2:m_size, :] - A1[0:m_size - 2, :]) * (2 * dt) ** (-1))
    
    return (A_t.T
           #).reshape(-1)
           ).flatten()

def first_derivative_z(a_mn,data,m_size,n_size):
    
    # Rearrange A as a matrix
    A2 = a_mn.reshape((n_size,m_size)).T
    
    # dz-step
    dz = data["zeta"][m_size] - data["zeta"][0]
    
    # d(V)/dz
    A_z = jnp.zeros_like(A2)
    # at i = 0
    A_z = A_z.at[:, 0].set( (A2[:, 1] - A2[:, n_size - 1]) * (2 * dz) ** (-1) )
    # at i = n_size
    A_z = A_z.at[:, n_size - 1].set((A2[:, 0] - A2[:, n_size - 2]) * (2 * dz) ** (-1) )
    # Intermediate steps
    A_z = A_z.at[:, 1:n_size - 1].set((A2[:, 2:n_size] - A2[:, 0:n_size - 2]) * (2 * dz) ** (-1) )
    
    return (A_z.T
           #).reshape(-1)
           ).flatten()

# First derivatives with shifted approximations at the edges
def first_derivative_t2(a_mn,data,m_size,n_size):
    
    # Rearrange A as a matrix
    A1_ = a_mn.reshape((n_size,m_size)).T
    
    # theta-step
    dt = data["theta"][1] - data["theta"][0]
    
    # d(V)/dz
    A_t_ = jnp.zeros_like(A1_)
    # i = 0
    A_t_ = A_t_.at[0,:].set( ( -3*A1_[0,:] + 4*A1_[1,:] - A1_[2,:] )*(2*dt)**(-1) )
    # i = n_size
    A_t_ = A_t_.at[m_size-1,:].set( ( 3*A1_[m_size-1,:] - 4*A1_[m_size-2,:] + A1_[m_size-3,:] )*(2*dt)**(-1) )
    # Intermediate steps
    A_t_ = A_t_.at[1:m_size - 1, :].set( ( A1_[2:m_size, :] - A1_[0:m_size - 2, :] ) * (2 * dt) ** (-1) )
    
    return (A_t_.T
           #).reshape(-1)
           ).flatten()

def first_derivative_z2(a_mn,data,m_size,n_size):
    
    # Rearrange A as a matrix
    A2_ = a_mn.reshape((n_size,m_size)).T
    
    # dz-step
    dz = data["zeta"][m_size] - data["zeta"][0]
    
    # d(V)/dz
    A_z_ = jnp.zeros_like(A2_)
    # at i = 0
    A_z_ = A_z_.at[:,0].set( ( -3*A2_[:,0] + 4*A2_[:,1] - A2_[:,2] )*(2*dz)**(-1) )
    # at i = n_size
    A_z_ = A_z_.at[:,n_size-1].set( ( 3*A2_[:,n_size-1] - 4*A2_[:,n_size-1-1] + A2_[:,n_size-1-2] )*(2*dz)**(-1) )
    # Intermediate steps
    A_z_ = A_z_.at[:, 1:n_size - 1].set( ( A2_[:, 2:n_size] - A2_[:, 0:n_size - 2] ) * (2 * dz) ** (-1) )
    
    return (A_z_.T
           #).reshape(-1)
           ).flatten()