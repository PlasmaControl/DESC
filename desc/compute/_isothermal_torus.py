"""Compute functions for isothermal coordinates and Harmonic Field.

Notes
-----
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""
from desc.backend import jnp
from .data_index import register_compute_fun

import jax
import jax.numpy as jnpå
from jax import jit, jacfwd

from desc.utils import flatten_list, cross, dot

from .data_index import register_compute_fun

from desc.derivatives import Derivative

@register_compute_fun(
    name="e_v_torus",
    label="\\e_{v, torus}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["u_iso_t_torus", "u_iso_z_torus",
          "v_iso_t_torus", "v_iso_z_torus",
          "e_theta", "e_zeta",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def e_v_torus(params, transforms, profiles, data, **kwargs):
    
    l = data["u_iso_t_torus"] * data["v_iso_z_torus"] - data["u_iso_z_torus"] * data["v_iso_t_torus"]

    data["e_v_torus"] = ( ( 1 / l ) * ( - data["u_iso_z"] * data["e_theta"].T + data["u_iso_t_torus"] * data["e_zeta"].T ) ).T
    
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_v_torus"] = rpz2xyz_vec(data["e_u_torus"], phi=data["phi"])
        
    return data

@register_compute_fun(
    name="e_u_torus",
    label="\\e_{u, torus}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["u_iso_t_torus", "u_iso_z_torus",
          "v_iso_t_torus", "v_iso_z_torus",
          "e_theta", "e_zeta",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def e_u_torus(params, transforms, profiles, data, **kwargs):
    
    l = data["u_iso_t_torus"] * data["v_iso_z_torus"] - data["u_iso_z_torus"] * data["v_iso_t_torus"]

    data["e_u_torus"] = ( ( 1 / l ) * ( data["v_iso_z_torus"] * data["e_theta"].T - data["v_iso_t_torus"] * data["e_zeta"].T ) ).T
    
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_u_torus"] = rpz2xyz_vec(data["e_u_torus"], phi=data["phi"])
        
    return data

@register_compute_fun(
    name="lambda_iso_v_torus",
    label="\\lambda_{iso,v, torus}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["u_iso_t", "u_iso_z",
          "v_iso_t", "v_iso_z",
          "lambda_iso_t","lambda_iso_z"
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def lambda_iso_v_torus(params, transforms, profiles, data, **kwargs):
    
    data["lambda_iso_v_torus"] = (( - data["u_iso_z_torus"]*data["lambda_iso_t_torus"] + data["u_iso_t_torus"]*data["lambda_iso_z_torus"]
                            ) / ( data["u_iso_t_torus"] * data["v_iso_z_torus"] - data["u_iso_z_torus"] * data["v_iso_t_torus"] )
                           )
    
    return data

@register_compute_fun(
    name="lambda_iso_u_torus",
    label="\\lambda_{iso,u, torus}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["u_iso_t_torus", "u_iso_z_torus",
          "v_iso_t_torus", "v_iso_z_torus",
          "lambda_iso_t_torus","lambda_iso_z_torus"
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def lambda_iso_u_torus(params, transforms, profiles, data, **kwargs):
    
    data["lambda_iso_u_torus"] = ( (data["v_iso_z_torus"] * data["lambda_iso_t_torus"] - data["v_iso_t_torus"] * data["lambda_iso_z_torus"]
                            ) / ( data["u_iso_t_torus"] * data["v_iso_z_torus"] - data["u_iso_z_torus"] * data["v_iso_t_torus"] )
                           )
    
    return data

@register_compute_fun(
    name="lambda_iso_z_torus",
    label="\\lambda_{iso,z, torus}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["lambda_iso_torus",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def lambda_iso_z_torus(params, transforms, profiles, data, **kwargs):
    
    data["lambda_iso_z_torus"] = first_derivative_z(data["lambda_iso_torus"], 
                                              data,
                                              2*(transforms["grid"].M)+1,
                                              2*(transforms["grid"].N)+1)
    
    return data

@register_compute_fun(
    name="lambda_iso_t_torus",
    label="\\lambda_{iso,t, torus}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["lambda_iso_torus",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def lambda_iso_t_torus(params, transforms, profiles, data, **kwargs):
    
    data["lambda_iso_t_torus"] = first_derivative_t(data["lambda_iso_torus"], 
                                              data,
                                              2*(transforms["grid"].M)+1,
                                              2*(transforms["grid"].N)+1)
    
    return data

@register_compute_fun(
    name="lambda_iso_torus",
    label="\\psi_{iso,tz, torus}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["H_1_torus",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def lambda_iso_torus(params, transforms, profiles, data, **kwargs):
    
    data["lambda_iso_torus"] = jnp.sum( data["H_1_torus"] * data["H_1_torus"], axis = -1 ) ** (-1/2)
    
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="v_iso_tt_torus",
    label="\\mathrm{H}_1_{torus}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["u_iso_tt_torus", "V_iso_tt_torus", "b_iso_torus", "lambda_ratio_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def v_iso_tt_torus(params, transforms, profiles, data, **kwargs):
    
    data["v_iso_tt_torus"] = - data["lambda_ratio_torus"] * ( data["u_iso_tt_torus"] + data["b_iso_torus"] * data["V_iso_tt_torus"] ) 
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="v_iso_tz_torus",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["u_iso_tz_torus", "V_iso_tz_torus", "b_iso_torus", "lambda_ratio_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def v_iso_tz_torus(params, transforms, profiles, data, **kwargs):
    
    data["v_iso_tz_torus"] = - data["lambda_ratio_torus"] * ( data["u_iso_tz_torus"] + data["b_iso_torus"] * data["V_iso_tz_torus"] ) 
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="v_iso_zz_torus",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["u_iso_zz_torus", "V_iso_zz_torus", "b_iso_torus", "lambda_ratio_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def v_iso_zz_torus(params, transforms, profiles, data, **kwargs):
    
    data["v_iso_zz_torus"] = - data["lambda_ratio_torus"] * ( data["u_iso_zz_torus"] + data["b_iso_torus"] * data["V_iso_zz_torus"] ) 
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="v_iso_t_torus",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["u_iso_t_torus", "V_iso_t_torus", "b_iso_torus", "lambda_ratio_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def v_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["v_iso_t_torus"] = data["lambda_ratio_torus"] * ( data["b_iso_torus"] * data["V_iso_t_torus"] ) 
    #- data["lambda_ratio_torus"] * ( data["u_iso_t_torus"] + data["b_iso_torus"] * data["V_iso_t_torus"] ) 
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="v_iso_z_torus",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["u_iso_z_torus","V_iso_z_torus", "b_iso_torus", "lambda_ratio_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def v_iso_z(params, transforms, profiles, data, **kwargs):
    data["v_iso_z_torus"] = data["lambda_ratio_torus"] * ( data["b_iso_torus"] * data["V_iso_z_torus"] ) 
    #- data["lambda_ratio_torus"] * ( data["u_iso_z_torus"] + data["b_iso_torus"] * data["V_iso_z_torus"] ) 
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="v_iso_torus",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["u_iso_torus", "V_iso_torus", "b_iso_torus", "lambda_ratio_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def v_iso(params, transforms, profiles, data, **kwargs):
    data["v_iso_torus"] = data["lambda_ratio_torus"] * ( data["b_iso_torus"] * data["V_iso_torus"] ) 
    #- data["lambda_ratio_torus"] * ( data["u_iso_torus"] + data["b_iso_torus"] * data["V_iso_torus"] ) 
                             
    return data

@register_compute_fun(
    name="u_iso_tt_torus",
    label="\\u_{iso,t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal derivative of toroidal isothermal coordinate ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","u_iso_t_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def u_iso_tt(params, transforms, profiles, data, **kwargs):
    
    data["u_iso_tt_torus"] = - first_derivative_t(data["u_iso_t_torus"], 
                                            data,
                                            2*(transforms["grid"].M)+1,
                                            2*(transforms["grid"].N)+1)
    
    return data

@register_compute_fun(
    name="u_iso_tz_torus",
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
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def u_iso_tz(params, transforms, profiles, data, **kwargs):
    
    data["u_iso_tz_torus"] = - first_derivative_z(data["u_iso_t_torus"], 
                                            data,
                                            2*(transforms["grid"].M)+1,
                                            2*(transforms["grid"].N)+1)
    
    return data

@register_compute_fun(
    name="u_iso_zz_torus",
    label="\\u_{iso,zz}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal derivative of toroidal isothermal coordinate ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","u_iso_z_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def u_iso_zz(params, transforms, profiles, data, **kwargs):
    
    data["u_iso_zz_torus"] = - first_derivative_z(data["u_iso_z_torus"], 
                                            data,
                                            2*(transforms["grid"].M)+1,
                                            2*(transforms["grid"].N)+1)
    
    return data

@register_compute_fun(
    name="u_iso_t_torus",
    label="\\u_{iso,t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal derivative of toroidal isothermal coordinate ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","phi_iso_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def u_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["u_iso_t_torus"] = - first_derivative_t(data["phi_iso_torus"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1)
    
    return data

@register_compute_fun(
    name="u_iso_z_torus",
    label="\\u_{iso,t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal derivative of toroidal isothermal coordinate ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","phi_iso_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def u_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["u_iso_z_torus"] = 1 - first_derivative_z(data["phi_iso_torus"], 
                                             data,
                                             2*(transforms["grid"].M)+1,
                                             2*(transforms["grid"].N)+1)
    
    return data

@register_compute_fun(
    name="u_iso_torus",
    label="\\u_{iso}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["zeta", "phi_iso_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def u_iso_torus(params, transforms, profiles, data, **kwargs):
    
    data["u_iso_torus"] = data["zeta"] - data["phi_iso_torus"]
    
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="V_iso_tt_torus",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["Psi_iso_tt_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def V_iso_tt_torus(params, transforms, profiles, data, **kwargs):
    
    data["V_iso_tt_torus"] = - data["Psi_iso_tt_torus"]
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="V_iso_tz_torus",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["Psi_iso_tz_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def V_iso_tz_torus(params, transforms, profiles, data, **kwargs):
    
    data["V_iso_tz_torus"] = - data["Psi_iso_tz_torus"]
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="V_iso_zz_torus",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["Psi_iso_zz_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def V_iso_zz_torus(params, transforms, profiles, data, **kwargs):
    
    data["V_iso_zz_torus"] = - data["Psi_iso_zz_torus"]
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="V_iso_t_torus",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["Psi_iso_t_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def V_iso_t_torus(params, transforms, profiles, data, **kwargs):
    
    data["V_iso_t_torus"] = 1 - data["Psi_iso_t_torus"]
                             
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="V_iso_z_torus",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["Psi_iso_z_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def V_iso_z_torus(params, transforms, profiles, data, **kwargs):
    
    data["V_iso_z_torus"] = - data["Psi_iso_z_torus"]
                             
    return data


# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="V_iso_torus",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta", "Psi_iso_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def V_iso_torus(params, transforms, profiles, data, **kwargs):
    
    data["V_iso_torus"] = data["theta"] - data["Psi_iso_torus"]
                             
    return data

# Find a poloidal harmonic vector on a surface
@register_compute_fun(
    name="H_2_torus",
    label="\\mathrm{H}_2",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta", "zeta", "e^theta_s", "e^zeta_s", "H_1_torus", "nabla_s_V_iso_torus", "lambda_ratio_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def H2_torus(params, transforms, profiles, data, **kwargs):
    
    # Normalize H_2 to match same magnitude of H_1 (?)
    data["H_2_torus"] = data["lambda_ratio_torus"] * ( data["b_iso_torus"] * data["nabla_s_V_iso_torus"] )
    #- data["lambda_ratio_torus"] * ( data["H_1_torus"] + data["b_iso_torus"] * data["nabla_s_V_iso_torus"] )
    
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["H_2_torus"] = rpz2xyz_vec(data["H_2_torus"], phi=data["phi"])
        
    return data

# Find a toroidal harmonic vector on a surface
@register_compute_fun(
    name="lambda_ratio_torus",
    label="\\mathrm{H}_1",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","e^theta_s","e^zeta_s","H_1_torus", "nabla_s_V_iso_torus","b_iso_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def lambda_ratio(params, transforms, profiles, data, **kwargs):
    
    lambda_temp = data["b_iso_torus"] * data["nabla_s_V_iso_torus"]
    #data["H_1_torus"] + data["b_iso_torus"] * data["nabla_s_V_iso_torus"]
    
    data["lambda_ratio_torus"] = jnp.mean( jnp.sqrt( jnp.sum( data["H_1_torus"] * data["H_1_torus"] , axis=-1 
                                                      ) / jnp.sum( lambda_temp * lambda_temp , axis=-1 )
                                             )
                                   )
                             
    return data

@register_compute_fun(
    name="psi_iso_torus",
    label="\\psi_{iso}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","Psi_iso_torus","b_iso_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def psi_iso_torus(params, transforms, profiles, data, **kwargs):
    
    data["psi_iso_torus"] = data["Psi_iso_torus"]
    #data["b_iso_torus"] * data["Psi_iso_torus"]
    
    return data

@register_compute_fun(
    name="b_iso_torus",
    label="\\mathrm{H}_2",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","e^theta_s","e^zeta_s","H_1_torus","nabla_s_V_iso_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def b_iso(params, transforms, profiles, data, **kwargs):

    data["b_iso_torus"] = 1
    #- jnp.mean( jnp.sum( data["H_1_torus"] * data["H_1_torus"] , axis=-1 
    #                                   ) / jnp.sum( data["nabla_s_V_iso_torus"] * data["H_1_torus"] , axis=-1 
    #                                              ) 
    #                          )

    return data

# Find a poloidal harmonic vector on a surface
@register_compute_fun(
    name="nabla_s_V_iso_torus",
    label="\\nabla_s V_{iso, torus}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","e^theta_s","e^zeta_s","Psi_iso_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def nabla_s_V_iso(params, transforms, profiles, data, **kwargs):

    data["nabla_s_V_iso_torus"] = data["e^theta_s"] - grad_(data["Psi_iso_torus"],
                                                      data,
                                                      2*(transforms["grid"].M)+1, 
                                                      2*(transforms["grid"].N)+1)
    
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["nabla_s_V_iso_torus"] = rpz2xyz_vec(data["nabla_s_V_iso_torus"], phi=data["phi"])
        
    return data

@register_compute_fun(
    name="Psi_iso_tt_torus",
    label="\\Psi_{iso,tt}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","Psi_iso_t_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def Psi_iso_tt(params, transforms, profiles, data, **kwargs):
    
    data["Psi_iso_tt_torus"] = first_derivative_t(data["Psi_iso_t_torus"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="Psi_iso_tz_torus",
    label="\\psi_{iso,tz}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","Psi_iso_t_torus"
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def Psi_iso_tz(params, transforms, profiles, data, **kwargs):
    
    data["Psi_iso_tz_torus"] = first_derivative_z(data["Psi_iso_t_torus"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="Psi_iso_zz_torus",
    label="\\Psi_{iso,zz}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","Psi_iso_z_torus"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def Psi_iso_zz(params, transforms, profiles, data, **kwargs):
    
    data["Psi_iso_zz_torus"] = first_derivative_z(data["Psi_iso_z_torus"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="Psi_iso_t_torus",
    label="\\Psi_{iso,t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta","psi_iso"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def Psi_iso_t_torus(params, transforms, profiles, data, **kwargs):
    
    data["Psi_iso_t_torus"] = first_derivative_t(data["Psi_iso_torus"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="Psi_iso_z_torus",
    label="\\Psi_{iso,z}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta", "Psi_iso_torus",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def Psi_iso_z_torus(params, transforms, profiles, data, **kwargs):
    
    data["Psi_iso_z_torus"] = first_derivative_z(data["Psi_iso_torus"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="Psi_iso_torus",
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
          "nabla_s^2_theta"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def Psi_iso_torus(params, transforms, profiles, data, **kwargs):
    
    data["Psi_iso_torus"] = find_phi(data, 
                               2*(transforms["grid"].M)+1,  
                               2*(transforms["grid"].N)+1, 
                               data["nabla_s^2_theta"]
                              )
    
    return data

@register_compute_fun(
    name="H_1_torus",
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
          "phi_iso_torus",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def H1(params, transforms, profiles, data, **kwargs):

    data["H_1_torus"] = data["e^zeta_s"] - grad_(data["phi_iso_torus"], 
                                           data,
                                           2*(transforms["grid"].M)+1, 
                                           2*(transforms["grid"].N)+1)
    
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["H_1_torus"] = rpz2xyz_vec(data["H_1_torus"], phi=data["phi"])
        
    return data


@register_compute_fun(
    name="phi_iso_t_torus",
    label="\\phi_{iso}_t_torus",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=[#"theta","zeta",
          "phi_iso_torus",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def phi_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["phi_iso_t_torus"] = first_derivative_t(data["phi_iso_torus"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data

@register_compute_fun(
    name="phi_iso_z_torus",
    label="\\phi_{iso}_z",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=[#"theta","zeta",
          "phi_iso_torus",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def phi_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["phi_iso_z_torus"] = first_derivative_z(data["phi_iso_torus"], 
                                         data,
                                         2*(transforms["grid"].M)+1,
                                         2*(transforms["grid"].N)+1,)
    
    return data
    
@register_compute_fun(
    name="phi_iso_torus",
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
          "nabla_s^2_zeta",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)

def phi_iso(params, transforms, profiles, data, **kwargs):
    
    data["phi_iso_torus"] = find_phi(data, 
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
    
    return nabla_s_2_f

def grad_(y,data,m_size,n_size):
    
    f_t_ = first_derivative_t(y, data,m_size,n_size,)
    f_z_ = first_derivative_z(y, data,m_size,n_size,)
    
    return (f_t_*data["e^theta_s"].T  + f_z_*data["e^zeta_s"].T).T

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
    
    return (A_t.T).flatten()

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
    
    return (A_z.T).flatten()

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
    
    return (A_t_.T).flatten()

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
    
    return (A_z_.T).flatten()