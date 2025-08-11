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

from ._isothermal import ( first_derivative_t, first_derivative_z, 
                         first_derivative_t2, first_derivative_z2
                        )

@register_compute_fun(
    name="theta_gamma",
    label="\\theta_\\gamma",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["E_iso","F_iso","G_iso",
          "alpha_iso",
         ],
)

def theta_gamma(params, transforms, profiles, data, **kwargs):
    
    data["theta_gamma"] = ( jnp.cos( data["alpha_iso"] ) / jnp.sqrt( data["E_iso"] ) 
                           - data["F_iso"] / jnp.sqrt( data["E_iso"]
                                                     ) * jnp.sin( data["alpha_iso"] 
                                                                ) / jnp.sqrt( data["E_iso"] * data["G_iso"] - data["F_iso"] ** 2)
                          )
        
    return data

@register_compute_fun(
    name="zeta_gamma",
    label="\\zeta_\\gamma",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["E_iso","F_iso","G_iso",
          "alpha_iso",
         ],
)

def zeta_gamma(params, transforms, profiles, data, **kwargs):
    
    data["zeta_gamma"] = ( jnp.sin( data["alpha_iso"] ) / jnp.sqrt( data["G_iso"] - data["F_iso"] ** 2 / data["E_iso"] )
                          )
        
    return data

@register_compute_fun(
    name="theta_beta",
    label="\\theta_\\beta",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["E_iso","F_iso","G_iso",
          "alpha_iso",
         ],
)

def theta_beta(params, transforms, profiles, data, **kwargs):
    
    data["theta_beta"] = ( - jnp.sin( data["alpha_iso"] ) / jnp.sqrt( data["E_iso"] ) 
                           - data["F_iso"] / jnp.sqrt( data["E_iso"] 
                                                     ) * jnp.cos( data["alpha_iso"] 
                                                                ) / jnp.sqrt( data["E_iso"] * data["G_iso"] - data["F_iso"] ** 2)
                          )
        
    return data

@register_compute_fun(
    name="zeta_beta",
    label="\\zeta_\\beta",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["E_iso","F_iso","G_iso",
          "alpha_iso",
         ],
)

def zeta_beta(params, transforms, profiles, data, **kwargs):
    
    data["zeta_beta"] = jnp.cos( data["alpha_iso"] ) / jnp.sqrt( data["G_iso"] - data["F_iso"] ** 2 / data["E_iso"] )
        
    return data

@register_compute_fun(
    name="alpha_iso",
    label="\\alpha_{iso}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["E_iso","F_iso","G_iso",
          "l_iso","m_iso","n_iso",
          "E_iso_t","F_iso_t","G_iso_t",
          "l_iso_t","m_iso_t","n_iso_t",
         ],
)

def alpha_iso(params, transforms, profiles, data, **kwargs):
    
    data["alpha_iso"] = ( 1 / 2 ) * jnp.arctan( 
        - 2 * ( - data["F_iso"] * data["l_iso"] + data["E_iso"] * data["m_iso"] 
              ) * jnp.sqrt( data["E_iso"] * data["G_iso"] - data["F_iso"] ** 2
                          ) / (
            ( 2 * data["F_iso"] ** 2 - data["E_iso"] * data["G_iso"]
            ) * data["l_iso"] 
            - 2 * data["E_iso"] * data["F_iso"] * data["m_iso"] 
            + data["E_iso"] ** 2 * data["n_iso"] 
        ) 
    )
        
    return data

@register_compute_fun(
    name="alpha_iso_t",
    label="\\alpha_{iso, t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["E_iso","F_iso","G_iso",
          "l_iso","m_iso","n_iso",
          "E_iso_t","F_iso_t","G_iso_t",
          "l_iso_t","m_iso_t","n_iso_t",
         ],
)

def alpha_iso_t(params, transforms, profiles, data, **kwargs):
    
    a_ = ( - data["F_iso_t"] * data["l_iso"] - data["F_iso"] * data["l_iso_t"] 
          + data["E_iso_t"] * data["m_iso"] + data["E_iso"] * data["m_iso_t"] 
         ) * jnp.sqrt( data["E_iso"] * data["G_iso"] - data["F_iso"] ** 2 ) 
    
    b_ = ( 1 / 2 ) * ( - data["F_iso"] * data["l_iso"] + data["E_iso"] * data["m_iso"] 
                     ) * ( data["E_iso"] * data["G_iso"] - data["F_iso"] ** 2 
                         ) ** ( - 1 / 2 ) * ( data["E_iso_t"] * data["G_iso"] 
                                             + data["E_iso"] * data["G_iso_t"] 
                                             - 2 * data["F_iso"] * data["F_iso_t"] )
    
    c_ = ( ( 2 * data["F_iso"] ** 2  - data["E_iso"] * data["G_iso"] ) * data["l_iso"] 
          - 2 * data["E_iso"] * data["F_iso"] * data["m_iso"] 
          + data["E_iso"] ** 2 * data["n_iso"]
        )
    
    d_ = ( ( 2 * (2 * data["F_iso"] * data["F_iso_t"])  - ( data["E_iso_t"] * data["G_iso"]
                                                           + data["E_iso"] * data["G_iso_t"] ) 
           ) * data["l_iso"]
          + ( - data["F_iso"] * data["l_iso"] + data["E_iso"] * data["m_iso"]
            ) * data["l_iso_t"]
          - 2 * ( data["E_iso_t"] * data["F_iso"] * data["m_iso"] 
                 + data["E_iso"] * data["F_iso_t"] * data["m_iso"] 
                 + data["E_iso"] * data["F_iso"] * data["m_iso_t"])
          + (2 * data["E_iso"] * data["E_iso_t"] * data["n_iso"] + data["E_iso"] ** 2 * data["n_iso_t"])
         )
    
    u_ = - 2 * ( - data["F_iso"] * data["l_iso"] + data["E_iso"] * data["m_iso"] 
               ) * jnp.sqrt( data["E_iso"] * data["G_iso"] - data["F_iso"] ** 2
                           ) / ( ( 2 * data["F_iso"] ** 2 - data["E_iso"] * data["G_iso"] ) * data["l_iso"] 
                                - 2 * data["E_iso"] * data["F_iso"] * data["m_iso"]
                                + data["E_iso"] ** 2 * data["n_iso"] 
                               )
    
    data["alpha_iso_t"] = - 1 / ( 1 + u_ ** 2 ) * ( ( a_ + b_ ) / c_ - d_ / c_ **2 )
        
    return data

@register_compute_fun(
    name="alpha_iso_z",
    label="\\alpha_{iso, z}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["E_iso","F_iso","G_iso",
          "l_iso","m_iso","n_iso",
          "E_iso_z","F_iso_z","G_iso_z",
          "l_iso_z","m_iso_z","n_iso_z",
         ],
)

def alpha_iso_z(params, transforms, profiles, data, **kwargs):
    
    _a_ = ( - data["F_iso_z"] * data["l_iso"] 
           - data["F_iso"] * data["l_iso_z"] 
           + data["E_iso_z"] * data["m_iso"] 
           + data["E_iso"] * data["m_iso_z"] ) * jnp.sqrt( data["E_iso"] * data["G_iso"] - data["F_iso"] ** 2 ) 
    
    _b_ = ( 1 / 2 ) * ( - data["F_iso"] * data["l_iso"] + data["E_iso"] * data["m_iso"] 
                      ) * ( data["E_iso"] * data["G_iso"] - data["F_iso"] ** 2
                          ) ** ( - 1 / 2 ) * ( data["E_iso_z"] * data["G_iso"] 
                                              + data["E_iso"] * data["G_iso_z"] 
                                              - 2 * data["F_iso"] * data["F_iso_z"] 
                                             )
    
    _c_ = ( ( 2 * data["F_iso"] ** 2  - data["E_iso"] * data["G_iso"]
            ) * data["l_iso"] 
           - 2 * data["E_iso"] * data["F_iso"] * data["m_iso"] 
           + data["E_iso"] ** 2 * data["n_iso"]
          )
    
    _d_ = ( ( 2 * (2 * data["F_iso"] * data["F_iso_z"])  - ( data["E_iso_z"] * data["G_iso"]
                                                            + data["E_iso"] * data["G_iso_z"] ) 
           ) * data["l_iso"]
          + ( - data["F_iso"] * data["l_iso"] + data["E_iso"] * data["m_iso"]
            ) * data["l_iso_z"]
          - 2 * ( data["E_iso_z"] * data["F_iso"] * data["m_iso"] 
                 + data["E_iso"] * data["F_iso_z"] * data["m_iso"] 
                 + data["E_iso"] * data["F_iso"] * data["m_iso_z"])
          + (2 * data["E_iso"] * data["E_iso_z"] * data["n_iso"] + data["E_iso"] ** 2 * data["n_iso_t"])
         )
    
    _u_ = - 2 * ( - data["F_iso"] * data["l_iso"] + data["E_iso"] * data["m_iso"] 
               ) * jnp.sqrt( data["E_iso"] * data["G_iso"] - data["F_iso"] ** 2
                           ) / ( ( 2 * data["F_iso"] ** 2 - data["E_iso"] * data["G_iso"] ) * data["l_iso"] 
                                - 2 * data["E_iso"] * data["F_iso"] * data["m_iso"]
                                + data["E_iso"] ** 2 * data["n_iso"] 
                               )
    
    data["alpha_iso_z"] = - 1 / ( 1 + _u_ ** 2 ) * ( ( _a_ + _b_ ) / _c_ - _d_ / _c_ **2 )
        
    return data

@register_compute_fun(
    name="E_iso",
    label="\\E",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_theta",
         ],
)

def E_iso(params, transforms, profiles, data, **kwargs):
    
    data["E_iso"] = jnp.sum( data["e_theta"] * data["e_theta"], axis = -1 )
        
    return data

@register_compute_fun(
    name="F_iso",
    label="\\F",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_theta","e_zeta",
         ],
)

def F_iso(params, transforms, profiles, data, **kwargs):
    
    data["F_iso"] = jnp.sum( data["e_theta"] * data["e_zeta"], axis = -1 )
        
    return data

@register_compute_fun(
    name="G_iso",
    label="\\G",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_zeta",
         ],
)

def G_iso(params, transforms, profiles, data, **kwargs):
    
    data["G_iso"] = jnp.sum( data["e_zeta"] * data["e_zeta"], axis = -1 )
        
    return data

@register_compute_fun(
    name="l_iso",
    label="\\l",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_theta_t","n_rho",
         ],
)

def l_iso(params, transforms, profiles, data, **kwargs):
    
    data["l_iso"] = jnp.sum( data["e_theta_t"] * data["n_rho"], axis = -1 )
        
    return data

@register_compute_fun(
    name="m_iso",
    label="\\m",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_theta_z","n_rho",
         ],
)

def m_iso(params, transforms, profiles, data, **kwargs):
    
    data["m_iso"] = jnp.sum( data["e_theta_z"] * data["n_rho"], axis = -1 )
        
    return data

@register_compute_fun(
    name="n_iso",
    label="\\n_{iso}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_zeta_z", "n_rho",
         ],
)

def n_iso(params, transforms, profiles, data, **kwargs):
    
    data["n_iso"] = jnp.sum( data["n_rho"] * data["e_zeta_z"], axis = -1 )
        
    return data

@register_compute_fun(
    name="E_iso_t",
    label="\\E_{t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_theta","e_theta_t",
         ],
)

def E_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["E_iso_t"] = 2 * jnp.sum( data["e_theta"] * data["e_theta_t"], axis = -1 )
        
    return data

@register_compute_fun(
    name="E_iso_z",
    label="\\E_{z}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_theta","e_theta_z",
         ],
)

def E_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["E_iso_z"] = 2 * jnp.sum( data["e_theta"] * data["e_theta_z"], axis = -1 )
        
    return data

@register_compute_fun(
    name="F_iso_t",
    label="\\F_{t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_theta","e_zeta",
          "e_theta_t","e_zeta_t",
         ],
)

def F_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["F_iso_t"] = ( jnp.sum( data["e_theta_t"] * data["e_zeta"], axis = -1 ) 
                       + jnp.sum( data["e_theta"] * data["e_zeta_t"], axis = -1 )
                      )
        
    return data

@register_compute_fun(
    name="F_iso_z",
    label="\\F_{z}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_theta","e_zeta",
          "e_theta_z","e_zeta_z",
         ],
)

def F_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["F_iso_z"] = ( jnp.sum( data["e_theta_z"] * data["e_zeta"], axis = -1 ) 
                       + jnp.sum( data["e_theta"] * data["e_zeta_z"], axis = -1 )
                      )
        
    return data

@register_compute_fun(
    name="G_iso_t",
    label="\\G_{t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_zeta","e_zeta_t",
         ],
)

def G_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["G_iso_t"] = 2 * jnp.sum( data["e_zeta"] * data["e_zeta_t"], axis = -1 )
        
    return data

@register_compute_fun(
    name="G_iso_z",
    label="\\G_{z}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="~",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_zeta","e_zeta_z",
         ],
)

def G_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["G_iso_z"] = 2 * jnp.sum( data["e_zeta"] * data["e_zeta_z"], axis = -1 )
    
    return data

@register_compute_fun(
    name="l_iso_t",
    label="\\l_{t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_theta_t","n_rho",
          "e_theta_tt","n_rho_t",
         ],
)

def l_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["l_iso_t"] = ( jnp.sum( data["e_theta_tt"] * data["n_rho"], axis = -1 ) 
                   + jnp.sum( data["e_theta_t"] * data["n_rho_t"], axis = -1 )
                  )
        
    return data

@register_compute_fun(
    name="l_iso_z",
    label="\\l_{z}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_theta_t","n_rho",
          "e_theta_tz","n_rho_z",
         ],
)

def l_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["l_iso_z"] = ( jnp.sum( data["e_theta_tz"] * data["n_rho"], axis = -1 ) 
                   + jnp.sum( data["e_theta_t"] * data["n_rho_z"], axis = -1 )
                  )
        
    return data

@register_compute_fun(
    name="m_iso_t",
    label="\\m_{t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_theta_z","n_rho",
          "e_theta_tz","n_rho_t",
         ],
)

def m_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["m_iso_t"] = (jnp.sum( data["e_theta_tz"] * data["n_rho"], axis = -1 ) 
                   + jnp.sum( data["e_theta_z"] * data["n_rho_t"], axis = -1 )
                  )
        
    return data

@register_compute_fun(
    name="m_iso_z",
    label="\\m_{z}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["e_theta_z","n_rho",
          "e_theta_zz","n_rho_z",
         ],
)

def m_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["m_iso_z"] = ( jnp.sum( data["e_theta_zz"] * data["n_rho"], axis = -1 ) 
                       + jnp.sum( data["e_theta_z"] * data["n_rho_z"], axis = -1 )
                      )
        
    return data

@register_compute_fun(
    name="n_iso_t",
    label="\\n_{t}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["n_rho","e_zeta_z",
          "n_rho_t","e_zeta_tz",
         ],
)

def n_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["n_iso_t"] = ( jnp.sum( data["n_rho_t"] * data["e_zeta_z"], axis = -1 ) 
                       + jnp.sum( data["n_rho"] * data["e_zeta_tz"], axis = -1 )
                      )
        
    return data

@register_compute_fun(
    name="n_iso_z",
    label="\\n_{\\zeta}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal Harmonic Vector on a given surface ",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["n_rho","e_zeta_z",
          "n_rho_z","e_zeta_zz",
         ],
)

def n_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["n_iso_z"] = ( jnp.sum( data["n_rho_z"] * data["e_zeta_z"], axis = -1 ) 
                       + jnp.sum( data["n_rho"] * data["e_zeta_zz"], axis = -1 )
                      )
        
    return data

@register_compute_fun(
    name="R_iso_t",
    label="\\R_{\\theta}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="~",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "R_iso",
         ],
)

def R_iso_t(params, transforms, profiles, data, **kwargs):
    
    data["R_iso_t"] = first_derivative_t( data["R_iso"], data,
                                        2 * (transforms["grid"].M) + 1 , 2 * (transforms["grid"].N) + 1 )
    
    return data

@register_compute_fun(
    name="R_iso_z",
    label="\\R_{\\zeta}",
    units="m^{-1}",
    units_long="Inverse meters",
    description="~",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rpz",
    data=["theta","zeta",
          "R_iso",
         ],
)

def R_iso_z(params, transforms, profiles, data, **kwargs):
    
    data["R_iso_z"] = first_derivative_z( data["R_iso"] , data, 
                                         2 * (transforms["grid"].M) + 1 , 2 * (transforms["grid"].N) + 1 )
    
    return data
    
@register_compute_fun(
    name="R_iso",
    label="R_{iso}",
    units="~",
    units_long="~",
    description="Log of scale factor K",
    dim=1,
    params=[],
    transforms={"grid":[],},
    profiles=[],
    coordinates="tz",
    data=["theta","zeta",
          "theta_gamma", "zeta_gamma",
          "theta_beta", "zeta_beta",
          "rhs1","rhs2",
         ],
)
def R_isothermic(params, transforms, profiles, data, **kwargs):
    
    data["R_iso"] = find_R( data , 2 * (transforms["grid"].M) + 1 , 2 * (transforms["grid"].N) + 1 )
        
    return data

@register_compute_fun(
    name="lhs1",
    label="lhs_{1}",
    units="~",
    units_long="~",
    description="~",
    dim=1,
    params=[],
    transforms={"grid":[],},
    profiles=[],
    coordinates="tz",
    data=["theta_gamma", "zeta_gamma",
          "theta_beta", "zeta_beta",
          "alpha_iso",
          "R_iso_t","R_iso_z",
         ],
)
def lhs1(params, transforms, profiles, data, **kwargs):
    
    data["lhs1"] = ( ( data["theta_beta"] * data["R_iso_t"] 
                      + data["zeta_beta"] * data["R_iso_z"] ) * jnp.cos(data["alpha_iso"] ) 
                    + ( data["theta_gamma"] * data["R_iso_t"] 
                       + data["zeta_gamma"] * data["R_iso_z"] ) * jnp.sin(data["alpha_iso"] )
                   )
        
    return data

@register_compute_fun(
    name="lhs2",
    label="lhs_{2}",
    units="~",
    units_long="~",
    description="~",
    dim=1,
    params=[],
    transforms={"grid":[],},
    profiles=[],
    coordinates="tz",
    data=["theta_gamma", "zeta_gamma",
          "theta_beta", "zeta_beta",
          "alpha_iso",
          "R_iso_t","R_iso_z",
         ],
)
def lhs2(params, transforms, profiles, data, **kwargs):
    
    data["lhs2"] = ( ( data["theta_beta"] * data["R_iso_t"] 
                      + data["zeta_beta"] * data["R_iso_z"] ) * jnp.sin(data["alpha_iso"] ) 
                    - ( data["theta_gamma"] * data["R_iso_t"] 
                       + data["zeta_gamma"] * data["R_iso_z"] ) * jnp.cos(data["alpha_iso"] )
                   )
    return data

@register_compute_fun(
    name="rhs1",
    label="rhs_{1}",
    units="~",
    units_long="~",
    description="Right hand side of first PDE for isothermic coordinates",
    dim=1,
    params=[],
    transforms={"grid":[],},
    profiles=[],
    coordinates="tz",
    data=["theta_gamma", "zeta_gamma",
          "theta_beta", "zeta_beta",
          "E_iso","F_iso","G_iso",
          "alpha_iso",
          "E_iso_t","E_iso_z",
          "F_iso_t","F_iso_z",
          "G_iso_t","G_iso_z",
          "alpha_iso_t","alpha_iso_z",
         ],
)
def rhs1(params, transforms, profiles, data, **kwargs):
    
    data["rhs1"] = ( - ( - data["E_iso_z"]
                        - ( data["F_iso"] / data["E_iso"] ) * data["E_iso_t"]
                        + 2 * data["F_iso_t"] ) / ( 2 * data["E_iso"] * jnp.sqrt(data["G_iso"]
                                                                                 - data["F_iso"] ** 2 / data["E_iso"] )
                                                  )
                    - data["alpha_iso_t"] / jnp.sqrt(data["E_iso"])
                   )
        
    return data

@register_compute_fun(
    name="rhs2",
    label="rhs_{2}",
    units="~",
    units_long="~",
    description="Right hand side of first PDE for isothermic coordinates",
    dim=1,
    params=[],
    transforms={"grid":[],},
    profiles=[],
    coordinates="tz",
    data=["theta_gamma", "zeta_gamma",
          "theta_beta", "zeta_beta",
          "E_iso","F_iso","G_iso",
          "alpha_iso",
          "E_iso_t","E_iso_z",
          "F_iso_t","F_iso_z",
          "G_iso_t","G_iso_z",
          "alpha_iso_t","alpha_iso_z",
         ],
)
def rhs2(params, transforms, profiles, data, **kwargs):
    
    data["rhs2"] = ( - ( ( data["F_iso"] ** 2 / data["E_iso"] ** 2 ) * data["E_iso_t"] 
                        - 2 * ( data["F_iso"] / data["E_iso"] ) * data["F_iso_t"]  
                        + data["G_iso_t"]) / (2 * ( data["G_iso"]  - data["F_iso"] ** 2 / data["E_iso"]
                                                  ) * jnp.sqrt(data["E_iso"])
                                  ) 
                    - ( data["alpha_iso_z"]   - ( data["F_iso"] / data["E_iso"] 
                                                ) * data["alpha_iso_t"] ) / jnp.sqrt( data["G_iso"] 
                                                                                     - data["F_iso"] ** 2 / data["E_iso"] 
                                                                                    )
                   )
    return data

# Invert the matrix and find b
def find_R(data,m_size,n_size):
    
    x = jnp.ones(data["theta"].shape[0])
    fun_wrapped = lambda x: R_residual(x,data,m_size,n_size)
    #A_ = jax.jacfwd(fun_wrapped)(x)
    
    return jnp.linalg.pinv( Derivative(fun_wrapped,deriv_mode="looped").compute(x)
                          ) @ jnp.concatenate( ( data["rhs1"] , data["rhs2"]  ))

# Function to find build a matrix to find the scalar b
def R_residual(y,data,m_size,n_size):
    
    f_t = first_derivative_t(y, data,m_size,n_size)
    f_z = first_derivative_z(y, data,m_size,n_size)
    
    return jnp.concatenate( ( ( (data["theta_beta"] * f_t + data["zeta_beta"] * f_z) * jnp.cos(data["alpha_iso"]) 
                               + (data["theta_gamma"] * f_t + data["zeta_gamma"] * f_z) * jnp.sin(data["alpha_iso"])
                              ),
                             ( (data["theta_beta"] * f_t + data["zeta_beta"] * f_z) * jnp.sin(data["alpha_iso"]) 
                               - (data["theta_gamma"] * f_t + data["zeta_gamma"] * f_z) * jnp.cos(data["alpha_iso"])
                              )
                            )
                          )