"""Compute functions for quasisymmetry objectives.

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


@register_compute_fun(
    name="psidot",
    label="\\dot{\\psi}",
    units="",
    units_long="",
    description="Time derivative of psi",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "B", "grad(|B|)", "grad(psi)"],
    vpar = "vpar",
    mu = "mu",
    m_q = "m_q"
)
def _psidot(params, transforms, profiles, data, **kwargs):
    m_q = kwargs.get("m_q", 1.673e-27/1.6e-19)
    mu = kwargs.get("mu")
    vpar = kwargs.get("vpar")
    data["psidot"] = (m_q*(1/(data["|B|"]**3))*(mu*data["|B|"] + vpar**2) * jnp.sum(jnp.cross(data["B"], data["grad(|B|)"], axis = -1) * data["grad(psi)"]))/params["Psi"]
    return data


@register_compute_fun(
    name="thetadot",
    label="\\dot{\\theta}",
    units="rad/s",
    units_long="radians / second",
    description="Time derivative of theta",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "B", "grad(|B|)", "e^theta"],
    vpar = "vpar",
    mu = "mu",
    m_q = "m_q"
)
def _thetadot(params, transforms, profiles, data, **kwargs):
    m_q = kwargs.get("m_q", 1.673e-27/1.6e-19)
    mu = kwargs.get("mu")
    vpar = kwargs.get("vpar")
    data["thetadot"] = (vpar/data["|B|"]) * jnp.sum(data["B"] * data["e^theta"]) + (m_q/(data["|B|"]**3))*(mu*data["|B|"] + vpar**2)*jnp.sum(jnp.cross(data["B"], data["grad(|B|)"], axis=-1) * data["e^theta"])
    return data

@register_compute_fun(
    name="zetadot",
    label="\\dot{\\zeta}",
    units="rad/s",
    units_long="radians / second",
    description="Time derivative of zeta",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "B", "e^zeta"],
    vpar = "vpar",
    mu = "mu",
    m_q = "m_q"
)
def _zetadot(params, transforms, profiles, data, **kwargs):
    m_q = kwargs.get("m_q", 1.673e-27/1.6e-19)
    mu = kwargs.get("mu")
    vpar = kwargs.get("vpar")
    data["zetadot"] = (vpar/data["|B|"]) * jnp.sum(data["B"] * data["e^zeta"]) 
    return data

@register_compute_fun(
    name="vpardot",
    label="\\dot{\\vpar}",
    units="m/s",
    units_long="meters / second",
    description="Time derivative of the parallel velocity",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "B", "grad(|B|)", "b"],
    vpar = "vpar",
    mu = "mu",
    m_q = "m_q"
)
def _vpardot(params, transforms, profiles, data, **kwargs):
    m_q = kwargs.get("m_q", 1.673e-27/1.6e-19)
    mu = kwargs.get("mu")
    vpar = kwargs.get("vpar")
    # data["vpardot"] = -mu*jnp.sum(data["b"] + m_q*(mu/data["|B|"]**2) * ((1/vpar)*jnp.cross(data["B"], data["grad(|B|)"], axis=-1).T).T * data["grad(|B|)"], axis=-1)
    
    data["vpardot"] = -mu * jnp.sum((data["b"] + (((m_q*mu)/(vpar*data["|B|"]**2))*(jnp.cross(data["B"], data["grad(|B|)"], axis=-1)).T).T) * data["grad(|B|)"], axis=-1)
    return data