from desc.backend import jnp

from .data_index import register_compute_fun


@register_compute_fun(
    name="psi_dot",
    label="\\dot{phi}",
    units="rad/s",
    units_long="Radians per second",
    description="Time derivative of psi",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["B", "|B|", "grad(psi)", "grad(B)"],
)
def _phi_dot(params, transforms, profiles, data, **kwargs):
    data["psi_dot"] = 
    return data

@register_compute_fun(
    name="theta_dot",
    label="\\dot{theta}",
    units="rad/s",
    units_long="Radians per second",
    description="Time derivative of theta",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["B", "|B|", "grad(psi)", "grad(B)"],
)
def _phi_dot(params, transforms, profiles, data, **kwargs):
    data["theta_dot"] = 
    return data

@register_compute_fun(
    name="zeta_dot",
    label="\\dot{theta}",
    units="rad/s",
    units_long="Radians per second",
    description="Time derivative of theta",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["B", "|B|", "grad(psi)", "grad(B)"],
)
def _phi_dot(params, transforms, profiles, data, **kwargs):
    data["zeta_dot"] = 
    return data
