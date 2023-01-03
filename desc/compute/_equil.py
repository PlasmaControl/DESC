"""Compute functions for equilibrium objectives, ie Force and MHD energy."""

from scipy.constants import mu_0

from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import dot


@register_compute_fun(
    name="J^rho",
    label="J^{\\rho}",
    units="A \\cdot m^{-3}",
    units_long="Amperes / cubic meter",
    description="Contravariant radial component of plasma current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)", "B_zeta_t", "B_theta_z"],
)
def _J_sup_rho(params, transforms, profiles, data, **kwargs):
    data["J^rho"] = (data["B_zeta_t"] - data["B_theta_z"]) / (mu_0 * data["sqrt(g)"])
    return data


@register_compute_fun(
    name="J^theta",
    label="J^{\\theta}",
    units="A \\cdot m^{-3}",
    units_long="Amperes / cubic meter",
    description="Contravariant poloidal component of plasma current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)", "B_rho_z", "B_zeta_r"],
)
def _J_sup_theta(params, transforms, profiles, data, **kwargs):
    data["J^theta"] = (data["B_rho_z"] - data["B_zeta_r"]) / (mu_0 * data["sqrt(g)"])
    return data


@register_compute_fun(
    name="J^zeta",
    label="J^{\\zeta}",
    units="A \\cdot m^{-3}",
    units_long="Amperes / cubic meter",
    description="Contravariant toroidal component of plasma current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)", "B_theta_r", "B_rho_t"],
)
def _J_sup_zeta(params, transforms, profiles, data, **kwargs):
    data["J^zeta"] = (data["B_theta_r"] - data["B_rho_t"]) / (mu_0 * data["sqrt(g)"])
    return data


@register_compute_fun(
    name="J",
    label="\\mathbf{J}",
    units="A \\cdot m^{-2}",
    units_long="Amperes / square meter",
    description="Plasma current density",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J^rho", "J^theta", "J^zeta", "e_rho", "e_theta", "e_zeta"],
)
def _J(params, transforms, profiles, data, **kwargs):
    data["J"] = (
        data["J^rho"] * data["e_rho"].T
        + data["J^theta"] * data["e_theta"].T
        + data["J^zeta"] * data["e_zeta"].T
    ).T
    return data


@register_compute_fun(
    name="J_R",
    label="J_{R}",
    units="A \\cdot m^{-2}",
    units_long="Amperes / square meter",
    description="Radial componenet of plasma current density in lab frame",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J"],
)
def _J_R(params, transforms, profiles, data, **kwargs):
    data["J_R"] = data["J"][:, 0]
    return data


@register_compute_fun(
    name="J_phi",
    label="J_{\\phi}",
    units="A \\cdot m^{-2}",
    units_long="Amperes / square meter",
    description="Toroidal componenet of plasma current density in lab frame",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J"],
)
def _J_phi(params, transforms, profiles, data, **kwargs):
    data["J_phi"] = data["J"][:, 1]
    return data


@register_compute_fun(
    name="J_Z",
    label="J_{Z}",
    units="A \\cdot m^{-2}",
    units_long="Amperes / square meter",
    description="Vertical componenet of plasma current density in lab frame",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J"],
)
def _J_Z(params, transforms, profiles, data, **kwargs):
    data["J_Z"] = data["J"][:, 2]
    return data


@register_compute_fun(
    name="|J|",
    label="|\\mathbf{J}|",
    units="A \\cdot m^{-2}",
    units_long="Amperes / square meter",
    description="Magnitue of plasma current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J"],
)
def _Jmag(params, transforms, profiles, data, **kwargs):
    data["|J|"] = jnp.linalg.norm(data["J"], axis=-1)
    return data


@register_compute_fun(
    name="J*B",
    label="\\mathbf{J} \\cdot \\mathbf{B}",
    units="N / m^{3}",
    units_long="Newtons / cubic meter",
    description="Bootstrap current (note units are not Amperes)",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J", "B"],
)
def _J_dot_B(params, transforms, profiles, data, **kwargs):
    data["J*B"] = dot(data["J"], data["B"])
    return data


@register_compute_fun(
    name="J_parallel",
    label="\\mathbf{J} \\cdot \\hat{\\mathbf{b}}",
    units="A \\cdot m^{-2}",
    units_long="Amperes / square meter",
    description="Plasma current density parallel to magnetic field",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J*B", "|B|"],
)
def _J_parallel(params, transforms, profiles, data, **kwargs):
    data["J_parallel"] = data["J*B"] / data["|B|"]
    return data


@register_compute_fun(
    name="F_rho",
    label="F_{\\rho}",
    units="N \\cdot m^{-2}",
    units_long="Newtons / square meter",
    description="Covariant radial component of force balance error",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["p_r", "sqrt(g)", "B^theta", "B^zeta", "J^theta", "J^zeta"],
)
def _F_rho(params, transforms, profiles, data, **kwargs):
    data["F_rho"] = -data["p_r"] + data["sqrt(g)"] * (
        data["B^zeta"] * data["J^theta"] - data["B^theta"] * data["J^zeta"]
    )
    return data


@register_compute_fun(
    name="F_theta",
    label="F_{\\theta}",
    units="N \\cdot m^{-2}",
    units_long="Newtons / square meter",
    description="Covariant poloidal component of force balance error",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)", "B^zeta", "J^rho"],
)
def _F_theta(params, transforms, profiles, data, **kwargs):
    data["F_theta"] = -data["sqrt(g)"] * data["B^zeta"] * data["J^rho"]
    return data


@register_compute_fun(
    name="F_zeta",
    label="F_{\\zeta}",
    units="N \\cdot m^{-2}",
    units_long="Newtons / square meter",
    description="Covariant toroidal component of force balance error",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)", "B^theta", "J^rho"],
)
def _F_zeta(params, transforms, profiles, data, **kwargs):
    data["F_zeta"] = data["sqrt(g)"] * data["B^theta"] * data["J^rho"]
    return data


@register_compute_fun(
    name="F_helical",
    label="F_{helical}",
    units="N \\cdot m^{-2}",
    units_long="Newtons / square meter",
    description="Covariant helical component of force balance error",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)", "J^rho"],
)
def _F_helical(params, transforms, profiles, data, **kwargs):
    data["F_helical"] = data["sqrt(g)"] * data["J^rho"]
    return data


@register_compute_fun(
    name="F",
    label="\\mathbf{J} \\times \\mathbf{B} - \\nabla p",
    units="N \\cdot m^{-3}",
    units_long="Newtons / cubic meter",
    description="Force balance error",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["F_rho", "F_theta", "F_zeta", "e^rho", "e^theta", "e^zeta"],
)
def _F(params, transforms, profiles, data, **kwargs):
    data["F"] = (
        data["F_rho"] * data["e^rho"].T
        + data["F_theta"] * data["e^theta"].T
        + data["F_zeta"] * data["e^zeta"].T
    ).T
    return data


@register_compute_fun(
    name="|F|",
    label="|\\mathbf{J} \\times \\mathbf{B} - \\nabla p|",
    units="N \\cdot m^{-3}",
    units_long="Newtons / cubic meter",
    description="Magnitude of force balance error",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["F"],
)
def _Fmag(params, transforms, profiles, data, **kwargs):
    data["|F|"] = jnp.linalg.norm(data["F"], axis=-1)
    return data


@register_compute_fun(
    name="<|F|>_vol",
    label="\\langle |\\mathbf{J} \\times \\mathbf{B} - \\nabla p| \\rangle_{vol}",
    units="N \\cdot m^{-3}",
    units_long="Newtons / cubic meter",
    description="Volume average of magnitude of force balance error",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["|F|", "sqrt(g)", "V"],
)
def _Fmag_vol(params, transforms, profiles, data, **kwargs):
    data["<|F|>_vol"] = (
        jnp.sum(data["|F|"] * jnp.abs(data["sqrt(g)"]) * transforms["grid"].weights)
        / data["V"]
    )
    return data


@register_compute_fun(
    name="e^helical",
    label="B^{\\theta} \\nabla \\zeta - B^{\\zeta} \\nabla \\theta",
    units="T \\cdot m^{-2}",
    units_long="Tesla / square meter",
    description="Helical basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B^theta", "B^zeta", "e^theta", "e^zeta"],
)
def _e_helical(params, transforms, profiles, data, **kwargs):
    data["e^helical"] = (
        data["B^zeta"] * data["e^theta"].T - data["B^theta"] * data["e^zeta"].T
    ).T
    return data


@register_compute_fun(
    name="|e^helical|",
    label="|B^{\\theta} \\nabla \\zeta - B^{\\zeta} \\nabla \\theta|",
    units="T \\cdot m^{-2}",
    units_long="Tesla / square meter",
    description="Magnitude of helical basis vector",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^helical"],
)
def _helical_mag(params, transforms, profiles, data, **kwargs):
    data["|e^helical|"] = jnp.linalg.norm(data["e^helical"], axis=-1)
    return data


@register_compute_fun(
    name="W_B",
    label="W_B",
    units="J",
    units_long="Joules",
    description="Plasma magnetic energy",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["|B|", "sqrt(g)"],
)
def _W_B(params, transforms, profiles, data, **kwargs):
    data["W_B"] = jnp.sum(
        data["|B|"] ** 2 * jnp.abs(data["sqrt(g)"]) * transforms["grid"].weights
    ) / (2 * mu_0)
    return data


@register_compute_fun(
    name="W_p",
    label="W_p",
    units="J",
    units_long="Joules",
    description="Plasma thermodynamic energy",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["p", "sqrt(g)"],
    gamma="gamma",
)
def _W_p(params, transforms, profiles, data, **kwargs):
    data["W_p"] = jnp.sum(
        data["p"] * jnp.abs(data["sqrt(g)"]) * transforms["grid"].weights
    ) / (kwargs.get("gamma", 0) - 1)
    return data


@register_compute_fun(
    name="W",
    label="W",
    units="J",
    units_long="Joules",
    description="Plasma total energy",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["W_B", "W_p"],
)
def _W(params, transforms, profiles, data, **kwargs):
    data["W"] = data["W_B"] + data["W_p"]
    return data


@register_compute_fun(
    name="<beta>_vol",
    label="\\langle \\beta \\rangle_{vol}",
    units="~",
    units_long="None",
    description="Normalized plasma pressure",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["W_p", "W_B"],
)
def _beta_vol(params, transforms, profiles, data, **kwargs):
    data["<beta>_vol"] = data["W_p"] / data["W_B"]
    return data
