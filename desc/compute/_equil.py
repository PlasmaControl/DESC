"""Compute functions for equilibrium objectives, i.e. Force and MHD energy.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from scipy.constants import mu_0

from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import dot, surface_averages


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
    axis_limit_data=["sqrt(g)_r", "B_zeta_rt", "B_theta_rz"],
    parameterization="desc.equilibrium.equilibrium.Equilibrium",
)
def _J_sup_rho(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis,
    # ∂_θ (𝐁 ⋅ 𝐞_ζ) - ∂_ζ (𝐁 ⋅ 𝐞_θ) = 𝐁 ⋅ (∂_θ 𝐞_ζ - ∂_ζ 𝐞_θ) = 0
    # because the partial derivatives commute. So 𝐉^ρ is of the indeterminate
    # form 0/0 and we may compute the limit as follows.
    data["J^rho"] = (
        transforms["grid"].replace_at_axis(
            (data["B_zeta_t"] - data["B_theta_z"]) / data["sqrt(g)"],
            lambda: (data["B_zeta_rt"] - data["B_theta_rz"]) / data["sqrt(g)_r"],
        )
    ) / mu_0
    return data


@register_compute_fun(
    name="J^theta*sqrt(g)",
    label="J^{\\theta} \\sqrt{g}",
    units="A",
    units_long="Amperes",
    description="Contravariant poloidal component of plasma current density,"
    " weighted by 3-D volume Jacobian",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B_rho_z", "B_zeta_r"],
)
def _J_sup_theta_sqrt_g(params, transforms, profiles, data, **kwargs):
    data["J^theta*sqrt(g)"] = (data["B_rho_z"] - data["B_zeta_r"]) / mu_0
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
    data=["sqrt(g)", "J^theta*sqrt(g)"],
)
def _J_sup_theta(params, transforms, profiles, data, **kwargs):
    data["J^theta"] = data["J^theta*sqrt(g)"] / data["sqrt(g)"]
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
    axis_limit_data=["sqrt(g)_r", "B_theta_rr", "B_rho_rt"],
)
def _J_sup_zeta(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis,
    # ∂ᵨ (𝐁 ⋅ 𝐞_θ) - ∂_θ (𝐁 ⋅ 𝐞ᵨ) = 𝐁 ⋅ (∂ᵨ 𝐞_θ - ∂_θ 𝐞ᵨ) = 0
    # because the partial derivatives commute. So 𝐉^ζ is of the indeterminate
    # form 0/0 and we may compute the limit as follows.
    data["J^zeta"] = (
        transforms["grid"].replace_at_axis(
            (data["B_theta_r"] - data["B_rho_t"]) / data["sqrt(g)"],
            lambda: (data["B_theta_rr"] - data["B_rho_rt"]) / data["sqrt(g)_r"],
        )
    ) / mu_0
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
    data=[
        "J^rho",
        "J^zeta",
        "J^theta*sqrt(g)",
        "e_rho",
        "e_zeta",
        "e_theta/sqrt(g)",
    ],
)
def _J(params, transforms, profiles, data, **kwargs):
    data["J"] = (
        data["J^rho"] * data["e_rho"].T
        + data["J^theta*sqrt(g)"] * data["e_theta/sqrt(g)"].T
        + data["J^zeta"] * data["e_zeta"].T
    ).T
    return data


@register_compute_fun(
    name="J*sqrt(g)",
    label="\\mathbf{J} \\sqrt{g}",
    units="A m",
    units_long="Ampere meters",
    description="Plasma current density weighted by 3-D volume Jacobian",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B_rho_z",
        "B_theta_r",
        "B_zeta_t",
        "B_rho_t",
        "B_theta_z",
        "B_zeta_r",
        "e_rho",
        "e_theta",
        "e_zeta",
    ],
)
def _J_sqrt_g(params, transforms, profiles, data, **kwargs):
    data["J*sqrt(g)"] = (
        (data["B_zeta_t"] - data["B_theta_z"]) * data["e_rho"].T
        + (data["B_rho_z"] - data["B_zeta_r"]) * data["e_theta"].T
        + (data["B_theta_r"] - data["B_rho_t"]) * data["e_zeta"].T
    ).T / mu_0
    return data


@register_compute_fun(
    name="(J*sqrt(g))_r",
    label="\\partial_{\\rho} (\\mathbf{J} \\sqrt{g})",
    units="A m",
    units_long="Ampere meters",
    description="Plasma current density weighted by 3-D volume Jacobian,"
    " radial derivative",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B_rho_z",
        "B_rho_rz",
        "B_theta_r",
        "B_theta_rr",
        "B_zeta_t",
        "B_zeta_rt",
        "B_rho_t",
        "B_rho_rt",
        "B_theta_z",
        "B_theta_rz",
        "B_zeta_r",
        "B_zeta_rr",
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_r",
        "e_theta_r",
        "e_zeta_r",
    ],
)
def _J_sqrt_g_r(params, transforms, profiles, data, **kwargs):
    data["(J*sqrt(g))_r"] = (
        (data["B_zeta_rt"] - data["B_theta_rz"]) * data["e_rho"].T
        + (data["B_zeta_t"] - data["B_theta_z"]) * data["e_rho_r"].T
        + (data["B_rho_rz"] - data["B_zeta_rr"]) * data["e_theta"].T
        + (data["B_rho_z"] - data["B_zeta_r"]) * data["e_theta_r"].T
        + (data["B_theta_rr"] - data["B_rho_rt"]) * data["e_zeta"].T
        + (data["B_theta_r"] - data["B_rho_t"]) * data["e_zeta_r"].T
    ).T / mu_0
    return data


@register_compute_fun(
    name="J_R",
    label="J_{R}",
    units="A \\cdot m^{-2}",
    units_long="Amperes / square meter",
    description="Radial component of plasma current density in lab frame",
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
    description="Toroidal component of plasma current density in lab frame",
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
    description="Vertical component of plasma current density in lab frame",
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
    description="Magnitude of plasma current density",
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
    name="J_rho",
    label="J_{\\rho}",
    units="A / m",
    units_long="Amperes / meter",
    description="Covariant radial component of plasma current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J", "e_rho"],
)
def _J_sub_rho(params, transforms, profiles, data, **kwargs):
    data["J_rho"] = dot(data["J"], data["e_rho"])
    return data


@register_compute_fun(
    name="J_theta",
    label="J_{\\theta}",
    units="A / m",
    units_long="Amperes / meter",
    description="Covariant poloidal component of plasma current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J", "e_theta"],
)
def _J_sub_theta(params, transforms, profiles, data, **kwargs):
    data["J_theta"] = dot(data["J"], data["e_theta"])
    return data


@register_compute_fun(
    name="J_zeta",
    label="J_{\\zeta}",
    units="A / m",
    units_long="Amperes / meter",
    description="Covariant toroidal component of plasma current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J", "e_zeta"],
)
def _J_sub_zeta(params, transforms, profiles, data, **kwargs):
    data["J_zeta"] = dot(data["J"], data["e_zeta"])
    return data


@register_compute_fun(
    name="J*B",
    label="\\mathbf{J} \\cdot \\mathbf{B}",
    units="N / m^{3}",
    units_long="Newtons / cubic meter",
    description="Current density parallel to magnetic field, times field strength "
    + "(note units are not Amperes)",
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
    name="<J*B>",
    label="\\langle \\mathbf{J} \\cdot \\mathbf{B} \\rangle",
    units="N / m^{3}",
    units_long="Newtons / cubic meter",
    description="Flux surface average of current density dotted into magnetic field "
    + "(note units are not Amperes)",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["J*sqrt(g)", "B", "V_r(r)"],
    axis_limit_data=["(J*sqrt(g))_r", "V_rr(r)"],
)
def _J_dot_B_fsa(params, transforms, profiles, data, **kwargs):
    J = transforms["grid"].replace_at_axis(
        data["J*sqrt(g)"], lambda: data["(J*sqrt(g))_r"], copy=True
    )
    data["<J*B>"] = surface_averages(
        transforms["grid"],
        dot(J, data["B"]),  # sqrt(g) factor pushed into J
        denominator=transforms["grid"].replace_at_axis(
            data["V_r(r)"], lambda: data["V_rr(r)"], copy=True
        ),
    )
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
    data=["p_r", "(curl(B)xB)_rho"],
)
def _F_rho(params, transforms, profiles, data, **kwargs):
    data["F_rho"] = data["(curl(B)xB)_rho"] / mu_0 - data["p_r"]
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
    data=["F_helical", "B^zeta"],
)
def _F_theta(params, transforms, profiles, data, **kwargs):
    data["F_theta"] = -data["B^zeta"] * data["F_helical"]
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
    data=["B^theta", "F_helical"],
)
def _F_zeta(params, transforms, profiles, data, **kwargs):
    data["F_zeta"] = data["B^theta"] * data["F_helical"]
    return data


@register_compute_fun(
    name="F_helical",
    label="F_{\\mathrm{helical}}",
    units="A",
    units_long="Amperes",
    description="Covariant helical component of force balance error",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B_zeta_t", "B_theta_z"],
)
def _F_helical(params, transforms, profiles, data, **kwargs):
    data["F_helical"] = (data["B_zeta_t"] - data["B_theta_z"]) / mu_0
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
    data=["F_rho", "F_zeta", "e^rho", "e^zeta", "B^zeta", "J^rho", "e^theta*sqrt(g)"],
)
def _F(params, transforms, profiles, data, **kwargs):
    # F_theta e^theta refactored as below to resolve indeterminacy at axis.
    data["F"] = (
        data["F_rho"] * data["e^rho"].T
        - data["B^zeta"] * data["J^rho"] * data["e^theta*sqrt(g)"].T
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
        jnp.sum(data["|F|"] * data["sqrt(g)"] * transforms["grid"].weights) / data["V"]
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
def _e_sup_helical(params, transforms, profiles, data, **kwargs):
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
def _e_sup_helical_mag(params, transforms, profiles, data, **kwargs):
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
        data["|B|"] ** 2 * data["sqrt(g)"] * transforms["grid"].weights
    ) / (2 * mu_0)
    return data


@register_compute_fun(
    name="W_Bpol",
    label="W_{B,pol}",
    units="J",
    units_long="Joules",
    description="Plasma magnetic energy in poloidal field",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["B", "sqrt(g)"],
)
def _W_Bpol(params, transforms, profiles, data, **kwargs):
    data["W_Bpol"] = jnp.sum(
        dot(data["B"][:, (0, 2)], data["B"][:, (0, 2)])
        * data["sqrt(g)"]
        * transforms["grid"].weights
    ) / (2 * mu_0)
    return data


@register_compute_fun(
    name="W_Btor",
    label="W_{B,tor}",
    units="J",
    units_long="Joules",
    description="Plasma magnetic energy in toroidal field",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["B", "sqrt(g)"],
)
def _W_Btor(params, transforms, profiles, data, **kwargs):
    data["W_Btor"] = jnp.sum(
        data["B"][:, 1] ** 2 * data["sqrt(g)"] * transforms["grid"].weights
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
    data["W_p"] = jnp.sum(data["p"] * data["sqrt(g)"] * transforms["grid"].weights) / (
        kwargs.get("gamma", 0) - 1
    )
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
    data["<beta>_vol"] = jnp.abs(data["W_p"] / data["W_B"])
    return data


@register_compute_fun(
    name="<beta_pol>_vol",
    label="\\langle \\beta_{pol} \\rangle_{vol}",
    units="~",
    units_long="None",
    description="Normalized poloidal plasma pressure",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["W_p", "W_Bpol"],
)
def _beta_volpol(params, transforms, profiles, data, **kwargs):
    data["<beta_pol>_vol"] = jnp.abs(data["W_p"] / data["W_Bpol"])
    return data


@register_compute_fun(
    name="<beta_tor>_vol",
    label="\\langle \\beta_{tor} \\rangle_{vol}",
    units="~",
    units_long="None",
    description="Normalized toroidal plasma pressure",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["W_p", "W_Btor"],
)
def _beta_voltor(params, transforms, profiles, data, **kwargs):
    data["<beta_tor>_vol"] = jnp.abs(data["W_p"] / data["W_Btor"])
    return data
