"""Compute functions for equilibrium objectives, i.e. Force and MHD energy.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from interpax import interp1d
from scipy.constants import elementary_charge, mu_0

from desc.backend import jnp

from ..integrals.surface_integral import surface_averages
from ..utils import cross, dot, safediv, safenorm
from .data_index import register_compute_fun


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
            safediv(data["B_zeta_t"] - data["B_theta_z"], data["sqrt(g)"]),
            lambda: safediv(data["B_zeta_rt"] - data["B_theta_rz"], data["sqrt(g)_r"]),
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
            safediv(data["B_theta_r"] - data["B_rho_t"], data["sqrt(g)"]),
            lambda: safediv(data["B_theta_rr"] - data["B_rho_rt"], data["sqrt(g)_r"]),
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
    data["|J|"] = safenorm(data["J"], axis=-1)
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
    resolution_requirement="tz",
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
    data["|F|"] = safenorm(data["F"], axis=-1)
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
    resolution_requirement="rtz",
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
    name="e^helical*sqrt(g)",
    label="\\sqrt{g}(B^{\\theta} \\nabla \\zeta - B^{\\zeta} \\nabla \\theta)",
    units="T \\cdot m^{2}",
    units_long="Tesla * square meter",
    description="Helical basis vector weighted by 3-D volume Jacobian",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B^theta", "B^zeta", "e^theta*sqrt(g)", "e^zeta", "sqrt(g)"],
)
def _e_sup_helical_times_sqrt_g(params, transforms, profiles, data, **kwargs):
    data["e^helical*sqrt(g)"] = (
        data["B^zeta"] * data["e^theta*sqrt(g)"].T
        - (data["sqrt(g)"] * data["B^theta"]) * data["e^zeta"].T
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
    name="|e^helical*sqrt(g)|",
    label="|\\sqrt{g}(B^{\\theta} \\nabla \\zeta - B^{\\zeta} \\nabla \\theta)|",
    units="T \\cdot m^{2}",
    units_long="Tesla * square meter",
    description="Magnitude of helical basis vector weighted by 3-D volume Jacobian",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^helical*sqrt(g)"],
)
def _e_sup_helical_times_sqrt_g_mag(params, transforms, profiles, data, **kwargs):
    data["|e^helical*sqrt(g)|"] = jnp.linalg.norm(data["e^helical*sqrt(g)"], axis=-1)
    return data


@register_compute_fun(
    name="F_anisotropic",
    label="F_{\\mathrm{anisotropic}}",
    units="N \\cdot m^{-3}",
    units_long="Newtons / cubic meter",
    description="Anisotropic force balance error",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J", "B", "grad(beta_a)", "beta_a", "grad(|B|^2)", "grad(p)"],
)
def _F_anisotropic(params, transforms, profiles, data, **kwargs):
    data["F_anisotropic"] = (
        (1 - data["beta_a"]) * cross(data["J"], data["B"]).T
        - dot(data["B"], data["grad(beta_a)"]) * data["B"].T / mu_0
        - data["beta_a"] * data["grad(|B|^2)"].T / (2 * mu_0)
        - data["grad(p)"].T
    ).T

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
    resolution_requirement="rtz",
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
    resolution_requirement="rtz",
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
    resolution_requirement="rtz",
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
    gamma="float: Adiabatic index. Default 0",
    resolution_requirement="rtz",
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


@register_compute_fun(
    name="P_ISS04",
    label="P_{ISS04}",
    units="W",
    units_long="Watts",
    description="Heating power required by the ISS04 energy confinement time scaling",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["a", "iota", "rho", "R0", "W_p", "<ne>_vol", "<|B|>_axis"],
    method="str: Interpolation method. Default 'cubic'.",
    H_ISS04="float: ISS04 confinement enhancement factor. Default 1.",
)
def _P_ISS04(params, transforms, profiles, data, **kwargs):
    rho = transforms["grid"].compress(data["rho"], surface_label="rho")
    iota = transforms["grid"].compress(data["iota"], surface_label="rho")
    method = kwargs.get("method", "cubic")
    fx = {}
    if "iota_r" in data and method == "cubic":
        # noqa: unused dependency
        fx["fx"] = transforms["grid"].compress(data["iota_r"])
    iota_23 = interp1d(2 / 3, rho, iota, method=method, **fx)
    data["P_ISS04"] = 1e6 * (  # MW -> W
        jnp.abs(data["W_p"] / 1e6)  # J -> MJ
        / (
            0.134
            * data["a"] ** 2.28  # m
            * data["R0"] ** 0.64  # m
            * (data["<ne>_vol"] / 1e19) ** 0.54  # 1/m^3 -> 1e19/m^3
            * data["<|B|>_axis"] ** 0.84  # T
            * jnp.abs(iota_23) ** 0.41
            * kwargs.get("H_ISS04", 1)
        )
    ) ** (1 / 0.39)
    return data


@register_compute_fun(
    name="P_fusion",
    label="P_{fusion}",
    units="W",
    units_long="Watts",
    description="Fusion power",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["ni", "<sigma*nu>", "sqrt(g)"],
    resolution_requirement="rtz",
    fuel="str: Fusion fuel, assuming a 50/50 mix. One of {'DT'}. Default is 'DT'.",
)
def _P_fusion(params, transforms, profiles, data, **kwargs):
    energies = {"DT": 3.52e6 + 14.06e6}  # eV
    fuel = kwargs.get("fuel", "DT")
    energy = energies.get(fuel)

    reaction_rate = jnp.sum(
        data["ni"] ** 2
        * data["<sigma*nu>"]
        * data["sqrt(g)"]
        * transforms["grid"].weights
    )  # reactions/s
    data["P_fusion"] = reaction_rate * energy * elementary_charge  # J/s
    return data
