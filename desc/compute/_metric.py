"""Compute functions related to the metric tensor of the coordinate system.

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
from .utils import cross, dot, safediv, safenorm


@register_compute_fun(
    name="sqrt(g)",
    label="\\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_theta", "e_zeta"],
)
def _sqrtg(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)"] = dot(data["e_rho"], cross(data["e_theta"], data["e_zeta"]))
    return data


@register_compute_fun(
    name="sqrt(g)_PEST",
    label="\\sqrt{g}_{PEST}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of (ρ,ϑ,ϕ) coordinate system or"
    " straight field line PEST coordinates. ϕ increases counterclockwise"
    " when viewed from above (cylindrical R,ϕ plane with Z out of page).",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)", "theta_PEST_t", "phi_z", "theta_PEST_z", "phi_t"],
)
def _sqrtg_pest(params, transforms, profiles, data, **kwargs):
    # Same as dot(data["e_rho|v,p"], cross(data["e_vartheta"], data["e_phi|r,v"])), but
    # more efficient as it avoids computing radial derivatives of the stream functions.
    data["sqrt(g)_PEST"] = data["sqrt(g)"] / (
        data["theta_PEST_t"] * data["phi_z"] - data["theta_PEST_z"] * data["phi_t"]
    )
    return data


@register_compute_fun(
    name="|e_theta x e_zeta|",
    label="|e_{\\theta} \\times e_{\\zeta}|",
    units="m^{2}",
    units_long="square meters",
    description="2D Jacobian determinant for constant rho surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_zeta"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _e_theta_x_e_zeta(params, transforms, profiles, data, **kwargs):
    data["|e_theta x e_zeta|"] = safenorm(
        cross(data["e_theta"], data["e_zeta"]), axis=-1
    )
    return data


@register_compute_fun(
    name="|e_theta x e_zeta|_r",
    label="\\partial_{\\rho} |e_{\\theta} \\times e_{\\zeta}|",
    units="m^{2}",
    units_long="square meters",
    description="2D Jacobian determinant for constant rho surface"
    + " derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_zeta", "e_theta_r", "e_zeta_r"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _e_theta_x_e_zeta_r(params, transforms, profiles, data, **kwargs):
    a = cross(data["e_theta"], data["e_zeta"])
    a_r = cross(data["e_theta_r"], data["e_zeta"]) + cross(
        data["e_theta"], data["e_zeta_r"]
    )
    # The limit of a sequence and the norm function can be interchanged
    # because norms are continuous functions. Likewise with dot product.
    # Then lim ‖𝐞^ρ‖ = ‖ lim 𝐞^ρ ‖ ≠ 0
    # lim (𝐞^ρ ⋅ a_r / ‖𝐞^ρ‖) = lim 𝐞^ρ ⋅ lim a_r / lim ‖𝐞^ρ‖
    # The vectors converge to be parallel.
    data["|e_theta x e_zeta|_r"] = transforms["grid"].replace_at_axis(
        safediv(dot(a, a_r), safenorm(a, axis=-1)), lambda: safenorm(a_r, axis=-1)
    )
    return data


@register_compute_fun(
    name="|e_theta x e_zeta|_rr",
    label="\\partial_{\\rho \\rho} |e_{\\theta} \\times e_{\\zeta}|",
    units="m^{2}",
    units_long="square meters",
    description="2D Jacobian determinant for constant rho surface"
    + " second derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_zeta", "e_theta_r", "e_zeta_r", "e_theta_rr", "e_zeta_rr"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _e_theta_x_e_zeta_rr(params, transforms, profiles, data, **kwargs):
    a = cross(data["e_theta"], data["e_zeta"])
    a_r = cross(data["e_theta_r"], data["e_zeta"]) + cross(
        data["e_theta"], data["e_zeta_r"]
    )
    a_rr = (
        cross(data["e_theta_rr"], data["e_zeta"])
        + 2 * cross(data["e_theta_r"], data["e_zeta_r"])
        + cross(data["e_theta"], data["e_zeta_rr"])
    )
    norm_a = safenorm(a, axis=-1)
    norm_a_r = safenorm(a_r, axis=-1)
    # The limit eventually reduces to a form where the technique used to compute
    # lim |e_theta x e_zeta|_r can be applied.
    data["|e_theta x e_zeta|_rr"] = transforms["grid"].replace_at_axis(
        safediv(norm_a_r**2 + dot(a, a_rr) - safediv(dot(a, a_r), norm_a) ** 2, norm_a),
        lambda: safediv(dot(a_r, a_rr), norm_a_r),
    )
    return data


@register_compute_fun(
    name="|e_theta x e_zeta|_z",
    label="\\partial_{\\zeta}|e_{\\theta} \\times e_{\\zeta}|",
    units="m^{2}",
    units_long="square meters",
    description="2D Jacobian determinant for constant rho surface,"
    "derivative wrt toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_theta_z", "e_zeta", "e_zeta_z", "|e_theta x e_zeta|"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _e_theta_x_e_zeta_z(params, transforms, profiles, data, **kwargs):
    data["|e_theta x e_zeta|_z"] = dot(
        (
            cross(data["e_theta_z"], data["e_zeta"])
            + cross(data["e_theta"], data["e_zeta_z"])
        ),
        cross(data["e_theta"], data["e_zeta"]),
    ) / (data["|e_theta x e_zeta|"])
    return data


@register_compute_fun(
    name="|e_zeta x e_rho|",
    label="|e_{\\zeta} \\times e_{\\rho}|",
    units="m^{2}",
    units_long="square meters",
    description="2D Jacobian determinant for constant theta surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^theta*sqrt(g)"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _e_zeta_x_e_rho(params, transforms, profiles, data, **kwargs):
    data["|e_zeta x e_rho|"] = safenorm(data["e^theta*sqrt(g)"], axis=-1)
    return data


@register_compute_fun(
    name="|e_rho x e_theta|",
    label="|e_{\\rho} \\times e_{\\theta}|",
    units="m^{2}",
    units_long="square meters",
    description="2D Jacobian determinant for constant zeta surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_theta"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _e_rho_x_e_theta(params, transforms, profiles, data, **kwargs):
    data["|e_rho x e_theta|"] = safenorm(cross(data["e_rho"], data["e_theta"]), axis=-1)
    return data


@register_compute_fun(
    name="|e_rho x e_theta|_r",
    label="\\partial_{\\rho} |e_{\\rho} \\times e_{\\theta}|",
    units="m^{2}",
    units_long="square meters",
    description="2D Jacobian determinant for constant zeta surface"
    " derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_theta", "e_rho_r", "e_theta_r"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _e_rho_x_e_theta_r(params, transforms, profiles, data, **kwargs):
    a = cross(data["e_rho"], data["e_theta"])
    a_r = cross(data["e_rho_r"], data["e_theta"]) + cross(
        data["e_rho"], data["e_theta_r"]
    )
    # The limit of a sequence and the norm function can be interchanged
    # because norms are continuous functions. Likewise with dot product.
    # Then lim ‖𝐞^ζ‖ = ‖ lim 𝐞^ζ ‖ ≠ 0
    # lim (𝐞^ζ ⋅ a_r / ‖𝐞^ζ‖) = lim 𝐞^ζ ⋅ lim a_r / lim ‖𝐞^ζ‖
    # The vectors converge to be parallel.
    data["|e_rho x e_theta|_r"] = transforms["grid"].replace_at_axis(
        safediv(dot(a, a_r), safenorm(a, axis=-1)), lambda: safenorm(a_r, axis=-1)
    )
    return data


@register_compute_fun(
    name="|e_rho x e_theta|_rr",
    label="\\partial_{\\rho \\rho} |e_{\\rho} \\times e_{\\theta}|",
    units="m^{2}",
    units_long="square meters",
    description="2D Jacobian determinant for constant zeta surface"
    + " second derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_theta", "e_rho_r", "e_theta_r", "e_rho_rr", "e_theta_rr"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _e_rho_x_e_theta_rr(params, transforms, profiles, data, **kwargs):
    a = cross(data["e_rho"], data["e_theta"])
    a_r = cross(data["e_rho_r"], data["e_theta"]) + cross(
        data["e_rho"], data["e_theta_r"]
    )
    a_rr = (
        cross(data["e_rho_rr"], data["e_theta"])
        + 2 * cross(data["e_rho_r"], data["e_theta_r"])
        + cross(data["e_rho"], data["e_theta_rr"])
    )
    norm_a = safenorm(a, axis=-1)
    norm_a_r = safenorm(a_r, axis=-1)
    # The limit eventually reduces to a form where the technique used to compute
    # lim |e_rho x e_theta|_r can be applied.
    data["|e_rho x e_theta|_rr"] = transforms["grid"].replace_at_axis(
        safediv(norm_a_r**2 + dot(a, a_rr) - safediv(dot(a, a_r), norm_a) ** 2, norm_a),
        lambda: safediv(dot(a_r, a_rr), norm_a_r),
    )
    return data


@register_compute_fun(
    name="sqrt(g)_r",
    label="\\partial_{\\rho} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, derivative wrt "
    + "radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_theta", "e_zeta", "e_rho_r", "e_theta_r", "e_zeta_r"],
)
def _sqrtg_r(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_r"] = (
        dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_r"]))
    )
    return data


@register_compute_fun(
    name="sqrt(g)_t",
    label="\\partial_{\\theta} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, derivative wrt "
    + "poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_theta", "e_zeta", "e_rho_t", "e_theta_t", "e_zeta_t"],
)
def _sqrtg_t(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_t"] = (
        dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_t"]))
    )
    return data


@register_compute_fun(
    name="sqrt(g)_z",
    label="\\partial_{\\zeta} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, derivative wrt "
    + "toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_theta", "e_zeta", "e_rho_z", "e_theta_z", "e_zeta_z"],
)
def _sqrtg_z(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_z"] = (
        dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_z"]))
    )
    return data


@register_compute_fun(
    name="sqrt(g)_rr",
    label="\\partial_{\\rho\\rho} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, second derivative wrt "
    + "radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_r",
        "e_theta_r",
        "e_zeta_r",
        "e_rho_rr",
        "e_theta_rr",
        "e_zeta_rr",
    ],
)
def _sqrtg_rr(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_rr"] = (
        dot(data["e_rho_rr"], cross(data["e_theta"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_rr"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_rr"]))
        + 2 * dot(data["e_rho_r"], cross(data["e_theta_r"], data["e_zeta"]))
        + 2 * dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta_r"]))
        + 2 * dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta_r"]))
    )
    return data


@register_compute_fun(
    name="sqrt(g)_rrr",
    label="\\partial_{\\rho\\rho\\rho} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, third derivative wrt "
    + "radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_r",
        "e_theta_r",
        "e_zeta_r",
        "e_rho_rr",
        "e_theta_rr",
        "e_zeta_rr",
        "e_rho_rrr",
        "e_theta_rrr",
        "e_zeta_rrr",
    ],
)
def _sqrtg_rrr(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_rrr"] = (
        dot(data["e_rho_rrr"], cross(data["e_theta"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_rrr"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_rrr"]))
        + 3 * dot(data["e_rho_rr"], cross(data["e_theta_r"], data["e_zeta"]))
        + 3 * dot(data["e_rho_rr"], cross(data["e_theta"], data["e_zeta_r"]))
        + 3 * dot(data["e_rho_r"], cross(data["e_theta_rr"], data["e_zeta"]))
        + 3 * dot(data["e_rho"], cross(data["e_theta_rr"], data["e_zeta_r"]))
        + 3 * dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta_rr"]))
        + 3 * dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta_rr"]))
        + 6 * dot(data["e_rho_r"], cross(data["e_theta_r"], data["e_zeta_r"]))
    )
    return data


@register_compute_fun(
    name="sqrt(g)_rrt",
    label="\\partial_{\\rho\\rho\\theta} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, third derivative wrt "
    + "radial coordinate twice and poloidal angle once",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_r",
        "e_theta_r",
        "e_zeta_r",
        "e_rho_t",
        "e_theta_t",
        "e_zeta_t",
        "e_rho_rt",
        "e_theta_rt",
        "e_zeta_rt",
        "e_rho_rr",
        "e_theta_rr",
        "e_zeta_rr",
        "e_rho_rrt",
        "e_theta_rrt",
        "e_zeta_rrt",
    ],
)
def _sqrtg_rrt(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_rrt"] = (
        dot(data["e_rho_rrt"], cross(data["e_theta"], data["e_zeta"]))
        + dot(
            data["e_rho_rr"],
            cross(data["e_theta_t"], data["e_zeta"])
            + cross(data["e_theta"], data["e_zeta_t"]),
        )
        + 2
        * dot(
            data["e_rho_rt"],
            cross(data["e_theta_r"], data["e_zeta"])
            + cross(data["e_theta"], data["e_zeta_r"]),
        )
        + 2
        * dot(
            data["e_rho_r"],
            cross(data["e_theta_rt"], data["e_zeta"])
            + cross(data["e_theta_r"], data["e_zeta_t"])
            + cross(data["e_theta_t"], data["e_zeta_r"])
            + cross(data["e_theta"], data["e_zeta_rt"]),
        )
        + dot(
            data["e_rho_t"],
            cross(data["e_theta_rr"], data["e_zeta"])
            + cross(data["e_theta"], data["e_zeta_rr"]),
        )
        + dot(
            data["e_rho"],
            cross(data["e_theta_rrt"], data["e_zeta"])
            + 2 * cross(data["e_theta_rt"], data["e_zeta_r"])
            + cross(data["e_theta_rr"], data["e_zeta_t"])
            + 2 * cross(data["e_theta_r"], data["e_zeta_rt"])
            + cross(data["e_theta_t"], data["e_zeta_rr"])
            + cross(data["e_theta"], data["e_zeta_rrt"]),
        )
    )
    return data


@register_compute_fun(
    name="sqrt(g)_tt",
    label="\\partial_{\\theta\\theta} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, second derivative wrt "
    + "poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_t",
        "e_theta_t",
        "e_zeta_t",
        "e_rho_tt",
        "e_theta_tt",
        "e_zeta_tt",
    ],
)
def _sqrtg_tt(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_tt"] = (
        dot(data["e_rho_tt"], cross(data["e_theta"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_tt"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_tt"]))
        + 2 * dot(data["e_rho_t"], cross(data["e_theta_t"], data["e_zeta"]))
        + 2 * dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta_t"]))
        + 2 * dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta_t"]))
    )
    return data


@register_compute_fun(
    name="sqrt(g)_rtt",
    label="\\partial_{\\rho\\theta\\theta} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, third derivative wrt"
    + " radial coordinate once and poloidal angle twice.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_r",
        "e_theta_r",
        "e_zeta_r",
        "e_rho_t",
        "e_theta_t",
        "e_zeta_t",
        "e_rho_rt",
        "e_theta_rt",
        "e_zeta_rt",
        "e_rho_tt",
        "e_theta_tt",
        "e_zeta_tt",
        "e_rho_rtt",
        "e_theta_rtt",
        "e_zeta_rtt",
    ],
)
def _sqrtg_rtt(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_rtt"] = (
        dot(data["e_rho_rtt"], cross(data["e_theta"], data["e_zeta"]))
        + dot(data["e_rho_r"], cross(data["e_theta_tt"], data["e_zeta"]))
        + dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta_tt"]))
        + 2 * dot(data["e_rho_rt"], cross(data["e_theta_t"], data["e_zeta"]))
        + 2 * dot(data["e_rho_rt"], cross(data["e_theta"], data["e_zeta_t"]))
        + 2 * dot(data["e_rho_r"], cross(data["e_theta_t"], data["e_zeta_t"]))
        + dot(data["e_rho_tt"], cross(data["e_theta_r"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_rtt"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta_tt"]))
        + 2 * dot(data["e_rho_t"], cross(data["e_theta_rt"], data["e_zeta"]))
        + 2 * dot(data["e_rho"], cross(data["e_theta_rt"], data["e_zeta_t"]))
        + dot(data["e_rho_tt"], cross(data["e_theta"], data["e_zeta_r"]))
        + dot(data["e_rho"], cross(data["e_theta_tt"], data["e_zeta_r"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_rtt"]))
        + 2 * dot(data["e_rho_t"], cross(data["e_theta_t"], data["e_zeta_r"]))
        + 2 * dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta_rt"]))
        + 2 * dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta_rt"]))
    )
    return data


@register_compute_fun(
    name="sqrt(g)_zz",
    label="\\partial_{\\zeta\\zeta} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, second derivative wrt "
    + "toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_z",
        "e_theta_z",
        "e_zeta_z",
        "e_rho_zz",
        "e_theta_zz",
        "e_zeta_zz",
    ],
)
def _sqrtg_zz(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_zz"] = (
        dot(data["e_rho_zz"], cross(data["e_theta"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_zz"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_zz"]))
        + 2 * dot(data["e_rho_z"], cross(data["e_theta_z"], data["e_zeta"]))
        + 2 * dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta_z"]))
        + 2 * dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta_z"]))
    )
    return data


@register_compute_fun(
    name="sqrt(g)_rzz",
    label="\\partial_{\\rho\\zeta\\zeta} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, third derivative wrt "
    + "radial coordinate once and toroidal angle twice",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_z",
        "e_theta_z",
        "e_zeta_z",
        "e_rho_zz",
        "e_theta_zz",
        "e_zeta_zz",
        "e_rho_r",
        "e_theta_r",
        "e_zeta_r",
        "e_rho_rz",
        "e_theta_rz",
        "e_zeta_rz",
        "e_rho_rzz",
        "e_theta_rzz",
        "e_zeta_rzz",
    ],
)
def _sqrtg_rzz(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_rzz"] = (
        dot(data["e_rho_rzz"], cross(data["e_theta"], data["e_zeta"]))
        + dot(data["e_rho_r"], cross(data["e_theta_zz"], data["e_zeta"]))
        + dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta_zz"]))
        + 2 * dot(data["e_rho_rz"], cross(data["e_theta_z"], data["e_zeta"]))
        + 2 * dot(data["e_rho_rz"], cross(data["e_theta"], data["e_zeta_z"]))
        + 2 * dot(data["e_rho_r"], cross(data["e_theta_z"], data["e_zeta_z"]))
        + dot(data["e_rho_zz"], cross(data["e_theta_r"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_rzz"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta_zz"]))
        + 2 * dot(data["e_rho_z"], cross(data["e_theta_rz"], data["e_zeta"]))
        + 2 * dot(data["e_rho_z"], cross(data["e_theta_r"], data["e_zeta_z"]))
        + 2 * dot(data["e_rho"], cross(data["e_theta_rz"], data["e_zeta_z"]))
        + dot(data["e_rho_zz"], cross(data["e_theta"], data["e_zeta_r"]))
        + dot(data["e_rho"], cross(data["e_theta_zz"], data["e_zeta_r"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_rzz"]))
        + 2 * dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta_rz"]))
        + 2 * dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta_rz"]))
    )
    return data


@register_compute_fun(
    name="sqrt(g)_rt",
    label="\\partial_{\\rho\\theta} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, second derivative wrt "
    + "radial coordinate and poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_r",
        "e_theta_r",
        "e_zeta_r",
        "e_rho_t",
        "e_theta_t",
        "e_zeta_t",
        "e_rho_rt",
        "e_theta_rt",
        "e_zeta_rt",
    ],
)
def _sqrtg_rt(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_rt"] = (
        dot(data["e_rho_rt"], cross(data["e_theta"], data["e_zeta"]))
        + dot(data["e_rho_r"], cross(data["e_theta_t"], data["e_zeta"]))
        + dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta_t"]))
        + dot(data["e_rho"], cross(data["e_theta_rt"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta_t"]))
        + dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta_r"]))
        + dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta_r"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_rt"]))
    )
    return data


@register_compute_fun(
    name="sqrt(g)_tz",
    label="\\partial_{\\theta\\zeta} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, second derivative wrt "
    + "poloidal and toroidal angles",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_t",
        "e_theta_t",
        "e_zeta_t",
        "e_rho_z",
        "e_theta_z",
        "e_zeta_z",
        "e_rho_tz",
        "e_theta_tz",
        "e_zeta_tz",
    ],
)
def _sqrtg_tz(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_tz"] = (
        dot(data["e_rho_tz"], cross(data["e_theta"], data["e_zeta"]))
        + dot(data["e_rho_z"], cross(data["e_theta_t"], data["e_zeta"]))
        + dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta_t"]))
        + dot(data["e_rho_t"], cross(data["e_theta_z"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_tz"], data["e_zeta"]))
        + dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta_z"]))
        + dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta_z"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_tz"]))
    )
    return data


@register_compute_fun(
    name="sqrt(g)_rtz",
    label="\\partial_{\\rho\\theta\\zeta} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, third derivative wrt "
    + "radial, poloidal, and toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_r",
        "e_theta_r",
        "e_zeta_r",
        "e_rho_t",
        "e_theta_t",
        "e_zeta_t",
        "e_rho_rt",
        "e_theta_rt",
        "e_zeta_rt",
        "e_rho_z",
        "e_theta_z",
        "e_zeta_z",
        "e_rho_rz",
        "e_theta_rz",
        "e_zeta_rz",
        "e_rho_tz",
        "e_theta_tz",
        "e_zeta_tz",
        "e_rho_rtz",
        "e_theta_rtz",
        "e_zeta_rtz",
    ],
)
def _sqrtg_rtz(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_rtz"] = (
        dot(data["e_rho_rtz"], cross(data["e_theta"], data["e_zeta"]))
        + dot(
            data["e_rho_rz"],
            cross(data["e_theta_t"], data["e_zeta"])
            + cross(data["e_theta"], data["e_zeta_t"]),
        )
        + dot(
            data["e_rho_rt"],
            cross(data["e_theta_z"], data["e_zeta"])
            + cross(data["e_theta"], data["e_zeta_z"]),
        )
        + dot(
            data["e_rho_r"],
            cross(data["e_theta_tz"], data["e_zeta"])
            + cross(data["e_theta_t"], data["e_zeta_z"])
            + cross(data["e_theta"], data["e_zeta_tz"]),
        )
        + dot(
            data["e_rho_tz"],
            cross(data["e_theta_r"], data["e_zeta"])
            + cross(data["e_theta"], data["e_zeta_r"]),
        )
        + dot(
            data["e_rho_z"],
            cross(data["e_theta_rt"], data["e_zeta"])
            + cross(data["e_theta_r"], data["e_zeta_t"])
            + cross(data["e_theta"], data["e_zeta_rt"]),
        )
        + dot(
            data["e_rho_t"],
            cross(data["e_theta_rz"], data["e_zeta"])
            + cross(data["e_theta_z"], data["e_zeta_r"])
            + cross(data["e_theta"], data["e_zeta_rz"]),
        )
        + dot(
            data["e_rho"],
            cross(data["e_theta_rtz"], data["e_zeta"])
            + cross(data["e_theta_tz"], data["e_zeta_r"])
            + cross(data["e_theta_rz"], data["e_zeta_t"])
            + cross(data["e_theta_z"], data["e_zeta_rt"])
            + cross(data["e_theta_rt"], data["e_zeta_z"])
            + cross(data["e_theta_t"], data["e_zeta_rz"])
            + cross(data["e_theta_r"], data["e_zeta_tz"])
            + cross(data["e_theta"], data["e_zeta_rtz"]),
        )
    )
    return data


@register_compute_fun(
    name="sqrt(g)_rz",
    label="\\partial_{\\rho\\zeta} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, second derivative wrt "
    + "radial coordinate and toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_r",
        "e_theta_r",
        "e_zeta_r",
        "e_rho_z",
        "e_theta_z",
        "e_zeta_z",
        "e_rho_rz",
        "e_theta_rz",
        "e_zeta_rz",
    ],
)
def _sqrtg_rz(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_rz"] = (
        dot(data["e_rho_rz"], cross(data["e_theta"], data["e_zeta"]))
        + dot(data["e_rho_r"], cross(data["e_theta_z"], data["e_zeta"]))
        + dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta_z"]))
        + dot(data["e_rho_z"], cross(data["e_theta_r"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_rz"], data["e_zeta"]))
        + dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta_z"]))
        + dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta_r"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_rz"]))
    )
    return data


@register_compute_fun(
    name="sqrt(g)_rrz",
    label="\\partial_{\\rho\\rho\\zeta} \\sqrt{g}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of flux coordinate system, third derivative wrt "
    + "radial coordinate twice and toroidal angle once",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_r",
        "e_theta_r",
        "e_zeta_r",
        "e_rho_z",
        "e_theta_z",
        "e_zeta_z",
        "e_rho_rr",
        "e_rho_rz",
        "e_theta_rr",
        "e_theta_rz",
        "e_zeta_rz",
        "e_zeta_rr",
        "e_rho_rrz",
        "e_theta_rrz",
        "e_zeta_rrz",
    ],
)
def _sqrtg_rrz(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_rrz"] = (
        dot(data["e_rho_rrz"], cross(data["e_theta"], data["e_zeta"]))
        + dot(
            data["e_rho_rr"],
            cross(data["e_theta_z"], data["e_zeta"])
            + cross(data["e_theta"], data["e_zeta_z"]),
        )
        + 2
        * dot(
            data["e_rho_rz"],
            cross(data["e_theta_r"], data["e_zeta"])
            + cross(data["e_theta"], data["e_zeta_r"]),
        )
        + 2
        * dot(
            data["e_rho_r"],
            cross(data["e_theta_rz"], data["e_zeta"])
            + cross(data["e_theta_r"], data["e_zeta_z"])
            + cross(data["e_theta_z"], data["e_zeta_r"])
            + cross(data["e_theta"], data["e_zeta_rz"]),
        )
        + dot(
            data["e_rho_z"],
            cross(data["e_theta_rr"], data["e_zeta"])
            + cross(data["e_theta"], data["e_zeta_rr"]),
        )
        + dot(
            data["e_rho"],
            cross(data["e_theta_rrz"], data["e_zeta"])
            + cross(data["e_theta_rr"], data["e_zeta_z"])
            + 2 * cross(data["e_theta_r"], data["e_zeta_rz"])
            + 2 * cross(data["e_theta_rz"], data["e_zeta_r"])
            + cross(data["e_theta_z"], data["e_zeta_rr"])
            + cross(data["e_theta"], data["e_zeta_rrz"]),
        )
    )
    return data


@register_compute_fun(
    name="g_rr",
    label="g_{\\rho\\rho}",
    units="m^{2}",
    units_long="square meters",
    description="Radial/Radial element of covariant metric tensor",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _g_sub_rr(params, transforms, profiles, data, **kwargs):
    data["g_rr"] = dot(data["e_rho"], data["e_rho"])
    return data


@register_compute_fun(
    name="g_tt",
    label="g_{\\theta\\theta}",
    units="m^{2}",
    units_long="square meters",
    description="Poloidal/Poloidal element of covariant metric tensor",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _g_sub_tt(params, transforms, profiles, data, **kwargs):
    data["g_tt"] = dot(data["e_theta"], data["e_theta"])
    return data


@register_compute_fun(
    name="g_zz",
    label="g_{\\zeta\\zeta}",
    units="m^{2}",
    units_long="square meters",
    description="Toroidal/Toroidal element of covariant metric tensor",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_zeta"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _g_sub_zz(params, transforms, profiles, data, **kwargs):
    data["g_zz"] = dot(data["e_zeta"], data["e_zeta"])
    return data


@register_compute_fun(
    name="g_rt",
    label="g_{\\rho\\theta}",
    units="m^{2}",
    units_long="square meters",
    description="Radial/Poloidal element of covariant metric tensor",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_theta"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _g_sub_rt(params, transforms, profiles, data, **kwargs):
    data["g_rt"] = dot(data["e_rho"], data["e_theta"])
    return data


@register_compute_fun(
    name="g_rz",
    label="g_{\\rho\\zeta}",
    units="m^{2}",
    units_long="square meters",
    description="Radial/Toroidal element of covariant metric tensor",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_zeta"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _g_sub_rz(params, transforms, profiles, data, **kwargs):
    data["g_rz"] = dot(data["e_rho"], data["e_zeta"])
    return data


@register_compute_fun(
    name="g_tz",
    label="g_{\\theta\\zeta}",
    units="m^{2}",
    units_long="square meters",
    description="Poloidal/Toroidal element of covariant metric tensor",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_zeta"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _g_sub_tz(params, transforms, profiles, data, **kwargs):
    data["g_tz"] = dot(data["e_theta"], data["e_zeta"])
    return data


@register_compute_fun(
    name="g_tt_r",
    label="\\partial_{\\rho} g_{\\theta\\theta}",
    units="m^{2}",
    units_long="square meters",
    description="Poloidal/Poloidal element of covariant metric tensor, derivative "
    + "wrt rho",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_theta_r"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _g_sub_tt_r(params, transforms, profiles, data, **kwargs):
    data["g_tt_r"] = 2 * dot(data["e_theta"], data["e_theta_r"])
    return data


@register_compute_fun(
    name="g_tz_r",
    label="\\partial_{\\rho} g_{\\theta\\zeta}",
    units="m^{2}",
    units_long="square meters",
    description="Poloidal/Toroidal element of covariant metric tensor, derivative "
    + "wrt rho",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_zeta", "e_theta_r", "e_zeta_r"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _g_sub_tz_r(params, transforms, profiles, data, **kwargs):
    data["g_tz_r"] = dot(data["e_theta_r"], data["e_zeta"]) + dot(
        data["e_theta"], data["e_zeta_r"]
    )
    return data


@register_compute_fun(
    name="g_tt_rr",
    label="\\partial_{\\rho\\rho} g_{\\theta\\theta}",
    units="m^{2}",
    units_long="square meters",
    description="Poloidal/Poloidal element of covariant metric tensor, second "
    + "derivative wrt rho",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_theta_r", "e_theta_rr"],
)
def _g_sub_tt_rr(params, transforms, profiles, data, **kwargs):
    data["g_tt_rr"] = 2 * (
        dot(data["e_theta_r"], data["e_theta_r"])
        + dot(data["e_theta"], data["e_theta_rr"])
    )
    return data


@register_compute_fun(
    name="g_tt_rrr",
    label="\\partial_{\\rho\\rho\\rho} g_{\\theta\\theta}",
    units="m^{2}",
    units_long="square meters",
    description="Poloidal/Poloidal element of covariant metric tensor, third "
    + "derivative wrt rho",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_theta_r", "e_theta_rr", "e_theta_rrr"],
)
def _g_sub_tt_rrr(params, transforms, profiles, data, **kwargs):
    data["g_tt_rrr"] = 6 * dot(data["e_theta_rr"], data["e_theta_r"]) + 2 * dot(
        data["e_theta"], data["e_theta_rrr"]
    )
    return data


@register_compute_fun(
    name="g_tz_rr",
    label="\\partial_{\\rho\\rho} g_{\\theta\\zeta}",
    units="m^{2}",
    units_long="square meters",
    description="Poloidal/Toroidal element of covariant metric tensor, second "
    + "derivative wrt rho",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_zeta", "e_theta_r", "e_zeta_r", "e_theta_rr", "e_zeta_rr"],
)
def _g_sub_tz_rr(params, transforms, profiles, data, **kwargs):
    data["g_tz_rr"] = (
        dot(data["e_theta_rr"], data["e_zeta"])
        + 2 * dot(data["e_theta_r"], data["e_zeta_r"])
        + dot(data["e_theta"], data["e_zeta_rr"])
    )
    return data


@register_compute_fun(
    name="g_tz_rrr",
    label="\\partial_{\\rho\\rho\\rho} g_{\\theta\\zeta}",
    units="m^{2}",
    units_long="square meters",
    description="Poloidal/Toroidal element of covariant metric tensor, third "
    + "derivative wrt rho",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_theta",
        "e_zeta",
        "e_theta_r",
        "e_zeta_r",
        "e_theta_rr",
        "e_zeta_rr",
        "e_theta_rrr",
        "e_zeta_rrr",
    ],
)
def _g_sub_tz_rrr(params, transforms, profiles, data, **kwargs):
    data["g_tz_rrr"] = (
        dot(data["e_theta_rrr"], data["e_zeta"])
        + 3 * dot(data["e_theta_rr"], data["e_zeta_r"])
        + 3 * dot(data["e_theta_r"], data["e_zeta_rr"])
        + dot(data["e_theta"], data["e_zeta_rrr"])
    )
    return data


@register_compute_fun(
    name="g^rr",
    label="g^{\\rho\\rho}",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Radial/Radial element of contravariant metric tensor",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho"],
)
def _g_sup_rr(params, transforms, profiles, data, **kwargs):
    data["g^rr"] = dot(data["e^rho"], data["e^rho"])
    return data


@register_compute_fun(
    name="g^tt",
    label="g^{\\theta\\theta}",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Poloidal/Poloidal element of contravariant metric tensor",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^theta"],
)
def _g_sup_tt(params, transforms, profiles, data, **kwargs):
    data["g^tt"] = dot(data["e^theta"], data["e^theta"])
    return data


@register_compute_fun(
    name="g^zz",
    label="g^{\\zeta\\zeta}",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Toroidal/Toroidal element of contravariant metric tensor",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^zeta"],
)
def _g_sup_zz(params, transforms, profiles, data, **kwargs):
    data["g^zz"] = dot(data["e^zeta"], data["e^zeta"])
    return data


@register_compute_fun(
    name="g^rt",
    label="g^{\\rho\\theta}",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Radial/Poloidal element of contravariant metric tensor",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^theta"],
)
def _g_sup_rt(params, transforms, profiles, data, **kwargs):
    data["g^rt"] = dot(data["e^rho"], data["e^theta"])
    return data


@register_compute_fun(
    name="g^rz",
    label="g^{\\rho\\zeta}",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Radial/Toroidal element of contravariant metric tensor",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^zeta"],
)
def _g_sup_rz(params, transforms, profiles, data, **kwargs):
    data["g^rz"] = dot(data["e^rho"], data["e^zeta"])
    return data


@register_compute_fun(
    name="g^tz",
    label="g^{\\theta\\zeta}",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Poloidal/Toroidal element of contravariant metric tensor",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^theta", "e^zeta"],
    aliases=["g^zt"],
)
def _g_sup_tz(params, transforms, profiles, data, **kwargs):
    data["g^tz"] = dot(data["e^theta"], data["e^zeta"])
    return data


@register_compute_fun(
    name="g^rr_r",
    label="g^{\\rho}{\\rho}_{\\rho}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Radial element of contravariant metric tensor, "
    + "first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^rho_r"],
)
def _g_sup_rr_r(params, transforms, profiles, data, **kwargs):
    data["g^rr_r"] = 2 * dot(data["e^rho_r"], data["e^rho"])
    return data


@register_compute_fun(
    name="g^rr_z",
    label="\\partial_{\\zeta} g^{\\rho}{\\rho}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Radial element of contravariant metric tensor, "
    + "first toroidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^rho_z"],
)
def _g_sup_rr_z(params, transforms, profiles, data, **kwargs):
    data["g^rr_z"] = 2 * dot(data["e^rho_z"], data["e^rho"])
    return data


@register_compute_fun(
    name="g^rr_zz",
    label="\\partial_{\\zeta \\zeta} g^{rr}",
    units="m^-2",
    units_long="None",
    description="Double zeta derivative of the flux expansion term",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^rho",
        "e^rho_z",
        "e^rho_zz",
    ],
)
def _g_sup_rr_sub_zz(params, transforms, profiles, data, **kwargs):
    data["g^rr_zz"] = 2 * (
        dot(data["e^rho_zz"], data["e^rho"]) + dot(data["e^rho_z"], data["e^rho_z"])
    )
    return data


@register_compute_fun(
    name="g^rr_t",
    label="\\partial_{\\theta} g^{\\rho}{\\rho}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Radial element of contravariant metric tensor, "
    + "first poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^rho_t"],
)
def _g_sup_rr_t(params, transforms, profiles, data, **kwargs):
    data["g^rr_t"] = 2 * dot(data["e^rho_t"], data["e^rho"])
    return data


@register_compute_fun(
    name="g^rr_tt",
    label="\\partial_{\\theta \\theta} g^{\\rho}{\\rho}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Radial element of contravariant metric tensor, "
    + "second poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^rho_t", "e^rho_tt"],
)
def _g_sup_rr_tt(params, transforms, profiles, data, **kwargs):
    data["g^rr_tt"] = 2 * (
        dot(data["e^rho_tt"], data["e^rho"]) + dot(data["e^rho_t"], data["e^rho_t"])
    )
    return data


@register_compute_fun(
    name="g^rr_tz",
    label="\\partial_{\\theta \\zeta} g^{\\rho}{\\rho}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Radial element of contravariant metric tensor, "
    + "mixed derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^rho_t", "e^rho_z", "e^rho_tz"],
)
def _g_sup_rr_tz(params, transforms, profiles, data, **kwargs):
    data["g^rr_tz"] = 2 * (
        dot(data["e^rho_tz"], data["e^rho"]) + dot(data["e^rho_t"], data["e^rho_z"])
    )
    return data


@register_compute_fun(
    name="g^rt_r",
    label="\\partial_{\\rho} g^{\\rho}{\\theta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Poloidal element of contravariant metric tensor, "
    + "first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^theta", "e^rho_r", "e^theta_r"],
)
def _g_sup_rt_r(params, transforms, profiles, data, **kwargs):
    data["g^rt_r"] = dot(data["e^rho_r"], data["e^theta"]) + dot(
        data["e^rho"], data["e^theta_r"]
    )
    return data


@register_compute_fun(
    name="g^rz_r",
    label="\\partial_{\\rho} g^{\\rho}{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Toroidal element of contravariant metric tensor, "
    + "first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^zeta", "e^rho_r", "e^zeta_r"],
)
def _g_sup_rz_r(params, transforms, profiles, data, **kwargs):
    data["g^rz_r"] = dot(data["e^rho_r"], data["e^zeta"]) + dot(
        data["e^rho"], data["e^zeta_r"]
    )
    return data


@register_compute_fun(
    name="g^tt_r",
    label="\\partial_{\\rho} g^{\\theta}{\\theta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Poloidal/Poloidal element of contravariant metric tensor, "
    + "first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^theta", "e^theta_r"],
)
def _g_sup_tt_r(params, transforms, profiles, data, **kwargs):
    data["g^tt_r"] = 2 * dot(data["e^theta_r"], data["e^theta"])
    return data


@register_compute_fun(
    name="g^tz_r",
    label="\\partial_{\\rho} g^{\\theta}{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Poloidal/Toroidal element of contravariant metric tensor, "
    + "first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^theta", "e^zeta", "e^theta_r", "e^zeta_r"],
)
def _g_sup_tz_r(params, transforms, profiles, data, **kwargs):
    data["g^tz_r"] = dot(data["e^theta_r"], data["e^zeta"]) + dot(
        data["e^theta"], data["e^zeta_r"]
    )
    return data


@register_compute_fun(
    name="g^zz_r",
    label="\\partial_{\\rho} g^{\\zeta}{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Toroidal/Toroidal element of contravariant metric tensor, "
    + "first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^zeta", "e^zeta_r"],
)
def _g_sup_zz_r(params, transforms, profiles, data, **kwargs):
    data["g^zz_r"] = 2 * dot(data["e^zeta_r"], data["e^zeta"])
    return data


@register_compute_fun(
    name="g^rt_t",
    label="\\partial_{\\theta} g^{\\rho}{\\theta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Poloidal element of contravariant metric tensor, "
    + "first poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^theta", "e^rho_t", "e^theta_t"],
)
def _g_sup_rt_t(params, transforms, profiles, data, **kwargs):
    data["g^rt_t"] = dot(data["e^rho_t"], data["e^theta"]) + dot(
        data["e^rho"], data["e^theta_t"]
    )
    return data


@register_compute_fun(
    name="g^rz_t",
    label="\\partial_{\\theta} g^{\\rho}{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Toroidal element of contravariant metric tensor, "
    + "first poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^zeta", "e^rho_t", "e^zeta_t"],
)
def _g_sup_rz_t(params, transforms, profiles, data, **kwargs):
    data["g^rz_t"] = dot(data["e^rho_t"], data["e^zeta"]) + dot(
        data["e^rho"], data["e^zeta_t"]
    )
    return data


@register_compute_fun(
    name="g^tt_t",
    label="\\partial_{\\theta} g^{\\theta}{\\theta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Poloidal/Poloidal element of contravariant metric tensor, "
    + "first poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^theta", "e^theta_t"],
)
def _g_sup_tt_t(params, transforms, profiles, data, **kwargs):
    data["g^tt_t"] = 2 * dot(data["e^theta_t"], data["e^theta"])
    return data


@register_compute_fun(
    name="g^tz_t",
    label="\\partial_{\\theta} g^{\\theta}{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Poloidal/Toroidal element of contravariant metric tensor, "
    + "first poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^theta", "e^zeta", "e^theta_t", "e^zeta_t"],
)
def _g_sup_tz_t(params, transforms, profiles, data, **kwargs):
    data["g^tz_t"] = dot(data["e^theta_t"], data["e^zeta"]) + dot(
        data["e^theta"], data["e^zeta_t"]
    )
    return data


@register_compute_fun(
    name="g^zz_t",
    label="\\partial_{\\theta} g^{\\zeta}{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Toroidal/Toroidal element of contravariant metric tensor, "
    + "first poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^zeta", "e^zeta_t"],
)
def _g_sup_zz_t(params, transforms, profiles, data, **kwargs):
    data["g^zz_t"] = 2 * dot(data["e^zeta_t"], data["e^zeta"])
    return data


@register_compute_fun(
    name="g^rt_z",
    label="\\partial_{\\zeta} g^{\\rho}{\\theta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Poloidal element of contravariant metric tensor, "
    + "first toroidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^theta", "e^rho_z", "e^theta_z"],
)
def _g_sup_rt_z(params, transforms, profiles, data, **kwargs):
    data["g^rt_z"] = dot(data["e^rho_z"], data["e^theta"]) + dot(
        data["e^rho"], data["e^theta_z"]
    )
    return data


@register_compute_fun(
    name="g^rz_z",
    label="\\partial_{\\zeta} g^{\\rho}{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Toroidal element of contravariant metric tensor, "
    + "first toroidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^zeta", "e^rho_z", "e^zeta_z"],
)
def _g_sup_rz_z(params, transforms, profiles, data, **kwargs):
    data["g^rz_z"] = dot(data["e^rho_z"], data["e^zeta"]) + dot(
        data["e^rho"], data["e^zeta_z"]
    )
    return data


@register_compute_fun(
    name="g^tt_z",
    label="\\partial_{\\zeta} g^{\\theta}{\\theta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Poloidal/Poloidal element of contravariant metric tensor, "
    + "first toroidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^theta", "e^theta_z"],
)
def _g_sup_tt_z(params, transforms, profiles, data, **kwargs):
    data["g^tt_z"] = 2 * dot(data["e^theta_z"], data["e^theta"])
    return data


@register_compute_fun(
    name="g^tz_z",
    label="\\partial_{\\zeta} g^{\\theta}{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Poloidal/Toroidal element of contravariant metric tensor, "
    + "first toroidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^theta", "e^zeta", "e^theta_z", "e^zeta_z"],
)
def _g_sup_tz_z(params, transforms, profiles, data, **kwargs):
    data["g^tz_z"] = dot(data["e^theta_z"], data["e^zeta"]) + dot(
        data["e^theta"], data["e^zeta_z"]
    )
    return data


@register_compute_fun(
    name="g^zz_z",
    label="\\partial_{\\zeta} g^{\\zeta}{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Toroidal/Toroidal element of contravariant metric tensor, "
    + "first toroidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^zeta", "e^zeta_z"],
)
def _g_sup_zz_z(params, transforms, profiles, data, **kwargs):
    data["g^zz_z"] = 2 * dot(data["e^zeta_z"], data["e^zeta"])
    return data


@register_compute_fun(
    name="|grad(rho)|",
    label="|\\nabla \\rho|",
    units="m^{-1}",
    units_long="inverse meters",
    description="Magnitude of contravariant radial basis vector",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["g^rr"],
)
def _gradrho(params, transforms, profiles, data, **kwargs):
    data["|grad(rho)|"] = jnp.sqrt(data["g^rr"])
    return data


@register_compute_fun(
    name="|grad(psi)|",
    label="|\\nabla\\psi|",
    units="Wb / m",
    units_long="Webers per meter",
    description="Toroidal flux gradient (normalized by 2pi) magnitude",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|grad(psi)|^2"],
)
def _gradpsi_mag(params, transforms, profiles, data, **kwargs):
    data["|grad(psi)|"] = jnp.sqrt(data["|grad(psi)|^2"])
    return data


@register_compute_fun(
    name="|grad(psi)|^2",
    label="|\\nabla\\psi|^{2}",
    units="(Wb / m)^{2}",
    units_long="Webers squared per square meter",
    description="Toroidal flux gradient (normalized by 2pi) magnitude squared",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["grad(psi)"],
)
def _gradpsi_mag2(params, transforms, profiles, data, **kwargs):
    data["|grad(psi)|^2"] = dot(data["grad(psi)"], data["grad(psi)"])
    return data


@register_compute_fun(
    name="|grad(theta)|",
    label="|\\nabla \\theta|",
    units="m^{-1}",
    units_long="inverse meters",
    description="Magnitude of contravariant poloidal basis vector",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["g^tt"],
)
def _gradtheta(params, transforms, profiles, data, **kwargs):
    data["|grad(theta)|"] = jnp.sqrt(data["g^tt"])
    return data


@register_compute_fun(
    name="|grad(zeta)|",
    label="|\\nabla \\zeta|",
    units="m^{-1}",
    units_long="inverse meters",
    description="Magnitude of contravariant toroidal basis vector",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["g^zz"],
)
def _gradzeta(params, transforms, profiles, data, **kwargs):
    data["|grad(zeta)|"] = jnp.sqrt(data["g^zz"])
    return data


@register_compute_fun(
    name="g^aa",
    label="g^{\\alpha \\alpha}",
    units="m^{-2}",
    units_long="None",
    description="Fieldline bending term1",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["grad(alpha)"],
)
def _g_sup_aa(params, transforms, profiles, data, **kwargs):
    data["g^aa"] = dot(data["grad(alpha)"], data["grad(alpha)"])
    return data


@register_compute_fun(
    name="g^aa_t",
    label="\\partial_{\\theta} g^{\\alpha \\alpha}",
    units="~",
    units_long="None",
    description="Poloidal derivative of the fieldline bending term",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "grad(alpha)",
        "grad(alpha)_t",
    ],
)
def _g_sup_aa_t(params, transforms, profiles, data, **kwargs):
    data["g^aa_t"] = 2 * dot(data["grad(alpha)"], data["grad(alpha)_t"])
    return data


@register_compute_fun(
    name="g^aa_z",
    label="\\partial_{\\zeta} g^{\\alpha \\alpha}",
    units="~",
    units_long="None",
    description="Toroidal derivative of the fieldline bending term",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "grad(alpha)",
        "grad(alpha)_z",
    ],
)
def _g_sup_aa_z(params, transforms, profiles, data, **kwargs):
    data["g^aa_z"] = 2 * dot(data["grad(alpha)_z"], data["grad(alpha)"])
    return data


@register_compute_fun(
    name="g^aa_zz",
    label="\\partial_{\\zeta \\zeta} g^{\\alpha \\alpha}",
    units="~",
    units_long="None",
    description="Toroidal derivative of the fieldline bending term",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "grad(alpha)",
        "grad(alpha)_z",
        "grad(alpha)_zz",
    ],
)
def _g_sup_aa_zz(params, transforms, profiles, data, **kwargs):
    data["g^aa_zz"] = 2 * (
        dot(data["grad(alpha)_z"], data["grad(alpha)_z"])
        + dot(data["grad(alpha)"], data["grad(alpha)_zz"])
    )
    return data


@register_compute_fun(
    name="g^aa_tt",
    label="\\partial_{\\theta \\theta} g^{\\alpha \\alpha}_{\\theta \\theta}",
    units="~",
    units_long="None",
    description="Poloidal derivative of the fieldline bending term",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "grad(alpha)",
        "grad(alpha)_t",
        "grad(alpha)_tt",
    ],
)
def _g_sup_aa_tt(params, transforms, profiles, data, **kwargs):
    data["g^aa_tt"] = 2 * (
        dot(data["grad(alpha)_t"], data["grad(alpha)_t"])
        + dot(data["grad(alpha)"], data["grad(alpha)_tt"])
    )
    return data


@register_compute_fun(
    name="g^aa_tz",
    label="\\partial_{\\theta \\zeta} g^{\\alpha \\alpha}",
    units="~",
    units_long="None",
    description="Mixed derivative of the fieldline bending term",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "grad(alpha)",
        "grad(alpha)_t",
        "grad(alpha)_z",
        "grad(alpha)_tz",
    ],
)
def _g_sup_aa_tz(params, transforms, profiles, data, **kwargs):
    data["g^aa_tz"] = 2 * (
        dot(data["grad(alpha)_tz"], data["grad(alpha)"])
        + dot(data["grad(alpha)_z"], data["grad(alpha)_t"])
    )

    return data


@register_compute_fun(
    name="g^ra",
    label="g^{\\rho \\alpha}",
    units="~",
    units_long="None",
    description="Fieldline bending term2",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["grad(alpha)", "e^rho"],
)
def _g_sup_ra(params, transforms, profiles, data, **kwargs):
    data["g^ra"] = dot(data["grad(alpha)"], data["e^rho"])
    return data


@register_compute_fun(
    name="g^ra_z",
    label="\\partial_{\\zeta} g^{\\rho \\alpha}",
    units="~",
    units_long="None",
    description="Single zeta derivative of a fieldline bending term2",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "grad(alpha)",
        "grad(alpha)_z",
        "e^rho",
        "e^rho_z",
    ],
)
def _g_sup_ra_z(params, transforms, profiles, data, **kwargs):
    data["g^ra_z"] = dot(data["grad(alpha)_z"], data["e^rho"]) + dot(
        data["grad(alpha)"], data["e^rho_z"]
    )
    return data


@register_compute_fun(
    name="g^ra_t",
    label="\\partial_{\\theta} g^{\\rho \\alpha}",
    units="~",
    units_long="None",
    description="Poloidal theta derivative of a fieldline bending term2",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "grad(alpha)",
        "grad(alpha)_t",
        "e^rho",
        "e^rho_t",
    ],
)
def _g_sup_ra_t(params, transforms, profiles, data, **kwargs):
    data["g^ra_t"] = dot(data["grad(alpha)_t"], data["e^rho"]) + dot(
        data["grad(alpha)"], data["e^rho_t"]
    )
    return data


@register_compute_fun(
    name="g^ra_zz",
    label="\\partial_{\\zeta \\zeta} g^{\\rho \\alpha}",
    units="~",
    units_long="None",
    description="Double zeta derivative of a fieldline bending term2",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "grad(alpha)",
        "grad(alpha)_z",
        "grad(alpha)_zz",
        "e^rho",
        "e^rho_z",
        "e^rho_zz",
    ],
)
def _g_sup_ra_zz(params, transforms, profiles, data, **kwargs):
    data["g^ra_zz"] = dot(data["grad(alpha)_zz"], data["e^rho"])
    +2 * dot(data["grad(alpha)_z"], data["e^rho_z"])
    +dot(data["grad(alpha)"], data["e^rho_zz"])
    return data


@register_compute_fun(
    name="g^ra_tt",
    label="\\partial_{\\theta \\theta} g^{\\rho \\alpha}",
    units="~",
    units_long="None",
    description="Double zeta derivative of a fieldline bending term2",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "grad(alpha)",
        "grad(alpha)_t",
        "grad(alpha)_tt",
        "e^rho",
        "e^rho_t",
        "e^rho_tt",
    ],
)
def _g_sup_ra_tt(params, transforms, profiles, data, **kwargs):
    data["g^ra_tt"] = dot(data["grad(alpha)_tt"], data["e^rho"])
    +2 * dot(data["grad(alpha)_t"], data["e^rho_t"])
    +dot(data["grad(alpha)"], data["e^rho_tt"])
    return data


@register_compute_fun(
    name="g^ra_tz",
    label="\\partial_{\\theta \\zeta} g^{\\rho \\alpha}",
    units="~",
    units_long="None",
    description="Mixed derivative of a fieldline bending term2",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "grad(alpha)",
        "grad(alpha)_t",
        "grad(alpha)_z",
        "grad(alpha)_tz",
        "e^rho",
        "e^rho_t",
        "e^rho_z",
        "e^rho_tz",
    ],
)
def _g_sup_ra_tz(params, transforms, profiles, data, **kwargs):
    data["g^ra_tz"] = (
        dot(data["grad(alpha)_tz"], data["e^rho"])
        + dot(data["grad(alpha)_z"], data["e^rho_t"])
        + dot(data["grad(alpha)"], data["e^rho_tz"])
        + dot(data["grad(alpha)_t"], data["e^rho_z"])
    )
    return data


@register_compute_fun(
    name="gbdrift",
    # Exact definition of the magnetic drifts taken from
    # eqn. 48 of Introduction to Quasisymmetry by Landreman
    # https://tinyurl.com/54udvaa4
    label="\\mathrm{gbdrift} = 1/B^{2} (\\mathbf{b}\\times\\nabla B) \\cdot"
    + "\\nabla \\alpha",
    units="1/(T-m^{2})",
    units_long="inverse Tesla meters^2",
    description="Binormal component of the geometric part of the gradB drift"
    + " used for local stability analyses, Gamma_c, epsilon_eff etc.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "b", "grad(alpha)", "grad(|B|)"],
)
def _gbdrift(params, transforms, profiles, data, **kwargs):
    data["gbdrift"] = (
        1
        / data["|B|"] ** 2
        * dot(data["b"], cross(data["grad(|B|)"], data["grad(alpha)"]))
    )
    return data


@register_compute_fun(
    name="cvdrift",
    # Exact definition of the magnetic drifts taken from
    # eqn. 48 of Introduction to Quasisymmetry by Landreman
    # https://tinyurl.com/54udvaa4
    label="\\mathrm{cvdrift} = 1/B^{3} (\\mathbf{b}\\times\\nabla(p + B^2/2))"
    + "\\cdot \\nabla \\alpha",
    units="1/(T-m^{2})",
    units_long="inverse Tesla meters^2",
    description="Binormal component of the geometric part of the curvature drift"
    + " used for local stability analyses, Gamma_c, epsilon_eff etc.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["p_r", "psi_r", "|B|", "gbdrift"],
)
def _cvdrift(params, transforms, profiles, data, **kwargs):
    dp_dpsi = mu_0 * data["p_r"] / data["psi_r"]
    data["cvdrift"] = 1 / data["|B|"] ** 2 * dp_dpsi + data["gbdrift"]
    return data


@register_compute_fun(
    name="cvdrift0",
    # Exact definition of the magnetic drifts taken from
    # eqn. 48 of Introduction to Quasisymmetry by Landreman
    # https://tinyurl.com/54udvaa4
    label="\\mathrm{cvdrift0} = 1/B^{2} (\\mathbf{b}\\times\\nabla B)"
    + "\\cdot \\nabla \\rho",
    units="1/(T-m^{2})",
    units_long="inverse Tesla meters^2",
    description="Radial component of the geometric part of the curvature drift"
    + " used for local stability analyses, Gamma_c, epsilon_eff etc.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "b", "e^rho", "grad(|B|)"],
)
def _cvdrift0(params, transforms, profiles, data, **kwargs):
    data["cvdrift0"] = (
        1 / data["|B|"] ** 2 * (dot(data["b"], cross(data["grad(|B|)"], data["e^rho"])))
    )
    return data
