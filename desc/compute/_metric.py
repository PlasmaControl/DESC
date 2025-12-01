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

from ..integrals.surface_integral import surface_averages
from ..utils import cross, dot, safediv, safenorm
from .data_index import register_compute_fun


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
    name="sqrt(g)_Clebsch",
    label="\\sqrt{g}_{\\text{Clebsch}}",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of Clebsch field line coordinate system (ρ,α,ζ)"
    " where ζ is the DESC toroidal coordinate.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)", "alpha_t"],
)
def _sqrtg_clebsch(params, transforms, profiles, data, **kwargs):
    # Same as dot(data["e_rho|a,z"], cross(data["e_alpha"], data["e_zeta|r,a"])), but
    # more efficient as it avoids computing radial derivative of alpha and hence iota.
    data["sqrt(g)_Clebsch"] = data["sqrt(g)"] / data["alpha_t"]
    return data


@register_compute_fun(
    name="|e_theta x e_zeta|",
    label="| \\mathbf{e}_{\\theta} \\times \\mathbf{e}_{\\zeta} |",
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
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _e_theta_x_e_zeta(params, transforms, profiles, data, **kwargs):
    data["|e_theta x e_zeta|"] = safenorm(
        cross(data["e_theta"], data["e_zeta"]), axis=-1
    )
    return data


@register_compute_fun(
    name="|e_theta x e_zeta|_r",
    label="\\partial_{\\rho} |\\mathbf{e}_{\\theta} \\times \\mathbf{e}_{\\zeta}|",
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
    label="\\partial_{\\rho\\rho}|\\mathbf{e}_{\\theta}\\times\\mathbf{e}_{\\zeta}|",
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
    label="\\partial_{\\zeta}|\\mathbf{e}_{\\theta} \\times \\mathbf{e}_{\\zeta}|",
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
        "desc.geometry.surface.FourierRZToroidalSurface",
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
    label="|\\mathbf{e}_{\\zeta} \\times \\mathbf{e}_{\\rho}|",
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
    ],
)
def _e_zeta_x_e_rho(params, transforms, profiles, data, **kwargs):
    data["|e_zeta x e_rho|"] = safenorm(data["e^theta*sqrt(g)"], axis=-1)
    return data


@register_compute_fun(
    name="|e_rho x e_theta|",
    label="|\\mathbf{e}_{\\rho} \\times \\mathbf{e}_{\\theta}|",
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
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _e_rho_x_e_theta(params, transforms, profiles, data, **kwargs):
    data["|e_rho x e_theta|"] = safenorm(cross(data["e_rho"], data["e_theta"]), axis=-1)
    return data


@register_compute_fun(
    name="|e_rho x e_alpha|",
    label="|\\mathbf{e}_{\\rho} \\times \\mathbf{e}_{\\alpha}|",
    units="m^{2}",
    units_long="square meters",
    description="2D Jacobian determinant for constant zeta surface in Clebsch "
    "field line coordinates (ρ,α,ζ) where ζ is the DESC toroidal coordinate.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|e_rho x e_theta|", "alpha_t"],
)
def _e_rho_x_e_alpha(params, transforms, profiles, data, **kwargs):
    # Same as safenorm(cross(data["e_rho|a,z"], data["e_alpha"]), axis=-1), but more
    # efficient as it avoids computing radial derivative of iota and stream functions.
    data["|e_rho x e_alpha|"] = data["|e_rho x e_theta|"] / jnp.abs(data["alpha_t"])
    return data


@register_compute_fun(
    name="|e_rho x e_theta|_r",
    label="\\partial_{\\rho} |\\mathbf{e}_{\\rho} \\times \\mathbf{e}_{\\theta}|",
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
        "desc.geometry.surface.ZernikeRZToroidalSection",
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
    label="\\partial_{\\rho \\rho} |\\mathbf{e}_{\\rho} \\times \\mathbf{e}_{\\theta}|",
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
        "desc.geometry.surface.ZernikeRZToroidalSection",
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
        "desc.geometry.surface.ZernikeRZToroidalSection",
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
        "desc.geometry.surface.FourierRZToroidalSurface",
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
        "desc.geometry.surface.ZernikeRZToroidalSection",
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
        "desc.geometry.surface.FourierRZToroidalSurface",
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
        "desc.geometry.surface.ZernikeRZToroidalSection",
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
    description="Radial/Poloidal (ρ, θ) element of contravariant metric tensor",
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
    label="\\partial_{\\rho} g^{\\rho \\rho}",
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
    name="g^rr_t",
    label="\\partial_{\\theta} g^{\\rho \\rho}",
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
    name="g^rr_z",
    label="\\partial_{\\zeta} g^{\\rho \\rho}",
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
    name="g^rt_r",
    label="\\partial_{\\rho} g^{\\rho \\theta}",
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
    label="\\partial_{\\rho} g^{\\rho \\zeta}",
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
    label="\\partial_{\\rho} g^{\\theta \\theta}",
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
    label="\\partial_{\\rho} g^{\\theta \\zeta}",
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
    label="\\partial_{\\rho} g^{\\zeta \\zeta}",
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
    label="\\partial_{\\theta} g^{\\rho \\theta}",
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
    label="\\partial_{\\theta} g^{\\rho \\zeta}",
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
    label="\\partial_{\\theta} g^{\\theta \\theta}",
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
    label="\\partial_{\\theta} g^{\\theta \\zeta}",
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
    label="\\partial_{\\theta} g^{\\zeta \\zeta}",
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
    label="\\partial_{\\zeta} g^{\\rho \\theta}",
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
    label="\\partial_{\\zeta} g^{\\rho \\zeta}",
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
    label="\\partial_{\\zeta} g^{\\theta \\theta}",
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
    label="\\partial_{\\zeta} g^{\\theta \\zeta}",
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
    label="\\partial_{\\zeta} g^{\\zeta \\zeta}",
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
    name="<|grad(rho)|>",  # same as S(r) / V_r(r)
    label="\\langle \\vert \\nabla \\rho \\vert \\rangle",
    units="m^{-1}",
    units_long="inverse meters",
    description="Magnitude of contravariant radial basis vector, flux surface average",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|grad(rho)|", "sqrt(g)"],
    axis_limit_data=["sqrt(g)_r"],
    resolution_requirement="tz",
)
def _gradrho_norm_fsa(params, transforms, profiles, data, **kwargs):
    data["<|grad(rho)|>"] = surface_averages(
        transforms["grid"],
        data["|grad(rho)|"],
        sqrt_g=transforms["grid"].replace_at_axis(
            data["sqrt(g)"], lambda: data["sqrt(g)_r"], copy=True
        ),
    )
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
    units_long="inverse square meters",
    description="Contravariant metric tensor grad alpha dot grad alpha",
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
    name="g^ra",
    label="g^{\\rho \\alpha}",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Contravariant metric tensor grad rho dot grad alpha",
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
    name="gbdrift",
    # Exact definition of the magnetic drifts taken from
    # eqn. 48 of Introduction to Quasisymmetry by Landreman
    # https://tinyurl.com/54udvaa4
    label="(\\nabla \\vert B \\vert)_{\\mathrm{drift}} = "
    "(\\mathbf{b} \\times \\nabla B) \\cdot \\nabla \\alpha / \\vert B \\vert^{2}",
    units="1 / Wb",
    units_long="Inverse webers",
    description="Binormal, geometric part of the gradB drift. "
    "Used for local stability analyses, gyrokinetics, and Gamma_c.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["gbdrift (periodic)", "gbdrift (secular)"],
)
def _gbdrift(params, transforms, profiles, data, **kwargs):
    data["gbdrift"] = data["gbdrift (periodic)"] + data["gbdrift (secular)"]
    return data


@register_compute_fun(
    name="gbdrift (periodic)",
    label="\\mathrm{periodic}(\\nabla \\vert B \\vert)_{\\mathrm{drift}}",
    units="1 / Wb",
    units_long="Inverse webers",
    description="Periodic, binormal, geometric part of the gradB drift.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|^2", "b", "grad(alpha) (periodic)", "grad(|B|)"],
)
def _periodic_gbdrift(params, transforms, profiles, data, **kwargs):
    data["gbdrift (periodic)"] = (
        dot(data["b"], cross(data["grad(|B|)"], data["grad(alpha) (periodic)"]))
        / data["|B|^2"]
    )
    return data


@register_compute_fun(
    name="gbdrift (secular)",
    label="\\mathrm{secular}(\\nabla \\vert B \\vert)_{\\mathrm{drift}}",
    units="1 / Wb",
    units_long="Inverse webers",
    description="Secular, binormal, geometric part of the gradB drift.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|^2", "b", "grad(alpha) (secular)", "grad(|B|)"],
)
def _secular_gbdrift(params, transforms, profiles, data, **kwargs):
    data["gbdrift (secular)"] = (
        dot(data["b"], cross(data["grad(|B|)"], data["grad(alpha) (secular)"]))
        / data["|B|^2"]
    )
    return data


@register_compute_fun(
    name="gbdrift (secular)/phi",
    label="\\mathrm{secular}(\\nabla \\vert B \\vert)_{\\mathrm{drift}} / \\phi",
    units="1 / Wb",
    units_long="Inverse webers",
    description="Secular, binormal, geometric part of the gradB drift divided "
    "by the toroidal angle. This quantity is periodic.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|^2", "b", "e^rho", "grad(|B|)", "iota_r"],
)
def _secular_gbdrift_over_phi(params, transforms, profiles, data, **kwargs):
    data["gbdrift (secular)/phi"] = (
        dot(data["b"], cross(data["e^rho"], data["grad(|B|)"]))
        * data["iota_r"]
        / data["|B|^2"]
    )
    return data


@register_compute_fun(
    name="cvdrift",
    # Exact definition of the magnetic drifts taken from
    # eqn. 48 of Introduction to Quasisymmetry by Landreman
    # https://tinyurl.com/54udvaa4
    label="\\mathrm{cvdrift} = 1/B^{3} (\\mathbf{b}\\times\\nabla( \\mu_0 p + B^2/2))"
    + "\\cdot \\nabla \\alpha",
    units="1 / Wb",
    units_long="Inverse webers",
    description="Binormal, geometric part of the curvature drift. "
    "Used for local stability analyses and gyrokinetics.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["cvdrift (periodic)", "gbdrift (secular)"],
)
def _cvdrift(params, transforms, profiles, data, **kwargs):
    data["cvdrift"] = data["cvdrift (periodic)"] + data["gbdrift (secular)"]
    return data


@register_compute_fun(
    name="cvdrift (periodic)",
    label="\\mathrm{cvdrift (periodic)}",
    units="1 / Wb",
    units_long="Inverse webers",
    description="Periodic, binormal, geometric part of the curvature drift.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["p_r", "psi_r", "|B|^2", "gbdrift (periodic)"],
)
def _periodic_cvdrift(params, transforms, profiles, data, **kwargs):
    data["cvdrift (periodic)"] = (
        mu_0 * data["p_r"] / data["psi_r"] / data["|B|^2"] + data["gbdrift (periodic)"]
    )
    return data


@register_compute_fun(
    name="cvdrift0",
    # Exact definition of the magnetic drifts taken from
    # eqn. 48 of Introduction to Quasisymmetry by Landreman
    # https://tinyurl.com/54udvaa4 up to dimensionless factors.
    label="\\mathrm{cvdrift0} = 1/B^{2} (\\mathbf{b}\\times\\nabla \\vert B \\vert)"
    + "\\cdot (2 \\rho \\nabla \\rho)",
    units="1 / Wb",
    units_long="Inverse webers",
    description="Radial, geometric part of the curvature drift."
    + " Used for local stability analyses, gyrokinetics, and Gamma_c.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["rho", "|B|^2", "b", "e^rho", "grad(|B|)"],
)
def _cvdrift0(params, transforms, profiles, data, **kwargs):
    data["cvdrift0"] = (
        2
        * data["rho"]
        / data["|B|^2"]
        * dot(data["b"], cross(data["grad(|B|)"], data["e^rho"]))
    )
    return data


################################################################################
##########-----------------METRIC ELEMENTS PEST----------------------###########
################################################################################


@register_compute_fun(
    name="g_rr|PEST",
    label="g_{\\rho\\rho}|PEST",
    units="m^{2}",
    units_long="square meters",
    description="Radial-Radial element of covariant metric tensor"
    + " PEST_coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho|v,p"],
)
def _g_sub_rr_PEST(params, transforms, profiles, data, **kwargs):
    data["g_rr|PEST"] = dot(data["e_rho|v,p"], data["e_rho|v,p"])
    return data


@register_compute_fun(
    name="g_rv|PEST",
    label="g_{\\rho\\vartheta}|PEST",
    units="m^{2}",
    units_long="square meters",
    description="Radial-Poloidal element of covariant metric tensor"
    + " PEST_coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho|v,p", "e_theta_PEST"],
    aliases=["g_vr|PEST"],
)
def _g_sub_rv_PEST(params, transforms, profiles, data, **kwargs):
    data["g_rv|PEST"] = dot(data["e_rho|v,p"], data["e_theta_PEST"])
    return data


@register_compute_fun(
    name="g_rp|PEST",
    label="g_{\\rho\\phi}|PEST",
    units="m^{2}",
    units_long="square meters",
    description="Radial-Toroidal element of covariant metric tensor"
    + " PEST_coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho|v,p", "e_phi|r,v"],
    aliases=["g_zr|PEST"],
)
def _g_sub_rp_PEST(params, transforms, profiles, data, **kwargs):
    data["g_rp|PEST"] = dot(data["e_rho|v,p"], data["e_phi|r,v"])
    return data


@register_compute_fun(
    name="g_vv|PEST",
    label="g_{\\vartheta \\vartheta}|PEST",
    units="m^{2}",
    units_long="square meters",
    description="Poloidal-Poloidal element of covariant metric tensor"
    + " PEST_coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta_PEST"],
)
def _g_sub_vv_PEST(params, transforms, profiles, data, **kwargs):
    data["g_vv|PEST"] = dot(data["e_theta_PEST"], data["e_theta_PEST"])
    return data


@register_compute_fun(
    name="g_vp|PEST",
    label="g_{\\vartheta \\phi}|PEST",
    units="m^{2}",
    units_long="square meters",
    description="Poloidal-Toroidal element of covariant metric tensor"
    + " PEST_coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta_PEST", "e_phi|r,v"],
    aliases=["g_zv|PEST"],
)
def _g_sub_vp_PEST(params, transforms, profiles, data, **kwargs):
    data["g_vp|PEST"] = dot(data["e_theta_PEST"], data["e_phi|r,v"])
    return data


@register_compute_fun(
    name="g_pp|PEST",
    label="g_{\\phi \\phi}|PEST",
    units="m^{2}",
    units_long="square meters",
    description="Toroidal-Toroidal element of covariant metric tensor"
    + " PEST_coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_phi|r,v"],
)
def _g_sub_pp_PEST(params, transforms, profiles, data, **kwargs):
    data["g_pp|PEST"] = dot(data["e_phi|r,v"], data["e_phi|r,v"])
    return data


@register_compute_fun(
    name="g^rv",
    label="g^{\\rho \\vartheta}",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Radial-Poloidal element of covariant metric tensor"
    + " PEST_coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^vartheta"],
)
def _g_sup_rv(params, transforms, profiles, data, **kwargs):
    data["g^rv"] = dot(data["e^rho"], data["e^vartheta"])
    return data


#################################################################################
############------------METRIC ELEMENT DERIVATIVES PEST---------------###########
#################################################################################


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(g_rr_v)|PEST",
    label="\\partial_{\\theta_PEST}|_{\\rho, \\phi} g_{\\rho\\rho}|PEST",
    units="m^{2}",
    units_long="square meters",
    description="Radial-Radial element of covariant metric tensor in PEST"
    + " coordinates, derivative w.r.t poloidal PEST coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho|v,p", "(e_rho_v)|PEST"],
)
def _g_sub_rr_v_PEST(params, transforms, profiles, data, **kwargs):
    data["(g_rr_v)|PEST"] = 2 * dot(data["e_rho|v,p"], data["(e_rho_v)|PEST"])
    return data


# TODO: Generalize for a general phi before #568
@register_compute_fun(
    name="(g_rr_p)|PEST",
    label="\\partial_{\\phi}|_{\\rho, \\vartheta} g_{\\rho\\rho}|PEST",
    units="m^{2}",
    units_long="square meters",
    description="Radial-Radial element of covariant metric tensor in PEST"
    + " coordinates, derivative w.r.t toroidal cylindrical coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho|v,p", "(e_rho_p)|PEST"],
)
def _g_sub_rr_p_PEST(params, transforms, profiles, data, **kwargs):
    data["(g_rr_p)|PEST"] = 2 * dot(data["e_rho|v,p"], data["(e_rho_p)|PEST"])
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(g_vv_r)|PEST",
    label="\\partial_{\\rho}|_{\\phi, \\vartheta} g_{\\vartheta \\vartheta}|PEST",
    units="m^{2}",
    units_long="square meters",
    description="Poloidal-Poloidal element of covariant metric tensor in PEST"
    + " coordinates, derivative w.r.t radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_vartheta|r,p", "(e_vartheta_r)|PEST"],
)
def _g_sub_vv_r_PEST(params, transforms, profiles, data, **kwargs):
    data["(g_vv_r)|PEST"] = 2 * dot(data["e_vartheta|r,p"], data["(e_vartheta_r)|PEST"])
    return data


# TODO: Generalize for a general phi before #568
@register_compute_fun(
    name="(g_vv_p)|PEST",
    label="\\partial_{\\phi}|_{\\rho, \\vartheta} g_{\\vartheta\\vartheta}|PEST",
    units="m^{2}",
    units_long="square meters",
    description="Poloidal-Poloidal element of covariant metric tensor in PEST"
    + " coordinates, derivative w.r.t toroidal cylindrical coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_vartheta|r,p", "(e_vartheta_p)|PEST"],
)
def _g_sub_vv_p_PEST(params, transforms, profiles, data, **kwargs):
    data["(g_vv_p)|PEST"] = 2 * dot(data["e_vartheta|r,p"], data["(e_vartheta_p)|PEST"])
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(g_pp_v)|PEST",
    label="\\partial_{\\theta_PEST}|_{\\rho, \\phi} g_{\\phi\\phi}|PEST",
    units="m^{2}",
    units_long="square meters",
    description="Toroidal-Toroidal element of covariant metric tensor in PEST"
    + " coordinates, derivative w.r.t poloidal PEST coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_phi|r,v", "(e_phi_v)|PEST"],
)
def _g_sub_pp_v_PEST(params, transforms, profiles, data, **kwargs):
    data["(g_pp_v)|PEST"] = 2 * dot(data["e_phi|r,v"], data["(e_phi_v)|PEST"])
    return data


# TODO: Generalize for a general phi before #568
@register_compute_fun(
    name="(g_rv_p)|PEST",
    label="a\\partial_{\\phi}|_{\\rho, \\vartheta} g_{\\rho\\vartheta}|PEST",
    units="m^{2}",
    units_long="square meters",
    description="Radial-Poloidal element of covariant metric tensor in PEST"
    + " coordinates, derivative w.r.t toroidal cylindrical coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho|v,p", "e_vartheta|r,p", "(e_rho_p)|PEST", "(e_vartheta_p)|PEST"],
)
def _g_sub_rv_p_PEST(params, transforms, profiles, data, **kwargs):
    data["(g_rv_p)|PEST"] = dot(data["e_rho|v,p"], data["(e_vartheta_p)|PEST"]) + dot(
        data["(e_rho_p)|PEST"], data["e_vartheta|r,p"]
    )
    return data


# TODO: Generalize for a general phi before #568
@register_compute_fun(
    name="(g^rr_p)|PEST",
    label="\\partial_{\\phi}|_{\\rho, \\vartheta} g^{\\rho\\rho}|PEST",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Radial-Radial element of contravariant metric tensor in PEST"
    + " coordinates, derivative w.r.t toroidal cylindrical coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "(e^rho_p)|PEST"],
)
def _g_sup_rr_p_PEST(params, transforms, profiles, data, **kwargs):
    data["(g^rr_p)|PEST"] = 2 * dot(data["e^rho"], data["(e^rho_p)|PEST"])
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(g^rr_v)|PEST",
    label="\\partial_{\\vartheta}|_{\\rho, \\phi} g^{\\rho\\rho}|PEST",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Radial-Radial element of contravariant metric tensor in PEST"
    + " coordinates, derivative w.r.t poloidal PEST coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "(e^rho_v)|PEST"],
)
def _g_sup_rr_v_PEST(params, transforms, profiles, data, **kwargs):
    data["(g^rr_v)|PEST"] = 2 * dot(data["e^rho"], data["(e^rho_v)|PEST"])
    return data


# TODO: Generalize for a general phi before #568
@register_compute_fun(
    name="(g^rv_p)|PEST",
    label="\\partial_{\\phi}|_{\\rho, \\vartheta} g^{\\rho\\vartheta}|PEST",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Radial-Poloidal element of contravariant metric tensor in PEST"
    + " coordinates, derivative w.r.t toroidal cylidrical coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^theta_PEST", "(e^rho_p)|PEST", "(e^theta_PEST_p)|PEST"],
)
def _g_sup_rv_p_PEST(params, transforms, profiles, data, **kwargs):
    data["(g^rv_p)|PEST"] = dot(data["e^rho"], data["(e^theta_PEST_p)|PEST"]) + dot(
        data["(e^rho_p)|PEST"], data["e^theta_PEST"]
    )
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(g^rv_v)|PEST",
    label="\\partial_{\\vartheta}|_{\\rho, \\phi} g^{\\rho\\vartheta}|PEST",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Radial-Poloidal element of contrvariant metric tensor in PEST"
    + " coordinates, derivative w.r.t poloidal PEST coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^theta_PEST", "(e^rho_v)|PEST", "(e^theta_PEST_v)|PEST"],
)
def _g_sup_rv_v_PEST(params, transforms, profiles, data, **kwargs):
    data["(g^rv_v)|PEST"] = dot(data["e^rho"], data["(e^theta_PEST_v)|PEST"]) + dot(
        data["(e^rho_v)|PEST"], data["e^theta_PEST"]
    )
    return data


# TODO: Generalize for a general phi before #568
@register_compute_fun(
    name="(g^rz_p)|PEST",
    label="\\partial_{\\phi}|_{\\rho, \\vartheta} g^{\\rho\\zeta}|PEST",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Radial-Toroidal element of contravariant metric tensor in PEST"
    + " coordinates, derivative w.r.t toroidal cylindrical coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^zeta", "(e^rho_p)|PEST", "(e^zeta_p)|PEST"],
)
def _g_sup_rz_p_PEST(params, transforms, profiles, data, **kwargs):
    data["(g^rz_p)|PEST"] = dot(data["e^rho"], data["(e^zeta_p)|PEST"]) + dot(
        data["(e^rho_p)|PEST"], data["e^zeta"]
    )
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(g^rz_v)|PEST",
    label="\\partial_{\\vartheta}|_{\\rho, \\phi} g^{\\rho\\zeta}|PEST",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Radial-Toroidal element of contrvariant metric tensor in PEST"
    + " coordinates, derivative w.r.t polodal PEST coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "(e^rho_v)|PEST", "(e^zeta_v)|PEST", "e^zeta"],
)
def _g_sup_rz_v_PEST(params, transforms, profiles, data, **kwargs):
    data["(g^rz_v)|PEST"] = dot(data["e^rho"], data["(e^zeta_v)|PEST"]) + dot(
        data["(e^rho_v)|PEST"], data["e^zeta"]
    )
    return data


################################################################################
#############--------------JACOBIAN DERIVATIVES PEST---------------#############
################################################################################


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(sqrt(g)_PEST_r)|PEST",
    label="\\partial_{\\rho}|_{\\phi, \\vartheta} \\sqrt{g}_PEST",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of PEST coordinate system"
    + " derivative w.r.t radial coordinate rho",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho|v,p",
        "e_theta_PEST",
        "e_phi|r,v",
        "(e_rho_r)|PEST",
        "(e_theta_PEST_r)|PEST",
        "(e_phi_r)|PEST",
    ],
)
def _sqrtg_PEST_r_PEST(params, transforms, profiles, data, **kwargs):
    data["(sqrt(g)_PEST_r)|PEST"] = (
        dot(data["(e_rho_r)|PEST"], cross(data["e_theta_PEST"], data["e_phi|r,v"]))
        + dot(
            data["e_rho|v,p"], cross(data["(e_theta_PEST_r)|PEST"], data["e_phi|r,v"])
        )
        + dot(data["e_rho|v,p"], cross(data["e_theta_PEST"], data["(e_phi_r)|PEST"]))
    )
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(sqrt(g)_PEST_v)|PEST",
    label="\\partial_{\\vartheta}|_{\\rho, \\phi} \\sqrt{g}_PEST",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of PEST coordinate system"
    + " derivative w.r.t PEST poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho|v,p",
        "e_theta_PEST",
        "e_phi|r,v",
        "(e_rho_v)|PEST",
        "(e_theta_PEST_v)|PEST",
        "(e_phi_v)|PEST",
    ],
)
def _sqrtg_PEST_theta_PEST_PEST(params, transforms, profiles, data, **kwargs):
    # TODO: This can be computed more efficiently without building radial
    #       derivatives of the stream functions by taking vartheta derivative
    #       of formula used to compute sqrt(g)_PEST.
    data["(sqrt(g)_PEST_v)|PEST"] = (
        dot(data["(e_rho_v)|PEST"], cross(data["e_theta_PEST"], data["e_phi|r,v"]))
        + dot(
            data["e_rho|v,p"], cross(data["(e_theta_PEST_v)|PEST"], data["e_phi|r,v"])
        )
        + dot(data["e_rho|v,p"], cross(data["e_theta_PEST"], data["(e_phi_v)|PEST"]))
    )
    return data


# TODO: Generalize for a general phi before #568
@register_compute_fun(
    name="(sqrt(g)_PEST_p)|PEST",
    label="\\partial_{\\phi}|_{\\rho, \\vartheta} \\sqrt{g}_PEST",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant of PEST coordinate system"
    + " derivative w.r.t cylindrical toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho|v,p",
        "e_theta_PEST",
        "e_phi|r,v",
        "(e_theta_PEST_p)|PEST",
        "(e_rho_p)|PEST",
        "(e_phi_p)|PEST",
    ],
)
def _sqrtg_PEST_phi_PEST(params, transforms, profiles, data, **kwargs):
    # TODO: This can be computed more efficiently without building radial
    #       derivatives of the stream functions by taking phi derivative
    #       of formula used to compute sqrt(g)_PEST.
    data["(sqrt(g)_PEST_p)|PEST"] = (
        dot(data["(e_rho_p)|PEST"], cross(data["e_theta_PEST"], data["e_phi|r,v"]))
        + dot(
            data["e_rho|v,p"], cross(data["(e_theta_PEST_p)|PEST"], data["e_phi|r,v"])
        )
        + dot(data["e_rho|v,p"], cross(data["e_theta_PEST"], data["(e_phi_p)|PEST"]))
    )
    return data


################################################################################
#############--------------FINITE-n STABILITY METRICS--------------#############
################################################################################


@register_compute_fun(
    name="(B*grad) grad(rho)",
    label="\\nabla(\\nabla(\\rho))",
    units="T m^{-2}",
    units_long="Tesla over square meters",
    description="Gradient of contravariant radial basis vector(grad rho)"
    "along the magnetic field scaled by the magnetic field strength",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho_t", "e^rho_z", "B^theta", "B^zeta"],
)
def _B_dot_grad_grad_rho(params, transforms, profiles, data, **kwargs):
    data["(B*grad) grad(rho)"] = (
        data["B^theta"][:, jnp.newaxis] * data["e^rho_t"]
        + data["B^zeta"][:, jnp.newaxis] * data["e^rho_z"]
    )
    return data


@register_compute_fun(
    name="finite-n instability drive",
    label="(\\mathbf{J} \\times (\\nabla \\rho))/{(g^{\\rho \\rho})}^2"
    + " \\mathbf{B} \\cdot \\mathbf{\\nabla} (\\mathbf{\\nabla} \\rho)",
    units="T A \\cdot m^{-1}",
    units_long="Tesla Amperes / meter",
    description="finite-n instability drive term",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J x grad(rho)", "g^rr", "(B*grad) grad(rho)"],
)
def _finite_n_instability_driver(params, transforms, profiles, data, **kwargs):
    """
    Taken from the TERPSICHORE paper.

    TERPSICHORE: A THREE-DIMENSIONAL IDEAL MAGNETOHYDRODYNAMIC STABILITY PROGRAM
    (https://doi.org/10.1007/978-1-4613-0659-7_8). Equation (5) on page 162.
    In the paper, the instability drive term uses s (= ρ²), but we have replaced
    all instances of s with ρ.
    """
    data["finite-n instability drive"] = (
        2 * dot(data["J x grad(rho)"], data["(B*grad) grad(rho)"]) / data["g^rr"] ** 2
    )
    return data
