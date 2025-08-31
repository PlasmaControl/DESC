"""Compute functions for coordinate system basis vectors.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from desc.backend import jnp

from ..utils import cross, dot, safediv
from .data_index import register_compute_fun


@register_compute_fun(
    name="b",
    label="\\hat{b}",
    units="~",
    units_long="None",
    description="Unit vector along magnetic field",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "|B|"],
)
def _b(params, transforms, profiles, data, **kwargs):
    data["b"] = (data["B"].T / data["|B|"]).T
    return data


@register_compute_fun(
    name="e^rho",  # ∇ρ is the same in any coordinate system.
    label="\\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant radial basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta/sqrt(g)", "e_zeta"],
    aliases=["grad(rho)"],
)
def _e_sup_rho(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { 𝐞^ρ | ρ=0 }.
    data["e^rho"] = cross(data["e_theta/sqrt(g)"], data["e_zeta"])
    return data


@register_compute_fun(
    name="e^rho_r",
    label="\\partial_{\\rho} \\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant radial basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_zeta", "e_theta_r", "e_zeta_r", "sqrt(g)", "sqrt(g)_r"],
    axis_limit_data=["e_theta_rr", "sqrt(g)_rr"],
)
def _e_sup_rho_r(params, transforms, profiles, data, **kwargs):
    a = cross(data["e_theta_r"], data["e_zeta"])
    data["e^rho_r"] = transforms["grid"].replace_at_axis(
        (
            safediv((a + cross(data["e_theta"], data["e_zeta_r"])).T, data["sqrt(g)"])
            - cross(data["e_theta"], data["e_zeta"]).T
            * safediv(data["sqrt(g)_r"], data["sqrt(g)"] ** 2)
        ).T,
        lambda: (
            safediv(
                (
                    cross(data["e_theta_rr"], data["e_zeta"])
                    + 2 * cross(data["e_theta_r"], data["e_zeta_r"])
                ).T,
                (2 * data["sqrt(g)_r"]),
            )
            - a.T * safediv(data["sqrt(g)_rr"], (2 * data["sqrt(g)_r"] ** 2))
        ).T,
    )
    return data


@register_compute_fun(
    name="e^rho_rr",
    label="\\partial_{\\rho\\rho} \\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Radial basis vector, 2nd derivative"
    " wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_theta",
        "e_theta_r",
        "e_theta_rr",
        "e_zeta",
        "e_zeta_r",
        "e_zeta_rr",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rr",
    ],
)
def _e_sup_rho_rr(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_theta"], data["e_zeta"])
    temp_r = cross(data["e_theta_r"], data["e_zeta"]) + cross(
        data["e_theta"], data["e_zeta_r"]
    )

    temp_rr = (
        cross(data["e_theta_rr"], data["e_zeta"])
        + cross(data["e_theta_r"], data["e_zeta_r"])
        + cross(data["e_theta_r"], data["e_zeta_r"])
        + cross(data["e_theta"], data["e_zeta_rr"])
    )

    data["e^rho_rr"] = (
        temp_rr.T / data["sqrt(g)"]
        - (
            temp_r.T * data["sqrt(g)_r"]
            + temp_r.T * data["sqrt(g)_r"]
            + temp.T * data["sqrt(g)_rr"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_r"] * data["sqrt(g)_r"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^rho_rt",
    label="\\partial_{\\rho\\theta} \\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Radial basis vector, derivative"
    " wrt radial and poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_theta",
        "e_theta_r",
        "e_theta_rt",
        "e_theta_t",
        "e_zeta",
        "e_zeta_r",
        "e_zeta_rt",
        "e_zeta_t",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rt",
        "sqrt(g)_t",
    ],
)
def _e_sup_rho_rt(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_theta"], data["e_zeta"])
    temp_r = cross(data["e_theta_r"], data["e_zeta"]) + cross(
        data["e_theta"], data["e_zeta_r"]
    )
    temp_t = cross(data["e_theta_t"], data["e_zeta"]) + cross(
        data["e_theta"], data["e_zeta_t"]
    )
    temp_rt = (
        cross(data["e_theta_rt"], data["e_zeta"])
        + cross(data["e_theta_r"], data["e_zeta_t"])
        + cross(data["e_theta_t"], data["e_zeta_r"])
        + cross(data["e_theta"], data["e_zeta_rt"])
    )

    data["e^rho_rt"] = (
        temp_rt.T / data["sqrt(g)"]
        - (
            temp_r.T * data["sqrt(g)_t"]
            + temp_t.T * data["sqrt(g)_r"]
            + temp.T * data["sqrt(g)_rt"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_r"] * data["sqrt(g)_t"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^rho_rz",
    label="\\partial_{\\rho\\zeta} \\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Radial basis vector, derivative"
    " wrt radial and toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_theta",
        "e_theta_r",
        "e_theta_rz",
        "e_theta_z",
        "e_zeta",
        "e_zeta_r",
        "e_zeta_rz",
        "e_zeta_z",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rz",
        "sqrt(g)_z",
    ],
)
def _e_sup_rho_rz(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_theta"], data["e_zeta"])
    temp_r = cross(data["e_theta_r"], data["e_zeta"]) + cross(
        data["e_theta"], data["e_zeta_r"]
    )
    temp_z = cross(data["e_theta_z"], data["e_zeta"]) + cross(
        data["e_theta"], data["e_zeta_z"]
    )
    temp_rz = (
        cross(data["e_theta_rz"], data["e_zeta"])
        + cross(data["e_theta_r"], data["e_zeta_z"])
        + cross(data["e_theta_z"], data["e_zeta_r"])
        + cross(data["e_theta"], data["e_zeta_rz"])
    )

    data["e^rho_rz"] = (
        temp_rz.T / data["sqrt(g)"]
        - (
            temp_r.T * data["sqrt(g)_z"]
            + temp_z.T * data["sqrt(g)_r"]
            + temp.T * data["sqrt(g)_rz"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_r"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^rho_t",
    label="\\partial_{\\theta} \\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant radial basis vector, derivative wrt poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_zeta", "e_theta_t", "e_zeta_t", "sqrt(g)", "sqrt(g)_t"],
    axis_limit_data=["e_theta_r", "e_theta_rt", "sqrt(g)_r", "sqrt(g)_rt"],
)
def _e_sup_rho_t(params, transforms, profiles, data, **kwargs):
    data["e^rho_t"] = transforms["grid"].replace_at_axis(
        (
            safediv(
                (
                    cross(data["e_theta_t"], data["e_zeta"])
                    + cross(data["e_theta"], data["e_zeta_t"])
                ).T,
                data["sqrt(g)"],
            )
            - safediv(
                cross(data["e_theta"], data["e_zeta"]).T * data["sqrt(g)_t"],
                data["sqrt(g)"] ** 2,
            )
        ).T,
        lambda: (
            safediv(cross(data["e_theta_rt"], data["e_zeta"]).T, data["sqrt(g)_r"])
            - safediv(
                cross(data["e_theta_r"], data["e_zeta"]).T * data["sqrt(g)_rt"],
                data["sqrt(g)_r"] ** 2,
            )
        ).T,
    )
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(e^rho_v)|PEST",  # ∇ρ is the same in any coordinate system.
    label="\\partial_{\\vartheta}|_{\\rho, \\phi} \\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant radial basis vector"
    + " derivative w.r.t the poloidal PEST coordinate.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho_t", "theta_PEST_t"],
)
def _e_sup_rho_v_PEST(params, transforms, profiles, data, **kwargs):
    data["(e^rho_v)|PEST"] = data["e^rho_t"] / data["theta_PEST_t"][:, None]
    return data


# TODO:Generalize for a general zeta before #568
@register_compute_fun(
    name="(e^rho_p)|PEST",  # ∇ρ is the same in any coordinate system.
    label="\\partial_{\\phi}|_{\\vartheta, \\rho} \\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant radial basis vector"
    + " derivative w.r.t the cylindrical toroidal angle.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho_t", "e^rho_z", "theta_PEST_t", "theta_PEST_z"],
)
def _e_sup_rho_p_PEST(params, transforms, profiles, data, **kwargs):
    data["(e^rho_p)|PEST"] = (
        data["e^rho_z"]
        - data["e^rho_t"] * (data["theta_PEST_z"] / data["theta_PEST_t"])[:, None]
    )
    return data


@register_compute_fun(
    name="e^rho_tt",
    label="\\partial_{\\theta\\theta} \\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Radial basis vector, 2nd derivative"
    " wrt poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_theta",
        "e_theta_t",
        "e_theta_tt",
        "e_zeta",
        "e_zeta_t",
        "e_zeta_tt",
        "sqrt(g)",
        "sqrt(g)_t",
        "sqrt(g)_tt",
    ],
)
def _e_sup_rho_tt(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_theta"], data["e_zeta"])
    temp_t = cross(data["e_theta_t"], data["e_zeta"]) + cross(
        data["e_theta"], data["e_zeta_t"]
    )

    temp_tt = (
        cross(data["e_theta_tt"], data["e_zeta"])
        + cross(data["e_theta_t"], data["e_zeta_t"])
        + cross(data["e_theta_t"], data["e_zeta_t"])
        + cross(data["e_theta"], data["e_zeta_tt"])
    )

    data["e^rho_tt"] = (
        temp_tt.T / data["sqrt(g)"]
        - (
            temp_t.T * data["sqrt(g)_t"]
            + temp_t.T * data["sqrt(g)_t"]
            + temp.T * data["sqrt(g)_tt"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_t"] * data["sqrt(g)_t"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^rho_tz",
    label="\\partial_{\\theta\\zeta} \\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Radial basis vector, derivative"
    " wrt poloidal and toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_theta",
        "e_theta_t",
        "e_theta_tz",
        "e_theta_z",
        "e_zeta",
        "e_zeta_t",
        "e_zeta_tz",
        "e_zeta_z",
        "sqrt(g)",
        "sqrt(g)_t",
        "sqrt(g)_tz",
        "sqrt(g)_z",
    ],
)
def _e_sup_rho_tz(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_theta"], data["e_zeta"])
    temp_t = cross(data["e_theta_t"], data["e_zeta"]) + cross(
        data["e_theta"], data["e_zeta_t"]
    )
    temp_z = cross(data["e_theta_z"], data["e_zeta"]) + cross(
        data["e_theta"], data["e_zeta_z"]
    )
    temp_tz = (
        cross(data["e_theta_tz"], data["e_zeta"])
        + cross(data["e_theta_t"], data["e_zeta_z"])
        + cross(data["e_theta_z"], data["e_zeta_t"])
        + cross(data["e_theta"], data["e_zeta_tz"])
    )

    data["e^rho_tz"] = (
        temp_tz.T / data["sqrt(g)"]
        - (
            temp_t.T * data["sqrt(g)_z"]
            + temp_z.T * data["sqrt(g)_t"]
            + temp.T * data["sqrt(g)_tz"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_t"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^rho_z",
    label="\\partial_{\\zeta} \\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant radial basis vector, derivative wrt toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_zeta", "e_theta_z", "e_zeta_z", "sqrt(g)", "sqrt(g)_z"],
    axis_limit_data=["e_theta_r", "e_theta_rz", "sqrt(g)_r", "sqrt(g)_rz"],
)
def _e_sup_rho_z(params, transforms, profiles, data, **kwargs):
    data["e^rho_z"] = transforms["grid"].replace_at_axis(
        (
            safediv(
                (
                    cross(data["e_theta_z"], data["e_zeta"])
                    + cross(data["e_theta"], data["e_zeta_z"])
                ).T,
                data["sqrt(g)"],
            )
            - cross(data["e_theta"], data["e_zeta"]).T
            * safediv(data["sqrt(g)_z"], data["sqrt(g)"] ** 2)
        ).T,
        lambda: (
            safediv(
                (
                    cross(data["e_theta_r"], data["e_zeta_z"])
                    + cross(data["e_theta_rz"], data["e_zeta"])
                ).T,
                data["sqrt(g)_r"],
            )
            - cross(data["e_theta_r"], data["e_zeta"]).T
            * safediv(data["sqrt(g)_rz"], data["sqrt(g)_r"] ** 2)
        ).T,
    )
    return data


@register_compute_fun(
    name="e^rho_zz",
    label="\\partial_{\\zeta\\zeta} \\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Radial basis vector, 2nd derivative"
    " wrt toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_theta",
        "e_theta_z",
        "e_theta_zz",
        "e_zeta",
        "e_zeta_z",
        "e_zeta_zz",
        "sqrt(g)",
        "sqrt(g)_z",
        "sqrt(g)_zz",
    ],
)
def _e_sup_rho_zz(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_theta"], data["e_zeta"])
    temp_z = cross(data["e_theta_z"], data["e_zeta"]) + cross(
        data["e_theta"], data["e_zeta_z"]
    )

    temp_zz = (
        cross(data["e_theta_zz"], data["e_zeta"])
        + cross(data["e_theta_z"], data["e_zeta_z"])
        + cross(data["e_theta_z"], data["e_zeta_z"])
        + cross(data["e_theta"], data["e_zeta_zz"])
    )

    data["e^rho_zz"] = (
        temp_zz.T / data["sqrt(g)"]
        - (
            temp_z.T * data["sqrt(g)_z"]
            + temp_z.T * data["sqrt(g)_z"]
            + temp.T * data["sqrt(g)_zz"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_z"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^theta",
    label="\\mathbf{e}^{\\theta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant poloidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^theta*sqrt(g)", "sqrt(g)"],
)
def _e_sup_theta(params, transforms, profiles, data, **kwargs):
    data["e^theta"] = (data["e^theta*sqrt(g)"].T / data["sqrt(g)"]).T
    return data


@register_compute_fun(
    name="e^vartheta",
    label="\\mathbf{e}^{\\vartheta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant PEST poloidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^theta", "e^rho", "e^zeta", "theta_PEST_r", "theta_PEST_t", "theta_PEST_z"],
    aliases=["e^theta_PEST"],
)
def _e_sup_theta_PEST(params, transforms, profiles, data, **kwargs):
    data["e^vartheta"] = (
        data["theta_PEST_r"][:, jnp.newaxis] * data["e^rho"]
        + data["theta_PEST_t"][:, jnp.newaxis] * data["e^theta"]
        + data["theta_PEST_z"][:, jnp.newaxis] * data["e^zeta"]
    )
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(e^vartheta_v)|PEST",
    label="\\partial_{\\vartheta}|_{\\rho, \\phi} \\mathbf{e}^{\\vartheta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant poloidal PEST basis vector"
    + " derivative wrt theta poloidal PEST coordinate ϑ.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^theta_t",
        "e^theta",
        "e^zeta",
        "e^zeta_t",
        "e^rho",
        "e^rho_t",
        "theta_PEST_r",
        "theta_PEST_t",
        "theta_PEST_z",
        "theta_PEST_rt",
        "theta_PEST_tz",
        "theta_PEST_tt",
    ],
    aliases=["(e^theta_PEST_v)|PEST"],
)
def _e_sup_vartheta_v_PEST(params, transforms, profiles, data, **kwargs):
    # ∂(𝐞^ϑ)/∂θ|(ρ,ϕ)
    # This is a mixed derivative so it has been removed as a compute function
    # until the convention for naming such variables becomes clear
    e_sup_vartheta_t = (
        data["e^theta_t"] * data["theta_PEST_t"][:, None]
        + data["e^zeta_t"] * data["theta_PEST_z"][:, None]
        + data["e^rho_t"] * data["theta_PEST_r"][:, None]
        + data["e^theta"] * data["theta_PEST_tt"][:, None]
        + data["e^zeta"] * data["theta_PEST_tz"][:, None]
        + data["e^rho"] * data["theta_PEST_rt"][:, None]
    )

    data["(e^vartheta_v)|PEST"] = e_sup_vartheta_t / (data["theta_PEST_t"])[:, None]
    return data


# TODO:Generalize for a general zeta before #568
@register_compute_fun(
    name="(e^vartheta_p)|PEST",
    label="\\partial_{\\phi}\\lvert_{\\rho, \\vartheta}(\\mathbf{e}^{\\vartheta})",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant poloidal PEST basis vector, derivative wrt"
    " cylindrical toroidal coordinate ϕ.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^rho",
        "e^theta",
        "e^zeta",
        "e^rho_z",
        "e^theta_z",
        "e^zeta_z",
        "e^rho_t",
        "theta_PEST_r",
        "theta_PEST_t",
        "theta_PEST_z",
        "theta_PEST_rt",
        "theta_PEST_tz",
        "theta_PEST_rz",
        "theta_PEST_tt",
        "theta_PEST_zz",
        "e^theta_t",
        "e^zeta_t",
    ],
    aliases=["(e^theta_PEST_p)|PEST"],
)
def _e_sup_vartheta_p_PEST(params, transforms, profiles, data, **kwargs):
    # This is a mixed derivative so it has been removed as a compute function
    # until the convention for naming such variables becomes clear
    e_sup_vartheta_t = (
        data["e^theta_t"] * data["theta_PEST_t"][:, None]
        + data["e^zeta_t"] * data["theta_PEST_z"][:, None]
        + data["e^rho_t"] * data["theta_PEST_r"][:, None]
        + data["e^theta"] * data["theta_PEST_tt"][:, None]
        + data["e^zeta"] * data["theta_PEST_tz"][:, None]
        + data["e^rho"] * data["theta_PEST_rt"][:, None]
    )
    e_sup_vartheta_z = (
        data["e^theta_z"] * data["theta_PEST_t"][:, None]
        + data["e^zeta_z"] * data["theta_PEST_z"][:, None]
        + data["e^rho_z"] * data["theta_PEST_r"][:, None]
        + data["e^theta"] * data["theta_PEST_tz"][:, None]
        + data["e^zeta"] * data["theta_PEST_zz"][:, None]
        + data["e^rho"] * data["theta_PEST_rz"][:, None]
    )
    data["(e^vartheta_p)|PEST"] = (
        e_sup_vartheta_z
        - (data["theta_PEST_z"] / data["theta_PEST_t"])[:, None] * e_sup_vartheta_t
    )
    return data


@register_compute_fun(
    name="e^theta*sqrt(g)",
    label="\\mathbf{e}^{\\theta} \\sqrt{g}",
    units="m^{2}",
    units_long="square meters",
    description="Contravariant poloidal basis vector weighted by 3-D volume Jacobian",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_zeta"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
    ],
)
def _e_sup_theta_times_sqrt_g(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { 𝐞^θ √g | ρ=0 }.
    data["e^theta*sqrt(g)"] = cross(data["e_zeta"], data["e_rho"])
    return data


@register_compute_fun(
    name="e^theta_r",
    label="\\partial_{\\rho} \\mathbf{e}^{\\theta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant poloidal basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_zeta", "e_rho", "e_zeta_r", "e_rho_r", "sqrt(g)", "sqrt(g)_r"],
)
def _e_sup_theta_r(params, transforms, profiles, data, **kwargs):
    data["e^theta_r"] = (
        (
            cross(data["e_zeta_r"], data["e_rho"])
            + cross(data["e_zeta"], data["e_rho_r"])
        ).T
        / data["sqrt(g)"]
        - cross(data["e_zeta"], data["e_rho"]).T
        * data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
    ).T
    return data


@register_compute_fun(
    name="e^theta_rr",
    label="\\partial_{\\rho\\rho} \\mathbf{e}^{\\theta}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Poloidal basis vector, 2nd derivative"
    " wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_rho_r",
        "e_rho_rr",
        "e_zeta",
        "e_zeta_r",
        "e_zeta_rr",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rr",
    ],
)
def _e_sup_theta_rr(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_zeta"], data["e_rho"])
    temp_r = cross(data["e_zeta_r"], data["e_rho"]) + cross(
        data["e_zeta"], data["e_rho_r"]
    )

    temp_rr = (
        cross(data["e_zeta_rr"], data["e_rho"])
        + cross(data["e_zeta_r"], data["e_rho_r"])
        + cross(data["e_zeta_r"], data["e_rho_r"])
        + cross(data["e_zeta"], data["e_rho_rr"])
    )

    data["e^theta_rr"] = (
        temp_rr.T / data["sqrt(g)"]
        - (
            temp_r.T * data["sqrt(g)_r"]
            + temp_r.T * data["sqrt(g)_r"]
            + temp.T * data["sqrt(g)_rr"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_r"] * data["sqrt(g)_r"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^theta_rt",
    label="\\partial_{\\rho\\theta} \\mathbf{e}^{\\theta}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Poloidal basis vector, derivative"
    " wrt radial and poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_rho_r",
        "e_rho_rt",
        "e_rho_t",
        "e_zeta",
        "e_zeta_r",
        "e_zeta_rt",
        "e_zeta_t",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rt",
        "sqrt(g)_t",
    ],
)
def _e_sup_theta_rt(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_zeta"], data["e_rho"])
    temp_r = cross(data["e_zeta_r"], data["e_rho"]) + cross(
        data["e_zeta"], data["e_rho_r"]
    )
    temp_t = cross(data["e_zeta_t"], data["e_rho"]) + cross(
        data["e_zeta"], data["e_rho_t"]
    )
    temp_rt = (
        cross(data["e_zeta_rt"], data["e_rho"])
        + cross(data["e_zeta_r"], data["e_rho_t"])
        + cross(data["e_zeta_t"], data["e_rho_r"])
        + cross(data["e_zeta"], data["e_rho_rt"])
    )

    data["e^theta_rt"] = (
        temp_rt.T / data["sqrt(g)"]
        - (
            temp_r.T * data["sqrt(g)_t"]
            + temp_t.T * data["sqrt(g)_r"]
            + temp.T * data["sqrt(g)_rt"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_r"] * data["sqrt(g)_t"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^theta_rz",
    label="\\partial_{\\rho\\zeta} \\mathbf{e}^{\\theta}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Poloidal basis vector, derivative"
    " wrt radial and toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_rho_r",
        "e_rho_rz",
        "e_rho_z",
        "e_zeta",
        "e_zeta_r",
        "e_zeta_rz",
        "e_zeta_z",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rz",
        "sqrt(g)_z",
    ],
)
def _e_sup_theta_rz(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_zeta"], data["e_rho"])
    temp_r = cross(data["e_zeta_r"], data["e_rho"]) + cross(
        data["e_zeta"], data["e_rho_r"]
    )
    temp_z = cross(data["e_zeta_z"], data["e_rho"]) + cross(
        data["e_zeta"], data["e_rho_z"]
    )
    temp_rz = (
        cross(data["e_zeta_rz"], data["e_rho"])
        + cross(data["e_zeta_r"], data["e_rho_z"])
        + cross(data["e_zeta_z"], data["e_rho_r"])
        + cross(data["e_zeta"], data["e_rho_rz"])
    )

    data["e^theta_rz"] = (
        temp_rz.T / data["sqrt(g)"]
        - (
            temp_r.T * data["sqrt(g)_z"]
            + temp_z.T * data["sqrt(g)_r"]
            + temp.T * data["sqrt(g)_rz"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_r"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^theta_t",
    label="\\partial_{\\theta} \\mathbf{e}^{\\theta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant poloidal basis vector, derivative wrt poloidal"
    " coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_zeta", "e_rho", "e_zeta_t", "e_rho_t", "sqrt(g)", "sqrt(g)_t"],
)
def _e_sup_theta_t(params, transforms, profiles, data, **kwargs):
    data["e^theta_t"] = (
        (
            cross(data["e_zeta_t"], data["e_rho"])
            + cross(data["e_zeta"], data["e_rho_t"])
        ).T
        / data["sqrt(g)"]
        - cross(data["e_zeta"], data["e_rho"]).T
        * data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
    ).T
    return data


@register_compute_fun(
    name="e^theta_tt",
    label="\\partial_{\\theta\\theta} \\mathbf{e}^{\\theta}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Poloidal basis vector, 2nd derivative"
    " wrt poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_rho_t",
        "e_rho_tt",
        "e_zeta",
        "e_zeta_t",
        "e_zeta_tt",
        "sqrt(g)",
        "sqrt(g)_t",
        "sqrt(g)_tt",
    ],
)
def _e_sup_theta_tt(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_zeta"], data["e_rho"])
    temp_t = cross(data["e_zeta_t"], data["e_rho"]) + cross(
        data["e_zeta"], data["e_rho_t"]
    )

    temp_tt = (
        cross(data["e_zeta_tt"], data["e_rho"])
        + cross(data["e_zeta_t"], data["e_rho_t"])
        + cross(data["e_zeta_t"], data["e_rho_t"])
        + cross(data["e_zeta"], data["e_rho_tt"])
    )

    data["e^theta_tt"] = (
        temp_tt.T / data["sqrt(g)"]
        - (
            temp_t.T * data["sqrt(g)_t"]
            + temp_t.T * data["sqrt(g)_t"]
            + temp.T * data["sqrt(g)_tt"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_t"] * data["sqrt(g)_t"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^theta_tz",
    label="\\partial_{\\theta\\zeta} \\mathbf{e}^{\\theta}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Poloidal basis vector, derivative"
    " wrt poloidal and toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_rho_t",
        "e_rho_tz",
        "e_rho_z",
        "e_zeta",
        "e_zeta_t",
        "e_zeta_tz",
        "e_zeta_z",
        "sqrt(g)",
        "sqrt(g)_t",
        "sqrt(g)_tz",
        "sqrt(g)_z",
    ],
)
def _e_sup_theta_tz(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_zeta"], data["e_rho"])
    temp_t = cross(data["e_zeta_t"], data["e_rho"]) + cross(
        data["e_zeta"], data["e_rho_t"]
    )
    temp_z = cross(data["e_zeta_z"], data["e_rho"]) + cross(
        data["e_zeta"], data["e_rho_z"]
    )
    temp_tz = (
        cross(data["e_zeta_tz"], data["e_rho"])
        + cross(data["e_zeta_t"], data["e_rho_z"])
        + cross(data["e_zeta_z"], data["e_rho_t"])
        + cross(data["e_zeta"], data["e_rho_tz"])
    )

    data["e^theta_tz"] = (
        temp_tz.T / data["sqrt(g)"]
        - (
            temp_t.T * data["sqrt(g)_z"]
            + temp_z.T * data["sqrt(g)_t"]
            + temp.T * data["sqrt(g)_tz"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_t"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^theta_z",
    label="\\partial_{\\zeta} \\mathbf{e}^{\\theta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant poloidal basis vector, derivative wrt toroidal"
    " coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_zeta", "e_rho", "e_zeta_z", "e_rho_z", "sqrt(g)", "sqrt(g)_z"],
)
def _e_sup_theta_z(params, transforms, profiles, data, **kwargs):
    data["e^theta_z"] = (
        (
            cross(data["e_zeta_z"], data["e_rho"])
            + cross(data["e_zeta"], data["e_rho_z"])
        ).T
        / data["sqrt(g)"]
        - cross(data["e_zeta"], data["e_rho"]).T
        * data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
    ).T
    return data


@register_compute_fun(
    name="e^theta_zz",
    label="\\partial_{\\zeta\\zeta} \\mathbf{e}^{\\theta}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Poloidal basis vector, 2nd derivative"
    " wrt toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_rho_z",
        "e_rho_zz",
        "e_zeta",
        "e_zeta_z",
        "e_zeta_zz",
        "sqrt(g)",
        "sqrt(g)_z",
        "sqrt(g)_zz",
    ],
)
def _e_sup_theta_zz(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_zeta"], data["e_rho"])
    temp_z = cross(data["e_zeta_z"], data["e_rho"]) + cross(
        data["e_zeta"], data["e_rho_z"]
    )

    temp_zz = (
        cross(data["e_zeta_zz"], data["e_rho"])
        + cross(data["e_zeta_z"], data["e_rho_z"])
        + cross(data["e_zeta_z"], data["e_rho_z"])
        + cross(data["e_zeta"], data["e_rho_zz"])
    )

    data["e^theta_zz"] = (
        temp_zz.T / data["sqrt(g)"]
        - (
            temp_z.T * data["sqrt(g)_z"]
            + temp_z.T * data["sqrt(g)_z"]
            + temp.T * data["sqrt(g)_zz"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_z"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^zeta",  # ∇ζ is the same in any coordinate system.
    label="\\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant toroidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_theta/sqrt(g)"],
)
def _e_sup_zeta(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { 𝐞^ζ | ρ=0 }.
    data["e^zeta"] = cross(data["e_rho"], data["e_theta/sqrt(g)"])
    return data


@register_compute_fun(
    name="e^zeta_r",
    label="\\partial_{\\rho} \\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant toroidal basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_rho_r", "e_theta", "e_theta_r", "sqrt(g)", "sqrt(g)_r"],
    axis_limit_data=["e_theta_rr", "sqrt(g)_rr"],
)
def _e_sup_zeta_r(params, transforms, profiles, data, **kwargs):
    b = cross(data["e_rho"], data["e_theta_r"])
    data["e^zeta_r"] = transforms["grid"].replace_at_axis(
        (
            safediv((cross(data["e_rho_r"], data["e_theta"]) + b).T, data["sqrt(g)"])
            - cross(data["e_rho"], data["e_theta"]).T
            * safediv(data["sqrt(g)_r"], data["sqrt(g)"] ** 2)
        ).T,
        lambda: (
            safediv(
                (
                    2 * cross(data["e_rho_r"], data["e_theta_r"])
                    + cross(data["e_rho"], data["e_theta_rr"])
                ).T,
                (2 * data["sqrt(g)_r"]),
            )
            - b.T * safediv(data["sqrt(g)_rr"], (2 * data["sqrt(g)_r"] ** 2))
        ).T,
    )
    return data


@register_compute_fun(
    name="e^zeta_rr",
    label="\\partial_{\\rho\\rho} \\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Toroidal basis vector, 2nd derivative"
    " wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_rho_r",
        "e_rho_rr",
        "e_theta",
        "e_theta_r",
        "e_theta_rr",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rr",
    ],
)
def _e_sup_zeta_rr(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_rho"], data["e_theta"])
    temp_r = cross(data["e_rho_r"], data["e_theta"]) + cross(
        data["e_rho"], data["e_theta_r"]
    )

    temp_rr = (
        cross(data["e_rho_rr"], data["e_theta"])
        + cross(data["e_rho_r"], data["e_theta_r"])
        + cross(data["e_rho_r"], data["e_theta_r"])
        + cross(data["e_rho"], data["e_theta_rr"])
    )

    data["e^zeta_rr"] = (
        temp_rr.T / data["sqrt(g)"]
        - (
            temp_r.T * data["sqrt(g)_r"]
            + temp_r.T * data["sqrt(g)_r"]
            + temp.T * data["sqrt(g)_rr"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_r"] * data["sqrt(g)_r"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^zeta_rt",
    label="\\partial_{\\rho\\theta} \\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Toroidal basis vector, derivative"
    " wrt radial and poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_rho_r",
        "e_rho_rt",
        "e_rho_t",
        "e_theta",
        "e_theta_r",
        "e_theta_rt",
        "e_theta_t",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rt",
        "sqrt(g)_t",
    ],
)
def _e_sup_zeta_rt(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_rho"], data["e_theta"])
    temp_r = cross(data["e_rho_r"], data["e_theta"]) + cross(
        data["e_rho"], data["e_theta_r"]
    )
    temp_t = cross(data["e_rho_t"], data["e_theta"]) + cross(
        data["e_rho"], data["e_theta_t"]
    )
    temp_rt = (
        cross(data["e_rho_rt"], data["e_theta"])
        + cross(data["e_rho_r"], data["e_theta_t"])
        + cross(data["e_rho_t"], data["e_theta_r"])
        + cross(data["e_rho"], data["e_theta_rt"])
    )

    data["e^zeta_rt"] = (
        temp_rt.T / data["sqrt(g)"]
        - (
            temp_r.T * data["sqrt(g)_t"]
            + temp_t.T * data["sqrt(g)_r"]
            + temp.T * data["sqrt(g)_rt"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_r"] * data["sqrt(g)_t"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^zeta_rz",
    label="\\partial_{\\rho\\zeta} \\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Toroidal basis vector, derivative"
    " wrt radial and toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_rho_r",
        "e_rho_rz",
        "e_rho_z",
        "e_theta",
        "e_theta_r",
        "e_theta_rz",
        "e_theta_z",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rz",
        "sqrt(g)_z",
    ],
)
def _e_sup_zeta_rz(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_rho"], data["e_theta"])
    temp_r = cross(data["e_rho_r"], data["e_theta"]) + cross(
        data["e_rho"], data["e_theta_r"]
    )
    temp_z = cross(data["e_rho_z"], data["e_theta"]) + cross(
        data["e_rho"], data["e_theta_z"]
    )
    temp_rz = (
        cross(data["e_rho_rz"], data["e_theta"])
        + cross(data["e_rho_r"], data["e_theta_z"])
        + cross(data["e_rho_z"], data["e_theta_r"])
        + cross(data["e_rho"], data["e_theta_rz"])
    )

    data["e^zeta_rz"] = (
        temp_rz.T / data["sqrt(g)"]
        - (
            temp_r.T * data["sqrt(g)_z"]
            + temp_z.T * data["sqrt(g)_r"]
            + temp.T * data["sqrt(g)_rz"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_r"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^zeta_t",
    label="\\partial_{\\theta} \\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant toroidal basis vector, derivative wrt poloidal"
    " coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_rho_t", "e_theta", "e_theta_t", "sqrt(g)", "sqrt(g)_t"],
    axis_limit_data=["e_theta_r", "e_theta_rt", "sqrt(g)_r", "sqrt(g)_rt"],
)
def _e_sup_zeta_t(params, transforms, profiles, data, **kwargs):
    data["e^zeta_t"] = transforms["grid"].replace_at_axis(
        (
            safediv(
                (
                    cross(data["e_rho_t"], data["e_theta"])
                    + cross(data["e_rho"], data["e_theta_t"])
                ).T,
                data["sqrt(g)"],
            )
            - cross(data["e_rho"], data["e_theta"]).T
            * safediv(data["sqrt(g)_t"], data["sqrt(g)"] ** 2)
        ).T,
        lambda: (
            safediv(
                (
                    cross(data["e_rho_t"], data["e_theta_r"])
                    + cross(data["e_rho"], data["e_theta_rt"])
                ).T,
                data["sqrt(g)_r"],
            )
            - cross(data["e_rho"], data["e_theta_r"]).T
            * safediv(data["sqrt(g)_rt"], data["sqrt(g)_r"] ** 2)
        ).T,
    )
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(e^zeta_v)|PEST",
    label="\\partial_{\\vartheta}_{\\rho,\\phi} \\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant toroidal basis vector, derivative wrt theta poloidal"
    " PEST coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^zeta_t", "theta_PEST_t"],
)
def _e_sup_zeta_v_PEST(params, transforms, profiles, data, **kwargs):
    data["(e^zeta_v)|PEST"] = data["e^zeta_t"] / data["theta_PEST_t"][:, None]
    return data


# TODO:Generalize for a general zeta before #568
@register_compute_fun(
    name="(e^zeta_p)|PEST",
    label="\\partial_{\\phi}|_{\\rho,\\vartheta} \\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant toroidal basis vector, derivative wrt cylindrical"
    " toroidal coordinate phi at constant rho and vartheta",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^zeta_t", "e^zeta_z", "theta_PEST_t", "theta_PEST_z"],
)
def _e_sup_zeta_p_PEST(params, transforms, profiles, data, **kwargs):
    data["(e^zeta_p)|PEST"] = (
        data["e^zeta_z"]
        - data["e^zeta_t"] * (data["theta_PEST_z"] / data["theta_PEST_t"])[:, None]
    )
    return data


@register_compute_fun(
    name="e^zeta_tt",
    label="\\partial_{\\theta\\theta} \\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Toroidal basis vector, 2nd derivative"
    " wrt poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_rho_t",
        "e_rho_tt",
        "e_theta",
        "e_theta_t",
        "e_theta_tt",
        "sqrt(g)",
        "sqrt(g)_t",
        "sqrt(g)_tt",
    ],
)
def _e_sup_zeta_tt(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_rho"], data["e_theta"])
    temp_t = cross(data["e_rho_t"], data["e_theta"]) + cross(
        data["e_rho"], data["e_theta_t"]
    )

    temp_tt = (
        cross(data["e_rho_tt"], data["e_theta"])
        + cross(data["e_rho_t"], data["e_theta_t"])
        + cross(data["e_rho_t"], data["e_theta_t"])
        + cross(data["e_rho"], data["e_theta_tt"])
    )

    data["e^zeta_tt"] = (
        temp_tt.T / data["sqrt(g)"]
        - (
            temp_t.T * data["sqrt(g)_t"]
            + temp_t.T * data["sqrt(g)_t"]
            + temp.T * data["sqrt(g)_tt"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_t"] * data["sqrt(g)_t"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^zeta_tz",
    label="\\partial_{\\theta\\zeta} \\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Toroidal basis vector, derivative"
    " wrt poloidal and toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_rho_t",
        "e_rho_tz",
        "e_rho_z",
        "e_theta",
        "e_theta_t",
        "e_theta_tz",
        "e_theta_z",
        "sqrt(g)",
        "sqrt(g)_t",
        "sqrt(g)_tz",
        "sqrt(g)_z",
    ],
)
def _e_sup_zeta_tz(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_rho"], data["e_theta"])
    temp_t = cross(data["e_rho_t"], data["e_theta"]) + cross(
        data["e_rho"], data["e_theta_t"]
    )
    temp_z = cross(data["e_rho_z"], data["e_theta"]) + cross(
        data["e_rho"], data["e_theta_z"]
    )
    temp_tz = (
        cross(data["e_rho_tz"], data["e_theta"])
        + cross(data["e_rho_t"], data["e_theta_z"])
        + cross(data["e_rho_z"], data["e_theta_t"])
        + cross(data["e_rho"], data["e_theta_tz"])
    )

    data["e^zeta_tz"] = (
        temp_tz.T / data["sqrt(g)"]
        - (
            temp_t.T * data["sqrt(g)_z"]
            + temp_z.T * data["sqrt(g)_t"]
            + temp.T * data["sqrt(g)_tz"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_t"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e^zeta_z",
    label="\\partial_{\\zeta} \\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant toroidal basis vector, derivative wrt toroidal"
    " coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_rho_z", "e_theta", "e_theta_z", "sqrt(g)", "sqrt(g)_z"],
    axis_limit_data=["e_theta_r", "e_theta_rz", "sqrt(g)_r", "sqrt(g)_rz"],
)
def _e_sup_zeta_z(params, transforms, profiles, data, **kwargs):
    data["e^zeta_z"] = transforms["grid"].replace_at_axis(
        (
            safediv(
                (
                    cross(data["e_rho_z"], data["e_theta"])
                    + cross(data["e_rho"], data["e_theta_z"])
                ).T,
                data["sqrt(g)"],
            )
            - cross(data["e_rho"], data["e_theta"]).T
            * safediv(data["sqrt(g)_z"], data["sqrt(g)"] ** 2)
        ).T,
        lambda: (
            safediv(
                (
                    cross(data["e_rho_z"], data["e_theta_r"])
                    + cross(data["e_rho"], data["e_theta_rz"])
                ).T,
                data["sqrt(g)_r"],
            )
            - cross(data["e_rho"], data["e_theta_r"]).T
            * safediv(data["sqrt(g)_rz"], data["sqrt(g)_r"] ** 2)
        ).T,
    )
    return data


@register_compute_fun(
    name="e^zeta_zz",
    label="\\partial_{\\zeta\\zeta} \\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse square meters",
    description="Contravariant Toroidal basis vector, 2nd derivative"
    " wrt toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho",
        "e_rho_z",
        "e_rho_zz",
        "e_theta",
        "e_theta_z",
        "e_theta_zz",
        "sqrt(g)",
        "sqrt(g)_z",
        "sqrt(g)_zz",
    ],
)
def _e_sup_zeta_zz(params, transforms, profiles, data, **kwargs):
    temp = cross(data["e_rho"], data["e_theta"])
    temp_z = cross(data["e_rho_z"], data["e_theta"]) + cross(
        data["e_rho"], data["e_theta_z"]
    )

    temp_zz = (
        cross(data["e_rho_zz"], data["e_theta"])
        + cross(data["e_rho_z"], data["e_theta_z"])
        + cross(data["e_rho_z"], data["e_theta_z"])
        + cross(data["e_rho"], data["e_theta_zz"])
    )

    data["e^zeta_zz"] = (
        temp_zz.T / data["sqrt(g)"]
        - (
            temp_z.T * data["sqrt(g)_z"]
            + temp_z.T * data["sqrt(g)_z"]
            + temp.T * data["sqrt(g)_zz"]
        )
        / data["sqrt(g)"] ** 2
        + 2 * temp.T * data["sqrt(g)_z"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 3
    ).T
    return data


@register_compute_fun(
    name="e_phi|r,t",
    label="\\mathbf{e}_{\\phi} |_{\\rho, \\theta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector in (ρ,θ,ϕ) coordinates",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_zeta", "phi_z"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    aliases=["e_phi"],
    # Our usual notation implies e_phi = (∂X/∂ϕ)|R,Z = R ϕ̂, but we need to alias e_phi
    # to e_phi|r,t = (∂X/∂ϕ)|ρ,θ for compatibility with older versions of the code.
)
def _e_sub_phi_rt(params, transforms, profiles, data, **kwargs):
    # (∂X/∂ϕ)|ρ,θ = (∂X/∂ζ)|ρ,θ / (∂ϕ/∂ζ)|ρ,θ
    data["e_phi|r,t"] = (data["e_zeta"].T / data["phi_z"]).T
    return data


@register_compute_fun(
    name="e_rho",
    label="\\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant Radial basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_r", "Z_r", "omega_r"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _e_sub_rho(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { 𝐞ᵨ | ρ=0 }.
    data["e_rho"] = jnp.array([data["R_r"], data["R"] * data["omega_r"], data["Z_r"]]).T
    return data


@register_compute_fun(
    name="e_rho_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant Radial basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_r", "R_rr", "Z_rr", "omega_r", "omega_rr"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _e_sub_rho_r(params, transforms, profiles, data, **kwargs):
    # e_rho_r = a^i e_i, where the a^i are the components specified below and the
    # e_i are the basis vectors of the polar lab frame. omega_r e_2, -omega_r e_1,
    # 0 are the derivatives with respect to rho of e_1, e_2, e_3, respectively.
    data["e_rho_r"] = jnp.array(
        [
            -data["R"] * data["omega_r"] ** 2 + data["R_rr"],
            2 * data["R_r"] * data["omega_r"] + data["R"] * data["omega_rr"],
            data["Z_rr"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, second derivative wrt radial and radial"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_rr",
        "R_rrr",
        "Z_rrr",
        "omega_r",
        "omega_rr",
        "omega_rrr",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _e_sub_rho_rr(params, transforms, profiles, data, **kwargs):
    data["e_rho_rr"] = jnp.array(
        [
            -3 * data["R_r"] * data["omega_r"] ** 2
            - 3 * data["R"] * data["omega_r"] * data["omega_rr"]
            + data["R_rrr"],
            3 * (data["omega_r"] * data["R_rr"] + data["R_r"] * data["omega_rr"])
            + data["R"] * (-data["omega_r"] ** 3 + data["omega_rrr"]),
            data["Z_rrr"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_rrr",
    label="\\partial_{\\rho \\rho \\rho} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, third derivative wrt radial coordinate"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_rr",
        "R_rrr",
        "R_rrrr",
        "Z_rrrr",
        "omega_r",
        "omega_rr",
        "omega_rrr",
        "omega_rrrr",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _e_sub_rho_rrr(params, transforms, profiles, data, **kwargs):
    data["e_rho_rrr"] = jnp.array(
        [
            -6 * data["R_rr"] * data["omega_r"] ** 2
            - 12 * data["R_r"] * data["omega_r"] * data["omega_rr"]
            + data["R"]
            * (
                -3 * data["omega_rr"] ** 2
                + data["omega_r"] ** 4
                - 4 * data["omega_rrr"] * data["omega_r"]
            )
            + data["R_rrrr"],
            4 * data["R_rrr"] * data["omega_r"]
            + 6 * data["R_rr"] * data["omega_rr"]
            - 4 * data["R_r"] * (data["omega_r"] ** 3 - data["omega_rrr"])
            + data["R_r"]
            * (data["omega_rrrr"] - 6 * data["omega_r"] ** 2 * data["omega_rr"]),
            data["Z_rrrr"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_rrt",
    label="\\partial_{\\rho \\rho \\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, third derivative wrt radial coordinate"
        " twice and poloidal once"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_rr",
        "R_rrr",
        "R_rrt",
        "R_rrrt",
        "R_rt",
        "R_t",
        "Z_rrrt",
        "omega_r",
        "omega_rr",
        "omega_rrr",
        "omega_rrt",
        "omega_rrrt",
        "omega_rt",
        "omega_t",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    aliases=["e_theta_rrr"],
)
def _e_sub_rho_rrt(params, transforms, profiles, data, **kwargs):
    data["e_rho_rrt"] = jnp.array(
        [
            -3 * data["R_rt"] * data["omega_r"] ** 2
            - 3 * data["R_t"] * data["omega_r"] * data["omega_rr"]
            - 4 * data["R_r"] * data["omega_r"] * data["omega_rt"]
            - 3
            * data["R"]
            * (
                data["omega_rr"] * data["omega_rt"]
                + data["omega_r"] * data["omega_rrt"]
            )
            - data["omega_rt"] * (2 * data["R_r"] * data["omega_r"])
            - data["omega_t"]
            * (
                3 * data["R_rr"] * data["omega_r"]
                + 3 * data["R_r"] * data["omega_rr"]
                + data["R"] * data["omega_rrr"]
            )
            + data["R_rrrt"]
            + data["omega_r"] * data["R"] * data["omega_t"] * data["omega_r"] ** 2,
            3 * data["omega_rr"] * data["R_rt"]
            + 3 * data["omega_r"] * data["R_rrt"]
            + 3 * data["R_rr"] * data["omega_rt"]
            + 2 * data["R_r"] * data["omega_rrt"]
            + data["omega_t"] * data["R_rrr"]
            + data["R_t"] * data["omega_rrr"]
            + data["R_r"]
            * (-3 * data["omega_t"] * data["omega_r"] ** 2 + data["omega_rrt"])
            + data["R"]
            * (
                -3 * data["omega_rt"] * data["omega_r"] ** 2
                - 3 * data["omega_t"] * data["omega_r"] * data["omega_rr"]
                + data["omega_rrrt"]
            )
            - data["R_t"] * data["omega_r"] ** 3,
            data["Z_rrrt"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_rrz",
    label="\\partial_{\\rho \\rho \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, third derivative wrt radial coordinate"
        " twice and toroidal once"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_rr",
        "R_rrr",
        "R_rrz",
        "R_rrrz",
        "R_rz",
        "R_z",
        "Z_rrrz",
        "omega_r",
        "omega_rr",
        "omega_rrr",
        "omega_rrz",
        "omega_rrrz",
        "omega_rz",
        "omega_z",
    ],
    aliases=["e_zeta_rrr"],
)
def _e_sub_rho_rrz(params, transforms, profiles, data, **kwargs):
    data["e_rho_rrz"] = jnp.array(
        [
            -3 * data["R"] * data["omega_rrz"] * data["omega_r"]
            - 3
            * data["omega_rz"]
            * (2 * data["R_r"] * data["omega_r"] + data["R"] * data["omega_rr"])
            - 3
            * data["omega_z"]
            * (data["R_rr"] * data["omega_r"] + data["R_r"] * data["omega_rr"])
            - (1 + data["omega_z"])
            * data["R"]
            * (data["omega_rrr"] - data["omega_r"] ** 3)
            - 3
            * data["omega_r"]
            * (
                data["R_rz"] * data["omega_r"]
                + data["R_z"] * data["omega_rr"]
                + data["R_rr"]
            )
            - 3 * data["R_r"] * data["omega_rr"]
            + data["R_rrrz"],
            3
            * data["R_r"]
            * (data["omega_rrz"] - (1 + data["omega_z"]) * data["omega_r"] ** 2)
            + 3 * data["omega_rz"] * data["R_rr"]
            + data["R"]
            * (
                data["omega_rrrz"]
                - 3
                * data["omega_r"]
                * (
                    data["omega_rz"] * data["omega_r"]
                    + (1 + data["omega_z"]) * data["omega_rr"]
                )
            )
            + (1 + data["omega_z"]) * data["R_rrr"]
            + 3 * data["R_rrz"] * data["omega_r"]
            + 3 * data["R_rz"] * data["omega_rr"]
            + data["R_z"] * (data["omega_rrr"] - data["omega_r"] ** 3),
            data["Z_rrrz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_rt",
    label="\\partial_{\\rho \\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, second derivative wrt radial and poloidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_rr",
        "R_rrt",
        "R_rt",
        "R_t",
        "Z_rrt",
        "omega_r",
        "omega_rr",
        "omega_rrt",
        "omega_rt",
        "omega_t",
    ],
    aliases=["x_rrt", "x_rtr", "x_trr", "e_theta_rr"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _e_sub_rho_rt(params, transforms, profiles, data, **kwargs):
    data["e_rho_rt"] = jnp.array(
        [
            -data["R_t"] * data["omega_r"] ** 2
            - 2 * data["R"] * data["omega_r"] * data["omega_rt"]
            - data["omega_t"]
            * (2 * data["R_r"] * data["omega_r"] + data["R"] * data["omega_rr"])
            + data["R_rrt"],
            2 * data["omega_r"] * data["R_rt"]
            + 2 * data["R_r"] * data["omega_rt"]
            + data["omega_t"] * data["R_rr"]
            + data["R_t"] * data["omega_rr"]
            + data["R"] * (-data["omega_t"] * data["omega_r"] ** 2 + data["omega_rrt"]),
            data["Z_rrt"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_rtt",
    label="\\partial_{\\rho \\theta \\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, third derivative wrt radial coordinate"
        "once and poloidal twice"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_rr",
        "R_rt",
        "R_rrt",
        "R_rtt",
        "R_rrtt",
        "R_t",
        "R_tt",
        "Z_rrtt",
        "omega_r",
        "omega_rr",
        "omega_rt",
        "omega_rrt",
        "omega_rtt",
        "omega_rrtt",
        "omega_t",
        "omega_tt",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    aliases=["e_theta_rrt"],
)
def _e_sub_rho_rtt(params, transforms, profiles, data, **kwargs):
    data["e_rho_rtt"] = jnp.array(
        [
            -data["R_rr"] * data["omega_t"] ** 2
            - 4 * data["R_rt"] * data["omega_r"] * data["omega_t"]
            - 4 * data["R_r"] * data["omega_rt"] * data["omega_t"]
            - 2 * data["R_t"] * data["omega_rr"] * data["omega_t"]
            - data["R_tt"] * data["omega_r"] ** 2
            - 4 * data["R_t"] * data["omega_r"] * data["omega_rt"]
            - data["omega_tt"]
            * (2 * data["R_r"] * data["omega_r"] + data["R"] * data["omega_rr"])
            + data["R"]
            * (
                (data["omega_r"] * data["omega_t"]) ** 2
                - 2 * (data["omega_rt"] ** 2 + data["omega_r"] * data["omega_rtt"])
                - 2 * (data["omega_t"] * data["omega_rrt"])
            )
            + data["R_rrtt"],
            -data["omega_t"] ** 2
            * (2 * data["R_r"] * data["omega_r"] + data["R"] * data["omega_rr"])
            + 2
            * data["omega_t"]
            * (
                data["R_rrt"]
                - data["omega_r"]
                * (data["R_t"] * data["omega_r"] + 2 * data["R"] * data["omega_rt"])
            )
            + 4 * data["R_rt"] * data["omega_rt"]
            + 2 * data["R_rtt"] * data["omega_r"]
            + 2 * data["R_r"] * data["omega_rtt"]
            + data["R_rr"] * data["omega_tt"]
            + data["R_tt"] * data["omega_rr"]
            + 2 * data["R_t"] * data["omega_rrt"]
            + data["R"]
            * (data["omega_rrtt"] - data["omega_tt"] * data["omega_r"] ** 2),
            data["Z_rrtt"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_rtz",
    label="\\partial_{\\rho \\theta \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, third derivative wrt radial, poloidal,"
        " and toroidal coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_rr",
        "R_rt",
        "R_rrt",
        "R_rtz",
        "R_rrtz",
        "R_rz",
        "R_rrz",
        "R_t",
        "R_tz",
        "R_z",
        "Z_rrtz",
        "omega_r",
        "omega_rr",
        "omega_rt",
        "omega_rrt",
        "omega_rtz",
        "omega_rrtz",
        "omega_rz",
        "omega_rrz",
        "omega_t",
        "omega_tz",
        "omega_z",
    ],
    aliases=["e_theta_rrz", "e_zeta_rrt"],
)
def _e_sub_rho_rtz(params, transforms, profiles, data, **kwargs):
    data["e_rho_rtz"] = jnp.array(
        [
            -data["omega_rz"] * data["R_t"] * data["omega_r"]
            - (1 + data["omega_z"])
            * (data["R_rt"] * data["omega_r"] + data["R_t"] * data["omega_rr"])
            - data["R_r"] * data["omega_tz"] * data["omega_r"]
            - data["R"]
            * (
                data["omega_rtz"] * data["omega_r"]
                + data["omega_tz"] * data["omega_rr"]
            )
            - data["omega_rt"]
            * (
                (1 + data["omega_z"]) * data["R_r"]
                + data["R_z"] * data["omega_r"]
                + data["R"] * data["omega_rz"]
            )
            - data["omega_t"]
            * (
                data["omega_rz"] * data["R_r"]
                + (1 + data["omega_z"]) * data["R_rr"]
                + data["R_rz"] * data["omega_r"]
                + data["R_z"] * data["omega_rr"]
                + data["R_r"] * data["omega_rz"]
                + data["R"] * data["omega_rrz"]
            )
            - data["R_r"] * (1 + data["omega_z"]) * data["omega_rt"]
            - data["R"]
            * (
                data["omega_rz"] * data["omega_rt"]
                + (1 + data["omega_z"]) * data["omega_rrt"]
            )
            + data["R_rrtz"]
            - data["omega_r"]
            * (
                data["omega_tz"] * data["R_r"]
                + data["R_tz"] * data["omega_r"]
                + data["omega_t"] * data["R_rz"]
                + data["R_t"] * data["omega_rz"]
                + (1 + data["omega_z"]) * data["R_rt"]
                + data["R_z"] * data["omega_rt"]
                + data["R"]
                * (
                    -(1 + data["omega_z"]) * data["omega_t"] * data["omega_r"]
                    + data["omega_rtz"]
                )
            ),
            data["omega_rtz"] * data["R_r"]
            + data["omega_tz"] * data["R_rr"]
            + data["R_rtz"] * data["omega_r"]
            + data["R_tz"] * data["omega_rr"]
            + data["omega_rt"] * data["R_rz"]
            + data["omega_t"] * data["R_rrz"]
            + data["R_rt"] * data["omega_rz"]
            + data["R_t"] * data["omega_rrz"]
            + data["omega_rz"] * data["R_rt"]
            + (1 + data["omega_z"]) * data["R_rrt"]
            + data["R_rz"] * data["omega_rt"]
            + data["R_z"] * data["omega_rrt"]
            + data["R_r"]
            * (
                -(1 + data["omega_z"]) * data["omega_t"] * data["omega_r"]
                + data["omega_rtz"]
            )
            + data["R"]
            * (
                -data["omega_rz"] * data["omega_t"] * data["omega_r"]
                - (1 + data["omega_z"])
                * (
                    data["omega_rt"] * data["omega_r"]
                    + data["omega_t"] * data["omega_rr"]
                )
                + data["omega_rrtz"]
            )
            + data["omega_r"]
            * (
                -((1 + data["omega_z"]) * data["R_t"] * data["omega_r"])
                - data["R"] * data["omega_tz"] * data["omega_r"]
                - data["omega_t"]
                * (
                    (1 + data["omega_z"]) * data["R_r"]
                    + data["R_z"] * data["omega_r"]
                    + data["R"] * data["omega_rz"]
                )
                - data["R"] * (1 + data["omega_z"]) * data["omega_rt"]
                + data["R_rtz"]
            ),
            data["Z_rrtz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_rz",
    label="\\partial_{\\rho \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, second derivative wrt radial and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_rr",
        "R_rrz",
        "R_rz",
        "R_z",
        "Z_rrz",
        "omega_r",
        "omega_rr",
        "omega_rrz",
        "omega_rz",
        "omega_z",
    ],
    aliases=["e_zeta_rr"],
)
def _e_sub_rho_rz(params, transforms, profiles, data, **kwargs):
    data["e_rho_rz"] = jnp.array(
        [
            -2 * (1 + data["omega_z"]) * data["R_r"] * data["omega_r"]
            - data["R_z"] * data["omega_r"] ** 2
            - 2 * data["R"] * data["omega_r"] * data["omega_rz"]
            - data["R"] * (1 + data["omega_z"]) * data["omega_rr"]
            + data["R_rrz"],
            2 * data["omega_r"] * data["R_rz"]
            + 2 * data["R_r"] * data["omega_rz"]
            + (1 + data["omega_z"]) * data["R_rr"]
            + data["R_z"] * data["omega_rr"]
            - data["R"]
            * ((1 + data["omega_z"]) * data["omega_r"] ** 2 - data["omega_rrz"]),
            data["Z_rrz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, third derivative wrt radial coordinate"
        " once and toroidal twice"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_rr",
        "R_rz",
        "R_rrz",
        "R_rzz",
        "R_rrzz",
        "R_z",
        "R_zz",
        "Z_rrzz",
        "omega_r",
        "omega_rr",
        "omega_rz",
        "omega_rrz",
        "omega_rzz",
        "omega_rrzz",
        "omega_z",
        "omega_zz",
    ],
    aliases=["e_zeta_rrz"],
)
def _e_sub_rho_rzz(params, transforms, profiles, data, **kwargs):
    data["e_rho_rzz"] = jnp.array(
        [
            -2 * (1 + data["omega_z"]) * data["omega_rz"] * data["R_r"]
            - (1 + data["omega_z"]) ** 2 * data["R_rr"]
            - 2 * data["R_rz"] * (1 + data["omega_z"]) * data["omega_r"]
            - 2
            * data["R_z"]
            * (
                data["omega_rz"] * data["omega_r"]
                + (1 + data["omega_z"]) * data["omega_rr"]
            )
            - data["R_r"] * data["omega_zz"] * data["omega_r"]
            - data["R"]
            * (
                data["omega_rzz"] * data["omega_r"]
                + data["omega_zz"] * data["omega_rr"]
            )
            - 2 * data["R_r"] * (1 + data["omega_z"]) * data["omega_rz"]
            - 2
            * data["R"]
            * (data["omega_rz"] ** 2 + (1 + data["omega_z"]) * data["omega_rrz"])
            + data["R_rrzz"]
            - data["omega_r"]
            * (
                data["omega_zz"] * data["R_r"]
                + data["R_zz"] * data["omega_r"]
                + 2 * (1 + data["omega_z"]) * data["R_rz"]
                + 2 * data["R_z"] * data["omega_rz"]
                - data["R"]
                * ((1 + data["omega_z"]) ** 2 * data["omega_r"] - data["omega_rzz"])
            ),
            data["omega_rzz"] * data["R_r"]
            + data["omega_zz"] * data["R_rr"]
            + data["R_rzz"] * data["omega_r"]
            + data["R_zz"] * data["omega_rr"]
            + 2 * data["omega_rz"] * data["R_rz"]
            + 2 * (1 + data["omega_z"]) * data["R_rrz"]
            + 2 * data["R_rz"] * data["omega_rz"]
            + 2 * data["R_z"] * data["omega_rrz"]
            - data["R_r"]
            * ((1 + data["omega_z"]) ** 2 * data["omega_r"] - data["omega_rzz"])
            - data["R"]
            * (
                2 * (1 + data["omega_z"]) * data["omega_rz"] * data["omega_r"]
                + (1 + data["omega_z"]) ** 2 * data["omega_rr"]
                - data["omega_rrzz"]
            )
            + data["omega_r"]
            * (
                -((1 + data["omega_z"]) ** 2) * data["R_r"]
                - 2 * data["R_z"] * (1 + data["omega_z"]) * data["omega_r"]
                - data["R"] * data["omega_zz"] * data["omega_r"]
                - 2 * data["R"] * (1 + data["omega_z"]) * data["omega_rz"]
                + data["R_rzz"]
            ),
            data["Z_rrzz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant Radial basis vector, derivative wrt poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_r", "R_rt", "R_t", "Z_rt", "omega_r", "omega_rt", "omega_t"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    aliases=["e_theta_r"],
)
def _e_sub_rho_t(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { ∂ᵨ 𝐞_θ | ρ=0 }
    data["e_rho_t"] = jnp.array(
        [
            -data["R"] * data["omega_t"] * data["omega_r"] + data["R_rt"],
            data["omega_t"] * data["R_r"]
            + data["R_t"] * data["omega_r"]
            + data["R"] * data["omega_rt"],
            data["Z_rt"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_tt",
    label="\\partial_{\\theta \\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, second derivative wrt poloidal and poloidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_rt",
        "R_rtt",
        "R_t",
        "R_tt",
        "Z_rtt",
        "omega_r",
        "omega_rt",
        "omega_rtt",
        "omega_t",
        "omega_tt",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    aliases=["e_theta_rt"],
)
def _e_sub_rho_tt(params, transforms, profiles, data, **kwargs):
    data["e_rho_tt"] = jnp.array(
        [
            -data["omega_t"] ** 2 * data["R_r"]
            - data["R"] * data["omega_tt"] * data["omega_r"]
            - 2
            * data["omega_t"]
            * (data["R_t"] * data["omega_r"] + data["R"] * data["omega_rt"])
            + data["R_rtt"],
            data["omega_tt"] * data["R_r"]
            + data["R_tt"] * data["omega_r"]
            + 2 * data["omega_t"] * data["R_rt"]
            + 2 * data["R_t"] * data["omega_rt"]
            + data["R"] * (-data["omega_t"] ** 2 * data["omega_r"] + data["omega_rtt"]),
            data["Z_rtt"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_tz",
    label="\\partial_{\\theta \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, second derivative wrt poloidal and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_rt",
        "R_rtz",
        "R_rz",
        "R_t",
        "R_tz",
        "R_z",
        "Z_rtz",
        "omega_r",
        "omega_rt",
        "omega_rtz",
        "omega_rz",
        "omega_t",
        "omega_tz",
        "omega_z",
    ],
    aliases=["e_theta_rz", "e_zeta_rt"],
)
def _e_sub_rho_tz(params, transforms, profiles, data, **kwargs):
    data["e_rho_tz"] = jnp.array(
        [
            -((1 + data["omega_z"]) * data["R_t"] * data["omega_r"])
            - data["R"] * data["omega_tz"] * data["omega_r"]
            - data["omega_t"]
            * (
                (1 + data["omega_z"]) * data["R_r"]
                + data["R_z"] * data["omega_r"]
                + data["R"] * data["omega_rz"]
            )
            - data["R"] * (1 + data["omega_z"]) * data["omega_rt"]
            + data["R_rtz"],
            data["omega_tz"] * data["R_r"]
            + data["R_tz"] * data["omega_r"]
            + data["omega_t"] * data["R_rz"]
            + data["R_t"] * data["omega_rz"]
            + (1 + data["omega_z"]) * data["R_rt"]
            + data["R_z"] * data["omega_rt"]
            + data["R"]
            * (
                -(1 + data["omega_z"]) * data["omega_t"] * data["omega_r"]
                + data["omega_rtz"]
            ),
            data["Z_rtz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant Radial basis vector, derivative wrt toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_r", "R_rz", "R_z", "Z_rz", "omega_r", "omega_rz", "omega_z"],
    aliases=["e_zeta_r"],
)
def _e_sub_rho_z(params, transforms, profiles, data, **kwargs):
    data["e_rho_z"] = jnp.array(
        [
            -data["R"] * (1 + data["omega_z"]) * data["omega_r"] + data["R_rz"],
            (1 + data["omega_z"]) * data["R_r"]
            + data["R_z"] * data["omega_r"]
            + data["R"] * data["omega_rz"],
            data["Z_rz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_rho_zz",
    label="\\partial_{\\zeta \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Radial basis vector, second derivative wrt toroidal and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_rz",
        "R_rzz",
        "R_z",
        "R_zz",
        "Z_rzz",
        "omega_r",
        "omega_rz",
        "omega_rzz",
        "omega_z",
        "omega_zz",
    ],
    aliases=["e_zeta_rz"],
)
def _e_sub_rho_zz(params, transforms, profiles, data, **kwargs):
    data["e_rho_zz"] = jnp.array(
        [
            -((1 + data["omega_z"]) ** 2) * data["R_r"]
            - 2 * data["R_z"] * (1 + data["omega_z"]) * data["omega_r"]
            - data["R"] * data["omega_zz"] * data["omega_r"]
            - 2 * data["R"] * (1 + data["omega_z"]) * data["omega_rz"]
            + data["R_rzz"],
            data["omega_zz"] * data["R_r"]
            + data["R_zz"] * data["omega_r"]
            + 2 * (1 + data["omega_z"]) * data["R_rz"]
            + 2 * data["R_z"] * data["omega_rz"]
            - data["R"]
            * ((1 + data["omega_z"]) ** 2 * data["omega_r"] - data["omega_rzz"]),
            data["Z_rzz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_theta",
    label="\\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant Poloidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_t", "Z_t", "omega_t"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _e_sub_theta(params, transforms, profiles, data, **kwargs):
    data["e_theta"] = jnp.array(
        [data["R_t"], data["R"] * data["omega_t"], data["Z_t"]]
    ).T

    return data


@register_compute_fun(
    name="e_theta/sqrt(g)",
    label="\\mathbf{e}_{\\theta} / \\sqrt{g}",
    units="m",
    units_long="meters",
    description="Covariant Poloidal basis vector divided by 3-D volume Jacobian",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "sqrt(g)"],
    axis_limit_data=["e_theta_r", "sqrt(g)_r"],
)
def _e_sub_theta_over_sqrt_g(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { 𝐞_θ / √g | ρ=0 }.
    data["e_theta/sqrt(g)"] = transforms["grid"].replace_at_axis(
        safediv(data["e_theta"].T, data["sqrt(g)"]).T,
        lambda: safediv(data["e_theta_r"].T, data["sqrt(g)_r"]).T,
    )
    return data


@register_compute_fun(
    name="e_theta_PEST",
    label="\\mathbf{e}_{\\vartheta} |_{\\rho, \\phi} = \\mathbf{e}_{\\theta_{PEST}}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector in (ρ,ϑ,ϕ) coordinates or"
    " straight field line PEST coordinates. ϕ increases counterclockwise"
    " when viewed from above (cylindrical R,ϕ plane with Z out of page).",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "theta_PEST_t", "e_zeta", "theta_PEST_z", "phi_t", "phi_z"],
    aliases=[
        "e_vartheta",
        "e_vartheta|r,p",
        "e_theta_PEST|r,p",
        "e_vartheta|p,r",
        "e_theta_PEST|p,r",
    ],
)
def _e_sub_vartheta_rp(params, transforms, profiles, data, **kwargs):
    # constant ρ and ϕ
    data["e_theta_PEST"] = (
        (data["e_theta"].T * data["phi_z"] - data["e_zeta"].T * data["phi_t"])
        / (data["theta_PEST_t"] * data["phi_z"] - data["theta_PEST_z"] * data["phi_t"])
    ).T
    return data


@register_compute_fun(
    name="e_phi|r,v",
    label="\\mathbf{e}_{\\phi} |_{\\rho, \\vartheta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector in (ρ,ϑ,ϕ) coordinates or"
    " straight field line PEST coordinates. ϕ increases counterclockwise"
    " when viewed from above (cylindrical R,ϕ plane with Z out of page).",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "theta_PEST_t", "e_zeta", "theta_PEST_z", "phi_t", "phi_z"],
    aliases=["e_phi|v,r"],
)
def _e_sub_phi_rv(params, transforms, profiles, data, **kwargs):
    # constant ρ and ϑ
    data["e_phi|r,v"] = (
        (
            data["e_zeta"].T * data["theta_PEST_t"]
            - data["e_theta"].T * data["theta_PEST_z"]
        )
        / (data["theta_PEST_t"] * data["phi_z"] - data["theta_PEST_z"] * data["phi_t"])
    ).T
    return data


@register_compute_fun(
    name="e_rho|v,p",
    label="\\mathbf{e}_{\\rho} |_{\\vartheta, \\phi}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector in (ρ,ϑ,ϕ) coordinates or"
    " straight field line PEST coordinates. ϕ increases counterclockwise"
    " when viewed from above (cylindrical R,ϕ plane with Z out of page).",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_vartheta|r,p", "e_phi|r,v", "theta_PEST_r", "phi_r"],
    aliases=["e_rho|p,v"],
)
def _e_sub_rho_vp(params, transforms, profiles, data, **kwargs):
    # constant ϑ and ϕ
    data["e_rho|v,p"] = (
        data["e_rho"]
        - data["e_vartheta|r,p"] * data["theta_PEST_r"][:, jnp.newaxis]
        - data["e_phi|r,v"] * data["phi_r"][:, jnp.newaxis]
    )
    return data


@register_compute_fun(
    name="e_theta_rtt",
    label="\\partial_{\\rho \\theta \\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, third derivative wrt radial coordinate"
        " once and poloidal twice"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_t",
        "R_rt",
        "R_tt",
        "R_rtt",
        "R_ttt",
        "R_rttt",
        "Z_rttt",
        "omega_r",
        "omega_t",
        "omega_rt",
        "omega_tt",
        "omega_rtt",
        "omega_ttt",
        "omega_rttt",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    aliases=["e_rho_ttt"],
)
def _e_sub_theta_rtt(params, transforms, profiles, data, **kwargs):
    data["e_theta_rtt"] = jnp.array(
        [
            -3 * data["R_rt"] * data["omega_t"] ** 2
            - 3
            * data["omega_t"]
            * (
                data["R_r"] * data["omega_tt"]
                + data["R_tt"] * data["omega_r"]
                + 2 * data["R_t"] * data["omega_rt"]
                + data["R"] * data["omega_rt"]
            )
            - data["omega_r"]
            * (3 * data["R_t"] * data["omega_tt"] + data["R"] * data["omega_ttt"])
            + data["R"]
            * (
                data["omega_r"] * data["omega_t"] ** 3
                - 3 * data["omega_rt"] * data["omega_tt"]
            )
            + data["R_rttt"],
            data["R_r"] * (data["omega_ttt"] - data["omega_t"] ** 3)
            + data["omega_r"]
            * (
                data["R_ttt"]
                - 3
                * data["omega_t"]
                * (data["R_t"] * data["omega_t"] + data["R"] * data["omega_tt"])
            )
            + 3
            * (
                data["R_rt"] * data["omega_tt"]
                + data["R_tt"] * data["omega_rt"]
                + data["R_rtt"] * data["omega_t"]
                + data["R_t"] * data["omega_rtt"]
            )
            + data["R"]
            * (data["omega_rttt"] - 3 * data["omega_t"] ** 2 * data["omega_rt"]),
            data["Z_rttt"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_theta_rtz",
    label="\\partial_{\\rho \\theta \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, third derivative wrt radial, poloidal,"
        " and toroidal coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_t",
        "R_rt",
        "R_tt",
        "R_rtt",
        "R_ttz",
        "R_rttz",
        "R_tz",
        "R_rtz",
        "R_z",
        "R_rz",
        "Z_rttz",
        "omega_r",
        "omega_t",
        "omega_rt",
        "omega_tt",
        "omega_rtt",
        "omega_ttz",
        "omega_rttz",
        "omega_tz",
        "omega_rtz",
        "omega_z",
        "omega_rz",
    ],
    aliases=["e_rho_ttz", "e_zeta_rtt"],
)
def _e_sub_theta_rtz(params, transforms, profiles, data, **kwargs):
    data["e_theta_rtz"] = jnp.array(
        [
            -2 * data["omega_rz"] * data["R_t"] * data["omega_t"]
            - 2
            * (1 + data["omega_z"])
            * (data["R_rt"] * data["omega_t"] + data["R_t"] * data["omega_rt"])
            - data["R_rz"] * data["omega_t"] ** 2
            - 2 * data["R_z"] * data["omega_t"] * data["omega_rt"]
            - 2 * data["R_r"] * data["omega_t"] * data["omega_tz"]
            - 2
            * data["R"]
            * (
                data["omega_rt"] * data["omega_tz"]
                + data["omega_t"] * data["omega_rtz"]
            )
            - data["R_r"] * (1 + data["omega_z"]) * data["omega_tt"]
            - data["R"]
            * (
                data["omega_rz"] * data["omega_tt"]
                + (1 + data["omega_z"]) * data["omega_rtt"]
            )
            + data["R_rttz"]
            - data["omega_r"]
            * (
                2 * data["omega_t"] * data["R_tz"]
                + 2 * data["R_t"] * data["omega_tz"]
                + (1 + data["omega_z"]) * data["R_tt"]
                + data["R_z"] * data["omega_tt"]
                - data["R"]
                * ((1 + data["omega_z"]) * data["omega_t"] ** 2 - data["omega_ttz"])
            ),
            2 * data["omega_rt"] * data["R_tz"]
            + 2 * data["omega_t"] * data["R_rtz"]
            + 2 * data["R_rt"] * data["omega_tz"]
            + 2 * data["R_t"] * data["omega_rtz"]
            + data["omega_rz"] * data["R_tt"]
            + (1 + data["omega_z"]) * data["R_rtt"]
            + data["R_rz"] * data["omega_tt"]
            + data["R_z"] * data["omega_rtt"]
            - data["R_r"]
            * ((1 + data["omega_z"]) * data["omega_t"] ** 2 - data["omega_ttz"])
            - data["R"]
            * (
                data["omega_rz"] * data["omega_t"] ** 2
                + (1 + data["omega_z"]) * 2 * data["omega_t"] * data["omega_rt"]
                - data["omega_rttz"]
            )
            + data["omega_r"]
            * (
                -2 * (1 + data["omega_z"]) * data["R_t"] * data["omega_t"]
                - data["R_z"] * data["omega_t"] ** 2
                - 2 * data["R"] * data["omega_t"] * data["omega_tz"]
                - data["R"] * (1 + data["omega_z"]) * data["omega_tt"]
                + data["R_ttz"]
            ),
            data["Z_rttz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_theta_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, third derivative wrt radial coordinate"
        " once and toroidal twice"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_t",
        "R_rt",
        "R_tz",
        "R_rtz",
        "R_tzz",
        "R_rtzz",
        "R_z",
        "R_rz",
        "R_zz",
        "R_rzz",
        "Z_rtzz",
        "omega_t",
        "omega_rt",
        "omega_tz",
        "omega_rtz",
        "omega_tzz",
        "omega_rtzz",
        "omega_r",
        "omega_z",
        "omega_zz",
        "omega_rz",
        "omega_rzz",
    ],
    aliases=["e_rho_tzz", "e_zeta_rtz"],
)
def _e_sub_theta_rzz(params, transforms, profiles, data, **kwargs):
    data["e_theta_rzz"] = jnp.array(
        [
            -2 * (1 + data["omega_z"]) * data["omega_rz"] * data["R_t"]
            - (1 + data["omega_z"]) ** 2 * data["R_rt"]
            - 2 * data["R_rz"] * (1 + data["omega_z"]) * data["omega_t"]
            - 2
            * data["R_z"]
            * (
                data["omega_rz"] * data["omega_t"]
                + (1 + data["omega_z"]) * data["omega_rt"]
            )
            - data["R_r"] * data["omega_zz"] * data["omega_t"]
            - data["R"]
            * (
                data["omega_rzz"] * data["omega_t"]
                + data["omega_zz"] * data["omega_rt"]
            )
            - 2 * data["R_r"] * (1 + data["omega_z"]) * data["omega_tz"]
            - 2
            * data["R"]
            * (
                data["omega_rz"] * data["omega_tz"]
                + (1 + data["omega_z"]) * data["omega_rtz"]
            )
            + data["R_rtzz"]
            - data["omega_r"]
            * (
                data["omega_zz"] * data["R_t"]
                + data["R_zz"] * data["omega_t"]
                + 2 * (1 + data["omega_z"]) * data["R_tz"]
                + 2 * data["R_z"] * data["omega_tz"]
                - data["R"]
                * ((1 + data["omega_z"]) ** 2 * data["omega_t"] - data["omega_tzz"])
            ),
            data["omega_rzz"] * data["R_t"]
            + data["omega_zz"] * data["R_rt"]
            + data["R_rzz"] * data["omega_t"]
            + data["R_zz"] * data["omega_rt"]
            + 2 * data["omega_rz"] * data["R_tz"]
            + 2 * (1 + data["omega_z"]) * data["R_rtz"]
            + 2 * data["R_rz"] * data["omega_tz"]
            + 2 * data["R_z"] * data["omega_rtz"]
            - data["R_r"]
            * ((1 + data["omega_z"]) ** 2 * data["omega_t"] - data["omega_tzz"])
            - data["R"]
            * (
                2 * (1 + data["omega_z"]) * data["omega_rz"] * data["omega_t"]
                + (1 + data["omega_z"]) ** 2 * data["omega_rt"]
                - data["omega_rtzz"]
            )
            + data["omega_r"]
            * (
                -((1 + data["omega_z"]) ** 2) * data["R_t"]
                - 2 * data["R_z"] * (1 + data["omega_z"]) * data["omega_t"]
                - data["R"] * data["omega_zz"] * data["omega_t"]
                - 2 * data["R"] * (1 + data["omega_z"]) * data["omega_tz"]
                + data["R_tzz"]
            ),
            data["Z_rtzz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_theta_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant Poloidal basis vector, derivative wrt poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_t", "R_tt", "Z_tt", "omega_t", "omega_tt"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _e_sub_theta_t(params, transforms, profiles, data, **kwargs):
    data["e_theta_t"] = jnp.array(
        [
            -data["R"] * data["omega_t"] ** 2 + data["R_tt"],
            2 * data["R_t"] * data["omega_t"] + data["R"] * data["omega_tt"],
            data["Z_tt"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_tt",
    label="\\partial_{\\theta \\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, second derivative wrt poloidal and poloidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_t",
        "R_tt",
        "R_ttt",
        "Z_ttt",
        "omega_t",
        "omega_tt",
        "omega_ttt",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _e_sub_theta_tt(params, transforms, profiles, data, **kwargs):
    data["e_theta_tt"] = jnp.array(
        [
            -3 * data["R_t"] * data["omega_t"] ** 2
            - 3 * data["R"] * data["omega_t"] * data["omega_tt"]
            + data["R_ttt"],
            3 * (data["omega_t"] * data["R_tt"] + data["R_t"] * data["omega_tt"])
            + data["R"] * (-data["omega_t"] ** 3 + data["omega_ttt"]),
            data["Z_ttt"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_theta_tz",
    label="\\partial_{\\theta \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, second derivative wrt poloidal and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_t",
        "R_tt",
        "R_ttz",
        "R_tz",
        "R_z",
        "Z_ttz",
        "omega_t",
        "omega_tt",
        "omega_ttz",
        "omega_tz",
        "omega_z",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    aliases=["e_zeta_tt"],
)
def _e_sub_theta_tz(params, transforms, profiles, data, **kwargs):
    data["e_theta_tz"] = jnp.array(
        [
            -2 * (1 + data["omega_z"]) * data["R_t"] * data["omega_t"]
            - data["R_z"] * data["omega_t"] ** 2
            - 2 * data["R"] * data["omega_t"] * data["omega_tz"]
            - data["R"] * (1 + data["omega_z"]) * data["omega_tt"]
            + data["R_ttz"],
            2 * data["omega_t"] * data["R_tz"]
            + 2 * data["R_t"] * data["omega_tz"]
            + (1 + data["omega_z"]) * data["R_tt"]
            + data["R_z"] * data["omega_tt"]
            - data["R"]
            * ((1 + data["omega_z"]) * data["omega_t"] ** 2 - data["omega_ttz"]),
            data["Z_ttz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_theta_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant Poloidal basis vector, derivative wrt toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_t", "R_tz", "R_z", "Z_tz", "omega_t", "omega_tz", "omega_z"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    aliases=["e_zeta_t"],
)
def _e_sub_theta_z(params, transforms, profiles, data, **kwargs):
    data["e_theta_z"] = jnp.array(
        [
            -data["R"] * (1 + data["omega_z"]) * data["omega_t"] + data["R_tz"],
            (1 + data["omega_z"]) * data["R_t"]
            + data["R_z"] * data["omega_t"]
            + data["R"] * data["omega_tz"],
            data["Z_tz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_theta_zz",
    label="\\partial_{\\zeta \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, second derivative wrt toroidal and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_t",
        "R_tz",
        "R_tzz",
        "R_z",
        "R_zz",
        "Z_tzz",
        "omega_t",
        "omega_tz",
        "omega_tzz",
        "omega_z",
        "omega_zz",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    aliases=["e_zeta_tz"],
)
def _e_sub_theta_zz(params, transforms, profiles, data, **kwargs):
    data["e_theta_zz"] = jnp.array(
        [
            -((1 + data["omega_z"]) ** 2) * data["R_t"]
            - 2 * data["R_z"] * (1 + data["omega_z"]) * data["omega_t"]
            - data["R"] * data["omega_zz"] * data["omega_t"]
            - 2 * data["R"] * (1 + data["omega_z"]) * data["omega_tz"]
            + data["R_tzz"],
            data["omega_zz"] * data["R_t"]
            + data["R_zz"] * data["omega_t"]
            + 2 * (1 + data["omega_z"]) * data["R_tz"]
            + 2 * data["R_z"] * data["omega_tz"]
            - data["R"]
            * ((1 + data["omega_z"]) ** 2 * data["omega_t"] - data["omega_tzz"]),
            data["Z_tzz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_zeta",
    label="\\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant Toroidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_z", "Z_z", "omega_z"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _e_sub_zeta(params, transforms, profiles, data, **kwargs):
    data["e_zeta"] = jnp.array(
        [data["R_z"], data["R"] * (1 + data["omega_z"]), data["Z_z"]]
    ).T

    return data


@register_compute_fun(
    name="e_zeta_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, third derivative wrt radial coordinate"
        " once and toroidal twice"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_r",
        "R_z",
        "R_rz",
        "R_zz",
        "R_rzz",
        "R_zzz",
        "R_rzzz",
        "Z_rzzz",
        "omega_r",
        "omega_z",
        "omega_rz",
        "omega_zz",
        "omega_rzz",
        "omega_zzz",
        "omega_rzzz",
    ],
)
def _e_sub_zeta_rzz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rzz"] = jnp.array(
        [
            -3 * data["R_rz"] * (1 + data["omega_z"]) ** 2
            - 6 * data["R_z"] * (1 + data["omega_z"]) * data["omega_rz"]
            - 3 * data["R_r"] * (1 + data["omega_z"]) * data["omega_zz"]
            - 3
            * data["R"]
            * (
                data["omega_rz"] * data["omega_zz"]
                + (1 + data["omega_z"]) * data["omega_rzz"]
            )
            + data["R_rzzz"]
            - data["omega_r"]
            * (
                3 * (1 + data["omega_z"]) * data["R_zz"]
                + 3 * data["R_z"] * data["omega_zz"]
                - data["R"]
                * (
                    1
                    + 3 * data["omega_z"]
                    + 3 * data["omega_z"] ** 2
                    + data["omega_z"] ** 3
                    - data["omega_zzz"]
                )
            ),
            3 * data["omega_rz"] * data["R_zz"]
            + 3 * (1 + data["omega_z"]) * data["R_rzz"]
            + 3 * data["R_rz"] * data["omega_zz"]
            + 3 * data["R_z"] * data["omega_rzz"]
            - data["R_r"]
            * (
                1
                + 3 * data["omega_z"]
                + 3 * data["omega_z"] ** 2
                + data["omega_z"] ** 3
                - data["omega_zzz"]
            )
            - data["R"]
            * (
                3 * data["omega_rz"] * (1 + data["omega_z"] * (1 + data["omega_z"]))
                - data["omega_rzzz"]
            )
            + data["omega_r"]
            * (
                -3 * data["R_z"] * (1 + data["omega_z"]) ** 2
                - 3 * data["R"] * (1 + data["omega_z"]) * data["omega_zz"]
                + data["R_zzz"]
            ),
            data["Z_rzzz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_zeta_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant Toroidal basis vector, derivative wrt toroidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_z", "R_zz", "Z_zz", "omega_z", "omega_zz"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _e_sub_zeta_z(params, transforms, profiles, data, **kwargs):
    data["e_zeta_z"] = jnp.array(
        [
            -data["R"] * (1 + data["omega_z"]) ** 2 + data["R_zz"],
            2 * data["R_z"] * (1 + data["omega_z"]) + data["R"] * data["omega_zz"],
            data["Z_zz"],
        ]
    ).T

    return data


@register_compute_fun(
    name="e_zeta_zz",
    label="\\partial_{\\zeta \\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, second derivative wrt toroidal and toroidal"
        " coordinates"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "R",
        "R_z",
        "R_zz",
        "R_zzz",
        "Z_zzz",
        "omega_z",
        "omega_zz",
        "omega_zzz",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _e_sub_zeta_zz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_zz"] = jnp.array(
        [
            -3 * data["R_z"] * (1 + data["omega_z"]) ** 2
            - 3 * data["R"] * (1 + data["omega_z"]) * data["omega_zz"]
            + data["R_zzz"],
            3 * (1 + data["omega_z"]) * data["R_zz"]
            + 3 * data["R_z"] * data["omega_zz"]
            - data["R"]
            * (
                1
                + 3 * data["omega_z"]
                + 3 * data["omega_z"] ** 2
                + data["omega_z"] ** 3
                - data["omega_zzz"]
            ),
            data["Z_zzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="grad(phi)",
    label="\\nabla \\phi",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Gradient of cylindrical toroidal angle ϕ.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _grad_phi(params, transforms, profiles, data, **kwargs):
    data["grad(phi)"] = jnp.column_stack([data["0"], 1 / data["R"], data["0"]])
    return data


@register_compute_fun(
    name="grad(alpha)",
    label="\\nabla \\alpha",
    units="m^{-1}",
    units_long="Inverse meters",
    description=(
        "Gradient of field line label, which is perpendicular to the field line"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["grad(alpha) (periodic)", "grad(alpha) (secular)"],
)
def _grad_alpha(params, transforms, profiles, data, **kwargs):
    data["grad(alpha)"] = data["grad(alpha) (periodic)"] + data["grad(alpha) (secular)"]
    return data


@register_compute_fun(
    name="grad(alpha) (periodic)",
    label="\\mathrm{periodic}(\\nabla \\alpha)",
    units="m^{-1}",
    units_long="Inverse meters",
    description=(
        "Gradient of field line label, which is perpendicular to the field line, "
        "periodic component"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^theta", "e^zeta", "alpha_r (periodic)", "alpha_t", "alpha_z"],
)
def _periodic_grad_alpha(params, transforms, profiles, data, **kwargs):
    data["grad(alpha) (periodic)"] = (
        data["alpha_r (periodic)"] * data["e^rho"].T
        + data["alpha_t"] * data["e^theta"].T
        + data["alpha_z"] * data["e^zeta"].T
    ).T
    return data


@register_compute_fun(
    name="grad(alpha) (secular)",
    label="\\mathrm{secular}(\\nabla \\alpha)",
    units="m^{-1}",
    units_long="Inverse meters",
    description=(
        "Gradient of field line label, which is perpendicular to the field line, "
        "secular component"
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "alpha_r (secular)"],
)
def _secular_grad_alpha(params, transforms, profiles, data, **kwargs):
    data["grad(alpha) (secular)"] = (
        data["alpha_r (secular)"][..., jnp.newaxis] * data["e^rho"]
    )
    return data


@register_compute_fun(
    name="grad(psi)",
    label="\\nabla\\psi",
    units="Wb / m",
    units_long="Webers per meter",
    description="Toroidal flux gradient (normalized by 2pi)",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "e^rho"],
)
def _gradpsi(params, transforms, profiles, data, **kwargs):
    data["grad(psi)"] = (data["psi_r"] * data["e^rho"].T).T
    return data


@register_compute_fun(
    name="n_rho",
    label="\\hat{\\mathbf{n}}_{\\rho}",
    units="~",
    units_long="None",
    description="Unit vector normal to constant rho surface (direction of e^rho)",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_zeta", "|e_theta x e_zeta|"],
    axis_limit_data=["e_theta_r", "|e_theta x e_zeta|_r"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
    ],
)
def _n_rho(params, transforms, profiles, data, **kwargs):
    # Equal to 𝐞^ρ / ‖𝐞^ρ‖ but works correctly for surfaces as well that don't
    # have contravariant basis defined.
    data["n_rho"] = transforms["grid"].replace_at_axis(
        safediv(cross(data["e_theta"], data["e_zeta"]).T, data["|e_theta x e_zeta|"]).T,
        # At the magnetic axis, this function returns the multivalued map whose
        # image is the set { 𝐞^ρ / ‖𝐞^ρ‖ | ρ=0 }.
        lambda: safediv(
            cross(data["e_theta_r"], data["e_zeta"]).T, data["|e_theta x e_zeta|_r"]
        ).T,
    )
    return data


@register_compute_fun(
    name="n_rho",
    label="\\hat{\\mathbf{n}}_{\\rho}",
    units="~",
    units_long="None",
    description="Unit vector normal to constant rho surface (direction of e^rho)",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_zeta", "|e_theta x e_zeta|"],
    parameterization=[
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _n_rho_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    # Equal to 𝐞^ρ / ‖𝐞^ρ‖ but works correctly for surfaces as well that don't
    # have contravariant basis defined.
    data["n_rho"] = safediv(
        cross(data["e_theta"], data["e_zeta"]).T, data["|e_theta x e_zeta|"]
    ).T

    return data


@register_compute_fun(
    name="n_rho_z",
    label="\\partial_{\\zeta}\\hat{\\mathbf{n}}_{\\rho}",
    units="~",
    units_long="None",
    description="Unit vector normal to constant rho surface (direction of e^rho),"
    " derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_theta",
        "e_theta_z",
        "e_zeta",
        "e_zeta_z",
        "|e_theta x e_zeta|",
        "|e_theta x e_zeta|_z",
        "n_rho",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _n_rho_z(params, transforms, profiles, data, **kwargs):
    data["n_rho_z"] = (
        cross(data["e_theta_z"], data["e_zeta"])
        + cross(data["e_theta"], data["e_zeta_z"])
    ) / data["|e_theta x e_zeta|"][:, None] - data["n_rho"] / (
        data["|e_theta x e_zeta|"][:, None]
    ) * (
        data["|e_theta x e_zeta|_z"][:, None]
    )
    return data


@register_compute_fun(
    name="n_theta",
    label="\\hat{\\mathbf{n}}_{\\theta}",
    units="~",
    units_long="None",
    description="Unit vector normal to constant theta surface (direction of e^theta)",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_zeta", "|e_zeta x e_rho|"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
    ],
)
def _n_theta(params, transforms, profiles, data, **kwargs):
    # Equal to 𝐞^θ / ‖𝐞^θ‖ but works correctly for surfaces as well that don't
    # have contravariant basis defined.
    data["n_theta"] = (
        cross(data["e_zeta"], data["e_rho"]).T / data["|e_zeta x e_rho|"]
    ).T
    return data


@register_compute_fun(
    name="n_zeta",
    label="\\hat{\\mathbf{n}}_{\\zeta}",
    units="~",
    units_long="None",
    description="Unit vector normal to constant zeta surface (direction of e^zeta)",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_theta", "|e_rho x e_theta|"],
    axis_limit_data=["e_theta_r", "|e_rho x e_theta|_r"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _n_zeta(params, transforms, profiles, data, **kwargs):
    # Equal to 𝐞^ζ / ‖𝐞^ζ‖ but works correctly for surfaces as well that don't
    # have contravariant basis defined.
    data["n_zeta"] = transforms["grid"].replace_at_axis(
        safediv(cross(data["e_rho"], data["e_theta"]).T, data["|e_rho x e_theta|"]).T,
        # At the magnetic axis, this function returns the multivalued map whose
        # image is the set { 𝐞^ζ / ‖𝐞^ζ‖ | ρ=0 }.
        lambda: safediv(
            cross(data["e_rho"], data["e_theta_r"]).T, data["|e_rho x e_theta|_r"]
        ).T,
    )
    return data


@register_compute_fun(
    name="e_theta|r,p",
    label="\\mathbf{e}_{\\theta} |_{\\rho, \\phi}",
    units="m",
    units_long="meters",
    description=(
        "Covariant poloidal basis vector in (ρ,θ,ϕ) coordinates. "
        "ϕ increases counterclockwise when viewed from above "
        "(cylindrical R,ϕ plane with Z out of page)."
    ),
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "e_phi|r,t", "phi_t"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _e_sub_theta_rp(params, transforms, profiles, data, **kwargs):
    data["e_theta|r,p"] = data["e_theta"] - (data["e_phi|r,t"].T * data["phi_t"]).T
    return data


@register_compute_fun(
    name="e_rho|a,z",
    label="\\mathbf{e}_{\\rho} |_{\\alpha, \\zeta}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector in (ρ, α, ζ) Clebsch coordinates.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_alpha", "alpha_r"],
)
def _e_rho_az(params, transforms, profiles, data, **kwargs):
    # constant α and ζ
    data["e_rho|a,z"] = (
        data["e_rho"] - data["e_alpha"] * data["alpha_r"][:, jnp.newaxis]
    )
    return data


@register_compute_fun(
    name="e_alpha",
    label="\\mathbf{e}_{\\alpha}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector in (ρ, α, ζ) Clebsch coordinates.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "alpha_t"],
)
def _e_alpha(params, transforms, profiles, data, **kwargs):
    # constant ρ and ζ
    data["e_alpha"] = data["e_theta"] / data["alpha_t"][:, jnp.newaxis]
    return data


@register_compute_fun(
    name="e_alpha_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\alpha}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector in (ρ, α, ζ) Clebsch coordinates,"
    " derivative wrt DESC poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "alpha_t", "e_theta_t", "alpha_tt"],
)
def _e_alpha_t(params, transforms, profiles, data, **kwargs):
    data["e_alpha_t"] = (
        data["e_theta_t"] / data["alpha_t"][:, jnp.newaxis]
        - data["e_theta"] * (data["alpha_tt"] / data["alpha_t"] ** 2)[:, jnp.newaxis]
    )
    return data


@register_compute_fun(
    name="e_alpha_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\alpha}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector in (ρ, α, ζ) Clebsch coordinates, "
    "derivative wrt DESC toroidal angle at fixed ρ,θ.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "alpha_t", "e_theta_z", "alpha_tz"],
)
def _e_alpha_z(params, transforms, profiles, data, **kwargs):
    data["e_alpha_z"] = (
        data["e_theta_z"] / data["alpha_t"][:, jnp.newaxis]
        - data["e_theta"] * (data["alpha_tz"] / data["alpha_t"] ** 2)[:, jnp.newaxis]
    )
    return data


@register_compute_fun(
    name="e_zeta|r,a",  # Same as B/(B⋅∇ζ).
    label="\\mathbf{e}_{\\zeta} |_{\\rho, \\alpha} "
    "= \\frac{\\mathbf{B}}{\\mathbf{B} \\cdot \\nabla \\zeta}",
    units="m",
    units_long="meters",
    description="Tangent vector along (collinear to) field line",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_zeta", "e_alpha", "alpha_z"],
)
def _e_zeta_ra(params, transforms, profiles, data, **kwargs):
    data["e_zeta|r,a"] = (
        data["e_zeta"] - data["e_alpha"] * data["alpha_z"][:, jnp.newaxis]
    )
    return data


@register_compute_fun(
    name="(e_zeta|r,a)_t",
    label="\\partial_{\\theta} (\\mathbf{e}_{\\zeta} |_{\\rho, \\alpha})",
    units="m",
    units_long="meters",
    description="Tangent vector along (collinear to) field line, "
    "derivative wrt DESC poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_zeta_t", "e_alpha", "alpha_z", "e_alpha_t", "alpha_tz"],
)
def _e_zeta_ra_t(params, transforms, profiles, data, **kwargs):
    data["(e_zeta|r,a)_t"] = data["e_zeta_t"] - (
        data["e_alpha_t"] * data["alpha_z"][:, jnp.newaxis]
        + data["e_alpha"] * data["alpha_tz"][:, jnp.newaxis]
    )
    return data


@register_compute_fun(
    name="(e_zeta|r,a)_a",
    label="\\partial_{\\alpha} (\\mathbf{e}_{\\zeta} |_{\\rho, \\alpha})",
    units="m",
    units_long="meters",
    description="Tangent vector along (collinear to) field line, derivative "
    "wrt field line poloidal label",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["(e_zeta|r,a)_t", "alpha_t"],
)
def _e_zeta_ra_a(params, transforms, profiles, data, **kwargs):
    data["(e_zeta|r,a)_a"] = data["(e_zeta|r,a)_t"] / data["alpha_t"][:, jnp.newaxis]
    return data


@register_compute_fun(
    name="(e_zeta|r,a)_z",
    label="\\partial_{\\zeta} (\\mathbf{e}_{\\zeta} |_{\\rho, \\alpha})",
    units="m",
    units_long="meters",
    description="Tangent vector along (collinear to) field line, "
    "derivative wrt DESC toroidal angle at fixed ρ,θ.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_zeta_z", "e_alpha", "alpha_z", "e_alpha_z", "alpha_zz"],
)
def _e_zeta_ra_z(params, transforms, profiles, data, **kwargs):
    data["(e_zeta|r,a)_z"] = data["e_zeta_z"] - (
        data["e_alpha_z"] * data["alpha_z"][:, jnp.newaxis]
        + data["e_alpha"] * data["alpha_zz"][:, jnp.newaxis]
    )
    return data


@register_compute_fun(
    name="(e_zeta|r,a)_z|r,a",
    label="\\partial_{\\zeta} (\\mathbf{e}_{\\zeta} |_{\\rho, \\alpha}) "
    "|_{\\rho, \\alpha}",
    units="m",
    units_long="meters",
    description="Curvature vector along field line",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["(e_zeta|r,a)_z", "(e_zeta|r,a)_a", "alpha_z"],
)
def _e_zeta_z_ra(params, transforms, profiles, data, **kwargs):
    data["(e_zeta|r,a)_z|r,a"] = (
        data["(e_zeta|r,a)_z"]
        - data["(e_zeta|r,a)_a"] * data["alpha_z"][:, jnp.newaxis]
    )
    return data


@register_compute_fun(
    name="|e_zeta|r,a|",  # Often written as |dℓ/dζ| = |B/(B⋅∇ζ)|.
    label="|\\mathbf{e}_{\\zeta} |_{\\rho, \\alpha}|"
    " = \\frac{|\\mathbf{B}|}{|\\mathbf{B} \\cdot \\nabla \\zeta|}",
    units="m",
    units_long="meters",
    description="Differential length along field line",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_zeta|r,a"],
)
def _d_ell_d_zeta(params, transforms, profiles, data, **kwargs):
    data["|e_zeta|r,a|"] = jnp.linalg.norm(data["e_zeta|r,a"], axis=-1)
    return data


@register_compute_fun(
    name="|e_zeta|r,a|_z|r,a",
    label="\\partial_{\\zeta} |\\mathbf{e}_{\\zeta} |_{\\rho, \\alpha}| "
    "|_{\\rho, \\alpha}",
    units="m",
    units_long="meters",
    description="Differential length along field line, derivative along field line",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|e_zeta|r,a|", "(e_zeta|r,a)_z|r,a", "e_zeta|r,a"],
)
def _d_ell_d_zeta_z(params, transforms, profiles, data, **kwargs):
    data["|e_zeta|r,a|_z|r,a"] = (
        dot(data["(e_zeta|r,a)_z|r,a"], data["e_zeta|r,a"]) / data["|e_zeta|r,a|"]
    )
    return data


@register_compute_fun(
    name="e_alpha|r,p",
    label="\\mathbf{e}_{\\alpha} |_{\\rho, \\phi}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector in (ρ, α, ϕ) Clebsch coordinates.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "alpha_t", "e_zeta", "alpha_z", "phi_t", "phi_z"],
)
def _e_alpha_rp(params, transforms, profiles, data, **kwargs):
    data["e_alpha|r,p"] = (
        (data["e_theta"].T * data["phi_z"] - data["e_zeta"].T * data["phi_t"])
        / (data["alpha_t"] * data["phi_z"] - data["alpha_z"] * data["phi_t"])
    ).T
    return data


@register_compute_fun(
    name="|e_alpha|r,p|",
    label="|\\mathbf{e}_{\\alpha} |_{\\rho, \\phi}|",
    units="m",
    units_long="meters",
    description="Norm of covariant poloidal basis vector in (ρ, α, ϕ) Clebsch "
    "coordinates.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_alpha|r,p"],
)
def _e_alpha_rp_norm(params, transforms, profiles, data, **kwargs):
    data["|e_alpha|r,p|"] = jnp.linalg.norm(data["e_alpha|r,p"], axis=-1)
    return data


##################################################################################
##########---------------HIGHER-ORDER DERIVATIVES (PEST)---------------###########
##################################################################################


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(e_theta_PEST_v)|PEST",
    label="(\\partial_{\\vartheta}|_{\\rho, \\phi}"
    "(\\mathbf{e}_{\\vartheta})|_{\\rho \\phi})",
    units="m",
    units_long="meters",
    description="Derivative of the covariant poloidal basis vector in"
    "straight field line PEST coordinates (ρ,ϑ,ϕ) w.r.t straight field"
    "line PEST theta coordinate. ϕ increases counterclockwise when viewed above"
    "(cylindrical R,ϕ plane with Z out of page).",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta_t", "e_theta_PEST", "theta_PEST_t", "theta_PEST_tt"],
    aliases=["(e_vartheta_v)|PEST"],
)
def _e_sub_vartheta_rp_vartheta_rp(params, transforms, profiles, data, **kwargs):
    # constant ρ and ϕ
    data["(e_theta_PEST_v)|PEST"] = (
        data["e_theta_t"] - data["e_theta_PEST"] * data["theta_PEST_tt"][:, jnp.newaxis]
    ) / (data["theta_PEST_t"] ** 2)[:, jnp.newaxis]
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(e_theta_PEST_p)|PEST",
    label="(\\partial_{\\phi} |_{\\rho, \\vartheta}"
    " (\\mathbf{e}_{\\vartheta}|_{\\rho, \\phi}))",
    units="m",
    units_long="meters",
    description="Derivative of the covariant poloidal basis vector in"
    "straight field line PEST coordinates (ρ,ϑ,ϕ) w.r.t the cylindrical"
    "toroidal angle. ϕ increases counterclockwise when viewed from above"
    "(cylindrical R,ϕ plane with Z out of page).",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_theta_z",
        "e_theta_PEST",
        "(e_theta_PEST_v)|PEST",
        "theta_PEST_t",
        "theta_PEST_z",
        "theta_PEST_tz",
    ],
    aliases=["(e_vartheta_p)|PEST", "(e_phi_v)|PEST"],
)
def _e_sub_vartheta_rz_phi_rvartheta(params, transforms, profiles, data, **kwargs):
    data["(e_theta_PEST_p)|PEST"] = (
        data["e_theta_z"]
        - data["e_theta_PEST"] * data["theta_PEST_tz"][:, jnp.newaxis]
        - data["(e_theta_PEST_v)|PEST"]
        * data["theta_PEST_t"][:, jnp.newaxis]
        * data["theta_PEST_z"][:, jnp.newaxis]
    ) / data["theta_PEST_t"][:, jnp.newaxis]
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(e_phi_p)|PEST",
    label="(\\partial_{\\phi} |_{\\rho, \\vartheta}"
    " \\mathbf{e}_{\\phi}) |_{\\rho, \\vartheta}",
    units="m",
    units_long="meters",
    description="Derivative of the covariant toroidal basis vector in"
    "straight field line PEST coordinates (ρ,ϑ,ϕ) w.r.t the cylindrical"
    "toroidal angle.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_zeta_z",  # TODO: 568
        "e_zeta_t",
        "e_theta_PEST",
        "(e_theta_PEST_p)|PEST",
        "theta_PEST_t",
        "theta_PEST_z",
        "theta_PEST_tz",
        "theta_PEST_zz",
    ],
)
def _e_sub_phi_rvartheta_phi_rvartheta(params, transforms, profiles, data, **kwargs):
    # ∂()/∂ϕ|r,ϑ = ∂()/∂ϕ|r,θ − ∂()/∂θ * ϑ_ζ/ϑ_θ ____________________________ (1)
    # ∂()/∂ϑ|r,ϕ = ∂()/∂θ 1/ϑ_θ _____________________________________________ (2)

    # Using (1) and (2), we can write
    # e_ϕ|r,ϑ = e_ϕ|r,θ − e_ϑ|r,ϕ ϑ_ζ _______________________________________ (3)

    # Applying just ∂()/∂ϕ|r,ϑ to both sides of (3),
    # ∂(e_ϕ|r,ϑ)/∂ϕ|r,ϑ = ∂(e_ϕ|r,θ)/∂ϕ|r,ϑ - ∂(e_ϑ|r,ϕ * ϑ_ζ)/∂ϕ|r,ϑ _______ (4)

    # Expanding the first term on the right side of (4) using (1), we get
    # ∂(e_ϕ|r,θ)/∂ϕ|r,ϑ = ∂(e_ϕ|r,θ)/∂ϕ|r,θ − ∂(e_ϕ|r,θ)/∂θ * ϑ_ζ/ϑ_θ

    # and expanding the second term on the right side of (4) without using (1)
    # ∂(e_ϑ|r,ϕ *ϑ_ζ)/∂ϕ|r,ϑ = (∂(e_ϑ|r,ϕ)/∂ϕ|r,ϑ)*ϑ_ζ + (e_ϑ|r,ϕ)*∂(ϑ_ζ)/∂ϕ|r,ϑ
    factor = data["theta_PEST_z"] / data["theta_PEST_t"]
    data["(e_phi_p)|PEST"] = (
        data["e_zeta_z"]
        - data["(e_theta_PEST_p)|PEST"] * data["theta_PEST_z"][:, jnp.newaxis]
        - data["e_theta_PEST"]
        * (data["theta_PEST_zz"] - data["theta_PEST_tz"] * factor)[:, jnp.newaxis]
        - data["e_zeta_t"] * factor[:, jnp.newaxis]
    )

    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(e_theta_PEST_r)|PEST",
    label="(\\partial_{\\rho} |_{\\phi, \\vartheta}"
    " \\mathbf{e}_{\\vartheta}) |_{\\rho, \\phi}",
    units="m",
    units_long="meters",
    description="Derivative of the covariant poloidal PEST basis vector in"
    "straight field line PEST coordinates (ρ,ϑ,ϕ) w.r.t rho.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_theta_r",  # in DESC coordinates
        "e_theta_PEST",
        "(e_theta_PEST_v)|PEST",
        "theta_PEST_t",
        "theta_PEST_r",
        "theta_PEST_rt",
    ],
    aliases=["(e_vartheta_r)|PEST", "(e_rho_v)|PEST"],
)
def _e_sub_vartheta_rz_rho_varthetaz(params, transforms, profiles, data, **kwargs):
    data["(e_theta_PEST_r)|PEST"] = (
        data["e_theta_r"]
        - data["e_theta_PEST"] * (data["theta_PEST_rt"])[:, jnp.newaxis]
        - data["(e_theta_PEST_v)|PEST"]
        * (data["theta_PEST_r"] * data["theta_PEST_t"])[:, jnp.newaxis]
    ) / data["theta_PEST_t"][:, jnp.newaxis]
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(e_phi_r)|PEST",
    label="\\partial_{\\rho} |_{\\phi, \\vartheta}"
    " (\\mathbf{e}_{\\phi} |_{\\rho, \\vartheta})",
    units="m",
    units_long="meters",
    description="Derivative of the covariant toroidal basis vector in"
    "straight field line PEST coordinates (ρ,ϑ,ϕ) w.r.t rho."
    "ϕ increases counterclockwise when viewed from above"
    "(cylindrical R,ϕ plane with Z out of page).",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_zeta_r",
        "e_theta",
        "e_theta_r",
        "(e_phi_v)|PEST",
        "theta_PEST_r",
        "theta_PEST_z",
        "theta_PEST_rz",
        "theta_PEST_t",
        "theta_PEST_rt",
    ],
    aliases=["(e_rho_p)|PEST"],
)
def _e_sub_phi_rvartheta_rho_varthetaz(params, transforms, profiles, data, **kwargs):
    # ∂/∂ρ|ϑ,ϕ = ∂/∂ρ|θ,ϕ − ∂/∂ϑ|ρ,ϕ ϑ_ρ ___________________________________ (1)
    # e_ϕ|ρ,ϑ = e_ϕ|ρ,θ − e_ϑ|r,ϕ ϑ_ζ ______________________________________ (2)
    # ∂(e_ϕ|ρ,ϑ)/∂ρ|ϑ,ϕ = ∂(e_ϕ|ρ,ϑ)/∂ρ|θ,ϕ - ∂(e_ϕ|ρ,ϑ)/∂ϑ|ρ,ϕ * ϑ_ρ ______ (3)

    # Expanding the two terms in (3), we get the relation below
    # The first term in (3) becomes
    # ∂(e_ϕ|ρ,ϑ)/∂ρ|θ,ϕ = ∂(e_ϕ|ρ,θ)/∂ρ|θ,ϕ − ∂(e_ϑ|r,ϕ ϑ_ζ)/∂ρ|θ,ϕ ________ (4)

    # The second term in (4) can be expanded to
    # ∂(e_ϑ|r,ϕ ϑ_ζ)/∂ρ|θ,ϕ = ∂(e_θ|ρ,ϕ)/∂ρ|θ,ϕ * (ϑ_ζ/ϑ_θ)
    #                       + e_θ|ρ,ϕ * ∂(ϑ_ζ/ϑ_θ)/∂ρ|θ,ϕ

    # The second term in (3) is implemented as it is.
    data["(e_phi_r)|PEST"] = (
        data["e_zeta_r"]
        - data["e_theta"]
        * (
            (
                data["theta_PEST_rz"] * data["theta_PEST_t"]
                - data["theta_PEST_z"] * data["theta_PEST_rt"]
            )
            / data["theta_PEST_t"] ** 2
        )[:, jnp.newaxis]
        - data["e_theta_r"]
        * (data["theta_PEST_z"] / data["theta_PEST_t"])[:, jnp.newaxis]
        - data["(e_phi_v)|PEST"] * (data["theta_PEST_r"])[:, jnp.newaxis]
    )
    return data


# TODO: Generalize for a general zeta before #568
@register_compute_fun(
    name="(e_rho_r)|PEST",
    label="\\partial_{\\rho} |_{\\phi, \\vartheta}"
    " (\\mathbf{e}_{\\rho} |_{\\phi, \\vartheta})",
    units="m",
    units_long="meters",
    description="Derivative of the covariant radial basis vector in"
    "straight field line PEST coordinates (ρ,ϑ,ϕ) w.r.t rho."
    "ϕ increases counterclockwise when viewed from above"
    "(cylindrical R,ϕ plane with Z out of page).",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e_rho_r",
        "e_rho_t",
        "(e_rho_v)|PEST",
        "e_theta_PEST",
        "theta_PEST_r",
        "theta_PEST_t",
        "theta_PEST_rr",
        "theta_PEST_rt",
    ],
)
def _e_sub_rho_varthetaz_rho_varthetaz(params, transforms, profiles, data, **kwargs):
    # ∂/∂ρ|ϑ,ϕ = ∂/∂ρ|θ,ϕ − ∂/∂ϑ|ρ,ϕ ϑ_ρ − ∂/∂ϕ|ρ,ϑ ϕ_ρ|θ,ϕ

    # Without generalizing the toroidal angle ϕ_ρ|θ,ϕ = 0, so
    # ∂/∂ρ|ϑ,ϕ = ∂/∂ρ|θ,ϕ − ∂/∂ϑ|ρ,ϕ ϑ_ρ ___________________________________ (1)
    # e_ρ|ϑ,ϕ = e_ρ|θ,ϕ − e_ϑ|ρ,ϕ ϑ_ρ ______________________________________ (2)

    # ∂(e_ρ|ϑ,ϕ)/∂ρ|ϑ,ϕ = ∂(e_ρ|θ,ϕ)/∂ρ|ϑ,ϕ - ∂(e_ϑ|ρ,ϕ * ϑ_ρ)/∂ρ|ϑ,ϕ ______ (3)

    # Expand first term on the right side of (3) using (1)
    # ∂(e_ρ|θ,ϕ)/∂ρ|ϑ,ϕ = ∂(e_ρ|θ,ϕ)/∂ρ|θ,ϕ - ∂(e_ρ|θ,ϕ)/∂θ|ρ,ϕ * (ϑ_ρ/ϑ_ϑ)

    # Now, use (1) again to expand the second term on the right side of (3)
    # ∂(e_ϑ|ρ,ϕ * ϑ_ρ)/∂ρ|ϑ,ϕ = ∂(e_ϑ|ρ,ϕ)/∂ρ|ϑ,ϕ * ϑ_ρ - e_ϑ|ρ,ϕ *(ϑ_ρρ+ϑ_ρθ (ϑ_ρ/ϑ_θ))
    factor = data["theta_PEST_r"] / data["theta_PEST_t"]
    data["(e_rho_r)|PEST"] = (
        data["e_rho_r"]
        - data["e_rho_t"] * factor[:, jnp.newaxis]
        - data["(e_rho_v)|PEST"] * data["theta_PEST_r"][:, jnp.newaxis]
        - data["e_theta_PEST"]
        * (data["theta_PEST_rr"] - data["theta_PEST_rt"] * factor)[:, jnp.newaxis]
    )
    return data
