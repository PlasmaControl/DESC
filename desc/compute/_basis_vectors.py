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

from .data_index import register_compute_fun
from .geom_utils import rpz2xyz_vec
from .utils import cross, safediv


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
    data=["B", "|B|", "phi"],
    basis="basis",
)
def _b(params, transforms, profiles, data, **kwargs):
    data["b"] = (data["B"].T / data["|B|"]).T
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["b"] = rpz2xyz_vec(data["b"], phi=data["phi"])
    return data


@register_compute_fun(
    name="e^rho",
    label="\\mathbf{e}^{\\rho}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant radial basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta/sqrt(g)", "e_zeta", "phi"],
    basis="basis",
)
def _e_sup_rho(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { 𝐞^ρ | ρ=0 }.
    data["e^rho"] = cross(data["e_theta/sqrt(g)"], data["e_zeta"])
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^rho"] = rpz2xyz_vec(data["e^rho"], phi=data["phi"])
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
    data=["e_theta", "e_zeta", "e_theta_r", "e_zeta_r", "sqrt(g)", "sqrt(g)_r", "phi"],
    axis_limit_data=["e_theta_rr", "sqrt(g)_rr"],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^rho_r"] = rpz2xyz_vec(data["e^rho_r"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^rho_rr"] = rpz2xyz_vec(data["e^rho_rr"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^rho_rt"] = rpz2xyz_vec(data["e^rho_rt"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^rho_rz"] = rpz2xyz_vec(data["e^rho_rz"], phi=data["phi"])
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
    data=["e_theta", "e_zeta", "e_theta_t", "e_zeta_t", "sqrt(g)", "sqrt(g)_t", "phi"],
    axis_limit_data=["e_theta_r", "e_theta_rt", "sqrt(g)_r", "sqrt(g)_rt"],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^rho_t"] = rpz2xyz_vec(data["e^rho_t"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^rho_tt"] = rpz2xyz_vec(data["e^rho_tt"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^rho_tz"] = rpz2xyz_vec(data["e^rho_tz"], phi=data["phi"])
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
    data=["e_theta", "e_zeta", "e_theta_z", "e_zeta_z", "sqrt(g)", "sqrt(g)_z", "phi"],
    axis_limit_data=["e_theta_r", "e_theta_rz", "sqrt(g)_r", "sqrt(g)_rz"],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^rho_z"] = rpz2xyz_vec(data["e^rho_z"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^rho_zz"] = rpz2xyz_vec(data["e^rho_zz"], phi=data["phi"])
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
    data=["e^theta*sqrt(g)", "sqrt(g)", "phi"],
    basis="basis",
)
def _e_sup_theta(params, transforms, profiles, data, **kwargs):
    data["e^theta"] = (data["e^theta*sqrt(g)"].T / data["sqrt(g)"]).T
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^theta"] = rpz2xyz_vec(data["e^theta"], phi=data["phi"])
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
    data=["e_rho", "e_zeta", "phi"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
    basis="basis",
)
def _e_sup_theta_times_sqrt_g(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { 𝐞^θ √g | ρ=0 }.
    data["e^theta*sqrt(g)"] = cross(data["e_zeta"], data["e_rho"])
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^theta*sqrt(g)"] = rpz2xyz_vec(data["e^theta*sqrt(g)"], phi=data["phi"])
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
    data=["e_zeta", "e_rho", "e_zeta_r", "e_rho_r", "sqrt(g)", "sqrt(g)_r", "phi"],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^theta_r"] = rpz2xyz_vec(data["e^theta_r"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^theta_rr"] = rpz2xyz_vec(data["e^theta_rr"], phi=data["phi"])

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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^theta_rt"] = rpz2xyz_vec(data["e^theta_rt"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^theta_rz"] = rpz2xyz_vec(data["e^theta_rz"], phi=data["phi"])
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
    data=["e_zeta", "e_rho", "e_zeta_t", "e_rho_t", "sqrt(g)", "sqrt(g)_t", "phi"],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^theta_t"] = rpz2xyz_vec(data["e^theta_t"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^theta_tt"] = rpz2xyz_vec(data["e^theta_tt"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^theta_tz"] = rpz2xyz_vec(data["e^theta_tz"], phi=data["phi"])
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
    data=["e_zeta", "e_rho", "e_zeta_z", "e_rho_z", "sqrt(g)", "sqrt(g)_z", "phi"],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^theta_z"] = rpz2xyz_vec(data["e^theta_z"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^theta_zz"] = rpz2xyz_vec(data["e^theta_zz"], phi=data["phi"])
    return data


@register_compute_fun(
    name="e^zeta",
    label="\\mathbf{e}^{\\zeta}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Contravariant toroidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_theta/sqrt(g)", "phi"],
    basis="basis",
)
def _e_sup_zeta(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { 𝐞^ζ | ρ=0 }.
    data["e^zeta"] = cross(data["e_rho"], data["e_theta/sqrt(g)"])
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^zeta"] = rpz2xyz_vec(data["e^zeta"], phi=data["phi"])
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
    data=["e_rho", "e_rho_r", "e_theta", "e_theta_r", "sqrt(g)", "sqrt(g)_r", "phi"],
    axis_limit_data=["e_theta_rr", "sqrt(g)_rr"],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^zeta_r"] = rpz2xyz_vec(data["e^zeta_r"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^zeta_rr"] = rpz2xyz_vec(data["e^zeta_rr"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^zeta_rt"] = rpz2xyz_vec(data["e^zeta_rt"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^zeta_rz"] = rpz2xyz_vec(data["e^zeta_rz"], phi=data["phi"])
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
    data=["e_rho", "e_rho_t", "e_theta", "e_theta_t", "sqrt(g)", "sqrt(g)_t", "phi"],
    axis_limit_data=["e_theta_r", "e_theta_rt", "sqrt(g)_r", "sqrt(g)_rt"],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^zeta_t"] = rpz2xyz_vec(data["e^zeta_t"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^zeta_tt"] = rpz2xyz_vec(data["e^zeta_tt"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^zeta_tz"] = rpz2xyz_vec(data["e^zeta_tz"], phi=data["phi"])
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
    data=["e_rho", "e_rho_z", "e_theta", "e_theta_z", "sqrt(g)", "sqrt(g)_z", "phi"],
    axis_limit_data=["e_theta_r", "e_theta_rz", "sqrt(g)_r", "sqrt(g)_rz"],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^zeta_z"] = rpz2xyz_vec(data["e^zeta_z"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e^zeta_zz"] = rpz2xyz_vec(data["e^zeta_zz"], phi=data["phi"])
    return data


@register_compute_fun(
    name="e_phi",
    label="\\mathbf{e}_{\\phi}",
    units="m",
    units_long="meters",
    description="Covariant cylindrical toroidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_zeta", "phi_z", "phi"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
        "desc.geometry.core.Surface",
    ],
    basis="basis",
)
def _e_sub_phi(params, transforms, profiles, data, **kwargs):
    # dX/dphi at const r,t = dX/dz * dz/dphi = dX/dz / (dphi/dz)
    data["e_phi"] = (data["e_zeta"].T / data["phi_z"]).T
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_phi"] = rpz2xyz_vec(data["e_phi"], phi=data["phi"])
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
    data=["R", "R_r", "Z_r", "omega_r", "phi"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    basis="basis",
)
def _e_sub_rho(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { 𝐞ᵨ | ρ=0 }.
    data["e_rho"] = jnp.array([data["R_r"], data["R"] * data["omega_r"], data["Z_r"]]).T
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho"] = rpz2xyz_vec(data["e_rho"], phi=data["phi"])
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
    data=["R", "R_r", "R_rr", "Z_rr", "omega_r", "omega_rr", "phi"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_r"] = rpz2xyz_vec(data["e_rho_r"], phi=data["phi"])
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
        "phi",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_rr"] = rpz2xyz_vec(data["e_rho_rr"], phi=data["phi"])
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
        "phi",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_rrr"] = rpz2xyz_vec(data["e_rho_rrr"], phi=data["phi"])
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
        "phi",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_rrt"] = rpz2xyz_vec(data["e_rho_rrt"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_rrz"] = rpz2xyz_vec(data["e_rho_rrz"], phi=data["phi"])
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
        "phi",
    ],
    aliases=["x_rrt", "x_rtr", "x_trr", "e_theta_rr"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_rt"] = rpz2xyz_vec(data["e_rho_rt"], phi=data["phi"])
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
        "phi",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_rtt"] = rpz2xyz_vec(data["e_rho_rtt"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_rtz"] = rpz2xyz_vec(data["e_rho_rtz"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_rz"] = rpz2xyz_vec(data["e_rho_rz"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_rzz"] = rpz2xyz_vec(data["e_rho_rzz"], phi=data["phi"])
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
    data=["R", "R_r", "R_rt", "R_t", "Z_rt", "omega_r", "omega_rt", "omega_t", "phi"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_t"] = rpz2xyz_vec(data["e_rho_t"], phi=data["phi"])
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
        "phi",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_tt"] = rpz2xyz_vec(data["e_rho_tt"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_tz"] = rpz2xyz_vec(data["e_rho_tz"], phi=data["phi"])
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
    data=["R", "R_r", "R_rz", "R_z", "Z_rz", "omega_r", "omega_rz", "omega_z", "phi"],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_z"] = rpz2xyz_vec(data["e_rho_z"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_rho_zz"] = rpz2xyz_vec(data["e_rho_zz"], phi=data["phi"])
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
    data=["R", "R_t", "Z_t", "omega_t", "phi"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
    basis="basis",
)
def _e_sub_theta(params, transforms, profiles, data, **kwargs):
    data["e_theta"] = jnp.array(
        [data["R_t"], data["R"] * data["omega_t"], data["Z_t"]]
    ).T
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_theta"] = rpz2xyz_vec(data["e_theta"], phi=data["phi"])
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
    data=["e_theta", "sqrt(g)", "phi"],
    axis_limit_data=["e_theta_r", "sqrt(g)_r"],
    basis="basis",
)
def _e_sub_theta_over_sqrt_g(params, transforms, profiles, data, **kwargs):
    # At the magnetic axis, this function returns the multivalued map whose
    # image is the set { 𝐞_θ / √g | ρ=0 }.
    data["e_theta/sqrt(g)"] = transforms["grid"].replace_at_axis(
        safediv(data["e_theta"].T, data["sqrt(g)"]).T,
        lambda: safediv(data["e_theta_r"].T, data["sqrt(g)_r"]).T,
    )
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_theta/sqrt(g)"] = rpz2xyz_vec(data["e_theta/sqrt(g)"], phi=data["phi"])
    return data


@register_compute_fun(
    name="e_theta_PEST",
    label="\\mathbf{e}_{\\theta_{PEST}}",
    units="m",
    units_long="meters",
    description="Covariant straight field line (PEST) poloidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_theta", "theta_PEST_t", "phi"],
    basis="basis",
)
def _e_sub_theta_pest(params, transforms, profiles, data, **kwargs):
    # dX/dv at const r,z = dX/dt * dt/dv / dX/dt / dv/dt
    data["e_theta_PEST"] = (data["e_theta"].T / data["theta_PEST_t"]).T
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_theta_PEST"] = rpz2xyz_vec(data["e_theta_PEST"], phi=data["phi"])
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
        "phi",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_theta_rtt"] = rpz2xyz_vec(data["e_theta_rtt"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_theta_rtz"] = rpz2xyz_vec(data["e_theta_rtz"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_theta_rzz"] = rpz2xyz_vec(data["e_theta_rzz"], phi=data["phi"])
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
    data=["R", "R_t", "R_tt", "Z_tt", "omega_t", "omega_tt", "phi"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
    basis="basis",
)
def _e_sub_theta_t(params, transforms, profiles, data, **kwargs):
    data["e_theta_t"] = jnp.array(
        [
            -data["R"] * data["omega_t"] ** 2 + data["R_tt"],
            2 * data["R_t"] * data["omega_t"] + data["R"] * data["omega_tt"],
            data["Z_tt"],
        ]
    ).T
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_theta_t"] = rpz2xyz_vec(data["e_theta_t"], phi=data["phi"])
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
        "phi",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_theta_tt"] = rpz2xyz_vec(data["e_theta_tt"], phi=data["phi"])
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
        "phi",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_theta_tz"] = rpz2xyz_vec(data["e_theta_tz"], phi=data["phi"])
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
    data=["R", "R_t", "R_tz", "R_z", "Z_tz", "omega_t", "omega_tz", "omega_z", "phi"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_theta_z"] = rpz2xyz_vec(data["e_theta_z"], phi=data["phi"])
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
        "phi",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_theta_zz"] = rpz2xyz_vec(data["e_theta_zz"], phi=data["phi"])
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
    data=["R", "R_z", "Z_z", "omega_z", "phi"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    basis="basis",
)
def _e_sub_zeta(params, transforms, profiles, data, **kwargs):
    data["e_zeta"] = jnp.array(
        [data["R_z"], data["R"] * (1 + data["omega_z"]), data["Z_z"]]
    ).T
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_zeta"] = rpz2xyz_vec(data["e_zeta"], phi=data["phi"])
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
        "phi",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_zeta_rzz"] = rpz2xyz_vec(data["e_zeta_rzz"], phi=data["phi"])
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
    data=["R", "R_z", "R_zz", "Z_zz", "omega_z", "omega_zz", "phi"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    basis="basis",
)
def _e_sub_zeta_z(params, transforms, profiles, data, **kwargs):
    data["e_zeta_z"] = jnp.array(
        [
            -data["R"] * (1 + data["omega_z"]) ** 2 + data["R_zz"],
            2 * data["R_z"] * (1 + data["omega_z"]) + data["R"] * data["omega_zz"],
            data["Z_zz"],
        ]
    ).T
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_zeta_z"] = rpz2xyz_vec(data["e_zeta_z"], phi=data["phi"])
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
        "phi",
    ],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
    basis="basis",
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
    if kwargs.get("basis", "rpz").lower() == "xyz":
        data["e_zeta_zz"] = rpz2xyz_vec(data["e_zeta_zz"], phi=data["phi"])
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
    data=["e^rho", "e^theta", "e^zeta", "alpha_r", "alpha_t", "alpha_z"],
)
def _grad_alpha(params, transforms, profiles, data, **kwargs):
    data["grad(alpha)"] = (
        data["alpha_r"] * data["e^rho"].T
        + data["alpha_t"] * data["e^theta"].T
        + data["alpha_z"] * data["e^zeta"].T
    ).T
    return data


@register_compute_fun(
    name="grad(alpha)_t",
    label="\\partial_{\\theta} \\nabla \\alpha",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal derivative of the gradient of the field line label",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^rho",
        "e^theta",
        "e^zeta",
        "e^rho_t",
        "e^theta_t",
        "e^zeta_t",
        "alpha_r",
        "alpha_t",
        "alpha_z",
        "alpha_rt",
        "alpha_tt",
        "alpha_tz",
    ],
)
def _grad_alpha_t(params, transforms, profiles, data, **kwargs):
    data["grad(alpha)_t"] = (
        data["alpha_r"] * data["e^rho_t"].T
        + data["alpha_rt"] * data["e^rho"].T
        + data["alpha_t"] * data["e^theta_t"].T
        + data["alpha_tt"] * data["e^theta"].T
        + data["alpha_z"] * data["e^zeta_t"].T
        + data["alpha_tz"] * data["e^zeta"].T
    ).T
    return data


@register_compute_fun(
    name="grad(alpha)_z",
    label="\\partial_{\\zeta} (\\nabla \\alpha)",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal derivative of the gradient of the field line label",
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
        "alpha_r",
        "alpha_t",
        "alpha_z",
        "alpha_rz",
        "alpha_tz",
        "alpha_zz",
    ],
)
def _grad_alpha_z(params, transforms, profiles, data, **kwargs):
    data["grad(alpha)_z"] = (
        data["alpha_r"] * data["e^rho_z"].T
        + data["alpha_rz"] * data["e^rho"].T
        + data["alpha_t"] * data["e^theta_z"].T
        + data["alpha_tz"] * data["e^theta"].T
        + data["alpha_z"] * data["e^zeta_z"].T
        + data["alpha_zz"] * data["e^zeta"].T
    ).T
    return data


@register_compute_fun(
    name="grad(alpha)_tt",
    label="\\partial_{\\theta \\theta} \\nabla \\alpha",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Poloidal derivative of the gradient of the field line label",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^rho",
        "e^theta",
        "e^zeta",
        "e^rho_t",
        "e^theta_t",
        "e^zeta_t",
        "e^rho_tt",
        "e^theta_tt",
        "e^zeta_tt",
        "alpha_r",
        "alpha_t",
        "alpha_z",
        "alpha_rt",
        "alpha_tt",
        "alpha_tz",
        "alpha_rtt",
        "alpha_ttt",
        "alpha_ttz",
    ],
)
def _grad_alpha_tt(params, transforms, profiles, data, **kwargs):
    data["grad(alpha)_tt"] = (
        data["alpha_r"] * data["e^rho_tt"].T
        + data["alpha_rtt"] * data["e^rho"].T
        + 2 * data["alpha_rt"] * data["e^rho_t"].T
        + data["alpha_t"] * data["e^theta_tt"].T
        + data["alpha_ttt"] * data["e^theta"].T
        + 2 * data["alpha_tt"] * data["e^theta_t"].T
        + data["alpha_z"] * data["e^zeta_tt"].T
        + data["alpha_ttz"] * data["e^zeta"].T
        + 2 * data["alpha_tz"] * data["e^zeta_t"].T
    ).T
    return data


@register_compute_fun(
    name="grad(alpha)_zz",
    label="\\partial_{\\zeta \\zeta} (\\nabla \\alpha)",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Toroidal derivative of the gradient of the field line label",
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
        "e^rho_zz",
        "e^theta_zz",
        "e^zeta_zz",
        "alpha_r",
        "alpha_t",
        "alpha_z",
        "alpha_rz",
        "alpha_tz",
        "alpha_zz",
        "alpha_rzz",
        "alpha_tzz",
        "alpha_zzz",
    ],
)
def _grad_alpha_zz(params, transforms, profiles, data, **kwargs):
    data["grad(alpha)_zz"] = (
        data["alpha_r"] * data["e^rho_zz"].T
        + data["alpha_rzz"] * data["e^rho"].T
        + 2 * data["alpha_rz"] * data["e^rho_z"].T
        + data["alpha_t"] * data["e^theta_zz"].T
        + data["alpha_tzz"] * data["e^theta"].T
        + 2 * data["alpha_tz"] * data["e^theta_z"].T
        + data["alpha_z"] * data["e^zeta_zz"].T
        + data["alpha_zzz"] * data["e^zeta"].T
        + 2 * data["alpha_zz"] * data["e^zeta_z"].T
    ).T
    return data


@register_compute_fun(
    name="grad(alpha)_tz",
    label="\\partial_{\\theta \\zeta} (\\nabla \\alpha)",
    units="m^{-1}",
    units_long="Inverse meters",
    description=(
        "Mixed toroidal and poloidal derivative of the gradient"
        " of the field line label"
    ),
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
        "e^theta_t",
        "e^zeta_t",
        "e^rho_tz",
        "e^theta_tz",
        "e^zeta_tz",
        "alpha_r",
        "alpha_t",
        "alpha_z",
        "alpha_rz",
        "alpha_tz",
        "alpha_tt",
        "alpha_rt",
        "alpha_rtz",
        "alpha_ttz",
        "alpha_tzz",
    ],
)
def _grad_alpha_tz(params, transforms, profiles, data, **kwargs):
    data["grad(alpha)_tz"] = (
        data["alpha_r"] * data["e^rho_tz"].T
        + data["alpha_rtz"] * data["e^rho"].T
        + data["alpha_rz"] * data["e^rho_t"].T
        + data["alpha_rt"] * data["e^rho_z"].T
        + data["alpha_t"] * data["e^theta_tz"].T
        + data["alpha_ttz"] * data["e^theta"].T
        + data["alpha_tz"] * data["e^theta_t"].T
        + data["alpha_tt"] * data["e^theta_z"].T
        + data["alpha_z"] * data["e^zeta_tz"].T
        + data["alpha_tzz"] * data["e^zeta"].T
        + data["alpha_tz"] * data["e^zeta_t"].T
        + data["alpha_tt"] * data["e^zeta_z"].T
    ).T
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
        "desc.geometry.core.Surface",
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
        "desc.geometry.core.Surface",
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
        "desc.geometry.core.Surface",
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
        "desc.geometry.core.Surface",
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
    data=["e_theta", "e_phi", "phi_t"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
        "desc.geometry.core.Surface",
    ],
)
def _e_sub_theta_rp(params, transforms, profiles, data, **kwargs):
    data["e_theta|r,p"] = data["e_theta"] - (data["e_phi"].T * data["phi_t"]).T
    return data
