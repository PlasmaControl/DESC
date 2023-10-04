"""Compute functions for magnetic field quantities.

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
from .utils import (
    cross,
    dot,
    surface_averages,
    surface_integrals_map,
    surface_max,
    surface_min,
)


@register_compute_fun(
    name="B0",
    label="\\psi' / \\sqrt{g}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "sqrt(g)"],
    axis_limit_data=["psi_rr", "sqrt(g)_r"],
)
def _B0(params, transforms, profiles, data, **kwargs):
    data["B0"] = transforms["grid"].replace_at_axis(
        data["psi_r"] / data["sqrt(g)"],
        lambda: data["psi_rr"] / data["sqrt(g)_r"],
    )
    return data


@register_compute_fun(
    name="B^rho",
    label="B^{\\rho}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="Contravariant radial component of magnetic field",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
)
def _B_sup_rho(params, transforms, profiles, data, **kwargs):
    data["B^rho"] = data["0"]
    return data


@register_compute_fun(
    name="B^theta",
    label="B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="Contravariant poloidal component of magnetic field",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "iota", "lambda_z", "omega_z"],
)
def _B_sup_theta(params, transforms, profiles, data, **kwargs):
    data["B^theta"] = data["B0"] * (
        data["iota"] * data["omega_z"] + data["iota"] - data["lambda_z"]
    )
    return data


@register_compute_fun(
    name="B^zeta",
    label="B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="Contravariant toroidal component of magnetic field",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "iota", "lambda_t", "omega_t"],
)
def _B_sup_zeta(params, transforms, profiles, data, **kwargs):
    data["B^zeta"] = data["B0"] * (
        -data["iota"] * data["omega_t"] + data["lambda_t"] + 1
    )
    return data


@register_compute_fun(
    name="B",
    label="B",
    units="T",
    units_long="Tesla",
    description="Magnetic field",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B^theta", "e_theta", "B^zeta", "e_zeta"],
)
def _B(params, transforms, profiles, data, **kwargs):
    data["B"] = (
        data["B^theta"] * data["e_theta"].T + data["B^zeta"] * data["e_zeta"].T
    ).T
    return data


@register_compute_fun(
    name="B_R",
    label="B_{R}",
    units="T",
    units_long="Tesla",
    description="Radial component of magnetic field in lab frame",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B"],
)
def _B_R(params, transforms, profiles, data, **kwargs):
    data["B_R"] = data["B"][:, 0]
    return data


@register_compute_fun(
    name="B_phi",
    label="B_{\\phi}",
    units="T",
    units_long="Tesla",
    description="Toroidal component of magnetic field in lab frame",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B"],
)
def _B_phi(params, transforms, profiles, data, **kwargs):
    data["B_phi"] = data["B"][:, 1]
    return data


@register_compute_fun(
    name="B_Z",
    label="B_{Z}",
    units="T",
    units_long="Tesla",
    description="Vertical component of magnetic field in lab frame",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B"],
)
def _B_Z(params, transforms, profiles, data, **kwargs):
    data["B_Z"] = data["B"][:, 2]
    return data


@register_compute_fun(
    name="B0_r",
    label="\\partial_{\\rho} (\\psi' / \\sqrt{g})",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "psi_rr", "sqrt(g)", "sqrt(g)_r"],
    axis_limit_data=["psi_rrr", "sqrt(g)_rr"],
)
def _B0_r(params, transforms, profiles, data, **kwargs):
    data["B0_r"] = transforms["grid"].replace_at_axis(
        (data["psi_rr"] * data["sqrt(g)"] - data["psi_r"] * data["sqrt(g)_r"])
        / data["sqrt(g)"] ** 2,
        lambda: (
            data["psi_rrr"] * data["sqrt(g)_r"] - data["psi_rr"] * data["sqrt(g)_rr"]
        )
        / (2 * data["sqrt(g)_r"] ** 2),
    )
    return data


@register_compute_fun(
    name="B^theta_r",
    label="\\partial_{\\rho} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant poloidal component of magnetic field, derivative wrt radial"
        " coordinate"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_r",
        "iota",
        "iota_r",
        "lambda_rz",
        "lambda_z",
        "omega_rz",
        "omega_z",
    ],
)
def _B_sup_theta_r(params, transforms, profiles, data, **kwargs):
    data["B^theta_r"] = data["B0"] * (
        data["iota"] * data["omega_rz"]
        + data["iota_r"] * data["omega_z"]
        + data["iota_r"]
        - data["lambda_rz"]
    ) + data["B0_r"] * (
        data["iota"] * data["omega_z"] + data["iota"] - data["lambda_z"]
    )
    return data


@register_compute_fun(
    name="B^zeta_r",
    label="\\partial_{\\rho} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant toroidal component of magnetic field, derivative wrt radial"
        " coordinate"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_r",
        "iota",
        "iota_r",
        "lambda_rt",
        "lambda_t",
        "omega_rt",
        "omega_t",
    ],
)
def _B_sup_zeta_r(params, transforms, profiles, data, **kwargs):
    data["B^zeta_r"] = data["B0"] * (
        -data["iota"] * data["omega_rt"]
        - data["iota_r"] * data["omega_t"]
        + data["lambda_rt"]
    ) + data["B0_r"] * (-data["iota"] * data["omega_t"] + data["lambda_t"] + 1)
    return data


@register_compute_fun(
    name="B_r",
    label="\\partial_{\\rho} \\mathbf{B}",
    units="T",
    units_long="Tesla",
    description="Magnetic field, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta_r",
        "B^theta",
        "B^zeta_r",
        "B^zeta",
        "e_theta",
        "e_theta_r",
        "e_zeta",
        "e_zeta_r",
    ],
)
def _B_r(params, transforms, profiles, data, **kwargs):
    data["B_r"] = (
        data["B^theta_r"] * data["e_theta"].T
        + data["B^theta"] * data["e_theta_r"].T
        + data["B^zeta_r"] * data["e_zeta"].T
        + data["B^zeta"] * data["e_zeta_r"].T
    ).T
    return data


@register_compute_fun(
    name="B0_t",
    label="\\partial_{\\theta} (\\psi' / \\sqrt{g})",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "sqrt(g)", "sqrt(g)_t"],
    axis_limit_data=["psi_rr", "sqrt(g)_r", "sqrt(g)_rt"],
)
def _B0_t(params, transforms, profiles, data, **kwargs):
    data["B0_t"] = transforms["grid"].replace_at_axis(
        -data["psi_r"] * data["sqrt(g)_t"] / data["sqrt(g)"] ** 2,
        lambda: -data["psi_rr"] * data["sqrt(g)_rt"] / data["sqrt(g)_r"] ** 2,
    )
    return data


@register_compute_fun(
    name="B^theta_t",
    label="\\partial_{\\theta} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant poloidal component of magnetic field, derivative wrt poloidal"
        " coordinate"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "B0_t", "iota", "lambda_tz", "lambda_z", "omega_tz", "omega_z"],
)
def _B_sup_theta_t(params, transforms, profiles, data, **kwargs):
    data["B^theta_t"] = data["B0"] * (
        data["iota"] * data["omega_tz"] - data["lambda_tz"]
    ) + data["B0_t"] * (
        data["iota"] * data["omega_z"] + data["iota"] - data["lambda_z"]
    )
    return data


@register_compute_fun(
    name="B^zeta_t",
    label="\\partial_{\\theta} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant toroidal component of magnetic field, derivative wrt poloidal"
        " coordinate"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "B0_t", "iota", "lambda_t", "lambda_tt", "omega_t", "omega_tt"],
)
def _B_sup_zeta_t(params, transforms, profiles, data, **kwargs):
    data["B^zeta_t"] = data["B0"] * (
        -data["iota"] * data["omega_tt"] + data["lambda_tt"]
    ) + data["B0_t"] * (-data["iota"] * data["omega_t"] + data["lambda_t"] + 1)
    return data


@register_compute_fun(
    name="B_t",
    label="\\partial_{\\theta} \\mathbf{B}",
    units="T",
    units_long="Tesla",
    description="Magnetic field, derivative wrt poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta_t",
        "B^theta",
        "B^zeta_t",
        "B^zeta",
        "e_theta",
        "e_theta_t",
        "e_zeta",
        "e_zeta_t",
    ],
)
def _B_t(params, transforms, profiles, data, **kwargs):
    data["B_t"] = (
        data["B^theta_t"] * data["e_theta"].T
        + data["B^theta"] * data["e_theta_t"].T
        + data["B^zeta_t"] * data["e_zeta"].T
        + data["B^zeta"] * data["e_zeta_t"].T
    ).T
    return data


@register_compute_fun(
    name="B0_z",
    label="\\partial_{\\zeta} (\\psi' / \\sqrt{g})",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "sqrt(g)", "sqrt(g)_z"],
    axis_limit_data=["psi_rr", "sqrt(g)_r", "sqrt(g)_rz"],
)
def _B0_z(params, transforms, profiles, data, **kwargs):
    data["B0_z"] = transforms["grid"].replace_at_axis(
        -data["psi_r"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 2,
        lambda: -data["psi_rr"] * data["sqrt(g)_rz"] / data["sqrt(g)_r"] ** 2,
    )
    return data


@register_compute_fun(
    name="B^theta_z",
    label="\\partial_{\\zeta} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant poloidal component of magnetic field, derivative wrt toroidal"
        " coordinate"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "B0_z", "iota", "lambda_z", "lambda_zz", "omega_z", "omega_zz"],
)
def _B_sup_theta_z(params, transforms, profiles, data, **kwargs):
    data["B^theta_z"] = data["B0"] * (
        data["iota"] * data["omega_zz"] - data["lambda_zz"]
    ) + data["B0_z"] * (
        data["iota"] * data["omega_z"] + data["iota"] - data["lambda_z"]
    )
    return data


@register_compute_fun(
    name="B^zeta_z",
    label="\\partial_{\\zeta} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant toroidal component of magnetic field, derivative wrt toroidal"
        " coordinate"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "B0_z", "iota", "lambda_t", "lambda_tz", "omega_t", "omega_tz"],
)
def _B_sup_zeta_z(params, transforms, profiles, data, **kwargs):
    data["B^zeta_z"] = data["B0"] * (
        -data["iota"] * data["omega_tz"] + data["lambda_tz"]
    ) + data["B0_z"] * (-data["iota"] * data["omega_t"] + data["lambda_t"] + 1)
    return data


@register_compute_fun(
    name="B_z",
    label="\\partial_{\\zeta} \\mathbf{B}",
    units="T",
    units_long="Tesla",
    description="Magnetic field, derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta_z",
        "B^theta",
        "B^zeta_z",
        "B^zeta",
        "e_theta",
        "e_theta_z",
        "e_zeta",
        "e_zeta_z",
    ],
)
def _B_z(params, transforms, profiles, data, **kwargs):
    data["B_z"] = (
        data["B^theta_z"] * data["e_theta"].T
        + data["B^theta"] * data["e_theta_z"].T
        + data["B^zeta_z"] * data["e_zeta"].T
        + data["B^zeta"] * data["e_zeta_z"].T
    ).T
    return data


@register_compute_fun(
    name="B0_rr",
    label="\\partial_{\\rho \\rho} (\\psi' / \\sqrt{g})",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "psi_rr", "psi_rrr", "sqrt(g)", "sqrt(g)_r", "sqrt(g)_rr"],
    axis_limit_data=["sqrt(g)_rrr"],
)
def _B0_rr(params, transforms, profiles, data, **kwargs):
    data["B0_rr"] = transforms["grid"].replace_at_axis(
        (
            data["psi_rrr"] * data["sqrt(g)"] ** 2
            - 2 * data["psi_rr"] * data["sqrt(g)_r"] * data["sqrt(g)"]
            - data["psi_r"] * data["sqrt(g)_rr"] * data["sqrt(g)"]
            + 2 * data["psi_r"] * data["sqrt(g)_r"] ** 2
        )
        / data["sqrt(g)"] ** 3,
        lambda: (
            3 * data["sqrt(g)_rr"] ** 2 * data["psi_rr"]
            - 2 * data["sqrt(g)_rrr"] * data["sqrt(g)_r"] * data["psi_rr"]
            - 3 * data["psi_rrr"] * data["sqrt(g)_r"] * data["sqrt(g)_rr"]
        )
        / (6 * data["sqrt(g)_r"] ** 3),
    )
    return data


@register_compute_fun(
    name="B^theta_rr",
    label="\\partial_{\\rho\\rho} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant poloidal component of magnetic field, second derivative wrt"
        " radial and radial coordinates"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_r",
        "B0_rr",
        "iota",
        "iota_r",
        "iota_rr",
        "lambda_rrz",
        "lambda_rz",
        "lambda_z",
        "omega_rrz",
        "omega_rz",
        "omega_z",
    ],
)
def _B_sup_theta_rr(params, transforms, profiles, data, **kwargs):
    data["B^theta_rr"] = (
        data["B0"]
        * (
            data["iota"] * data["omega_rrz"]
            + 2 * data["iota_r"] * data["omega_rz"]
            + data["iota_rr"] * data["omega_z"]
            + data["iota_rr"]
            - data["lambda_rrz"]
        )
        + 2
        * data["B0_r"]
        * (
            data["iota"] * data["omega_rz"]
            + data["iota_r"] * data["omega_z"]
            + data["iota_r"]
            - data["lambda_rz"]
        )
        + data["B0_rr"]
        * (data["iota"] * data["omega_z"] + data["iota"] - data["lambda_z"])
    )
    return data


@register_compute_fun(
    name="B^zeta_rr",
    label="\\partial_{\\rho\\rho} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant toroidal component of magnetic field, second derivative wrt"
        " radial and radial coordinates"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_r",
        "B0_rr",
        "iota",
        "iota_r",
        "iota_rr",
        "lambda_rrt",
        "lambda_rt",
        "lambda_t",
        "omega_rrt",
        "omega_rt",
        "omega_t",
    ],
)
def _B_sup_zeta_rr(params, transforms, profiles, data, **kwargs):
    data["B^zeta_rr"] = (
        -data["B0"]
        * (
            data["iota"] * data["omega_rrt"]
            + 2 * data["iota_r"] * data["omega_rt"]
            + data["iota_rr"] * data["omega_t"]
            - data["lambda_rrt"]
        )
        - 2
        * data["B0_r"]
        * (
            data["iota"] * data["omega_rt"]
            + data["iota_r"] * data["omega_t"]
            - data["lambda_rt"]
        )
        + data["B0_rr"] * (-data["iota"] * data["omega_t"] + data["lambda_t"] + 1)
    )
    return data


@register_compute_fun(
    name="B_rr",
    label="\\partial_{\\rho\\rho} \\mathbf{B}",
    units="T",
    units_long="Tesla",
    description="Magnetic field, second derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^theta_r",
        "B^theta_rr",
        "B^zeta",
        "B^zeta_r",
        "B^zeta_rr",
        "e_theta",
        "e_theta_r",
        "e_theta_rr",
        "e_zeta",
        "e_zeta_r",
        "e_zeta_rr",
    ],
)
def _B_rr(params, transforms, profiles, data, **kwargs):
    data["B_rr"] = (
        data["B^theta_rr"] * data["e_theta"].T
        + 2 * data["B^theta_r"] * data["e_theta_r"].T
        + data["B^theta"] * data["e_theta_rr"].T
        + data["B^zeta_rr"] * data["e_zeta"].T
        + 2 * data["B^zeta_r"] * data["e_zeta_r"].T
        + data["B^zeta"] * data["e_zeta_rr"].T
    ).T
    return data


@register_compute_fun(
    name="B0_tt",
    label="\\partial_{\\theta \\theta} (\\psi' / \\sqrt{g})",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "sqrt(g)", "sqrt(g)_t", "sqrt(g)_tt"],
    axis_limit_data=["psi_rr", "sqrt(g)_r", "sqrt(g)_rt", "sqrt(g)_rtt"],
)
def _B0_tt(params, transforms, profiles, data, **kwargs):
    data["B0_tt"] = transforms["grid"].replace_at_axis(
        data["psi_r"]
        * (2 * data["sqrt(g)_t"] ** 2 - data["sqrt(g)"] * data["sqrt(g)_tt"])
        / data["sqrt(g)"] ** 3,
        lambda: data["psi_rr"]
        * (2 * data["sqrt(g)_rt"] ** 2 - data["sqrt(g)_r"] * data["sqrt(g)_rtt"])
        / data["sqrt(g)_r"] ** 3,
    )
    return data


@register_compute_fun(
    name="B^theta_tt",
    label="\\partial_{\\theta\\theta} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant poloidal component of magnetic field, second derivative wrt"
        " poloidal and poloidal coordinates"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_t",
        "B0_tt",
        "iota",
        "lambda_ttz",
        "lambda_tz",
        "lambda_z",
        "omega_ttz",
        "omega_tz",
        "omega_z",
    ],
)
def _B_sup_theta_tt(params, transforms, profiles, data, **kwargs):
    data["B^theta_tt"] = (
        data["B0"] * (data["iota"] * data["omega_ttz"] - data["lambda_ttz"])
        + 2 * data["B0_t"] * (data["iota"] * data["omega_tz"] - data["lambda_tz"])
        + data["B0_tt"]
        * (data["iota"] * data["omega_z"] + data["iota"] - data["lambda_z"])
    )
    return data


@register_compute_fun(
    name="B^zeta_tt",
    label="\\partial_{\\theta\\theta} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant toroidal component of magnetic field, second derivative wrt"
        " poloidal and poloidal coordinates"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_t",
        "B0_tt",
        "iota",
        "lambda_t",
        "lambda_tt",
        "lambda_ttt",
        "omega_t",
        "omega_tt",
        "omega_ttt",
    ],
)
def _B_sup_zeta_tt(params, transforms, profiles, data, **kwargs):
    data["B^zeta_tt"] = (
        -data["B0"] * (data["iota"] * data["omega_ttt"] - data["lambda_ttt"])
        - 2 * data["B0_t"] * (data["iota"] * data["omega_tt"] - data["lambda_tt"])
        + data["B0_tt"] * (-data["iota"] * data["omega_t"] + data["lambda_t"] + 1)
    )
    return data


@register_compute_fun(
    name="B_tt",
    label="\\partial_{\\theta\\theta} \\mathbf{B}",
    units="T",
    units_long="Tesla",
    description="Magnetic field, second derivative wrt poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^theta_t",
        "B^theta_tt",
        "B^zeta",
        "B^zeta_t",
        "B^zeta_tt",
        "e_theta",
        "e_theta_t",
        "e_theta_tt",
        "e_zeta",
        "e_zeta_t",
        "e_zeta_tt",
    ],
)
def _B_tt(params, transforms, profiles, data, **kwargs):
    data["B_tt"] = (
        data["B^theta_tt"] * data["e_theta"].T
        + 2 * data["B^theta_t"] * data["e_theta_t"].T
        + data["B^theta"] * data["e_theta_tt"].T
        + data["B^zeta_tt"] * data["e_zeta"].T
        + 2 * data["B^zeta_t"] * data["e_zeta_t"].T
        + data["B^zeta"] * data["e_zeta_tt"].T
    ).T
    return data


@register_compute_fun(
    name="B0_zz",
    label="\\partial_{\\zeta \\zeta} (\\psi' / \\sqrt{g})",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "sqrt(g)", "sqrt(g)_z", "sqrt(g)_zz"],
    axis_limit_data=["psi_rr", "sqrt(g)_r", "sqrt(g)_rz", "sqrt(g)_rzz"],
)
def _B0_zz(params, transforms, profiles, data, **kwargs):
    data["B0_zz"] = transforms["grid"].replace_at_axis(
        data["psi_r"]
        * (2 * data["sqrt(g)_z"] ** 2 - data["sqrt(g)"] * data["sqrt(g)_zz"])
        / data["sqrt(g)"] ** 3,
        lambda: data["psi_rr"]
        * (2 * data["sqrt(g)_rz"] ** 2 - data["sqrt(g)_r"] * data["sqrt(g)_rzz"])
        / data["sqrt(g)_r"] ** 3,
    )
    return data


@register_compute_fun(
    name="B^theta_zz",
    label="\\partial_{\\zeta\\zeta} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant poloidal component of magnetic field, second derivative wrt"
        " toroidal and toroidal coordinates"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_z",
        "B0_zz",
        "iota",
        "lambda_z",
        "lambda_zz",
        "lambda_zzz",
        "omega_z",
        "omega_zz",
        "omega_zzz",
    ],
)
def _B_sup_theta_zz(params, transforms, profiles, data, **kwargs):
    data["B^theta_zz"] = (
        data["B0"] * (data["iota"] * data["omega_zzz"] - data["lambda_zzz"])
        + 2 * data["B0_z"] * (data["iota"] * data["omega_zz"] - data["lambda_zz"])
        + data["B0_zz"]
        * (data["iota"] * data["omega_z"] + data["iota"] - data["lambda_z"])
    )
    return data


@register_compute_fun(
    name="B^zeta_zz",
    label="\\partial_{\\zeta\\zeta} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant toroidal component of magnetic field, second derivative wrt"
        " toroidal and toroidal coordinates"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_z",
        "B0_zz",
        "iota",
        "lambda_t",
        "lambda_tz",
        "lambda_tzz",
        "omega_t",
        "omega_tz",
        "omega_tzz",
    ],
)
def _B_sup_zeta_zz(params, transforms, profiles, data, **kwargs):
    data["B^zeta_zz"] = (
        -data["B0"] * (data["iota"] * data["omega_tzz"] - data["lambda_tzz"])
        - 2 * data["B0_z"] * (data["iota"] * data["omega_tz"] - data["lambda_tz"])
        + data["B0_zz"] * (-data["iota"] * data["omega_t"] + data["lambda_t"] + 1)
    )
    return data


@register_compute_fun(
    name="B_zz",
    label="\\partial_{\\zeta\\zeta} \\mathbf{B}",
    units="T",
    units_long="Tesla",
    description="Magnetic field, second derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^theta_z",
        "B^theta_zz",
        "B^zeta",
        "B^zeta_z",
        "B^zeta_zz",
        "e_theta",
        "e_theta_z",
        "e_theta_zz",
        "e_zeta",
        "e_zeta_z",
        "e_zeta_zz",
    ],
)
def _B_zz(params, transforms, profiles, data, **kwargs):
    data["B_zz"] = (
        data["B^theta_zz"] * data["e_theta"].T
        + 2 * data["B^theta_z"] * data["e_theta_z"].T
        + data["B^theta"] * data["e_theta_zz"].T
        + data["B^zeta_zz"] * data["e_zeta"].T
        + 2 * data["B^zeta_z"] * data["e_zeta_z"].T
        + data["B^zeta"] * data["e_zeta_zz"].T
    ).T
    return data


@register_compute_fun(
    name="B0_rt",
    label="\\partial_{\\rho\\theta} (\\psi' / \\sqrt{g})",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "psi_rr", "sqrt(g)", "sqrt(g)_r", "sqrt(g)_t", "sqrt(g)_rt"],
    axis_limit_data=["psi_rrr", "sqrt(g)_rr", "sqrt(g)_rrt"],
)
def _B0_rt(params, transforms, profiles, data, **kwargs):
    data["B0_rt"] = transforms["grid"].replace_at_axis(
        (
            -data["sqrt(g)"]
            * (data["psi_rr"] * data["sqrt(g)_t"] + data["psi_r"] * data["sqrt(g)_rt"])
            + 2 * data["psi_r"] * data["sqrt(g)_r"] * data["sqrt(g)_t"]
        )
        / data["sqrt(g)"] ** 3,
        lambda: (
            -data["sqrt(g)_r"]
            * (
                data["psi_rrr"] * data["sqrt(g)_rt"]
                + data["psi_rr"] * data["sqrt(g)_rrt"]
            )
            + 2 * data["psi_rr"] * data["sqrt(g)_rr"] * data["sqrt(g)_rt"]
        )
        / (2 * data["sqrt(g)_r"] ** 3),
    )
    return data


@register_compute_fun(
    name="B^theta_rt",
    label="\\partial_{\\rho\\theta} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant poloidal component of magnetic field, second derivative wrt"
        " radial and poloidal coordinates"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_r",
        "B0_rt",
        "B0_t",
        "iota",
        "iota_r",
        "lambda_rtz",
        "lambda_rz",
        "lambda_tz",
        "lambda_z",
        "omega_rtz",
        "omega_rz",
        "omega_tz",
        "omega_z",
    ],
)
def _B_sup_theta_rt(params, transforms, profiles, data, **kwargs):
    data["B^theta_rt"] = (
        data["B0"]
        * (
            data["iota"] * data["omega_rtz"]
            + data["iota_r"] * data["omega_tz"]
            - data["lambda_rtz"]
        )
        + data["B0_r"] * (data["iota"] * data["omega_tz"] - data["lambda_tz"])
        + data["B0_t"]
        * (
            data["iota"] * data["omega_rz"]
            + data["iota_r"] * data["omega_z"]
            + data["iota_r"]
            - data["lambda_rz"]
        )
        + data["B0_rt"]
        * (data["iota"] * data["omega_z"] + data["iota"] - data["lambda_z"])
    )
    return data


@register_compute_fun(
    name="B^zeta_rt",
    label="\\partial_{\\rho\\theta} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant toroidal component of magnetic field, second derivative wrt"
        " radial and poloidal coordinates"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_r",
        "B0_rt",
        "B0_t",
        "iota",
        "iota_r",
        "lambda_rt",
        "lambda_rtt",
        "lambda_t",
        "lambda_tt",
        "omega_rt",
        "omega_rtt",
        "omega_t",
        "omega_tt",
    ],
)
def _B_sup_zeta_rt(params, transforms, profiles, data, **kwargs):
    data["B^zeta_rt"] = (
        -data["B0"]
        * (
            data["iota"] * data["omega_rtt"]
            + data["iota_r"] * data["omega_tt"]
            - data["lambda_rtt"]
        )
        - data["B0_r"] * (data["iota"] * data["omega_tt"] - data["lambda_tt"])
        - data["B0_t"]
        * (
            data["iota"] * data["omega_rt"]
            + data["iota_r"] * data["omega_t"]
            - data["lambda_rt"]
        )
        + data["B0_rt"] * (-data["iota"] * data["omega_t"] + data["lambda_t"] + 1)
    )
    return data


@register_compute_fun(
    name="B_rt",
    label="\\partial_{\\rho\\theta} \\mathbf{B}",
    units="T",
    units_long="Tesla",
    description="Magnetic field, second derivative wrt radial coordinate and poloidal "
    + "angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^theta_r",
        "B^theta_t",
        "B^theta_rt",
        "B^zeta",
        "B^zeta_r",
        "B^zeta_t",
        "B^zeta_rt",
        "e_theta",
        "e_theta_r",
        "e_theta_t",
        "e_theta_rt",
        "e_zeta",
        "e_zeta_r",
        "e_zeta_t",
        "e_zeta_rt",
    ],
)
def _B_rt(params, transforms, profiles, data, **kwargs):
    data["B_rt"] = (
        data["B^theta_rt"] * data["e_theta"].T
        + data["B^theta_r"] * data["e_theta_t"].T
        + data["B^theta_t"] * data["e_theta_r"].T
        + data["B^theta"] * data["e_theta_rt"].T
        + data["B^zeta_rt"] * data["e_zeta"].T
        + data["B^zeta_r"] * data["e_zeta_t"].T
        + data["B^zeta_t"] * data["e_zeta_r"].T
        + data["B^zeta"] * data["e_zeta_rt"].T
    ).T
    return data


@register_compute_fun(
    name="B0_tz",
    label="\\partial_{\\theta\\zeta} (\\psi' / \\sqrt{g})",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "sqrt(g)", "sqrt(g)_t", "sqrt(g)_z", "sqrt(g)_tz"],
    axis_limit_data=["psi_rr", "sqrt(g)_r", "sqrt(g)_rt", "sqrt(g)_rz", "sqrt(g)_rtz"],
)
def _B0_tz(params, transforms, profiles, data, **kwargs):
    data["B0_tz"] = transforms["grid"].replace_at_axis(
        data["psi_r"]
        * (
            2 * data["sqrt(g)_t"] * data["sqrt(g)_z"]
            - data["sqrt(g)_tz"] * data["sqrt(g)"]
        )
        / data["sqrt(g)"] ** 3,
        lambda: data["psi_rr"]
        * (
            2 * data["sqrt(g)_rt"] * data["sqrt(g)_rz"]
            - data["sqrt(g)_rtz"] * data["sqrt(g)_r"]
        )
        / data["sqrt(g)_r"] ** 3,
    )
    return data


@register_compute_fun(
    name="B^theta_tz",
    label="\\partial_{\\theta\\zeta} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant poloidal component of magnetic field, second derivative wrt"
        " poloidal and toroidal coordinates"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_t",
        "B0_tz",
        "B0_z",
        "iota",
        "lambda_tz",
        "lambda_tzz",
        "lambda_z",
        "lambda_zz",
        "omega_tz",
        "omega_tzz",
        "omega_z",
        "omega_zz",
    ],
)
def _B_sup_theta_tz(params, transforms, profiles, data, **kwargs):
    data["B^theta_tz"] = (
        data["B0"] * (data["iota"] * data["omega_tzz"] - data["lambda_tzz"])
        + data["B0_t"] * (data["iota"] * data["omega_zz"] - data["lambda_zz"])
        + data["B0_z"] * (data["iota"] * data["omega_tz"] - data["lambda_tz"])
        + data["B0_tz"]
        * (data["iota"] * data["omega_z"] + data["iota"] - data["lambda_z"])
    )
    return data


@register_compute_fun(
    name="B^zeta_tz",
    label="\\partial_{\\theta\\zeta} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant toroidal component of magnetic field, second derivative wrt"
        " poloidal and toroidal coordinates"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_t",
        "B0_tz",
        "B0_z",
        "iota",
        "lambda_t",
        "lambda_tt",
        "lambda_ttz",
        "lambda_tz",
        "omega_t",
        "omega_tt",
        "omega_ttz",
        "omega_tz",
    ],
)
def _B_sup_zeta_tz(params, transforms, profiles, data, **kwargs):
    data["B^zeta_tz"] = (
        -data["B0"] * (data["iota"] * data["omega_ttz"] - data["lambda_ttz"])
        - data["B0_t"] * (data["iota"] * data["omega_tz"] - data["lambda_tz"])
        - data["B0_z"] * (data["iota"] * data["omega_tt"] - data["lambda_tt"])
        + data["B0_tz"] * (-data["iota"] * data["omega_t"] + data["lambda_t"] + 1)
    )
    return data


@register_compute_fun(
    name="B_tz",
    label="\\partial_{\\theta\\zeta} \\mathbf{B}",
    units="T",
    units_long="Tesla",
    description="Magnetic field, second derivative wrt poloidal and toroidal angles",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^theta_t",
        "B^theta_z",
        "B^theta_tz",
        "B^zeta",
        "B^zeta_t",
        "B^zeta_z",
        "B^zeta_tz",
        "e_theta",
        "e_theta_t",
        "e_theta_z",
        "e_theta_tz",
        "e_zeta",
        "e_zeta_t",
        "e_zeta_z",
        "e_zeta_tz",
    ],
)
def _B_tz(params, transforms, profiles, data, **kwargs):
    data["B_tz"] = (
        data["B^theta_tz"] * data["e_theta"].T
        + data["B^theta_t"] * data["e_theta_z"].T
        + data["B^theta_z"] * data["e_theta_t"].T
        + data["B^theta"] * data["e_theta_tz"].T
        + data["B^zeta_tz"] * data["e_zeta"].T
        + data["B^zeta_t"] * data["e_zeta_z"].T
        + data["B^zeta_z"] * data["e_zeta_t"].T
        + data["B^zeta"] * data["e_zeta_tz"].T
    ).T
    return data


@register_compute_fun(
    name="B0_rz",
    label="\\partial_{\\rho\\zeta} (\\psi' / \\sqrt{g})",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "psi_rr", "sqrt(g)", "sqrt(g)_r", "sqrt(g)_z", "sqrt(g)_rz"],
    axis_limit_data=["psi_rrr", "sqrt(g)_rr", "sqrt(g)_rrz"],
)
def _B0_rz(params, transforms, profiles, data, **kwargs):
    data["B0_rz"] = transforms["grid"].replace_at_axis(
        (
            -data["sqrt(g)"]
            * (data["psi_rr"] * data["sqrt(g)_z"] + data["psi_r"] * data["sqrt(g)_rz"])
            + 2 * data["psi_r"] * data["sqrt(g)_r"] * data["sqrt(g)_z"]
        )
        / data["sqrt(g)"] ** 3,
        lambda: (
            -data["sqrt(g)_r"]
            * (
                data["psi_rrr"] * data["sqrt(g)_rz"]
                + data["psi_rr"] * data["sqrt(g)_rrz"]
            )
            + 2 * data["psi_rr"] * data["sqrt(g)_rr"] * data["sqrt(g)_rz"]
        )
        / (2 * data["sqrt(g)_r"] ** 3),
    )
    return data


@register_compute_fun(
    name="B^theta_rz",
    label="\\partial_{\\rho\\zeta} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant poloidal component of magnetic field, second derivative wrt"
        " radial and toroidal coordinates"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_r",
        "B0_rz",
        "B0_z",
        "iota",
        "iota_r",
        "lambda_rz",
        "lambda_rzz",
        "lambda_z",
        "lambda_zz",
        "omega_rz",
        "omega_rzz",
        "omega_z",
        "omega_zz",
    ],
)
def _B_sup_theta_rz(params, transforms, profiles, data, **kwargs):
    data["B^theta_rz"] = (
        data["B0"]
        * (
            data["iota"] * data["omega_rzz"]
            + data["iota_r"] * data["omega_zz"]
            - data["lambda_rzz"]
        )
        + data["B0_r"] * (data["iota"] * data["omega_zz"] - data["lambda_zz"])
        + data["B0_z"]
        * (
            data["iota"] * data["omega_rz"]
            + data["iota_r"] * data["omega_z"]
            + data["iota_r"]
            - data["lambda_rz"]
        )
        + data["B0_rz"]
        * (data["iota"] * data["omega_z"] + data["iota"] - data["lambda_z"])
    )
    return data


@register_compute_fun(
    name="B^zeta_rz",
    label="\\partial_{\\rho\\zeta} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description=(
        "Contravariant toroidal component of magnetic field, second derivative wrt"
        " radial and toroidal coordinates"
    ),
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_r",
        "B0_rz",
        "B0_z",
        "iota",
        "iota_r",
        "lambda_rt",
        "lambda_rtz",
        "lambda_t",
        "lambda_tz",
        "omega_rt",
        "omega_rtz",
        "omega_t",
        "omega_tz",
    ],
)
def _B_sup_zeta_rz(params, transforms, profiles, data, **kwargs):
    data["B^zeta_rz"] = (
        -data["B0"]
        * (
            data["iota"] * data["omega_rtz"]
            + data["iota_r"] * data["omega_tz"]
            - data["lambda_rtz"]
        )
        - data["B0_r"] * (data["iota"] * data["omega_tz"] - data["lambda_tz"])
        - data["B0_z"]
        * (
            data["iota"] * data["omega_rt"]
            + data["iota_r"] * data["omega_t"]
            - data["lambda_rt"]
        )
        + data["B0_rz"] * (-data["iota"] * data["omega_t"] + data["lambda_t"] + 1)
    )
    return data


@register_compute_fun(
    name="B_rz",
    label="\\partial_{\\rho\\zeta} \\mathbf{B}",
    units="T",
    units_long="Tesla",
    description="Magnetic field, second derivative wrt radial coordinate and toroidal "
    + "angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^theta_r",
        "B^theta_z",
        "B^theta_rz",
        "B^zeta",
        "B^zeta_r",
        "B^zeta_z",
        "B^zeta_rz",
        "e_theta",
        "e_theta_r",
        "e_theta_z",
        "e_theta_rz",
        "e_zeta",
        "e_zeta_r",
        "e_zeta_z",
        "e_zeta_rz",
    ],
)
def _B_rz(params, transforms, profiles, data, **kwargs):
    data["B_rz"] = (
        data["B^theta_rz"] * data["e_theta"].T
        + data["B^theta_r"] * data["e_theta_z"].T
        + data["B^theta_z"] * data["e_theta_r"].T
        + data["B^theta"] * data["e_theta_rz"].T
        + data["B^zeta_rz"] * data["e_zeta"].T
        + data["B^zeta_r"] * data["e_zeta_z"].T
        + data["B^zeta_z"] * data["e_zeta_r"].T
        + data["B^zeta"] * data["e_zeta_rz"].T
    ).T
    return data


@register_compute_fun(
    name="B_rho",
    label="B_{\\rho}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant radial component of magnetic field",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "e_rho"],
)
def _B_sub_rho(params, transforms, profiles, data, **kwargs):
    data["B_rho"] = dot(data["B"], data["e_rho"])
    return data


@register_compute_fun(
    name="B_theta",
    label="B_{\\theta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant poloidal component of magnetic field",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "e_theta"],
)
def _B_sub_theta(params, transforms, profiles, data, **kwargs):
    data["B_theta"] = dot(data["B"], data["e_theta"])
    return data


@register_compute_fun(
    name="B_zeta",
    label="B_{\\zeta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant toroidal component of magnetic field",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "e_zeta"],
)
def _B_sub_zeta(params, transforms, profiles, data, **kwargs):
    data["B_zeta"] = dot(data["B"], data["e_zeta"])
    return data


@register_compute_fun(
    name="B_rho_r",
    label="\\partial_{\\rho} B_{\\rho}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant radial component of magnetic field, derivative "
    + "wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_r", "e_rho", "e_rho_r"],
)
def _B_sub_rho_r(params, transforms, profiles, data, **kwargs):
    data["B_rho_r"] = dot(data["B_r"], data["e_rho"]) + dot(data["B"], data["e_rho_r"])
    return data


@register_compute_fun(
    name="B_theta_r",
    label="\\partial_{\\rho} B_{\\theta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant poloidal component of magnetic field, derivative "
    + "wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_r", "e_theta", "e_theta_r"],
)
def _B_sub_theta_r(params, transforms, profiles, data, **kwargs):
    data["B_theta_r"] = dot(data["B_r"], data["e_theta"]) + dot(
        data["B"], data["e_theta_r"]
    )
    return data


@register_compute_fun(
    name="B_zeta_r",
    label="\\partial_{\\rho} B_{\\zeta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant toroidal component of magnetic field, derivative "
    + "wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_r", "e_zeta", "e_zeta_r"],
)
def _B_sub_zeta_r(params, transforms, profiles, data, **kwargs):
    data["B_zeta_r"] = dot(data["B_r"], data["e_zeta"]) + dot(
        data["B"], data["e_zeta_r"]
    )
    return data


@register_compute_fun(
    name="B_rho_t",
    label="\\partial_{\\theta} B_{\\rho}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant radial component of magnetic field, derivative "
    + "wrt poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_t", "e_rho", "e_rho_t"],
)
def _B_sub_rho_t(params, transforms, profiles, data, **kwargs):
    data["B_rho_t"] = dot(data["B_t"], data["e_rho"]) + dot(data["B"], data["e_rho_t"])
    return data


@register_compute_fun(
    name="B_theta_t",
    label="\\partial_{\\theta} B_{\\theta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant poloidal component of magnetic field, derivative "
    + "wrt poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_t", "e_theta", "e_theta_t"],
)
def _B_sub_theta_t(params, transforms, profiles, data, **kwargs):
    data["B_theta_t"] = dot(data["B_t"], data["e_theta"]) + dot(
        data["B"], data["e_theta_t"]
    )
    return data


@register_compute_fun(
    name="B_zeta_t",
    label="\\partial_{\\theta} B_{\\zeta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant toroidal component of magnetic field, derivative "
    + "wrt poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_t", "e_zeta", "e_zeta_t"],
)
def _B_sub_zeta_t(params, transforms, profiles, data, **kwargs):
    data["B_zeta_t"] = dot(data["B_t"], data["e_zeta"]) + dot(
        data["B"], data["e_zeta_t"]
    )
    return data


@register_compute_fun(
    name="B_rho_z",
    label="\\partial_{\\zeta} B_{\\rho}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant radial component of magnetic field, derivative "
    + "wrt toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_z", "e_rho", "e_rho_z"],
)
def _B_sub_rho_z(params, transforms, profiles, data, **kwargs):
    data["B_rho_z"] = dot(data["B_z"], data["e_rho"]) + dot(data["B"], data["e_rho_z"])
    return data


@register_compute_fun(
    name="B_theta_z",
    label="\\partial_{\\zeta} B_{\\theta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant poloidal component of magnetic field, derivative "
    + "wrt toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_z", "e_theta", "e_theta_z"],
)
def _B_sub_theta_z(params, transforms, profiles, data, **kwargs):
    data["B_theta_z"] = dot(data["B_z"], data["e_theta"]) + dot(
        data["B"], data["e_theta_z"]
    )
    return data


@register_compute_fun(
    name="B_zeta_z",
    label="\\partial_{\\zeta} B_{\\zeta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant toroidal component of magnetic field, derivative "
    + "wrt toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_z", "e_zeta", "e_zeta_z"],
)
def _B_sub_zeta_z(params, transforms, profiles, data, **kwargs):
    data["B_zeta_z"] = dot(data["B_z"], data["e_zeta"]) + dot(
        data["B"], data["e_zeta_z"]
    )
    return data


@register_compute_fun(
    name="B_rho_rr",
    label="\\partial_{\\rho\\rho} B_{\\rho}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant radial component of magnetic field, second derivative "
    + "wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_r", "B_rr", "e_rho", "e_rho_r", "e_rho_rr"],
)
def _B_sub_rho_rr(params, transforms, profiles, data, **kwargs):
    data["B_rho_rr"] = (
        dot(data["B_rr"], data["e_rho"])
        + 2 * dot(data["B_r"], data["e_rho_r"])
        + dot(data["B"], data["e_rho_rr"])
    )
    return data


@register_compute_fun(
    name="B_theta_rr",
    label="\\partial_{\\rho\\rho} B_{\\theta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant poloidal component of magnetic field, second derivative "
    + "wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_r", "B_rr", "e_theta", "e_theta_r", "e_theta_rr"],
)
def _B_sub_theta_rr(params, transforms, profiles, data, **kwargs):
    data["B_theta_rr"] = (
        dot(data["B_rr"], data["e_theta"])
        + 2 * dot(data["B_r"], data["e_theta_r"])
        + dot(data["B"], data["e_theta_rr"])
    )
    return data


@register_compute_fun(
    name="B_zeta_rr",
    label="\\partial_{\\rho\\rho} B_{\\zeta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant toroidal component of magnetic field, second derivative "
    + "wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_r", "B_rr", "e_zeta", "e_zeta_r", "e_zeta_rr"],
)
def _B_sub_zeta_rr(params, transforms, profiles, data, **kwargs):
    data["B_zeta_rr"] = (
        dot(data["B_rr"], data["e_zeta"])
        + 2 * dot(data["B_r"], data["e_zeta_r"])
        + dot(data["B"], data["e_zeta_rr"])
    )
    return data


@register_compute_fun(
    name="B_rho_tt",
    label="\\partial_{\\theta\\theta} B_{\\rho}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant radial component of magnetic field, second derivative "
    + "wrt poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_t", "B_tt", "e_rho", "e_rho_t", "e_rho_tt"],
)
def _B_sub_rho_tt(params, transforms, profiles, data, **kwargs):
    data["B_rho_tt"] = (
        dot(data["B_tt"], data["e_rho"])
        + 2 * dot(data["B_t"], data["e_rho_t"])
        + dot(data["B"], data["e_rho_tt"])
    )
    return data


@register_compute_fun(
    name="B_theta_tt",
    label="\\partial_{\\theta\\theta} B_{\\theta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant poloidal component of magnetic field, second derivative "
    + "wrt poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_t", "B_tt", "e_theta", "e_theta_t", "e_theta_tt"],
)
def _B_sub_theta_tt(params, transforms, profiles, data, **kwargs):
    data["B_theta_tt"] = (
        dot(data["B_tt"], data["e_theta"])
        + 2 * dot(data["B_t"], data["e_theta_t"])
        + dot(data["B"], data["e_theta_tt"])
    )
    return data


@register_compute_fun(
    name="B_zeta_tt",
    label="\\partial_{\\theta\\theta} B_{\\zeta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant toroidal component of magnetic field, second derivative "
    + "wrt poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_t", "B_tt", "e_zeta", "e_zeta_t", "e_zeta_tt"],
)
def _B_sub_zeta_tt(params, transforms, profiles, data, **kwargs):
    data["B_zeta_tt"] = (
        dot(data["B_tt"], data["e_zeta"])
        + 2 * dot(data["B_t"], data["e_zeta_t"])
        + dot(data["B"], data["e_zeta_tt"])
    )
    return data


@register_compute_fun(
    name="B_rho_zz",
    label="\\partial_{\\zeta\\zeta} B_{\\rho}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant radial component of magnetic field, second derivative "
    + "wrt toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_z", "B_zz", "e_rho", "e_rho_z", "e_rho_zz"],
)
def _B_sub_rho_zz(params, transforms, profiles, data, **kwargs):
    data["B_rho_zz"] = (
        dot(data["B_zz"], data["e_rho"])
        + 2 * dot(data["B_z"], data["e_rho_z"])
        + dot(data["B"], data["e_rho_zz"])
    )
    return data


@register_compute_fun(
    name="B_theta_zz",
    label="\\partial_{\\zeta\\zeta} B_{\\theta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant poloidal component of magnetic field, second derivative "
    + "wrt toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_z", "B_zz", "e_theta", "e_theta_z", "e_theta_zz"],
)
def _B_sub_theta_zz(params, transforms, profiles, data, **kwargs):
    data["B_theta_zz"] = (
        dot(data["B_zz"], data["e_theta"])
        + 2 * dot(data["B_z"], data["e_theta_z"])
        + dot(data["B"], data["e_theta_zz"])
    )
    return data


@register_compute_fun(
    name="B_zeta_zz",
    label="\\partial_{\\zeta\\zeta} B_{\\zeta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant toroidal component of magnetic field, second derivative "
    + "wrt toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_z", "B_zz", "e_zeta", "e_zeta_z", "e_zeta_zz"],
)
def _B_sub_zeta_zz(params, transforms, profiles, data, **kwargs):
    data["B_zeta_zz"] = (
        dot(data["B_zz"], data["e_zeta"])
        + 2 * dot(data["B_z"], data["e_zeta_z"])
        + dot(data["B"], data["e_zeta_zz"])
    )
    return data


@register_compute_fun(
    name="B_rho_rt",
    label="\\partial_{\\rho\\theta} B_{\\rho}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant radial component of magnetic field, second derivative "
    + "wrt radial coordinate and poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_r", "B_t", "B_rt", "e_rho", "e_rho_r", "e_rho_t", "e_rho_rt"],
)
def _B_sub_rho_rt(params, transforms, profiles, data, **kwargs):
    data["B_rho_rt"] = (
        dot(data["B_rt"], data["e_rho"])
        + dot(data["B_r"], data["e_rho_t"])
        + dot(data["B_t"], data["e_rho_r"])
        + dot(data["B"], data["e_rho_rt"])
    )
    return data


@register_compute_fun(
    name="B_theta_rt",
    label="\\partial_{\\rho\\theta} B_{\\theta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant poloidal component of magnetic field, second derivative "
    + "wrt radial coordinate and poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_r", "B_t", "B_rt", "e_theta", "e_theta_r", "e_theta_t", "e_theta_rt"],
)
def _B_sub_theta_rt(params, transforms, profiles, data, **kwargs):
    data["B_theta_rt"] = (
        dot(data["B_rt"], data["e_theta"])
        + dot(data["B_r"], data["e_theta_t"])
        + dot(data["B_t"], data["e_theta_r"])
        + dot(data["B"], data["e_theta_rt"])
    )
    return data


@register_compute_fun(
    name="B_zeta_rt",
    label="\\partial_{\\rho\\theta} B_{\\zeta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant toroidal component of magnetic field, second derivative "
    + "wrt radial coordinate and poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_r", "B_t", "B_rt", "e_zeta", "e_zeta_r", "e_zeta_t", "e_zeta_rt"],
)
def _B_sub_zeta_rt(params, transforms, profiles, data, **kwargs):
    data["B_zeta_rt"] = (
        dot(data["B_rt"], data["e_zeta"])
        + dot(data["B_r"], data["e_zeta_t"])
        + dot(data["B_t"], data["e_zeta_r"])
        + dot(data["B"], data["e_zeta_rt"])
    )
    return data


@register_compute_fun(
    name="B_rho_tz",
    label="\\partial_{\\theta\\zeta} B_{\\rho}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant radial component of magnetic field, second derivative "
    + "wrt poloidal and toroidal angles",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_t", "B_z", "B_tz", "e_rho", "e_rho_t", "e_rho_z", "e_rho_tz"],
)
def _B_sub_rho_tz(params, transforms, profiles, data, **kwargs):
    data["B_rho_tz"] = (
        dot(data["B_tz"], data["e_rho"])
        + dot(data["B_t"], data["e_rho_z"])
        + dot(data["B_z"], data["e_rho_t"])
        + dot(data["B"], data["e_rho_tz"])
    )
    return data


@register_compute_fun(
    name="B_theta_tz",
    label="\\partial_{\\theta\\zeta} B_{\\theta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant poloidal component of magnetic field, second derivative "
    + "wrt poloidal and toroidal angles",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_t", "B_z", "B_tz", "e_theta", "e_theta_t", "e_theta_z", "e_theta_tz"],
)
def _B_sub_theta_tz(params, transforms, profiles, data, **kwargs):
    data["B_theta_tz"] = (
        dot(data["B_tz"], data["e_theta"])
        + dot(data["B_t"], data["e_theta_z"])
        + dot(data["B_z"], data["e_theta_t"])
        + dot(data["B"], data["e_theta_tz"])
    )
    return data


@register_compute_fun(
    name="B_zeta_tz",
    label="\\partial_{\\theta\\zeta} B_{\\zeta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant toroidal component of magnetic field, second derivative "
    + "wrt poloidal and toroidal angles",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_t", "B_z", "B_tz", "e_zeta", "e_zeta_t", "e_zeta_z", "e_zeta_tz"],
)
def _B_sub_zeta_tz(params, transforms, profiles, data, **kwargs):
    data["B_zeta_tz"] = (
        dot(data["B_tz"], data["e_zeta"])
        + dot(data["B_t"], data["e_zeta_z"])
        + dot(data["B_z"], data["e_zeta_t"])
        + dot(data["B"], data["e_zeta_tz"])
    )
    return data


@register_compute_fun(
    name="B_rho_rz",
    label="\\partial_{\\rho\\zeta} B_{\\rho}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant radial component of magnetic field, second derivative "
    + "wrt radial coordinate and toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_r", "B_z", "B_rz", "e_rho", "e_rho_r", "e_rho_z", "e_rho_rz"],
)
def _B_sub_rho_rz(params, transforms, profiles, data, **kwargs):
    data["B_rho_rz"] = (
        dot(data["B_rz"], data["e_rho"])
        + dot(data["B_r"], data["e_rho_z"])
        + dot(data["B_z"], data["e_rho_r"])
        + dot(data["B"], data["e_rho_rz"])
    )
    return data


@register_compute_fun(
    name="B_theta_rz",
    label="\\partial_{\\rho\\zeta} B_{\\theta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant poloidal component of magnetic field, second derivative "
    + "wrt radial coordinate and toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_r", "B_z", "B_rz", "e_theta", "e_theta_r", "e_theta_z", "e_theta_rz"],
)
def _B_sub_theta_rz(params, transforms, profiles, data, **kwargs):
    data["B_theta_rz"] = (
        dot(data["B_rz"], data["e_theta"])
        + dot(data["B_r"], data["e_theta_z"])
        + dot(data["B_z"], data["e_theta_r"])
        + dot(data["B"], data["e_theta_rz"])
    )
    return data


@register_compute_fun(
    name="B_zeta_rz",
    label="\\partial_{\\rho\\zeta} B_{\\zeta}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant toroidal component of magnetic field, second derivative "
    + "wrt radial coordinate and toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "B_r", "B_z", "B_rz", "e_zeta", "e_zeta_r", "e_zeta_z", "e_zeta_rz"],
)
def _B_sub_zeta_rz(params, transforms, profiles, data, **kwargs):
    data["B_zeta_rz"] = (
        dot(data["B_rz"], data["e_zeta"])
        + dot(data["B_r"], data["e_zeta_z"])
        + dot(data["B_z"], data["e_zeta_r"])
        + dot(data["B"], data["e_zeta_rz"])
    )
    return data


@register_compute_fun(
    name="|B|^2",
    label="|\\mathbf{B}|^{2}",
    units="T^2",
    units_long="Tesla squared",
    description="Magnitude of magnetic field, squared",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B"],
)
def _B_mag2(params, transforms, profiles, data, **kwargs):
    data["|B|^2"] = dot(data["B"], data["B"])
    return data


@register_compute_fun(
    name="|B|",
    label="|\\mathbf{B}|",
    units="T",
    units_long="Tesla",
    description="Magnitude of magnetic field",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|^2"],
)
def _B_mag(params, transforms, profiles, data, **kwargs):
    data["|B|"] = jnp.sqrt(data["|B|^2"])
    return data


@register_compute_fun(
    name="|B|_r",
    label="\\partial_{\\rho} |\\mathbf{B}|",
    units="T",
    units_long="Tesla",
    description="Magnitude of magnetic field, derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B_theta",
        "B_zeta",
        "B^theta_r",
        "B^zeta_r",
        "B_theta_r",
        "B_zeta_r",
        "|B|",
    ],
)
def _B_mag_r(params, transforms, profiles, data, **kwargs):
    data["|B|_r"] = (
        data["B^theta_r"] * data["B_theta"]
        + data["B^theta"] * data["B_theta_r"]
        + data["B^zeta_r"] * data["B_zeta"]
        + data["B^zeta"] * data["B_zeta_r"]
    ) / (2 * data["|B|"])
    return data


@register_compute_fun(
    name="|B|_t",
    label="\\partial_{\\theta} |\\mathbf{B}|",
    units="T",
    units_long="Tesla",
    description="Magnitude of magnetic field, derivative wrt poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B_theta",
        "B_zeta",
        "B^theta_t",
        "B^zeta_t",
        "B_theta_t",
        "B_zeta_t",
        "|B|",
    ],
)
def _B_mag_t(params, transforms, profiles, data, **kwargs):
    data["|B|_t"] = (
        data["B^theta_t"] * data["B_theta"]
        + data["B^theta"] * data["B_theta_t"]
        + data["B^zeta_t"] * data["B_zeta"]
        + data["B^zeta"] * data["B_zeta_t"]
    ) / (2 * data["|B|"])
    return data


@register_compute_fun(
    name="|B|_z",
    label="\\partial_{\\zeta} |\\mathbf{B}|",
    units="T",
    units_long="Tesla",
    description="Magnitude of magnetic field, derivative wrt toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B_theta",
        "B_zeta",
        "B^theta_z",
        "B^zeta_z",
        "B_theta_z",
        "B_zeta_z",
        "|B|",
    ],
)
def _B_mag_z(params, transforms, profiles, data, **kwargs):
    data["|B|_z"] = (
        data["B^theta_z"] * data["B_theta"]
        + data["B^theta"] * data["B_theta_z"]
        + data["B^zeta_z"] * data["B_zeta"]
        + data["B^zeta"] * data["B_zeta_z"]
    ) / (2 * data["|B|"])
    return data


@register_compute_fun(
    name="|B|_rr",
    label="\\partial_{\\rho\\rho} |\\mathbf{B}|",
    units="T",
    units_long="Tesla",
    description="Magnitude of magnetic field, second derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B_theta",
        "B_zeta",
        "B^theta_r",
        "B^zeta_r",
        "B_theta_r",
        "B_zeta_r",
        "B^theta_rr",
        "B^zeta_rr",
        "B_theta_rr",
        "B_zeta_rr",
        "|B|",
        "|B|_r",
    ],
)
def _B_mag_rr(params, transforms, profiles, data, **kwargs):
    data["|B|_rr"] = (
        data["B^theta_rr"] * data["B_theta"]
        + 2 * data["B^theta_r"] * data["B_theta_r"]
        + data["B^theta"] * data["B_theta_rr"]
        + data["B^zeta_rr"] * data["B_zeta"]
        + 2 * data["B^zeta_r"] * data["B_zeta_r"]
        + data["B^zeta"] * data["B_zeta_rr"]
    ) / (2 * data["|B|"]) - data["|B|_r"] ** 2 / data["|B|"]
    return data


@register_compute_fun(
    name="|B|_tt",
    label="\\partial_{\\theta\\theta} |\\mathbf{B}|",
    units="T",
    units_long="Tesla",
    description="Magnitude of magnetic field, second derivative wrt poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B_theta",
        "B_zeta",
        "B^theta_t",
        "B^zeta_t",
        "B_theta_t",
        "B_zeta_t",
        "B^theta_tt",
        "B^zeta_tt",
        "B_theta_tt",
        "B_zeta_tt",
        "|B|",
        "|B|_t",
    ],
)
def _B_mag_tt(params, transforms, profiles, data, **kwargs):
    data["|B|_tt"] = (
        data["B^theta_tt"] * data["B_theta"]
        + 2 * data["B^theta_t"] * data["B_theta_t"]
        + data["B^theta"] * data["B_theta_tt"]
        + data["B^zeta_tt"] * data["B_zeta"]
        + 2 * data["B^zeta_t"] * data["B_zeta_t"]
        + data["B^zeta"] * data["B_zeta_tt"]
    ) / (2 * data["|B|"]) - data["|B|_t"] ** 2 / data["|B|"]
    return data


@register_compute_fun(
    name="|B|_zz",
    label="\\partial_{\\zeta\\zeta} |\\mathbf{B}|",
    units="T",
    units_long="Tesla",
    description="Magnitude of magnetic field, second derivative wrt toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B_theta",
        "B_zeta",
        "B^theta_z",
        "B^zeta_z",
        "B_theta_z",
        "B_zeta_z",
        "B^theta_zz",
        "B^zeta_zz",
        "B_theta_zz",
        "B_zeta_zz",
        "|B|",
        "|B|_z",
    ],
)
def _B_mag_zz(params, transforms, profiles, data, **kwargs):
    data["|B|_zz"] = (
        data["B^theta_zz"] * data["B_theta"]
        + 2 * data["B^theta_z"] * data["B_theta_z"]
        + data["B^theta"] * data["B_theta_zz"]
        + data["B^zeta_zz"] * data["B_zeta"]
        + 2 * data["B^zeta_z"] * data["B_zeta_z"]
        + data["B^zeta"] * data["B_zeta_zz"]
    ) / (2 * data["|B|"]) - data["|B|_z"] ** 2 / data["|B|"]
    return data


@register_compute_fun(
    name="|B|_rt",
    label="\\partial_{\\rho\\theta} |\\mathbf{B}|",
    units="T",
    units_long="Tesla",
    description="Magnitude of magnetic field, derivative wrt radial coordinate and "
    + "poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B_theta",
        "B_zeta",
        "B^theta_r",
        "B^zeta_r",
        "B_theta_r",
        "B_zeta_r",
        "B^theta_t",
        "B^zeta_t",
        "B_theta_t",
        "B_zeta_t",
        "B^theta_rt",
        "B^zeta_rt",
        "B_theta_rt",
        "B_zeta_rt",
        "|B|",
        "|B|_r",
        "|B|_t",
    ],
)
def _B_mag_rt(params, transforms, profiles, data, **kwargs):
    data["|B|_rt"] = (
        data["B^theta_rt"] * data["B_theta"]
        + data["B^theta_r"] * data["B_theta_t"]
        + data["B^theta_t"] * data["B_theta_r"]
        + data["B^theta"] * data["B_theta_rt"]
        + data["B^zeta_rt"] * data["B_zeta"]
        + data["B^zeta_r"] * data["B_zeta_t"]
        + data["B^zeta_t"] * data["B_zeta_r"]
        + data["B^zeta"] * data["B_zeta_rt"]
    ) / (2 * data["|B|"]) - data["|B|_r"] * data["|B|_t"] / data["|B|"]
    return data


@register_compute_fun(
    name="|B|_tz",
    label="\\partial_{\\theta\\zeta} |\\mathbf{B}|",
    units="T",
    units_long="Tesla",
    description="Magnitude of magnetic field, derivative wrt poloidal and "
    + "toroidal angles",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B_theta",
        "B_zeta",
        "B^theta_t",
        "B^zeta_t",
        "B_theta_t",
        "B_zeta_t",
        "B^theta_z",
        "B^zeta_z",
        "B_theta_z",
        "B_zeta_z",
        "B^theta_tz",
        "B^zeta_tz",
        "B_theta_tz",
        "B_zeta_tz",
        "|B|",
        "|B|_t",
        "|B|_z",
    ],
)
def _B_mag_tz(params, transforms, profiles, data, **kwargs):
    data["|B|_tz"] = (
        data["B^theta_tz"] * data["B_theta"]
        + data["B^theta_t"] * data["B_theta_z"]
        + data["B^theta_z"] * data["B_theta_t"]
        + data["B^theta"] * data["B_theta_tz"]
        + data["B^zeta_tz"] * data["B_zeta"]
        + data["B^zeta_t"] * data["B_zeta_z"]
        + data["B^zeta_z"] * data["B_zeta_t"]
        + data["B^zeta"] * data["B_zeta_tz"]
    ) / (2 * data["|B|"]) - data["|B|_t"] * data["|B|_z"] / data["|B|"]
    return data


@register_compute_fun(
    name="|B|_rz",
    label="\\partial_{\\rho\\zeta} |\\mathbf{B}|",
    units="T",
    units_long="Tesla",
    description="Magnitude of magnetic field, derivative wrt radial coordinate and "
    + "toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B_theta",
        "B_zeta",
        "B^theta_r",
        "B^zeta_r",
        "B_theta_r",
        "B_zeta_r",
        "B^theta_z",
        "B^zeta_z",
        "B_theta_z",
        "B_zeta_z",
        "B^theta_rz",
        "B^zeta_rz",
        "B_theta_rz",
        "B_zeta_rz",
        "|B|",
        "|B|_r",
        "|B|_z",
    ],
)
def _B_mag_rz(params, transforms, profiles, data, **kwargs):
    data["|B|_rz"] = (
        data["B^theta_rz"] * data["B_theta"]
        + data["B^theta_r"] * data["B_theta_z"]
        + data["B^theta_z"] * data["B_theta_r"]
        + data["B^theta"] * data["B_theta_rz"]
        + data["B^zeta_rz"] * data["B_zeta"]
        + data["B^zeta_r"] * data["B_zeta_z"]
        + data["B^zeta_z"] * data["B_zeta_r"]
        + data["B^zeta"] * data["B_zeta_rz"]
    ) / (2 * data["|B|"]) - data["|B|_r"] * data["|B|_z"] / data["|B|"]
    return data


@register_compute_fun(
    name="grad(|B|)",
    label="\\nabla |\\mathbf{B}|",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Gradient of magnetic field magnitude",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|_r", "|B|_t", "|B|_z", "e^rho", "e^theta*sqrt(g)", "e^zeta", "sqrt(g)"],
    axis_limit_data=["|B|_rt", "sqrt(g)_r"],
)
def _grad_B(params, transforms, profiles, data, **kwargs):
    data["grad(|B|)"] = (
        data["|B|_r"] * data["e^rho"].T
        + transforms["grid"].replace_at_axis(
            data["|B|_t"] / data["sqrt(g)"], lambda: data["|B|_rt"] / data["sqrt(g)_r"]
        )
        * data["e^theta*sqrt(g)"].T
        + data["|B|_z"] * data["e^zeta"].T
    ).T
    return data


@register_compute_fun(
    name="<|B|>_vol",
    label="\\langle |B| \\rangle_{vol}",
    units="T",
    units_long="Tesla",
    description="Volume average magnetic field",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["sqrt(g)", "|B|", "V"],
)
def _B_vol(params, transforms, profiles, data, **kwargs):
    data["<|B|>_vol"] = (
        jnp.sum(data["|B|"] * data["sqrt(g)"] * transforms["grid"].weights) / data["V"]
    )
    return data


@register_compute_fun(
    name="<|B|>_rms",
    label="\\langle |B| \\rangle_{rms}",
    units="T",
    units_long="Tesla",
    description="Volume average magnetic field, root mean square",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["sqrt(g)", "|B|", "V"],
)
def _B_rms(params, transforms, profiles, data, **kwargs):
    data["<|B|>_rms"] = jnp.sqrt(
        jnp.sum(data["|B|"] ** 2 * data["sqrt(g)"] * transforms["grid"].weights)
        / data["V"]
    )
    return data


@register_compute_fun(
    name="<|B|>",
    label="\\langle |B| \\rangle",
    units="T",
    units_long="Tesla",
    description="Flux surface average magnetic field",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["sqrt(g)", "|B|"],
    axis_limit_data=["sqrt(g)_r"],
)
def _B_fsa(params, transforms, profiles, data, **kwargs):
    data["<|B|>"] = surface_averages(
        transforms["grid"],
        data["|B|"],
        sqrt_g=transforms["grid"].replace_at_axis(
            data["sqrt(g)"], lambda: data["sqrt(g)_r"], copy=True
        ),
    )
    return data


@register_compute_fun(
    name="<|B|^2>",
    label="\\langle |B|^2 \\rangle",
    units="T^2",
    units_long="Tesla squared",
    description="Flux surface average magnetic field squared",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["sqrt(g)", "|B|^2"],
    axis_limit_data=["sqrt(g)_r"],
)
def _B2_fsa(params, transforms, profiles, data, **kwargs):
    data["<|B|^2>"] = surface_averages(
        transforms["grid"],
        data["|B|^2"],
        sqrt_g=transforms["grid"].replace_at_axis(
            data["sqrt(g)"], lambda: data["sqrt(g)_r"], copy=True
        ),
    )
    return data


@register_compute_fun(
    name="<1/|B|>",
    label="\\langle 1/|B| \\rangle",
    units="T^{-1}",
    units_long="1 / Tesla",
    description="Flux surface averaged inverse field strength",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["sqrt(g)", "|B|"],
    axis_limit_data=["sqrt(g)_r"],
)
def _1_over_B_fsa(params, transforms, profiles, data, **kwargs):
    data["<1/|B|>"] = surface_averages(
        transforms["grid"],
        1 / data["|B|"],
        sqrt_g=transforms["grid"].replace_at_axis(
            data["sqrt(g)"], lambda: data["sqrt(g)_r"], copy=True
        ),
    )
    return data


@register_compute_fun(
    name="<|B|^2>_r",
    label="\\partial_{\\rho} \\langle |B|^2 \\rangle",
    units="T^2",
    units_long="Tesla squared",
    description="Flux surface average magnetic field squared, radial derivative",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["sqrt(g)", "sqrt(g)_r", "B", "B_r", "|B|^2", "V_r(r)", "V_rr(r)"],
    axis_limit_data=["sqrt(g)_rr", "V_rrr(r)"],
)
def _B2_fsa_r(params, transforms, profiles, data, **kwargs):
    integrate = surface_integrals_map(transforms["grid"])
    B2_r = 2 * dot(data["B"], data["B_r"])
    num = integrate(data["sqrt(g)"] * data["|B|^2"])
    num_r = integrate(data["sqrt(g)_r"] * data["|B|^2"] + data["sqrt(g)"] * B2_r)
    data["<|B|^2>_r"] = transforms["grid"].replace_at_axis(
        (num_r * data["V_r(r)"] - num * data["V_rr(r)"]) / data["V_r(r)"] ** 2,
        lambda: (
            integrate(data["sqrt(g)_rr"] * data["|B|^2"] + 2 * data["sqrt(g)_r"] * B2_r)
            * data["V_rr(r)"]
            - num_r * data["V_rrr(r)"]
        )
        / (2 * data["V_rr(r)"] ** 2),
    )
    return data


@register_compute_fun(
    name="grad(|B|^2)_rho",
    label="(\\nabla |B|^{2})_{\\rho}",
    units="T^{2}",
    units_long="Tesla squared",
    description="Covariant radial component of magnetic pressure gradient",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B^theta_r",
        "B^zeta_r",
        "B_theta",
        "B_zeta",
        "B_theta_r",
        "B_zeta_r",
    ],
)
def _gradB2_rho(params, transforms, profiles, data, **kwargs):
    data["grad(|B|^2)_rho"] = (
        data["B^theta"] * data["B_theta_r"]
        + data["B_theta"] * data["B^theta_r"]
        + data["B^zeta"] * data["B_zeta_r"]
        + data["B_zeta"] * data["B^zeta_r"]
    )
    return data


@register_compute_fun(
    name="grad(|B|^2)_theta",
    label="(\\nabla |B|^{2})_{\\theta}",
    units="T^{2}",
    units_long="Tesla squared",
    description="Covariant poloidal component of magnetic pressure gradient",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B^theta_t",
        "B^zeta_t",
        "B_theta",
        "B_zeta",
        "B_theta_t",
        "B_zeta_t",
    ],
)
def _gradB2_theta(params, transforms, profiles, data, **kwargs):
    data["grad(|B|^2)_theta"] = (
        data["B^theta"] * data["B_theta_t"]
        + data["B_theta"] * data["B^theta_t"]
        + data["B^zeta"] * data["B_zeta_t"]
        + data["B_zeta"] * data["B^zeta_t"]
    )
    return data


@register_compute_fun(
    name="grad(|B|^2)_zeta",
    label="(\\nabla |B|^{2})_{\\zeta}",
    units="T^{2}",
    units_long="Tesla squared",
    description="Covariant toroidal component of magnetic pressure gradient",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B^theta_z",
        "B^zeta_z",
        "B_theta",
        "B_zeta",
        "B_theta_z",
        "B_zeta_z",
    ],
)
def _gradB2_zeta(params, transforms, profiles, data, **kwargs):
    data["grad(|B|^2)_zeta"] = (
        data["B^theta"] * data["B_theta_z"]
        + data["B_theta"] * data["B^theta_z"]
        + data["B^zeta"] * data["B_zeta_z"]
        + data["B_zeta"] * data["B^zeta_z"]
    )
    return data


@register_compute_fun(
    name="grad(|B|^2)",
    label="\\nabla |B|^{2}",
    units="T^{2} \\cdot m^{-1}",
    units_long="Tesla squared / meters",
    description="Magnetic pressure gradient",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "grad(|B|)"],
)
def _gradB2(params, transforms, profiles, data, **kwargs):
    data["grad(|B|^2)"] = 2 * (data["|B|"] * data["grad(|B|)"].T).T
    return data


@register_compute_fun(
    name="|grad(|B|^2)|/2mu0",
    label="|\\nabla |B|^{2}/(2\\mu_0)|",
    units="N \\cdot m^{-3}",
    units_long="Newton / cubic meter",
    description="Magnitude of magnetic pressure gradient",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["grad(|B|^2)"],
)
def _gradB2mag(params, transforms, profiles, data, **kwargs):
    data["|grad(|B|^2)|/2mu0"] = jnp.linalg.norm(data["grad(|B|^2)"], axis=-1) / (
        2 * mu_0
    )
    return data


@register_compute_fun(
    name="<|grad(|B|^2)|/2mu0>_vol",
    label="\\langle |\\nabla |B|^{2}/(2\\mu_0)| \\rangle_{vol}",
    units="N \\cdot m^{-3}",
    units_long="Newtons per cubic meter",
    description="Volume average of magnitude of magnetic pressure gradient",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["|grad(|B|^2)|/2mu0", "sqrt(g)", "V"],
)
def _gradB2mag_vol(params, transforms, profiles, data, **kwargs):
    data["<|grad(|B|^2)|/2mu0>_vol"] = (
        jnp.sum(
            data["|grad(|B|^2)|/2mu0"] * data["sqrt(g)"] * transforms["grid"].weights
        )
        / data["V"]
    )
    return data


@register_compute_fun(
    name="(curl(B)xB)_rho",
    label="((\\nabla \\times \\mathbf{B}) \\times \\mathbf{B})_{\\rho}",
    units="T^{2}",
    units_long="Tesla squared",
    description="Covariant radial component of Lorentz force",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B^theta", "B^zeta", "B_rho_z", "B_zeta_r", "B_theta_r", "B_rho_t"],
)
def _curl_B_x_B_rho(params, transforms, profiles, data, **kwargs):
    data["(curl(B)xB)_rho"] = data["B^zeta"] * (
        data["B_rho_z"] - data["B_zeta_r"]
    ) - data["B^theta"] * (data["B_theta_r"] - data["B_rho_t"])
    return data


@register_compute_fun(
    name="(curl(B)xB)_theta",
    label="((\\nabla \\times \\mathbf{B}) \\times \\mathbf{B})_{\\theta}",
    units="T^{2}",
    units_long="Tesla squared",
    description="Covariant poloidal component of Lorentz force",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B^zeta", "B_zeta_t", "B_theta_z"],
)
def _curl_B_x_B_theta(params, transforms, profiles, data, **kwargs):
    data["(curl(B)xB)_theta"] = -data["B^zeta"] * (data["B_zeta_t"] - data["B_theta_z"])
    return data


@register_compute_fun(
    name="(curl(B)xB)_zeta",
    label="((\\nabla \\times \\mathbf{B}) \\times \\mathbf{B})_{\\zeta}",
    units="T^{2}",
    units_long="Tesla squared",
    description="Covariant toroidal component of Lorentz force",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B^theta", "B_zeta_t", "B_theta_z"],
)
def _curl_B_x_B_zeta(params, transforms, profiles, data, **kwargs):
    data["(curl(B)xB)_zeta"] = data["B^theta"] * (data["B_zeta_t"] - data["B_theta_z"])
    return data


@register_compute_fun(
    name="curl(B)xB",
    label="(\\nabla \\times \\mathbf{B}) \\times \\mathbf{B}",
    units="T^{2} \\cdot m^{-1}",
    units_long="Tesla squared / meters",
    description="Lorentz force",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "(curl(B)xB)_rho",
        "B^zeta",
        "J^rho",
        "(curl(B)xB)_zeta",
        "e^rho",
        "e^theta*sqrt(g)",
        "e^zeta",
    ],
)
def _curl_B_x_B(params, transforms, profiles, data, **kwargs):
    # (curl(B)xB)_theta e^theta refactored to resolve indeterminacy at axis.
    data["curl(B)xB"] = (
        data["(curl(B)xB)_rho"] * data["e^rho"].T
        - mu_0 * data["B^zeta"] * data["J^rho"] * data["e^theta*sqrt(g)"].T
        + data["(curl(B)xB)_zeta"] * data["e^zeta"].T
    ).T
    return data


@register_compute_fun(
    name="(B*grad)B",
    label="(\\mathbf{B} \\cdot \\nabla) \\mathbf{B}",
    units="T^{2} \\cdot m^{-1}",
    units_long="Tesla squared / meters",
    description="Magnetic tension",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["curl(B)xB", "grad(|B|^2)"],
)
def _B_dot_grad_B(params, transforms, profiles, data, **kwargs):
    data["(B*grad)B"] = data["curl(B)xB"] + data["grad(|B|^2)"] / 2
    return data


@register_compute_fun(
    name="((B*grad)B)_rho",
    label="((\\mathbf{B} \\cdot \\nabla) \\mathbf{B})_{\\rho}",
    units="T^{2}",
    units_long="Tesla squared",
    description="Covariant radial component of magnetic tension",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["(B*grad)B", "e_rho"],
)
def _B_dot_grad_B_rho(params, transforms, profiles, data, **kwargs):
    data["((B*grad)B)_rho"] = dot(data["(B*grad)B"], data["e_rho"])
    return data


@register_compute_fun(
    name="((B*grad)B)_theta",
    label="((\\mathbf{B} \\cdot \\nabla) \\mathbf{B})_{\\theta}",
    units="T^{2}",
    units_long="Tesla squared",
    description="Covariant poloidal component of magnetic tension",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["(B*grad)B", "e_theta"],
)
def _B_dot_grad_B_theta(params, transforms, profiles, data, **kwargs):
    data["((B*grad)B)_theta"] = dot(data["(B*grad)B"], data["e_theta"])
    return data


@register_compute_fun(
    name="((B*grad)B)_zeta",
    label="((\\mathbf{B} \\cdot \\nabla) \\mathbf{B})_{\\zeta}",
    units="T^{2}",
    units_long="Tesla squared",
    description="Covariant toroidal component of magnetic tension",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["(B*grad)B", "e_zeta"],
)
def _B_dot_grad_B_zeta(params, transforms, profiles, data, **kwargs):
    data["((B*grad)B)_zeta"] = dot(data["(B*grad)B"], data["e_zeta"])
    return data


@register_compute_fun(
    name="|(B*grad)B|",
    label="|(\\mathbf{B} \\cdot \\nabla) \\mathbf{B}|",
    units="T^2 \\cdot m^{-1}",
    units_long="Tesla squared / meters",
    description="Magnitude of magnetic tension",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["(B*grad)B"],
)
def _B_dot_grad_B_mag(params, transforms, profiles, data, **kwargs):
    data["|(B*grad)B|"] = jnp.linalg.norm(data["(B*grad)B"], axis=-1)
    return data


@register_compute_fun(
    name="<|(B*grad)B|>_vol",
    label="\\langle |(\\mathbf{B} \\cdot \\nabla) \\mathbf{B}| \\rangle_{vol}",
    units="T^{2} \\cdot m^{-1}",
    units_long="Tesla squared / meters",
    description="Volume average magnetic tension magnitude",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["|(B*grad)B|", "sqrt(g)", "V"],
)
def _B_dot_grad_B_mag_vol(params, transforms, profiles, data, **kwargs):
    data["<|(B*grad)B|>_vol"] = (
        jnp.sum(data["|(B*grad)B|"] * data["sqrt(g)"] * transforms["grid"].weights)
        / data["V"]
    )
    return data


@register_compute_fun(
    name="B*grad(|B|)",
    label="\\mathbf{B} \\cdot \\nabla B",
    units="T^2 \\cdot m^{-1}",
    units_long="Tesla squared / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B^theta", "B^zeta", "|B|_t", "|B|_z"],
)
def _B_dot_gradB(params, transforms, profiles, data, **kwargs):
    data["B*grad(|B|)"] = (
        data["B^theta"] * data["|B|_t"] + data["B^zeta"] * data["|B|_z"]
    )
    return data


@register_compute_fun(
    name="(B*grad(|B|))_r",
    label="\\partial_{\\theta} (\\mathbf{B} \\cdot \\nabla B)",
    units="T^2 \\cdot m^{-1}",
    units_long="Tesla squared / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "|B|_t",
        "|B|_z",
        "B^theta_r",
        "B^zeta_r",
        "|B|_rt",
        "|B|_rz",
    ],
)
def _B_dot_gradB_r(params, transforms, profiles, data, **kwargs):
    data["(B*grad(|B|))_r"] = (
        data["B^theta_r"] * data["|B|_t"]
        + data["B^theta"] * data["|B|_rt"]
        + data["B^zeta_r"] * data["|B|_z"]
        + data["B^zeta"] * data["|B|_rz"]
    )
    return data


@register_compute_fun(
    name="(B*grad(|B|))_t",
    label="\\partial_{\\theta} (\\mathbf{B} \\cdot \\nabla B)",
    units="T^2 \\cdot m^{-1}",
    units_long="Tesla squared / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B^theta_t",
        "B^zeta_t",
        "|B|_t",
        "|B|_z",
        "|B|_tt",
        "|B|_tz",
    ],
)
def _B_dot_gradB_t(params, transforms, profiles, data, **kwargs):
    data["(B*grad(|B|))_t"] = (
        data["B^theta_t"] * data["|B|_t"]
        + data["B^zeta_t"] * data["|B|_z"]
        + data["B^theta"] * data["|B|_tt"]
        + data["B^zeta"] * data["|B|_tz"]
    )
    return data


@register_compute_fun(
    name="(B*grad(|B|))_z",
    label="\\partial_{\\zeta} (\\mathbf{B} \\cdot \\nabla B)",
    units="T^2 \\cdot m^{-1}",
    units_long="Tesla squared / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B^theta",
        "B^zeta",
        "B^theta_z",
        "B^zeta_z",
        "|B|_t",
        "|B|_z",
        "|B|_tz",
        "|B|_zz",
    ],
)
def _B_dot_gradB_z(params, transforms, profiles, data, **kwargs):
    data["(B*grad(|B|))_z"] = (
        data["B^theta_z"] * data["|B|_t"]
        + data["B^zeta_z"] * data["|B|_z"]
        + data["B^theta"] * data["|B|_tz"]
        + data["B^zeta"] * data["|B|_zz"]
    )
    return data


@register_compute_fun(
    name="max_tz |B|",
    label="\\max_{\\theta \\zeta} |B|",
    units="T",
    units_long="Tesla",
    description="Maximum field strength on each flux surface",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|B|"],
)
def _max_tz_modB(params, transforms, profiles, data, **kwargs):
    data["max_tz |B|"] = surface_max(transforms["grid"], data["|B|"])
    return data


@register_compute_fun(
    name="min_tz |B|",
    label="\\min_{\\theta \\zeta} |B|",
    units="T",
    units_long="Tesla",
    description="Minimum field strength on each flux surface",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|B|"],
)
def _min_tz_modB(params, transforms, profiles, data, **kwargs):
    data["min_tz |B|"] = surface_min(transforms["grid"], data["|B|"])
    return data


@register_compute_fun(
    name="effective r/R0",
    label="(r / R_0)_{\\mathrm{effective}}",
    units="~",
    units_long="None",
    description="Effective local inverse aspect ratio, based on max and min |B|",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["max_tz |B|", "min_tz |B|"],
)
def _effective_r_over_R0(params, transforms, profiles, data, **kwargs):
    """Compute an effective local inverse aspect ratio.

    This effective local inverse aspect ratio epsilon is defined by
    Bmax / Bmin = (1 + ε) / (1 − ε).

    This definition is motivated by the fact that this formula would
    be true in the case of circular cross-section surfaces in
    axisymmetry with B ∝ 1/R and R = (1 + ε cos θ) R₀.
    """
    w = data["max_tz |B|"] / data["min_tz |B|"]
    data["effective r/R0"] = (w - 1) / (w + 1)
    return data


@register_compute_fun(
    name="kappa",
    label="\\kappa",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Curvature vector of magnetic field lines",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["J", "|B|", "b", "grad(|B|)"],
)
def _kappa(params, transforms, profiles, data, **kwargs):
    data["kappa"] = -(
        cross(data["b"], mu_0 * data["J"] + cross(data["b"], data["grad(|B|)"])).T
        / data["|B|"]
    ).T
    return data


@register_compute_fun(
    name="kappa_n",
    label="\\kappa_n",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Normal curvature vector of magnetic field lines",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["kappa", "n_rho"],
)
def _kappa_n(params, transforms, profiles, data, **kwargs):
    data["kappa_n"] = dot(data["kappa"], data["n_rho"])
    return data


@register_compute_fun(
    name="kappa_g",
    label="\\kappa_g",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Geodesic curvature vector of magnetic field lines",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["kappa", "n_rho", "b"],
)
def _kappa_g(params, transforms, profiles, data, **kwargs):
    data["kappa_g"] = dot(data["kappa"], cross(data["n_rho"], data["b"]))
    return data


@register_compute_fun(
    name="grad(B)",
    label="\\nabla \\mathbf{B}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="Gradient of magnetic field vector",
    dim=(3, 3),
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B_r", "B_t", "B_z", "e^rho", "e^theta*sqrt(g)", "e^zeta", "sqrt(g)"],
    axis_limit_data=["B_rt", "sqrt(g)_r"],
)
def _grad_B_vec(params, transforms, profiles, data, **kwargs):
    B_t_over_sqrt_g = transforms["grid"].replace_at_axis(
        (data["B_t"].T / data["sqrt(g)"]).T,
        lambda: (data["B_rt"].T / data["sqrt(g)_r"]).T,
    )
    data["grad(B)"] = (
        (data["B_r"][:, jnp.newaxis, :] * data["e^rho"][:, :, jnp.newaxis])
        + (
            B_t_over_sqrt_g[:, jnp.newaxis, :]
            * data["e^theta*sqrt(g)"][:, :, jnp.newaxis]
        )
        + (data["B_z"][:, jnp.newaxis, :] * data["e^zeta"][:, :, jnp.newaxis])
    )
    return data


@register_compute_fun(
    name="|grad(B)|",
    label="|\\nabla \\mathbf{B}|",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="Frobenius norm of gradient of magnetic field vector",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["grad(B)"],
)
def _grad_B_vec_fro(params, transforms, profiles, data, **kwargs):
    data["|grad(B)|"] = jnp.linalg.norm(data["grad(B)"], axis=(1, 2), ord="fro")
    return data


@register_compute_fun(
    name="L_grad(B)",
    label="L_{\\nabla \\mathbf{B}} = \\frac{\\sqrt{2}|B|}{|\\nabla \\mathbf{B}|}",
    units="m",
    units_long="meters",
    description="Magnetic field length scale based on Frobenius norm of gradient "
    + "of magnetic field vector",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|grad(B)|", "|B|"],
)
def _L_grad_B(params, transforms, profiles, data, **kwargs):
    data["L_grad(B)"] = jnp.sqrt(2) * data["|B|"] / data["|grad(B)|"]
    return data
