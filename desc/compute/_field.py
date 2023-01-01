"""Compute functions for magnetic field quantities."""

from scipy.constants import mu_0

from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import dot, surface_averages, surface_integrals


@register_compute_fun(
    name="B0",
    label="\\psi' / \\sqrt{g}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "sqrt(g)"],
)
def _B0(params, transforms, profiles, data, **kwargs):
    data["B0"] = data["psi_r"] / data["sqrt(g)"]
    return data


@register_compute_fun(
    name="B^rho",
    label="B^{\\rho}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant radial component of magnetic field",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["0"],
)
def _B_sup_rho(params, transforms, profiles, data, **kwargs):
    data["B^rho"] = data["0"]
    return data


@register_compute_fun(
    name="B^theta",
    label="B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant poloidal component of magnetic field",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "iota", "lambda_z"],
)
def _B_sup_theta(params, transforms, profiles, data, **kwargs):
    data["B^theta"] = data["B0"] * (data["iota"] - data["lambda_z"])
    return data


@register_compute_fun(
    name="B^zeta",
    label="B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant toroidal component of magnetic field",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "lambda_t"],
)
def _B_sup_zeta(params, transforms, profiles, data, **kwargs):
    data["B^zeta"] = data["B0"] * (1 + data["lambda_t"])
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
    label="\\psi'' / \\sqrt{g} - \\psi' \\partial_{\\rho} \\sqrt{g} / g",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "psi_rr", "sqrt(g)", "sqrt(g)_r"],
)
def _B0_r(params, transforms, profiles, data, **kwargs):
    data["B0_r"] = (
        data["psi_rr"] / data["sqrt(g)"]
        - data["psi_r"] * data["sqrt(g)_r"] / data["sqrt(g)"] ** 2
    )
    return data


@register_compute_fun(
    name="B^theta_r",
    label="\\partial_{\\rho} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant poloidal component of magnetic field, derivative wrt "
    + "radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0_r", "iota", "lambda_z", "B0", "iota_r", "lambda_rz"],
)
def _B_sup_theta_r(params, transforms, profiles, data, **kwargs):
    data["B^theta_r"] = data["B0_r"] * (data["iota"] - data["lambda_z"]) + data[
        "B0"
    ] * (data["iota_r"] - data["lambda_rz"])
    return data


@register_compute_fun(
    name="B^zeta_r",
    label="\\partial_{\\rho} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant toroidal component of magnetic field, derivative wrt "
    + "radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0_r", "lambda_t", "B0", "lambda_rt"],
)
def _B_sup_zeta_r(params, transforms, profiles, data, **kwargs):
    data["B^zeta_r"] = (
        data["B0_r"] * (1 + data["lambda_t"]) + data["B0"] * data["lambda_rt"]
    )
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
    label="-\\psi' \\partial_{\\theta} \\sqrt{g} / g",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "sqrt(g)_t", "sqrt(g)"],
)
def _B0_t(params, transforms, profiles, data, **kwargs):
    data["B0_t"] = -data["psi_r"] * data["sqrt(g)_t"] / data["sqrt(g)"] ** 2
    return data


@register_compute_fun(
    name="B^theta_t",
    label="\\partial_{\\theta} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant poloidal component of magnetic field, derivative wrt "
    + "poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "B0_t", "iota", "lambda_z", "lambda_tz"],
)
def _B_sup_theta_t(params, transforms, profiles, data, **kwargs):
    data["B^theta_t"] = (
        data["B0_t"] * (data["iota"] - data["lambda_z"])
        - data["B0"] * data["lambda_tz"]
    )
    return data


@register_compute_fun(
    name="B^zeta_t",
    label="\\partial_{\\theta} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant toroidal component of magnetic field, derivative wrt "
    + "poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "B0_t", "lambda_t", "lambda_tt"],
)
def _B_sup_zeta_t(params, transforms, profiles, data, **kwargs):
    data["B^zeta_t"] = (
        data["B0_t"] * (1 + data["lambda_t"]) + data["B0"] * data["lambda_tt"]
    )
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
    label="-\\psi' \\partial_{\\zeta} \\sqrt{g} / g",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "sqrt(g)", "sqrt(g)_z"],
)
def _B0_z(params, transforms, profiles, data, **kwargs):
    data["B0_z"] = -data["psi_r"] * data["sqrt(g)_z"] / data["sqrt(g)"] ** 2
    return data


@register_compute_fun(
    name="B^theta_z",
    label="\\partial_{\\zeta} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant poloidal component of magnetic field, derivative wrt "
    + "toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "B0_z", "iota", "lambda_z", "lambda_zz"],
)
def _B_sup_theta_z(params, transforms, profiles, data, **kwargs):
    data["B^theta_z"] = (
        data["B0_z"] * (data["iota"] - data["lambda_z"])
        - data["B0"] * data["lambda_zz"]
    )
    return data


@register_compute_fun(
    name="B^zeta_z",
    label="\\partial_{\\zeta} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant toroidal component of magnetic field, derivative wrt "
    + "toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "B0_z", "lambda_t", "lambda_tz"],
)
def _B_sup_zeta_z(params, transforms, profiles, data, **kwargs):
    data["B^zeta_z"] = (
        data["B0_z"] * (1 + data["lambda_t"]) + data["B0"] * data["lambda_tz"]
    )
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
    name="B0_tt",
    label="-\\psi' \\partial_{\\theta\\theta} \\sqrt{g} / g + "
    + "2 \\psi' (\\partial_{\\theta} \\sqrt{g})^2 / (\\sqrt{g})^{3}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "sqrt(g)", "sqrt(g)_t", "sqrt(g)_tt"],
)
def _B0_tt(params, transforms, profiles, data, **kwargs):
    data["B0_tt"] = -(
        data["psi_r"]
        / data["sqrt(g)"] ** 2
        * (data["sqrt(g)_tt"] - 2 * data["sqrt(g)_t"] ** 2 / data["sqrt(g)"])
    )
    return data


@register_compute_fun(
    name="B^theta_tt",
    label="\\partial_{\\theta\\theta} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant poloidal component of magnetic field, second "
    + "derivative wrt poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "B0_t", "B0_tt", "iota", "lambda_z", "lambda_tz", "lambda_ttz"],
)
def _B_sup_theta_tt(params, transforms, profiles, data, **kwargs):
    data["B^theta_tt"] = (
        data["B0_tt"] * (data["iota"] - data["lambda_z"])
        - 2 * data["B0_t"] * data["lambda_tz"]
        - data["B0"] * data["lambda_ttz"]
    )
    return data


@register_compute_fun(
    name="B^zeta_tt",
    label="\\partial_{\\theta\\theta} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant toroidal component of magnetic field, second "
    + "derivative wrt poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "B0_t", "B0_tt", "lambda_t", "lambda_tt", "lambda_ttt"],
)
def _B_sup_zeta_tt(params, transforms, profiles, data, **kwargs):
    data["B^zeta_tt"] = (
        data["B0_tt"] * (1 + data["lambda_t"])
        + 2 * data["B0_t"] * data["lambda_tt"]
        + data["B0"] * data["lambda_ttt"]
    )
    return data


@register_compute_fun(
    name="B0_zz",
    label="-\\psi' \\partial_{\\zeta\\zeta} \\sqrt{g} / g + "
    + "2 \\psi' (\\partial_{\\zeta} \\sqrt{g})^2 / (\\sqrt{g})^{3}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "sqrt(g)", "sqrt(g)_z", "sqrt(g)_zz"],
)
def _B0_zz(params, transforms, profiles, data, **kwargs):
    data["B0_zz"] = -(
        data["psi_r"]
        / data["sqrt(g)"] ** 2
        * (data["sqrt(g)_zz"] - 2 * data["sqrt(g)_z"] ** 2 / data["sqrt(g)"])
    )
    return data


@register_compute_fun(
    name="B^theta_zz",
    label="\\partial_{\\zeta\\zeta} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant poloidal component of magnetic field, second "
    + "derivative wrt toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "B0_z", "B0_zz", "iota", "lambda_z", "lambda_zz", "lambda_zzz"],
)
def _B_sup_theta_zz(params, transforms, profiles, data, **kwargs):
    data["B^theta_zz"] = (
        data["B0_zz"] * (data["iota"] - data["lambda_z"])
        - 2 * data["B0_z"] * data["lambda_zz"]
        - data["B0"] * data["lambda_zzz"]
    )
    return data


@register_compute_fun(
    name="B^zeta_zz",
    label="\\partial_{\\zeta\\zeta} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant toroidal component of magnetic field, second "
    + "derivative wrt toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B0", "B0_z", "B0_zz", "lambda_t", "lambda_tz", "lambda_tzz"],
)
def _B_sup_zeta_zz(params, transforms, profiles, data, **kwargs):
    data["B^zeta_zz"] = (
        data["B0_zz"] * (1 + data["lambda_t"])
        + 2 * data["B0_z"] * data["lambda_tz"]
        + data["B0"] * data["lambda_tzz"]
    )
    return data


@register_compute_fun(
    name="B0_tz",
    label="-\\psi' \\partial_{\\theta\\zeta} \\sqrt{g} / g + "
    + "2 \\psi' \\partial_{\\theta} \\sqrt{g} \\partial_{\\zeta} \\sqrt{g} / "
    + "(\\sqrt{g})^{3}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r", "sqrt(g)", "sqrt(g)_t", "sqrt(g)_z", "sqrt(g)_tz"],
)
def _B0_tz(params, transforms, profiles, data, **kwargs):
    data["B0_tz"] = -(
        data["psi_r"]
        / data["sqrt(g)"] ** 2
        * (
            data["sqrt(g)_tz"]
            - 2 * data["sqrt(g)_t"] * data["sqrt(g)_z"] / data["sqrt(g)"]
        )
    )
    return data


@register_compute_fun(
    name="B^theta_tz",
    label="\\partial_{\\theta\\zeta} B^{\\theta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant poloidal component of magnetic field, second "
    + "derivative wrt poloidal and toroidal angles",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_t",
        "B0_z",
        "B0_tz",
        "iota",
        "lambda_z",
        "lambda_zz",
        "lambda_tz",
        "lambda_tzz",
    ],
)
def _B_sup_theta_tz(params, transforms, profiles, data, **kwargs):
    data["B^theta_tz"] = (
        data["B0_tz"] * (data["iota"] - data["lambda_z"])
        - data["B0_t"] * data["lambda_zz"]
        - data["B0_z"] * data["lambda_tz"]
        - data["B0"] * data["lambda_tzz"]
    )
    return data


@register_compute_fun(
    name="B^zeta_tz",
    label="\\partial_{\\theta\\zeta} B^{\\zeta}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meters",
    description="Contravariant toroidal component of magnetic field, second "
    + "derivative wrt poloidal and toroidal angles",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "B0",
        "B0_t",
        "B0_z",
        "B0_tz",
        "lambda_t",
        "lambda_tt",
        "lambda_tz",
        "lambda_ttz",
    ],
)
def _B_sup_zeta_tz(params, transforms, profiles, data, **kwargs):
    data["B^zeta_tz"] = (
        data["B0_tz"] * (1 + data["lambda_t"])
        + data["B0_t"] * data["lambda_tz"]
        + data["B0_z"] * data["lambda_tt"]
        + data["B0"] * data["lambda_ttz"]
    )
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


# TODO: |B|_r
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
        "B^theta_t",
        "B^zeta_t",
        "g_tt",
        "g_tz",
        "g_zz",
        "e_theta",
        "e_zeta",
        "e_theta_t",
        "e_zeta_t",
        "|B|",
    ],
)
def _B_mag_t(params, transforms, profiles, data, **kwargs):
    data["|B|_t"] = (
        data["B^theta"]
        * (
            data["B^zeta_t"] * data["g_tz"]
            + data["B^theta_t"] * data["g_tt"]
            + data["B^theta"] * dot(data["e_theta_t"], data["e_theta"])
        )
        + data["B^zeta"]
        * (
            data["B^theta_t"] * data["g_tz"]
            + data["B^zeta_t"] * data["g_zz"]
            + data["B^zeta"] * dot(data["e_zeta_t"], data["e_zeta"])
        )
        + data["B^theta"]
        * data["B^zeta"]
        * (
            dot(data["e_theta_t"], data["e_zeta"])
            + dot(data["e_zeta_t"], data["e_theta"])
        )
    ) / data["|B|"]
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
        "B^theta_z",
        "B^zeta_z",
        "g_tt",
        "g_tz",
        "g_zz",
        "e_theta",
        "e_zeta",
        "e_theta_z",
        "e_zeta_z",
        "|B|",
    ],
)
def _B_mag_z(params, transforms, profiles, data, **kwargs):
    data["|B|_z"] = (
        data["B^theta"]
        * (
            data["B^zeta_z"] * data["g_tz"]
            + data["B^theta_z"] * data["g_tt"]
            + data["B^theta"] * dot(data["e_theta_z"], data["e_theta"])
        )
        + data["B^zeta"]
        * (
            data["B^theta_z"] * data["g_tz"]
            + data["B^zeta_z"] * data["g_zz"]
            + data["B^zeta"] * dot(data["e_zeta_z"], data["e_zeta"])
        )
        + data["B^theta"]
        * data["B^zeta"]
        * (
            dot(data["e_theta_z"], data["e_zeta"])
            + dot(data["e_zeta_z"], data["e_theta"])
        )
    ) / data["|B|"]
    return data


# TODO: |B|_rr
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
        "B^theta_t",
        "B^zeta_t",
        "B^theta_tt",
        "B^zeta_tt",
        "g_tt",
        "g_tz",
        "g_zz",
        "e_theta",
        "e_zeta",
        "e_theta_t",
        "e_zeta_t",
        "e_theta_tt",
        "e_zeta_tt",
        "|B|",
        "|B|_t",
    ],
)
def _B_mag_tt(params, transforms, profiles, data, **kwargs):
    data["|B|_tt"] = (
        data["B^theta_t"]
        * (
            data["B^zeta_t"] * data["g_tz"]
            + data["B^theta_t"] * data["g_tt"]
            + data["B^theta"] * dot(data["e_theta_t"], data["e_theta"])
        )
        + data["B^theta"]
        * (
            data["B^zeta_tt"] * data["g_tz"]
            + data["B^theta_tt"] * data["g_tt"]
            + data["B^theta_t"] * dot(data["e_theta_t"], data["e_theta"])
        )
        + data["B^theta"]
        * (
            data["B^zeta_t"]
            * (
                dot(data["e_theta_t"], data["e_zeta"])  # TODO: this is just g_tz_t
                + dot(data["e_theta"], data["e_zeta_t"])
            )
            + 2 * data["B^theta_t"] * dot(data["e_theta_t"], data["e_theta"])
            + data["B^theta"]
            * (
                dot(data["e_theta_tt"], data["e_theta"])
                + dot(data["e_theta_t"], data["e_theta_t"])
            )
        )
        + data["B^zeta_t"]
        * (
            data["B^theta_t"] * data["g_tz"]
            + data["B^zeta_t"] * data["g_zz"]
            + data["B^zeta"] * dot(data["e_zeta_t"], data["e_zeta"])
        )
        + data["B^zeta"]
        * (
            data["B^theta_tt"] * data["g_tz"]
            + data["B^zeta_tt"] * data["g_zz"]
            + data["B^zeta_t"] * dot(data["e_zeta_t"], data["e_zeta"])
        )
        + data["B^zeta"]
        * (
            data["B^theta_t"]
            * (
                dot(data["e_theta_t"], data["e_zeta"])
                + dot(data["e_theta"], data["e_zeta_t"])
            )
            + 2 * data["B^zeta_t"] * dot(data["e_zeta_t"], data["e_zeta"])
            + data["B^zeta"]
            * (
                dot(data["e_zeta_tt"], data["e_zeta"])
                + dot(data["e_zeta_t"], data["e_zeta_t"])
            )
        )
        + (data["B^theta_t"] * data["B^zeta"] + data["B^theta"] * data["B^zeta_t"])
        * (
            dot(data["e_theta_t"], data["e_zeta"])
            + dot(data["e_zeta_t"], data["e_theta"])
        )
        + data["B^theta"]
        * data["B^zeta"]
        * (
            dot(data["e_theta_tt"], data["e_zeta"])
            + dot(data["e_zeta_tt"], data["e_theta"])
            + 2 * dot(data["e_zeta_t"], data["e_theta_t"])
        )
    ) / data["|B|"] - data["|B|_t"] ** 2 / data["|B|"]
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
        "B^theta_z",
        "B^zeta_z",
        "B^theta_zz",
        "B^zeta_zz",
        "g_tt",
        "g_tz",
        "g_zz",
        "e_theta",
        "e_zeta",
        "e_theta_z",
        "e_zeta_z",
        "e_theta_zz",
        "e_zeta_zz",
        "|B|",
        "|B|_z",
    ],
)
def _B_mag_zz(params, transforms, profiles, data, **kwargs):
    data["|B|_zz"] = (
        data["B^theta_z"]
        * (
            data["B^zeta_z"] * data["g_tz"]
            + data["B^theta_z"] * data["g_tt"]
            + data["B^theta"] * dot(data["e_theta_z"], data["e_theta"])
        )
        + data["B^theta"]
        * (
            data["B^zeta_zz"] * data["g_tz"]
            + data["B^theta_zz"] * data["g_tt"]
            + data["B^theta_z"] * dot(data["e_theta_z"], data["e_theta"])
        )
        + data["B^theta"]
        * (
            data["B^zeta_z"]
            * (
                dot(data["e_theta_z"], data["e_zeta"])
                + dot(data["e_theta"], data["e_zeta_z"])
            )
            + 2 * data["B^theta_z"] * dot(data["e_theta_z"], data["e_theta"])
            + data["B^theta"]
            * (
                dot(data["e_theta_zz"], data["e_theta"])
                + dot(data["e_theta_z"], data["e_theta_z"])
            )
        )
        + data["B^zeta_z"]
        * (
            data["B^theta_z"] * data["g_tz"]
            + data["B^zeta_z"] * data["g_zz"]
            + data["B^zeta"] * dot(data["e_zeta_z"], data["e_zeta"])
        )
        + data["B^zeta"]
        * (
            data["B^theta_zz"] * data["g_tz"]
            + data["B^zeta_zz"] * data["g_zz"]
            + data["B^zeta_z"] * dot(data["e_zeta_z"], data["e_zeta"])
        )
        + data["B^zeta"]
        * (
            data["B^theta_z"]
            * (
                dot(data["e_theta_z"], data["e_zeta"])
                + dot(data["e_theta"], data["e_zeta_z"])
            )
            + 2 * data["B^zeta_z"] * dot(data["e_zeta_z"], data["e_zeta"])
            + data["B^zeta"]
            * (
                dot(data["e_zeta_zz"], data["e_zeta"])
                + dot(data["e_zeta_z"], data["e_zeta_z"])
            )
        )
        + (data["B^theta_z"] * data["B^zeta"] + data["B^theta"] * data["B^zeta_z"])
        * (
            dot(data["e_theta_z"], data["e_zeta"])
            + dot(data["e_zeta_z"], data["e_theta"])
        )
        + data["B^theta"]
        * data["B^zeta"]
        * (
            dot(data["e_theta_zz"], data["e_zeta"])
            + dot(data["e_zeta_zz"], data["e_theta"])
            + 2 * dot(data["e_theta_z"], data["e_zeta_z"])
        )
    ) / data["|B|"] - data["|B|_z"] ** 2 / data["|B|"]
    return data


# TODO: |B|_rt
# TODO: |B|_rz
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
        "B^theta_t",
        "B^zeta_t",
        "B^theta_z",
        "B^zeta_z",
        "B^theta_tz",
        "B^zeta_tz",
        "g_tt",
        "g_tz",
        "g_zz",
        "e_theta",
        "e_zeta",
        "e_theta_t",
        "e_zeta_t",
        "e_theta_z",
        "e_zeta_z",
        "e_theta_tz",
        "e_zeta_tz",
        "|B|",
        "|B|_t",
        "|B|_z",
    ],
)
def _B_mag_tz(params, transforms, profiles, data, **kwargs):
    data["|B|_tz"] = (
        data["B^theta_z"]
        * (
            data["B^zeta_t"] * data["g_tz"]
            + data["B^theta_t"] * data["g_tt"]
            + data["B^theta"] * dot(data["e_theta_t"], data["e_theta"])
        )
        + data["B^theta"]
        * (
            data["B^zeta_tz"] * data["g_tz"]
            + data["B^theta_tz"] * data["g_tt"]
            + data["B^theta_z"] * dot(data["e_theta_t"], data["e_theta"])
        )
        + data["B^theta"]
        * (
            data["B^zeta_t"]
            * (
                dot(data["e_theta_z"], data["e_zeta"])
                + dot(data["e_theta"], data["e_zeta_z"])
            )
            + 2 * data["B^theta_t"] * dot(data["e_theta_z"], data["e_theta"])
            + data["B^theta"]
            * (
                dot(data["e_theta_tz"], data["e_theta"])
                + dot(data["e_theta_t"], data["e_theta_z"])
            )
        )
        + data["B^zeta_z"]
        * (
            data["B^theta_t"] * data["g_tz"]
            + data["B^zeta_t"] * data["g_zz"]
            + data["B^zeta"] * dot(data["e_zeta_t"], data["e_zeta"])
        )
        + data["B^zeta"]
        * (
            data["B^theta_tz"] * data["g_tz"]
            + data["B^zeta_tz"] * data["g_zz"]
            + data["B^zeta_z"] * dot(data["e_zeta_t"], data["e_zeta"])
        )
        + data["B^zeta"]
        * (
            data["B^theta_t"]
            * (
                dot(data["e_theta_z"], data["e_zeta"])
                + dot(data["e_theta"], data["e_zeta_z"])
            )
            + 2 * data["B^zeta_t"] * dot(data["e_zeta_z"], data["e_zeta"])
            + data["B^zeta"]
            * (
                dot(data["e_zeta_tz"], data["e_zeta"])
                + dot(data["e_zeta_t"], data["e_zeta_z"])
            )
        )
        + (data["B^theta_z"] * data["B^zeta"] + data["B^theta"] * data["B^zeta_z"])
        * (
            dot(data["e_theta_t"], data["e_zeta"])
            + dot(data["e_zeta_t"], data["e_theta"])
        )
        + data["B^theta"]
        * data["B^zeta"]
        * (
            dot(data["e_theta_tz"], data["e_zeta"])
            + dot(data["e_zeta_tz"], data["e_theta"])
            + dot(data["e_theta_t"], data["e_zeta_z"])
            + dot(data["e_zeta_t"], data["e_theta_z"])
        )
    ) / data["|B|"] - data["|B|_t"] * data["|B|_z"] / data["|B|"]
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
        jnp.sum(transforms["grid"].weights * data["sqrt(g)"] * data["|B|"]) / data["V"]
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
    data=["sqrt(g)", "|B|", "V_r(r)"],
)
def _B_fsa(params, transforms, profiles, data, **kwargs):
    data["<|B|>"] = surface_averages(
        transforms["grid"],
        data["|B|"],
        jnp.abs(data["sqrt(g)"]),
        denominator=data["V_r(r)"],
    )
    return data


@register_compute_fun(
    name="<B^2>",
    label="\\langle B^2 \\rangle",
    units="T^2",
    units_long="Tesla squared",
    description="Flux surface average magnetic field squared",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["sqrt(g)", "|B|^2", "V_r(r)"],
)
def _B2_fsa(params, transforms, profiles, data, **kwargs):
    data["<B^2>"] = surface_averages(
        transforms["grid"],
        data["|B|^2"],
        jnp.abs(data["sqrt(g)"]),
        denominator=data["V_r(r)"],
    )
    return data


@register_compute_fun(
    name="<B^2>_r",
    label="\\partial_{\\rho} \\langle B^2 \\rangle",
    units="T^2",
    units_long="Tesla squared",
    description="Flux surface average magnetic field squared, radial derivative",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "sqrt(g)",
        "sqrt(g)_r",
        "B",
        "B_r",
        "|B|^2",
        "<B^2>",
        "V_r(r)",
        "V_rr(r)",
    ],
)
def _B2_fsa_r(params, transforms, profiles, data, **kwargs):
    data["<B^2>_r"] = (
        surface_integrals(
            transforms["grid"],
            data["sqrt(g)_r"] * jnp.sign(data["sqrt(g)"]) * data["|B|^2"]
            + jnp.abs(data["sqrt(g)"]) * 2 * dot(data["B"], data["B_r"]),
        )
        - data["V_rr(r)"] * data["<B^2>"]
    ) / data["V_r(r)"]
    return data


@register_compute_fun(
    name="grad(|B|^2)_rho",
    label="(\\nabla B^{2})_{\\rho}",
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
    label="(\\nabla B^{2})_{\\theta}",
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
    label="(\\nabla B^{2})_{\\zeta}",
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
    label="\\nabla B^{2}",
    units="T^{2} \\cdot m^{-1}",
    units_long="Tesla squared / meters",
    description="Magnetic pressure gradient",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "grad(|B|^2)_rho",
        "grad(|B|^2)_theta",
        "grad(|B|^2)_zeta",
        "e^rho",
        "e^theta",
        "e^zeta",
    ],
)
def _gradB2(params, transforms, profiles, data, **kwargs):
    data["grad(|B|^2)"] = (
        data["grad(|B|^2)_rho"] * data["e^rho"].T
        + data["grad(|B|^2)_theta"] * data["e^theta"].T
        + data["grad(|B|^2)_zeta"] * data["e^zeta"].T
    ).T
    return data


@register_compute_fun(
    name="|grad(|B|^2)|/2mu0",
    label="|\\nabla B^{2}/(2\\mu_0)|",
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
    data=["sqrt(g)", "B^theta", "B^zeta", "J^theta", "J^zeta"],
)
def _curl_B_x_B_rho(params, transforms, profiles, data, **kwargs):
    data["(curl(B)xB)_rho"] = (
        mu_0
        * data["sqrt(g)"]
        * (data["B^zeta"] * data["J^theta"] - data["B^theta"] * data["J^zeta"])
    )
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
    data=["sqrt(g)", "B^zeta", "J^rho"],
)
def _curl_B_x_B_theta(params, transforms, profiles, data, **kwargs):
    data["(curl(B)xB)_theta"] = -mu_0 * data["sqrt(g)"] * data["B^zeta"] * data["J^rho"]
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
    data=["sqrt(g)", "B^theta", "J^rho"],
)
def _curl_B_x_B_zeta(params, transforms, profiles, data, **kwargs):
    data["(curl(B)xB)_zeta"] = mu_0 * data["sqrt(g)"] * data["B^theta"] * data["J^rho"]
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
        "(curl(B)xB)_theta",
        "(curl(B)xB)_zeta",
        "e^rho",
        "e^theta",
        "e^zeta",
    ],
)
def _curl_B_x_B(params, transforms, profiles, data, **kwargs):
    data["curl(B)xB"] = (
        data["(curl(B)xB)_rho"] * data["e^rho"].T
        + data["(curl(B)xB)_theta"] * data["e^theta"].T
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


# TODO: (B*grad(|B|))_r
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
