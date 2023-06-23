from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import cross, dot


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
    description="Jacobian determinant of PEST flux coordinate system",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e_rho", "e_theta_PEST", "e_zeta"],
)
def _sqrtg_pest(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_PEST"] = dot(
        data["e_rho"], cross(data["e_theta_PEST"], data["e_zeta"])
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
)
def _e_theta_x_e_zeta(params, transforms, profiles, data, **kwargs):
    data["|e_theta x e_zeta|"] = jnp.linalg.norm(
        cross(data["e_theta"], data["e_zeta"]), axis=1
    )
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
    data=["e_rho", "e_zeta"],
)
def _e_zeta_x_e_rho(params, transforms, profiles, data, **kwargs):
    data["|e_zeta x e_rho|"] = jnp.linalg.norm(
        cross(data["e_zeta"], data["e_rho"]), axis=1
    )
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
)
def _e_rho_x_e_theta(params, transforms, profiles, data, **kwargs):
    data["|e_rho x e_theta|"] = jnp.linalg.norm(
        cross(data["e_rho"], data["e_theta"]), axis=1
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
        + dot(data["e_rho_t"], cross(data["e_theta_r"], data["e_zeta"]))
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
        + dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta_t"]))
        + dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta_z"]))
        + dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta_z"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_tz"]))
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
        + dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta_r"]))
        + dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta_r"]))
        + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_rz"]))
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
    data=[
        "e^rho",
        "sqrt(g)_r",
        "sqrt(g)",
        "e_rho",
        "e_theta",
        "e_rho_r",
        "e_theta_r",
    ],
)
def _g_sup_rr_r(params, transforms, profiles, data, **kwargs):
    data["g^rr_r"] = (
        -data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^rho"], cross(data["e_theta"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_theta_r"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_theta"], data["e_zeta_r"]))
        - data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^rho"], cross(data["e_theta"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_theta_r"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_theta"], data["e_zeta_r"]))
    )
    return data


@register_compute_fun(
    name="g^rt_r",
    label="g^{\\rho}{\\theta}_{\\rho}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Poloidal element of contravariant metric tensor, "
    + "first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^rho",
        "e^theta",
        "sqrt(g)_r",
        "sqrt(g)",
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_r",
        "e_theta_r",
        "e_zeta_r",
    ],
)
def _g_sup_rt_r(params, transforms, profiles, data, **kwargs):
    data["g^rt_r"] = (
        -data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^theta"], cross(data["e_theta"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_theta_r"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_zeta_r"]))
        - data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^rho"], cross(data["e_zeta"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_zeta_r"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_zeta"], data["e_rho_r"]))
    )
    return data


@register_compute_fun(
    name="g^rz_r",
    label="g^{\\rho}{\\zeta}_{\\rho}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Toroidal element of contravariant metric tensor, "
    + "first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^rho",
        "e^zeta",
        "sqrt(g)_r",
        "sqrt(g)",
        "e_rho",
        "e_zeta",
        "e_theta",
        "e_rho_r",
        "e_zeta_r",
        "e_theta_r",
    ],
)
def _g_sup_rz_r(params, transforms, profiles, data, **kwargs):
    data["g^rz_r"] = (
        -data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^zeta"], cross(data["e_theta"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_theta_r"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_zeta_r"]))
        - data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^rho"], cross(data["e_rho"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_rho_r"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_rho"], data["e_theta_r"]))
    )
    return data


@register_compute_fun(
    name="g^tt_r",
    label="g^{\\theta}{\\theta}_{\\rho}",
    units="m^-2",
    units_long="inverse square meters",
    description="Poloidal/Poloidal element of contravariant metric tensor, "
    + "first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^theta",
        "e^theta",
        "sqrt(g)_r",
        "sqrt(g)",
        "e_theta",
        "e_theta",
        "e_rho",
        "e_theta_r",
        "e_theta_r",
        "e_rho_r",
    ],
)
def _g_sup_tt_r(params, transforms, profiles, data, **kwargs):
    data["g^tt_r"] = (
        -data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta_r"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_rho_r"]))
        - data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta_r"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_rho_r"]))
    )
    return data


@register_compute_fun(
    name="g^tz_r",
    label="g^{\\theta}{\\zeta}_{\\rho}",
    units="m^-2",
    units_long="inverse square meters",
    description="Poloidal/Toroidal element of contravariant metric tensor, "
    + "first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^theta",
        "e^zeta",
        "sqrt(g)_r",
        "sqrt(g)",
        "e_theta",
        "e_zeta",
        "e_rho",
        "e_theta_r",
        "e_zeta_r",
        "e_rho_r",
    ],
)
def _g_sup_tz_r(params, transforms, profiles, data, **kwargs):
    data["g^tz_r"] = (
        -data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^zeta"], cross(data["e_zeta"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_zeta_r"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_zeta"], data["e_rho_r"]))
        - data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^theta"], cross(data["e_rho"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_rho_r"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_rho"], data["e_theta_r"]))
    )
    return data


@register_compute_fun(
    name="g^zz_r",
    label="g^{\\zeta}{\\zeta}_{\\rho}",
    units="m^-2",
    units_long="inverse square meters",
    description="Toroidal/Toroidal element of contravariant metric tensor, "
    + "first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^zeta",
        "e^zeta",
        "sqrt(g)_r",
        "sqrt(g)",
        "e_zeta",
        "e_zeta",
        "e_rho",
        "e_zeta_r",
        "e_zeta_r",
        "e_rho_r",
    ],
)
def _g_sup_zz_r(params, transforms, profiles, data, **kwargs):
    data["g^zz_r"] = (
        -data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho_r"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_theta_r"]))
        - data["sqrt(g)_r"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho_r"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_theta_r"]))
    )
    return data


@register_compute_fun(
    name="g^rr_t",
    label="g^{\\rho}{\\rho}_{\\theta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Radial element of contravariant metric tensor, "
    + "first poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^rho",
        "e^rho",
        "sqrt(g)_t",
        "sqrt(g)",
        "e_rho",
        "e_rho",
        "e_theta",
        "e_rho_t",
        "e_rho_t",
        "e_theta_t",
    ],
)
def _g_sup_rr_t(params, transforms, profiles, data, **kwargs):
    data["g^rr_t"] = (
        -data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^rho"], cross(data["e_theta"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_theta_t"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_theta"], data["e_zeta_t"]))
        - data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^rho"], cross(data["e_theta"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_theta_t"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_theta"], data["e_zeta_t"]))
    )
    return data


@register_compute_fun(
    name="g^rt_t",
    label="g^{\\rho}{\\theta}_{\\theta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Poloidal element of contravariant metric tensor, "
    + "first poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^rho",
        "e^theta",
        "sqrt(g)_t",
        "sqrt(g)",
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_t",
        "e_theta_t",
        "e_zeta_t",
    ],
)
def _g_sup_rt_t(params, transforms, profiles, data, **kwargs):
    data["g^rt_t"] = (
        -data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^theta"], cross(data["e_theta"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_theta_t"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_theta"], data["e_zeta_t"]))
        - data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^rho"], cross(data["e_zeta"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_zeta_t"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_zeta"], data["e_rho_t"]))
    )
    return data


@register_compute_fun(
    name="g^rz_t",
    label="g^{\\rho}{\\zeta}_{\\theta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Toroidal element of contravariant metric tensor, "
    + "first poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^rho",
        "e^zeta",
        "sqrt(g)_t",
        "sqrt(g)",
        "e_rho",
        "e_zeta",
        "e_theta",
        "e_rho_t",
        "e_zeta_t",
        "e_theta_t",
    ],
)
def _g_sup_rz_t(params, transforms, profiles, data, **kwargs):
    data["g^rz_t"] = (
        -data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^zeta"], cross(data["e_theta"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_theta_t"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_theta"], data["e_zeta_t"]))
        - data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^rho"], cross(data["e_rho"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_rho_t"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_rho"], data["e_theta_t"]))
    )
    return data


@register_compute_fun(
    name="g^tt_t",
    label="g^{\\theta}{\\theta}_{\\theta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Poloidal/Poloidal element of contravariant metric tensor, "
    + "first poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^theta",
        "e^theta",
        "sqrt(g)_t",
        "sqrt(g)",
        "e_theta",
        "e_theta",
        "e_rho",
        "e_theta_t",
        "e_theta_t",
        "e_rho_t",
    ],
)
def _g_sup_tt_t(params, transforms, profiles, data, **kwargs):
    data["g^tt_t"] = (
        -data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta_t"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_rho_t"]))
        - data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta_t"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_rho_t"]))
    )
    return data


@register_compute_fun(
    name="g^tz_t",
    label="g^{\\theta}{\\zeta}_{\\theta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Poloidal/Toroidal element of contravariant metric tensor, "
    + "first poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^theta",
        "e^zeta",
        "sqrt(g)_t",
        "sqrt(g)",
        "e_theta",
        "e_zeta",
        "e_rho",
        "e_theta_t",
        "e_zeta_t",
        "e_rho_t",
    ],
)
def _g_sup_tz_t(params, transforms, profiles, data, **kwargs):
    data["g^tz_t"] = (
        -data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^zeta"], cross(data["e_zeta"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_zeta_t"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_zeta"], data["e_rho_t"]))
        - data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^theta"], cross(data["e_rho"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_rho_t"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_rho"], data["e_theta_t"]))
    )
    return data


@register_compute_fun(
    name="g^zz_t",
    label="g^{\\zeta}{\\zeta}_{\\theta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Toroidal/Toroidal element of contravariant metric tensor, "
    + "first poloidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^zeta",
        "e^zeta",
        "sqrt(g)_t",
        "sqrt(g)",
        "e_zeta",
        "e_zeta",
        "e_rho",
        "e_zeta_t",
        "e_zeta_t",
        "e_rho_t",
    ],
)
def _g_sup_zz_t(params, transforms, profiles, data, **kwargs):
    data["g^zz_t"] = (
        -data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho_t"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_theta_t"]))
        - data["sqrt(g)_t"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho_t"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_theta_t"]))
    )
    return data


@register_compute_fun(
    name="g^rr_z",
    label="g^{\\rho}{\\rho}_{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Radial element of contravariant metric tensor, "
    + "first toroidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^rho",
        "e^rho",
        "sqrt(g)_z",
        "sqrt(g)",
        "e_rho",
        "e_rho",
        "e_theta",
        "e_rho_z",
        "e_rho_z",
        "e_theta_z",
    ],
)
def _g_sup_rr_z(params, transforms, profiles, data, **kwargs):
    data["g^rr_z"] = (
        -data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^rho"], cross(data["e_theta"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_theta_z"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_theta"], data["e_zeta_z"]))
        - data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^rho"], cross(data["e_theta"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_theta_z"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_theta"], data["e_zeta_z"]))
    )
    return data


@register_compute_fun(
    name="g^rt_z",
    label="g^{\\rho}{\\theta}_{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Poloidal element of contravariant metric tensor, "
    + "first toroidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^rho",
        "e^theta",
        "sqrt(g)_z",
        "sqrt(g)",
        "e_rho",
        "e_theta",
        "e_zeta",
        "e_rho_z",
        "e_theta_z",
        "e_zeta_z",
    ],
)
def _g_sup_rt_z(params, transforms, profiles, data, **kwargs):
    data["g^rt_z"] = (
        -data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^theta"], cross(data["e_theta"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_theta_z"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_theta"], data["e_zeta_z"]))
        - data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^rho"], cross(data["e_zeta"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_zeta_z"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_zeta"], data["e_rho_z"]))
    )
    return data


@register_compute_fun(
    name="g^rz_z",
    label="g^{\\rho}{\\zeta}_{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Radial/Toroidal element of contravariant metric tensor, "
    + "first toroidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^rho",
        "e^zeta",
        "sqrt(g)_z",
        "sqrt(g)",
        "e_rho",
        "e_zeta",
        "e_theta",
        "e_rho_z",
        "e_zeta_z",
        "e_theta_z",
    ],
)
def _g_sup_rz_z(params, transforms, profiles, data, **kwargs):
    data["g^rz_z"] = (
        -data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^zeta"], cross(data["e_theta"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_theta_z"], data["e_zeta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_theta"], data["e_zeta_z"]))
        - data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^rho"], cross(data["e_rho"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_rho_z"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^rho"], cross(data["e_rho"], data["e_theta_z"]))
    )
    return data


@register_compute_fun(
    name="g^tt_z",
    label="g^{\\theta}{\\theta}_{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Poloidal/Poloidal element of contravariant metric tensor, "
    + "first toroidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^theta",
        "e^theta",
        "sqrt(g)_z",
        "sqrt(g)",
        "e_theta",
        "e_theta",
        "e_rho",
        "e_theta_z",
        "e_theta_z",
        "e_rho_z",
    ],
)
def _g_sup_tt_z(params, transforms, profiles, data, **kwargs):
    data["g^tt_z"] = (
        -data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta_z"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_rho_z"]))
        - data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta_z"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_zeta"], data["e_rho_z"]))
    )
    return data


@register_compute_fun(
    name="g^tz_z",
    label="g^{\\theta}{\\zeta}_{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Poloidal/Toroidal element of contravariant metric tensor, "
    + "first toroidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^theta",
        "e^zeta",
        "sqrt(g)_z",
        "sqrt(g)",
        "e_theta",
        "e_zeta",
        "e_rho",
        "e_theta_z",
        "e_zeta_z",
        "e_rho_z",
    ],
)
def _g_sup_tz_z(params, transforms, profiles, data, **kwargs):
    data["g^tz_z"] = (
        -data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^zeta"], cross(data["e_zeta"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_zeta_z"], data["e_rho"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_zeta"], data["e_rho_z"]))
        - data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^theta"], cross(data["e_rho"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_rho_z"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^theta"], cross(data["e_rho"], data["e_theta_z"]))
    )
    return data


@register_compute_fun(
    name="g^zz_z",
    label="g^{\\zeta}{\\zeta}_{\\zeta}",
    units="m^-2",
    units_long="inverse square meters",
    description="Toroidal/Toroidal element of contravariant metric tensor, "
    + "first toroidal derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "e^zeta",
        "e^zeta",
        "sqrt(g)_z",
        "sqrt(g)",
        "e_zeta",
        "e_zeta",
        "e_rho",
        "e_zeta_z",
        "e_zeta_z",
        "e_rho_z",
    ],
)
def _g_sup_zz_z(params, transforms, profiles, data, **kwargs):
    data["g^zz_z"] = (
        -data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho_z"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_theta_z"]))
        - data["sqrt(g)_z"]
        / data["sqrt(g)"] ** 2
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho_z"], data["e_theta"]))
        + 1
        / data["sqrt(g)"]
        * dot(data["e^zeta"], cross(data["e_rho"], data["e_theta_z"]))
    )
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
