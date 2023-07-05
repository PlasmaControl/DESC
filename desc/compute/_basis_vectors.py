from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import cross


@register_compute_fun(
    name="e_rho",
    label="\\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_r", "Z_r"],
)
def _e_sub_rho(params, transforms, profiles, data, **kwargs):
    data["e_rho"] = jnp.array([data["R_r"], data["0"], data["Z_r"]]).T
    return data


@register_compute_fun(
    name="e_theta",
    label="\\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_t", "Z_t"],
)
def _e_sub_theta(params, transforms, profiles, data, **kwargs):
    data["e_theta"] = jnp.array([data["R_t"], data["0"], data["Z_t"]]).T
    return data


@register_compute_fun(
    name="e_zeta",
    label="\\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_z", "Z_z"],
)
def _e_sub_zeta(params, transforms, profiles, data, **kwargs):
    data["e_zeta"] = jnp.array([data["R_z"], data["R"], data["Z_z"]]).T
    return data


@register_compute_fun(
    name="e_rho_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rr", "Z_rr"],
)
def _e_sub_rho_r(params, transforms, profiles, data, **kwargs):
    data["e_rho_r"] = jnp.array([data["R_rr"], data["0"], data["Z_rr"]]).T
    return data


@register_compute_fun(
    name="e_rho_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, derivative wrt poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rt", "Z_rt"],
)
def _e_sub_rho_t(params, transforms, profiles, data, **kwargs):
    data["e_rho_t"] = jnp.array([data["R_rt"], data["0"], data["Z_rt"]]).T
    return data


@register_compute_fun(
    name="e_rho_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rz", "Z_rz"],
)
def _e_sub_rho_z(params, transforms, profiles, data, **kwargs):
    data["e_rho_z"] = jnp.array([data["R_rz"], data["0"], data["Z_rz"]]).T
    return data


@register_compute_fun(
    name="e_theta_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rt", "Z_rt"],
)
def _e_sub_theta_r(params, transforms, profiles, data, **kwargs):
    data["e_theta_r"] = jnp.array([data["R_rt"], data["0"], data["Z_rt"]]).T
    return data


@register_compute_fun(
    name="e_theta_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, derivative wrt poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_tt", "Z_tt"],
)
def _e_sub_theta_t(params, transforms, profiles, data, **kwargs):
    data["e_theta_t"] = jnp.array([data["R_tt"], data["0"], data["Z_tt"]]).T
    return data


@register_compute_fun(
    name="e_theta_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_tz", "Z_tz"],
)
def _e_sub_theta_z(params, transforms, profiles, data, **kwargs):
    data["e_theta_z"] = jnp.array([data["R_tz"], data["0"], data["Z_tz"]]).T
    return data


@register_compute_fun(
    name="e_zeta_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_r", "R_rz", "Z_rz"],
)
def _e_sub_zeta_r(params, transforms, profiles, data, **kwargs):
    data["e_zeta_r"] = jnp.array([data["R_rz"], data["R_r"], data["Z_rz"]]).T
    return data


@register_compute_fun(
    name="e_zeta_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, derivative wrt poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_t", "R_tz", "Z_tz"],
)
def _e_sub_zeta_t(params, transforms, profiles, data, **kwargs):
    data["e_zeta_t"] = jnp.array([data["R_tz"], data["R_t"], data["Z_tz"]]).T
    return data


@register_compute_fun(
    name="e_zeta_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_z", "R_zz", "Z_zz"],
)
def _e_sub_zeta_z(params, transforms, profiles, data, **kwargs):
    data["e_zeta_z"] = jnp.array([data["R_zz"], data["R_z"], data["Z_zz"]]).T
    return data


@register_compute_fun(
    name="e_rho_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, second derivative wrt radial "
    + "coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rrr", "Z_rrr"],
)
def _e_sub_rho_rr(params, transforms, profiles, data, **kwargs):
    data["e_rho_rr"] = jnp.array([data["R_rrr"], data["0"], data["Z_rrr"]]).T
    return data


@register_compute_fun(
    name="e_rho_tt",
    label="\\partial_{\\theta \\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, second derivative wrt poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rtt", "Z_rtt"],
)
def _e_sub_rho_tt(params, transforms, profiles, data, **kwargs):
    data["e_rho_tt"] = jnp.array([data["R_rtt"], data["0"], data["Z_rtt"]]).T
    return data


@register_compute_fun(
    name="e_rho_zz",
    label="\\partial_{\\zeta \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, second derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rzz", "Z_rzz"],
)
def _e_sub_rho_zz(params, transforms, profiles, data, **kwargs):
    data["e_rho_zz"] = jnp.array([data["R_rzz"], data["0"], data["Z_rzz"]]).T
    return data


@register_compute_fun(
    name="e_rho_rt",
    label="\\partial_{\\rho \\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, second derivative wrt radial "
    + "coordinate and poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rrt", "Z_rrt"],
)
def _e_sub_rho_rt(params, transforms, profiles, data, **kwargs):
    data["e_rho_rt"] = jnp.array([data["R_rrt"], data["0"], data["Z_rrt"]]).T
    return data


@register_compute_fun(
    name="e_rho_rz",
    label="\\partial_{\\rho \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, second derivative wrt radial "
    + "coordinate and toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rrz", "Z_rrz"],
)
def _e_sub_rho_rz(params, transforms, profiles, data, **kwargs):
    data["e_rho_rz"] = jnp.array([data["R_rrz"], data["0"], data["Z_rrz"]]).T
    return data


@register_compute_fun(
    name="e_rho_tz",
    label="\\partial_{\\theta \\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, second derivative wrt poloidal "
    + "and toroidal angles",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rtz", "Z_rtz"],
)
def _e_sub_rho_tz(params, transforms, profiles, data, **kwargs):
    data["e_rho_tz"] = jnp.array([data["R_rtz"], data["0"], data["Z_rtz"]]).T
    return data


@register_compute_fun(
    name="e_theta_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, second derivative wrt radial "
    + "coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rrt", "Z_rrt"],
)
def _e_sub_theta_rr(params, transforms, profiles, data, **kwargs):
    data["e_theta_rr"] = jnp.array([data["R_rrt"], data["0"], data["Z_rrt"]]).T
    return data


@register_compute_fun(
    name="e_theta_tt",
    label="\\partial_{\\theta \\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, second derivative wrt poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_ttt", "Z_ttt"],
)
def _e_sub_theta_tt(params, transforms, profiles, data, **kwargs):
    data["e_theta_tt"] = jnp.array([data["R_ttt"], data["0"], data["Z_ttt"]]).T
    return data


@register_compute_fun(
    name="e_theta_zz",
    label="\\partial_{\\zeta \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, second derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_tzz", "Z_tzz"],
)
def _e_sub_theta_zz(params, transforms, profiles, data, **kwargs):
    data["e_theta_zz"] = jnp.array([data["R_tzz"], data["0"], data["Z_tzz"]]).T
    return data


@register_compute_fun(
    name="e_theta_rt",
    label="\\partial_{\\rho \\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, second derivative wrt radial "
    + "coordinate and poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rtt", "Z_rtt"],
)
def _e_sub_theta_rt(params, transforms, profiles, data, **kwargs):
    data["e_theta_rt"] = jnp.array([data["R_rtt"], data["0"], data["Z_rtt"]]).T
    return data


@register_compute_fun(
    name="e_theta_rz",
    label="\\partial_{\\rho \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, second derivative wrt radial "
    + "coordinate and toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_rtz", "Z_rtz"],
)
def _e_sub_theta_rz(params, transforms, profiles, data, **kwargs):
    data["e_theta_rz"] = jnp.array([data["R_rtz"], data["0"], data["Z_rtz"]]).T
    return data


@register_compute_fun(
    name="e_theta_tz",
    label="\\partial_{\\theta \\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, second derivative wrt poloidal "
    + "and toroidal angles",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0", "R_ttz", "Z_ttz"],
)
def _e_sub_theta_tz(params, transforms, profiles, data, **kwargs):
    data["e_theta_tz"] = jnp.array([data["R_ttz"], data["0"], data["Z_ttz"]]).T
    return data


@register_compute_fun(
    name="e_zeta_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, second derivative wrt radial "
    + "coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_rr", "R_rrz", "Z_rrz"],
)
def _e_sub_zeta_rr(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rr"] = jnp.array([data["R_rrz"], data["R_rr"], data["Z_rrz"]]).T
    return data


@register_compute_fun(
    name="e_zeta_tt",
    label="\\partial_{\\theta \\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, second derivative wrt poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_tt", "R_ttz", "Z_ttz"],
)
def _e_sub_zeta_tt(params, transforms, profiles, data, **kwargs):
    data["e_zeta_tt"] = jnp.array([data["R_ttz"], data["R_tt"], data["Z_ttz"]]).T
    return data


@register_compute_fun(
    name="e_zeta_zz",
    label="\\partial_{\\zeta \\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, second derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_zz", "R_zzz", "Z_zzz"],
)
def _e_sub_zeta_zz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_zz"] = jnp.array([data["R_zzz"], data["R_zz"], data["Z_zzz"]]).T
    return data


@register_compute_fun(
    name="e_zeta_rt",
    label="\\partial_{\\rho \\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, second derivative wrt radial "
    + "coordinate and poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_rt", "R_rtz", "Z_rtz"],
)
def _e_sub_zeta_rt(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rt"] = jnp.array([data["R_rtz"], data["R_rt"], data["Z_rtz"]]).T
    return data


@register_compute_fun(
    name="e_zeta_rz",
    label="\\partial_{\\rho \\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, second derivative wrt radial "
    + "coordinate and toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_rz", "R_rzz", "Z_rzz"],
)
def _e_sub_zeta_rz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rz"] = jnp.array([data["R_rzz"], data["R_rz"], data["Z_rzz"]]).T
    return data


@register_compute_fun(
    name="e_zeta_tz",
    label="\\partial_{\\theta \\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, second derivative wrt poloidal "
    + "and toroidal angles",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R_tz", "R_tzz", "Z_tzz"],
)
def _e_sub_zeta_tz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_tz"] = jnp.array([data["R_tzz"], data["R_tz"], data["Z_tzz"]]).T
    return data


@register_compute_fun(
    name="e_theta_PEST",
    label="\\mathbf{e}_{\\theta_{PEST}}",
    units="m",
    units_long="meters",
    description="Covariant straight field line poloidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "0",
        "R_t",
        "Z_t",
        "lambda_t",
    ],
)
def _e_sub_theta_pest(params, transforms, profiles, data, **kwargs):
    dt_dv = 1 / (1 + data["lambda_t"])
    data["e_theta_PEST"] = jnp.array(
        [data["R_t"] * dt_dv, data["0"], data["Z_t"] * dt_dv]
    ).T
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
    data=["e_theta", "e_zeta", "sqrt(g)"],
)
def _e_sup_rho(params, transforms, profiles, data, **kwargs):
    data["e^rho"] = (cross(data["e_theta"], data["e_zeta"]).T / data["sqrt(g)"]).T
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
    data=["e_rho", "e_zeta", "sqrt(g)"],
)
def _e_sup_theta(params, transforms, profiles, data, **kwargs):
    data["e^theta"] = (cross(data["e_zeta"], data["e_rho"]).T / data["sqrt(g)"]).T
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
    data=["e_rho", "e_theta", "sqrt(g)"],
)
def _e_sup_zeta(params, transforms, profiles, data, **kwargs):
    data["e^zeta"] = (cross(data["e_rho"], data["e_theta"]).T / data["sqrt(g)"]).T
    return data


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
    data=["B"],
)
def _b(params, transforms, profiles, data, **kwargs):
    data["b"] = (data["B"].T / jnp.linalg.norm(data["B"], axis=-1)).T
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
    data=["e^rho"],
)
def _n_rho(params, transforms, profiles, data, **kwargs):
    data["n_rho"] = (data["e^rho"].T / jnp.linalg.norm(data["e^rho"], axis=-1)).T
    return data


@register_compute_fun(
    name="grad(alpha)",
    label="\\nabla \\alpha",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Unit vector normal to flux surface",
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
