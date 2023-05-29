from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import cross


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
)
def _e_sub_rho_(params, transforms, profiles, data, **kwargs):
    data["e_rho"] = jnp.array([data["R_r"], data["R"] * data["omega_r"], data["Z_r"]]).T
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
)
def _e_sub_theta_(params, transforms, profiles, data, **kwargs):
    data["e_theta"] = jnp.array(
        [data["R_t"], data["R"] * data["omega_t"], data["Z_t"]]
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
)
def _e_sub_zeta_(params, transforms, profiles, data, **kwargs):
    data["e_zeta"] = jnp.array(
        [data["R_z"], data["R"] * (1 + data["omega_z"]), data["Z_z"]]
    ).T
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
    data=["e_zeta", "phi_z"],
)
def _e_sub_phi_(params, transforms, profiles, data, **kwargs):
    # dX/dphi at const r,t = dX/dz * dz/dphi = dX/dz / (dphi/dz)
    data["e_phi"] = (data["e_zeta"].T / data["phi_z"]).T
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
)
def _e_sub_rho_r(params, transforms, profiles, data, **kwargs):
    data["e_rho_r"] = jnp.array(
        [
            -data["R"] * data["omega_r"] ** 2 + data["R_rr"],
            2 * data["R_r"] * data["omega_r"] + data["R"] * data["omega_rr"],
            data["Z_rr"],
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
)
def _e_sub_rho_t(params, transforms, profiles, data, **kwargs):
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
    name="e_rho_rr",
    label="\\partial_{\\rho}{\\rho} \\mathbf{e}_{\\rho}",
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
    data=["R", "R_r", "R_rr", "R_rrr", "Z_rrr", "omega_r", "omega_rr", "omega_rrr"],
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
    name="e_rho_rt",
    label="\\partial_{\\rho}{\\theta} \\mathbf{e}_{\\rho}",
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
    name="e_rho_rz",
    label="\\partial_{\\rho}{\\zeta} \\mathbf{e}_{\\rho}",
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
)
def _e_sub_rho_rz(params, transforms, profiles, data, **kwargs):
    data["e_rho_rz"] = jnp.array(
        [
            -2 * (1 + data["omega_z"]) * data["R_r"] * data["omega_r"]
            - data["R_z"] * data["omega_r"] ** 2
            - 2 * data["R"] * data["omega_r"] * data["omega_rz"]
            - data["R"] * data["omega_rr"]
            - data["R"] * data["omega_z"] * data["omega_rr"]
            + data["R_rrz"],
            2 * data["omega_r"] * data["R_rz"]
            + 2 * data["R_r"] * data["omega_rz"]
            + data["R_rr"]
            + data["omega_z"] * data["R_rr"]
            + data["R_z"] * data["omega_rr"]
            - data["R"]
            * ((1 + data["omega_z"]) * data["omega_r"] ** 2 - data["omega_rrz"]),
            data["Z_rrz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_tt",
    label="\\partial_{\\theta}{\\theta} \\mathbf{e}_{\\rho}",
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
    label="\\partial_{\\theta}{\\zeta} \\mathbf{e}_{\\rho}",
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
            - data["R"] * data["omega_rt"]
            - data["R"] * data["omega_z"] * data["omega_rt"]
            + data["R_rtz"],
            data["omega_tz"] * data["R_r"]
            + data["R_tz"] * data["omega_r"]
            + data["omega_t"] * data["R_rz"]
            + data["R_t"] * data["omega_rz"]
            + data["R_rt"]
            + data["omega_z"] * data["R_rt"]
            + data["R_z"] * data["omega_rt"]
            + data["R"]
            * (
                -((1 + data["omega_z"]) * data["omega_t"] * data["omega_r"])
                + data["omega_rtz"]
            ),
            data["Z_rtz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_rho_zz",
    label="\\partial_{\\zeta}{\\zeta} \\mathbf{e}_{\\rho}",
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
)
def _e_sub_rho_zz(params, transforms, profiles, data, **kwargs):
    data["e_rho_zz"] = jnp.array(
        [
            -((1 + data["omega_z"]) ** 2) * data["R_r"]
            - 2 * data["R_z"] * (1 + data["omega_z"]) * data["omega_r"]
            - data["R"] * data["omega_zz"] * data["omega_r"]
            - 2 * data["R"] * data["omega_rz"]
            - 2 * data["R"] * data["omega_z"] * data["omega_rz"]
            + data["R_rzz"],
            data["omega_zz"] * data["R_r"]
            + data["R_zz"] * data["omega_r"]
            + 2 * data["R_rz"]
            + 2 * data["omega_z"] * data["R_rz"]
            + 2 * data["R_z"] * data["omega_rz"]
            - data["R"]
            * ((1 + data["omega_z"]) ** 2 * data["omega_r"] - data["omega_rzz"]),
            data["Z_rzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant Poloidal basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_r", "R_rt", "R_t", "Z_rt", "omega_r", "omega_rt", "omega_t"],
)
def _e_sub_theta_r(params, transforms, profiles, data, **kwargs):
    data["e_theta_r"] = jnp.array(
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
    name="e_theta_rr",
    label="\\partial_{\\rho}{\\rho} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, second derivative wrt radial and radial"
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
)
def _e_sub_theta_rr(params, transforms, profiles, data, **kwargs):
    data["e_theta_rr"] = jnp.array(
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
    name="e_theta_rt",
    label="\\partial_{\\rho}{\\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, second derivative wrt radial and poloidal"
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
)
def _e_sub_theta_rt(params, transforms, profiles, data, **kwargs):
    data["e_theta_rt"] = jnp.array(
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
    name="e_theta_rz",
    label="\\partial_{\\rho}{\\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Poloidal basis vector, second derivative wrt radial and toroidal"
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
)
def _e_sub_theta_rz(params, transforms, profiles, data, **kwargs):
    data["e_theta_rz"] = jnp.array(
        [
            -((1 + data["omega_z"]) * data["R_t"] * data["omega_r"])
            - data["R"] * data["omega_tz"] * data["omega_r"]
            - data["omega_t"]
            * (
                (1 + data["omega_z"]) * data["R_r"]
                + data["R_z"] * data["omega_r"]
                + data["R"] * data["omega_rz"]
            )
            - data["R"] * data["omega_rt"]
            - data["R"] * data["omega_z"] * data["omega_rt"]
            + data["R_rtz"],
            data["omega_tz"] * data["R_r"]
            + data["R_tz"] * data["omega_r"]
            + data["omega_t"] * data["R_rz"]
            + data["R_t"] * data["omega_rz"]
            + data["R_rt"]
            + data["omega_z"] * data["R_rt"]
            + data["R_z"] * data["omega_rt"]
            + data["R"]
            * (
                -((1 + data["omega_z"]) * data["omega_t"] * data["omega_r"])
                + data["omega_rtz"]
            ),
            data["Z_rtz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_tt",
    label="\\partial_{\\theta}{\\theta} \\mathbf{e}_{\\theta}",
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
    data=["R", "R_t", "R_tt", "R_ttt", "Z_ttt", "omega_t", "omega_tt", "omega_ttt"],
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
    label="\\partial_{\\theta}{\\zeta} \\mathbf{e}_{\\theta}",
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
)
def _e_sub_theta_tz(params, transforms, profiles, data, **kwargs):
    data["e_theta_tz"] = jnp.array(
        [
            -2 * (1 + data["omega_z"]) * data["R_t"] * data["omega_t"]
            - data["R_z"] * data["omega_t"] ** 2
            - 2 * data["R"] * data["omega_t"] * data["omega_tz"]
            - data["R"] * data["omega_tt"]
            - data["R"] * data["omega_z"] * data["omega_tt"]
            + data["R_ttz"],
            2 * data["omega_t"] * data["R_tz"]
            + 2 * data["R_t"] * data["omega_tz"]
            + data["R_tt"]
            + data["omega_z"] * data["R_tt"]
            + data["R_z"] * data["omega_tt"]
            - data["R"]
            * ((1 + data["omega_z"]) * data["omega_t"] ** 2 - data["omega_ttz"]),
            data["Z_ttz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_theta_zz",
    label="\\partial_{\\zeta}{\\zeta} \\mathbf{e}_{\\theta}",
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
)
def _e_sub_theta_zz(params, transforms, profiles, data, **kwargs):
    data["e_theta_zz"] = jnp.array(
        [
            -((1 + data["omega_z"]) ** 2) * data["R_t"]
            - 2 * data["R_z"] * (1 + data["omega_z"]) * data["omega_t"]
            - data["R"] * data["omega_zz"] * data["omega_t"]
            - 2 * data["R"] * data["omega_tz"]
            - 2 * data["R"] * data["omega_z"] * data["omega_tz"]
            + data["R_tzz"],
            data["omega_zz"] * data["R_t"]
            + data["R_zz"] * data["omega_t"]
            + 2 * data["R_tz"]
            + 2 * data["omega_z"] * data["R_tz"]
            + 2 * data["R_z"] * data["omega_tz"]
            - data["R"]
            * ((1 + data["omega_z"]) ** 2 * data["omega_t"] - data["omega_tzz"]),
            data["Z_tzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant Toroidal basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_r", "R_rz", "R_z", "Z_rz", "omega_r", "omega_rz", "omega_z"],
)
def _e_sub_zeta_r(params, transforms, profiles, data, **kwargs):
    data["e_zeta_r"] = jnp.array(
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
    name="e_zeta_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant Toroidal basis vector, derivative wrt poloidal coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_t", "R_tz", "R_z", "Z_tz", "omega_t", "omega_tz", "omega_z"],
)
def _e_sub_zeta_t(params, transforms, profiles, data, **kwargs):
    data["e_zeta_t"] = jnp.array(
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
    name="e_zeta_rr",
    label="\\partial_{\\rho}{\\rho} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, second derivative wrt radial and radial"
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
)
def _e_sub_zeta_rr(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rr"] = jnp.array(
        [
            -2 * (1 + data["omega_z"]) * data["R_r"] * data["omega_r"]
            - data["R_z"] * data["omega_r"] ** 2
            - 2 * data["R"] * data["omega_r"] * data["omega_rz"]
            - data["R"] * data["omega_rr"]
            - data["R"] * data["omega_z"] * data["omega_rr"]
            + data["R_rrz"],
            2 * data["omega_r"] * data["R_rz"]
            + 2 * data["R_r"] * data["omega_rz"]
            + data["R_rr"]
            + data["omega_z"] * data["R_rr"]
            + data["R_z"] * data["omega_rr"]
            - data["R"]
            * ((1 + data["omega_z"]) * data["omega_r"] ** 2 - data["omega_rrz"]),
            data["Z_rrz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_rt",
    label="\\partial_{\\rho}{\\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, second derivative wrt radial and poloidal"
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
)
def _e_sub_zeta_rt(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rt"] = jnp.array(
        [
            -((1 + data["omega_z"]) * data["R_t"] * data["omega_r"])
            - data["R"] * data["omega_tz"] * data["omega_r"]
            - data["omega_t"]
            * (
                (1 + data["omega_z"]) * data["R_r"]
                + data["R_z"] * data["omega_r"]
                + data["R"] * data["omega_rz"]
            )
            - data["R"] * data["omega_rt"]
            - data["R"] * data["omega_z"] * data["omega_rt"]
            + data["R_rtz"],
            data["omega_tz"] * data["R_r"]
            + data["R_tz"] * data["omega_r"]
            + data["omega_t"] * data["R_rz"]
            + data["R_t"] * data["omega_rz"]
            + data["R_rt"]
            + data["omega_z"] * data["R_rt"]
            + data["R_z"] * data["omega_rt"]
            + data["R"]
            * (
                -((1 + data["omega_z"]) * data["omega_t"] * data["omega_r"])
                + data["omega_rtz"]
            ),
            data["Z_rtz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_rz",
    label="\\partial_{\\rho}{\\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, second derivative wrt radial and toroidal"
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
)
def _e_sub_zeta_rz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_rz"] = jnp.array(
        [
            -((1 + data["omega_z"]) ** 2) * data["R_r"]
            - 2 * data["R_z"] * (1 + data["omega_z"]) * data["omega_r"]
            - data["R"] * data["omega_zz"] * data["omega_r"]
            - 2 * data["R"] * data["omega_rz"]
            - 2 * data["R"] * data["omega_z"] * data["omega_rz"]
            + data["R_rzz"],
            data["omega_zz"] * data["R_r"]
            + data["R_zz"] * data["omega_r"]
            + 2 * data["R_rz"]
            + 2 * data["omega_z"] * data["R_rz"]
            + 2 * data["R_z"] * data["omega_rz"]
            - data["R"]
            * ((1 + data["omega_z"]) ** 2 * data["omega_r"] - data["omega_rzz"]),
            data["Z_rzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_tt",
    label="\\partial_{\\theta}{\\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, second derivative wrt poloidal and poloidal"
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
)
def _e_sub_zeta_tt(params, transforms, profiles, data, **kwargs):
    data["e_zeta_tt"] = jnp.array(
        [
            -2 * (1 + data["omega_z"]) * data["R_t"] * data["omega_t"]
            - data["R_z"] * data["omega_t"] ** 2
            - 2 * data["R"] * data["omega_t"] * data["omega_tz"]
            - data["R"] * data["omega_tt"]
            - data["R"] * data["omega_z"] * data["omega_tt"]
            + data["R_ttz"],
            2 * data["omega_t"] * data["R_tz"]
            + 2 * data["R_t"] * data["omega_tz"]
            + data["R_tt"]
            + data["omega_z"] * data["R_tt"]
            + data["R_z"] * data["omega_tt"]
            - data["R"]
            * ((1 + data["omega_z"]) * data["omega_t"] ** 2 - data["omega_ttz"]),
            data["Z_ttz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_tz",
    label="\\partial_{\\theta}{\\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description=(
        "Covariant Toroidal basis vector, second derivative wrt poloidal and toroidal"
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
)
def _e_sub_zeta_tz(params, transforms, profiles, data, **kwargs):
    data["e_zeta_tz"] = jnp.array(
        [
            -((1 + data["omega_z"]) ** 2) * data["R_t"]
            - 2 * data["R_z"] * (1 + data["omega_z"]) * data["omega_t"]
            - data["R"] * data["omega_zz"] * data["omega_t"]
            - 2 * data["R"] * data["omega_tz"]
            - 2 * data["R"] * data["omega_z"] * data["omega_tz"]
            + data["R_tzz"],
            data["omega_zz"] * data["R_t"]
            + data["R_zz"] * data["omega_t"]
            + 2 * data["R_tz"]
            + 2 * data["omega_z"] * data["R_tz"]
            + 2 * data["R_z"] * data["omega_tz"]
            - data["R"]
            * ((1 + data["omega_z"]) ** 2 * data["omega_t"] - data["omega_tzz"]),
            data["Z_tzz"],
        ]
    ).T
    return data


@register_compute_fun(
    name="e_zeta_zz",
    label="\\partial_{\\zeta}{\\zeta} \\mathbf{e}_{\\zeta}",
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
    data=["R", "R_z", "R_zz", "R_zzz", "Z_zzz", "omega_z", "omega_zz", "omega_zzz"],
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
    data=[
        "e_theta",
        "theta_PEST_t",
    ],
)
def _e_sub_theta_pest(params, transforms, profiles, data, **kwargs):
    # dX/dv at const r,z = dX/dt * dt/dv / dX/dt / dv/dt
    data["e_theta_PEST"] = (data["e_theta"].T / data["theta_PEST_t"]).T
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
    name="n",
    label="\\hat{n}",
    units="~",
    units_long="None",
    description="Unit vector normal to flux surface",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["e^rho"],
)
def _n(params, transforms, profiles, data, **kwargs):
    data["n"] = (data["e^rho"].T / jnp.linalg.norm(data["e^rho"], axis=-1)).T
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
