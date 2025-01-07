"""Core compute functions, for polar, flux, and cartesian coordinates."""

from desc.backend import jnp

from .data_index import register_compute_fun


@register_compute_fun(
    name="0",
    label="0",
    units="~",
    units_long="None",
    description="Zeros",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
        "desc.geometry.core.Curve",
    ],
    aliases=["rho_t", "rho_z", "theta_r", "theta_z", "zeta_r", "zeta_t"],
)
def _0(params, transforms, profiles, data, **kwargs):
    data["0"] = jnp.zeros(transforms["grid"].num_nodes)
    return data


@register_compute_fun(
    name="1",
    label="1",
    units="~",
    units_long="None",
    description="Ones",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
        "desc.geometry.core.Curve",
    ],
    aliases=["rho_r", "theta_t", "zeta_z"],
)
def _1(params, transforms, profiles, data, **kwargs):
    data["1"] = jnp.ones(transforms["grid"].num_nodes)
    return data


@register_compute_fun(
    name="R",
    label="R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R(params, transforms, profiles, data, **kwargs):
    data["R"] = transforms["R"].transform(params["R_lmn"], 0, 0, 0)
    return data


@register_compute_fun(
    name="R_r",
    label="\\partial_{\\rho} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, first radial derivative",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[1, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _R_r(params, transforms, profiles, data, **kwargs):
    data["R_r"] = transforms["R"].transform(params["R_lmn"], 1, 0, 0)
    return data


@register_compute_fun(
    name="R_rr",
    label="\\partial_{\\rho \\rho} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, second radial derivative",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[2, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _R_rr(params, transforms, profiles, data, **kwargs):
    data["R_rr"] = transforms["R"].transform(params["R_lmn"], 2, 0, 0)
    return data


@register_compute_fun(
    name="R_rrr",
    label="\\partial_{\\rho \\rho \\rho} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third radial derivative",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[3, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _R_rrr(params, transforms, profiles, data, **kwargs):
    data["R_rrr"] = transforms["R"].transform(params["R_lmn"], 3, 0, 0)
    return data


@register_compute_fun(
    name="R_rrrr",
    label="\\partial_{\\rho \\rho \\rho \\rho} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, fourth radial derivative",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[4, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _R_rrrr(params, transforms, profiles, data, **kwargs):
    data["R_rrrr"] = transforms["R"].transform(params["R_lmn"], 4, 0, 0)
    return data


@register_compute_fun(
    name="R_rrrt",
    label="\\partial_{\\rho \\rho \\rho \\theta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, fourth derivative wrt"
    " radial coordinate thrice and poloidal once",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[3, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _R_rrrt(params, transforms, profiles, data, **kwargs):
    data["R_rrrt"] = transforms["R"].transform(params["R_lmn"], 3, 1, 0)
    return data


@register_compute_fun(
    name="R_rrrz",
    label="\\partial_{\\rho \\rho \\rho \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, fourth derivative wrt"
    " radial coordinate thrice and toroidal once",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[3, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rrrz(params, transforms, profiles, data, **kwargs):
    data["R_rrrz"] = transforms["R"].transform(params["R_lmn"], 3, 0, 1)
    return data


@register_compute_fun(
    name="R_rrt",
    label="\\partial_{\\rho \\rho \\theta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative, wrt radius twice "
    "and poloidal angle",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[2, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _R_rrt(params, transforms, profiles, data, **kwargs):
    data["R_rrt"] = transforms["R"].transform(params["R_lmn"], 2, 1, 0)
    return data


@register_compute_fun(
    name="R_rrtt",
    label="\\partial_{\\rho \\rho \\theta \\theta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, fourth derivative, wrt radius twice "
    "and poloidal angle twice",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[2, 2, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _R_rrtt(params, transforms, profiles, data, **kwargs):
    data["R_rrtt"] = transforms["R"].transform(params["R_lmn"], 2, 2, 0)
    return data


@register_compute_fun(
    name="R_rrtz",
    label="\\partial_{\\rho \\rho \\theta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, fourth derivative wrt radius twice,"
    " poloidal angle, and toroidal angle",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[2, 1, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _R_rrtz(params, transforms, profiles, data, **kwargs):
    data["R_rrtz"] = transforms["R"].transform(params["R_lmn"], 2, 1, 1)
    return data


@register_compute_fun(
    name="R_rrz",
    label="\\partial_{\\rho \\rho \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative, wrt radius twice "
    "and toroidal angle",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[2, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rrz(params, transforms, profiles, data, **kwargs):
    data["R_rrz"] = transforms["R"].transform(params["R_lmn"], 2, 0, 1)
    return data


@register_compute_fun(
    name="R_rrzz",
    label="\\partial_{\\rho \\rho \\zeta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, fourth derivative, wrt radius twice "
    "and toroidal angle twice",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[2, 0, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rrzz(params, transforms, profiles, data, **kwargs):
    data["R_rrzz"] = transforms["R"].transform(params["R_lmn"], 2, 0, 2)
    return data


@register_compute_fun(
    name="R_rt",
    label="\\partial_{\\rho \\theta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, second derivative wrt radius "
    "and poloidal angle",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[1, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _R_rt(params, transforms, profiles, data, **kwargs):
    data["R_rt"] = transforms["R"].transform(params["R_lmn"], 1, 1, 0)
    return data


@register_compute_fun(
    name="R_rtt",
    label="\\partial_{\\rho \\theta \\theta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative wrt radius and "
    "poloidal angle twice",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[1, 2, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _R_rtt(params, transforms, profiles, data, **kwargs):
    data["R_rtt"] = transforms["R"].transform(params["R_lmn"], 1, 2, 0)
    return data


@register_compute_fun(
    name="R_rttt",
    label="\\partial_{\\rho \\theta \\theta \\theta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, fourth derivative wrt radius and "
    "poloidal angle thrice",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[1, 3, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _R_rttt(params, transforms, profiles, data, **kwargs):
    data["R_rttt"] = transforms["R"].transform(params["R_lmn"], 1, 3, 0)
    return data


@register_compute_fun(
    name="R_rttz",
    label="\\partial_{\\rho \\theta \\theta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, fourth derivative wrt radius once, "
    "poloidal angle twice, and toroidal angle once",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[1, 2, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rttz(params, transforms, profiles, data, **kwargs):
    data["R_rttz"] = transforms["R"].transform(params["R_lmn"], 1, 2, 1)
    return data


@register_compute_fun(
    name="R_rtz",
    label="\\partial_{\\rho \\theta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative wrt radius, poloidal "
    "angle, and toroidal angle",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[1, 1, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rtz(params, transforms, profiles, data, **kwargs):
    data["R_rtz"] = transforms["R"].transform(params["R_lmn"], 1, 1, 1)
    return data


@register_compute_fun(
    name="R_rtzz",
    label="\\partial_{\\rho \\theta \\zeta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, fourth derivative wrt radius, poloidal "
    "angle, and toroidal angle twice",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[1, 1, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rtzz(params, transforms, profiles, data, **kwargs):
    data["R_rtzz"] = transforms["R"].transform(params["R_lmn"], 1, 1, 2)
    return data


@register_compute_fun(
    name="R_rz",
    label="\\partial_{\\rho \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, second derivative wrt radius "
    "and toroidal angle",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[1, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rz(params, transforms, profiles, data, **kwargs):
    data["R_rz"] = transforms["R"].transform(params["R_lmn"], 1, 0, 1)
    return data


@register_compute_fun(
    name="R_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative wrt radius and "
    "toroidal angle twice",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[1, 0, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rzz(params, transforms, profiles, data, **kwargs):
    data["R_rzz"] = transforms["R"].transform(params["R_lmn"], 1, 0, 2)
    return data


@register_compute_fun(
    name="R_rzzz",
    label="\\partial_{\\rho \\zeta \\zeta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, fourth derivative wrt radius and "
    "toroidal angle thrice",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[1, 0, 3]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rzzz(params, transforms, profiles, data, **kwargs):
    data["R_rzzz"] = transforms["R"].transform(params["R_lmn"], 1, 0, 3)
    return data


@register_compute_fun(
    name="R_t",
    label="\\partial_{\\theta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, first poloidal derivative",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _R_t(params, transforms, profiles, data, **kwargs):
    data["R_t"] = transforms["R"].transform(params["R_lmn"], 0, 1, 0)
    return data


@register_compute_fun(
    name="R_tt",
    label="\\partial_{\\theta \\theta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, second poloidal derivative",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 2, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _R_tt(params, transforms, profiles, data, **kwargs):
    data["R_tt"] = transforms["R"].transform(params["R_lmn"], 0, 2, 0)
    return data


@register_compute_fun(
    name="R_ttt",
    label="\\partial_{\\theta \\theta \\theta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third poloidal derivative",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 3, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _R_ttt(params, transforms, profiles, data, **kwargs):
    data["R_ttt"] = transforms["R"].transform(params["R_lmn"], 0, 3, 0)
    return data


@register_compute_fun(
    name="R_ttz",
    label="\\partial_{\\theta \\theta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative wrt poloidal angle "
    "twice and toroidal angle",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 2, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _R_ttz(params, transforms, profiles, data, **kwargs):
    data["R_ttz"] = transforms["R"].transform(params["R_lmn"], 0, 2, 1)
    return data


@register_compute_fun(
    name="R_tz",
    label="\\partial_{\\theta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, second derivative wrt poloidal "
    "and toroidal angles",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 1, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _R_tz(params, transforms, profiles, data, **kwargs):
    data["R_tz"] = transforms["R"].transform(params["R_lmn"], 0, 1, 1)
    return data


@register_compute_fun(
    name="R_tzz",
    label="\\partial_{\\theta \\zeta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative wrt poloidal angle "
    "and toroidal angle twice",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 1, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _R_tzz(params, transforms, profiles, data, **kwargs):
    data["R_tzz"] = transforms["R"].transform(params["R_lmn"], 0, 1, 2)
    return data


@register_compute_fun(
    name="R_z",
    label="\\partial_{\\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, first toroidal derivative",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _R_z(params, transforms, profiles, data, **kwargs):
    data["R_z"] = transforms["R"].transform(params["R_lmn"], 0, 0, 1)
    return data


@register_compute_fun(
    name="R_zz",
    label="\\partial_{\\zeta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, second toroidal derivative",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 0, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _R_zz(params, transforms, profiles, data, **kwargs):
    data["R_zz"] = transforms["R"].transform(params["R_lmn"], 0, 0, 2)
    return data


@register_compute_fun(
    name="R_zzz",
    label="\\partial_{\\zeta \\zeta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third toroidal derivative",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 0, 3]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _R_zzz(params, transforms, profiles, data, **kwargs):
    data["R_zzz"] = transforms["R"].transform(params["R_lmn"], 0, 0, 3)
    return data


@register_compute_fun(
    name="X",
    label="X = R \\cos{\\phi}",
    units="m",
    units_long="meters",
    description="Cartesian X coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "phi"],
)
def _X(params, transforms, profiles, data, **kwargs):
    data["X"] = data["R"] * jnp.cos(data["phi"])
    return data


@register_compute_fun(
    name="X_r",
    label="\\partial_{\\rho} X",
    units="m",
    units_long="meters",
    description="Cartesian X coordinate, derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_r", "phi", "phi_r"],
)
def _X_r(params, transforms, profiles, data, **kwargs):
    data["X_r"] = (
        data["R_r"] * jnp.cos(data["phi"])
        - data["R"] * jnp.sin(data["phi"]) * data["phi_r"]
    )
    return data


@register_compute_fun(
    name="X_t",
    label="\\partial_{\\theta} X",
    units="m",
    units_long="meters",
    description="Cartesian X coordinate, derivative wrt poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_t", "phi", "phi_t"],
)
def _X_t(params, transforms, profiles, data, **kwargs):
    data["X_t"] = (
        data["R_t"] * jnp.cos(data["phi"])
        - data["R"] * jnp.sin(data["phi"]) * data["phi_t"]
    )
    return data


@register_compute_fun(
    name="X_z",
    label="\\partial_{\\zeta} X",
    units="m",
    units_long="meters",
    description="Cartesian X coordinate, derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_z", "phi", "phi_z"],
)
def _X_z(params, transforms, profiles, data, **kwargs):
    data["X_z"] = (
        data["R_z"] * jnp.cos(data["phi"])
        - data["R"] * jnp.sin(data["phi"]) * data["phi_z"]
    )
    return data


@register_compute_fun(
    name="Y",
    label="Y = R \\sin{\\phi}",
    units="m",
    units_long="meters",
    description="Cartesian Y coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "phi"],
)
def _Y(params, transforms, profiles, data, **kwargs):
    data["Y"] = data["R"] * jnp.sin(data["phi"])
    return data


@register_compute_fun(
    name="Y_r",
    label="\\partial_{\\rho} Y",
    units="m",
    units_long="meters",
    description="Cartesian Y coordinate, derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_r", "phi", "phi_r"],
)
def _Y_r(params, transforms, profiles, data, **kwargs):
    data["Y_r"] = (
        data["R_r"] * jnp.sin(data["phi"])
        + data["R"] * jnp.cos(data["phi"]) * data["phi_r"]
    )
    return data


@register_compute_fun(
    name="Y_t",
    label="\\partial_{\\theta} Y",
    units="m",
    units_long="meters",
    description="Cartesian Y coordinate, derivative wrt poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_t", "phi", "phi_t"],
)
def _Y_t(params, transforms, profiles, data, **kwargs):
    data["Y_t"] = (
        data["R_t"] * jnp.sin(data["phi"])
        + data["R"] * jnp.cos(data["phi"]) * data["phi_t"]
    )
    return data


@register_compute_fun(
    name="Y_z",
    label="\\partial_{\\zeta} Y",
    units="m",
    units_long="meters",
    description="Cartesian Y coordinate, derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["R", "R_z", "phi", "phi_z"],
)
def _Y_z(params, transforms, profiles, data, **kwargs):
    data["Y_z"] = (
        data["R_z"] * jnp.sin(data["phi"])
        + data["R"] * jnp.cos(data["phi"]) * data["phi_z"]
    )
    return data


@register_compute_fun(
    name="Z",
    label="Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z(params, transforms, profiles, data, **kwargs):
    data["Z"] = transforms["Z"].transform(params["Z_lmn"], 0, 0, 0)
    return data


@register_compute_fun(
    name="Z_r",
    label="\\partial_{\\rho} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, first radial derivative",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[1, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _Z_r(params, transforms, profiles, data, **kwargs):
    data["Z_r"] = transforms["Z"].transform(params["Z_lmn"], 1, 0, 0)
    return data


@register_compute_fun(
    name="Z_rr",
    label="\\partial_{\\rho \\rho} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, second radial derivative",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[2, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _Z_rr(params, transforms, profiles, data, **kwargs):
    data["Z_rr"] = transforms["Z"].transform(params["Z_lmn"], 2, 0, 0)
    return data


@register_compute_fun(
    name="Z_rrr",
    label="\\partial_{\\rho \\rho \\rho} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third radial derivative",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[3, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _Z_rrr(params, transforms, profiles, data, **kwargs):
    data["Z_rrr"] = transforms["Z"].transform(params["Z_lmn"], 3, 0, 0)
    return data


@register_compute_fun(
    name="Z_rrrr",
    label="\\partial_{\\rho \\rho \\rho \\rho} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, fourth radial derivative",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[4, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _Z_rrrr(params, transforms, profiles, data, **kwargs):
    data["Z_rrrr"] = transforms["Z"].transform(params["Z_lmn"], 4, 0, 0)
    return data


@register_compute_fun(
    name="Z_rrrt",
    label="\\partial_{\\rho \\rho \\rho \\theta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, fourth derivative wrt "
    " radial coordinate thrice and poloidal once",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[3, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _Z_rrrt(params, transforms, profiles, data, **kwargs):
    data["Z_rrrt"] = transforms["Z"].transform(params["Z_lmn"], 3, 1, 0)
    return data


@register_compute_fun(
    name="Z_rrrz",
    label="\\partial_{\\rho \\rho \\rho \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, fourth derivative wrt "
    " radial coordinate thrice and toroidal once",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[3, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rrrz(params, transforms, profiles, data, **kwargs):
    data["Z_rrrz"] = transforms["Z"].transform(params["Z_lmn"], 3, 0, 1)
    return data


@register_compute_fun(
    name="Z_rrt",
    label="\\partial_{\\rho \\rho \\theta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative, wrt radius "
    "twice and poloidal angle",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[2, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _Z_rrt(params, transforms, profiles, data, **kwargs):
    data["Z_rrt"] = transforms["Z"].transform(params["Z_lmn"], 2, 1, 0)
    return data


@register_compute_fun(
    name="Z_rrtt",
    label="\\partial_{\\rho \\rho \\theta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, fourth derivative, wrt radius "
    "twice and poloidal angle twice",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[2, 2, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _Z_rrtt(params, transforms, profiles, data, **kwargs):
    data["Z_rrtt"] = transforms["Z"].transform(params["Z_lmn"], 2, 2, 0)
    return data


@register_compute_fun(
    name="Z_rrtz",
    label="\\partial_{\\rho \\rho \\theta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, fourth derivative wrt radius"
    "twice, poloidal angle, and toroidal angle",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[2, 1, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rrtz(params, transforms, profiles, data, **kwargs):
    data["Z_rrtz"] = transforms["Z"].transform(params["Z_lmn"], 2, 1, 1)
    return data


@register_compute_fun(
    name="Z_rrz",
    label="\\partial_{\\rho \\rho \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative, wrt radius "
    "twice and toroidal angle",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[2, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rrz(params, transforms, profiles, data, **kwargs):
    data["Z_rrz"] = transforms["Z"].transform(params["Z_lmn"], 2, 0, 1)
    return data


@register_compute_fun(
    name="Z_rrzz",
    label="\\partial_{\\rho \\rho \\zeta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, fourth derivative, wrt radius "
    "twice and toroidal angle twice",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[2, 0, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rrzz(params, transforms, profiles, data, **kwargs):
    data["Z_rrzz"] = transforms["Z"].transform(params["Z_lmn"], 2, 0, 2)
    return data


@register_compute_fun(
    name="Z_rt",
    label="\\partial_{\\rho \\theta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, second derivative wrt radius "
    "and poloidal angle",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[1, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _Z_rt(params, transforms, profiles, data, **kwargs):
    data["Z_rt"] = transforms["Z"].transform(params["Z_lmn"], 1, 1, 0)
    return data


@register_compute_fun(
    name="Z_rtt",
    label="\\partial_{\\rho \\theta \\theta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative wrt radius "
    "and poloidal angle twice",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[1, 2, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _Z_rtt(params, transforms, profiles, data, **kwargs):
    data["Z_rtt"] = transforms["Z"].transform(params["Z_lmn"], 1, 2, 0)
    return data


@register_compute_fun(
    name="Z_rttt",
    label="\\partial_{\\rho \\theta \\theta \\theta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative wrt radius "
    "and poloidal angle thrice",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[1, 3, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
)
def _Z_rttt(params, transforms, profiles, data, **kwargs):
    data["Z_rttt"] = transforms["Z"].transform(params["Z_lmn"], 1, 3, 0)
    return data


@register_compute_fun(
    name="Z_rttz",
    label="\\partial_{\\rho \\theta \\theta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, fourth derivative wrt radius "
    "once, poloidal angle twice, and toroidal angle once",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[1, 2, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rttz(params, transforms, profiles, data, **kwargs):
    data["Z_rttz"] = transforms["Z"].transform(params["Z_lmn"], 1, 2, 1)
    return data


@register_compute_fun(
    name="Z_rtz",
    label="\\partial_{\\rho \\theta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative wrt radius, "
    "poloidal angle, and toroidal angle",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[1, 1, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rtz(params, transforms, profiles, data, **kwargs):
    data["Z_rtz"] = transforms["Z"].transform(params["Z_lmn"], 1, 1, 1)
    return data


@register_compute_fun(
    name="Z_rtzz",
    label="\\partial_{\\rho \\theta \\zeta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, fourth derivative wrt radius, "
    "poloidal angle, and toroidal angle twice",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[1, 1, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rtzz(params, transforms, profiles, data, **kwargs):
    data["Z_rtzz"] = transforms["Z"].transform(params["Z_lmn"], 1, 1, 2)
    return data


@register_compute_fun(
    name="Z_rz",
    label="\\partial_{\\rho \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, second derivative wrt radius "
    "and toroidal angle",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[1, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rz(params, transforms, profiles, data, **kwargs):
    data["Z_rz"] = transforms["Z"].transform(params["Z_lmn"], 1, 0, 1)
    return data


@register_compute_fun(
    name="Z_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative wrt radius "
    "and toroidal angle twice",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[1, 0, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rzz(params, transforms, profiles, data, **kwargs):
    data["Z_rzz"] = transforms["Z"].transform(params["Z_lmn"], 1, 0, 2)
    return data


@register_compute_fun(
    name="Z_rzzz",
    label="\\partial_{\\rho \\zeta \\zeta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative wrt radius "
    "and toroidal angle thrice",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[1, 0, 3]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rzzz(params, transforms, profiles, data, **kwargs):
    data["Z_rzzz"] = transforms["Z"].transform(params["Z_lmn"], 1, 0, 3)
    return data


@register_compute_fun(
    name="Z_t",
    label="\\partial_{\\theta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, first poloidal derivative",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _Z_t(params, transforms, profiles, data, **kwargs):
    data["Z_t"] = transforms["Z"].transform(params["Z_lmn"], 0, 1, 0)
    return data


@register_compute_fun(
    name="Z_tt",
    label="\\partial_{\\theta \\theta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, second poloidal derivative",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 2, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _Z_tt(params, transforms, profiles, data, **kwargs):
    data["Z_tt"] = transforms["Z"].transform(params["Z_lmn"], 0, 2, 0)
    return data


@register_compute_fun(
    name="Z_ttt",
    label="\\partial_{\\theta \\theta \\theta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third poloidal derivative",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 3, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _Z_ttt(params, transforms, profiles, data, **kwargs):
    data["Z_ttt"] = transforms["Z"].transform(params["Z_lmn"], 0, 3, 0)
    return data


@register_compute_fun(
    name="Z_ttz",
    label="\\partial_{\\theta \\theta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative wrt poloidal "
    "angle twice and toroidal angle",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 2, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _Z_ttz(params, transforms, profiles, data, **kwargs):
    data["Z_ttz"] = transforms["Z"].transform(params["Z_lmn"], 0, 2, 1)
    return data


@register_compute_fun(
    name="Z_tz",
    label="\\partial_{\\theta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, second derivative wrt poloidal "
    "and toroidal angles",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 1, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _Z_tz(params, transforms, profiles, data, **kwargs):
    data["Z_tz"] = transforms["Z"].transform(params["Z_lmn"], 0, 1, 1)
    return data


@register_compute_fun(
    name="Z_tzz",
    label="\\partial_{\\theta \\zeta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative wrt poloidal "
    "angle and toroidal angle twice",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 1, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _Z_tzz(params, transforms, profiles, data, **kwargs):
    data["Z_tzz"] = transforms["Z"].transform(params["Z_lmn"], 0, 1, 2)
    return data


@register_compute_fun(
    name="Z_z",
    label="\\partial_{\\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, first toroidal derivative",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _Z_z(params, transforms, profiles, data, **kwargs):
    data["Z_z"] = transforms["Z"].transform(params["Z_lmn"], 0, 0, 1)
    return data


@register_compute_fun(
    name="Z_zz",
    label="\\partial_{\\zeta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, second toroidal derivative",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 0, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _Z_zz(params, transforms, profiles, data, **kwargs):
    data["Z_zz"] = transforms["Z"].transform(params["Z_lmn"], 0, 0, 2)
    return data


@register_compute_fun(
    name="Z_zzz",
    label="\\partial_{\\zeta \\zeta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third toroidal derivative",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 0, 3]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.surface.FourierRZToroidalSurface",
    ],
)
def _Z_zzz(params, transforms, profiles, data, **kwargs):
    data["Z_zzz"] = transforms["Z"].transform(params["Z_lmn"], 0, 0, 3)
    return data


@register_compute_fun(
    name="alpha",
    label="\\alpha",
    units="~",
    units_long="None",
    description="Field line label",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta_PEST", "phi", "iota"],
)
def _alpha(params, transforms, profiles, data, **kwargs):
    data["alpha"] = data["theta_PEST"] - data["iota"] * data["phi"]
    return data


@register_compute_fun(
    name="alpha_r",
    label="\\partial_\\rho \\alpha",
    units="~",
    units_long="None",
    description="Field line label, derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["alpha_r (periodic)", "alpha_r (secular)"],
)
def _alpha_r(params, transforms, profiles, data, **kwargs):
    data["alpha_r"] = data["alpha_r (periodic)"] + data["alpha_r (secular)"]
    return data


@register_compute_fun(
    name="alpha_r (periodic)",
    label="\\mathrm{periodic}(\\partial_\\rho \\alpha)",
    units="~",
    units_long="None",
    description="Field line label, derivative wrt radial coordinate, "
    "periodic component",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta_PEST_r", "iota", "phi_r"],
)
def _periodic_alpha_r(params, transforms, profiles, data, **kwargs):
    data["alpha_r (periodic)"] = data["theta_PEST_r"] - data["iota"] * data["phi_r"]
    return data


@register_compute_fun(
    name="alpha_r (secular)",
    label="\\mathrm{secular}(\\partial_\\rho \\alpha)",
    units="~",
    units_long="None",
    description="Field line label, derivative wrt radial coordinate, "
    "secular component",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["iota_r", "phi"],
)
def _secular_alpha_r(params, transforms, profiles, data, **kwargs):
    data["alpha_r (secular)"] = -data["iota_r"] * data["phi"]
    return data


@register_compute_fun(
    name="alpha_t",
    label="\\partial_\\theta \\alpha",
    units="~",
    units_long="None",
    description="Field line label, derivative wrt poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta_PEST_t", "phi_t", "iota"],
)
def _alpha_t(params, transforms, profiles, data, **kwargs):
    data["alpha_t"] = data["theta_PEST_t"] - data["iota"] * data["phi_t"]
    return data


@register_compute_fun(
    name="alpha_tz",
    label="\\partial_{\\theta \\zeta} \\alpha",
    units="~",
    units_long="None",
    description="Field line label, derivative wrt poloidal and toroidal coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta_PEST_tz", "phi_tz", "iota"],
)
def _alpha_tz(params, transforms, profiles, data, **kwargs):
    data["alpha_tz"] = data["theta_PEST_tz"] - data["iota"] * data["phi_tz"]
    return data


@register_compute_fun(
    name="alpha_z",
    label="\\partial_\\zeta \\alpha",
    units="~",
    units_long="None",
    description="Field line label, derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta_PEST_z", "phi_z", "iota"],
)
def _alpha_z(params, transforms, profiles, data, **kwargs):
    data["alpha_z"] = data["theta_PEST_z"] - data["iota"] * data["phi_z"]
    return data


@register_compute_fun(
    name="alpha_tt",
    label="\\partial_{\\theta \\theta} \\alpha",
    units="~",
    units_long="None",
    description="Field line label, second-order derivative wrt poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["phi_tt", "iota", "theta_PEST_tt"],
)
def _alpha_tt(params, transforms, profiles, data, **kwargs):
    data["alpha_tt"] = data["theta_PEST_tt"] - data["iota"] * data["phi_tt"]
    return data


@register_compute_fun(
    name="alpha_zz",
    label="\\partial_{\\zeta \\zeta} \\alpha",
    units="~",
    units_long="None",
    description="Field line label, second-order derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta_PEST_zz", "phi_zz", "iota"],
)
def _alpha_zz(params, transforms, profiles, data, **kwargs):
    data["alpha_zz"] = data["theta_PEST_zz"] - data["iota"] * data["phi_zz"]
    return data


@register_compute_fun(
    name="lambda",
    label="\\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda(params, transforms, profiles, data, **kwargs):
    data["lambda"] = transforms["L"].transform(params["L_lmn"], 0, 0, 0)
    return data


@register_compute_fun(
    name="lambda_r",
    label="\\partial_{\\rho} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, first radial derivative",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[1, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_r(params, transforms, profiles, data, **kwargs):
    data["lambda_r"] = transforms["L"].transform(params["L_lmn"], 1, 0, 0)
    return data


@register_compute_fun(
    name="lambda_rr",
    label="\\partial_{\\rho \\rho} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, second radial derivative",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[2, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_rr(params, transforms, profiles, data, **kwargs):
    data["lambda_rr"] = transforms["L"].transform(params["L_lmn"], 2, 0, 0)
    return data


@register_compute_fun(
    name="lambda_rrr",
    label="\\partial_{\\rho \\rho \\rho} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third radial derivative",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[3, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_rrr(params, transforms, profiles, data, **kwargs):
    data["lambda_rrr"] = transforms["L"].transform(params["L_lmn"], 3, 0, 0)
    return data


@register_compute_fun(
    name="lambda_rrrt",
    label="\\partial_{\\rho \\rho \\rho \\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third radial derivative and"
    " first poloidal derivative",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[3, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_rrrt(params, transforms, profiles, data, **kwargs):
    data["lambda_rrrt"] = transforms["L"].transform(params["L_lmn"], 3, 1, 0)
    return data


@register_compute_fun(
    name="lambda_rrrz",
    label="\\partial_{\\rho \\rho \\rho \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third radial derivative and"
    " first toroidal derivative",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[3, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_rrrz(params, transforms, profiles, data, **kwargs):
    data["lambda_rrrz"] = transforms["L"].transform(params["L_lmn"], 3, 0, 1)
    return data


@register_compute_fun(
    name="lambda_rrt",
    label="\\partial_{\\rho \\rho \\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative, wrt radius twice "
    "and poloidal angle",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[2, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_rrt(params, transforms, profiles, data, **kwargs):
    data["lambda_rrt"] = transforms["L"].transform(params["L_lmn"], 2, 1, 0)
    return data


@register_compute_fun(
    name="lambda_rrz",
    label="\\partial_{\\rho \\rho \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative, wrt radius twice "
    "and toroidal angle",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[2, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_rrz(params, transforms, profiles, data, **kwargs):
    data["lambda_rrz"] = transforms["L"].transform(params["L_lmn"], 2, 0, 1)
    return data


@register_compute_fun(
    name="lambda_rt",
    label="\\partial_{\\rho \\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, second derivative wrt radius and "
    "poloidal angle",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[1, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_rt(params, transforms, profiles, data, **kwargs):
    data["lambda_rt"] = transforms["L"].transform(params["L_lmn"], 1, 1, 0)
    return data


@register_compute_fun(
    name="lambda_rtt",
    label="\\partial_{\\rho \\theta \\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative wrt radius and "
    "poloidal angle twice",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[1, 2, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_rtt(params, transforms, profiles, data, **kwargs):
    data["lambda_rtt"] = transforms["L"].transform(params["L_lmn"], 1, 2, 0)
    return data


@register_compute_fun(
    name="lambda_rtz",
    label="\\partial_{\\rho \\theta \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative wrt radius, poloidal "
    " angle, and toroidal angle",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[1, 1, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_rtz(params, transforms, profiles, data, **kwargs):
    data["lambda_rtz"] = transforms["L"].transform(params["L_lmn"], 1, 1, 1)
    return data


@register_compute_fun(
    name="lambda_rz",
    label="\\partial_{\\rho \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, second derivative wrt radius and "
    "toroidal angle",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[1, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_rz(params, transforms, profiles, data, **kwargs):
    data["lambda_rz"] = transforms["L"].transform(params["L_lmn"], 1, 0, 1)
    return data


@register_compute_fun(
    name="lambda_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative wrt radius and "
    "toroidal angle twice",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[1, 0, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_rzz(params, transforms, profiles, data, **kwargs):
    data["lambda_rzz"] = transforms["L"].transform(params["L_lmn"], 1, 0, 2)
    return data


@register_compute_fun(
    name="lambda_t",
    label="\\partial_{\\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, first poloidal derivative",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_t(params, transforms, profiles, data, **kwargs):
    data["lambda_t"] = transforms["L"].transform(params["L_lmn"], 0, 1, 0)
    return data


@register_compute_fun(
    name="lambda_tt",
    label="\\partial_{\\theta \\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, second poloidal derivative",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 2, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_tt(params, transforms, profiles, data, **kwargs):
    data["lambda_tt"] = transforms["L"].transform(params["L_lmn"], 0, 2, 0)
    return data


@register_compute_fun(
    name="lambda_ttt",
    label="\\partial_{\\theta \\theta \\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third poloidal derivative",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 3, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_ttt(params, transforms, profiles, data, **kwargs):
    data["lambda_ttt"] = transforms["L"].transform(params["L_lmn"], 0, 3, 0)
    return data


@register_compute_fun(
    name="lambda_ttz",
    label="\\partial_{\\theta \\theta \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative wrt poloidal angle "
    "twice and toroidal angle",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 2, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_ttz(params, transforms, profiles, data, **kwargs):
    data["lambda_ttz"] = transforms["L"].transform(params["L_lmn"], 0, 2, 1)
    return data


@register_compute_fun(
    name="lambda_tz",
    label="\\partial_{\\theta \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, second derivative wrt poloidal and "
    "toroidal angles",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 1, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_tz(params, transforms, profiles, data, **kwargs):
    data["lambda_tz"] = transforms["L"].transform(params["L_lmn"], 0, 1, 1)
    return data


@register_compute_fun(
    name="lambda_tzz",
    label="\\partial_{\\theta \\zeta \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative wrt poloidal angle "
    "and toroidal angle twice",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 1, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_tzz(params, transforms, profiles, data, **kwargs):
    data["lambda_tzz"] = transforms["L"].transform(params["L_lmn"], 0, 1, 2)
    return data


@register_compute_fun(
    name="lambda_z",
    label="\\partial_{\\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, first toroidal derivative",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_z(params, transforms, profiles, data, **kwargs):
    data["lambda_z"] = transforms["L"].transform(params["L_lmn"], 0, 0, 1)
    return data


@register_compute_fun(
    name="lambda_zz",
    label="\\partial_{\\zeta \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, second toroidal derivative",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 0, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_zz(params, transforms, profiles, data, **kwargs):
    data["lambda_zz"] = transforms["L"].transform(params["L_lmn"], 0, 0, 2)
    return data


@register_compute_fun(
    name="lambda_zzz",
    label="\\partial_{\\zeta \\zeta \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third toroidal derivative",
    dim=1,
    params=["L_lmn"],
    transforms={"L": [[0, 0, 3]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _lambda_zzz(params, transforms, profiles, data, **kwargs):
    data["lambda_zzz"] = transforms["L"].transform(params["L_lmn"], 0, 0, 3)
    return data


@register_compute_fun(
    name="omega",
    label="\\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function",
    dim=1,
    params=[],  # ["W_lmn"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega(params, transforms, profiles, data, **kwargs):
    data["omega"] = data["0"]
    return data


@register_compute_fun(
    name="omega_r",
    label="\\partial_{\\rho} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, first radial derivative",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_r(params, transforms, profiles, data, **kwargs):
    data["omega_r"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rr",
    label="\\partial_{\\rho \\rho} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, second radial derivative",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rr(params, transforms, profiles, data, **kwargs):
    data["omega_rr"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rrr",
    label="\\partial_{\\rho \\rho \\rho} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, third radial derivative",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rrr(params, transforms, profiles, data, **kwargs):
    data["omega_rrr"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rrrr",
    label="\\partial_{\\rho \\rho \\rho \\rho} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, fourth radial derivative",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rrrr(params, transforms, profiles, data, **kwargs):
    data["omega_rrrr"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rrrt",
    label="\\partial_{\\rho \\rho \\rho \\theta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, fourth derivative wrt radial coordinate"
    " thrice and poloidal once",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rrrt(params, transforms, profiles, data, **kwargs):
    data["omega_rrrt"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rrrz",
    label="\\partial_{\\rho \\rho \\rho \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, fourth derivative wrt radial coordinate"
    " thrice and toroidal once",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rrrz(params, transforms, profiles, data, **kwargs):
    data["omega_rrrz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rrt",
    label="\\partial_{\\rho \\rho \\theta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, third derivative, wrt radius twice "
    "and poloidal angle",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rrt(params, transforms, profiles, data, **kwargs):
    data["omega_rrt"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rrtt",
    label="\\partial_{\\rho \\rho \\theta \\theta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, fourth derivative, wrt radius twice "
    "and poloidal angle twice",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rrtt(params, transforms, profiles, data, **kwargs):
    data["omega_rrtt"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rrtz",
    label="\\partial_{\\rho \\theta \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, fourth derivative wrt radius twice,"
    " poloidal angle, and toroidal angle",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rrtz(params, transforms, profiles, data, **kwargs):
    data["omega_rrtz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rrz",
    label="\\partial_{\\rho \\rho \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, third derivative, wrt radius twice "
    "and toroidal angle",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rrz(params, transforms, profiles, data, **kwargs):
    data["omega_rrz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rrzz",
    label="\\partial_{\\rho \\rho \\zeta \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, fourth derivative, wrt radius twice "
    "and toroidal angle twice",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rrzz(params, transforms, profiles, data, **kwargs):
    data["omega_rrzz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rt",
    label="\\partial_{\\rho \\theta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, second derivative wrt radius and "
    "poloidal angle",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rt(params, transforms, profiles, data, **kwargs):
    data["omega_rt"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rtt",
    label="\\partial_{\\rho \\theta \\theta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, third derivative wrt radius and "
    "poloidal angle twice",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rtt(params, transforms, profiles, data, **kwargs):
    data["omega_rtt"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rttt",
    label="\\partial_{\\rho \\theta \\theta \\theta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, third derivative wrt radius and "
    "poloidal angle thrice",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rttt(params, transforms, profiles, data, **kwargs):
    data["omega_rttt"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rttz",
    label="\\partial_{\\rho \\theta \\theta \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, fourth derivative wrt radius once, "
    "poloidal angle twice, and toroidal angle once",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rttz(params, transforms, profiles, data, **kwargs):
    data["omega_rttz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rtz",
    label="\\partial_{\\rho \\theta \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, third derivative wrt radius, poloidal"
    " angle, and toroidal angle",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rtz(params, transforms, profiles, data, **kwargs):
    data["omega_rtz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rtzz",
    label="\\partial_{\\rho \\theta \\zeta \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, fourth derivative wrt radius, poloidal"
    " angle, and toroidal angle twice",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rtzz(params, transforms, profiles, data, **kwargs):
    data["omega_rtzz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rz",
    label="\\partial_{\\rho \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, second derivative wrt radius and "
    "toroidal angle",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rz(params, transforms, profiles, data, **kwargs):
    data["omega_rz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, third derivative wrt radius and "
    "toroidal angle twice",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rzz(params, transforms, profiles, data, **kwargs):
    data["omega_rzz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_rzzz",
    label="\\partial_{\\rho \\zeta \\zeta \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, third derivative wrt radius and "
    "toroidal angle thrice",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_rzzz(params, transforms, profiles, data, **kwargs):
    data["omega_rzzz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_t",
    label="\\partial_{\\theta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, first poloidal derivative",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_t(params, transforms, profiles, data, **kwargs):
    data["omega_t"] = data["0"]
    return data


@register_compute_fun(
    name="omega_tt",
    label="\\partial_{\\theta \\theta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, second poloidal derivative",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_tt(params, transforms, profiles, data, **kwargs):
    data["omega_tt"] = data["0"]
    return data


@register_compute_fun(
    name="omega_ttt",
    label="\\partial_{\\theta \\theta \\theta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, third poloidal derivative",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_ttt(params, transforms, profiles, data, **kwargs):
    data["omega_ttt"] = data["0"]
    return data


@register_compute_fun(
    name="omega_ttz",
    label="\\partial_{\\theta \\theta \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, third derivative wrt poloidal angle "
    "twice and toroidal angle",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_ttz(params, transforms, profiles, data, **kwargs):
    data["omega_ttz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_tz",
    label="\\partial_{\\theta \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, second derivative wrt poloidal and "
    "toroidal angles",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_tz(params, transforms, profiles, data, **kwargs):
    data["omega_tz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_tzz",
    label="\\partial_{\\theta \\zeta \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, third derivative wrt poloidal angle "
    "and toroidal angle twice",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_tzz(params, transforms, profiles, data, **kwargs):
    data["omega_tzz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_z",
    label="\\partial_{\\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, first toroidal derivative",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_z(params, transforms, profiles, data, **kwargs):
    data["omega_z"] = data["0"]
    return data


@register_compute_fun(
    name="omega_zz",
    label="\\partial_{\\zeta \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, second toroidal derivative",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_zz(params, transforms, profiles, data, **kwargs):
    data["omega_zz"] = data["0"]
    return data


@register_compute_fun(
    name="omega_zzz",
    label="\\partial_{\\zeta \\zeta \\zeta} \\omega",
    units="rad",
    units_long="radians",
    description="Toroidal stream function, third toroidal derivative",
    dim=1,
    params=[],  # ["W_lmn"]
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _omega_zzz(params, transforms, profiles, data, **kwargs):
    data["omega_zzz"] = data["0"]
    return data


@register_compute_fun(
    name="phi",
    label="\\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["zeta", "omega"],
)
def _phi(params, transforms, profiles, data, **kwargs):
    data["phi"] = data["zeta"] + data["omega"]
    return data


@register_compute_fun(
    name="phi_r",
    label="\\partial_{\\rho} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_r"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_r(params, transforms, profiles, data, **kwargs):
    data["phi_r"] = data["omega_r"]
    return data


@register_compute_fun(
    name="phi_rr",
    label="\\partial_{\\rho \\rho} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_rr"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_rr(params, transforms, profiles, data, **kwargs):
    data["phi_rr"] = data["omega_rr"]
    return data


@register_compute_fun(
    name="phi_rrz",
    label="\\partial_{\\rho \\rho \\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt radial coordinate "
    "and first wrt DESC toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_rrz"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_rrz(params, transforms, profiles, data, **kwargs):
    data["phi_rrz"] = data["omega_rrz"]
    return data


@register_compute_fun(
    name="phi_rt",
    label="\\partial_{\\rho \\theta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt radial and "
    "poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_rt"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_rt(params, transforms, profiles, data, **kwargs):
    data["phi_rt"] = data["omega_rt"]
    return data


@register_compute_fun(
    name="phi_rtz",
    label="\\partial_{\\rho \\theta \\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, third derivative wrt radial, "
    "poloidal, and toroidal coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_rtz"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_rtz(params, transforms, profiles, data, **kwargs):
    data["phi_rtz"] = data["omega_rtz"]
    return data


@register_compute_fun(
    name="phi_rz",
    label="\\partial_{\\rho \\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt radial and "
    "toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_rz"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_rz(params, transforms, profiles, data, **kwargs):
    data["phi_rz"] = data["omega_rz"]
    return data


@register_compute_fun(
    name="phi_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, first derivative wrt radial and "
    "second derivative wrt DESC toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_rzz"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_rzz(params, transforms, profiles, data, **kwargs):
    data["phi_rzz"] = data["omega_rzz"]
    return data


@register_compute_fun(
    name="phi_t",
    label="\\partial_{\\theta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, derivative wrt poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_t"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_t(params, transforms, profiles, data, **kwargs):
    data["phi_t"] = data["omega_t"]
    return data


@register_compute_fun(
    name="phi_tt",
    label="\\partial_{\\theta \\theta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt poloidal "
    "coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_tt"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_tt(params, transforms, profiles, data, **kwargs):
    data["phi_tt"] = data["omega_tt"]
    return data


@register_compute_fun(
    name="phi_ttt",
    label="\\partial_{\\theta \\theta \\theta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, third derivative wrt poloidal "
    "coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_ttt"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_ttt(params, transforms, profiles, data, **kwargs):
    data["phi_ttt"] = data["omega_ttt"]
    return data


@register_compute_fun(
    name="phi_ttz",
    label="\\partial_{\\theta \\theta \\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt poloidal "
    "coordinate and first derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_ttz"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_ttz(params, transforms, profiles, data, **kwargs):
    data["phi_ttz"] = data["omega_ttz"]
    return data


@register_compute_fun(
    name="phi_tz",
    label="\\partial_{\\theta \\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, derivative wrt poloidal and "
    "toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_tz"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_tz(params, transforms, profiles, data, **kwargs):
    data["phi_tz"] = data["omega_tz"]
    return data


@register_compute_fun(
    name="phi_tzz",
    label="\\partial_{\\theta \\zeta \\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, derivative wrt poloidal coordinate and "
    "second derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_tzz"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_tzz(params, transforms, profiles, data, **kwargs):
    data["phi_tzz"] = data["omega_tzz"]
    return data


@register_compute_fun(
    name="phi_z",
    label="\\partial_{\\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_z"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_z(params, transforms, profiles, data, **kwargs):
    data["phi_z"] = 1 + data["omega_z"]
    return data


@register_compute_fun(
    name="phi_zz",
    label="\\partial_{\\zeta \\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt toroidal "
    "coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_zz"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_zz(params, transforms, profiles, data, **kwargs):
    data["phi_zz"] = data["omega_zz"]
    return data


@register_compute_fun(
    name="phi_zzz",
    label="\\partial_{\\zeta \\zeta \\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, third derivative wrt toroidal "
    "coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["omega_zzz"],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _phi_zzz(params, transforms, profiles, data, **kwargs):
    data["phi_zzz"] = data["omega_zzz"]
    return data


@register_compute_fun(
    name="rho",
    label="\\rho",
    units="~",
    units_long="None",
    description="Radial coordinate, proportional to the square root "
    + "of the toroidal flux",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
        "desc.magnetic_fields._core.OmnigenousField",
    ],
)
def _rho(params, transforms, profiles, data, **kwargs):
    data["rho"] = transforms["grid"].nodes[:, 0]
    return data


@register_compute_fun(
    name="theta",
    label="\\theta",
    units="rad",
    units_long="radians",
    description="Poloidal angular coordinate (geometric, not magnetic)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="t",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _theta(params, transforms, profiles, data, **kwargs):
    data["theta"] = transforms["grid"].nodes[:, 1]
    return data


@register_compute_fun(
    name="theta_PEST",
    label="\\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta", "lambda"],
)
def _theta_PEST(params, transforms, profiles, data, **kwargs):
    data["theta_PEST"] = data["theta"] + data["lambda"]
    return data


@register_compute_fun(
    name="theta_PEST_r",
    label="\\partial_{\\rho} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, derivative wrt "
    "radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_r"],
)
def _theta_PEST_r(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_r"] = data["lambda_r"]
    return data


@register_compute_fun(
    name="theta_PEST_rt",
    label="\\partial_{\\rho \\theta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate,"
    "derivative wrt poloidal and radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_rt"],
)
def _theta_PEST_rt(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_rt"] = data["lambda_rt"]
    return data


@register_compute_fun(
    name="theta_PEST_rz",
    label="\\partial_{\\rho \\zeta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, derivative wrt "
    "radial and DESC toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_rz"],
)
def _theta_PEST_rz(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_rz"] = data["lambda_rz"]
    return data


@register_compute_fun(
    name="theta_PEST_rrt",
    label="\\partial_{\\rho \\rho \\theta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, second "
    "derivative wrt radial coordinate and first derivative wrt DESC poloidal "
    "coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_rrt"],
)
def _theta_PEST_rrt(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_rrt"] = data["lambda_rrt"]
    return data


@register_compute_fun(
    name="theta_PEST_rtz",
    label="\\partial_{\\rho \\theta \\zeta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, derivative wrt "
    "radial and DESC poloidal and toroidal coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_rtz"],
)
def _theta_PEST_rtz(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_rtz"] = data["lambda_rtz"]
    return data


@register_compute_fun(
    name="theta_PEST_rtt",
    label="\\partial_{\\rho \\theta \\theta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, derivative wrt "
    "radial coordinate once and DESC poloidal coordinate twice",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_rtt"],
)
def _theta_PEST_rtt(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_rtt"] = data["lambda_rtt"]
    return data


@register_compute_fun(
    name="theta_PEST_t",
    label="\\partial_{\\theta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, derivative wrt "
    "poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_t"],
)
def _theta_PEST_t(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_t"] = 1 + data["lambda_t"]
    return data


@register_compute_fun(
    name="theta_PEST_tt",
    label="\\partial_{\\theta \\theta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate,"
    "second derivative wrt poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_tt"],
)
def _theta_PEST_tt(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_tt"] = data["lambda_tt"]
    return data


@register_compute_fun(
    name="theta_PEST_ttt",
    label="\\partial_{\\theta \\theta \\theta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, third "
    "derivative wrt poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_ttt"],
)
def _theta_PEST_ttt(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_ttt"] = data["lambda_ttt"]
    return data


@register_compute_fun(
    name="theta_PEST_tz",
    label="\\partial_{\\theta \\zeta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, derivative wrt "
    "poloidal and toroidal coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_tz"],
)
def _theta_PEST_tz(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_tz"] = data["lambda_tz"]
    return data


@register_compute_fun(
    name="theta_PEST_tzz",
    label="\\partial_{\\theta \\zeta \\zeta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, derivative wrt "
    "poloidal coordinate once and toroidal coordinate twice",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_tzz"],
)
def _theta_PEST_tzz(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_tzz"] = data["lambda_tzz"]
    return data


@register_compute_fun(
    name="theta_PEST_z",
    label="\\partial_{\\zeta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate,"
    " derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_z"],
)
def _theta_PEST_z(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_z"] = data["lambda_z"]
    return data


@register_compute_fun(
    name="theta_PEST_zz",
    label="\\partial_{\\zeta \\zeta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, second "
    "derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_zz"],
)
def _theta_PEST_zz(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_zz"] = data["lambda_zz"]
    return data


@register_compute_fun(
    name="theta_PEST_ttz",
    label="\\partial_{\\theta \\theta \\zeta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, second "
    "derivative wrt poloidal coordinate and derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_ttz"],
)
def _theta_PEST_ttz(params, transforms, profiles, data, **kwargs):
    data["theta_PEST_ttz"] = data["lambda_ttz"]
    return data


@register_compute_fun(
    name="zeta",
    label="\\zeta",
    units="rad",
    units_long="radians",
    description="Toroidal angular coordinate",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="z",
    data=[],
    parameterization=[
        "desc.equilibrium.equilibrium.Equilibrium",
        "desc.geometry.core.Surface",
    ],
)
def _zeta(params, transforms, profiles, data, **kwargs):
    data["zeta"] = transforms["grid"].nodes[:, 2]
    return data
