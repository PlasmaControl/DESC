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
)
def _0(params, transforms, profiles, data, **kwargs):
    data["0"] = jnp.zeros(transforms["grid"].num_nodes)
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
)
def _rho(params, transforms, profiles, data, **kwargs):
    data["rho"] = transforms["grid"].nodes[:, 0]
    return data


@register_compute_fun(
    name="rho_r",
    label="\\partial_{\\rho} \\rho",
    units="~",
    units_long="None",
    description="Radial coordinate, proportional to the square root "
    + "of the toroidal flux, derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["0"],
)
def _rho_r(params, transforms, profiles, data, **kwargs):
    data["rho_r"] = jnp.ones_like(data["0"])
    return data


@register_compute_fun(
    name="rho_t",
    label="\\partial_{\\theta} \\rho",
    units="~",
    units_long="None",
    description="Radial coordinate, proportional to the square root "
    + "of the toroidal flux, derivative wrt poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["0"],
)
def _rho_t(params, transforms, profiles, data, **kwargs):
    data["rho_t"] = data["0"]
    return data


@register_compute_fun(
    name="rho_z",
    label="\\partial_{\\zeta} \\rho",
    units="~",
    units_long="None",
    description="Radial coordinate, proportional to the square root "
    + "of the toroidal flux, derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["0"],
)
def _rho_z(params, transforms, profiles, data, **kwargs):
    data["rho_z"] = data["0"]
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
)
def _theta(params, transforms, profiles, data, **kwargs):
    data["theta"] = transforms["grid"].nodes[:, 1]
    return data


@register_compute_fun(
    name="theta_r",
    label="\\partial_{\\rho} \\theta",
    units="rad",
    units_long="radians",
    description="Poloidal angular coordinate (geometric, not magnetic), "
    + "derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="t",
    data=["0"],
)
def _theta_r(params, transforms, profiles, data, **kwargs):
    data["theta_r"] = data["0"]
    return data


@register_compute_fun(
    name="theta_t",
    label="\\partial_{\\theta} \\theta",
    units="rad",
    units_long="radians",
    description="Poloidal angular coordinate (geometric, not magnetic), "
    + "derivative wrt poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="t",
    data=["0"],
)
def _theta_t(params, transforms, profiles, data, **kwargs):
    data["theta_t"] = jnp.ones_like(data["0"])
    return data


@register_compute_fun(
    name="theta_z",
    label="\\partial_{\\zeta} \\theta",
    units="rad",
    units_long="radians",
    description="Poloidal angular coordinate (geometric, not magnetic), "
    + "derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="t",
    data=["0"],
)
def _theta_z(params, transforms, profiles, data, **kwargs):
    data["theta_z"] = data["0"]
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
)
def _zeta(params, transforms, profiles, data, **kwargs):
    data["zeta"] = transforms["grid"].nodes[:, 2]
    return data


@register_compute_fun(
    name="zeta_r",
    label="\\partial_{\\rho} \\zeta",
    units="rad",
    units_long="radians",
    description="Toroidal angular coordinate derivative, wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="z",
    data=["0"],
)
def _zeta_r(params, transforms, profiles, data, **kwargs):
    data["zeta_r"] = data["0"]
    return data


@register_compute_fun(
    name="zeta_t",
    label="\\partial_{\\theta} \\zeta",
    units="rad",
    units_long="radians",
    description="Toroidal angular coordinate, derivative wrt poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="z",
    data=["0"],
)
def _zeta_t(params, transforms, profiles, data, **kwargs):
    data["zeta_t"] = data["0"]
    return data


@register_compute_fun(
    name="zeta_z",
    label="\\partial_{\\zeta} \\zeta",
    units="rad",
    units_long="radians",
    description="Toroidal angular coordinate, derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="z",
    data=["0"],
)
def _zeta_z(params, transforms, profiles, data, **kwargs):
    data["zeta_z"] = jnp.ones_like(data["0"])
    return data


@register_compute_fun(
    name="theta_sfl",
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
def _theta_sfl(params, transforms, profiles, data, **kwargs):
    data["theta_sfl"] = (data["theta"] + data["lambda"]) % (2 * jnp.pi)
    return data


@register_compute_fun(
    name="theta_sfl_r",
    label="\\partial_{\\rho} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, derivative wrt "
    + "radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_r"],
)
def _theta_sfl_r(params, transforms, profiles, data, **kwargs):
    data["theta_sfl_r"] = data["lambda_r"]
    return data


@register_compute_fun(
    name="theta_sfl_t",
    label="\\partial_{\\theta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, derivative wrt "
    + "poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_t"],
)
def _theta_sfl_t(params, transforms, profiles, data, **kwargs):
    data["theta_sfl_t"] = 1 + data["lambda_t"]
    return data


@register_compute_fun(
    name="theta_sfl_z",
    label="\\partial_{\\zeta} \\vartheta",
    units="rad",
    units_long="radians",
    description="PEST straight field line poloidal angular coordinate, derivative wrt "
    + "toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_z"],
)
def _theta_sfl_z(params, transforms, profiles, data, **kwargs):
    data["theta_sfl_z"] = data["lambda_z"]
    return data


@register_compute_fun(
    name="alpha",
    label="\\alpha",
    units="~",
    units_long="None",
    description="Field line label, defined on [0, 2pi)",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta_sfl", "zeta", "iota"],
)
def _alpha(params, transforms, profiles, data, **kwargs):
    data["alpha"] = (data["theta_sfl"] - data["iota"] * data["zeta"]) % (2 * jnp.pi)
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
    data=["lambda_r", "zeta", "iota_r"],
)
def _alpha_r(params, transforms, profiles, data, **kwargs):
    data["alpha_r"] = data["lambda_r"] - data["iota_r"] * data["zeta"]
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
    data=["lambda_t"],
)
def _alpha_t(params, transforms, profiles, data, **kwargs):
    data["alpha_t"] = 1 + data["lambda_t"]
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
    data=["lambda_z", "iota"],
)
def _alpha_z(params, transforms, profiles, data, **kwargs):
    data["alpha_z"] = data["lambda_z"] - data["iota"]
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
)
def _R_r(params, transforms, profiles, data, **kwargs):
    data["R_r"] = transforms["R"].transform(params["R_lmn"], 1, 0, 0)
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
)
def _R_t(params, transforms, profiles, data, **kwargs):
    data["R_t"] = transforms["R"].transform(params["R_lmn"], 0, 1, 0)
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
)
def _R_z(params, transforms, profiles, data, **kwargs):
    data["R_z"] = transforms["R"].transform(params["R_lmn"], 0, 0, 1)
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
)
def _R_rr(params, transforms, profiles, data, **kwargs):
    data["R_rr"] = transforms["R"].transform(params["R_lmn"], 2, 0, 0)
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
)
def _R_tt(params, transforms, profiles, data, **kwargs):
    data["R_tt"] = transforms["R"].transform(params["R_lmn"], 0, 2, 0)
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
)
def _R_zz(params, transforms, profiles, data, **kwargs):
    data["R_zz"] = transforms["R"].transform(params["R_lmn"], 0, 0, 2)
    return data


@register_compute_fun(
    name="R_rt",
    label="\\partial_{\\rho \\theta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, second derivative wrt radius "
    + "and poloidal angle",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[1, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rt(params, transforms, profiles, data, **kwargs):
    data["R_rt"] = transforms["R"].transform(params["R_lmn"], 1, 1, 0)
    return data


@register_compute_fun(
    name="R_rz",
    label="\\partial_{\\rho \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, second derivative wrt radius "
    + "and toroidal angle",
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
    name="R_tz",
    label="\\partial_{\\theta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, second derivative wrt poloidal "
    + "and toroidal angles",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 1, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_tz(params, transforms, profiles, data, **kwargs):
    data["R_tz"] = transforms["R"].transform(params["R_lmn"], 0, 1, 1)
    return data


@register_compute_fun(
    name="R_rrr",
    label="\\partial_{\rho \\rho \\rho} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third radial derivative",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[3, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rrr(params, transforms, profiles, data, **kwargs):
    data["R_rrr"] = transforms["R"].transform(params["R_lmn"], 3, 0, 0)
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
)
def _R_ttt(params, transforms, profiles, data, **kwargs):
    data["R_ttt"] = transforms["R"].transform(params["R_lmn"], 0, 3, 0)
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
)
def _R_zzz(params, transforms, profiles, data, **kwargs):
    data["R_zzz"] = transforms["R"].transform(params["R_lmn"], 0, 0, 3)
    return data


@register_compute_fun(
    name="R_rrt",
    label="\\partial_{\\rho \\rho \\theta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative, wrt radius twice "
    + "and poloidal angle",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[2, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rrt(params, transforms, profiles, data, **kwargs):
    data["R_rrt"] = transforms["R"].transform(params["R_lmn"], 2, 1, 0)
    return data


@register_compute_fun(
    name="R_rtt",
    label="\\partial_{\\rho \\theta \\theta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative wrt radius and "
    + "poloidal angle twice",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[1, 2, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_rtt(params, transforms, profiles, data, **kwargs):
    data["R_rtt"] = transforms["R"].transform(params["R_lmn"], 1, 2, 0)
    return data


@register_compute_fun(
    name="R_rrz",
    label="\\partial_{\\rho \\rho \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative, wrt radius twice "
    + "and toroidal angle",
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
    name="R_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative wrt radius and "
    + "toroidal angle twice",
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
    name="R_ttz",
    label="\\partial_{\\theta \\theta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative wrt poloidal angle "
    + "twice and toroidal angle",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 2, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_ttz(params, transforms, profiles, data, **kwargs):
    data["R_ttz"] = transforms["R"].transform(params["R_lmn"], 0, 2, 1)
    return data


@register_compute_fun(
    name="R_tzz",
    label="\\partial_{\\theta \\zeta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative wrt poloidal angle "
    + "and toroidal angle twice",
    dim=1,
    params=["R_lmn"],
    transforms={"R": [[0, 1, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _R_tzz(params, transforms, profiles, data, **kwargs):
    data["R_tzz"] = transforms["R"].transform(params["R_lmn"], 0, 1, 2)
    return data


@register_compute_fun(
    name="R_rtz",
    label="\\partial_{\\rho \\theta \\zeta} R",
    units="m",
    units_long="meters",
    description="Major radius in lab frame, third derivative wrt radius, poloidal "
    + "angle, and toroidal angle",
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
)
def _Z_r(params, transforms, profiles, data, **kwargs):
    data["Z_r"] = transforms["Z"].transform(params["Z_lmn"], 1, 0, 0)
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
)
def _Z_t(params, transforms, profiles, data, **kwargs):
    data["Z_t"] = transforms["Z"].transform(params["Z_lmn"], 0, 1, 0)
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
)
def _Z_z(params, transforms, profiles, data, **kwargs):
    data["Z_z"] = transforms["Z"].transform(params["Z_lmn"], 0, 0, 1)
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
)
def _Z_rr(params, transforms, profiles, data, **kwargs):
    data["Z_rr"] = transforms["Z"].transform(params["Z_lmn"], 2, 0, 0)
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
)
def _Z_tt(params, transforms, profiles, data, **kwargs):
    data["Z_tt"] = transforms["Z"].transform(params["Z_lmn"], 0, 2, 0)
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
)
def _Z_zz(params, transforms, profiles, data, **kwargs):
    data["Z_zz"] = transforms["Z"].transform(params["Z_lmn"], 0, 0, 2)
    return data


@register_compute_fun(
    name="Z_rt",
    label="\\partial_{\\rho \\theta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, second derivative wrt radius "
    + "and poloidal angle",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[1, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rt(params, transforms, profiles, data, **kwargs):
    data["Z_rt"] = transforms["Z"].transform(params["Z_lmn"], 1, 1, 0)
    return data


@register_compute_fun(
    name="Z_rz",
    label="\\partial_{\\rho \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, second derivative wrt radius "
    + "and toroidal angle",
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
    name="Z_tz",
    label="\\partial_{\\theta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, second derivative wrt poloidal "
    + "and toroidal angles",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 1, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_tz(params, transforms, profiles, data, **kwargs):
    data["Z_tz"] = transforms["Z"].transform(params["Z_lmn"], 0, 1, 1)
    return data


@register_compute_fun(
    name="Z_rrr",
    label="\\partial_{\rho \\rho \\rho} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third radial derivative",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[3, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rrr(params, transforms, profiles, data, **kwargs):
    data["Z_rrr"] = transforms["Z"].transform(params["Z_lmn"], 3, 0, 0)
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
)
def _Z_ttt(params, transforms, profiles, data, **kwargs):
    data["Z_ttt"] = transforms["Z"].transform(params["Z_lmn"], 0, 3, 0)
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
)
def _Z_zzz(params, transforms, profiles, data, **kwargs):
    data["Z_zzz"] = transforms["Z"].transform(params["Z_lmn"], 0, 0, 3)
    return data


@register_compute_fun(
    name="Z_rrt",
    label="\\partial_{\\rho \\rho \\theta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative, wrt radius "
    + "twice and poloidal angle",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[2, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rrt(params, transforms, profiles, data, **kwargs):
    data["Z_rrt"] = transforms["Z"].transform(params["Z_lmn"], 2, 1, 0)
    return data


@register_compute_fun(
    name="Z_rtt",
    label="\\partial_{\\rho \\theta \\theta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative wrt radius "
    + "and poloidal angle twice",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[1, 2, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_rtt(params, transforms, profiles, data, **kwargs):
    data["Z_rtt"] = transforms["Z"].transform(params["Z_lmn"], 1, 2, 0)
    return data


@register_compute_fun(
    name="Z_rrz",
    label="\\partial_{\\rho \\rho \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative, wrt radius "
    + "twice and toroidal angle",
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
    name="Z_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative wrt radius "
    + "and toroidal angle twice",
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
    name="Z_ttz",
    label="\\partial_{\\theta \\theta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative wrt poloidal "
    + "angle twice and toroidal angle",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 2, 1]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_ttz(params, transforms, profiles, data, **kwargs):
    data["Z_ttz"] = transforms["Z"].transform(params["Z_lmn"], 0, 2, 1)
    return data


@register_compute_fun(
    name="Z_tzz",
    label="\\partial_{\\theta \\zeta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative wrt poloidal "
    + "angle and toroidal angle twice",
    dim=1,
    params=["Z_lmn"],
    transforms={"Z": [[0, 1, 2]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _Z_tzz(params, transforms, profiles, data, **kwargs):
    data["Z_tzz"] = transforms["Z"].transform(params["Z_lmn"], 0, 1, 2)
    return data


@register_compute_fun(
    name="Z_rtz",
    label="\\partial_{\\rho \\theta \\zeta} Z",
    units="m",
    units_long="meters",
    description="Vertical coordinate in lab frame, third derivative wrt radius, "
    + "poloidal angle, and toroidal angle",
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
    data=["zeta"],
)
def _phi(params, transforms, profiles, data, **kwargs):
    data["phi"] = data["zeta"]
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
    data=["0"],
)
def _phi_r(params, transforms, profiles, data, **kwargs):
    data["phi_r"] = data["0"]
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
    data=["0"],
)
def _phi_t(params, transforms, profiles, data, **kwargs):
    data["phi_t"] = data["0"]
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
    data=["0"],
)
def _phi_z(params, transforms, profiles, data, **kwargs):
    data["phi_z"] = jnp.ones_like(data["0"])
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
    data=["0"],
)
def _phi_rr(params, transforms, profiles, data, **kwargs):
    data["phi_rr"] = data["0"]
    return data


@register_compute_fun(
    name="phi_rt",
    label="\\partial_{\\rho \\theta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt radial and "
    + "poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
)
def _phi_rt(params, transforms, profiles, data, **kwargs):
    data["phi_rt"] = data["0"]
    return data


@register_compute_fun(
    name="phi_rz",
    label="\\partial_{\\rho \\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt radial and "
    + "toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
)
def _phi_rz(params, transforms, profiles, data, **kwargs):
    data["phi_rz"] = data["0"]
    return data


@register_compute_fun(
    name="phi_tt",
    label="\\partial_{\\theta \\theta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt poloidal "
    + "coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
)
def _phi_tt(params, transforms, profiles, data, **kwargs):
    data["phi_tt"] = data["0"]
    return data


@register_compute_fun(
    name="phi_tz",
    label="\\partial_{\\theta \\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt poloidal and "
    + "toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
)
def _phi_tz(params, transforms, profiles, data, **kwargs):
    data["phi_tz"] = data["0"]
    return data


@register_compute_fun(
    name="phi_zz",
    label="\\partial_{\\zeta \\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt toroidal "
    + "coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["0"],
)
def _phi_zz(params, transforms, profiles, data, **kwargs):
    data["phi_zz"] = data["0"]
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
    name="lambda_rt",
    label="\\partial_{\\rho \\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, second derivative wrt radius and "
    + "poloidal angle",
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
    name="lambda_rz",
    label="\\partial_{\\rho \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, second derivative wrt radius and "
    + "toroidal angle",
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
    name="lambda_tz",
    label="\\partial_{\\theta \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, second derivative wrt poloidal and "
    + "toroidal angles",
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
    name="lambda_rrr",
    label="\\partial_{\rho \\rho \\rho} \\lambda",
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
    name="lambda_rrt",
    label="\\partial_{\\rho \\rho \\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative, wrt radius twice "
    + "and poloidal angle",
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
    name="lambda_rtt",
    label="\\partial_{\\rho \\theta \\theta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative wrt radius and "
    + "poloidal angle twice",
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
    name="lambda_rrz",
    label="\\partial_{\\rho \\rho \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative, wrt radius twice "
    + "and toroidal angle",
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
    name="lambda_rzz",
    label="\\partial_{\\rho \\zeta \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative wrt radius and "
    + "toroidal angle twice",
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
    name="lambda_ttz",
    label="\\partial_{\\theta \\theta \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative wrt poloidal angle "
    + "twice and toroidal angle",
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
    name="lambda_tzz",
    label="\\partial_{\\theta \\zeta \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative wrt poloidal angle "
    + "and toroidal angle twice",
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
    name="lambda_rtz",
    label="\\partial_{\\rho \\theta \\zeta} \\lambda",
    units="rad",
    units_long="radians",
    description="Poloidal stream function, third derivative wrt radius, poloidal "
    + " angle, and toroidal angle",
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
