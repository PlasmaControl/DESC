from scipy.constants import elementary_charge, mu_0

from desc.backend import jnp, put

from .data_index import register_compute_fun
from .utils import compress, cumtrapz, dot, expand, surface_averages


@register_compute_fun(
    name="psi",
    label="\\psi = \\Psi / (2 \\pi)",
    units="Wb",
    units_long="Webers",
    description="Toroidal flux (normalized by 2pi)",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["rho"],
)
def _psi(params, transforms, profiles, data, **kwargs):
    data["psi"] = params["Psi"] * data["rho"] ** 2 / (2 * jnp.pi)
    return data


@register_compute_fun(
    name="psi_r",
    label="\\partial_{\\rho} \\psi = \\partial_{\\rho} \\Psi / (2 \\pi)",
    units="Wb",
    units_long="Webers",
    description="Toroidal flux (normalized by 2pi), first radial derivative",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["rho"],
)
def _psi_r(params, transforms, profiles, data, **kwargs):
    data["psi_r"] = params["Psi"] * data["rho"] / jnp.pi
    return data


@register_compute_fun(
    name="psi_rr",
    label="\\partial_{\\rho\\rho} \\psi = \\partial_{\\rho\\rho} \\Psi / (2 \\pi)",
    units="Wb",
    units_long="Webers",
    description="Toroidal flux (normalized by 2pi), second radial derivative",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["rho"],
)
def _psi_rr(params, transforms, profiles, data, **kwargs):
    data["psi_rr"] = params["Psi"] * jnp.ones_like(data["rho"]) / jnp.pi
    return data


@register_compute_fun(
    name="psi_rrr",
    label="\\partial_{\\rho\\rho\\rho} \\psi = \\partial_{\\rho\\rho\\rho} \\Psi / "
    + "(2 \\pi)",
    units="Wb",
    units_long="Webers",
    description="Toroidal flux (normalized by 2pi), third radial derivative",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["0"],
)
def _psi_rrr(params, transforms, profiles, data, **kwargs):
    data["psi_rrr"] = data["0"]
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
    name="chi_r",
    label="\\partial_{\\rho} \\chi",
    units="Wb",
    units_long="Webers",
    description="Poloidal flux (normalized by 2pi), first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["psi_r", "iota"],
)
def _chi_r(params, transforms, profiles, data, **kwargs):
    data["chi_r"] = data["psi_r"] * data["iota"]
    return data


@register_compute_fun(
    name="chi",
    label="\\chi",
    units="Wb",
    units_long="Webers",
    description="Poloidal flux (normalized by 2pi)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "chi_r",
    ],
)
def _chi(params, transforms, profiles, data, **kwargs):
    chi_r = compress(transforms["grid"], data["chi_r"], surface_label="rho")
    rho = transforms["grid"].nodes[transforms["grid"].unique_rho_idx, 0]
    chi = cumtrapz(chi_r, rho, initial=0)
    data["chi"] = expand(transforms["grid"], chi, surface_label="rho")
    return data


@register_compute_fun(
    name="Te",
    label="T_e",
    units="eV",
    units_long="electron-Volts",
    description="Electron temperature",
    dim=1,
    params=["Te_l"],
    transforms={},
    profiles=["electron_temperature"],
    coordinates="r",
    data=["0"],
)
def _Te(params, transforms, profiles, data, **kwargs):
    if profiles["electron_temperature"] is not None:
        data["Te"] = profiles["electron_temperature"].compute(params["Te_l"], dr=0)
    else:
        data["Te"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="Te_r",
    label="\\partial_{\\rho} T_e",
    units="eV",
    units_long="electron-Volts",
    description="Electron temperature, first radial derivative",
    dim=1,
    params=["Te_l"],
    transforms={},
    profiles=["electron_temperature"],
    coordinates="r",
    data=["0"],
)
def _Te_r(params, transforms, profiles, data, **kwargs):
    if profiles["electron_temperature"] is not None:
        data["Te_r"] = profiles["electron_temperature"].compute(params["Te_l"], dr=1)
    else:
        data["Te_r"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="ne",
    label="n_e",
    units="m^{-3}",
    units_long="1 / cubic meters",
    description="Electron density",
    dim=1,
    params=["ne_l"],
    transforms={},
    profiles=["electron_density"],
    coordinates="r",
    data=["0"],
)
def _ne(params, transforms, profiles, data, **kwargs):
    if profiles["electron_density"] is not None:
        data["ne"] = profiles["electron_density"].compute(params["ne_l"], dr=0)
    else:
        data["ne"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="ne_r",
    label="\\partial_{\\rho} n_e",
    units="m^{-3}",
    units_long="1 / cubic meters",
    description="Electron density, first radial derivative",
    dim=1,
    params=["ne_l"],
    transforms={},
    profiles=["electron_density"],
    coordinates="r",
    data=["0"],
)
def _ne_r(params, transforms, profiles, data, **kwargs):
    if profiles["electron_density"] is not None:
        data["ne_r"] = profiles["electron_density"].compute(params["ne_l"], dr=1)
    else:
        data["ne_r"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="Ti",
    label="T_i",
    units="eV",
    units_long="electron-Volts",
    description="Ion temperature",
    dim=1,
    params=["Ti_l"],
    transforms={},
    profiles=["ion_temperature"],
    coordinates="r",
    data=["0"],
)
def _Ti(params, transforms, profiles, data, **kwargs):
    if profiles["ion_temperature"] is not None:
        data["Ti"] = profiles["ion_temperature"].compute(params["Ti_l"], dr=0)
    else:
        data["Ti"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="Ti_r",
    label="\\partial_{\\rho} T_i",
    units="eV",
    units_long="electron-Volts",
    description="Ion temperature, first radial derivative",
    dim=1,
    params=["Ti_l"],
    transforms={},
    profiles=["ion_temperature"],
    coordinates="r",
    data=["0"],
)
def _Ti_r(params, transforms, profiles, data, **kwargs):
    if profiles["ion_temperature"] is not None:
        data["Ti_r"] = profiles["ion_temperature"].compute(params["Ti_l"], dr=1)
    else:
        data["Ti_r"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="Zeff",
    label="Z_{eff}",
    units="~",
    units_long="None",
    description="Effective atomic number",
    dim=1,
    params=["Zeff_l"],
    transforms={},
    profiles=["atomic_number"],
    coordinates="r",
    data=["0"],
)
def _Zeff(params, transforms, profiles, data, **kwargs):
    if profiles["atomic_number"] is not None:
        data["Zeff"] = profiles["atomic_number"].compute(params["Zeff_l"], dr=0)
    else:
        data["Zeff"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="Zeff_r",
    label="\\partial_{\\rho} Z_{eff}",
    units="~",
    units_long="None",
    description="Effective atomic number, first radial derivative",
    dim=1,
    params=["Zeff_l"],
    transforms={},
    profiles=["atomic_number"],
    coordinates="r",
    data=["0"],
)
def _Zeff_r(params, transforms, profiles, data, **kwargs):
    if profiles["atomic_number"] is not None:
        data["Zeff_r"] = profiles["atomic_number"].compute(params["Zeff_l"], dr=1)
    else:
        data["Zeff_r"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="p",
    label="p",
    units="Pa",
    units_long="Pascals",
    description="Pressure",
    dim=1,
    params=["p_l"],
    transforms={},
    profiles=["pressure"],
    coordinates="r",
    data=["Te", "ne", "Ti", "Zeff"],
)
def _p(params, transforms, profiles, data, **kwargs):
    if profiles["pressure"] is not None:
        data["p"] = profiles["pressure"].compute(params["p_l"], dr=0)
    else:
        data["p"] = elementary_charge * (
            data["ne"] * data["Te"] + data["Ti"] * data["ne"] / data["Zeff"]
        )
    return data


@register_compute_fun(
    name="p_r",
    label="\\partial_{\\rho} p",
    units="Pa",
    units_long="Pascals",
    description="Pressure, first radial derivative",
    dim=1,
    params=["p_l"],
    transforms={},
    profiles=["pressure"],
    coordinates="r",
    data=["Te", "Te_r", "ne", "ne_r", "Ti", "Ti_r", "Zeff", "Zeff_r"],
)
def _p_r(params, transforms, profiles, data, **kwargs):
    if profiles["pressure"] is not None:
        data["p_r"] = profiles["pressure"].compute(params["p_l"], dr=1)
    else:
        data["p_r"] = elementary_charge * (
            data["ne_r"] * data["Te"]
            + data["ne"] * data["Te_r"]
            + data["Ti_r"] * data["ne"] / data["Zeff"]
            + data["Ti"] * data["ne_r"] / data["Zeff"]
            - data["Ti"] * data["ne"] * data["Zeff_r"] / data["Zeff"] ** 2
        )
    return data


@register_compute_fun(
    name="grad(p)",
    label="\\nabla p",
    units="N \\cdot m^{-3}",
    units_long="Newtons / cubic meter",
    description="Pressure gradient",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["p_r", "e^rho"],
)
def _gradp(params, transforms, profiles, data, **kwargs):
    data["grad(p)"] = (data["p_r"] * data["e^rho"].T).T
    return data


@register_compute_fun(
    name="|grad(p)|^2",
    label="|\\nabla p|^{2}",
    units="N^2 \\cdot m^{-6}",
    units_long="Newtons per cubic meter squared",
    description="Magnitude of pressure gradient squared",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["grad(p)"],
)
def _gradp_mag2(params, transforms, profiles, data, **kwargs):
    data["|grad(p)|^2"] = dot(data["grad(p)"], data["grad(p)"])
    return data


@register_compute_fun(
    name="|grad(p)|",
    label="|\\nabla p|",
    units="N \\cdot m^{-3}",
    units_long="Newtons per cubic meter",
    description="Magnitude of pressure gradient",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|grad(p)|^2"],
)
def _gradp_mag(params, transforms, profiles, data, **kwargs):
    data["|grad(p)|"] = jnp.sqrt(data["|grad(p)|^2"])
    return data


@register_compute_fun(
    name="<|grad(p)|>_vol",
    label="\\langle |\\nabla p| \\rangle_{vol}",
    units="N \\cdot m^{-3}",
    units_long="Newtons per cubic meter",
    description="Volume average of magnitude of pressure gradient",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["|grad(p)|", "sqrt(g)", "V"],
)
def _gradp_mag_vol(params, transforms, profiles, data, **kwargs):
    data["<|grad(p)|>_vol"] = (
        jnp.sum(data["|grad(p)|"] * data["sqrt(g)"] * transforms["grid"].weights)
        / data["V"]
    )
    return data


@register_compute_fun(
    name="iota",
    label="\\iota",
    units="~",
    units_long="None",
    description="Rotational transform (normalized by 2pi)",
    dim=1,
    params=["i_l", "c_l"],
    transforms={"grid": []},
    profiles=["iota", "current"],
    coordinates="r",
    data=["psi_r", "iota_zero_current_num", "iota_zero_current_den"],
    axis_limit_data=["psi_rr"],
)
def _iota(params, transforms, profiles, data, **kwargs):
    # The rotational transform is computed from the toroidal current profile using
    # equation 11 in S.P. Hishman & J.T. Hogan (1986)
    # doi:10.1016/0021-9991(86)90197-X. Their "zero current algorithm" is supplemented
    # with an additional term to account for finite net toroidal currents. Note that
    # the flux surface average integrals in their formula should not be weighted by a
    # coordinate Jacobian factor, meaning the sqrt(g) terms in the denominators of
    # these averages will not be canceled out.
    if profiles["iota"] is not None:
        data["iota"] = profiles["iota"].compute(params["i_l"], dr=0)
    elif profiles["current"] is not None:
        # current_term = 2*pi * I / params["Psi"]_r = mu_0 / 2*pi * current / psi_r
        current_term = (
            mu_0
            / (2 * jnp.pi)
            * profiles["current"].compute(params["c_l"], dr=0)
            / data["psi_r"]
        )
        if transforms["grid"].axis.size:
            limit = (
                mu_0
                / (2 * jnp.pi)
                * profiles["current"].compute(params["c_l"], dr=2)
                / data["psi_rr"]
            )
            current_term = put(
                current_term, transforms["grid"].axis, limit[transforms["grid"].axis]
            )
        data["iota"] = (current_term + data["iota_zero_current_num"]) / data[
            "iota_zero_current_den"
        ]
    return data


@register_compute_fun(
    name="iota_r",
    label="\\partial_{\\rho} \\iota",
    units="~",
    units_long="None",
    description="Rotational transform (normalized by 2pi), first radial derivative",
    dim=1,
    params=["i_l", "c_l"],
    transforms={"grid": []},
    profiles=["iota", "current"],
    coordinates="r",
    data=[
        "iota",
        "psi_r",
        "psi_rr",
        "iota_zero_current_num",
        "iota_zero_current_num_r",
        "iota_zero_current_den",
        "iota_zero_current_den_r",
    ],
)
def _iota_r(params, transforms, profiles, data, **kwargs):
    # The rotational transform is computed from the toroidal current profile using
    # equation 11 in S.P. Hishman & J.T. Hogan (1986)
    # doi:10.1016/0021-9991(86)90197-X. Their "zero current algorithm" is supplemented
    # with an additional term to account for finite net toroidal currents. Note that
    # the flux surface average integrals in their formula should not be weighted by a
    # coordinate Jacobian factor, meaning the sqrt(g) terms in the denominators of
    # these averages will not be canceled out.
    if profiles["iota"] is not None:
        data["iota_r"] = profiles["iota"].compute(params["i_l"], dr=1)
    elif profiles["current"] is not None:
        current_term = (
            mu_0
            / (2 * jnp.pi)
            * profiles["current"].compute(params["c_l"], dr=0)
            / data["psi_r"]
        )
        current_term_r = (
            mu_0 / (2 * jnp.pi) * profiles["current"].compute(params["c_l"], dr=1)
            - current_term * data["psi_rr"]
        ) / data["psi_r"]
        data["iota_r"] = (
            current_term_r
            + data["iota_zero_current_num_r"]
            - data["iota"] * data["iota_zero_current_den_r"]
        ) / data["iota_zero_current_den"]
    return data


@register_compute_fun(
    name="iota_rr",
    label="\\partial_{\\rho\\rho} \\iota",
    units="~",
    units_long="None",
    description="Rotational transform (normalized by 2pi), second radial derivative",
    dim=1,
    params=["i_l", "c_l"],
    transforms={"grid": []},
    profiles=["iota", "current"],
    coordinates="r",
    data=[
        "iota",
        "iota_r",
        "psi_r",
        "psi_rr",
        "psi_rrr",
        "iota_zero_current_num",
        "iota_zero_current_num_r",
        "iota_zero_current_num_rr",
        "iota_zero_current_den",
        "iota_zero_current_den_r",
        "iota_zero_current_den_rr",
    ],
)
def _iota_rr(params, transforms, profiles, data, **kwargs):
    # The rotational transform is computed from the toroidal current profile using
    # equation 11 in S.P. Hishman & J.T. Hogan (1986)
    # doi:10.1016/0021-9991(86)90197-X. Their "zero current algorithm" is supplemented
    # with an additional term to account for finite net toroidal currents. Note that
    # the flux surface average integrals in their formula should not be weighted by a
    # coordinate Jacobian factor, meaning the sqrt(g) terms in the denominators of
    # these averages will not be canceled out.
    if profiles["iota"] is not None:
        data["iota_rr"] = profiles["iota"].compute(params["i_l"], dr=2)
    elif profiles["current"] is not None:
        current_term = (
            mu_0
            / (2 * jnp.pi)
            * profiles["current"].compute(params["c_l"], dr=0)
            / data["psi_r"]
        )
        current_term_r = (
            mu_0 / (2 * jnp.pi) * profiles["current"].compute(params["c_l"], dr=1)
            - current_term * data["psi_rr"]
        ) / data["psi_r"]
        current_term_rr = (
            mu_0 / (2 * jnp.pi) * profiles["current"].compute(params["c_l"], dr=2)
            - 2 * current_term_r * data["psi_rr"]
            - current_term * data["psi_rrr"]
        ) / data["psi_r"]
        data["iota_rr"] = (
            current_term_rr
            + data["iota_zero_current_num_rr"]
            - 2 * data["iota_r"] * data["iota_zero_current_den_r"]
            - data["iota"] * data["iota_zero_current_den_rr"]
        ) / data["iota_zero_current_den"]
    return data


@register_compute_fun(
    name="iota_zero_current_num",
    label="\\iota_{0~\\mathrm{numerator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Zero toroidal current rotational transform numerator",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["lambda_t", "lambda_z", "g_tt", "g_tz", "sqrt(g)"],
    axis_limit_data=[
        "lambda_rt",
        "g_tt_rr",
        "g_tz_r",
        "g_tz_rr",
        "sqrt(g)_r",
        "sqrt(g)_rr",
    ],
)
def _iota_zero_current_num(params, transforms, profiles, data, **kwargs):
    num = (
        data["lambda_z"] * data["g_tt"] - (1 + data["lambda_t"]) * data["g_tz"]
    ) / data["sqrt(g)"]
    data["iota_zero_current_num"] = surface_averages(transforms["grid"], num)

    if transforms["grid"].axis.size:
        limit = (
            (data["g_tz_r"] * data["sqrt(g)_rr"] * (1 + data["lambda_t"]))
            / data["sqrt(g)_r"] ** 2
        ) + (
            data["lambda_z"] * data["g_tt_rr"]
            - 2 * data["g_tz_r"] * data["lambda_rt"]
            - (1 + data["lambda_t"]) * data["g_tz_rr"]
        ) / data[
            "sqrt(g)_r"
        ]
        limit = surface_averages(transforms["grid"], limit)
        data["iota_zero_current_num"] = put(
            data["iota_zero_current_num"],
            transforms["grid"].axis,
            limit[transforms["grid"].axis],
        )

    return data


@register_compute_fun(
    name="iota_zero_current_num_r",
    label="\\partial_{\\rho} \\iota_{0~\\mathrm{numerator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Zero toroidal current rotational transform numerator,"
    " first radial derivative",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "lambda_t",
        "lambda_rt",
        "lambda_z",
        "lambda_rz",
        "g_tt",
        "g_tt_r",
        "g_tz",
        "g_tz_r",
        "sqrt(g)",
        "sqrt(g)_r",
    ],
)
def _iota_zero_current_num_r(params, transforms, profiles, data, **kwargs):
    num = (
        data["lambda_z"] * data["g_tt"] - (1 + data["lambda_t"]) * data["g_tz"]
    ) / data["sqrt(g)"]
    num_r = (
        data["lambda_rz"] * data["g_tt"]
        + data["lambda_z"] * data["g_tt_r"]
        - data["lambda_rt"] * data["g_tz"]
        - (1 + data["lambda_t"]) * data["g_tz_r"]
        - num * data["sqrt(g)_r"]
    ) / data["sqrt(g)"]
    data["iota_zero_current_num_r"] = surface_averages(transforms["grid"], num_r)
    # TODO: limit at axis
    return data


@register_compute_fun(
    name="iota_zero_current_num_rr",
    label="\\partial_{\\rho\\rho} \\iota_{0~\\mathrm{numerator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Zero toroidal current rotational transform numerator,"
    " second radial derivative",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "lambda_t",
        "lambda_rt",
        "lambda_rrt",
        "lambda_z",
        "lambda_rz",
        "lambda_rrz",
        "g_tt",
        "g_tt_r",
        "g_tt_rr",
        "g_tz",
        "g_tz_r",
        "g_tz_rr",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rr",
    ],
)
def _iota_zero_current_num_rr(params, transforms, profiles, data, **kwargs):
    num = (
        data["lambda_z"] * data["g_tt"] - (1 + data["lambda_t"]) * data["g_tz"]
    ) / data["sqrt(g)"]
    num_r = (
        data["lambda_rz"] * data["g_tt"]
        + data["lambda_z"] * data["g_tt_r"]
        - data["lambda_rt"] * data["g_tz"]
        - (1 + data["lambda_t"]) * data["g_tz_r"]
        - num * data["sqrt(g)_r"]
    ) / data["sqrt(g)"]
    num_rr = (
        data["lambda_rrz"] * data["g_tt"]
        + 2 * data["lambda_rz"] * data["g_tt_r"]
        + data["lambda_z"] * data["g_tt_rr"]
        - data["lambda_rrt"] * data["g_tz"]
        - 2 * data["lambda_rt"] * data["g_tz_r"]
        - (1 + data["lambda_t"]) * data["g_tz_rr"]
        - 2 * num_r * data["sqrt(g)_r"]
        - num * data["sqrt(g)_rr"]
    ) / data["sqrt(g)"]
    data["iota_zero_current_num_rr"] = surface_averages(transforms["grid"], num_rr)
    # TODO: limit at axis
    return data


@register_compute_fun(
    name="iota_zero_current_den",
    label="\\iota_{0~\\mathrm{denominator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Zero toroidal current rotational transform denominator",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["g_tt", "sqrt(g)"],
    axis_limit_data=["g_tt_rr", "sqrt(g)_r"],
)
def _iota_zero_current_den(params, transforms, profiles, data, **kwargs):
    den = data["g_tt"] / data["sqrt(g)"]
    data["iota_zero_current_den"] = surface_averages(transforms["grid"], den)

    if transforms["grid"].axis.size:
        limit = surface_averages(
            transforms["grid"], data["g_tt_rr"] / data["sqrt(g)_r"]
        )
        data["iota_zero_current_den"] = put(
            data["iota_zero_current_den"],
            transforms["grid"].axis,
            limit[transforms["grid"].axis],
        )

    return data


@register_compute_fun(
    name="iota_zero_current_den_r",
    label="\\partial_{\\rho} \\iota_{0~\\mathrm{denominator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Zero toroidal current rotational transform denominator,"
    " first radial derivative",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["g_tt", "g_tt_r", "sqrt(g)", "sqrt(g)_r"],
)
def _iota_zero_current_den_r(params, transforms, profiles, data, **kwargs):
    den = data["g_tt"] / data["sqrt(g)"]
    den_r = (data["g_tt_r"] - den * data["sqrt(g)_r"]) / data["sqrt(g)"]
    data["iota_zero_current_den_r"] = surface_averages(transforms["grid"], den_r)
    # TODO: limit at axis
    return data


@register_compute_fun(
    name="iota_zero_current_den_rr",
    label="\\partial_{\\rho\\rho} \\iota_{0~\\mathrm{denominator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Zero toroidal current rotational transform denominator,"
    " second radial derivative",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["g_tt", "g_tt_r", "g_tt_rr", "sqrt(g)", "sqrt(g)_r", "sqrt(g)_rr"],
)
def _iota_zero_current_den_rr(params, transforms, profiles, data, **kwargs):
    den = data["g_tt"] / data["sqrt(g)"]
    den_r = (data["g_tt_r"] - den * data["sqrt(g)_r"]) / data["sqrt(g)"]
    den_rr = (
        data["g_tt_rr"] - 2 * den_r * data["sqrt(g)_r"] - den * data["sqrt(g)_rr"]
    ) / data["sqrt(g)"]
    data["iota_zero_current_den_rr"] = surface_averages(transforms["grid"], den_rr)
    # TODO: limit at axis
    return data


@register_compute_fun(
    name="q",
    label="q = 1/\\iota",
    units="~",
    units_long="None",
    description="Safety factor 'q', inverse of rotational transform.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["iota"],
)
def _q(params, transforms, profiles, data, **kwargs):
    data["q"] = 1 / data["iota"]
    return data


# TODO: add K(rho,theta,zeta)*grad(rho) term
@register_compute_fun(
    name="I",
    label="I",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant poloidal component of magnetic field in Boozer coordinates "
    + "(proportional to toroidal current)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B_theta"],
)
def _I(params, transforms, profiles, data, **kwargs):
    data["I"] = surface_averages(transforms["grid"], data["B_theta"])
    return data


@register_compute_fun(
    name="I_r",
    label="\\partial_{\\rho} I",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant poloidal component of magnetic field in Boozer coordinates "
    + "(proportional to toroidal current), derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B_theta_r"],
)
def _I_r(params, transforms, profiles, data, **kwargs):
    data["I_r"] = surface_averages(transforms["grid"], data["B_theta_r"])
    return data


@register_compute_fun(
    name="I_rr",
    label="\\partial_{\\rho\\rho} I",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Boozer toroidal current enclosed by flux surfaces, second derivative "
    + "wrt radial coordinate",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B_theta_rr"],
)
def _I_rr(params, transforms, profiles, data, **kwargs):
    data["I_rr"] = surface_averages(transforms["grid"], data["B_theta_rr"])
    return data


@register_compute_fun(
    name="G",
    label="G",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant toroidal component of magnetic field in Boozer coordinates "
    + "(proportional to poloidal current)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B_zeta"],
)
def _G(params, transforms, profiles, data, **kwargs):
    data["G"] = surface_averages(transforms["grid"], data["B_zeta"])
    return data


@register_compute_fun(
    name="G_r",
    label="\\partial_{\\rho} G",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Covariant toroidal component of magnetic field in Boozer coordinates "
    + "(proportional to poloidal current), derivative wrt radial coordinate",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B_zeta_r"],
)
def _G_r(params, transforms, profiles, data, **kwargs):
    data["G_r"] = surface_averages(transforms["grid"], data["B_zeta_r"])
    return data


@register_compute_fun(
    name="G_rr",
    label="\\partial_{\\rho\\rho} G",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Boozer poloidal current enclosed by flux surfaces, second derivative "
    + "wrt radial coordinate",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["B_zeta_rr"],
)
def _G_rr(params, transforms, profiles, data, **kwargs):
    data["G_rr"] = surface_averages(transforms["grid"], data["B_zeta_rr"])
    return data


@register_compute_fun(
    name="current",
    label="\\frac{2\\pi}{\\mu_0} I",
    units="A",
    units_long="Amperes",
    description="Net toroidal current enclosed by flux surfaces",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["I"],
)
def _current(params, transforms, profiles, data, **kwargs):
    data["current"] = 2 * jnp.pi / mu_0 * data["I"]
    return data


@register_compute_fun(
    name="current_r",
    label="\\frac{2\\pi}{\\mu_0} \\partial_{\\rho} I",
    units="A",
    units_long="Amperes",
    description="Net toroidal current enclosed by flux surfaces, derivative "
    + "wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["I_r"],
)
def _current_r(params, transforms, profiles, data, **kwargs):
    data["current_r"] = 2 * jnp.pi / mu_0 * data["I_r"]
    return data


@register_compute_fun(
    name="current_rr",
    label="\\frac{2\\pi}{\\mu_0} \\partial_{\\rho\\rho} I",
    units="A",
    units_long="Amperes",
    description="Net toroidal current enclosed by flux surfaces, second derivative "
    + "wrt radial coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["I_rr"],
)
def _current_rr(params, transforms, profiles, data, **kwargs):
    data["current_rr"] = 2 * jnp.pi / mu_0 * data["I_rr"]
    return data
