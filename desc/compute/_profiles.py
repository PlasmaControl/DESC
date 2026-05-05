"""Compute functions for profiles.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from quadax import cumulative_simpson
from scipy.constants import elementary_charge, mu_0

from desc.backend import cond, jnp

from ..integrals.surface_integral import surface_averages, surface_integrals
from ..utils import dot, safediv
from .data_index import register_compute_fun


@register_compute_fun(
    name="Psi",
    label="\\Psi",
    units="Wb",
    units_long="Webers",
    description="Toroidal flux",
    dim=1,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["rho"],
)
def _Psi(params, transforms, profiles, data, **kwargs):
    data["Psi"] = params["Psi"] * data["rho"] ** 2
    return data


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
    data=["1"],
)
def _psi_rr(params, transforms, profiles, data, **kwargs):
    data["psi_rr"] = data["1"] * params["Psi"] / jnp.pi
    return data


@register_compute_fun(
    name="psi_rrr",
    label="\\partial_{\\rho\\rho\\rho} \\psi = \\partial_{\\rho\\rho\\rho} \\Psi / "
    + "(2 \\pi)",
    units="Wb",
    units_long="Webers",
    description="Toroidal flux (normalized by 2pi), third radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["0"],
)
def _psi_rrr(params, transforms, profiles, data, **kwargs):
    data["psi_rrr"] = data["0"]
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
    data=["chi_r", "rho"],
    resolution_requirement="r",
)
def _chi(params, transforms, profiles, data, **kwargs):
    chi = cumulative_simpson(
        y=transforms["grid"].compress(data["chi_r"]),
        x=transforms["grid"].compress(data["rho"]),
        initial=0,
    )
    data["chi"] = transforms["grid"].expand(chi)
    return data


@register_compute_fun(
    name="Te",
    label="T_e",
    units="eV",
    units_long="electron-Volts",
    description="Electron temperature",
    dim=1,
    params=["Te_l"],
    transforms={"grid": []},
    profiles=["electron_temperature"],
    coordinates="r",
    data=["0"],
)
def _Te(params, transforms, profiles, data, **kwargs):
    if profiles["electron_temperature"] is not None:
        data["Te"] = profiles["electron_temperature"].compute(
            transforms["grid"], params["Te_l"], dr=0
        )
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
    transforms={"grid": []},
    profiles=["electron_temperature"],
    coordinates="r",
    data=["0"],
)
def _Te_r(params, transforms, profiles, data, **kwargs):
    if profiles["electron_temperature"] is not None:
        data["Te_r"] = profiles["electron_temperature"].compute(
            transforms["grid"], params["Te_l"], dr=1
        )
    else:
        data["Te_r"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="Te_rr",
    label="\\partial_{\\rho \\rho} T_e",
    units="eV",
    units_long="electron-Volts",
    description="Electron temperature, second radial derivative",
    dim=1,
    params=["Te_l"],
    transforms={"grid": []},
    profiles=["electron_temperature"],
    coordinates="r",
    data=["0"],
)
def _Te_rr(params, transforms, profiles, data, **kwargs):
    if profiles["electron_temperature"] is not None:
        data["Te_rr"] = profiles["electron_temperature"].compute(
            transforms["grid"], params["Te_l"], dr=2
        )
    else:
        data["Te_rr"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="<ne>_vol",
    label="\\langle n_e \\rangle_{vol}",
    units="m^{-3}",
    units_long="1 / cubic meters",
    description="Volume average electron density",
    dim=0,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="",
    data=["ne", "sqrt(g)", "V"],
    resolution_requirement="rtz",
)
def _ne_vol(params, transforms, profiles, data, **kwargs):
    data["<ne>_vol"] = (
        jnp.sum(data["ne"] * data["sqrt(g)"] * transforms["grid"].weights) / data["V"]
    )
    return data


@register_compute_fun(
    name="ne",
    label="n_e",
    units="m^{-3}",
    units_long="1 / cubic meters",
    description="Electron density",
    dim=1,
    params=["ne_l"],
    transforms={"grid": []},
    profiles=["electron_density"],
    coordinates="r",
    data=["0"],
)
def _ne(params, transforms, profiles, data, **kwargs):
    if profiles["electron_density"] is not None:
        data["ne"] = profiles["electron_density"].compute(
            transforms["grid"], params["ne_l"], dr=0
        )
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
    transforms={"grid": []},
    profiles=["electron_density"],
    coordinates="r",
    data=["0"],
)
def _ne_r(params, transforms, profiles, data, **kwargs):
    if profiles["electron_density"] is not None:
        data["ne_r"] = profiles["electron_density"].compute(
            transforms["grid"], params["ne_l"], dr=1
        )
    else:
        data["ne_r"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="ne_rr",
    label="\\partial_{\\rho \\rho} n_e",
    units="m^{-3}",
    units_long="1 / cubic meters",
    description="Electron density, second radial derivative",
    dim=1,
    params=["ne_l"],
    transforms={"grid": []},
    profiles=["electron_density"],
    coordinates="r",
    data=["0"],
)
def _ne_rr(params, transforms, profiles, data, **kwargs):
    if profiles["electron_density"] is not None:
        data["ne_rr"] = profiles["electron_density"].compute(
            transforms["grid"], params["ne_l"], dr=2
        )
    else:
        data["ne_rr"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="Ti",
    label="T_i",
    units="eV",
    units_long="electron-Volts",
    description="Ion temperature",
    dim=1,
    params=["Ti_l"],
    transforms={"grid": []},
    profiles=["ion_temperature"],
    coordinates="r",
    data=["0"],
)
def _Ti(params, transforms, profiles, data, **kwargs):
    if profiles["ion_temperature"] is not None:
        data["Ti"] = profiles["ion_temperature"].compute(
            transforms["grid"], params["Ti_l"], dr=0
        )
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
    transforms={"grid": []},
    profiles=["ion_temperature"],
    coordinates="r",
    data=["0"],
)
def _Ti_r(params, transforms, profiles, data, **kwargs):
    if profiles["ion_temperature"] is not None:
        data["Ti_r"] = profiles["ion_temperature"].compute(
            transforms["grid"], params["Ti_l"], dr=1
        )
    else:
        data["Ti_r"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="Ti_rr",
    label="\\partial_{\\rho \\rho} T_i",
    units="eV",
    units_long="electron-Volts",
    description="Ion temperature, second radial derivative",
    dim=1,
    params=["Ti_l"],
    transforms={"grid": []},
    profiles=["ion_temperature"],
    coordinates="r",
    data=["0"],
)
def _Ti_rr(params, transforms, profiles, data, **kwargs):
    if profiles["ion_temperature"] is not None:
        data["Ti_rr"] = profiles["ion_temperature"].compute(
            transforms["grid"], params["Ti_l"], dr=2
        )
    else:
        data["Ti_rr"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="ni",
    label="n_i",
    units="m^{-3}",
    units_long="1 / cubic meters",
    description="Ion density",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["ne", "Zeff"],
)
def _ni(params, transforms, profiles, data, **kwargs):
    data["ni"] = data["ne"] / data["Zeff"]
    return data


@register_compute_fun(
    name="ni_r",
    label="\\partial_{\\rho} n_i",
    units="m^{-3}",
    units_long="1 / cubic meters",
    description="Ion density, first radial derivative",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["ne", "ne_r", "Zeff", "Zeff_r"],
)
def _ni_r(params, transforms, profiles, data, **kwargs):
    data["ni_r"] = (data["ne_r"] * data["Zeff"] - data["ne"] * data["Zeff_r"]) / data[
        "Zeff"
    ] ** 2
    return data


@register_compute_fun(
    name="Zeff",
    label="Z_{eff}",
    units="~",
    units_long="None",
    description="Effective atomic number",
    dim=1,
    params=["Zeff_l"],
    transforms={"grid": []},
    profiles=["atomic_number"],
    coordinates="r",
    data=["0"],
)
def _Zeff(params, transforms, profiles, data, **kwargs):
    if profiles["atomic_number"] is not None:
        data["Zeff"] = profiles["atomic_number"].compute(
            transforms["grid"], params["Zeff_l"], dr=0
        )
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
    transforms={"grid": []},
    profiles=["atomic_number"],
    coordinates="r",
    data=["0"],
)
def _Zeff_r(params, transforms, profiles, data, **kwargs):
    if profiles["atomic_number"] is not None:
        data["Zeff_r"] = profiles["atomic_number"].compute(
            transforms["grid"], params["Zeff_l"], dr=1
        )
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
    transforms={"grid": []},
    profiles=["pressure"],
    coordinates="r",
    data=["ne", "ni", "Te", "Ti"],
    aliases=["pressure"],
)
def _p(params, transforms, profiles, data, **kwargs):
    if profiles["pressure"] is not None:
        data["p"] = profiles["pressure"].compute(
            transforms["grid"], params["p_l"], dr=0
        )
    else:
        data["p"] = elementary_charge * (
            data["ne"] * data["Te"] + data["ni"] * data["Ti"]
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
    transforms={"grid": []},
    profiles=["pressure"],
    coordinates="r",
    data=["ne", "ne_r", "ni", "ni_r", "Te", "Te_r", "Ti", "Ti_r"],
)
def _p_r(params, transforms, profiles, data, **kwargs):
    if profiles["pressure"] is not None:
        data["p_r"] = profiles["pressure"].compute(
            transforms["grid"], params["p_l"], dr=1
        )
    else:
        data["p_r"] = elementary_charge * (
            data["ne_r"] * data["Te"]
            + data["ne"] * data["Te_r"]
            + data["ni_r"] * data["Ti"]
            + data["ni"] * data["Ti_r"]
        )
    return data


@register_compute_fun(
    name="p_t",
    label="\\partial_{\\theta} p",
    units="Pa",
    units_long="Pascals",
    description="Pressure, first poloidal derivative",
    dim=1,
    params=["p_l"],
    transforms={"grid": []},
    profiles=["pressure"],
    coordinates="rtz",
    data=[],
)
def _p_t(params, transforms, profiles, data, **kwargs):
    data["p_t"] = profiles["pressure"].compute(transforms["grid"], params["p_l"], dt=1)
    return data


@register_compute_fun(
    name="p_z",
    label="\\partial_{\\zeta} p",
    units="Pa",
    units_long="Pascals",
    description="Pressure, first toroidal derivative",
    dim=1,
    params=["p_l"],
    transforms={"grid": []},
    profiles=["pressure"],
    coordinates="rtz",
    data=[],
)
def _p_z(params, transforms, profiles, data, **kwargs):
    data["p_z"] = profiles["pressure"].compute(transforms["grid"], params["p_l"], dz=1)
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
    data=["p_r", "p_t", "p_z", "e^rho", "e^theta", "e^zeta"],
)
def _gradp(params, transforms, profiles, data, **kwargs):
    data["grad(p)"] = (
        data["p_r"] * data["e^rho"].T
        # e^theta is blows up at the axis but the limit should go to zero
        # if pressure is analytic
        + data["p_t"] * jnp.where(data["p_t"] == 0, 0, data["e^theta"].T)
        + data["p_z"] * data["e^zeta"].T
    ).T
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
    resolution_requirement="rtz",
)
def _gradp_mag_vol(params, transforms, profiles, data, **kwargs):
    data["<|grad(p)|>_vol"] = (
        jnp.sum(data["|grad(p)|"] * data["sqrt(g)"] * transforms["grid"].weights)
        / data["V"]
    )
    return data


@register_compute_fun(
    name="beta_a",
    label="\\beta_a = \\mu_0 (p_{||} - p_{\\perp})/B^2",
    units="~",
    units_long="None",
    description="Pressure anisotropy",
    dim=1,
    params=["a_lmn"],
    transforms={"grid": []},
    profiles=["anisotropy"],
    coordinates="rtz",
    data=["0"],
)
def _beta_a(params, transforms, profiles, data, **kwargs):
    if profiles["anisotropy"] is not None:
        data["beta_a"] = profiles["anisotropy"].compute(
            transforms["grid"], params["a_lmn"], dr=0
        )
    else:
        data["beta_a"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="beta_a_r",
    label="\\partial_{\\rho} \\beta_a = \\mu_0 (p_{||} - p_{\\perp})/B^2",
    units="~",
    units_long="None",
    description="Pressure anisotropy, first radial derivative",
    dim=1,
    params=["a_lmn"],
    transforms={"grid": []},
    profiles=["anisotropy"],
    coordinates="rtz",
    data=["0"],
)
def _beta_a_r(params, transforms, profiles, data, **kwargs):
    if profiles["anisotropy"] is not None:
        data["beta_a_r"] = profiles["anisotropy"].compute(
            transforms["grid"], params["a_lmn"], dr=1
        )
    else:
        data["beta_a_r"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="beta_a_t",
    label="\\partial_{\\theta} \\beta_a = \\mu_0 (p_{||} - p_{\\perp})/B^2",
    units="~",
    units_long="None",
    description="Pressure anisotropy, first poloidal derivative",
    dim=1,
    params=["a_lmn"],
    transforms={"grid": []},
    profiles=["anisotropy"],
    coordinates="rtz",
    data=["0"],
)
def _beta_a_t(params, transforms, profiles, data, **kwargs):
    if profiles["anisotropy"] is not None:
        data["beta_a_t"] = profiles["anisotropy"].compute(
            transforms["grid"], params["a_lmn"], dt=1
        )
    else:
        data["beta_a_t"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="beta_a_z",
    label="\\partial_{\\zeta} \\beta_a = \\mu_0 (p_{||} - p_{\\perp})/B^2",
    units="~",
    units_long="None",
    description="Pressure anisotropy, first toroidal derivative",
    dim=1,
    params=["a_lmn"],
    transforms={"grid": []},
    profiles=["anisotropy"],
    coordinates="rtz",
    data=["0"],
)
def _beta_a_z(params, transforms, profiles, data, **kwargs):
    if profiles["anisotropy"] is not None:
        data["beta_a_z"] = profiles["anisotropy"].compute(
            transforms["grid"], params["a_lmn"], dz=1
        )
    else:
        data["beta_a_z"] = jnp.nan * data["0"]
    return data


@register_compute_fun(
    name="grad(beta_a)",
    label="\\nabla \\beta_a = \\nabla \\mu_0 (p_{||} - p_{\\perp})/B^2",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Pressure anisotropy gradient",
    dim=3,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["beta_a_r", "beta_a_t", "beta_a_z", "e^rho", "e^theta", "e^zeta"],
)
def _gradbeta_a(params, transforms, profiles, data, **kwargs):
    data["grad(beta_a)"] = (
        data["beta_a_r"] * data["e^rho"].T
        + data["beta_a_t"] * data["e^theta"].T
        + data["beta_a_z"] * data["e^zeta"].T
    ).T
    return data


@register_compute_fun(
    name="iota",
    label="\\iota",
    units="~",
    units_long="None",
    description="Rotational transform (normalized by 2pi)",
    dim=1,
    params=["i_l"],
    transforms={"grid": []},
    profiles=["iota", "current"],
    coordinates="r",
    data=["iota_den", "iota_num"],
    axis_limit_data=["iota_den_r", "iota_num_r"],
)
def _iota(params, transforms, profiles, data, **kwargs):
    if profiles["iota"] is not None:
        data["iota"] = profiles["iota"].compute(transforms["grid"], params["i_l"], dr=0)
    elif profiles["current"] is not None:
        # See the document attached to GitHub pull request #556 for the math.
        # Assumes ζ = ϕ − ω and θ = ϑ − λ.
        data["iota"] = transforms["grid"].replace_at_axis(
            safediv(data["iota_num"], data["iota_den"]),
            lambda: safediv(data["iota_num_r"], data["iota_den_r"]),
        )
    return data


@register_compute_fun(
    name="iota_r",
    label="\\partial_{\\rho} \\iota",
    units="~",
    units_long="None",
    description="Rotational transform (normalized by 2pi), first radial derivative",
    dim=1,
    params=["i_l"],
    transforms={"grid": []},
    profiles=["iota", "current"],
    coordinates="r",
    data=["iota_den", "iota_den_r", "iota_num", "iota_num_r"],
    axis_limit_data=["iota_den_rr", "iota_num_rr"],
)
def _iota_r(params, transforms, profiles, data, **kwargs):
    if profiles["iota"] is not None:
        data["iota_r"] = profiles["iota"].compute(
            transforms["grid"], params["i_l"], dr=1
        )
    elif profiles["current"] is not None:
        # See the document attached to GitHub pull request #556 for the math.
        # Assumes ζ = ϕ − ω and θ = ϑ − λ.
        data["iota_r"] = transforms["grid"].replace_at_axis(
            safediv(
                data["iota_num_r"] * data["iota_den"]
                - data["iota_num"] * data["iota_den_r"],
                data["iota_den"] ** 2,
            ),
            lambda: safediv(
                data["iota_num_rr"] * data["iota_den_r"]
                - data["iota_num_r"] * data["iota_den_rr"],
                (2 * data["iota_den_r"] ** 2),
            ),
        )
    return data


@register_compute_fun(
    name="iota_rr",
    label="\\partial_{\\rho\\rho} \\iota",
    units="~",
    units_long="None",
    description="Rotational transform (normalized by 2pi), second radial derivative",
    dim=1,
    params=["i_l"],
    transforms={"grid": []},
    profiles=["iota", "current"],
    coordinates="r",
    data=[
        "iota_den",
        "iota_den_r",
        "iota_den_rr",
        "iota_num",
        "iota_num_r",
        "iota_num_rr",
    ],
    axis_limit_data=["iota_den_rrr", "iota_num_rrr"],
)
def _iota_rr(params, transforms, profiles, data, **kwargs):
    if profiles["iota"] is not None:
        data["iota_rr"] = profiles["iota"].compute(
            transforms["grid"], params["i_l"], dr=2
        )
    elif profiles["current"] is not None:
        # See the document attached to GitHub pull request #556 for the math.
        # Assumes ζ = ϕ − ω and θ = ϑ − λ.
        data["iota_rr"] = transforms["grid"].replace_at_axis(
            safediv(
                data["iota_num_rr"] * data["iota_den"] ** 2
                - 2 * data["iota_num_r"] * data["iota_den"] * data["iota_den_r"]
                + 2 * data["iota_num"] * data["iota_den_r"] ** 2
                - data["iota_num"] * data["iota_den"] * data["iota_den_rr"],
                data["iota_den"] ** 3,
            ),
            lambda: safediv(
                2 * data["iota_num_rrr"] * data["iota_den_r"] ** 2
                - 3 * data["iota_num_rr"] * data["iota_den_r"] * data["iota_den_rr"]
                + 3 * data["iota_num_r"] * data["iota_den_rr"] ** 2
                - 2 * data["iota_num_r"] * data["iota_den_r"] * data["iota_den_rrr"],
                (6 * data["iota_den_r"] ** 3),
            ),
        )
    return data


@register_compute_fun(
    name="iota current",
    label="\\iota~\\mathrm{from~current}",
    units="~",
    units_long="None",
    description="Rotational transform (normalized by 2pi), current contribution",
    dim=1,
    params=["i_l"],
    transforms={"grid": []},
    profiles=["iota", "current"],
    coordinates="r",
    data=["iota vacuum", "iota_den", "iota_num current"],
    axis_limit_data=["iota_den_r", "iota_num_r current"],
)
def _iota_current(params, transforms, profiles, data, **kwargs):
    if profiles["iota"] is not None:
        iota = profiles["iota"].compute(transforms["grid"], params["i_l"], dr=0)
        data["iota current"] = iota - data["iota vacuum"]
    elif profiles["current"] is not None:
        data["iota current"] = transforms["grid"].replace_at_axis(
            safediv(data["iota_num current"], data["iota_den"]),
            lambda: safediv(data["iota_num_r current"], data["iota_den_r"]),
        )
    return data


@register_compute_fun(
    name="iota vacuum",
    label="\\iota~\\mathrm{in~vacuum}",
    units="~",
    units_long="None",
    description="Rotational transform (normalized by 2pi), vacuum contribution",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["iota_den", "iota_num vacuum"],
    axis_limit_data=["iota_den_r", "iota_num_r vacuum"],
)
def _iota_vacuum(params, transforms, profiles, data, **kwargs):
    data["iota vacuum"] = transforms["grid"].replace_at_axis(
        safediv(data["iota_num vacuum"], data["iota_den"]),
        lambda: safediv(data["iota_num_r vacuum"], data["iota_den_r"]),
    )
    return data


@register_compute_fun(
    name="iota_num current",
    label="\\iota_{\\mathrm{numerator}}~\\mathrm{from~current}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Numerator of rotational transform formula, current contribution",
    dim=1,
    params=["c_l", "i_l"],
    transforms={"grid": []},
    profiles=["current", "iota"],
    coordinates="r",
    data=["psi_r", "iota_den", "iota_num vacuum"],
    axis_limit_data=["psi_rr"],
)
def _iota_num_current(params, transforms, profiles, data, **kwargs):
    """Current contribution to the numerator of rotational transform formula."""
    if profiles["iota"] is not None:
        iota = profiles["iota"].compute(transforms["grid"], params["i_l"], dr=0)
        data["iota_num current"] = iota * data["iota_den"] - data["iota_num vacuum"]
    elif profiles["current"] is not None:
        # 4π² I = 4π² (μ₀ current / 2π) = 2π μ₀ current
        current = profiles["current"].compute(transforms["grid"], params["c_l"], dr=0)
        current_r = profiles["current"].compute(transforms["grid"], params["c_l"], dr=1)
        data["iota_num current"] = (
            2
            * jnp.pi
            * mu_0
            * transforms["grid"].replace_at_axis(
                safediv(current, data["psi_r"]),
                lambda: safediv(current_r, data["psi_rr"]),
            )
        )
    return data


@register_compute_fun(
    name="iota_num vacuum",
    label="\\iota_{\\mathrm{numerator}}~\\mathrm{in~vacuum}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Numerator of rotational transform formula, vacuum contribution",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["lambda_z", "g_tt", "lambda_t", "g_tz", "sqrt(g)"],
    axis_limit_data=["g_tz_r", "sqrt(g)_r"],
    resolution_requirement="tz",
)
def _iota_num_vacuum(params, transforms, profiles, data, **kwargs):
    """Vacuum contribution to the numerator of rotational transform formula."""
    # Assumes ζ = ϕ − ω and θ = ϑ − λ.
    iota_num_vacuum = transforms["grid"].replace_at_axis(
        safediv(
            data["lambda_z"] * data["g_tt"] - (1 + data["lambda_t"]) * data["g_tz"],
            data["sqrt(g)"],
        ),
        lambda: safediv(-(1 + data["lambda_t"]) * data["g_tz_r"], data["sqrt(g)_r"]),
    )
    data["iota_num vacuum"] = surface_integrals(transforms["grid"], iota_num_vacuum)
    return data


@register_compute_fun(
    name="iota_num_r current",
    label="\\partial_{\\rho} \\iota_{\\mathrm{numerator}}~\\mathrm{from~current}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Numerator of rotational transform formula, current contribution, "
    + "first radial derivative",
    dim=1,
    params=["c_l", "i_l"],
    transforms={"grid": []},
    profiles=["current", "iota"],
    coordinates="r",
    data=["psi_r", "psi_rr", "iota_den", "iota_den_r", "iota_num", "iota_num_r vacuum"],
)
def _iota_num_r_current(params, transforms, profiles, data, **kwargs):
    if profiles["iota"] is not None:
        iota_r = profiles["iota"].compute(transforms["grid"], params["i_l"], dr=1)
        data["iota_num_r current"] = (
            iota_r * data["iota_den"] ** 2 + data["iota_num"] * data["iota_den_r"]
        ) / data["iota_den"] - data["iota_num_r vacuum"]
    elif profiles["current"] is not None:
        # 4π^2 I = 4π^2 (mu_0 current / 2π) = 2π mu_0 current
        current = profiles["current"].compute(transforms["grid"], params["c_l"], dr=0)
        current_r = profiles["current"].compute(transforms["grid"], params["c_l"], dr=1)
        current_rr = profiles["current"].compute(
            transforms["grid"], params["c_l"], dr=2
        )
        data["iota_num_r current"] = (
            2
            * jnp.pi
            * mu_0
            * transforms["grid"].replace_at_axis(
                safediv(
                    current_r * data["psi_r"] - current * data["psi_rr"],
                    data["psi_r"] ** 2,
                ),
                lambda: safediv(current_rr, (2 * data["psi_rr"])),
            )
        )
    return data


@register_compute_fun(
    name="iota_num_r vacuum",
    label="\\partial_{\\rho} \\iota_{\\mathrm{numerator}}~\\mathrm{in~vacuum}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Numerator of rotational transform formula, vacuum contribution, "
    + "first radial derivative",
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
    axis_limit_data=["g_tt_rr", "g_tz_rr", "sqrt(g)_rr"],
    resolution_requirement="tz",
)
def _iota_num_r_vacuum(params, transforms, profiles, data, **kwargs):
    # Assumes ζ = ϕ − ω and θ = ϑ − λ.
    iota_num_vacuum = safediv(
        data["lambda_z"] * data["g_tt"] - (1 + data["lambda_t"]) * data["g_tz"],
        data["sqrt(g)"],
    )
    iota_num_r_vacuum = transforms["grid"].replace_at_axis(
        safediv(
            data["lambda_rz"] * data["g_tt"]
            + data["lambda_z"] * data["g_tt_r"]
            - data["lambda_rt"] * data["g_tz"]
            - (1 + data["lambda_t"]) * data["g_tz_r"]
            - iota_num_vacuum * data["sqrt(g)_r"],
            data["sqrt(g)"],
        ),
        lambda: (
            safediv(
                (1 + data["lambda_t"]) * data["g_tz_r"] * data["sqrt(g)_rr"],
                (2 * data["sqrt(g)_r"] ** 2),
            )
            + safediv(
                data["lambda_z"] * data["g_tt_rr"]
                - 2 * data["lambda_rt"] * data["g_tz_r"]
                - (1 + data["lambda_t"]) * data["g_tz_rr"],
                (2 * data["sqrt(g)_r"]),
            )
        ),
    )
    data["iota_num_r vacuum"] = surface_integrals(transforms["grid"], iota_num_r_vacuum)
    return data


@register_compute_fun(
    name="iota_num",
    label="\\iota_{\\mathrm{numerator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Numerator of rotational transform formula",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["iota_num current", "iota_num vacuum"],
)
def _iota_num(params, transforms, profiles, data, **kwargs):
    """Numerator of rotational transform formula.

    Computes 𝛼 + 𝛽 as defined in the document attached to the description
    of GitHub pull request #556. 𝛼 supplements the rotational transform with an
    additional term to account for the enclosed net toroidal current.
    """
    data["iota_num"] = data["iota_num current"] + data["iota_num vacuum"]
    return data


@register_compute_fun(
    name="iota_num_r",
    label="\\partial_{\\rho} \\iota_{\\mathrm{numerator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Numerator of rotational transform formula, first radial derivative",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["iota_num_r current", "iota_num_r vacuum"],
)
def _iota_num_r(params, transforms, profiles, data, **kwargs):
    """Numerator of rotational transform formula, first radial derivative.

    Computes d(𝛼+𝛽)/d𝜌 as defined in the document attached to the description
    of GitHub pull request #556. 𝛼 supplements the rotational transform with an
    additional term to account for the enclosed net toroidal current.
    """
    data["iota_num_r"] = data["iota_num_r current"] + data["iota_num_r vacuum"]
    return data


@register_compute_fun(
    name="iota_num_rr",
    label="\\partial_{\\rho\\rho} \\iota_{\\mathrm{numerator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Numerator of rotational transform formula, second radial derivative",
    dim=1,
    params=["c_l"],
    transforms={"grid": []},
    profiles=["current", "iota"],
    coordinates="r",
    data=[
        "0",
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
        "psi_r",
        "psi_rr",
        "psi_rrr",
    ],
    axis_limit_data=["sqrt(g)_rrr", "g_tt_rrr", "g_tz_rrr"],
    resolution_requirement="tz",
)
def _iota_num_rr(params, transforms, profiles, data, **kwargs):
    """Numerator of rotational transform formula, second radial derivative.

    Computes d2(𝛼+𝛽)/d𝜌2 as defined in the document attached to the description
    of GitHub pull request #556. 𝛼 supplements the rotational transform with an
    additional term to account for the enclosed net toroidal current.
    Assumes ζ = ϕ − ω and θ = ϑ − λ.
    """
    if profiles["iota"] is not None:
        data["iota_num_rr"] = jnp.nan * data["0"]
    elif profiles["current"] is not None:
        current = profiles["current"].compute(transforms["grid"], params["c_l"], dr=0)
        current_r = profiles["current"].compute(transforms["grid"], params["c_l"], dr=1)
        current_rr = profiles["current"].compute(
            transforms["grid"], params["c_l"], dr=2
        )
        current_rrr = profiles["current"].compute(
            transforms["grid"], params["c_l"], dr=3
        )
        # 4π² I = 4π² (μ₀ current / 2π) = 2π μ₀ current
        alpha_rr = (
            jnp.pi
            * mu_0
            * transforms["grid"].replace_at_axis(
                2 * safediv(current_rr, data["psi_r"])
                - safediv(4 * current_r * data["psi_rr"], data["psi_r"] ** 2)
                + 2
                * current
                * safediv(
                    2 * data["psi_rr"] ** 2 - data["psi_rrr"] * data["psi_r"],
                    data["psi_r"] ** 3,
                ),
                lambda: safediv(2 * current_rrr, (3 * data["psi_rr"]))
                - safediv(current_rr * data["psi_rrr"], data["psi_rr"] ** 2)
                + safediv(current_r * data["psi_rrr"] ** 2, data["psi_rr"] ** 3),
            )
        )
        beta = safediv(
            data["lambda_z"] * data["g_tt"] - (1 + data["lambda_t"]) * data["g_tz"],
            data["sqrt(g)"],
        )
        beta_r = safediv(
            data["lambda_rz"] * data["g_tt"]
            + data["lambda_z"] * data["g_tt_r"]
            - data["lambda_rt"] * data["g_tz"]
            - (1 + data["lambda_t"]) * data["g_tz_r"]
            - beta * data["sqrt(g)_r"],
            data["sqrt(g)"],
        )
        beta_rr = transforms["grid"].replace_at_axis(
            safediv(
                data["lambda_rrz"] * data["g_tt"]
                + 2 * data["lambda_rz"] * data["g_tt_r"]
                + data["lambda_z"] * data["g_tt_rr"]
                - data["lambda_rrt"] * data["g_tz"]
                - 2 * data["lambda_rt"] * data["g_tz_r"]
                - (1 + data["lambda_t"]) * data["g_tz_rr"]
                - 2 * beta_r * data["sqrt(g)_r"]
                - beta * data["sqrt(g)_rr"],
                data["sqrt(g)"],
            ),
            lambda: safediv(
                2
                * data["sqrt(g)_r"] ** 2
                * (
                    3 * data["g_tt_rr"] * data["lambda_rz"]
                    + data["g_tt_rrr"] * data["lambda_z"]
                    - 3 * data["g_tz_rr"] * data["lambda_rt"]
                    - 3 * data["g_tz_r"] * data["lambda_rrt"]
                    - data["g_tz_rrr"] * (1 + data["lambda_t"])
                )
                + data["sqrt(g)_r"]
                * (
                    3
                    * data["sqrt(g)_rr"]
                    * (
                        2 * data["g_tz_r"] * data["lambda_rt"]
                        - data["g_tt_rr"] * data["lambda_t"]
                        + data["g_tz_rr"] * (1 + data["lambda_t"])
                    )
                    + 2 * data["sqrt(g)_rrr"] * data["g_tz_r"] * (1 + data["lambda_t"])
                )
                - 3 * data["sqrt(g)_rr"] ** 2 * data["g_tz_r"] * (1 + data["lambda_t"]),
                (6 * data["sqrt(g)_r"] ** 3),
            ),
        )
        beta_rr = surface_integrals(transforms["grid"], beta_rr)
        data["iota_num_rr"] = alpha_rr + beta_rr
    return data


@register_compute_fun(
    name="iota_num_rrr",
    label="\\partial_{\\rho\\rho\\rho} \\iota_{\\mathrm{numerator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Numerator of rotational transform formula, third radial derivative",
    dim=1,
    params=["c_l"],
    transforms={"grid": []},
    profiles=["current", "iota"],
    coordinates="r",
    data=[
        "0",
        "lambda_t",
        "lambda_rt",
        "lambda_rrt",
        "lambda_rrrt",
        "lambda_z",
        "lambda_rz",
        "lambda_rrz",
        "lambda_rrrz",
        "g_tt",
        "g_tt_r",
        "g_tt_rr",
        "g_tt_rrr",
        "g_tz",
        "g_tz_r",
        "g_tz_rr",
        "g_tz_rrr",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rr",
        "sqrt(g)_rrr",
        "psi_r",
        "psi_rr",
        "psi_rrr",
    ],
    resolution_requirement="tz",
)
def _iota_num_rrr(params, transforms, profiles, data, **kwargs):
    """Numerator of rotational transform formula, third radial derivative.

    Computes d3(𝛼+𝛽)/d𝜌3 as defined in the document attached to the description
    of GitHub pull request #556. 𝛼 supplements the rotational transform with an
    additional term to account for the enclosed net toroidal current.
    Assumes ζ = ϕ − ω and θ = ϑ − λ.
    """
    if profiles["iota"] is not None:
        data["iota_num_rrr"] = jnp.nan * data["0"]
    elif profiles["current"] is not None:
        current = profiles["current"].compute(transforms["grid"], params["c_l"], dr=0)
        current_r = profiles["current"].compute(transforms["grid"], params["c_l"], dr=1)
        current_rr = profiles["current"].compute(
            transforms["grid"], params["c_l"], dr=2
        )
        current_rrr = profiles["current"].compute(
            transforms["grid"], params["c_l"], dr=3
        )
        current_rrrr = profiles["current"].compute(
            transforms["grid"], params["c_l"], dr=4
        )
        # 4π² I = 4π² (μ₀ current / 2π) = 2π μ₀ current
        alpha_rrr = (
            jnp.pi
            * mu_0
            * transforms["grid"].replace_at_axis(
                safediv(2 * current_rrr, data["psi_r"])
                - safediv(6 * current_rr * data["psi_rr"], data["psi_r"] ** 2)
                + safediv(
                    6
                    * current_r
                    * (
                        2 * data["psi_r"] * data["psi_rr"] ** 2
                        - data["psi_rrr"] * data["psi_r"] ** 2
                    ),
                    data["psi_r"] ** 4,
                )
                + safediv(
                    12
                    * current
                    * (
                        data["psi_rrr"] * data["psi_rr"] * data["psi_r"]
                        - data["psi_rr"] ** 3
                    ),
                    data["psi_r"] ** 4,
                ),
                lambda: safediv(current_rrrr, (2 * data["psi_rr"]))
                - safediv(current_rrr * data["psi_rrr"], data["psi_rr"] ** 2)
                + safediv(
                    3 * current_rr * data["psi_rrr"] ** 2, (2 * data["psi_rr"] ** 3)
                )
                - safediv(
                    3 * current_r * data["psi_rrr"] ** 3, (2 * data["psi_rr"] ** 4)
                ),
            )
        )
        beta = safediv(
            data["lambda_z"] * data["g_tt"] - (1 + data["lambda_t"]) * data["g_tz"],
            data["sqrt(g)"],
        )
        beta_r = safediv(
            data["lambda_rz"] * data["g_tt"]
            + data["lambda_z"] * data["g_tt_r"]
            - data["lambda_rt"] * data["g_tz"]
            - (1 + data["lambda_t"]) * data["g_tz_r"]
            - beta * data["sqrt(g)_r"],
            data["sqrt(g)"],
        )
        beta_rr = safediv(
            data["lambda_rrz"] * data["g_tt"]
            + 2 * data["lambda_rz"] * data["g_tt_r"]
            + data["lambda_z"] * data["g_tt_rr"]
            - data["lambda_rrt"] * data["g_tz"]
            - 2 * data["lambda_rt"] * data["g_tz_r"]
            - (1 + data["lambda_t"]) * data["g_tz_rr"]
            - 2 * beta_r * data["sqrt(g)_r"]
            - beta * data["sqrt(g)_rr"],
            data["sqrt(g)"],
        )
        beta_rrr = transforms["grid"].replace_at_axis(
            safediv(
                data["lambda_rrrz"] * data["g_tt"]
                + 3 * data["lambda_rrz"] * data["g_tt_r"]
                + 3 * data["lambda_rz"] * data["g_tt_rr"]
                + data["lambda_z"] * data["g_tt_rrr"]
                - data["lambda_rrrt"] * data["g_tz"]
                - 3 * data["lambda_rrt"] * data["g_tz_r"]
                - 3 * data["lambda_rt"] * data["g_tz_rr"]
                - (1 + data["lambda_t"]) * data["g_tz_rrr"]
                - 3 * beta_rr * data["sqrt(g)_r"]
                - 3 * beta_r * data["sqrt(g)_rr"]
                - beta * data["sqrt(g)_rrr"],
                data["sqrt(g)"],
            ),
            # TODO(#587): axis limit of beta_rrr
            #   Computed with four applications of l’Hôpital’s rule.
            #   Requires sqrt(g)_rrrr and fourth derivatives of basis vectors.
            jnp.nan,
        )
        beta_rrr = surface_integrals(transforms["grid"], beta_rrr)
        # force limit to nan until completed because integration replaces nan with 0
        data["iota_num_rrr"] = alpha_rrr + transforms["grid"].replace_at_axis(
            beta_rrr, jnp.nan
        )
    return data


@register_compute_fun(
    name="iota_den",
    label="\\iota_{\\mathrm{denominator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Denominator of rotational transform formula",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["g_tt", "g_tz", "sqrt(g)", "omega_t", "omega_z"],
    resolution_requirement="tz",
)
def _iota_den(params, transforms, profiles, data, **kwargs):
    """Denominator of rotational transform formula.

    Computes 𝛾 as defined in the document attached to the description
    of GitHub pull request #556. Assumes ζ = ϕ − ω and θ = ϑ − λ.
    """
    gamma = safediv(
        (1 + data["omega_z"]) * data["g_tt"] - data["omega_t"] * data["g_tz"],
        data["sqrt(g)"],
    )
    # Assumes toroidal stream function behaves such that the magnetic axis limit
    # of γ is zero (as it would if ω = 0 identically).
    gamma = transforms["grid"].replace_at_axis(
        surface_integrals(transforms["grid"], gamma), 0
    )
    data["iota_den"] = gamma
    return data


@register_compute_fun(
    name="iota_den_r",
    label="\\partial_{\\rho} \\iota_{\\mathrm{denominator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Denominator of rotational transform formula, first radial derivative",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "g_tt",
        "g_tt_r",
        "g_tz",
        "g_tz_r",
        "sqrt(g)",
        "sqrt(g)_r",
        "omega_t",
        "omega_rt",
        "omega_z",
        "omega_rz",
    ],
    axis_limit_data=["sqrt(g)_rr", "g_tt_rr", "g_tz_rr"],
    resolution_requirement="tz",
)
def _iota_den_r(params, transforms, profiles, data, **kwargs):
    """Denominator of rotational transform formula, first radial derivative.

    Computes d𝛾/d𝜌 as defined in the document attached to the description
    of GitHub pull request #556. Assumes ζ = ϕ − ω and θ = ϑ − λ.
    """
    gamma = safediv(
        (1 + data["omega_z"]) * data["g_tt"] - data["omega_t"] * data["g_tz"],
        data["sqrt(g)"],
    )
    gamma_r = transforms["grid"].replace_at_axis(
        safediv(
            data["omega_rz"] * data["g_tt"]
            + (1 + data["omega_z"]) * data["g_tt_r"]
            - data["omega_rt"] * data["g_tz"]
            - data["omega_t"] * data["g_tz_r"]
            - gamma * data["sqrt(g)_r"],
            data["sqrt(g)"],
        ),
        lambda: (
            data["omega_t"]
            * data["g_tz_r"]
            * safediv(data["sqrt(g)_rr"], (2 * data["sqrt(g)_r"] ** 2))
            + safediv(
                (1 + data["omega_z"]) * data["g_tt_rr"]
                - 2 * data["omega_rt"] * data["g_tz_r"]
                - data["omega_t"] * data["g_tz_rr"],
                (2 * data["sqrt(g)_r"]),
            )
        ),
    )
    gamma_r = surface_integrals(transforms["grid"], gamma_r)
    data["iota_den_r"] = gamma_r
    return data


@register_compute_fun(
    name="iota_den_rr",
    label="\\partial_{\\rho\\rho} \\iota_{\\mathrm{denominator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Denominator of rotational transform formula, second radial derivative",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "g_tt",
        "g_tt_r",
        "g_tt_rr",
        "g_tz",
        "g_tz_r",
        "g_tz_rr",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rr",
        "omega_t",
        "omega_rt",
        "omega_rrt",
        "omega_z",
        "omega_rz",
        "omega_rrz",
    ],
    axis_limit_data=["sqrt(g)_rrr", "g_tt_rrr", "g_tz_rrr"],
    resolution_requirement="tz",
)
def _iota_den_rr(params, transforms, profiles, data, **kwargs):
    """Denominator of rotational transform formula, second radial derivative.

    Computes d2𝛾/d𝜌2 as defined in the document attached to the description
    of GitHub pull request #556. Assumes ζ = ϕ − ω and θ = ϑ − λ.
    """
    gamma = safediv(
        (1 + data["omega_z"]) * data["g_tt"] - data["omega_t"] * data["g_tz"],
        data["sqrt(g)"],
    )
    gamma_r = safediv(
        data["omega_rz"] * data["g_tt"]
        + (1 + data["omega_z"]) * data["g_tt_r"]
        - data["omega_rt"] * data["g_tz"]
        - data["omega_t"] * data["g_tz_r"]
        - gamma * data["sqrt(g)_r"],
        data["sqrt(g)"],
    )
    gamma_rr = transforms["grid"].replace_at_axis(
        safediv(
            data["omega_rrz"] * data["g_tt"]
            + 2 * data["omega_rz"] * data["g_tt_r"]
            + (1 + data["omega_z"]) * data["g_tt_rr"]
            - data["omega_rrt"] * data["g_tz"]
            - 2 * data["omega_rt"] * data["g_tz_r"]
            - data["omega_t"] * data["g_tz_rr"]
            - 2 * gamma_r * data["sqrt(g)_r"]
            - gamma * data["sqrt(g)_rr"],
            data["sqrt(g)"],
        ),
        lambda: safediv(
            2
            * data["sqrt(g)_r"] ** 2
            * (
                3 * data["g_tt_rr"] * data["omega_rz"]
                + data["g_tt_rrr"] * (1 + data["omega_z"])
                - 3 * data["g_tz_rr"] * data["omega_rt"]
                - 3 * data["g_tz_r"] * data["omega_rrt"]
                - data["g_tz_rrr"] * data["omega_t"]
            )
            + data["sqrt(g)_r"]
            * (
                3
                * data["sqrt(g)_rr"]
                * (
                    2 * data["g_tz_r"] * data["omega_rt"]
                    - data["g_tt_rr"] * (1 + data["omega_z"])
                    + data["g_tz_rr"] * data["omega_t"]
                )
                + 2 * data["sqrt(g)_rrr"] * data["g_tz_r"] * data["omega_t"]
            )
            - 3 * data["sqrt(g)_rr"] ** 2 * data["g_tz_r"] * data["omega_t"],
            (6 * data["sqrt(g)_r"] ** 3),
        ),
    )
    gamma_rr = surface_integrals(transforms["grid"], gamma_rr)
    data["iota_den_rr"] = gamma_rr
    return data


@register_compute_fun(
    name="iota_den_rrr",
    label="\\partial_{\\rho\\rho\\rho} \\iota_{\\mathrm{denominator}}",
    units="m^{-1}",
    units_long="inverse meters",
    description="Denominator of rotational transform formula, third radial derivative",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "g_tt",
        "g_tt_r",
        "g_tt_rr",
        "g_tt_rrr",
        "g_tz",
        "g_tz_r",
        "g_tz_rr",
        "g_tz_rrr",
        "sqrt(g)",
        "sqrt(g)_r",
        "sqrt(g)_rr",
        "sqrt(g)_rrr",
        "omega_t",
        "omega_rt",
        "omega_rrt",
        "omega_rrrt",
        "omega_z",
        "omega_rz",
        "omega_rrz",
        "omega_rrrz",
    ],
    resolution_requirement="tz",
)
def _iota_den_rrr(params, transforms, profiles, data, **kwargs):
    """Denominator of rotational transform formula, third radial derivative.

    Computes d3𝛾/d𝜌3 as defined in the document attached to the description
    of GitHub pull request #556. Assumes ζ = ϕ − ω and θ = ϑ − λ.
    """
    gamma = safediv(
        (1 + data["omega_z"]) * data["g_tt"] - data["omega_t"] * data["g_tz"],
        data["sqrt(g)"],
    )
    gamma_r = safediv(
        data["omega_rz"] * data["g_tt"]
        + (1 + data["omega_z"]) * data["g_tt_r"]
        - data["omega_rt"] * data["g_tz"]
        - data["omega_t"] * data["g_tz_r"]
        - gamma * data["sqrt(g)_r"],
        data["sqrt(g)"],
    )
    gamma_rr = safediv(
        data["omega_rrz"] * data["g_tt"]
        + 2 * data["omega_rz"] * data["g_tt_r"]
        + (1 + data["omega_z"]) * data["g_tt_rr"]
        - data["omega_rrt"] * data["g_tz"]
        - 2 * data["omega_rt"] * data["g_tz_r"]
        - data["omega_t"] * data["g_tz_rr"]
        - 2 * gamma_r * data["sqrt(g)_r"]
        - gamma * data["sqrt(g)_rr"],
        data["sqrt(g)"],
    )
    gamma_rrr = transforms["grid"].replace_at_axis(
        safediv(
            data["omega_rrrz"] * data["g_tt"]
            + 3 * data["omega_rrz"] * data["g_tt_r"]
            + 3 * data["omega_rz"] * data["g_tt_rr"]
            + (1 + data["omega_z"]) * data["g_tt_rrr"]
            - data["omega_rrrt"] * data["g_tz"]
            - 3 * data["omega_rrt"] * data["g_tz_r"]
            - 3 * data["omega_rt"] * data["g_tz_rr"]
            - data["omega_t"] * data["g_tz_rrr"]
            - 3 * gamma_rr * data["sqrt(g)_r"]
            - 3 * gamma_r * data["sqrt(g)_rr"]
            - gamma * data["sqrt(g)_rrr"],
            data["sqrt(g)"],
        ),
        # TODO(#587): axis limit
        #   Computed with four applications of l’Hôpital’s rule.
        #   Requires sqrt(g)_rrrr and fourth derivatives of basis vectors.
        jnp.nan,
    )
    gamma_rrr = surface_integrals(transforms["grid"], gamma_rrr)
    # force limit to nan until completed because integration replaces nan with 0
    data["iota_den_rrr"] = transforms["grid"].replace_at_axis(gamma_rrr, jnp.nan)
    return data


@register_compute_fun(
    name="iota_psi",
    label="\\partial_{\\psi} \\iota",
    units="Wb^{-1}",
    units_long="Inverse Webers",
    description="Rotational transform, radial derivative wrt toroidal flux",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["iota_r", "psi_r"],
    axis_limit_data=["iota_rr", "psi_rr"],
)
def _iota_psi(params, transforms, profiles, data, **kwargs):
    """∂ι/∂ψ.

    Existence of limit at magnetic axis requires ∂ι/∂ρ = 0 at axis.
    Assume ι may be expanded as an even power series of ρ so that this
    condition is satisfied.
    """
    data["iota_psi"] = transforms["grid"].replace_at_axis(
        safediv(data["iota_r"], data["psi_r"]),
        lambda: safediv(data["iota_rr"], data["psi_rr"]),
    )
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


# TODO (#1381): add K(rho,theta,zeta)*grad(rho) term
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
    resolution_requirement="tz",
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
    resolution_requirement="tz",
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
    resolution_requirement="tz",
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
    resolution_requirement="tz",
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
    resolution_requirement="tz",
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
    resolution_requirement="tz",
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


@register_compute_fun(
    name="shear",
    label="-\\rho \\frac{\\partial_{\\rho}\\iota}{\\iota}",
    units="~",
    units_long="None",
    description="Global magnetic shear",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["0", "iota", "iota_r", "rho"],
)
def _shear(params, transforms, profiles, data, **kwargs):
    """Global magnetic shear, as defined in the tokamak literature: -dι/dρ * (ρ/ι).

    When ι=0 ∀ρ (such as in a vacuum tokamak) the shear is defined to be 0 everywhere.
    This implementation is undefined whenever ι=0 otherwise, which is the correct value
    in the case of a Reverse Field Configuration.
    The case where both ι=0 and dι/dρ=0 is rare, and that limit is not implemented.
    """
    eps = 1e2 * jnp.finfo(data["iota"].dtype).eps
    data["shear"] = cond(
        jnp.all(jnp.abs(data["iota"]) < eps),
        lambda _: data["0"],  # if iota is 0 everywhere, set shear to 0 everywhere
        lambda _: -data["rho"] * data["iota_r"] / data["iota"],
        None,
    )
    return data


@register_compute_fun(
    name="<sigma*nu>",
    label="\\langle\\sigma\\nu\\rangle",
    units="m^3 \\cdot s^{-1}",
    units_long="cubic meters / second",
    description="Thermal reactivity from Bosch-Hale parameterization",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["Ti"],
    fuel="str: Fusion fuel, assuming a 50/50 mix. One of {'DT'}. Default is 'DT'.",
)
def _reactivity(params, transforms, profiles, data, **kwargs):
    # Bosch and Hale. “Improved Formulas for Fusion Cross-Sections and Thermal
    # Reactivities.” Nuclear Fusion 32 (April 1992): 611-631.
    # https://doi.org/10.1088/0029-5515/32/4/I07.
    coefficients = {
        "DT": {
            "B_G": 34.382,
            "mc2": 1124656,
            "C1": 1.17302e-9,
            "C2": 1.51361e-2,
            "C3": 7.51886e-2,
            "C4": 4.60643e-3,
            "C5": 1.35000e-2,
            "C6": -1.06750e-4,
            "C7": 1.36600e-5,
        }
    }
    fuel = kwargs.get("fuel", "DT")
    coeffs = coefficients.get(fuel)

    T = data["Ti"] / 1e3  # keV
    theta = T / (
        1
        - (T * (coeffs["C2"] + T * (coeffs["C4"] + T * coeffs["C6"])))
        / (1 + T * (coeffs["C3"] + T * (coeffs["C5"] + T * coeffs["C7"])))
    )
    xi = (coeffs["B_G"] ** 2 / (4 * theta)) ** (1 / 3)
    sigma_nu = (
        coeffs["C1"] * theta * jnp.sqrt(xi / (coeffs["mc2"] * T**3)) * jnp.exp(-3 * xi)
    )  # cm^3/s
    data["<sigma*nu>"] = sigma_nu / 1e6  # m^3/s
    return data
