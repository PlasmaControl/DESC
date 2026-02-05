import jax

from desc.backend import jnp
from desc.compute._isothermal import (
    first_derivative_t,
    first_derivative_t2,
    first_derivative_z,
    first_derivative_z2,
)

# TODO: review when zeta no longer equals phi
from desc.derivatives import Derivative
from .data_index import register_compute_fun

def sqr_int(f,data,grid):
    
    integrand = grid.spacing[:, 1] * grid.spacing[:, 2] * f
    desired_rho_surface = 1.0
    indices = jnp.where(grid.nodes[:, 0] == desired_rho_surface)[0]
    integrand = integrand[indices]
    
    return integrand.sum()
    
@register_compute_fun(
    name="e_rho",
    label="\\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
)
def _e_rho_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_rho"] = coords
    return data


@register_compute_fun(
    name="e_rho_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
)
def _e_rho_r_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_rho_r"] = coords
    return data


@register_compute_fun(
    name="e_rho_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector,"
    " second derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
)
def _e_rho_rr_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_rho_rr"] = coords
    return data


@register_compute_fun(
    name="e_rho_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, derivative wrt poloidal angle",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    aliases=["e_theta_r"],
)
def _e_rho_t_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_rho_t"] = coords
    return data


@register_compute_fun(
    name="e_rho_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.geometry.surface.FourierRZToroidalSurface",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    aliases=["e_zeta_r"],
)
def _e_rho_z_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_rho_z"] = coords
    return data


@register_compute_fun(
    name="e_theta_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector,"
    " second derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    aliases=["e_rho_rt"],
)
def _e_theta_rr_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_theta_rr"] = coords
    return data


@register_compute_fun(
    name="e_zeta_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector,"
    " second derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.geometry.surface.FourierRZToroidalSurface",
        "desc.geometry.surface.ZernikeRZToroidalSection",
    ],
    aliases=["e_rho_rz"],
)
def _e_zeta_rr_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_zeta_rr"] = coords
    return data


@register_compute_fun(
    name="Phi",
    label="\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential",
    dim=1,
    params=["I", "G", "Phi_mn"],
    transforms={"Phi": [[0, 0, 0]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _Phi_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi"] = (
        transforms["Phi"].transform(params["Phi_mn"])
        + params["G"] * transforms["Phi"].nodes[:, 2].flatten(order="F") / 2 / jnp.pi
        + params["I"] * transforms["Phi"].nodes[:, 1].flatten(order="F") / 2 / jnp.pi
    )
    return data


@register_compute_fun(
    name="Phi_t",
    label="\\partial_{\\theta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, poloidal derivative",
    dim=1,
    params=["I", "Phi_mn"],
    transforms={"Phi": [[0, 1, 0]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _Phi_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_t"] = (
        transforms["Phi"].transform(params["Phi_mn"], dt=1) + params["I"] / 2 / jnp.pi
    )
    return data


@register_compute_fun(
    name="Phi_z",
    label="\\partial_{\\zeta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, toroidal derivative",
    dim=1,
    params=["G", "Phi_mn"],
    transforms={"Phi": [[0, 0, 1]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _Phi_z_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_z"] = (
        transforms["Phi"].transform(params["Phi_mn"], dz=1) + params["G"] / 2 / jnp.pi
    )
    return data


@register_compute_fun(
    name="Phi",
    label="\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential",
    dim=1,
    params=[],
    transforms={"grid": [], "potential": [], "params": []},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.magnetic_fields._current_potential.CurrentPotentialField",
)
def _Phi_CurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi"] = transforms["potential"](
        transforms["grid"].nodes[:, 1],
        transforms["grid"].nodes[:, 2],
        **transforms["params"]
    )
    return data


@register_compute_fun(
    name="Laplace_Beltrami(Phi)",
    label="LB(\\Phi)",
    units="A/m2",
    units_long="Amperes per square meter",
    description="Laplace Beltrami Operator on Surface current potential",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=[
        "Phi_t",
        "Phi_z",
        "Phi_tt",
        "Phi_tz",
        "Phi_zz",
        "e_theta",
        "e^zeta_s",
        "e^theta_s_t",
        "e^theta_s_z",
        "e^zeta_s_t",
        "e^zeta_s_z",
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _Laplace_Beltrami_Phi_FourierCurrentPotentialField(
    params, transforms, profiles, data, **kwargs
):
    data["Laplace_Beltrami(Phi)"] = (
        jnp.sum(data["e^theta_s"] * data["e^theta_s_t"], axis=-1) * data["Phi_t"]
        + jnp.sum(data["e^theta_s"] * data["e^theta_s"], axis=-1) * data["Phi_tt"]
        + jnp.sum(data["e^theta_s"] * data["e^zeta_s_t"], axis=-1) * data["Phi_z"]
        + jnp.sum(data["e^theta_s"] * data["e^zeta_s"], axis=-1) * data["Phi_tz"]
        + jnp.sum(data["e^zeta_s"] * data["e^theta_s_z"], axis=-1) * data["Phi_t"]
        + jnp.sum(data["e^theta_s"] * data["e^zeta_s"], axis=-1) * data["Phi_tz"]
        + jnp.sum(data["e^zeta_s"] * data["e^zeta_s_z"], axis=-1) * data["Phi_z"]
        + jnp.sum(data["e^zeta_s"] * data["e^zeta_s"], axis=-1) * data["Phi_zz"]
    )
    return data


@register_compute_fun(
    name="Phi_t",
    label="\\partial_{\\theta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, poloidal derivative",
    dim=1,
    params=[],
    transforms={"grid": [], "potential_dtheta": [], "params": []},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.magnetic_fields._current_potential.CurrentPotentialField",
)
def _Phi_t_CurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_t"] = transforms["potential_dtheta"](
        transforms["grid"].nodes[:, 1],
        transforms["grid"].nodes[:, 2],
        **transforms["params"]
    )
    return data


@register_compute_fun(
    name="Phi_z",
    label="\\partial_{\\zeta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, toroidal derivative",
    dim=1,
    params=[],
    transforms={"grid": [], "potential_dzeta": [], "params": []},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.magnetic_fields._current_potential.CurrentPotentialField",
)
def _Phi_z_CurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_z"] = transforms["potential_dzeta"](
        transforms["grid"].nodes[:, 1],
        transforms["grid"].nodes[:, 2],
        **transforms["params"]
    )
    return data


@register_compute_fun(
    name="K^theta",
    label="K^{\\theta}",
    units="A/m^2",
    units_long="Amperes per square meter",
    description="Contravariant poloidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["Phi_z", "|e_theta x e_zeta|"],
    parameterization=[
        "desc.magnetic_fields._current_potential.CurrentPotentialField",
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_theta_CurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K^theta"] = -data["Phi_z"] * (1 / data["|e_theta x e_zeta|"])
    return data


@register_compute_fun(
    name="K^zeta",
    label="K^{\\zeta}",
    units="A/m^2",
    units_long="Amperes per square meter",
    description="Contravariant toroidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["Phi_t", "|e_theta x e_zeta|"],
    parameterization=[
        "desc.magnetic_fields._current_potential.CurrentPotentialField",
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_zeta_CurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K^zeta"] = data["Phi_t"] * (1 / data["|e_theta x e_zeta|"])
    return data


@register_compute_fun(
    name="K",
    label="\\mathbf{K}",
    units="A/m",
    units_long="Amperes per meter",
    description="Surface current density, defined as the"
    "surface normal vector cross the gradient of the current potential.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["K^theta", "K^zeta", "e_theta", "e_zeta"],
    parameterization=[
        "desc.magnetic_fields._current_potential.CurrentPotentialField",
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_CurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K"] = (data["K^zeta"] * data["e_zeta"].T).T + (
        data["K^theta"] * data["e_theta"].T
    ).T
    return data


@register_compute_fun(
    name="grad_s(Phi)",
    label="\\nabla_s \\Phi",
    units="A/m",
    units_long="Amperes per meter",
    description="Surface gradient of the current potential.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=[
        "Phi_t",
        "Phi_z",
        "e^theta_s",
        "e^zeta_s",
    ],  # , "Phi_z", "e^theta_s", "e^zeta_s"],
    parameterization=[
        # "desc.magnetic_fields._current_potential.CurrentPotentialField",
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _grad_s_Phi_FourierCurrentPotentialField(
    params, transforms, profiles, data, **kwargs
):
    data["grad_s(Phi)"] = (
        data["Phi_t"] * data["e^theta_s"].T + data["Phi_z"] * data["e^zeta_s"].T
    ).T
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[0, 0, 0]],
        "Z": [[0, 0, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
)
def _x_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"])
    Z = transforms["Z"].transform(params["Z_lmn"])
    phi = transforms["grid"].nodes[:, 2]
    coords = jnp.stack([R, phi, Z], axis=1)
    data["x"] = coords
    return data


@register_compute_fun(
    name="e_zeta",
    label="\\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
)
def _e_zeta_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_zeta"] = coords
    return data


@register_compute_fun(
    name="e_theta_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    aliases=["e_zeta_t"],
)
def _e_theta_z_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_theta_z"] = coords
    return data


@register_compute_fun(
    name="e_zeta_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
)
def _e_zeta_z_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_zeta_z"] = coords
    return data


##########################################################################################################
# More derivatives of surface current
##########################################################################################################
@register_compute_fun(
    name="Phi_tt",
    label="\\partial_{\\theta \\theta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, second poloidal derivative",
    dim=1,
    params=["Phi_mn"],
    transforms={"Phi": [[0, 2, 0]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _Phi_tt_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_tt"] = transforms["Phi"].transform(
        params["Phi_mn"],
        dt=2,
    )
    return data


@register_compute_fun(
    name="Phi_tz",
    label="\\partial_{\\theta \\zeta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, poloidal-toroidal derivative",
    dim=1,
    params=["Phi_mn"],
    transforms={"Phi": [[0, 1, 1]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _Phi_tz_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_tz"] = transforms["Phi"].transform(
        params["Phi_mn"],
        dt=1,
        dz=1,
    )
    return data


@register_compute_fun(
    name="Phi_zz",
    label="\\partial_{\\zeta \\zeta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, second toroidal derivative",
    dim=1,
    params=["Phi_mn"],
    transforms={"Phi": [[0, 0, 2]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _Phi_zz_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_zz"] = transforms["Phi"].transform(
        params["Phi_mn"],
        dz=2,
    )
    return data


@register_compute_fun(
    name="Phi_ttt",
    label="\\partial_{\\theta \\theta \\theta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, second poloidal derivative",
    dim=1,
    params=["Phi_mn"],
    transforms={"Phi": [[0, 3, 0]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _Phi_ttt_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_ttt"] = transforms["Phi"].transform(
        params["Phi_mn"],
        dt=3,
    )
    return data


@register_compute_fun(
    name="Phi_ttz",
    label="\\partial_{\\theta \\theta \\zeta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, second poloidal - toroidal derivative",
    dim=1,
    params=["Phi_mn"],
    transforms={"Phi": [[0, 2, 1]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _Phi_tz_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_ttz"] = transforms["Phi"].transform(
        params["Phi_mn"],
        dt=2,
        dz=1,
    )
    return data


@register_compute_fun(
    name="Phi_tzz",
    label="\\partial_{\\theta \\zeta \\zeta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, poloidal - second toroidal derivative",
    dim=1,
    params=["Phi_mn"],
    transforms={"Phi": [[0, 1, 2]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _Phi_tz_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_tzz"] = transforms["Phi"].transform(
        params["Phi_mn"],
        dt=1,
        dz=2,
    )
    return data


@register_compute_fun(
    name="Phi_zzz",
    label="\\partial_{\\zeta \\zeta \\zeta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, third toroidal derivative",
    dim=1,
    params=["Phi_mn"],
    transforms={"Phi": [[0, 0, 3]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _Phi_zzz_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_zz"] = transforms["Phi"].transform(
        params["Phi_mn"],
        dz=3,
    )
    return data


@register_compute_fun(
    name="K^theta_t",
    label="K^{\\theta}_t",
    units="A/m^3",
    units_long="Amperes per cubic meter",
    description="Poloidal derivative of Contravariant poloidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=[
        "Phi_z",
        "|e_theta x e_zeta|",
        "Phi_tz",
        "|e_theta x e_zeta|_t",
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_theta_t_FourierCurrentPotentialField(
    params, transforms, profiles, data, **kwargs
):
    data["K^theta_t"] = -(
        data["Phi_tz"] * (1 / data["|e_theta x e_zeta|"])
        + data["Phi_z"]
        * (-data["|e_theta x e_zeta|_t"] / data["|e_theta x e_zeta|"] ** 2)
    )
    return data


@register_compute_fun(
    name="K^theta_z",
    label="K^{\\theta}_z",
    units="A/m^3",
    units_long="Amperes per cubic meter",
    description="Toroidal derivative of Contravariant poloidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["Phi_z", "|e_theta x e_zeta|", "Phi_zz", "|e_theta x e_zeta|_z"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_theta_z_FourierCurrentPotentialField(
    params, transforms, profiles, data, **kwargs
):
    data["K^theta_z"] = -(
        data["Phi_zz"] * (1 / data["|e_theta x e_zeta|"])
        + data["Phi_z"]
        * (-data["|e_theta x e_zeta|_z"] / data["|e_theta x e_zeta|"] ** 2)
    )
    return data


@register_compute_fun(
    name="K^zeta_t",
    label="K^{\\zeta}_t",
    units="A/m^3",
    units_long="Amperes per cubic meter",
    description="Contravariant toroidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["Phi_t", "|e_theta x e_zeta|", "Phi_tt", "|e_theta x e_zeta|_t"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_zeta_t_FourierCurrentPotentialField(
    params, transforms, profiles, data, **kwargs
):
    data["K^zeta_t"] = data["Phi_tt"] * (1 / data["|e_theta x e_zeta|"]) + data[
        "Phi_t"
    ] * (-data["|e_theta x e_zeta|_t"] / data["|e_theta x e_zeta|"] ** 2)
    return data


@register_compute_fun(
    name="K^zeta_z",
    label="K^{\\zeta}_z",
    units="A/m^3",
    units_long="Amperes per cubic meter",
    description="Contravariant toroidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["Phi_t", "|e_theta x e_zeta|", "Phi_tz", "|e_theta x e_zeta|_z"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_zeta_z_FourierCurrentPotentialField(
    params, transforms, profiles, data, **kwargs
):
    data["K^zeta_z"] = data["Phi_tz"] * (1 / data["|e_theta x e_zeta|"]) + data[
        "Phi_t"
    ] * (-data["|e_theta x e_zeta|_z"] / data["|e_theta x e_zeta|"] ** 2)
    return data


@register_compute_fun(
    name="K_t",
    label="K_z",
    units="A/m^3",
    units_long="Amperes per cubic meter",
    description="Contravariant toroidal component of surface current density",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=[
        "e_theta",
        "e_zeta",
        "K^theta",
        "K^zeta",
        "e_theta_t",
        "e_zeta_t",
        "K^theta_t",
        "K^zeta_t",
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K_t"] = (
        data["K^theta_t"] * data["e_theta"].T
        + data["K^theta"] * data["e_theta_t"].T
        + data["K^zeta_t"] * data["e_zeta"].T
        + data["K^zeta"] * data["e_zeta_t"].T
    ).T
    return data


@register_compute_fun(
    name="K_z",
    label="K_z",
    units="A/m^3",
    units_long="Amperes per cubic meter",
    description="Contravariant toroidal component of surface current density",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=[
        "e_theta",
        "e_zeta",
        "K^theta",
        "K^zeta",
        "e_theta_z",
        "e_zeta_z",
        "K^theta_z",
        "K^zeta_z",
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_z_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K_z"] = (
        data["K^theta_z"] * data["e_theta"].T
        + data["K^theta"] * data["e_theta_z"].T
        + data["K^zeta_z"] * data["e_zeta"].T
        + data["K^zeta"] * data["e_zeta_z"].T
    ).T
    return data


@register_compute_fun(
    name="y_s",
    label="y_s",
    units="A/m^2",
    units_long="Amperes per square meter",
    description="Contravariant poloidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=[  # "K^theta",
        "K",
        "e_theta",
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _y_s_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["y_s"] = jnp.sum(data["K"] * data["e_theta"], axis=-1)
    # data["y_s"] = data["K^theta"]
    return data


@register_compute_fun(
    name="y_s_t",
    label="y_s_t",
    units="A/m^2",
    units_long="Amperes per square meter",
    description="Poloidal derivative of Contravariant poloidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["K_t", "e_theta", "e_theta_t", "K^theta_t"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _y_s_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["y_s_t"] = jnp.sum(data["K_t"] * data["e_theta"], axis=-1) + jnp.sum(
        data["K"] * data["e_theta_t"], axis=-1
    )
    return data


@register_compute_fun(
    name="y_s_z",
    label="y_s_z",
    units="A/m^2",
    units_long="Amperes per square meter",
    description="Toroidal derivative of Contravariant poloidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=[
        "K",
        "K_z",
        "e_theta",
        "e_theta_z",
        # "K^theta_z"
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _y_s_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["y_s_z"] = jnp.sum(data["K_z"] * data["e_theta"], axis=-1) + jnp.sum(
        data["K"] * data["e_theta_z"], axis=-1
    )

    # data["y_s_z"] = data["K^theta_z"]
    return data


@register_compute_fun(
    name="x_s",
    label="x_s",
    units="A/m^2",
    units_long="Amperes per square meter",
    description="Contravariant toroidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=[  # "K^zeta",
        "K",
        "e_zeta",
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _x_s_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["x_s"] = jnp.sum(data["K"] * data["e_zeta"], axis=-1)
    # data["x_s"] = data["K^zeta"]
    return data


@register_compute_fun(
    name="x_s_t",
    label="x_s_t",
    units="A/m^2",
    units_long="Amperes per square meter",
    description="Poloidal derivative of Contravariant toroidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=[
        "K",
        "K_t",
        "e_zeta",
        "e_zeta_t",
        # "K^zeta_t"
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _x_s_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["x_s_t"] = jnp.sum(data["K_t"] * data["e_zeta"], axis=-1) + jnp.sum(
        data["K"] * data["e_zeta_t"], axis=-1
    )

    # data["x_s_t"] = data["K^zeta_t"]
    return data


@register_compute_fun(
    name="x_s_z",
    label="x_s_z",
    units="A/m^2",
    units_long="Amperes per square meter",
    description="Toroidal derivative of Contravariant toroidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=[  # "K^zeta_z",
        "K",
        "K_z",
        "e_zeta",
        "e_zeta_z",
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _x_s_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    # data["x_s_z"] = data["K^zeta_z"]

    data["x_s_z"] = jnp.sum(data["K_z"] * data["e_zeta"], axis=-1) + jnp.sum(
        data["K"] * data["e_zeta_z"], axis=-1
    )

    return data


@register_compute_fun(
    name="z_s",
    label="z_s",
    units="A/m^2",
    units_long="Amperes per square meter",
    description="~",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["x_s_t", "y_s_z"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _z_s_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["z_s"] = data["x_s_t"] - data["y_s_z"]
    return data


################################################################################################################
# Functions to find variable sigma
################################################################################################################

# Thickness through Fourier modes
@register_compute_fun(
    name="bf",
    label="b",
    units="~",
    units_long="~",
    description="Surface current potential",
    dim=1,
    params=["b_I", "b_G", "b_mn"],
    transforms={"bf": [[0, 0, 0]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _bf_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["bf"] = (
        transforms["bf"].transform(params["b_mn"])
        + params["b_G"]
        * transforms["bf"].nodes[:, 2].flatten(order="F")  # / 2 / jnp.pi
        + params["b_I"]
        * transforms["bf"].nodes[:, 1].flatten(order="F")  # / 2 / jnp.pi
    )
    return data


@register_compute_fun(
    name="bf_t",
    label="\\partial_{\\theta}\\ bf",
    units="~",
    units_long="~",
    description="Derivative of non-dimensional thickness",
    dim=1,
    params=["b_I", "b_mn"],
    transforms={"bf": [[0, 1, 0]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _bf_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["bf_t"] = (
        transforms["bf"].transform(params["b_mn"], dt=1) + params["b_I"]  # / 2 / jnp.pi
    )
    return data


@register_compute_fun(
    name="bf_z",
    label="\\partial_{\\zeta} bf",
    units="~",
    units_long="~",
    description="Surface current potential, toroidal derivative",
    dim=1,
    params=["b_G", "b_mn"],
    transforms={"bf": [[0, 0, 1]]},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
    ],
)
def _bf_z_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["bf_z"] = (
        transforms["bf"].transform(params["b_mn"], dz=1) + params["b_G"]  # / 2 / jnp.pi
    )
    return data


##############################################################################################
# Finite differences
@register_compute_fun(
    name="b_t",
    label="b_t",
    units="~",
    units_long="~",
    description="Poloidal derivative of Log of electrical conductivity",
    dim=1,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=["theta", "zeta", "b_s"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _b_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):

    data["b_t"] = first_derivative_t(
        data["b_s"],
        data,
        2 * (transforms["grid"].M) + 1,
        2 * (transforms["grid"].N) + 1,
    )

    return data


@register_compute_fun(
    name="b_z",
    label="b_z",
    units="~",
    units_long="~",
    description="Toroidal derivative of Log of electrical conductivity",
    dim=1,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=["theta", "zeta", "b_s"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _b_z_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):

    data["b_z"] = first_derivative_z(
        data["b_s"],
        data,
        2 * (transforms["grid"].M) + 1,
        2 * (transforms["grid"].N) + 1,
    )
    return data


@register_compute_fun(
    name="sigma",
    label="\sigma",
    units="S",
    units_long="Siemens",
    description="Variable conductivity.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["b_s"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _sigma_CurrentPotentialField(params, transforms, profiles, data, **kwargs):

    data["sigma"] = jnp.exp(data["b_s"])
    return data

    
@register_compute_fun(
    name="b_s",
    label="b_s",
    units="~",
    units_long="~",
    description="Log of electrical conductivity",
    dim=1,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[
        "theta",
        "zeta",
        "x_s",
        "y_s",
        "z_s",
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _b_s_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):

    data["b_s"] = find_b(
        data, 2 * (transforms["grid"].M) + 1, 2 * (transforms["grid"].N) + 1
    )

    return data


##############################################################################################
# Lele Finite differences
@register_compute_fun(
    name="bl_t",
    label="bl_t",
    units="~",
    units_long="~",
    description="Poloidal derivative of Log of electrical conductivity",
    dim=1,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=["theta", "zeta", "b_s"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _bl_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["bl_t"] = (
        lele_matrix(
            data, 2 * (transforms["grid"].M) + 1, 2 * (transforms["grid"].N) + 1
        )
        @ data["b_s"]
    )

    return data


@register_compute_fun(
    name="bl_z",
    label="bl_z",
    units="~",
    units_long="~",
    description="Toroidal derivative of Log of electrical conductivity",
    dim=1,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=["theta", "zeta", "b_s"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _bl_z_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):

    lele_z = (
        (data["theta"][1] - data["theta"][0])
        / (data["zeta"][2 * (transforms["grid"].M) + 1] - data["zeta"][0])
    ) * lele_matrix(
        data, 2 * (transforms["grid"].M) + 1, 2 * (transforms["grid"].N) + 1
    )

    data["bl_z"] = (
        (
            lele_z
            @ (
                data["b_s"].reshape(
                    (2 * (transforms["grid"].N) + 1, 2 * (transforms["grid"].M) + 1)
                )
            ).T.flatten()
        ).reshape(
            (
                2 * (transforms["grid"].N) + 1,
                2 * (transforms["grid"].M) + 1,
            )
        )
    ).T.flatten()

    return data


@register_compute_fun(
    name="bl_s",
    label="bl_s",
    units="~",
    units_long="~",
    description="Log of electrical conductivity",
    dim=1,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[
        "theta",
        "zeta",
        "x_s",
        "y_s",
        "z_s",
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _bl_s_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):

    data["bl_s"] = find_b_lele(
        data, 2 * (transforms["grid"].M) + 1, 2 * (transforms["grid"].N) + 1
    )

    return data


###
@register_compute_fun(
    name="K_eng",
    label="\\mathbf{K}_{eng}",
    units="A/m",
    units_long="Amperes per meter",
    description="Engineering current with variable conductivity.",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=[
        "e_theta",
        "e_zeta",
        "b_s",
        "V_t",
        "V_z",
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _Keng_CurrentPotentialField(params, transforms, profiles, data, **kwargs):

    den = (
        jnp.sum(data["e_theta"] * data["e_theta"], axis=-1)
        * jnp.sum(data["e_zeta"] * data["e_zeta"], axis=-1)
        - jnp.sum(data["e_theta"] * data["e_zeta"], axis=-1) ** 2
    )

    data["K_eng"] = (
        (jnp.exp(data["b_s"]) / den)
        * (
            (
                jnp.sum(data["e_zeta"] * data["e_zeta"], axis=-1) * data["V_t"]
                - jnp.sum(data["e_theta"] * data["e_zeta"], axis=-1) * data["V_z"]
            )
            * data["e_theta"].T
            + (
                -jnp.sum(data["e_theta"] * data["e_zeta"], axis=-1) * data["V_t"]
                + jnp.sum(data["e_theta"] * data["e_theta"], axis=-1) * data["V_z"]
            )
            * data["e_zeta"].T
        )
    ).T
    return data

@register_compute_fun(
    name="I_V",
    label="I_V",
    units="~",
    units_long="~",
    description="Poloidal current from voltage",
    dim=1,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=['V_t',#"theta", "zeta",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _I_V_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    
    data["I_V"] = sqr_int(data['V_t'],data, transforms["grid"]) * (2 * jnp.pi) ** (-2)

    return data

@register_compute_fun(
    name="G_V",
    label="G_V",
    units="~",
    units_long="~",
    description="Toroidal current from voltage",
    dim=1,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=['V_z',#"theta", "zeta",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _G_V_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    
    data["G_V"] = - sqr_int(data['V_z'],data, transforms["grid"]) * (2 * jnp.pi) ** (-2)

    return data

@register_compute_fun(
    name="V_t",
    label="V_t",
    units="~",
    units_long="~",
    description="Poloidal derivative of Log of electrical conductivity",
    dim=1,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=["theta", "zeta",
          'sigma', 'y_s',
          # "V_s",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _V_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    
    data["V_t"] = data['sigma'] ** (-1) * data['y_s'] 
    
    #data["V_t"] = first_derivative_t2(
    #    data["V_s"],
    #    data,
    #    2 * (transforms["grid"].M) + 1,
    #    2 * (transforms["grid"].N) + 1,
    #)

    return data


@register_compute_fun(
    name="V_z",
    label="V_z",
    units="~",
    units_long="~",
    description="Toroidal derivatiev of Log of electrical conductivity",
    dim=1,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=["theta", "zeta",
          'sigma', "x_s",
          #'V_s',
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _V_z_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):

    data["V_z"] = data['sigma'] ** (-1) * data['x_s'] 
    
    #data["V_z"] = first_derivative_z2(
    #    data["V_s"],
    #    data,
    #    2 * (transforms["grid"].M) + 1,
    #    2 * (transforms["grid"].N) + 1,
    #)

    return data


###


@register_compute_fun(
    name="V_s",
    label="V_s",
    units="V",
    units_long="~",
    description="Voltage distribution",
    dim=1,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[
        "theta",
        "zeta",
        "x_s",
        "y_s",
        "b_s",
    ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _V_s_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    # b = find_b(data,2*(transforms["grid"].M)+1, 2*(transforms["grid"].N)+1)

    data["V_s"] = find_V(
        data, 2 * (transforms["grid"].M) + 1, 2 * (transforms["grid"].N) + 1
    )

    return data


# Invert the matrix and find b
def find_V(data, m_size, n_size):

    x = jnp.ones(data["theta"].shape[0])
    fun_wrapped = lambda x: V_residual(x, data, m_size, n_size)
    # A_ = Derivative(fun_wrapped,deriv_mode="looped").compute(x)
    # A_ = jax.jacfwd(fun_wrapped)(x)

    return jnp.linalg.pinv(
        Derivative(fun_wrapped, deriv_mode="looped").compute(x)
    ) @ jnp.concatenate(
        (jnp.exp(-data["b_s"]) * data["y_s"], jnp.exp(-data["b_s"]) * data["x_s"])
    )


# Invert the matrix and find b
def find_b(data, m_size, n_size):

    x = jnp.ones(data["theta"].shape[0])
    fun_wrapped = lambda x: b_residual(x, data, m_size, n_size)
    A_ = Derivative(fun_wrapped, deriv_mode="looped").compute(x)
    # A_ = jax.jacfwd(fun_wrapped)(x)

    return jnp.linalg.pinv(A_) @ data["z_s"]


# Invert the matrix and find b
def find_b_lele(data, m_size, n_size):

    x = jnp.ones(data["theta"].shape[0])
    fun_wrapped = lambda x: b_residual_lele(x, data, m_size, n_size)
    A_ = Derivative(fun_wrapped, deriv_mode="looped").compute(x)
    # A_ = jax.jacfwd(fun_wrapped)(x)

    return jnp.linalg.pinv(A_) @ data["z_s"]


# Function to find build a matrix to find the scalar b
def V_residual(y, data, m_size, n_size):

    # f_t = first_derivative_t(y, data,m_size,n_size)
    # f_z = first_derivative_z(y, data,m_size,n_size)

    return jnp.concatenate(
        (
            first_derivative_t2(y, data, m_size, n_size),
            first_derivative_z2(y, data, m_size, n_size),
        )
    )


# Function to find build a matrix to find the scalar b
def b_residual(y, data, m_size, n_size):

    f_t = first_derivative_t(y, data, m_size, n_size)
    f_z = first_derivative_z(y, data, m_size, n_size)

    return data["x_s"] * f_t - data["y_s"] * f_z


##### Lele matrix for first derivatives
def lele_matrix(data, m_size, n_size):

    # Create LHS matrix A
    A = jnp.zeros((m_size, m_size)).T
    # Diagonal terms
    A = A.at[jnp.arange(m_size), jnp.arange(m_size)].set(jnp.ones(m_size))

    # Boundary treatment
    alpha = 1 / 3
    A = A.at[0, m_size - 1].set(alpha)
    A = A.at[m_size - 1, 0].set(alpha)

    # Off diagonal terms
    interior = jnp.arange(1, m_size)
    A = A.at[interior, interior - 1].set(jnp.ones(m_size - 1) * (alpha))
    A = A.at[interior - 1, interior].set(jnp.ones(m_size - 1) * (alpha))

    Ainv = jnp.linalg.pinv(A)

    # For loop to concatenate block matrices along the diagonal
    B = jnp.block(jnp.block([Ainv, jnp.zeros((m_size, (n_size - 1) * m_size))]))

    for i in range(1, n_size + 0):
        F = jnp.block(
            [
                jnp.zeros((m_size, (i - 0) * m_size)),
                Ainv,
                jnp.zeros((m_size, (n_size - i - 1) * m_size)),
            ]
        )
        B = jnp.block([B.T, F.T]).T

    # Build D matrix
    h = data["theta"][1] - data["theta"][0]
    # h = data["zeta"][m_size] - data["zeta"][0]
    a = 14 / 9
    b = 1 / 9

    # Create C matrix
    C = jnp.zeros((m_size, m_size)).T

    # Boundary treatment
    C = C.at[0, m_size - 1].set(-a / (2 * h))
    C = C.at[0, m_size - 2].set(-b / (4 * h))
    C = C.at[m_size - 1, 0].set(a / (2 * h))
    C = C.at[m_size - 1, 1].set(b / (4 * h))

    C = C.at[1, m_size - 1].set(-b / (4 * h))
    C = C.at[m_size - 2, 0].set(b / (4 * h))

    # Off diagonal terms
    interior = jnp.arange(0, m_size - 1)
    C = C.at[interior, interior + 1].set(jnp.ones(m_size - 1) * a / (2 * h))
    C = C.at[interior, interior + 2].set(jnp.ones(m_size - 1) * b / (4 * h))
    C = C.at[interior + 1, interior].set(jnp.ones(m_size - 1) * -a / (2 * h))
    C = C.at[interior + 2, interior].set(jnp.ones(m_size - 1) * -b / (4 * h))

    # For loop to concatenate block matrices along the diagonal
    D = jnp.block(jnp.block([C, jnp.zeros((m_size, (n_size - 1) * m_size))]))

    for i in range(1, n_size + 0):
        E = jnp.block(
            [
                jnp.zeros((m_size, (i - 0) * m_size)),
                C,
                jnp.zeros((m_size, (n_size - i - 1) * m_size)),
            ]
        )
        D = jnp.block([D.T, E.T]).T

    return B @ D


# Invert the matrix and find b
def find_b_lele(data, m_size, n_size):

    x = jnp.ones(data["theta"].shape[0])
    fun_wrapped = lambda x: b_residual_lele(x, data, m_size, n_size)
    A_ = Derivative(fun_wrapped, deriv_mode="looped").compute(x)
    # A_ = jax.jacfwd(fun_wrapped)(x)

    return jnp.linalg.pinv(A_) @ data["z_s"]


# Function to find build a matrix to find the scalar b
def b_residual_lele(y, data, m_size, n_size):

    lele_t = lele_matrix(data, m_size, n_size)
    lele_z = (
        (data["theta"][1] - data["theta"][0])
        / (data["zeta"][m_size] - data["zeta"][0])
        * lele_t
    )
    f_t = lele_t @ y
    f_z = (
        (lele_z @ (y.reshape((n_size, m_size))).T.flatten()).reshape((n_size, m_size))
    ).T.flatten()

    return data["x_s"] * f_t - data["y_s"] * f_z
