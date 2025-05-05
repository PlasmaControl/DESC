from desc.backend import jnp

from .data_index import register_compute_fun

# TODO(#568): review when zeta no longer equals phi


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
