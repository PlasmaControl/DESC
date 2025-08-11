import jax
from desc.backend import jnp

from .data_index import register_compute_fun
from .geom_utils import rpz2xyz

# TODO: review when zeta no longer equals phi


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
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
)
def _x_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"])
    Z = transforms["Z"].transform(params["Z_lmn"])
    # TODO: change when zeta no longer equals phi
    phi = transforms["grid"].nodes[:, 2]
    coords = jnp.stack([R, phi, Z], axis=1)
    # default basis for "x" is rpz, the conversion will be done
    # in the main _compute function
    data["x"] = coords
    return data


@register_compute_fun(
    name="X",
    label="X",
    units="m",
    units_long="meters",
    description="Cartesian X coordinate.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["x"],
    parameterization="desc.geometry.core.Surface",
)
def _X_Surface(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    coords = rpz2xyz(coords)
    data["X"] = coords[:, 0]
    return data


@register_compute_fun(
    name="Y",
    label="Y",
    units="m",
    units_long="meters",
    description="Cartesian Y coordinate.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["x"],
    parameterization="desc.geometry.core.Surface",
)
def _Y_Surface(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    coords = rpz2xyz(coords)
    data["Y"] = coords[:, 1]
    return data


@register_compute_fun(
    name="R",
    label="R",
    units="m",
    units_long="meters",
    description="Cylindrical radial position along surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["x"],
    parameterization="desc.geometry.core.Surface",
)
def _R_Surface(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    data["R"] = coords[:, 0]
    return data


@register_compute_fun(
    name="phi",
    label="\\phi",
    units="rad",
    units_long="radians",
    description="Toroidal phi position along surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["x"],
    parameterization="desc.geometry.core.Surface",
)
def _phi_Surface(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    data["phi"] = coords[:, 1]
    return data


@register_compute_fun(
    name="Z",
    label="Z",
    units="m",
    units_long="meters",
    description="Cylindrical vertical position along surface",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["x"],
    parameterization="desc.geometry.core.Surface",
)
def _Z_Surface(params, transforms, profiles, data, **kwargs):
    data["Z"] = data["x"][:, 2]
    return data


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
    data["Phi_tt"] = (
        transforms["Phi"].transform(params["Phi_mn"], dt=2,)
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
    data["Phi_tz"] = (
        transforms["Phi"].transform(params["Phi_mn"], dt=1, dz = 1,)
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
    data["Phi_zz"] = (
        transforms["Phi"].transform(params["Phi_mn"], dz = 2,)
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
    data["Phi_ttt"] = (
        transforms["Phi"].transform(params["Phi_mn"], dt=3,)
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
    data["Phi_ttz"] = (
        transforms["Phi"].transform(params["Phi_mn"], dt=2, dz = 1,)
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
    data["Phi_tzz"] = (
        transforms["Phi"].transform(params["Phi_mn"], dt=1, dz = 2,)
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
    data["Phi_zz"] = (
        transforms["Phi"].transform(params["Phi_mn"], dz = 3,)
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
    data=["Phi_z", 
          "|e_theta x e_zeta|",
          "Phi_tz",
          "|e_theta x e_zeta|_t",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_theta_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K^theta_t"] = -(data["Phi_tz"] * (1 / data["|e_theta x e_zeta|"])
                          + data["Phi_z"] * (- data["|e_theta x e_zeta|_t"] / data["|e_theta x e_zeta|"]**2)
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
    data=["Phi_z", "|e_theta x e_zeta|",
          "Phi_zz", "|e_theta x e_zeta|_z"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_theta_z_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K^theta_z"] = -(data["Phi_zz"] * (1 / data["|e_theta x e_zeta|"])
                          + data["Phi_z"] * (- data["|e_theta x e_zeta|_z"] / data["|e_theta x e_zeta|"]**2)
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
    data=["Phi_t", "|e_theta x e_zeta|",
         "Phi_tt", "|e_theta x e_zeta|_t"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_zeta_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K^zeta_t"] = (data["Phi_tt"] * (1 / data["|e_theta x e_zeta|"]) 
                        + data["Phi_t"] * ( - data["|e_theta x e_zeta|_t"] / data["|e_theta x e_zeta|"]**2)
                       )
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
    data=["Phi_t", "|e_theta x e_zeta|",
         "Phi_tz", "|e_theta x e_zeta|_z"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_zeta_z_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K^zeta_z"] = (data["Phi_tz"] * (1 / data["|e_theta x e_zeta|"]) 
                        + data["Phi_t"] * ( - data["|e_theta x e_zeta|_z"] / data["|e_theta x e_zeta|"]**2)
                       )
    return data

@register_compute_fun(
    name="K^theta_tz",
    label="K^{\\theta}_tz",
    units="A/m^3",
    units_long="Amperes per cubic meter",
    description="Poloidal-Toroidal derivative of Contravariant poloidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["Phi_z", 
          "Phi_tz",
          "|e_theta x e_zeta|",
          "|e_theta x e_zeta|_t",
          "|e_theta x e_zeta|_z",
          "|e_theta x e_zeta|_tz",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_theta_tz_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K^theta_tz"] = -(data["Phi_tzz"] * 1 / data["|e_theta x e_zeta|"]
                           - data["Phi_tz"] * (data["|e_theta x e_zeta|_z"] / data["|e_theta x e_zeta|"]**2)
                           - data["Phi_zz"] * ( data["|e_theta x e_zeta|_t"] / data["|e_theta x e_zeta|"]**2) 
                           - data["Phi_z"] * ( data["|e_theta x e_zeta|_tz"] / data["|e_theta x e_zeta|"]**2 
                                              - 2*data["|e_theta x e_zeta|_t"]*data["|e_theta x e_zeta|_z"] / data["|e_theta x e_zeta|"]**3
                                             )
                          )

    return data

@register_compute_fun(
    name="K^theta_zz",
    label="K^{\\theta}_zz",
    units="A/m^3",
    units_long="Amperes per cubic meter",
    description="Second Toroidal derivative of Contravariant poloidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["Phi_z", 
          "Phi_z",
          "|e_theta x e_zeta|",
          "|e_theta x e_zeta|_t",
          "|e_theta x e_zeta|_z",
          "|e_theta x e_zeta|_tz",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_theta_zz_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K^theta_zz"] = -(data["Phi_zzz"] * 1 / data["|e_theta x e_zeta|"]
                           - 2*data["Phi_zz"] * data["|e_theta x e_zeta|_z"] / data["|e_theta x e_zeta|"]**2
                           - data["Phi_z"] * ( data["|e_theta x e_zeta|_zz"] / data["|e_theta x e_zeta|"]**2 
                                              - 2*data["|e_theta x e_zeta|_z"]**2/ data["|e_theta x e_zeta|"]**3
                                             )
                          )
    return data

@register_compute_fun(
    name="K^zeta_tt",
    label="K^{\\zeta}_tt",
    units="A/m^3",
    units_long="Amperes per cubic meter",
    description="Second Poloidal Derivative of Contravariant toroidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["Phi_t", "|e_theta x e_zeta|",
          "Phi_tt", "|e_theta x e_zeta|_t"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_zeta_tt_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K^zeta_tt"] = (data["Phi_ttt"] * 1 / data["|e_theta x e_zeta|"]
                         - 2*data["Phi_tt"] * data["|e_theta x e_zeta|_t"] / data["|e_theta x e_zeta|"]**2
                         - data["Phi_t"] * ( data["|e_theta x e_zeta|_tt"] / data["|e_theta x e_zeta|"]**2 
                                            - 2*data["|e_theta x e_zeta|_t"]**2/ data["|e_theta x e_zeta|"]**3
                                           )
                        )
    
    return data

@register_compute_fun(
    name="K^zeta_tz",
    label="K^{\\zeta}_tt",
    units="A/m^3",
    units_long="Amperes per cubic meter",
    description="Second Poloidal Derivative of Contravariant toroidal component of surface current density",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["Phi_t", "|e_theta x e_zeta|",
         "Phi_tt", "|e_theta x e_zeta|_t"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_sup_zeta_tz_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K^zeta_tz"] = (data["Phi_ttz"] * 1 / data["|e_theta x e_zeta|"]
                         - data["Phi_tt"] * data["|e_theta x e_zeta|_z"] / data["|e_theta x e_zeta|"]**2 
                         - data["Phi_tz"] * data["|e_theta x e_zeta|_t"] / data["|e_theta x e_zeta|"]**2
                         - data["Phi_t"] * ( data["|e_theta x e_zeta|_tz"] / data["|e_theta x e_zeta|"]**2 
                                            - 2*data["|e_theta x e_zeta|_t"]*data["|e_theta x e_zeta|_z"] / data["|e_theta x e_zeta|"]**3
                                           )
                        )
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
    data=["e_theta", "e_zeta","K^theta", "K^zeta",
          "e_theta_t", "e_zeta_t", "K^theta_t", "K^zeta_t"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K_t"] = (data["K^theta_t"]*data["e_theta"].T
                   + data["K^theta"]*data["e_theta_t"].T 
                   + data["K^zeta_t"]*data["e_zeta"].T
                   + data["K^zeta"]*data["e_zeta_t"].T 
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
    data=["e_theta", "e_zeta", "K^theta", "K^zeta",
          "e_theta_z", "e_zeta_z", "K^theta_z", "K^zeta_z"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _K_z_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K_z"] = (data["K^theta_z"]*data["e_theta"].T
                   + data["K^theta"]*data["e_theta_z"].T 
                   + data["K^zeta_z"]*data["e_zeta"].T
                   + data["K^zeta"]*data["e_zeta_z"].T 
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
    data=["K^theta",
          #"K", 
          #"e_theta",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _y_s_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["y_s"] = data["K^theta"]
    #jnp.sum(data["K"]*data["e_theta"], axis=-1) 
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
    data=["K^theta_t",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _y_s_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["y_s_t"] = data["K^theta_t"]
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
    data=["K^theta_z",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _y_s_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["y_s_z"] = data["K^theta_z"]
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
    data=["K^zeta",
          #"K", 
          #"e_theta",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _x_s_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["x_s"] = data["K^zeta"]
    #jnp.sum(data["K"]*data["e_theta"], axis=-1) 
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
    data=["K^zeta_t",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _x_s_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["x_s_t"] = data["K^zeta_t"]
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
    data=["K^zeta_z",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _x_s_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["x_s_z"] = data["K^zeta_z"]
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
    data=["K^theta_z",
          "K^zeta_t",
          #"K", 
          #"e_theta",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _z_s_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["z_s"] = data["K^zeta_t"] - data["K^theta_z"]
    #(jnp.sum(data["K_t"]*data["e_zeta"], axis=-1) 
    # + jnp.sum(data["K"]*data["e_zeta_t"], axis=-1) 
    #- ( jnp.sum(data["K_z"]*data["e_theta"], axis=-1) 
    #   + jnp.sum(data["K"]*data["e_theta_z"], axis=-1)
    #  )
    #)
    return data

@register_compute_fun(
    name="z_s_t",
    label="z_s_t",
    units="A/m^2",
    units_long="Amperes per square meter",
    description="~",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["K^theta_tz",
          "K^zeta_tt",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _z_s_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["z_s_t"] = data["K^zeta_tt"] - data["K^theta_tz"]
    return data

@register_compute_fun(
    name="z_s_z",
    label="z_s_z",
    units="A/m^2",
    units_long="Amperes per square meter",
    description="~",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["K^theta_zz",
          "K^zeta_tz",
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _z_s_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["z_s_z"] = data["K^zeta_tz"] - data["K^theta_zz"]
    return data

################################################################################################################
# Functions to find variable sigma
################################################################################################################
@register_compute_fun(
    name="b_t",
    label="b_t",
    units="~",
    units_long="~",
    description="Poloidal derivatiev of Log of electrical conductivity",
    dim=1,
    params=[],
    transforms={"grid":[],},
    profiles=[],
    coordinates="tz",
    data=["theta","zeta","b_s"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _b_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["b_t"] = first_derivative_t(data["b_s"], data, 2*(transforms["grid"].N)+1)
        
    return data

@register_compute_fun(
    name="b_z",
    label="b_z",
    units="~",
    units_long="~",
    description="Toroidal derivatiev of Log of electrical conductivity",
    dim=1,
    params=[],
    transforms={"grid":[],},
    profiles=[],
    coordinates="tz",
    data=["theta","zeta","b_s"],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _b_t_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["b_z"] = first_derivative_z(data["b_s"], data, 2*(transforms["grid"].N)+1)
        
    return data

@register_compute_fun(
    name="b_s",
    label="b_s",
    units="~",
    units_long="~",
    description="Log of electrical conductivity",
    dim=1,
    params=[],
    transforms={"grid":[],},
    profiles=[],
    coordinates="tz",
    data=["theta","zeta",
          "e_theta", "e_zeta",
          "e_theta_z", "e_zeta_t",
          "K","K_t", "K_z"
         ],
    parameterization=[
        "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
    ],
)
def _b_s_FourierCurrentPotentialField(params, transforms, profiles, data, **kwargs):
    b = find_b(data,2*(transforms["grid"].N)+1)
    
    data["b_s"] = b
        
    return data

# Invert the matrix and find b
def find_b(data,n_size):
    x = jnp.ones(data["theta"].shape[0])
    rhs = (jnp.sum(data["K_t"]*data["e_zeta"], axis=-1) 
           + jnp.sum(data["K"]*data["e_zeta_t"], axis=-1) 
           - ( jnp.sum(data["K_z"]*data["e_theta"], axis=-1) 
              + jnp.sum(data["K"]*data["e_theta_z"], axis=-1)
             )
          )
    
    fun_wrapped = lambda x: b_residual(x,data,n_size)
    A_ = jax.jacfwd(fun_wrapped)(x)
    
    return jnp.linalg.pinv(A_)@rhs

# Function to find build a matrix to find the scalar b
def b_residual(y,data,n_size):
    f_t = first_derivative_t(y, data,n_size)
    f_z = first_derivative_z(y, data,n_size)
    
    return data["K^zeta"]*f_t - data["K^theta"]*f_z

def first_derivative_t(a_mn,data,n_size):
    # Expecting square grids
    
    # Rearrange A as a matrix
    A1 = a_mn.reshape((n_size, n_size)).T
    
    # theta-step
    dt = data["theta"][1] - data["theta"][0]
                                     
    # d(sigma)/dt
    A_t = jnp.zeros_like(A1)
    # i = 0
    A_t = A_t.at[0, :].set( (A1[1, :] - A1[n_size-1, :]) * (2 * dt) ** (-1) )
    # i = n_size
    A_t = A_t.at[n_size-1, :].set( (A1[0, :] - A1[n_size-2, :]) * (2 * dt) ** (-1) )
    # Intermediate steps
    A_t = A_t.at[1:n_size - 1, :].set((A1[2:n_size, :] - A1[0:n_size - 2, :]) * (2 * dt) ** (-1))
    
    return (A_t.T).reshape(-1)#.flatten()

def first_derivative_z(a_mn,data,n_size):
    # Expecting square grids
    
    # Rearrange A as a matrix
    A2 = a_mn.reshape((n_size, n_size)).T
    
    # dz-step
    dz = data["zeta"][n_size] - data["zeta"][0]
    # d(V)/dz
    A_z = jnp.zeros_like(A2)
    # at i = 0
    A_z = A_z.at[:, 0].set( (A2[:, 1] - A2[:, n_size - 1]) * (2 * dz) ** (-1) )
    # at i = n_size
    A_z = A_z.at[:, n_size - 1].set((A2[:, 0] - A2[:, n_size - 2]) * (2 * dz) ** (-1) )
    # Intermediate steps
    A_z = A_z.at[:, 1:n_size - 1].set((A2[:, 2:n_size] - A2[:, 0:n_size - 2]) * (2 * dz) ** (-1) )
    
    return (A_z.T).reshape(-1)#flatten()