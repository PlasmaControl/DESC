from desc.backend import jnp

from .data_index import register_compute_fun
from .geom_utils import rpz2xyz, rpz2xyz_vec, xyz2rpz


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
    basis="basis",
)
def _x_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"])
    Z = transforms["Z"].transform(params["Z_lmn"])
    phi = transforms["grid"].nodes[:, 2]
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
    data["x"] = coords
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
    coordinates="tz",
    data=["x"],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _R_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    if kwargs.get("basis", "rpz").lower() == "xyz":
        # if basis is xyz, then "x" is xyz and we must convert to rpz
        coords = xyz2rpz(coords)
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
    coordinates="tz",
    data=["x"],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _phi_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    if kwargs.get("basis", "rpz").lower() == "xyz":
        # if basis is xyz, then "x" is xyz and we must convert to rpz
        coords = xyz2rpz(coords)
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
    coordinates="tz",
    data=["x"],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
)
def _Z_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
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
    basis="basis",
)
def _e_rho_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_rho"] = coords
    return data


@register_compute_fun(
    name="e_theta",
    label="\\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[0, 1, 0]],
        "Z": [[0, 1, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_theta_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dt=1)
    Z = transforms["Z"].transform(params["Z_lmn"], dt=1)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["e_theta"] = coords
    return data


@register_compute_fun(
    name="e_zeta",
    label="\\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[0, 0, 0], [0, 0, 1]],
        "Z": [[0, 0, 1]],
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_zeta_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_lmn"], dz=0)
    dR = transforms["R"].transform(params["R_lmn"], dz=1)
    dZ = transforms["Z"].transform(params["Z_lmn"], dz=1)
    dphi = R0
    coords = jnp.stack([dR, dphi, dZ], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["e_zeta"] = coords
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
    basis="basis",
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
    basis="basis",
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
    basis="basis",
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
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_rho_z_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_rho_z"] = coords
    return data


@register_compute_fun(
    name="e_theta_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_theta_r_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_theta_r"] = coords
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
    basis="basis",
)
def _e_theta_rr_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_theta_rr"] = coords
    return data


@register_compute_fun(
    name="e_theta_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, derivative wrt poloidal angle",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[0, 2, 0]],
        "Z": [[0, 2, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_theta_t_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dt=2)
    Z = transforms["Z"].transform(params["Z_lmn"], dt=2)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["e_theta_t"] = coords
    return data


@register_compute_fun(
    name="e_theta_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, derivative wrt toroidal angle",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[0, 1, 1]],
        "Z": [[0, 1, 1]],
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_theta_z_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    dR = transforms["R"].transform(params["R_lmn"], dt=1, dz=1)
    dZ = transforms["Z"].transform(params["Z_lmn"], dt=1, dz=1)
    dphi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([dR, dphi, dZ], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["e_theta_z"] = coords
    return data


@register_compute_fun(
    name="e_zeta_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_zeta_r_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_zeta_r"] = coords
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
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_zeta_rr_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_zeta_rr"] = coords
    return data


@register_compute_fun(
    name="e_zeta_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, derivative wrt poloidal angle",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[0, 1, 1]],
        "Z": [[0, 1, 1]],
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_zeta_t_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    dR = transforms["R"].transform(params["R_lmn"], dt=1, dz=1)
    dZ = transforms["Z"].transform(params["Z_lmn"], dt=1, dz=1)
    dphi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([dR, dphi, dZ], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["e_zeta_t"] = coords
    return data


@register_compute_fun(
    name="e_zeta_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, derivative wrt toroidal angle",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[0, 0, 0], [0, 0, 1], [0, 0, 2]],
        "Z": [[0, 0, 2]],
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_zeta_z_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_lmn"], dz=0)
    dR = transforms["R"].transform(params["R_lmn"], dz=1)
    d2R = transforms["R"].transform(params["R_lmn"], dz=2)
    d2Z = transforms["Z"].transform(params["Z_lmn"], dz=2)
    dphi = 2 * dR
    coords = jnp.stack([d2R - R0, dphi, d2Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["e_zeta_z"] = coords
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
    parameterization="desc.magnetic_fields.FourierCurrentPotentialField",
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
    parameterization="desc.magnetic_fields.FourierCurrentPotentialField",
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
    parameterization="desc.magnetic_fields.FourierCurrentPotentialField",
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
    params=["params"],
    transforms={"grid": [], "potential": []},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.magnetic_fields.CurrentPotentialField",
)
def _Phi_CurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi"] = transforms["potential"](
        transforms["grid"].nodes[:, 1],
        transforms["grid"].nodes[:, 2],
        **params["params"]
    )
    return data


@register_compute_fun(
    name="Phi_t",
    label="\\partial_{\\theta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, poloidal derivative",
    dim=1,
    params=["params"],
    transforms={"grid": [], "potential_dtheta": []},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.magnetic_fields.CurrentPotentialField",
)
def _Phi_t_CurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_t"] = transforms["potential_dtheta"](
        transforms["grid"].nodes[:, 1],
        transforms["grid"].nodes[:, 2],
        **params["params"]
    )
    return data


@register_compute_fun(
    name="Phi_z",
    label="\\partial_{\\zeta}\\Phi",
    units="A",
    units_long="Amperes",
    description="Surface current potential, toroidal derivative",
    dim=1,
    params=["params"],
    transforms={"grid": [], "potential_dzeta": []},
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.magnetic_fields.CurrentPotentialField",
)
def _Phi_z_CurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["Phi_z"] = transforms["potential_dzeta"](
        transforms["grid"].nodes[:, 1],
        transforms["grid"].nodes[:, 2],
        **params["params"]
    )
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
    data=["Phi_t", "Phi_z", "e_theta", "e_zeta", "|e_theta x e_zeta|"],
    parameterization=[
        "desc.magnetic_fields.CurrentPotentialField",
        "desc.magnetic_fields.FourierCurrentPotentialField",
    ],
)
def _K_CurrentPotentialField(params, transforms, profiles, data, **kwargs):
    data["K"] = (
        data["Phi_t"] * (1 / data["|e_theta x e_zeta|"]) * data["e_zeta"].T
    ).T - (data["Phi_z"] * (1 / data["|e_theta x e_zeta|"]) * data["e_theta"].T).T
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
    basis="basis",
)
def _x_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"])
    Z = transforms["Z"].transform(params["Z_lmn"])
    phi = transforms["grid"].nodes[:, 2]
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
    data["x"] = coords
    return data


@register_compute_fun(
    name="e_rho",
    label="\\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[1, 0, 0]],
        "Z": [[1, 0, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_rho_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dr=1)
    Z = transforms["Z"].transform(params["Z_lmn"], dr=1)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
    data["e_rho"] = coords
    return data


@register_compute_fun(
    name="e_theta",
    label="\\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[0, 1, 0]],
        "Z": [[0, 1, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_theta_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dt=1)
    Z = transforms["Z"].transform(params["Z_lmn"], dt=1)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["e_theta"] = coords
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
    basis="basis",
)
def _e_zeta_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_zeta"] = coords
    return data


@register_compute_fun(
    name="e_rho_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, derivative wrt radial coordinate",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[2, 0, 0]],
        "Z": [[2, 0, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_rho_r_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dr=2)
    Z = transforms["Z"].transform(params["Z_lmn"], dr=2)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
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
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[3, 0, 0]],
        "Z": [[3, 0, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_rho_rr_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dr=3)
    Z = transforms["Z"].transform(params["Z_lmn"], dr=3)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
    data["e_rho_rr"] = coords
    return data


@register_compute_fun(
    name="e_rho_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\rho}",
    units="m",
    units_long="meters",
    description="Covariant radial basis vector, derivative wrt poloidal angle",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[1, 1, 0]],
        "Z": [[1, 1, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_rho_t_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dr=1, dt=1)
    Z = transforms["Z"].transform(params["Z_lmn"], dr=1, dt=1)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
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
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_rho_z_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_rho_z"] = coords
    return data


@register_compute_fun(
    name="e_theta_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, derivative wrt radial coordinate",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[1, 1, 0]],
        "Z": [[1, 1, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_theta_r_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dr=1, dt=1)
    Z = transforms["Z"].transform(params["Z_lmn"], dr=1, dt=1)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
    data["e_theta_r"] = coords
    return data


@register_compute_fun(
    name="e_theta_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector,"
    " second derivative wrt radial coordinate",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[2, 1, 0]],
        "Z": [[2, 1, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_theta_rr_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dr=2, dt=1)
    Z = transforms["Z"].transform(params["Z_lmn"], dr=2, dt=1)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
    data["e_theta_rr"] = coords
    return data


@register_compute_fun(
    name="e_theta_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, derivative wrt poloidal angle",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[0, 2, 0]],
        "Z": [[0, 2, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_theta_t_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dt=2)
    Z = transforms["Z"].transform(params["Z_lmn"], dt=2)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["e_theta_t"] = coords
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
    basis="basis",
)
def _e_theta_z_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_theta_z"] = coords
    return data


@register_compute_fun(
    name="e_zeta_r",
    label="\\partial_{\\rho} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_zeta_r_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_zeta_r"] = coords
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
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_zeta_rr_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_zeta_rr"] = coords
    return data


@register_compute_fun(
    name="e_zeta_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\zeta}",
    units="m",
    units_long="meters",
    description="Covariant toroidal basis vector, derivative wrt poloidal angle",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_zeta_t_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_zeta_t"] = coords
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
    basis="basis",
)
def _e_zeta_z_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["e_zeta_z"] = coords
    return data
