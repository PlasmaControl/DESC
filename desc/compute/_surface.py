from desc.backend import jnp

from .data_index import register_compute_fun
from .geom_utils import rpz2xyz, rpz2xyz_vec


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
