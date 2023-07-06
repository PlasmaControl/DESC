from desc.backend import jnp
from desc.geometry.utils import rpz2xyz, rpz2xyz_vec

from .data_index import register_compute_fun


@register_compute_fun(
    name="r",
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
    parameterization="desc.geometry.FourierRZToroidalSurface",
)
def _r_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"])
    Z = transforms["Z"].transform(params["Z_lmn"])
    phi = transforms["grid"].nodes[:, 2]
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
    data["r"] = coords
    return data


@register_compute_fun(
    name="r_r",
    label="\\partial_{\\rho} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, radial derivative",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.FourierRZToroidalSurface",
)
def _r_r_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["r_r"] = coords
    return data


@register_compute_fun(
    name="r_t",
    label="\\partial_{\\theta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, poloidal derivative",
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
    parameterization="desc.geometry.FourierRZToroidalSurface",
)
def _r_t_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dt=1)
    Z = transforms["Z"].transform(params["Z_lmn"], dt=1)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["r_t"] = coords
    return data


@register_compute_fun(
    name="r_z",
    label="\\partial_{\\zeta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, toroidal derivative",
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
    parameterization="desc.geometry.FourierRZToroidalSurface",
)
def _r_z_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_lmn"], dz=0)
    dR = transforms["R"].transform(params["R_lmn"], dz=1)
    dZ = transforms["Z"].transform(params["Z_lmn"], dz=1)
    dphi = R0
    coords = jnp.stack([dR, dphi, dZ], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["r_z"] = coords
    return data


@register_compute_fun(
    name="r_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, second radial derivative",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.FourierRZToroidalSurface",
)
def _r_rr_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["r_rr"] = coords
    return data


@register_compute_fun(
    name="r_tt",
    label="\\partial_{\theta \\theta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, second poloidal derivative",
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
    parameterization="desc.geometry.FourierRZToroidalSurface",
)
def _r_tt_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dt=2)
    Z = transforms["Z"].transform(params["Z_lmn"], dt=2)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["r_tt"] = coords
    return data


@register_compute_fun(
    name="r_zz",
    label="\\partial_{\\zeta \\zeta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, toroidal derivative",
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
    parameterization="desc.geometry.FourierRZToroidalSurface",
)
def _r_zz_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_lmn"], dz=0)
    dR = transforms["R"].transform(params["R_lmn"], dz=1)
    d2R = transforms["R"].transform(params["R_lmn"], dz=2)
    d2Z = transforms["Z"].transform(params["Z_lmn"], dz=2)
    dphi = 2 * dR
    coords = jnp.stack([d2R - R0, dphi, d2Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["r_zz"] = coords
    return data


@register_compute_fun(
    name="r_rt",
    label="\\partial_{\\rho \\theta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, radial/poloidal derivative",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.FourierRZToroidalSurface",
)
def _r_rt_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["r_rt"] = coords
    return data


@register_compute_fun(
    name="r_rz",
    label="\\partial_{\\rho \\zeta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, radial/toroidal derivative",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.FourierRZToroidalSurface",
)
def _r_rz_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["r_rz"] = coords
    return data


@register_compute_fun(
    name="r_tz",
    label="\\partial_{\\theta \\zeta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, poloidal/toroidal derivative",
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
    parameterization="desc.geometry.FourierRZToroidalSurface",
)
def _r_tz_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    dR = transforms["R"].transform(params["R_lmn"], dt=1, dz=1)
    dZ = transforms["Z"].transform(params["Z_lmn"], dt=1, dz=1)
    dphi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([dR, dphi, dZ], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["r_tz"] = coords
    return data


@register_compute_fun(
    name="r",
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
    parameterization="desc.geometry.ZernikeRZToroidalSection",
)
def _r_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"])
    Z = transforms["Z"].transform(params["Z_lmn"])
    phi = transforms["grid"].nodes[:, 2]
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
    data["r"] = coords
    return data


@register_compute_fun(
    name="r_r",
    label="\\partial_{\\rho} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, radial derivative",
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
    parameterization="desc.geometry.ZernikeRZToroidalSection",
)
def _r_r_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dr=1)
    Z = transforms["Z"].transform(params["Z_lmn"], dr=1)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
    data["r_r"] = coords
    return data


@register_compute_fun(
    name="r_t",
    label="\\partial_{\\theta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, poloidal derivative",
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
    parameterization="desc.geometry.ZernikeRZToroidalSection",
)
def _r_t_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dt=1)
    Z = transforms["Z"].transform(params["Z_lmn"], dt=1)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["r_t"] = coords
    return data


@register_compute_fun(
    name="r_z",
    label="\\partial_{\\zeta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, toroidal derivative",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.ZernikeRZToroidalSection",
)
def _r_z_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["r_z"] = coords
    return data


@register_compute_fun(
    name="r_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, second radial derivative",
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
    parameterization="desc.geometry.ZernikeRZToroidalSection",
)
def _r_rr_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dr=2)
    Z = transforms["Z"].transform(params["Z_lmn"], dr=2)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
    data["r_rr"] = coords
    return data


@register_compute_fun(
    name="r_tt",
    label="\\partial_{\theta \\theta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, second poloidal derivative",
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
    parameterization="desc.geometry.ZernikeRZToroidalSection",
)
def _r_tt_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dt=2)
    Z = transforms["Z"].transform(params["Z_lmn"], dt=2)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["r_tt"] = coords
    return data


@register_compute_fun(
    name="r_zz",
    label="\\partial_{\\zeta \\zeta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, toroidal derivative",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.ZernikeRZToroidalSection",
)
def _r_zz_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["r_zz"] = coords
    return data


@register_compute_fun(
    name="r_rt",
    label="\\partial_{\\rho \\theta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, radial/poloidal derivative",
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
    parameterization="desc.geometry.ZernikeRZToroidalSection",
)
def _r_rt_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dr=1, dt=1)
    Z = transforms["Z"].transform(params["Z_lmn"], dr=1, dt=1)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz(coords)
    data["r_rt"] = coords
    return data


@register_compute_fun(
    name="r_rz",
    label="\\partial_{\\rho \\zeta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, radial/toroidal derivative",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.ZernikeRZToroidalSection",
)
def _r_rz_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["r_rz"] = coords
    return data


@register_compute_fun(
    name="r_tz",
    label="\\partial_{\\theta \\zeta} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along surface, poloidal/toroidal derivative",
    dim=3,
    params=[],
    transforms={
        "grid": [],
    },
    profiles=[],
    coordinates="rt",
    data=[],
    parameterization="desc.geometry.ZernikeRZToroidalSection",
)
def _r_tz_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    coords = jnp.zeros((transforms["grid"].num_nodes, 3))
    data["r_tz"] = coords
    return data
