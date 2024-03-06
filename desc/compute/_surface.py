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
    basis="basis",
)
def _X_Surface(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    if kwargs.get("basis", "rpz").lower() == "rpz":
        # if basis is rpz, then "x" is rpz and we must convert to xyz
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
    basis="basis",
)
def _Y_Surface(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    if kwargs.get("basis", "rpz").lower() == "rpz":
        # if basis is rpz, then "x" is rpz and we must convert to xyz
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
    basis="basis",
)
def _R_Surface(params, transforms, profiles, data, **kwargs):
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
    coordinates="rtz",
    data=["x"],
    parameterization="desc.geometry.core.Surface",
    basis="basis",
)
def _phi_Surface(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    if kwargs.get("basis", "rpz").lower() == "xyz":
        # if basis is xyz, then "x" is xyz and we must convert to rpz
        coords = xyz2rpz(coords)
    data["phi"] = coords[:, 1]
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
    data=["phi"],
    parameterization="desc.geometry.core.Surface",
    basis="basis",
)
def _phi_r_Surface(params, transforms, profiles, data, **kwargs):
    data["phi_r"] = jnp.zeros_like(data["phi"])
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
    data=["phi"],
    parameterization="desc.geometry.core.Surface",
    basis="basis",
)
def _phi_t_Surface(params, transforms, profiles, data, **kwargs):
    data["phi_t"] = jnp.zeros_like(data["phi"])
    return data


@register_compute_fun(
    name="phi_tt",
    label="\\partial_{\\theta\\theta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt"
    " poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["phi"],
    parameterization="desc.geometry.core.Surface",
    basis="basis",
)
def _phi_tt_Surface(params, transforms, profiles, data, **kwargs):
    data["phi_tt"] = jnp.zeros_like(data["phi"])
    return data


@register_compute_fun(
    name="phi_tz",
    label="\\partial_{\\theta\\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, first derivative wrt"
    " poloidal coordinate and first derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["phi"],
    parameterization="desc.geometry.core.Surface",
    basis="basis",
)
def _phi_tz_Surface(params, transforms, profiles, data, **kwargs):
    data["phi_tz"] = jnp.zeros_like(data["phi"])
    return data


@register_compute_fun(
    name="phi_ttt",
    label="\\partial_{\\theta\\theta\\theta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, third derivative wrt"
    " poloidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["phi"],
    parameterization="desc.geometry.core.Surface",
    basis="basis",
)
def _phi_ttt_Surface(params, transforms, profiles, data, **kwargs):
    data["phi_ttt"] = jnp.zeros_like(data["phi"])
    return data


@register_compute_fun(
    name="phi_ttz",
    label="\\partial_{\\theta\\theta\\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative wrt"
    " poloidal coordinate and first derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["phi"],
    parameterization="desc.geometry.core.Surface",
    basis="basis",
)
def _phi_ttz_Surface(params, transforms, profiles, data, **kwargs):
    data["phi_ttz"] = jnp.zeros_like(data["phi"])
    return data


@register_compute_fun(
    name="phi_tzz",
    label="\\partial_{\\theta\\zeta\\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, first derivative wrt"
    " poloidal coordinate and second derivative wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["phi"],
    parameterization="desc.geometry.core.Surface",
    basis="basis",
)
def _phi_tzz_Surface(params, transforms, profiles, data, **kwargs):
    data["phi_tzz"] = jnp.zeros_like(data["phi"])
    return data


@register_compute_fun(
    name="phi_zzz",
    label="\\partial_{\\zeta\\zeta\\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, third derivative wrt"
    " toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["phi"],
    parameterization="desc.geometry.core.Surface",
    basis="basis",
)
def _phi_zzz_Surface(params, transforms, profiles, data, **kwargs):
    data["phi_zzz"] = jnp.zeros_like(data["phi"])
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
    data=["phi"],
    parameterization="desc.geometry.core.Surface",
    basis="basis",
)
def _phi_z_Surface(params, transforms, profiles, data, **kwargs):
    # TODO: if surfaces eventually get an omega for generalized toroidal angle,
    # this (and everything else in this file) must be changed, for now this
    # assumes zeta = phi
    data["phi_z"] = jnp.ones_like(data["phi"])
    return data


@register_compute_fun(
    name="phi_zz",
    label="\\partial_{\\zeta} \\phi",
    units="rad",
    units_long="radians",
    description="Toroidal angle in lab frame, second derivative"
    " wrt toroidal coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["phi"],
    parameterization="desc.geometry.core.Surface",
    basis="basis",
)
def _phi_zz_Surface(params, transforms, profiles, data, **kwargs):
    data["phi_zz"] = jnp.zeros_like(data["phi"])
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
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["R_t", "Z_t", "phi"],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_theta_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R = data["R_t"]
    Z = data["Z_t"]
    phi = jnp.zeros_like(R)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
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
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["R", "R_z", "Z_z", "phi"],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_zeta_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R0 = data["R"]
    dR = data["R_z"]
    dZ = data["Z_z"]
    dphi = R0
    coords = jnp.stack([dR, dphi, dZ], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
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
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["R_tt", "Z_tt", "phi"],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_theta_t_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R = data["R_tt"]
    Z = data["Z_tt"]
    phi = jnp.zeros_like(R)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
    data["e_theta_t"] = coords
    return data


@register_compute_fun(
    name="e_theta_tt",
    label="\\partial^2_{\\theta} \\mathbf{e}_{\\theta\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, 2nd derivative wrt poloidal angle",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[0, 3, 0]],
        "Z": [[0, 3, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_theta_tt_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dt=3)
    Z = transforms["Z"].transform(params["Z_lmn"], dt=3)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["e_theta_tt"] = coords
    return data


@register_compute_fun(
    name="e_theta_tz",
    label="\\partial_{\\theta\\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, 2nd derivative wrt poloidal angle"
    " and toroidal angle",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[0, 2, 1]],
        "Z": [[0, 2, 1]],
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_theta_tz_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_lmn"], dt=2, dz=1)
    Z = transforms["Z"].transform(params["Z_lmn"], dt=2, dz=1)
    phi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["e_theta_tz"] = coords
    return data


@register_compute_fun(
    name="e_theta_z",
    label="\\partial_{\\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, derivative wrt toroidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["R_tz", "Z_tz", "R_t", "phi"],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_theta_z_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    dR = data["R_tz"]
    dZ = data["Z_tz"]
    dphi = data["R_t"]
    coords = jnp.stack([dR, dphi, dZ], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
    data["e_theta_z"] = coords
    return data


@register_compute_fun(
    name="e_theta_zz",
    label="\\partial_{\\zeta\\zeta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, 2nd derivative wrt toroidal angle",
    dim=3,
    params=["R_lmn", "Z_lmn"],
    transforms={
        "R": [[0, 1, 2]],
        "Z": [[0, 1, 2]],
        "grid": [],
    },
    profiles=[],
    coordinates="tz",
    data=[],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_theta_zz_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    dR = transforms["R"].transform(params["R_lmn"], dt=1, dz=2)
    dZ = transforms["Z"].transform(params["Z_lmn"], dt=1, dz=2)
    dphi = jnp.zeros(transforms["grid"].num_nodes)
    coords = jnp.stack([dR, dphi, dZ], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["e_theta_zz"] = coords
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
    params=[],
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["R_tz", "Z_tz", "R_t", "phi"],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_zeta_t_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    dR = data["R_tz"]
    dZ = data["Z_tz"]
    dphi = data["R_t"]
    coords = jnp.stack([dR, dphi, dZ], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
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
    transforms={},
    profiles=[],
    coordinates="tz",
    data=["R", "R_z", "R_zz", "Z_zz", "phi"],
    parameterization="desc.geometry.surface.FourierRZToroidalSurface",
    basis="basis",
)
def _e_zeta_z_FourierRZToroidalSurface(params, transforms, profiles, data, **kwargs):
    R0 = data["R"]
    dR = data["R_z"]
    d2R = data["R_zz"]
    d2Z = data["Z_zz"]
    dphi = 2 * dR
    coords = jnp.stack([d2R - R0, dphi, d2Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
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
    params=[],
    transforms={},
    profiles=[],
    coordinates="rt",
    data=["R_r", "Z_r", "phi"],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_rho_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = data["R_r"]
    Z = data["Z_r"]
    phi = jnp.zeros_like(R)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
    data["e_rho"] = coords
    return data


@register_compute_fun(
    name="e_theta",
    label="\\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rt",
    data=["R_t", "Z_t", "phi"],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_theta_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = data["R_t"]
    Z = data["Z_t"]
    phi = jnp.zeros_like(R)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
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
    params=[],
    transforms={},
    profiles=[],
    coordinates="rt",
    data=["R_rr", "Z_rr", "phi"],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_rho_r_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = data["R_rr"]
    Z = data["Z_rr"]
    phi = jnp.zeros_like(R)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
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
    transforms={},
    profiles=[],
    coordinates="rt",
    data=["R_rrr", "Z_rrr", "phi"],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_rho_rr_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = data["R_rrr"]
    Z = data["Z_rrr"]
    phi = jnp.zeros_like(R)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
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
    transforms={},
    profiles=[],
    coordinates="rt",
    data=["R_rt", "Z_rt", "phi"],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
    aliases=["e_theta_r"],
)
def _e_rho_t_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = data["R_rt"]
    Z = data["Z_rt"]
    phi = jnp.zeros_like(R)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
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
    name="e_theta_rr",
    label="\\partial_{\\rho \\rho} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector,"
    " second derivative wrt radial coordinate",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rt",
    data=["R_rrt", "Z_rrt", "phi"],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_theta_rr_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = data["R_rrt"]
    Z = data["Z_rrt"]
    phi = jnp.zeros_like(R)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
    data["e_theta_rr"] = coords
    return data


@register_compute_fun(
    name="e_theta_t",
    label="\\partial_{\\theta} \\mathbf{e}_{\\theta}",
    units="m",
    units_long="meters",
    description="Covariant poloidal basis vector, derivative wrt poloidal angle",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rt",
    data=["R_tt", "Z_tt", "phi"],
    parameterization="desc.geometry.surface.ZernikeRZToroidalSection",
    basis="basis",
)
def _e_theta_t_ZernikeRZToroidalSection(params, transforms, profiles, data, **kwargs):
    R = data["R_tt"]
    Z = data["Z_tt"]
    phi = jnp.zeros_like(R)
    coords = jnp.stack([R, phi, Z], axis=1)
    if kwargs.get("basis", "rpz").lower() == "xyz":
        coords = rpz2xyz_vec(coords, phi=data["phi"])
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
