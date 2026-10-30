from interpax import interp1d

from desc.backend import jnp, root_scalar, scan, sign, vmap

from ..utils import (
    cross,
    dot,
    rotation_matrix,
    rpz2xyz,
    rpz2xyz_vec,
    xyz2rpz,
    xyz2rpz_vec,
)
from .data_index import register_compute_fun
from .spline_utils import b_p_deriv3, chord_length_knots, uniform_knots


@register_compute_fun(
    name="s",
    label="s",
    units="~",
    units_long="None",
    description="Curve parameter, on [0, 2pi)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.core.Curve",
)
def _s(params, transforms, profiles, data, **kwargs):
    data["s"] = transforms["grid"].nodes[:, 2]
    return data


@register_compute_fun(
    name="ds",
    label="ds",
    units="~",
    units_long="None",
    description=(
        "Quadrature weights for integration along the curve,"
        + " i.e. an alias for ``grid.spacing[:,2]``"
    ),
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.core.Curve",
)
def _ds(params, transforms, profiles, data, **kwargs):
    data["ds"] = transforms["grid"].spacing[:, 2]
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
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
)
def _X_curve(params, transforms, profiles, data, **kwargs):
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
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
)
def _Y_Curve(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    coords = rpz2xyz(coords)
    data["Y"] = coords[:, 1]
    return data


@register_compute_fun(
    name="R",
    label="R",
    units="m",
    units_long="meters",
    description="Cylindrical radial position along curve",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
)
def _R_Curve(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    data["R"] = coords[:, 0]
    return data


@register_compute_fun(
    name="phi",
    label="\\phi",
    units="rad",
    units_long="radians",
    description="Toroidal phi position along curve",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
)
def _phi_Curve(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    data["phi"] = coords[:, 1]
    return data


@register_compute_fun(
    name="Z",
    label="Z",
    units="m",
    units_long="meters",
    description="Cylindrical vertical position along curve",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
)
def _Z_Curve(params, transforms, profiles, data, **kwargs):
    data["Z"] = data["x"][:, 2]
    return data


@register_compute_fun(
    name="center",
    label="\\langle\\mathbf{x}\\rangle",
    units="m",
    units_long="meters",
    description="Centroid of the curve",
    dim=3,
    params=["center", "rotmat", "shift"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization=[
        "desc.geometry.curve.FourierPlanarCurve",
        "desc.geometry.curve.FourierXYCurve",
    ],
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _center_PlanarCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        center = rpz2xyz(params["center"])
    else:
        center = params["center"]
    # displacement and rotation
    center = jnp.matmul(center, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert back to rpz
    data["center"] = xyz2rpz(center) * jnp.ones_like(data["x"])
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["r_n", "center", "normal", "rotmat", "shift"],
    transforms={"r": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        center = rpz2xyz(params["center"])
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        center = params["center"]
        normal = params["normal"]
    # create planar curve at Z==0
    r = transforms["r"].transform(params["r_n"], dz=0)
    Z = jnp.zeros_like(r)
    X = r * jnp.cos(data["s"])
    Y = r * jnp.sin(data["s"])
    coords = jnp.array([X, Y, Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    A = rotation_matrix(Zaxis, normal)
    coords = jnp.matmul(coords, A.T) + center
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert back to rpz
    coords = xyz2rpz(coords)
    data["x"] = coords
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["r_n", "center", "normal", "rotmat"],
    transforms={"r": [[0, 0, 0], [0, 0, 1]]},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_s_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]
    r = transforms["r"].transform(params["r_n"], dz=0)
    dr = transforms["r"].transform(params["r_n"], dz=1)
    dX = dr * jnp.cos(data["s"]) - r * jnp.sin(data["s"])
    dY = dr * jnp.sin(data["s"]) + r * jnp.cos(data["s"])
    dZ = jnp.zeros_like(dX)
    coords = jnp.array([dX, dY, dZ]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    A = rotation_matrix(Zaxis, normal)

    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    # convert back to rpz
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_s"] = coords
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["r_n", "center", "normal", "rotmat"],
    transforms={"r": [[0, 0, 0], [0, 0, 1], [0, 0, 2]]},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_ss_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]
    r = transforms["r"].transform(params["r_n"], dz=0)
    dr = transforms["r"].transform(params["r_n"], dz=1)
    d2r = transforms["r"].transform(params["r_n"], dz=2)
    d2X = (
        d2r * jnp.cos(data["s"]) - 2 * dr * jnp.sin(data["s"]) - r * jnp.cos(data["s"])
    )
    d2Y = (
        d2r * jnp.sin(data["s"]) + 2 * dr * jnp.cos(data["s"]) - r * jnp.sin(data["s"])
    )
    d2Z = jnp.zeros_like(d2X)
    coords = jnp.array([d2X, d2Y, d2Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    A = rotation_matrix(Zaxis, normal)

    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    # convert back to rpz
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_ss"] = coords
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["r_n", "center", "normal", "rotmat"],
    transforms={"r": [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_sss_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]
    r = transforms["r"].transform(params["r_n"], dz=0)
    dr = transforms["r"].transform(params["r_n"], dz=1)
    d2r = transforms["r"].transform(params["r_n"], dz=2)
    d3r = transforms["r"].transform(params["r_n"], dz=3)
    d3X = (
        d3r * jnp.cos(data["s"])
        - 3 * d2r * jnp.sin(data["s"])
        - 3 * dr * jnp.cos(data["s"])
        + r * jnp.sin(data["s"])
    )
    d3Y = (
        d3r * jnp.sin(data["s"])
        + 3 * d2r * jnp.cos(data["s"])
        - 3 * dr * jnp.sin(data["s"])
        - r * jnp.cos(data["s"])
    )
    d3Z = jnp.zeros_like(d3X)
    coords = jnp.array([d3X, d3Y, d3Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    A = rotation_matrix(Zaxis, normal)

    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    # convert back to rpz
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_sss"] = coords
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["X_n", "Y_n", "center", "normal", "rotmat", "shift"],
    transforms={"X": [[0, 0, 0]], "Y": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierXYCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_FourierXYCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        center = rpz2xyz(params["center"])
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        center = params["center"]
        normal = params["normal"]
    # create planar curve at Z==0
    X = transforms["X"].transform(params["X_n"], dz=0)
    Y = transforms["Y"].transform(params["Y_n"], dz=0)
    Z = jnp.zeros_like(X)
    coords = jnp.array([X, Y, Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    A = rotation_matrix(Zaxis, normal)

    coords = jnp.matmul(coords, A.T) + center
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert back to rpz
    coords = xyz2rpz(coords)
    data["x"] = coords
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["X_n", "Y_n", "center", "normal", "rotmat"],
    transforms={"X": [[0, 0, 1]], "Y": [[0, 0, 1]]},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierXYCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_s_FourierXYCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]
    dX = transforms["X"].transform(params["X_n"], dz=1)
    dY = transforms["Y"].transform(params["Y_n"], dz=1)
    dZ = jnp.zeros_like(dX)
    coords = jnp.array([dX, dY, dZ]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    A = rotation_matrix(Zaxis, normal)

    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    # convert back to rpz
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_s"] = coords
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["X_n", "Y_n", "center", "normal", "rotmat"],
    transforms={"X": [[0, 0, 2]], "Y": [[0, 0, 2]]},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierXYCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_ss_FourierXYCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]
    d2X = transforms["X"].transform(params["X_n"], dz=2)
    d2Y = transforms["Y"].transform(params["Y_n"], dz=2)
    d2Z = jnp.zeros_like(d2X)
    coords = jnp.array([d2X, d2Y, d2Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    A = rotation_matrix(Zaxis, normal)

    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    # convert back to rpz
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_ss"] = coords
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["X_n", "Y_n", "center", "normal", "rotmat"],
    transforms={"X": [[0, 0, 3]], "Y": [[0, 0, 3]]},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierXYCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_sss_FourierXYCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]
    d3X = transforms["X"].transform(params["X_n"], dz=3)
    d3Y = transforms["Y"].transform(params["Y_n"], dz=3)
    d3Z = jnp.zeros_like(d3X)
    coords = jnp.array([d3X, d3Y, d3Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    A = rotation_matrix(Zaxis, normal)

    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    # convert back to rpz
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_sss"] = coords
    return data


@register_compute_fun(
    name="center",
    label="\\langle\\mathbf{x}\\rangle",
    units="m",
    units_long="meters",
    description="Centroid of the curve",
    dim=3,
    params=["R_n", "Z_n", "rotmat", "shift"],
    transforms={"R": [[0, 0, 0]], "Z": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.curve.FourierRZCurve",
)
def _center_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    idx_Rc = transforms["R"].basis.get_idx(N=1, error=False)
    idx_Rs = transforms["R"].basis.get_idx(N=-1, error=False)
    idx_Z = transforms["Z"].basis.get_idx(N=0, error=False)
    X0 = params["R_n"][idx_Rc] / 2 if isinstance(idx_Rc, int) else 0
    Y0 = params["R_n"][idx_Rs] / 2 if isinstance(idx_Rs, int) else 0
    Z0 = params["Z_n"][idx_Z] if isinstance(idx_Z, int) else 0
    center = jnp.array([X0, Y0, Z0])
    # displacement and rotation
    center = jnp.matmul(center, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert back to rpz
    data["center"] = xyz2rpz(center) * jnp.ones_like(data["x"])
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["R_n", "Z_n", "rotmat", "shift"],
    transforms={"R": [[0, 0, 0]], "Z": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierRZCurve",
)
def _x_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_n"], dz=0)
    Z = transforms["Z"].transform(params["Z_n"], dz=0)
    phi = transforms["grid"].nodes[:, 2]
    coords = jnp.stack([R, phi, Z], axis=1)
    # convert to xyz for displacement and rotation
    coords = rpz2xyz(coords)
    coords = (
        coords @ params["rotmat"].reshape((3, 3)).T + params["shift"][jnp.newaxis, :]
    )
    # convert back to rpz
    coords = xyz2rpz(coords)
    data["x"] = coords
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["R_n", "Z_n", "rotmat"],
    transforms={"R": [[0, 0, 0], [0, 0, 1]], "Z": [[0, 0, 1]], "grid": []},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierRZCurve",
)
def _x_s_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_n"], dz=0)
    dR = transforms["R"].transform(params["R_n"], dz=1)
    dZ = transforms["Z"].transform(params["Z_n"], dz=1)
    coords = jnp.stack([dR, R0, dZ], axis=1)
    # convert to xyz for rotation using phi=s
    coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    # convert back to rpz using real phi to account for displacement
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_s"] = coords
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["R_n", "Z_n", "rotmat"],
    transforms={"R": [[0, 0, 0], [0, 0, 1], [0, 0, 2]], "Z": [[0, 0, 2]], "grid": []},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierRZCurve",
)
def _x_ss_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_n"], dz=0)
    d1R = transforms["R"].transform(params["R_n"], dz=1)
    d2R = transforms["R"].transform(params["R_n"], dz=2)
    d2Z = transforms["Z"].transform(params["Z_n"], dz=2)
    coords = jnp.stack([d2R - R0, 2 * d1R, d2Z], axis=1)
    # convert to xyz for rotation using phi=s
    coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    # convert back to rpz using real phi to account for displacement
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_ss"] = coords
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["R_n", "Z_n", "rotmat"],
    transforms={
        "R": [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
        "Z": [[0, 0, 3]],
        "grid": [],
    },
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierRZCurve",
)
def _x_sss_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_n"], dz=0)
    d1R = transforms["R"].transform(params["R_n"], dz=1)
    d2R = transforms["R"].transform(params["R_n"], dz=2)
    d3R = transforms["R"].transform(params["R_n"], dz=3)
    d3Z = transforms["Z"].transform(params["Z_n"], dz=3)
    coords = jnp.stack([d3R - 3 * d1R, 3 * d2R - R0, d3Z], axis=1)
    # convert to xyz for rotation using phi=s
    coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    # convert back to rpz using real phi to account for displacement
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_sss"] = coords
    return data


@register_compute_fun(
    name="center",
    label="\\langle\\mathbf{x}\\rangle",
    units="m",
    units_long="meters",
    description="Centroid of the curve",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat", "shift"],
    transforms={"X": [[0, 0, 0]], "Y": [[0, 0, 0]], "Z": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.curve.FourierXYZCurve",
)
def _center_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    idx_X = transforms["X"].basis.get_idx(N=0, error=False)
    idx_Y = transforms["Y"].basis.get_idx(N=0, error=False)
    idx_Z = transforms["Z"].basis.get_idx(N=0, error=False)
    X0 = params["X_n"][idx_X] if isinstance(idx_X, int) else 0
    Y0 = params["Y_n"][idx_Y] if isinstance(idx_Y, int) else 0
    Z0 = params["Z_n"][idx_Z] if isinstance(idx_Z, int) else 0
    center = jnp.array([X0, Y0, Z0])
    # displacement and rotation
    center = jnp.matmul(center, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert to rpz
    data["center"] = xyz2rpz(center) * jnp.ones_like(data["x"])
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat", "shift"],
    transforms={"X": [[0, 0, 0]], "Y": [[0, 0, 0]], "Z": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierXYZCurve",
)
def _x_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    X = transforms["X"].transform(params["X_n"], dz=0)
    Y = transforms["Y"].transform(params["Y_n"], dz=0)
    Z = transforms["Z"].transform(params["Z_n"], dz=0)
    coords = jnp.stack([X, Y, Z], axis=1)
    coords = (
        coords @ params["rotmat"].reshape((3, 3)).T + params["shift"][jnp.newaxis, :]
    )
    coords = xyz2rpz(coords)
    data["x"] = coords
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat"],
    transforms={"X": [[0, 0, 1]], "Y": [[0, 0, 1]], "Z": [[0, 0, 1]]},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierXYZCurve",
)
def _x_s_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    dX = transforms["X"].transform(params["X_n"], dz=1)
    dY = transforms["Y"].transform(params["Y_n"], dz=1)
    dZ = transforms["Z"].transform(params["Z_n"], dz=1)
    coords = jnp.stack([dX, dY, dZ], axis=1)
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_s"] = coords
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat"],
    transforms={"X": [[0, 0, 2]], "Y": [[0, 0, 2]], "Z": [[0, 0, 2]]},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierXYZCurve",
)
def _x_ss_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    d2X = transforms["X"].transform(params["X_n"], dz=2)
    d2Y = transforms["Y"].transform(params["Y_n"], dz=2)
    d2Z = transforms["Z"].transform(params["Z_n"], dz=2)
    coords = jnp.stack([d2X, d2Y, d2Z], axis=1)
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_ss"] = coords
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat"],
    transforms={"X": [[0, 0, 3]], "Y": [[0, 0, 3]], "Z": [[0, 0, 3]]},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierXYZCurve",
)
def _x_sss_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    d3X = transforms["X"].transform(params["X_n"], dz=3)
    d3Y = transforms["Y"].transform(params["Y_n"], dz=3)
    d3Z = transforms["Z"].transform(params["Z_n"], dz=3)
    coords = jnp.stack([d3X, d3Y, d3Z], axis=1)
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_sss"] = coords
    return data


@register_compute_fun(
    name="center",
    label="\\langle\\mathbf{x}\\rangle",
    units="m",
    units_long="meters",
    description="Centroid of the curve",
    dim=3,
    params=["X", "Y", "Z", "rotmat", "shift"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
)
def _center_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    # center is average of xyz knots
    xyz = jnp.stack([params["X"], params["Y"], params["Z"]], axis=1)
    center = jnp.mean(xyz, axis=0)
    # displacement and rotation
    center = jnp.matmul(center, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert to rpz
    data["center"] = xyz2rpz(center) * jnp.ones_like(data["x"])
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["X", "Y", "Z", "rotmat", "shift"],
    transforms={"knots": []},
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    method="Interpolation type, Default 'cubic'. See SplineXYZCurve docs for options.",
)
def _x_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    xq = data["s"]
    Xq = interp1d(
        xq,
        transforms["knots"],
        params["X"],
        method=kwargs["method"],
        derivative=0,
        period=2 * jnp.pi,
    )
    Yq = interp1d(
        xq,
        transforms["knots"],
        params["Y"],
        method=kwargs["method"],
        derivative=0,
        period=2 * jnp.pi,
    )
    Zq = interp1d(
        xq,
        transforms["knots"],
        params["Z"],
        method=kwargs["method"],
        derivative=0,
        period=2 * jnp.pi,
    )
    coords = jnp.stack([Xq, Yq, Zq], axis=1)
    coords = (
        coords @ params["rotmat"].reshape((3, 3)).T + params["shift"][jnp.newaxis, :]
    )
    coords = xyz2rpz(coords)
    data["x"] = coords
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["X", "Y", "Z", "rotmat"],
    transforms={"knots": []},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    method="Interpolation type, Default 'cubic'. See SplineXYZCurve docs for options.",
)
def _x_s_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    xq = data["s"]
    dXq = interp1d(
        xq,
        transforms["knots"],
        params["X"],
        method=kwargs["method"],
        derivative=1,
        period=2 * jnp.pi,
    )
    dYq = interp1d(
        xq,
        transforms["knots"],
        params["Y"],
        method=kwargs["method"],
        derivative=1,
        period=2 * jnp.pi,
    )
    dZq = interp1d(
        xq,
        transforms["knots"],
        params["Z"],
        method=kwargs["method"],
        derivative=1,
        period=2 * jnp.pi,
    )
    coords = jnp.stack([dXq, dYq, dZq], axis=1)
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_s"] = coords
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["X", "Y", "Z", "rotmat"],
    transforms={"knots": []},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    method="Interpolation type, Default 'cubic'. See SplineXYZCurve docs for options.",
)
def _x_ss_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    xq = data["s"]
    d2Xq = interp1d(
        xq,
        transforms["knots"],
        params["X"],
        method=kwargs["method"],
        derivative=2,
        period=2 * jnp.pi,
    )
    d2Yq = interp1d(
        xq,
        transforms["knots"],
        params["Y"],
        method=kwargs["method"],
        derivative=2,
        period=2 * jnp.pi,
    )
    d2Zq = interp1d(
        xq,
        transforms["knots"],
        params["Z"],
        method=kwargs["method"],
        derivative=2,
        period=2 * jnp.pi,
    )
    coords = jnp.stack([d2Xq, d2Yq, d2Zq], axis=1)
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_ss"] = coords
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["X", "Y", "Z", "rotmat"],
    transforms={"knots": []},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    method="Interpolation type, Default 'cubic'. See SplineXYZCurve docs for options.",
)
def _x_sss_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    xq = data["s"]
    d3Xq = interp1d(
        xq,
        transforms["knots"],
        params["X"],
        method=kwargs["method"],
        derivative=3,
        period=2 * jnp.pi,
    )
    d3Yq = interp1d(
        xq,
        transforms["knots"],
        params["Y"],
        method=kwargs["method"],
        derivative=3,
        period=2 * jnp.pi,
    )
    d3Zq = interp1d(
        xq,
        transforms["knots"],
        params["Z"],
        method=kwargs["method"],
        derivative=3,
        period=2 * jnp.pi,
    )
    coords = jnp.stack([d3Xq, d3Yq, d3Zq], axis=1)
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_sss"] = coords
    return data


@register_compute_fun(
    name="frenet_tangent",
    label="\\mathbf{T}_{\\mathrm{Frenet-Serret}}",
    units="~",
    units_long="None",
    description="Tangent unit vector to curve in Frenet-Serret frame",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x_s"],
    parameterization="desc.geometry.core.Curve",
)
def _frenet_tangent(params, transforms, profiles, data, **kwargs):
    data["frenet_tangent"] = (
        data["x_s"] / jnp.linalg.norm(data["x_s"], axis=-1)[:, None]
    )
    return data


@register_compute_fun(
    name="frenet_normal",
    label="\\mathbf{N}_{\\mathrm{Frenet-Serret}}",
    units="~",
    units_long="None",
    description="Normal unit vector to curve in Frenet-Serret frame",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x_s", "x_ss"],
    parameterization="desc.geometry.core.Curve",
)
def _frenet_normal(params, transforms, profiles, data, **kwargs):
    normal = cross(data["x_s"], cross(data["x_ss"], data["x_s"]))
    data["frenet_normal"] = normal / jnp.linalg.norm(normal, axis=-1)[:, None]
    return data


@register_compute_fun(
    name="frenet_binormal",
    label="\\mathbf{B}_{\\mathrm{Frenet-Serret}}",
    units="~",
    units_long="None",
    description="Binormal unit vector to curve in Frenet-Serret frame",
    dim=3,
    params=["rotmat"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["frenet_tangent", "frenet_normal"],
    parameterization="desc.geometry.core.Curve",
)
def _frenet_binormal(params, transforms, profiles, data, **kwargs):
    data["frenet_binormal"] = cross(
        data["frenet_tangent"], data["frenet_normal"]
    ) * jnp.linalg.det(params["rotmat"].reshape((3, 3)))
    return data


@register_compute_fun(
    name="double_reflection_rmf",
    label="\\mathbf{T}_{\\mathrm{Bishop}}",
    units="~",
    units_long="None",
    description="Tangent unit vector to curve in Bishop frame",
    dim=3,
    params=[],
    transforms={"nfp": [], "sym": []},
    profiles=[],
    coordinates="s",
    data=["x", "frenet_tangent", "phi"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
)
def _double_reflection_rmf(params, transforms, profiles, data, **kwargs):
    # chord/reflection math below is only valid in true Cartesian coordinates,
    # not rpz components -- convert in, and back out, explicitly
    x = rpz2xyz(data["x"])
    T = rpz2xyz_vec(data["frenet_tangent"], phi=data["phi"])
    N = _double_reflection(x, T, _seed_normal(x[0], T[0]))
    B = jnp.cross(T, N)  # complete the right-handed frame

    data["double_reflection_rmf"] = (T, N, B)  # Cartesian; converted back by the caller
    return data


def _seed_normal(x0, T0):
    """R-hat at x0, projected orthogonal to the tangent T0 (xyz)."""
    normal0 = jnp.array([x0[0], x0[1], 0.0])
    normal0 = normal0 - (normal0 @ T0) * T0
    return normal0 / jnp.linalg.norm(normal0)


def _double_reflection(x, T, normal0):
    """Propagate a rotation-minimizing normal along points x with tangents T (xyz)."""

    def _step(N_i, step_in):
        x_i, x_ip1, T_i, T_ip1 = step_in
        v1 = x_ip1 - x_i  # chord to the next point
        c1 = v1 @ v1
        r_L = N_i - (2 / c1) * (v1 @ N_i) * v1  # reflect N through the chord's bisector
        t_L = T_i - (2 / c1) * (v1 @ T_i) * v1  # reflect T the same way

        v2 = T_ip1 - t_L  # mismatch between reflected and actual tangent
        c2 = v2 @ v2
        N_ip1 = jnp.where(
            c2 > jnp.finfo(jnp.float64).eps,
            r_L - (2 / c2) * (v2 @ r_L) * v2,  # second reflection corrects the mismatch
            r_L,  # tangent already matches, skip the second reflection
        )
        N_ip1 = N_ip1 / jnp.linalg.norm(N_ip1)  # renormalize
        return N_ip1, N_ip1

    _, N_rest = scan(
        _step, normal0, (x[:-1], x[1:], T[:-1], T[1:])
    )  # propagate along the curve
    return jnp.concatenate([normal0[None], N_rest])  # prepend the seed vector


def _nurbs_curve_closed_bishop_frame(
    control_points_xyz, weights, degree, knots, nfp, s, n_prop=2000
):
    """Closed Bishop frame (T, N, B) in xyz at curve parameters s.

    Propagated on a fixed grid over one field period, closed with a twist linear
    in toroidal angle, then interpolated to s.
    """
    period = 2 * jnp.pi / nfp
    s_dense = jnp.linspace(0, period, n_prop)
    x_d, x_s_d, _, _ = _nurbs_curve(
        control_points_xyz, weights, degree, b_p_deriv3(s_dense, degree, knots)
    )
    T_d = x_s_d / jnp.linalg.norm(x_s_d, axis=1, keepdims=True)
    N_d = _double_reflection(x_d, T_d, _seed_normal(x_d[0], T_d[0]))
    B_d = jnp.cross(T_d, N_d)

    # holonomy defect: angle between the propagated end frame and the rotated start
    c, sn = jnp.cos(period), jnp.sin(period)
    Rzp = jnp.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    rot_defect = jnp.arctan2(N_d[-1] @ (Rzp @ B_d[0]), N_d[-1] @ (Rzp @ N_d[0]))
    phi_d = jnp.unwrap(jnp.arctan2(x_d[:, 1], x_d[:, 0]))
    beta = -rot_defect * (phi_d - phi_d[0]) / period
    N_corr = jnp.cos(beta)[:, None] * N_d + jnp.sin(beta)[:, None] * B_d

    k = jnp.floor(s / period)
    N_q = interp1d(s - k * period, s_dense, N_corr, method="cubic", extrap=True)
    # rotate into each query point's own field period
    ca, sa = jnp.cos(k * period), jnp.sin(k * period)
    N_q = jnp.stack(
        [ca * N_q[:, 0] - sa * N_q[:, 1], sa * N_q[:, 0] + ca * N_q[:, 1], N_q[:, 2]],
        axis=1,
    )

    _, x_s, _, _ = _nurbs_curve(
        control_points_xyz, weights, degree, b_p_deriv3(s, degree, knots)
    )
    T = x_s / jnp.linalg.norm(x_s, axis=1, keepdims=True)
    N = N_q - jnp.sum(N_q * T, axis=1, keepdims=True) * T
    N = N / jnp.linalg.norm(N, axis=1, keepdims=True)
    return T, N, jnp.cross(T, N)


@register_compute_fun(
    name="closed_bishop_frame",
    label="{\\mathbf{T}, \\mathbf{N}, \\mathbf{B}}_{\\mathrm{Bishop, closed}}",
    units="~",
    units_long="None",
    description="Closed Bishop/rotation minimizing frame",
    dim=3,
    params=[],
    transforms={"nfp": [], "degree": []},
    profiles=[],
    coordinates="s",
    data=["s", "phi", "full_control_net", "full_weights", "knots"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
)
def _closed_bishop_frame(params, transforms, profiles, data, **kwargs):
    T, N, B = _nurbs_curve_closed_bishop_frame(
        data["full_control_net"],
        data["full_weights"],
        transforms["degree"],
        data["knots"],
        transforms["nfp"],
        data["s"],
    )
    # convert back to rpz components before storing, matching the usual convention
    data["closed_bishop_frame"] = (
        xyz2rpz_vec(T, phi=data["phi"]),
        xyz2rpz_vec(N, phi=data["phi"]),
        xyz2rpz_vec(B, phi=data["phi"]),
    )
    return data


@register_compute_fun(
    name="closed_bishop_tangent",
    label="\\mathbf{T}_{\\mathrm{Bishop, closed}}",
    units="~",
    units_long="None",
    description="Tangent unit vector in the closed Bishop frame",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["closed_bishop_frame"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
)
def _closed_bishop_tangent(params, transforms, profiles, data, **kwargs):
    data["closed_bishop_tangent"] = data["closed_bishop_frame"][
        0
    ]  # T is the first element
    return data


@register_compute_fun(
    name="closed_bishop_normal",
    label="\\mathbf{N}_{\\mathrm{Bishop, closed}}",
    units="~",
    units_long="None",
    description="Rotation-minimizing normal unit vector in the closed Bishop frame",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["closed_bishop_frame"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
)
def _closed_bishop_normal(params, transforms, profiles, data, **kwargs):
    data["closed_bishop_normal"] = data["closed_bishop_frame"][
        1
    ]  # N is the second element
    return data


@register_compute_fun(
    name="closed_bishop_binormal",
    label="\\mathbf{B}_{\\mathrm{Bishop, closed}}",
    units="~",
    units_long="None",
    description="Binormal unit vector in the closed Bishop frame",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["closed_bishop_frame"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
)
def _closed_bishop_binormal(params, transforms, profiles, data, **kwargs):
    data["closed_bishop_binormal"] = data["closed_bishop_frame"][
        2
    ]  # B is the third element
    return data


@register_compute_fun(
    name="curvature",
    label="\\kappa",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Scalar curvature of the curve, with the sign denoting the convexity/"
    + "concavity relative to the center of the curve (a circle has positive curvature)",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["center", "x", "x_s", "x_ss", "frenet_normal", "phi"],
    parameterization="desc.geometry.core.Curve",
)
def _curvature(params, transforms, profiles, data, **kwargs):
    # magnitude of curvature
    dxn = jnp.linalg.norm(data["x_s"], axis=-1)[:, jnp.newaxis]
    curvature = jnp.linalg.norm(cross(data["x_s"], data["x_ss"]) / dxn**3, axis=-1)
    # sign of curvature (positive = "convex", negative = "concave")
    r = rpz2xyz(data["center"]) - rpz2xyz(data["x"])
    r = xyz2rpz_vec(r, phi=data["phi"])
    data["curvature"] = curvature * sign(dot(r, data["frenet_normal"]))
    return data


@register_compute_fun(
    name="torsion",
    label="\\tau",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Scalar torsion of the curve",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x_s", "x_ss", "x_sss"],
    parameterization="desc.geometry.core.Curve",
)
def _torsion(params, transforms, profiles, data, **kwargs):
    dxd2x = cross(data["x_s"], data["x_ss"])
    data["torsion"] = dot(dxd2x, data["x_sss"]) / jnp.linalg.norm(dxd2x, axis=-1) ** 2
    return data


@register_compute_fun(
    name="length",
    label="L",
    units="m",
    units_long="meters",
    description="Length of the curve",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["ds", "x_s"],
    parameterization=["desc.geometry.core.Curve"],
)
def _length(params, transforms, profiles, data, **kwargs):
    T = jnp.linalg.norm(data["x_s"], axis=-1)
    # this is equivalent to jnp.trapz(T, s) for a closed curve,
    # but also works if grid.endpoint is False
    data["length"] = jnp.sum(T * data["ds"])
    return data


@register_compute_fun(
    name="length",
    label="L",
    units="m",
    units_long="meters",
    description="Length of the curve",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["ds", "x", "x_s"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    method="Interpolation type, Default 'cubic'. See SplineXYZCurve docs for options.",
)
def _length_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    if kwargs["method"] == "nearest":  # cannot use derivative method as deriv=0
        coords = data["x"]
        if kwargs.get("basis", "rpz").lower() == "rpz":
            coords = rpz2xyz(coords)
        # ensure curve is closed
        # if it's already closed this doesn't add any length since ds will be zero
        coords = jnp.concatenate([coords, coords[:1]])
        X = coords[:, 0]
        Y = coords[:, 1]
        Z = coords[:, 2]
        lengths = jnp.sqrt(jnp.diff(X) ** 2 + jnp.diff(Y) ** 2 + jnp.diff(Z) ** 2)
        data["length"] = jnp.sum(lengths)
    else:
        T = jnp.linalg.norm(data["x_s"], axis=-1)
        # this is equivalent to jnp.trapz(T, s) for a closed curve
        # but also works if grid.endpoint is False
        data["length"] = jnp.sum(T * data["ds"])
    return data


def _nurbs_curve_control_net(R, phi, Z, sym, nfp):
    """Mirror and tile raw (R, phi, Z) control points to the full device, in xyz."""
    if not sym:
        raise NotImplementedError
    r_ctrl_1fp = jnp.append(R, R[-2::-1])[:-1]
    z_ctrl_1fp = jnp.append(Z, -Z[-2::-1])[:-1]
    phi_ctrl_1fp = jnp.append(phi, (2 * jnp.pi / nfp) - phi[-2::-1])[:-1]
    R = jnp.tile(r_ctrl_1fp, nfp)
    Z = jnp.tile(z_ctrl_1fp, nfp)
    phi = jnp.concatenate([phi_ctrl_1fp + n * 2 * jnp.pi / nfp for n in range(nfp)])
    return rpz2xyz(jnp.stack([R, phi, Z], axis=1))


def _nurbs_curve_weights(W, sym, nfp):
    """Mirror and tile raw control point weights to the full device."""
    if not sym:
        raise NotImplementedError
    return jnp.tile(jnp.append(W, W[-2::-1])[:-1], nfp)


def _nurbs_curve_knots(control_points_xyz, degree, knot_parametrization):
    """Periodic knot vector for the full control net."""
    n = len(control_points_xyz) - 1
    if knot_parametrization == "uniform":
        return uniform_knots(n, degree, domain=2 * jnp.pi)
    if knot_parametrization == "chord":
        return chord_length_knots(control_points_xyz, degree, domain=2 * jnp.pi)


def _nurbs_curve(control_points_xyz, weights, degree, basis):
    """Periodic NURBS curve and its first three s-derivatives, in xyz.

    ``basis`` is the (b, b_s, b_ss, b_sss) tuple returned by ``b_p_deriv3``.
    """
    p = degree
    padded = jnp.concatenate(
        [control_points_xyz[-p:], control_points_xyz, control_points_xyz[:p]], axis=0
    )
    padded_w = jnp.concatenate([weights[-p:], weights, weights[:p]])
    pw = padded * padded_w[:, None]

    b, b_s, b_ss, b_sss = basis
    num, den = b @ pw, b @ padded_w
    num_s, den_s = b_s @ pw, b_s @ padded_w
    num_ss, den_ss = b_ss @ pw, b_ss @ padded_w
    num_sss, den_sss = b_sss @ pw, b_sss @ padded_w

    c = num / den[:, None]
    c_s = (num_s - c * den_s[:, None]) / den[:, None]
    c_ss = (num_ss - 2 * c_s * den_s[:, None] - c * den_ss[:, None]) / den[:, None]
    c_sss = (
        num_sss
        - 3 * c_ss * den_s[:, None]
        - 3 * c_s * den_ss[:, None]
        - c * den_sss[:, None]
    ) / den[:, None]
    return c, c_s, c_ss, c_sss


@register_compute_fun(
    name="phi2s",
    label="s(\\varphi)",
    units="~",
    units_long="None",
    description="Curve parameter s corresponding to a "
    + "given physical toroidal angle phi",
    dim=0,
    params=[],
    transforms={"degree": []},
    profiles=[],
    coordinates="",
    data=["full_control_net", "full_weights", "knots"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
    public=False,
    phi="ndarray: target phi value(s) to invert for s. Required.",
)
def _phi2s_NurbsRPZCurve(params, transforms, profiles, data, **kwargs):
    data["phi2s"] = _nurbs_curve_phi2s(
        data["full_control_net"],
        data["full_weights"],
        transforms["degree"],
        data["knots"],
        jnp.atleast_1d(kwargs["phi"]),
    )
    return data


def _nurbs_curve_phi2s(control_points_xyz, weights, degree, knots, phi_target):
    """Curve parameter s at which the curve has toroidal angle phi_target."""

    def _phi_and_dphi_ds(s):
        basis = b_p_deriv3(jnp.atleast_1d(s), degree, knots)
        xcart, xcart_s, _, _ = _nurbs_curve(control_points_xyz, weights, degree, basis)
        x, y = xcart[0, 0], xcart[0, 1]
        x_s, y_s = xcart_s[0, 0], xcart_s[0, 1]
        phi = jnp.arctan2(y, x)  # wraps to (-pi, pi]
        dphi_ds = (x * y_s - y * x_s) / (x**2 + y**2)
        return phi, dphi_ds

    def _residual(s, target):
        phi, _ = _phi_and_dphi_ds(s)
        return ((phi - target + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    def _jac(s, target):
        _, dphi_ds = _phi_and_dphi_ds(s)
        return dphi_ds

    s0 = phi_target  # initial guess: s ~= phi works for near-circular axes
    return vmap(
        lambda s0_i, target: root_scalar(
            _residual, s0_i, jac=_jac, args=(target,), tol=1e-12
        )
    )(s0, phi_target)


@register_compute_fun(
    name="full_control_net",
    label="X_i, Y_i, Z_i",
    units="m",
    units_long="meters",
    description=("Full Cartesian control net (with wrapping for stellarator symmetry)"),
    dim=0,
    params=["R", "phi", "Z"],
    transforms={"degree": [], "sym": [], "nfp": [], "knot_parametrization": []},
    profiles=[],
    coordinates="",
    data=[],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
    public=False,
)
def _full_control_net(params, transforms, profiles, data, **kwargs):
    data["full_control_net"] = _nurbs_curve_control_net(
        params["R"], params["phi"], params["Z"], transforms["sym"], transforms["nfp"]
    )
    return data


@register_compute_fun(
    name="full_weights",
    label="w_i",
    units="~",
    units_long="not applicable",
    description="Full NURBS weight vector (with wrapping for stellarator symmetry)",
    dim=1,
    params=["W"],
    transforms={"sym": [], "nfp": []},
    profiles=[],
    coordinates="",
    data=[],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
    public=False,
)
def _full_weights(params, transforms, profiles, data, **kwargs):
    data["full_weights"] = _nurbs_curve_weights(
        params["W"], transforms["sym"], transforms["nfp"]
    )
    return data


@register_compute_fun(
    name="knots",
    label="s_i",
    units="~",
    units_long="not applicable",
    description="get knots for a nurbs curve",
    dim=0,
    params=[],
    transforms={"degree": [], "knot_parametrization": []},
    profiles=[],
    coordinates="",
    data=["full_control_net"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
    public=False,
)
def _knots(params, transforms, profiles, data, **kwargs):
    data["knots"] = _nurbs_curve_knots(
        data["full_control_net"],
        transforms["degree"],
        transforms["knot_parametrization"],
    )
    return data


@register_compute_fun(
    name="spline_derivs_till_3",
    label="B_{i,p},B'_{i,p},B''_{i,p},B'''_{i,p}",
    units="~",
    units_long="not applicable",
    description="helper function for nurbs derivatives",
    dim=1,
    params=[],
    transforms={"degree": []},
    profiles=[],
    coordinates="s",
    data=["s", "knots"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
    public=False,
)
def _spline_derivs_till_3(params, transforms, profiles, data, **kwargs):
    # evaluate spline on native parameter grid
    sq = data["s"]
    degree = transforms["degree"]
    knots = data["knots"]

    b, b_s, b_ss, b_sss = b_p_deriv3(sq, degree, knots)

    data["spline_derivs_till_3"] = (b, b_s, b_ss, b_sss)
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["rotmat", "shift"],
    transforms={"degree": []},
    profiles=[],
    coordinates="s",
    data=["full_control_net", "full_weights", "spline_derivs_till_3"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
)
def _x_NurbsRPZCurve(params, transforms, profiles, data, **kwargs):
    # TODO: check if sq is within [0, 2pi]?
    c, _, _, _ = _nurbs_curve(
        data["full_control_net"],
        data["full_weights"],
        transforms["degree"],
        data["spline_derivs_till_3"],
    )
    data["x"] = xyz2rpz(c)
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["rotmat", "shift"],
    transforms={"degree": []},
    profiles=[],
    coordinates="s",
    data=["full_control_net", "full_weights", "spline_derivs_till_3", "phi"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
)
def _x_s_NurbsRPZCurve(params, transforms, profiles, data, **kwargs):
    # TODO: check if sq is within [0, 2pi]?
    _, c_s, _, _ = _nurbs_curve(
        data["full_control_net"],
        data["full_weights"],
        transforms["degree"],
        data["spline_derivs_till_3"],
    )
    data["x_s"] = xyz2rpz_vec(c_s, phi=data["phi"])
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["rotmat", "shift"],
    transforms={"degree": []},
    profiles=[],
    coordinates="s",
    data=["full_control_net", "full_weights", "spline_derivs_till_3", "phi"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
)
def _x_ss_NurbsRPZCurve(params, transforms, profiles, data, **kwargs):
    # TODO: check if sq is within [0, 2pi]?
    _, _, c_ss, _ = _nurbs_curve(
        data["full_control_net"],
        data["full_weights"],
        transforms["degree"],
        data["spline_derivs_till_3"],
    )
    data["x_ss"] = xyz2rpz_vec(c_ss, phi=data["phi"])
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["rotmat", "shift"],
    transforms={"degree": []},
    profiles=[],
    coordinates="s",
    data=["full_control_net", "full_weights", "spline_derivs_till_3", "phi"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
)
def _x_sss_NurbsRPZCurve(params, transforms, profiles, data, **kwargs):
    # TODO: check if sq is within [0, 2pi]?
    _, _, _, c_sss = _nurbs_curve(
        data["full_control_net"],
        data["full_weights"],
        transforms["degree"],
        data["spline_derivs_till_3"],
    )
    data["x_sss"] = xyz2rpz_vec(c_sss, phi=data["phi"])
    return data


@register_compute_fun(
    name="center",
    label="\\langle\\mathbf{x}\\rangle",
    units="m",
    units_long="meters",
    description="Centroid of the curve",
    dim=3,
    params=["rotmat", "shift"],
    transforms={"degree": []},
    profiles=[],
    coordinates="s",
    data=["full_control_net", "spline_derivs_till_3", "x"],
    parameterization="desc.geometry.curve.NurbsRPZCurve",
)
def _center_NurbsRPZCurve(params, transforms, profiles, data, **kwargs):
    # center is average of xyz control points
    xyz = data["full_control_net"]
    center = jnp.mean(xyz, axis=0)
    # displacement and rotation
    center = jnp.matmul(center, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert to rpz
    data["center"] = xyz2rpz(center) * jnp.ones_like(data["x"])
    return data
