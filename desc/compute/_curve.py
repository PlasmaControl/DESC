import numpy as np
from interpax import interp1d

from desc.backend import jnp, sign

from ..utils import cross, dot, safenormalize
from .data_index import register_compute_fun
from .geom_utils import rotation_matrix, rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec


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
        "desc.geometry.curve.C0FourierPlanarCurve",
    ],
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _center_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
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


def get_closest(arr, val):
    """Get closest index to value in array."""
    arr = jnp.asarray(arr)
    return (jnp.argmin(jnp.abs(arr - val))).astype(int)


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve",
    dim=3,
    params=["r_n", "center", "normal", "rotmat", "shift"],
    transforms={"r": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.C0FourierPlanarCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
    len_rn=1,
)
def _x_C0FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        center = rpz2xyz(params["center"])
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        center = params["center"]
        normal = params["normal"]

    len_rn = kwargs.get("len_rn", 1)
    s = jnp.asarray(data["s"])

    # 0 to pi
    mid_point = jnp.pi
    istart = 0
    istop = get_closest(s, mid_point)
    r1 = transforms["r"].transform(jnp.atleast_1d(params["r_n"][:len_rn]), dz=0)
    Z1 = jnp.zeros_like(r1)
    X1 = r1 * jnp.cos(data["s"])
    Y1 = r1 * jnp.sin(data["s"])
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis1 = cross(Zaxis, normal)
    angle1 = jnp.arccos(dot(Zaxis, safenormalize(normal)))
    A1 = rotation_matrix(axis=axis1, angle=angle1)
    coords1 = jnp.array([X1, Y1, Z1]).T  # full coil still
    coords1 = jnp.matmul(coords1, A1.T) + center
    coords1 = jnp.matmul(coords1, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert back to rpz

    # now take half the coil
    # TODO: nothing lower than 0 but need to generalize
    X1_half = jnp.where(s >= s[istop], coords1[:, 0], 0)
    Y1_half = jnp.where(s >= s[istop], coords1[:, 1], 0)
    Z1_half = jnp.where(s >= s[istop], coords1[:, 2], 0)

    # pi to 2pi
    istart = get_closest(s, mid_point)
    istop = -1
    r2 = transforms["r"].transform(jnp.atleast_1d(params["r_n"][len_rn:]), dz=0)
    Z2 = jnp.zeros_like(r2)
    X2 = r2 * jnp.cos(data["s"])
    Y2 = r2 * jnp.sin(data["s"])
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis2 = cross(Zaxis, normal)
    angle2 = jnp.arccos(dot(Zaxis, safenormalize(normal)))
    A2 = rotation_matrix(axis=axis2, angle=angle2)
    coords2 = jnp.array([X2, Y2, Z2]).T  # full coil still
    coords2 = jnp.matmul(coords2, A2.T) + center
    coords2 = jnp.matmul(coords2, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert back to rpz

    # now take half the coil
    X2_half = jnp.where(s < s[istart], coords2[:, 0], 0)
    Y2_half = jnp.where(s < s[istart], coords2[:, 1], 0)
    Z2_half = jnp.where(s < s[istart], coords2[:, 2], 0)

    X = X1_half + X2_half
    Y = Y1_half + Y2_half
    Z = Z1_half + Z2_half

    coords = jnp.array([X, Y, Z]).T

    # # convert back to rpz
    coords = xyz2rpz(coords)
    data["x"] = coords

    return data


@register_compute_fun(
    name="x_s",
    label="\\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["r_n", "center", "normal", "rotmat", "shift"],
    transforms={"r": [[0, 0, 0], [0, 0, 1]]},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.C0FourierPlanarCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_s_C0FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]

    len_rn = kwargs.get("len_rn", 1)
    s = jnp.asarray(data["s"])

    istart = 0
    istop = get_closest(data["s"], np.pi)
    r1 = transforms["r"].transform(params["r_n"][:len_rn], dz=0)
    dr1 = transforms["r"].transform(params["r_n"][:len_rn], dz=1)
    dX1 = dr1 * jnp.cos(data["s"]) - r1 * jnp.sin(data["s"])
    dY1 = dr1 * jnp.sin(data["s"]) + r1 * jnp.cos(data["s"])
    dZ1 = jnp.zeros_like(dX1)

    coords1 = jnp.array([dX1, dY1, dZ1]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis1 = cross(Zaxis, normal)
    angle1 = jnp.arccos(dot(Zaxis, safenormalize(normal)))
    A1 = rotation_matrix(axis=axis1, angle=angle1)
    coords1 = jnp.array([dX1, dY1, dZ1]).T
    coords1 = jnp.matmul(coords1, A1.T)
    coords1 = jnp.matmul(coords1, params["rotmat"].reshape((3, 3)).T)

    dX1_half = jnp.where(s >= s[istop], coords1[:, 0], 0)
    dY1_half = jnp.where(s >= s[istop], coords1[:, 1], 0)
    dZ1_half = jnp.where(s >= s[istop], coords1[:, 2], 0)

    # pi to 2pi
    istart = get_closest(data["s"], np.pi)
    istop = -1
    r2 = transforms["r"].transform(params["r_n"][len_rn:], dz=0)
    dr2 = transforms["r"].transform(params["r_n"][len_rn:], dz=1)
    dX2 = dr2 * jnp.cos(data["s"]) - r2 * jnp.sin(data["s"])
    dY2 = dr2 * jnp.sin(data["s"]) + r2 * jnp.cos(data["s"])
    dZ2 = jnp.zeros_like(dX2)

    coords2 = jnp.array([dX2, dY2, dZ2]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis2 = cross(Zaxis, normal)
    angle2 = jnp.arccos(dot(Zaxis, safenormalize(normal)))
    A2 = rotation_matrix(axis=axis2, angle=angle2)
    coords2 = jnp.array([dX2, dY2, dZ2]).T
    coords2 = jnp.matmul(coords2, A2.T)
    coords2 = jnp.matmul(coords2, params["rotmat"].reshape((3, 3)).T)

    dX2_half = jnp.where(s < s[istart], coords2[:, 0], 0)
    dY2_half = jnp.where(s < s[istart], coords2[:, 1], 0)
    dZ2_half = jnp.where(s < s[istart], coords2[:, 2], 0)

    dX = dX1_half + dX2_half
    dY = dY1_half + dY2_half
    dZ = dZ1_half + dZ2_half
    # convert back to rpz
    coords = jnp.array([dX, dY, dZ]).T
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_s"] = coords
    return data


@register_compute_fun(
    name="x_ss",
    label="\\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve",
    dim=3,
    params=["r_n", "center", "normal", "rotmat", "shift"],
    transforms={"r": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.C0FourierPlanarCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_ss_C0FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]

    len_rn = kwargs.get("len_rn", 1)

    s = jnp.asarray(data["s"])
    mid_point = jnp.pi

    # 0 to pi
    istart = 0
    istop = get_closest(s, mid_point)

    r1 = transforms["r"].transform(params["r_n"][:len_rn], dz=0)
    dr1 = transforms["r"].transform(params["r_n"][:len_rn], dz=1)
    d2r1 = transforms["r"].transform(params["r_n"][:len_rn], dz=2)
    d2X1 = (
        d2r1 * jnp.cos(data["s"])
        - 2 * dr1 * jnp.sin(data["s"])
        - r1 * jnp.cos(data["s"])
    )
    d2Y1 = (
        d2r1 * jnp.sin(data["s"])
        + 2 * dr1 * jnp.cos(data["s"])
        - r1 * jnp.sin(data["s"])
    )
    d2Z1 = jnp.zeros_like(d2X1)
    coords1 = jnp.array([d2X1, d2Y1, d2Z1]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis1 = cross(Zaxis, normal)
    angle1 = jnp.arccos(dot(Zaxis, safenormalize(normal)))
    A1 = rotation_matrix(axis=axis1, angle=angle1)
    coords1 = jnp.matmul(coords1, A1.T)
    coords1 = jnp.matmul(coords1, params["rotmat"].reshape((3, 3)).T)

    d2X1_half = jnp.where(s >= s[istop], coords1[:, 0], 0)
    d2Y1_half = jnp.where(s >= s[istop], coords1[:, 1], 0)
    d2Z1_half = jnp.where(s >= s[istop], coords1[:, 2], 0)

    # pi to 2pi
    istart = get_closest(data["s"], np.pi)
    istop = -1
    r2 = transforms["r"].transform(params["r_n"][len_rn:], dz=0)
    dr2 = transforms["r"].transform(params["r_n"][len_rn:], dz=1)
    d2r2 = transforms["r"].transform(params["r_n"][len_rn:], dz=2)
    d2X2 = (
        d2r2 * jnp.cos(data["s"])
        - 2 * dr2 * jnp.sin(data["s"])
        - r2 * jnp.cos(data["s"])
    )
    d2Y2 = (
        d2r2 * jnp.sin(data["s"])
        + 2 * dr2 * jnp.cos(data["s"])
        - r2 * jnp.sin(data["s"])
    )
    d2Z2 = jnp.zeros_like(d2X2)
    coords2 = jnp.array([d2X2, d2Y2, d2Z2]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis2 = cross(Zaxis, normal)
    angle2 = jnp.arccos(dot(Zaxis, safenormalize(normal)))
    A2 = rotation_matrix(axis=axis2, angle=angle2)
    coords2 = jnp.matmul(coords2, A2.T)
    coords2 = jnp.matmul(coords2, params["rotmat"].reshape((3, 3)).T)
    d2X2_half = jnp.where(s < s[istart], coords2[:, 0], 0)
    d2Y2_half = jnp.where(s < s[istart], coords2[:, 1], 0)
    d2Z2_half = jnp.where(s < s[istart], coords2[:, 2], 0)

    d2X = d2X1_half + d2X2_half
    d2Y = d2Y1_half + d2Y2_half
    d2Z = d2Z1_half + d2Z2_half
    # convert back to rpz
    coords = jnp.array([d2X, d2Y, d2Z]).T
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_ss"] = coords
    return data


@register_compute_fun(
    name="x_sss",
    label="\\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve",
    dim=3,
    params=["r_n", "center", "normal", "rotmat", "shift"],
    transforms={"r": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.C0FourierPlanarCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_sss_C0FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        center = rpz2xyz(params["center"])
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        center = params["center"]
        normal = params["normal"]

    params["r_n"]

    # enforce that all curves have same resolution
    r = transforms["r"].transform(params["r_n"], dz=0)
    Z = jnp.zeros_like(r)

    # 0 to pi
    X1 = r[0] * jnp.cos(data["s"])
    X1 = jnp.where(data["s"] > jnp.pi, X1, 0)
    Y1 = r[0] * jnp.sin(data["s"])
    Y1 = jnp.where(data["s"] > jnp.pi, Y1, 0)

    # pi to 2pi
    X2 = r[1] * jnp.cos(data["s"])
    X2 = jnp.where(data["s"] < jnp.pi, X2, 0)
    Y2 = r[1] * jnp.sin(data["s"])
    Y2 = jnp.where(data["s"] < jnp.pi, Y2, 0)

    X = X1 + X2
    Y = Y1 + Y2

    coords = jnp.array([X, Y, Z]).T

    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis = cross(Zaxis, normal)
    angle = jnp.arccos(dot(Zaxis, safenormalize(normal)))
    A = rotation_matrix(axis=axis, angle=angle)
    coords = jnp.matmul(coords, A.T) + center
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert back to rpz
    coords = xyz2rpz(coords)
    data["x"] = coords


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve",
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
    axis = cross(Zaxis, normal)
    angle = jnp.arccos(dot(Zaxis, safenormalize(normal)))
    A = rotation_matrix(axis=axis, angle=angle)
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
    axis = cross(Zaxis, normal)
    angle = jnp.arccos(dot(Zaxis, safenormalize(normal)))
    A = rotation_matrix(axis=axis, angle=angle)
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
    axis = cross(Zaxis, normal)
    angle = jnp.arccos(dot(Zaxis, safenormalize(normal)))
    A = rotation_matrix(axis=axis, angle=angle)
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
    axis = cross(Zaxis, normal)
    angle = jnp.arccos(dot(Zaxis, safenormalize(normal)))
    A = rotation_matrix(axis=axis, angle=angle)
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
    units="m",
    units_long="meters",
    description="Position vector along curve",
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
    units="m",
    units_long="meters",
    description="Position vector along curve",
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
    transforms={
        "X": [[0, 0, 1]],
        "Y": [[0, 0, 1]],
        "Z": [[0, 0, 1]],
    },
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
    transforms={
        "X": [[0, 0, 0], [0, 0, 2]],
        "Y": [[0, 0, 0], [0, 0, 2]],
        "Z": [[0, 0, 2]],
    },
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
    transforms={
        "X": [[0, 0, 0], [0, 0, 3]],
        "Y": [[0, 0, 0], [0, 0, 3]],
        "Z": [[0, 0, 3]],
    },
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
    units="m",
    units_long="meters",
    description="Position vector along curve",
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
    data=["center", "x", "x_s", "x_ss", "frenet_normal"],
    parameterization="desc.geometry.core.Curve",
)
def _curvature(params, transforms, profiles, data, **kwargs):
    # magnitude of curvature
    dxn = jnp.linalg.norm(data["x_s"], axis=-1)[:, jnp.newaxis]
    curvature = jnp.linalg.norm(cross(data["x_s"], data["x_ss"]) / dxn**3, axis=-1)
    # sign of curvature (positive = "convex", negative = "concave")
    r = data["center"] - data["x"]
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
