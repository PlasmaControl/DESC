from interpax import interp1d

from desc.backend import fori_loop, jnp

from .data_index import register_compute_fun
from .geom_utils import rotation_matrix, rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from .utils import cross, dot, safenormalize


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
    description="Spacing of curve parameter",
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
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _X_curve(params, transforms, profiles, data, **kwargs):
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
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _Y_Curve(params, transforms, profiles, data, **kwargs):
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
    description="Cylindrical radial position along curve",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _R_Curve(params, transforms, profiles, data, **kwargs):
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
    description="Toroidal phi position along curve",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _phi_Curve(params, transforms, profiles, data, **kwargs):
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
    name="x",
    label="\\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve",
    dim=3,
    params=["r_n", "center", "normal", "rotmat", "shift"],
    transforms={
        "r": [[0, 0, 0]],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _x_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    # create planar curve at Z==0
    r = transforms["r"].transform(params["r_n"], dz=0)
    Z = jnp.zeros_like(r)
    X = r * jnp.cos(data["s"])
    Y = r * jnp.sin(data["s"])
    coords = jnp.array([X, Y, Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis = cross(Zaxis, params["normal"])
    angle = jnp.arccos(dot(Zaxis, safenormalize(params["normal"])))
    A = rotation_matrix(axis=axis, angle=angle)
    coords = jnp.matmul(coords, A.T) + params["center"]
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    if kwargs.get("basis", "rpz").lower() == "rpz":
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
    params=["r_n", "center", "normal", "rotmat", "shift"],
    transforms={
        "r": [[0, 0, 0], [0, 0, 1]],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _x_s_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    r = transforms["r"].transform(params["r_n"], dz=0)
    dr = transforms["r"].transform(params["r_n"], dz=1)
    dX = dr * jnp.cos(data["s"]) - r * jnp.sin(data["s"])
    dY = dr * jnp.sin(data["s"]) + r * jnp.cos(data["s"])
    dZ = jnp.zeros_like(dX)
    coords = jnp.array([dX, dY, dZ]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis = cross(Zaxis, params["normal"])
    angle = jnp.arccos(dot(Zaxis, safenormalize(params["normal"])))
    A = rotation_matrix(axis=axis, angle=angle)
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    if kwargs.get("basis", "rpz").lower() == "rpz":
        X = r * jnp.cos(data["s"])
        Y = r * jnp.sin(data["s"])
        Z = jnp.zeros_like(X)
        xyzcoords = jnp.array([X, Y, Z]).T
        xyzcoords = jnp.matmul(xyzcoords, A.T) + params["center"]
        xyzcoords = (
            jnp.matmul(xyzcoords, params["rotmat"].reshape((3, 3)).T) + params["shift"]
        )
        x, y, z = xyzcoords.T
        coords = xyz2rpz_vec(coords, x=x, y=y)
    data["x_s"] = coords
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["r_n", "center", "normal", "rotmat", "shift"],
    transforms={
        "r": [[0, 0, 0], [0, 0, 1], [0, 0, 2]],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _x_ss_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
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
    axis = cross(Zaxis, params["normal"])
    angle = jnp.arccos(dot(Zaxis, safenormalize(params["normal"])))
    A = rotation_matrix(axis=axis, angle=angle)
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    if kwargs.get("basis", "rpz").lower() == "rpz":
        X = r * jnp.cos(data["s"])
        Y = r * jnp.sin(data["s"])
        Z = jnp.zeros_like(X)
        xyzcoords = jnp.array([X, Y, Z]).T
        xyzcoords = jnp.matmul(xyzcoords, A.T) + params["center"]
        xyzcoords = (
            jnp.matmul(xyzcoords, params["rotmat"].reshape((3, 3)).T) + params["shift"]
        )
        x, y, z = xyzcoords.T
        coords = xyz2rpz_vec(coords, x=x, y=y)
    data["x_ss"] = coords
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["r_n", "center", "normal", "rotmat", "shift"],
    transforms={
        "r": [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _x_sss_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
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
    axis = cross(Zaxis, params["normal"])
    angle = jnp.arccos(dot(Zaxis, safenormalize(params["normal"])))
    A = rotation_matrix(axis=axis, angle=angle)
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    if kwargs.get("basis", "rpz").lower() == "rpz":
        X = r * jnp.cos(data["s"])
        Y = r * jnp.sin(data["s"])
        Z = jnp.zeros_like(X)
        xyzcoords = jnp.array([X, Y, Z]).T
        xyzcoords = jnp.matmul(xyzcoords, A.T) + params["center"]
        xyzcoords = (
            jnp.matmul(xyzcoords, params["rotmat"].reshape((3, 3)).T) + params["shift"]
        )
        x, y, z = xyzcoords.T
        coords = xyz2rpz_vec(coords, x=x, y=y)
    data["x_sss"] = coords
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve",
    dim=3,
    params=["R_n", "Z_n", "rotmat", "shift"],
    transforms={
        "R": [[0, 0, 0]],
        "Z": [[0, 0, 0]],
        "grid": [],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierRZCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
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
    if kwargs.get("basis", "rpz").lower() == "rpz":
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
    transforms={
        "R": [[0, 0, 0], [0, 0, 1]],
        "Z": [[0, 0, 1]],
        "grid": [],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierRZCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _x_s_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_n"], dz=0)
    dR = transforms["R"].transform(params["R_n"], dz=1)
    dZ = transforms["Z"].transform(params["Z_n"], dz=1)
    dphi = R0
    coords = jnp.stack([dR, dphi, dZ], axis=1)
    # convert to xyz for displacement and rotation
    coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    if kwargs.get("basis", "rpz").lower() == "rpz":
        coords = xyz2rpz_vec(coords, phi=transforms["grid"].nodes[:, 2])
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
    transforms={
        "R": [[0, 0, 0], [0, 0, 1], [0, 0, 2]],
        "Z": [[0, 0, 2]],
        "grid": [],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierRZCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _x_ss_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_n"], dz=0)
    dR = transforms["R"].transform(params["R_n"], dz=1)
    d2R = transforms["R"].transform(params["R_n"], dz=2)
    d2Z = transforms["Z"].transform(params["Z_n"], dz=2)
    R = d2R - R0
    Z = d2Z
    # 2nd derivative wrt phi = 0
    phi = 2 * dR
    coords = jnp.stack([R, phi, Z], axis=1)
    # convert to xyz for displacement and rotation
    coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    if kwargs.get("basis", "rpz").lower() == "rpz":
        coords = xyz2rpz_vec(coords, phi=transforms["grid"].nodes[:, 2])
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
    data=[],
    parameterization="desc.geometry.curve.FourierRZCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _x_sss_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_n"], dz=0)
    dR = transforms["R"].transform(params["R_n"], dz=1)
    d2R = transforms["R"].transform(params["R_n"], dz=2)
    d3R = transforms["R"].transform(params["R_n"], dz=3)
    d3Z = transforms["Z"].transform(params["Z_n"], dz=3)
    R = d3R - 3 * dR
    Z = d3Z
    phi = 3 * d2R - R0
    coords = jnp.stack([R, phi, Z], axis=1)
    # convert to xyz for displacement and rotation
    coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    if kwargs.get("basis", "rpz").lower() == "rpz":
        coords = xyz2rpz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["x_sss"] = coords
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat", "shift"],
    transforms={
        "X": [[0, 0, 0]],
        "Y": [[0, 0, 0]],
        "Z": [[0, 0, 0]],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierXYZCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _x_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    X = transforms["X"].transform(params["X_n"], dz=0)
    Y = transforms["Y"].transform(params["Y_n"], dz=0)
    Z = transforms["Z"].transform(params["Z_n"], dz=0)
    coords = jnp.stack([X, Y, Z], axis=1)
    coords = (
        coords @ params["rotmat"].reshape((3, 3)).T + params["shift"][jnp.newaxis, :]
    )
    if kwargs.get("basis", "rpz").lower() == "rpz":
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
    params=["X_n", "Y_n", "Z_n", "rotmat", "shift"],
    transforms={
        "X": [[0, 0, 0], [0, 0, 1]],
        "Y": [[0, 0, 0], [0, 0, 1]],
        "Z": [[0, 0, 1]],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierXYZCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _x_s_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    dX = transforms["X"].transform(params["X_n"], dz=1)
    dY = transforms["Y"].transform(params["Y_n"], dz=1)
    dZ = transforms["Z"].transform(params["Z_n"], dz=1)
    coords = jnp.stack([dX, dY, dZ], axis=1)
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    if kwargs.get("basis", "rpz").lower() == "rpz":
        coords = xyz2rpz_vec(
            coords,
            x=transforms["X"].transform(params["X_n"]) + params["shift"][0],
            y=transforms["Y"].transform(params["Y_n"]) + params["shift"][1],
        )
    data["x_s"] = coords
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat", "shift"],
    transforms={
        "X": [[0, 0, 0], [0, 0, 2]],
        "Y": [[0, 0, 0], [0, 0, 2]],
        "Z": [[0, 0, 2]],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierXYZCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _x_ss_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    d2X = transforms["X"].transform(params["X_n"], dz=2)
    d2Y = transforms["Y"].transform(params["Y_n"], dz=2)
    d2Z = transforms["Z"].transform(params["Z_n"], dz=2)
    coords = jnp.stack([d2X, d2Y, d2Z], axis=1)
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    if kwargs.get("basis", "rpz").lower() == "rpz":
        coords = xyz2rpz_vec(
            coords,
            x=transforms["X"].transform(params["X_n"]) + params["shift"][0],
            y=transforms["Y"].transform(params["Y_n"]) + params["shift"][1],
        )
    data["x_ss"] = coords
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat", "shift"],
    transforms={
        "X": [[0, 0, 0], [0, 0, 3]],
        "Y": [[0, 0, 0], [0, 0, 3]],
        "Z": [[0, 0, 3]],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierXYZCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
)
def _x_sss_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    d3X = transforms["X"].transform(params["X_n"], dz=3)
    d3Y = transforms["Y"].transform(params["Y_n"], dz=3)
    d3Z = transforms["Z"].transform(params["Z_n"], dz=3)
    coords = jnp.stack([d3X, d3Y, d3Z], axis=1)
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    if kwargs.get("basis", "rpz").lower() == "rpz":
        coords = xyz2rpz_vec(
            coords,
            x=transforms["X"].transform(params["X_n"]) + params["shift"][0],
            y=transforms["Y"].transform(params["Y_n"]) + params["shift"][1],
        )
    data["x_sss"] = coords
    return data


def inner_body(f, knots, xq, derivative, method, period=None):
    """Interpolation for spline curves."""
    Xq = interp1d(
        xq,
        knots,
        f[0],
        method=method,
        derivative=derivative,
        period=period,
    )
    Yq = interp1d(
        xq,
        knots,
        f[1],
        method=method,
        derivative=derivative,
        period=period,
    )
    Zq = interp1d(
        xq,
        knots,
        f[2],
        method=method,
        derivative=derivative,
        period=period,
    )

    return [Xq, Yq, Zq]


def get_XYZ_in_interval(X, Y, Z, knots, istart, istop):
    """Get X, Y, and Z in given interval."""

    def get_interval(arr, knots, istart, istop):
        arr_temp = jnp.where(knots > knots[istop], arr[istop], arr)
        arr_temp = jnp.where(knots < knots[istart], arr[istart], arr_temp)

        return arr_temp

    X_in_interval = get_interval(X, knots, istart, istop)
    Y_in_interval = get_interval(Y, knots, istart, istop)
    Z_in_interval = get_interval(Z, knots, istart, istop)

    return [X_in_interval, Y_in_interval, Z_in_interval]


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve",
    dim=3,
    params=["X", "Y", "Z", "knots", "rotmat", "shift"],
    transforms={
        "intervals": [],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
    method="{'cubic', 'linear'}: Interpolation type, Default 'cubic'",
)
def _x_SplineXYZCurve(params, transforms, profiles, data, **kwargs):

    derivative = 0
    xq = data["s"]
    method = kwargs.get("method", "cubic")
    transforms["intervals"] = jnp.asarray(transforms["intervals"])
    is_discontinuous = len(transforms["intervals"][0])

    def body(i, fq):
        istart, istop = transforms["intervals"][i]
        # catch end-point
        istop = jnp.where(istop == 0, -1, istop)

        f_in_interval = get_XYZ_in_interval(X, Y, Z, knots, istart, istop)
        fq_temp = inner_body(
            f_in_interval,
            knots,
            xq,
            derivative,
            method,
        )

        xq_range = (xq >= knots[istart]) & (xq <= knots[istop])
        for i in range(3):
            fq = fq.at[i].set(jnp.where(xq_range, fq_temp[i], fq[i]))

        return fq

    if is_discontinuous:
        # manually add endpoint for discontinuous
        knots = jnp.append(params["knots"], params["knots"][0] + 2 * jnp.pi)
        X = jnp.append(params["X"], params["X"][0])
        Y = jnp.append(params["Y"], params["Y"][0])
        Z = jnp.append(params["Z"], params["Z"][0])
        # query points for Xq, Yq, Zq
        fq = jnp.zeros((3, len(xq)))
        fq = fori_loop(0, len(transforms["intervals"]), body, fq)
    else:
        # regular interpolation
        fq = inner_body(
            [params["X"], params["Y"], params["Z"]],
            params["knots"],
            xq,
            derivative,
            method=method,
            period=2 * jnp.pi,
        )
        fq = jnp.array(fq)

    coords = jnp.stack(fq, axis=1)
    coords = (
        coords @ params["rotmat"].reshape((3, 3)).T + params["shift"][jnp.newaxis, :]
    )
    if kwargs.get("basis", "rpz").lower() == "rpz":
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
    params=["X", "Y", "Z", "knots", "rotmat"],
    transforms={
        "intervals": [],
    },
    profiles=[],
    coordinates="s",
    data=["s", "x"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
    method="{'cubic', 'linear'}: Interpolation type, Default 'cubic'",
)
def _x_s_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    derivative = 1
    xq = data["s"]
    transforms["intervals"] = jnp.asarray(transforms["intervals"])
    method = kwargs.get("method", "cubic")

    is_discontinuous = len(transforms["intervals"][0])

    def body(i, fq):
        istart, istop = transforms["intervals"][i]
        # catch end-point
        istop = jnp.where(istop == 0, -1, istop)

        f_in_interval = get_XYZ_in_interval(X, Y, Z, knots, istart, istop)
        fq_temp = inner_body(
            f_in_interval,
            knots,
            xq,
            derivative,
            method,
        )

        xq_range = (xq >= knots[istart]) & (xq <= knots[istop])
        for i in range(3):
            fq = fq.at[i].set(jnp.where(xq_range, fq_temp[i], fq[i]))

        return fq

    if is_discontinuous:
        # manually add endpoint for discontinuous
        knots = jnp.append(params["knots"], params["knots"][0] + 2 * jnp.pi)
        X = jnp.append(params["X"], params["X"][0])
        Y = jnp.append(params["Y"], params["Y"][0])
        Z = jnp.append(params["Z"], params["Z"][0])
        # query points for Xq, Yq, Zq
        fq = jnp.zeros((3, len(xq)))
        fq = fori_loop(0, len(transforms["intervals"]), body, fq)
    else:
        # regular interpolation
        fq = inner_body(
            [params["X"], params["Y"], params["Z"]],
            params["knots"],
            xq,
            derivative,
            method=method,
            period=2 * jnp.pi,
        )
        fq = jnp.array(fq)

    coords_s = jnp.stack(fq, axis=1)
    coords_s = coords_s @ params["rotmat"].reshape((3, 3)).T

    if kwargs.get("basis", "rpz").lower() == "rpz":
        coords = data["x"]
        coords_s = xyz2rpz_vec(coords_s, phi=coords[:, 1])

    data["x_s"] = coords_s
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["X", "Y", "Z", "knots", "rotmat"],
    transforms={
        "intervals": [],
    },
    profiles=[],
    coordinates="s",
    data=["s", "x"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
    method="{'cubic', 'linear'}: Interpolation type, Default 'cubic'",
)
def _x_ss_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    derivative = 2
    xq = data["s"]
    transforms["intervals"] = jnp.asarray(transforms["intervals"])
    method = kwargs.get("method", "cubic")

    is_discontinuous = len(transforms["intervals"][0])

    def body(i, fq):
        istart, istop = transforms["intervals"][i]
        # catch end-point
        istop = jnp.where(istop == 0, -1, istop)

        f_in_interval = get_XYZ_in_interval(X, Y, Z, knots, istart, istop)
        fq_temp = inner_body(
            f_in_interval,
            knots,
            xq,
            derivative,
            method,
        )

        xq_range = (xq >= knots[istart]) & (xq <= knots[istop])
        for i in range(3):
            fq = fq.at[i].set(jnp.where(xq_range, fq_temp[i], fq[i]))

        return fq

    if is_discontinuous:
        # manually add endpoint for discontinuous
        knots = jnp.append(params["knots"], params["knots"][0] + 2 * jnp.pi)
        X = jnp.append(params["X"], params["X"][0])
        Y = jnp.append(params["Y"], params["Y"][0])
        Z = jnp.append(params["Z"], params["Z"][0])
        # query points for Xq, Yq, Zq
        fq = jnp.zeros((3, len(xq)))
        fq = fori_loop(0, len(transforms["intervals"]), body, fq)
    else:
        # regular interpolation
        fq = inner_body(
            [params["X"], params["Y"], params["Z"]],
            params["knots"],
            xq,
            derivative,
            method=method,
            period=2 * jnp.pi,
        )
        fq = jnp.array(fq)

    coords_ss = jnp.stack(fq, axis=1)
    coords_ss = coords_ss @ params["rotmat"].reshape((3, 3)).T

    if kwargs.get("basis", "rpz").lower() == "rpz":
        coords = data["x"]
        coords_ss = xyz2rpz_vec(coords_ss, phi=coords[:, 1])
    data["x_ss"] = coords_ss
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["X", "Y", "Z", "knots", "rotmat"],
    transforms={
        "intervals": [],
    },
    profiles=[],
    coordinates="s",
    data=["s", "x"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    basis="{'rpz', 'xyz'}: Basis for returned vectors, Default 'rpz'",
    method="{'cubic', 'linear'}: Interpolation type, Default 'cubic'",
)
def _x_sss_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    derivative = 3
    xq = data["s"]
    transforms["intervals"] = jnp.asarray(transforms["intervals"])
    method = kwargs.get("method", "cubic")

    is_discontinuous = len(transforms["intervals"][0])

    def body(i, fq):
        istart, istop = transforms["intervals"][i]
        # catch end-point
        istop = jnp.where(istop == 0, -1, istop)

        f_in_interval = get_XYZ_in_interval(X, Y, Z, knots, istart, istop)
        fq_temp = inner_body(
            f_in_interval,
            knots,
            xq,
            derivative,
            method,
        )

        xq_range = (xq >= knots[istart]) & (xq <= knots[istop])
        for i in range(3):
            fq = fq.at[i].set(jnp.where(xq_range, fq_temp[i], fq[i]))

        return fq

    if is_discontinuous:
        # manually add endpoint for discontinuous
        knots = jnp.append(params["knots"], params["knots"][0] + 2 * jnp.pi)
        X = jnp.append(params["X"], params["X"][0])
        Y = jnp.append(params["Y"], params["Y"][0])
        Z = jnp.append(params["Z"], params["Z"][0])
        # query points for Xq, Yq, Zq
        fq = jnp.zeros((3, len(xq)))
        fq = fori_loop(0, len(transforms["intervals"]), body, fq)
    else:
        # regular interpolation
        fq = inner_body(
            [params["X"], params["Y"], params["Z"]],
            params["knots"],
            xq,
            derivative,
            method=method,
            period=2 * jnp.pi,
        )
        fq = jnp.array(fq)

    coords_sss = jnp.stack(fq, axis=1)
    coords_sss = coords_sss @ params["rotmat"].reshape((3, 3)).T

    if kwargs.get("basis", "rpz").lower() == "rpz":
        coords = data["x"]
        coords_sss = xyz2rpz_vec(coords_sss, phi=coords[:, 1])
    data["x_sss"] = coords_sss

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
    data=["x_ss"],
    parameterization="desc.geometry.core.Curve",
)
def _frenet_normal(params, transforms, profiles, data, **kwargs):
    data["frenet_normal"] = (
        data["x_ss"] / jnp.linalg.norm(data["x_ss"], axis=-1)[:, None]
    )
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
    description="Scalar curvature of the curve",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x_s", "x_ss"],
    parameterization="desc.geometry.core.Curve",
)
def _curvature(params, transforms, profiles, data, **kwargs):
    dxn = jnp.linalg.norm(data["x_s"], axis=-1)[:, jnp.newaxis]
    data["curvature"] = jnp.linalg.norm(
        cross(data["x_s"], data["x_ss"]) / dxn**3, axis=-1
    )
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
    parameterization=[
        "desc.geometry.curve.FourierRZCurve",
        "desc.geometry.curve.FourierXYZCurve",
        "desc.geometry.curve.FourierPlanarCurve",
    ],
)
def _length(params, transforms, profiles, data, **kwargs):
    T = jnp.linalg.norm(data["x_s"], axis=-1)
    # this is equivalent to jnp.trapz(T, s) for a closed curve, but also works
    # if grid.endpoint is False
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
    transforms={"method": []},
    profiles=[],
    coordinates="",
    data=["ds", "x", "x_s"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
)
def _length_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    if transforms["method"] == "nearest":  # cannot use derivative method as deriv=0
        coords = data["x"]
        if kwargs.get("basis", "rpz").lower() == "rpz":
            coords = rpz2xyz(coords)
        # ensure curve is closed. If it's already closed this doesn't add any length
        # since ds will be zero
        coords = jnp.concatenate([coords, coords[:1]])
        X = coords[:, 0]
        Y = coords[:, 1]
        Z = coords[:, 2]
        lengths = jnp.sqrt(jnp.diff(X) ** 2 + jnp.diff(Y) ** 2 + jnp.diff(Z) ** 2)
        data["length"] = jnp.sum(lengths)
    else:
        T = jnp.linalg.norm(data["x_s"], axis=-1)
        # this is equivalent to jnp.trapz(T, s) for a closed curve, but also works
        # if grid.endpoint is False
        data["length"] = jnp.sum(T * data["ds"])
    return data
