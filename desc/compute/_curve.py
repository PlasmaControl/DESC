from desc.backend import jnp
from desc.interpolate import interp1d

from .data_index import register_compute_fun
from .geom_utils import (
    _rotation_matrix_from_normal,
    rpz2xyz,
    rpz2xyz_vec,
    xyz2rpz,
    xyz2rpz_vec,
)
from .utils import cross, dot


@register_compute_fun(
    name="s",
    label="s",
    units="~",
    units_long="None",
    description="Curve parameter, on [0, 2pi)",
    dim=3,
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
    name="x",
    label="\\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve",
    dim=3,
    params=["r_n", "center", "normal"],
    transforms={
        "r": [[0, 0, 0]],
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis="basis",
)
def _x_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    # create planar curve at z==0
    r = transforms["r"].transform(params["r_n"], dz=0)
    Z = jnp.zeros_like(r)
    X = r * jnp.cos(data["s"])
    Y = r * jnp.sin(data["s"])
    coords = jnp.array([X, Y, Z]).T
    # rotate into place
    R = _rotation_matrix_from_normal(params["normal"])
    coords = jnp.matmul(coords, R.T) + params["center"]
    coords = jnp.matmul(coords, transforms["rotmat"].T) + transforms["shift"]
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
    params=["r_n", "center", "normal"],
    transforms={
        "r": [[0, 0, 0], [0, 0, 1]],
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis="basis",
)
def _x_s_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    r = transforms["r"].transform(params["r_n"], dz=0)
    dr = transforms["r"].transform(params["r_n"], dz=1)
    dX = dr * jnp.cos(data["s"]) - r * jnp.sin(data["s"])
    dY = dr * jnp.sin(data["s"]) + r * jnp.cos(data["s"])
    dZ = jnp.zeros_like(dX)
    coords = jnp.array([dX, dY, dZ]).T
    A = _rotation_matrix_from_normal(params["normal"])
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, transforms["rotmat"].T)
    if kwargs.get("basis", "rpz").lower() == "rpz":
        X = r * jnp.cos(data["s"])
        Y = r * jnp.sin(data["s"])
        Z = jnp.zeros_like(X)
        xyzcoords = jnp.array([X, Y, Z]).T
        xyzcoords = jnp.matmul(xyzcoords, A.T) + params["center"]
        xyzcoords = jnp.matmul(xyzcoords, transforms["rotmat"].T) + transforms["shift"]
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
    params=["r_n", "center", "normal"],
    transforms={
        "r": [[0, 0, 0], [0, 0, 1], [0, 0, 2]],
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis="basis",
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
    A = _rotation_matrix_from_normal(params["normal"])
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, transforms["rotmat"].T)
    if kwargs.get("basis", "rpz").lower() == "rpz":
        X = r * jnp.cos(data["s"])
        Y = r * jnp.sin(data["s"])
        Z = jnp.zeros_like(X)
        xyzcoords = jnp.array([X, Y, Z]).T
        xyzcoords = jnp.matmul(xyzcoords, A.T) + params["center"]
        xyzcoords = jnp.matmul(xyzcoords, transforms["rotmat"].T) + transforms["shift"]
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
    params=["r_n", "center", "normal"],
    transforms={
        "r": [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis="basis",
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
    A = _rotation_matrix_from_normal(params["normal"])
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, transforms["rotmat"].T)
    if kwargs.get("basis", "rpz").lower() == "rpz":
        X = r * jnp.cos(data["s"])
        Y = r * jnp.sin(data["s"])
        Z = jnp.zeros_like(X)
        xyzcoords = jnp.array([X, Y, Z]).T
        xyzcoords = jnp.matmul(xyzcoords, A.T) + params["center"]
        xyzcoords = jnp.matmul(xyzcoords, transforms["rotmat"].T) + transforms["shift"]
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
    params=["R_n", "Z_n"],
    transforms={
        "R": [[0, 0, 0]],
        "Z": [[0, 0, 0]],
        "grid": [],
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierRZCurve",
    basis="basis",
)
def _x_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_n"], dz=0)
    Z = transforms["Z"].transform(params["Z_n"], dz=0)
    phi = transforms["grid"].nodes[:, 2]
    coords = jnp.stack([R, phi, Z], axis=1)
    # convert to xyz for displacement and rotation
    coords = rpz2xyz(coords)
    coords = coords @ transforms["rotmat"].T + transforms["shift"][jnp.newaxis, :]
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
    params=["R_n", "Z_n"],
    transforms={
        "R": [[0, 0, 0], [0, 0, 1]],
        "Z": [[0, 0, 1]],
        "grid": [],
        "rotmat": [],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierRZCurve",
    basis="basis",
)
def _x_s_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_n"], dz=0)
    dR = transforms["R"].transform(params["R_n"], dz=1)
    dZ = transforms["Z"].transform(params["Z_n"], dz=1)
    dphi = R0
    coords = jnp.stack([dR, dphi, dZ], axis=1)
    # convert to xyz for displacement and rotation
    coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    coords = coords @ transforms["rotmat"].T
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
    params=["R_n", "Z_n"],
    transforms={
        "R": [[0, 0, 0], [0, 0, 1], [0, 0, 2]],
        "Z": [[0, 0, 2]],
        "grid": [],
        "rotmat": [],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierRZCurve",
    basis="basis",
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
    coords = coords @ transforms["rotmat"].T
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
    params=["R_n", "Z_n"],
    transforms={
        "R": [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
        "Z": [[0, 0, 3]],
        "grid": [],
        "rotmat": [],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierRZCurve",
    basis="basis",
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
    coords = coords @ transforms["rotmat"].T
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
    params=["X_n", "Y_n", "Z_n"],
    transforms={
        "X": [[0, 0, 0]],
        "Y": [[0, 0, 0]],
        "Z": [[0, 0, 0]],
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierXYZCurve",
    basis="basis",
)
def _x_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    X = transforms["X"].transform(params["X_n"], dz=0)
    Y = transforms["Y"].transform(params["Y_n"], dz=0)
    Z = transforms["Z"].transform(params["Z_n"], dz=0)
    coords = jnp.stack([X, Y, Z], axis=1)
    coords = coords @ transforms["rotmat"].T + transforms["shift"][jnp.newaxis, :]
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
    params=["X_n", "Y_n", "Z_n"],
    transforms={
        "X": [[0, 0, 0], [0, 0, 1]],
        "Y": [[0, 0, 0], [0, 0, 1]],
        "Z": [[0, 0, 1]],
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierXYZCurve",
    basis="basis",
)
def _x_s_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    dX = transforms["X"].transform(params["X_n"], dz=1)
    dY = transforms["Y"].transform(params["Y_n"], dz=1)
    dZ = transforms["Z"].transform(params["Z_n"], dz=1)
    coords = jnp.stack([dX, dY, dZ], axis=1)
    coords = coords @ transforms["rotmat"].T
    if kwargs.get("basis", "rpz").lower() == "rpz":
        coords = xyz2rpz_vec(
            coords,
            x=transforms["X"].transform(params["X_n"]) + transforms["shift"][0],
            y=transforms["Y"].transform(params["Y_n"]) + transforms["shift"][1],
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
    params=["X_n", "Y_n", "Z_n"],
    transforms={
        "X": [[0, 0, 0], [0, 0, 2]],
        "Y": [[0, 0, 0], [0, 0, 2]],
        "Z": [[0, 0, 2]],
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierXYZCurve",
    basis="basis",
)
def _x_ss_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    d2X = transforms["X"].transform(params["X_n"], dz=2)
    d2Y = transforms["Y"].transform(params["Y_n"], dz=2)
    d2Z = transforms["Z"].transform(params["Z_n"], dz=2)
    coords = jnp.stack([d2X, d2Y, d2Z], axis=1)
    coords = coords @ transforms["rotmat"].T
    if kwargs.get("basis", "rpz").lower() == "rpz":
        coords = xyz2rpz_vec(
            coords,
            x=transforms["X"].transform(params["X_n"]) + transforms["shift"][0],
            y=transforms["Y"].transform(params["Y_n"]) + transforms["shift"][1],
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
    params=["X_n", "Y_n", "Z_n"],
    transforms={
        "X": [[0, 0, 0], [0, 0, 3]],
        "Y": [[0, 0, 0], [0, 0, 3]],
        "Z": [[0, 0, 3]],
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierXYZCurve",
    basis="basis",
)
def _x_sss_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    d3X = transforms["X"].transform(params["X_n"], dz=3)
    d3Y = transforms["Y"].transform(params["Y_n"], dz=3)
    d3Z = transforms["Z"].transform(params["Z_n"], dz=3)
    coords = jnp.stack([d3X, d3Y, d3Z], axis=1)
    coords = coords @ transforms["rotmat"].T
    if kwargs.get("basis", "rpz").lower() == "rpz":
        coords = xyz2rpz_vec(
            coords,
            x=transforms["X"].transform(params["X_n"]) + transforms["shift"][0],
            y=transforms["Y"].transform(params["Y_n"]) + transforms["shift"][1],
        )
    data["x_sss"] = coords
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve",
    dim=3,
    params=["X", "Y", "Z", "knots", "method"],
    transforms={
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    basis="basis",
)
def _x_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    xq = data["s"]

    Xq = interp1d(
        xq,
        params["knots"],
        params["X"],
        method=params["method"],
        derivative=0,
        period=2 * jnp.pi,
    )
    Yq = interp1d(
        xq,
        params["knots"],
        params["Y"],
        method=params["method"],
        derivative=0,
        period=2 * jnp.pi,
    )
    Zq = interp1d(
        xq,
        params["knots"],
        params["Z"],
        method=params["method"],
        derivative=0,
        period=2 * jnp.pi,
    )

    coords = jnp.stack([Xq, Yq, Zq], axis=1)
    coords = coords @ transforms["rotmat"].T + transforms["shift"][jnp.newaxis, :]
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
    params=["X", "Y", "Z", "knots", "method"],
    transforms={
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    basis="basis",
)
def _x_s_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    xq = data["s"]

    dXq = interp1d(
        xq,
        params["knots"],
        params["X"],
        method=params["method"],
        derivative=1,
        period=2 * jnp.pi,
    )
    dYq = interp1d(
        xq,
        params["knots"],
        params["Y"],
        method=params["method"],
        derivative=1,
        period=2 * jnp.pi,
    )
    dZq = interp1d(
        xq,
        params["knots"],
        params["Z"],
        method=params["method"],
        derivative=1,
        period=2 * jnp.pi,
    )

    coords_s = jnp.stack([dXq, dYq, dZq], axis=1)
    coords_s = coords_s @ transforms["rotmat"].T

    if kwargs.get("basis", "rpz").lower() == "rpz":
        # calculate the xy coordinates to rotate to rpz
        Xq = interp1d(
            xq,
            params["knots"],
            params["X"],
            method=params["method"],
            derivative=0,
            period=2 * jnp.pi,
        )
        Yq = interp1d(
            xq,
            params["knots"],
            params["Y"],
            method=params["method"],
            derivative=0,
            period=2 * jnp.pi,
        )
        Zq = interp1d(
            xq,
            params["knots"],
            params["Z"],
            method=params["method"],
            derivative=0,
            period=2 * jnp.pi,
        )

        coords = jnp.stack([Xq, Yq, Zq], axis=1)
        coords = coords @ transforms["rotmat"].T + transforms["shift"][jnp.newaxis, :]

        coords_s = xyz2rpz_vec(coords_s, x=coords[:, 0], y=coords[:, 1])
    data["x_s"] = coords_s
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["X", "Y", "Z", "knots", "method"],
    transforms={
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    basis="basis",
)
def _x_ss_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    xq = data["s"]

    d2Xq = interp1d(
        xq,
        params["knots"],
        params["X"],
        method=params["method"],
        derivative=2,
        period=2 * jnp.pi,
    )
    d2Yq = interp1d(
        xq,
        params["knots"],
        params["Y"],
        method=params["method"],
        derivative=2,
        period=2 * jnp.pi,
    )
    d2Zq = interp1d(
        xq,
        params["knots"],
        params["Z"],
        method=params["method"],
        derivative=2,
        period=2 * jnp.pi,
    )

    coords_ss = jnp.stack([d2Xq, d2Yq, d2Zq], axis=1)
    coords_ss = coords_ss @ transforms["rotmat"].T

    if kwargs.get("basis", "rpz").lower() == "rpz":
        # calculate the xy coordinates to rotate to rpz
        Xq = interp1d(
            xq,
            params["knots"],
            params["X"],
            method=params["method"],
            derivative=0,
            period=2 * jnp.pi,
        )
        Yq = interp1d(
            xq,
            params["knots"],
            params["Y"],
            method=params["method"],
            derivative=0,
            period=2 * jnp.pi,
        )
        Zq = interp1d(
            xq,
            params["knots"],
            params["Z"],
            method=params["method"],
            derivative=0,
            period=2 * jnp.pi,
        )
        coords = jnp.stack([Xq, Yq, Zq], axis=1)
        coords = coords @ transforms["rotmat"].T + transforms["shift"][jnp.newaxis, :]

        coords_ss = xyz2rpz_vec(coords_ss, x=coords[:, 0], y=coords[:, 1])
    data["x_ss"] = coords_ss
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["X", "Y", "Z", "knots", "method"],
    transforms={
        "rotmat": [],
        "shift": [],
    },
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    basis="basis",
)
def _x_sss_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    xq = data["s"]

    d3Xq = interp1d(
        xq,
        params["knots"],
        params["X"],
        method=params["method"],
        derivative=3,
        period=2 * jnp.pi,
    )
    d3Yq = interp1d(
        xq,
        params["knots"],
        params["Y"],
        method=params["method"],
        derivative=3,
        period=2 * jnp.pi,
    )
    d3Zq = interp1d(
        xq,
        params["knots"],
        params["Z"],
        method=params["method"],
        derivative=3,
        period=2 * jnp.pi,
    )

    coords_sss = jnp.stack([d3Xq, d3Yq, d3Zq], axis=1)
    coords_sss = coords_sss @ transforms["rotmat"].T

    if kwargs.get("basis", "rpz").lower() == "rpz":
        # calculate the xy coordinates to rotate to rpz
        Xq = interp1d(
            xq,
            params["knots"],
            params["X"],
            method=params["method"],
            derivative=0,
            period=2 * jnp.pi,
        )
        Yq = interp1d(
            xq,
            params["knots"],
            params["Y"],
            method=params["method"],
            derivative=0,
            period=2 * jnp.pi,
        )
        Zq = interp1d(
            xq,
            params["knots"],
            params["Z"],
            method=params["method"],
            derivative=0,
            period=2 * jnp.pi,
        )
        coords = jnp.stack([Xq, Yq, Zq], axis=1)
        coords = coords @ transforms["rotmat"].T + transforms["shift"][jnp.newaxis, :]

        coords_sss = xyz2rpz_vec(coords_sss, x=coords[:, 0], y=coords[:, 1])
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
    params=[],
    transforms={"rotmat": []},
    profiles=[],
    coordinates="s",
    data=["frenet_tangent", "frenet_normal"],
    parameterization="desc.geometry.core.Curve",
)
def _frenet_binormal(params, transforms, profiles, data, **kwargs):
    data["frenet_binormal"] = cross(
        data["frenet_tangent"], data["frenet_normal"]
    ) * jnp.linalg.det(transforms["rotmat"])
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
    data=["s", "x_s"],
    parameterization=[
        "desc.geometry.curve.FourierRZCurve",
        "desc.geometry.curve.FourierXYZCurve",
        "desc.geometry.curve.FourierPlanarCurve",
    ],
)
def _length(params, transforms, profiles, data, **kwargs):
    T = jnp.linalg.norm(data["x_s"], axis=-1)
    data["length"] = jnp.trapz(T, data["s"])
    return data


@register_compute_fun(
    name="length",
    label="L",
    units="m",
    units_long="meters",
    description="Length of the curve",
    dim=0,
    params=["method"],
    transforms={},
    profiles=[],
    coordinates="",
    data=["s", "x", "x_s"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
)
def _length_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    if params["method"] == "nearest":  # cannot use derivative method as deriv=0
        coords = data["x"]
        if kwargs.get("basis", "rpz").lower() == "rpz":
            coords = rpz2xyz(coords)
        X = coords[:, 0]
        Y = coords[:, 1]
        Z = coords[:, 2]
        lengths = jnp.sqrt(jnp.diff(X) ** 2 + jnp.diff(Y) ** 2 + jnp.diff(Z) ** 2)
        data["length"] = jnp.sum(lengths)
    else:
        T = jnp.linalg.norm(data["x_s"], axis=-1)
        data["length"] = jnp.trapz(T, data["s"])
    return data
