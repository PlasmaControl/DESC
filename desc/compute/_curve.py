from desc.backend import jnp
from desc.compute.utils import _has_params, _has_transforms, has_dependencies
from desc.geometry.utils import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

curve_data_index = {}


def register_curve_compute_fun(
    name,
    label,
    units,
    units_long,
    description,
    dim,
    params,
    transforms,
    data,
    parameterization=None,
    **kwargs,
):
    """Decorator to wrap a function and add it to the list of things we can compute.

    Parameters
    ----------
    name : str
        Name of the quantity. This will be used as the key used to compute the
        quantity in `compute` and its name in the data dictionary.
    label : str
        Title of the quantity in LaTeX format.
    units : str
        Units of the quantity in LaTeX format.
    units_long : str
        Full units without abbreviations.
    description : str
        Description of the quantity.
    dim : int
        Dimension of the quantity: 0-D (global qty), 1-D (local scalar qty),
        or 3-D (local vector qty).
    params : list of str
        Parameters of equilibrium needed to compute quantity, eg "R_lmn", "Z_lmn"
    transforms : dict
        Dictionary of keys and derivative orders [rho, theta, zeta] for R, Z, etc.
    data : list of str
        Names of other items in the data index needed to compute qty.
    parameterization: str or None
        Name of curve types the method is valid for. eg 'FourierXYZCurve'.
        None means it is valid for any curve type (calculation does not depend on
        parameterization).

    Notes
    -----
    Should only list *direct* dependencies. The full dependencies will be built
    recursively at runtime using each quantity's direct dependencies.
    """
    deps = {
        "params": params,
        "transforms": transforms,
        "data": data,
        "kwargs": list(kwargs.values()),
    }

    def _decorator(func):
        d = {
            "label": label,
            "units": units,
            "units_long": units_long,
            "description": description,
            "fun": func,
            "dim": dim,
            "dependencies": deps,
        }
        if parameterization not in curve_data_index:
            curve_data_index[parameterization] = {}
        curve_data_index[parameterization][name] = d
        return func

    return _decorator


def compute(names, params, transforms, data=None, **kwargs):
    """Compute the quantity given by name on grid.

    Parameters
    ----------
    names : str or array-like of str
        Name(s) of the quantity(s) to compute.
    params : dict of ndarray
        Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc
        Defaults to attributes of self.
    transforms : dict of Transform
        Transforms for R, Z, lambda, etc. Default is to build from grid
    data : dict of ndarray
        Data computed so far, generally output from other compute functions

    Returns
    -------
    data : dict of ndarray
        Computed quantity and intermediate variables.

    """
    if isinstance(names, str):
        names = [names]
    for name in names:
        if name not in curve_data_index:
            raise ValueError("Unrecognized value '{}'.".format(name))
    allowed_kwargs = {}  # might have some in the future
    bad_kwargs = kwargs.keys() - allowed_kwargs
    if len(bad_kwargs) > 0:
        raise ValueError(f"Unrecognized argument(s): {bad_kwargs}")

    for name in names:
        assert _has_params(name, params), f"Don't have params to compute {name}"
        assert _has_transforms(
            name, transforms
        ), f"Don't have transforms to compute {name}"

    if data is None:
        data = {}

    data = _compute(
        names,
        params=params,
        transforms=transforms,
        data=data,
        **kwargs,
    )
    return data


def _compute(names, params, transforms, data=None, **kwargs):
    """Same as above but without checking inputs for faster recursion."""
    for name in names:
        if name in data:
            # don't compute something that's already been computed
            continue
        if not has_dependencies(name, params, transforms, data):
            # then compute the missing dependencies
            data = _compute(
                curve_data_index[name]["dependencies"]["data"],
                params=params,
                transforms=transforms,
                data=data,
                **kwargs,
            )
        # now compute the quantity
        data = curve_data_index[name]["fun"](params, transforms, data, **kwargs)
    return data


@register_curve_compute_fun(
    name="s",
    label="s",
    units="~",
    units_long="None",
    description="Curve parameter",
    dim=3,
    params=[],
    transforms={"grid": []},
    data=[],
)
def _s(params, data, transforms, basis="rpz"):
    data["s"] = transforms["grid"].nodes[:, 2]
    return data


def _rotation_matrix_from_normal(normal):
    nx, ny, nz = normal
    nxny = jnp.sqrt(nx**2 + ny**2)
    R = jnp.array(
        [
            [ny / nxny, -nx / nxny, 0],
            [nx * nx / nxny, ny * nz / nxny, -nxny],
            [nx, ny, nz],
        ]
    ).T
    R = jnp.where(nxny == 0, jnp.eye(3), R)
    return R


@register_curve_compute_fun(
    name="r",
    label="\\mathbf{r}",
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
    data=["s"],
    parameterization="FourierPlanarCurve",
)
def _r_FourierPlanarCurve(params, data, transforms, basis="rpz"):
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
    if basis.lower() == "rpz":
        xyzcoords = jnp.array([X, Y, Z]).T
        xyzcoords = jnp.matmul(xyzcoords, R.T) + params["center"]
        xyzcoords = jnp.matmul(xyzcoords, transforms["rotmat"].T) + transforms["shift"]
        x, y, z = xyzcoords.T
        coords = xyz2rpz_vec(coords, x=x, y=y)
    data["r"] = coords
    return data


@register_curve_compute_fun(
    name="r_s",
    label="\\partial_{s} \\mathbf{r}",
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
    data=["s"],
    parameterization="FourierPlanarCurve",
)
def _r_s_FourierPlanarCurve(params, data, transforms, basis="rpz"):
    r = transforms["r"].transform(params["r_n"], dz=0)
    dr = transforms["r"].transform(params["r_n"], dz=1)
    dX = dr * jnp.cos(data["s"]) - r * jnp.sin(data["s"])
    dY = dr * jnp.sin(data["s"]) + r * jnp.cos(data["s"])
    dZ = jnp.zeros_like(dX)
    coords = jnp.array([dX, dY, dZ]).T
    A = _rotation_matrix_from_normal(params["normal"])
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, transforms["rotmat"].T)
    if basis.lower() == "rpz":
        X = r * jnp.cos(data["s"])
        Y = r * jnp.sin(data["s"])
        Z = jnp.zeros_like(X)
        xyzcoords = jnp.array([X, Y, Z]).T
        xyzcoords = jnp.matmul(xyzcoords, A.T) + params["center"]
        xyzcoords = jnp.matmul(xyzcoords, transforms["rotmat"].T) + transforms["shift"]
        x, y, z = xyzcoords.T
        coords = xyz2rpz_vec(coords, x=x, y=y)
    data["r_s"] = coords
    return data


@register_curve_compute_fun(
    name="r_ss",
    label="\\partial_{ss} \\mathbf{r}",
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
    data=["s"],
    parameterization="FourierPlanarCurve",
)
def _r_ss_FourierPlanarCurve(params, data, transforms, basis="rpz"):
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
    if basis.lower() == "rpz":
        X = r * jnp.cos(data["s"])
        Y = r * jnp.sin(data["s"])
        Z = jnp.zeros_like(X)
        xyzcoords = jnp.array([X, Y, Z]).T
        xyzcoords = jnp.matmul(xyzcoords, A.T) + params["center"]
        xyzcoords = jnp.matmul(xyzcoords, transforms["rotmat"].T) + transforms["shift"]
        x, y, z = xyzcoords.T
        coords = xyz2rpz_vec(coords, x=x, y=y)
    data["r_ss"] = coords
    return data


@register_curve_compute_fun(
    name="r_sss",
    label="\\partial_{sss} \\mathbf{r}",
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
    data=["s"],
    parameterization="FourierPlanarCurve",
)
def _r_sss_FourierPlanarCurve(params, data, transforms, basis="rpz"):
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
    if basis.lower() == "rpz":
        X = r * jnp.cos(data["s"])
        Y = r * jnp.sin(data["s"])
        Z = jnp.zeros_like(X)
        xyzcoords = jnp.array([X, Y, Z]).T
        xyzcoords = jnp.matmul(xyzcoords, A.T) + params["center"]
        xyzcoords = jnp.matmul(xyzcoords, transforms["rotmat"].T) + transforms["shift"]
        x, y, z = xyzcoords.T
        coords = xyz2rpz_vec(coords, x=x, y=y)
    data["r_sss"] = coords
    return data


@register_curve_compute_fun(
    name="r",
    label="\\mathbf{r}",
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
    data=[],
    parameterization="FourierRZCurve",
)
def _r_FourierRZCurve(params, data, transforms, basis="rpz"):
    R = transforms["R"].transform(params["R_n"], dz=0)
    Z = transforms["Z"].transform(params["Z_n"], dz=0)
    phi = transforms["grid"].nodes[:, 2]
    coords = jnp.stack([R, phi, Z], axis=1)
    # convert to xyz for displacement and rotation
    coords = rpz2xyz(coords)
    coords = coords @ transforms["rotmat"].T + transforms["shift"][jnp.newaxis, :]
    if basis.lower() == "rpz":
        coords = xyz2rpz(coords)
    data["r"] = coords
    return data


@register_curve_compute_fun(
    name="r_s",
    label="\\partial_{s} \\mathbf{r}",
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
    data=[],
    parameterization="FourierRZCurve",
)
def _r_s_FourierRZCurve(params, data, transforms, basis="rpz"):

    R0 = transforms["R"].transform(params["R_n"], dz=0)
    dR = transforms["R"].transform(params["R_n"], dz=1)
    dZ = transforms["Z"].transform(params["Z_n"], dz=1)
    dphi = R0
    coords = jnp.stack([dR, dphi, dZ], axis=1)
    # convert to xyz for displacement and rotation
    coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    coords = coords @ transforms["rotmat"].T
    if basis.lower() == "rpz":
        coords = xyz2rpz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["r_s"] = coords
    return data


@register_curve_compute_fun(
    name="r_ss",
    label="\\partial_{ss} \\mathbf{r}",
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
    data=[],
    parameterization="FourierRZCurve",
)
def _r_ss_FourierRZCurve(params, data, transforms, basis="rpz"):
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
    if basis.lower() == "rpz":
        coords = xyz2rpz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["r_ss"] = coords
    return data


@register_curve_compute_fun(
    name="r_sss",
    label="\\partial_{sss} \\mathbf{r}",
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
    data=[],
    parameterization="FourierRZCurve",
)
def _r_sss_FourierRZCurve(params, data, transforms, basis="rpz"):
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
    if basis.lower() == "rpz":
        coords = xyz2rpz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    data["r_sss"] = coords
    return data


@register_curve_compute_fun(
    name="r",
    label="\\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along curve",
    dim=3,
    params=["X_n", "Y_n", "Z_n"],
    transforms={"r": [[0, 0, 0]], "rotmat": [], "shift": []},
    data=[],
    parameterization="FourierXYZCurve",
)
def _r_FourierXYZCurve(params, data, transforms, basis="rpz"):
    X = transforms["r"].transform(params["X_n"], dz=0)
    Y = transforms["r"].transform(params["Y_n"], dz=0)
    Z = transforms["r"].transform(params["Z_n"], dz=0)
    coords = jnp.stack([X, Y, Z], axis=1)
    coords = coords @ transforms["rotmat"].T + transforms["shift"][jnp.newaxis, :]
    if basis.lower() == "rpz":
        coords = xyz2rpz(coords)
    data["r"] = coords
    return data


@register_curve_compute_fun(
    name="r_s",
    label="\\partial_{s} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["X_n", "Y_n", "Z_n"],
    transforms={"r": [[0, 0, 1]], "rotmat": [], "shift": []},
    data=[],
    parameterization="FourierXYZCurve",
)
def _r_s_FourierXYZCurve(params, data, transforms, basis="rpz"):
    X = transforms["r"].transform(params["X_n"], dz=1)
    Y = transforms["r"].transform(params["Y_n"], dz=1)
    Z = transforms["r"].transform(params["Z_n"], dz=1)
    coords = jnp.stack([X, Y, Z], axis=1)
    coords = coords @ transforms["rotmat"].T
    if basis.lower() == "rpz":
        coords = xyz2rpz_vec(
            coords,
            x=coords[:, 1] + transforms["shift"][0],
            y=coords[:, 1] + transforms["shift"][1],
        )
    data["r_s"] = coords
    return data


@register_curve_compute_fun(
    name="r_ss",
    label="\\partial_{ss} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["X_n", "Y_n", "Z_n"],
    transforms={"r": [[0, 0, 2]], "rotmat": [], "shift": []},
    data=[],
    parameterization="FourierXYZCurve",
)
def _r_ss_FourierXYZCurve(params, data, transforms, basis="rpz"):
    X = transforms["r"].transform(params["X_n"], dz=2)
    Y = transforms["r"].transform(params["Y_n"], dz=2)
    Z = transforms["r"].transform(params["Z_n"], dz=2)
    coords = jnp.stack([X, Y, Z], axis=1)
    coords = coords @ transforms["rotmat"].T
    if basis.lower() == "rpz":
        coords = xyz2rpz_vec(
            coords,
            x=coords[:, 1] + transforms["shift"][0],
            y=coords[:, 1] + transforms["shift"][1],
        )
    data["r_ss"] = coords
    return data


@register_curve_compute_fun(
    name="r_sss",
    label="\\partial_{sss} \\mathbf{r}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["X_n", "Y_n", "Z_n"],
    transforms={"r": [[0, 0, 3]], "rotmat": [], "shift": []},
    data=[],
    parameterization="FourierXYZCurve",
)
def _r_sss_FourierXYZCurve(params, data, transforms, basis="rpz"):
    X = transforms["r"].transform(params["X_n"], dz=3)
    Y = transforms["r"].transform(params["Y_n"], dz=3)
    Z = transforms["r"].transform(params["Z_n"], dz=3)
    coords = jnp.stack([X, Y, Z], axis=1)
    coords = coords @ transforms["rotmat"].T
    if basis.lower() == "rpz":
        coords = xyz2rpz_vec(
            coords,
            x=coords[:, 1] + transforms["shift"][0],
            y=coords[:, 1] + transforms["shift"][1],
        )
    data["r_sss"] = coords
    return data


@register_curve_compute_fun(
    name="frenet_tangent",
    label="T",
    units="~",
    units_long="None",
    description="Tangent unit vector to curve in Frenet-Serret frame",
    dim=3,
    params=[],
    transforms={},
    data=["r_s"],
)
def _frenet_tangent(params, data, transforms, basis="rpz"):
    data["frenet_tangent"] = data["r_s"] / jnp.linalg.norm(data["r_s"])[:, None]
    return data


@register_curve_compute_fun(
    name="frenet_normal",
    label="N",
    units="~",
    units_long="None",
    description="Normal unit vector to curve in Frenet-Serret frame",
    dim=3,
    params=[],
    transforms={},
    data=["r_ss"],
)
def _frenet_normal(params, data, transforms, basis="rpz"):
    data["frenet_normal"] = data["r_ss"] / jnp.linalg.norm(data["r_ss"])[:, None]
    return data


@register_curve_compute_fun(
    name="frenet_binormal",
    label="B",
    units="~",
    units_long="None",
    description="Binormal unit vector to curve in Frenet-Serret frame",
    dim=3,
    params=[],
    transforms={"rotmat": []},
    data=["T", "N"],
)
def _frenet_binormal(params, data, transforms, basis="rpz"):
    data["frenet_binormal"] = jnp.cross(data["T"], data["N"], axis=1) * jnp.linalg.det(
        transforms["rotmat"]
    )
    return data


@register_curve_compute_fun(
    name="curvature",
    label="\\kappa",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Scalar curvature of the curve",
    dim=1,
    params=[],
    transforms={},
    data=["r_s", "r_ss"],
)
def _curvature(params, data, transforms, basis="rpz"):
    dxn = jnp.linalg.norm(data["r_s"], axis=1)[:, jnp.newaxis]
    data["curvature"] = jnp.linalg.norm(
        jnp.cross(data["r_s"], data["r_ss"], axis=1) / dxn**3, axis=1
    )
    return data


@register_curve_compute_fun(
    name="torsion",
    label="\\tau",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Scalar torsion of the curve",
    dim=1,
    params=[],
    transforms={},
    data=["r_s", "r_ss", "r_sss"],
)
def _torsion(params, data, transforms, basis="rpz"):
    dxd2x = jnp.cross(data["r_s"], data["r_ss"], axis=1)
    data["torsion"] = (
        jnp.sum(dxd2x * data["r_sss"], axis=1)
        / jnp.linalg.norm(dxd2x, axis=1)[:, jnp.newaxis] ** 2
    )
    return data


@register_curve_compute_fun(
    name="length",
    label="L",
    units="m",
    units_long="meters",
    description="Length of the curve",
    dim=0,
    params=[],
    transforms={},
    data=["s", "r_s"],
)
def _length(params, data, transforms, basis="rpz"):
    T = jnp.linalg.norm(data["r_s"], axis=1)
    data["length"] = jnp.trapz(T, data["s"])
    return data
