from .data_index import register_compute_fun


@register_compute_fun(
    name="UC",
    label="umbilic curve",
    units="~",
    units_long="None",
    description="angular difference between theta and phi on a surface"
    + r", equal to n_umbilic \theta - m_umbilic NFP \zeta",
    dim=1,
    params=["a_n"],
    transforms={"UC": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="",
    data=[],
    parameterization="desc.geometry.fluxsurfacecurve.FourierUmbilicCurve",
)
def _UC_FourierUmbilicCurve(params, transforms, profiles, data, **kwargs):
    # If the grid is non-uniform, the transform has to be modified
    data["UC"] = transforms["UC"].transform(params["a_n"], dz=0)
    return data


@register_compute_fun(
    name="phi",
    label="phi",
    units="~",
    units_long="Radians",
    description="Values of toroidal angle phi along curve",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="phi",
    data=[],
    parameterization="desc.geometry.fluxsurfacecurve.FourierUmbilicCurve",
)
def _phi(params, transforms, profiles, data, **kwargs):
    data["phi"] = transforms["grid"].nodes[:, 2]
    return data


@register_compute_fun(
    name="theta",
    label="theta",
    units="~",
    units_long="Radians",
    description="Values of poloidal angle theta along curve",
    dim=1,
    params=["n_umbilic", "m_umbilic", "NFP"],
    transforms={"grid": []},
    profiles=[],
    coordinates="phi",
    data=["UC", "phi"],
    parameterization="desc.geometry.fluxsurfacecurve.FourierUmbilicCurve",
)
def _theta(params, transforms, profiles, data, **kwargs):
    data["theta"] = (
        1
        / params["n_umbilic"]
        * (params["m_umbilic"] * params["NFP"] * data["phi"] + data["UC"])
    )
    return data
