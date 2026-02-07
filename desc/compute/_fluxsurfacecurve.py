from .data_index import register_compute_fun

kwargs_FourierUmbilicCurve = {
    "n_umbilic": """int:
        Prefactor of the form 1/n_umbilic modifying NFP.
        Curve closes after n_umbilic/gcd(n_umbilic, NFP) transits.
        Default is n_umbilic = 1.
        """,
    "m_umbilic": """int:
        Parameter arising from umbilic torus parameterization, determining
        the average slope of the curve in the (theta,zeta) plane.
        Should satisfy gcd(n_umbilic, m_umbilic)=1.
        Default is m_umbilic = 1.
        """,
    "NFP": """int:
        Number of field periods.""",
}


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
    coordinates="phi",
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
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="phi",
    data=["UC", "phi"],
    parameterization="desc.geometry.fluxsurfacecurve.FourierUmbilicCurve",
    **kwargs_FourierUmbilicCurve
)
def _theta(params, transforms, profiles, data, **kwargs):
    data["theta"] = (
        1
        / kwargs.get("n_umbilic")
        * (kwargs.get("m_umbilic") * kwargs.get("NFP") * data["phi"] + data["UC"])
    )
    return data
