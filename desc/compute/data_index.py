"""data_index contains all the quantities calculated by the compute functions."""


def register_compute_fun(
    name,
    label,
    units,
    units_long,
    description,
    dim,
    params,
    transforms,
    profiles,
    coordinates,
    data,
    parameterization="desc.equilibrium.equilibrium.Equilibrium",
    axis_limit_data=None,
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
    profiles : list of str
        Names of profiles needed, eg "iota", "pressure"
    coordinates : str
        Coordinate dependency. IE, "rtz" for a function of rho, theta, zeta, or "r" for
        a flux function, etc.
    data : list of str
        Names of other items in the data index needed to compute qty.
    parameterization: str or list of str
        Name of desc types the method is valid for. eg 'desc.geometry.FourierXYZCurve'
        or `desc.equilibrium.Equilibrium`.
    axis_limit_data : list of str
        Names of other items in the data index needed to compute axis limit of qty.

    Notes
    -----
    Should only list *direct* dependencies. The full dependencies will be built
    recursively at runtime using each quantity's direct dependencies.
    """
    if not isinstance(parameterization, (tuple, list)):
        parameterization = [parameterization]

    deps = {
        "params": params,
        "transforms": transforms,
        "profiles": profiles,
        "data": data,
        "axis_limit_data": [] if axis_limit_data is None else axis_limit_data,
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
            "coordinates": coordinates,
            "dependencies": deps,
        }
        for p in parameterization:
            flag = False
            for base_class, superclasses in _class_inheritance.items():
                if p in superclasses or p == base_class:
                    if name in data_index[base_class]:
                        raise ValueError(
                            f"Already registered function with parameterization {p} and name {name}."
                        )
                    data_index[base_class][name] = d.copy()
                    flag = True
            if not flag:
                raise ValueError(
                    f"Can't register function with unknown parameterization: {p}"
                )
        return func

    return _decorator


# This allows us to handle subclasses whose data_index stuff should inherit
# from parent classes.
# This is the least bad solution I've found, since everything else requires
# crazy circular imports
# could maybe make this fancier with a registry of compute-able objects?
_class_inheritance = {
    "desc.equilibrium.equilibrium.Equilibrium": [],
    "desc.geometry.curve.FourierRZCurve": [
        "desc.geometry.core.Curve",
    ],
    "desc.geometry.curve.FourierXYZCurve": [
        "desc.geometry.core.Curve",
    ],
    "desc.geometry.curve.FourierPlanarCurve": [
        "desc.geometry.core.Curve",
    ],
    "desc.geometry.curve.SplineXYZCurve": [
        "desc.geometry.core.Curve",
    ],
    "desc.geometry.surface.FourierRZToroidalSurface": [
        "desc.geometry.core.Surface",
    ],
    "desc.geometry.surface.ZernikeRZToroidalSection": [
        "desc.geometry.core.Surface",
    ],
    "desc.coils.FourierRZCoil": [
        "desc.geometry.curve.FourierRZCurve",
        "desc.geometry.core.Curve",
    ],
    "desc.coils.FourierXYZCoil": [
        "desc.geometry.curve.FourierXYZCurve",
        "desc.geometry.core.Curve",
    ],
    "desc.coils.FourierPlanarCoil": [
        "desc.geometry.curve.FourierPlanarCurve",
        "desc.geometry.core.Curve",
    ],
    "desc.coils.SplineXYZCoil": [
        "desc.geometry.curve.SplineXYZCurve",
        "desc.geometry.core.Curve",
    ],
}

data_index = {p: {} for p in _class_inheritance.keys()}
