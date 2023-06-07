"""data_index contains all the quantities calculated by the compute functions."""

data_index = {}


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
    axis_limit_data=None,
    **kwargs
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
    axis_limit_data : list of str
        Names of other items in the data index needed to compute axis limit of qty.

    Notes
    -----
    Should only list *direct* dependencies. The full dependencies will be built
    recursively at runtime using each quantity's direct dependencies.
    """
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
        data_index[name] = d
        return func

    return _decorator
