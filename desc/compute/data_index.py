"""data_index contains all the quantities calculated by the compute functions."""

import functools
from collections import deque

import numpy as np


def find_permutations(primary, separator="_"):
    """Finds permutations of quantity names for aliases."""
    prefix, primary_permutation = primary.rsplit(separator, 1)
    primary_permutation = deque(primary_permutation)

    new_permutations = []
    for i in range(len(primary_permutation)):
        primary_permutation.rotate(1)
        new_permutations.append(list(primary_permutation))

    # join new permutation to form alias keys
    aliases = [prefix + separator + "".join(perm) for perm in new_permutations]
    aliases = np.unique(aliases)
    aliases = np.delete(aliases, np.where(aliases == primary))

    return aliases


def assign_alias_data(
    alias, primary, fun, params, profiles, transforms, data, **kwargs
):
    """Assigns primary data to alias.

    Parameters
    ----------
    alias : `str`
        data_index key for alias of primary
    primary : `str`
        key defined in compute function

    Returns
    -------
    data : `dict`
        computed data dictionary (includes both alias and primary)

    """
    data = fun(params, transforms, profiles, data, **kwargs)
    data[alias] = data[primary].copy()
    return data


def register_compute_fun(  # noqa: C901
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
    aliases=None,
    parameterization="desc.equilibrium.equilibrium.Equilibrium",
    resolution_requirement="",
    grid_requirement=None,
    source_grid_requirement=None,
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
    axis_limit_data : list of str
        Names of other items in the data index needed to compute axis limit of qty.
    aliases : list of str
        Aliases of `name`. Will be stored in the data dictionary as a copy of `name`s
        data.
    parameterization : str or list of str
        Name of desc types the method is valid for. eg `'desc.geometry.FourierXYZCurve'`
        or `'desc.equilibrium.Equilibrium'`.
    resolution_requirement : str
        Resolution requirements in coordinates. I.e. "r" expects radial resolution
        in the grid. Likewise, "rtz" is shorthand for "rho, theta, zeta" and indicates
        the computation expects a grid with radial, poloidal, and toroidal resolution.
        If the computation simply performs pointwise operations, instead of a
        reduction (such as integration) over a coordinate, then an empty string may
        be used to indicate no requirements.
    grid_requirement : dict
        Attributes of the grid that the compute function requires.
        Also assumes dependencies were computed on such a grid.
        As an example, quantities that require tensor product grids over 2 or more
        coordinates may specify ``grid_requirement={"is_meshgrid": True}``.
    source_grid_requirement : dict
        Attributes of the source grid that the compute function requires.
        Also assumes dependencies were computed on such a grid.
        By default, the source grid is assumed to be ``transforms["grid"]`` and
        no requirements are expected of it. As an example, quantities that require
        integration along field lines may specify
        ``source_grid_requirement={"coordinates": "raz"}``.
        which will allow accessing the Clebsch-Type rho, alpha, zeta coordinates in
        ``transforms["grid"].source_grid``` that correspond to the DESC rho, theta,
        zeta coordinates in ``transforms["grid"]``.

    Notes
    -----
    Should only list *direct* dependencies. The full dependencies will be built
    recursively at runtime using each quantity's direct dependencies.
    """
    if aliases is None:
        aliases = []
    if source_grid_requirement is None:
        source_grid_requirement = {}
    if grid_requirement is None:
        grid_requirement = {}
    if not isinstance(parameterization, (tuple, list)):
        parameterization = [parameterization]
    if not isinstance(aliases, (tuple, list)):
        aliases = [aliases]

    deps = {
        "params": params,
        "transforms": transforms,
        "profiles": profiles,
        "data": data,
        "axis_limit_data": [] if axis_limit_data is None else axis_limit_data,
        "kwargs": list(kwargs.keys()),
    }
    for kw in kwargs:
        allowed_kwargs.add(kw)
    splits = name.rsplit("_", 1)
    if (
        len(splits) > 1
        # Only look for permutations of partial derivatives of same coordinate system.
        and {"r", "t", "z"}.issuperset(splits[-1])
    ):
        aliases_temp = np.append(np.array(aliases), find_permutations(name))
        for alias in aliases:
            aliases_temp = np.append(aliases_temp, find_permutations(alias))
        aliases = np.unique(aliases_temp)

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
            "aliases": aliases,
            "resolution_requirement": resolution_requirement,
            "grid_requirement": grid_requirement,
            "source_grid_requirement": source_grid_requirement,
        }
        for p in parameterization:
            flag = False
            for base_class, superclasses in _class_inheritance.items():
                if p in superclasses or p == base_class:
                    # already registered ?
                    if name in data_index[base_class]:
                        if p == data_index[base_class][name]["parameterization"]:
                            raise ValueError(
                                f"Already registered function with parameterization {p}"
                                f" and name {name}."
                            )
                        # if it was already registered from a parent class, we
                        # prefer the child class.
                        inheritance_order = [base_class] + superclasses
                        if inheritance_order.index(p) > inheritance_order.index(
                            data_index[base_class][name]["parameterization"]
                        ):
                            continue
                    d["parameterization"] = p
                    data_index[base_class][name] = d.copy()
                    all_kwargs[base_class][name] = kwargs
                    for alias in aliases:
                        data_index[base_class][alias] = d.copy()
                        # assigns alias compute func to generator to be used later
                        data_index[base_class][alias]["fun"] = functools.partial(
                            assign_alias_data,
                            alias=alias,
                            primary=name,
                            fun=data_index[base_class][name]["fun"],
                        )
                        all_kwargs[base_class][alias] = kwargs

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
    "desc.magnetic_fields._current_potential.CurrentPotentialField": [
        "desc.geometry.surface.FourierRZToroidalSurface",
        "desc.geometry.core.Surface",
        "desc.magnetic_fields._core.MagneticField",
    ],
    "desc.magnetic_fields._current_potential.FourierCurrentPotentialField": [
        "desc.geometry.surface.FourierRZToroidalSurface",
        "desc.geometry.core.Surface",
        "desc.magnetic_fields._core.MagneticField",
    ],
    "desc.coils.SplineXYZCoil": [
        "desc.geometry.curve.SplineXYZCurve",
        "desc.geometry.core.Curve",
    ],
    "desc.magnetic_fields._core.OmnigenousField": [],
    "desc.magnetic_fields._core.PiecewiseOmnigenousField": [],
}
data_index = {p: {} for p in _class_inheritance.keys()}
all_kwargs = {p: {} for p in _class_inheritance.keys()}
allowed_kwargs = {"basis"}
# dictionary of {deprecated_name: new_name} for deprecated compute quantities
deprecated_names = {
    "sqrt(g)_B": "sqrt(g)_Boozer_DESC",
    "|B|_mn": "|B|_mn_B",
}


def is_0d_vol_grid(name, p="desc.equilibrium.equilibrium.Equilibrium"):
    """Is name constant throughout plasma volume and needs full volume to compute?."""
    # Should compute on a grid that samples entire plasma volume.
    # In particular, a QuadratureGrid for accurate radial integration.
    return (
        data_index[p][name]["coordinates"] == ""
        and data_index[p][name]["resolution_requirement"] != ""
    )


def is_1dr_rad_grid(name, p="desc.equilibrium.equilibrium.Equilibrium"):
    """Is name constant over radial surfaces and needs full surface to compute?."""
    return (
        data_index[p][name]["coordinates"] == "r"
        and data_index[p][name]["resolution_requirement"] == "tz"
    )


def is_1dz_tor_grid(name, p="desc.equilibrium.equilibrium.Equilibrium"):
    """Is name constant over toroidal surfaces and needs full surface to compute?."""
    return (
        data_index[p][name]["coordinates"] == "z"
        and data_index[p][name]["resolution_requirement"] == "rt"
    )
