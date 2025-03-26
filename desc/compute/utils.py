"""Functions for flux surface averages and vector algebra operations."""

import copy
import inspect
import warnings

import numpy as np

from desc.backend import execute_on_cpu, jnp
from desc.grid import Grid

from ..utils import errorif
from .data_index import allowed_kwargs, data_index, deprecated_names

# map from profile name to equilibrium parameter name
profile_names = {
    "pressure": "p_l",
    "iota": "i_l",
    "current": "c_l",
    "electron_temperature": "Te_l",
    "electron_density": "ne_l",
    "ion_temperature": "Ti_l",
    "atomic_number": "Zeff_l",
}


def _parse_parameterization(p):
    if isinstance(p, str):
        return p
    klass = p.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


def compute(  # noqa: C901
    parameterization, names, params, transforms, profiles, data=None, **kwargs
):
    """Compute the quantity given by name on grid.

    Parameters
    ----------
    parameterization : str, class, or instance
        Type of object to compute for, eg Equilibrium, Curve, etc.
    names : str or array-like of str
        Name(s) of the quantity(s) to compute.
    params : dict[str, jnp.ndarray]
        Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc.
        Defaults to attributes of self.
    transforms : dict of Transform
        Transforms for R, Z, lambda, etc. Default is to build from grid
    profiles : dict of Profile
        Profile objects for pressure, iota, current, etc. Defaults to attributes
        of self
    data : dict[str, jnp.ndarray]
        Data computed so far, generally output from other compute functions.
        Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
        v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
        of the cylindrical coordinates R, ϕ, Z.

    Returns
    -------
    data : dict of ndarray
        Computed quantity and intermediate variables.

    """
    basis = kwargs.pop("basis", "rpz").lower()
    errorif(basis not in {"rpz", "xyz"}, NotImplementedError)
    p = _parse_parameterization(parameterization)
    if isinstance(names, str):
        names = [names]
    if basis == "xyz" and "phi" not in names:
        names = names + ["phi"]
    # this allows the DeprecationWarning to be thrown in this file
    with warnings.catch_warnings():
        warnings.simplefilter("always", DeprecationWarning)
        for name in names:
            if name not in data_index[p]:
                raise ValueError(
                    f"Unrecognized value '{name}' for parameterization {p}."
                )
            if name in list(deprecated_names.keys()):
                warnings.warn(
                    f"Variable name {name} is deprecated and will be removed in a "
                    f"future DESC version, use name {deprecated_names[name]} "
                    "instead.",
                    DeprecationWarning,
                )
    bad_kwargs = kwargs.keys() - allowed_kwargs
    if len(bad_kwargs) > 0:
        raise ValueError(f"Unrecognized argument(s): {bad_kwargs}")

    for name in names:
        assert _has_params(name, params, p), f"Don't have params to compute {name}"
        assert _has_profiles(
            name, profiles, p
        ), f"Don't have profiles to compute {name}"
        assert _has_transforms(
            name, transforms, p
        ), f"Don't have transforms to compute {name}"

    if "grid" in transforms:

        def check_fun(name):
            reqs = data_index[p][name]["grid_requirement"]
            for req in reqs:
                errorif(
                    not hasattr(transforms["grid"], req)
                    or reqs[req] != getattr(transforms["grid"], req),
                    AttributeError,
                    f"Expected grid with '{req}:{reqs[req]}' to compute {name}.",
                )

            reqs = data_index[p][name]["source_grid_requirement"]
            errorif(
                reqs and not hasattr(transforms["grid"], "source_grid"),
                AttributeError,
                f"Expected grid with attribute 'source_grid' to compute {name}. "
                f"Source grid should have coordinates: {reqs.get('coordinates')}.",
            )
            for req in reqs:
                errorif(
                    not hasattr(transforms["grid"].source_grid, req)
                    or reqs[req] != getattr(transforms["grid"].source_grid, req),
                    AttributeError,
                    f"Expected grid with '{req}:{reqs[req]}' to compute {name}.",
                )

        _ = _get_deps(
            p, names, set(), data, transforms["grid"].axis.size, check_fun=check_fun
        )

    if data is None:
        data = {}

    data = _compute(
        p,
        names,
        params=params,
        transforms=transforms,
        profiles=profiles,
        data=data,
        **kwargs,
    )

    # convert data from default 'rpz' basis to 'xyz' basis, if requested by the user
    if basis == "xyz":
        from .geom_utils import rpz2xyz, rpz2xyz_vec

        for name in data.keys():
            errorif(
                data_index[p][name]["dim"] == (3, 3),
                NotImplementedError,
                "Tensor quantities cannot be converted to Cartesian coordinates.",
            )
            if data_index[p][name]["dim"] == 3:  # only convert vector data
                if name in ["x", "center"]:
                    data[name] = rpz2xyz(data[name])
                else:
                    data[name] = rpz2xyz_vec(data[name], phi=data["phi"])

    return data


def _compute(
    parameterization, names, params, transforms, profiles, data=None, **kwargs
):
    """Same as above but without checking inputs for faster recursion.

    Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
    v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
    of the cylindrical coordinates R, ϕ, Z.

    We need to directly call this function in objectives, since the checks in above
    function are not compatible with JIT. This function computes given names while
    using recursion to compute dependencies. If you want to call this function, you
    cannot give the argument basis='xyz' since that will break the recursion. In that
    case, either call above function or manually convert the output to xyz basis.
    """
    assert kwargs.get("basis", "rpz") == "rpz", "_compute only works in rpz coordinates"
    parameterization = _parse_parameterization(parameterization)
    if isinstance(names, str):
        names = [names]
    if data is None:
        data = {}

    names += get_data_deps(
        names, parameterization, has_axis=transforms["grid"].axis.size
    )
    names_tiers = [(data_index[parameterization][name]["tier"], name) for name in names]
    names_tiers = sorted(list(set(names_tiers)))

    for _, name in names_tiers:
        if name in data:
            # don't compute something that's already been computed
            continue
        data = data_index[parameterization][name]["fun"](
            params=params, transforms=transforms, profiles=profiles, data=data, **kwargs
        )

    return data


@execute_on_cpu
def get_data_deps(keys, obj, has_axis=False, basis="rpz", data=None):
    """Get list of keys needed to compute ``keys`` given already computed data.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index
    obj : Equilibrium, Curve, Surface, Coil, etc.
        Object to compute quantity for.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.
    basis : {"rpz", "xyz"}
        Basis of computed quantities.
    data : dict[str, jnp.ndarray]
        Data computed so far, generally output from other compute functions

    Returns
    -------
    deps : list[str]
        Names of quantities needed to compute key.

    """
    p = _parse_parameterization(obj)
    keys = [keys] if isinstance(keys, str) else keys
    if not data:
        out = []
        for key in keys:
            out += _get_deps_1_key(p, key, has_axis)
        out = set(out)
    else:
        out = _get_deps(p, keys, deps=set(), data=data, has_axis=has_axis)
        out.difference_update(keys)
    if basis.lower() == "xyz":
        out.add("phi")
    return sorted(out)


def _get_deps_1_key(p, key, has_axis):
    """Gather all quantities required to compute ``key``.

    Parameters
    ----------
    p : str
        Type of object to compute for, eg Equilibrium, Curve, etc.
    key : str
        Name of the quantity to compute.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.

    Returns
    -------
    deps_1_key : list of str
        Dependencies required to compute ``key``.


    """
    if has_axis:
        if "full_with_axis_dependencies" in data_index[p][key]:
            return data_index[p][key]["full_with_axis_dependencies"]["data"]
    elif "full_dependencies" in data_index[p][key]:
        return data_index[p][key]["full_dependencies"]["data"]

    deps = data_index[p][key]["dependencies"]["data"]
    if len(deps) == 0:
        return deps
    out = deps.copy()  # to avoid modifying the data_index
    for dep in deps:
        out += _get_deps_1_key(p, dep, has_axis)
    if has_axis:
        axis_limit_deps = data_index[p][key]["dependencies"]["axis_limit_data"]
        out += axis_limit_deps.copy()  # to be safe
        for dep in axis_limit_deps:
            out += _get_deps_1_key(p, dep, has_axis)

    return sorted(set(out))


def _get_deps(parameterization, names, deps, data=None, has_axis=False, check_fun=None):
    """Gather all quantities required to compute ``names`` given already computed data.

    Parameters
    ----------
    parameterization : str, class, or instance
        Type of object to compute for, eg Equilibrium, Curve, etc.
    names : str or array-like of str
        Name(s) of the quantity(s) to compute.
    deps : set[str]
        Dependencies gathered so far.
    data : dict[str, jnp.ndarray]
        Data computed so far, generally output from other compute functions.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.
    check_fun : callable
        If provided, ``check_fun(name)`` is called before adding name to ``deps``.

    Returns
    -------
    deps : set[str]
        All additional quantities required to compute ``names``.

    """
    p = _parse_parameterization(parameterization)
    for name in names:
        if name not in deps and (data is None or name not in data):
            if check_fun is not None:
                check_fun(name)
            deps.add(name)
            deps = _get_deps(
                p,
                data_index[p][name]["dependencies"]["data"],
                deps,
                data,
                has_axis,
                check_fun,
            )
            if has_axis:
                deps = _get_deps(
                    p,
                    data_index[p][name]["dependencies"]["axis_limit_data"],
                    deps,
                    data,
                    has_axis,
                    check_fun,
                )
    return deps


def _grow_seeds(parameterization, seeds, search_space, has_axis=False):
    """Return ``seeds`` plus keys in ``search_space`` with dependency in ``seeds``.

    Parameters
    ----------
    parameterization : str, class, or instance
        Type of object to compute for, eg Equilibrium, Curve, etc.
    seeds : set[str]
        Keys to find paths toward.
    search_space : iterable of str
        Additional keys besides ``seeds`` to consider returning.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.

    Returns
    -------
    out : set[str]
        All keys in ``search_space`` that have a dependency in ``seeds``
        plus ``seeds``.

    """
    p = _parse_parameterization(parameterization)
    out = seeds.copy()
    for key in search_space:
        deps = data_index[p][key][
            "full_with_axis_dependencies" if has_axis else "full_dependencies"
        ]["data"]
        if not seeds.isdisjoint(deps):
            out.add(key)
    return out


@execute_on_cpu
def get_derivs(keys, obj, has_axis=False, basis="rpz"):
    """Get dict of derivative orders needed to compute a given quantity.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index
    obj : Equilibrium, Curve, Surface, Coil, etc.
        Object to compute quantity for.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.
    basis : {"rpz", "xyz"}
        Basis of computed quantities.

    Returns
    -------
    derivs : dict[list, str]
        Orders of derivatives needed to compute key.
        Keys for R, Z, L, etc

    """
    p = _parse_parameterization(obj)
    keys = [keys] if isinstance(keys, str) else keys

    def _get_derivs_1_key(key):
        if has_axis:
            if "full_with_axis_dependencies" in data_index[p][key]:
                return data_index[p][key]["full_with_axis_dependencies"]["transforms"]
        elif "full_dependencies" in data_index[p][key]:
            return data_index[p][key]["full_dependencies"]["transforms"]
        deps = [key] + get_data_deps(key, p, has_axis=has_axis, basis=basis)
        derivs = {}
        for dep in deps:
            for key, val in data_index[p][dep]["dependencies"]["transforms"].items():
                if key not in derivs:
                    derivs[key] = []
                derivs[key] += val
        return derivs

    derivs = {}
    for key in keys:
        derivs1 = _get_derivs_1_key(key)
        for key1, val in derivs1.items():
            if key1 not in derivs:
                derivs[key1] = []
            derivs[key1] += val
    return {key: np.unique(val, axis=0).tolist() for key, val in derivs.items()}


def get_profiles(keys, obj, grid=None, has_axis=False, basis="rpz"):
    """Get profiles needed to compute a given quantity on a given grid.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index.
    obj : Equilibrium, Curve, Surface, Coil, etc.
        Object to compute quantity for.
    grid : Grid
        Grid to compute quantity on.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.
    basis : {"rpz", "xyz"}
        Basis of computed quantities.

    Returns
    -------
    profiles : list of str or dict of Profile
        Profiles needed to compute key.
        if eq is None, returns a list of the names of profiles needed
        otherwise, returns a dict of Profiles
        Keys for pressure, iota, etc.

    """
    p = _parse_parameterization(obj)
    keys = [keys] if isinstance(keys, str) else keys
    has_axis = has_axis or (grid is not None and grid.axis.size)
    deps = list(keys) + get_data_deps(keys, p, has_axis=has_axis, basis=basis)
    profs = []
    for key in deps:
        profs += data_index[p][key]["dependencies"]["profiles"]
    profs = sorted(set(profs))
    if isinstance(obj, str) or inspect.isclass(obj):
        return profs
    # need to use copy here because profile may be None
    profiles = {name: copy.deepcopy(getattr(obj, name)) for name in profs}
    return profiles


@execute_on_cpu
def get_params(keys, obj, has_axis=False, basis="rpz"):
    """Get parameters needed to compute a given quantity.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index
    obj : Equilibrium, Curve, Surface, Coil, etc.
        Object to compute quantity for.
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.
    basis : {"rpz", "xyz"}
        Basis of computed quantities.

    Returns
    -------
    params : list[str] or dict[str, jnp.ndarray]
        Parameters needed to compute key.
        If eq is None, returns a list of the names of params needed
        otherwise, returns a dict of ndarray with keys for R_lmn, Z_lmn, etc.

    """
    p = _parse_parameterization(obj)
    keys = [keys] if isinstance(keys, str) else keys
    deps = list(keys) + get_data_deps(keys, p, has_axis=has_axis, basis=basis)
    params = []
    for key in deps:
        params += data_index[p][key]["dependencies"]["params"]
    if isinstance(obj, str) or inspect.isclass(obj):
        return params
    temp_params = {}
    for name in params:
        p = getattr(obj, name)
        if isinstance(p, dict):
            temp_params[name] = p.copy()
        else:
            temp_params[name] = jnp.atleast_1d(p)
    return temp_params


@execute_on_cpu
def get_transforms(
    keys, obj, grid, jitable=False, has_axis=False, basis="rpz", **kwargs
):
    """Get transforms needed to compute a given quantity on a given grid.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index
    obj : Equilibrium, Curve, Surface, Coil, etc.
        Object to compute quantity for.
    grid : Grid
        Grid to compute quantity on
    jitable: bool
        Whether to skip certain checks so that this operation works under JIT
    has_axis : bool
        Whether the grid to compute on has a node on the magnetic axis.
    basis : {"rpz", "xyz"}
        Basis of computed quantities.

    Returns
    -------
    transforms : dict of Transform
        Transforms needed to compute key.
        Keys for R, Z, L, etc

    """
    from desc.basis import DoubleFourierSeries
    from desc.grid import LinearGrid
    from desc.transform import Transform

    method = "jitable" if jitable or kwargs.get("method") == "jitable" else "auto"
    keys = [keys] if isinstance(keys, str) else keys
    has_axis = has_axis or (grid is not None and grid.axis.size)
    derivs = get_derivs(keys, obj, has_axis=has_axis, basis=basis)
    transforms = {"grid": grid}
    for c in derivs.keys():
        if hasattr(obj, c + "_basis"):  # regular stuff like R, Z, lambda etc.
            basis = getattr(obj, c + "_basis")
            # first check if we already have a transform with a compatible basis
            if not jitable:
                for transform in transforms.values():
                    if basis.equiv(getattr(transform, "basis", None)):
                        ders = np.unique(
                            np.vstack([derivs[c], transform.derivatives]), axis=0
                        ).astype(int)
                        # don't build until we know all the derivs we need
                        transform.change_derivatives(ders, build=False)
                        c_transform = transform
                        break
                else:  # if we didn't exit the loop early
                    c_transform = Transform(
                        grid,
                        basis,
                        derivs=derivs[c],
                        build=False,
                        method=method,
                    )
            else:  # don't perform checks if jitable=True as they are not jit-safe
                c_transform = Transform(
                    grid,
                    basis,
                    derivs=derivs[c],
                    build=False,
                    method=method,
                )
            transforms[c] = c_transform
        elif c == "B":  # used for Boozer transform
            # assume grid is a meshgrid but only care about a single surface
            if grid.num_rho > 1:
                theta = grid.nodes[grid.unique_theta_idx, 1]
                zeta = grid.nodes[grid.unique_zeta_idx, 2]
                grid_B = LinearGrid(theta=theta, zeta=zeta, NFP=grid.NFP, sym=grid.sym)
            else:
                grid_B = grid
            transforms["B"] = Transform(
                grid_B,
                kwargs.get(
                    "B_basis",
                    DoubleFourierSeries(
                        M=kwargs.get("M_booz", 2 * obj.M),
                        N=kwargs.get("N_booz", 2 * obj.N),
                        NFP=obj.NFP,
                        sym=kwargs.get("sym", obj.R_basis.sym),
                    ),
                ),
                derivs=derivs["B"],
                build=False,
                build_pinv=True,
                method=method,
            )
        elif c == "w":  # used for Boozer transform
            # assume grid is a meshgrid but only care about a single surface
            if grid.num_rho > 1:
                theta = grid.nodes[grid.unique_theta_idx, 1]
                zeta = grid.nodes[grid.unique_zeta_idx, 2]
                grid_w = LinearGrid(theta=theta, zeta=zeta, NFP=grid.NFP, sym=grid.sym)
            else:
                grid_w = grid
            transforms["w"] = Transform(
                grid_w,
                DoubleFourierSeries(
                    M=kwargs.get("M_booz", 2 * obj.M),
                    N=kwargs.get("N_booz", 2 * obj.N),
                    NFP=obj.NFP,
                    sym=kwargs.get("sym", obj.Z_basis.sym),
                ),
                derivs=derivs["w"],
                build=False,
                build_pinv=True,
                method=method,
            )
        elif c == "h":  # used for omnigenity
            rho = grid.nodes[:, 0]
            eta = (grid.nodes[:, 1] - np.pi) / 2
            alpha = grid.nodes[:, 2] * grid.NFP
            nodes = jnp.array([rho, eta, alpha]).T
            transforms["h"] = Transform(
                Grid(nodes, jitable=jitable),
                obj.x_basis,
                derivs=derivs["h"],
                build=True,
                build_pinv=False,
                method=method,
            )
        elif c not in transforms:  # possible other stuff lumped in with transforms
            transforms[c] = getattr(obj, c)

    # now build them
    for t in transforms.values():
        if hasattr(t, "build"):
            t.build()

    return transforms


def has_data_dependencies(parameterization, qty, data, axis=False):
    """Determine if we have the data needed to compute qty."""
    return _has_data(qty, data, parameterization) and (
        not axis or _has_axis_limit_data(qty, data, parameterization)
    )


def has_dependencies(parameterization, qty, params, transforms, profiles, data):
    """Determine if we have the ingredients needed to compute qty.

    Parameters
    ----------
    parameterization : str or class
        Type of thing we're checking dependencies for. eg desc.equilibrium.Equilibrium
    qty : str
        Name of something from the data index.
    params : dict[str, jnp.ndarray]
        Dictionary of parameters we have.
    transforms : dict[str, Transform]
        Dictionary of transforms we have.
    profiles : dict[str, Profile]
        Dictionary of profiles we have.
    data : dict[str, jnp.ndarray]
        Dictionary of what we've computed so far.

    Returns
    -------
    has_dependencies : bool
        Whether we have what we need.
    """
    return (
        _has_data(qty, data, parameterization)
        and (
            not transforms["grid"].axis.size
            or _has_axis_limit_data(qty, data, parameterization)
        )
        and _has_params(qty, params, parameterization)
        and _has_profiles(qty, profiles, parameterization)
        and _has_transforms(qty, transforms, parameterization)
    )


def _has_data(qty, data, parameterization):
    p = _parse_parameterization(parameterization)
    deps = data_index[p][qty]["dependencies"]["data"]
    return all(d in data for d in deps)


def _has_axis_limit_data(qty, data, parameterization):
    p = _parse_parameterization(parameterization)
    deps = data_index[p][qty]["dependencies"]["axis_limit_data"]
    return all(d in data for d in deps)


def _has_params(qty, params, parameterization):
    p = _parse_parameterization(parameterization)
    deps = data_index[p][qty]["dependencies"]["params"]
    return all(d in params for d in deps)


def _has_profiles(qty, profiles, parameterization):
    p = _parse_parameterization(parameterization)
    deps = data_index[p][qty]["dependencies"]["profiles"]
    return all(d in profiles for d in deps)


def _has_transforms(qty, transforms, parameterization):
    p = _parse_parameterization(parameterization)
    flags = {}
    derivs = data_index[p][qty]["dependencies"]["transforms"]
    for key in derivs.keys():
        if key not in transforms:
            return False
        else:
            flags[key] = np.array(
                [d in transforms[key].derivatives.tolist() for d in derivs[key]]
            ).all()
    return all(flags.values())
