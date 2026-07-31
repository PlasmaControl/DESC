"""Utilities for computing dependencies and transforms."""

import copy
import inspect
import warnings

import numpy as np

from desc.backend import execute_on_cpu, jnp
from desc.grid import Grid

from ..utils import errorif, rpz2xyz, rpz2xyz_vec, setdefault, warnif
from .data_index import _topological_order, allowed_kwargs, data_index, deprecated_names

# map from profile name to equilibrium parameter name
profile_names = {
    "pressure": "p_l",
    "iota": "i_l",
    "current": "c_l",
    "electron_temperature": "Te_l",
    "electron_density": "ne_l",
    "ion_temperature": "Ti_l",
    "ion_density": "ni_l",
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
    parameterization,
    names,
    params,
    transforms,
    profiles,
    data=None,
    RpZ_data=None,
    **kwargs,
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
        Transforms for R, Z, lambda, etc. Default is to build from grid.
    profiles : dict of Profile
        Profile objects for pressure, iota, current, etc. Defaults to attributes
        of self
    data : dict[str, jnp.ndarray]
        Data computed so far, generally output from other compute functions.
        Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
        v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
        of the cylindrical coordinates R, ϕ, Z.
    RpZ_data : dict[str, jnp.ndarray]
        Data evaluated so far on the (R, ϕ, Z) coordinates in this dictionary.
        Should store the three entries ``"R"``, ``"phi"``, and ``"Z"``
        if the intention is to compute something at these coordinates.

    Returns
    -------
    data : dict[str, jnp.ndarray]
        Quantities and intermediate variables computed on the
        grid attached to the transforms.
    RpZ_data : dict[str, jnp.ndarray]
        Quantities and intermediate variables computed on the
        (R, ϕ, Z) coordinates in ``RpZ_data``. If ``RpZ_data``
        was not given then this dictionary will not be returned.

    """
    return_RpZ_data = RpZ_data is not None

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
            errorif(
                name not in data_index[p],
                msg=f"Unrecognized value '{name}' for parameterization {p}.",
            )
            warnif(
                name in list(deprecated_names.keys()),
                DeprecationWarning,
                f"Variable name {name} is deprecated and will be removed in a future "
                f"DESC version, use name {deprecated_names.get(name, None)} instead.",
            )
    bad_kwargs = kwargs.keys() - allowed_kwargs
    errorif(bad_kwargs, msg=f"Unrecognized argument(s): {bad_kwargs}")

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

        # this call is purely for validation of the grid/deps consistency
        _ = _get_deps(p, names, data, transforms["grid"].axis.size, check_fun=check_fun)

    if data is None:
        data = {}

    # Need to query this before JIT barriers detach them from each other.
    same_grid = data is RpZ_data

    data = _compute(
        p,
        names,
        params=params,
        transforms=transforms,
        profiles=profiles,
        data=data,
        RpZ_data=RpZ_data,
        **kwargs,
    )
    if return_RpZ_data:
        if same_grid:
            RpZ_data = data
        RpZ_data = _compute_RpZ_data(
            p,
            names,
            params=params,
            transforms=transforms,
            profiles=profiles,
            data=data,
            RpZ_data=RpZ_data,
            **kwargs,
        )
        if same_grid:
            data = RpZ_data

    data = _convert_basis(p, data, basis)

    if return_RpZ_data:
        if data is not RpZ_data:
            RpZ_data = _convert_basis(p, RpZ_data, basis)
        return data, RpZ_data
    else:
        return data


def _convert_basis(p, data, basis):
    # convert data from default 'rpz' basis to 'xyz' basis, if requested by the user
    if basis == "xyz":
        for name in data.keys():
            if name == "potential data":
                continue
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
    """Compute quantities without input validation.

    Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
    v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
    of the cylindrical coordinates R, ϕ, Z.

    This function is called directly by objectives because the validation in
    :func:`compute` is not compatible with JIT. Dependencies are evaluated in
    topological order. The argument ``basis="xyz"`` is not supported; call
    :func:`compute` or convert the output manually instead.

    Returns
    -------
    data : dict[str, jnp.ndarray]
        Quantities and intermediate variables computed on the
        grid attached to the transforms.

    """
    assert kwargs.get("basis", "rpz") == "rpz", "_compute only works in rpz coordinates"
    p = _parse_parameterization(parameterization)
    if isinstance(names, str):
        names = [names]
    if data is None:
        data = {}

    has_axis = bool(transforms["grid"].axis.size)
    needed = _get_deps(p, names, data=data, has_axis=has_axis)
    needed = sorted(needed, key=_topological_order[p].__getitem__)

    for name in needed:
        if name in data:
            # a previously-called fun may have populated this already
            continue
        # RpZ quantities are evaluated separately on the coordinates in RpZ_data.
        if data_index[p][name]["coordinates"] != "RpZ":
            data = data_index[p][name]["fun"](
                params=params,
                transforms=transforms,
                profiles=profiles,
                data=data,
                **kwargs,
            )
    return data


def _compute_RpZ_data(
    parameterization,
    names,
    params,
    transforms,
    profiles,
    data,
    RpZ_data,
    **kwargs,
):
    """Compute quantities on the coordinates in ``RpZ_data`` without validation.

    Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
    v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
    of the cylindrical coordinates R, ϕ, Z.

    Dependencies are evaluated in topological order. Quantities evaluated on the
    transform grid are read from ``data``; quantities whose coordinates are ``RpZ``
    are evaluated into ``RpZ_data``.

    Returns
    -------
    RpZ_data : dict[str, jnp.ndarray]
        Quantities and intermediate variables computed on the
        (R, ϕ, Z) coordinates in ``RpZ_data``.

    """
    assert kwargs.get("basis", "rpz") == "rpz", "_compute only works in rpz coordinates"
    p = _parse_parameterization(parameterization)
    if isinstance(names, str):
        names = [names]

    # A quantity evaluated on the transform grid cannot satisfy the same quantity
    # evaluated at RpZ_data's coordinates. Other transform-grid quantities can be
    # reused as dependencies, as can everything already evaluated in RpZ_data.
    available = {
        name
        for name in data
        if name not in data_index[p] or data_index[p][name]["coordinates"] != "RpZ"
    }
    available.update(RpZ_data)
    has_axis = bool(transforms["grid"].axis.size)
    needed = _get_deps(p, names, data=available, has_axis=has_axis)
    needed = sorted(needed, key=_topological_order[p].__getitem__)

    for name in needed:
        if data_index[p][name]["coordinates"] != "RpZ" or name in RpZ_data:
            continue
        RpZ_data = data_index[p][name]["fun"](
            params=params,
            transforms=transforms,
            profiles=profiles,
            data=data,
            RpZ_data=RpZ_data,
            **kwargs,
        )
    return RpZ_data


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
    data : dict[str, jnp.ndarray] or set[str]
        Data computed so far, generally output from other compute functions

    Returns
    -------
    deps : list[str]
        Names of quantities needed to compute key.

    """
    p = _parse_parameterization(obj)
    keys = [keys] if isinstance(keys, str) else keys
    deps_type = "full_with_axis_dependencies" if has_axis else "full_dependencies"
    if not data:
        out = []
        for key in keys:
            out += data_index[p][key][deps_type]["data"]
        out = set(out)
    else:
        out = _get_deps(p, keys, data=data, has_axis=has_axis)
        out.difference_update(keys)
    if basis.lower() == "xyz":
        out.add("phi")
    return sorted(out)


def _get_deps(parameterization, names, data=None, has_axis=False, check_fun=None):
    """Gather all quantities required to compute ``names`` given already computed data.

    Parameters
    ----------
    parameterization : str, class, or instance
        Type of object to compute for, eg Equilibrium, Curve, etc.
    names : str or array-like of str
        Name(s) of the quantity(s) to compute.
    data : dict[str, jnp.ndarray] or set[str]
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
    deps = set()
    # below while loop expands each direct dependency if they are not
    # in data or they are already added to the set before
    stack = [n for n in names if data is None or n not in data]
    while stack:
        name = stack.pop()
        if name in deps:
            continue
        if check_fun is not None:
            check_fun(name)
        deps.add(name)
        direct = data_index[p][name]["dependencies"]
        for dep in direct["data"]:
            if dep in deps:
                continue
            if data is not None and dep in data:
                continue
            stack.append(dep)
        if has_axis:
            for dep in direct["axis_limit_data"]:
                if dep in deps:
                    continue
                if data is not None and dep in data:
                    continue
                stack.append(dep)
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
    deps_type = "full_with_axis_dependencies" if has_axis else "full_dependencies"
    for key in search_space:
        deps = data_index[p][key][deps_type]["data"]
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
    deps_type = "full_with_axis_dependencies" if has_axis else "full_dependencies"

    derivs = {}
    for key in keys:
        derivs1 = data_index[p][key][deps_type]["transforms"]
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
    deps_type = "full_with_axis_dependencies" if has_axis else "full_dependencies"
    profs = set()
    # below loop doesn't consider extra "phi" in basis="xyz" case
    # but since "phi" doesn't have any profiles, no problem
    # this way we skip calling get_data_deps again
    for key in keys:
        profs.update(data_index[p][key][deps_type]["profiles"])
    profs = sorted(profs)
    if isinstance(obj, str) or inspect.isclass(obj):
        return profs
    # need to use copy here because profile may be None
    profiles = {name: copy.deepcopy(getattr(obj, name)) for name in profs}
    return profiles


@execute_on_cpu
def get_params(keys, obj, has_axis=False, basis="rpz", params=None):
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
    params : dict[str, jnp.ndarray]
        Params computed so far.

    Returns
    -------
    params : list[str] or dict[str, jnp.ndarray]
        Parameters needed to compute key.
        If eq is None, returns a list of the names of params needed
        otherwise, returns a dict of ndarray with keys for R_lmn, Z_lmn, etc.

    """
    p = _parse_parameterization(obj)
    keys = [keys] if isinstance(keys, str) else keys
    deps_type = "full_with_axis_dependencies" if has_axis else "full_dependencies"
    param_names = set()
    # below loop doesn't consider extra "phi" in basis="xyz" case
    # but since "phi" doesn't have any params, no problem
    # this way we skip calling get_data_deps again
    # TODO (#568): This will probably need w_lmn
    for key in keys:
        param_names.update(data_index[p][key][deps_type]["params"])
    param_names = sorted(param_names)

    if isinstance(obj, str) or inspect.isclass(obj):
        return param_names

    params = setdefault(params, {})
    for name in param_names:
        if name not in params:
            value = getattr(obj, name)
            params[name] = (
                value.copy()
                if isinstance(value, dict)
                else (None if value is None else jnp.atleast_1d(value))
            )
    return params


@execute_on_cpu
def get_transforms(  # noqa: C901
    keys,
    obj,
    grid,
    jitable=False,
    has_axis=False,
    basis="rpz",
    transforms=None,
    **kwargs,
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
    transforms : dict[str, Transform]
        Transforms that are already computed.

    Returns
    -------
    transforms : dict[str, Transform]
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

    transforms = setdefault(transforms, {})
    transforms.setdefault("grid", grid)
    p = _parse_parameterization(obj)

    for c in derivs.keys():
        if c in transforms:
            continue
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
                        if c == "Phi_tilde" and p in (
                            "desc.magnetic_fields._laplace.SourceFreeField",
                            "desc.magnetic_fields._laplace.FreeSurfaceOuterField",
                        ):
                            transform.build_pinv()
                        c_transform = transform
                        break
                else:  # if we didn't exit the loop early
                    c_transform = Transform(
                        grid,
                        basis,
                        derivs=derivs[c],
                        build=False,
                        build_pinv=c == "Phi_tilde"
                        and (
                            p == "desc.magnetic_fields._laplace.SourceFreeField"
                            or p
                            == "desc.magnetic_fields._laplace.FreeSurfaceOuterField"
                        ),
                        method=method,
                    )
            else:  # don't perform checks if jitable=True as they are not jit-safe
                c_transform = Transform(
                    grid,
                    basis,
                    derivs=derivs[c],
                    build=False,
                    build_pinv=c == "Phi_tilde"
                    and (
                        p == "desc.magnetic_fields._laplace.SourceFreeField"
                        or p == "desc.magnetic_fields._laplace.FreeSurfaceOuterField"
                    ),
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
